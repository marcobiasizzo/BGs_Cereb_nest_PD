#!/usr/bin/env python3
"""
"""

__author__ = 'marco'

import time
import numpy as np
import scipy.io
from datetime import datetime
from marco_nest_utils import utils, visualizer as vsl
from multiprocessing import cpu_count
import os
import pickle
from pathlib import Path
import sys
from matplotlib.pyplot import close

# NEST modules
import nest

if str(Path.home()) == '/home/gambosi':
    MODULE_PATH = str(Path.home()) + '/nest_env/nest/lib/nest/ml_module'
    nest.Install(MODULE_PATH)  # Import my_BGs module
    MODULE_PATH = str(Path.home()) + '/nest_env/nest/lib/nest/cerebmodule'
    nest.Install(MODULE_PATH)  # Import CerebNEST
else:
    MODULE_PATH = str(Path.home()) + '/nest/lib/nest/ml_module'
    nest.Install(MODULE_PATH)  # Import my_BGs module
    MODULE_PATH = str(Path.home()) + '/nest/lib/nest/cerebmodule'
    nest.Install(MODULE_PATH)  # Import CerebNEST

# path to h5py spatial distribution
hdf5_path = 'Cereb_nest/scaffold_full_IO_400.0x400.0_microzone.hdf5'

# my modules
from Cereb_nest.Cereb import Cereb_class as C_c
from BGs_nest.BGs import BGs_class as B_c
from nest_multiscale.nest_multiscale import sim_handler, generate_ode_dictionary
from experiments import conditioning

print(f'CPU = {cpu_count()}')

# simulation parameters
if str(Path.home()) == '/home/marco':
    CORES = 4
    VIRTUAL_CORES = 4
    run_on_vm = False
elif str(Path.home()) == '/home/gambosi':
    CORES = 32
    VIRTUAL_CORES = 32
    run_on_vm = True
else:
    CORES = 24
    run_on_vm = True

# IO stimulation every trial
t_start = 300
t_end = 400
stimulation_frequency = 50  # [sp/s]

N_BGs = 20000
N_Cereb = 96767
load_from_file = False       # load results from directory or simulate and save
dopa_depl_level = -0.2      # between 0. and -0.8
sol_n = 17
if dopa_depl_level != 0.:
    dopa_depl = True
else:
    dopa_depl = False

mode_list = ['external_dopa', 'internal_dopa', 'both_dopa']
experiment_list = ['active', 'EBCC']
mode = mode_list[2]
experiment = experiment_list[0]

if experiment == 'active':
    settling_time = 1000.
    sim_time = 3000.
    start_time = 0.  # starting time for histograms data
    sim_period = 1.  # ms
    trials = 1
elif experiment == 'EBCC':
    settling_time = 500.
    sim_time = 400.
    start_time = 0.  # starting time for histograms data
    sim_period = 10.  # ms
    trials = 10
else:
    assert False, 'Select a correct experiment'

# defines where the dopamine is depleted
dopa_depl_cereb = 0.
dopa_depl_BGs = 0.
if mode != 'external_dopa':
    dopa_depl_cereb = dopa_depl_level
if mode != 'internal_dopa':
    dopa_depl_BGs = dopa_depl_level

# set number of kernels
nest.ResetKernel()
nest.SetKernelStatus({"total_num_virtual_procs": CORES})
nest.set_verbosity("M_ERROR")  # reduce plotted info

# set saving directory
# date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
# savings_dir = f'shared_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}'  # f'savings/{date_time}'
savings_dir = f'savings/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}'  # f'savings/{date_time}'
if dopa_depl: savings_dir = savings_dir + f'_dopadepl_{(str(int(-dopa_depl_level*10)))}'

if len(sys.argv) > 1:
    n_trial = int(sys.argv[1])
    savings_dir = savings_dir + f'_trial_{n_trial}'

# create folder if not present
if not load_from_file:
    if not os.path.exists(savings_dir):
        os.makedirs(savings_dir)
        print(f'\nWriting to {savings_dir}\n')
    else:
        print(f'\nATTENTION: subscribing to {savings_dir}\n')

''' Set up multi-scale simulation: order is important| '''

# Define all the NEST populations:
Cereb_pop_names = ['golgi', 'glomerulus', 'granule', 'purkinje', 'basket', 'stellate', 'dcn', 'dcnp', 'io']
BGs_pop_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI', 'STN', 'SNr']
# Select the NEST populations you will attach to a spike detector:
if experiment == 'active':
    Cereb_recorded_names = Cereb_pop_names
    BGs_recorded_names = BGs_pop_names
elif experiment == 'EBCC':
    Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn', 'io']
    BGs_recorded_names = ['STN', 'SNr']
recorded_names = Cereb_recorded_names + BGs_recorded_names
# Define the names of the mass-model populations:
ode_names = ['CTX', 'thalamus', 'nRT']

# Nest populations connected to mass models:
Cereb_pop_names_to_ode = ['dcn']
BGs_pop_names_to_ode = ['SNr']
# Mass model population connected to NEST:
Cereb_pop_names_to_nest = ['CTX']
BGs_pop_names_to_nest = ['CF', 'CM1_1', 'CM1_2', 'CM2_1', 'CM2_2', 'CS_1', 'CS_2']

# Mass models connections:
w = [20., 5., 8., 25., 15., 5., 19., 5., 15., 20., 20.]
if sol_n == 2: b_c_params = [192.669,  88.011,  98.1135, 114.351]   # 2 - bad average fr
if sol_n == 7: b_c_params = [191.817,  88.011,  98.422, 114.351]    # 7 - ok but different from genetic
if sol_n == 11: b_c_params = [191.817,  88.011,  96.298, 140.390]   # 11 -
if sol_n == 17: b_c_params = [170.676,  84.751,  77.478, 174.500]

# with bground
b1 = w[3] / b_c_params[0]     # DCN -> Thal  # 2900
b2 = -w[4] / b_c_params[1]     # SNr -> Thal
c1 = b_c_params[2]             # CTX -> MF
c5 = b_c_params[3]            # CTX -> STN
# maintain the same oscillation ratio of Lindahl
c2 = c5 / 175. * 173.14     # CTX -> FSN
c3 = c5 / 175. * 120.12     # CTX -> MSN1
c4 = c5 / 175. * 158.84     # CTX -> MSN2
c5 = c5                     # CTX -> STN

# Define ODEs matrices
# follow the order in ode_names = ['CTX', 'thalamus', 'nRT']
A_mat = np.array([[0, w[0], 0],
                  [w[1], 0, -w[2]],
                  [w[5], 0, 0]])
# follow also the order in Cereb_pop_names_to_ode = ['dcn'] + BGs_pop_names_to_ode = ['SNr']
B_mat = np.array([[0, 0],
                  [b1, b2],
                  [0, 0]])
# follow also the order in Cereb_pop_names_to_nest = ['CTX'] + BGs_pop_names_to_nest
C_mat = np.array([[c1, 0, 0],
                  [c2, 0, 0],
                  [c3, 0, 0],
                  [c3, 0, 0],
                  [c4, 0, 0],
                  [c4, 0, 0],
                  [c5, 0, 0],
                  [c5, 0, 0]])

# ODE params from Yousif et al. 2020
b_e = 1.3
b_i = 2.0
theta_e = 4.0
theta_i = 3.7
k_e = 1 - 1/(1+np.exp(b_e*theta_e))
k_i = 1 - 1/(1+np.exp(b_i*theta_i))

# define sigmoid params
theta = np.array([theta_e, theta_e, theta_i])
lambda_max = lambda x: np.array([k_e, k_e, k_i]) - x
a = np.array([b_e, b_e, b_i])  # []
q_exp = lambda b, th: -1. / (1 + np.exp(b * th))
q = q_exp(a, theta)

# define characteristic time
tau = 10.  # [ms]

# create a dictionary with all the defined prams
params_dic = generate_ode_dictionary(A_matrix=A_mat, B_matrix=B_mat, C_matrix=C_mat, theta_vec=theta,
                                     lambda_max_vec=lambda_max, a_vec=a, q_vec=q, tau_val=tau, const_in=None)

if __name__ == "__main__":
    if not load_from_file:
        # create an instance of the populations and inputs
        Cereb_class = C_c(nest, hdf5_path, 'spike_generator', n_spike_generators=500,
                          mode=mode, experiment=experiment, dopa_depl=dopa_depl_cereb, LTD=-1.0e-3*0.02)
        BGs_class = B_c(nest, N_BGs, 'active', 'BGs_nest/default_params.csv', dopa_depl=dopa_depl_BGs,
                        cortex_type='spike_generator', in_vitro=False,
                        n_spike_generators={'FS': 250, 'M1': 1250, 'M2': 1250, 'ST': 50})

        if experiment == 'active':
            additional_classes = []
        if experiment == 'EBCC':
            cond_exp = conditioning(nest, Cereb_class, t_start=t_start, t_end=t_end, stimulation=50)
            additional_classes = [cond_exp]

        recorded_list = [Cereb_class.Cereb_pops[name] for name in Cereb_recorded_names] + \
                        [BGs_class.BGs_pops[name] for name in BGs_recorded_names]
        pop_list_to_ode = [Cereb_class.Cereb_pops[name] for name in Cereb_pop_names_to_ode] + \
                          [BGs_class.BGs_pops[name] for name in BGs_pop_names_to_ode]
        pop_list_to_nest = [Cereb_class.CTX_pops[name] for name in Cereb_pop_names_to_nest] + \
                           [BGs_class.CTX_pops[name] for name in BGs_pop_names_to_nest]

        # initiate the simulation handler
        s_h = sim_handler(nest, pop_list_to_ode, pop_list_to_nest,
                          params_dic, sim_time, sim_period_=sim_period, additional_classes=additional_classes)

        # record membrane potential from the first neuron of the population
        # MF parrots neurons cannot be connected to vm
        # vm_list = utils.attach_voltmeter(nest, recorded_list[1:], sampling_resolution=2., target_neurons=0)

        # record spikes neurons
        sd_list = utils.attach_spikedetector(nest, recorded_list,
                                             pop_list_to_ode=pop_list_to_ode,   # possibility to load sd from s_h
                                             sd_list_to_ode=s_h.sd_list)

        # min and max index for every population
        pop_ids = {**Cereb_class.Cereb_pop_ids, **BGs_class.BGs_pop_ids}
        # dictionary of the population params
        model_dic = utils.create_model_dictionary(N_BGs+N_Cereb, recorded_names, pop_ids, sim_time,
                                                  sample_time=sim_period, settling_time=settling_time,
                                                  trials=trials, b_c_params=b_c_params)
        print('Starting the simulation ...')
        tic = time.time()
        s_h.simulate(tot_trials=trials, pre_sim_time=settling_time)
        toc = time.time()
        print(f'Elapsed simulation time with {CORES} cores: {int((toc - tic) / 60)} min, {(toc - tic) % 60:.0f} sec')

        # potentials = utils.get_voltage_values(nest, vm_list, recorded_names[1:])
        rasters = utils.get_spike_values(nest, sd_list, recorded_names)
        # load mass models states and inputs
        mass_models_sol = {'mass_frs': s_h.ode_sol,
                           'mass_frs_times': s_h.ode_sol_t,
                           'in_frs': s_h.u_sol}

        with open(f'{savings_dir}/model_dic', 'wb') as pickle_file:
            pickle.dump(model_dic, pickle_file)
        # with open(f'{savings_dir}/potentials', 'wb') as pickle_file:
        #     pickle.dump(potentials, pickle_file)
        with open(f'{savings_dir}/rasters', 'wb') as pickle_file:
            pickle.dump(rasters, pickle_file)
        with open(f'{savings_dir}/mass_models_sol', 'wb') as pickle_file:
            pickle.dump(mass_models_sol, pickle_file)

    elif load_from_file:
        print(f'Simulation results loaded from files')

        with open(f'{savings_dir}/model_dic', 'rb') as pickle_file:
            model_dic = pickle.load(pickle_file)
        # with open(f'{savings_dir}/potentials', 'rb') as pickle_file:
        #     potentials = pickle.load(pickle_file)
        with open(f'{savings_dir}/rasters', 'rb') as pickle_file:
            rasters = pickle.load(pickle_file)
        with open(f'{savings_dir}/mass_models_sol', 'rb') as pickle_file:
            mass_models_sol = pickle.load(pickle_file)

    print(f'Showing results obtained from {model_dic["b_c_params"]}')

    # show results
    # fig1, ax1 = vsl.plot_potential_multiple(potentials, clms=1, t_start=start_time)
    # fig1.show()

    fig2, ax2 = vsl.raster_plots_multiple(rasters, clms=1, start_stop_times=[0, sim_time*trials], t_start=start_time)
    # fig2.show()

    fig3, ax3 = vsl.plot_mass_frs(mass_models_sol, ode_names, u_array=None, # xlim=[0, settling_time+sim_time*trials],
                                  ylim=[None, None])
    fig3.show()

    # fig4, ax4 = vsl.plot_mass_frs(mass_frs[:, :3], [0, sim_time], ode_names + ['DCN_in', 'SNr_in'],
    #                               u_array=in_frs / np.array([w[3], -w[4]]) * np.array([b1, b2]),
    #                               xlim=[0, 1000], ylim=[None, None])
    # fig4.show()

    fig5, ax5 = vsl.combine_axes_in_figure(rasters, mass_models_sol, clms=1,
                                           legend_labels=ode_names, t_start=start_time, ylim=[None, None])
    # fig5.show()

    print(f'mean f.r.  = {mass_models_sol["mass_frs"].mean(axis=0)}')
    print(f'mean input = {mass_models_sol["in_frs"].mean() / np.array([w[3], -w[4]]) * np.array([b1, b2])}')

    fr_stats = utils.calculate_fr_stats(rasters, model_dic['pop_ids'], t_start=start_time)
    # name_list = ['Glomerulus', 'Purkinje', 'DCNp', 'GPeTA', 'GPeTI', 'STN', 'SNr']

    # ['glomerulus', 'purkinje', 'dcn']
    Cereb_target = np.array([25.445, 114.332, 46.073])
    # ['GPeTA', 'GPeTI', 'STN', 'SNr']
    # BGs_target = np.array([9.30, 38.974, 12.092, 24.402])
    BGs_target = np.array([12.092, 24.402])     # ['STN', 'SNr']
    fr_target = np.concatenate((Cereb_target, BGs_target))

    # scale errors according to standard deviation:
    # fr_weights = np.array([1. / 0.4398, 1. / 0.3276, 1. / 0.6918, 1. / 0.4017, 1. / 0.31366, 1 / 0.276, 1 / 0.242])
    fr_weights = np.array([1. / 0.4398, 1. / 0.3276, 1. / 0.6918, 1 / 0.276, 1 / 0.242])

    # fr = np.concatenate((fr_stats['fr'][0:5], fr_stats['fr'][6:8]))
    # flags = [True if n != 'io' else False for n in recorded_names]
    # fr = np.array(fr_stats['fr'])[flags]

    # print the fitness
    # filter_range = [30, 50]     # [Hz]
    # filter_sd = 6               # [Hz]
    # utils.fitness_function(fr, fr_target, mass_models_sol["mass_frs"], sim_period,
    #                        filter_range=filter_range, filter_sd=filter_sd,
    #                        t_start=start_time, fr_weights=fr_weights)

    # fr_target = np.concatenate((fr_target[0:5], fr_target[5:]))
    # fig6, ax6 =vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['name'], CV_list=fr_stats['CV'],
    #                           target_fr=fr_target)
    # fig6.show()

    # fig7, ax7 = vsl.plot_fourier_transform(mass_models_sol["mass_fr"][:, :], sim_period, ode_names,
    #                                        mean=sum(filter_range)/2, sd=filter_sd, t_start=start_time)
    # fig7.show()

    # fig8, ax8, _ = vsl.plot_wavelet_transform(mass_models_sol, sim_period, ode_names,
    #                                        mean=sum(filter_range)/2, sd=filter_sd, t_start=start_time, y_range=[0, 580])
    # fig8.show()

    # fig8, ax8, _ = vsl.plot_wavelet_transform_and_mass(mass_models_sol, sim_period, ode_names,
    #                                            mean=sum(filter_range)/2, sd=filter_sd, t_start=start_time, t_end=sim_time,
    #                                            y_range=[0, 580])
    # fig8.show()

    instant_fr = utils.fr_window_step(rasters, model_dic['pop_ids'], settling_time + sim_time*trials, window=10., step=5.)
    fig9, ax9 = vsl.plot_instant_fr_multiple(instant_fr, clms=1, t_start=start_time)
    fig9.show()

    if experiment == 'EBCC':
        average_fr_per_trial = utils.average_fr_per_trial([rasters], model_dic['pop_ids'], t_end, t_end, settling_time, trials)
        POP_NAME = 'purkinje'
        io_idx = [i for i, n in enumerate(recorded_names) if n == POP_NAME][0]
        fig10, ax10 = vsl.plot_fr_learning1([average_fr_per_trial], recorded_names, POP_NAME, labels=[dopa_depl_level])
        # fig10, ax10 = vsl.plot_fr_learning2([instant_fr[io_idx]], t_start, t_end, settling_time, trials, POP_NAME, labels=[dopa_depl_level])
        fig10.show()

