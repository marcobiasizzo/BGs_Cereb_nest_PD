# -*- coding: utf-8 -*-
"""
"""

__author__ = 'marco'

import time
import numpy as np
from datetime import datetime
from marco_nest_utils import utils, visualizer as vsl
import os
import pickle
import pygad
from pathlib import Path
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

# path to h5py spatial distribution - Cereb
hdf5_path = 'Cereb_nest/scaffold_full_IO_400.0x400.0_microzone.hdf5'

# my modules
from Cereb_nest.Cereb import Cereb_class as C_c
from BGs_nest.BGs import BGs_class as B_c
from nest_multiscale.nest_multiscale import sim_handler, generate_ode_dictionary

# simulation parameters
CORES = 24  # cpu_count()

sim_time = 3000.    # total simulation time
start_time = 1000.  # starting time for averaged data
sim_period = 1      # sim time interval
trials = 1
N_BGs = 20000
N_Cereb = 96767
load_from_file = False      # load results from directory or simulate and save
dopa_depl_level = 0.        # between 0. and -0.8
if dopa_depl_level != 0.:
    dopa_depl = True
else:
    dopa_depl = False

# set number of kernels
nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": CORES, "total_num_virtual_procs": CORES})  # , 'resolution': 0.1})
nest.set_verbosity("M_ERROR")  # reduce plotted info

# set saving directory
# date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
savings_dir = f'savings/complete_{int(sim_time)}ms'  # f'savings/{date_time}'
if dopa_depl: savings_dir = savings_dir + f'_dopadepl_{(str(int(-dopa_depl_level*10)))}'
# create folder if not present
if not load_from_file:
    if not os.path.exists(savings_dir):
        os.makedirs(savings_dir)  # create folder if not present

''' Set up multi-scale simulation: order is important! '''

# Define all the NEST populations:
Cereb_pop_names = ['golgi', 'glomerulus', 'granule', 'purkinje', 'basket', 'stellate', 'dcn', 'dcnp', 'io']
BGs_pop_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI', 'STN', 'SNr']
# Select the NEST populations you will attach to a spike detector:
Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn']
BGs_recorded_names = ['GPeTA', 'GPeTI', 'STN', 'SNr']
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

# Define ODEs matrices
# follow the order in ode_names = ['CTX', 'thalamus', 'nRT']
A_mat = np.array([[0, w[0], 0],
                  [w[1], 0, -w[2]],
                  [w[5], 0, 0]])


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

idx = 0

def fitness_func(solution, solution_idx):
    global idx
    idx += 1
    print(f'Evaluating solution #{idx}')

    b1 = w[3] / solution[0]     # DCN -> Thal
    b2 = -w[4] / solution[1]    # SNr -> Thal
    c1 = solution[2]            # CTX -> MF
    c5 = solution[3]            # CTX -> STN
    # maintain the same oscillation ratio of Lindahl
    c2 = c5 / 175. * 173.14     # CTX -> FSN
    c3 = c5 / 175. * 120.12     # CTX -> MSN1
    c4 = c5 / 175. * 158.84     # CTX -> MSN2
    c5 = c5                     # CTX -> STN

    print(f'Solution = {solution}')
    print(f'b1 = {b1}, b2 = {b2}')
    print(f'c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}')

    # set number of kernels
    nest.ResetKernel()
    round_time = round(100 * time.time())
    nest.SetKernelStatus({'grng_seed': round_time + 1,
                          'rng_seeds': [round_time + 1+i for i in range(1,25)],
                        #   'rng_seeds': [round_time + 2, round_time + 3, round_time + 4, round_time + 5],
                          'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
    nest.set_verbosity("M_ERROR")  # reduce plotted info

    if not load_from_file:

        B_mat = np.array([[0, 0],
                          [b1, b2],
                          [0, 0]])
        C_mat = np.array([[c1, 0, 0],
                          [c2, 0, 0],
                          [c3, 0, 0],
                          [c3, 0, 0],
                          [c4, 0, 0],
                          [c4, 0, 0],
                          [c5, 0, 0],
                          [c5, 0, 0]])

        # create a dictionary with all the defined prams
        params_dic = generate_ode_dictionary(A_matrix=A_mat, B_matrix=B_mat, C_matrix=C_mat, theta_vec=theta,
                                             lambda_max_vec=lambda_max, a_vec=a, q_vec=q, tau_val=tau, const_in=None)

        # create an instance of the Cereb populations and inputs
        # Cereb_class = C_c(nest, hdf5_path, 'spike_generator', n_spike_generators=500)
        Cereb_class = C_c(nest, hdf5_path, 'spike_generator', n_spike_generators=500,
                             dopa_depl=dopa_depl_level)
        BGs_class = B_c(nest, N_BGs, 'active', 'BGs_nest/default_params.csv', dopa_depl=dopa_depl_level,
                        cortex_type='spike_generator', in_vitro=False,
                        n_spike_generators={'FS': 250, 'M1': 1250, 'M2': 1250, 'ST': 50})

        recorded_list = [Cereb_class.Cereb_pops[name] for name in Cereb_recorded_names] + \
                        [BGs_class.BGs_pops[name] for name in BGs_recorded_names]
        pop_list_to_ode = [Cereb_class.Cereb_pops[name] for name in Cereb_pop_names_to_ode] + \
                          [BGs_class.BGs_pops[name] for name in BGs_pop_names_to_ode]
        pop_list_to_nest = [Cereb_class.CTX_pops[name] for name in Cereb_pop_names_to_nest] + \
                           [BGs_class.CTX_pops[name] for name in BGs_pop_names_to_nest]

        # initiate the simulation handler
        # s_h = sim_handler(nest, pop_list_to_ode, pop_list_to_nest,
        #                   params_dic, sim_time, sim_period_=sim_period)
        s_h = sim_handler(nest, pop_list_to_ode, pop_list_to_nest,
                                params_dic, sim_time, sim_period_=sim_period, resolution=0.1, additional_classes=[])
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
        # model_dic = utils.create_model_dictionary(N_BGs+N_Cereb, recorded_names, pop_ids, sim_time, solution)
        
        model_dic = utils.create_model_dictionary(N_BGs+N_Cereb, recorded_names, pop_ids, sim_time,
                                                  sample_time=sim_period, settling_time=start_time,
                                                  trials=trials, b_c_params=solution)
        print('Starting the simulation ...')
        tic = time.time()
        s_h.simulate(tot_trials=trials, pre_sim_time=start_time)
        toc = time.time()
        print(f'Elapsed simulation time with {CORES} cores: {int((toc - tic) / 60)} min, {(toc - tic) % 60:.0f} sec')

        # potentials = utils.get_voltage_values(nest, vm_list, recorded_names)
        rasters = utils.get_spike_values(nest, sd_list, recorded_names)
        # load mass models states and inputs
        mass_frs = s_h.ode_sol
        in_frs = s_h.u_sol

        with open(f'{savings_dir}/model_dic_{idx}', 'wb') as pickle_file:
            pickle.dump(model_dic, pickle_file)
        # with open(f'{savings_dir}/potentials', 'wb') as pickle_file:
        #     pickle.dump(potentials, pickle_file)
        with open(f'{savings_dir}/rasters_{idx}', 'wb') as pickle_file:
            pickle.dump(rasters, pickle_file)
        with open(f'{savings_dir}/mass_frs_{idx}', 'wb') as pickle_file:
            pickle.dump(mass_frs, pickle_file)
        with open(f'{savings_dir}/in_frs_{idx}', 'wb') as pickle_file:
            pickle.dump(in_frs, pickle_file)

    elif load_from_file:
        print(f'Simulation results loaded from files')

        with open(f'{savings_dir}/model_dic_{idx}', 'rb') as pickle_file:
            model_dic = pickle.load(pickle_file)
        # with open(f'{savings_dir}/potentials', 'rb') as pickle_file:
        #     potentials = pickle.load(pickle_file)
        with open(f'{savings_dir}/rasters_{idx}', 'rb') as pickle_file:
            rasters = pickle.load(pickle_file)
        with open(f'{savings_dir}/mass_frs_{idx}', 'rb') as pickle_file:
            mass_frs = pickle.load(pickle_file)
        with open(f'{savings_dir}/in_frs_{idx}', 'rb') as pickle_file:
            in_frs = pickle.load(pickle_file)


    # show results
    # fig1, ax1 = vsl.plot_potential_multiple(potentials, clms=1, t_start=start_time)
    # fig1.show()

    # fig2, ax2 = vsl.raster_plots_multiple(rasters, clms=1, start_stop_times=[0, sim_time], t_start=start_time)
    # fig2.show()

    # fig3, ax3 = vsl.plot_mass_frs(mass_frs[:, :], [0, sim_time], ode_names, u_array=None, xlim=[1000, sim_time],
    #                               ylim=[None, None], title=f'solution_idx = {idx}')
    # fig3.show()

    # fig4, ax4 = vsl.plot_mass_frs(mass_frs[:, :3], [0, sim_time], ode_names + ['DCN_in', 'SNr_in'],
    #                               u_array=in_frs / np.array([w[3], -w[4]]) * np.array([b1, b2]),
    #                               xlim=[0, 1000], ylim=[None, None])
    # # fig4.show()

    # fig5, ax5 = vsl.combine_axes_in_figure(rasters, mass_frs[:, :3], clms=1, start_stop_times=[0, sim_time],
    #                                        legend_labels=ode_names, t_start=start_time, ylim=[None, None])
    # fig5.show()

    print(f'mean f.r.  = {mass_frs.mean(axis=0)}')
    print(f'mean input = {in_frs.mean() / np.array([w[3], -w[4]]) * np.array([b1, b2])}')

    fr_stats = utils.calculate_fr_stats(rasters, model_dic['pop_id«s'], t_start=start_time)

    # ['glomerulus', 'purkinje', 'dcn']
    # Cereb_target = np.array([25.445, 114.332, 46.073])
    Cereb_target = np.array([23.538, 151.228,  43.043])

    # ['GPeTA', 'GPeTI', 'STN', 'SNr']
    BGs_target = np.array([9.30, 38.974, 12.092, 24.402])
    fr_target = np.concatenate((Cereb_target, BGs_target))

    # scale errors according to standard deviation:
    # fr_weights = np.array([1. / 0.4398, 1. / 0.3276, 1. / 0.6918, 1. / 0.4017, 1. / 0.31366, 1 / 0.276, 1 / 0.242])
    fr_weights = np.array([1. / 0.931, 1. / 0.224, 1. / 0.432, 1. / 0.4017, 1. / 0.31366, 1 / 0.276, 1 / 0.242])

    # fr = np.concatenate((fr_stats['fr'][0:5], fr_stats['fr'][6:8]))
    fr = fr_stats['fr']

    # print the fitness
    filter_range = [30, 50]     # [Hz]
    filter_sd = 6               # [Hz]
    fitness = utils.fitness_function(fr, fr_target, mass_frs, sim_period,
                           filter_range=filter_range, filter_sd=filter_sd,
                           t_start=start_time, fr_weights=fr_weights)

    fig6, ax6 = vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['CV'], fr_stats['name'], 
                                          fr_target)
    # fig6.show()
    import matplotlib.pyplot as plt
    
    plt.savefig("./savings/genetic_alg/hist_"+str(idx)+".png")
    # fig7, ax7 = vsl.plot_fourier_transform(mass_frs[:, :], sim_period, ode_names,
    #                                        mean=sum(filter_range) / 2, sd=filter_sd, t_start=start_time)
    # # fig7.show()

    # fig8, ax8 = vsl.plot_wavelet_transform(mass_frs[:, :], sim_period, ode_names,
    #                                        mean=sum(filter_range) / 2, sd=filter_sd, t_start=start_time)
    # fig8.show()

    print(' ')

    return fitness


if __name__ == "__main__":
    fitness_function = fitness_func
    num_generations = 3  # 10
    num_parents_mating = 2  # 4 number of pop to be selected as parents

    sol_per_pop = 3  # 8 pop dimension
    num_genes = 4  # len([b], [c])

    # values for b1, b2, c1, c2
    gene_space = [{'low': 155., 'high': 205.}, {'low': 35., 'high': 105.}, {'low': 62., 'high': 112.}, {'low': 80., 'high': 220.}]

    parent_selection_type = "sss"
    keep_parents = 1

    # crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 50

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           keep_parents=keep_parents,
                           # crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


    ga_instance.plot_fitness()

    ga_instance.save(f'savings/genetic_alg')