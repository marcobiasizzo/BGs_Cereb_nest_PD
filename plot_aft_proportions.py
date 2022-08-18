# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import close

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

settling_time = 1000.
sim_time = 3000.
start_time = 0.  # starting time for histograms data
sim_period = 1.  # ms
trials = 1

t_start = 300
t_end = 400

N_BGs = 20000
N_Cereb = 96767
load_from_file = True  # load results from directory or simulate and save
sol_n = 17

mode_list = ['external_dopa', 'internal_dopa', 'both_dopa']
experiment_list = ['active', 'EBCC']
mode = mode_list[2]
experiment = experiment_list[0]

# set saving directory
# date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
savings_dir = f'shared_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_external_dopa_{experiment}'  # f'savings/{date_time}'
saving_dir_list = []
savings_dir = f'shared_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}'  # f'savings/{date_time}'
for dopa_depl_level in [-0.1, -0.2]:    # , -0.4, -0.8]:
    saving_dir_list += [savings_dir + f'_dopadepl_{(str(int(-dopa_depl_level*10)))}']

''' Set up multi-scale simulation: order is important| '''

# Define all the NEST populations:
Cereb_pop_names = ['golgi', 'glomerulus', 'granule', 'purkinje', 'basket', 'stellate', 'dcn', 'dcnp', 'io']
BGs_pop_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI', 'STN', 'SNr']
# Select the NEST populations you will attach to a spike detector:
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
b1 = w[3] / b_c_params[0]       # DCN -> Thal  # 2900
b2 = -w[4] / b_c_params[1]      # SNr -> Thal
c1 = b_c_params[2]              # CTX -> MF
c5 = b_c_params[3]              # CTX -> STN
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

average_fr_per_trial_list = []
average_fr_sd_per_trial_list = []

if __name__ == "__main__":
    for sd in saving_dir_list:
        rasters_list = []
        for trial_idx in range(1, 3):
            sdt = sd + f'_trial_{trial_idx}'
            print(f'Simulation results loaded from {sdt}')

            with open(f'{sdt}/model_dic', 'rb') as pickle_file:
                model_dic = pickle.load(pickle_file)
            with open(f'{sdt}/potentials', 'rb') as pickle_file:
                potentials = pickle.load(pickle_file)
            with open(f'{sdt}/rasters', 'rb') as pickle_file:
                rasters = pickle.load(pickle_file)
            with open(f'{sdt}/mass_models_sol', 'rb') as pickle_file:
                mass_frs = pickle.load(pickle_file)

            TARGET_POP = 'glomerulus'

            # instant_fr = utils.fr_window_step(rasters, model_dic['pop_ids'], pre_sim_time + sim_time*trials, window=10., step=5.)
            # io_idx = [i for i, n in enumerate(recorded_names) if n == TARGET_POP]
            # io_fr_list += [instant_fr[io_idx[0]]]

            rasters_list += [rasters]

        fr_stats = utils.calculate_fr_stats(rasters_list, model_dic['pop_ids'], t_start=start_time, multiple_trials=True)
        average_fr_per_trial_list += [fr_stats['fr']]
        average_fr_sd_per_trial_list += [fr_stats['fr_sd']]

    titles_list = {'external_dopa': 'BGs dopa depl', 'internal_dopa': 'Cereb dopa depl',
                   'both_dopa': 'BGs & Cereb dopa depl'}

    ######### figure  ############
    fig_width = 8.0
    plot_height = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))

    # name_list = utils.calculate_fr_stats(rasters0, model_dic['pop_ids'], t_start=start_time)['name']
    name_list = ['Glomerulus', 'Purkinje', 'DCNp', 'GPeTA', 'GPeTI', 'STN'] # , 'SNr']

    width = 0.3  # columns width
    x = np.array(list(range(len(average_fr_per_trial_list[0])))) * 1.5
    ax.set_xticks(x)
    x1 = x - width * 1.5
    # x2 = x - width / 2
    # x4 = x + width / 2
    # x8 = x + width * 1.5

    width = width * 0.9
    alpha = 0.9

    relative_fr = []
    for fr in average_fr_per_trial_list[1:]:
        relative_fr += [(fr - average_fr_per_trial_list[0]) / average_fr_per_trial_list[0]]

    bars1 = ax.bar(x1, relative_fr[0], width, alpha=alpha * 0.4)  # color=cmap.to_rgba(2))  # , color='tab:red')
    # bars2 = ax.bar(x2, relative_fr[1], width, alpha=alpha * 0.6)  # color=cmap.to_rgba(2))  # , color='tab:red')
    # bars4 = ax.bar(x4, relative_fr[2], width, alpha=alpha * 0.8)  # color=cmap.to_rgba(2))  # , color='tab:red')
    # bars8 = ax.bar(x8, relative_fr[3], width, alpha=alpha * 1.0)  # color=cmap.to_rgba(2))  # , color='tab:red')

    bars_null = ax.bar(x, -0.003 * np.sign(relative_fr[0]), width, alpha=0., color='white')
    ax.bar_label(bars_null, name_list)

    ax.axhline(0., color='black')

    ax.legend(title="Dopamine depletion level")
    ax.set_ylabel('Relative variation')
    ax.set_title(f'Relative variation between parkinsonian and healthy average firing rates')  # in mode: {mode}')

    ax.tick_params(bottom=False, labelbottom=False)

    fig.tight_layout()

    fig.show()
