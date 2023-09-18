#%%
from Cereb_nest.Cereb import Cereb_class
from nest_multiscale.nest_multiscale import sim_handler, generate_ode_dictionary

import sys
sys.path.append('/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/')
import nest
from pathlib import Path
from marco_nest_utils import utils
import pickle as p
import numpy as np
CORES = 24
VIRTUAL_CORES = 24
RESOLUTION = 1.
run_on_vm = False

#%%

# set number of kernels
nest.ResetKernel()
nest.SetKernelStatus({"total_num_virtual_procs": CORES, "resolution": RESOLUTION})
nest.set_verbosity("M_ERROR")  # reduce plotted info
MODULE_PATH = str(Path.home()) + '/nest/lib/nest/ml_module'
nest.Install(MODULE_PATH)  # Import my_BGs module
MODULE_PATH = str(Path.home()) + '/nest/lib/nest/cerebmodule'
nest.Install(MODULE_PATH)  # Import CerebNEST

#%%
path = "/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/shared_results/"
name = "complete_1280ms_x_20_sol17_both_dopa_EBCC_trial_13"
# path = "/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/cereb_test"

# name ="complete_580ms_x_101_sol17_both_dopa_EBCC_test34_dcn_io"
name ="complete_580ms_x_1_sol17_both_dopa_EBCC_test11_ctx_diff_pc"
name1=''
f = open(path + name + "/rasters"+name1,"rb")
rster = p.load(f)
f.close()

f = open(path + name + "/model_dic"+name1,"rb")
model = p.load(f)
f.close()

i=0
ids = rster[i]["neurons_idx"].reshape(-1,1)
times = rster[i]["times"].reshape(-1,1)
spk = np.concatenate((ids, times), axis=1)

#%%
neurons = np.unique(spk[:,0])


#%%

hdf5_file_name = "Cereb_nest/scaffold_full_IO_400.0x400.0_microzone.hdf5"
Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn','dcnp', 'io']

# CS ={"start":100., "stop":380., "freq":50.}
# US ={"start":350., "stop":380., "freq":500.}

tot_trials = 101
baseline = 200
len_trial = 380. + baseline
set_time = 0

# ltd = np.logspace(-4,0,base=2,num=5)
# ltp = np.logspace(-10,-6,base=2,num=5)

# ltp = [0.000022]
# ltd = [0.00008]
# i=0
# tests_dict = {0: [-0.00001, 0.000009],
#               1: [-0.000022, 0.00008],
#                 }
# tests_dict = {2: [-0.00001, 0.000011],
#               3: [-0.000012, 0.00001],
#               4: [-0.00001, 0.000012]
#                 }
tests_dict = {5: [-0.00001, 0.000013],
              6: [-0.00001, 0.000014],
              7: [-0.000009, 0.000012]
                }
for key in tests_dict:
    LTP = tests_dict[key][1]
    LTD = tests_dict[key][0]
    i +=1

    nest.ResetKernel()
    cereb = Cereb_class(nest, hdf5_file_name, n_spike_generators='n_glomeruli',
                mode='external_dopa', experiment='EBCC', dopa_depl=0, LTD=LTD, LTP=LTP)

    #ct = cereb.create_ctxinput(nest, pos_file=hdf5_file_name, in_spikes="background")
    
    for neu in neurons:
        CS_spk = spk[spk[:,0]==neu][:,1]
        CS_matrix = np.concatenate(
                    [
                        CS_spk
                        + len_trial * t
                        for t in range(tot_trials)
                    ]
                )
        CS_stim = nest.Create("spike_generator", 1, {"spike_times":CS_matrix})
        nest.Connect(CS_stim, [int(neu)], "one_to_one")

    IO_id = cereb.Cereb_pops['io']
    US_matrix = np.concatenate(
                    [
                        np.arange(350., 380. + 3, 3)
                        + len_trial * t
                        for t in range(tot_trials)
                    ]
                )
    
    US_stim = nest.Create("spike_generator", len(IO_id), {"spike_times":US_matrix})
    
    nest.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.}) #10.





    recorded_list = [cereb.Cereb_pops[name] for name in Cereb_recorded_names]
    sd_list = utils.attach_spikedetector(nest, recorded_list)
    
    model_dict = utils.create_model_dictionary(0, Cereb_recorded_names, {**cereb.Cereb_pop_ids}, len_trial,
                                                sample_time=1., settling_time=set_time,
                                                trials=tot_trials, b_c_params=[])
    

    print("Simulating settling time: " + str(set_time) )

    #nest.Simulate(set_time)

    
    for trial in range(tot_trials):
        
        '''
        # CS_spk = np.around(np.linspace(CS["start"]+ set_time +(trial*len_trial),CS["stop"]+ set_time +(trial*len_trial),22), decimals=1)
        # CS_stim = nest.Create("spike_generator", len(glom_id), {"spike_times":CS_spk})

        # CS_stim = nest.Create("poisson_generator", len(glom_id), {"start":500.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":36.})
        #nest.Connect(CS_stim, glom_id, "one_to_one")
        '''

        '''
        # US_spk = np.around(np.linspace(US["start"]+ set_time +(trial*len_trial),US["stop"]+ set_time +(trial*len_trial),int(US["freq"]*1000/(US["stop"]-US["start"]))), decimals=1)
        # US_stim = nest.Create("spike_generator", len(IO_id), {"spike_times":US_spk})
        
        # US_stim = nest.Create("poisson_generator", len(IO_id), {"start":750.+(trial*len_trial), "stop":760.+(trial*len_trial), "rate":200.})
        nest.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.})
        '''
        
        print("Simulating trial: " + str(trial +1) +" di "+ str(tot_trials))
        
        print(key)
        nest.Simulate(len_trial)

        
    rasters = utils.get_spike_values(nest, sd_list, Cereb_recorded_names)
    with open(f'./learning_test/rasters_'+str(tot_trials)+'_test_'+str(key), 'wb') as pickle_file:
        p.dump(rasters, pickle_file)


    with open(f'./learning_test/model_dict_'+str(tot_trials)+'_test_'+str(key), 'wb') as pickle_file:
        p.dump(model_dict, pickle_file)
# %%
