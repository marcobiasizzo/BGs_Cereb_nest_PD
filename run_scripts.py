import subprocess

trials = 5
exp = 1
for idx in range(trials):
    for mod in [1]: #range(3):
        # for dopa in [0,3,4]: #range(5):
        for dopa in range(5):
            program = [f'./main.py {str(idx + 1)} {str(mod)} {str(exp)} {str(dopa)}']

            subprocess.call(program, shell=True)    # , capture_output=True)
            print("Finished:" + program[0])


            # dopa_depl_level_list = [0.,-0.1,-0.2,-0.4,-0.8]     # between 0. and -0.8

            # mode_list = ['external_dopa', 'internal_dopa', 'both_dopa']     # external = only BGs dopa depl, internal = only Cereb dopa depl
            # experiment_list = ['active', 'EBCC']
            # mode_i = mod
            # experiment_i = exp
            # dopa_depl_level_i = dopa
            # mode = mode_list[mode_i]                 # dopa depl location
            # experiment = experiment_list[experiment_i]     # cortical activation or EBCC
            # dopa_depl_level = dopa_depl_level_list[dopa_depl_level_i]      # between 0. and -0.8
            # sim_time = 3000
            
            # sol_n = 18
            # savings_dir = f'shared_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}'  # f'savings/{date_time}'
            # # savings_dir = f'shared_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}_test{key}_all_plast_ctx'  # f'savings/{date_time}'
            
            # if dopa_depl_level != 0.:
            #     dopa_depl = True
            # else:
            #     dopa_depl = False

            # if dopa_depl: savings_dir = savings_dir + f'_dopadepl_{(str(int(-dopa_depl_level*10)))}'
            # # if load_from_file: savings_dir += '_trial_1'

            # savings_dir = savings_dir + f'_trial_{idx}_test'

            # print(savings_dir)

            
