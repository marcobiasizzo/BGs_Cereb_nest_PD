# -*- coding: utf-8 -*-
"""
"""

from marco_nest_utils import utils

__author__ = 'marco'

import numpy as np
from time import time

from nest_multiscale.nest_multiscale import set_poisson_fr

class conditioning():
    def __init__(self, nest_, cereb_class, t_start_MF=100, t_start_IO=350, t_end=380, stimulation_IO=500, resolution=0.1, t_sim= 1000., tot_trials = 1):
        self.t_start_MF = t_start_MF
        self.t_start_IO = t_start_IO
        self.t_end = t_end
        self.nest_ = nest_
        self.stimulation_IO = stimulation_IO
        self.n_wind = cereb_class.n_wind

        self.resolution = resolution

        self.rng = np.random.default_rng(round(time() * 1000))

        # US = ()
        # for ii in range(int(len(cereb_class.Cereb_pops['io']))): # / 2)):  # uncomment to have different IO input in microzones
        #     US_new = nest_.Create('spike_generator')
        #     US = US + US_new

        # # Connection to first half of IO, corresponding to first microzone
        # syn_param = {"model": "static_synapse", "weight": 500.0, "receptor_type": 1}
        # nest_.Connect(US, cereb_class.Cereb_pops['io'][:int(len(cereb_class.Cereb_pops['io']))], # / 2)],
        #               {'rule': 'one_to_one'}, syn_param)

        # self.US = [US]

        # IO_id = cereb_class.Cereb_pops['io']
        # glom_id, _ = cereb_class.get_glom_indexes(cereb_class.Cereb_pops['glomerulus'], "EBCC")
        # id_stim = sorted(list(set(glom_id)))

        # US_matrix = np.concatenate(
        #                 [
        #                     np.arange(t_start_IO, t_end + 2, 2)
        #                     + t_sim * t
        #                     for t in range(tot_trials)
        #                 ]
        #             )
        
        # US_stim = nest_.Create("spike_generator", len(IO_id), {"spike_times":US_matrix})
        
        # nest_.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.}) #10.

        # self.US = US_stim

        # n_mf = 25
        # self.bins = int((t_end - t_start_MF)/n_mf)

        # n_glom_x_mf = len(glom_id)/n_mf
        # splits = [int(n_glom_x_mf)*i for i in range(1,n_mf+1)]
        # glom_mf = np.split(np.asarray(glom_id),splits)
        # self.map_glom = {}
        # self.CS_stim = nest_.Create("spike_generator", n_mf)
        # for sg in range(len(self.CS_stim)):	
        #         #nest.SetStatus(CS_stim[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()})
        #         nest_.Connect(self.CS_stim[sg : sg + 1], glom_mf[sg].tolist())
            

            

    def start(self, sim_handler, Sim_time, T_sample):
        self.T = T_sample

    def before_loop(self, sim_handler):
        ...

    def beginning_loop(self, sim_handler, trial_time, total_time, in_spikes):
        if trial_time >= self.t_start_IO and trial_time < self.t_end:
            set_poisson_fr(sim_handler.nest, self.stimulation_IO, self.US, total_time,
                           self.T, self.rng, self.resolution, sin_weight=1., in_spikes=in_spikes, n_wind=sim_handler.n_wind)
            # set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[0]], total_time + self.T,
            #                self.T, sim_handler.rng, self.resolution)
            # self.define_input_glom(trial_time, sim_handler.yT[0])
            # sg = trial_time/self.bins
            set_poisson_fr(sim_handler.nest, 0., sim_handler.pop_list_to_nest[:self.n_wind], total_time + self.T,
                           self.T, sim_handler.rng, self.resolution, in_spikes, n_wind=sim_handler.n_wind)
            # set_poisson_fr(sim_handler.nest, sim_handler.yT[0], self.CS_stim[sg : sg + 1], total_time + self.T,
            #                self.T, sim_handler.rng, self.resolution)
            
    # def CS(self, sim_handler, yT, trial_time, total_time):
    #     n_mf = 2
    #     bins = int((self.t_end - self.t_start_MF)/n_mf) -1
    #     sg = int((trial_time - self.t_start_MF)/bins)
    #     sg = int(len(sim_handler.CS_stim["CTX"])/2)
    #     if trial_time > self.t_start_MF and trial_time < self.t_start_MF + (self.t_end-self.t_start_MF)/2:
    #         # set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[0]], total_time + self.T,
    #         #                self.T, sim_handler.rng, self.resolution)
    #         # self.define_input_glom(trial_time, sim_handler.yT[0])
            
    #         # set_poisson_fr(sim_handler.nest, yT[0], [sim_handler.CS_stim["CTX"][ : sg]], total_time + self.T,
    #         #                self.T, sim_handler.rng, self.resolution, in_spikes = "EBCC")
    #         set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[0]], total_time + self.T,
    #                        self.T, sim_handler.rng, self.resolution, in_spikes = "EBCC")
        
    #     if trial_time >= self.t_start_MF + (self.t_end-self.t_start_MF)/2 and trial_time<self.t_end:
    #         # set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[0]], total_time + self.T,
    #         #                self.T, sim_handler.rng, self.resolution)
    #         # self.define_input_glom(trial_time, sim_handler.yT[0])
            
    #         set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[1]], total_time + self.T,
    #                        self.T, sim_handler.rng, self.resolution, in_spikes = "EBCC")
    def CS(self, sim_handler, yT, trial_time, total_time):
        
        dt = int((self.t_end - self.t_start_MF)/self.n_wind)
        CS_dict = {}
        for i in range(self.n_wind):
            t_start = self.t_start_MF + i*dt
            CS_dict[t_start] = i
        for i_t in CS_dict.keys():
            i_wind = CS_dict[i_t]
            if trial_time >= i_t and trial_time < i_t+dt:
                pops = [ip for ip in range(self.n_wind)]
                pops.remove(i_wind)
                id_gen = [sim_handler.pop_list_to_nest[id_gen_i] for id_gen_i in pops]
                # id_gen = id_gen.remove(sim_handler.pop_list_to_nest[i_wind])
                set_poisson_fr(sim_handler.nest, 0., id_gen, total_time + self.T,
                            self.T, sim_handler.rng, self.resolution, in_spikes = "EBCC", n_wind=sim_handler.n_wind)
           
            
        
    def ending_loop(self, sim_handler, trial_time, total_time, in_spikes="active"):
        if trial_time < self.t_start_MF or trial_time >= self.t_end:
            set_poisson_fr(sim_handler.nest, [0., 0.], sim_handler.pop_list_to_nest[:self.n_wind], total_time + self.T,
                           self.T, sim_handler.rng, self.resolution, in_spikes=in_spikes, n_wind=sim_handler.n_wind)
            
    def define_input_glom(self, trial_time,fr):
        n_spk = 1000/fr*self.bins
        resto = trial_time%self.bins
        spk = np.arange(trial_time-resto, trial_time-resto +self.bins ,n_spk)
        sg = trial_time/self.bins
        self.nest_.SetStatus(self.CS_stim[sg : sg + 1], params={"spike_times": spk})

