# -*- coding: utf-8 -*-
"""
"""

from marco_nest_utils import utils

__author__ = 'marco'

import numpy as np
from time import time

from nest_multiscale.nest_multiscale import set_poisson_fr


class conditioning():
    def __init__(self, nest_, cereb_class, t_start_MF=300, t_start_IO=400, t_end=450, stimulation_IO=500, resolution=0.1):
        self.t_start_MF = t_start_MF
        self.t_start_IO = t_start_IO
        self.t_end = t_end

        self.stimulation_IO = stimulation_IO

        self.resolution = resolution

        self.rng = np.random.default_rng(round(time() * 1000))

        US = ()
        for ii in range(int(len(cereb_class.Cereb_pops['io']))): # / 2)):  # uncomment to have different IO input in microzones
            US_new = nest_.Create('spike_generator')
            US = US + US_new

        # Connection to first half of IO, corresponding to first microzone
        syn_param = {"model": "static_synapse", "weight": 500.0, "receptor_type": 1}
        nest_.Connect(US, cereb_class.Cereb_pops['io'][:int(len(cereb_class.Cereb_pops['io']))], # / 2)],
                      {'rule': 'one_to_one'}, syn_param)

        self.US = [US]


    def start(self, sim_handler, Sim_time, T_sample):
        self.T = T_sample

    def before_loop(self, sim_handler):
        ...

    def beginning_loop(self, sim_handler, trial_time, total_time):
        if trial_time >= self.t_start_IO and trial_time < self.t_end:
            set_poisson_fr(sim_handler.nest, self.stimulation_IO, self.US, total_time,
                           self.T, self.rng, self.resolution)

    def ending_loop(self, sim_handler, trial_time, total_time):
        if trial_time < self.t_start_MF or trial_time >= self.t_end:
            set_poisson_fr(sim_handler.nest, 0., [sim_handler.pop_list_to_nest[0]], total_time + self.T,
                           self.T, sim_handler.rng, self.resolution)


