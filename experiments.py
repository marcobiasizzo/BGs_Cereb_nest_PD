# -*- coding: utf-8 -*-
"""
"""

from marco_nest_utils import utils

__author__ = 'marco'

import numpy as np
from time import time

from nest_multiscale.nest_multiscale import set_poisson_fr


class conditioning():
    def __init__(self, nest_, cereb_class, t_start=300, t_end=400, stimulation=10):
        self.nest_ = nest_

        self.t_start = t_start
        self.t_end = t_end

        self.stimulation = stimulation

        self.rng = np.random.default_rng(round(time() * 1000))

        US = ()
        for ii in range(int(len(cereb_class.Cereb_pops['io']))): # / 2)):  # uncomment to have different IO input in microzones
            US_new = nest_.Create('spike_generator')
            US = US + US_new

        # Connection to first half of IO, corresponding to first microzone
        syn_param = {"model": "static_synapse", "weight": 500.0, "delay": 0.1, "receptor_type": 1}
        nest_.Connect(US, cereb_class.Cereb_pops['io'][:int(len(cereb_class.Cereb_pops['io']))], # / 2)],
                      {'rule': 'one_to_one'}, syn_param)

        self.US = [US]


    def start(self, Sim_time, T_sample):
        self.T = T_sample

    def before_loop(self):
        ...

    def beginning_loop(self, trial_time, total_time):
        if trial_time >= self.t_start and trial_time < self.t_end:
            set_poisson_fr(self.nest_, self.stimulation, self.US, total_time,
                           self.T, self.rng)

    def ending_loop(self, trial_time, total_time):
        ...


