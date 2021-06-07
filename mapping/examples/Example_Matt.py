#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:44:54 2021

@author: renamatt
"""

import sys
sys.path.append('..')

from src import exp
from src import state as state
from src import mod as mod
from src import obs as obs
from src import ana as ana
import numpy as np


# path of configuration file
path_config = 'config_Example_Matt_Id.py'

config = exp.exp(path_config)

print('\n ** create State **\n')
State = state.State(config)

print('\n ** create Model **\n')
Model = mod.Model(config,State)


print('\n ** create dict_obs **\n')
# obs dictionnary
dict_obs = obs.obs(config,State)

ana.ana_4Dvar_QG(config, State, Model, dict_obs=dict_obs)

print('\n ** assimilation **\n')
res = ana.window_4D(config, State, Model,dict_obs=dict_obs,date_ini=config.init_date,date_final=config.final_date)
