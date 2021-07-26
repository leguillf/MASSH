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


# path of configuration file
path_config = 'config_Example_4Dvar_QG.py'

config = exp.exp(path_config)

print('\n ** create State **\n')
State = state.State(config)

print('\n ** create Model **\n')
Model = mod.Model(config,State)

print('\n ** create dict_obs **\n')
# obs dictionnary
dict_obs = obs.obs(config,State)

print('\n** assimilation **\n')
# run the 4Dvar assimilation
ana.ana_4Dvar_QG(config, State, Model, dict_obs=dict_obs)
