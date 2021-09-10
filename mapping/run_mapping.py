#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:41:43 2021

@author: leguillou
"""

import os,sys,time

from src import exp as exp
from src import state as state
from src import mod as mod
from src import obs as obs
from src import ana as ana

if __name__ == "__main__":
    
    start = time.time()
    
    # check number of arguments
    if len(sys.argv)==2:
        exp_config_file = sys.argv[1]
    elif len(sys.argv)==1:
        os.chdir('examples')
        exp_config_file = 'config_Example_Matt.py'
    elif len(sys.argv)>2:
        sys.exit('Wrong number of argument')
        
    # Experiment config file
    print("* Experimental configuration file")
    config = exp.exp(exp_config_file)
    cmd = 'cp ' + exp_config_file + ' ' + config.tmp_DA_path + '/config.py'
    os.system(cmd)
    
    # Init
    print("* State initialization")
    State = state.State(config)
    
    # Model
    print('* Model Initialization')
    Model = mod.Model(config,State)
        
    # Observations
    print('* Observations')
    dict_obs = obs.obs(config,State)
    
    # Analysis
    print('* Analysis')
    ana.ana(config,State,Model,dict_obs=dict_obs)
    
    end = time.time()
    print('computation time:',end-start,'seconds')
    