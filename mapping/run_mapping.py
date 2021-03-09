#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:41:43 2021

@author: leguillou
"""

import sys,time

if __name__ == "__main__":
    
    start = time.time()
    
    # check number of arguments
    if  len(sys.argv)!=2:
        sys.exit('Wrong number of argument')
    # Experiment config file
    print("* Experimental configuration file")
    exp_config_file = sys.argv[1]
    from src import exp as exp
    config = exp.exp(exp_config_file)
    # Init
    print("* State initialization")
    from src import state as state
    State = state.State(config)
    # Model
    print('* Model Initialization')
    from src import mod as mod
    Model = mod.Model(config,State)
    # Observations
    print('* Observations')
    from src import obs as obs
    dict_obs = obs.obs(config,State)
    # Analysis
    print('* Analysis')
    from src import ana as ana
    ana.ana(config,State,Model,dict_obs=dict_obs)
    
    end = time.time()
    print('computation time:',end-start,'seconds')
    