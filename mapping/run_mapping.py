#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:41:43 2021

@author: leguillou
"""

import sys,os,time

def run_bfn(config):
    
    return

if __name__ == "__main__":
    
    start = time.time()
    
    # check number of arguments
    if  len(sys.argv)!=2:
        sys.exit('Wrong number of argument')
    # Experiment config file
    print("* Experimental configuration file")
    exp_config_file = sys.argv[1]
    _dir,_config = os.path.split(os.path.abspath(sys.argv[1]))
    sys.path.append(_dir)
    if _config[-3:]=='.py':
        _config = _config[:-3]
    config_exp = __import__(_config)
    # Merge with default config file
    from src import config_default as config
    config.__dict__.update(config_exp.__dict__)
    # Temporary directory
    print("* Temporary directory")
    if not os.path.exists(config.tmp_DA_path):
        os.makedirs(config.tmp_DA_path)
    print(config.tmp_DA_path)
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
    