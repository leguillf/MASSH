#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:02:31 2021

@author: leguillou
"""

import sys,time
import xarray as xr
import numpy as np

from src import state as state
from src import exp as exp
from src import mod as mod
from src import obs as obs
from src import ana as ana

import gc

def update_config(config,i,i0=0):
    if i==i0:
        # Init paths
        config.tmp_DA_path +=  'iteration_' + str(i) +'/' 
        config.path_save +=  'iteration_' + str(i) +'/' 
    else:
        if config2.name_model=='SW1L' and config2.name_analysis=='4Dvar':
            config.path_init_4Dvar = config.tmp_DA_path + 'Xini.pic'
        config.tmp_DA_path = config.tmp_DA_path[:-2] + str(i) + "/"
        config.path_save = config.path_save[:-2] + str(i) + "/"
    
def compute_new_obs(dict_obs,config):
    # Load maps
    ds = xr.open_mfdataset(config.path_save+config.name_exp_save+'*.nc',
                           combine='nested',concat_dim='t')
    if config.name_model=='QG1L':
        name_var = config.name_mod_var[0]
    elif config.name_model=='SW1L':
        name_var = config.name_mod_var[2]
    times = [(dt64 - np.datetime64(config.init_date)) / np.timedelta64(1, 's')
            for dt64 in ds['time'].values]
    maps = ds[name_var]
    ds.close()
    del ds
    maps = maps.assign_coords({"t": ('t',times)})
    maps = maps.chunk(chunks=(len(times),1,1))
    # Get observed date and interpolate maps
    times_obs = [(np.datetime64(dt) - np.datetime64(config.init_date))/ np.timedelta64(1, 's')
                 for dt in dict_obs]
    maps = maps.interp(t=times_obs)
    # For each observation, remove the corresponding estimated map
    for i,date in enumerate(dict_obs):
        # Open obs
        path_obs = dict_obs[date]['obs_name']
        sat =  dict_obs[date]['satellite']
        for _sat,_path_obs in zip(sat,path_obs):
            ds = xr.open_dataset(_path_obs)
            if _sat.kind=='fullSSH':
                # No grid interpolation
                dsout = ds.copy()
                ds.close()
                del ds
                dsout[_sat.name_obs_var[0]] -= maps[i,:,:].values
                dsout.to_netcdf(_path_obs,engine='scipy')
                dsout.close()
                del dsout
            
def compute_convergence_criteria(config,i):
    path_save_i = config.path_save
    path_save_i1 = config.path_save[:-2] + str(i-1) + '/'
    maps_i = xr.open_mfdataset(path_save_i+config.name_exp_save+'*.nc',
                               combine='nested',concat_dim='t', engine='h5netcdf')
    maps_i1 = xr.open_mfdataset(path_save_i1+config.name_exp_save+'*.nc',
                               combine='nested',concat_dim='t', engine='h5netcdf')
    K = 0
    nc = 0
    for name in config.name_mod_var:
        for t in range(maps_i.t.size):
            var_i = maps_i[name][t,:,:].values
            var_i1 = maps_i1[name][t,:,:].values
            if ( np.max(np.max(var_i1))-np.min(np.min(var_i1)) )!=0:
                K += np.sqrt(np.sum(np.sum(np.square(var_i-var_i1)))/var_i1.size) /\
                    ( np.max(np.max(var_i1))-np.min(np.min(var_i1)) )
                nc += 1
        
    K /= nc
    
    maps_i.close()
    maps_i1.close()
    del maps_i,maps_i1

    return K

if __name__ == "__main__":
    
    start = time.time()
    
    # check number of arguments
    if  len(sys.argv)!=3:
        sys.exit('Wrong number of argument')
    # Experiment config file
    print("* Experimental configuration files")
    exp_config_file_1 = sys.argv[1]
    exp_config_file_2 = sys.argv[2]
    
    config1 = exp.exp(exp_config_file_1)
    config2 = exp.exp(exp_config_file_2)
    

    K = np.inf
    i0 = 0
    i = i0
    while K>1e-3:
        print('\n\n\
        *****************************************************************\n\
        *****************************************************************\n\
                      1st Experiment (iteration ' + str(i+1) +')\n\
        *****************************************************************\n\
        *****************************************************************\n')
        # Updtade configuration file
        update_config(config1,i,i0)
        # State
        State1 = state.State(config1)
        # Model
        print('* Model Initialization')
        Model1 = mod.Model(config1,State1)
        # Observations
        print('* Observations')
        dict_obs1 = obs.obs(config1,State1)
        # Compute new observations taking into account previous estimation
        print('* Compute new observations')
        if i>0:
            compute_new_obs(dict_obs1,config2)
        # Analysis
        print('* Analysis')
        ana.ana(config1,State1,Model1,dict_obs=dict_obs1)
        # Clean-up
        del State1,Model1,dict_obs1
        gc.collect()
        
        print('\n\n\
        *****************************************************************\n\
        *****************************************************************\n\
                      2nd Experiment (iteration ' + str(i+1) +')\n\
        *****************************************************************\n\
        *****************************************************************\n')
        # Updtade configuration file
        update_config(config2,i,i0)
        # State
        State2 = state.State(config2)
        # Model
        print('* Model Initialization')
        Model2 = mod.Model(config2,State2)
        # Observations
        print('* Observations')
        dict_obs2 = obs.obs(config2,State2)
        # Compute new observations taking into account previous estimation
        print('* Compute new observations')
        compute_new_obs(dict_obs2,config1)
        # Analysis
        print('* Analysis')
        ana.ana(config2,State2,Model2,dict_obs=dict_obs2)
        # Clean-up
        del State2,Model2,dict_obs2
        gc.collect()
        
        print('\n\n\
        *****************************************************************\n\
        *****************************************************************\n\
                      Convergence test (iteration ' + str(i+1) +')\n\
        *****************************************************************\n\
        *****************************************************************\n')
        if i>0:
            K1 = compute_convergence_criteria(config1,i)
            K2 = compute_convergence_criteria(config2,i)
            K = (K1+K2)/2
            print('K1=',K1)
            print('K2=',K2)
            print('K=',K)
    
        i += 1
        
        
    
