#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:44:46 2021

@author: leguillou
"""
import sys, os 
import xarray as xr
import numpy as np
import time 
from  scipy import interpolate 
from datetime import datetime
from glob import glob

from src import state as state
from src import exp as exp
from src import mod as mod
from src import obs as obs
from src import ana as ana

def update_config(config,i):
    name_it = 'iteration_' + str(i) 
    config.tmp_DA_path = '/'.join(config.tmp_DA_path.split('/')[:-1]+[name_it])
    config.path_save = '/'.join(config.path_save.split('/')[:-1]+[name_it])
    if i>0:
        if config.name_model=='QG1L' and config.name_analysis=='BFN':
            # Use first output of previous iteration as initialization 
            name_prev = 'iteration_' + str(i-1) 
            path_output_prev = '/'.join(config.path_save.split('/')[:-1]+[name_prev])
            # First file
            file_init = sorted(glob(os.path.join(path_output_prev,'*.nc')))[0]
            # Update config
            config.name_init = 'from_file'
            config.name_init_grid = file_init
            config.name_init_lon = config.name_mod_lon
            config.name_init_lat = config.name_mod_lat
            config.name_init_var = config.name_mod_var[0]
            
        if config.name_model=='SW1L' and config.name_analysis=='4Dvar':
            # Use converged state from previous iteration as initialization 
            name_prev = 'iteration_' + str(i-1) 
            path_tmp_prev = '/'.join(config.tmp_DA_path.split('/')[:-1]+[name_prev])
            config.path_init_4Dvar = os.path.join(path_tmp_prev,'Xini.pic')
        
    

def compute_new_obs(it,dict_obs,config,State):
    
    # Read grid
    lon = State.lon 
    lat = State.lat
    
    # Maps timestamps
    maps_date = []
    date = config.init_date
    while date<=config.final_date:
        maps_date.append(date)
        date += config.saveoutput_time_step
        
    # For each observation, remove the corresponding estimated map
    for i,date in enumerate(dict_obs):
        # Load corresponding map(s)
        if date in maps_date:
            # Cool: the observation date matches exactly an estimated map
            ssh_now = State.load_output(date=date).ssh
        else:
            # Don't panic: we just have to perform a time interpolation
            date_prev = min(maps_date, key=lambda x: (x<date, abs(x-date)) )
            date_next = min(maps_date, key=lambda x: (x>date, abs(x-date)) )
            ssh_prev = State.load_output(date=date_prev).ssh.values
            ssh_next = State.load_output(date=date_next).ssh.values
            # Time interpolation
            Wprev = 1/abs(date_prev - date).total_seconds()
            Wnext = 1/abs(date_next - date).total_seconds()
            ssh_now = (Wprev*ssh_prev + Wnext*ssh_next)/(Wprev+Wnext)

        # Open obs
        path_obs = dict_obs[date]['obs_name']
        sat =  dict_obs[date]['satellite']
        new_path_obs = []
        for _sat,_path_obs in zip(sat,path_obs):
            ds = xr.open_dataset(_path_obs)
            dsout = ds.copy().load()
            ds.close()
            del ds
            # Load current state
            if _sat.kind=='fullSSH':
                # No grid interpolation
                dsout[_sat.name_obs_var[0]].data -= ssh_now
            elif _sat.kind=='swot_simulator':
                # grid interpolation 
                lon_obs = dsout[_sat.name_obs_lon].values
                lat_obs = dsout[_sat.name_obs_lat].values
                ssh_on_obs = interpolate.griddata((lon.ravel(),lat.ravel()),
                                                ssh_now.ravel(),
                                                (lon_obs.ravel(),lat_obs.ravel()))
                dsout[_sat.name_obs_var[0]].data -=  ssh_on_obs.reshape(lon_obs.shape).data
            # Writing new obs file
            _dir,_name = os.path.split(_path_obs)
            name_iteration = 'iteration_' + str(it) 
            _new_dir = '/'.join(_dir.split('/')[:-1]+[name_iteration])
            _new_path_obs = os.path.join(_new_dir,_name)
            new_path_obs.append(_new_path_obs)
            dsout.to_netcdf(_new_path_obs)
            dsout.close()
            del dsout
        # Update dict_obs
        dict_obs[date]['obs_name'] = new_path_obs
        
            
            
def compute_convergence_criteria(config,State,i):
    
    # Maps timestamps
    maps_date = []
    date = config.init_date
    while date<=config.final_date:
        maps_date.append(date)
        date += config.saveoutput_time_step
    
    # Set State for previous iteration
    name_it_prev = 'iteration_' + str(i-1) 
    path_save_prev = '/'.join(config.path_save.split('/')[:-1]+[name_it_prev])
    State_prev = State.copy()
    State_prev.path_save = path_save_prev
    
    # Compute convergence criteria
    K,c = 0,0
    for i,date in enumerate(maps_date):
        # Load corresponding maps
        ssh_curr = State.load_output(date=date).ssh.values
        ssh_prev = State_prev.load_output(date=date).ssh.values
        # Mask
        mask = (np.isnan(ssh_curr)) | (np.isnan(ssh_prev))
        ssh_curr[mask] = 0
        ssh_prev[mask] = 0
        # Compare maps
        K_t = np.sqrt(np.sum(np.sum(np.square(ssh_curr-ssh_prev)))/ssh_prev.size) /\
            ( np.max(np.max(ssh_prev))-np.min(np.min(ssh_prev)) )    
        if np.isfinite(K_t):
            K += K_t
            c += 1
    K /= c

    return K


if __name__ == "__main__":
    
    start = time.time()
    
    # check number of arguments
    if len(sys.argv)!=5:
        sys.exit('Wrong number of argument')
    # Parsing
    exp_config_file_1 = sys.argv[1]
    exp_config_file_2 = sys.argv[2]
    iteration = int(sys.argv[3])
    path_K = sys.argv[4]
    
    config1 = exp.exp(exp_config_file_1)
    config2 = exp.exp(exp_config_file_2)
    
    print('\n\n\
    *****************************************************************\n\
    *****************************************************************\n\
                  1st Experiment (iteration ' + str(iteration) +')\n\
    *****************************************************************\n\
    *****************************************************************\n')
    time0 = datetime.now()
    # Updtade configuration file
    update_config(config1,iteration)
    # State
    print('* State Initialization')
    State1 = state.State(config1)
    # Model
    print('* Model Initialization')
    Model1 = mod.Model(config1,State1)
    # Observations
    print('* Observations')
    dict_obs1 = obs.obs(config1,State1)
    # Compute new observations taking into account previous estimation
    print('* Compute new observations')
    if iteration>0:
        update_config(config2,iteration-1)
        State2 = state.State(config2)
        compute_new_obs(iteration,dict_obs1,config2,State2)
    # Analysis
    print('* Analysis')
    ana.ana(config1,State1,Model1,dict_obs=dict_obs1)
    # Computational time
    time1 = datetime.now()
    print('1st Experiment (iteration ' + str(iteration) + ' took',
          time1-time0)
    
    print('\n\n\
    *****************************************************************\n\
    *****************************************************************\n\
                  2nd Experiment (iteration ' + str(iteration) +')\n\
    *****************************************************************\n\
    *****************************************************************\n')
    time0 = datetime.now()
    # Updtade configuration file
    print('* State Initialization')
    update_config(config2,iteration)
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
    compute_new_obs(iteration,dict_obs2,config1,State1)
    # Analysis
    print('* Analysis')
    ana.ana(config2,State2,Model2,dict_obs=dict_obs2)
    # Computational time
    time1 = datetime.now()
    print('2nd Experiment (iteration ' + str(iteration) + ' took',
          time1-time0)
    
    if iteration>0:
        print('\n\n\
        *****************************************************************\n\
        *****************************************************************\n\
                      Convergence test (iteration ' + str(iteration) +')\n\
        *****************************************************************\n\
        *****************************************************************\n')
        K1 = compute_convergence_criteria(config1,State1,iteration)
        K2 = compute_convergence_criteria(config2,State2,iteration)
        K = (K1+K2)/2
        print('K1=',K1)
        print('K2=',K2)
        print('K=',K)
        with open(path_K,'a') as f:
            f.write(str(K1) + " " + str(K2) + " " + str(K) + '\n') 
    
    # Clean-up
    del State1,Model1,dict_obs1
    del State2,Model2,dict_obs2
    
