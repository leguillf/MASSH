#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:44:46 2021

@author: leguillou
"""
import sys, os 
import argparse
import xarray as xr
import numpy as np
import time 
from  scipy import interpolate 
from datetime import datetime
from glob import glob
import pickle


from src import state as state
from src import exp as exp
from src import mod as mod
from src import obs as obs
from src import ana as ana


def update_config(config,i,params=None):
    """
    NAME
        update_config

    DESCRIPTION
        Update experimental config file for ith iteration 
        of the alternation minimization algorithm

        Args:
            config : configuration module specific to one experiment
            i : iteration number
            params : list of parameters that are iterated  

    """
    
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
        
    # Iterable parameter update
    if params is not None:
        for param in params:
            try:
                print(param,'==',config[param][i])
                config[param] = config[param][i]
            except:
                try:
                    print(param,'==',config[param][-1])
                    config[param] = config[param][-1]
                    print('Warning: maximum number of iteration reached for updating '\
                          +param+' parameter')
                    print('We keep the last prescribed value')
                except:
                    sys.exit('Impossible to update parameter '+param)
                        
                        

def get_dict_obs(config,State):
    date1 = config.init_date.strftime('%Y%m%d')
    date2 = config.final_date.strftime('%Y%m%d')
    box = f'{int(State.lon.min())}_{int(State.lon.max())}_{int(State.lat.min())}_{int(State.lat.max())}'
    name_dict_obs = os.path.join(config.path_obs,
                                 f'dict_obs_{"_".join(config.satellite)}_{date1}_{date2}_{box}.pic')
    if not os.path.exists(name_dict_obs):
        dict_obs = obs.obs(config,State)
        # Save obs for next iterations
        for date in dict_obs:
            for obs_name,sat in zip(dict_obs[date]['obs_name'],dict_obs[date]['satellite']):
                file_obs = os.path.basename(obs_name)
                new_obs_name = os.path.join(config.path_obs,file_obs)
                # Copy to *tmp_DA_path* directory
                os.system(f'cp {obs_name} {new_obs_name}')
    else:
                
        with open(name_dict_obs, 'rb') as f:
            
            dict_obs0 = pickle.load(f)
            dict_obs = {}
            
            for date in dict_obs0:
                # Create new dict_obs by copying the obs files in tmp_DA_dir directory 
                dict_obs[date] = {'obs_name':[],'satellite':[]}
                for obs_name,sat in zip(dict_obs0[date]['obs_name'],dict_obs0[date]['satellite']):
                    file_obs = os.path.basename(obs_name)
                    first_obs_name = os.path.join(config.path_obs,file_obs)
                    new_obs_name = os.path.join(config.tmp_DA_path,file_obs)
                    # Copy to *tmp_DA_path* directory
                    os.system(f'cp {first_obs_name} {new_obs_name}')
                    # Update new dictionary 
                    dict_obs[date]['obs_name'].append(new_obs_name)
                    dict_obs[date]['satellite'].append(sat)
                
    return dict_obs
    


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
    name_ssh = State.name_var[State.get_indobs()]
    for i,date in enumerate(dict_obs):
        # Load corresponding map(s)
        if date in maps_date:
            # Cool: the observation date matches exactly an estimated map
            ssh_now = State.load_output(date=date)[name_ssh].values
        else:
            # Don't panic: we just have to perform a time interpolation
            date_prev = min(maps_date, key=lambda x: (x<date, abs(x-date)) )
            date_next = min(maps_date, key=lambda x: (x>date, abs(x-date)) )
            ssh_prev = State.load_output(date=date_prev)[name_ssh].values
            ssh_next = State.load_output(date=date_next)[name_ssh].values
            # Time interpolation
            Wprev = 1/abs(date_prev - date).total_seconds()
            Wnext = 1/abs(date_next - date).total_seconds()
            ssh_now = (Wprev*ssh_prev + Wnext*ssh_next)/(Wprev+Wnext)
        
        # Open obs
        path_obs = dict_obs[date]['obs_name']
        sat = dict_obs[date]['satellite']
        for _sat,_path_obs in zip(sat,path_obs):
	    # Current obs dataset 
            ds = xr.open_dataset(_path_obs)
            ssh_obs0 = ds[_sat.name_obs_var[0]].values.squeeze()
            # New obs dataset
            dsout = ds.copy().load()
            ds.close()
            del ds
            # Compute new obs by removing estimated map from previous run 
            if _sat.kind=='fullSSH':
                # No grid interpolation
                dsout[_sat.name_obs_var[0]].data = ssh_obs0 - ssh_now
            elif _sat.kind=='swot_simulator':
                # grid interpolation 
                lon_obs = dsout[_sat.name_obs_lon].values
                lat_obs = dsout[_sat.name_obs_lat].values
                ssh_on_obs = interpolate.griddata((lon.ravel(),lat.ravel()),
                                                   ssh_now.ravel(),
                                                  (lon_obs.ravel(),lat_obs.ravel()))
                dsout[_sat.name_obs_var[0]].data = ssh_obs0 - ssh_on_obs.reshape(lon_obs.shape)
                
            # Writing new obs file
            dsout.to_netcdf(_path_obs)
            dsout.close()
            del dsout 
        
            
            
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
    name_ssh = State.name_var[State.get_indobs()]
    K,c = 0,0
    for i,date in enumerate(maps_date):
        # Load corresponding maps
        ssh_curr = State.load_output(date=date)[name_ssh].values
        ssh_prev = State_prev.load_output(date=date)[name_ssh].values
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
    
    # Parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--c1',type=str) # Path of 1st config file
    parser.add_argument('--c2',type=str) # Path of 2nd config file  
    parser.add_argument('--i', type=int) # iteration number
    parser.add_argument('--K', type=str) # path of convergence file
    parser.add_argument('--params1', type=str, default=None) # names of iterable parameters for 1st experiment
    parser.add_argument('--params2', type=str, default=None) # names of iterable parameters for 2nd experiment
    
    opts = parser.parse_args()
    
    exp_config_file_1 = opts.c1
    exp_config_file_2 = opts.c2
    config1 = exp.exp(exp_config_file_1)
    config2 = exp.exp(exp_config_file_2)
    iteration = opts.i
    path_K = opts.K
    params1 = opts.params1
    if params1 is not None:
        params1 = params1.split(' ')
    params2 = opts.params2
    if params2 is not None:
        params2 = params2.split(' ')
    
    
    print('\n\n\
    *****************************************************************\n\
    *****************************************************************\n\
                  1st Experiment (iteration ' + str(iteration) +')\n\
    *****************************************************************\n\
    *****************************************************************\n')
    time0 = datetime.now()
    # Updtade configuration file
    print('* Updtade configuration file')
    update_config(config1,iteration,params=params1)
    # State
    print('* State Initialization')
    State1 = state.State(config1)
    # Model
    print('* Model Initialization')
    Model1 = mod.Model(config1,State1)
    # Observations
    print('* Observations')
    dict_obs1 = get_dict_obs(config1,State1)
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
    update_config(config2,iteration,params=params2)
    # State
    State2 = state.State(config2)
    # Model
    print('* Model Initialization')
    Model2 = mod.Model(config2,State2)
    # Observations
    print('* Observations')
    dict_obs2 = get_dict_obs(config2,State2)
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
    
