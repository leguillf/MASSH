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

from src import state as state
from src import exp as exp
from src import mod as mod
from src import obs as obs
from src import ana as ana

def update_config(config,i):
    name_it = 'iteration_' + str(i) 
    config.tmp_DA_path = '/'.join(config.tmp_DA_path.split('/')[:-1]+[name_it])
    config.path_save = '/'.join(config.path_save.split('/')[:-1]+[name_it])
    if i>0 and config.name_model=='SW1L' and config.name_analysis=='4Dvar':
        name_prev = 'iteration_' + str(i-1) 
        path_tmp_prev = '/'.join(config.tmp_DA_path.split('/')[:-1]+[name_prev])
        config.path_init_4Dvar = os.path.join(path_tmp_prev,'Xini.pic')
    

def compute_new_obs(dict_obs,config):
    # Load maps
    ds = xr.open_mfdataset(
        os.path.join(config.path_save,config.name_exp_save+'*.nc'),
        combine='nested',concat_dim='t')
    if config.name_model=='QG1L':
        name_var = config.name_mod_var[0]
    elif config.name_model=='SW1L':
        name_var = config.name_mod_var[2]
    maps = ds[name_var]
    # Convert map times
    times = [(dt64 - np.datetime64(config.init_date)) / np.timedelta64(1, 's')
            for dt64 in ds['time'].values]
    # Read map  grid
    lon = ds[config.name_mod_lon]
    lat = ds[config.name_mod_lat]
    if len(lon.shape)==3:
        lon = lon[0,:,:].values
        lat = lat[0,:,:].values
    else:
        lon = lon.values
        lat = lat.values
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
        map_grd = maps[i].values
        # Open obs
        path_obs = dict_obs[date]['obs_name']
        sat =  dict_obs[date]['satellite']
        for _sat,_path_obs in zip(sat,path_obs):
            ds = xr.open_dataset(_path_obs)
            dsout = ds.copy()
            ds.close()
            del ds
            if _sat.kind=='fullSSH':
                # No grid interpolation
                dsout[_sat.name_obs_var[0]] -= map_grd 
            elif _sat.kind=='swot_simulator':
                # grid interpolation 
                lon_obs = dsout[_sat.name_obs_lon].values
                lat_obs = dsout[_sat.name_obs_lat].values
                map_obs = interpolate.griddata((lon.ravel(),lat.ravel()),
                                               map_grd.ravel(),
                                               (lon_obs.ravel(),lat_obs.ravel()))
                dsout[_sat.name_obs_var[0]] -=  map_obs.reshape(lon_obs.shape)
            dsout.to_netcdf(_path_obs,engine='scipy')
            dsout.close()
            del dsout
            
def compute_convergence_criteria(config,i):
    path_save_i = config.path_save
    name_it1 = 'iteration_' + str(i-1)
    path_save_i1 = '/'.join(config.path_save.split('/')[:-1]+[name_it1])
    maps_i = xr.open_mfdataset(
        os.path.join(path_save_i,config.name_exp_save+'*.nc'),
        combine='nested',concat_dim='t', engine='h5netcdf')
    maps_i1 = xr.open_mfdataset(
        os.path.join(path_save_i1,config.name_exp_save+'*.nc'),
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
    # Updtade configuration file
    update_config(config1,iteration)
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
    if iteration>0:
        update_config(config2,iteration-1)
        compute_new_obs(dict_obs1,config2)
    # Analysis
    print('* Analysis')
    ana.ana(config1,State1,Model1,dict_obs=dict_obs1)
    # Clean-up
    del State1,Model1,dict_obs1
    
    print('\n\n\
    *****************************************************************\n\
    *****************************************************************\n\
                  2nd Experiment (iteration ' + str(iteration) +')\n\
    *****************************************************************\n\
    *****************************************************************\n')
    # Updtade configuration file
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
    compute_new_obs(dict_obs2,config1)
    # Analysis
    print('* Analysis')
    ana.ana(config2,State2,Model2,dict_obs=dict_obs2)
    # Clean-up
    del State2,Model2,dict_obs2
    
    if iteration>0:
        print('\n\n\
        *****************************************************************\n\
        *****************************************************************\n\
                      Convergence test (iteration ' + str(iteration) +')\n\
        *****************************************************************\n\
        *****************************************************************\n')
        K1 = compute_convergence_criteria(config1,iteration)
        K2 = compute_convergence_criteria(config2,iteration)
        K = (K1+K2)/2
        print('K1=',K1)
        print('K2=',K2)
        print('K=',K)
        with open(path_K,'a') as f:
            f.write(str(K1) + " " + str(K2) + " " + str(K) + '\n') 