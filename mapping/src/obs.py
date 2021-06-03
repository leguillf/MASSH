#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:15:09 2021

@author: leguillou
"""
import os 
import pickle
import xarray as xr
import numpy as np
from datetime import datetime
from scipy import interpolate
import pandas as pd
import glob 
import matplotlib.pylab as plt 

from .sat import read_satellite_info
from .tools import detrendn

def obs(config, State, *args, **kwargs):
    """
    NAME
        obs

    DESCRIPTION
        Main function calling subfunctions considering the kind of satellite observations
        Args:
            config (module): configuration module
            State (class): class of model state

        Param:

        Returns:
            dict_obs (dictionary): the keys are the dates of observations, and the values are dictionaries gathering all information
            needed to assimilate these observations
    """
    
    # Check if previous *dict_obs* has been computed
    dir_obs = config.path_obs
    if config.path_obs is None:
        dir_obs = config.tmp_DA_path
    if config.write_obs and os.path.exists(os.path.join(dir_obs,'dict_obs.pic')):
        print('Reading *dict_obs* from previous run')
        with open(os.path.join(dir_obs,'dict_obs.pic'), 'rb') as f:
            dict_obs = pickle.load(f)
            return dict_obs
        
    # Read grid
    lon = State.lon
    lat = State.lat
    bbox = [lon.min(),lon.max(),lat.min(),lat.max()]
    
    # Open MDT map if provided
    if config.path_mdt is not None and os.path.exists(config.path_mdt):
        ds = xr.open_dataset(config.path_mdt)
        ds[config.name_var_mdt['lon']] = ds[config.name_var_mdt['lon']] % 360
        if ds[config.name_var_mdt['var']].shape[0]!=ds[config.name_var_mdt['lat']].shape[0]:
            ds[config.name_var_mdt['var']] = ds[config.name_var_mdt['var']].transpose()
        ds = ds.sel({config.name_var_mdt['lon']:slice(lon.min(),lon.max()),
                     config.name_var_mdt['lat']:slice(lat.min(),lat.max())})
        if len(ds[config.name_var_mdt['lon']].shape)==1:
            
            lon_mdt,lat_mdt = np.meshgrid(ds[config.name_var_mdt['lon']].values,
                                          ds[config.name_var_mdt['lat']].values)
            
        mdt = ds[config.name_var_mdt['var']].values
        
    else:
        lon_mdt = lat_mdt = mdt = None
                                      
    
    # Compute output observation dictionnary
    dict_obs = {}
    
    if config.satellite is None:
        print('None observation has been provided')
        return dict_obs
    
    assim_dates = []
    date = config.init_date
    while date<=config.final_date:
        assim_dates.append(date)
        date += config.assimilation_time_step
        
    for sat in config.satellite:
        
        print('* for sat',sat,':')
        
        # Read satellite info
        sat_info = read_satellite_info(config,sat)
        print(sat_info)
        
        # Read observation
        path_obs = os.path.join(sat_info.path,sat_info.name)
        if '.nc' in path_obs:
            ds = xr.open_dataset(path_obs)
        else:
            files = glob.glob(os.path.join(sat_info.path,sat_info.name+'*.nc'))
            # Get time dimension to concatenate
            ds0 = xr.open_dataset(files[0])
            name_time_dim = ds0[sat_info.name_obs_time].dims[0]
            ds = xr.open_mfdataset(os.path.join(sat_info.path,sat_info.name+'*.nc'),
                                   combine='nested',concat_dim=name_time_dim,lock=False)
            
        # Run subfunction specific to the kind of satellite
        if sat_info.kind in ['swot_simulator','CMEMS']:
            _obs_swot_simulator(ds, assim_dates, dict_obs, sat_info, 
                                config.assimilation_time_step, 
                                config.tmp_DA_path,bbox,
                                lon_mdt,lat_mdt,mdt)
        elif sat_info.kind=='fullSSH':
            _obs_fullSSH(ds, assim_dates,dict_obs, sat_info,
                         config.assimilation_time_step,
                         config.tmp_DA_path,bbox)
    
    # Write *dict_obs* for next experiment
    if config.write_obs:
        if not os.path.exists(dir_obs):
            os.makedirs(dir_obs)
        with open(os.path.join(dir_obs,'dict_obs.pic'), 'wb') as f:
            pickle.dump(dict_obs,f)
        
    return dict_obs


def _obs_swot_simulator(ds, dt_list, dict_obs, sat_info, dt_timestep, out_path,
                        bbox=None,lon_mdt=None,lat_mdt=None,mdt=None):
    """
    NAME
        _obs_swot_simulator

    DESCRIPTION
        Subfunction handling observations generated from swotsimulator module
        
    """
    ds = ds[sat_info.name_obs_grd + sat_info.name_obs_var]
    
    # Select sub area
    lon_obs = ds[sat_info.name_obs_lon] % 360
    lat_obs = ds[sat_info.name_obs_lat]
    ds = ds.where((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs), drop=True)
                  
    # Time loop
    for dt_curr in dt_list:
        
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)
        _ds = ds.sel({sat_info.name_obs_time:slice(dt1,dt2)})
        
        lon = _ds[sat_info.name_obs_lon].values
        lat = _ds[sat_info.name_obs_lat].values
        
        is_obs = np.any(~np.isnan(lon.ravel()*lat.ravel())) * (lon.size>0)
                    
        if is_obs:
            # Add MDT if provided
            if mdt is not None:
                mdt_on_track = interpolate.griddata(
                    (lon_mdt.ravel(),lat_mdt.ravel()),mdt.ravel(),
                    (lon.ravel(),lat.ravel())).reshape(lon.shape)
                _ds[sat_info.name_obs_var[0]] += mdt_on_track
            # Save the selected dataset in a new nc file
        
            varobs = {}
            for namevar in sat_info.name_obs_var:
                varobs[namevar] = _ds[namevar]
            coords = {sat_info.name_obs_time:_ds[sat_info.name_obs_time].values}
            if len(lon.shape)==1:
                coords[sat_info.name_obs_lon] = lon
            else:
                varobs[sat_info.name_obs_lon] = _ds[sat_info.name_obs_lon]
            if len(lat.shape)==1:
                coords[sat_info.name_obs_lat] = lat
            else:
                varobs[sat_info.name_obs_lat] = _ds[sat_info.name_obs_lat]
            if sat_info.name_obs_xac is not None:
                varobs[sat_info.name_obs_xac] = _ds[sat_info.name_obs_xac]
                
            dsout = xr.Dataset(varobs,
                               coords=coords
                               )
            
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = os.path.join(out_path, 'obs_' + sat_info.satellite + '_' +\
                '_'.join(sat_info.name_obs_var) + '_' + date + '.nc')
            print(dt_curr,': '+path)
            dsout[sat_info.name_obs_time].encoding.pop("_FillValue", None)
            dsout.to_netcdf(path)#, encoding={'my_variable': {'_FillValue': 1e35}})
            dsout.close()
            _ds.close()
            del dsout,_ds
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                    dict_obs[dt_curr]['satellite'].append(sat_info)
                    dict_obs[dt_curr]['obs_name'].append(path)
            else:
                dict_obs[dt_curr] = {}
                dict_obs[dt_curr]['satellite'] = [sat_info]
                dict_obs[dt_curr]['obs_name'] = [path]
    
    
def _obs_fullSSH(ds, dt_list, dict_obs, sat_info, dt_timestep, out_path, bbox=None):
    
    name_dim_time_obs = ds[sat_info.name_obs_time].dims[0]
    # read time variable
    times_obs = pd.to_datetime(ds[sat_info.name_obs_time].values)
    # convert to datetime objects
    ts = times_obs - np.datetime64('1970-01-01T00:00:00Z')
    ts /= np.timedelta64(1, 's')
    dt_obs = np.asarray([datetime.utcfromtimestamp(t) for t in ts])
    # time loop on model datetime
    for dt_curr in dt_list:
        # Get time interval for this spectific date
        dt_min = dt_curr - dt_timestep/2
        dt_max = dt_curr + dt_timestep/2
        # Get indexes of observation times that fall into this time interval
        idx_obs = np.where((dt_obs>=dt_min) & (dt_obs<dt_max))[0]
        # Select the data for theses indexes
        ds1 = ds.isel(**{name_dim_time_obs: idx_obs},drop=True)
        if len(ds1[sat_info.name_obs_time])>0:
            # Save the selected dataset in a new nc file
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = os.path.join(out_path,'obs_' + date + '.nc')
            print(dt_curr,': '+path)
            ds1[sat_info.name_obs_time].encoding.pop("_FillValue", None)
            ds1.to_netcdf(path)
            ds1.close()
            del ds1
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                    dict_obs[dt_curr]['satellite'].append(sat_info)
                    dict_obs[dt_curr]['obs_name'].append(path)
            else:
                dict_obs[dt_curr] = {}
                dict_obs[dt_curr]['satellite'] = [sat_info]
                dict_obs[dt_curr]['obs_name'] = [path]
    ds.close()
    del ds
    
    return    


def detrend_obs(dict_obs):
    
    for t in dict_obs:
        # Read obs
        sat_info_list = dict_obs[t]['satellite']
        obs_file_list = dict_obs[t]['obs_name']
        for sat_info,obs_file in zip(sat_info_list,obs_file_list):
            # Read obs file
            ncin = xr.open_dataset(obs_file)
            ncout = ncin.copy().load()
            ncin.close()
            del ncin
            # Load ssh
            ssh = ncout[sat_info.name_obs_var[0]].squeeze().values
            # Fill Masked pixels 
            mask = np.isnan(ssh)
            ssh[mask] = 0
            # Detrend data in all directions
            ssh_detrended = detrendn(ssh)
            # Re-mask 
            ssh_detrended[mask] = np.nan
            # Write detrended observation
            ncout[sat_info.name_obs_var[0]].data = ssh_detrended.reshape(ncout[sat_info.name_obs_var[0]].shape)
            ncout.to_netcdf(obs_file)
            ncout.close()
            del ncout
                
