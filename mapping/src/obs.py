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
from datetime import datetime,timedelta
from scipy import interpolate,signal
import pandas as pd
import glob 
import matplotlib.pylab as plt

from .sat import read_satellite_info
from .tools import detrendn,read_auxdata_mdt

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
    
    if config.satellite is None or len(config.satellite) == 0:
        return {}
    
    date1 = config.init_date.strftime('%Y%m%d')
    date2 = config.final_date.strftime('%Y%m%d')
    box = f'{int(State.lon.min())}_{int(State.lon.max())}_{int(State.lat.min())}_{int(State.lat.max())}'
    name_dict_obs = f'dict_obs_{"_".join(config.satellite)}_{date1}_{date2}_{box}.pic'
    
    # Check if previous *dict_obs* has been computed
    if config.path_obs is None:
        path_save_obs = config.tmp_DA_path
    else:
        path_save_obs = config.path_obs
    if config.write_obs and os.path.exists(os.path.join(path_save_obs,name_dict_obs)):
        print(f'Reading {name_dict_obs} from previous run')
        with open(os.path.join(path_save_obs,name_dict_obs), 'rb') as f:
            dict_obs = pickle.load(f)
            return _new_dict_obs(dict_obs,config.tmp_DA_path)
        
    # Read grid
    lon = State.lon
    lat = State.lat
    bbox = [lon.min(),lon.max(),lat.min(),lat.max()]
    
    # MDT
    finterpmdt = None
    if config.path_mdt is not None and os.path.exists(config.path_mdt):
        finterpmdt = read_auxdata_mdt(config.path_mdt,config.name_var_mdt)
       
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
            if len(files)==0:
                continue
            ds0 = xr.open_dataset(files[0])
            name_time_dim = ds0[sat_info.name_obs_time].dims[0]
            ds = xr.open_mfdataset(os.path.join(sat_info.path,sat_info.name+'*.nc'),
                                   combine='nested',concat_dim=name_time_dim,lock=False)
            
        # Run subfunction specific to the kind of satellite
        if sat_info.kind in ['swot_simulator','CMEMS']:
            _obs_swot_simulator(ds, assim_dates, dict_obs, sat_info, 
                                config.assimilation_time_step, 
                                config.tmp_DA_path,bbox,finterpmdt)
        elif sat_info.kind=='fullSSH':
            _obs_fullSSH(ds, assim_dates,dict_obs, sat_info,
                         config.assimilation_time_step,
                         config.tmp_DA_path,bbox,finterpmdt)
    
    # Write *dict_obs* for next experiment
    if config.write_obs:
        if not os.path.exists(path_save_obs):
            os.makedirs(path_save_obs)
        with open(os.path.join(path_save_obs,name_dict_obs), 'wb') as f:
            pickle.dump(_new_dict_obs(dict_obs,path_save_obs),f)
            
    return dict_obs

def _new_dict_obs(dict_obs,new_dir):
    """
    NAME
        _new_dict_obs

    DESCRIPTION
        Subfunction creating a new dict_obs, similar as dict_obs, except that 
        the obs files are stored in *new_dir*
        
        Args: 
            dict_obs(dict): initial dictionary
            new_dir(str): new directory where the obs will be copied
        Returns:
            new_dict_obs (dictionary)
    """
    
    new_dict_obs = {}
    for date in dict_obs:
        # Create new dict_obs by copying the obs files in *new_dir* directory 
        new_dict_obs[date] = {'obs_name':[],'satellite':[]}
        for obs,sat in zip(dict_obs[date]['obs_name'],dict_obs[date]['satellite']):
            file_obs = os.path.basename(obs)
            new_obs = os.path.join(new_dir,file_obs)
            # Copy to *tmp_DA_path* directory
            if os.path.normpath(obs)!=os.path.normpath(new_obs): 
                os.system(f'cp {obs} {new_obs}')
            # Update new dictionary 
            new_dict_obs[date]['obs_name'].append(new_obs)
            new_dict_obs[date]['satellite'].append(sat)
            
    return new_dict_obs
                    
    

def _obs_swot_simulator(ds, dt_list, dict_obs, sat_info, dt_timestep, out_path,
                        bbox=None,finterpmdt=None):
    """
    NAME
        _obs_swot_simulator

    DESCRIPTION
        Subfunction handling observations generated from swotsimulator module
        
    """
    ds = ds[sat_info.name_obs_grd + sat_info.name_obs_var]
    
    ds = ds.assign_coords({sat_info.name_obs_time:ds[sat_info.name_obs_time]})

    ds = ds.rename({ds[sat_info.name_obs_time].dims[0]:sat_info.name_obs_time})
    
    # Select sub area
    lon_obs = ds[sat_info.name_obs_lon] % 360
    lat_obs = ds[sat_info.name_obs_lat]
    ds = ds.where((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs), drop=True)
    

                  
    # Time loop
    for dt_curr in dt_list:
        
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)
       
        try:
            _ds = ds.sel({sat_info.name_obs_time:slice(dt1,dt2)})
        except:
            try:
                _ds = ds.where((ds[sat_info.name_obs_time]<dt2) &\
                        (ds[sat_info.name_obs_time]>=dt1),drop=True)
            except:
                print(dt_curr,': Warning: impossible to select data for this time')
                continue
        

        lon = _ds[sat_info.name_obs_lon].values
        lat = _ds[sat_info.name_obs_lat].values
        
        is_obs = np.any(~np.isnan(lon.ravel()*lat.ravel())) * (lon.size>0)
                    
        if is_obs:
            # Save the selected dataset in a new nc file
            varobs = {}
            for namevar in sat_info.name_obs_var:
                varobs[namevar] = _ds[namevar]
                if finterpmdt is not None and True in [sat_info.add_mdt,sat_info.substract_mdt]:
                    mdt_on_obs = finterpmdt((lon,lat))
                    if sat_info.add_mdt:
                        sign = 1
                    else:
                        sign = -1
                    varobs[namevar].data = varobs[namevar].data + sign*mdt_on_obs
                    
            coords = {sat_info.name_obs_time:_ds[sat_info.name_obs_time].values}
            varobs[sat_info.name_obs_lon] = _ds[sat_info.name_obs_lon]
            varobs[sat_info.name_obs_lat] = _ds[sat_info.name_obs_lat]
            if sat_info.name_obs_xac is not None:
                varobs[sat_info.name_obs_xac] = _ds[sat_info.name_obs_xac]
            
            dsout = xr.Dataset(varobs,
                               coords=coords
                               )
            
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = os.path.join(out_path, 'obs_' + sat_info.satellite + '_' +\
                '_'.join(sat_info.name_obs_var) + '_' + date)
            if finterpmdt is not None:
                if sat_info.add_mdt:
                    path += '_addmdt'
                elif sat_info.substract_mdt:
                    path += '_submdt'
            path += '.nc'
            print(dt_curr,': '+path)
            #dsout[sat_info.name_obs_time].encoding.pop("_FillValue", None)
            dsout.to_netcdf(path, encoding={sat_info.name_obs_time: {'_FillValue': None},
                                            sat_info.name_obs_lon: {'_FillValue': None},
                                            sat_info.name_obs_lat: {'_FillValue': None}})
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
    
    
def _obs_fullSSH(ds, dt_list, dict_obs, sat_info, dt_timestep, out_path, bbox=None,finterpmdt=None):
    
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
            if finterpmdt is not None:
                mdt_on_obs = finterpmdt((ds1[sat_info.name_obs_lon],ds1[sat_info.name_obs_lat]))
                for namevar in sat_info.name_obs_var:
                    ds1[namevar].data = ds1[namevar].data - mdt_on_obs
            # Save the selected dataset in a new nc file
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = os.path.join(out_path,'obs_' + date + '.nc')
            print(dt_curr,': '+path)
           
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
            if len(ssh.shape)==0:
                ssh_detrended = +ssh
            elif len(ssh.shape)==1:
                ssh_detrended = signal.detrend(ssh)
            else:
                ssh_detrended = detrendn(ssh)
            # Re-mask
            if mask.size>1:
                ssh_detrended[mask] = np.nan
            # Write detrended observation
            ncout[sat_info.name_obs_var[0]].data = ssh_detrended.reshape(ncout[sat_info.name_obs_var[0]].shape)
            ncout.to_netcdf(obs_file)
            ncout.close()
            del ncout
            
            
            
    

def get_obs(dict_obs,box,subsampling=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        
        time0 = datetime(2003,1,1) + timedelta(days=box[4]-19358)
        time1 = datetime(2003,1,1) + timedelta(days=box[5]-19358)

        lon= np.array([])
        lat= np.array([])
        time= np.array([])
        ssh= np.array([])
        
        for dt in dict_obs:
            
            if (dt<=time1) & (dt>=time0):
                
                    path_obs = dict_obs[dt]['obs_name']
                    sat =  dict_obs[dt]['satellite']
                    
                    for _sat,_path_obs in zip(sat,path_obs):
                        
                        ds = xr.open_dataset(_path_obs).squeeze() 
                        lon_obs = ds[_sat.name_obs_lon] % 360
                        lat_obs = ds[_sat.name_obs_lat]
                        
                        ds = ds.where((lon0<=lon_obs) & (lon1>=lon_obs) & 
                  (lat0<=lat_obs) & (lat1>=lat_obs), drop=True)
                        time_obs = ds[_sat.name_obs_time].values
                        time_obs = (time_obs-np.datetime64(time0))/np.timedelta64(1, 'D')

                        if _sat.kind=='fullSSH':
                            if len(ds[_sat.name_obs_lon].shape)==1:
                                lon_obs = ds[_sat.name_obs_lon].values[::subsampling]
                                lat_obs = ds[_sat.name_obs_lat].values[::subsampling]
                                lon_obs,lat_obs = np.meshgrid(lon_obs,lat_obs)
                            else:
                                lon_obs = ds[_sat.name_obs_lon].values[::subsampling,::subsampling]
                                lat_obs = ds[_sat.name_obs_lat].values[::subsampling,::subsampling]
                            ssh_obs = ds[_sat.name_obs_var[0]].values[::subsampling,::subsampling]
                            time_obs = time_obs * np.ones_like(ssh_obs)
                        
                        elif _sat.kind in ['swot_simulator','CMEMS']:
                            lon_obs = ds[_sat.name_obs_lon].values
                            lat_obs = ds[_sat.name_obs_lat].values
                            ssh_obs = ds[_sat.name_obs_var[0]].values
                            if _sat.kind=='CMEMS' and len(_sat.name_obs_var)==2:
                                ssh_obs += ds[_sat.name_obs_var[1]].values
                            if len(ssh_obs.shape)==2:
                                # SWATH data
                                if ssh_obs.shape[0]==time_obs.size:
                                    dim = 1
                                else:
                                    dim = 0
                                time_obs = time_obs.repeat(ssh_obs.shape[dim],axis=0)
                        ds.close()
                        del ds
                        
                        # Flattening
                        time1d = time_obs.ravel()
                        lon1d = lon_obs.ravel()
                        lat1d = lat_obs.ravel()
                        ssh1d = ssh_obs.ravel()

                        # Remove NaN pixels
                        indNoNan= ~np.isnan(ssh1d)
                        time1d = time1d[indNoNan]
                        lon1d = lon1d[indNoNan]
                        lat1d = lat1d[indNoNan]
                        ssh1d = ssh1d[indNoNan]    
                        
                        # Append to arrays
                        time = np.append(time,time1d)
                        lon = np.append(lon,lon1d)
                        lat = np.append(lat,lat1d)
                        ssh = np.append(ssh,ssh1d)
        
        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
        values=None

        if len(time)>0:
            indsort = np.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]   
                lat=lat[indsort]
                time=time[indsort]
                ssh=ssh[indsort]

            coords[coords_att['lon']] = lon
            coords[coords_att['lat']] = lat
            coords[coords_att['time']] = time      
            values =  ssh

                    
        return [values, coords, coords_att]
