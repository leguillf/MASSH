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

import datetime 
from scipy import signal
import pandas as pd
import glob 

from .tools import detrendn,read_auxdata_mdt
from .exp import Config

def Obs(config, State, *args, **kwargs):
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
    
    if config.OBS is None:
        print('None observation has been provided')
        return {}
    
    if config.EXP.time_obs_min is not None:
        time_obs_min = config.EXP.time_obs_min
    else:
        time_obs_min = config.EXP.init_date
    
    if config.EXP.time_obs_max is not None:
        time_obs_max = config.EXP.time_obs_max
    else:
        time_obs_max = config.EXP.final_date
        
    date1 = time_obs_min.strftime('%Y%m%d')
    date2 = time_obs_max.strftime('%Y%m%d')
    box = f'{int(State.lon_min)}_{int(State.lon_max)}_{int(State.lat_min)}_{int(State.lat_max)}'
    
    name_dict_obs = f'dict_obs_{"_".join(config.OBS.keys())}_{date1}_{date2}_{box}.txt'
    print('Observation information will be saved in',name_dict_obs)
    
    # Check if previous *dict_obs* has been computed
    if config.EXP.path_obs is None:
        path_save_obs = config.EXP.tmp_DA_path
    else:
        path_save_obs = config.EXP.path_obs
    if config.EXP.write_obs and os.path.exists(os.path.join(path_save_obs,name_dict_obs)) and not config.EXP.compute_obs:
        print(f'Reading {name_dict_obs} from previous run')
        with open(os.path.join(path_save_obs,name_dict_obs), 'rb') as f:
            dict_obs = eval(f.read())
            return _new_dict_obs(dict_obs,config.EXP.tmp_DA_path)
        
    # Read grid
    bbox = [State.lon_min,State.lon_max,State.lat_min,State.lat_max]
    
       
    # Compute output observation dictionnary
    dict_obs = {}
    
    assim_dates = []
    date = time_obs_min
    while date<=time_obs_max:
        assim_dates.append(date)
        date += config.EXP.assimilation_time_step
        
    for name_obs, OBS in config.OBS.items():

        print(f'\n{name_obs}:\n{OBS}')
        
        # Read observation
        if '.nc' in OBS.path and '*' not in OBS.path:
            ds = xr.open_dataset(OBS.path)
        else:
            try:
                if '*' in OBS.path:
                    ds = xr.open_mfdataset(f'{OBS.path}',combine='nested',concat_dim=OBS.name_concat_dim)
                else:
                    ds = xr.open_mfdataset(f'{OBS.path}*.nc',combine='nested',concat_dim=OBS.name_concat_dim)
            except:
                print('Warning: unable to properly open multiple netcdf files. Please check the parameters and name_concat_dim.')
                if '*' in OBS.path:
                    files = glob.glob(f'{OBS.path}')
                else:
                    files = glob.glob(f'{OBS.path}*.nc')
                # Get time dimension to concatenate
                if len(files)==0:
                    continue
                ds0 = xr.open_dataset(files[0])
                name_time_dim = ds0[OBS.name_time].dims[0]
                ds = xr.open_mfdataset(f'{OBS.path}*.nc',
                                    combine='nested',concat_dim=name_time_dim,lock=False)
            
        # Run subfunction specific to the kind of satellite
        if OBS.super in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
            _obs_alti(ds, assim_dates, dict_obs, name_obs, OBS, 
                                config.EXP.assimilation_time_step, 
                                config.EXP.tmp_DA_path,bbox)
        elif OBS.super=='OBS_MODEL':
            _obs_model(ds, assim_dates,dict_obs, OBS,
                         config.EXP.assimilation_time_step,
                         config.EXP.tmp_DA_path,bbox)
    
    # Write *dict_obs* for next experiment
    if config.EXP.write_obs:
        if not os.path.exists(path_save_obs):
            os.makedirs(path_save_obs)
        new_dict_obs = _new_dict_obs(dict_obs,path_save_obs)
        with open(os.path.join(path_save_obs,name_dict_obs), 'w') as f:
            #pickle.dump(new_dict_obs,f)
            f.write(str(new_dict_obs))
            
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
            new_dict_obs[date]['satellite'].append(Config(sat))
            
    return new_dict_obs
                    
    

def _obs_alti(ds, dt_list, dict_obs, obs_name, obs_attr, dt_timestep, out_path, bbox=None):
    """
    NAME
        _obs_swot_simulator

    DESCRIPTION
        Subfunction handling observations generated from swotsimulator module
        
    """

    ds = ds.assign_coords({obs_attr.name_time:ds[obs_attr.name_time]})
    ds = ds.swap_dims({ds[obs_attr.name_time].dims[0]:obs_attr.name_time})
    
    # Select sub area
    lon_obs = ds[obs_attr.name_lon] % 360
    lat_obs = ds[obs_attr.name_lat]
    ds = ds.where((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs), drop=True)

    # MDT 
    if True in [obs_attr.add_mdt,obs_attr.substract_mdt]:
        finterpmdt = read_auxdata_mdt(obs_attr.path_mdt,obs_attr.name_var_mdt)
    else:
        finterpmdt = None
    
    # Time loop
    count = 0
    for dt_curr in dt_list:
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)

        try:
            _ds = ds.sel({obs_attr.name_time:slice(dt1,dt2)})
        except:
            try:
                _ds = ds.where((ds[obs_attr.name_time]<dt2) &\
                        (ds[obs_attr.name_time]>=dt1),drop=True)
            except:
                print(dt_curr,': Warning: impossible to select data for this time')
                continue
        

        lon = _ds[obs_attr.name_lon].values
        lat = _ds[obs_attr.name_lat].values
        
        is_obs = np.any(~np.isnan(lon.ravel()*lat.ravel())) * (lon.size>0)
        


        if is_obs:
            # Save the selected dataset in a new nc file
            varobs = {}
            for name in obs_attr.name_var:
                varobs[name] = _ds[obs_attr.name_var[name]]
                if finterpmdt is not None:
                    mdt_on_obs = finterpmdt((lon,lat))
                    if obs_attr.add_mdt:
                        sign = 1
                    else:
                        sign = -1
                    varobs[name].data = varobs[name].data + sign*mdt_on_obs
                    
            coords = {obs_attr.name_time:_ds[obs_attr.name_time].values}
            varobs[obs_attr.name_lon] = _ds[obs_attr.name_lon]
            varobs[obs_attr.name_lat] = _ds[obs_attr.name_lat]
            if obs_attr.super=='OBS_SSH_SWATH' and obs_attr.name_xac is not None:
                coords[obs_attr.name_xac] = _ds[obs_attr.name_xac] # Accross track distance 
            
            dsout = xr.Dataset(varobs,
                               coords=coords
                               )
            
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = os.path.join(out_path, 'obs_' + obs_name + '_' +\
                '_'.join(obs_attr.name_var) + '_' + date)
            if finterpmdt is not None:
                if obs_attr.add_mdt:
                    path += '_addmdt'
                elif obs_attr.substract_mdt:
                    path += '_submdt'
            path += '.nc'

            dsout.to_netcdf(path, encoding={obs_attr.name_time: {'_FillValue': None},
                                            obs_attr.name_lon: {'_FillValue': None},
                                            obs_attr.name_lat: {'_FillValue': None}})
            dsout.close()
            _ds.close()
            del dsout,_ds
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                    dict_obs[dt_curr]['satellite'].append(obs_attr)
                    dict_obs[dt_curr]['obs_name'].append(path)
            else:
                dict_obs[dt_curr] = Config({})
                dict_obs[dt_curr]['satellite'] = [obs_attr]
                dict_obs[dt_curr]['obs_name'] = [path]
            
            count +=1

        
    print(f'--> {count} tracks selected')
    
    
def _obs_model(ds, dt_list, dict_obs, sat_info, dt_timestep, out_path, bbox=None,finterpmdt=None):
    
    name_dim_time_obs = ds[sat_info.name_obs_time].dims[0]
    # read time variable
    times_obs = pd.to_datetime(ds[sat_info.name_obs_time].values)
    # convert to datetime objects
    ts = times_obs - np.datetime64('1970-01-01T00:00:00Z')
    ts /= np.timedelta64(1, 's')
    dt_obs = np.asarray([datetime.datetime.utcfromtimestamp(t) for t in ts])
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
            
            
            
    

def get_obs(dict_obs,box,time_init,subsampling=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        
        time0 = box[4]
        time1 = box[5]

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
                        time_obs = (time_obs-np.datetime64(time_init))/np.timedelta64(1, 'D')

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
