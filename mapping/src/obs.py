#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:15:09 2021

@author: leguillou
"""
import os, sys
import xarray as xr
import numpy as np

import datetime 
from scipy import signal
import matplotlib.pylab as plt
import glob 

from .tools import detrendn,read_auxdata
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
    
    name_dict_obs = f'dict_obs_{"_".join(config.OBS.keys())}_{date1}_{date2}_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}.txt'
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
    dlon = np.nanmax(State.lon[:,1:] - State.lon[:,:-1])
    dlat = np.nanmax(State.lat[1:,:] - State.lat[:-1,:])
    bbox = [State.lon_min-2*dlon,State.lon_max+2*dlon,State.lat_min-2*dlat,State.lat_max+2*dlat]
    
    # Compute output observation dictionnary
    dict_obs = {}
    assim_dates = []
    date = time_obs_min
    while date<=time_obs_max:
        assim_dates.append(date)
        date += config.EXP.assimilation_time_step
        
    for name_obs, OBS in config.OBS.items():

        print(f'\n{name_obs}:\n{OBS}')
        
        # Preprocessing function to select only variables of interest
        def preprocess(ds):
            name_var = [OBS.name_time, OBS.name_lon, OBS.name_lat]
            for key in OBS.name_var:
                name_var.append(OBS.name_var[key])
            ds = ds[name_var]
            return ds
        
        # Read observation files
        if '.nc' in OBS.path and '*' not in OBS.path:
            _ds = xr.open_dataset(OBS.path)
        else:
            if '*' in OBS.path:
                path = OBS.path
            else:
                path = f'{OBS.path}*.nc'    
            try:
                _ds = xr.open_mfdataset(path,preprocess=preprocess,compat='override',coords='minimal')
            except:
                try:
                    print('opening with combine==nested')
                    files = glob.glob(path)
                    # Get time dimension to concatenate
                    if len(files)==0:
                        continue
                    _ds0 = xr.open_dataset(files[0])
                    name_time_dim = _ds0[OBS.name_time].dims[0]
                    _ds0.close()
                    # Open nested files
                    _ds = xr.open_mfdataset(path,combine='nested',concat_dim=name_time_dim,preprocess=preprocess,compat='override',coords='minimal')
                except:
                    print('Error: unable to open multiple netcdf files')
                    continue
        
        # Copy and close dataset
        ds = _ds.copy()
        _ds.close()
        
        # Name of obs files
        out_name = f'obs_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}'

        # Run subfunction specific to the kind of satellite
        if OBS.super in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
            _obs_alti(ds, assim_dates, dict_obs, name_obs, OBS, 
                                config.EXP.assimilation_time_step, 
                                config.EXP.tmp_DA_path,out_name,State.lon_unit,bbox)
        elif OBS.super=='OBS_L4':
            _obs_l4(ds, assim_dates, dict_obs, name_obs, OBS, 
                                config.EXP.assimilation_time_step, 
                                config.EXP.tmp_DA_path,out_name,State.lon_unit,bbox)
    
    # Write *dict_obs* for next experiment
    if config.EXP.write_obs:
        if not os.path.exists(path_save_obs):
            os.makedirs(path_save_obs)
        new_dict_obs = _new_dict_obs(dict_obs,path_save_obs)
        with open(os.path.join(path_save_obs,name_dict_obs), 'w') as f:
            f.write(str(new_dict_obs))
            
    return dict_obs

def _obs_alti(ds, dt_list, dict_obs, obs_name, obs_attr, dt_timestep, out_path, out_name, lon_unit='0_360', bbox=None):
    """
    NAME
        _obs_alti

    DESCRIPTION
        Subfunction handling observations generated from altimetric observations
        
    """

    ds = ds.assign_coords({obs_attr.name_time:ds[obs_attr.name_time]})
    ds = ds.swap_dims({ds[obs_attr.name_time].dims[0]:obs_attr.name_time})

    # Convert longitude
    if np.sign(ds[obs_attr.name_lon].data.min())==-1 and lon_unit=='0_360':
        ds = ds.assign_coords({obs_attr.name_lon:((ds[obs_attr.name_lon].dims, ds[obs_attr.name_lon].data % 360))})
    elif np.sign(ds[obs_attr.name_lon].data.min())>=0 and lon_unit=='-180_180':
        ds = ds.assign_coords({obs_attr.name_lon:((ds[obs_attr.name_lon].dims, (ds[obs_attr.name_lon].data + 180) % 360 - 180))})

    
    # Select sub area
    lon_obs = ds[obs_attr.name_lon] 
    lat_obs = ds[obs_attr.name_lat]
    ds = ds.where(((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs)).compute(), drop=True)

    # MDT 
    if True in [obs_attr.add_mdt,obs_attr.substract_mdt]:
        finterpmdt = read_auxdata(obs_attr.path_mdt,obs_attr.name_var_mdt, lon_unit)
    else:
        finterpmdt = None
    
    # Error file
    if obs_attr.path_err is not None:
        finterperr = read_auxdata(obs_attr.path_err,obs_attr.name_var_err, lon_unit)
    else:
        finterperr = None
        
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
                # Add/Remove MDT
                if finterpmdt is not None:
                    mdt_on_obs = finterpmdt((lon,lat))
                    if obs_attr.add_mdt:
                        sign = 1
                    else:
                        sign = -1
                    varobs[name].data = varobs[name].data + sign*mdt_on_obs
                # Add synthetic noise to the data
                if 'synthetic_noise' in obs_attr and obs_attr.synthetic_noise is not None:
                    varobs[name].data = varobs[name].data + np.random.normal(0,obs_attr.synthetic_noise,varobs[name].size).reshape(varobs[name].shape) 
                # Remove high values
                if 'varmax' in obs_attr and obs_attr.varmax is not None:
                    varobs[name][(np.abs(varobs[name])>obs_attr.varmax).compute()] = np.nan
                # Error
                if finterperr is not None:
                    err_on_obs = finterperr((lon,lat))
                    varobs[name + '_err'] = varobs[name].copy()
                    varobs[name + '_err'].data = err_on_obs
                
            # Build netcdf
            coords = {obs_attr.name_time:_ds[obs_attr.name_time].values}
            coords[obs_attr.name_lon] = _ds[obs_attr.name_lon]
            coords[obs_attr.name_lat] = _ds[obs_attr.name_lat]
            if obs_attr.super=='OBS_SSH_SWATH' and obs_attr.name_xac is not None:
                coords[obs_attr.name_xac] = _ds[obs_attr.name_xac] # Accross track distance 
            dsout = xr.Dataset(varobs,
                               coords=coords
                               )

            # Write netcdf
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = f"{out_path}/{out_name}_{obs_name}_{'_'.join(obs_attr.name_var)}_{date}"
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
                dict_obs[dt_curr]['obs_name'].append(obs_name)
                dict_obs[dt_curr]['obs_path'].append(path)
                dict_obs[dt_curr]['attributes'].append(obs_attr)
            else:
                dict_obs[dt_curr] = Config({})
                dict_obs[dt_curr]['obs_name'] = [obs_name]
                dict_obs[dt_curr]['obs_path'] = [path]
                dict_obs[dt_curr]['attributes'] = [obs_attr]
                
            count +=1

        
    print(f'--> {count} tracks selected')
    
    
def _obs_l4(ds, dt_list, dict_obs, obs_name, obs_attr, dt_timestep, out_path, out_name, lon_unit='0_360', bbox=None ):
    
    ds = ds.assign_coords({obs_attr.name_time:ds[obs_attr.name_time]})
    ds = ds.swap_dims({ds[obs_attr.name_time].dims[0]:obs_attr.name_time})

    # Subsampling
    if obs_attr.subsampling is not None:
        ds = ds.isel({obs_attr.name_time:slice(None,None,obs_attr.subsampling)})
    
    # Convert longitude
    if np.sign(ds[obs_attr.name_lon].data.min())==-1 and lon_unit=='0_360':
            ds = ds.assign_coords({obs_attr.name_lon:((obs_attr.name_lon, ds[obs_attr.name_lon].data % 360))})
    elif np.sign(ds[obs_attr.name_lon].data.min())>=0 and lon_unit=='-180_180':
        ds = ds.assign_coords({obs_attr.name_lon:((obs_attr.name_lon, (ds[obs_attr.name_lon].data + 180) % 360 - 180))})
    
    # Select sub area
    lon_obs = ds[obs_attr.name_lon] 
    lat_obs = ds[obs_attr.name_lat]
    ds = ds.where(((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs)).compute(), drop=True)

    lon_obs = ds[obs_attr.name_lon].values
    lat_obs = ds[obs_attr.name_lat].values
    if len(lon_obs.shape)==1:
        lon_obs,lat_obs = np.meshgrid(lon_obs,lat_obs)

    # Time loop
    count = 0
    for dt_curr in dt_list:
        
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)
       
        try:
            _ds = ds.sel({obs_attr.name_time:slice(dt1,dt2)})
        except:
            try:
                _ds = ds.where(((ds[obs_attr.name_time]<dt2) &\
                        (ds[obs_attr.name_time]>=dt1)).compute(),drop=True)
            except:
                print(dt_curr,': Warning: impossible to select data for this time')
                continue
        
        if _ds[obs_attr.name_time].size>0:
            # Time mean if several timestep selected
            if _ds[obs_attr.name_time].size>1:
                _ds = _ds.mean(dim=obs_attr.name_time)
            # Read variables
            varobs = {}
            for name in obs_attr.name_var:
                # Observed variable
                varobs[name] = (('y','x'), _ds[obs_attr.name_var[name]].data.squeeze())
                # Error variable
                if obs_attr.name_err is not None and name in obs_attr.name_err:
                    varobs[name+'_err'] = (('y','x'), _ds[obs_attr.name_err[name]].data.squeeze())
            # Coords
            varobs[obs_attr.name_lon] = (('y','x'), lon_obs)
            varobs[obs_attr.name_lat] = (('y','x'), lat_obs)
            # Save to netcdf
            dsout = xr.Dataset(varobs)
            
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = f"{out_path}/{out_name}_{obs_name}_{'_'.join(obs_attr.name_var)}_{date}.nc"
            dsout.to_netcdf(path, encoding={obs_attr.name_lon: {'_FillValue': None},
                                            obs_attr.name_lat: {'_FillValue': None}})
            dsout.close()
            _ds.close()
            del dsout,_ds
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                dict_obs[dt_curr]['obs_name'].append(obs_name)
                dict_obs[dt_curr]['obs_path'].append(path)
                dict_obs[dt_curr]['attributes'].append(obs_attr)
            else:
                dict_obs[dt_curr] = Config({})
                dict_obs[dt_curr]['obs_name'] = [obs_name]
                dict_obs[dt_curr]['obs_path'] = [path]
                dict_obs[dt_curr]['attributes'] = [obs_attr]
            
            count +=1
    
    print(f'--> {count} fields selected')
            


def _new_dict_obs(dict_obs, new_dir, date_min=None, date_max=None):
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
        if date_min is not None and date<date_min:
            continue
        if date_max is not None and date>date_max:
            continue
        # Create new dict_obs by copying the obs files in *new_dir* directory 
        new_dict_obs[date] = {'obs_name':[],'obs_path':[],'attributes':[]}
        for obs_name,obs_path,attributes in zip(dict_obs[date]['obs_name'],dict_obs[date]['obs_path'],dict_obs[date]['attributes']):
            file_obs = os.path.basename(obs_path)
            new_obs_path = os.path.join(new_dir,file_obs)
            # Copy to *tmp_DA_path* directory
            if os.path.normpath(obs_path)!=os.path.normpath(new_obs_path): 
                os.system(f'cp {obs_path} {new_obs_path}')
            # Update new dictionary 
            new_dict_obs[date]['obs_name'].append(obs_name)
            new_dict_obs[date]['obs_path'].append(new_obs_path)
            new_dict_obs[date]['attributes'].append(Config(attributes))
            
    return new_dict_obs
                    
    


def detrend_obs(dict_obs):

    sys.exit('obs.detrend is depreciated')
    
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
            
            
            
def get_obs(dict_obs,box,time_init,name_var='SSH'):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        
        time0 = box[4]
        time1 = box[5]

        lon = np.array([])
        lat = np.array([])
        time = np.array([])
        var = np.array([])
        
        for dt in dict_obs:
            
            if (dt<=time1) & (dt>=time0):
                
                    path_obs = dict_obs[dt]['obs_path']
                    attrs =  dict_obs[dt]['attributes']

                    for _attrs,_path_obs in zip(attrs,path_obs):
                        
                        ds = xr.open_dataset(_path_obs).squeeze() 
                        
                        if name_var not in ds.variables:
                            print(f'Warning: {name_var} not in {_path_obs}')
                            continue

                        lon_obs = ds[_attrs.name_lon] % 360
                        lat_obs = ds[_attrs.name_lat]
                        
                        ds = ds.where(((lon0<=lon_obs) & (lon1>=lon_obs) & 
                  (lat0<=lat_obs) & (lat1>=lat_obs)).compute(), drop=True)
                        time_obs = ds[_attrs.name_time].values
                        time_obs = (time_obs-np.datetime64(time_init))/np.timedelta64(1, 'D')

                        if _attrs.super=='OBS_L4':
                            if len(ds[_attrs.name_lon].shape)==1:
                                lon_obs = ds[_attrs.name_lon].values
                                lat_obs = ds[_attrs.name_lat].values
                                lon_obs,lat_obs = np.meshgrid(lon_obs,lat_obs)
                            else:
                                lon_obs = ds[_attrs.name_lon].values
                                lat_obs = ds[_attrs.name_lat].values
                            var_obs = ds[name_var].values
                            time_obs = time_obs * np.ones_like(var_obs)
                        
                        elif _attrs.super in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
                            lon_obs = ds[_attrs.name_lon].values
                            lat_obs = ds[_attrs.name_lat].values
                            var_obs = ds[name_var].values
                            if len(var_obs.shape)==2:
                                # SWATH data
                                if var_obs.shape[0]==time_obs.size:
                                    dim = 1
                                else:
                                    dim = 0
                                time_obs = time_obs.repeat(var_obs.shape[dim],axis=0)
                        ds.close()
                        del ds
                        
                        # Flattening
                        time1d = time_obs.ravel()
                        lon1d = lon_obs.ravel()
                        lat1d = lat_obs.ravel()
                        var1d = var_obs.ravel()

                        # Remove NaN pixels
                        indNoNan= ~np.isnan(var1d)
                        time1d = time1d[indNoNan]
                        lon1d = lon1d[indNoNan]
                        lat1d = lat1d[indNoNan]
                        var1d = var1d[indNoNan]    
                        
                        # Append to arrays
                        time = np.append(time,time1d)
                        lon = np.append(lon,lon1d)
                        lat = np.append(lat,lat1d)
                        var = np.append(var,var1d)
        
        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }

        if len(time)>0:
            indsort = np.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]   
                lat=lat[indsort]
                time=time[indsort]
                var=var[indsort]

            coords[coords_att['lon']] = lon
            coords[coords_att['lat']] = lat
            coords[coords_att['time']] = time      
        
        return [var, coords, coords_att]
