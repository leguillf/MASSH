#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import os,sys
import xarray as xr 
import numpy as np 
from src import grid as grid
import pickle
import matplotlib.pylab as plt
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist
import pandas as pd
from jax.experimental import sparse
import jax.numpy as jnp 
from jax import jit
import jax
jax.config.update("jax_enable_x64", True)

def Obsop(config, State, dict_obs, Model, verbose=1, *args, **kwargs):
    """
    NAME
        basis

    DESCRIPTION
        Main function calling obsopt for specific observational operators
    """
    
    if config.OBSOP is None:
        return 
    
    if verbose:
        print(config.OBSOP)
    
    if config.OBSOP.super is None:
        return Obsop_multi(config, State, dict_obs, Model)
    
    elif config.OBSOP.super=='OBSOP_INTERP_L3':
        return Obsop_interp_l3(config, State, dict_obs, Model)

    elif config.OBSOP.super=='OBSOP_INTERP_L3_GEOCUR':
        return Obsop_interp_l3_geocur(config, State, dict_obs, Model)
    
    elif config.OBSOP.super=='OBSOP_INTERP_L4':
        return Obsop_interp_l4(config, State, dict_obs, Model)
    
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')



class Obsop_interp:

    def __init__(self,config,State,dict_obs,Model):
        
        self.compute_H = config.OBSOP.compute_op

        # Date obs list
        self.date_obs = []
        
        # Pattern for saving files
        box = f'{int(State.lon_min)}_{int(State.lon_max)}_{int(State.lat_min)}_{int(State.lat_max)}'
        self.name_H = f'H_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}_{int(State.dx)}_{int(State.dy)}'

        # Path to save operators
        self.compute_op = config.OBSOP.compute_op
        self.write_op = config.OBSOP.write_op
        if self.write_op:
            # We'll save or read operator data to *path_save*
            self.path_save = config.OBSOP.path_save
            if not os.path.exists(self.path_save):
                os.makedirs(self.path_save)
        else:
            self.path_save = config.EXP.tmp_DA_path

        # Temporary path where to save misfit
        self.tmp_DA_path = config.EXP.tmp_DA_path
        
        # Model variable
        self.name_mod_var = Model.name_var
        
        # For grid interpolation:
        lon = +State.lon
        lat = +State.lat
        self.shape_grid = (State.ny, State.nx)
        self.coords_geo = np.column_stack((lon.ravel(), lat.ravel()))
        self.coords_car = grid.geo2cart(self.coords_geo)

        # Mask land
        if State.mask is not None:
            self.ind_mask = np.where(State.mask)[1]
        else:
            self.ind_mask = []
        
        # Mask boundary pixels
        self.ind_borders = []
        if config.OBSOP.mask_borders:
            coords_geo_borders = np.column_stack((
                np.concatenate((State.lon[0,:],State.lon[1:-1,-1],State.lon[-1,:],State.lon[:,0])),
                np.concatenate((State.lat[0,:],State.lat[1:-1,-1],State.lat[-1,:],State.lat[:,0]))
                ))
            if len(coords_geo_borders)>0:
                for i in range(self.coords_geo.shape[0]):
                    if np.any(np.all(np.isclose(coords_geo_borders,self.coords_geo[i]), axis=1)):
                        self.ind_borders.append(i)
        
        # Process obs
        self.dict_obs = dict_obs
        

    def process_obs(self, var_bc=None):

        return

    def is_obs(self,t):

        return t in self.date_obs
                
    def misfit(self,t,State):

        return

    def adj(self, t, adState, R):

        return

class Obsop_interp_l3(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.date_obs = []
        self.name_var_obs = {}
        self.name_obs = []
        t_obs = [tobs for tobs in dict_obs.keys()] 
        for t in Model.timestamps:
            delta_t = [(t - tobs).total_seconds() for tobs in dict_obs.keys()]
            if len(delta_t)>0:
                t_obs = [tobs for tobs in dict_obs.keys()] 
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[t_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[t_obs[ind_obs]]['attributes']):
                        
                        # Check if this observation class is wanted
                        if sat_info.super not in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
                            continue
                        if config.OBSOP.name_obs is None or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.name_var_obs:
                                self.name_var_obs[t] = []
                                self.date_obs.append(t_obs[ind_obs])
                            # Get obs variable names (SSH,U,V,SST...) at this time
                            for name in sat_info['name_var']:
                                if name not in self.name_var_obs[t]:
                                    self.name_var_obs[t].append(name)
        
        # For grid interpolation:
        self.Npix = config.OBSOP.Npix
        self.dmax = self.Npix*np.mean(np.sqrt(State.DX**2 + State.DY**2))*1e-3*np.sqrt(2)/2 # maximal distance for space interpolation

        self.name_H += f'_L3_{config.OBSOP.Npix}'
    
    def _sparse_op(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        row = [] # indexes of observation grid
        col = [] # indexes of state grid
        data = [] # interpolation coefficients
        Nobs = coords_geo_obs.shape[0]

        for iobs in range(Nobs):
            _dist = cdist(coords_car_obs[iobs][np.newaxis,:], self.coords_car, metric="euclidean")[0]
            # Npix closest
            ind_closest = np.argsort(_dist)
            # Get Npix closest pixels (ignoring boundary pixels)
            weights = []
            for ipix in range(self.Npix):
                if (not ind_closest[ipix] in self.ind_borders) and (not ind_closest[ipix] in self.ind_mask) and (_dist[ind_closest[ipix]]<=self.dmax):
                    weights.append(np.exp(-(_dist[ind_closest[ipix]]**2/(2*(.5*self.dmax)**2))))
                    row.append(iobs)
                    col.append(ind_closest[ipix])
            sum_weights = np.sum(weights)
            # Fill interpolation coefficients 
            for w in weights:
                data.append(w/sum_weights)

        return csc_matrix((data, (row, col)), shape=(Nobs, self.coords_geo.shape[0]))

    def process_obs(self, var_bc=None):

        self.varobs = {}
        self.errobs = {}
        self.Hop = {}

        for i,t in enumerate(self.date_obs):

            self.varobs[t] = {}
            self.errobs[t] = {}
            self.Hop[t] = {}

            sat_info_list = self.dict_obs[t]['attributes']
            obs_file_list = self.dict_obs[t]['obs_path']
            obs_name_list = self.dict_obs[t]['obs_name']

        
            # Concatenate obs from different sensors
            lon_obs = {}
            lat_obs = {}
            var_obs = {}
            err_obs = {}

            for sat_info,obs_file,obs_name in zip(sat_info_list,obs_file_list,obs_name_list):

                if sat_info.super not in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
                        continue
                
                ####################
                # Merge observations
                ####################
                with xr.open_dataset(obs_file) as ncin:
                    lon = ncin[sat_info['name_lon']].values.ravel() 
                    lat = ncin[sat_info['name_lat']].values.ravel()

                    for name in sat_info['name_var']:
                        # Observed variable
                        var = ncin[name].values.ravel() 
                        # Observed error
                        name_err = name + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values.ravel() 
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)                        
                        if name in lon_obs:
                            var_obs[name] = np.concatenate((var_obs[name],var))
                            err_obs[name] = np.concatenate((err_obs[name],err))
                            lon_obs[name] = np.concatenate((lon_obs[name],lon))
                            lat_obs[name] = np.concatenate((lat_obs[name],lat))
                        else:
                            var_obs[name] = +var
                            err_obs[name] = +err
                            lon_obs[name] = +lon
                            lat_obs[name] = +lat
            
            for name in lon_obs:
                coords_obs = np.column_stack((lon_obs[name], lat_obs[name]))
                file_L3 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
                if var_bc is not None and name in var_bc:
                    mask = np.any(np.isnan(self.coords_geo),axis=1)
                    var_bc_interp = griddata(self.coords_geo[~mask], var_bc[name][i].flatten()[~mask], coords_obs, method='cubic')
                    var_obs[name] -= var_bc_interp

                # Fill dictionnaries
                self.varobs[t][name] = var_obs[name]
                self.errobs[t][name] = err_obs[name]

                # Compute Sparse operator
                if not self.compute_op and self.write_op and os.path.exists(file_L3):
                    with open(file_L3, "rb") as f:
                        self.Hop[t][name] = pickle.load(f)
                else:
                    # Compute operator
                    _H = self._sparse_op(lon_obs[name],lat_obs[name])
                    self.Hop[t][name] = _H
                    # Save operator if asked
                    if self.write_op:
                        with open(file_L3, "wb") as f:
                            pickle.dump(_H, f)

    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Project model state to obs space
            HX = self.Hop[t][name] @ X

            # Compute misfit & errors
            _misfit = (HX-self.varobs[t][name])
            _inverr = 1/self.errobs[t][name]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0
        
            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit": (("Nobs"), _inverr*_inverr*_misfit),
                    }
                    )
            dsout.to_netcdf(
                os.path.join(self.tmp_DA_path,f"misfit_L3_{t.strftime('%Y%m%d_%H%M')}.nc"), 
                mode=mode, 
                group=name
                )
            dsout.close()
            mode = 'a'

            # Concatenate
            misfit = np.concatenate((misfit,_inverr*_misfit))

        return misfit

    def adj(self, t, adState, R):

        for name in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L3_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit = ds['misfit'].values
            ds.close()
            del ds

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            adX = self.Hop[t][name].T @ misfit

            # Update adjoint variable
            adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[name])  

class Obsop_interp_l3_geocur(Obsop_interp_l3):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)
    
    def _compute_geovel_sat(self,time_alongtrack,lon_alongtrack, lat_alongtrack, sla_alongtrack, delta_t, delta_x):

        dt = np.float64(time_alongtrack[1:] - time_alongtrack[:-1])*1e-9
        ind = np.where(dt>1.1*delta_t)[0]
        ind += 1
        ind = np.append([0],ind)


        vel_alongtrack = sla_alongtrack*np.nan
        angle_alongtrack = sla_alongtrack*0.

        cn = [
            np.array([-0.1,0,0.3,0.8]),
            np.array([0.2,0,0.2,0.6]),
            np.array([0.4,0.1,0.1,0.4]),
            np.array([0.6,0.2,0.,0.2]),
            np.array([0.8,0.3,0.,-0.1])
        ]
            
        n = [
            np.array([1,2,3,4]),
            np.array([-1,1,2,3]),
            np.array([-2,-1,1,2]),
            np.array([-3,-2,-1,1]),
            np.array([-4,-3,-2,-1])
        ]

        for idx in range(len(ind)):
            if idx==0:
                i0 = 0
                if len(ind)>1:
                    i1 = ind[1]
                else:
                    i1 = sla_alongtrack.size
            elif idx==len(ind)-1:
                i0 = ind[idx]
                i1 = sla_alongtrack.size
            else:
                i0 = ind[idx]
                i1 = ind[idx+1]
            if i1-i0<=4:
                continue
            _sla_alongtrack = sla_alongtrack[i0:i1]
            _lon_alongtrack = lon_alongtrack[i0:i1]
            _lat_alongtrack = lat_alongtrack[i0:i1]

            
            f = 4*np.pi/86164*np.sin(_lat_alongtrack*np.pi/180)
            
            _vel_alongtrack = _sla_alongtrack*np.nan
            
            _vel_alongtrack[0] = 9.81/f[0] * (np.sum(-cn[0]/(n[0]*delta_x)) * _sla_alongtrack[0] + np.sum(cn[0]*_sla_alongtrack[0+n[0]]/(n[0]*delta_x)))
            _vel_alongtrack[1] = 9.81/f[1] * (np.sum(-cn[1]/(n[1]*delta_x)) * _sla_alongtrack[1] + np.sum(cn[1]*_sla_alongtrack[1+n[1]]/(n[1]*delta_x)))
            
            for i in range(2,_sla_alongtrack.size-2): 
                _vel_alongtrack[i] = 9.81/f[i] * (np.sum(-cn[2]/(n[2]*delta_x)) * _sla_alongtrack[i] + np.sum(cn[2]*_sla_alongtrack[i+n[2]]/(n[2]*delta_x)))
            _vel_alongtrack[-2] = 9.81/f[-2] * (np.sum(-cn[-2]/(n[-2]*delta_x)) * _sla_alongtrack[-2] + np.sum(cn[-2]*_sla_alongtrack[-2+n[-2]]/(n[-2]*delta_x)))
            _vel_alongtrack[-1] = 9.81/f[-1] * (np.sum(-cn[-1]/(n[-1]*delta_x)) * _sla_alongtrack[-1] + np.sum(cn[-1]*_sla_alongtrack[-1+n[-1]]/(n[-1]*delta_x)))
            
            vel_alongtrack[i0:i1] = _vel_alongtrack
            angle_alongtrack[i0+1:i1-1] = np.arctan2((_lat_alongtrack[2:] - _lat_alongtrack[:-2]), ((_lon_alongtrack[2:] - _lon_alongtrack[:-2])*np.cos(np.deg2rad(_lat_alongtrack[1:-1]))))
            angle_alongtrack[i0] = angle_alongtrack[i0+1] 
            angle_alongtrack[i1-1] = angle_alongtrack[i1-2] 
        
        return vel_alongtrack, angle_alongtrack

    def process_obs(self, var_bc=None):

        self.velobs = {} # Geostrophic current velocity norm
        self.angobs = {} # Angle of the satellite track 
        self.errobs = {} # Error on observed SSH 
        self.Hop = {}

        for i,t in enumerate(self.date_obs):
            
            lon_obs = np.array([])
            lat_obs = np.array([])
            self.velobs[t] = np.array([])
            self.angobs[t] = np.array([])
            self.errobs[t] = np.array([])
            self.Hop[t] = np.array([])

            sat_info_list = self.dict_obs[t]['attributes']
            obs_file_list = self.dict_obs[t]['obs_path']
            obs_name_list = self.dict_obs[t]['obs_name']
    
            for sat_info,obs_file,obs_name in zip(sat_info_list,obs_file_list,obs_name_list):

                if sat_info.super not in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
                    continue
                if sat_info.super=='OBS_SSH_SWATH':
                    print('OBSOP_INTERP_L3_GEOCUR not yet implemented for OBS_SSH_SWATH')
                    continue

                ####################
                # Merge observations
                ####################
                with xr.open_dataset(obs_file) as ncin:
                    time = ncin[sat_info['name_time']].values
                    lon = ncin[sat_info['name_lon']].values.ravel() 
                    lat = ncin[sat_info['name_lat']].values.ravel()

                    for name in sat_info['name_var']:

                        if name!='SSH':
                            continue

                        # Observed variable (SSH)
                        var = ncin[name].values.ravel() 

                        # Convert SSH to geostrophic current velocity
                        vel, ang = self._compute_geovel_sat(time,lon, lat,var, sat_info['delta_t'], sat_info['delta_t']*sat_info['velocity'])

                        # Observed error
                        name_err = name + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values.ravel() 
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)  

                        self.velobs[t] = np.concatenate((self.velobs[t],vel))
                        self.angobs[t] = np.concatenate((self.angobs[t],ang))
                        self.errobs[t] = np.concatenate((self.errobs[t],err))
                        lon_obs = np.concatenate((lon_obs,lon))
                        lat_obs = np.concatenate((lat_obs,lat))
            
            
                file_L3 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{t.strftime('%Y%m%d_%H%M')}_SSH.pic"

                # Compute Sparse operator
                if not self.compute_op and self.write_op and os.path.exists(file_L3):
                    with open(file_L3, "rb") as f:
                        self.Hop[t] = pickle.load(f)
                else:
                    # Compute operator
                    _H = self._sparse_op(lon_obs,lat_obs)
                    self.Hop[t] = _H
                    # Save operator if asked
                    if self.write_op:
                        with open(file_L3, "wb") as f:
                            pickle.dump(_H, f)

    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        if 'SSH' in self.name_var_obs[t]:

            # Get model velocities
            u = State.getvar(self.name_mod_var['U']).ravel() 
            v = State.getvar(self.name_mod_var['V']).ravel() 

            # Project model state to obs space
            HX = - np.sin(self.angobs[t]) * (self.Hop[t] @ u) + np.cos(self.angobs[t]) * (self.Hop[t] @ v)

            # Compute misfit & errors
            _misfit = HX-self.velobs[t]
            _inverr = 1/self.errobs[t]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0
        
            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit": (("Nobs"), _inverr*_inverr*_misfit),
                    }
                    )
            dsout.to_netcdf(
                os.path.join(self.tmp_DA_path,f"misfit_L3_GEOCUR_{t.strftime('%Y%m%d_%H%M')}.nc"), 
                )
            dsout.close()

            # Concatenate
            misfit = np.concatenate((misfit,_inverr*_misfit))

        return misfit

    def adj(self, t, adState, R):

        if 'SSH' in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L3_GEOCUR_{t.strftime('%Y%m%d_%H%M')}.nc")))
            misfit = ds['misfit'].values
            ds.close()
            del ds

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            adu = adState.getvar(self.name_mod_var['U'])
            adv = adState.getvar(self.name_mod_var['V'])

            # Compute adjoint operation of y = Hx
            adu += (self.Hop[t].T @ (-np.sin(self.angobs[t])*misfit)).reshape(adu.shape)
            adv += (self.Hop[t].T @ (np.cos(self.angobs[t])*misfit)).reshape(adv.shape)

            # Update adjoint variable
            adState.setvar(adu, self.name_mod_var['U'])      
            adState.setvar(adv, self.name_mod_var['V'])      


class Obsop_interp_l4(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.name_var_obs = {}
        self.name_obs = []
        t_obs = [tobs for tobs in dict_obs.keys()] 
        for t in Model.timestamps:
            delta_t = [(t - tobs).total_seconds() for tobs in dict_obs.keys()]
            if len(delta_t)>0:
                t_obs = [tobs for tobs in dict_obs.keys()] 
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[t_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[t_obs[ind_obs]]['attributes']):
                        
                        # Check if this observation class is wanted
                        if sat_info.super!='OBS_L4':
                            continue
                        if config.OBSOP.name_obs is None or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.name_var_obs:
                                self.name_var_obs[t] = []
                                self.date_obs.append(t_obs[ind_obs])
                            # Get obs variable names (SSH,U,V,SST...) at this time
                            for name in sat_info['name_var']:
                                if name not in self.name_var_obs[t]:
                                    self.name_var_obs[t].append(name)
        
        # For grid interpolation:
        self.interp_method = config.OBSOP.interp_method

        self.name_H += f'_L4_{config.OBSOP.interp_method}'

    def process_obs(self, var_bc=None):

        self.varobs = {}
        self.errobs = {}

        for i,t in enumerate(self.date_obs):

            self.varobs[t] = {}
            self.errobs[t] = {}

            sat_info_list = self.dict_obs[t]['attributes']
            obs_file_list = self.dict_obs[t]['obs_path']
            obs_name_list = self.dict_obs[t]['obs_name']

        
            # Concatenate obs from different sensors
            lon_obs = {}
            lat_obs = {}
            var_obs = {}
            err_obs = {}

            for sat_info,obs_file,obs_name in zip(sat_info_list,obs_file_list,obs_name_list):

                if sat_info.super!='OBS_L4':
                    continue

                ####################
                # Merge observations
                ####################
                with xr.open_dataset(obs_file) as ncin:
                    lon = ncin[sat_info['name_lon']].values.ravel() 
                    lat = ncin[sat_info['name_lat']].values.ravel()

                    for name in sat_info['name_var']:
                        # Observed variable
                        var = ncin[name].values.ravel() 
                        # Observed error
                        name_err = name + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values.ravel() 
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)                        
                        if name in lon_obs:
                            var_obs[name] = np.concatenate((var_obs[name],var))
                            err_obs[name] = np.concatenate((err_obs[name],err))
                            lon_obs[name] = np.concatenate((lon_obs[name],lon))
                            lat_obs[name] = np.concatenate((lat_obs[name],lat))
                        else:
                            var_obs[name] = +var
                            err_obs[name] = +err
                            lon_obs[name] = +lon
                            lat_obs[name] = +lat
            
            for name in lon_obs:
                coords_obs = np.column_stack((lon_obs[name], lat_obs[name]))
                ################
                # Process L4 obs
                ################
                file_L4 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
                if not self.compute_op and self.write_op and os.path.exists(file_L4):
                    with open(file_L4, "rb") as f:
                        var_obs_interp, err_obs_interp = pickle.load(f)
                else:
                    # Grid interpolation: performing spatial interpolation now
                    var_obs_interp = griddata(coords_obs, var_obs[name], self.coords_geo, method=self.interp_method)
                    err_obs_interp = griddata(coords_obs, err_obs[name], self.coords_geo, method=self.interp_method)
                    # Save operator if asked
                    if self.write_op:
                        with open(file_L4, "wb") as f:
                            pickle.dump((var_obs_interp,err_obs_interp), f)

                if var_bc is not None and name in var_bc:
                    var_obs_interp -= var_bc[name][i].flatten()

                # Fill dictionnaries
                self.varobs[t][name] = var_obs_interp
                self.errobs[t][name] = err_obs_interp

    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Project model state to obs space
            HX = +X

            # Compute misfit & errors
            _misfit = (HX-self.varobs[t][name])
            _inverr = 1/self.errobs[t][name]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0
        
            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit": (("Nobs"), _inverr*_inverr*_misfit),
                    }
                    )
            dsout.to_netcdf(
                os.path.join(self.tmp_DA_path,f"misfit_L4_{t.strftime('%Y%m%d_%H%M')}.nc"), 
                mode=mode, 
                group=name
                )
            dsout.close()
            mode = 'a'

            # Concatenate
            misfit = np.concatenate((misfit,_inverr*_misfit))

        return misfit

    def adj(self, t, adState, R):

        for name in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L4_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit = ds['misfit'].values
            ds.close()
            del ds

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            adX = +misfit

            # Update adjoint variable
            adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[name])      


###############################################################################
#                            Multi-Operators                                  #
###############################################################################      

class Obsop_multi:

    def __init__(self,config,State,dict_obs,Model):

        self.Obsop = []
        _config = config.copy()

        for _OBSOP in config.OBSOP:
            _config.OBSOP = config.OBSOP[_OBSOP]
            self.Obsop.append(Obsop(_config,State,dict_obs,Model))

    def is_obs(self,t):

        for _Obsop in self.Obsop:
            if _Obsop.is_obs(t):
                return True
        return False

    def process_obs(self, var_bc=None):

        for _Obsop in self.Obsop:
            _Obsop.process_obs(var_bc)
                

    def misfit(self,t,State):

        misfit = np.array([])

        for _Obsop in self.Obsop:
            if _Obsop.is_obs(t):
                _misfit = _Obsop.misfit(t,State)
                misfit = np.concatenate((misfit,_misfit))
        
        return misfit

    def adj(self, t, adState, R):
    
        for _Obsop in self.Obsop:
            if _Obsop.is_obs(t):
                _Obsop.adj(t,adState,R)

    
class Obsop_interp_jax(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):
        super().__init__(config,State,dict_obs,Model)

        self.H_jit = jit(self.H, static_argnums=[1,2])
        self.misfit_jit = jit(self.misfit, static_argnums=[1])

    def _sparse_op(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        row = [] # indexes of observation grid
        col = [] # indexes of state grid
        data = [] # interpolation coefficients
        Nobs = coords_geo_obs.shape[0]

        for iobs in range(Nobs):
            _dist = cdist(coords_car_obs[iobs][np.newaxis,:], self.coords_car, metric="euclidean")[0]
            # Npix closest
            ind_closest = np.argsort(_dist)
            # Get Npix closest pixels (ignoring boundary pixels)
            weights = []
            for ipix in range(self.Npix):
                if (not ind_closest[ipix] in self.ind_borders) and (_dist[ind_closest[ipix]]<=self.dmax):
                    weights.append(np.exp(-(_dist[ind_closest[ipix]]**2/(2*(.5*self.dmax)**2))))
                    row.append(iobs)
                    col.append(ind_closest[ipix])
            sum_weights = np.sum(weights)
            # Fill interpolation coefficients 
            for w in weights:
                data.append(w/sum_weights)

        data = jnp.array(data)
        row = jnp.array(row)
        col = jnp.array(col)
        indexes = jnp.column_stack((row,col))

        sparse_matrix = sparse.BCOO((data, indexes), shape=(Nobs, self.coords_geo.shape[0]))

        return sparse_matrix
    
    def misfit(self,State_var,t):

        # Initialization
        misfit = np.array([])

        for name in self.name_var_obs[t]:

            # Get model state
            X = State_var[self.name_mod_var[name]].ravel() 

            # Project model state to obs space
            if name in self.Hop[t]:
                HX = self.Hop[t][name] @ X
            else:
                HX = +X

            # Compute misfit & errors
            _misfit = (HX-self.varobs[t][name])
            _inverr = 1/self.errobs[t][name]
            _misfit = jnp.where(jnp.isnan(_misfit),0,_misfit) 
            _inverr = jnp.where(jnp.isnan(_inverr),0,_inverr) 

            # Concatenate
            misfit = jnp.concatenate((misfit,_inverr*_misfit))

        return misfit
