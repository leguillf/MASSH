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
from scipy.spatial import distance_matrix, cKDTree
import pandas as pd
from jax.experimental import sparse
import jax.numpy as jnp 
from jax import jit
import jax
import datetime
from datetime import datetime 
from joblib import Parallel
from joblib import delayed as jb_delayed

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
    
    if verbose and not config.OBSOP.super is None:
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
            self.ind_mask = np.where(State.mask)
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

        #number of observations 
        self.n_obs = 0 

        # number of workers 
        self.n_workers = config.EXP.n_workers


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

        # misfit normalization 
        self.normalize_misfit = config.OBSOP.normalize_misfit

    ## NEW VERSION OF OBSOP INTERPL3 (PARALLELIZED) ## 
    def process_obs(self, var_bc=None):

        self.var_obs = {}
        self.err_obs = {}
        self.lon_obs = {}
        self.lat_obs = {}
        self.Hop = {}

        data_obs = Parallel(n_jobs=self.n_workers,backend='threading')(jb_delayed(compute_obs_L3)(i=i,
                                                                                                    t=t,
                                                                                                    dict_obs=self.dict_obs[t],
                                                                                                    name_obs=self.name_obs,
                                                                                                    path_save=self.path_save,
                                                                                                    name_H=self.name_H,
                                                                                                    compute_op=self.compute_op,
                                                                                                    write_op=self.write_op,
                                                                                                    coords_car=self.coords_car,
                                                                                                    coords_geo=self.coords_geo,
                                                                                                    Npix=self.Npix,
                                                                                                    ind_borders=self.ind_borders,
                                                                                                    ind_mask=self.ind_mask,
                                                                                                    dmax=self.dmax,
                                                                                                    var_bc=var_bc) for i,t in enumerate(self.date_obs))
        
        for i,t in enumerate(self.date_obs):
            self.lon_obs[t], self.lat_obs[t], self.var_obs[t], self.err_obs[t], self.Hop[t], _n_obs = data_obs[i]

            self.n_obs+=_n_obs

    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])
        misfit_to_save = np.array([])

        #mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Project model state to obs space
            HX = self.Hop[t][name] @ X

            # Compute misfit & errors
            _misfit = (HX-self.var_obs[t][name])
            _inverr = 1/self.err_obs[t][name]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0

            # Concatenate
            misfit = np.concatenate((misfit,_inverr*_misfit))
            misfit_to_save = np.concatenate((misfit_to_save,_inverr*_inverr*_misfit))

            if self.normalize_misfit: 
               misfit/=np.sqrt(self.n_obs/1628021)
               misfit_to_save/=self.n_obs/1628021
        
        return misfit, misfit_to_save

    def adj(self, t, adState, misfit, R):

        for name in self.name_var_obs[t]:

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            #print(name,t,"misfit_shape in grad : ",misfit.shape)
            #print(tprint("misfit shape",misfit.shape)

            # Compute adjoint operation of y = Hx
            adX = self.Hop[t][name].T @ misfit

            # Update adjoint variable
            adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[name])  

def compute_obs_L3(i,t,dict_obs,name_obs,path_save,name_H,compute_op,write_op,coords_car,coords_geo,Npix,ind_borders,ind_mask,dmax,var_bc):
    
    # variables to return # 
    _lon_obs = {}
    _lat_obs = {}
    _var_obs = {}
    _err_obs = {}
    _Hop = {}
    _n_obs = 0

    sat_info_list = dict_obs['attributes']
    obs_file_list = dict_obs['obs_path']
    obs_name_list = dict_obs['obs_name']

    for sat_info,obs_file,obs_name in zip(sat_info_list,obs_file_list,obs_name_list):
        
        if obs_name not in name_obs or sat_info.super not in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:  
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
                if name in _lon_obs:
                    _var_obs[name] = np.concatenate((_var_obs[name],var))
                    _err_obs[name] = np.concatenate((_err_obs[name],err))
                    _lon_obs[name] = np.concatenate((_lon_obs[name],lon))
                    _lat_obs[name] = np.concatenate((_lat_obs[name],lat))
                else:
                    _var_obs[name] = +var
                    _err_obs[name] = +err
                    _lon_obs[name] = +lon
                    _lat_obs[name] = +lat

    for name in _lon_obs:
        coords_obs = np.column_stack((_lon_obs[name], _lon_obs[name]))
        file_L3 = f"{path_save}/{name_H}_{'_'.join(name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
        if var_bc is not None and name in var_bc:
            mask = np.any(np.isnan(coords_geo),axis=1)
            var_bc_interp = griddata(coords_geo[~mask], var_bc[name][i].flatten()[~mask], coords_obs, method='cubic')
            _var_obs[name] -= var_bc_interp

        # Fill dictionnaries
        _var_obs[name] = _var_obs[name]
        _err_obs[name] = _err_obs[name]
        _lon_obs[name] = _lon_obs[name]
        _lat_obs[name] = _lat_obs[name]

        # Compute Sparse operator
        if not compute_op and write_op and os.path.exists(file_L3):
            with open(file_L3, "rb") as f:
                _Hop[name] = pickle.load(f)
        else:
            # Compute operator
            _H = _sparse_op(_lon_obs[name],_lat_obs[name],coords_car,coords_geo,Npix,ind_borders,ind_mask,dmax)
            _Hop[name] = _H
            # Save operator if asked
            if write_op:
                with open(file_L3, "wb") as f:
                    pickle.dump(_H, f)


        #updating number of obs 
        _n_obs+=_var_obs[name].size
    
    return _lon_obs, _lat_obs, _var_obs, _err_obs, _Hop, _n_obs

def _sparse_op(lon_obs,lat_obs,coords_car,coords_geo,Npix,ind_borders,ind_mask,dmax):
        
    coords_geo_obs = np.column_stack((lon_obs, lat_obs))
    coords_car_obs = grid.geo2cart(coords_geo_obs)

    row = [] # indexes of observation grid
    col = [] # indexes of state grid
    data = [] # interpolation coefficients
    Nobs = coords_geo_obs.shape[0]

    for iobs in range(Nobs):
        _dist = cdist(coords_car_obs[iobs][np.newaxis,:], coords_car, metric="euclidean")[0]
        # Npix closest
        ind_closest = np.argsort(_dist)
        # Get Npix closest pixels (ignoring boundary pixels)
        weights = []
        for ipix in range(Npix):
            if (not ind_closest[ipix] in ind_borders) and (not ind_closest[ipix] in ind_mask) and (_dist[ind_closest[ipix]]<=dmax):
                weights.append(np.exp(-(_dist[ind_closest[ipix]]**2/(2*(.5*dmax)**2))))
                row.append(iobs)
                col.append(ind_closest[ipix])
        sum_weights = np.sum(weights)
        # Fill interpolation coefficients 
        for w in weights:
            data.append(w/sum_weights)

    return csc_matrix((data, (row, col)), shape=(Nobs, coords_geo.shape[0]))

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
        misfit_to_save = np.array([])

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
            misfit_to_save = np.concatenate((misfit,_inverr*_inverr*_misfit))
            
        return misfit, misfit_to_save

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
                        #if sat_info.super!='OBS_L4':
                        if sat_info.super not in ["OBS_L4", "OBS_SSH_SWATH"]:
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

        self.dist_min = .5*np.sqrt(State.dx**2+State.dy**2)*1e-3 # Minimum distance to consider an observation inside a model pixel

        self.name_H += f'_L4_{config.OBSOP.interp_method}'


    # NEWER VERSION OF PROCESS_OBS L4 (5TH JUNE 2024) - BETTER PARALLELIZATION #
    def process_obs(self, var_bc=None):

        self.lon_obs = {}
        self.lat_obs = {}
        self.var_obs = {}
        self.err_obs = {}

        # for t in self.date_obs:
        #     self.lon_obs[t], self.lat_obs[t], self.var_obs[t], self.err_obs[t] = compute_obs_L4(t=t,dict_obs=self.dict_obs[t],name_obs=self.name_obs,path_save=self.path_save,
        #                                                                  name_H=self.name_H,compute_op=self.compute_op,write_op=self.write_op,
        #                                                                  coords_geo=self.coords_geo,interp_method=self.interp_method,
        #                                                                  dist_min = self.dist_min, var_bc=var_bc,date_obs=self.date_obs)


        data_obs = Parallel(n_jobs=self.n_workers,backend='threading')(jb_delayed(compute_obs_L4)(t=t,dict_obs=self.dict_obs[t],name_obs=self.name_obs,path_save=self.path_save,
                                                                         name_H=self.name_H,compute_op=self.compute_op,write_op=self.write_op,
                                                                         coords_geo=self.coords_geo,interp_method=self.interp_method,
                                                                         dist_min = self.dist_min, var_bc=var_bc,date_obs=self.date_obs) for t in self.date_obs)
        
        for i,t in enumerate(self.date_obs):
            self.lon_obs[t], self.lat_obs[t], self.var_obs[t], self.err_obs[t] = data_obs[i]

    
    def misfit(self,t,State):

        '''
        Computes the misfit.

                Parameters:
                        t (datetime): time stamp to compute misfit from 
                        State (class State): State class containing the variables 

                Returns:
                        misfit (array): misfit used for cost function in tools_4Dvar.py
                        misfit_to_save (array) : misfit saved for grad function in tools_4Dvar.py
        '''
        
        # Initialization
        misfit = np.array([])
        misfit_to_save = np.array([])

        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Project model state to obs space
            HX = +X

            # Compute misfit & errors
            _misfit = (HX-self.var_obs[t][name])
            _inverr = 1/self.err_obs[t][name]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0

            # Output
            misfit = np.concatenate((misfit,_inverr*_misfit))
            misfit_to_save = np.concatenate((misfit_to_save,_inverr*_inverr*_misfit))

        return misfit, misfit_to_save

    def adj(self, t, adState, misfit, R):

        for name in self.name_var_obs[t]:

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            adX = +misfit

            # Update adjoint variable
            adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[name])


def compute_obs_L4(t,dict_obs,name_obs,path_save,name_H,compute_op,write_op,coords_geo,interp_method,dist_min,var_bc,date_obs):

    """
    
    ARGUMENTS : 
        - t : timestep
        - dict_obs : obs information in a directory 
        - name_obs : observation names that are in observational operator
    
    """

    # variables to return # 
    _lon_obs = {}
    _lat_obs = {}
    _var_obs = {}
    _err_obs = {}

    # variables for satellite info #
    _sat_info_super_obs = {}

    sat_info_list = dict_obs['attributes']
    obs_file_list = dict_obs['obs_path']
    obs_name_list = dict_obs['obs_name']

    for sat_info,obs_file,obs_name in zip(sat_info_list,obs_file_list,obs_name_list):

        
        #if sat_info.super!='OBS_L4':
        if obs_name not in name_obs or sat_info.super not in ["OBS_L4", "OBS_SSH_SWATH"]:

            continue

        ####################
        # Merge observations
        ####################

        with xr.open_dataset(obs_file) as ncin:
            lon = ncin[sat_info['name_lon']].values.ravel() 
            lat = ncin[sat_info['name_lat']].values.ravel()

            for name in sat_info['name_var']:
                # Observed information
                _sat_info_super_obs[name] = sat_info.super
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
                if name in _lon_obs:
                    _var_obs[name] = np.concatenate((_var_obs[name],var))
                    _err_obs[name] = np.concatenate((_err_obs[name],err))
                    _lon_obs[name] = np.concatenate((_lon_obs[name],lon))
                    _lat_obs[name] = np.concatenate((_lat_obs[name],lat))
                else:
                    _var_obs[name] = +var
                    _err_obs[name] = +err
                    _lon_obs[name] = +lon
                    _lat_obs[name] = +lat    
        
    for name in _lon_obs:

        # check if obsop have been previously saved for this timestep # 
        file_L4 = f"{path_save}/{name_H}_{'_'.join(name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
        if not compute_op and os.path.exists(file_L4):
            with open(file_L4, "rb") as f:
                _var_obs[name], _err_obs[name] = pickle.load(f)

        else : 

            _var_obs[name] = grid_interp(_lon_obs[name],_lat_obs[name],_var_obs[name],coords_geo,interp_method,dist_min,_sat_info_super_obs[name])
            _err_obs[name] = grid_interp(_lon_obs[name],_lat_obs[name],_err_obs[name],coords_geo,interp_method,dist_min,_sat_info_super_obs[name])
            
            if write_op:
                file_L4 = f"{path_save}/{name_H}_{'_'.join(name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
                with open(file_L4, "wb") as f:
                    pickle.dump((_var_obs[name],_err_obs[name]), f)
            if var_bc is not None and name in var_bc:
                _var_obs[name] -= var_bc[name][date_obs.index(t)].flatten()

    return _lon_obs, _lat_obs, _var_obs, _err_obs

def grid_interp(_lon_obs,_lat_obs,_array_obs,_coords_geo,interp_method,dist_min,sat_info_super):

    """
        NAME
            grid_interp
    
        DESCRIPTION
            Function to interpolate obsevrations on L4 grid. It is defined as a function outside the class because it used with joblib for parallelisation. 

            Args:
                _lon_obs (array): longitude of observations 
                _lat_obs (array) : latitude of observations 
                _array_obs (array) : values of observations 
                _coords_geo (array) : coordinates onto which values need to be interpolated 
                interp_method (str) : interpolation method 
                
    """     

    _coords_obs = np.column_stack((_lon_obs, _lat_obs))
    
    if interp_method == "hybrid":

        _array_obs_interp = griddata(_coords_obs,_array_obs, _coords_geo, method="nearest") # first interpolation with "nearest" method
        _array_obs_interp_linear = griddata(_coords_obs,_array_obs, _coords_geo, method="linear") # second interpolation with "linear" method
        _array_obs_interp_cubic = griddata(_coords_obs, _array_obs, _coords_geo, method="cubic") # third interpolation with "cubic" method

        _array_obs_interp[~np.isnan(_array_obs_interp_linear)]=_array_obs_interp_linear[~np.isnan(_array_obs_interp_linear)] # filling out first interpolation with all available values of second method 
        _array_obs_interp[~np.isnan(_array_obs_interp_cubic)] = _array_obs_interp_cubic[~np.isnan(_array_obs_interp_cubic)] # filling out first interpolation with all available values of third method 
    
    else : 
        
        _array_obs_interp = griddata(_coords_obs,_array_obs,_coords_geo,method=interp_method)

    # Special processing for SWOT data (Masking pixels outside of swath) #
    if sat_info_super == "OBS_SSH_SWATH":

        obs_tree = cKDTree(grid.geo2cart(_coords_obs))
        mod_tree = cKDTree(grid.geo2cart(_coords_geo))
        dist_mx = mod_tree.sparse_distance_matrix(obs_tree, dist_min)
        keys = np.array(list(dist_mx.keys()))
        ind_mod_in = keys[:, 0] # Index of model grid inside the swath
        mask_mod_in = np.ones_like(_array_obs_interp, dtype=bool) # mask to be applied on interpoled fields
        mask_mod_in[ind_mod_in] = 0
        _array_obs_interp[mask_mod_in] = np.nan

    return _array_obs_interp

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
        misfit_to_save = {}

        for i,_Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs(t):
                _misfit, _misfit_to_save = _Obsop.misfit(t,State)
                misfit = np.concatenate((misfit,_misfit))
                misfit_to_save[i] = _misfit_to_save
        
        return misfit, misfit_to_save

        # FORMER VERSION OF MISFIT FOR OBSOP MULTI # 
        # misfit = np.array([])

        # for _Obsop in self.Obsop:
        #     if _Obsop.is_obs(t):
        #         _misfit = _Obsop.misfit(t,State)
        #         misfit = np.concatenate((misfit,_misfit))
        
        # return misfit


    def adj(self, t, adState, misfits, R):
    
        for i,_Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs(t):
                _Obsop.adj(t,adState,misfits[i],R)

    
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


