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
import matplotlib.pylab as plt
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
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
    
    if config.OBSOP.super=='OBSOP_INTERP':
        return Obsop_interp(config,State,dict_obs,Model)
    elif config.OBSOP.super=='OBSOP_INTERP_JAX':
        return Obsop_interp_jax(config,State,dict_obs,Model)
    
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')


class Obsop_interp:

    def __init__(self,config,State,dict_obs,Model):
        
        self.compute_H = config.OBSOP.compute_op
        
        # Pattern for saving files
        date1 = config.EXP.init_date.strftime('%Y%m%d')
        date2 = config.EXP.final_date.strftime('%Y%m%d')
        box = f'{int(State.lon_min)}_{int(State.lon_max)}_{int(State.lat_min)}_{int(State.lat_max)}'
        self.name_H = f'H_{"_".join(config.OBS.keys())}_{date1}_{date2}_{box}_{int(State.dx)}_{int(State.dy)}_{config.OBSOP.Npix}'

        # Temporary path where to save H operators
        self.tmp_DA_path = config.EXP.tmp_DA_path
        if config.OBSOP.path_save is not None:
            # We'll save to *path_H* or read in *path_H* from previous run
            self.path_H = config.OBSOP.path_save
            self.read_H = True
            if not os.path.exists(self.path_H):
                os.makedirs(self.path_H)
        else:
            # We'll use temporary directory to read/save the files
            self.path_H = self.tmp_DA_path
            if self.compute_H:
                self.read_H = False
            else:
                self.read_H = True
        
        # Date obs
        self.date_obs = []
        self.name_var_obs = {}
        t_obs = [tobs for tobs in dict_obs.keys()] 
        for t in Model.timestamps:
            delta_t = [(t - tobs).total_seconds() for tobs in dict_obs.keys()]
            if len(delta_t)>0:
                t_obs = [tobs for tobs in dict_obs.keys()] 
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    ind_obs = np.argmin(np.abs(delta_t))
                    self.date_obs.append(t_obs[ind_obs])
                    # Get obs variable names (SSH,U,V,SST...) at this time
                    self.name_var_obs[t] = []
                    for sat_info in dict_obs[t_obs[ind_obs]]['satellite']:
                        for name in sat_info['name_var']:
                            if name not in self.name_var_obs[t]:
                                self.name_var_obs[t].append(name)
        
        # Model variable
        self.name_mod_var = Model.name_var
        
        # For grid interpolation:
        self.interp_method = config.OBSOP.interp_method
        self.Npix = config.OBSOP.Npix
        lon = +State.lon
        lat = +State.lat
        self.coords_geo = np.column_stack((lon.ravel(), lat.ravel()))
        self.coords_car = grid.geo2cart(self.coords_geo)
        self.dmax = self.Npix*np.mean(np.sqrt(State.DX**2 + State.DY**2))*1e-3*np.sqrt(2)/2 # maximal distance for space interpolation

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

        return coo_matrix((data, (row, col)), shape=(Nobs, self.coords_geo.shape[0]))

    def process_obs(self, var_bc=None):

        self.varobs = {}
        self.errobs = {}
        self.Hop = {}

        for i,t in enumerate(self.date_obs):

            self.varobs[t] = {}
            self.errobs[t] = {}
            self.Hop[t] = {}

            sat_info_list = self.dict_obs[t]['satellite']
            obs_file_list = self.dict_obs[t]['obs_name']

        
            # Concatenate obs from different sensors
            lon_obs = {}
            lat_obs = {}
            var_obs = {}
            err_obs = {}
            is_full = {}
            for sat_info,obs_file in zip(sat_info_list,obs_file_list):

                with xr.open_dataset(obs_file) as ncin:
                    lon = ncin[sat_info['name_lon']].values.ravel() %360
                    lat = ncin[sat_info['name_lat']].values.ravel()

                    for name in sat_info['name_var']:
                        if sat_info.super=='OBS_MODEL':
                            is_full[name] = True # We'll interpolate the observations to the state grid to accelerate the misfit computations
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
                if name in is_full:
                    # Grid interpolation: performing spatial interpolation now
                    var_obs_interp = griddata(coords_obs, var_obs[name], self.coords_geo, method=self.interp_method)
                    err_obs_interp = griddata(coords_obs, err_obs[name], self.coords_geo, method=self.interp_method)

                    if var_bc is not None and name in var_bc:
                        var_obs_interp -= var_bc[name][i].flatten()

                    # Fill dictionnaries
                    self.varobs[t][name] = var_obs_interp
                    self.errobs[t][name] = err_obs_interp

                else:
                    if var_bc is not None and name in var_bc:
                        mask = np.any(np.isnan(self.coords_geo),axis=1)
                        var_bc_interp = griddata(self.coords_geo[~mask], var_bc[name][i].flatten()[~mask], coords_obs, method='cubic')
                        var_obs[name] -= var_bc_interp

                    # Fill dictionnaries
                    self.varobs[t][name] = var_obs[name]
                    self.errobs[t][name] = err_obs[name]
                    # Compute Sparse operator 
                    self.Hop[t][name] = self._sparse_op(lon_obs[name],lat_obs[name])
                
    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 


            # Project model state to obs space
            if name in self.Hop[t]:
                HX = self.Hop[t][name] @ X
            else:
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
                os.path.join(self.tmp_DA_path,f"misfit_{t.strftime('%Y%m%d_%H%M')}.nc"), 
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
                os.path.join(self.tmp_DA_path,f"misfit_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit = ds['misfit'].values
            ds.close()
            del ds

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            if name in self.Hop[t]:
                adX = self.Hop[t][name].T @ misfit
            else:
                adX = +misfit

            # Update adjoint variable
            adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[name])      



        
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
