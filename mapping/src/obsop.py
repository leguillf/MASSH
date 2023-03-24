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
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd

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
        self.obs_sparse = {}
        
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
        self.Npix = config.OBSOP.Npix
        self.coords_geo = np.column_stack((State.lon.ravel(), State.lat.ravel()))
        self.coords_car = grid.geo2cart(self.coords_geo)
        self.dmax = self.Npix*np.mean(np.sqrt(State.DX**2 + State.DY**2))*1e-3*np.sqrt(2)/2 # maximal distance for space interpolation
        
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
        
        # Mask coast pixels
        self.dist_coast = config.H_dist_coast
        if config.OBSOP.mask_coast and self.dist_coast is not None and State.mask is not None and np.any(State.mask):
            self.flag_mask_coast = True
            lon_land = State.lon[State.mask].ravel()
            lat_land = State.lat[State.mask].ravel()
            coords_geo_land = np.column_stack((lon_land,lat_land))
            self.coords_car_land = grid.geo2cart(coords_geo_land)
            
        else: self.flag_mask_coast = False
        
        # Process obs
        self.dict_obs = dict_obs
        

            
    def process_obs(self, var_bc=None):

        for i,t in enumerate(self.date_obs):

            sat_info_list = self.dict_obs[t]['satellite']
            obs_file_list = self.dict_obs[t]['obs_name']

            name_file_H = f"{self.name_H}_{t.strftime('%Y%m%d_%H%M.nc')}"

            if self.read_H:
                file_H = os.path.join(self.path_H,name_file_H)
                if os.path.exists(file_H) and not self.compute_H:
                    new_file_H = os.path.join(self.tmp_DA_path,name_file_H)
                    if new_file_H != file_H:
                        os.system(f"cp {file_H} {new_file_H}")
                    self.obs_sparse[t] = True
                    return t
            else:
                file_H = os.path.join(self.tmp_DA_path,name_file_H)
            
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
                            is_full[name] = True
                        var = ncin[name].values.ravel()
                        if sat_info['sigma_noise'] is not None:
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
            
            mode = 'w'
            for name in lon_obs:
                coords_obs = np.column_stack((lon_obs[name], lat_obs[name]))
                if name in is_full:
                    # Grid interpolation: performing spatial interpolation now
                    var_obs_interp = griddata(coords_obs, var_obs[name], self.coords_geo, method='cubic')
                    err_obs_interp = griddata(coords_obs, err_obs[name], self.coords_geo, method='cubic')

                    if var_bc is not None and name in var_bc:
                        var_obs_interp -= var_bc[name][i].flatten()
                        
                    # Write in netcdf
                    dsout = xr.Dataset(
                        { "var_obs": (("Nobs"), var_obs_interp),
                        "err_obs": (("Nobs"), err_obs_interp)})
                    dsout.to_netcdf(file_H, mode=mode, group=name)
                    dsout.close()
                    mode = 'a'
                else:
                    # Sparse interpolation: compute indexes, weights and masks 
                    indexes, weights = self.interpolator(lon_obs[name],lat_obs[name])
                    maskobs = np.isnan(lon_obs[name])*np.isnan(lat_obs[name])
                    if self.flag_mask_coast:
                        coords_geo_obs = np.column_stack((lon_obs[name],lat_obs[name]))
                        coords_car_obs = grid.geo2cart(coords_geo_obs)
                        for i in range(lon_obs.size):
                            _dist = np.min(np.sqrt(np.sum(np.square(coords_car_obs[i]-self.coords_car_land),axis=1)))
                            if (self.flag_mask_coast and _dist<self.dist_coast):
                                maskobs[i] = True
                
                    if var_bc is not None and name in var_bc:
                        mask = np.any(np.isnan(self.coords_geo),axis=1)
                        var_bc_interp = griddata(self.coords_geo[~mask], var_bc[name][i].flatten()[~mask], coords_obs, method='cubic')
                        var_obs[name] -= var_bc_interp

                    # Write in netcdf
                    dsout = xr.Dataset(
                        {
                            "var_obs": (("Nobs"), var_obs[name]),
                            "err_obs": (("Nobs"), err_obs[name]),
                            "indexes": (("Nobs","Npix"), indexes),
                                        "weights": (("Nobs","Npix"), weights),
                                        "maskobs": (("Nobs"), maskobs)},                
                                    )
                    dsout.to_netcdf(file_H, mode=mode, group=name,
                        encoding={'indexes': {'dtype': int}})
                    dsout.close()
                    mode = 'a'
                    
            if self.read_H:
                new_file_H = os.path.join(self.tmp_DA_path,name_file_H)
                if file_H!=new_file_H:
                    os.system(f"cp {file_H} {new_file_H}")

    

    def interpolator(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        indexes = np.zeros((lon_obs.size,self.Npix),dtype=int)
        weights = np.zeros((lon_obs.size,self.Npix))
        for iobs in range(lon_obs.size):
            _dist = cdist(coords_car_obs[iobs][np.newaxis,:], self.coords_car, metric="euclidean")[0]
            # Npix closest
            ind_closest = np.argsort(_dist)
            # Get Npix closest pixels (ignoring boundary pixels)
            for ipix in range(self.Npix):
                if (not ind_closest[ipix] in self.ind_borders) and (_dist[ind_closest[ipix]]<=self.dmax):
                    #Ignoring boundary pixels 
                    weights[iobs,ipix] = np.exp(-(_dist[ind_closest[ipix]]**2/(2*(.5*self.dmax)**2)))
                    indexes[iobs,ipix] = ind_closest[ipix]

        return indexes,weights
     
    def H(self,X,indexes,weights,maskobs):
        
        # Compute inerpolation of X to obs space
        HX = np.zeros(indexes.shape[0])
        
        for i,(mask,ind,w) in enumerate(zip(maskobs,indexes,weights)):
            if not mask:
                # Average
                if ind.size>1:
                    try:
                        HX[i] = np.average(X[ind],weights=w)
                    except:
                        HX[i] = np.nan
                else:
                    HX[i] = X[ind[0]]
            else:
                HX[i] = np.nan
        
        return HX

    def misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Read obs
            ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,f"{self.name_H}_{t.strftime('%Y%m%d_%H%M.nc')}"), group=name)

            # Get obs values & errors
            var_obs = ds['var_obs'].values
            err_obs = ds['err_obs'].values

            if 'indexes' in ds:
                # Get obs indexes & weights of neighbour grid pixels
                indexes = ds['indexes'].values
                weights = ds['weights'].values
                maskobs = ds['maskobs'].values

                # Project model state to obs space
                HX = self.H(X,indexes,weights,maskobs)
            else:
                # Observations are already interpolated onto the model grid
                HX = +X

            # Compute misfit & errors
            _misfit = (HX-var_obs)
            _inverr = 1/err_obs
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

            # Read observational operator
            ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc')), 
                group=name)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            if 'indexes' in ds:
                # Get obs indexes & weights of neighbour grid pixels
                indexes = ds['indexes'].values
                weights = ds['weights'].values
                maskobs = ds['maskobs'].values

                # Project misfit to model space
                adH = np.zeros(advar.size)
                Nobs,Npix = indexes.shape
                for i in range(Nobs):
                    if not maskobs[i]:
                        # Average
                        for j in range(Npix):
                            if weights[i].sum()!=0:
                                adH[indexes[i,j]] += weights[i,j]*misfit[i]/(weights[i].sum())
            else:
                adH = +misfit

            # Update adjoint variable
            adState.setvar(advar + adH.reshape(advar.shape), self.name_mod_var[name])
            

        
      