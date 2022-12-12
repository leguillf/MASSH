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

def Obsop(config, State, dict_obs, Model, *args, **kwargs):
    """
    NAME
        basis

    DESCRIPTION
        Main function calling obsopt for specific observational operators
    """
    
    if config.OBSOP is None:
        return 
    
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
        self.date_obs = {}
        self.name_var_obs = {}
        t_obs = [tobs for tobs in dict_obs.keys()] 
        for t in Model.timestamps:
            delta_t = [(t - tobs).total_seconds() for tobs in dict_obs.keys()]
            if len(delta_t)>0:
                t_obs = [tobs for tobs in dict_obs.keys()] 
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    ind_obs = np.argmin(np.abs(delta_t))
                    self.date_obs[t] = t_obs[ind_obs]
                    # Get obs variable names (SSH,U,V,SST...) at this time
                    self.name_var_obs[t] = []
                    for sat_info in dict_obs[self.date_obs[t]]['satellite']:
                        for name in sat_info['name_var']:
                            if name not in self.name_var_obs[t]:
                                self.name_var_obs[t].append(name)
        
        # Model variable
        self.name_mod_var = Model.name_var
        
        # For grid interpolation:
        self.Npix = config.OBSOP.Npix
        coords_geo = np.column_stack((State.lon.ravel(), State.lat.ravel()))
        self.coords_car = grid.geo2cart(coords_geo)
        #self.coords_car = coords_car[~np.isnan(coords_car)]
        
        # Mask boundary pixels
        self.ind_borders = []
        if config.OBSOP.mask_borders:
            coords_geo_borders = np.column_stack((
                np.concatenate((State.lon[0,:],State.lon[1:-1,-1],State.lon[-1,:],State.lon[:,0])),
                np.concatenate((State.lat[0,:],State.lat[1:-1,-1],State.lat[-1,:],State.lat[:,0]))
                ))
            if len(coords_geo_borders)>0:
                for i in range(coords_geo.shape[0]):
                    if np.any(np.all(np.isclose(coords_geo_borders,coords_geo[i]), axis=1)):
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
        for t in self.date_obs:
            self.process_obs(
                t,
                dict_obs[self.date_obs[t]]['satellite'],
                dict_obs[self.date_obs[t]]['obs_name']
                )


            
    def process_obs(self,t,sat_info_list,obs_file_list):

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
        for sat_info,obs_file in zip(sat_info_list,obs_file_list):
            with xr.open_dataset(obs_file) as ncin:
                lon = ncin[sat_info['name_lon']].values.ravel()
                lat = ncin[sat_info['name_lat']].values.ravel()
                for name in sat_info['name_var']:
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
        
        # Compute indexes, weights and masks for spatial interpolations 
        indexes = {}
        weights = {}
        maskobs = {}
        for name in lon_obs:
            _indexes, _weights = self.interpolator(lon_obs[name],lat_obs[name])
            _maskobs = np.isnan(lon_obs[name])*np.isnan(lat_obs[name])
            if self.flag_mask_coast:
                coords_geo_obs = np.column_stack((lon_obs[name],lat_obs[name]))
                coords_car_obs = grid.geo2cart(coords_geo_obs)
                for i in range(lon_obs.size):
                    _dist = np.min(np.sqrt(np.sum(np.square(coords_car_obs[i]-self.coords_car_land),axis=1)))
                    if _dist<self.dist_coast:
                        _maskobs[i] = True
            indexes[name] = _indexes
            weights[name] = _weights
            maskobs[name] = _maskobs
        
        # Write in netcdf
        for name in lon_obs:
            
            dsout = xr.Dataset(
                {
                    "var_obs": (("Nobs"), var_obs[name]),
                    "err_obs": (("Nobs"), err_obs[name]),
                    "indexes": (("Nobs","Npix"), indexes[name]),
                                "weights": (("Nobs","Npix"), weights[name]),
                                "maskobs": (("Nobs"), maskobs[name])},                
                               )
            dsout.to_netcdf(file_H, group=name,
                encoding={'indexes': {'dtype': 'int16'}})
            dsout.close()

        if self.read_H:
            new_file_H = os.path.join(self.tmp_DA_path,name_file_H)
            if file_H!=new_file_H:
                os.system(f"cp {file_H} {new_file_H}")
                        
        return t
    
    def interpolator(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        indexes = []
        weights = []
        for iobs in range(lon_obs.size):
            _dist = np.sqrt(np.sum(np.square(coords_car_obs[iobs]-self.coords_car),axis=1))
            # Npix closest
            ind_closest = np.argsort(_dist)
            # Get Npix closest pixels (ignoring boundary pixels)
            n = 0
            i = 0
            ind = []
            w = []
            while n<self.Npix:
                if ind_closest[i] in self.ind_borders:
                    #Ignoring boundary pixels 
                    w.append(0.)
                else:
                    w.append(1/_dist[ind_closest[i]])
                ind.append(ind_closest[i])
                n += 1
                i +=1 
            indexes.append(ind)
            weights.append(w)   
            
        return np.asarray(indexes),np.asarray(weights)
     
    def H(self,t,X,indexes,weights,maskobs):
        
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
        inv_err2 = np.array([])

        for name in self.name_var_obs[t]:

            # Read obs
            ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,f"{self.name_H}_{t.strftime('%Y%m%d_%H%M.nc')}"), group=name)

            # Get obs values, errors and indexes & weights of neighbour grid pixels
            var_obs = ds['var_obs'].values
            err_obs = ds['err_obs'].values
            indexes = ds['indexes'].values
            weights = ds['weights'].values
            maskobs = ds['maskobs'].values

            # Get model state
            X = State.getvar(self.name_mod_var[name]).ravel() 

            # Project model state to obs space
            HX = self.H(t,X,indexes,weights,maskobs)

            # Compute misfit
            misfit = np.concatenate((misfit,HX-var_obs))
            inv_err2 = np.concatenate((inv_err2,err_obs**(-2)))

        # Remove nan
        misfit[np.isnan(misfit)] = 0
        
        # Save misfit
        with open(
            os.path.join(self.tmp_DA_path,f"misfit_{t.strftime('%Y%m%d_%H%M')}.pic"),'wb') as f:
            pickle.dump((misfit,inv_err2),f)

        return misfit,inv_err2
    
    def load_misfit(self,t):
        
        with open(
            os.path.join(self.tmp_DA_path,f"misfit_{t.strftime('%Y%m%d_%H%M')}.pic"),'rb') as f:
            misfit,inv_err2 = pickle.load(f)
        
        return misfit,inv_err2

    def adj(self,t,adState,misfit):

        for name in self.name_var_obs[t]:

            # get adjoint model variable
            advar = adState.getvar(self.name_mod_var[name])

            # Read obs
            ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc')), group=name)

            # Get indexes & weights of neighbour grid pixels
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

            # Update adjoint variable
            adState.setvar(advar + adH.reshape(advar.shape), self.name_mod_var[name])
            

        
      