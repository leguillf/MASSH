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
        self.shape_grid = [State.ny, State.nx]
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
                        if sat_info.super not in ['OBS_L4', 'OBS_SSH_SWATH']:
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

        self.DX = State.DX
        self.DY = State.DY

        self.DX = State.DX
        self.DY = State.DY

        # Misfit on gradients
        self.gradients = config.OBSOP.gradients
        if self.gradients:
            self.name_H += f'_L4_grad_{config.OBSOP.interp_method}'
        else:
            self.name_H += f'_L4_{config.OBSOP.interp_method}'

    def process_obs(self, var_bc=None):

        self.varobs = {}
        self.errobs = {}

        #############################
        # Loop on observation dates #
        #############################
        for i,t in enumerate(self.date_obs):

            self.varobs[t] = {}
            self.errobs[t] = {}

            sat_info_list = self.dict_obs[t]['attributes']
            obs_file_list = self.dict_obs[t]['obs_path']

            # Concatenate obs from different sensors
            lon_obs = {}
            lat_obs = {}
            var_obs = {}
            err_obs = {}
            type_obs = {}

            for sat_info,obs_file in zip(sat_info_list,obs_file_list):

                # Check if this observation class is wanted
                if sat_info.super not in ['OBS_L4', 'OBS_SSH_SWATH']:
                    continue
                
                ####################
                # Merge observations
                ####################
                with xr.open_dataset(obs_file) as ncin:
                    lon = ncin[sat_info['name_lon']].values
                    lat = ncin[sat_info['name_lat']].values
                    for name in sat_info['name_var']:
                        # Observed variable
                        var = ncin[name].values 
                        # Observed error
                        name_err = name + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)                        

                        # Append to lists
                        if name in lon_obs:
                            var_obs[name].append(+var)
                            err_obs[name].append(+err)
                            lon_obs[name].append(+lon)
                            lat_obs[name].append(+lat)
                        else:
                            var_obs[name] = [+var]
                            err_obs[name] = [+err]
                            lon_obs[name] = [+lon]
                            lat_obs[name] = [+lat]
                    
            
            for name in lon_obs:
                ################
                # Process L4 obs
                ################
                file_L4 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{t.strftime('%Y%m%d_%H%M')}_{name}.pic"
                # Check if spatial interpolations have already been performed
                if not self.compute_op and self.write_op and os.path.exists(file_L4):
                    with open(file_L4, "rb") as f:
                        var_obs_interp, err_obs_interp = pickle.load(f)
                else:
                    # Grid interpolation: performing spatial interpolation now
                    # Loop on different obs for this date and this variable name
                    var_obs_interp = np.zeros([len(var_obs[name]),]+self.shape_grid)
                    err_obs_interp = np.zeros([len(var_obs[name]),]+self.shape_grid)
                    for iobs in range(len(var_obs[name])):
                        _coords_obs = np.column_stack((lon_obs[name][iobs].flatten(), lat_obs[name][iobs].flatten()))
                        if self.interp_method=='hybrid':
                            # We perform first nearest, then linear, and then cubic interpolations
                            _var_obs_interp = griddata(_coords_obs, var_obs[name][iobs].flatten(), self.coords_geo, method='nearest')
                            _err_obs_interp = griddata(_coords_obs, err_obs[name][iobs].flatten(), self.coords_geo, method='nearest')
                            _var_obs_interp_linear = griddata(_coords_obs, var_obs[name][iobs].flatten(), self.coords_geo, method='linear')
                            _err_obs_interp_linear = griddata(_coords_obs, err_obs[name][iobs].flatten(), self.coords_geo, method='linear')
                            _var_obs_interp[~np.isnan(_var_obs_interp_linear)] = _var_obs_interp_linear[~np.isnan(_var_obs_interp_linear)]
                            _err_obs_interp[~np.isnan(_err_obs_interp_linear)] = _err_obs_interp_linear[~np.isnan(_err_obs_interp_linear)]
                            _var_obs_interp_cubic = griddata(_coords_obs, var_obs[name][iobs].flatten(), self.coords_geo, method='cubic')
                            _err_obs_interp_cubic = griddata(_coords_obs, err_obs[name][iobs].flatten(), self.coords_geo, method='cubic')
                            _var_obs_interp[~np.isnan(_var_obs_interp_cubic)] = _var_obs_interp_linear[~np.isnan(_var_obs_interp_cubic)]
                            _err_obs_interp[~np.isnan(_err_obs_interp_cubic)] = _err_obs_interp_linear[~np.isnan(_err_obs_interp_cubic)]
                        else:
                            _var_obs_interp = griddata(_coords_obs, var_obs[name][iobs].flatten(), self.coords_geo, method=self.interp_method)
                            _err_obs_interp = griddata(_coords_obs, err_obs[name][iobs].flatten(), self.coords_geo, method=self.interp_method)
                        
                        # Add error due to interpolation (resolutions ratio)
                        dx,dy = grid.lonlat2dxdy(lon_obs[name][iobs],lat_obs[name][iobs])
                        dx = griddata(_coords_obs, dx.flatten(), self.coords_geo)
                        dy = griddata(_coords_obs, dy.flatten(), self.coords_geo)
                        _err_res = (dx * dy) / (self.DX * self.DY).flatten()
                        _err_res = np.where(_err_res<1,1,_err_res)
                        _err_obs_interp *= _err_res
                    
                        var_obs_interp[iobs] = _var_obs_interp.reshape(self.shape_grid)
                        err_obs_interp[iobs] = _err_obs_interp.reshape(self.shape_grid)
                        
                    # Save operator if asked
                    if self.write_op:
                        with open(file_L4, "wb") as f:
                            pickle.dump((var_obs_interp,err_obs_interp), f)

                if var_bc is not None and name in var_bc:
                    var_obs_interp -= var_bc[name][i].flatten()
                
                if self.gradients:
                     # Compute gradients
                    var_obs_interp_grady = np.zeros_like(var_obs_interp)*np.nan
                    var_obs_interp_gradx = np.zeros_like(var_obs_interp)*np.nan
                    var_obs_interp_grady[:,1:-1,1:-1] = (var_obs_interp[:,2:,1:-1] - var_obs_interp[:,:-2,1:-1]) / (2 * self.DY[np.newaxis,1:-1,1:-1])
                    var_obs_interp_gradx[:,1:-1,1:-1] = (var_obs_interp[:,1:-1,2:] - var_obs_interp[:,1:-1,:-2]) / (2 * self.DX[np.newaxis,1:-1,1:-1])

                    # Fill dictionnaries
                    self.varobs[t][name+'_grady'] = var_obs_interp_grady
                    self.varobs[t][name+'_gradx'] = var_obs_interp_gradx
                    self.errobs[t][name] = .5* err_obs_interp /  (self.DY[np.newaxis,:,:]**2 + self.DX[np.newaxis,:,:]**2)**.5
                else:
                    # Fill dictionnaries
                    self.varobs[t][name] = var_obs_interp
                    self.errobs[t][name] = err_obs_interp
                
    def misfit(self,t,State):
        if self.gradients:
            return self._misfit_grad(t, State)
        else:
            return self._misfit(t, State)

    def _misfit(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name])

            # Project model state to obs space
            HX = +X[np.newaxis,:,:]

            # Compute misfit & errors
            _misfit = (HX-self.varobs[t][name])
            _inverr = 1/self.errobs[t][name]
            _misfit[np.isnan(_misfit)] = 0
            _inverr[np.isnan(_inverr)] = 0

            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit": (('Nobs','Ny','Nx'), _misfit),
                    "inverr": (('Nobs','Ny','Nx'), _inverr),
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
            for iobs in range(_misfit.shape[0]):
                misfit = np.concatenate((misfit,(_inverr[iobs]*_misfit[iobs]).flatten()))

        return misfit
    
    def _misfit_grad(self,t,State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state
            X = State.getvar(self.name_mod_var[name])

            # Compute gradients
            HX_grady = np.zeros_like(self.DY)
            HX_gradx = np.zeros_like(self.DY)
            HX_grady[1:-1,1:-1] = ((X[2:,1:-1] - X[:-2,1:-1]) / (2 * self.DY[1:-1,1:-1]))
            HX_gradx[1:-1,1:-1] = ((X[1:-1,2:] - X[1:-1,:-2]) / (2 * self.DX[1:-1,1:-1]))
            HX_grady = HX_grady[np.newaxis,:,:]
            HX_gradx = HX_gradx[np.newaxis,:,:]

            # Compute misfit & errors
            _misfit_grady = (HX_grady-self.varobs[t][name+'_grady']) 
            _misfit_gradx = (HX_gradx-self.varobs[t][name+'_gradx']) 
            _inverr = 1/self.errobs[t][name]
            _misfit_grady[np.isnan(_misfit_grady)] = 0
            _misfit_gradx[np.isnan(_misfit_gradx)] = 0
            _inverr[np.isnan(_inverr)] = 0
        
            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit_grady": (('Nobs','Ny','Nx'), _misfit_grady),
                    "misfit_gradx": (('Nobs','Ny','Nx'), _misfit_gradx),
                    "inverr" : (('Nobs','Ny','Nx'), _inverr)
                    }
                    )
            dsout.to_netcdf(
                os.path.join(self.tmp_DA_path,f"misfit_L4_grad_{t.strftime('%Y%m%d_%H%M')}.nc"), 
                mode=mode, 
                group=name
                )
            dsout.close()
            mode = 'a'

            # Concatenate
            for iobs in range(_inverr.shape[0]):
                misfit = np.concatenate((misfit,(_inverr[iobs]*_misfit_grady[iobs]).flatten(),(_inverr[iobs]*_misfit_gradx[iobs]).flatten()))

        return misfit
    
    def adj(self, t, adState, R):

        if self.gradients:
            return self._adj_grad(t, adState, R)
        else:
            return self._adj(t, adState, R)

    def _adj(self, t, adState, R):

        for name in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L4_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit = ds['misfit'].values
            inverr = ds['inverr'].values
            ds.close()
            del ds

            # Apply R operator
            misfit = R.inv(misfit)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            for iobs in range(misfit.shape[0]):
                advar += (inverr[iobs]* inverr[iobs] * misfit[iobs])

            # Update adjoint variable
            adState.setvar(advar, self.name_mod_var[name])      
    
    def _adj_grad(self, t, adState, R):

        for name in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L4_grad_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit_grady = ds['misfit_grady'].values
            misfit_gradx = ds['misfit_gradx'].values
            inverr = ds['inverr'].values
            ds.close()
            del ds

            # Apply R operator
            misfit_grady = R.inv(misfit_grady)
            misfit_gradx = R.inv(misfit_gradx)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = Hx
            for iobs in range(inverr.shape[0]):
                advar[2:,1:-1] += inverr[iobs,1:-1,1:-1]* inverr[iobs,1:-1,1:-1] * misfit_grady[iobs,1:-1,1:-1] / (2 * self.DY[1:-1,1:-1])
                advar[:-2,1:-1] += -inverr[iobs,1:-1,1:-1]* inverr[iobs,1:-1,1:-1] * misfit_grady[iobs,1:-1,1:-1] / (2 * self.DY[1:-1,1:-1])
                advar[1:-1,2:] += inverr[iobs,1:-1,1:-1]* inverr[iobs,1:-1,1:-1] * misfit_gradx[iobs,1:-1,1:-1] / (2 * self.DX[1:-1,1:-1])
                advar[1:-1,:-2] += -inverr[iobs,1:-1,1:-1]* inverr[iobs,1:-1,1:-1] * misfit_gradx[iobs,1:-1,1:-1] / (2 * self.DX[1:-1,1:-1])

            # Update adjoint variable
            adState.setvar(advar, self.name_mod_var[name])      

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
