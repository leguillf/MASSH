#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
from .config import USE_FLOAT64
import os,sys
import xarray as xr 
import numpy as np 
from src import grid as grid
import pickle
import matplotlib.pylab as plt
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import pyinterp
from jax.experimental import sparse
import jax.numpy as jnp 
from jax.lax import dynamic_slice
from jax import jit
import jax

jax.config.update("jax_enable_x64", USE_FLOAT64)



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
    
    elif config.OBSOP.super=='OBSOP_INTERP_L3_JAX':
        return Obsop_interp_l3_jax(config, State, dict_obs, Model)
    
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
        if config.EXP.lon_obs_max is not None:
            lon_obs_max = config.EXP.lon_obs_max
        else:
            lon_obs_max = State.lon_max

        if config.EXP.lon_obs_min is not None:
            lon_obs_min = config.EXP.lon_obs_min
        else:
            lon_obs_min = State.lon_min
        
        if config.EXP.lat_obs_max is not None:
            lat_obs_max = config.EXP.lat_obs_max
        else:
            lat_obs_max = State.lat_max

        if config.EXP.lat_obs_min is not None:
            lat_obs_min = config.EXP.lat_obs_min
        else:
            lat_obs_min = State.lat_min

        box = f'{int(lon_obs_min)}_{int(lon_obs_max)}_{int(lat_obs_min)}_{int(lat_obs_max)}'
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
    
    def is_obs_time(self,t):

        return self.is_obs(t)
                
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

class Obsop_interp_l3_jax(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.name_var = config.OBSOP.name_var
        self.date_obs = []
        self.t_obs = []
        self.name_obs = []
        date_obs = list(dict_obs.keys())
        for t,timestamp in zip(Model.T,Model.timestamps):
            delta_t = [(timestamp - date).total_seconds() for date in date_obs]
            if len(delta_t)>0:
                
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[date_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[date_obs[ind_obs]]['attributes']):
                        
                        if (self.name_var in sat_info['name_var']) and ((config.OBSOP.name_obs is None) or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs)):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.t_obs:
                                self.date_obs.append(date_obs[ind_obs])
                                self.t_obs.append(t)
                    
        # For grid interpolation:
        self.Npix = config.OBSOP.Npix
        self.dmax = self.Npix*np.mean(np.sqrt(State.DX**2 + State.DY**2))*1e-3*np.sqrt(2)/2 # maximal distance for space interpolation

        self.t_obs = np.array(self.t_obs)
        self.t_obs_jax = jnp.array(self.t_obs)

        self.name_H += f'_L3-JAX_{self.name_var}_{config.OBSOP.Npix}'

        self._misfit_reduced_jit = jit(self._misfit_reduced)
        self._misfit_jit = jit(self._misfit)
    
    def _sparse_op(self, lon_obs, lat_obs):
        """Optimized sparse observation operator using KDTree."""
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        # KDTree for nearest neighbor search
        tree = KDTree(self.coords_car)
        D, ind_closest = tree.query(coords_car_obs, k=self.Npix)

        # Apply distance threshold
        valid_mask = (D <= self.dmax)

        if np.any(valid_mask):

            if self.Npix>1:
                # Compute weights
                weights = np.exp(-D**2 / (2 * (0.5 * self.dmax)**2)) * valid_mask

                # Normalize
                sum_weights = np.sum(weights, axis=1, keepdims=True)
                weights = np.divide(
                                weights,
                                sum_weights,
                                out=np.zeros_like(weights),
                                where=sum_weights > 0
                            )
            else:
                weights = np.ones_like(D) 

        else:
            data = D * 0.

        # Extract valid indices
        try:
            row, col = np.where(valid_mask)
            data = weights[row, col]
            indices = jnp.array([row, ind_closest[row, col]])
            data = jnp.array(data)
        except:
            data = jnp.array([])
            indices = jnp.array([])
            
        return data, indices
    
    def explicit_proj_operation(self, data, indices, X, n_obs):
        """
        Perform the projection operation explicitly without using csc_matrix.

        Parameters:
        - data: list or numpy array of interpolation coefficients (non-zero values of the sparse matrix).
        - row: list or numpy array of observation grid indices corresponding to each non-zero entry.
        - col: list or numpy array of state grid indices corresponding to each non-zero entry.
        - X: numpy array representing the state grid field to be projected.
        - n_obs: number of observation points (rows of the sparse matrix).

        Returns:
        - proj_X: projected observation values as a numpy array.
        """
        
        row = indices[0]
        col = indices[1]

        # Compute contributions of all sparse matrix entries to the corresponding observation indices
        proj_X = jnp.zeros(n_obs)
        proj_X = jnp.add.at(proj_X, row, data * X[col], inplace=False)

        return proj_X
    
    def process_obs(self, var_bc=None):

        self.varobs = {}
        self.errobs = {}
        self.data = {}
        self.indices = {}

        for i,(date,t) in enumerate(zip(self.date_obs, self.t_obs)):

            self.varobs[t] = {}
            self.errobs[t] = {}
            self.data[t] = {}
            self.indices[t] = {}

            sat_info_list = self.dict_obs[date]['attributes']
            obs_file_list = self.dict_obs[date]['obs_path']
            obs_name_list = self.dict_obs[date]['obs_name']

            # Init lists
            var_obs = []
            err_obs = []
            lon_obs = []
            lat_obs = []

            ####################
            # Merge observations
            ####################
            for obs_name, sat_info, obs_file in zip(obs_name_list, sat_info_list, obs_file_list):

                if obs_name not in self.name_obs:
                    continue

                with xr.open_dataset(obs_file) as ncin:

                    lon = ncin[sat_info['name_lon']].values
                    lat = ncin[sat_info['name_lat']].values

                    if self.name_var not in ncin:
                        continue

                    # Observed variable
                    var = ncin[self.name_var].values

                    # Reshape to 1D arrays
                    lon = lon.ravel()
                    lat = lat.ravel()
                    var = var.ravel()

                    # Observed error
                    name_err = self.name_var + '_err'
                    if name_err in ncin:
                        err = ncin[name_err].values.ravel() 
                    elif sat_info['sigma_noise'] is not None:
                        err = sat_info['sigma_noise'] * np.ones_like(var)
                    else:
                        err = np.ones_like(var)    

                    # Append to lists
                    var_obs.append(var)
                    err_obs.append(err)
                    lon_obs.append(lon)
                    lat_obs.append(lat)
            
            # Concatenations of lists
            var_obs = np.concatenate(var_obs)
            err_obs = np.concatenate(err_obs)
            lon_obs = np.concatenate(lon_obs)
            lat_obs = np.concatenate(lat_obs)

            coords_obs = np.column_stack((lon_obs, lat_obs))
            file_L3 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{date.strftime('%Y%m%d_%H%M')}.pic"
            if var_bc is not None and self.name_var in var_bc:
                mask = np.any(np.isnan(self.coords_geo),axis=1)
                var_bc_interp = griddata(self.coords_geo[~mask], var_bc[self.name_var][i].flatten()[~mask], coords_obs, method='cubic')
                var_obs -= var_bc_interp

            # Fill dictionnaries
            self.varobs[t] = var_obs
            self.errobs[t] = err_obs

            # Compute Sparse operator
            if not self.compute_op and self.write_op and os.path.exists(file_L3):
                with open(file_L3, "rb") as f:
                    data, indices = pickle.load(f)
                    self.data[t] = data
                    self.indices[t] = indices
            else:
                # Compute operator
                result = self._sparse_op(lon_obs,lat_obs)
                data, indices = result
                self.data[t] = data
                self.indices[t] = indices
                # Save operator if asked
                if self.write_op:
                    with open(file_L3, "wb") as f:
                        pickle.dump([data, indices], f)
        
        self.n_data = np.array([self.data[t].size for t in self.t_obs])
        self.n_obs = np.array([self.varobs[t].size for t in self.t_obs])
        
        if self.n_data.size>0:
            self.data_arr = np.zeros((self.t_obs.size, self.n_data.max()))
            self.indices_arr = np.zeros((self.t_obs.size, 2, self.n_data.max()))
            self.varobs_arr = np.zeros((self.t_obs.size, self.n_obs.max()))
            self.errobs_arr = np.ones((self.t_obs.size, self.n_obs.max())) * 1e7
        
            for i,t in enumerate(self.t_obs):

                self.data_arr[i,:self.n_data[i]] = self.data[t]
                self.indices_arr[i,:,:self.n_data[i]] = self.indices[t]
                self.varobs_arr[i,:self.n_obs[i]] = self.varobs[t]
                self.errobs_arr[i,:self.n_obs[i]] = self.errobs[t] 
            
            self.n_data = jnp.array(self.n_data)
            self.n_obs = jnp.array(self.n_obs)
            self.data_arr = jnp.array(self.data_arr)
            self.indices_arr = jnp.array(self.indices_arr, dtype=int)
            self.varobs_arr = jnp.array(self.varobs_arr)
            self.errobs_arr = jnp.array(self.errobs_arr)
        
    def is_obs_time(self,t):
        """Check if t is in observation times."""
        return jnp.any(jnp.isclose(t, self.t_obs_jax))

    def _misfit(self, t, X):
        
        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]
        data_t = self.data_arr[idt][0]
        indices_t = self.indices_arr[idt][0]
        varobs_t = self.varobs_arr[idt][0]
        errobs_t = self.errobs_arr[idt][0]

        # Project model state to obs space
        HX = self.explicit_proj_operation(data_t, indices_t, X, varobs_t.size)

        # Compute misfit & errors
        misfit = HX - varobs_t
        inverr = 1/errobs_t
        misfit = jnp.where(jnp.isnan(misfit), 0, misfit) 
        inverr = jnp.where(jnp.isnan(inverr), 0, inverr) 

        return inverr * misfit
    
    def _misfit_reduced(self, t, grad_obs, X):

        """
        Projects a gradient in observation space back to the model state space.
        
        Parameters:
            t: Current time
            grad_obs: Gradient in observation space (adjoint variable associated with misfit).
            X: Current model state (used to ensure consistent adjoint mapping).
            
        Returns:
            Gradient in the model state space (reduced space).
        """

        # Define a wrapper for _misfit that computes the forward misfit
        def misfit_func(X):
            return self._misfit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward operation
        _, vjp_func = jax.vjp(misfit_func, X)

        # Use the vjp function to project the gradient from observation space to model state space
        grad_model_state, = vjp_func(grad_obs)

        return grad_model_state
    
    def misfit(self, t, State):

        X = State.var[self.name_mod_var[self.name_var]]

        return self._misfit_jit(t, X.ravel() )
    
    def misfit_jax(self, t, State_var):

        X = State_var[self.name_mod_var[self.name_var]].ravel() 

        return self._misfit_jit(t, X)
    
    def adj(self, t, adState, State, misfit):

        # Read adjoint variable
        advar = adState.var[self.name_mod_var[self.name_var]]
        var = State.var[self.name_mod_var[self.name_var]]

        # Compute adjoint operation of y = Hx
        adX = self._misfit_reduced_jit(t, misfit , var.ravel())

        # Update adjoint variable
        adState.setvar(advar + adX.reshape(advar.shape), self.name_mod_var[self.name_var])  
    
class Obsop_interp_l4(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.name_var = config.OBSOP.name_var
        self.name_var_obs = {}
        self.name_obs = []
        self.t_obs = [] 
        date_obs = list(dict_obs.keys())
        for t,timestamp in zip(Model.T,Model.timestamps):
            delta_t = [(timestamp - date).total_seconds() for date in date_obs]
            if len(delta_t)>0:
                
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[date_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[date_obs[ind_obs]]['attributes']):
                        
                        if (self.name_var in sat_info['name_var']) and ((config.OBSOP.name_obs is None) or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs)):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.t_obs:
                                self.date_obs.append(date_obs[ind_obs])
                                self.t_obs.append(t)
    
        # For grid interpolation:
        self.interp_method = config.OBSOP.interp_method
        self.dist_min = .5*np.sqrt(State.dx**2+State.dy**2)*1e-3 # Minimum distance to consider an observation inside a model pixel

        self.DX = State.DX
        self.DY = State.DY

        self.DX = State.DX
        self.DY = State.DY
        
        # Mask
        self.mask = State.mask

        # Misfit on gradients
        self.gradients = config.OBSOP.gradients
        if self.gradients:
            self.name_H += f'_L4_grad_{self.name_var}_{config.OBSOP.interp_method}'
        else:
            self.name_H += f'_L4_{self.name_var}_{config.OBSOP.interp_method}'
        
        self.t_obs = np.array(self.t_obs)
        self.t_obs_jax = jnp.array(self.t_obs)

        self._misfit_reduced_jit = jit(self._misfit_reduced)
        self._misfit_jit = jit(self._misfit)

        self.flag_plot = config.EXP.flag_plot

    def process_obs(self, var_bc=None):
        
        # Initialize dictionnaries
        if self.gradients:
            self.varobs_grady = np.zeros((self.t_obs.size, self.DX.size))
            self.varobs_gradx = np.zeros((self.t_obs.size, self.DX.size))
        else:
            self.varobs = np.zeros((self.t_obs.size, self.DX.size))
        self.errobs = np.zeros((self.t_obs.size, self.DX.size))

        #############################
        # Loop on observation dates #
        #############################
        for i,(date,t) in enumerate(zip(self.date_obs,self.t_obs)):

            print(f"Processing observations at date {date} for variable {self.name_var}...")

            sat_info_list = self.dict_obs[date]['attributes']
            obs_file_list = self.dict_obs[date]['obs_path']
            obs_name_list = self.dict_obs[date]['obs_name']

            # Concatenate obs from different sensors
            lon_obs = []
            lat_obs = []
            var_obs = []
            err_obs = []

            ####################
            # Merge observations
            ####################
            for sat_info, obs_file, obs_name in zip(sat_info_list, obs_file_list, obs_name_list):

                if obs_name not in self.name_obs:
                    continue

                try:
                
                    with xr.open_dataset(obs_file) as ncin:

                        lon = ncin[sat_info['name_lon']].values
                        lat = ncin[sat_info['name_lat']].values

                        # Check if this observation class is wanted
                        if self.name_var not in ncin :
                            continue

                        # Observed variable
                        var = ncin[self.name_var].values

                        if lon.size != var.size and len(lon.shape)==1: # 2D regular grid
                            lon, lat = np.meshgrid(lon, lat)

                        # Observed error
                        name_err = self.name_var + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)
                        err[np.isnan(var)] = np.nan

                        # Add error due to interpolation (resolutions ratio)
                        dx, dy = grid.lonlat2dxdy(lon,lat)
                        _err_res = np.nanmean(dx * dy) / np.nanmean(self.DX * self.DY)
                        if _err_res>1:
                            err *= _err_res
                                        
                        # Append to lists
                        var_obs.append(+var.flatten())
                        err_obs.append(+err.flatten())
                        lon_obs.append(+lon.flatten())
                        lat_obs.append(+lat.flatten())
                except:
                    print(f"Warning: problem reading {obs_file}")
                    continue
            
            if len(var_obs)==0:
                # remove date from list
                print(f"Warning: no observation found at date {date} for variable {self.name_var}, skipping this date.")
                self.date_obs.pop(i)
                self.t_obs = np.delete(self.t_obs, i)
                self.t_obs_jax = jnp.delete(self.t_obs_jax, i)
                self.varobs = np.delete(self.varobs, i, axis=0)
                self.errobs = np.delete(self.errobs, i, axis=0)
                continue
            
            # Concatenations of lists
            var_obs = np.concatenate(var_obs)
            err_obs = np.concatenate(err_obs)
            lon_obs = np.concatenate(lon_obs)
            lat_obs = np.concatenate(lat_obs)
            
            ################
            # Process L4 obs
            ################
            file_L4 = f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_{date.strftime('%Y%m%d_%H%M')}.pic"
            # Check if spatial interpolations have already been performed
            if not self.compute_op and self.write_op and os.path.exists(file_L4):
                with open(file_L4, "rb") as f:
                    var_obs_interp, err_obs_interp = pickle.load(f)
            else:
                # Grid interpolation: performing spatial interpolation now
                # Loop on different obs for this date and this variable name
                _coords_obs = np.column_stack((lon_obs, lat_obs))

                if self.interp_method=='hybrid':
                    # We perform first nearest, then linear, and then cubic interpolations
                    _var_obs_interp = griddata(_coords_obs, var_obs, self.coords_geo, method='nearest')
                    _err_obs_interp = griddata(_coords_obs, err_obs, self.coords_geo, method='nearest')
                    _var_obs_interp_linear = griddata(_coords_obs, var_obs, self.coords_geo, method='linear')
                    _err_obs_interp_linear = griddata(_coords_obs, err_obs, self.coords_geo, method='linear')
                    _var_obs_interp[~np.isnan(_var_obs_interp_linear)] = _var_obs_interp_linear[~np.isnan(_var_obs_interp_linear)]
                    _err_obs_interp[~np.isnan(_err_obs_interp_linear)] = _err_obs_interp_linear[~np.isnan(_err_obs_interp_linear)]
                    _var_obs_interp_cubic = griddata(_coords_obs, var_obs, self.coords_geo, method='cubic')
                    _err_obs_interp_cubic = griddata(_coords_obs, err_obs, self.coords_geo, method='cubic')
                    _var_obs_interp[~np.isnan(_var_obs_interp_cubic)] = _var_obs_interp_linear[~np.isnan(_var_obs_interp_cubic)]
                    _err_obs_interp[~np.isnan(_err_obs_interp_cubic)] = _err_obs_interp_linear[~np.isnan(_err_obs_interp_cubic)]
                
                elif self.interp_method=='rtree': 

                    def _regrid_unstructured(lon_target, lat_target, lon, lat, var):
                        
                        # Spatial interpolation 
                        mesh = pyinterp.RTree() 
                        if len(lon_target.shape)==1:
                            lon_target, lat_target = np.meshgrid(lon_target, lat_target)
                        lons = lon.ravel()
                        lats = lat.ravel()
                        var_regridded = np.zeros((lat_target.shape[0],lon_target.shape[1])) 
                        data = np.array(var) 
                        mask = np.isnan(lons) | np.isnan(lats) | np.isnan(data) 
                        data = data[~mask]
                        mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
                        idw, _ = mesh.radial_basis_function(
                            np.vstack((lon_target.ravel(), lat_target.ravel())).T,
                            within=True,
                            k=11,
                            rbf='multiquadric',
                            epsilon=None,
                            smooth=0,
                            num_threads=0)
                        var_regridded[:,:] = idw.reshape(lon_target.shape) 

                        return var_regridded
                     
                    lon_target,lat_target = self.coords_geo.T
                    lon_target_grid,lat_target_grid = lon_target.reshape(self.shape_grid),lat_target.reshape(self.shape_grid)

                    _var_obs_interp = _regrid_unstructured(lon_target_grid,lat_target_grid,lon_obs,lat_obs,var_obs) 
                    _err_obs_interp = _regrid_unstructured(lon_target_grid,lat_target_grid,lon_obs,lat_obs,err_obs) 

                else:
                    _var_obs_interp = griddata(_coords_obs, var_obs, self.coords_geo, method=self.interp_method)
                    _err_obs_interp = griddata(_coords_obs, err_obs, self.coords_geo, method=self.interp_method)

                if np.all(np.isnan(_var_obs_interp)):
                    # remove date from list
                    print(f"Warning: all interpolated values are NaN at date {date} for variable {self.name_var}, skipping this date.")
                    print(self.date_obs[i], self.t_obs[i])
                    self.date_obs.pop(i)
                    self.t_obs = np.delete(self.t_obs, i)
                    self.t_obs_jax = jnp.delete(self.t_obs_jax, i)
                    self.varobs = np.delete(self.varobs, i, axis=0)
                    self.errobs = np.delete(self.errobs, i, axis=0)
                    
                    continue

                # Mask values outside obs range
                var_obs_interp = _var_obs_interp.reshape(self.shape_grid)
                err_obs_interp = _err_obs_interp.reshape(self.shape_grid)
                mask = (var_obs_interp<np.nanmin(var_obs)) | (var_obs_interp>np.nanmax(var_obs)) | (self.mask)
                var_obs_interp[mask] = np.nan
                err_obs_interp[mask] = np.nan
                
                # Save operator if asked
                if self.write_op:
                    with open(file_L4, "wb") as f:
                        pickle.dump((var_obs_interp,err_obs_interp), f)

                if var_bc is not None and self.name_var in var_bc:
                    var_obs_interp -= var_bc[self.name_var][i].flatten()
                
            if self.gradients:
                    # Compute gradients
                var_obs_interp_grady = np.zeros_like(var_obs_interp)*np.nan
                var_obs_interp_gradx = np.zeros_like(var_obs_interp)*np.nan
                var_obs_interp_grady[1:-1,1:-1] = (var_obs_interp[2:,1:-1] - var_obs_interp[:-2,1:-1]) / (2 * self.DY[1:-1,1:-1])
                var_obs_interp_gradx[1:-1,1:-1] = (var_obs_interp[1:-1,2:] - var_obs_interp[1:-1,:-2]) / (2 * self.DX[1:-1,1:-1])

                # Fill dictionnaries
                self.varobs_grady[i] = var_obs_interp_grady.flatten()
                self.varobs_gradx[i] = var_obs_interp_gradx.flatten()
                self.errobs[i] = (.5* err_obs_interp /  (self.DY[np.newaxis,:,:]**2 + self.DX[np.newaxis,:,:]**2)**.5).flatten()

            else:
                # Fill dictionnaries
                self.varobs[i] = var_obs_interp.flatten()
                self.errobs[i] = err_obs_interp.flatten()
            
            if self.flag_plot>1:
                lon_grid,lat_grid = self.coords_geo.T
                fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
                im1 = ax1.scatter(lon_obs,lat_obs,c=var_obs)
                plt.colorbar(im1,ax=ax1)
                im2 = ax2.pcolormesh(lon_grid.reshape(self.shape_grid),lat_grid.reshape(self.shape_grid), var_obs_interp)
                plt.colorbar(im2,ax=ax2)
                im3 = ax3.pcolormesh(lon_grid.reshape(self.shape_grid),lat_grid.reshape(self.shape_grid), err_obs_interp)
                plt.colorbar(im3,ax=ax3)
                fig.suptitle(date.strftime('%Y-%m-%d %H:%M'))
                plt.show()

        self.varobs_arr = jnp.array(self.varobs)
        self.errobs_arr = jnp.array(self.errobs)

        mask = jnp.isnan(self.varobs_arr) | jnp.isnan(self.errobs_arr) | (self.errobs_arr<1e-7) | (self.varobs_arr>1e7) 
        self.varobs_arr = jnp.where(mask, 0., self.varobs_arr)
        self.errobs_arr = jnp.where(mask, 1e15, self.errobs_arr)

    
    def is_obs_time(self,t):
        """Check if t is in observation times."""
        return jnp.any(jnp.isclose(t, self.t_obs_jax))
                
    def misfit(self, t, State):

        X = State.var[self.name_mod_var[self.name_var]].flatten()

        if self.gradients:
            return self._misfit_grad(t, X)
        else:
            return self._misfit(t, X)

    def _misfit(self, t, X):

        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]

        # Compute misfit & errors
        misfit = X - self.varobs_arr[idt]
        inverr = 1/self.errobs_arr[idt]
        
        res = inverr * misfit
        res = jnp.where(jnp.isnan(res), 0., res) 

        return res.flatten()

    def _misfit_reduced(self, t, grad_obs, X):

        """
        Projects a gradient in observation space back to the model state space.
        
        Parameters:
            t: Current time
            grad_obs: Gradient in observation space (adjoint variable associated with misfit).
            X: Current model state (used to ensure consistent adjoint mapping).
            
        Returns:
            Gradient in the model state space (reduced space).
        """

        # Define a wrapper for _misfit that computes the forward misfit
        def misfit_func(X):
            return self._misfit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward operation
        _, vjp_func = jax.vjp(misfit_func, X)

        # Use the vjp function to project the gradient from observation space to model state space
        grad_model_state, = vjp_func(grad_obs)

        return grad_model_state
    
    def _misfit_grad(self, t, State):

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
    
    def adj(self, t, adState, State, misfit):

        if self.gradients:
            return self._adj_grad(t, adState, State, misfit)
        else:
            return self._adj(t, adState, State, misfit)

    def _adj(self, t, adState, State, misfit):

        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]
        inverr = 1/self.errobs_arr[idt]
        _advar = (inverr * misfit)
        _advar = jnp.where(jnp.isnan(_advar), 0., _advar)

        # Read adjoint variable
        advar = adState.var[self.name_mod_var[self.name_var]]
        advar += _advar.reshape(advar.shape)
    
        # Update adjoint variable
        adState.setvar(advar, self.name_mod_var[self.name_var])  
  
    def _adj_grad(self, t, adState, State, misfit):

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
        
        self.misfit_indt = {}

    def is_obs_time(self,t):

        for _Obsop in self.Obsop:
            if _Obsop.is_obs_time(t):
                return True
        return False

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
        
        self.misfit_indt[t] = [None for _ in self.Obsop]
        indt = 0
        for i, _Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs_time(t):
                _misfit = _Obsop.misfit(t,State)
                self.misfit_indt[t][i] = slice(indt, indt+_misfit.size)
                indt += _misfit.size
                misfit = np.concatenate((misfit,_misfit))

        if len(misfit)==0:
            print(t)
            misfit = np.array([1e-7]) # To avoid problems with empty misfit
    
        return misfit

    def adj(self, t, adState, State, misfit):
    
        for i, _Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs_time(t):
                _mistfit = misfit[self.misfit_indt[t][i]]
                _Obsop.adj(t, adState, State, _mistfit)
