#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import sys
import xarray as xr 
import numpy as np 
import pyinterp 
import pyinterp.fill
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd
from . import grid

import matplotlib.pylab as plt

def Bc(config, State=None, verbose=1, *args, **kwargs):
    """
    NAME
        Bc

    DESCRIPTION
        Main function for specific boundary conditions
    """
    
    if config.BC is None:
        return 
    
    elif config.BC.super is None:
        return Bc_multi(config,State=State,verbose=verbose)
    
    if verbose:
        print(config.BC)
    
    if config.BC.super=='BC_EXT':
        return Bc_ext(config,State=State)
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')


class Bc_ext:

    def __init__(self,config, State=None):

        if State is None:
            from . import state
            State = state.State(config, verbose=False)

        # Get grid coordinates
        self.lon = State.lon
        self.lat = State.lat
        self.mask = State.mask

        # Study domain borders
        lon_min = np.nanmin(self.lon)
        lon_max = np.nanmax(self.lon)
        lat_min = np.nanmin(self.lat)
        lat_max = np.nanmax(self.lat)

        # Grid spacing
        dlon = np.nanmean(self.lon[:,1:]-self.lon[:,:-1])
        dlat = np.nanmean(self.lat[1:,:]-self.lat[:-1,:])

        # Read netcdf
        _ds = xr.open_mfdataset(config.BC.file)

        # Copy dataset
        ds = _ds.copy()#.load()
        _ds.close()

        # Convert longitude 
        try:
            if np.sign(ds[config.BC.name_lon].data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, ds[config.BC.name_lon].data % 360))})
            elif np.sign(ds[config.BC.name_lon].data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, (ds[config.BC.name_lon].data + 180) % 360 - 180 ))})
            ds = ds.sortby(ds[config.BC.name_lon])    
        except:
            print("Warning: can't convert longitude coordinates in " + config.BC.file)
        
        # Select study domain
        time_bc = ds[config.BC.name_time].values
        self.name_time_bc = config.BC.name_time
        lon_bc = ds[config.BC.name_lon].values
        lat_bc = ds[config.BC.name_lat].values
        dlon += np.nanmean(lon_bc[1:]-lon_bc[:-1])
        dlat += np.nanmean(lat_bc[1:]-lat_bc[:-1])
        dtime = time_bc[1]-time_bc[0]
        try:
            ds = ds.sel({
                config.BC.name_time:slice(np.datetime64(config.EXP.init_date)-dtime,np.datetime64(config.EXP.final_date)+dtime),
                })
        except:
            ds = ds.where(
                (ds[config.BC.name_time] >= np.datetime64(config.EXP.init_date)-dtime) & (ds[config.BC.name_time] <= np.datetime64(config.EXP.final_date)+dtime),
                drop=True
            )
        if len(ds[config.BC.name_lon].shape)==1 and len(ds[config.BC.name_lat].shape)==1:
            ds = ds.sel({
                config.BC.name_lon:slice(lon_min-2*dlon,lon_max+2*dlon),
                config.BC.name_lat:slice(lat_min-2*dlat,lat_max+2*dlat),
                })
        
        self.c_grid = config.BC.c_grid if 'c_grid' in config.BC else False
        
        # Get BC coordinates
        self.lon_bc = ds[config.BC.name_lon].values
        self.lat_bc = ds[config.BC.name_lat].values
        if config.BC.name_time is not None:
            self.time_bc = ds[config.BC.name_time].values
        else:
            self.time_bc = None

        # Get BC variables
        self.var = {}
        for name in config.BC.name_var:
            self.var[name] = ds[config.BC.name_var[name]].load()
        ds.close()        
    
    def interp(self, time):
        """
        Interpolate boundary conditions on the model grid
        """
        
        # Check time dimension
        if self.time_bc is not None and self.time_bc.size>1:
            if len(time.shape)==0:
                time = np.array([time])
            elif len(time.shape)==1:
                time = np.ascontiguousarray(time)
            else:
                sys.exit('Time dimension must be 1D or 0D')
        
        # Convert time to np.datetime64
        if time.size>1 and type(time[0])!=np.datetime64:
            time = np.array([np.datetime64(dt) for dt in time])
        elif time.size==1 and type(time)!=np.datetime64:
            time = np.array([np.datetime64(time)])
        
        # Interpolate
        if len(self.lon_bc.shape)==1 and len(self.lat_bc.shape)==1:
            var_interp = self._interp_3D(time)
        elif len(self.lon_bc.shape)==2 and len(self.lat_bc.shape)==2 and np.all(self.lon_bc==self.lon) and np.all(self.lat_bc==self.lat):
            var_interp = self._interp_1D(time)
        else:
            sys.exit('Boundary conditions coordinates must be 1D, 2D. If 2D, they must match the model grid coordinates')

        return var_interp
            
    def _interp_3D(self,time):
        """
        Interpolate boundary conditions on the model grid. It works only if the boundary conditions grid is regular
        """

        # Select timestamps
        if self.time_bc is not None and self.time_bc.size>1:
            dtbc = self.time_bc[1] - self.time_bc[0]
            time0 = time[0] - dtbc
            time1 = time[-1] + dtbc
            time_bc = self.time_bc[(self.time_bc>=time0) & (self.time_bc<=time1)]
            
        # Define source grid
        x_source_axis = pyinterp.Axis(self.lon_bc, is_circle=True)
        y_source_axis = pyinterp.Axis(self.lat_bc)
        if self.time_bc is not None and self.time_bc.size>1:
            z_source_axis = pyinterp.TemporalAxis(time_bc)

        # Define target grid
        if self.time_bc is not None and self.time_bc.size>1:
            time_target = z_source_axis.safe_cast(time)
            nt = len(time_target)
        else:
            nt = 1

        # Interpolation
        var_interp = {}
        for name in self.var:
            if not self.c_grid or (name!='U' and name!='V'):
                x_target = np.repeat(self.lon.transpose()[:,:,np.newaxis],nt,axis=2)
                y_target = np.repeat(self.lat.transpose()[:,:,np.newaxis],nt,axis=2)
                if self.time_bc is not None and self.time_bc.size>1:
                    z_target = np.tile(time_target,(self.lon.shape[1],self.lat.shape[0],1))
            elif self.c_grid and name=='U':
                lon_u = np.zeros((self.lon.shape[0], self.lon.shape[1]+1))
                lat_u = np.zeros((self.lat.shape[0], self.lat.shape[1]+1))
                lon_u[:,1:-1] = (self.lon[:,1:] + self.lon[:,:-1])/2. 
                lon_u[:,0] = self.lon[:,0] - (self.lon[:,1]-self.lon[:,0])/2.
                lon_u[:,-1] = self.lon[:,-1] + (self.lon[:,-1]-self.lon[:,-2])/2.
                lat_u[:,1:-1] = (self.lat[:,1:] + self.lat[:,:-1])/2. 
                lat_u[:,0] = self.lat[:,0] - (self.lat[:,1]-self.lat[:,0])/2.
                lat_u[:,-1] = self.lat[:,-1] + (self.lat[:,-1]-self.lat[:,-2])/2.
                x_target = np.repeat(lon_u.transpose()[:,:,np.newaxis],nt,axis=2)
                y_target = np.repeat(lat_u.transpose()[:,:,np.newaxis],nt,axis=2)
                if self.time_bc is not None and self.time_bc.size>1:
                    z_target = np.tile(time_target,(lon_u.shape[1],lat_u.shape[0],1))
            elif self.c_grid and name=='V':
                lon_v = np.zeros((self.lat.shape[0]+1, self.lat.shape[1]))
                lat_v = np.zeros((self.lat.shape[0]+1, self.lat.shape[1]))
                lon_v[1:-1,:] = (self.lon[1:,:] + self.lon[:-1,:])/2. 
                lon_v[0,:] = self.lon[0,:] - (self.lon[1,:]-self.lon[0,:])/2.
                lon_v[-1,:] = self.lon[-1,:] + (self.lon[-1,:]-self.lon[-2,:])/2.
                lat_v[1:-1,:] = (self.lat[1:,:] + self.lat[:-1,:])/2. 
                lat_v[0,:] = self.lat[0,:] - (self.lat[1,:]-self.lat[0,:])/2.
                lat_v[-1,:] = self.lat[-1,:] + (self.lat[-1,:]-self.lat[-2,:])/2.
                x_target = np.repeat(lon_v.transpose()[:,:,np.newaxis],nt,axis=2)
                y_target = np.repeat(lat_v.transpose()[:,:,np.newaxis],nt,axis=2)
                if self.time_bc is not None and self.time_bc.size>1:
                    z_target = np.tile(time_target,(lon_v.shape[1],lat_v.shape[0],1))

            if self.time_bc is not None and self.time_bc.size>1:
                var = +self.var[name].sel({self.name_time_bc:slice(time0,time1)}).squeeze()
                grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, var.T)
                # Remove NaN
                if np.isnan(var).any():
                    _, var = pyinterp.fill.gauss_seidel(grid_source)
                    grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, var)
                # Interpolate
                _var_interp = pyinterp.trivariate(grid_source,
                                            x_target.flatten(),
                                            y_target.flatten(),
                                            z_target.flatten(),
                                            bounds_error=False).reshape(x_target.shape).T
                
                for t in range(len(time)):
                    if not self.c_grid or name not in ['U', 'V']:
                        _var_interp[t][self.mask] = np.nan
                    elif name == 'U':
                        mask_u = np.zeros((self.mask.shape[0], self.mask.shape[1]+1), dtype=bool)
                        mask_u[:, 1:-1] = self.mask[:, :-1] | self.mask[:, 1:]
                        mask_u[:, 0] = self.mask[:, 0]
                        mask_u[:, -1] = self.mask[:, -1]
                        _var_interp[t][mask_u] = np.nan
                    elif name == 'V':
                        mask_v = np.zeros((self.mask.shape[0]+1, self.mask.shape[1]), dtype=bool)
                        mask_v[1:-1, :] = self.mask[:-1, :] | self.mask[1:, :]
                        mask_v[0, :] = self.mask[0, :]
                        mask_v[-1, :] = self.mask[-1, :]
                        _var_interp[t][mask_v] = np.nan
                    if time[t]<self.time_bc[0]:
                        ind_t = np.argmin(np.abs(time-self.time_bc[0]))
                        _var_interp[t] = _var_interp[ind_t]
                    if time[t]>self.time_bc[-1]:
                        ind_t = np.argmin(np.abs(time-self.time_bc[-1]))
                        _var_interp[t] = _var_interp[ind_t]
            else:
                grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, self.var[name].T)
                _var_interp = pyinterp.bivariate(grid_source,
                                                x_target[:,:,0].flatten(),
                                                y_target[:,:,0].flatten(),
                                                bounds_error=False).reshape(x_target[:,:,0].shape).T

                _var_interp = _var_interp[np.newaxis,:,:].repeat(len(time),axis=0) 
                _var_interp[self.mask] = np.nan
            #_var_interp[np.isnan(_var_interp)] = 0.
            var_interp[name] = _var_interp
        
        return var_interp
    
    def _interp_1D(self, time):
        """
        Interpolate boundary conditions in time only (spatial grid matches model grid)
        """
        var_interp = {}
        # Ensure time is array
        if len(time.shape) == 0:
            time = np.array([time])
        elif len(time.shape) == 1:
            time = np.ascontiguousarray(time)
        else:
            sys.exit('Time dimension must be 1D or 0D')

        for name in self.var:
            var_data = self.var[name]
            time_bc = self.time_bc

            nt = len(time)
            out_shape = (nt,) + var_data.shape[1:]

            if time_bc is None or np.size(time_bc) <= 1:
                # Repeat the same array for each requested time
                interp_data = np.repeat(var_data.values[np.newaxis, ...], nt, axis=0)
            else:
                # Use xarray's interp method for time interpolation
                interp_result = var_data.interp({self.name_time_bc: xr.DataArray(time, dims=[self.name_time_bc])}, method="linear")
                interp_data = interp_result.values

            # Mask if needed
            if hasattr(self, 'mask'):
                mask = self.mask
                if mask.shape == interp_data.shape[1:]:
                    interp_data[:, mask] = np.nan

            var_interp[name] = interp_data

        return var_interp


class Bc_multi:

    def __init__(self,config,State=None,verbose=1):

        self.Bc = []
        _config = config.copy()

        for name_bc in config.BC:
            _config.BC = config.BC[name_bc]
            _Bc = Bc(_config,State=State,verbose=verbose)
            self.Bc.append(_Bc)

    
    def interp(self,time):

        var_interp = {}

        for _Bc in self.Bc:

            _var_interp = _Bc.interp(time)

            for name in _var_interp:
                var_interp[name] = _var_interp[name]

        return var_interp

