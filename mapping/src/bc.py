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
        self.mask = +State.mask

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

        # Convert longitude 
        if np.sign(_ds[config.BC.name_lon].data.min())==-1 and State.lon_unit=='0_360':
            _ds = _ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, _ds[config.BC.name_lon].data % 360))})
        elif np.sign(_ds[config.BC.name_lon].data.min())==1 and State.lon_unit=='-180_180':
            _ds = _ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, (_ds[config.BC.name_lon].data + 180) % 360 - 180))})
        _ds = _ds.sortby(_ds[config.BC.name_lon])    

        # Copy dataset
        ds = _ds.copy()
        _ds.close()
        
        # Select study domain
        lon_bc = ds[config.BC.name_lon].values
        lat_bc = ds[config.BC.name_lat].values
        dlon += np.nanmean(lon_bc[1:]-lon_bc[:-1])
        dlat += np.nanmean(lat_bc[1:]-lat_bc[:-1])
        ds = ds.sel({
            config.BC.name_lon:slice(lon_min-dlon,lon_max+2*dlon),
            config.BC.name_lat:slice(lat_min-dlat,lat_max+2*dlat)})
        
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
            
    def interp(self,time):

        # Define source grid
        x_source_axis = pyinterp.Axis(self.lon_bc, is_circle=True)
        y_source_axis = pyinterp.Axis(self.lat_bc)
        if self.time_bc is not None and self.time_bc.size>1:
            z_source_axis = pyinterp.TemporalAxis(self.time_bc)

        # Define target grid
        if self.time_bc is not None and self.time_bc.size>1:
            time_target = z_source_axis.safe_cast(np.ascontiguousarray(time))
            z_target = np.tile(time_target,(self.lon.shape[1],self.lat.shape[0],1))
            nt = len(time_target)
        else:
            nt = 1
        x_target = np.repeat(self.lon.transpose()[:,:,np.newaxis],nt,axis=2)
        y_target = np.repeat(self.lat.transpose()[:,:,np.newaxis],nt,axis=2)

        # Interpolation
        var_interp = {}
        for name in self.var:
            if self.time_bc is not None and self.time_bc.size>1:
                grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, self.var[name].T)
                _var_interp = pyinterp.trivariate(grid_source,
                                            x_target.flatten(),
                                            y_target.flatten(),
                                            z_target.flatten(),
                                            bounds_error=False).reshape(x_target.shape).T
                _var_interp[np.isnan(_var_interp)] = 0
                for t in range(len(time)):
                    _var_interp[t][self.mask] = np.nan
            else:
                grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, self.var[name].T)
                _var_interp = pyinterp.bivariate(grid_source,
                                                x_target[:,:,0].flatten(),
                                                y_target[:,:,0].flatten(),
                                                bounds_error=False).reshape(x_target[:,:,0].shape).T

                _var_interp = _var_interp[np.newaxis,:,:].repeat(len(time),axis=0) 
                _var_interp[np.isnan(_var_interp)] = 0
                _var_interp[self.mask] = np.nan

            
            
            
            
            var_interp[name] = _var_interp
        
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

