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
from . import  grid

import matplotlib.pylab as plt

def Bc(config, *args, **kwargs):
    """
    NAME
        Bc

    DESCRIPTION
        Main function for specific boundary conditions
    """
    
    if config.BC is None:
        return 
    
    elif config.BC.super is None:
        return Bc_multi(config)
    
    print(config.BC)
    
    if config.BC.super=='BC_EXT':
        return Bc_ext(config)
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')


class Bc_ext:

    def __init__(self,config):

        ds = xr.open_mfdataset(config.BC.file)
        print(ds)
        self.lon = ds[config.BC.name_lon].values %360
        self.lat = ds[config.BC.name_lat].values
        if config.BC.name_time is not None:
            self.time = ds[config.BC.name_time].values
        else:
            self.time = None

        self.var = {}
        for name in config.BC.name_var:
            self.var[name] = ds[config.BC.name_var[name]].data.squeeze()

        ds.close()        

        self.name_mod_var = config.BC.name_mod_var
        self.dist_sponge = config.BC.dist_sponge
        

    def interp(self,time,lon,lat):

        # Define source grid
        x_source_axis = pyinterp.Axis(self.lon, is_circle=True)
        y_source_axis = pyinterp.Axis(self.lat)
        if self.time is not None and self.time.size>1:
            z_source_axis = pyinterp.TemporalAxis(self.time)

        # Define target grid
        if self.time is not None and self.time.size>1:
            time = z_source_axis.safe_cast(np.ascontiguousarray(time))
            z_target = np.tile(time,(lon.shape[1],lat.shape[0],1))
            nt = time.size
        else:
            nt = 1
        x_target = np.repeat(lon.transpose()[:,:,np.newaxis],nt,axis=2)
        y_target = np.repeat(lat.transpose()[:,:,np.newaxis],nt,axis=2)

        # Interpolation
        var_interp = {}
        for name in self.var:
            if self.time is not None and self.time.size>1:
                grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, self.var[name].T)
                _var_interp = pyinterp.trivariate(grid_source,
                                            x_target.flatten(),
                                            y_target.flatten(),
                                            z_target.flatten(),
                                            bounds_error=False).reshape(x_target.shape).T
            else:
                grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, self.var[name].T)
                _var_interp = pyinterp.bivariate(grid_source,
                                                x_target[:,:,0].flatten(),
                                                y_target[:,:,0].flatten(),
                                                bounds_error=False).reshape(x_target[:,:,0].shape).T

                _var_interp = _var_interp[np.newaxis,:,:].repeat(time.size,axis=0) 

            _var_interp[np.isnan(_var_interp)] = 0
            
            var_interp[self.name_mod_var[name]] = _var_interp
        
        return var_interp

    def compute_weight_map(self,lon,lat,mask=None,bc=True):

        if self.dist_sponge is None:
            print('No sponge distance set in the configuration')
            return np.zeros_like(lon)
        
        if mask is None:
            mask = np.zeros_like(lon,dtype=bool)

        #####################
        # Compute weights map
        #####################
        coords = np.column_stack((lon.ravel(), lat.ravel()))
        # construct KD-tree
        ground_pixel_tree = spatial.cKDTree(grid.geo2cart(coords))
        subdomain = grid.geo2cart(coords)[:100]
        eucl_dist = cdist(subdomain, subdomain, metric="euclidean")
        dist_threshold = np.min(eucl_dist[np.nonzero(eucl_dist)])
        # Add boundary pixels to mask
        if bc:
            mask[0,:] = True
            mask[-1,:] = True
            mask[:,0] = True
            mask[:,-1] = True
        
        # get boundary coordinates
        lon_bc = lon[mask]
        lat_bc = lat[mask]
        coords_bc = np.column_stack((lon_bc, lat_bc))
        bc_tree = spatial.cKDTree(grid.geo2cart(coords_bc))
        
        # Compute distance between model pixels and boundary pixels
        dist_mx = ground_pixel_tree.sparse_distance_matrix(bc_tree,2*self.dist_sponge)
        
        # Initialize weight map
        bc_weight = np.zeros(lon.size)
        #
        keys = np.array(list(dist_mx.keys()))
        ind_mod = keys[:, 0]
        dist = np.array(list(dist_mx.values()))
        dist = np.maximum(dist-0.5*dist_threshold, 0)
        # Dataframe initialized without nan values in var
        df = pd.DataFrame({'ind_mod': ind_mod,
                        'dist': dist,
                        'weight':np.ones_like(dist)})
        # Remove external values in the boundary pixels
        ind_dist = (df.dist == 0)
        df = df[np.logical_or(ind_dist,
                            np.isin(df.ind_mod,
                                    df[ind_dist].ind_mod,
                                    invert=True))]
        # Compute tapering
        df['tapering'] = np.exp(-(df['dist']**2/(2*(0.5*self.dist_sponge)**2)))
        # Nudge values out of pixels
        df.loc[df.dist > 0, "weight"] *= df.loc[df.dist > 0, "tapering"]
        # Compute weight average and save it
        df['tapering'] = df['tapering']**10
        wa = lambda x: np.average(x, weights=df.loc[x.index, "tapering"])
        dfg = df.groupby('ind_mod')
        weights = dfg['weight'].apply(wa)
        bc_weight[weights.index] = np.array(weights)
        bc_weight = bc_weight.reshape(lon.shape)
        
        return bc_weight

class Bc_multi:

    def __init__(self,config):

        self.Bc = []
        _config = config.copy()

        for _BC in config.BC:
            _config.BC = config.BC[_BC]
            self.Bc.append(Bc(_config))

    
    def interp(self,time,lon,lat):

        var_interp = {}

        for i,_Bc in enumerate(self.Bc):

            _var_interp = _Bc.interp(time,lon,lat)

            for name in _var_interp:
                var_interp[name] = _var_interp[name]

        return var_interp

