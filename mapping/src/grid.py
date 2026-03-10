#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:02:05 2019

@author: leguillou
"""

import os
import xarray as xr
import numpy as np
from scipy import interpolate
import pyinterp 
import pyinterp.fill
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pylab as plt

from scipy.spatial import KDTree

def lonlat2dxdy(lon,lat):
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    dx = np.sqrt((dlon[1]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[1]*111000)**2)
    dy = np.sqrt((dlon[0]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[0]*111000)**2)
    dx[0,:] = dx[1,:]
    dx[-1,: ]= dx[-2,:] 
    dx[:,0] = dx[:,1]
    dx[:,-1] = dx[:,-2]
    dy[0,:] = dy[1,:]
    dy[-1,:] = dy[-2,:] 
    dy[:,0] = dy[:,1]
    dy[:,-1] = dy[:,-2]
    
    return dx,dy

def dxdy2xy(dx,dy,x0=0,y0=0):
    ny,nx = dx.shape
    X = np.zeros((ny,nx))
    Y = np.zeros((ny,nx))
    for i in range(ny):
        for j in range(nx):
            X[i,j] = x0 + np.sum(dx[i,:j])
            Y[i,j] = y0 + np.sum(dy[:i,j])   
    return X,Y

def ds(lon, lat):
    n = lon.size
    ds = np.zeros((n,)) * np.nan
    for i in range(0, n-1):
        dlon = lon[i+1] - lon[i]
        dlat = lat[i+1] - lat[i]
        ds[i] = np.sqrt((dlon*111000*np.cos(np.deg2rad(lat[i])))**2
                        + (dlat*111000)**2)
    return ds


def geo2cart(coords):
    """
    NAME
        geo2cart

    DESCRIPTION
        Transform coordinates from geodetic to cartesian

        Args:
            coords : a set of lan/lon coordinates (e.g. a tuple or
             an array of tuples)


        Returns: a set of cartesian coordinates (x,y,z)

    """

    # WGS 84 reference coordinate system parameters
    A = 6378.137  # major axis [km]
    E2 = 6.69437999014e-3  # eccentricity squared

    coords = np.asarray(coords).astype(float)

    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])

    # convert to radiants
    lat_rad = np.radians(coords[:, 1])
    lon_rad = np.radians(coords[:, 0])

    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))


def laplacian(u, dx, dy):
    """
    Calculates the laplacian of u using the divergence and gradient functions and gives
    as output Ml.
    """
    Ml = np.gradient(np.gradient(u, dx, axis=0), dx, axis=0)\
         +  np.gradient(np.gradient(u, dy, axis=1), dy, axis=1);
    return Ml

def interp2d(ds,name_vars,lon_out,lat_out):

    ds = ds.squeeze()
    
    ds = ds.assign_coords(
                 {name_vars['lon']:(ds[name_vars['lon']]),
                  name_vars['lat']:ds[name_vars['lat']]})
            
    if ds[name_vars['var']].shape[0]!=ds[name_vars['lat']].shape[0]:
        ds[name_vars['var']] = ds[name_vars['var']].transpose()
        
    if len(ds[name_vars['lon']].shape)==2:
        dlon = (ds[name_vars['lon']][:,1:].values - ds[name_vars['lon']][:,:-1].values).max()
        dlat = (ds[name_vars['lat']][1:,:].values - ds[name_vars['lat']][:-1,:].values).max()

        
        ds = ds.where((ds[name_vars['lon']]<=lon_out.max()+dlon) &\
                      (ds[name_vars['lon']]>=lon_out.min()-dlon) &\
                      (ds[name_vars['lat']]<=lat_out.max()+dlat) &\
                      (ds[name_vars['lat']]>=lat_out.min()-dlat),drop=True)
            
        lon_sel = ds[name_vars['lon']].values
        lat_sel = ds[name_vars['lat']].values
            
    else:
        dlon = (ds[name_vars['lon']][1:].values - ds[name_vars['lon']][:-1].values).max()
        dlat = (ds[name_vars['lat']][1:].values - ds[name_vars['lat']][:-1].values).max()
        
        ds = ds.where((ds[name_vars['lon']]<=lon_out.max()+dlon) &\
                      (ds[name_vars['lon']]>=lon_out.min()-dlon) &\
                      (ds[name_vars['lat']]<=lat_out.max()+dlat) &\
                      (ds[name_vars['lat']]>=lat_out.min()-dlat),drop=True)
            
        lon_sel,lat_sel = np.meshgrid(
            ds[name_vars['lon']].values,
            ds[name_vars['lat']].values)
    
    var_sel = ds[name_vars['var']].values

    # Interpolate to state grid 
    var_out = interpolate.griddata((lon_sel.ravel(),lat_sel.ravel()),
                   var_sel.ravel(),
                   (lon_out.ravel(),lat_out.ravel())).reshape((lat_out.shape))
    
    return var_out


def interp3d_geo(lon, lat, time, var, lon_target, lat_target, time_target, remove_NaN=False):

        # Define source grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        if time is not None:
            z_source_axis = pyinterp.TemporalAxis(time)
        ssh_source = var.T
        if time is not None:    
            grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
        else:
            grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, ssh_source.data)
        
        # Define target grid
        mx_target, my_target, mz_target = np.meshgrid(
            lon_target.flatten(),
            lat_target.flatten(),
            z_source_axis.safe_cast(np.ascontiguousarray(time_target)),
            indexing="ij")

        # Spatio-temporal Interpolation
        if time is not None:  
            ssh_interp = pyinterp.trivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            mz_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        else:
            ssh_interp = pyinterp.bivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        
        # Extrapolation in NaN values if needed
        if remove_NaN and np.isnan(ssh_interp).any():
            x_source_axis = pyinterp.Axis(lon_target, is_circle=False)
            y_source_axis = pyinterp.Axis(lat_target)
            z_source_axis = pyinterp.TemporalAxis(np.ascontiguousarray(time_target))
            grid = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis,  ssh_interp.T)
            _, filled = pyinterp.fill.gauss_seidel(grid)
        else:
            filled = ssh_interp.T
        
        # Save to dataset
        return filled

def interp3d_unstructured(lon, lat, time, var, lon_target, lat_target, time_target):
        
    # Spatial interpolation 
    mesh = pyinterp.RTree()
    if len(lon_target.shape)==1:
        lon_target, lat_target = np.meshgrid(lon_target, lat_target)
    lons = lon.ravel()
    lats = lat.ravel()
    var_regridded = np.zeros((time.size,lat_target.shape[0],lon_target.shape[1]))
    for i in range(time.size):
        data = var[i].data.ravel()
        mask = np.isnan(lons) + np.isnan(lats) + np.isnan(data)
        data = data[~mask]
        mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
        idw, _ = mesh.inverse_distance_weighting(
            np.vstack((lon_target.ravel(), lat_target.ravel())).T,
            within=False,  # Extrapolation is forbidden
            k=11,  # We are looking for at most 11 neighbors
            radius=600000,
            num_threads=0)
        var_regridded[i,:,:] = idw.reshape(lon_target.shape)


    # Time interpolation
    f = interpolate.interp1d(time,var_regridded,axis=0)
    var_regridded = f(time_target)

    return var_regridded


def boundary_conditions(file_bc, dist_bc, name_var_bc, timestamps,
                        lon2d, lat2d, flag_plot=0,
                        mask=None, coast=False,file_depth=None,name_var_depth=None,mindepth=None):

    # Read depth
    if file_depth is not None and os.path.exists(file_depth):
        ds = xr.open_dataset(file_depth).squeeze()
        depth = interp2d(ds,name_var_depth,lon2d,lat2d)
    else: depth = None

    if type(timestamps) in [int,float]:
        timestamps = np.array([timestamps])

    NT = timestamps.size
    ny,nx = lon2d.shape

    var_bc_interpTime = np.zeros((NT,ny,nx))

    try:

        if file_bc is not None:
            #####################
            # Read file
            #####################
            
            ds = xr.open_mfdataset(file_bc,combine='by_coords').squeeze()
    
            
            if 'time' in name_var_bc:
                flag = '3D'
                bc_times = ds[name_var_bc['time']].values
                dtime = (ds[name_var_bc['time']][1:].values-ds[name_var_bc['time']][:-1].values).max()
                # Convert to timestamps
                ds = ds.sel(
                    {name_var_bc['time']:slice(timestamps.min()-dtime,
                                               timestamps.max()+dtime)
                     }
                    )
                bc_times = ds[name_var_bc['time']]
            else:
                flag = '2D'
            
            # Read and extract BC grid
            ds = ds.assign_coords({name_var_bc['lon']:(ds[name_var_bc['lon']] % 360),
                                   name_var_bc['lat']:ds[name_var_bc['lat']]})
            
            if len(ds[name_var_bc['lon']].shape)==2:
                dlon = (ds[name_var_bc['lon']][:,1:].values - ds[name_var_bc['lon']][:,:-1].values).max()
                dlat = (ds[name_var_bc['lat']][1:,:].values - ds[name_var_bc['lat']][:-1,:].values).max()
    
                
                ds = ds.where((ds[name_var_bc['lon']]<=lon2d.max()+dlon) &\
                              (ds[name_var_bc['lon']]>=lon2d.min()-dlon) &\
                              (ds[name_var_bc['lat']]<=lat2d.max()+dlat) &\
                              (ds[name_var_bc['lat']]>=lat2d.min()-dlat),drop=True)
                    
                lon_bc = ds[name_var_bc['lon']].values
                lat_bc = ds[name_var_bc['lat']].values
            
            else:
                dlon = (ds[name_var_bc['lon']][1:].values - ds[name_var_bc['lon']][:-1].values).max()
                dlat = (ds[name_var_bc['lat']][1:].values - ds[name_var_bc['lat']][:-1].values).max()
                
                ds = ds.where((ds[name_var_bc['lon']]<=lon2d.max()+dlon) &\
                              (ds[name_var_bc['lon']]>=lon2d.min()-dlon) &\
                              (ds[name_var_bc['lat']]<=lat2d.max()+dlat) &\
                              (ds[name_var_bc['lat']]>=lat2d.min()-dlat),drop=True)
    
                lon_bc,lat_bc = np.meshgrid(ds[name_var_bc['lon']].values,
                                            ds[name_var_bc['lat']].values)
            
            lon_bc = lon_bc % 360
            
            # Read BC fields
            var_bc = ds[name_var_bc['var']]
            var_bc = var_bc.where(var_bc<10,drop=True)
            
            #####################
            # Grid processing
            #####################
            
            if np.all(lon_bc==lon2d) and np.all(lat_bc==lat2d):
                var_bc_interp2d = var_bc.copy().values
                
            elif flag == '2D':
                var_bc_interp2d = interpolate.griddata(
                    (lon_bc.ravel(),lat_bc.ravel()),
                    var_bc.values.ravel(),
                    (lon2d.ravel(),lat2d.ravel())
                    ).reshape((ny,nx))
                
            elif flag == '3D':
                
                var_bc_interp2d = xr.DataArray(
                    np.empty((bc_times.size,ny,nx)),
                    dims=[name_var_bc['time'],'y','x'],
                    coords={name_var_bc['time']:bc_times.values})
                
                for t in range(var_bc.shape[0]):
                    var_bc_interp2d[t] = interpolate.griddata(
                        (lon_bc.ravel(),lat_bc.ravel()),
                        var_bc[t].values.ravel(),
                        (lon2d.ravel(),lat2d.ravel())
                        ).reshape((ny,nx))
            
            #####################
            # Time processing
            #####################
            if flag == '2D':
                # Only one field, use it at every timestamps
                for t in range(NT):
                    var_bc_interpTime[t] = var_bc_interp2d.reshape((ny,nx))
            elif flag == '3D':
                # Time interpolations
                try:
                    var_bc_interpTime = var_bc_interp2d.interp(
                        {name_var_bc['time']:timestamps}).values
                    # Get the closest valid map for each non valid maps
                    ind_all_nan = np.array([],dtype=int)
                    ind_no_nan = np.array([],dtype=int)
                    for t in range(NT):
                        if np.all(np.isnan(var_bc_interpTime[t])):
                            ind_all_nan = np.append(ind_all_nan,t)  
                        else:
                            ind_no_nan = np.append(ind_no_nan,t)  
                    
                    # We replace by 0 for missing timesteps
                    for ind in ind_all_nan:
                        ind_to_replace = ind_no_nan[np.argmin(np.abs(ind-ind_no_nan))]
                        var_bc_interpTime[ind,:,:] = var_bc_interpTime[ind_to_replace,:,:]
                    
                except:
                    print('Warning: impossible to interpolate boundary conditions')
        else:
            print('Warning: no boundary conditions provided')
    except Exception as e:
        
        print('Warning: an error occured in the boundary condition processing.')
        print('We set to 0')
        print('Code crashed because:',e)
        
    if NT == 1:
        var_bc_interpTime = var_bc_interpTime[0]
    
    var_bc_interpTime[np.abs(var_bc_interpTime)>10e10] = np.nan
    
    #####################
    # Weight map
    #####################
    if coast:
        if mask is not None:
            mask = +mask
        else:
            if NT==1:
                mask = np.isnan(var_bc_interpTime)
            else:
                mask = np.isnan(np.sum(var_bc_interpTime,axis=0))
        if depth is not None and mindepth is not None:
            mask[depth>mindepth] = True
    else:
        mask = np.zeros(lon2d.shape,dtype=bool)
    
            
    bc_weight = compute_weight_map(lon2d,lat2d,
                                   mask,
                                   dist_bc) 
    
    var_bc_interpTime[np.isnan(var_bc_interpTime)] = 0
    
    
    if flag_plot >= 1:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7))
        if NT == 1:
            im0 = ax0.pcolormesh(lon2d, lat2d, var_bc_interpTime)
        else:
            im0 = ax0.pcolormesh(lon2d, lat2d, var_bc_interpTime[0],cmap='RdBu_r')
        ax0.set_title('BC field')
        plt.colorbar(im0, ax=ax0)
        im1 = ax1.pcolormesh(lon2d, lat2d, bc_weight)
        ax1.set_title('BC weights')
        plt.colorbar(im1, ax=ax1)
        plt.show()
        plt.close()
    
    
        
    return var_bc_interpTime, bc_weight

    
    
def compute_weight_map(lon2d,lat2d,mask,dist_scale,bc=True,slope=10):
    
    #####################
    # Compute weights map
    #####################
    coords = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    coords_cart = geo2cart(coords)
    # construct KD-tree
    ground_pixel_tree = spatial.cKDTree(coords_cart)
    # Compute dist_threshold (≈ half a grid spacing) via nearest-neighbor query
    n_probe = min(100, len(coords_cart))
    dd, _ = ground_pixel_tree.query(coords_cart[:n_probe], k=2)
    dist_threshold = dd[:, 1].min()
    # Add boundary pixels to mask (work on a copy to avoid mutating caller's array)
    mask = mask.copy()
    if bc:
        mask[0,:] = True
        mask[-1,:] = True
        mask[:,0] = True
        mask[:,-1] = True
    
    # get boundary coordinates
    lon_bc = lon2d[mask]
    lat_bc = lat2d[mask]
    coords_bc = np.column_stack((lon_bc, lat_bc))
    bc_tree = spatial.cKDTree(geo2cart(coords_bc))
    
    # For each pixel find its distance to the NEAREST boundary pixel.
    # Shifting by half a grid-spacing ensures the first row/column of interior 
    # pixels (immediately adjacent to the boundary) is treated as distance 0,
    # i.e. they receive weight = 1 — no discontinuous jump from land/boundary.
    dist_to_bc, _ = bc_tree.query(coords_cart, k=1)
    dist_to_bc = np.maximum(dist_to_bc - 0.5 * dist_threshold, 0.0)

    # Quintic smoothstep (C² at both ends):
    #   weight = 1  at the boundary  (dist = 0)
    #   weight = 0  at dist_scale    (interior)
    # Zero first AND second derivatives at both endpoints, giving a more
    # progressive transition than a cosine taper — the gradient is smallest
    # at the inner edge where interior waves first encounter the sponge,
    # reducing spurious reflections.
    x = np.clip(dist_to_bc / dist_scale, 0.0, 1.0)
    bc_weight = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    bc_weight = bc_weight.reshape(lon2d.shape)
    
    return bc_weight

    
