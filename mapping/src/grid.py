#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:02:05 2019

@author: leguillou
"""

import os
import sys
import numpy as np
import xarray as xr
from scipy import interpolate
from .tools import gaspari_cohn
from datetime import datetime
import calendar
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pylab as plt

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

    coords = np.asarray(coords).astype(np.float)

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


def boundary_conditions(file_bc, dist_bc, name_var_bc, timestamps,
                        lon2d, lat2d, flag_plot=0, sponge='gaspari-cohn'):

    if type(timestamps) in [int,float]:
        timestamps = np.array([timestamps])

    NT = timestamps.size
    ny,nx = lon2d.shape

    bc_field_interpTime = np.zeros((NT,ny,nx))


    if file_bc is not None:
        #####################
        # Read file
        #####################
        ds = xr.open_mfdataset(file_bc,combine='by_coords')
        if 'time' in name_var_bc:
            flag = '3D'
            bc_times = ds[name_var_bc['time']].values
            # Convert to timestamps
            bc_times =  np.asarray([calendar.timegm(datetime.utcfromtimestamp(dt64.astype(int) * 1e-9).timetuple()) for dt64 in bc_times])
            dtime = np.max(bc_times[1:]-bc_times[:-1])
            # Extract relevant timestamps
            ind_time = (timestamps.min()-dtime<=bc_times) & (bc_times<=timestamps.max()+dtime)
            bc_times = bc_times[ind_time]

        else:
            flag = '2D'
        # Read and extract BC grid
        bc_lon = ds[name_var_bc['lon']].values % 360
        bc_lat = ds[name_var_bc['lat']].values
        if len(bc_lon.shape)==1:
            # Meshgrid
            bc_lon, bc_lat = np.meshgrid(bc_lon,bc_lat)
        # Extract relevant pixels
        dlon =  np.max(bc_lon[:,1:] - bc_lon[:,:-1])
        dlat =  np.max(bc_lat[1:,:] - bc_lat[:-1,:])
        ind_lonlat = (lon2d.min()-dlon<=bc_lon) & (bc_lon<=lon2d.max()+dlon) & (lat2d.min()-dlat<=bc_lat) & (bc_lat<=lat2d.max()+dlat)

        bc_lon = bc_lon[ind_lonlat]
        bc_lat = bc_lat[ind_lonlat]
        # Read BC fields
        if flag=='2D':
            bc_field = ds[name_var_bc['var']].values[ind_lonlat]
        else:
            bc_field = []
            for i in np.where(ind_time)[0]:
                bc_field.append(ds[name_var_bc['var']][i].values[ind_lonlat])
            bc_field = np.asarray(bc_field)


        #####################
        # Grid processing
        #####################
        if np.all(bc_lon==lon2d.ravel()) and np.all(bc_lat==lat2d.ravel()):
            bc_field_interp2d = bc_field.reshape((bc_field.size//(ny*nx),ny,nx))
        elif flag == '2D':
            bc_field_interp2d = interpolate.griddata((bc_lon,bc_lat), 
                                                     bc_field, 
                                                     (lon2d.ravel(),lat2d.ravel()))
        elif flag == '3D':
            bc_field_interp2d = np.zeros((bc_field.shape[0], ny, nx))
            for t in range(bc_field.shape[0]):
                bc_field_interp2d[t] = interpolate.griddata((bc_lon,bc_lat), 
                                                            bc_field[t], 
                                                            (lon2d.ravel(),lat2d.ravel())).reshape((ny,nx))

        #####################
        # Time processing
        #####################
        if flag == '2D':
            # Only one field, use it at every timestamps
            for t in range(NT):
                bc_field_interpTime[t] = bc_field_interp2d.reshape((ny,nx))
        elif flag == '3D':
            # Time interpolations
            f_interpTime = interpolate.interp1d(bc_times, bc_field_interp2d, axis=0)
            for i,t in enumerate(timestamps):
                if bc_times.min() <= t <= bc_times.max():
                    bc_field_interpTime[i] = f_interpTime(t)

    else:
        print('Warning: no boundary conditions provided')
        
    if NT == 1:
        bc_field_interpTime = bc_field_interpTime[0]

    bc_field_interpTime[np.abs(bc_field_interpTime)>10] = np.nan
    
    #####################
    # Compute weights map
    #####################
    coords = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    # construct KD-tree
    ground_pixel_tree = spatial.cKDTree(geo2cart(coords))
    subdomain = geo2cart(coords)[:100]
    eucl_dist = cdist(subdomain, subdomain, metric="euclidean")
    dist_threshold = np.min(eucl_dist[np.nonzero(eucl_dist)])
    # Compute boundary pixels, including land pixels
    mask = np.isnan(bc_field_interpTime[0])
    mask[0,:] = True
    mask[-1,:] = True
    mask[:,0] = True
    mask[:,-1] = True
    
    # get boundary coordinates$
    lon_bc = lon2d[mask]
    lat_bc = lat2d[mask]
    coords_bc = np.column_stack((lon_bc, lat_bc))
    bc_tree = spatial.cKDTree(geo2cart(coords_bc))
    # Compute distance between model pixels and boundary pixels
    dist_mx = ground_pixel_tree.sparse_distance_matrix(bc_tree,2*dist_bc)
    # Initialize weight map
    bc_weight = np.zeros(lon2d.size)
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
    df['tapering'] = np.exp(-(df['dist']**2/(2*(0.5*dist_bc)**2)))
    # Nudge values out of pixels
    df.loc[df.dist > 0, "weight"] *= df.loc[df.dist > 0, "tapering"]
    # Compute weight average and save it
    df['tapering'] = df['tapering']**10
    wa = lambda x: np.average(x, weights=df.loc[x.index, "tapering"])
    dfg = df.groupby('ind_mod')
    weights = dfg['weight'].apply(wa)
    bc_weight[weights.index] = np.array(weights)
    bc_weight = bc_weight.reshape(lon2d.shape)

    if flag_plot > 1:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7))
        if NT == 1:
            im0 = ax0.pcolormesh(lon2d, lat2d, bc_field_interpTime)
        else:
            im0 = ax0.pcolormesh(lon2d, lat2d, bc_field_interpTime[NT//2],cmap='RdBu_r')
        ax0.set_title('BC field')
        plt.colorbar(im0, ax=ax0)
        im1 = ax1.pcolormesh(lon2d, lat2d, bc_weight)
        ax1.set_title('BC weights')
        plt.colorbar(im1, ax=ax1)
        plt.show()
        plt.close()
    
    bc_field_interpTime[np.isnan(bc_field_interpTime)] = 0
        
    return bc_field_interpTime, bc_weight
