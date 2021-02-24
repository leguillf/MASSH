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
import matplotlib.pylab as plt

class Grid():


  def __init__(self, X, Y, cartesian=False):

    ny, nx, = np.shape(X)

    mask = np.zeros((ny, nx)) + 2
    mask[:2,:] = 1
    mask[:,:2] = 1
    mask[-3:,:] = 1
    mask[:,-3:] = 1

    if cartesian:
        ##DEV## to generalize f0 computation
        from swotda.swotda_params_specific import N, L0, Rom, L, U
        dx = np.zeros((ny, nx)) + L0 * L / N
        dy = np.zeros((ny, nx)) + L0 * L / N
        #dx = np.gradient(X)[1]
        #dy = np.gradient(Y)[0]
        f0 = np.zeros((ny, nx)) - U/L/Rom
    else:
        dX = np.gradient(X)
        dY = np.gradient(Y)
        dx = np.sqrt((dX[1]*111000*np.cos(np.deg2rad(Y)))**2
                     + (dY[1]*111000)**2)
        dy = np.sqrt((dX[0]*111000*np.cos(np.deg2rad(Y)))**2
                     + (dY[0]*111000)**2)
        f0 = 2*2*np.pi/86164*np.sin(np.deg2rad(Y))

    np0 = np.shape(np.where(mask >= 1))[1]
    np2 = np.shape(np.where(mask == 2))[1]
    np1 = np.shape(np.where(mask == 1))[1]
    self.mask1d = np.zeros((np0))
    self.H = np.zeros((np0))
    self.c1d = np.zeros((np0))
    self.f01d = np.zeros((np0))
    self.dx1d = np.zeros((np0))
    self.dy1d = np.zeros((np0))
    self.indi = np.zeros((np0), dtype=np.int)
    self.indj = np.zeros((np0), dtype=np.int)
    self.vp1 = np.zeros((np1), dtype=np.int)
    self.vp2 = np.zeros((np2), dtype=np.int)
    self.vp2n = np.zeros((np2), dtype=np.int)
    self.vp2nn = np.zeros((np2), dtype=np.int)
    self.vp2s = np.zeros((np2), dtype=np.int)
    self.vp2ss = np.zeros((np2), dtype=np.int)
    self.vp2e = np.zeros((np2), dtype=np.int)
    self.vp2ee = np.zeros((np2), dtype=np.int)
    self.vp2w = np.zeros((np2), dtype=np.int)
    self.vp2ww = np.zeros((np2), dtype=np.int)
    self.vp2nw = np.zeros((np2), dtype=np.int)
    self.vp2ne = np.zeros((np2), dtype=np.int)
    self.vp2se = np.zeros((np2), dtype=np.int)
    self.vp2sw = np.zeros((np2), dtype=np.int)
    self.indp = np.zeros((ny,nx), dtype=np.int)

    p = -1
    for i in range(ny):
      for j in range(nx):
        if (mask[i,j] >= 1):
          p += 1
          self.mask1d[p] = mask[i,j]
          self.dx1d[p] = dx[i,j]
          self.dy1d[p] = dy[i,j]
          self.f01d[p] = f0[i,j]
          self.indi[p] = i
          self.indj[p] = j
          self.indp[i,j] = p


    p2 = -1
    p1 = -1
    for p in range(np0):
      if (self.mask1d[p] == 2):
        p2 += 1
        i = self.indi[p]
        j = self.indj[p]
        self.vp2[p2] = p
        self.vp2n[p2] = self.indp[i+1,j]
        self.vp2nn[p2] = self.indp[i+2,j]
        self.vp2s[p2] = self.indp[i-1,j]
        self.vp2ss[p2] = self.indp[i-2,j]
        self.vp2e[p2] = self.indp[i,j+1]
        self.vp2ee[p2] = self.indp[i,j+2]
        self.vp2w[p2] = self.indp[i,j-1]
        self.vp2ww[p2] = self.indp[i,j-2]
        self.vp2nw[p2] = self.indp[i+1,j-1]
        self.vp2ne[p2] = self.indp[i+1,j+1]
        self.vp2se[p2] = self.indp[i-1,j+1]
        self.vp2sw[p2] = self.indp[i-1,j-1]
      if (self.mask1d[p] == 1):
        p1 += 1
        i = self.indi[p]
        j = self.indj[p]
        self.vp1[p1] = p
    self.mask = mask
    self.f0 = f0
    self.dx = dx
    self.dy = dy
    self.np0 = np0
    self.np2 = np2
    self.nx = nx
    self.ny = ny
    self.X = X
    self.Y = Y
    self.g = 9.81


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
            Y[i,j] = y0 + np.sum(dx[:i,j])   
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


def laplacian(u, dx, dy):
    """
    Calculates the laplacian of u using the divergence and gradient functions and gives
    as output Ml.
    """
    Ml = np.gradient(np.gradient(u, dx, axis=0), dx, axis=0)\
         +  np.gradient(np.gradient(u, dy, axis=1), dy, axis=1);
    return Ml


def boundary_conditions(file_bc, lenght_bc, name_var_bc, timestamps,
                        lon2d, lat2d, flag_plot=0, sponge='gaspari-cohn'):

    if type(timestamps) in [int,float]:
        timestamps = np.array([timestamps])

    NT = timestamps.size
    ny,nx = lon2d.shape

    bc_field_interpTime = np.zeros((NT,ny,nx))


    if file_bc is not None and os.path.exists(file_bc):
        #####################
        # Read file
        #####################
        ds = xr.open_dataset(file_bc)
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
            bc_field = ds[name_var_bc['var']][ind_lonlat].values
        else:
            bc_field = []
            for i in np.where(ind_time)[0]:
                bc_field.append(ds[name_var_bc['var']][i].values[ind_lonlat])
            bc_field = np.asarray(bc_field)


        #####################
        # Grid processing
        #####################
        if np.all(bc_lon==lon2d.ravel()) and np.all(bc_lat==lat2d.ravel()):
            bc_field_interp2d = bc_field.reshape((bc_field.shape[0],ny,nx))
        elif flag == '2D':
            bc_field_interp2d = interpolate.griddata((bc_lon,bc_lat), bc_field, (lon2d.ravel(),lat2d.ravel()))
        elif flag == '3D':
            bc_field_interp2d = np.zeros((bc_field.shape[0], ny, nx))
            for t in range(bc_field.shape[0]):
                bc_field_interp2d[t] = interpolate.griddata((bc_lon,bc_lat), bc_field[t], (lon2d.ravel(),lat2d.ravel())).reshape((ny,nx))

        #####################
        # Time processing
        #####################
        if flag == '2D':
            # Only one field, use it at every timestamps
            for t in range(NT):
                bc_field_interpTime[t] = bc_field
        elif flag == '3D':
            # Time interpolations
            f_interpTime = interpolate.interp1d(bc_times, bc_field_interp2d, axis=0)
            for i,t in enumerate(timestamps):
                if bc_times.min() < t < bc_times.max():
                    bc_field_interpTime[i] = f_interpTime(t)

    if NT == 1:
        bc_field_interpTime = bc_field_interpTime[0]

    bc_field_interpTime[np.isnan(bc_field_interpTime)] = 0
    #####################
    # Compute weights map
    #####################
    bc_weight = np.ones((ny, nx))

    if sponge == 'gaspari-cohn':
        for r in range(np.max((ny, nx))):
            X = np.arange(r, nx-r)
            Y = np.arange(r, ny-r)
            if r < ny:
                bc_weight[r, X] = gaspari_cohn(np.array([r]), lenght_bc)
                bc_weight[ny-r-1, X] = gaspari_cohn(np.array([r]), lenght_bc)
            if r < nx:
                bc_weight[Y, r] = gaspari_cohn(np.array([r]), lenght_bc)
                bc_weight[Y, nx-r-1] = gaspari_cohn(np.array([r]), lenght_bc)
            if len(X) == 0 or len(Y) == 0:
                break

    elif sponge == 'linear':
        spn = np.linspace(0, 1, lenght_bc, endpoint=False)
        bc_weight[:lenght_bc, :] *= spn[:, None]
        bc_weight[:, :lenght_bc] *= spn[None, :]
        bc_weight[(-lenght_bc):, :] *= spn[::-1][:, None]
        bc_weight[:, (-lenght_bc):] *= spn[::-1][None, :]
        bc_weight = 1- bc_weight

    else:
        sys.exit('Error: No ' + sponge + ' sponge implemented'
              + 'for boundary conditions')

    if flag_plot > 1:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7))
        if NT == 1:
            im0 = ax0.pcolormesh(lon2d, lat2d, bc_field_interpTime)
        else:
            im0 = ax0.pcolormesh(lon2d, lat2d, bc_field_interpTime[0])
        ax0.set_title('BC field')
        plt.colorbar(im0, ax=ax0)
        im1 = ax1.pcolormesh(lon2d, lat2d, bc_weight)
        ax1.set_title('BC weights')
        plt.colorbar(im1, ax=ax1)
        plt.show()
        plt.close()

    return bc_field_interpTime, bc_weight
