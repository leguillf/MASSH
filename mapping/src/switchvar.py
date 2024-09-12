#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:13:23 2019

@author: leguillou
"""

import numpy as np

from . import grid

def ssh2uv(ssh, State=None, lon=None, lat=None, xac=None,g=9.81):

    # if lon and lat are provided, we compute grid spacing. 
    # Otherwise, state grid will be used
    if lon is not None and lat is not None:
        f = 4*np.pi/86164*np.sin(lat*np.pi/180)
        dx,dy = grid.lonlat2dxdy(lon,lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY
        
    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx
        _dy = dy
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:,:,np.newaxis]
        _dx = dx[:,:,np.newaxis]
        _dy = dy[:,:,np.newaxis]

    # Initialization
    u = np.zeros_like(ssh)*np.nan
    v = np.zeros_like(ssh)*np.nan
    # Compute velocies
    u[1:-1,1:] = - g/f[1:-1,1:]*\
            (ssh[2:,:-1]+ssh[2:,1:]-ssh[:-2,1:]-ssh[:-2,:-1])/(4*_dy[1:-1,1:])
             
    v[1:,1:-1] = + g/f[1:,1:-1]*\
        (ssh[1:,2:]+ssh[:-1,2:]-ssh[:-1,:-2]-ssh[1:,:-2])/(4*_dx[1:,1:-1])
        
    if xac is not None:
        u = _masked_edge(u, xac)
        v = _masked_edge(v, xac)

    if ssh_shapelen == 3:
        u = np.moveaxis(u, -1, 0)
        v = np.moveaxis(v, -1, 0)

    return u,v

def ssh2rv(ssh, State=None, lon=None, lat=None, xac=None,g=9.81,norm=False):

    # if lon and lat are provided, we compute grid spacing. 
    # Otherwise, state grid will be used
    if lon is not None and lat is not None:
        f = 4*np.pi/86164*np.sin(lat*np.pi/180)
        dx,dy = grid.lonlat2dxdy(lon,lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY
        
    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx[1:-1,1:-1]
        _dy = dy[1:-1,1:-1]
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:,:,np.newaxis]
        _dx = dx[1:-1,1:-1,np.newaxis]
        _dy = dy[1:-1,1:-1,np.newaxis]

    # Initialization
    rv = np.zeros_like(ssh)*np.nan
    # Compute relative vorticity
    rv[1:-1,1:-1] = g/f[1:-1,1:-1] *\
        ((ssh[2:,1:-1]+ssh[:-2,1:-1]-2*ssh[1:-1,1:-1])/_dy**2 \
        +(ssh[1:-1,2:]+ssh[1:-1,:-2]-2*ssh[1:-1,1:-1])/_dx**2)
    if norm:
        rv /= f
        
    if xac is not None:
        rv = _masked_edge(rv, xac)
    

    if ssh_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)
    
    

    return rv


def uv2rv(UV, State=None, lon=None, lat=None, xac=None):
    # if lon and lat are provided, we compute grid spacing. 
    # Otherwise, state grid will be used
    try:
        dx,dy = grid.lonlat2dxdy(lon,lat)
    except:
        dx = State.DX
        dy = State.DY

    # Initialization
    u = UV[0]
    v = UV[1]

    uv_shapelen = len(u.shape)
    if uv_shapelen == 2:
        _dx = dx
        _dy = dy
    elif uv_shapelen == 3:
        u = np.moveaxis(u, 0, -1)
        v = np.moveaxis(v, 0, -1)
        _dx = dx[:,:,np.newaxis]
        _dy = dy[:,:,np.newaxis]

    mask = np.isnan(u)

    rv = np.zeros_like(u)
    
    rv = np.gradient(v, _dx, axis=1) - np.gradient(u, _dy, axis=0)

    if xac is not None:
        rv = _masked_edge(rv,xac)

    rv[mask] = np.nan

    if uv_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def _masked_edge(var,xac):
    
    """ _masked_edge

    mask the edges of the swath gap


    """

    if np.any(xac>0):
        ind_gap = (xac==np.nanmin(xac[xac>0]))
        if ind_gap.size==var.size:
            if ind_gap.shape!=var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size==var.shape[1]:
            var[:,ind_gap] = np.nan
    if np.any(xac<0):
        ind_gap = (xac==np.nanmax(xac[xac<0]))
        if ind_gap.size==var.size:
            if ind_gap.shape!=var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size==var.shape[1]:
            var[:,ind_gap] = np.nan

    return var
