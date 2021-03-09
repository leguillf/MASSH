#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:13:23 2019

@author: leguillou
"""

import numpy as np
import pickle
import os.path

from .grid import Grid

def ssh2pv(ssh, lon, lat, c, name_grd=None, xac=None):
    if name_grd is not None:
        if os.path.isfile(name_grd):
            with open(name_grd, 'rb') as f:
                grd = pickle.load(f)
        else:
            grd = Grid(lon, lat)
            with open(name_grd, 'wb') as f:
                pickle.dump(grd, f)
                f.close()
    else:
        grd = Grid(lon, lat)

    g = grd.g
    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        f0 = grd.f0
        dx = grd.dx[1:-1,1:-1]
        dy = grd.dy[1:-1,1:-1]
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f0 = grd.f0[:,:,np.newaxis]
        dx = grd.dx[1:-1,1:-1,np.newaxis]
        dy = grd.dy[1:-1,1:-1,np.newaxis]

    # Initialization
    pv = np.zeros_like(ssh)

    # Compute relative vorticity
    #pv[t] = laplacian(factor*ssh[t],dx,dy) - g*f0/(c**2) * ssh[t]
    pv[1:-1,1:-1] = g/f0[1:-1,1:-1]*((ssh[2:,1:-1]+ssh[:-2,1:-1]-2*ssh[1:-1,1:-1])/dy**2 \
                                      + (ssh[1:-1,2:]+ssh[1:-1,:-2]-2*ssh[1:-1,1:-1])/dx**2) \
                                      - g*f0[1:-1,1:-1]/(c**2) * ssh[1:-1,1:-1]
    if xac is not None:
        pv = _masked_edge(pv, xac)

    if ssh_shapelen == 3:
        pv = np.moveaxis(pv, -1, 0)

    return pv


def ssh2rv(ssh, lon, lat, name_grd=None, xac=None):
    if name_grd is not None:
        if os.path.isfile(name_grd):
            with open(name_grd, 'rb') as f:
                grd = pickle.load(f)
        else:
            grd = Grid(lon, lat)
            with open(name_grd, 'wb') as f:
                pickle.dump(grd, f)
                f.close()
    else:
        grd = Grid(lon, lat)

    g = grd.g
    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        f0 = grd.f0
        dx = grd.dx[1:-1,1:-1]
        dy = grd.dy[1:-1,1:-1]
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f0 = grd.f0[:,:,np.newaxis]
        dx = grd.dx[1:-1,1:-1,np.newaxis]
        dy = grd.dy[1:-1,1:-1,np.newaxis]

    # Initialization
    rv = np.zeros_like(ssh)*np.nan
    # Compute relative vorticity
    rv[1:-1,1:-1] = g/f0[1:-1,1:-1] *\
        ((ssh[2:,1:-1]+ssh[:-2,1:-1]-2*ssh[1:-1,1:-1])/dy**2 \
        +(ssh[1:-1,2:]+ssh[1:-1,:-2]-2*ssh[1:-1,1:-1])/dx**2)
    if xac is not None:
        rv = _masked_edge(rv, xac)

    if ssh_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def uv2rv(UV, lon, lat, name_grd=None, xac=None):
    if name_grd is not None:
        if os.path.isfile(name_grd):
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = Grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else:
        grd = Grid(lon,lat)

    # Initialization
    u = UV[0]
    v = UV[1]

    uv_shapelen = len(u.shape)
    if uv_shapelen.len == 2:
        dx = grd.dx
        dy = grd.dy
    elif uv_shapelen == 3:
        u = np.moveaxis(u, 0, -1)
        v = np.moveaxis(v, 0, -1)
        dx = grd.dx[:,:,np.newaxis]
        dy = grd.dy[:,:,np.newaxis]

    mask = np.isnan(u)

    rv = np.zeros_like(u)
    
    rv = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)

    if xac is not None:
        rv = _masked_edge(rv,xac)

    rv[mask] = np.nan

    if uv_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv




def pv2ssh(lon, lat, q, hg, c, nitr=1, name_grd=''):
    """ Q to SSH

    This code solve a linear system of equations using Conjugate Gradient method

    Args:
        q (2D array): Potential Vorticity field
        hg (2D array): SSH guess
        grd (Grid() object): check modgrid.py

    Returns:
        h (2D array): SSH field.
    """
    def compute_avec(vec,aaa,bbb,grd):

        avec=np.empty(grd.np0,)
        avec[grd.vp2] = aaa[grd.vp2]*((vec[grd.vp2e]+vec[grd.vp2w]-2*vec[grd.vp2])/(grd.dx1d[grd.vp2]**2)+(vec[grd.vp2n]+vec[grd.vp2s]-2*vec[grd.vp2])/(grd.dy1d[grd.vp2]**2)) + bbb[grd.vp2]*vec[grd.vp2]
        avec[grd.vp1] = vec[grd.vp1]

        return avec,
    if name_grd is not None:
        if os.path.isfile(name_grd):
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = Grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else:
        grd = Grid(lon,lat)

    ny,nx,=np.shape(hg)
    g=grd.g


    x=hg[grd.indi,grd.indj]
    q1d=q[grd.indi,grd.indj]

    aaa=g/grd.f01d
    bbb=-g*grd.f01d/c**2
    ccc=+q1d

    aaa[grd.vp1]=0
    bbb[grd.vp1]=1
    ccc[grd.vp1]=x[grd.vp1]  ##boundary condition

    vec=+x

    avec,=compute_avec(vec,aaa,bbb,grd)
    gg=avec-ccc
    p=-gg

    for itr in range(nitr-1):
        vec=+p
        avec,=compute_avec(vec,aaa,bbb,grd)
        tmp=np.dot(p,avec)

        if tmp!=0. : s=-np.dot(p,gg)/tmp
        else: s=1.

        a1=np.dot(gg,gg)
        x=x+s*p
        vec=+x
        avec,=compute_avec(vec,aaa,bbb,grd)
        gg=avec-ccc
        a2=np.dot(gg,gg)

        if a1!=0: beta=a2/a1
        else: beta=1.

        p=-gg+beta*p

    vec=+p
    avec,=compute_avec(vec,aaa,bbb,grd)
    val1=-np.dot(p,gg)
    val2=np.dot(p,avec)
    if (val2==0.):
        s=1.
    else:
        s=val1/val2

    a1=np.dot(gg,gg)
    x=x+s*p

    # back to 2D
    h=np.empty((ny,nx))
    h[:,:]=np.NAN
    h[grd.indi,grd.indj]=x[:]


    return h


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
