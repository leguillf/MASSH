#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:15:36 2019

@author: leguillou
"""
import numpy as np 
import xarray as xr
import scipy.linalg as spl
from netCDF4 import Dataset
import scipy



def gaspari_cohn(r,c):
    """
    NAME 
        bfn_gaspari_cohn

    DESCRIPTION 
        Gaspari-Cohn function. Inspired from E.Cosmes.
        
        Args: 
            r : array of value whose the Gaspari-Cohn function will be applied
            c : Distance above which the return values are zeros


        Returns:  smoothed values 
            
    """ 
    if type(r) is float or type(r) is int:
        ra = np.array([r])
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp


def hat_function(r,c):
    """
    NAME 
        hat_function
        
    DESCRIPTION 
        linear hat function         
        Args: 
            r : array of value whose the hat function will be applied
            c : Distance above which the return values are zeros
            
        Returns:  smoothed values
            
    """ 

    return (c - np.abs(r))/c

def L2_scalar_prod(v1,v2,coeff=1):
    """
    NAME 
        L2_scalar_prod
        
    DESCRIPTION 
        compute L2 scalar product between two L2 vectors        
        Args: 
            v1, v2: the two L2 vectors 
            coeff (default 1): multiplicating coefficient
            
        Returns:  scalar product
            
    """ 
    if np.shape(v1)==() and np.shape(v2)==() : 
        res = coeff*v1*v2
    else :
        ind = np.isfinite(v1+v2)
        res = coeff*np.sum(np.multiply(v1[np.where(ind)],v2[np.where(ind)]))
    return res


def detrendn(da, axes=None):
    
    """
    Detrend by subtracting out the least-square plane or least-square cubic fit
    depending on the number of axis.
    Parameters
    ----------
    da : `dask.array`
        The data to be detrended
    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
    
    if axes is None:
        axes = range(len(da.shape))
        
    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])
            
    if len(N) == 2:
        G = np.ones((N[0]*N[1],3))
        for i in range(N[0]):
            G[N[1]*i:N[1]*i+N[1], 1] = i+1
            G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
        if type(da) == xr.DataArray:
            d_obs = np.reshape(da.copy().values, (N[0]*N[1],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    elif len(N) == 3:
        if type(da) == xr.DataArray:
            if da.ndim > 3:
                raise NotImplementedError("Cubic detrend is not implemented "
                                         "for 4-dimensional `xarray.DataArray`."
                                         " We suggest converting it to "
                                         "`dask.array`.")
            else:
                d_obs = np.reshape(da.copy().values, (N[0]*N[1]*N[2],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1]*N[2],1))

        G = np.ones((N[0]*N[1]*N[2],4))
        G[:,3] = np.tile(np.arange(1,N[2]+1), N[0]*N[1])
        ys = np.zeros(N[1]*N[2])
        for i in range(N[1]):
            ys[N[2]*i:N[2]*i+N[2]] = i+1
        G[:,2] = np.tile(ys, N[0])
        for i in range(N[0]):
            G[len(ys)*i:len(ys)*i+len(ys),1] = i+1
    else:
        raise NotImplementedError("Detrending over more than 4 axes is "
                                 "not implemented.")

    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend


def read_auxdata_geos(file_aux):

    # Read spectrum database
    fcid = Dataset(file_aux, 'r')
    ff1 = np.array(fcid.variables['f'][:])
    lon = np.array(fcid.variables['lon'][:])
    lat = np.array(fcid.variables['lat'][:])
    NOISEFLOOR = np.array(fcid.variables['NOISEFLOOR'][:,:])
    PSDS = np.array(fcid.variables['PSDS'][:,:,:])
    tdec = np.array(fcid.variables['tdec'][:,:,:])

    finterpPSDS = scipy.interpolate.RegularGridInterpolator((ff1,lat,lon),PSDS,bounds_error=False,fill_value=None)
    finterpTDEC = scipy.interpolate.RegularGridInterpolator((ff1,lat,lon),tdec,bounds_error=False,fill_value=None)
    #finterpTDEC = []
    finterpNOISEFLOOR = scipy.interpolate.RegularGridInterpolator((lat,lon),NOISEFLOOR,bounds_error=False,fill_value=None)

    return finterpPSDS,finterpTDEC,  finterpNOISEFLOOR


def read_auxdata_geosc(filec_aux):

    # Read spectrum database
    fcid = Dataset(filec_aux, 'r')
    lon = np.array(fcid.variables['lon'][:])
    lat = np.array(fcid.variables['lat'][:])
    C1 = np.array(fcid.variables['c1'][:,:])
    finterpC = scipy.interpolate.RegularGridInterpolator((lon,lat),C1,bounds_error=False,fill_value=None)
    return finterpC

def read_auxdata_depth(filec_aux):

    # Read spectrum database
    fcid = Dataset(filec_aux, 'r')
    lon = np.array(fcid.variables['lon'][:])
    lat = np.array(fcid.variables['lat'][:])
    DEPTH=np.array(fcid.variables['H'][:,:])
    finterpDEPTH = scipy.interpolate.RegularGridInterpolator((lon,lat),DEPTH,bounds_error=False,fill_value=None)
    return finterpDEPTH    

def read_auxdata_varcit(file_aux):

    # Read spectrum database
    fcid = Dataset(file_aux, 'r')
    lon = np.array(fcid.variables['lon'][:])
    lat = np.array(fcid.variables['lat'][:])
    VARIANCE=np.array(fcid.variables['variance'][:,:])
    finterpVARIANCE = scipy.interpolate.RegularGridInterpolator((lon,lat),VARIANCE.T,bounds_error=False,fill_value=None)
    return finterpVARIANCE   


def read_auxdata_mdt(filemdt_aux,name_var):

    # Read spectrum database
    fcid = Dataset(filemdt_aux, 'r')
    lon = np.array(fcid.variables[name_var['lon']][:])
    lat = np.array(fcid.variables[name_var['lat']][:])
    mdt = np.array(fcid.variables[name_var['mdt']]).squeeze()
    if mdt.shape[1]==lon.size:
        mdt = mdt.transpose()
    finterpMDT = scipy.interpolate.RegularGridInterpolator((lon,lat),mdt,bounds_error=False,fill_value=None)
    return finterpMDT

