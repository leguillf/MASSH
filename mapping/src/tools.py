#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:15:36 2019

@author: leguillou
"""
import numpy as np 
import os
import xarray as xr
import pandas as pd


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


