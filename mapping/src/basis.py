#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:24:24 2021

@author: leguillou
"""
import os, sys
import numpy as np
import logging
import pickle 
import xarray as xr
import scipy
from scipy.sparse import csc_matrix
import matplotlib.pylab as plt
from scipy.integrate import quad
from scipy.interpolate import griddata
import jax.numpy as jnp 
from jax.experimental import sparse
from jax import jit
from jax import vjp
import jax
from functools import partial
from jax.lax import scan
jax.config.update("jax_enable_x64", True)

import matplotlib.pylab as plt

def Basis(config, State, verbose=True, *args, **kwargs):
    """
    NAME
        Basis

    DESCRIPTION
        Main function calling subfunctions for specific Reduced Basis functions
    """
    
    if config.BASIS is None:
        return 
    
    elif config.BASIS.super is None:
        return Basis_multi(config, State, verbose=verbose)

    else:
        if verbose:
            print(config.BASIS)

        if config.BASIS.super=='BASIS_BM':
            return Basis_bm(config, State)

        elif config.BASIS.super=='BASIS_GEOCUR':
            return Basis_geocur(config,State)

        elif config.BASIS.super=='BASIS_GAUSS3D':
            return Basis_gauss3d(config,State)
        
        elif config.BASIS.super=='BASIS_BM_JAX':
            return Basis_bm_jax(config, State)
        
        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_bmaux(config,State)
        
        elif config.BASIS.super=='BASIS_LS':
            return BASIS_ls(config, State)
        
        elif config.BASIS.super=='BASIS_IT':
            return Basis_it(config, State)

        elif config.BASIS.super=='BASIS_CONSTANT':
            return Basis_constant(config, State)

        else:
            sys.exit(config.BASIS.super + ' not implemented yet')


class Basis_bm:
   
    def __init__(self,config,State):

        self.km2deg=1./110
        
        # Internal params
        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns 
        self.facnlt = config.BASIS.facnlt
        self.npsp = config.BASIS.npsp 
        self.facpsp = config.BASIS.facpsp 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tdecmin = config.BASIS.tdecmin
        self.tdecmax = config.BASIS.tdecmax
        self.factdec = config.BASIS.factdec
        self.sloptdec = config.BASIS.sloptdec
        self.Qmax = config.BASIS.Qmax
        self.facQ = config.BASIS.facQ
        self.slopQ = config.BASIS.slopQ
        self.lmeso = config.BASIS.lmeso
        self.tmeso = config.BASIS.tmeso
        self.name_mod_var = config.BASIS.name_mod_var
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background
        
        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Mask
        if State.mask is not None and np.any(State.mask):
            self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Depth data
        if config.BASIS.file_depth is not None:
            ds = xr.open_dataset(config.BASIS.file_depth)
            lon_depth = ds[config.BASIS.name_var_depth['lon']].values
            lat_depth = ds[config.BASIS.name_var_depth['lat']].values
            var_depth = ds[config.BASIS.name_var_depth['var']].values
            finterpDEPTH = scipy.interpolate.RegularGridInterpolator((lon_depth,lat_depth),var_depth,bounds_error=False,fill_value=None)
            self.depth = -finterpDEPTH((self.lon1d,self.lat1d))
            self.depth[np.isnan(self.depth)] = 0.
            self.depth[np.isnan(self.depth)] = 0.

            self.depth1 = config.BASIS.depth1
            self.depth2 = config.BASIS.depth2
        else:
            self.depth = None

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow

    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        
        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./self.lmin),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ff = np.exp(logff)
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int32') # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        self.nwavemeso = 0

    
        for iff in range(nf):
            
            if 1/ff[iff]<self.lmeso:
                self.nwavemeso = nwave
                
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                _ENSLON = np.arange(
                        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        LON_MAX+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)
                _ENSLAT = np.repeat(ENSLAT1[I],len(_ENSLON))
                if self.mask1d is None:
                    _ENSLON1 = _ENSLON
                    _ENSLAT1 = _ENSLAT
                else:
                    # Avoid wave component for which the state grid points are full masked
                    _ENSLON1 = []
                    _ENSLAT1 = []
                    for (lon,lat) in zip(_ENSLON,_ENSLAT):
                        indphys = np.where(
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= .5/ff[iff]) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= .5/ff[iff])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    
                ENSLAT[iff] = np.concatenate(([ENSLAT[iff],_ENSLAT1]))
                ENSLON[iff] = np.concatenate(([ENSLON[iff],_ENSLON1]))

            NP[iff] = len(ENSLON[iff])
            tdec[iff] = self.tmeso*self.lmeso**(self.sloptdec) * ff[iff]**self.sloptdec
            if tdec[iff]<self.tdecmin:
                    tdec[iff] = self.tdecmin
            if tdec[iff]>self.tdecmax:
                tdec[iff] = self.tdecmax 
            tdec[iff] *= self.factdec
            enst[iff] = np.arange(-tdec[iff]/self.facnlt,deltat+tdec[iff]/self.facnlt , tdec[iff]/self.facnlt) 
            # Compute time integral for each frequency for normalization
            tt = np.linspace(-tdec[iff],tdec[iff])
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff])*(tt[i+1]-tt[i])
            norm_fact[iff] = tmp.max()

            nwave += ntheta*2*len(enst[iff])*NP[iff]
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
         
        Q = np.array([]) 
        iwave = 0
        self.iff_wavebounds = [None]*(nf+1)
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave
            if NP[iff]>0:
                _nwavet = 2*len(enst[iff])*ntheta*NP[iff]
                if 1/ff[iff]>self.lmeso:
                    # Constant
                    Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5*np.ones((_nwavet,))))
                else:
                    # Slope
                    Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5 * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavet,)))) 
                iwave += _nwavet
                    
                print(f'lambda={1/ff[iff]:.1E}',
                    f'nlocs={NP[iff]:.1E}',
                    f'tdec={tdec[iff]:.1E}',
                    f'Q={Q[-1]:.1E}')
        self.iff_wavebounds[-1] = iwave
        
    
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)

        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.tdec=tdec
        self.norm_fact = norm_fact
        self.enst=enst
        self.nbasis=Q.size
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff


        # Compute basis components
        self.Gx, self.Nx = self._compute_component_space() # in space
        self.Gt, self.Nt = self._compute_component_time(time) # in time
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            
        if return_q:
            return Xb, Q
    
    def _compute_component_space(self):

        Gx = [None,]*self.nf
        Nx = [None,]*self.nf

        for iff in range(self.nf):

            data = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,))
            indices = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((2*self.ntheta*self.NP[iff],),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                # Spatial tapering shape of the wavelet 
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]
                facd = np.ones((indphys.size))
                if self.depth is not None:
                    facd = (self.depth[indphys]-self.depth1)/(self.depth2-self.depth1)
                    facd[facd>1]=1.
                    facd[facd<0]=0.
                    indphys = indphys[facd>0]
                    xx = xx[facd>0]
                    yy = yy[facd>0]
                    facd = facd[facd>0]

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd

                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.cos(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.sin(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            sizes = sizes[:nwaves]
            indices = indices[:ind_tmp]
            data = data[:ind_tmp]

            indptr = np.zeros((nwaves+1),dtype=int)
            indptr[1:] = np.cumsum(sizes)

            Gx[iff] = csc_matrix((data, indices, indptr), shape=(self.nphys, nwaves))

        return Gx, Nx
    
    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency 
        Nt = {} # Number of wave times tw such as abs(tw-t)<tdec

        for t in time:

            Gt[t] = [None,]*self.nf
            Nt[t] = [0,]*self.nf

            for iff in range(self.nf):
                Gt[t][iff] = np.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],))
                ind_tmp = 0
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    if abs(dt) < self.tdec[iff]:
                        fact = self.window(dt / self.tdec[iff]) 
                        fact /= self.norm_fact[iff]
                        if fact!=0:   
                            Nt[t][iff] += 1
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta*self.NP[iff]] = fact   
                    ind_tmp += 2*self.ntheta*self.NP[iff]
        return Gt, Nt     

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            ind0 = np.nonzero(self.Gt[t][iff])[0]
            if ind0.size>0:
                GtXf = GtXf[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                phi += self.Gx[iff].dot(GtXf.sum(axis=0))
        phi = phi.reshape(self.shape_phys)

        # Update State
        if State is not None:
            State.params[self.name_mod_var] = phi
        else:
            return phi

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState.params[self.name_mod_var] is None:
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState.params[self.name_mod_var].ravel()
        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX

class Basis_geocur:
   
    def __init__(self,config,State):

        self.km2deg=1./110
        
        # Internal params
        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns 
        self.facnlt = config.BASIS.facnlt
        self.npsp = config.BASIS.npsp 
        self.facpsp = config.BASIS.facpsp 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tdecmin = config.BASIS.tdecmin
        self.tdecmax = config.BASIS.tdecmax
        self.factdec = config.BASIS.factdec
        self.sloptdec = config.BASIS.sloptdec
        self.Qmax = config.BASIS.Qmax
        self.facQ = config.BASIS.facQ
        self.slopQ = config.BASIS.slopQ
        self.lmeso = config.BASIS.lmeso
        self.tmeso = config.BASIS.tmeso
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()
        self.g = State.g

        # Mask
        if State.mask is not None:
            self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Depth data
        if config.BASIS.file_depth is not None:
            ds = xr.open_dataset(config.BASIS.file_depth)
            lon_depth = ds[config.BASIS.name_var_depth['lon']].values
            lat_depth = ds[config.BASIS.name_var_depth['lat']].values
            var_depth = ds[config.BASIS.name_var_depth['var']].values
            finterpDEPTH = scipy.interpolate.RegularGridInterpolator((lon_depth,lat_depth),var_depth,bounds_error=False,fill_value=None)
            self.depth = -finterpDEPTH((self.lon1d,self.lat1d))
            self.depth[np.isnan(self.depth)] = 0.
            self.depth[np.isnan(self.depth)] = 0.

            self.depth1 = config.BASIS.depth1
            self.depth2 = config.BASIS.depth2
        else:
            self.depth = None

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow

    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        
        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./self.lmin),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ff = np.exp(logff)
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int32') # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        self.nwavemeso = 0

    
        for iff in range(nf):
            
            if 1/ff[iff]<self.lmeso:
                self.nwavemeso = nwave
                
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                _ENSLON = np.arange(
                        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        LON_MAX+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)
                _ENSLAT = np.repeat(ENSLAT1[I],len(_ENSLON))
                if self.mask1d is None:
                    _ENSLON1 = _ENSLON
                    _ENSLAT1 = _ENSLAT
                else:
                    # Avoid wave component for which the state grid points are full masked
                    _ENSLON1 = []
                    _ENSLAT1 = []
                    for (lon,lat) in zip(_ENSLON,_ENSLAT):
                        indphys = np.where(
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= .5/ff[iff]) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= .5/ff[iff])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    
                    ENSLAT[iff] = np.concatenate(([ENSLAT[iff],_ENSLAT1]))
                    ENSLON[iff] = np.concatenate(([ENSLON[iff],_ENSLON1]))

            NP[iff] = len(ENSLON[iff])
            tdec[iff] = self.tmeso*self.lmeso**(self.sloptdec) * ff[iff]**self.sloptdec
            if tdec[iff]<self.tdecmin:
                    tdec[iff] = self.tdecmin
            if tdec[iff]>self.tdecmax:
                tdec[iff] = self.tdecmax 
            tdec[iff] *= self.factdec
            enst[iff] = np.arange(-tdec[iff]/self.facnlt,deltat+tdec[iff]/self.facnlt , tdec[iff]/self.facnlt) 
            # Compute time integral for each frequency for normalization
            tt = np.linspace(-tdec[iff],tdec[iff])
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff])*(tt[i+1]-tt[i])
            norm_fact[iff] = tmp.max()

            nwave += ntheta*2*len(enst[iff])*NP[iff]
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
         
        Q = np.array([]) 
        iwave = 0
        self.iff_wavebounds = [None]*(nf+1)
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave
            if NP[iff]>0:
                _nwavet = 2*len(enst[iff])*ntheta*NP[iff]
                if 1/ff[iff]>self.lmeso:
                    # Constant
                    Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5*np.ones((_nwavet,))))
                else:
                    # Slope
                    Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5 * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavet,)))) 
                iwave += _nwavet
                    
                print(f'lambda={1/ff[iff]:.1E}',
                    f'nlocs={NP[iff]:.1E}',
                    f'tdec={tdec[iff]:.1E}',
                    f'Q={Q[-1]:.1E}')
        self.iff_wavebounds[-1] = iwave
        
    
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)

        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.tdec=tdec
        self.norm_fact = norm_fact
        self.enst=enst
        self.nbasis=Q.size
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff


        # Compute basis components
        self.Gx_u, self.Nx = self._compute_component_space(angle=0) # in space
        self.Gx_v, self.Nx = self._compute_component_space(angle=np.pi/2) # in space
        self.Gt, self.Nt = self._compute_component_time(time) # in time
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            
        if return_q:
            return Xb, Q
    
    def _compute_component_space(self, angle):

        eps = 0.01 # in km, to convert the H wavelets into equivalent current wavelets
        epsx = eps * np.cos(angle - np.pi/2)
        epsy = eps * np.sin(angle - np.pi/2)

        Gx = [None,]*self.nf
        Nx = [None,]*self.nf

        for iff in range(self.nf):

            data = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,))
            indices = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((2*self.ntheta*self.NP[iff],),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg 
                # Spatial tapering shape of the wavelet 
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]
                facd = np.ones((indphys.size))
                if self.depth is not None:
                    facd = (self.depth[indphys]-self.depth1)/(self.depth2-self.depth1)
                    facd[facd>1]=1.
                    facd[facd<0]=0.
                    indphys = indphys[facd>0]
                    xx = xx[facd>0]
                    yy = yy[facd>0]
                    facd = facd[facd>0]

                facs = mywindow((xx+epsx) / self.DX[iff]) * mywindow((yy+epsy) / self.DX[iff]) * facd
                fc = 2*2*np.pi/86164*np.sin(self.ENSLAT[iff][P]*np.pi/180.) # Coriolis parameter
                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = self.g/fc * np.sqrt(2) * facs *\
                        (np.cos(kx*(xx+epsx)+ky*(yy+epsy)) - np.cos(kx*(xx)+ky*(yy))) / (eps*1000)
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = self.g/fc * np.sqrt(2) * facs *\
                        (np.sin(kx*(xx+epsx)+ky*(yy+epsy)) - np.sin(kx*(xx)+ky*(yy))) / (eps*1000)
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            sizes = sizes[:nwaves]
            indices = indices[:ind_tmp]
            data = data[:ind_tmp]

            indptr = np.zeros((nwaves+1),dtype=int)
            indptr[1:] = np.cumsum(sizes)

            Gx[iff] = csc_matrix((data, indices, indptr), shape=(self.nphys, nwaves))

        return Gx, Nx
    
    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency 
        Nt = {} # Number of wave times tw such as abs(tw-t)<tdec

        for t in time:

            Gt[t] = [None,]*self.nf
            Nt[t] = [0,]*self.nf

            for iff in range(self.nf):
                Gt[t][iff] = np.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],))
                ind_tmp = 0
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    if abs(dt) < self.tdec[iff]:
                        fact = self.window(dt / self.tdec[iff]) 
                        fact /= self.norm_fact[iff]
                        if fact!=0:   
                            Nt[t][iff] += 1
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta*self.NP[iff]] = fact   
                    ind_tmp += 2*self.ntheta*self.NP[iff]
        return Gt, Nt     

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        #############
        # u-component
        #############
        # Projection
        phi_u = np.zeros((self.nphys,))
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            ind0 = np.nonzero(self.Gt[t][iff])[0]
            if ind0.size>0:
                GtXf = GtXf[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                phi_u += self.Gx_u[iff].dot(GtXf.sum(axis=0))
        phi_u = phi_u.reshape(self.shape_phys)

        # Update State
        State.params[self.name_mod_u] = phi_u

        #############
        # v-component
        #############
        # Projection
        phi_v = np.zeros((self.nphys,))
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            ind0 = np.nonzero(self.Gt[t][iff])[0]
            if ind0.size>0:
                GtXf = GtXf[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                phi_v += self.Gx_v[iff].dot(GtXf.sum(axis=0))
        phi_v = phi_v.reshape(self.shape_phys)

        # Update State
        State.params[self.name_mod_v] = phi_v



    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState.params[self.name_mod_u] is None:
            adState.params[self.name_mod_u] = np.zeros((self.nphys,))
        if adState.params[self.name_mod_v] is None:
            adState.params[self.name_mod_v] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)

        #############
        # u-component
        #############
        adparams_u = adState.params[self.name_mod_u].ravel()
        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx_u[iff].T.dot(adparams_u)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        adState.params[self.name_mod_u] *= 0.

        #############
        # v-component
        #############
        adparams_v = adState.params[self.name_mod_v].ravel()
        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx_v[iff].T.dot(adparams_v)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        adState.params[self.name_mod_v] *= 0.
        
        return adX
      

class Basis_bm_offset: 

    def __init__(self,config,State):

        self.km2deg=1./110
        
        # Internal params
        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns # factor for wavelet spacing= space
        self.facnlt = config.BASIS.facnlt
        self.npsp = config.BASIS.npsp # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp = config.BASIS.facpsp # 1.5 # factor to fix df between wavelets 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tdecmin = config.BASIS.tdecmin
        self.tdecmax = config.BASIS.tdecmax
        self.factdec = config.BASIS.factdec
        self.sloptdec = config.BASIS.sloptdec
        self.Qmax = config.BASIS.Qmax
        self.facQ = config.BASIS.facQ
        self.slopQ = config.BASIS.slopQ
        self.lmeso = config.BASIS.lmeso
        self.tmeso = config.BASIS.tmeso
        self.name_mod_var = config.BASIS.name_mod_var

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Offset parameter 
        self.offset = config.BASIS.offset 
        self.tdec_offset = config.BASIS.tdec_offset 
        self.facnlt_offset = config.BASIS.facnlt_offset

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow

    def set_basis(self,time,offset,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        
        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./self.lmin),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ff = np.exp(logff)
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        self.nwavemeso = 0
        lonmax = LON_MAX
        if (LON_MAX<LON_MIN): lonmax = LON_MAX+360.
        
    
        for iff in range(nf):
            
            if 1/ff[iff]<self.lmeso:
                self.nwavemeso = nwave
                
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = np.mod(
                    np.arange(
                        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        lonmax+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                    360)
                ENSLAT[iff] = np.concatenate(([ENSLAT[iff],np.repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON[iff] = np.concatenate(([ENSLON[iff],ENSLON1]))
            NP[iff] = len(ENSLON[iff])
            tdec[iff] = self.tmeso*self.lmeso**(self.sloptdec) * ff[iff]**self.sloptdec
            if tdec[iff]<self.tdecmin:
                    tdec[iff] = self.tdecmin
            if tdec[iff]>self.tdecmax:
                tdec[iff] = self.tdecmax 
            tdec[iff] *= self.factdec
            enst[iff] = np.arange(-tdec[iff]/self.facnlt,deltat+tdec[iff]/self.facnlt , tdec[iff]/self.facnlt) 

            # Compute time integral for each frequency
            tt = np.linspace(-tdec[iff],tdec[iff])
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff])*(tt[i+1]-tt[i])
            norm_fact[iff] = tmp.max()


            nwave += ntheta*2*len(enst[iff])*NP[iff]
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
        
        Q = np.array([]) # Initial state      

        iwave = 0
        self.iff_wavebounds = [None]*nf
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave
            _nwavet = 2*len(enst[iff])*ntheta*NP[iff]
            if 1/ff[iff]>self.lmeso:
                Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5*np.ones((_nwavet,))))
                iwave += _nwavet
            else:
                Q = np.concatenate((Q,self.Qmax/(self.facns*self.facnlt)**.5 * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavet,)))) 
                iwave += _nwavet
                
            print(f'lambda={1/ff[iff]:.1E}',
                  f'nlocs={NP[iff]:.1E}',
                  f'tdec={tdec[iff]:.1E}',
                  f'Q={Q[-1]:.1E}')
        
        if offset : # creating enst_offset and Q values if offset = True

            enst_offset = np.arange(-self.tdec_offset/self.facnlt,deltat+self.tdec_offset/self.facnlt , self.tdec_offset/self.facnlt) # array of time centers for offsets 
            self.enst_offset=enst_offset

            _nwave_offset = len(self.enst_offset) # number of offset waves for the model error 

            Q = np.concatenate((Q,self.Qmax/(self.facnlt)**.5*np.ones((_nwave_offset,))))
            iwave += _nwave_offset

        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.tdec=tdec
        self.norm_fact = norm_fact
        self.enst=enst
        self.nbasis=Q.size
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff


        # Compute basis components

        ## In space
        self.indx, self.facGx = self._compute_component_space()

        ## In time
        self.facGt = {}
        for t in time:
            facGt = self._compute_component_time(t)
            self.facGt[t] = facGt      
        
        ## For offset 
        if offset : 
            self.facGo = {}
            for t in time :
                facGo = self._compute_component_time(t,offset = True) # add the parameter for offset 
                self.facGo[t] = facGo

        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            
        if return_q:
            return Q
        
    def _compute_component_space(self):

        indx = [None,]*self.nf
        facGx = [None,]*self.nf

        for iff in range(self.nf):
            indx[iff] = [None,]*self.NP[iff]
            facGx[iff] = [None,]*self.NP[iff]
            for P in range(self.NP[iff]):
                # Obs selection around point P
                iobs = np.where(
                    (np.abs((np.mod(self.lon1d - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (np.mod(self.lon1d[iobs] - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[iobs] - self.ENSLAT[iff][P]) / self.km2deg

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff])

                indx[iff][P] = iobs
                
                facGx[iff][P] = [None,]*self.ntheta
                if iobs.shape[0] > 0:
                    for itheta in range(self.ntheta):
                        facGx[iff][P][itheta] = [[],[]]
                        kx = self.k[iff] * np.cos(self.theta[itheta])
                        ky = self.k[iff] * np.sin(self.theta[itheta])
                        facGx[iff][P][itheta][0] = np.sqrt(2) * facs * np.cos(kx*(xx)+ky*(yy))
                        facGx[iff][P][itheta][1] = np.sqrt(2) * facs * np.sin(kx*(xx)+ky*(yy))

        return indx, facGx

    def _compute_component_time(self, t, offset=False):

        if not offset :

            facGt = [None,]*self.nf

            for iff in range(self.nf):
                facGt[iff] = [None,]*(len(self.enst[iff])) 
                # Time spread wavelets
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    if abs(dt) < self.tdec[iff]:
                        fact = self.window(dt / self.tdec[iff]) 
                        fact /= self.norm_fact[iff]    
                        facGt[iff][it] = fact   

        else : 

            facGt = [None,]*(self.enst_offset)

            for it in range(len(self.enst_offset)):
                dt = t - self.enst_offset[it]
                try:
                    if abs(dt) < self.tdec:
                        fact = self.window(dt / self.tdec) 
                        fact /= self.norm_fact
                        facGt[it] = fact
                except:
                    print(f'Warning: an error occured at t={t}, enstloc={self.enst_offset[it]}')

        return facGt     

    def _proj(self, phi, X, t, transpose):

        iwave = 0

        if self.offset : 
            for it in range(len(self.enst_offset)):
                if self.facGo[t][it] is not None:
                    if transpose:
                        phi[iwave] = np.sum(X * self.facGo[t][it])
                    else:
                        phi += X[iwave] * self.facGo[t][it]
                iwave += 1

        for iff in range(self.nf):
            enstloc = self.enst[iff]
            for P in range(self.NP[iff]):
                iobs = self.indx[iff][P]
                if iobs.shape[0] > 0:
                    # Initial State
                    if t==0 and self.wavelet_init:
                        for itheta in range(self.ntheta):
                            for iphase in range(2):
                                if transpose:
                                    phi[iwave] = np.sum(X[iobs] * self.facGx[iff][P][itheta][iphase])
                                else:
                                    phi[iobs] += X[iwave] * self.facGx[iff][P][itheta][iphase]
                                iwave += 1
                    elif self.wavelet_init:
                        iwave += 2*self.ntheta

                    # Time spread wavelets
                    for it in range(len(enstloc)):
                        if self.facGt[t][iff][it] is None:
                            iwave += 2*self.ntheta
                        else:
                            for itheta in range(self.ntheta):
                                for iphase in range(2):
                                    if transpose:
                                        phi[iwave] = np.sum(X[iobs] * self.facGx[iff][P][itheta][iphase] * self.facGt[t][iff][it])
                                    else:
                                        phi[iobs] += X[iwave] * self.facGx[iff][P][itheta][iphase] * self.facGt[t][iff][it]
                                    iwave += 1        

    def operg(self, t, X, transpose=False,State=None):
        
        """
            Project to physicial space
        """

        # Projection
        if transpose:
            X = X.flatten()
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.nphys,))

        self._proj(phi, X, t, transpose)
        
        # Reshaping
        if not transpose:
            phi = phi.reshape(self.shape_phys)

        if State is not None:
            State.params[self.name_mod_var] = phi
        else:
            return phi

    def operg_transpose(self, t, adState):

        """
            Project to reduced space
        """
        
        if adState.params[self.name_mod_var] is None:
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))
        adX = self.operg(t, adState.params[self.name_mod_var], transpose=True)
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX


class Basis_constant: 

    def __init__(self,config,State):
        
        # Internal params 
        self.flux = config.BASIS.flux
        self.Qmax = config.BASIS.Qmax
        self.facnlt = config.BASIS.facnlt
        self.name_mod_var = config.BASIS.name_mod_var
        self.tdec = config.BASIS.tdec 

        # Grid params 
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
    
    def set_basis(self,time,return_q=False):

        TIME_MIN = time.min()
        TIME_MAX = time.max()
        
        deltat = (TIME_MAX-TIME_MIN)

        enst = np.arange(-self.tdec/self.facnlt,deltat+self.tdec/self.facnlt , self.tdec/self.facnlt) # array of time centers for offsets 
        self.enst=enst


        # Compute time integral for each frequency
        tt = np.linspace(-self.tdec,self.tdec)
        tmp = np.zeros_like(tt)
        for i in range(tt.size-1):
            tmp[i+1] = tmp[i] + self.window(tt[i]/self.tdec)*(tt[i+1]-tt[i])
        norm_fact = tmp.max()
        self.norm_fact =norm_fact

        _nwavet = len(enst) # number of offset for the model error 


        # creating Q matrix 

        Q = self.Qmax/(self.facnlt)**.5*np.ones((_nwavet,))

        self.nbasis=Q.size

        # Compute basis components


        self.facG = {}

        for t in time:
            self.facG[t] = self._compute_component(t)
 
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        if return_q:
            return Q


    def _compute_component(self,t):

        enstloc = self.enst # time centers of wavelets  
        facGt = [None,]*(len(enstloc))

        for it in range(len(enstloc)):
            dt = t - enstloc[it]
            try:
                if abs(dt) < self.tdec:
                    fact = self.window(dt / self.tdec) 
                    fact /= self.norm_fact
                    facGt[it] = fact
            except:
                print(f'Warning: an error occured at t={t}, enstloc={enstloc[it]}')

        return facGt
    
    def _proj(self, phi, X, t, transpose):

        iwave = 0

        enstloc = self.enst

        for it in range(len(enstloc)):
            if self.facG[t][it] is not None:
                if transpose:
                    phi[iwave] = np.sum(X * self.facG[t][it])
                else:
                    phi += X[iwave] * self.facG[t][it]
            iwave += 1
                

    def operg(self, t, X, transpose=False,State=None):
            
            """
                Project to physicial space
            """

            # Projection
            if transpose:
                X = X.flatten()
                phi = np.zeros((self.nbasis,))
            else:
                phi = np.zeros((self.nphys,))

            self._proj(phi, X, t, transpose)
            
            # Reshaping
            if not transpose:
                phi = phi.reshape(self.shape_phys)

            if State is not None:
                State.params[self.name_mod_var] = phi
            else:
                return phi

    def operg_transpose(self, t, adState):
    
        """
            Project to reduced space
        """
        
        if adState.params[self.name_mod_var] is None:
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))
        adX = self.operg(t, adState.params[self.name_mod_var], transpose=True)
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX




class Basis_gauss3d:
   
    def __init__(self, config, State):

        self.km2deg =1./110

        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns
        self.facnlt = config.BASIS.facnlt
        self.sigma_D = config.BASIS.sigma_D
        self.sigma_T = config.BASIS.sigma_T
        self.sigma_Q = config.BASIS.sigma_Q
        self.name_mod_var = config.BASIS.name_mod_var

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
    
    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time
        
        # coordinates in space
        ENSLAT1 = np.arange(
            LAT_MIN - self.sigma_D*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.sigma_D/self.facns*self.km2deg, self.sigma_D/self.facns*self.km2deg)
        ENSLAT = []
        ENSLON = []
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(
                np.arange(
                    LON_MIN - self.sigma_D*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.sigma_D/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    self.sigma_D/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                360)
            ENSLAT = np.concatenate(([ENSLAT,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON = np.concatenate(([ENSLON,ENSLON1]))
        self.ENSLAT = ENSLAT
        self.ENSLON = ENSLON
        
        # coordinates in time
        ENST = np.arange(-self.sigma_T*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.sigma_T/self.facnlt , self.sigma_T/self.facnlt)
        self.ENST = ENST
        
        # Gaussian functions in space
        data = np.empty((ENSLAT.size*self.lon1d.size,))
        indices = np.empty((ENSLAT.size*self.lon1d.size,),dtype=int)
        sizes = np.zeros((ENSLAT.size,),dtype=int)
        ind_tmp = 0
        for i,(lat0,lon0) in enumerate(zip(ENSLAT,ENSLON)):
            indphys = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
                    )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[indphys] - lat0) / self.km2deg

            sizes[i] = indphys.size
            indices[ind_tmp:ind_tmp+indphys.size] = indphys
            data[ind_tmp:ind_tmp+indphys.size] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
            ind_tmp += indphys.size
        indptr = np.zeros((i+2),dtype=int)
        indptr[1:] = np.cumsum(sizes)
        Gauss_xy = csc_matrix((data, indices, indptr), shape=(self.lon1d.size, ENSLAT.size))
        
        # Gaussian functions in time
        Gauss_t = {}
        Nt = {}
        for t in time:
            Gauss_t[t] = np.zeros((ENSLAT.size*ENST.size))
            Nt[t] = 0
            ind_tmp = 0
            for it in range(len(ENST)):
                dt = t - ENST[it]
                if abs(dt) < self.sigma_T:
                    fact = self.window(dt / self.sigma_T) / self.sigma_T 
                    if fact!=0:   
                        Nt[t] += 1
                        Gauss_t[t][ind_tmp:ind_tmp+ENSLAT.size] = fact   
                ind_tmp += ENSLAT.size
        
        
        self.Gauss_xy = Gauss_xy
        self.Gauss_t = Gauss_t
        self.Nt = Nt
        self.Nx = ENSLAT.size
        self.nbasis = ENST.size * ENSLAT.size
        self.nphys = self.lon1d.size
        self.shape_phys = [self.ny, self.nx]
        self.shape_basis = [ENST.size,ENSLAT.size]
        
        # Fill Q matrix
        Q = self.sigma_Q / (self.facns*self.facnlt)  * np.ones((self.nbasis))

        if return_q:
            return np.zeros_like(Q), Q
        

        
    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """
        
        phi = np.zeros(self.nphys)
        GtX = self.Gauss_t[t] * X
        ind0 = np.nonzero(self.Gauss_t[t])[0]
        if ind0.size>0:
                GtX = GtX[ind0].reshape(self.Nt[t],self.Nx)
                phi += self.Gauss_xy.dot(GtX.sum(axis=0))
        phi = phi.reshape(self.shape_phys)
    
        if State is not None:
            State.params[self.name_mod_var] = phi
        else:
            return phi


    def operg_transpose(self,t,adState):
        """
            Project to reduced space
        """

        if adState.params[self.name_mod_var] is None:
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState.params[self.name_mod_var].ravel()
        Gt = self.Gauss_t[t]
        ind0 = np.nonzero(Gt)[0]
        if ind0.size>0:
            Gt = Gt[ind0].reshape(self.Nt[t],self.Nx)
            adGtX = self.Gauss_xy.T.dot(adparams)
            adGtX = np.repeat(adGtX[np.newaxis,:],self.Nt[t],axis=0)
            adX[ind0] += (Gt*adGtX).ravel()

        adState.params[self.name_mod_var] *= 0.

        return adX

class Basis_bm_jax(Basis_bm): 


    def __init__(self,config,State):
        super().__init__(config,State)
        self.operg_jit = jit(self.operg, static_argnums=[0])
    
    def _compute_component_space(self):

        Gx = [None,]*self.nf
        Nx = [None,]*self.nf

        for iff in range(self.nf):

            data = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,))
            row = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)
            col = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((np.mod(self.lon1d - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (np.mod(self.lon1d[indphys] - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff])

                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    col[ind_tmp:ind_tmp+indphys.size] = iwave
                    row[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.cos(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    col[ind_tmp:ind_tmp+indphys.size] = iwave
                    row[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.sin(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            indexes = np.column_stack((row,col))

            Gx[iff] = sparse.BCOO((jnp.array(data), jnp.array(indexes)), 
                                  shape=(self.nphys, nwaves))

        return Gx,Nx
    
    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency 
        Nt = {} # Number of wave times tw such as abs(tw-t)<tdec
        indt = {}

        for t in time:

            Gt[t] = [None,]*self.nf
            Nt[t] = [0,]*self.nf
            indt[t] = [None,]*self.nf

            for iff in range(self.nf):
                Gt[t][iff] = jnp.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],))
                ind_tmp = 0
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    if abs(dt) < self.tdec[iff]:
                        Nt[t][iff] += 1
                        fact = self.window(dt / self.tdec[iff]) 
                        fact /= self.norm_fact[iff]   
                        Gt[t][iff] = Gt[t][iff].at[ind_tmp:ind_tmp+2*self.ntheta*self.NP[iff]].set(fact)   
                    ind_tmp += 2*self.ntheta*self.NP[iff]

                indt[t][iff] = jnp.nonzero(Gt[t][iff])[0]

        self.indt = indt

        return Gt, Nt     
    

    @sparse.sparsify
    def _sparse_op(self,Gx,X):
        return Gx @ X
    
    def operg(self, t, X, State_params):
        
        """
            Project to physicial space
        """

        phi = jnp.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            
            if self.indt[t][iff].size>0:
                GtXf = GtXf[self.indt[t][iff]].reshape(self.Nt[t][iff],self.Nx[iff])
                GtXf_sum = GtXf.sum(axis=0)
                phi0 = self._sparse_op(self.Gx[iff], GtXf_sum)
                phi += phi0
        phi = phi.reshape(self.shape_phys)

        # Update State parameters 
        State_params[self.name_mod_var] = phi

        return State_params
    
class Basis_bmaux:
   
    def __init__(self,config,State):

        self.km2deg=1./110
        
        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns # factor for wavelet spacing= space
        self.facnlt = config.BASIS.facnlt
        self.npsp = config.BASIS.npsp # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp = config.BASIS.facpsp # 1.5 # factor to fix df between wavelets 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tdecmin = config.BASIS.tdecmin
        self.tdecmax = config.BASIS.tdecmax
        self.factdec = config.BASIS.factdec
        self.facQ = config.BASIS.facQ
        self.save_wave_basis = config.BASIS.save_wave_basis
        self.file_aux = config.BASIS.file_aux
        self.filec_aux = config.BASIS.filec_aux
        self.Romax = config.BASIS.Romax
        self.facRo = config.BASIS.facRo
        self.cutRo = config.BASIS.cutRo
        self.tssr = config.BASIS.tssr
        self.distortion_eq = config.BASIS.distortion_eq
        self.distortion_eq_law = config.BASIS.distortion_eq_law
        self.lat_distortion_eq = config.BASIS.lat_distortion_eq
        self.name_mod_var = config.BASIS.name_mod_var
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Mask
        if State.mask is not None:
            self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow

    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon1d.min()
        LON_MAX = self.lon1d.max()
        LAT_MIN = self.lat1d.min()
        LAT_MAX = self.lat1d.max()
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        lat_tmp = np.arange(-90,90,0.1)
        alpha = (self.distortion_eq-1)*np.sin(self.lat_distortion_eq*np.pi/180)**self.distortion_eq_law
        finterpdist = scipy.interpolate.interp1d(
            lat_tmp, 1+alpha/(np.sin(np.maximum(self.lat_distortion_eq,np.abs(lat_tmp))*np.pi/180)**self.distortion_eq_law))
        
        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./self.lmin),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ff = np.exp(logff)
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Read aux data
        aux = xr.open_dataset(self.file_aux)
        daPSDS = aux['PSDS']
        auxc = xr.open_dataset(self.filec_aux)
        daC = auxc['c1']

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        Cb1 = list() # First baroclinic phase speed
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        self.nwavemeso = 0

        for iff in range(nf):
                
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = np.mod(
                    np.arange(
                        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg*finterpdist(ENSLAT1[I]),
                        LON_MAX + DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg*finterpdist(ENSLAT1[I]),
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg*finterpdist(ENSLAT1[I])),
                    360)
                ENSLAT[iff]=np.concatenate(([ENSLAT[iff],np.repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON[iff]=np.concatenate(([ENSLON[iff],ENSLON1]))
            
            NP[iff] = len(ENSLON[iff])
            enst.append(list())
            tdec.append(list())
            Cb1.append(list())
            for P in range(NP[iff]):
                enst[-1].append(list())
                tdec[-1].append(list()) 
                Cb1[-1].append(list())

                # First baroclinic phase speed
                dlon = DX[iff]*self.km2deg/np.cos(ENSLAT[iff][P] * np.pi / 180.)*finterpdist(ENSLAT[iff][P])
                dlat = DX[iff]*self.km2deg
                elon = np.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat = np.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2 = np.meshgrid(elon,elat)
                Ctmp = daC.interp(lon=elon2.flatten(),lat=elat2.flatten()).values
                Ctmp = Ctmp[np.isnan(Ctmp)==False]
                if len(Ctmp)>0:
                    C = np.nanmean(Ctmp)
                else: 
                    C = np.nan
                if np.isnan(C): 
                    C=0.
                Cb1[-1][-1] = C

                # Decorrelation time
                fc = 2*2*np.pi/86164 * np.sin(ENSLAT[iff][P]*np.pi/180.)
                Ro = C / np.abs(fc) /1000. # Rossby radius (km)
                if Ro>self.Romax: 
                    Ro = self.Romax
                if C>0: 
                    td1 = self.factdec / (1./(self.facRo*Ro)*C/1000*86400)
                else: 
                    td1 = np.nan
                PSDS = daPSDS.interp(f=ff[iff],lat=ENSLAT[iff][P],lon=ENSLON[iff][P]).values
                if Ro>0: 
                    PSDSR = daPSDS.interp(f=1./(self.facRo*Ro),lat=ENSLAT[iff][P],lon=ENSLON[iff][P]).values
                else: 
                    PSDSR = np.nan
                if PSDS<=PSDSR: 
                    tdec[-1][-1] = td1 * (PSDS/PSDSR)**self.tssr
                else: 
                    tdec[-1][-1] = td1
                if tdec[-1][-1]>self.tdecmax: 
                    tdec[-1][-1] = self.tdecmax
                cp = 1./(2*2*np.pi/86164*np.sin(max(10,np.abs(ENSLAT[iff][P]))*np.pi/180.))/300000
                tdecp = (1./ff[iff])*1000/cp/86400/4
                if tdecp<tdec[-1][-1]: 
                    tdec[-1][-1] = tdecp

                enst[-1][-1] = np.arange(-tdec[-1][-1]*(1-1./self.facnlt),deltat+tdec[-1][-1]/self.facnlt , tdec[-1][-1]/self.facnlt)
                nwave += ntheta*2*len(enst[iff][P])
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)            
        Q = np.zeros((nwave))
        self.wavetest=[None]*nf
        # Loop on all wavelets of given pseudo-period 
        iwave=-1 
        self.iff_wavebounds = [None]*(nf+1)
        self.P_wavebounds = [None]*(nf+1)
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave+1
            self.P_wavebounds[iff] = [None]*(NP[iff]+1)
            self.wavetest[iff] = np.ones((NP[iff]), dtype=bool)
            for P in range(NP[iff]):
                self.P_wavebounds[iff][P] = iwave+1
                PSDLOC = abs(daPSDS.interp(f=ff[iff],lat=ENSLAT[iff][P],lon=ENSLON[iff][P]).values)
                C = Cb1[iff][P]
                fc = (2*2*np.pi/86164*np.sin(ENSLAT[iff][P]*np.pi/180.))
                if fc==0: 
                    Ro=self.Romax
                else:
                    Ro = C / np.abs(fc) /1000.  # Rossby radius (km)
                    if Ro>self.Romax: 
                        Ro=self.Romax
                # Tests
                if ((1./ff[iff] < self.cutRo * Ro)): 
                    self.wavetest[iff][P]=False
                if tdec[iff][P]<self.tdecmin: 
                    self.wavetest[iff][P]=False
                if np.isnan(PSDLOC): 
                    self.wavetest[iff][P]=False
                if ((np.isnan(Cb1[iff][P]))|(Cb1[iff][P]==0)): 
                    self.wavetest[iff][P]=False

                if self.wavetest[iff][P]==True:
                    for it in range(len(enst[iff][P])):
                        for itheta in range(len(theta)):
                            iwave += 1
                            Q[iwave] = (PSDLOC*ff[iff]**2 * self.facQ * np.exp(-3*(self.cutRo * Ro*ff[iff])**3))**.5
                            iwave += 1
                            Q[iwave] = (PSDLOC*ff[iff]**2 * self.facQ* np.exp(-3*(self.cutRo * Ro*ff[iff])**3))**.5
                
            print(f'lambda={1/ff[iff]:.1E}',
                  f'nlocs={P:.1E}',
                  f'tdec={np.nanmean(tdec[iff]):.1E}',
                  f'Q={np.nanmean(Q[self.iff_wavebounds[iff]:iwave+1]):.1E}')
                  
            self.P_wavebounds[iff][P+1] = iwave +1
        self.iff_wavebounds[-1] = iwave +1
            
        nwave = iwave+1
        Q = Q[:nwave]

        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)
        

        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nbasis=nwave
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff
        self.tdec=tdec
        
        # Compute basis components
        self.G, self.N = self._compute_component(time) # in space
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            
        if return_q:
            return Xb, Q
    
    def _compute_component(self,time):

        G = {}
        N = {}

        for t in time:

            G[t] = [None,]*self.nf
            N[t] = [None,]*self.nf
            
            for iff in range(self.nf):

                data = np.empty(((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff])*self.nphys,))
                indices = np.empty(((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff])*self.nphys,),dtype=int)
                sizes = np.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],),dtype=int)

                N[t][iff] = np.zeros(self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],dtype=bool)

                ind_tmp = 0
                iwave = 0

                for P in range(self.NP[iff]):

                    if self.wavetest[iff][P]:
                        # Obs selection around point P
                        indphys = np.where(
                            (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                            (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                            )[0]
                        xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                        yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                        # Spatial tapering shape of the wavelet 
                        if self.mask1d is not None:
                            indmask = self.mask1d[indphys]
                            indphys = indphys[~indmask]
                            xx = xx[~indmask]
                            yy = yy[~indmask]
        
                        facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) 

                        for it in range(len(self.enst[iff][P])):
                            dt = t - self.enst[iff][P][it]
                            if abs(dt) < self.tdec[iff][P]:
                                fact = self.window(dt / self.tdec[iff][P]) 
                                tt = np.linspace(-self.tdec[iff][P],self.tdec[iff][P])
                                # Normalization
                                I =  np.sum(mywindow(tt/self.tdec[iff][P]))*(tt[1]-tt[0])
                                fact /= I
                                for itheta in range(self.ntheta):
                                    # Wave vector components
                                    kx = self.k[iff] * np.cos(self.theta[itheta])
                                    ky = self.k[iff] * np.sin(self.theta[itheta])
                                    # Cosine component
                                    N[t][iff][iwave] = True
                                    sizes[iwave] = indphys.size
                                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * fact * np.cos(kx*(xx)+ky*(yy))
                                    ind_tmp += indphys.size
                                    iwave += 1
                                    # Sine component
                                    N[t][iff][iwave] = True
                                    sizes[iwave] = indphys.size
                                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * fact * np.sin(kx*(xx)+ky*(yy))
                                    ind_tmp += indphys.size
                                    iwave += 1
                            else:
                                iwave += 2*self.ntheta

                sizes = sizes[N[t][iff]]
                nwaves = sizes.size
                indices = indices[:ind_tmp]
                data = data[:ind_tmp]

                indptr = np.zeros((nwaves+1),dtype=int)
                indptr[1:] = np.cumsum(sizes)

                G[t][iff] = csc_matrix((data, indices, indptr), shape=(self.nphys, nwaves))

        return G, N
 

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):

            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][self.N[t][iff]]
            phi += self.G[t][iff] @ Xf

        phi = phi.reshape(self.shape_phys)

        # Update State
        if State is not None:
            State.params[self.name_mod_var] = phi
        else:
            return phi

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState.params[self.name_mod_var] is None:
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState.params[self.name_mod_var].ravel()
        for iff in range(self.nf):
            adGtXf = self.G[t][iff].T.dot(adparams)
            adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][self.N[t][iff]] += adGtXf.ravel()
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX
    

class BASIS_ls:
   
    def __init__(self,config,State):

        self.km2deg=1./110
    
        self.name_mod_var = config.BASIS.name_mod_var
        self.facnls = config.BASIS.facnls
        self.facnlt = config.BASIS.facnlt
        self.lambda_lw = config.BASIS.lambda_lw
        self.tdec_lw = config.BASIS.tdec_lw
        self.fcor = config.BASIS.fcor
        self.std_lw = config.BASIS.std_lw
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()
        
        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path
        self.indx = {}
        self.facG = {}

    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): self.LON_MAX = self.LON_MAX+360.

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        
        DX = self.lambda_lw # wavelet extension
        DXG = DX / self.facnls # distance (km) between the wavelets grid in space
        
        ENSLON = []
        ENSLAT = []

        ENSLAT1 = np.arange(LAT_MIN-(DX-DXG)*self.km2deg,LAT_MAX+DX*self.km2deg,DXG*self.km2deg)
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(np.arange(LON_MIN -(DX-DXG)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX+DX/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    DXG/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg) , 360)
            ENSLAT = np.concatenate(([ENSLAT,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON = np.concatenate(([ENSLON,ENSLON1]))
        
        NP = len(ENSLON)

        enst = [None]*NP
        tdec = [None]*NP
        nwave=0
        for P in range(NP):
            tdec[P] = self.tdec_lw
            enst[P] = np.arange(-tdec[P]*(1-1./self.facnlt) , deltat+tdec[P]/self.facnlt , tdec[P]/self.facnlt)
            nt = len(enst[P])
            nwave += nt
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)            
        Q = np.zeros((nwave))
        iwave = -1
        self.P_wavebounds = [None]*(NP+1)
        varHlw = self.std_lw**2 * self.fcor
        for P in range(NP):
            self.P_wavebounds[P] = iwave+1
            for it in range(len(enst[P])):
                iwave += 1
                Q[iwave] = (varHlw/(self.facnls*self.facnlt))**.5
        self.P_wavebounds[P+1] = iwave +1

        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)
        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nbasis=nwave
        self.tdec=tdec
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        if return_q:
            return Xb, Q
    
    def operg(self, t, X, transpose=False,State=None):
        
        """
            Project to physicial space
        """
        
        # Initialize projected vector
        if transpose:
            X = X.flatten()
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.lon1d.size,))
        
        # Compute projection
        iwave = 0
        for P in range(self.NP):
                
            # Obs selection around point P
            iobs = np.where(
                (np.abs((np.mod(self.lon1d - self.ENSLON[P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[P] * np.pi / 180.)) <= self.DX) &
                (np.abs((self.lat1d - self.ENSLAT[P]) / self.km2deg) <= self.DX)
                )[0]
            xx = (np.mod(self.lon1d[iobs] - self.ENSLON[P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[P] * np.pi / 180.) 
            yy = (self.lat1d[iobs] - self.ENSLAT[P]) / self.km2deg

            facs = mywindow(xx / self.DX) * mywindow(yy / self.DX)

            enstloc = self.enst[P]
            for it in range(len(enstloc)):
                dt = t - enstloc[it]
                try:
                    if iobs.shape[0] > 0 and abs(dt) < self.tdec[P]:
                        if t==0:
                            fact = mywindow(dt / self.tdec[P])
                        else:
                            fact = mywindow_flux(dt / self.tdec[P])
                            fact /= self.tdec[P]
                        
                        if transpose:
                            phi[iwave] = np.sum(X[iobs] * (fact * facs)**2)
                        else:
                            phi[iobs] += X[iwave] * (fact * facs)**2
                    iwave += 1
                except:
                    print(f'Warning: an error occured at t={t},  P={P}, enstloc={enstloc[it]}')

        # Reshaping
        if not transpose:
            phi = phi.reshape(self.shape_phys)
        
        # Update State
        if State is not None:
            if t==0:
                State.setvar(phi,self.name_mod_var)
                State.params[self.name_mod_var] = np.zeros(self.shape_phys)
            else:
                State.params[self.name_mod_var] = phi
        else:
            return phi
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        if t==0:
            adX = self.operg(t, adState.getvar(self.name_mod_var), transpose=True)
        else:
            if adState.params is None:
                adState.params[self.name_mod_var] = np.zeros((self.shape_phys))
            adX = self.operg(t, adState.params[self.name_mod_var], transpose=True)
        
        adState.params[self.name_mod_var] = np.zeros((self.shape_phys))
        
        return adX

### FLO VERSION OF BASIS ITG ### 

class Basis_it:
   
    def __init__(self,config, State):
        self.km2deg =1./110
    
        self.facns = config.BASIS.facgauss
        self.facnlt = config.BASIS.facgauss
        self.D_He = config.BASIS.D_He
        self.T_He = config.BASIS.T_He
        self.D_bc = config.BASIS.D_bc
        self.T_bc = config.BASIS.T_bc
        
        self.sigma_B_He = config.BASIS.sigma_B_He
        self.sigma_B_bc = config.BASIS.sigma_B_bc
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background
        
        if config.BASIS.Ntheta>0:
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°
            
        self.Nwaves = config.BASIS.Nwaves

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()
        self.lonS = State.lon[0,:]
        self.lonN = State.lon[-1,:]
        self.latE = State.lat[:,0]
        self.latW = State.lat[:,-1]
    
    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time
        
        ##########################
        # He 
        ##########################
        # coordinates in space
        ENSLAT1 = np.arange(
            LAT_MIN - self.D_He*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_He/self.facns*self.km2deg, self.D_He/self.facns*self.km2deg)
        ENSLAT_He = []
        ENSLON_He = []
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(
                np.arange(
                    LON_MIN - self.D_He*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_He/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    self.D_He/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                360)
            ENSLAT_He = np.concatenate(([ENSLAT_He,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON_He = np.concatenate(([ENSLON_He,ENSLON1]))
        self.ENSLAT_He = ENSLAT_He
        self.ENSLON_He = ENSLON_He
        
        # coordinates in time
        ENST_He = np.arange(-self.T_He*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_He/self.facnlt , self.T_He/self.facnlt)
        
        
        # Gaussian functions in space
        He_xy_gauss = np.zeros((ENSLAT_He.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_He,ENSLON_He)):
            iobs = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_He) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.D_He)
                    )[0]
            xx = (np.mod(self.lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[iobs] - lat0) / self.km2deg
            
            He_xy_gauss[i,iobs] = mywindow(xx / self.D_He) * mywindow(yy / self.D_He)

        He_xy_gauss = He_xy_gauss.reshape((ENSLAT_He.size,self.ny,self.nx))
        
        # Gaussian functions in time
        He_t_gauss = np.zeros((ENST_He.size,time.size))
        for i,time0 in enumerate(ENST_He):
            iobs = np.where(abs(time-time0) < self.T_He)
            He_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_He)
        
        self.He_xy_gauss = He_xy_gauss
        self.He_t_gauss = He_t_gauss
        self.nHe = ENST_He.size * ENSLAT_He.size
        self.sliceHe = slice(0,self.nHe)
        self.shapeHe = [ENST_He.size,ENSLAT_He.size]
        print('nHe:',self.nHe)
        
        ##########################
        # bc 
        ##########################
        ## in space
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)
        
        # South
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # North
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # East
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # West 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        
        self.bc_S_gauss = bc_S_gauss
        self.bc_N_gauss = bc_N_gauss
        self.bc_E_gauss = bc_E_gauss
        self.bc_W_gauss = bc_W_gauss
        
        ## in time
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        bc_t_gauss = np.zeros((ENST_bc.size,time.size))
        for i,time0 in enumerate(ENST_bc):
            iobs = np.where(abs(time-time0) < self.T_bc)
            bc_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_bc)
        self.bc_t_gauss = bc_t_gauss
        
        self.nbcS = self.Nwaves * 2 * self.Ntheta * ENST_bc.size * bc_S_gauss.shape[0]
        self.nbcN = self.Nwaves * 2 * self.Ntheta * ENST_bc.size * bc_N_gauss.shape[0]
        self.nbcE = self.Nwaves * 2 * self.Ntheta * ENST_bc.size * bc_E_gauss.shape[0]
        self.nbcW = self.Nwaves * 2 * self.Ntheta * ENST_bc.size * bc_W_gauss.shape[0]
        print("nbcS : ",self.nbcS)
        print("nbcN : ",self.nbcN)
        print("nbcE : ",self.nbcE)
        print("nbcW : ",self.nbcW)
        self.nbc = self.nbcS + self.nbcN + self.nbcE + self.nbcW
        print('nbc:',self.nbc)
        
        
        self.shapehbcS = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_S_gauss.shape[0]]
        self.shapehbcN = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_N_gauss.shape[0]]
        self.shapehbcE = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_E_gauss.shape[0]]
        self.shapehbcW = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_W_gauss.shape[0]]
        
        self.slicebcS = slice(self.nHe,
                              self.nHe + self.nbcS)
        self.slicebcN = slice(self.nHe+ self.nbcS,
                              self.nHe + self.nbcS + self.nbcN)
        self.slicebcE = slice(self.nHe+ self.nbcS + self.nbcN,
                              self.nHe + self.nbcS + self.nbcN + self.nbcE)
        self.slicebcW = slice(self.nHe+ self.nbcS + self.nbcN + self.nbcE,
                              self.nHe + self.nbcS + self.nbcN + self.nbcE + self.nbcW)
        self.slicebc = slice(self.nHe,
                             self.nHe + self.nbc)
        
        self.nbasis = self.nHe + self.nbc
        
        # OUTPUT SHAPES (physical space)
        self.shapeHe_phys = (self.ny,self.nx)
        self.shapehbcx_phys = [self.Nwaves, # tide frequencies
                          2, # North/South
                          2, # cos/sin
                          self.Ntheta, # Angles
                          self.nx # NX
                          ]
        self.shapehbcy_phys = [self.Nwaves, # tide frequencies
                          2, # North/South
                          2, # cos/sin
                          self.Ntheta, # Angles
                          self.ny # NY
                          ]
        self.nphys = np.prod(self.shapeHe_phys) + np.prod(self.shapehbcx_phys) + np.prod(self.shapehbcy_phys)
        self.sliceHe_phys = slice(0,np.prod(self.shapeHe_phys))
        self.slicehbcx_phys = slice(np.prod(self.shapeHe_phys),
                               np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys))
        self.slicehbcy_phys = slice(np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys),
                               np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys)+np.prod(self.shapehbcy_phys))
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        # Fill Q matrix
        if return_q:
            if None not in [self.sigma_B_He, self.sigma_B_bc]:
                Q = np.zeros((self.nbasis,)) 
                # variance on He
                Q[self.sliceHe] = self.sigma_B_He 
                if hasattr(self.sigma_B_bc,'__len__'):
                    if len(self.sigma_B_bc)==self.Nwaves:
                        # Different background values for each frequency
                        nw = self.nbc//self.Nwaves
                        for iw in range(self.Nwaves):
                                slicew = slice(iw*nw,(iw+1)*nw)
                                Q[self.slicebc][slicew] = self.sigma_B_bc[iw]
                    else:
                        # Not the right number of frequency prescribed in the config file 
                        # --> we use only the first one
                        Q[self.slicebc] = self.sigma_B_bc[0]
                else:
                    Q[self.slicebc] = self.sigma_B_bc
            else:
                Q = None
            
            # Background
            if self.path_background is not None and os.path.exists(self.path_background):
                with xr.open_dataset(self.path_background) as ds:
                    print(f'Load background from file: {self.path_background}')
                    Xb = ds[self.var_background].values
            else:
                Xb = np.zeros_like(Q)

            return Xb, Q
        
        
    def operg(self,t,X,State=None):
        """
            Project to physicial space
        """
        
        # Get variables in reduced space
        X_He = X[self.sliceHe].reshape(self.shapeHe)
        X_bcS = X[self.slicebcS].reshape(self.shapehbcS)
        X_bcN = X[self.slicebcN].reshape(self.shapehbcN)
        X_bcE = X[self.slicebcE].reshape(self.shapehbcE)
        X_bcW = X[self.slicebcW].reshape(self.shapehbcW)
        
        # Project to physical space
        indt = np.argmin(np.abs(self.time-t))        
        He = np.tensordot(
            np.tensordot(X_He,self.He_xy_gauss,(1,0)),
                                self.He_t_gauss[:,indt],(0,0))
    
        hbcx = np.zeros(self.shapehbcx_phys)
        hbcy = np.zeros(self.shapehbcy_phys)
        
        hbcx[:,0] = np.tensordot(
            np.tensordot(X_bcS,self.bc_S_gauss,(-1,0)),
                                 self.bc_t_gauss[:,indt],(-2,0))
        hbcx[:,1] = np.tensordot(
            np.tensordot(X_bcN,self.bc_N_gauss,(-1,0)),
                                 self.bc_t_gauss[:,indt],(-2,0))
        hbcy[:,0] = np.tensordot(
            np.tensordot(X_bcE,self.bc_E_gauss,(-1,0)),
                                 self.bc_t_gauss[:,indt],(-2,0))
        hbcy[:,1] = np.tensordot(
            np.tensordot(X_bcW,self.bc_W_gauss,(-1,0)),
                                 self.bc_t_gauss[:,indt],(-2,0))
        
        if State is not None:
            State.params['He'] = +He
            State.params['hbcx'] = +hbcx
            State.params['hbcy'] = +hbcy
        else:
            phi = np.concatenate((He.flatten(),hbcx.flatten(),hbcy.flatten()))
            return phi


    def operg_transpose(self,t,phi=None,adState=None):
        """
            Project to reduced space
        """
        
        # Get variable in physical space
        if phi is not None:
            He = phi[self.sliceHe_phys].reshape(self.shapeHe_phys)
            hbcx = phi[self.slicehbcx_phys].reshape(self.shapehbcx_phys)
            hbcy = phi[self.slicehbcy_phys].reshape(self.shapehbcy_phys)
        elif adState is not None:
            He = +adState.params['He'].reshape(self.shapeHe_phys)
            hbcx = +adState.params['hbcx'].reshape(self.shapehbcx_phys)
            hbcy = +adState.params['hbcy'].reshape(self.shapehbcy_phys)
            adState.params['He'] *= 0
            adState.params['hbcx'] *= 0
            adState.params['hbcy'] *= 0

        else:
            sys.exit('Provide either phi or adState')
        
        # Project to reduced space
        indt = np.argmin(np.abs(self.time-t))   
        
        adX_He = np.tensordot(
            He[:,:,np.newaxis]*self.He_t_gauss[:,indt],
                                   self.He_xy_gauss[:,:,:],([0,1],[1,2])) 
        adX_bcS = np.tensordot(
               hbcx[:,0,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                              self.bc_S_gauss,(-2,-1))
        adX_bcN = np.tensordot(
               hbcx[:,1,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                              self.bc_N_gauss,(-2,-1))
        adX_bcE = np.tensordot(
               hbcy[:,0,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                              self.bc_E_gauss,(-2,-1))
        adX_bcW = np.tensordot(
               hbcy[:,1,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                              self.bc_W_gauss,(-2,-1))
        
        adX = np.concatenate((adX_He.flatten(),
                              adX_bcS.flatten(),
                              adX_bcN.flatten(),
                              adX_bcE.flatten(),
                              adX_bcW.flatten()))
            
        return adX
    
    def test_operg(self, t, State):
        
        np.random.seed(40)
        State0 = State.random()
        # Setting a fixed version of State # 
        #State0 = State.copy()
        #np.random.seed(30)
        #State0.params['He'] = np.random.random(State0.params['He'].shape)
        #np.random.seed(31)
        #State0.params['hbcx'] = np.random.random(State0.params['hbcx'].shape)
        #np.random.seed(32)
        #State0.params['hbcy'] = np.random.random(State0.params['hbcy'].shape)

        #np.random.seed(33)
        phi0 = np.random.random((self.nbasis,))

        adState1 = State.random()
        # Setting a fixed version of adState1 # 
        #adState1 = State.copy()
        #np.random.seed(34)
        #adState1.params['He'] = np.random.random(State0.params['He'].shape)
        #np.random.seed(35)
        #adState1.params['hbcx'] = np.random.random(State0.params['hbcx'].shape)
        #np.random.seed(36)
        #adState1.params['hbcy'] = np.random.random(State0.params['hbcy'].shape)
        
        psi1 = adState1.getparams(vect=True)

        phi1 = self.operg_transpose(t,adState=adState1)
        self.operg(t,phi0,State=State0)
        psi0 = State0.getparams(vect=True)
        
        ps1 = np.inner(psi0,psi1)
        ps2 = np.inner(phi0,phi1)
            
        print(f'test G[{t}]:', ps1/ps2)


### VERSION OF BASIS ITG DEVELOPPED BY VALENTIN ### 
'''
class Basis_it:
   
    def __init__(self,config, State):
        self.km2deg =1./110
    
        self.facns = config.BASIS.facgauss
        self.facnlt = config.BASIS.facgauss
        self.D_He = config.BASIS.D_He
        self.T_He = config.BASIS.T_He
        self.D_bc = config.BASIS.D_bc
        self.T_bc = config.BASIS.T_bc
        # for itg (internal tide generation)
        self.D_itg = config.BASIS.D_itg
        self.T_itg = config.BASIS.T_itg
        self.reduced_basis_itg = config.BASIS.reduced_basis_itg
        
        self.sigma_B_He = config.BASIS.sigma_B_He
        self.sigma_B_bc = config.BASIS.sigma_B_bc
        self.sigma_B_itg = config.BASIS.sigma_B_itg
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background
        
        if config.BASIS.Ntheta>0:
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°
        
        self.Nwaves = config.BASIS.Nwaves

        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()
        self.lonS = State.lon[0,:]
        self.lonN = State.lon[-1,:]
        self.latE = State.lat[:,0]
        self.latW = State.lat[:,-1]

        self.name_params = config.MOD.name_params 
    
    def set_basis(self,time,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time

        self.bc_gauss = {} # gaussian basis for boundary conditions 
        self.shape_params = {} # dictionary with the shapes in the reduced space of each of the parameters 
        self.shape_params_phys = {} # dictionary with the shapes in the physical space of each of the parameters 
 
        for name in self.name_params : 
            if name == "He": 
                self.shape_params["He"], self.shape_params_phys["He"] = self.set_He(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX) 

            if name == "hbcx":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_hbcx(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX)

            if name == "hbcy": 
                self.shape_params["hbcE"], self.shape_params["hbcW"], self.shape_params_phys["hbcE"], self.shape_params_phys["hbcW"] = self.set_hbcy(time, LAT_MIN, LAT_MAX, TIME_MIN, TIME_MAX)

            if name == "itg": 
                #self.shape_params["itg"], self.shape_params_phys["itg"] = self.set_itg(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX)
                ### UNCOMMENT FOR ITG INDEPENDANT ON TIME ###
                self.shape_params["itg"], self.shape_params_phys["itg"] = self.set_itg(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        
        self.n_params = dict(zip(self.shape_params.keys(), map(np.prod, self.shape_params.values()))) # dictionary with the number of each of the parameters in reduced space 
        self.n_params_phys = dict(zip(self.shape_params_phys.keys(), map(np.prod, self.shape_params_phys.values()))) # dictionary with the number of each of the parameters in physical space
        
        self.nbasis = sum(self.n_params.values()) # total number of parameters in the reduced space
        self.nphys = sum(self.n_params_phys.values()) # total number of parameters in the physical space
        self.nphystot = 0 # total number of parameters in the physical space (for printing reduced order)
        for param in self.n_params_phys.keys():
            if param == "itg":
                self.nphystot += self.n_params_phys[param] 
            else : 
                self.nphystot += self.n_params_phys[param]*time.size

        interval = 0 ; interval_phys = 0 
        self.slice_params = {} # dictionary with the slices of each of the parameters in the reduced space
        self.slice_params_phys = {} # dictionary with the slices of each of the parameters in the physical space
        for name in self.shape_params.keys() :
            self.slice_params[name]=slice(interval,interval+self.n_params[name])
            self.slice_params_phys[name]=slice(interval_phys,interval_phys+self.n_params_phys[name])
            interval += self.n_params[name]; interval_phys += self.n_params_phys[name]

        # PRINTING REDUCED ORDER : #     
        print(f'reduced order: {self.nphystot} --> {self.nbasis}\nreduced factor: {int(self.nphystot/self.nbasis)}')
    
        if return_q :
            if None not in [self.sigma_B_He, self.sigma_B_bc, self.sigma_B_itg]:
                Q = np.zeros((self.nbasis,))
                for name in self.slice_params.keys() :
                    if name == "He" : 
                        Q[self.slice_params[name]]=self.sigma_B_He
                    if name in ["hbcS","hbcN","hbcW","hbcE"] : 
                        if hasattr(self.sigma_B_bc,'__len__'):
                            if len(self.sigma_B_bc)==self.Nwaves:
                                # Different background values for each frequency
                                nw = self.nbc//self.Nwaves
                                for iw in range(self.Nwaves):
                                    slicew = slice(iw*nw,(iw+1)*nw)
                                    Q[self.slice_params[name]][slicew]=self.sigma_B_bc[iw]
                            else:
                                # Not the right number of frequency prescribed in the config file 
                                # --> we use only the first one
                                Q[self.slice_params[name]]=self.sigma_B_bc[0]
                        else:
                            Q[self.slice_params[name]]=self.sigma_B_bc
                    if name == "itg" : 
                        Q[self.slice_params[name]]=self.sigma_B_itg
            else:
                Q = None

            # Background
            if self.path_background is not None and os.path.exists(self.path_background):
                with xr.open_dataset(self.path_background) as ds:
                    print(f'Load background from file: {self.path_background}')
                    Xb = ds[self.var_background].values
            else:
                Xb = np.zeros_like(Q)

            return Xb, Q

    def set_He(self,time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX):
                    
        ##########################
        # He 
        ##########################
        # coordinates in space
        ENSLAT1 = np.arange(
            LAT_MIN - self.D_He*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_He/self.facns*self.km2deg, self.D_He/self.facns*self.km2deg)
        ENSLAT_He = []
        ENSLON_He = []
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(
                np.arange(
                    LON_MIN - self.D_He*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_He/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    self.D_He/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                360)
            ENSLAT_He = np.concatenate(([ENSLAT_He,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON_He = np.concatenate(([ENSLON_He,ENSLON1]))
        self.ENSLAT_He = ENSLAT_He
        self.ENSLON_He = ENSLON_He
        
        # coordinates in time
        ENST_He = np.arange(-self.T_He*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_He/self.facnlt , self.T_He/self.facnlt)
        
        
        # Gaussian functions in space
        He_xy_gauss = np.zeros((ENSLAT_He.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_He,ENSLON_He)):
            iobs = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_He) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.D_He)
                    )[0]
            xx = (np.mod(self.lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[iobs] - lat0) / self.km2deg
            
            He_xy_gauss[i,iobs] = mywindow(xx / self.D_He) * mywindow(yy / self.D_He)

        He_xy_gauss = He_xy_gauss.reshape((ENSLAT_He.size,self.ny,self.nx))
        
        # Gaussian functions in time
        He_t_gauss = np.zeros((ENST_He.size,time.size))
        for i,time0 in enumerate(ENST_He):
            iobs = np.where(abs(time-time0) < self.T_He)
            He_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_He)
        
        self.He_xy_gauss = He_xy_gauss
        self.He_t_gauss = He_t_gauss

        shapeHe = [ENST_He.size,ENSLAT_He.size]
        shapeHe_phys = (self.ny,self.nx)

        print('nHe:',np.prod(shapeHe))

        return shapeHe, shapeHe_phys
        
    
    def set_hbcx(self,time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX):
        
        # South
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # North
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 

        self.bc_gauss["hbcS"] = bc_S_gauss
        self.bc_gauss["hbcN"] = bc_N_gauss

        ## in time
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        bc_t_gauss = np.zeros((ENST_bc.size,time.size))
        for i,time0 in enumerate(ENST_bc):
            iobs = np.where(abs(time-time0) < self.T_bc)
            bc_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_bc)
        
        self.bc_t_gauss = bc_t_gauss

        shapehbcS = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_S_gauss.shape[0]]
        shapehbcN = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_N_gauss.shape[0]]

        shapehbcS_phys = shapehbcN_phys = [self.Nwaves, 2, self.Ntheta, self.nx]
        
        print('nbcx:',np.prod(shapehbcS)+np.prod(shapehbcN))

        return shapehbcS, shapehbcN, shapehbcS_phys, shapehbcN_phys


    def set_hbcy(self,time, LAT_MIN, LAT_MAX, TIME_MIN, TIME_MAX): 
    
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)

        # East
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # West 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        self.bc_gauss["hbcE"] = bc_E_gauss
        self.bc_gauss["hbcW"] = bc_W_gauss
        
        ## in time
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        bc_t_gauss = np.zeros((ENST_bc.size,time.size))
        for i,time0 in enumerate(ENST_bc):
            iobs = np.where(abs(time-time0) < self.T_bc)
            bc_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_bc)
        
        self.bc_t_gauss = bc_t_gauss
        
        shapehbcE = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_E_gauss.shape[0]]
        shapehbcW = [self.Nwaves, 2, self.Ntheta, ENST_bc.size, bc_W_gauss.shape[0]]

        shapehbcE_phys = shapehbcW_phys = [self.Nwaves, 2, self.Ntheta, self.ny]

        print('nbcy:',np.prod(shapehbcE)+np.prod(shapehbcW))

        return shapehbcE, shapehbcW, shapehbcE_phys, shapehbcW_phys
    
    """
    ### VERSION OF ITG DEPENDING ON TIME ### 
    def set_itg(self,time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX):
                    
        # coordinates in space
        ENSLAT1 = np.arange(
            LAT_MIN - self.D_itg*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_itg/self.facns*self.km2deg, self.D_itg/self.facns*self.km2deg)
        ENSLAT_itg = []
        ENSLON_itg = []
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(
                np.arange(
                    LON_MIN - self.D_itg*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_itg/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    self.D_itg/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                360)
            ENSLAT_itg = np.concatenate(([ENSLAT_itg,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON_itg = np.concatenate(([ENSLON_itg,ENSLON1]))
        self.ENSLAT_itg = ENSLAT_itg
        self.ENSLON_itg = ENSLON_itg
        
        # coordinates in time
        ENST_itg = np.arange(-self.T_itg*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_itg/self.facnlt , self.T_itg/self.facnlt)
        
        
        # Gaussian functions in space
        itg_xy_gauss = np.zeros((ENSLAT_itg.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_itg,ENSLON_itg)):
            iobs = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_itg) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.D_itg)
                    )[0]
            xx = (np.mod(self.lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[iobs] - lat0) / self.km2deg
            
            itg_xy_gauss[i,iobs] = mywindow(xx / self.D_itg) * mywindow(yy / self.D_itg)

        itg_xy_gauss = itg_xy_gauss.reshape((ENSLAT_itg.size,self.ny,self.nx))
        
        # Gaussian functions in time
        itg_t_gauss = np.zeros((ENST_itg.size,time.size))
        for i,time0 in enumerate(ENST_itg):
            iobs = np.where(abs(time-time0) < self.T_itg)
            itg_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_itg)
        
        self.itg_xy_gauss = itg_xy_gauss
        self.itg_t_gauss = itg_t_gauss

        shapeitg = [2,ENST_itg.size,ENSLAT_itg.size]
        shapeitg_phys = (2,self.ny,self.nx)

        print('nitg:',np.prod(shapeitg))

        return shapeitg, shapeitg_phys
    """

    
    ### VERSION OF ITG INDEPENDANT ON TIME ### 
    def set_itg(self,LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):

        if not self.reduced_basis_itg :
            ### if no reduced basis for the itg parameters ###
            shapeitg = (2,self.nx,self.ny)
            shapeitg_phys = (2,self.nx,self.ny)

            print('nitg:',np.prod(shapeitg))

            return shapeitg, shapeitg_phys
        
        ENSLAT1 = np.arange(
            LAT_MIN - self.D_itg*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_itg/self.facns*self.km2deg, self.D_itg/self.facns*self.km2deg)
        ENSLAT_itg = []
        ENSLON_itg = []
        for I in range(len(ENSLAT1)):
            ENSLON1 = np.mod(
                np.arange(
                    LON_MIN - self.D_itg*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_itg/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    self.D_itg/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                360)
            ENSLAT_itg = np.concatenate(([ENSLAT_itg,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON_itg = np.concatenate(([ENSLON_itg,ENSLON1]))
        self.ENSLAT_itg = ENSLAT_itg
        self.ENSLON_itg = ENSLON_itg

        # Gaussian functions in space
        itg_xy_gauss = np.zeros((ENSLAT_itg.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_itg,ENSLON_itg)):
            iobs = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_itg) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.D_itg)
                    )[0]
            xx = (np.mod(self.lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[iobs] - lat0) / self.km2deg
            
            itg_xy_gauss[i,iobs] = mywindow(xx / self.D_itg) * mywindow(yy / self.D_itg)

        itg_xy_gauss = itg_xy_gauss.reshape((ENSLAT_itg.size,self.ny,self.nx))
        #itg_xy_gauss = itg_xy_gauss.reshape((ENSLAT_itg.size,self.ny,self.nx))
        #itg_xy_gauss = np.repeat(np.expand_dims(itg_xy_gauss,axis=1),axis=1,repeats=2)

        self.itg_xy_gauss = itg_xy_gauss

        shapeitg = [2,ENSLAT_itg.size]
        shapeitg_phys = (2,self.nx,self.ny)
        
        print('nitg:',np.prod(shapeitg))

        return shapeitg, shapeitg_phys
    

    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """

        indt = np.argmin(np.abs(self.time-t))   

        phi = np.zeros((self.nphys,))

        for name in self.slice_params_phys.keys():
            if name == "He":
                phi[self.slice_params_phys[name]] = np.tensordot(
                                                        np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.He_xy_gauss,(1,0)),
                                                        self.He_t_gauss[:,indt],(0,0)).flatten()
            if name in ["hbcS","hbcN","hbcW","hbcE"]:
                phi[self.slice_params_phys[name]] = np.tensordot(
                                                        np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.bc_gauss[name],(-1,0)),
                                                        self.bc_t_gauss[:,indt],(-2,0)).flatten()
            #if name == "itg":
            #    phi[self.slice_params_phys[name]] = np.tensordot(
            #                                            np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.itg_xy_gauss,(-1,0)),
            #                                            self.itg_t_gauss[:,indt],(1,0)).flatten()
            ### UNCOMMENT FOR ITG INDEPENDANT ON TIME ###
            if name == "itg": 
                if not self.reduced_basis_itg : 
                    phi[self.slice_params_phys[name]] = X[self.slice_params[name]].reshape(self.shape_params[name]).flatten()
                else : 
                    phi[self.slice_params_phys[name]] = np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.itg_xy_gauss,(-1,0)).flatten()

        if State is not None:
            for name in self.name_params : 
                if name == "hbcx" : 
                    State.params['hbcx'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                           np.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                elif name == "hbcy" : 
                    State.params['hbcy'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1),
                                                           np.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1)),axis=1)
                else : 
                    State.params[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])
        else: 
            return phi

    def operg_transpose(self,t,phi=None,adState=None):

        """
            Project to reduced space
        """

        param = {} # dictionary containing the values of alle the params 
        # Get variable in physical space
        if phi is not None:
            for name in self.slice_params_phys.keys():
                param[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])
        elif adState is not None:
            for name in self.name_params:
                if name == "hbcx" : 
                    param["hbcS"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcS"])
                    param["hbcN"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcN"])
                elif name == "hbcy" : 
                    param["hbcE"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcE"])
                    param["hbcW"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcW"])
                else : 
                    param[name] = adState.params[name].reshape(self.shape_params_phys[name])
        else: 
            sys.exit('Provide either phi or adState')

        # Project to reduced space
        indt = np.argmin(np.abs(self.time-t)) 

        adX = np.zeros((self.nbasis,))  

        for name in self.slice_params.keys():
            if name == "He":
                adX[self.slice_params[name]] = np.tensordot(param[name][:,:,np.newaxis]*self.He_t_gauss[:,indt],
                                                            self.He_xy_gauss[:,:,:],([0,1],[1,2])).flatten()
            if name in ["hbcS","hbcN","hbcW","hbcE"]:
                adX[self.slice_params[name]] = np.tensordot(param[name][:,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                                            self.bc_gauss[name],(-2,-1)).flatten()
            #if name == "itg":
            #    adX[self.slice_params[name]] = np.tensordot(param[name][:,:,:,np.newaxis]*self.itg_t_gauss[:,indt],
            #                                                self.itg_xy_gauss[:,:,:],([1,2],[-2,-1])).flatten()
            ### UNCOMMENT FOR ITG INDEPENDANT ON TIME ###
            if name == "itg":
                if not self.reduced_basis_itg :
                    adX[self.slice_params[name]] = param[name].flatten()
                else : 
                    adX[self.slice_params[name]] = np.tensordot(param[name][:,:,:,np.newaxis],
                                                            self.itg_xy_gauss[:,:,:],([1,2],[-2,-1])).flatten()

        return adX
        
    def test_operg(self, t, State):

        np.random.seed(40)
        State0 = State.random()
        # Setting a fixed version of State # 
        #State0 = State.copy()
        #np.random.seed(40)
        #State0.params['He'] = np.random.random(State0.params['He'].shape)
        #np.random.seed(41)
        #State0.params['hbcx'] = np.random.random(State0.params['hbcx'].shape)
        #np.random.seed(42)
        #State0.params['hbcy'] = np.random.random(State0.params['hbcy'].shape)

        #np.random.seed(43)
        phi0 = np.random.random((self.nbasis,))

        adState1 = State.random()
        # Setting a fixed version of adState1 # 
        #adState1 = State.copy()
        #np.random.seed(44)
        #adState1.params['He'] = np.random.random(State0.params['He'].shape)
        #np.random.seed(45)
        #adState1.params['hbcx'] = np.random.random(State0.params['hbcx'].shape)
        #np.random.seed(46)
        #adState1.params['hbcy'] = np.random.random(State0.params['hbcy'].shape)
        
        psi1 = adState1.getparams(vect=True)

        phi1 = self.operg_transpose(t,adState=adState1)
        self.operg(t,phi0,State=State0)
        psi0 = State0.getparams(vect=True)
        
        ps1 = np.inner(psi0,psi1)
        ps2 = np.inner(phi0,phi1)
            
        print(f'test G[{t}]:', ps1/ps2)
'''

###############################################################################
#                              Multi-Basis                                    #
###############################################################################      

class Basis_multi:

    def __init__(self,config,State,verbose=True):

        self.Basis = []
        _config = config.copy()

        for _BASIS in config.BASIS:
            _config.BASIS = config.BASIS[_BASIS]
            self.Basis.append(Basis(_config,State,verbose=verbose))

    def set_basis(self,time,return_q=False):

        self.nbasis = 0
        self.slice_basis = []

        if return_q:
            Xb = np.array([])
            Q = np.array([])

        for B in self.Basis:
            _Xb,_Q = B.set_basis(time,return_q=return_q)
            self.slice_basis.append(slice(self.nbasis,self.nbasis+B.nbasis))
            self.nbasis += B.nbasis
            
            if return_q:
                Xb = np.concatenate((Xb,_Xb))
                Q = np.concatenate((Q,_Q))
        
        if return_q:
            return Xb,Q

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        phi = np.array([])

        for i,B in enumerate(self.Basis):
            _X = X[self.slice_basis[i]]
            phi = np.append(phi, B.operg(t, _X, State=State))
        
        if State is None:
            return phi


    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        adX = np.array([])
        for B in self.Basis:
            adX = np.concatenate((adX, B.operg_transpose(t, adState=adState)))

        return adX


def mywindow(x): # x must be between -1 and 1
     y  = np.cos(x*0.5*np.pi)**2
     return y
  
def mywindow_flux(x): # x must be between -1 and 1
     y = -np.pi*np.sin(x*0.5*np.pi)*np.cos(x*0.5*np.pi)
     return y

def integrand(x,f):
    y  = quad(f, -1, x)[0]
    return y

def test_operg(Basis,t=0):
        
    psi = np.random.random((Basis.nbasis,))
    phi = np.random.random((Basis.shape_phys))
    
    ps1 = np.inner(psi,Basis.operg(phi,t,transpose=True))
    ps2 = np.inner(Basis.operg(psi,t).flatten(),phi.flatten())
        
    print(f'test G[{t}]:', ps1/ps2)

