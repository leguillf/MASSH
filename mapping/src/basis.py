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
from scipy.integrate import quad
import jax.numpy as jnp 
from jax.experimental import sparse as sparse
from jax import jit, lax
import jax

jax.config.update("jax_enable_x64", True)

from .tools import gaspari_cohn


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

        elif config.BASIS.super=='BASIS_WAVELET3D':
            return Basis_wavelet3d(config,State)
    
        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_bmaux(config,State)
        
        elif config.BASIS.super=='BASIS_BMaux_JAX':
            return Basis_bmaux_jax(config,State)
        
        elif config.BASIS.super=='BASIS_IT':
            return Basis_it(config, State)

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
        self.lon_min = State.lon.min()
        self.lon_max = State.lon.max()
        self.lat_min = State.lat.min()
        self.lat_max = State.lat.max()
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

    def set_basis(self,time,return_q=False,**kwargs):
        
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
        ff = ff[1/ff<=self.lmax]
        dff = ff[1:] - ff[:-1]
        print(ff)
        print(dff)
        
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
                
            ENSLON[iff] = []
            ENSLAT[iff] = []

            #ENSLAT1 = np.arange(
            #   LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
            #    LAT_MAX + DX[iff]*self.km2deg,
            #    DXG[iff]*self.km2deg)
            
            ENSLAT1 = np.arange(
                (LAT_MIN+LAT_MAX)/2,
                LAT_MIN-DX[iff]*self.km2deg,
                -DXG[iff]*self.km2deg)[::-1]
            
            ENSLAT1 = np.concatenate((ENSLAT1,
                                    np.arange(
                (LAT_MIN+LAT_MAX)/2,
                LAT_MAX+DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)[1:]))
                
                
            for I in range(len(ENSLAT1)):


                #_ENSLON = np.arange(
                #        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                #        LON_MAX+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                #        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)

                _ENSLON = np.arange(
                    (LON_MIN+LON_MAX)/2,
                    LON_MIN-DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    -DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)[::-1]
                _ENSLON = np.concatenate((_ENSLON,
                                        np.arange(
                    (LON_MIN+LON_MAX)/2,
                    LON_MAX+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)[1:]))
                    
                
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
            tdec[iff] *= self.factdec
            if tdec[iff]<self.tdecmin:
                    tdec[iff] = self.tdecmin
            if tdec[iff]>self.tdecmax:
                tdec[iff] = self.tdecmax 
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
                if return_q:
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
        
        if return_q:
            print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
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

class Basis_bmaux:
   
    def __init__(self,config,State):

        self.km2deg=1./110
        
        # Internal params
        self.file_aux = config.BASIS.file_aux
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
        self.facQ = config.BASIS.facQ
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

    def set_basis(self,time,return_q=False,**kwargs):
        
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
        ff = ff[1/ff<=self.lmax]
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        aux = xr.open_dataset(self.file_aux,decode_times=False)
        daTdec = aux['Tdec']
        daStd = aux['Std']

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        tdec_max = [None]*nf
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int32') # Nomber of spatial wavelet locations for a given frequency

        for iff in range(nf):
            
            # Spatial coordinates of wavelet components
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
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg
                        )
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
                            (np.abs((self.lat1d - lat) / self.km2deg) <= 1/ff[iff])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    
                ENSLAT[iff] = np.concatenate(([ENSLAT[iff],_ENSLAT1]))
                ENSLON[iff] = np.concatenate(([ENSLON[iff],_ENSLON1]))
            NP[iff] = len(ENSLON[iff])

            # Time decorrelation
            tdec[iff] = [None]*NP[iff]
            norm_fact[iff] = [None]*NP[iff]
            for P in range(NP[iff]):
                dlon = DX[iff]*self.km2deg/np.cos(ENSLAT[iff][P] * np.pi / 180.)
                dlat = DX[iff]*self.km2deg
                elon = np.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat = np.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2 = np.meshgrid(elon,elat)
                tdec_tmp = daTdec.interp(f=ff[iff],lon=elon2.flatten(),lat=elat2.flatten()).values
                if np.all(np.isnan(tdec_tmp)):
                    tdec[iff][P] = 0
                else:
                    tdec[iff][P] = np.nanmean(tdec_tmp)
                tdec[iff][P] *= self.factdec
                if tdec[iff][P]<self.tdecmin:
                        tdec[iff][P] = self.tdecmin
                if tdec[iff][P]>self.tdecmax:
                    tdec[iff][P] = self.tdecmax 
                # Compute time integral for each frequency for normalization
                tt = np.linspace(-tdec[iff][P],tdec[iff][P])
                tmp = np.zeros_like(tt)
                for i in range(tt.size-1):
                    tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff][P])*(tt[i+1]-tt[i])
                norm_fact[iff][P] = tmp.max()
            
            if len(tdec[iff])>0:
                tdec_max[iff] = np.max(tdec[iff])
                # Time coordinates, uniform for all points, set with the minimum tdec
                tdec_min = np.min(tdec[iff]) 
                enst[iff] = np.arange(-tdec_min/self.facnlt,deltat+tdec_min/self.facnlt , tdec_min/self.facnlt)
            else:
                enst[iff] = []
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
        Q = np.array([]) 
        iwave = 0
        self.iff_wavebounds = [None]*(nf+1)
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave
            for P in range(NP[iff]):
                dlon = DX[iff]*self.km2deg/np.cos(ENSLAT[iff][P] * np.pi / 180.)
                dlat = DX[iff]*self.km2deg
                elon = np.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat = np.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2 = np.meshgrid(elon,elat)
                Q_tmp = daStd.interp(f=ff[iff],lon=elon2.flatten(),lat=elat2.flatten()).values
                if np.all(np.isnan(Q_tmp)):
                    Q_tmp = 10**-10 # Not zero otherwise a ZeroDivisionError exception will be raised
                else:
                    Q_tmp = np.nanmean(Q_tmp)
                Q_tmp *= self.facQ
                # Fill Q
                _nwavet = 2*len(enst[iff])*ntheta
                Q = np.concatenate((Q,Q_tmp*np.ones((_nwavet,))))
                iwave += _nwavet

            print(f'lambda={1/ff[iff]:.1E}',
                    f'nlocs={NP[iff]:.1E}',
                    f'tdec={np.mean(tdec[iff]):.1E}',
                    f'Q={np.mean(Q[self.iff_wavebounds[iff]:]):.1E}')
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
        self.tdec_max = tdec_max
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
                Gt[t][iff] = np.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],)) * np.nan
                ind_tmp = 0
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    for P in range(self.NP[iff]):
                        if abs(dt) < self.tdec_max[iff]:
                            fact = self.window(dt / self.tdec[iff][P]) 
                            fact /= self.norm_fact[iff][P]
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta] = fact   
                            if P==0:
                                Nt[t][iff] += 1
                        ind_tmp += 2*self.ntheta
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
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                GtXf = GtXf[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
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
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                Gt = Gt[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][indNoNan] += (Gt*adGtXf).ravel()
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX

class Basis_bmaux_jax(Basis_bmaux):
    def __init__(self,config, State):
        super().__init__(config, State)

        self.operg_jit = jit(self.operg, static_argnums=0)

    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)

        # Convert dictionary to keys and values arrays
        self.Gt_keys = jnp.array(list(self.Gt.keys()))
        self.Gt_values = jnp.array(list(self.Gt.values()))
        self.Nt_values = jnp.array(list(self.Nt.values()))

        return res

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

            Gx[iff] = sparse.CSC((data, indices, indptr), shape=(self.nphys, nwaves))
                        

        return Gx, Nx


    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency 
        Nt = {} # Number of wave times tw such as abs(tw-t)<tdec

        for t in time:

            Gt[t] = [None,]*self.nf
            Nt[t] = [0,]*self.nf

            for iff in range(self.nf):
                Gt[t][iff] = np.zeros((self.nbasis,)) 
                ind_tmp = self.iff_wavebounds[iff]
                for it in range(len(self.enst[iff])):
                    dt = t - self.enst[iff][it]
                    for P in range(self.NP[iff]):
                        if abs(dt) < self.tdec_max[iff]:
                            fact = self.window(dt / self.tdec[iff][P]) 
                            fact /= self.norm_fact[iff][P]
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta] = fact   
                            if P==0:
                                Nt[t][iff] += 1
                        ind_tmp += 2*self.ntheta
        return Gt, Nt     
    
    def get_Gt_value(self, t):
        idx = jnp.where(self.Gt_keys == t, size=1)[0]  # Find index
        return self.Gt_values[idx][0], self.Nt_values[idx][0]  # Get corresponding value
    
    def operg(self, t, X, State_params=None):
        """
            Project to physical space
        """

        # Initialize phi
        phi = jnp.zeros(self.shape_phys).ravel()

        for iff in range(self.nf):
            # Get Gt value
            Gt, Nt = self.get_Gt_value(t)

            # Compute GtXf
            GtXf = (Gt[iff] * X)[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]

            # Replace NaNs with 0 (use jnp.nan_to_num for JAX compatibility)
            GtXf_no_nan = jnp.nan_to_num(GtXf)

            # Use shape-safe slicing instead of boolean indexing
            Nx_val = self.Nx[iff]

            # Dynamically reshape the sliced array
            reshaped_GtXf = GtXf_no_nan.reshape((-1, Nx_val))  # Ensure reshaping works dynamically

            # Update phi
            phi += self.Gx[iff] @ reshaped_GtXf.sum(axis=0)

        # Reshape phi back to physical space shape
        phi = phi.reshape(self.shape_phys)

        if State_params is not None:
            State_params[self.name_mod_var] = phi

        return phi
    
    
    
class Basis_wavelet3d:
   
    def __init__(self,config,State):

        self.km2deg=1./110
        
        # Internal params
        self.name_mod_var = config.BASIS.name_mod_var
        self.facnst = config.BASIS.facnst
        self.npsp = config.BASIS.npsp 
        self.facpsp = config.BASIS.facpsp 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tmin = config.BASIS.tmin
        self.tmax = config.BASIS.tmax
        self.sigma_Q = config.BASIS.sigma_Q
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background
        
        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon.min()
        self.lon_max = State.lon.max()
        self.lat_min = State.lat.min()
        self.lat_max = State.lat.max()
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Mask
        if State.mask is not None and np.any(State.mask):
            self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Path to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path

    def set_basis(self,time,return_q=False,**kwargs):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logfs = np.arange(
            np.log(1./self.lmin),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        fs = np.exp(logfs)
        fs = fs[1/fs<=self.lmax]
        dfs = fs[1:] - fs[:-1]
        nfs = len(fs)

        # Ensemble of pseudo-frequencies for the wavelets (time)
        logft = np.arange(
            np.log(1./self.tmin),
            np.log(1. / self.tmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ft = np.exp(logft)
        ft = ft[1/ft<=self.tmax]
        nft = len(ft)
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * fs[0] / dfs[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        
        print(f'Spatial wavelength: {1./np.exp(logfs)}')
        print(f'Time periods: {1./np.exp(logft)}')
        print(f'ntheta: {ntheta}')

        # Lon/Lat coordinates
        NS = np.empty(nfs, dtype='int32') # Nomber of spatial wavelet locations for a given frequency
        ENSLON = [None]*nfs # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nfs # Ensemble of latitudes of the center of each wavelets
        DXs = 1./fs*self.npsp * 0.5 # wavelet extension in space
        for ifs in range(nfs):
                
            ENSLON[ifs] = []
            ENSLAT[ifs] = []
            
            ENSLAT1 = np.arange(
                (LAT_MIN+LAT_MAX)/2,
                LAT_MIN-DXs[ifs]*self.km2deg,
                -DXs[ifs]/self.facnst*self.km2deg)[::-1]
            
            ENSLAT1 = np.concatenate((ENSLAT1,
                                    np.arange(
                (LAT_MIN+LAT_MAX)/2,
                LAT_MAX+DXs[ifs]*self.km2deg,
                DXs[ifs]/self.facnst*self.km2deg)[1:]))
                
                
            for I in range(len(ENSLAT1)):

                _ENSLON = np.arange(
                    (LON_MIN+LON_MAX)/2,
                    LON_MIN-DXs[ifs]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    -DXs[ifs]/self.facnst/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)[::-1]
                _ENSLON = np.concatenate((_ENSLON,
                                        np.arange(
                    (LON_MIN+LON_MAX)/2,
                    LON_MAX+DXs[ifs]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                    DXs[ifs]/self.facnst/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg)[1:]))
                    
                
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
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= .5/fs[ifs]) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= .5/fs[ifs])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    
                ENSLAT[ifs] = np.concatenate(([ENSLAT[ifs],_ENSLAT1]))
                ENSLON[ifs] = np.concatenate(([ENSLON[ifs],_ENSLON1]))
        
            NS[ifs] = len(ENSLON[ifs])

        # Time coordinates
        NT = np.empty(nft, dtype='int32') # Nomber of time wavelet locations for a given frequency
        DXt = 1./ft*self.npsp * 0.5 # wavelet extension in time
        ENST = [None,]*nft #  Ensemble of times of the center of each wavelets
        norm_fact = [None,]*nft 
        for ift in range(nft):
            _ENST = np.arange(
                (TIME_MIN+TIME_MAX)/2,
                TIME_MIN-DXt[ift],
                -DXt[ift]/self.facnst)[::-1]
            _ENST = np.concatenate((_ENST,
                                    np.arange(
                (TIME_MIN+TIME_MAX)/2,
                TIME_MAX+DXt[ift],
                DXt[ift]/self.facnst)[1:]))
        
            ENST[ift] = _ENST
            NT[ift] = _ENST.size

            tt = np.linspace(-DXt[ift],DXt[ift])
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + gaspari_cohn(tt[i],DXt[ift])*(tt[i+1]-tt[i])
            norm_fact[ift] = tmp.max()
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
        #nbasis = (NT[:,np.newaxis] * NS[np.newaxis,:]).sum() * 2 * ntheta
        #Q = self.sigma_Q * np.ones(nbasis,)
        Q = np.array([]) 
        iwave = 0
        self.fs_wavebounds = [None]*(nfs+1)
        for iff in range(nfs):
            self.fs_wavebounds[iff] = iwave
            if NS[iff]>0:
                _nwavet = 2*NT.sum()*ntheta*NS[iff]
                Q = np.concatenate((Q,self.sigma_Q*np.ones((_nwavet,)))) 
                iwave += _nwavet
        self.fs_wavebounds[-1] = iwave
    
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)

        self.DXs=DXs
        self.DXt=DXt
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.ENST=ENST
        self.NS=NS
        self.NT=NT
        self.nbasis=Q.size
        self.nfs=nfs
        self.nft=nft
        self.theta=theta
        self.ntheta=ntheta
        self.fs=fs
        self.ft=ft
        self.k = 2 * np.pi * fs
        self.norm_fact = norm_fact

        # Compute basis components
        self.Gx, self.Nx = self._compute_component_space() # in space
        self.Gt, self.Nt = self._compute_component_time(time) # in time
        
        if return_q:
            print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            return Xb, Q
    
    def _compute_component_space(self):

        Gx = [None,]*self.nfs
        Nx = [None,]*self.nfs

        for iff in range(self.nfs):

            data = np.empty((2*self.ntheta*self.NS[iff]*self.nphys,))
            indices = np.empty((2*self.ntheta*self.NS[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((2*self.ntheta*self.NS[iff],),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NS[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DXs[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DXs[iff])
                    )[0]
                xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                # Spatial tapering shape of the wavelet 
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]
                
                facs = mywindow(xx / self.DXs[iff]) * mywindow(yy / self.DXs[iff]) 

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

            Gt[t] = [None,]*self.nfs
            Nt[t] = [0,]*self.nfs

            for ifs in range(self.nfs):
                Gt[t][ifs] = np.zeros((self.fs_wavebounds[ifs+1]-self.fs_wavebounds[ifs],))
                ind_tmp = 0
                for ift in range(self.nft):
                    for P in range(len(self.ENST[ift])):
                        dt = t - self.ENST[ift][P]
                        if abs(dt) < self.DXt[ift]:
                            fact = gaspari_cohn(dt,self.DXt[ift]) / self.norm_fact[ift]
                            if fact!=0:   
                                Nt[t][ifs] += 1
                                Gt[t][ifs][ind_tmp:ind_tmp+2*self.ntheta*self.NS[ifs]] = fact   
                        ind_tmp += 2*self.ntheta*self.NS[ifs]
        return Gt, Nt     

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nfs):
            Xf = X[self.fs_wavebounds[iff]:self.fs_wavebounds[iff+1]]
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
        for iff in range(self.nfs):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.fs_wavebounds[iff]:self.fs_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX
    
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
    
    def set_basis(self,time,return_q=False,**kwargs):
        
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


        State0 = State.random()
        phi0 = np.random.random((self.nbasis,))
        adState1 = State.random()
        psi1 = adState1.getparams(vect=True)

        phi1 = self.operg_transpose(t,adState=adState1)
        self.operg(t,phi0,State=State0)
        psi0 = State0.getparams(vect=True)
        
        ps1 = np.inner(psi0,psi1)
        ps2 = np.inner(phi0,phi1)
            
        print(f'test G[{t}]:', ps1/ps2)


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

    def set_basis(self,time,return_q=False,**kwargs):

        self.nbasis = 0
        self.slice_basis = []

        if return_q:
            Xb = np.array([])
            Q = np.array([])

        for B in self.Basis:
            _Xb,_Q = B.set_basis(time,return_q=return_q,**kwargs)
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

