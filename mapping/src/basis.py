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
import pyinterp

from jax import debug

import matplotlib.pylab as plt

import datetime 

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
        
        elif config.BASIS.super=='BASIS_BM_JAX':
            return Basis_bm_jax(config, State)

        elif config.BASIS.super=='BASIS_GEOCUR':
            return Basis_geocur(config,State)

        elif config.BASIS.super=='BASIS_GAUSS3D':
            return Basis_gauss3d(config,State)

        elif config.BASIS.super=='BASIS_GAUSS3D_JAX':
            return Basis_gauss3d_jax(config,State)

        elif config.BASIS.super=='BASIS_GAUSS_ITG':
            return Basis_gauss_itg(config,State) 

        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_bmaux(config,State)

        elif config.BASIS.super=='BASIS_BMaux_JAX':
            return Basis_bmaux_jax(config,State)
        
        elif config.BASIS.super=='BASIS_LS':
            return BASIS_ls(config, State)
        
        elif config.BASIS.super=='BASIS_IT':
            return Basis_it(config, State)
        
        elif config.BASIS.super=='BASIS_IT_FLO':
            return Basis_it_flo(config, State)
        
        elif config.BASIS.super == 'BASIS_OFFSET':
            return Basis_offset(config,State)
        
        elif config.BASIS.super == 'BASIS_HBC':
            return Basis_hbc(config,State)

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

        # Reference time to have fixed time coordinates
        self.delta_time_ref = (config.EXP.init_date - datetime.datetime(1950,1,1,0)).total_seconds() / 24/3600

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
        #ff = ff[1/ff<=self.lmax]
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
                
            ENSLON[iff] = []
            ENSLAT[iff] = []

            # Latitudes
            dlat = DXG[iff]*self.km2deg
            lat0 = LAT_MIN - LAT_MIN%dlat - DX[iff]*self.km2deg  # To start at a fix latitude
            lat1 = LAT_MAX + DX[iff]*self.km2deg * 1.5
            ENSLAT1 = np.arange(lat0, lat1, dlat)
            
            # Longitudes
            for I in range(len(ENSLAT1)):
                dlon = DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.) *self.km2deg
                lon0 = LON_MIN - LON_MIN%dlon - DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg # To start at a fix longitude
                lon1 = LON_MAX + DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg * 1.5
                _ENSLON = np.arange(lon0, lon1, dlon)
                _ENSLAT = np.repeat(ENSLAT1[I],len(_ENSLON))

                # Mask
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
                            (np.abs((self.lat1d - lat) / self.km2deg) <= 1./ff[iff])
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

            t0 = -self.delta_time_ref % tdec[iff] # To start at a fix time
            enst[iff] = np.arange(t0 - tdec[iff]/self.facnlt,deltat+tdec[iff]/self.facnlt , tdec[iff]/self.facnlt) 
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

                facs = gaspari_cohn(xx, self.DX[iff]) * gaspari_cohn(yy, self.DX[iff]) 

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

class Basis_bm_jax(Basis_bm):
    def __init__(self,config, State):
        super().__init__(config, State)

        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

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
                    if abs(dt) < self.tdec[iff]:
                        fact = self.window(dt / self.tdec[iff]) 
                        fact /= self.norm_fact[iff]
                        if fact!=0:   
                            Nt[t][iff] += 1
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta*self.NP[iff]] = fact   
                    ind_tmp += 2*self.ntheta*self.NP[iff]
        return Gt, Nt    
    
    def get_Gt_value(self, t):
        idx = jnp.where(self.Gt_keys == t, size=1)[0]  # Find index
        return self.Gt_values[idx][0], self.Nt_values[idx][0]  # Get corresponding value
    
    def _operg(self, t, X):

        """
            Project to physicial space
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

        return phi

    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced


    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

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
        adparams = adState.params[self.name_mod_var]
        adX = self._operg_reduced_jit(t, adparams)
        
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

class Basis_gauss_itg():

    def __init__(self, config, State):

        self.km2deg = 1./110

        self.facns = config.BASIS.facns
        self.D_itg = config.BASIS.D_itg
        # self.sigma_T = config.BASIS.sigma_T
        self.sigma_Q = config.BASIS.sigma_Q
        self.name_mod_var = config.BASIS.name_mod_var

        # Grid params
        self.shape_phys = State.params[self.name_mod_var].shape
        self.nphys = np.prod(self.shape_phys)
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        self.Nwaves = config.BASIS.Nwaves

        self._operg_jit = jit(self._operg)
        self._operg_transpose_jit = jit(self._operg_transpose)
        self._operg_reduced_jit = jit(self._operg_reduced)
    
    def set_basis(self,time,return_q=False,**kwargs):
        
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.
        
        # Ensemble of reduced basis coordinates
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

        # Computing reduced basis elements gaussian supports 
        itg_xy_gauss = np.zeros((ENSLAT_itg.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_itg,ENSLON_itg)):
            iobs = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_itg) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.D_itg)
                    )[0]
            xx = (np.mod(self.lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[iobs] - lat0) / self.km2deg
            
            itg_xy_gauss[i,iobs] = mywindow(xx / self.D_itg) * mywindow(yy / self.D_itg)

        itg_xy_gauss = itg_xy_gauss.reshape((ENSLAT_itg.size,self.ny*self.nx))

        itg_xy_gauss = jnp.array(itg_xy_gauss)

        # self.itg_xy_gauss_operg = sparse.bcsr_fromdense(itg_xy_gauss.T)
        self.itg_xy_gauss_operg = sparse.CSR.fromdense(itg_xy_gauss.T)
        
        self.itg_xy_gauss_operg_transpose = sparse.CSR.fromdense(itg_xy_gauss)
        
        # Shape and number of basis elements 
        self.shape_basis = [self.Nwaves,            # - Number of tidal frequency components 
                            4,                      # - Number of estimated parameter (cos and sin for x and y axis)
                            self.ENSLAT_itg.size]       # - Number of basis spatial points 
        self.nbasis = np.prod(self.shape_basis)
        
        # Fill Q matrix
        Q = self.sigma_Q / (self.facns)  * np.ones((self.nbasis))

        if return_q:
            return np.zeros_like(Q), Q
    
    def _operg(self,X):

        """
            Project to physicial space
        """

        X = X.reshape(self.shape_basis)

        X_reshaped = X.reshape(self.shape_basis[0]*self.shape_basis[1],self.shape_basis[2])  
        
        result = sparse.csr_matmat(self.itg_xy_gauss_operg ,X_reshaped.T)

        result = result.T

        phi = result.reshape(self.shape_phys)

        return phi
    
    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def _operg_transpose(self,adparams):
        """
            Project to reduced space
        """

        adparams_reshaped = adparams.reshape(self.shape_phys[0]*self.shape_phys[1],self.shape_phys[2]*self.shape_phys[3])

        adX = sparse.csr_matmat(self.itg_xy_gauss_operg_transpose ,adparams_reshaped.T)

        adX = adX.T

        adX = adX.ravel()

        return adX
    
    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(X)

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
        adparams = adState.params[self.name_mod_var]

        # adX = self._operg_transpose_jit(adparams)
        adX = self._operg_reduced_jit(t, adparams) # a bit quicker to compute 
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX

class Basis_gauss3d:
   
    def __init__(self, config, State):

        self.km2deg = 1./110

        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns
        self.facnlt = config.BASIS.facnlt
        self.sigma_D = config.BASIS.sigma_D
        self.sigma_T = config.BASIS.sigma_T
        self.sigma_Q = config.BASIS.sigma_Q
        self.normalize_fact = config.BASIS.normalize_fact
        self.name_mod_var = config.BASIS.name_mod_var
        self.time_spinup = config.BASIS.time_spinup
        self.fcor = config.BASIS.fcor
        self.flag_variable_Q = config.BASIS.flag_variable_Q
        self.path_sad = config.BASIS.path_sad
        self.name_var_sad = config.BASIS.name_var_sad

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

        # Time dependancy 
        self.time_dependant = config.BASIS.time_dependant 

        # For time normalization
        if self.normalize_fact:
            tt = np.linspace(-self.sigma_T,self.sigma_T)
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/self.sigma_T)*(tt[i+1]-tt[i])
            self.norm_fact = tmp.max()
        
        # Longitude unit
        self.lon_unit = State.lon_unit
        
    
    def set_basis(self,time,return_q=False,**kwargs):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time
        
        # Coordinates in space
        dlat = self.sigma_D/self.facns*self.km2deg
        lat0 = LAT_MIN - LAT_MIN%dlat - self.sigma_D*(1-1./self.facns)*self.km2deg  # To start at a fix latitude
        lat1 = LAT_MAX + 1.5*dlat
        ENSLAT1 = np.arange(lat0, lat1, dlat)
        ENSLAT = []
        ENSLON = []
        for I in range(len(ENSLAT1)):
            dlon = self.sigma_D/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg
            lon0 = LON_MIN - LON_MIN%dlon - self.sigma_D*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg # To start at a fix longitude
            lon1 = LON_MAX + dlon * 1.5
            ENSLON1 = np.arange(lon0, lon1, dlon)
            ENSLAT = np.concatenate(([ENSLAT,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON = np.concatenate(([ENSLON,ENSLON1]))
        self.ENSLAT = ENSLAT
        self.ENSLON = ENSLON
        
        # Coordinates in time
        if self.time_dependant : 
            ENST = np.arange(-self.sigma_T*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.sigma_T/self.facnlt , self.sigma_T/self.facnlt)
            self.ENST = ENST

        # BASIS PROPERTIES 
        if self.time_dependant:
            self.nbasis = ENST.size * ENSLAT.size
            self.shape_basis = [ENST.size,ENSLAT.size]
        else : 
            self.nbasis = ENSLAT.size
            self.shape_basis = [ENSLAT.size]
        # PARAMETER PROPERTIES 
        self.nphys = self.lon1d.size
        self.shape_phys = [self.ny, self.nx]
        
        
        # Fill Q matrix
        if self.flag_variable_Q:
            Q = np.zeros(self.nbasis)
            sad = xr.open_dataset(self.path_sad)[self.name_var_sad['var']]**2 # Std -> Variance
            # Convert longitude 
            if np.sign(sad[self.name_var_sad[self.name_var_sad['lon']]].data.min())==-1 and self.lon_unit=='0_360':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], sad[self.name_var_sad['lon']].data % 360))})
            elif np.sign(sad[self.name_var_sad[self.name_var_sad[self.name_var_sad['lon']]]].data.min())>=0 and self.lon_unit=='-180_180':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], (sad[self.name_var_sad['lon']].data + 180) % 360 - 180 ))})
            sad = sad.sortby(sad[self.name_var_sad['lon']])    
            grid = pyinterp.Grid2D(pyinterp.Axis(sad[self.name_var_sad['lon']], is_circle=False), pyinterp.Axis(sad[self.name_var_sad['lat']]), sad.T)  # Note: Transpose required
            i = 0
            range_ENST = len(self.ENST) if self.time_dependant else 1
            for _ in range(range_ENST):
                for (lon,lat) in zip(ENSLON,ENSLAT):
                    indphys = np.where(
                            (np.abs((np.mod(self.lon1d - lon+180,360)-180) / self.km2deg * np.cos(lat * np.pi / 180.)) <= self.sigma_D) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= self.sigma_D)
                            )[0]
                    xx = (np.mod(self.lon1d[indphys] - lon+180,360)-180) / self.km2deg * np.cos(lat * np.pi / 180.) 
                    yy = (self.lat1d[indphys] - lat) / self.km2deg
                    facS = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
                    Q_tmp = pyinterp.bivariate(grid, self.lon1d[indphys],self.lat1d[indphys], bounds_error=False)
                    if np.all(np.isnan(Q_tmp)):
                        Q_tmp = 10**-10 # Not zero otherwise a ZeroDivisionError exception will be raised
                    else:
                        Q_tmp = (np.average(Q_tmp, weights=facS) * self.fcor / (self.facns*self.facnlt))**.5 
                    Q[i] = Q_tmp 
                    i += 1
        else:
            if self.time_dependant :
                Q = (self.fcor * self.sigma_Q**2 / (self.facns*self.facnlt))**.5   * np.ones((self.nbasis))
            else : # not normalizing by self.facnlt 
                Q = (self.fcor * self.sigma_Q**2 / (self.facns))**.5   * np.ones((self.nbasis))
        
        size_ENST = ENST.size if self.time_dependant else 0
        print(f'lambda={self.sigma_D:.1E}',
            f'nlocs={ENSLAT.size:.1E}',
            f'tdec={self.sigma_T:.1E}',
            f'ntime={size_ENST:.1E}',
            f'Q={np.mean(Q):.1E}')
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')

        # Compute basis components
        # In space 
        Gauss_xy = self._compute_component_space()
        self.Gauss_xy = Gauss_xy
        self.Nx = ENSLAT.size
        # In time 
        if self.time_dependant:
            Gauss_t, Nt = self._compute_component_time(time)
            self.Gauss_t = Gauss_t
            self.Nt = Nt
        

        if return_q:
            return np.zeros_like(Q), Q
        

    def _compute_component_space(self):
        """
            Gaussian functions in space
        """

        data = np.empty((self.ENSLAT.size*self.lon1d.size,))
        indices = np.empty((self.ENSLAT.size*self.lon1d.size,),dtype=int)
        sizes = np.zeros((self.ENSLAT.size,),dtype=int)
        ind_tmp = 0
        for i,(lat0,lon0) in enumerate(zip(self.ENSLAT,self.ENSLON)):
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

        return csc_matrix((data, indices, indptr), shape=(self.lon1d.size, self.ENSLAT.size))
    
    def _compute_component_time(self, time):
        """
            Gaussian functions in time
        """
        Gauss_t = {}
        Nt = {}
        for t in time:
            Gauss_t[t] = np.zeros((self.ENSLAT.size*self.ENST.size))
            Nt[t] = 0
            ind_tmp = 0
            for it in range(len(self.ENST)):
                dt = t - self.ENST[it]
                if abs(dt) < self.sigma_T:
                    fact = self.window(dt / self.sigma_T) 
                    if self.normalize_fact:
                        fact /= self.norm_fact
                    if self.time_spinup is not None and t<self.time_spinup:
                        fact *= (1-self.window(t / self.time_spinup))
                    if fact!=0:   
                        Nt[t] += 1
                        Gauss_t[t][ind_tmp:ind_tmp+self.ENSLAT.size] = fact   
                ind_tmp += self.ENSLAT.size

        return Gauss_t, Nt

    def operg(self, t, X, State=None):

        """
            Project to physicial space
        """
        
        phi = np.zeros(self.nphys)
        if self.time_dependant:
            GtX = self.Gauss_t[t] * X
            ind0 = np.nonzero(self.Gauss_t[t])[0]
            if ind0.size>0:
                GtX = GtX[ind0].reshape(self.Nt[t],self.Nx)
                phi += self.Gauss_xy.dot(GtX.sum(axis=0))
        else: 
            phi = self.Gauss_xy.dot(X)

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

        adX = np.zeros(self.nbasis)
        adparams = adState.params[self.name_mod_var].ravel()

        if self.time_dependant:
            Gt = self.Gauss_t[t]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t],self.Nx)
                adGtX = self.Gauss_xy.T.dot(adparams)
                adGtX = np.repeat(adGtX[np.newaxis,:],self.Nt[t],axis=0)
                adX[ind0] += (Gt*adGtX).ravel()
        
        else : 
            adX += self.Gauss_xy.T.dot(adparams)

        adState.params[self.name_mod_var] *= 0.

        return adX
 
class Basis_gauss3d_jax(Basis_gauss3d):

    def __init__(self,config, State):
        super().__init__(config, State)

        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)
        
    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)

        self.time = time
        self.vect_time = jnp.eye(time.size)

        return res
    
    def _compute_component_space(self):
        """
            Gaussian functions in space
        """

        Gauss_2d = np.zeros((self.ENSLAT.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(self.ENSLAT,self.ENSLON)):
            indphys = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
                    )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[indphys] - lat0) / self.km2deg
            Gauss_2d[i,indphys] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
        Gauss_2d = jnp.array(Gauss_2d)
        return sparse.CSR.fromdense(Gauss_2d.T)

    def _compute_component_time(self, time):

        Gt_np = np.zeros((time.size,self.nbasis))
        ind_tmp = 0
        for it in range(len(self.ENST)):
            for _ in range(self.ENSLAT.size):
                for i,t in enumerate(time) :
                    dt = t - self.ENST[it]
                    if abs(dt) < self.sigma_T:
                        fact = self.window(dt / self.sigma_T) 
                        if self.normalize_fact:
                            fact /= self.norm_fact
                        if self.time_spinup is not None and t<self.time_spinup:
                            fact *= (1-self.window(t / self.time_spinup))
                        if fact!=0:   
                            Gt_np[i,ind_tmp:ind_tmp+1] = fact
                ind_tmp += 1
        Gt = sparse.csr_fromdense(jnp.array(Gt_np).T)

        return Gt, None
    
    def get_Gt_value(self, t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gauss_t @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):

        """
            Project to physicial space
        """

        # Initialize phi
        phi = jnp.zeros(self.nphys)

        if self.time_dependant:

            # Get Gt value
            Gt = self.get_Gt_value(t)
            GtX = Gt * X

            reshaped_GtX = GtX.reshape((-1, self.Nx))

            phi += self.Gauss_xy @ (reshaped_GtX.sum(axis=0))

        else : 

            phi += self.Gauss_xy @ X

        
        phi = phi.reshape(self.shape_phys)
        
        return phi

    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

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
        adparams = adState.params[self.name_mod_var]
        adX = self._operg_reduced_jit(t, adparams)
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX

class Basis_bmaux:
   
    def __init__(self,config,State,multi_mode=False):

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
        self.norm_time = config.BASIS.norm_time
        
        # Grid params
        self.nphys= State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.lon1d = State.lon.flatten()
        self.lat1d = State.lat.flatten()

        # Reference time to have fixed time coordinates
        self.delta_time_ref = (config.EXP.init_date - datetime.datetime(1950,1,1,0)).total_seconds() / 24/3600

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

        # Longitude unit
        self.lon_unit = State.lon_unit

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
        
        self.multi_mode = multi_mode

    def set_basis(self,time,return_q=False,**kwargs):

        print('Setting Basis BMaux...')
        
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
        # Convert longitude 
        if np.sign(aux['lon'].data.min())==-1 and self.lon_unit=='0_360':
            aux = aux.assign_coords({'lon':(('lon', aux['lon'].data % 360))})
        elif np.sign(aux['lon'].data.min())>=0 and self.lon_unit=='-180_180':
            aux = aux.assign_coords({'lon':(('lon', (aux['lon'].data + 180) % 360 - 180 ))})
        aux = aux.sortby(aux['lon'])    
        daTdec = aux['Tdec']
        daStd = aux['Std']

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        #DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int32') # Nomber of spatial wavelet locations for a given frequency

        for iff in range(nf):
            
            # Spatial coordinates of wavelet components
            ENSLON[iff] = []
            ENSLAT[iff] = []

            facns = self.facns
            DXG = DX / facns

            # Latitudes
            dlat = DXG[iff]*self.km2deg
            lat0 = LAT_MIN - LAT_MIN%dlat - DX[iff]*self.km2deg  # To start at a fix latitude
            lat1 = LAT_MAX + DX[iff]*self.km2deg 
            ENSLAT1 = np.arange(lat0, lat1, dlat)
            
            # Longitudes
            for I in range(len(ENSLAT1)):
                dlon = DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.) *self.km2deg
                lon0 = LON_MIN - LON_MIN%dlon - DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg # To start at a fix longitude
                lon1 = LON_MAX + DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg 
                _ENSLON = np.arange(lon0, lon1, dlon)
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
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= 1/ff[iff]) &
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
            enst[iff] = [None]*NP[iff]
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
                if self.norm_time:
                    tt = np.linspace(-tdec[iff][P],tdec[iff][P])
                    tmp = np.zeros_like(tt)
                    for i in range(tt.size-1):
                        tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff][P])*(tt[i+1]-tt[i])
                    norm_fact[iff][P] = tmp.max()
                else:
                    norm_fact[iff][P] = 1
                # Time decorrelation
                t0 = -self.delta_time_ref % tdec[iff][P] # To start at a fix time
                enst[iff][P] = np.arange(t0 - tdec[iff][P]/self.facnlt, deltat+tdec[iff][P]/self.facnlt , tdec[iff][P]/self.facnlt)
                
        # Harmonize the wavelet time center dimensions for all point by adding NaN if needed 
        # (we must do that for the time operator Gt to be independent from the space operator Gx)
        enst_same_dim = [None]*nf
        for iff in range(nf):
            max_number_enst_iff = np.max([enst[iff][P].size for P in range(NP[iff])])
            enst_same_dim[iff] = np.zeros((NP[iff], max_number_enst_iff)) * np.nan
            for P in range(NP[iff]):
                enst_same_dim[iff][P, :enst[iff][P].size] = enst[iff][P]
        
        # Fill the Q diagonal matrix (expected variance for each wavelet)   
        print('Computing Q')  

        iwave = 0
        self.iff_wavebounds = [None]*(nf+1)
        Q = np.array([])
        facQ = self.facQ  # Move outside the loop for efficiency

        std = []
        for iff in range(nf):
            std.append([])
            std[iff] = []
            for P in range(NP[iff]):
                
                dlon = DX[iff] * self.km2deg / np.cos(ENSLAT[iff][P] * np.pi / 180.0)
                dlat = DX[iff] * self.km2deg

                # Precompute interpolation grid once
                elon = np.linspace(ENSLON[iff][P] - dlon, ENSLON[iff][P] + dlon, 10)
                elat = np.linspace(ENSLAT[iff][P] - dlat, ENSLAT[iff][P] + dlat, 10)
                elon2, elat2 = np.meshgrid(elon, elat)

                std_tmp_values = daStd.interp(f=ff[iff], lon=elon2.ravel(), lat=elat2.ravel()).values
                std_tmp = np.nanmean(std_tmp_values) if not np.all(np.isnan(std_tmp_values)) else 10**-10
                std[iff].append(std_tmp)

        for iff in range(nf):

            self.iff_wavebounds[iff] = iwave
            _nwavef = 0
            Qf_list = []  # Use a list instead of np.concatenate in loops

            enst_data = enst_same_dim[iff]  # Store reference to avoid repeated access
            num_it = enst_data.shape[1]

            for it in range(num_it):
                for P in range(NP[iff]):
                    enst_value = enst_data[P, it]
                    if np.isnan(enst_value):
                        Q_tmp = 10**-10  # Small nonzero value to avoid division errors
                    else:
                        Q_tmp = +std[iff][P]

                    Q_tmp *= facQ  # Multiply after NaN check

                    # Store Q_tmp values in list for later concatenation
                    Qf_list.append(Q_tmp * np.ones(2 * ntheta))
                    _nwavef += 2 * ntheta

            # Convert list to numpy array once
            if Qf_list:
                Qf = np.concatenate(Qf_list)
                Q = np.concatenate((Q, Qf))

            iwave += _nwavef

            print(f'lambda={1/ff[iff]:.1E}',
                f'nlocs={NP[iff]:.1E}',
                f'tdec={np.mean(tdec[iff]):.1E}',
                f'Q={np.mean(Q[self.iff_wavebounds[iff]:iwave]):.1E}')

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
        self.enst=enst_same_dim
        self.nbasis=Q.size
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff

        # Compute basis components
        print('Computing Spatial components')
        self.Gx, self.Nx = self._compute_component_space() # in space
        print('Computing Time components')
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
                for it in range(self.enst[iff].shape[1]):
                    for P in range(self.NP[iff]):
                        dt = t - self.enst[iff][P,it]
                        if abs(dt)>self.tdec[iff][P] or np.isnan(self.enst[iff][P,it]):
                            fact = 0
                        else:
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
            if not self.multi_mode:
                State[self.name_mod_var] = phi
            else:
                State[self.name_mod_var] += phi
        else:
            return phi

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState[self.name_mod_var].ravel()
        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                Gt = Gt[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][indNoNan] += (Gt*adGtXf).ravel()
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
        
        return adX

class Basis_bmaux_jax(Basis_bmaux):

    def __init__(self,config, State, multi_mode=False):
        super().__init__(config, State,multi_mode=multi_mode)

        # JIT 
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)
        self.time = time
        self.vect_time = jnp.eye(time.size)

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
        
        for iff in range(self.nf):
            nbasis_f = self.iff_wavebounds[iff+1] - self.iff_wavebounds[iff]
            Gt_np = np.zeros((time.size,nbasis_f))
            ind_tmp = 0
            for it in range(self.enst[iff].shape[1]):
                for P in range(self.NP[iff]):
                    for i,t in enumerate(time) :
                        dt = t - self.enst[iff][P,it]
                        if not (abs(dt)>self.tdec[iff][P] or np.isnan(self.enst[iff][P,it])):
                            fact = self.window(dt / self.tdec[iff][P])
                            fact /= self.norm_fact[iff][P]
                            Gt_np[i,ind_tmp:ind_tmp+2*self.ntheta] = fact
                    ind_tmp += 2*self.ntheta
            Gt[iff] = sparse.csr_fromdense(jnp.array(Gt_np).T)

        return Gt, None

    def get_Gt_value(self, t, iff):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt[iff] @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):
        """
            Project to physicial space
        """

        # Initialize phi
        phi = jnp.zeros(self.shape_phys).ravel()

        for iff in range(self.nf):

            Gt = self.get_Gt_value(t,iff)
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = Gt * Xf

            # Replace NaNs with 0 (use jnp.nan_to_num for JAX compatibility)
            GtXf_no_nan = jnp.nan_to_num(GtXf)

            # # Use shape-safe slicing instead of boolean indexing
            Nx_val = self.Nx[iff]

            # # Dynamically reshape the sliced array
            reshaped_GtXf = GtXf_no_nan.reshape((-1, Nx_val))  # Ensure reshaping works dynamically

            # Update phi
            phi += self.Gx[iff] @ reshaped_GtXf.sum(axis=0)

        # Reshape phi back to physical space shape
        phi = phi.reshape(self.shape_phys)

        return phi


    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Update State
        if State is not None:
            if not self.multi_mode:
                # State[self.name_mod_var] = phi
                State.params[self.name_mod_var] = phi
            else:
                # State[self.name_mod_var] += phi
                State.params[self.name_mod_var] += phi
        else:
            return phi
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        # if adState[self.name_mod_var] is None:
        if adState.params[self.name_mod_var] is None:
            # adState[self.name_mod_var] = np.zeros((self.nphys,))
            adState.params[self.name_mod_var] = np.zeros((self.nphys,))
        # adparams = adState[self.name_mod_var]
        adparams = adState.params[self.name_mod_var]
        
        adX = self._operg_reduced_jit(t, adparams)
        
        if not self.multi_mode:
            # adState[self.name_mod_var] *= 0.
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

class Basis_it:
   
    def __init__(self,config, State):

        ##################
        ### - COMMON - ###
        ##################

        # Grid specs
        self.km2deg =1./110 # Kilometer to deg factor 
        self.nphys = State.lon.size
        self.shape_phys = (State.ny,State.nx)
        self.ny = State.ny
        self.nx = State.nx
        self.X = State.X 
        self.Y = State.Y 
        self.lon = State.lon
        self.lat = State.lat
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

        # Name of controlled parameters
        self.name_params = config.BASIS.name_params 

        # Basis reduction factor
        self.facns = config.BASIS.facgauss # Factor for gaussian spacing in space
        self.facnlt = config.BASIS.facgauss # Factor for gaussian spacing in time

        # Tidal frequencies 
        # self.Nwaves = len(config.BASIS.w_waves) # Number
        # self.omegas = np.asarray(config.BASIS.w_waves) # List of frequencies 
        self.Nwaves = config.BASIS.Nwaves

        # Information for setting parameter background
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # Information for seeting initial control parameter vector 
        self.path_restart = config.BASIS.path_restart # Path to the get the vector at the start of the minimization for the specified Basis 

        ################################
        ### - EQUIVALENT HEIGHT He - ###
        ################################

        self.D_He = config.BASIS.D_He # Space scale of gaussian decomposition for He (in km)
        self.T_He = config.BASIS.T_He # Time scale of gaussian decomposition for He (in days)

        self.control_He_offset = config.BASIS.control_He_offset # if True an offset on the equivalent height is controlled
        self.control_He_variation = config.BASIS.control_He_variation # if True the spatial variations of equivalent height are controlled 

        if "He" in self.name_params and not(self.control_He_offset or self.control_He_variation):
            print("Warning : He is in the controlled parameters but neither control_He_offset or control_He_variation are set to True")

        if (self.control_He_offset or self.control_He_variation) and "He" not in self.name_params:
            print("Warning : He is set up being controlled in the reduced basis but it is not in the controlled parameters.")

        self.He_time_dependant = config.BASIS.He_time_dependant # True if the equivalent height variation parameters are time dependant (True by default)

        self.sigma_B_He = config.BASIS.sigma_B_He # Covariance sigma for equivalent height He parameter
        self.sigma_B_He_offset = config.BASIS.sigma_B_He_offset # Covariance sigma for equivalent height He parameter (used if self.control_He_offset = True)

        ##########################################
        ### - HEIGHT BOUNDARY CONDITIONS hbc - ###
        ##########################################

        self.D_bc = config.BASIS.D_bc # Space scale of gaussian decomposition for hbc (in km)
        self.T_bc = config.BASIS.T_bc # Time scale of gaussian decomposition for hbc (in days)

        # Number of angles (computed from the normal of the border) of incoming waves
        if config.BASIS.Ntheta>0: 
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°

        self.sigma_B_bc = config.BASIS.sigma_B_bc # Covariance sigma for hbc parameter

        ########################################
        ### - INTERNAL TIDE GENERATION itg - ###
        ########################################

        self.D_itg = config.BASIS.D_itg # Space scale of gaussian decomposition for itg (in km)
        self.T_itg = config.BASIS.T_itg # Time scale of gaussian decomposition for itg (in days)

        self.itg_time_dependant = config.BASIS.itg_time_dependant # True if the itg parameter is time dependant (False by default)

        self.sigma_B_itg = config.BASIS.sigma_B_itg # Covariance sigma for itg parameter
        
    def set_basis(self,time,return_q=False):

        """
        Set the basis for the controlled parameters of the model and calculate reduced basis functions.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        return_q : bool, optional
            If True, returns the covariance matrix Q and the background vector array Xb, by default False.

        Returns:
        --------
        tuple of np.ndarray
            If return_q is True, returns a tuple containing:
                - Xb : np.ndarray
                    Background vector array Xb.
                - Q : np.ndarray or None
                    Covariance matrix Q.
        
        Notes:
        ------
        - This function initializes and sets the basis for controlled model parameters by 
        calculating their shapes in both the reduced and physical spaces.
        - It handles chosen parameters in config file between : "He", "hbcx", "hbcy", and "itg".
        - It prints the reduced order of the recued basis.
        - If return_q is True and necessary sigma_B values are set, it creates and returns 
        the Q array based on these sigma_B values.
        - If path_background is set and the file exists, it loads background vector from 
        the specified file.
        """
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time 

        self.bc_gauss = {} # Dictionary containing gaussian basis elements for each parameters. 
        self.shape_params = {} # Dictionary containing the shapes in the reduced space of each of the parameters.
        self.shape_params_phys = {} # Dictionary containing the shapes in the physical space of each of the parameters.
        
        #############################################
        ### SETTING UP THE REDUCED BASIS ELEMENTS ###
        #############################################

        for name in self.name_params : 

            # - Equivalent Height He - #
            if name == "He": 
                self.shape_params["He"], self.shape_params_phys["He"] = self.set_He(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX) 

            # - X height boundary conditions - #
            if name == "hbcx":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_hbcx(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX)
            
            # - Y height boundary conditions - #
            if name == "hbcy": 
                self.shape_params["hbcE"], self.shape_params["hbcW"], self.shape_params_phys["hbcE"], self.shape_params_phys["hbcW"] = self.set_hbcy(time, LAT_MIN, LAT_MAX, TIME_MIN, TIME_MAX)
            
            # - Internal tide generation itg - #
            if name == "itg": 
                self.shape_params["itg"], self.shape_params_phys["itg"] = self.set_itg(time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX)
        
        ############################################
        ### REDUCED BASIS INFORMATION ATTRIBUTES ###
        ############################################

        # Dictionary with the number of parameters in reduced space 
        self.n_params = {}
        for param in self.shape_params.keys():
            # if param == "He" :#and (self.control_He_offset and not self.control_He_variation):
                
            #     self.n_params[param] = int(1) 
            # else : 
            if self.shape_params[param] == []:
                self.n_params[param] = 0
            else :
                self.n_params[param] = np.prod(self.shape_params[param])

        if "He" in self.name_params and self.control_He_offset : # Adding the offset in the He control parameter 
            self.n_params["He"] += 1 
        # self.n_params = dict(zip(self.shape_params.keys(), map(np.prod, self.shape_params.values()))) 
        # if "He" in self.name_params and self.control_He_offset : # Adding the offset in the He control parameter 
        #     self.n_params["He"] += 1 
        # Dictionary with the number of parameters in pysical space
        self.n_params_phys = dict(zip(self.shape_params_phys.keys(), map(np.prod, self.shape_params_phys.values()))) 
        # Total number of parameters in the reduced space
        self.nbasis = sum(self.n_params.values()) 
        # Total number of parameters in the physical space
        self.nphys = sum(self.n_params_phys.values()) 
        # Total number of parameters in the physical space (including time dimension)
        self.nphystot = 0 
        for param in self.n_params_phys.keys():
            if (param == "itg" and not self.itg_time_dependant) or (param == "He" and not self.He_time_dependant):
                self.nphystot += self.n_params_phys[param] 
            else : 
                self.nphystot += self.n_params_phys[param]*time.size
        # Setting up slice information for parameters 
        interval = 0 ; interval_phys = 0 
        self.slice_params = {} # Dictionary with the slices of parameters in the reduced space
        self.slice_params_phys = {} # Dictionary with the slices of parameters in the physical space
        for name in self.shape_params.keys():
            self.slice_params[name]=slice(interval,interval+self.n_params[name])
            self.slice_params_phys[name]=slice(interval_phys,interval_phys+self.n_params_phys[name])
            interval += self.n_params[name]; interval_phys += self.n_params_phys[name]
        # PRINTING REDUCED ORDER : #     
        print(f'reduced order: {self.nphystot} --> {self.nbasis}\nreduced factor: {int(self.nphystot/self.nbasis)}')

        #########################################
        ### COMPUTING THE COVARIANCE MATRIX Q ###
        #########################################        

        if return_q :
            if None not in [self.sigma_B_He, self.sigma_B_bc, self.sigma_B_itg]:
                Q = np.zeros((self.nbasis,)) # Initializing
                for name in self.slice_params.keys() :

                    # - Equivalent Height He - #
                    if name == "He" : 
                        Q[self.slice_params[name]]=self.sigma_B_He
                        if self.control_He_offset : 
                            Q[0] = self.sigma_B_He_offset

                    # - Height boundary conditions hbc - #
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

                    # - Internal tide generation itg - #
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

        """
        Set the equivalent height He parameter recuced basis elements.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapeHe : list
                    Shape of the He parameter in the reduced space.
                - shapeHe_phys : list
                    Shape of the He parameter in the physical space.
        
        Notes:
        ------
        - This function sets the basis elements for the He parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of He parameters in the reduced space.
        """

        ###############################
        ###   - SPACE DIMENSION -   ###
        ###############################

        if self.control_He_variation: 

            # - COORDINATES - # 
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

            # - GAUSSIAN FUNCTIONS - # 
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
            self.He_xy_gauss = He_xy_gauss
        
        ##############################
        ###   - TIME DIMENSION -   ###
        ##############################
        
        if self. control_He_variation and self.He_time_dependant:

            # - COORDINATES - # 
            ENST_He = np.arange(-self.T_He*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_He/self.facnlt , self.T_He/self.facnlt)
            
            # - GAUSSIAN FUNCTIONS - # 
            He_t_gauss = np.zeros((ENST_He.size,time.size))
            for i,time0 in enumerate(ENST_He):
                iobs = np.where(abs(time-time0) < self.T_He)
                He_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_He)

            self.He_t_gauss = He_t_gauss

        ####################################  
        ###   - PARAMETER DIMENSIONS -   ###
        ####################################

        # Note : shapeHe is the shape of the He variation control. The He offset isn't taken into account into it. 

        nHe = 0
        if self.control_He_variation: 

            if self.He_time_dependant:
            
                shapeHe = [ENST_He.size,ENSLAT_He.size]

            else : 

                shapeHe = [ENSLAT_He.size]
            
            nHe += np.prod(shapeHe)
            if self.control_He_offset:
                nHe +=1

        elif self.control_He_offset:

            shapeHe = []
            nHe +=1

        shapeHe_phys = [self.nx,self.ny]

        print('nHe:',nHe)

        return shapeHe, shapeHe_phys
        
    
    def set_hbcx(self,time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX):

        """
        Set the height boundary conditions hbcx parameter recuced basis elements for both the South and North boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcS : list
                    Shape of the South boundary hbcx parameter in the reduced space.
                - shapehbcN : list
                    Shape of the North boundary hbcx parameter in the reduced space.
                - shapehbcS_phys : list
                    Shape of the South boundary hbcx parameter in the physical space.
                - shapehbcN_phys : list
                    Shape of the North boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcx parameters in the reduced space.
        """

        ###############################
        ###   - SPACE DIMENSION -   ###
        ###############################

        # - SOUTH - # 
        # Ensemble of reduced basis longitudes
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # - NORTH - #
        # Ensemble of reduced basis longitudes
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 

        # Saving gaussian reduced basis elements 
        self.bc_gauss["hbcS"] = bc_S_gauss # For South boundary 
        self.bc_gauss["hbcN"] = bc_N_gauss # For North boundary 

        ##############################
        ###   - TIME DIMENSION -   ###
        ##############################

        # Ensemble of reduced basis timesteps
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        bc_t_gauss = np.zeros((ENST_bc.size,time.size))
        for i,time0 in enumerate(ENST_bc):
            iobs = np.where(abs(time-time0) < self.T_bc)
            bc_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_bc)
        
        # Gaussian reduced basis element
        self.bc_t_gauss = bc_t_gauss

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################

        # - Shapes of the hbcy parameters in the reduced space.
        shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                     2,                     # - Number of controlled components (cos & sin)
                     self.Ntheta,           # - Number of angles
                     ENST_bc.size,          # - Number of basis timesteps
                     bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
        
        shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                     2,                     # - Number of controlled components (cos & sin)
                     self.Ntheta,           # - Number of angles
                     ENST_bc.size,          # - Number of basis timesteps
                     bc_N_gauss.shape[0]]   # - Number of basis spatial elements 

        # - Shapes of the hbcy parameters in the physical space.
        shapehbcS_phys = shapehbcN_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.nx]         # - Number of gridpoints along x axis
        
        print('nbcx:',np.prod(shapehbcS)+np.prod(shapehbcN))

        return shapehbcS, shapehbcN, shapehbcS_phys, shapehbcN_phys


    def set_hbcy(self,time, LAT_MIN, LAT_MAX, TIME_MIN, TIME_MAX): 

        """
        Set the height boundary conditions hbcy parameter recuced basis elements for both the East and West boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcE : list
                    Shape of the East boundary hbcx parameter in the reduced space.
                - shapehbcW : list
                    Shape of the West boundary hbcx parameter in the reduced space.
                - shapehbcE_phys : list
                    Shape of the East boundary hbcx parameter in the physical space.
                - shapehbcW_phys : list
                    Shape of the West boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcy parameters in the reduced space.
        """
        
        #########################################
        ###   - COMPUTING SPACE DIMENSION -   ###
        #########################################

        # Ensemble of reduced basis latitudes (common for each boundaries)
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)

        # - EAST - #
        # Computing reduced basis elements gaussian supports 
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # - WEST - # 
        # Computing reduced basis elements gaussian supports 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # Gaussian reduced basis elements
        self.bc_gauss["hbcE"] = bc_E_gauss # For East boundary 
        self.bc_gauss["hbcW"] = bc_W_gauss # For West boundary 
        
        ########################################
        ###   - COMPUTING TIME DIMENSION -   ###
        ########################################

        # Ensemble of reduced basis timesteps
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        bc_t_gauss = np.zeros((ENST_bc.size,time.size))
        for i,time0 in enumerate(ENST_bc):
            iobs = np.where(abs(time-time0) < self.T_bc)
            bc_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_bc)
        
        # Gaussian reduced basis element
        self.bc_t_gauss = bc_t_gauss

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################
        
        # Shapes of the hbcx parameters in the reduced space.
        shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                     2,                         # - Number of controlled components (cos & sin)
                     self.Ntheta,               # - Number of angles
                     ENST_bc.size,              # - Number of basis timesteps
                     bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
        
        shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                     2,                         # - Number of controlled components (cos & sin)
                     self.Ntheta,               # - Number of angles
                     ENST_bc.size,              # - Number of basis timesteps
                     bc_W_gauss.shape[0]]       # - Number of basis spatial elements 

        # Shapes of the hbcx parameters in the physical space.
        shapehbcE_phys = shapehbcW_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.ny]         # - Number of gridpoints along x axis

        print('nbcy:',np.prod(shapehbcE)+np.prod(shapehbcW))

        return shapehbcE, shapehbcW, shapehbcE_phys, shapehbcW_phys
    
    def set_itg(self,time, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TIME_MIN, TIME_MAX):

        """
        Set the internal tide generation itg parameter recuced basis elements.

        Parameters:
        ----------
        time : ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude for the grid.
        LAT_MAX : float
            Maximum latitude for the grid.
        LON_MIN : float
            Minimum longitude for the grid.
        LON_MAX : float
            Maximum longitude for the grid.
        TIME_MIN : float
            Minimum time value for the grid.
        TIME_MAX : float
            Maximum time value for the grid.

        Returns:
        -------
        tuple
            - shapeitg: Shape of the itg parameter in the reduced space.
            - shapeitg_phys: Shape of the itg parameter in the physical space.
        """

        #########################################
        ###   - COMPUTING SPACE DIMENSION -   ###
        #########################################

        # Ensemble of reduced basis coordinates
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

        # Computing reduced basis elements gaussian supports 
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
        self.itg_xy_gauss = itg_xy_gauss

        ########################################
        ###   - COMPUTING TIME DIMENSION -   ###
        ########################################

        if self.itg_time_dependant:

            # Ensemble of reduced basis coordinates
            ENST_itg = np.arange(-self.T_itg*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_itg/self.facnlt , self.T_itg/self.facnlt)
        
            # Computing reduced basis elements gaussian supports 
            itg_t_gauss = np.zeros((ENST_itg.size,time.size))
            for i,time0 in enumerate(ENST_itg):
                iobs = np.where(abs(time-time0) < self.T_itg)
                itg_t_gauss[i,iobs] = mywindow(abs(time-time0)[iobs]/self.T_itg)

            self.itg_t_gauss = itg_t_gauss

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################

        # Shapes of the itg parameter in the reduced space.
        if self.itg_time_dependant: # If parameter is time dependant 
            shapeitg = [self.Nwaves,            # - Number of tidal frequency components 
                        4,                      # - Number of estimated parameter (cos and sin for x and y axis)
                        ENST_itg.size,          # - Number of basis timesteps 
                        ENSLAT_itg.size]        # - Number of basis spatial points 
        else : # If parameter is not time dependant
            shapeitg = [self.Nwaves,            # - Number of tidal frequency components 
                        4,                      # - Number of estimated parameter (cos and sin for x and y axis)
                        ENSLAT_itg.size]        # - Number of basis spatial points 

        # Shapes of the itg parameter in the physical space.
        shapeitg_phys = [self.Nwaves,       # - Number of tidal frequency components 
                         4,                 # - Number of estimated parameter (cos and sin for x and y axis)
                         self.nx,           # - Number of grid points along x axis. 
                         self.ny]           # - Number of grid points along y axis. 
                          
        print('nitg:',np.prod(shapeitg))

        return shapeitg, shapeitg_phys
    
    def operg(self,t,X,State=None):

        """
        Perform the basis projection operation for a given time and parameter vector.

        This method projects the given parameter vector X from reduced basis onto the model grid, at the provided time t. The results can be stored in the provided State object.
                
        operg : | REDUCED SPACE >>>>>> PHYSICAL SPACE | (Model Grid) 

        Parameters:
        ----------
        t : float
            The time at which the projection is performed.
        X : ndarray
            The parameter vector to be projected.
        State : object, optional
            State object to store the parameters after projection. If not provided, the method returns the projected vector onto physical space.

        Returns:
        -------
        phi : ndarray
            The projected parameter vector if State is not provided. Otherwise, updates the State object in place.

        """

        ##############################
        ###   - INITIALIZATION -   ###
        ##############################

        # Index of timestep t in self.time array
        indt = np.argmin(np.abs(self.time-t))  
        # Variable to return  
        phi = np.zeros((self.nphys,))

        ##########################################
        ###   - BASIS PROJECTION OPERATION -   ###
        ##########################################

        for name in self.slice_params_phys.keys():

            # - Equivalent Height He - # 
            if name == "He":

                if self.control_He_offset:
                    X_offset = X[self.slice_params[name]][0] # He offset value
                    phi[self.slice_params_phys[name]] += X_offset

                if self.control_He_variation:

                    # extracting the He variation control vector 
                    if self.control_He_offset: # if offset is controled, first element is withdrawed
                        X_variation = X[self.slice_params[name]][1:]
                    else : 
                        X_variation = X[self.slice_params[name]]

                    # computing He in physical space 
                    if self.He_time_dependant:
                        phi[self.slice_params_phys[name]] += np.tensordot(
                                                                np.tensordot(X_variation.reshape(self.shape_params[name]),self.He_xy_gauss,(1,0)),
                                                                self.He_t_gauss[:,indt],(0,0)).flatten()
                    else : 
                        phi[self.slice_params_phys[name]] += np.tensordot(X_variation.reshape(self.shape_params[name]),self.He_xy_gauss,(-1,0)).flatten()

            # - Height boundary conditions hbc - #
            if name in ["hbcS","hbcN","hbcW","hbcE"]:
                phi[self.slice_params_phys[name]] = np.tensordot(
                                                        np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.bc_gauss[name],(-1,0)),
                                                        self.bc_t_gauss[:,indt],(-2,0)).flatten()
                
            # - Internal Tide Generation itg - #
            if name == "itg":
                if self.itg_time_dependant == True :
                    phi[self.slice_params_phys[name]] = np.tensordot(
                                                            np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.itg_xy_gauss,(-1,0)),
                                                            self.itg_t_gauss[:,indt],(2,0)).flatten() 
                else : 
                    phi[self.slice_params_phys[name]] = np.tensordot(X[self.slice_params[name]].reshape(self.shape_params[name]),self.itg_xy_gauss,(-1,0)).flatten()


        ##############################
        ###   - SAVING OUTPUTS -   ###
        ##############################

        if State is not None:
            for name in self.name_params : 

                # - Height boundary conditions hbcx - #
                if name == "hbcx" : 
                    State.params['hbcx'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                           np.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                # - Height boundary conditions hbcy - #
                elif name == "hbcy" : 
                    State.params['hbcy'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1),
                                                           np.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1)),axis=1)
                # - Equivalent Height He - OR - Internal Tide Generation itg - #
                else : 
                    State.params[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])
        else: 
            return phi

    def operg_transpose(self,t,phi=None,adState=None):

        """
        Perform the transpose basis projection operation for a given time and parameters.

        This method computes the transposed operation of the basis projection operation operg. 
                
        operg_transpose : | PHYSICAL SPACE  >>>>>> REDUCED SPACE | 

        Parameters:
        ----------
        t : float
            The time at which the projection is performed.
        phi : ndarray, optional
            The vector of concatenated parameters in the physical space. 
        adState : object, optional
            State object containing the parameters.

        Returns:
        -------
        adX : ndarray
            The result of the transposed basis projection of the parameter vector.

        """

        ##############################
        ###   - INITIALIZATION -   ###
        ##############################

        # Dictionary containing the values of alle the params 
        param = {} 
        # Index of timestep t in self.time array
        indt = np.argmin(np.abs(self.time-t)) 
        # Variable to return 
        adX = np.zeros((self.nbasis,)) 

        # Getting the parameters 
        if phi is not None: # If provided through phi ndarray argument 
            for name in self.slice_params_phys.keys():
                param[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])
        elif adState is not None: # If provided through adState object argument 
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

        ####################################
        ###   - TRANSPOSED OPERATION -   ###
        ####################################

        for name in self.slice_params.keys():

            # - Equivalent Height He - #  
            if name == "He":

                if self.control_He_offset:
                    adX[self.slice_params[name]][0] = np.sum(param[name])
                if self.control_He_variation:

                    slice_variation = self.slice_params[name]
                    if self.control_He_offset: # if offset is controled, first element is withdrawed from slice
                        slice_variation = slice(slice_variation.start+1,slice_variation.stop) 

                    if self.He_time_dependant:
                        adX[slice_variation] = np.tensordot(param[name][:,:,np.newaxis]*self.He_t_gauss[:,indt],
                                                                    self.He_xy_gauss[:,:,:],([0,1],[1,2])).flatten()
                    else : 
                        adX[slice_variation] = np.tensordot(param[name],
                                                                    self.He_xy_gauss,([0,1],[1,2])).flatten()
            
            # - Height boundary conditions hbc - #        
            if name in ["hbcS","hbcN","hbcW","hbcE"]:
                adX[self.slice_params[name]] = np.tensordot(param[name][:,:,:,:,np.newaxis]*self.bc_t_gauss[:,indt],
                                                            self.bc_gauss[name],(-2,-1)).flatten()
            
            # - Internal Tide Generation itg - #
            if name == "itg":
                if self.itg_time_dependant == True :
                    adX[self.slice_params[name]] = np.tensordot(param[name][:,:,:,:,np.newaxis]*self.itg_t_gauss[:,indt],
                                                                        self.itg_xy_gauss[:,:,:],([2,3],[-2,-1])).flatten()
                else : 
                    adX[self.slice_params[name]] = np.tensordot(param[name],
                                                                self.itg_xy_gauss,([-2,-1],[-2,-1])).flatten()

        # Setting adState parameters to 0 
        if adState is not None:
            for name in self.name_params:
                adState.params[name] *= 0

        return adX
        
    def test_operg(self, t, State):

        """
        Test the operg function for consistency.

        This method performs a consistency check for the `operg` and `operg_transpose`
        functions by comparing the inner products of the state vectors and their projections.

        Parameters:
        ----------
        t : float
            The time at which the test is performed.
        State : object
            The state object containing the parameters shape and information to be used for testing.

        Returns:
        -------
        None

        Notes:
        -----
        - The method generates random states and projections, applies the `operg` and
        `operg_transpose` functions, and compares the inner products of the results.
        - The ratio of the inner products is printed to verify it equals to 0..

        """

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

class Basis_hbc: 

    def __init__(self,config, State):

        ##################
        ### - COMMON - ###
        ##################

        # Grid specs
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.ny = State.ny
        self.nx = State.nx
        self.lonS = State.lon[0,:]
        self.lonN = State.lon[-1,:]
        self.latE = State.lat[:,0]
        self.latW = State.lat[:,-1]
        self.km2deg =1./110 # Kilometer to deg factor 

        # Name of controlled parameters
        # self.name_params = config.BASIS.name_params 
        self.name_mod_var = config.BASIS.name_mod_var

        # Basis reduction factor
        self.facns = config.BASIS.facns # Factor for gaussian spacing in space
        self.facnlt = config.BASIS.facnlt # Factor for gaussian spacing in time

        # Tidal frequencies 
        self.Nwaves = config.BASIS.Nwaves # Number of tidal components

        # Time dependancy 
        self.time_dependant = config.BASIS.time_dependant

        ##########################################
        ### - HEIGHT BOUNDARY CONDITIONS hbc - ###
        ##########################################

        self.D_bc = config.BASIS.D_bc # Space scale of gaussian decomposition for hbc (in km)
        self.T_bc = config.BASIS.T_bc # Time scale of gaussian decomposition for hbc (in days)

        # Number of angles (computed from the normal of the border) of incoming waves
        if config.BASIS.Ntheta>0: 
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°

        self.sigma_B_bc = config.BASIS.sigma_B_bc # Covariance sigma for hbc parameter

        self.window = mywindow

        # JIT
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

    def set_basis(self,time,return_q=False):

        """
        Set the basis for the controlled parameters of the model and calculate reduced basis functions.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        return_q : bool, optional
            If True, returns the covariance matrix Q and the background vector array Xb, by default False.

        Returns:
        --------
        tuple of np.ndarray
            If return_q is True, returns a tuple containing:
                - Xb : np.ndarray
                    Background vector array Xb.
                - Q : np.ndarray or None
                    Covariance matrix Q.
        
        """
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time 
        self.vect_time = jnp.eye(time.size)

        self.Gxy = {} # Dictionary containing gaussian basis elements for each parameters. 
        self.shape_params = {} # Dictionary containing the shapes in the reduced space of each of the parameters.
        self.shape_params_phys = {} # Dictionary containing the shapes in the physical space of each of the parameters.
        
        #############################################
        ### SETTING UP THE REDUCED BASIS ELEMENTS ###
        #############################################

        # - In Time - #
        if self.time_dependant:
            self.set_bc_gauss_t(time, TIME_MIN, TIME_MAX) 

        # - In Space - # 
        for name in self.name_mod_var : 

            # - X height boundary conditions - #
            if name == "HBCX":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_bc_gauss_hbcx(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            
            # - Y height boundary conditions - #
            if name == "HBCY": 
                self.shape_params["hbcE"], self.shape_params["hbcW"], self.shape_params_phys["hbcE"], self.shape_params_phys["hbcW"] = self.set_bc_gauss_hbcy(LAT_MIN, LAT_MAX)

        ############################################
        ### REDUCED BASIS INFORMATION ATTRIBUTES ###
        ############################################

        # Dictionary with the number of parameters in reduced space 
        self.n_params = {}
        for param in self.shape_params.keys():
            if self.shape_params[param] == []:
                self.n_params[param] = 0
            else :
                self.n_params[param] = np.prod(self.shape_params[param])

        # Dictionary with the number of parameters in pysical space
        self.n_params_phys = dict(zip(self.shape_params_phys.keys(), map(np.prod, self.shape_params_phys.values()))) 
        # Total number of parameters in the reduced space
        self.nbasis = sum(self.n_params.values()) 
        # Total number of parameters in the physical space
        self.nphys = sum(self.n_params_phys.values()) 
        # Total number of parameters in the physical space (including time dimension)
        self.nphystot = 0 
        for param in self.n_params_phys.keys():
            self.nphystot += self.n_params_phys[param]*time.size
        # Setting up slice information for parameters 
        interval = 0 ; interval_phys = 0 
        self.slice_params = {} # Dictionary with the slices of parameters in the reduced space
        self.slice_params_phys = {} # Dictionary with the slices of parameters in the physical space
        for name in self.shape_params.keys():
            self.slice_params[name]=slice(interval,interval+self.n_params[name])
            self.slice_params_phys[name]=slice(interval_phys,interval_phys+self.n_params_phys[name])
            interval += self.n_params[name]; interval_phys += self.n_params_phys[name]
        # PRINTING REDUCED ORDER : #     
        print(f'reduced order: {self.nphystot} --> {self.nbasis}\nreduced factor: {int(self.nphystot/self.nbasis)}')

        #########################################
        ### COMPUTING THE COVARIANCE MATRIX Q ###
        #########################################        

        if return_q :
            if self.sigma_B_bc is not None:
                Q = np.zeros((self.nbasis,)) # Initializing
                for name in self.slice_params.keys() :

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
                        # Q[self.slice_params[name]]=self.sigma_B_bc
                        Q[self.slice_params[name]]=self.sigma_B_bc/(self.facnlt*self.facns)

            else:
                Q = None

            Xb = np.zeros_like(Q)

            return Xb, Q
    
    def set_bc_gauss_hbcx(self, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):

        """
        Set the height boundary conditions hbcx parameter recuced basis elements for both the South and North boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcS : list
                    Shape of the South boundary hbcx parameter in the reduced space.
                - shapehbcN : list
                    Shape of the North boundary hbcx parameter in the reduced space.
                - shapehbcS_phys : list
                    Shape of the South boundary hbcx parameter in the physical space.
                - shapehbcN_phys : list
                    Shape of the North boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcx parameters in the reduced space.
        """

        ###############################
        ###   - SPACE DIMENSION -   ###
        ###############################

        # - SOUTH - # 
        # Ensemble of reduced basis longitudes
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # - NORTH - #
        # Ensemble of reduced basis longitudes
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 

        # Saving gaussian reduced basis elements 
        self.Gxy["hbcS"] = sparse.CSR.fromdense(jnp.array(bc_S_gauss.T)) # For South boundary 
        self.Gxy["hbcN"] = sparse.CSR.fromdense(jnp.array(bc_N_gauss.T)) # For South boundary

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################

        # - Shapes of the hbcy parameters in the reduced space.

        if self.time_dependant : # the parameters include the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,     # - Number of basis timesteps
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,          # - Number of basis timesteps
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 

        # - Shapes of the hbcy parameters in the physical space.
        shapehbcS_phys = shapehbcN_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.nx]         # - Number of gridpoints along x axis
        
        print('nbcx:',np.prod(shapehbcS)+np.prod(shapehbcN))

        return shapehbcS, shapehbcN, shapehbcS_phys, shapehbcN_phys

    def set_bc_gauss_hbcy(self,LAT_MIN, LAT_MAX): 

        """
        Set the height boundary conditions hbcy parameter recuced basis elements for both the East and West boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcE : list
                    Shape of the East boundary hbcx parameter in the reduced space.
                - shapehbcW : list
                    Shape of the West boundary hbcx parameter in the reduced space.
                - shapehbcE_phys : list
                    Shape of the East boundary hbcx parameter in the physical space.
                - shapehbcW_phys : list
                    Shape of the West boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcy parameters in the reduced space.
        """
        
        #########################################
        ###   - COMPUTING SPACE DIMENSION -   ###
        #########################################

        # Ensemble of reduced basis latitudes (common for each boundaries)
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)

        # - EAST - #
        # Computing reduced basis elements gaussian supports 
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # - WEST - # 
        # Computing reduced basis elements gaussian supports 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # Gaussian reduced basis elements
        self.Gxy["hbcE"] = sparse.CSR.fromdense(jnp.array(bc_E_gauss.T)) # For East boundary 
        self.Gxy["hbcW"] = sparse.CSR.fromdense(jnp.array(bc_W_gauss.T)) # For West boundary 

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################
        
        # Shapes of the hbcx parameters in the reduced space.
        if self.time_dependant : # the parameters include the time dependency

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements

        # Shapes of the hbcx parameters in the physical space.
        shapehbcE_phys = shapehbcW_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.ny]         # - Number of gridpoints along x axis

        print('nbcy:',np.prod(shapehbcE)+np.prod(shapehbcW))

        return shapehbcE, shapehbcW, shapehbcE_phys, shapehbcW_phys
    
    def set_bc_gauss_t(self,time, TIME_MIN, TIME_MAX):
        # Ensemble of reduced basis timesteps
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        # bc_t_gauss = np.zeros((time.size,ENST_bc.size))
        # for i,time0 in enumerate(ENST_bc):
        #     iobs = np.where(abs(time-time0) < self.T_bc)
        #     bc_t_gauss[iobs,i] = mywindow(abs(time-time0)[iobs]/self.T_bc)

        # self.bc_t_gauss = bc_t_gauss

        self.ENST_bc = ENST_bc

        Gt = np.zeros((time.size,self.ENST_bc.size))

        for i,t in enumerate(time) :
            for it in range(len(self.ENST_bc)):
                dt = t - self.ENST_bc[it]
                if abs(dt) < self.T_bc:
                    fact = self.window(dt / self.T_bc) 
                    if fact!=0:   
                        Gt[i,it] = fact
        
        self.Gt = sparse.csr_fromdense(jnp.array(Gt).T)

    def get_bc_t_gauss_value(self,t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt @ self.vect_time[idt[0]] # Get corresponding value

    def _operg(self,t,X):

        """
        Perform the basis projection operation for a given time and parameter vector.

        This method projects the given parameter vector X from reduced basis onto the model grid, at the provided time t. The results can be stored in the provided State object.
                
        operg : | REDUCED SPACE >>>>>> PHYSICAL SPACE | (Model Grid) 

        Parameters:
        ----------
        t : float
            The time at which the projection is performed.
        X : ndarray
            The parameter vector to be projected.
        State : object, optional
            State object to store the parameters after projection. If not provided, the method returns the projected vector onto physical space.

        Returns:
        -------
        phi : ndarray
            The projected parameter vector if State is not provided. Otherwise, updates the State object in place.

        """

        ##############################
        ###   - INITIALIZATION -   ###
        ##############################

        # Time gaussian function
        if self.time_dependant:
            _Gt = self.get_bc_t_gauss_value(t) 

        # Variable to return  
        phi = jnp.zeros((self.nphys,))

        ##########################################
        ###   - BASIS PROJECTION OPERATION -   ###
        ##########################################

        for name in self.slice_params_phys.keys():

            _X = X[self.slice_params[name]]
            _X = _X.reshape(self.shape_params[name])

            if self.time_dependant:
                _X = (_Gt[None,None,None,:, None]*_X).sum(axis=3, keepdims=False)

            _X_t = _X.T

            Gxy_X_t = sparse.csr_matmat(self.Gxy[name],_X_t.reshape(_X_t.shape[0],-1))

            Gxy_X_t = Gxy_X_t.reshape((Gxy_X_t.shape[0],)+_X_t.shape[1:])

            Gxy_X = Gxy_X_t.T

            phi = phi.at[self.slice_params_phys[name]].set(Gxy_X.flatten())#.reshape(self.shape_params_phys[name]))
        
        return phi

    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Update State
        if State is not None:
            for name in self.name_mod_var:
                # - Height boundary conditions hbcx - #
                if name == "HBCX" : 
                    State.params[self.name_mod_var[name]] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                            np.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                # - Height boundary conditions hbcy - #
                elif name == "HBCY" : 
                    State.params[self.name_mod_var[name]] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1),
                                                            np.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1)),axis=1)
            # State.params[self.name_mod_var] = phi
        else:
            return phi
    
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        # if adState.params[self.name_mod_var] is None:
        #     adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        # Getting the parameters 
        # if phi is not None: # If provided through phi ndarray argument 
        #     for name in self.slice_params_phys.keys():
        #         param[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])

        adparams = np.zeros((self.nphys))
        if adState is not None: # If provided through adState object argument 
            for name in self.name_mod_var:
                if name == "HBCX" : 
                    # adparams["hbcS"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcS"])
                    # adparams["hbcN"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcN"])
                    adparams[self.slice_params_phys["hbcS"]] = adState.params[self.name_mod_var[name]][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcN"]] = adState.params[self.name_mod_var[name]][:,1,:,:,:].flatten()
                elif name == "HBCY" : 
                    # adparams["hbcE"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcE"])
                    # adparams["hbcW"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcW"])
                    adparams[self.slice_params_phys["hbcE"]] = adState.params[self.name_mod_var[name]][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcW"]] = adState.params[self.name_mod_var[name]][:,1,:,:,:].flatten()
                # else :
                #     param[name] = adState.params[name].reshape(self.shape_params_phys[name])
        # adparams = adparams.flatten()
        # adparams = adState.getparams(self.name_params,vect=True)

        adX = self._operg_reduced_jit(t, adparams)
        
        for name in self.name_mod_var: 
            adState.params[self.name_mod_var[name]] *= 0.
        
        return adX

class Basis_offset:

    def __init__(self,config, State):
        
        self.name_mod_var = config.BASIS.name_mod_var
        self.shape_phys = State.params[self.name_mod_var].shape
        self.nphys = np.prod(self.shape_phys)
        self.ny = State.ny
        self.nx = State.nx
        self.sigma_B = config.BASIS.sigma_B
        
        if self.sigma_B == None : 
            print("Warning, please prescribe sigma_B for Basis Offset") 
    
    def set_basis(self,time,return_q=False,**kwargs):
        self.nbasis = 1
        self.shape_basis = [1]

        # Fill Q matrix
        Q = self.sigma_B * np.ones((self.nbasis))

        if return_q:
            return np.zeros_like(Q), Q

    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """

        phi = X*np.ones(self.shape_phys)

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
        adparams = adState.params[self.name_mod_var]

        adX = [np.sum(adparams)]
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX

class Basis_it_flo:
   
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
            # t0 = datetime.datetime.now()
            _X = X[self.slice_basis[i]]
            phi = np.append(phi, B.operg(t, _X, State=State))
            # print(f"Basis operg time for Basis {i} : ",datetime.datetime.now()-t0)
        
        if State is None:
            return phi


    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        adX = np.array([])
        for i,B in enumerate(self.Basis):
            # t0 = datetime.datetime.now()
            adX = np.concatenate((adX, B.operg_transpose(t, adState=adState)))
            # print(f"Basis operg_transpose time for Basis {i} : ",datetime.datetime.now()-t0)

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

