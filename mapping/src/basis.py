#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:24:24 2021

@author: leguillou
"""
from .config import USE_FLOAT64
import os, sys
import numpy as np
import logging
import pickle 
import datetime 
import xarray as xr
import scipy
from scipy.sparse import csc_matrix
from scipy.integrate import quad
import jax.numpy as jnp 
from jax.experimental import sparse as sparse
from jax import jit, lax
import jax
import pyinterp
import matplotlib.pylab as plt 

from .tools import gaspari_cohn

jax.config.update("jax_enable_x64", USE_FLOAT64)


def Basis(config, State, verbose=True, multi_mode=False, *args, **kwargs):
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
            return Basis_bm(config, State,multi_mode=multi_mode)
          
        elif config.BASIS.super=='BASIS_BM_JAX':
            return Basis_bm_jax(config, State,multi_mode=multi_mode)

        elif config.BASIS.super=='BASIS_GAUSS3D':
            return Basis_gauss3d(config,State,multi_mode=multi_mode)

        elif config.BASIS.super=='BASIS_GAUSS3D_JAX':
            return Basis_gauss3d_jax(config,State,multi_mode=multi_mode)
          
        elif config.BASIS.super=='BASIS_GAUSSV2':
            return Basis_gaussv2(config, State) 

        elif config.BASIS.super=='BASIS_WAVELET3D':
            return Basis_wavelet3d(config,State)
    
        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_bmaux(config,State,multi_mode=multi_mode)
        
        elif config.BASIS.super=='BASIS_BMaux_JAX':
            return Basis_bmaux_jax(config,State,multi_mode=multi_mode)
        
        elif config.BASIS.super == 'BASIS_OFFSET':
            return Basis_offset(config,State,multi_mode=multi_mode)
        
        elif config.BASIS.super == 'BASIS_OFFSET_JAX':
            return Basis_offset_jax(config,State,multi_mode=multi_mode)
        
        elif config.BASIS.super == 'BASIS_HBC_JAX':
            return Basis_hbc_jax(config,State)
        
        elif config.BASIS.super == 'BASIS_HBC_CST_JAX':
            return Basis_hbc_cst_jax(config,State)

        else:
            sys.exit(config.BASIS.super + ' not implemented yet')

 
###############################################################################
#                          Balanced-Motions                                   #
###############################################################################    

class Basis_gaussv2:
   
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
         
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff)) 

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

            nwave += len(enst[iff])*NP[iff]
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
         
        Q = np.array([]) 
        iwave = 0
        self.iff_wavebounds = [None]*(nf+1)
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave
            if NP[iff]>0:
                _nwavet = len(enst[iff])*NP[iff]
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

            data = np.empty((self.NP[iff]*self.nphys,))
            indices = np.empty((self.NP[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((self.NP[iff],),dtype=int)

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
 
                # Gaspari-Cohn 
                sizes[iwave] = indphys.size
                indices[ind_tmp:ind_tmp+indphys.size] = indphys 
                data[ind_tmp:ind_tmp+indphys.size] = gaspari_cohn(xx, self.DX[iff]) * gaspari_cohn(yy, self.DX[iff]) 
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
                            Gt[t][iff][ind_tmp:ind_tmp+self.NP[iff]] = fact   
                    ind_tmp += self.NP[iff]
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
 
class Basis_bm:
   
    def __init__(self,config,State,multi_mode=False):

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
        self.norm_time = config.BASIS.norm_time
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

        # Compute geostrophic velocoties
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1,0),(1,0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])
        
        # Gravity 
        self.g = 9.81

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])

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

        self.multi_mode = multi_mode

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
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= 1./ff[iff]) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= 1./ff[iff])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    
                ENSLAT[iff] = np.concatenate(([ENSLAT[iff],_ENSLAT1]))
                ENSLON[iff] = np.concatenate(([ENSLON[iff],_ENSLON1]))
            

            NP[iff] = len(ENSLON[iff])
            tdec[iff] = self.tmeso*self.lmeso**self.sloptdec * ff[iff]**self.sloptdec
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
                    Q = np.concatenate((Q,self.Qmax/self.facns * np.ones((_nwavet,))))
                else:
                    # Slope
                    Q = np.concatenate((Q,self.Qmax/self.facns * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavet,)))) 
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
        print('Computing Spatial components')
        self.Gx, self.Nx = self._compute_component_space() # in space
        print('Computing Time components')
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

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) 

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
                        if self.norm_time: 
                            fact /= self.norm_fact[iff]
                        if fact!=0:   
                            Nt[t][iff] += 1
                            Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta*self.NP[iff]] = fact   
                    ind_tmp += 2*self.ntheta*self.NP[iff]
        return Gt, Nt

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        return _u, _v
    
    def _ssh2uv_adj(self, adu, adv):

        """
        Adjoint of geostrophic velocity computation.
        """

        # _adssh lives on padded grid: (ny+1, nx+1)
        _adssh = np.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))

        _adssh[1:,:]  += -self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:-1,:] +=  self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:,1:]  +=  self.g / self.f_on_v * adv / self.dx_on_v
        _adssh[:,:-1] += -self.g / self.f_on_v * adv / self.dx_on_v

        # map padded grid back to physical ssh grid:
        # physical ssh[i,j] == _ssh[i+1,j+1]
        adssh = _adssh[1:,1:].copy()

        # contributions from the padded first row/col (mode='edge' duplicates edge values)
        # add the padded southern row (index 0) to physical southern row (adssh[0,:])
        adssh[0,:] += _adssh[0,1:]
        # add the padded western column (index 0) to physical western column (adssh[:,0])
        adssh[:,0] += _adssh[1:,0]
        # the padded corner (0,0) was duplicated as well — add it into adssh[0,0]
        adssh[0,0] += _adssh[0,0]

        return adssh
    
    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        ssh = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            ind0 = np.nonzero(self.Gt[t][iff])[0]
            if ind0.size>0:
                GtXf = GtXf[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                ssh += self.Gx[iff].dot(GtXf.sum(axis=0))
        ssh = ssh.reshape(self.shape_phys)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = ssh
            else:
                State[self.name_mod_var] += ssh
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros((self.nphys,))
            adState[self.name_mod_v] = np.zeros((self.nphys,))
            
        adX = np.zeros(self.nbasis)

        adssh = adState[self.name_mod_var]
        if self.compute_velocities:
            adssh += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])

        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adssh.ravel())
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
        
        return adX

class Basis_bm_jax(Basis_bm):
    def __init__(self,config, State, multi_mode=False):
        super().__init__(config, State, multi_mode=multi_mode)

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
            for it in range(len(self.enst[iff])):
                for _ in range(self.NP[iff]):
                    for i,t in enumerate(time) :
                        dt = t - self.enst[iff][it]
                        if not (abs(dt)>self.tdec[iff] or np.isnan(self.enst[iff][it])):
                            fact = self.window(dt / self.tdec[iff])
                            if self.norm_time: 
                                fact /= self.norm_fact[iff]
                            Gt_np[i,ind_tmp:ind_tmp+2*self.ntheta] = fact
                    ind_tmp += 2*self.ntheta
            Gt[iff] = sparse.csr_fromdense(jnp.array(Gt_np).T)

        return Gt, None
    
    def _get_Gt_value(self, t):
        idx = jnp.where(self.Gt_keys == t, size=1)[0]  # Find index
        return self.Gt_values[idx][0], self.Nt_values[idx][0]  # Get corresponding value
    
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
        ssh = self._operg_jit(t, X)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = ssh
            else:
                State[self.name_mod_var] += ssh
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros((self.nphys,))
            adState[self.name_mod_v] = np.zeros((self.nphys,))

        adssh = adState[self.name_mod_var]
        if self.compute_velocities:
            adssh += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])
        adX = self._operg_reduced_jit(t, adssh)

        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
    
        return adX

class Basis_gauss3d:
   
    def __init__(self, config, State, multi_mode=False):

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
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

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

        # Gravity 
        self.g = 9.81

        # Compute geostrophic velocoties
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1,0),(1,0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])

        # Mask
        if State.mask is not None and np.any(State.mask):
            self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
 
        # For time normalization
        if self.normalize_fact:
            tt = np.linspace(-self.sigma_T,self.sigma_T)
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/self.sigma_T)*(tt[i+1]-tt[i])
            self.norm_fact = tmp.max()
        
        # Longitude unit
        self.lon_unit = State.lon_unit
        
        # For multi-basis
        self.multi_mode = multi_mode
         
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
        ENST = np.arange(-self.sigma_T*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.sigma_T/self.facnlt , self.sigma_T/self.facnlt)
        self.ENST = ENST
    
        self.nbasis = ENST.size * ENSLAT.size
        self.nphys = self.lon1d.size
        self.shape_phys = [self.ny, self.nx]
        self.shape_basis = [ENST.size,ENSLAT.size]
        
        # Fill Q matrix
        if self.flag_variable_Q:
            Q = []
            sad = xr.open_dataset(self.path_sad)[self.name_var_sad['var']] 
            # Convert longitude 
            if np.sign(sad[self.name_var_sad['lon']].data.min())==-1 and self.lon_unit=='0_360':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], sad[self.name_var_sad['lon']].data % 360))})
            elif (np.sign(sad[self.name_var_sad['lon']].data.min())>=0  or sad[self.name_var_sad['lon']].data.max()>180) and self.lon_unit=='-180_180':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], (sad[self.name_var_sad['lon']].data + 180) % 360 - 180 ))})
            sad = sad.sortby(sad[self.name_var_sad['lon']])    
            for (lon,lat) in zip(ENSLON,ENSLAT):
                # Precompute interpolation grid once
                dlon = .5 * self.sigma_D/np.cos(lat*np.pi/180.)
                dlat = .5 * self.sigma_D
                elon = np.linspace(lon - dlon, lon + dlon, 10)
                elat = np.linspace(lat - dlat, lat + dlat, 10)
                elon2, elat2 = np.meshgrid(elon, elat)
                std_tmp_values = sad.interp({self.name_var_sad['lon']:elon2.ravel(), 
                                            self.name_var_sad['lat']:elat2.ravel()}).values
                std_tmp = np.nanmean(std_tmp_values) if not np.all(np.isnan(std_tmp_values)) else 10**-10
                Q_tmp = std_tmp / ((self.facns*self.facnlt))**.5 
                Q.append(Q_tmp) 
            # Repeat for all time centers
            Q = np.tile(Q, len(self.ENST))
        else:
            Q = self.sigma_Q / ((self.facns*self.facnlt))**.5 * np.ones((self.nbasis))


        
        Xb = np.zeros_like(Q)
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print('gauss3d np.shape(Xb)',np.shape(Xb))
                print('gauss3d np.shape(ds[self.var_background].values)',np.shape(ds[self.var_background].values))
                print(f'Load background from file: {self.path_background}') 
                Xb = ds[self.var_background].values[:len(Xb)] 

        
        print(f'lambda={self.sigma_D:.1E}',
            f'nlocs={ENSLAT.size:.1E}',
            f'tdec={self.sigma_T:.1E}',
            f'ntime={ENST.size:.1E}',
            f'Q={np.mean(Q):.1E}')
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')

        # Compute basis components
        Gauss_xy = self._compute_component_space()
        Gauss_t, Nt = self._compute_component_time(time)
        self.Gauss_xy = Gauss_xy
        self.Gauss_t = Gauss_t
        self.Nt = Nt
        self.Nx = ENSLAT.size

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
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
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

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        return _u, _v
    
    def _ssh2uv_adj(self, adu, adv):

        """
        Adjoint of geostrophic velocity computation.
        """

        # _adssh lives on padded grid: (ny+1, nx+1)
        _adssh = np.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))

        _adssh[1:,:]  += -self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:-1,:] +=  self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:,1:]  +=  self.g / self.f_on_v * adv / self.dx_on_v
        _adssh[:,:-1] += -self.g / self.f_on_v * adv / self.dx_on_v

        # map padded grid back to physical ssh grid:
        # physical ssh[i,j] == _ssh[i+1,j+1]
        adssh = _adssh[1:,1:].copy()

        # contributions from the padded first row/col (mode='edge' duplicates edge values)
        # add the padded southern row (index 0) to physical southern row (adssh[0,:])
        adssh[0,:] += _adssh[0,1:]
        # add the padded western column (index 0) to physical western column (adssh[:,0])
        adssh[:,0] += _adssh[1:,0]
        # the padded corner (0,0) was duplicated as well — add it into adssh[0,0]
        adssh[0,0] += _adssh[0,0]

        return adssh
    
    def operg(self, t, X, State=None):

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

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v
    
        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = phi
            else:
                State[self.name_mod_var] += phi
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi

    def operg_transpose(self, t, adState):
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))
        
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros((self.nphys,))
            adState[self.name_mod_v] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState[self.name_mod_var].ravel()

        if self.compute_velocities:
            adparams += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])

        Gt = self.Gauss_t[t]
        ind0 = np.nonzero(Gt)[0]
        if ind0.size>0:
            Gt = Gt[ind0].reshape(self.Nt[t],self.Nx)
            adGtX = self.Gauss_xy.T.dot(adparams)
            adGtX = np.repeat(adGtX[np.newaxis,:],self.Nt[t],axis=0)
            adX[ind0] += (Gt*adGtX).ravel()

        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.

        return adX
 
class Basis_gauss3d_jax(Basis_gauss3d):

    def __init__(self,config, State, multi_mode=False):
        super().__init__(config, State,multi_mode=multi_mode)

        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)
        
    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)

        self.time = time
        self.vect_time = jnp.eye(time.size)

        self.zero_basis = jnp.zeros((self.nbasis,))
        self.zero_phys = jnp.zeros((self.nphys,))

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
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
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
    
    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = jnp.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        return _u, _v
    
    def get_Gt_value(self, t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gauss_t @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):

        """
            Project to physicial space
        """

        # Initialize phi
        phi = self.zero_phys.ravel()

        # Get Gt value
        Gt = self.get_Gt_value(t)
        GtX = Gt * X

        reshaped_GtX = GtX.reshape((-1, self.Nx))

        phi += self.Gauss_xy @ (reshaped_GtX.sum(axis=0))
        
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
        _, vjp_func = jax.vjp(operg_func, self.zero_basis)  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = phi
            else:
                State[self.name_mod_var] += phi
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = self.zero_phys
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = self.zero_phys
            adState[self.name_mod_v] = self.zero_phys

        adparams = adState[self.name_mod_var]
        adX = self._operg_reduced_jit(t, adparams)
        if self.compute_velocities:
            adparams += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])
        adX = self._operg_reduced_jit(t, adparams)
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
        
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
        self.facQ_aux_path = config.BASIS.facQ_aux_path
        self.l_largescale = config.BASIS.l_largescale
        self.facQ_largescale = config.BASIS.facQ_largescale
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

        # Gravity 
        self.g = 9.81

        # Compute geostrophic velocoties
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1,0),(1,0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])

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
        
        # FacQ_aux file (e.g. from background error)
        if config.BASIS.file_facQaux is not None:
            self.file_facQaux = config.BASIS.file_facQaux
            self.name_var_facQaux = config.BASIS.name_var_facQaux
        else:
            self.file_facQaux = None

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

        Mutltiple_basis_exp = True
        if Mutltiple_basis_exp:  
            L_MIN = 30
            L_MAX = 1000 

            TIME_MIN = time.min()
            TIME_MAX = time.max()
            LON_MIN = self.lon_min
            LON_MAX = self.lon_max
            LAT_MIN = self.lat_min
            LAT_MAX = self.lat_max
            if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

            # Ensemble of pseudo-frequencies for the wavelets (spatial)
            logff_all = np.arange(
                np.log(1./L_MIN),
                np.log(1. / L_MAX) - np.log(1 + self.facpsp / self.npsp),
                -np.log(1 + self.facpsp / self.npsp))[::-1]
            #print('_',logff_all)
            #print('A',self.lmax)
            #print('B',(logff_all>1/self.lmax))
            #print('C',self.lmin)
            #print('D',(logff_all<1/self.lmin))
            #print('E',(logff_all>1/self.lmax) & (logff_all<1/self.lmin))
            logff = logff_all[(logff_all>=np.log(1/self.lmax)) & (logff_all<=np.log(1/self.lmin))] 
            ff = np.exp(logff)
            ff = ff[1/ff<=self.lmax]
            dff = ff[1:] - ff[:-1]

        else: 
        
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

        # Auxiliary data
        aux = xr.open_dataset(self.file_aux,decode_times=False)
        if np.sign(aux['lon'].data.min())==-1 and self.lon_unit=='0_360':
            aux = aux.assign_coords({'lon':(('lon', aux['lon'].data % 360))})
        elif (np.sign(aux['lon'].data.min())>=0 or aux['lon'].data.max()>180) and self.lon_unit=='-180_180':
            aux = aux.assign_coords({'lon':(('lon', (aux['lon'].data + 180) % 360 - 180 ))})
        aux = aux.sortby(aux['lon'])    
        daTdec = aux['Tdec']
        daStd = aux['Std']

        # Auxiliary for Q
        if self.file_facQaux is not None:
            auxQ = xr.open_dataset(self.file_facQaux,decode_times=False)
            if np.sign(auxQ['lon'].data.min())==-1 and self.lon_unit=='0_360':
                auxQ = auxQ.assign_coords({'lon':(('lon', auxQ['lon'].data % 360))})
            elif (np.sign(auxQ['lon'].data.min())>=0 or auxQ['lon'].data.max()>180) and self.lon_unit=='-180_180':
                auxQ = auxQ.assign_coords({'lon':(('lon', (auxQ['lon'].data + 180) % 360 - 180 ))})
            auxQ = auxQ.sortby(auxQ['lon'])    
            daFacQ = auxQ[self.name_var_facQaux['var']]

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = [None]*nf #  Ensemble of times of the center of each wavelets
        tdec = [None]*nf # Ensemble of equivalent decorrelation times. Used to define enst.
        norm_fact = [None]*nf # integral of the time component (for normalization)
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        #DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int64') # Nomber of spatial wavelet locations for a given frequency

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
        facQ_largescale = self.facQ_largescale 
        l_largescale = self.l_largescale 

        std = []
        facQaux = []
        facQaux = []
        for iff in range(nf):
            std.append([])
            std[iff] = []
            if self.file_facQaux is not None:
                facQaux.append([])
                facQaux[iff] = []
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

                if self.file_facQaux is not None:
                    facQaux_tmp_values = daFacQ.interp({self.name_var_facQaux['wavenumber']:ff[iff], 
                                                        self.name_var_facQaux['lon']:elon2.ravel(), 
                                                        self.name_var_facQaux['lat']:elat2.ravel()}).values
                    facQaux_tmp = np.nanmean(facQaux_tmp_values) if not np.all(np.isnan(facQaux_tmp_values)) else 1.0
                    facQaux[iff].append(facQaux_tmp)

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

                    Q_tmp *= facQ   # Multiply after NaN check

                    # Include facQaux if available
                    if self.file_facQaux is not None:
                        Q_tmp *= np.sqrt(facQaux[iff][P])

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

        Xb = np.zeros_like(Q)
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print('bmaux np.shape(Xb)',np.shape(Xb))
                print('bmaux np.shape(ds[self.var_background].values)',np.shape(ds[self.var_background].values))
                print(f'Load background from file: {self.path_background}') 
                Xb = ds[self.var_background].values[-len(Xb):] 

            

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

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        return _u, _v
    
    def _ssh2uv_adj(self, adu, adv):

        """
        Adjoint of geostrophic velocity computation.
        """

        # _adssh lives on padded grid: (ny+1, nx+1)
        _adssh = np.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))

        _adssh[1:,:]  += -self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:-1,:] +=  self.g / self.f_on_u * adu / self.dy_on_u
        _adssh[:,1:]  +=  self.g / self.f_on_v * adv / self.dx_on_v
        _adssh[:,:-1] += -self.g / self.f_on_v * adv / self.dx_on_v

        # map padded grid back to physical ssh grid:
        # physical ssh[i,j] == _ssh[i+1,j+1]
        adssh = _adssh[1:,1:].copy()

        # contributions from the padded first row/col (mode='edge' duplicates edge values)
        # add the padded southern row (index 0) to physical southern row (adssh[0,:])
        adssh[0,:] += _adssh[0,1:]
        # add the padded western column (index 0) to physical western column (adssh[:,0])
        adssh[:,0] += _adssh[1:,0]
        # the padded corner (0,0) was duplicated as well — add it into adssh[0,0]
        adssh[0,0] += _adssh[0,0]

        return adssh
        
    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        ssh = np.zeros(self.shape_phys).ravel()
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                GtXf = GtXf[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                phi += self.Gx[iff].dot(GtXf.sum(axis=0))
        ssh = ssh.reshape(self.shape_phys)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = ssh
            else:
                State[self.name_mod_var] += ssh
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros((self.nphys,))
            adState[self.name_mod_v] = np.zeros((self.nphys,))
            
        adX = np.zeros(self.nbasis)

        adssh = adState[self.name_mod_var]

        if self.compute_velocities:
            adssh += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])

        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                Gt = Gt[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adssh.ravel())
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][indNoNan] += (Gt*adGtXf).ravel()
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
        
        return adX

class Basis_bmaux_jax(Basis_bmaux):

    def __init__(self, config, State, multi_mode=False):
        super().__init__(config, State, multi_mode=multi_mode)

        # JIT 
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)


    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)
        self.time = time
        self.vect_time = jnp.eye(time.size)

        self.zero_basis = jnp.zeros((self.nbasis,))
        self.zero_phys = jnp.zeros((self.nphys,))

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

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = jnp.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        return _u, _v

    def get_Gt_value(self, t, iff):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt[iff] @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):
        """
            Project to physicial space
        """

        # Initialize phi
        phi = self.zero_phys.ravel()

        for iff in range(self.nf):

            Gt = self.get_Gt_value(t,iff)
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = Gt * Xf

            # # Use shape-safe slicing instead of boolean indexing
            Nx_val = self.Nx[iff]

            # # Dynamically reshape the sliced array
            reshaped_GtXf = GtXf.reshape((-1, Nx_val))  # Ensure reshaping works dynamically

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
        _, vjp_func = jax.vjp(operg_func, self.zero_basis)  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        ssh = self._operg_jit(t, X)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v
        
        # Update State
        if State is not None:
            if not self.multi_mode:
                State[self.name_mod_var] = ssh
            else:
                State[self.name_mod_var] += ssh
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = self.zero_phys
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = self.zero_phys
            adState[self.name_mod_v] = self.zero_phys

        adssh = adState[self.name_mod_var]
        if self.compute_velocities:
            adssh += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])
        adX = self._operg_reduced_jit(t, adssh)

        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
    
        return adX

###############################################################################
#                     Wavelet - multiple scale                                #
###############################################################################
      
class Basis_wavelet3d:
   
    def __init__(self, config, State, multi_mode=False):

        self.km2deg = 1./110
        
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

        # For multi-basis
        self.multi_mode = multi_mode

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
        for iff in range(self.nfs):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.fs_wavebounds[iff]:self.fs_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
        
        return adX
    
class Basis_wavelet3d_jax(Basis_wavelet3d):
   
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

            #Gx[iff] = csc_matrix((data, indices, indptr), shape=(self.nphys, nwaves))
            Gx[iff] = sparse.CSC((data, indices, indptr), shape=(self.nphys, nwaves))

        return Gx, Nx
    
    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency
        
        for iff in range(self.nf):
            nbasis_f = self.iff_wavebounds[iff+1] - self.iff_wavebounds[iff]
            Gt_np = np.zeros((time.size, nbasis_f))
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
    
    def __compute_component_time(self, time):

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
            State[self.name_mod_var] = phi
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
        for iff in range(self.nfs):
            Gt = +self.Gt[t][iff]
            ind0 = np.nonzero(Gt)[0]
            if ind0.size>0:
                Gt = Gt[ind0].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.fs_wavebounds[iff]:self.fs_wavebounds[iff+1]][ind0] += (Gt*adGtXf).ravel()
        
        adState[self.name_mod_var] *= 0.
        
        return adX
    
###############################################################################
#                            Internal-tides                                   #
###############################################################################   

class Basis_hbc_jax: 

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
        self.name_params = config.BASIS.name_params 

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

    def set_basis(self,time,return_q=False,**kwargs):

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
        for name in self.name_params : 

            # - X height boundary conditions - #
            if name == "hbcx":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_bc_gauss_hbcx(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            
            # - Y height boundary conditions - #
            if name == "hbcy": 
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
                                Q[self.slice_params[name]][slicew]=self.sigma_B_bc[iw]/(self.facnlt*self.facns)**.5
                        else:
                            # Not the right number of frequency prescribed in the config file 
                            # --> we use only the first one
                            Q[self.slice_params[name]]=self.sigma_B_bc[0]/(self.facnlt*self.facns)**.5
                    else:
                        # Q[self.slice_params[name]]=self.sigma_B_bc
                        Q[self.slice_params[name]]=self.sigma_B_bc/(self.facnlt*self.facns)**.5

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
        self.Gxy["hbcN"] = sparse.CSR.fromdense(jnp.array(bc_N_gauss.T)) # For North boundary

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
            for name in self.name_params:
                # - Height boundary conditions hbcx - #
                if name == "hbcx" : 
                    State['hbcx'] = jnp.concatenate((jnp.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                            jnp.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                # - Height boundary conditions hbcy - #
                elif name == "hbcy" : 
                    State['hbcy'] = jnp.concatenate((jnp.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1),
                                                        jnp.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1)),axis=1)
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
            for name in self.name_params:
                if name == "hbcx" : 
                    # adparams["hbcS"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcS"])
                    # adparams["hbcN"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcN"])
                    adparams[self.slice_params_phys["hbcS"]] = adState[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcN"]] = adState[name][:,1,:,:,:].flatten()
                elif name == "hbcy" : 
                    # adparams["hbcE"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcE"])
                    # adparams["hbcW"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcW"])
                    adparams[self.slice_params_phys["hbcW"]] = adState[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcE"]] = adState[name][:,1,:,:,:].flatten()
                # else :
                #     param[name] = adState.params[name].reshape(self.shape_params_phys[name])
        # adparams = adparams.flatten()
        # adparams = adState.getparams(self.name_params,vect=True)

        adX = self._operg_reduced_jit(t, adparams)
        
        for _param in self.name_params : 
            adState[_param] *= 0.
        
        return adX

class Basis_hbc_cst_jax: 

    def __init__(self,config, State):


        # Grid specs
        self.ny = State.ny
        self.nx = State.nx


        # Tidal frequencies 
        self.Nwaves = config.BASIS.Nwaves # Number of tidal components


        # Number of angles (computed from the normal of the border) of incoming waves
        if config.BASIS.Ntheta>0: 
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°

        self.sigma_B_bc = config.BASIS.sigma_B_bc # Covariance sigma for hbc parameter

        # JIT
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

    def set_basis(self,time,return_q=False,**kwargs):

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
        # Shapes of the hbcx parameters in the reduced space.
        self.shapehbcx = [
            self.Nwaves,    # - Number of tidal frequency components 
            2,              # - South/North 
            2,              # - Number of controlled components (cos & sin)
            self.Ntheta     # - Number of angles
            ]
        
        # Shapes of the hbcy parameters in the reduced space.
        self.shapehbcy = [
            self.Nwaves,    # - Number of tidal frequency components 
            2,              # - West/East
            2,              # - Number of controlled components (cos & sin)
            self.Ntheta     # - Number of angles
            ]
        
        self.nbasis = np.prod(self.shapehbcx) + np.prod(self.shapehbcy)
        self.slice_params_hbcx = slice(0, np.prod(self.shapehbcx))
        self.slice_params_hbcy = slice(np.prod(self.shapehbcx), self.nbasis)
        
        
        # - Shapes of the hbcy parameters in the physical space.
        self.shapehbcx_phys = [
            self.Nwaves,    # - Number of tidal frequency components 
            2,              # South/North
            2,              # - Number of controlled components (cos & sin)
            self.Ntheta,    # - Number of angles
            self.nx         # - Number of gridpoints along x axis
            ]
        
        # Shapes of the hbcx parameters in the physical space.
        self.shapehbcy_phys = [
            self.Nwaves,    # - Number of tidal frequency components 
            2,              # West/East
            2,              # - Number of controlled components (cos & sin)
            self.Ntheta,    # - Number of angles
            self.ny         # - Number of gridpoints along x axis
            ]
        
        self.nphys = np.prod(self.shapehbcx_phys) + np.prod(self.shapehbcy_phys)
        self.slice_params_phys_hbcx = slice(0, np.prod(self.shapehbcx_phys))
        self.slice_params_phys_hbcy = slice(np.prod(self.shapehbcx_phys), self.nphys)

        self.ones_nx = jnp.ones((1,1,1,1,self.nx))
        self.ones_ny = jnp.ones((1,1,1,1,self.ny))

        print(f'reduced order: {self.nphys} --> {self.nbasis}\nreduced factor: {int(self.nphys/self.nbasis)}')

        #########################################
        ### COMPUTING THE COVARIANCE MATRIX Q ###
        #########################################        

        if return_q :
            if self.sigma_B_bc is not None:
                Q = self.sigma_B_bc * np.ones((self.nbasis,)) # Initializing
            else:
                Q = np.ones((self.nbasis,)) # Initializing
                
            Xb = np.zeros_like(Q)

            return Xb, Q

    def _operg(self,t,X):

        """
        Perform the basis projection operation for a given time and parameter vector.
        """


        # Variable to return  
        phi = jnp.zeros((self.nphys,))

        X_hbcx = X[self.slice_params_hbcx].reshape(self.shapehbcx)
        X_hbcy = X[self.slice_params_hbcy].reshape(self.shapehbcy)

        phi = phi.at[self.slice_params_phys_hbcx].set(
            (X_hbcx[:,:, :, :, None] * self.ones_nx).reshape(-1)
        )

        phi = phi.at[self.slice_params_phys_hbcy].set(
            (X_hbcy[:, :, :, :, None] * self.ones_ny).reshape(-1)
        )

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
            # - Height boundary conditions hbcx - #
            State['hbcx'] = phi[self.slice_params_phys_hbcx].reshape(self.shapehbcx_phys)
            # - Height boundary conditions hbcy - #
            State['hbcy'] = phi[self.slice_params_phys_hbcy].reshape(self.shapehbcy_phys)
        else:
            return phi
    
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        adparams = jnp.concatenate([adState['hbcx'].reshape(-1), adState['hbcy'].reshape(-1)], axis=0) 
        adX = self._operg_reduced_jit(t, adparams)
    
        adState['hbcx'] *= 0.
        adState['hbcy'] *= 0.
        
        return adX

class Basis_offset:

    def __init__(self,config, State, multi_mode=False):
        
        self.name_mod_var = config.BASIS.name_mod_var
        self.shape_phys = State.params[self.name_mod_var].shape
        self.nphys = np.prod(self.shape_phys)
        self.ny = State.ny
        self.nx = State.nx
        self.sigma_B = config.BASIS.sigma_B
        
        if self.sigma_B == None : 
            print("Warning, please prescribe sigma_B for Basis Offset") 
        
        self.multi_mode = multi_mode
    
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
        adparams = adState[self.name_mod_var]

        adX = [np.sum(adparams)]
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
        
        return adX

class Basis_offset_jax(Basis_offset):
   
    def __init__(self,config, State, multi_mode=False):

        super().__init__(config, State,multi_mode=multi_mode)
    
    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """

        phi = X*jnp.ones(self.shape_phys)

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
            adState[self.name_mod_var] = jnp.zeros((self.nphys,))
        adparams = adState[self.name_mod_var]

        adX = jnp.expand_dims(jnp.sum(adparams), axis=0)
        
        if not self.multi_mode:
            adState[self.name_mod_var] *= 0.
        
        return adX
    
###############################################################################
#                              Multi-Basis                                    #
###############################################################################      

class Basis_multi:

    def __init__(self,config,State,verbose=True):

        self.Basis = []
        _config = config.copy()

        self.jax = True
        self.name_mod_var = []
        for _BASIS in config.BASIS:
            _config.BASIS = config.BASIS[_BASIS]

            self.Basis.append(Basis(_config,State,verbose=verbose, multi_mode=True))
            if 'name_mod_var' in _config.BASIS and _config.BASIS.name_mod_var is not None and _config.BASIS.name_mod_var not in self.name_mod_var:
                self.name_mod_var.append(_config.BASIS.name_mod_var)
                if 'compute_velocities' in _config.BASIS and _config.BASIS.compute_velocities:
                    if 'name_mod_u' in _config.BASIS and _config.BASIS.name_mod_u is not None and _config.BASIS.name_mod_u not in self.name_mod_var:
                        self.name_mod_var.append(_config.BASIS.name_mod_u)
                    if 'name_mod_v' in _config.BASIS and _config.BASIS.name_mod_v is not None and _config.BASIS.name_mod_v not in self.name_mod_var:
                        self.name_mod_var.append(_config.BASIS.name_mod_v)

            if '_JAX' not in _config.BASIS.super:
                self.jax = False

        if self.jax:
            print('Basis_multi: Full JAX mode')
        
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

        if State is None:
            if self.jax:
                phi = jnp.array([])
            else:
                phi = np.array([])

        if State is not None:
            for name_mod_var in self.name_mod_var:
                State[name_mod_var] *= 0.

        for i,B in enumerate(self.Basis):
            _X = X[self.slice_basis[i]]
            _phi = B.operg(t, _X, State=State)
            if State is None:
                if self.jax:
                    phi = jnp.append(phi, _phi)
                else:
                    phi = np.append(phi, _phi)
        
        if State is None:
            return phi


    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        for i,B in enumerate(self.Basis):
            _adX = B.operg_transpose(t, adState=adState)
            if i==0:
                adX = +_adX
            else:
                if self.jax:
                    adX = jnp.concatenate((adX, _adX))
                else:
                    adX = np.concatenate((adX, _adX))
        
        for name_mod_var in self.name_mod_var:
            adState[name_mod_var] *= 0.

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

