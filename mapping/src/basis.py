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
import matplotlib.pylab as plt
import xarray as xr
import scipy
from scipy.integrate import quad

def Basis(config, State, *args, **kwargs):
    """
    NAME
        Basis

    DESCRIPTION
        Main function calling subfunctions for specific Reduced Basis functions
    """
    
    if config.BASIS is None:
        return 
    
    elif config.BASIS.super is None:
        return Basis_multi(config, State)

    else:
        
        print(config.BASIS)

        if config.BASIS.super=='BASIS_BM':
            return Basis_BM(config, State)

        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_BMaux(config)
        
        elif config.BASIS.super=='BASIS_LS':
            return BASIS_LS(config, State)
        
        elif config.BASIS.super=='BASIS_IT':
            return Basis_IT(config, State)
    
        else:
            sys.exit(config.BASIS.super + ' not implemented yet')

class Basis_IT:
   
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
        
        if config.BASIS.Ntheta>0:
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°
            
        self.w_it = config.BASIS.w_it

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
        
        self.nbcS = len(self.w_it) * 2 * self.Ntheta * ENST_bc.size * bc_S_gauss.shape[0]
        self.nbcN = len(self.w_it) * 2 * self.Ntheta * ENST_bc.size * bc_N_gauss.shape[0]
        self.nbcE = len(self.w_it) * 2 * self.Ntheta * ENST_bc.size * bc_E_gauss.shape[0]
        self.nbcW = len(self.w_it) * 2 * self.Ntheta * ENST_bc.size * bc_W_gauss.shape[0]
        self.nbc = self.nbcS + self.nbcN + self.nbcE + self.nbcW
        print('nbc:',self.nbc)
        
        
        self.shapehbcS = [len(self.w_it), 2, self.Ntheta, ENST_bc.size, bc_S_gauss.shape[0]]
        self.shapehbcN = [len(self.w_it), 2, self.Ntheta, ENST_bc.size, bc_N_gauss.shape[0]]
        self.shapehbcE = [len(self.w_it), 2, self.Ntheta, ENST_bc.size, bc_E_gauss.shape[0]]
        self.shapehbcW = [len(self.w_it), 2, self.Ntheta, ENST_bc.size, bc_W_gauss.shape[0]]
        
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
        self.shapehbcx_phys = [len(self.w_it), # tide frequencies
                          2, # North/South
                          2, # cos/sin
                          self.Ntheta, # Angles
                          self.nx # NX
                          ]
        self.shapehbcy_phys = [len(self.w_it), # tide frequencies
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
                    if len(self.sigma_B_bc)==len(self.w_it):
                        # Different background values for each frequency
                        nw = self.nbc//len(self.w_it)
                        for iw in range(len(self.w_it)):
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
            
            return Q
        
        
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
        
class Basis_BM:
   
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
        self.save_wave_basis = config.BASIS.save_wave_basis
        self.wavelet_init = config.BASIS.wavelet_init
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

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path
        self.indx = {}
        self.facG = {}

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
            tt = np.linspace(-tdec[iff],tdec[iff])
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/tdec[iff])*(tt[i+1]-tt[i])
            norm_fact[iff] = tmp.max()


            nwave += ntheta*2*len(enst[iff])*NP[iff]
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)     
         
        Qi = np.array([])
        Qt = np.array([]) # Initial state      

        for iff in range(nf):
            _nwavei = 2*ntheta*NP[iff] # just in space
            _nwavet = 2*len(enst[iff])*ntheta*NP[iff]
            if 1/ff[iff]>self.lmeso:
                if self.wavelet_init:
                    Qi = np.concatenate((Qi,self.Qmax/(self.facns*self.facnlt)**.5*np.ones((_nwavei,))))
                Qt = np.concatenate((Qt,self.Qmax/(self.facns*self.facnlt)**.5*np.ones((_nwavet,))))
            else:
                if self.wavelet_init:
                    Qi = np.concatenate((Qi,self.Qmax/(self.facns*self.facnlt)**.5 * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavei,)))) 
                Qt = np.concatenate((Qt,self.Qmax/(self.facns*self.facnlt)**.5 * self.lmeso**self.slopQ * ff[iff]**self.slopQ*np.ones((_nwavet,)))) 
                
            print(f'lambda={1/ff[iff]:.1E}',
                  f'nlocs={NP[iff]:.1E}',
                  f'tdec={tdec[iff]:.1E}',
                  f'Q={Qt[-1]:.1E}')
        
        Q = np.concatenate((Qi,Qt))

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
        if not self.save_wave_basis:
            self.facG = {}
            self.indx = {}
        
        for t in time:
            facGt = {}
            indxt = {}

            for iff in range(self.nf):
                for P in range(self.NP[iff]):
                    
                    facGt[(iff,P)] = {}
                    
                    # Obs selection around point P
                    iobs = np.where(
                        (np.abs((np.mod(self.lon1d - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                        (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                        )[0]
                    xx = (np.mod(self.lon1d[iobs] - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                    yy = (self.lat1d[iobs] - self.ENSLAT[iff][P]) / self.km2deg

                    facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff])
                    
                    indxt[(iff,P)] = iobs
                    
                    if iobs.shape[0] > 0:
                        # Initial State
                        if t==0 and self.wavelet_init:
                            facGt[(iff,P)][-1] = [None,]*self.ntheta # -1 stands for initial state
                            for itheta in range(self.ntheta):
                                facGt[(iff,P)][-1][itheta] = [[],[]]
                                kx = self.k[iff] * np.cos(self.theta[itheta])
                                ky = self.k[iff] * np.sin(self.theta[itheta])
                                facGt[(iff,P)][-1][itheta][0] = np.sqrt(2)* facs * np.cos(kx*(xx)+ky*(yy))
                                facGt[(iff,P)][-1][itheta][1] = np.sqrt(2)* facs * np.cos(kx*(xx)+ky*(yy)-np.pi/2)

                        # Time spread wavelets
                        enstloc = self.enst[iff]
                        for it in range(len(enstloc)):
                            dt = t - enstloc[it]
                            try:
                                if abs(dt) < self.tdec[iff]:
                                    fact = self.window(dt / self.tdec[iff]) 
                                    fact /= self.norm_fact[iff]
                                        
                                    facGt[(iff,P)][it] = [None,]*self.ntheta
                                    for itheta in range(self.ntheta):
                                        facGt[(iff,P)][it][itheta] = [[],[]]
                                        kx = self.k[iff] * np.cos(self.theta[itheta])
                                        ky = self.k[iff] * np.sin(self.theta[itheta])
                                        facGt[(iff,P)][it][itheta][0] = np.sqrt(2)* fact * facs * np.cos(kx*(xx)+ky*(yy))
                                        facGt[(iff,P)][it][itheta][1] = np.sqrt(2)* fact * facs * np.cos(kx*(xx)+ky*(yy)-np.pi/2)
                            except:
                                print(f'Warning: an error occured at t={t}, iff={iff}, P={P}, enstloc={enstloc[it]}')

            if self.save_wave_basis:
                name_facG = os.path.join(self.path_save_tmp,f'facG_{t}.pic')
                name_indx = os.path.join(self.path_save_tmp,'indx.pic')
                if not os.path.exists(name_facG):
                    with open(name_facG, 'wb') as f:
                        pickle.dump(facGt,f)  
                if not os.path.exists(name_indx):
                    with open(name_indx, 'wb') as f:
                        pickle.dump(indxt,f)     
            else:
                self.facG[t] = facGt
                self.indx = indxt        
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        if return_q:
            return Q
    
    def operg(self, t, X, transpose=False,State=None):
        
        """
            Project to physicial space
        """

        # Load basis components 
        if self.save_wave_basis:
            # Offline
            name_facG = os.path.join(self.path_save_tmp,f'facG_{t}.pic')
            name_indx = os.path.join(self.path_save_tmp,'indx.pic')
            if os.path.exists(name_facG) and os.path.exists(name_indx):
                with open(name_facG, 'rb') as f:
                    facG = pickle.load(f)
                with open(name_indx, 'rb') as f:
                    indx = pickle.load(f)
        else: 
            # Inline
            facG = self.facG[t]
            indx = self.indx

        # Projection
        if transpose:
            X = X.flatten()
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.nphys,))
        
        iwave = 0
        for iff in range(self.nf):
            enstloc = self.enst[iff]
            for P in range(self.NP[iff]):
                iobs = indx[(iff,P)]
                if iobs.shape[0] > 0:
                    # Initial State
                    if t==0 and self.wavelet_init:
                        for itheta in range(self.ntheta):
                            for iphase in range(2):
                                if transpose:
                                    phi[iwave] = np.sum(X[iobs] * facG[(iff,P)][-1][itheta][iphase])
                                else:
                                    phi[iobs] += X[iwave] * facG[(iff,P)][-1][itheta][iphase]
                                iwave += 1
                    elif self.wavelet_init:
                        iwave += 2*self.ntheta

                    # Time spread wavelets
                    for it in range(len(enstloc)):
                        if it not in facG[(iff,P)]:
                            iwave += 2*self.ntheta
                        else:
                            for itheta in range(self.ntheta):
                                for iphase in range(2):
                                    if transpose:
                                        phi[iwave] = np.sum(X[iobs] * facG[(iff,P)][it][itheta][iphase])
                                    else:
                                        phi[iobs] += X[iwave] * facG[(iff,P)][it][itheta][iphase]
                                    iwave += 1
        
        if iwave!=self.nbasis:
                print(f'Warning: not the right number of wavelet: {iwave}≠{self.nbasis}')
        
        # Reshaping
        if not transpose:
            phi = phi.reshape(self.shape_phys)

        if State is not None:
            if t==0:
                if self.wavelet_init:
                    State.setvar(phi,self.name_mod_var)
                    State.params[self.name_mod_var] = np.zeros(self.shape_phys)
                else:
                    State.params[self.name_mod_var] = phi
            else:
                State.params[self.name_mod_var] = phi
        else:
            return phi
        

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        if t==0:
            if self.wavelet_init:
                adX = self.operg(t, adState.getvar(self.name_mod_var), transpose=True)
            else:
                adX = self.operg(t, adState.params[self.name_mod_var], transpose=True)
        else:
            if adState.params[self.name_mod_var] is None:
                adState.params[self.name_mod_var] = np.zeros((self.nphys,))
            adX = self.operg(t, adState.params[self.name_mod_var], transpose=True)
        
        adState.params[self.name_mod_var] *= 0.
        
        return adX
        
class Basis_BMaux:
   
    def __init__(self,config):

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
        self.wavelet_init = config.BASIS.wavelet_init
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

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path
        self.indx = {}
        self.facG = {}

    def set_basis(self,time,lon,lat,return_q=False):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = lon.min()
        LON_MAX = lon.max()
        LAT_MIN = lat.min()
        LAT_MAX = lat.max()
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.
        lon1d = lon.flatten()
        lat1d = lat.flatten()

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
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

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
        lonmax = LON_MAX
        if (LON_MAX<LON_MIN): lonmax = LON_MAX+360.
            
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
                        lonmax+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg*finterpdist(ENSLAT1[I]),
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
        self.iff_wavebounds[iff+1] = iwave +1
            
        nwave = iwave+1
        Q = Q[:nwave]
        
        
        self.lon1d = lon1d
        self.lat1d = lat1d
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nbasis=nwave
        self.nphys= lon1d.size
        self.shape_phys = lon.shape
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff
        self.tdec=tdec
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        if return_q:
            return Q
        

    def operg(self, X, t, transpose=False,State=None):
        
        """
            Project to physicial space
        """

        
            
        compute_basis = False
        # Load basis components if already computed
        if self.save_wave_basis:
            # Offline
            name_facG = os.path.join(self.path_save_tmp,f'facG_{t}.pic')
            name_indx = os.path.join(self.path_save_tmp,'indx.pic')
            if os.path.exists(name_facG) and os.path.exists(name_indx):
                with open(name_facG, 'rb') as f:
                    facG = pickle.load(f)
                with open(name_indx, 'rb') as f:
                    indx = pickle.load(f)
            else: 
                compute_basis = True
        
        elif (t in self.facG) and (self.indx!={}):
            # Inline
            facG = self.facG[t]
            indx = self.indx
        else: 
            compute_basis = True
            
        # Compute basis components
        if compute_basis:

            facG = {}
            indx = {}
            
            for iff in range(self.nf):
                for P in range(self.NP[iff]):

                    if self.wavetest[iff][P]:
                    
                        facG[(iff,P)] = {}
                        
                        # Obs selection around point P
                        iobs = np.where(
                            (np.abs((np.mod(self.lon1d - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                            (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                            )[0]
                        xx = (np.mod(self.lon1d[iobs] - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                        yy = (self.lat1d[iobs] - self.ENSLAT[iff][P]) / self.km2deg
    
                        
                        # facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd
                        facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff])
                        
                        indx[(iff,P)] = iobs

                        enstloc = self.enst[iff][P]
                        
                        if iobs.shape[0] > 0:
                            for it in range(len(enstloc)):
                                dt = t - enstloc[it]
                                try:
                                    if abs(dt) < self.tdec[iff][P]:
                                        if t==0 and self.wavelet_init:
                                            fact = mywindow(dt / self.tdec[iff][P])
                                        else:
                                            if self.flux:
                                                fact = mywindow_flux(dt / self.tdec[iff][P])
                                            else:
                                                fact = mywindow(dt / self.tdec[iff][P])
                                            tt = np.linspace(-self.tdec[iff][P],self.tdec[iff][P])
                                            I =  np.sum(mywindow(tt/self.tdec[iff][P]))*(tt[1]-tt[0])
                                            fact /= I 
                                            
                                        facG[(iff,P)][it] = [None,]*self.ntheta
                                        for itheta in range(self.ntheta):
                                            facG[(iff,P)][it][itheta] = [[],[]]
                                            kx = self.k[iff] * np.cos(self.theta[itheta])
                                            ky = self.k[iff] * np.sin(self.theta[itheta])
                                            facG[(iff,P)][it][itheta][0] = np.sqrt(2)* fact * facs * np.cos(kx*(xx)+ky*(yy))
                                            facG[(iff,P)][it][itheta][1] = np.sqrt(2)* fact * facs * np.cos(kx*(xx)+ky*(yy)-np.pi/2)
                                except:
                                    print(f'Warning: an error occured at t={t}, iff={iff}, P={P}, enstloc={enstloc[it]}')

            if self.save_wave_basis:
                if not os.path.exists(name_facG):
                    with open(name_facG, 'wb') as f:
                        pickle.dump(facG,f)  
                if not os.path.exists(name_indx):
                    with open(name_indx, 'wb') as f:
                        pickle.dump(indx,f)     
            else:
                self.facG[t] = facG
                self.indx = indx
        
        # Projection
        if transpose:
            # Flattening
            X = X.flatten()
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.lon1d.size,))
        
        iwave = 0
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                if self.wavetest[iff][P]:
                    enstloc = self.enst[iff][P]
                    iobs = indx[(iff,P)]
                    #if iobs.shape[0] > 0:
                    for it in range(len(enstloc)):
            
                        if it not in facG[(iff,P)]:
                            iwave += 2*self.ntheta
                        else:
                            for itheta in range(self.ntheta):
                                for iphase in range(2):
                                    if transpose:
                                        phi[iwave] = np.sum(X[iobs] * facG[(iff,P)][it][itheta][iphase])
                                    else:
                                        phi[iobs] += X[iwave] * facG[(iff,P)][it][itheta][iphase]
                            
                                    iwave += 1
        
        if iwave!=self.nbasis:
            print(f'Warning: not the right number of wavelet: {iwave}≠{self.nbasis}')

        # Reshaping
        if not transpose:
            phi = phi.reshape(self.shape_phys)

        if State is not None:
            if t==0:
                if self.wavelet_init:
                    State.setvar(phi,self.name_mod_var)
                    State.params[self.name_mod_var] = np.zeros(self.shape_phys)
                else:
                    State.params[self.name_mod_var] = phi
            else:
                State.params[self.name_mod_var] = phi
        else:
            return phi
        
    
    def operg_transpose(self, adState, t):
        
        """
            Project to reduced space
        """
        
        if t==0:
            if self.wavelet_init:
                adX = self.operg(adState.getvar(self.name_mod_var), t, transpose=True)
            else:
                adX = self.operg(adState.params[self.name_mod_var], t, transpose=True)
        else:
            if adState.params[self.name_mod_var] is None:
                adState.params[self.name_mod_var] = np.zeros((self.nphys,))
            adX = self.operg(adState.params[self.name_mod_var], t, transpose=True)

        adState.params[self.name_mod_var] *= 0.
        
        return adX

class BASIS_LS:
   
    def __init__(self,config,State):

        self.km2deg=1./110
    
        self.name_mod_var = config.BASIS.name_mod_var
        self.facnls = config.BASIS.facnls
        self.facnlt = config.BASIS.facnlt
        self.lambda_lw = config.BASIS.lambda_lw
        self.tdec_lw = config.BASIS.tdec_lw
        self.fcor = config.BASIS.fcor
        self.std_lw = config.BASIS.std_lw

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
        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nbasis=nwave
        self.tdec=tdec
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
        
        if return_q:
            return Q
    
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


###############################################################################
#                              Multi-Basis                                    #
###############################################################################      

class Basis_multi:

    def __init__(self,config,State):

        self.Basis = []
        _config = config.copy()

        for _BASIS in config.BASIS:
            _config.BASIS = config.BASIS[_BASIS]
            self.Basis.append(Basis(_config,State))

    def set_basis(self,time,return_q=False):

        self.nbasis = 0
        self.slice_basis = []

        if return_q:
            Q = np.array([])

        for B in self.Basis:
            _Q = B.set_basis(time,return_q=return_q)
            self.slice_basis.append(slice(self.nbasis,self.nbasis+B.nbasis))
            self.nbasis += B.nbasis
            
            if return_q:
                Q = np.concatenate((Q,_Q))
        
        if return_q:
            return Q

    def operg(self, t, X, transpose=False, State=None):
        
        """
            Project to physicial space
        """

        phi = np.array([])

        for i,B in enumerate(self.Basis):
            _X = X[self.slice_basis[i]]
            phi = np.append(phi,B.operg(t, _X, State=State, transpose=transpose))
        
        if State is None:
            return phi


    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        adX = np.array([])
        for B in self.Basis:
            adX = np.concatenate((adX,B.operg_transpose(t, adState)))

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