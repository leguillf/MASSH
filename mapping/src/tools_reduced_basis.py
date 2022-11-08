#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:24:24 2021

@author: leguillou
"""
import os
from datetime import datetime
import numpy as np
import logging
import pickle 
import matplotlib.pylab as plt
import xarray as xr
import scipy

class RedBasis_IT:
   
    def __init__(self,config):

        self.km2deg =1./110
    
        self.facns = config.facgauss
        self.facnlt = config.facgauss
        self.D_He = config.D_He
        self.T_He = config.T_He
        self.D_bc = config.D_bc
        self.T_bc = config.T_bc
        
        self.sigma_B_He = config.sigma_B_He
        self.sigma_B_bc = config.sigma_B_bc
        
        
        if config.Ntheta>0:
            self.Ntheta = 2*(config.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°
            
        self.w_it = config.w_igws
    
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
        He_xy_gauss = np.zeros((ENSLAT_He.size,lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(ENSLAT_He,ENSLON_He)):
            iobs = np.where(
                    (np.abs((np.mod(lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.D_He) &
                    (np.abs((lat1d - lat0) / self.km2deg) <= self.D_He)
                    )[0]
            xx = (np.mod(lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (lat1d[iobs] - lat0) / self.km2deg
            
            He_xy_gauss[i,iobs] = mywindow(xx / self.D_He) * mywindow(yy / self.D_He)

        He_xy_gauss = He_xy_gauss.reshape((ENSLAT_He.size,lon.shape[0],lon.shape[1]))
        
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
        bc_S_gauss = np.zeros((ENSLON_S.size,lon.shape[1]))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(lon[0,:] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(lon[0,:][iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # North
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        bc_N_gauss = np.zeros((ENSLON_N.size,lon.shape[1]))
        
        # East
        bc_E_gauss = np.zeros((ENSLAT.size,lon.shape[0]))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((lat[:,0] - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (lat[:,0][iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # West 
        bc_W_gauss = np.zeros((ENSLAT.size,lon.shape[0]))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((lat[:,-1] - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (lat[:,-1][iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(lon[-1,:] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(lon[-1,:][iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
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
        self.shapeHe_phys = lon.shape
        self.shapehbcx_phys = [len(self.w_it), # tide frequencies
                          2, # North/South
                          2, # cos/sin
                          self.Ntheta, # Angles
                          lon.shape[1] # NX
                          ]
        self.shapehbcy_phys = [len(self.w_it), # tide frequencies
                          2, # North/South
                          2, # cos/sin
                          self.Ntheta, # Angles
                          lon.shape[0] # NY
                          ]
        self.nphys = np.prod(self.shapeHe_phys) + np.prod(self.shapehbcx_phys) + np.prod(self.shapehbcy_phys)
        self.sliceHe_phys = slice(0,np.prod(self.shapeHe_phys))
        self.slicehbcx_phys = slice(np.prod(self.shapeHe_phys),
                               np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys))
        self.slicehbcy_phys = slice(np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys),
                               np.prod(self.shapeHe_phys)+np.prod(self.shapehbcx_phys)+np.prod(self.shapehbcy_phys))
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
    
        #self.test_operg()
        
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
        
        
    def operg(self,X,t,State=None):
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
        
        phi = np.concatenate((He.flatten(),hbcx.flatten(),hbcy.flatten()))
        
        
        if State is not None:
            State.params = phi
        
        else:
            return phi


    def operg_transpose(self,adState, adX, t):
        """
            Project to reduced space
        """
        
        # Get variable in physical space
        He = adState.params[self.sliceHe_phys].reshape(self.shapeHe_phys)
        hbcx = adState.params[self.slicehbcx_phys].reshape(self.shapehbcx_phys)
        hbcy = adState.params[self.slicehbcy_phys].reshape(self.shapehbcy_phys)
        
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
        
        adX += np.concatenate((adX_He.flatten(),
                              adX_bcS.flatten(),
                              adX_bcN.flatten(),
                              adX_bcE.flatten(),
                              adX_bcW.flatten()))
        

        
    def test_operg(self,t=0):
        psi = np.random.random((self.nbasis,))
        phi = np.random.random((self.nphys,))
        
        ps1 = np.inner(psi,self.operg_transpose(phi,t))
        ps2 = np.inner(self.operg(psi,t),phi)
            
        print(f'test G[{t}]:', ps1/ps2)
        
class RedBasis_BM:
   
    def __init__(self,config):

        self.km2deg=1./110
        
        self.flux = config.flux
        self.facns = config.facns # factor for wavelet spacing= space
        self.facnlt = config.facnlt
        self.npsp = config.npsp # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp = config.facpsp # 1.5 # factor to fix df between wavelets 
        self.lmin = config.lmin 
        self.lmax = config.lmax
        self.tdecmin = config.tdecmin
        self.tdecmax = config.tdecmax
        self.factdec = config.factdec
        self.sloptdec = config.sloptdec
        self.Qmax = config.Qmax
        self.facQ = config.facQ
        self.slopQ = config.slopQ
        self.lmeso = config.lmeso
        self.tmeso = config.tmeso
        self.save_wave_basis = config.save_wave_basis
        self.wavelet_init = config.wavelet_init

        for param in ['flux','facns','facnlt','lmin','lmax','tdecmin','tdecmax','factdec','sloptdec','Qmax','facQ','slopQ','lmeso','tmeso']:
            print(param,'=',config[param])

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.tmp_DA_path
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
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        
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
                ENSLAT[iff]=np.concatenate(([ENSLAT[iff],np.repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON[iff]=np.concatenate(([ENSLON[iff],ENSLON1]))
            

            NP[iff] = len(ENSLON[iff])
            enst.append(list())
            tdec.append(list())
            for P in range(NP[iff]):
                enst[-1].append(list())
                tdec[-1].append(list()) 
                tdec[-1][-1] = self.tmeso*self.lmeso**(self.sloptdec) * ff[iff]**self.sloptdec
                if tdec[-1][-1]<self.tdecmin:
                    tdec[-1][-1] = self.tdecmin
                if tdec[-1][-1]>self.tdecmax:
                    tdec[-1][-1] = self.tdecmax
                tdec[-1][-1] *= self.factdec
                enst[-1][-1] = np.arange(-tdec[-1][-1]*(1-1./self.facnlt),deltat+tdec[-1][-1]/self.facnlt , tdec[-1][-1]/self.facnlt)
                nwave += ntheta*2*len(enst[iff][P])
                
            
                
                
        # Fill the Q diagonal matrix (expected variance for each wavelet)            
        Q = np.zeros((nwave))
        iwave = 0
        # Loop on all wavelets of given pseudo-period 
        for iff in range(nf):
            for P in range(NP[iff]):
                
                _nwave = 2*len(enst[iff][P])*ntheta
                
                if 1/ff[iff]>self.lmeso:
                    Q[iwave:iwave+_nwave] = self.Qmax 
                else:
                    Q[iwave:iwave+_nwave] = self.Qmax * self.lmeso**self.slopQ * ff[iff]**self.slopQ
                iwave += _nwave
                
            print(f'lambda={1/ff[iff]:.1E}',
                  f'nlocs={P:.1E}',
                  f'tdec={tdec[iff][-1]:.1E}',
                  f'Q={Q[iwave-_nwave]:.1E}')
            
        nwave = iwave
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
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.lon1d.size,))
        
        iwave = 0
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                enstloc = self.enst[iff][P]
                iobs = indx[(iff,P)]
                if iobs.shape[0] > 0:
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
        
        if State is not None:
            if t==0:
                if self.wavelet_init:
                    State.setvar(phi.reshape((State.ny,State.nx)),ind=0)
                    State.params = np.zeros_like(phi)
                else:
                    State.params = phi
            else:
                State.params = phi
        else:
            return phi
        
    
    def operg_transpose(self, adState, adX, t):
        
        """
            Project to reduced space
        """
        
        if t==0:
            if self.wavelet_init:
                adX += self.operg(adState.getvar(ind=0).flatten(), t, transpose=True)
            else:
                adX += self.operg(adState.params, t, transpose=True)
        else:
            if adState.params is None:
                adState.params = np.zeros((self.nphys,))
            adX += self.operg(adState.params, t, transpose=True)
        
class RedBasis_BMaux:
   
    def __init__(self,config):

        self.km2deg=1./110
        
        self.flux = config.flux
        self.facns = config.facns # factor for wavelet spacing= space
        self.facnlt = config.facnlt
        self.npsp = config.npsp # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp = config.facpsp # 1.5 # factor to fix df between wavelets 
        self.lmin = config.lmin 
        self.lmax = config.lmax
        self.tdecmin = config.tdecmin
        self.tdecmax = config.tdecmax
        self.factdec = config.factdec
        self.facQ = config.facQ
        self.save_wave_basis = config.save_wave_basis
        self.wavelet_init = config.wavelet_init
        self.file_aux = config.file_aux
        self.filec_aux = config.filec_aux
        self.Romax = config.Romax
        self.facRo = config.facRo
        self.cutRo = config.cutRo
        self.tssr = config.tssr
        self.distortion_eq = config.distortion_eq
        self.distortion_eq_law = config.distortion_eq_law
        self.lat_distortion_eq = config.lat_distortion_eq

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.tmp_DA_path
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
        daTdec = aux['tdec']
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
        iwave = 0
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
            phi = np.zeros((self.nbasis,))
        else:
            phi = np.zeros((self.lon1d.size,))
        
        iwave = 0
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                if self.wavetest[iff][P]:
                    enstloc = self.enst[iff][P]
                    iobs = indx[(iff,P)]
                    if iobs.shape[0] > 0:
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
        
        if State is not None:
            if t==0:
                if self.wavelet_init:
                    State.setvar(phi.reshape((State.ny,State.nx)),ind=0)
                    State.params = np.zeros_like(phi)
                else:
                    State.params = phi
            else:
                State.params = phi
        else:
            return phi
        
    
    def operg_transpose(self, adState, adX, t):
        
        """
            Project to reduced space
        """
        
        if t==0:
            if self.wavelet_init:
                adX += self.operg(adState.getvar(ind=0).flatten(), t, transpose=True)
            else:
                adX += self.operg(adState.params, t, transpose=True)
        else:
            if adState.params is None:
                adState.params = np.zeros((self.nphys,))
            adX += self.operg(adState.params, t, transpose=True)
                   
class RedBasis_BM_IT:
   
    def __init__(self,config):
        self.RedBasis_BM = RedBasis_BM(config)
        self.RedBasis_IT = RedBasis_IT(config)
        
    def set_basis(self,time,lon,lat,return_q=False):
        
        print('* Reduced basis for BM:')
        Qbm = self.RedBasis_BM.set_basis(time,lon,lat,return_q=return_q)
        
        print('* Reduced basis for IT:')
        Qit = self.RedBasis_IT.set_basis(time,lon,lat,return_q=return_q)
        
        self.nbasis = self.RedBasis_BM.nbasis + self.RedBasis_IT.nbasis
        self.slicebm = slice(0,self.RedBasis_BM.nbasis)
        self.sliceit = slice(self.RedBasis_BM.nbasis,
                             self.RedBasis_BM.nbasis + self.RedBasis_IT.nbasis)
        self.slicebm_phys = slice(0,self.RedBasis_BM.nphys)
        self.sliceit_phys = slice(self.RedBasis_BM.nphys,
                                  self.RedBasis_BM.nphys + self.RedBasis_IT.nphys)
        
        self.RedBasis_IT.sliceHe_phys = slice(self.RedBasis_BM.nphys,
                                              self.RedBasis_BM.nphys + np.prod(self.RedBasis_IT.shapeHe_phys))
        self.RedBasis_IT.slicehbcx_phys = slice(self.RedBasis_BM.nphys + np.prod(self.RedBasis_IT.shapeHe_phys),
                               self.RedBasis_BM.nphys + np.prod(self.RedBasis_IT.shapeHe_phys)+np.prod(self.RedBasis_IT.shapehbcx_phys))
        self.RedBasis_IT.slicehbcy_phys = slice(self.RedBasis_BM.nphys + np.prod(self.RedBasis_IT.shapeHe_phys)+np.prod(self.RedBasis_IT.shapehbcx_phys),
                               self.RedBasis_BM.nphys + np.prod(self.RedBasis_IT.shapeHe_phys)+np.prod(self.RedBasis_IT.shapehbcx_phys)+np.prod(self.RedBasis_IT.shapehbcy_phys))
        
        
        if return_q:
            return np.concatenate((Qbm,Qit))
        
    
    def operg(self, X, t, transpose=False,State=None):
        
        """
            Project to physicial space
        """
        
        phi_bm = self.RedBasis_BM.operg(X[self.slicebm],t)
        phi_it = self.RedBasis_IT.operg(X[self.sliceit],t)
        
        phi = np.concatenate((phi_bm,phi_it))
        
        if State is not None:
            if t==0:
                if self.RedBasis_BM.wavelet_init:
                    State.setvar(phi_bm.reshape((State.ny,State.nx)),ind=0)
                    State.params[self.slicebm_phys] = np.zeros_like(phi_bm)
                    State.params[self.sliceit_phys] = phi_it
                else:
                    State.params = phi
            else:
                State.params = phi
        
        else:
            return phi
        
    def operg_transpose(self, adState, adX, t):
        
        """
            Project to reduced space
        """
        
        self.RedBasis_BM.operg_transpose(adState,adX[self.slicebm],t)
        self.RedBasis_IT.operg_transpose(adState,adX[self.sliceit],t)
    
class RedBasis_BM_2scales:
   
    def __init__(self,config):
        
        lmin = config.lmin
        lmax = config.lmax
        
        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./config.lmin),
            np.log(1. / config.lmax) - np.log(1 + config.facpsp / config.npsp),
            -np.log(1 + config.facpsp / config.npsp))[::-1]
        ff = np.exp(logff)
        
        ind_ls = np.where((1/ff>=config.lmeso))[0]
        ind_ss = np.where((1/ff<config.lmeso))[0]
    
        # Large scales
        config_ls = config
        config_ls.lmin = 1/ff[ind_ls[-1]]
        self.RedBasis_ls = RedBasis_BM(config_ls)
        # Small scales
        config_ss = config
        config_ss.lmin = lmin
        config_ss.lmax = 1/ff[ind_ss[1]]
        self.RedBasis_ss = RedBasis_BM(config_ss)
        config.lmax = lmax
        
        self.wavelet_init = config.wavelet_init
        
        
        
    def set_basis(self,time,lon,lat,return_q=False):
        
        print('* Reduced basis for large scales:')
        Qls = self.RedBasis_ls.set_basis(time,lon,lat,return_q=return_q)
        
        print('* Reduced basis for small scales:')
        Qss = self.RedBasis_ss.set_basis(time,lon,lat,return_q=return_q)
        
        self.nbasis = self.RedBasis_ls.nbasis + self.RedBasis_ss.nbasis
        self.nphys = self.RedBasis_ls.nphys + self.RedBasis_ss.nphys
        self.slice_ls = slice(0,self.RedBasis_ls.nbasis)
        self.slice_ss = slice(self.RedBasis_ls.nbasis,
                             self.RedBasis_ls.nbasis + self.RedBasis_ss.nbasis)
        self.slice_ls_phys = slice(0,self.RedBasis_ls.nphys)
        self.slice_ss_phys = slice(self.RedBasis_ls.nphys,
                                  self.RedBasis_ss.nphys + self.RedBasis_ss.nphys)
        
        # print("Test operg:")
        # for t in time:
        #     self.test_operg(t)
            
        if return_q:
            return np.concatenate((Qls,Qss))
        
        
        
    def operg(self, X, t, transpose=False,State=None):
        
        """
            Project to physicial space
        """
        
        if transpose:
            phi_ls = self.RedBasis_ls.operg(X[self.slice_ls_phys],t,transpose=True)
            phi_ss = self.RedBasis_ss.operg(X[self.slice_ss_phys],t,transpose=True)
        else:
            phi_ls = self.RedBasis_ls.operg(X[self.slice_ls],t)
            phi_ss = self.RedBasis_ss.operg(X[self.slice_ss],t)

        phi = np.concatenate((phi_ls,phi_ss))
        
        if State is not None:
            if t==0 :
                if self.wavelet_init:
                    State.setvar(phi_ls.reshape((State.ny,State.nx)),ind=0)
                    State.setvar(phi_ss.reshape((State.ny,State.nx)),ind=1)
                    State.params = np.zeros(self.nphys)
                else:
                    State.params = phi
            else:
                State.params = phi
        
        else:
            return phi
        
    def operg_transpose(self, adState, adX, t):
        
        """
            Project to reduced space
        """
        
        if t==0:
            if self.wavelet_init:
                adX += self.operg(adState.getvar(ind=[0,1],vect=True), t, transpose=True)
            else:
                adX += self.operg(adState.params, t, transpose=True)
        else:
            if adState.params is None:
                adState.params = np.zeros((self.nphys,))
            adX += self.operg(adState.params, t, transpose=True)
    
    def test_operg(self,t=0):
        
        psi = np.random.random((self.nbasis,))
        phi = np.random.random((self.nphys,))
        
        ps1 = np.inner(psi,self.operg(phi,t,transpose=True))
        ps2 = np.inner(self.operg(psi,t),phi)
            
        print(f'test G[{t}]:', ps1/ps2)
    
        

def mywindow(x): #xloc must be between -1 and 1
     y  = np.cos(x*0.5*np.pi)**2
     return y
  
def mywindow_flux(x): #xloc must be between -1 and 1
     y = -np.pi*np.sin(x*0.5*np.pi)*np.cos(x*0.5*np.pi)
     return y

