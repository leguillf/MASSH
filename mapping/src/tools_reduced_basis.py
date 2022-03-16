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


class RedBasis_IT:
   
    def __init__(self,config):

        self.km2deg =1./110
    
        self.facns = config.facns
        self.facnlt = config.facnlt
        self.D_He = config.D_He
        self.T_He = config.T_He
        self.D_bc = config.D_bc
        self.T_bc = config.T_bc
        
        self.sigma_B_He = config.sigma_B_He
        self.sigma_B_bc = config.sigma_B_bc
        
        self.Ntheta = config.Ntheta
        if self.Ntheta>0:
            self.Ntheta += 2 # We add -pi/2,pi/2
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
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}')
    
        self.test_operg()
        
        # Fill Q matrix
        if return_q:
            if None not in [self.sigma_B_He, self.sigma_B_bc]:
                Q = np.zeros((self.nbasis,)) 
                # variance on He
                Q[self.sliceHe] = self.sigma_B_He 
                Q[self.slicebc] = self.sigma_B_bc 
            else:
                Q = None
            
            return Q
        
        
    def operg(self,psi,t):
        """
            Project to physicial space
        """
        
        # Get variables in reduced space
        psi_He = psi[self.sliceHe].reshape(self.shapeHe)
        psi_bcS = psi[self.slicebcS].reshape(self.shapehbcS)
        psi_bcN = psi[self.slicebcN].reshape(self.shapehbcN)
        psi_bcE = psi[self.slicebcE].reshape(self.shapehbcE)
        psi_bcW = psi[self.slicebcW].reshape(self.shapehbcW)
        
        # Project to physical space
        He = np.tensordot(
            np.tensordot(psi_He,self.He_xy_gauss,(1,0)),
                                self.He_t_gauss[:,t],(0,0))
    
        hbcx = np.zeros(self.shapehbcx_phys)
        hbcy = np.zeros(self.shapehbcy_phys)
        
        hbcx[:,0] = np.tensordot(
            np.tensordot(psi_bcS,self.bc_S_gauss,(-1,0)),
                                 self.bc_t_gauss[:,t],(-2,0))
        hbcx[:,1] = np.tensordot(
            np.tensordot(psi_bcN,self.bc_N_gauss,(-1,0)),
                                 self.bc_t_gauss[:,t],(-2,0))
        hbcy[:,0] = np.tensordot(
            np.tensordot(psi_bcE,self.bc_E_gauss,(-1,0)),
                                 self.bc_t_gauss[:,t],(-2,0))
        hbcy[:,1] = np.tensordot(
            np.tensordot(psi_bcW,self.bc_W_gauss,(-1,0)),
                                 self.bc_t_gauss[:,t],(-2,0))
        
        phi = np.concatenate((He.flatten(),hbcx.flatten(),hbcy.flatten()))
        
        return phi
    
    def operg_transpose(self,phi,t):
        """
            Project to reduced space
        """
        
        # Get variable in physical space
        phi_He = phi[self.sliceHe_phys].reshape(self.shapeHe_phys)
        phi_hbcx = phi[self.slicehbcx_phys].reshape(self.shapehbcx_phys)
        phi_hbcy = phi[self.slicehbcy_phys].reshape(self.shapehbcy_phys)
        
        # Project to reduced space
        psi_He = np.tensordot(
            phi_He[:,:,np.newaxis]*self.He_t_gauss[:,t],
                                   self.He_xy_gauss[:,:,:],([0,1],[1,2])) 
        psi_bcS = np.tensordot(
               phi_hbcx[:,0,:,:,:,np.newaxis]*self.bc_t_gauss[:,t],
                                              self.bc_S_gauss,(-2,-1))
        psi_bcN = np.tensordot(
               phi_hbcx[:,1,:,:,:,np.newaxis]*self.bc_t_gauss[:,t],
                                              self.bc_N_gauss,(-2,-1))
        psi_bcE = np.tensordot(
               phi_hbcy[:,0,:,:,:,np.newaxis]*self.bc_t_gauss[:,t],
                                              self.bc_E_gauss,(-2,-1))
        psi_bcW = np.tensordot(
               phi_hbcy[:,1,:,:,:,np.newaxis]*self.bc_t_gauss[:,t],
                                              self.bc_W_gauss,(-2,-1))
        
        psi = np.concatenate((psi_He.flatten(),
                              psi_bcS.flatten(),
                              psi_bcN.flatten(),
                              psi_bcE.flatten(),
                              psi_bcW.flatten()))
        
        return psi
        
    def test_operg(self,t=0):
        psi = np.random.random((self.nbasis,))
        phi = np.random.random((self.nphys,))
        
        ps1 = np.inner(psi,self.operg_transpose(phi,t))
        ps2 = np.inner(self.operg(psi,t),phi)
            
        print(f'test G[{t}]:', ps1/ps2)
        
        
        
                
class RedBasis_BM:
   
    def __init__(self,config):

        self.km2deg=1./110
        
        
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
        self.gsize_max = config.gsize_max
        self.lmeso = config.lmeso
        self.tmeso = config.tmeso        
     

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
        self.wave_xy = [None,]*nf
        self.wave_t_norm = [None,]*nf
        self.wave_t = [None,]*nf
        self.slicef = [None,]*nf
        NT = np.empty(nf, dtype='int32') # Nomber of time locations for a given frequency
        NS = np.empty(nf, dtype='int32') # Nomber of spatial wavelet locations for a given frequency
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        
        nbasis = 0        
        for iff in range(nf):
            
            # Coordinates in space
            ENSLON = []
            ENSLAT = []
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = np.mod(
                    np.arange(
                        LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        LON_MAX+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg),
                    360)
                ENSLAT = np.concatenate(([ENSLAT,np.repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON = np.concatenate(([ENSLON,ENSLON1]))
            
            # Coordinates in time 
            tdec = self.tmeso*self.lmeso**(self.sloptdec) * ff[iff]**self.sloptdec
            if tdec<self.tdecmin:
                tdec = self.tdecmin
            if tdec>self.tdecmax:
                tdec = self.tdecmax
            tdec *= self.factdec
            ENST = np.arange(-tdec*(1-1./self.facnlt),deltat+tdec/self.facnlt , tdec/self.facnlt)
            print(iff,1/ff[iff],tdec)
            
            NT[iff] = len(ENST)
            NS[iff] = 2*ntheta * len(ENSLAT)
            self.slicef[iff] = slice(nbasis,nbasis + NT[iff]*NS[iff])
            nbasis += NT[iff] * NS[iff]
            
            # Wavelet shape in space
            wave_xy_f = np.zeros((ENSLAT.size,2*ntheta,lon1d.size))
            for i,(lat0,lon0) in enumerate(zip(ENSLAT,ENSLON)):
                iobs = np.where(
                        (np.abs((np.mod(lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= DX[iff]) &
                        (np.abs((lat1d - lat0) / self.km2deg) <= DX[iff])
                        )[0]
                xx = (np.mod(lon1d[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
                yy = (lat1d[iobs] - lat0) / self.km2deg
                facs = mywindow(xx / DX[iff]) * mywindow(yy / DX[iff])
                for itheta in range(ntheta):
                    kx = 2 * np.pi * ff[iff] * np.cos(theta[itheta])
                    ky = 2 * np.pi * ff[iff] * np.sin(theta[itheta])
                    wave_xy_f[i,2*itheta,iobs] = facs * np.cos(kx*(xx)+ky*(yy))
                    wave_xy_f[i,2*itheta+1,iobs] = facs * np.cos(kx*(xx)+ky*(yy)-np.pi / 2)
            self.wave_xy[iff] = wave_xy_f.reshape((ENSLAT.size*2*ntheta,lon1d.size))
            
            # Wavelet shape in time
            wave_t_f = np.zeros((ENST.size,time.size))
            for i,time0 in enumerate(ENST):
                iobs = np.where(abs(time-time0) < tdec)
                t = np.linspace(-tdec,tdec)
                I =  np.sum(mywindow(t/tdec))*(t[1]-t[0])
                wave_t_f[i,iobs] = mywindow(abs(time-time0)[iobs]/tdec)   
            self.wave_t_norm[iff] = wave_t_f/I  
            self.wave_t[iff] = wave_t_f

        # Fill the Q diagonal matrix (expected variance for each wavelet)            
        Q = np.zeros((nbasis,))
        # Loop on all wavelets of given pseudo-period
        for iff in range(nf):
            if 1/ff[iff]>self.lmeso:
                Q[self.slicef[iff]] = self.Qmax   
            else:
                Q[self.slicef[iff]] = self.Qmax * self.lmeso**self.slopQ * ff[iff]**self.slopQ

        
        plt.figure()
        plt.plot(Q)
        plt.yscale('log')
        plt.show()
        
        self.nf = nf
        self.NS = NS
        self.NT = NT
        self.ntheta = ntheta
        self.nbasis = nbasis
        self.nphys = lon1d.size
        
        print('nbasis=',self.nbasis)

        self.test_operg()
        
        if return_q:
            return np.sqrt(Q)
        
    
    def operg(self,psi,t,norm=True):
        
        """
            Project to physicial space
        """
        
        phi = np.zeros((self.nphys))
        for iff in range(self.nf):
            psi_f = psi[self.slicef[iff]]
            psi_f = psi_f.reshape((self.NT[iff],self.NS[iff]))
            if norm:
                phi += np.tensordot(np.tensordot(psi_f,self.wave_xy[iff],(-1,0)),
                                    self.wave_t_norm[iff][:,t],(0,0))
            else:
                phi += np.tensordot(np.tensordot(psi_f,self.wave_xy[iff],(-1,0)),
                                    self.wave_t[iff][:,t],(0,0))
        return phi
    
    def operg_transpose(self,phi,t,norm=True):
        
        """
            Project to reduced space
        """
        
        psi = np.zeros((self.nbasis,))
        for iff in range(self.nf):
            if norm:
                psi[self.slicef[iff]] = np.tensordot(
                    phi[:,np.newaxis]*self.wave_t_norm[iff][:,t],
                                           self.wave_xy[iff],[0,1]).flatten() 
            else:
                psi[self.slicef[iff]] = np.tensordot(
                    phi[:,np.newaxis]*self.wave_t[iff][:,t],
                                           self.wave_xy[iff],[0,1]).flatten() 
        return psi
    
    def test_operg(self,t=0):
        psi = np.random.random((self.nbasis,))
        phi = np.random.random((self.nphys,))
        
        ps1 = np.inner(psi,self.operg_transpose(phi,t))
        ps2 = np.inner(self.operg(psi,t),phi)
            
        print(f'test G[{t}]:', ps1/ps2)
        
class RedBasis_BM_2:
   
    def __init__(self,config):

        self.km2deg=1./110
        
        
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
        self.gsize_max = config.gsize_max
        self.lmeso = config.lmeso
        self.tmeso = config.tmeso

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.tmp_DA_path
        self.indx = {}
        self.indt = {}
        self.facGeta = {}
        self.facGeta_flux = {}

        
     

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
                
            print(iff,P,1/ff[iff],tdec[-1][-1])
                
                
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
                
            print(1/ff[iff],Q[iwave-_nwave])
            
        nwave = iwave
        Q=Q[:nwave]
        
        plt.figure()
        plt.plot(Q)
        plt.yscale('log')
        plt.show()
        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nwave=nwave
        self.nf=nf
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff
        self.k = 2 * np.pi * ff
        self.tdec=tdec
        
        
        print('nwaves=',self.nwave)
        print('nf=',self.nf)
        print('ntheta=',self.ntheta)

        if return_q:
            return np.sqrt(Q)
        
        
        
        
    def operg(self, coords=None, coords_name=None, compute_g=False, 
            compute_geta=False, eta=None, transpose=False,
            coordtype='scattered', iwave0=0, iwave1=None,
            int_type='i8', float_type='f8',save_wave_basis=True,mode=None):
        
        
        
        if iwave1==None: iwave1=self.nwave

        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]
        
        if hasattr(time,'__len__'):
            nt = len(time)
        else:
            nt = 1
            time = [time]
            
        Geta = None
        if compute_geta:
            if transpose:
                Geta = np.zeros((self.nwave))
            else:
                Geta = np.zeros((nt,lon.size))
        
        G=[None]*3
        if compute_g:
            G[0]=np.zeros((iwave1-iwave0), dtype=int_type)
            G[1]=np.empty((self.gsize_max), dtype=int_type)
            G[2]=np.empty((self.gsize_max), dtype=float_type)
            ind_tmp = 0
            
            
        
        compute_basis = False
        if save_wave_basis:
            # Offline
            name_facGeta = os.path.join(self.path_save_tmp,f'facGeta_{time[0]}_{mode}.pic')
            name_indt = os.path.join(self.path_save_tmp,f'indt_{time[0]}.pic')
            name_indx = os.path.join(self.path_save_tmp,'indx.pic')
            if os.path.exists(name_facGeta) and os.path.exists(name_indt) and os.path.exists(name_indx):
                with open(name_facGeta, 'rb') as f:
                    facGeta = pickle.load(f)
                with open(name_indt, 'rb') as f:
                    indt = pickle.load(f)
                with open(name_indx, 'rb') as f:
                    indx = pickle.load(f)
            else: 
                compute_basis = True

        
        elif ((mode!='flux' and time[0] in self.facGeta) or (mode=='flux' and time[0] in self.facGeta_flux))\
            and (time[0] in self.indt) and self.indx!={}:
            # Inline
            if mode=='flux':
                facGeta = self.facGeta_flux[time[0]]
            else:
                facGeta = self.facGeta[time[0]]
            indt = self.indt[time[0]]
            indx = self.indx
        else: 
            compute_basis = True
        if compute_basis:
            facGeta = {}
            indt = {}
            indx = {}
            
            for iff in range(self.nf):
                for P in range(self.NP[iff]):
                        
                    indt[(iff,P)] = {}
                    facGeta[(iff,P)] = {}
                    
                    # Obs selection around point P
                    iobs = np.where(
                        (np.abs((np.mod(lon - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                        (np.abs((lat - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                        )[0]
                    xx = (np.mod(lon[iobs] - self.ENSLON[iff][P]+180,360)-180) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                    yy = (lat[iobs] - self.ENSLAT[iff][P]) / self.km2deg
  
                    
                    # facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd
                    facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff])
                    
                    indx[(iff,P)] = iobs

                    enstloc = self.enst[iff][P]
                    
                    if iobs.shape[0] > 0:
                        for it in range(len(enstloc)):
                            nobs = 0
                            iiobs=[]
                            diff = time - enstloc[it]
                            iobs2 = np.where(abs(diff) < self.tdec[iff][P])[0] 
                            for i2 in iobs2:
                                for i in iobs:
                                    iiobs.append(np.ravel_multi_index(
                                        (i2,i), (nt,len(lon))))
                            nobs = len(iiobs)
                            if nobs > 0:
                                tt2 = diff[iobs2]
                                
                                fact = mywindow(tt2 / self.tdec[iff][P])
                                
                                if mode=='flux':
                                    t = np.linspace(-self.tdec[iff][P],self.tdec[iff][P])
                                    I =  np.sum(mywindow(t/self.tdec[iff][P]))*(t[1]-t[0])
                                    fact /= I 
                                    
                            else:
                                fact = None
                            indt[iff,P][it] = (iobs2,iiobs,nobs)
                                
                            if ((nobs == 0)):
                                pass
                            else:
                                facGeta[(iff,P)][it] = [None,]*self.ntheta
                                for itheta in range(self.ntheta):
                                    facGeta[(iff,P)][it][itheta] = [[],[]]
                                    kx = self.k[iff] * np.cos(self.theta[itheta])
                                    ky = self.k[iff] * np.sin(self.theta[itheta])
                                    facGeta[(iff,P)][it][itheta][0] = np.sqrt(2)* np.outer(fact , np.cos(kx*(xx)+ky*(yy))*facs)
                                    facGeta[(iff,P)][it][itheta][1] = np.sqrt(2)* np.outer(fact , np.cos(kx*(xx)+ky*(yy)-np.pi / 2)*facs)
                                    

                
                    
            if save_wave_basis:
                if not os.path.exists(name_facGeta):
                    with open(name_facGeta, 'wb') as f:
                        pickle.dump(facGeta,f)  
                if not os.path.exists(name_indt):
                    with open(name_indt, 'wb') as f:
                        pickle.dump(indt,f)
                if not os.path.exists(name_indx):
                    with open(name_indx, 'wb') as f:
                        pickle.dump(indx,f)     
            else:
                if mode=='flux':
                    self.facGeta_flux[time[0]] = facGeta
                else:
                    self.facGeta[time[0]] = facGeta
                self.indt[time[0]] = indt
                self.indx = indx
                                        
        iwave = 0
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                enstloc = self.enst[iff][P]
                iobs = indx[(iff,P)]
                if iobs.shape[0] > 0:
                    for it in range(len(enstloc)):
                        iobs2,iiobs,nobs = indt[iff,P][it]
                        if ((nobs == 0)):
                            iwave += 2*self.ntheta
                        else:
                            for itheta in range(self.ntheta):
                                for iphase in range(2):
                                    if compute_g:
                                        G[0][iwave-iwave0] = nobs
                                        G[1][ind_tmp:ind_tmp+nobs] = iiobs
                                        G[2][ind_tmp:ind_tmp+nobs] = facGeta[(iff,P)][it][itheta][iphase].flatten()                              
                                    if compute_geta:
                                        if transpose:
                                            Geta[iwave] = np.sum(eta[iobs2[0]:iobs2[-1]+1,iobs] * facGeta[(iff,P)][it][itheta][iphase])
                                        else:
                                            Geta[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * facGeta[(iff,P)][it][itheta][iphase]
                            
                                        iwave += 1  
                                
                            
                            
                            
                            
        if compute_g and compute_geta:           
            return [np.copy(G[0]), np.copy(G[1][:ind_tmp]), np.copy(G[2][:ind_tmp])],Geta
        elif compute_g and not compute_geta:
            return [np.copy(G[0]), np.copy(G[1][:ind_tmp]), np.copy(G[2][:ind_tmp])]
        elif compute_geta and not compute_g:
            return Geta
        else:
            return  



    
    

def mywindow(x): #xloc must be between -1 and 1
     y  = np.cos(x*0.5*np.pi)**2
     return y
 
    
def mywindow_flux(x): #xloc must be between -1 and 1
     y = -np.pi*np.sin(x*0.5*np.pi)*np.cos(x*0.5*np.pi)
     return y







