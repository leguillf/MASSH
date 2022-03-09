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



        
        
class RedBasis_BM:
   
    def __init__(self,config,State):

        self.TIME_MIN = (config.init_date - datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX = (config.final_date - datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.LON_MIN = State.lon.min()
        self.LON_MAX = State.lon.max()
        self.LAT_MIN = State.lat.min()
        self.LAT_MAX = State.lat.max()  
        
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

        
     

    def set_basis(self,return_q=False):
        
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
        deltat = self.TIME_MAX - self.TIME_MIN

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        lonmax = self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax = self.LON_MAX+360.
            
        for iff in range(nf):
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(
                self.LAT_MIN - (DX[iff]-DXG[iff])*self.km2deg,
                self.LAT_MAX + DX[iff]*self.km2deg,
                DXG[iff]*self.km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = np.mod(
                    np.arange(
                        self.LON_MIN - (DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg,
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
            return Q
        
        
        
        
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







