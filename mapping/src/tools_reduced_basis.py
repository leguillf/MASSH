#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:24:24 2021

@author: leguillou
"""

from datetime import datetime
import numpy as np
import scipy
import logging
import matplotlib.pylab as plt
from copy import deepcopy

from . import tools

class RedBasis_QG:
   
    def __init__(self,config,State):

        self.TIME_MIN = (config.init_date - datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX = (config.final_date - datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.LON_MIN = config.lon_min
        self.LON_MAX = config.lon_max
        self.LAT_MIN = config.lat_min
        self.LAT_MAX = config.lat_max   
        
        self.C = config.c
        
        self.facns= 1. #factor for wavelet spacing= space
        self.facnlt= 2.
        self.npsp= 3.5 # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp= 1.5 #1.5 # factor to fix df between wavelets 
        self.lmin= config.lmin 
        self.lmax= config.lmax
        self.cutRo= 1.6
        self.factdec= config.factdec
        self.tdecmin= config.tdecmin
        self.tdecmax= config.tdecmax
        self.tssr= 0.5
        self.facRo= 8.
        self.Romax= 150. #bidouille avec maxime
        self.facQ= config.facQ
        self.depth1= 0.
        self.depth2= 30.
        self.distortion_eq= 2.
        self.lat_distortion_eq= 5.
        self.distortion_eq_law= 2.
        self.file_aux = config.file_aux
        self.filec_aux = config.filec_aux
        self.gsize_max = config.gsize_max
        
     

    def set_basis(self,return_qinv=False):

        km2deg=1./110
        
        # Definition of the wavelet basis in the domain
        lat_tmp = np.arange(-90,90,0.1)
        alpha=(self.distortion_eq-1)*np.sin(self.lat_distortion_eq*np.pi/180)**self.distortion_eq_law
        finterpdist = scipy.interpolate.interp1d(lat_tmp, 1+alpha/(np.sin(np.maximum(self.lat_distortion_eq,np.abs(lat_tmp))*np.pi/180)**self.distortion_eq_law))

        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = np.arange(
            np.log(1./self.lmin ),
            np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
            -np.log(1 + self.facpsp / self.npsp))[::-1]
        ff = np.exp(logff)
        dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf=len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)
        
        # Auxillary data 
        finterpPSDS, finterpTDEC, finterpNOISEFLOOR = tools.read_auxdata_geos(self.file_aux)
        finterpC = tools.read_auxdata_geosc(self.filec_aux)
        finterpDEPTH = tools.read_auxdata_depth(self.filec_aux)

        # Global time window
        deltat = self.TIME_MAX - self.TIME_MIN

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        Cb1 = list() # First baroclinic phase speed
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave=0
        lonmax=self.LON_MAX
        
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.
        for iff in range(nf):
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            ENSLAT1 = np.arange(self.LAT_MIN-(DX[iff]-DXG[iff])*km2deg,self.LAT_MAX+DX[iff]*km2deg,DXG[iff]*km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = np.mod(np.arange(self.LON_MIN -(DX[iff]-DXG[iff])/np.cos(ENSLAT1[I]*np.pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        lonmax+DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*km2deg*finterpdist(ENSLAT1[I])) , 360)
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

                dlon = DX[iff]*km2deg/np.cos(ENSLAT[iff][P] * np.pi / 180.)*finterpdist(ENSLAT[iff][P])
                dlat = DX[iff]*km2deg
                elon = np.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat = np.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2 = np.meshgrid(elon,elat)
                
                tmp = finterpC((elon2.flatten(),elat2.flatten()))
                tmp = tmp[np.isnan(tmp)==False]
                if len(tmp)>0:
                    C = np.nanmean(finterpC((elon2.flatten(),elat2.flatten()))) # vitesse phase
                else: C = np.nan
                if np.isnan(C): C=0.
                Cb1[-1][-1] = C
                
                fc = (2*2*np.pi/86164*np.sin(ENSLAT[iff][P]*np.pi/180.))
                Ro = C / np.abs(fc) /1000. # Rossby radius (km)
                if Ro>self.Romax: Ro=self.Romax
                
                
                if C>0: td1=self.factdec / (1./(self.facRo*Ro)*C/1000*86400)
                else: td1 = np.nan
                
                PSDS = finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P]))
                if Ro>0: PSDSR = finterpPSDS((1./(self.facRo*Ro),ENSLAT[iff][P],ENSLON[iff][P]))
                else: PSDSR = np.nan
                if PSDS<=PSDSR: tdec[-1][-1] = td1 * (PSDS/PSDSR)**self.tssr
                else: tdec[-1][-1] = td1
                if tdec[-1][-1]>self.tdecmax: tdec[-1][-1]=self.tdecmax
                
                cp=1./(2*2*np.pi/86164*np.sin(max(10,np.abs(ENSLAT[iff][P]))*np.pi/180.))/300000
                tdecp=(1./ff[iff])*1000/cp/86400/4
                if tdecp<tdec[-1][-1]: tdec[-1][-1]=tdecp
                
                try: enst[-1][-1] = np.arange(-tdec[-1][-1]*(1-1./self.facnlt),deltat+tdec[-1][-1]/self.facnlt , tdec[-1][-1]/self.facnlt)
                except: pass
                nt = len(enst[iff][P])
                
                
                nwave += ntheta*2*nt
                

        

        # Fill the Q diagonal matrix (expected variace for each wavelet)
        self.wavetest = [None]*nf
        Q = np.zeros((nwave))
        iwave = -1
        self.iff_wavebounds = [None]*(nf+1)
        self.P_wavebounds = [None]*(nf+1)
          
        # Loop on all wavelets of given pseudo-period
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave+1
            self.P_wavebounds[iff] = [None]*(NP[iff]+1)
            self.wavetest[iff] = np.ones((NP[iff]), dtype=bool)
            for P in range(NP[iff]):
                self.P_wavebounds[iff][P] = iwave+1
                PSDLOC = abs(finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P])))
                C = Cb1[iff][P]
                fc = (2*2*np.pi/86164*np.sin(ENSLAT[iff][P]*np.pi/180.))
                if fc==0: Ro = self.Romax
                else:
                    Ro = C / np.abs(fc) /1000.  # Rossby radius (km)
                    if Ro>self.Romax: Ro = self.Romax
                if ((1./ff[iff] < self.cutRo * Ro) ): self.wavetest[iff][P]=False
                if tdec[iff][P]<self.tdecmin: 
                    self.wavetest[iff][P]=False
                if np.isnan(PSDLOC): self.wavetest[iff][P]=False
                if ((np.isnan(Cb1[iff][P]))|(Cb1[iff][P]==0)): self.wavetest[iff][P]=False
                if self.wavetest[iff][P]==True:
                    for it in range(len(enst[iff][P])):
                        for itheta in range(len(theta)):
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ * np.exp(-3*(self.cutRo * Ro*ff[iff])**3)
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ* np.exp(-3*(self.cutRo * Ro*ff[iff])**3)

            self.P_wavebounds[iff][P+1] = iwave +1
        self.iff_wavebounds[iff+1] = iwave +1

        nwave = iwave+1
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
        self.finterpDEPTH=finterpDEPTH

        self.finterpdist=finterpdist
        self.Cb1 = Cb1
        
        print('nwaves=',self.nwave)
        print('nf=',self.nf)
        print('ntheta=',self.ntheta)

        if return_qinv:
            return 1./Q
        
        
        
        
    def operg(self, coords=None, coords_name=None, compute_g=False, 
            compute_geta=False, eta=None, 
            coordtype='scattered', iwave0=0, iwave1=None,
            int_type='i8', float_type='f8'):
        

        if iwave1==None: iwave1=self.nwave

        km2deg=1./110 # A bouger, redondant

    
        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]

        if compute_geta:
            if coordtype=='reg':
                Geta = np.zeros((len(time),len(lon)))
            else:
                Geta = np.zeros((len(time)))
                
        if compute_g:
            G=[None]*3
            G[0]=np.zeros((iwave1-iwave0), dtype=int_type)
            G[1]=np.empty((self.gsize_max), dtype=int_type)
            G[2]=np.empty((self.gsize_max), dtype=float_type)
            ind_tmp = 0

        iwave = -1
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                    if self.wavetest[iff][P]:
                        distortion=self.finterpdist(self.ENSLAT[iff][P])
                        # Obs selection around point P
                        iobs = np.where((np.abs((np.mod(lon - self.ENSLON[iff][P]+180,360)-180) / km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.))/distortion <= self.DX[iff]) &
                                    (np.abs((lat - self.ENSLAT[iff][P]) / km2deg) <= self.DX[iff]))[0]
                        xx = (np.mod(lon[iobs] - self.ENSLON[iff][P]+180,360)-180) / km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) /distortion
                        yy = (lat[iobs] - self.ENSLAT[iff][P]) / km2deg
    
                        # Spatial tapering shape of the wavelet and its derivative if velocity
                        facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) 
                    
                        enstloc = self.enst[iff][P]
                        for it in range(len(enstloc)):
                            nobs = 0
                            iiobs=[]
                            if iobs.shape[0] > 0:
                                if coordtype=='reg':
                                    diff = time - enstloc[it]
                                    iobs2 = np.where(abs(diff) < self.tdec[iff][P])[0] 
                                    for i2 in iobs2:
                                        for i in iobs:
                                            iiobs.append(np.ravel_multi_index(
                                                (i2,i), (len(time),len(lon))))
                                    nobs = len(iiobs)
                                else:
                                    diff = time[iobs] - enstloc[it]
                                    iobs2 = np.where(abs(diff) < self.tdec[iff][P])[0]
                                    iiobs = iobs[iobs2]
                                    nobs = iiobs.shape[0]
    
                                if nobs > 0:
                                    tt2 = diff[iobs2]
                                    fact = mywindow_flux(tt2 / self.tdec[iff][P])
    
                            for itheta in range(self.ntheta):
                                kx = self.k[iff] * np.cos(self.theta[itheta])
                                ky = self.k[iff] * np.sin(self.theta[itheta])
                                for phase in [0, np.pi / 2]:
                                    iwave += 1
                                    if ((iwave >= iwave0) & (iwave <iwave1)):
                                        if ((nobs > 0)):
                                            if compute_g:
                                                G[0][iwave-iwave0] = nobs
                                                if coordtype=='reg':
                                                    G[1][ind_tmp:ind_tmp+nobs] = iiobs
                                                    G[2][ind_tmp:ind_tmp+nobs] = np.sqrt(2) * np.outer(fact,np.cos(kx*(xx)+ky*(yy)-phase)*facs).flatten()
                                                ind_tmp += nobs
                                  
                                            if compute_geta:
                                                if coordtype=='reg':
                                                    Geta[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * np.sqrt(2)* np.outer(fact , np.cos(kx*(xx)+ky*(yy)-phase)*facs)
                                                else:
                                                    Geta[iiobs] += eta[iwave] * np.sqrt(2)*np.cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2]*fact
                                        
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