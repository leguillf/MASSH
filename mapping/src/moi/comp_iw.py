# -*- coding: utf-8 -*-
"""
"""
import logging
import _pickle as pickle
#from numpy import pi, cos, sin, arange, log, linspace, array, where, meshgrid, concatenate, repeat, zeros, sqrt, empty, exp
from scipy.interpolate import griddata
from netCDF4 import Dataset
from scipy.fftpack import ifft, ifft2, fft, fft2
from os import path
from copy import deepcopy
from allcomps import Comp
from tools import mywindow
import yaml
from rw import read_auxdata_geos, read_auxdata_geosc, read_auxdata_depth, read_auxdata_varcit
from numpy import pi, sqrt, arange, log, zeros, exp, linspace, array, where, cos, concatenate, outer, sin, repeat, minimum, empty, full, real, imag, mod, append, abs
import pdb
from scipy.sparse import csc_matrix, coo_matrix, hstack, vstack
import numpy
#import pytide 
import pickle
import scipy.interpolate
import datetime



        
        
class Comp_cit(Comp, yaml.YAMLObject):

    """
    """



    #def __init__(self, lat_max=90.,**kwargs):
    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'lat_max' not in kwargs.keys(): self.lat_max = 90.
        #     kwargs['write']=True        logging.info('HELLO %s', lat_max)

        ## super(Comp_cit,self).__init__(**kwargs)

        self.ens_nature = ['sla', 'rcur'] # Ensemble of obs nature that project on the component
        self.name='iw_'+self.tidal_comp+'_mode'+str(self.mode)
        


    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   
        
    
    def set_basis(self, return_qinv=False):

        km2deg=1./110   

        deltat = self.TIME_MAX - self.TIME_MIN
                            
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2)) 

        finterpC = read_auxdata_geosc(self.filec_aux)

        DX = self.lambda_lw #* 0.5 #wavelet extension
        DXG = DX / self.facnls

        nwave=0
        lonmax=self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.       
        #for iff in range(nf):
        ENSLON=[]
        ENSLAT=[]  
        # if iff<nf-1:
        ENSLAT1 = arange(self.LAT_MIN-(DX-DXG)*km2deg,self.LAT_MAX+DX*km2deg,DXG*km2deg)
        ENSLAT1 = ENSLAT1[numpy.abs(ENSLAT1)<self.lat_max]
        for I in range(len(ENSLAT1)):
            ENSLON1 = mod(arange(self.LON_MIN -(DX-DXG)/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    lonmax+DX/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    DXG/cos(ENSLAT1[I]*pi/180.)*km2deg) , 360)
            ENSLAT=concatenate(([ENSLAT,repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON=concatenate(([ENSLON,ENSLON1]))


        NP = len(ENSLON)


        # enst=[None]*NP
        # tdec=[None]*NP
        # for P in range(NP):
        #     tdec[P] = self.tdec_lw

        #     enst[P] = arange(-tdec[P]*(1-1./self.facnlt) , deltat+tdec[P]/self.facnlt , tdec[P]/self.facnlt)
        #     nt = len(enst[P])
        #     nwave += nt


        # Fill the Q diagonal matrix (expected variace for each wavelet)


        ###self.C = finterpC((ENSLON,ENSLAT))

        #####self.wave_table=pytide.WaveTable([self.tidal_comp])
        data_astr = pickle.load( open( self.file_data_astr, "rb" ) )
        self.freq_tide = data_astr[self.tidal_comp]['freq']
        self.Ttide = 2*numpy.pi/self.freq_tide/3600 # Hours

        self.Ltide = numpy.zeros((NP))
        #self.Ttide*3600*self.C/1000./self.mode # km

        self.theta=[None]*NP
        self.wavetest=numpy.full((NP),False, dtype=bool)
        iwave=-1
        for P in range(NP):
            dlon=DX*km2deg/cos(ENSLAT[P]*pi/180.)
            dlat=DX*km2deg
            elon=numpy.linspace(ENSLON[P]-dlon,ENSLON[P]+dlon/2,10)
            elat=numpy.linspace(ENSLAT[P]-dlat,ENSLAT[P]+dlat/2,10)
            elon2,elat2=numpy.meshgrid(elon,elat)
            tmp = finterpC((elon2.flatten(),elat2.flatten()))
            tmp = tmp[numpy.isnan(tmp)==False]
            if len(tmp)>0:
                C = numpy.nanmean(tmp)
            else: C=0.
            if ((C>self.cmin)): 
                self.wavetest[P]=True
                self.Ltide[P] = self.Ttide*3600*C/1000./self.mode # km

                npsp = 2*DX / self.Ltide[P]
                ntheta=max(int(npsp*self.facntheta), 1)
                self.theta[P] = linspace(0, pi, ntheta+1)[:-1]

                iwave += len(self.theta[P])*4

        nwave = iwave+1
        Q=zeros((nwave))

        varHlw = self.std_lw**2 * fcor
        Q[:]=varHlw/(self.facnls)



        self.data_Qinv = 1./Q

        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        #self.enst=enst
        self.nwave=nwave
        #self.tdec=tdec
        self.time0 = self.TIME_MIN # Reference in CNES julian days, will be used for astronomical coefficients
        


        if return_qinv:
            return 1./Q



    
    def operg(self, coords=None, coords_name=None, cdir=None, config=None, nature=None, compute_g=False, 
                compute_geta=False, eta=None, coordtype='scattered', iwave0=0, iwave1=None, obs_name=None, gsize_max=None, int_type='i8', float_type='f4', label=None):

        if iwave1==None: iwave1=self.nwave

        if compute_g==True: logging.debug('START computing G: %s %s %s', self.name, obs_name, label )

        km2deg=1./110 # A bouger, redondant

        compute_gh=False
        compute_gheta=False
        compute_gc=False
        compute_gceta=False
        if nature=='sla':
            if compute_g:
                compute_gh=True
            if compute_geta:
                compute_gheta=True
        if nature=='rcur':
            if compute_g:
                compute_gc=True
            if compute_geta:
                compute_gceta=True

        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]

        timejd=self.time0+time
        Amp_astr, Phi_astr = jd2ap(self.file_data_astr, self.tidal_comp, timejd)

        if (nature=='rcur'):
            if cdir is None: angle = coords[coords_name['angle']]
            else: angle = full((len(lon)), cdir)
            eps = 0.1 # in km to convert the H wavelets into equivalent current wavelets


        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(lon)))
            else:
                result = zeros((len(time)))

        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0

        iwave = -1
        # for iff in range(self.nf):
        #     logging.debug('progress: %d/%d', iff+1, self.nf)
        for P in range(self.NP):
            if self.wavetest[P]==True:

                # if (nature=='rcur'):
                #     fc = 2*2*pi/86164*sin(self.ENSLAT[iff][P]*pi/180.) # Coriolis parameter
                #     if cdir is None: angle = coords[coords_name['angle']]
                #     else: angle = full((len(lon)), cdir) 
                #     epsx = 1j*self.freq_tide*cos(angle) + fc*sin(angle)
                #     epsy = -fc*cos(angle) + 1j*self.freq_tide*sin(angle)
                #     norm = ( numpy.abs(epsx)**2 + numpy.abs(epsy)**2 )**0.5
                #     eps = 1. / norm # in m, to convert the H wavelets into equivalent current wavelets
                #     epsx *= eps
                #     epsy *= eps

                # Obs selection around point P
                iobs = where((abs((mod(lon - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)) < self.DX) & 
                            (abs((lat - self.ENSLAT[P]) / km2deg) < self.DX))[0]
                xx = (mod(lon[iobs] - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)
                yy = (lat[iobs] - self.ENSLAT[P]) / km2deg

                #tt = time[iobs]

                # Spatial tapering shape of the wavelet and its derivative if velocity
                facs = mywindow(xx / self.DX) * mywindow(yy / self.DX)
                if ((compute_gc)|(compute_gceta)):
                    facs_epsx = (mywindow((xx+eps)/self.DX)*mywindow((yy)/self.DX)) 
                    facs_epsy = (mywindow((xx)/self.DX)*mywindow((yy+eps)/self.DX))
                    fc = 2*2*pi/86164*sin(self.ENSLAT[P]*pi/180.) # Coriolis parameter
                nobs=len(iobs)


                ntheta = len(self.theta[P])
                k=2*pi/self.Ltide[P]
                for itheta in range(ntheta):
                    kx=k*numpy.cos(self.theta[P][itheta])
                    ky=k*numpy.sin(self.theta[P][itheta])
                    for phasel in [0, pi / 2]:
                        for way in [1., -1.]:
                            iwave += 1
                            if ((iwave >= iwave0) & (iwave <iwave1)):
                                if (nobs > 0):
                                    if compute_gh:
                                        result[0][iwave-iwave0] = nobs
                                        result[1][ind_tmp:ind_tmp+nobs] = iobs
                                        # result[2][ind_tmp:ind_tmp+nobs] = Amp_astr[iobs]*facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])
                                        hh = Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))
                                        result[2][ind_tmp:ind_tmp+nobs] = numpy.real(hh*numpy.exp(1j*self.freq_tide*timejd[iobs]*86400)) 
                                        ind_tmp += nobs
                                    if compute_gc:
                                        result[0][iwave-iwave0] = nobs
                                        result[1][ind_tmp:ind_tmp+nobs] = iobs
                                        #hh = Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))
                                        hhx = ( Amp_astr[iobs]*facs_epsx * numpy.exp(1j*way*(kx*(xx+eps)+ky*(yy)+phasel + Phi_astr[iobs])) -
                                               Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))  ) / (eps*1000)
                                        hhy = ( Amp_astr[iobs]*facs_epsy * numpy.exp(1j*way*(kx*(xx)+ky*(yy+eps)+phasel + Phi_astr[iobs])) -
                                               Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))  ) / (eps*1000)                                  
                                        tmp = -10./(self.freq_tide**2-fc**2) * Amp_astr[iobs] * ( (-numpy.cos(angle[iobs])*fc + numpy.sin(angle[iobs])*1j*self.freq_tide)*hhy + (numpy.cos(angle[iobs])*1j*self.freq_tide + numpy.sin(angle[iobs])*fc)*hhx  ) 

                                        result[2][ind_tmp:ind_tmp+nobs] = numpy.real(tmp*numpy.exp(1j*self.freq_tide*timejd[iobs]*86400)) 


                                        ind_tmp += nobs

                                    if compute_gheta:
                                        if coordtype=='reg':
                                            result[iobs] += eta[iwave] * Amp_astr * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd*86400 + Phi_astr)
                                        else:
                                            result[iobs] += eta[iwave] * Amp_astr[iobs] * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])

                                    if compute_gceta:
                                        if coordtype=='reg':
                                            hhx = ( Amp_astr*facs_epsx * numpy.exp(1j*way*(kx*(xx+eps)+ky*(yy)+phasel + Phi_astr)) -
                                                Amp_astr*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr))  ) / (eps*1000)
                                            hhy = ( Amp_astr*facs_epsy * numpy.exp(1j*way*(kx*(xx)+ky*(yy+eps)+phasel + Phi_astr)) -
                                                Amp_astr*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr))  ) / (eps*1000)                            
                                            tmp = -10./(self.freq_tide**2-fc**2) * Amp_astr * ( (-numpy.cos(angle[iobs])*fc + numpy.sin(angle[iobs])*1j*self.freq_tide)*hhy + (numpy.cos(angle[iobs])*1j*self.freq_tide + numpy.sin(angle[iobs])*fc)*hhx ) 
                                            result[iobs] += eta[iwave] * numpy.real(tmp*numpy.exp(1j*self.freq_tide*timejd*86400))
                                        else:
                                            hhx = ( Amp_astr[iobs]*facs_epsx * numpy.exp(1j*way*(kx*(xx+eps)+ky*(yy)+phasel + Phi_astr[iobs])) -
                                                Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))  ) / (eps*1000)
                                            hhy = ( Amp_astr[iobs]*facs_epsy * numpy.exp(1j*way*(kx*(xx)+ky*(yy+eps)+phasel + Phi_astr[iobs])) -
                                                Amp_astr[iobs]*facs * numpy.exp(1j*way*(kx*(xx)+ky*(yy)+phasel + Phi_astr[iobs]))  ) / (eps*1000)                                  
                                            tmp = -10./(self.freq_tide**2-fc**2) * Amp_astr[iobs] * ( (-numpy.cos(angle[iobs])*fc + numpy.sin(angle[iobs])*1j*self.freq_tide)*hhy + (numpy.cos(angle[iobs])*1j*self.freq_tide + numpy.sin(angle[iobs])*fc)*hhx )  
                                            result[iobs] += eta[iwave] * numpy.real(tmp*numpy.exp(1j*self.freq_tide*timejd[iobs]*86400))                               


        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True: 
            logging.debug('END computing G: %s %s %s', self.name, obs_name, label )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result
        
        
class Comp_citv2(Comp, yaml.YAMLObject):

    """
    """



    #def __init__(self, lat_max=90.,**kwargs):
    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'lat_max' not in kwargs.keys(): self.lat_max = 90.
        #     kwargs['write']=True        logging.info('HELLO %s', lat_max)

        ## super(Comp_cit,self).__init__(**kwargs)

        self.ens_nature = ['sla'] # Ensemble of obs nature that project on the component
        self.name='iw_'+self.tidal_comp+'_mode'+str(self.mode)
        


    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   
        
    
    def set_basis(self, return_qinv=False):

        km2deg=1./110   

        deltat = self.TIME_MAX - self.TIME_MIN
                            
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2)) 

        finterpC = read_auxdata_geosc(self.filec_aux)

        DX = self.lambda_lw #* 0.5 #wavelet extension
        DXG = DX / self.facnls

        nwave=0
        lonmax=self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.       
        #for iff in range(nf):
        ENSLON=[]
        ENSLAT=[]  
        # if iff<nf-1:
        ENSLAT1 = arange(self.LAT_MIN-(DX-DXG)*km2deg,self.LAT_MAX+DX*km2deg,DXG*km2deg)
        ENSLAT1 = ENSLAT1[numpy.abs(ENSLAT1)<self.lat_max]
        for I in range(len(ENSLAT1)):
            ENSLON1 = mod(arange(self.LON_MIN -(DX-DXG)/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    lonmax+DX/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    DXG/cos(ENSLAT1[I]*pi/180.)*km2deg) , 360)
            ENSLAT=concatenate(([ENSLAT,repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON=concatenate(([ENSLON,ENSLON1]))


        NP = len(ENSLON)


        data_astr = pickle.load( open( self.file_data_astr, "rb" ) )
        self.freq_tide = data_astr[self.tidal_comp]['freq']
        self.Ttide = 2*numpy.pi/self.freq_tide/3600 # Hours

        self.Ltide = numpy.zeros((NP))
        #self.Ttide*3600*self.C/1000./self.mode # km

        self.theta=[None]*NP
        self.wavetest=numpy.full((NP),False, dtype=bool)
        iwave=-1
        for P in range(NP):
            dlon=DX*km2deg/cos(ENSLAT[P]*pi/180.)
            dlat=DX*km2deg
            elon=numpy.linspace(ENSLON[P]-dlon,ENSLON[P]+dlon/2,10)
            elat=numpy.linspace(ENSLAT[P]-dlat,ENSLAT[P]+dlat/2,10)
            elon2,elat2=numpy.meshgrid(elon,elat)
            tmp = finterpC((elon2.flatten(),elat2.flatten()))
            tmp = tmp[numpy.isnan(tmp)==False]
            if len(tmp)>0:
                C = numpy.nanmean(tmp)
            else: C=0.
            fc = 2*2*pi/86164*sin(ENSLAT[P]*pi/180.) # Coriolis parameter
            Cp = self.freq_tide / (self.freq_tide**2 - fc**2)**0.5 *C
            if ((Cp>self.cmin)): 
                self.wavetest[P]=True
                self.Ltide[P] = self.Ttide*3600*Cp/1000./self.mode # km

                npsp = 2*DX / self.Ltide[P]
                ntheta=max(int(npsp*self.facntheta), 1)
                self.theta[P] = linspace(0, pi, ntheta+1)[:-1]

                iwave += len(self.theta[P])*4

        nwave = iwave+1
        Q=zeros((nwave))

        varHlw = self.std_lw**2 * fcor
        Q[:]=varHlw/(self.facnls)



        self.data_Qinv = 1./Q

        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        #self.enst=enst
        self.nwave=nwave
        #self.tdec=tdec
        self.time0 = self.TIME_MIN # Reference in CNES julian days, will be used for astronomical coefficients
        


        if return_qinv:
            return 1./Q



    
    def operg(self, coords=None, coords_name=None, cdir=None, config=None, nature=None, compute_g=False, 
                compute_geta=False, eta=None, coordtype='scattered', iwave0=0, iwave1=None, obs_name=None, gsize_max=None, int_type='i8', float_type='f4', label=None):

        if iwave1==None: iwave1=self.nwave

        if compute_g==True: logging.debug('START computing G: %s %s %s', self.name, obs_name, label )

        km2deg=1./110 # A bouger, redondant

        compute_gh=False
        compute_gheta=False
        compute_gc=False
        compute_gceta=False
        if nature=='sla':
            if compute_g:
                compute_gh=True
            if compute_geta:
                compute_gheta=True
        if nature=='rcur':
            if compute_g:
                compute_gc=True
            if compute_geta:
                compute_gceta=True

        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]

        timejd=self.time0+time
        Amp_astr, Phi_astr = jd2ap(self.file_data_astr, self.tidal_comp, timejd)


        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(lon)))
            else:
                result = zeros((len(time)))

        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0

        iwave = -1
        # for iff in range(self.nf):
        #     logging.debug('progress: %d/%d', iff+1, self.nf)
        for P in range(self.NP):
            if self.wavetest[P]==True:
                # Obs selection around point P
                iobs = where((abs((mod(lon - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)) < self.DX) & 
                            (abs((lat - self.ENSLAT[P]) / km2deg) < self.DX))[0]
                xx = (mod(lon[iobs] - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)
                yy = (lat[iobs] - self.ENSLAT[P]) / km2deg

                #tt = time[iobs]

                # Spatial tapering shape of the wavelet and its derivative if velocity
                facs = mywindow(xx / self.DX) * mywindow(yy / self.DX)
                if ((compute_gc)|(compute_gceta)):
                    facs_eps = (mywindow((xx+epsx[iobs])/self.DX)*mywindow((yy+epsy[iobs])/self.DX))
                nobs=len(iobs)


                ntheta = len(self.theta[P])
                k=2*pi/self.Ltide[P]
                for itheta in range(ntheta):
                    kx=k*numpy.cos(self.theta[P][itheta])
                    ky=k*numpy.sin(self.theta[P][itheta])
                    for phasel in [0, pi / 2]:
                        for way in [1., -1.]:
                            iwave += 1
                            if ((iwave >= iwave0) & (iwave <iwave1)):
                                if (nobs > 0):
                                    if compute_gh:
                                        result[0][iwave-iwave0] = nobs
                                        result[1][ind_tmp:ind_tmp+nobs] = iobs
                                        result[2][ind_tmp:ind_tmp+nobs] = Amp_astr[iobs]*facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])
                                        ind_tmp += nobs
                                    if compute_gheta:
                                        if coordtype=='reg':
                                            result[iobs] += eta[iwave] * Amp_astr * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd*86400 + Phi_astr)
                                        else:
                                            result[iobs] += eta[iwave] * Amp_astr[iobs] * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])


        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True: 
            logging.debug('END computing G: %s %s %s', self.name, obs_name, label )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result
        
class Comp_citr(Comp, yaml.YAMLObject):

    """
    """



    #def __init__(self, lat_max=90.,**kwargs):
    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'lat_max' not in kwargs.keys(): self.lat_max = 90.
        #     kwargs['write']=True        logging.info('HELLO %s', lat_max)

        ## super(Comp_cit,self).__init__(**kwargs)

        self.ens_nature = ['sla'] # Ensemble of obs nature that project on the component
        self.name='iw_'+self.tidal_comp+'_mode'+str(self.mode)
        


    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   
        
    
    def set_basis(self, return_qinv=False):

        km2deg=1./110   

        deltat = self.TIME_MAX - self.TIME_MIN
                            
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2)) 

        finterpC = read_auxdata_geosc(self.filec_aux)
        finterpVARIANCE = read_auxdata_varcit(self.filevarcit_aux)

        DX = self.lambda_lw #* 0.5 #wavelet extension
        DXG = DX / self.facnls

        nwave=0
        lonmax=self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.       
        #for iff in range(nf):
        ENSLON=[]
        ENSLAT=[]  
        # if iff<nf-1:
        ENSLAT1 = arange(self.LAT_MIN-(DX-DXG)*km2deg,self.LAT_MAX+DX*km2deg,DXG*km2deg)
        ENSLAT1 = ENSLAT1[numpy.abs(ENSLAT1)<self.lat_max]
        for I in range(len(ENSLAT1)):
            ENSLON1 = mod(arange(self.LON_MIN -(DX-DXG)/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    lonmax+DX/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    DXG/cos(ENSLAT1[I]*pi/180.)*km2deg) , 360)
            ENSLAT=concatenate(([ENSLAT,repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON=concatenate(([ENSLON,ENSLON1]))


        NP = len(ENSLON)


        data_astr = pickle.load( open( self.file_data_astr, "rb" ) )
        self.freq_tide = data_astr[self.tidal_comp]['freq']
        self.Ttide = 2*numpy.pi/self.freq_tide/3600 # Hours

        self.Ltide = numpy.zeros((NP))
        #self.Ttide*3600*self.C/1000./self.mode # km

        self.theta=[None]*NP
        self.wavetest=numpy.full((NP),False, dtype=bool)
        Q=[]
        iwave=-1
        for P in range(NP):
            dlon=DX*km2deg/cos(ENSLAT[P]*pi/180.)
            dlat=DX*km2deg
            #
            elon=numpy.linspace(ENSLON[P]-dlon/2,ENSLON[P]+dlon/2,10)
            elat=numpy.linspace(ENSLAT[P]-dlat/2,ENSLAT[P]+dlat/2,10)
            elon2,elat2=numpy.meshgrid(elon,elat)
            tmp = finterpC((elon2.flatten(),elat2.flatten()))
            tmp = tmp[numpy.isnan(tmp)==False]
            if len(tmp)>0:
                C = numpy.nanmean(tmp)
            else: C=0.
            #
            elon=numpy.linspace(ENSLON[P]-dlon,ENSLON[P]+dlon,10)
            elat=numpy.linspace(ENSLAT[P]-dlat,ENSLAT[P]+dlat,10)
            elon2,elat2=numpy.meshgrid(elon,elat)
            tmp = finterpVARIANCE((elon2.flatten(),elat2.flatten()))
            tmp = tmp[numpy.isnan(tmp)==False]
            if len(tmp)>0:
                VARIANCE = numpy.maximum(numpy.nanmean(tmp), self.var_min)
            else: VARIANCE = self.var_min
            #
            fc = 2*2*pi/86164*sin(ENSLAT[P]*pi/180.) # Coriolis parameter
            Cp = self.freq_tide / (self.freq_tide**2 - fc**2)**0.5 *C
            if ((Cp>self.cmin)): 
                self.wavetest[P]=True
                self.Ltide[P] = self.Ttide*3600*Cp/1000./self.mode # km

                npsp = 2*DX / self.Ltide[P]
                ntheta=max(int(npsp*self.facntheta), 1)
                self.theta[P] = linspace(0, pi, ntheta+1)[:-1]
                Q.append(numpy.full((len(self.theta[P])*4), VARIANCE/self.facnls*self.fac_var))
                iwave += len(self.theta[P])*4

        nwave = iwave+1
        if nwave>0: Q=numpy.concatenate(Q)
        else: Q=numpy.array((0))




        self.data_Qinv = 1./Q # A retirer non?

        
        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        #self.enst=enst
        self.nwave=nwave
        #self.tdec=tdec
        self.time0 = self.TIME_MIN # Reference in CNES julian days, will be used for astronomical coefficients
        


        if return_qinv:
            return 1./Q



    
    def operg(self, coords=None, coords_name=None, cdir=None, config=None, nature=None, compute_g=False, 
                compute_geta=False, eta=None, coordtype='scattered', iwave0=0, iwave1=None, obs_name=None, gsize_max=None, int_type='i8', float_type='f4', label=None):

        if iwave1==None: iwave1=self.nwave

        if compute_g==True: logging.debug('START computing G: %s %s %s', self.name, obs_name, label )

        km2deg=1./110 # A bouger, redondant

        compute_gh=False
        compute_gheta=False
        compute_gc=False
        compute_gceta=False
        if nature=='sla':
            if compute_g:
                compute_gh=True
            if compute_geta:
                compute_gheta=True
        if nature=='rcur':
            if compute_g:
                compute_gc=True
            if compute_geta:
                compute_gceta=True

        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]

        timejd=self.time0+time
        Amp_astr, Phi_astr = jd2ap(self.file_data_astr, self.tidal_comp, timejd)


        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(lon)))
            else:
                result = zeros((len(time)))

        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0

        iwave = -1
        # for iff in range(self.nf):
        #     logging.debug('progress: %d/%d', iff+1, self.nf)
        for P in range(self.NP):
            if self.wavetest[P]==True:
                # Obs selection around point P
                iobs = where((abs((mod(lon - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)) < self.DX) & 
                            (abs((lat - self.ENSLAT[P]) / km2deg) < self.DX))[0]
                xx = (mod(lon[iobs] - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)
                yy = (lat[iobs] - self.ENSLAT[P]) / km2deg

                #tt = time[iobs]

                # Spatial tapering shape of the wavelet and its derivative if velocity
                facs = mywindow(xx / self.DX) * mywindow(yy / self.DX)
                if ((compute_gc)|(compute_gceta)):
                    facs_eps = (mywindow((xx+epsx[iobs])/self.DX)*mywindow((yy+epsy[iobs])/self.DX))
                nobs=len(iobs)


                ntheta = len(self.theta[P])
                k=2*pi/self.Ltide[P]
                for itheta in range(ntheta):
                    kx=k*numpy.cos(self.theta[P][itheta])
                    ky=k*numpy.sin(self.theta[P][itheta])
                    for phasel in [0, pi / 2]:
                        for way in [1., -1.]:
                            iwave += 1
                            if ((iwave >= iwave0) & (iwave <iwave1)):
                                if (nobs > 0):
                                    if compute_gh:
                                        result[0][iwave-iwave0] = nobs
                                        result[1][ind_tmp:ind_tmp+nobs] = iobs
                                        result[2][ind_tmp:ind_tmp+nobs] = Amp_astr[iobs]*facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])
                                        ind_tmp += nobs
                                    if compute_gheta:
                                        if coordtype=='reg':
                                            result[iobs] += eta[iwave] * Amp_astr * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd*86400 + Phi_astr)
                                        else:
                                            result[iobs] += eta[iwave] * Amp_astr[iobs] * facs * numpy.cos(kx*(xx)+ky*(yy)+phasel + way*self.freq_tide*timejd[iobs]*86400 + Phi_astr[iobs])


        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True: 
            logging.debug('END computing G: %s %s %s', self.name, obs_name, label )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result
      
        

def jd2ap(file_data_astr, tidal_comp, jd):  
  # CNES Julian days to Amp_astr and Phi_astr
  data_astr = pickle.load( open( file_data_astr, "rb" ) )
  freq_tide = data_astr[tidal_comp]['freq']

  finterpA = scipy.interpolate.interp1d(data_astr['time_jd'], data_astr[tidal_comp]['amp_astr'])
  finterpDP = scipy.interpolate.interp1d(data_astr['time_jd'], numpy.mod(data_astr[tidal_comp]['phi_astr']-data_astr[tidal_comp]['phi_astr'][0]+numpy.pi, 2*numpy.pi)-numpy.pi)


  Amp_astr = finterpA(jd)

  Phi_astr = finterpDP(jd) + data_astr[tidal_comp]['phi_astr'][0]


  return Amp_astr, Phi_astr
