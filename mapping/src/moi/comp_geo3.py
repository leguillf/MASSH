# -*- coding: utf-8 -*-
"""
"""
from numpy import pi, sqrt, arange, log, zeros, exp, linspace, array, where, cos, concatenate, outer, sin, repeat, minimum, empty, full, real, imag, mod, append, abs
from scipy.interpolate import interp2d
from netCDF4 import Dataset
from os import path, mkdir
import _pickle as pickle
#import h5py
import logging
from rw import read_auxdata_geos, read_auxdata_geosc, read_auxdata_depth
import pdb
from allcomps import Comp 
from tools import mywindow

import yaml
import scipy
import numpy

import dask.array



class Comp_geo3ss6d(Comp):
    # Comme geo3ss avec tdec seuillé autre Tdec et cut Ro non brutal With zonal distortion

    """
    """
    # __slots__ = ('Hvarname', 'Romax', 'cutRo', 'depth1', 'depth2',
    #              'distortion_eq', 'distortion_eq_law', 'facQ', 'facRo',
    #              'facnlt', 'facns', 'facpsp', 'factdec', 'file_aux',
    #              'filec_aux', 'lat_distortion_eq', 'lmax', 'lmin', 'npsp',
    #              'tdecmax', 'tdecmin', 'tssr', 'write', 'ens_nature','name','wavetest','DX','ENSLON','ENSLAT','NP','enst','nf',
    #              'theta','ntheta','ff','k','tdec','finterpDEPTH','finterpdist','nwave',
    #              'data_eta')


    # def __setstate__(self, state):
    #     self.__init__(**state)

    # def __getstate__(self):
    #     result = dict()
    #     for item in self.__slots__:
    #         if hasattr(self,item):
    #             result[item] = getattr(self, item)
    #         else:
    #             result[item] = None
    #     return result


    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        ###if 'facRo' not in kwargs.keys(): kwargs['facRo']=8.

        self.ens_nature = ['sla','rcur'] # Ensemble of obs nature that project on the component
        self.name='geo3ss'


    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   
     

    def set_basis(self, return_qinv=False):

        km2deg=1./110
        # Definition of the wavelet basis in the domain
        #params = config['PHYS_COMP_PROP']['GEOS']

        lat_tmp = numpy.arange(-90,90,0.1)
        alpha=(self.distortion_eq-1)*numpy.sin(self.lat_distortion_eq*numpy.pi/180)**self.distortion_eq_law
        finterpdist = scipy.interpolate.interp1d(lat_tmp, 1+alpha/(numpy.sin(numpy.maximum(self.lat_distortion_eq,numpy.abs(lat_tmp))*numpy.pi/180)**self.distortion_eq_law))
        #finterpdist = scipy.interpolate.interp1d(lat_tmp, 1+0.025/(numpy.sin(numpy.maximum(5,numpy.abs(lat_tmp))*numpy.pi/180)**2))

        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = arange(
            log(1./self.lmin ),
            log(1. / self.lmax) - log(1 + self.facpsp / self.npsp),
            -log(1 + self.facpsp / self.npsp))[::-1]
        # ff = zeros(logff.shape[0] + 1)
        # # Last frequency set to zero (the associated wavelet is just the taper function)
        # ff[:-1] = exp(logff)
        ff = exp(logff)
        #k = 2 * pi * ff
        dff = ff[1:] - ff[:-1]
        # Ensemble of directions for the wavelets (2D plane)
        theta = linspace(0, pi, int(pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf=len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./exp(logff))
        logging.info('ntheta: %s', ntheta)
        #pdb.set_trace()

        # Global time window
        deltat = self.TIME_MAX - self.TIME_MIN

        finterpPSDS, finterpTDEC, finterpNOISEFLOOR = read_auxdata_geos(self.file_aux)
        finterpC = read_auxdata_geosc(self.filec_aux)
        finterpDEPTH = read_auxdata_depth(self.filec_aux)

        # correction factor to compensate from amplitude increase with time-superimposing
        #dd = 0.4 # spacing factor between consecutive wavelets in time
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2))
        ###ns = 4 # spacing factor between consecutive wavelets in space, for large scales


        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        Cb1 = list() # First baroclinic phase speed

        DX = 1./ff*self.npsp * 0.5 #wavelet extension
        DXG = DX / self.facns #distance (km) between the wavelets grid in space



        NP = empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave=0
        lonmax=self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.
        for iff in range(nf):
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            # if iff<nf-1:
            ENSLAT1 = arange(self.LAT_MIN-(DX[iff]-DXG[iff])*km2deg,self.LAT_MAX+DX[iff]*km2deg,DXG[iff]*km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = mod(arange(self.LON_MIN -(DX[iff]-DXG[iff])/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        lonmax+DX[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        DXG[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I])) , 360)
                ENSLAT[iff]=concatenate(([ENSLAT[iff],repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON[iff]=concatenate(([ENSLON[iff],ENSLON1]))


            NP[iff] = len(ENSLON[iff])


            enst.append(list())
            tdec.append(list())
            Cb1.append(list())

            for P in range(NP[iff]):
                enst[-1].append(list())
                tdec[-1].append(list())
                Cb1[-1].append(list())

                # if iff==nf-1:
                #     tdec[-1][-1] = self.tdec_lw
                # else:
                dlon=DX[iff]*km2deg/cos(ENSLAT[iff][P] * pi / 180.)*finterpdist(ENSLAT[iff][P])
                dlat=DX[iff]*km2deg
                elon=numpy.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat=numpy.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2=numpy.meshgrid(elon,elat)
                tmp = finterpC((elon2.flatten(),elat2.flatten()))
                tmp = tmp[numpy.isnan(tmp)==False]
                if len(tmp)>0:
                    C = numpy.nanmean(finterpC((elon2.flatten(),elat2.flatten())))
                else: C=numpy.nan
                #tdec[-1][-1] = self.tdec_sw * 3./C
                if numpy.isnan(C): C=0.
                if C>100: pdb.set_trace()
                #if C<1: C=1.sparse_matrix
                Cb1[-1][-1] = C
                #test = abs(finterpPSDS((ff[1],ENSLAT[iff][P],ENSLON[iff][P])))
                #if numpy.isnan(test)==False: PSDLOC=test
                #print('PSDLOC',PSDLOC)
                #tdec_sw_lmin=self.tdec_sw_lmin * (PSDLOC/self.PSDR)**self.fpt
                #print('tdec_sw_lmin',tdec_sw_lmin)
                #print('Cref/C',self.Cref/C)
                #tdec[-1][-1] = (self.tdec_sw_lmax -(ff[iff]-ff[0])/(ff[-2]-ff[0])*(self.tdec_sw_lmax-tdec_sw_lmin)  ) * self.Cref/C

                fc=(2*2*pi/86164*sin(ENSLAT[iff][P]*pi/180.))
                Ro = C / numpy.abs(fc) /1000. # Rossby radius (km)
                #print('Ro',Ro)
                #print('C',C)
                if Ro>self.Romax: Ro=self.Romax
                if C>0: td1=self.factdec / (1./(self.facRo*Ro)*C/1000*86400)
                else: td1=numpy.nan
                #if C==0.: pdb.set_trace()
                PSDS = finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P]))
                if Ro>0: PSDSR = finterpPSDS((1./(self.facRo*Ro),ENSLAT[iff][P],ENSLON[iff][P]))
                else: PSDSR = numpy.nan
                if PSDS<=PSDSR: tdec[-1][-1] = td1 * (PSDS/PSDSR)**self.tssr
                else: tdec[-1][-1] = td1
                if tdec[-1][-1]>self.tdecmax: tdec[-1][-1]=self.tdecmax
                #if tdec[-1][-1]>100: pdb.set_trace()
                cp=1./(2*2*numpy.pi/86164*numpy.sin(max(10,numpy.abs(ENSLAT[iff][P]))*numpy.pi/180.))/300000
                tdecp=(1./ff[iff])*1000/cp/86400/4
                #print('tdecP', tdecp)
                if tdecp<tdec[-1][-1]: tdec[-1][-1]=tdecp

                #print('tdec',tdec[-1][-sparse_matrix1])
                #enst[-1][-1] = arange(-1.5*tdec[-1][-1],deltat+1.5*tdec[-1][-1],dd*tdec[-1][-1])
                try: enst[-1][-1] = arange(-tdec[-1][-1]*(1-1./self.facnlt) , deltat+tdec[-1][-1]/self.facnlt , tdec[-1][-1]/self.facnlt)
                except: pass
                nt = len(enst[iff][P])
                nwave += ntheta*2*nt


        # Fill the Q diagonal matrix (expected variace for each wavelet)
        self.wavetest=[None]*nf
        Q=zeros((nwave))
        iwave=-1
        ffx = outer(ff[:-1],cos(theta))
        ffy = outer(ff[:-1],sin(theta))
        self.iff_wavebounds = [None]*(nf+1)
        self.P_wavebounds = [None]*(nf+1)
        # Loop on all wavelets of given pseudo-period
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave+1
            self.P_wavebounds[iff] = [None]*(NP[iff]+1)
            self.wavetest[iff]=numpy.ones((NP[iff]), dtype=bool)
            for P in range(NP[iff]):
                self.P_wavebounds[iff][P] = iwave+1
                try: PSDLOC = abs(finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P])))
                except: pdb.set_trace()
                C = Cb1[iff][P]
                fc=(2*2*pi/86164*sin(ENSLAT[iff][P]*pi/180.))
                if fc==0: Ro=self.Romax
                else:
                    Ro = C / numpy.abs(fc) /1000.  # Rossby radius (km)
                    if Ro>self.Romax: Ro=self.Romax
                #if ((1./ff[iff] < self.cutRo * Ro) & (1./ff[iff] <self.lminmax)): self.wavetest[iff][P]=False
                if ((1./ff[iff] < self.cutRo * Ro) ): self.wavetest[iff][P]=False
                if tdec[iff][P]<self.tdecmin: self.wavetest[iff][P]=False
                if numpy.isnan(PSDLOC): self.wavetest[iff][P]=False
                if ((numpy.isnan(Cb1[iff][P]))|(Cb1[iff][P]==0)): self.wavetest[iff][P]=False
                if PSDLOC<=0:
                    pdb.set_trace()
                if self.wavetest[iff][P]==True:
                    #print('Cb1',Cb1[iff][P])
                    #print('tdec',tdec[iff][P])
                    for it in range(len(enst[iff][P])):
                        for itheta in range(len(theta)):
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ * numpy.exp(-3*(self.cutRo * Ro*ff[iff])**3)
                            if Q[iwave]<1e-20: pdb.set_trace()
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ* numpy.exp(-3*(self.cutRo * Ro*ff[iff])**3)
                            if numpy.isnan(Q[iwave]): pdb.set_trace()
            print(f'lambda={1/ff[iff]:.1E}',
                  f'nlocs={P:.1E}',
                  f'tdec={numpy.nanmean(tdec[iff]):.1E}',
                  f'Q={numpy.nanmean(Q[self.iff_wavebounds[iff]:iwave+1]):.1E}')

            self.P_wavebounds[iff][P+1] = iwave +1
        self.iff_wavebounds[iff+1] = iwave +1


        #if iwave+1 != nwave: pdb.set_trace()
        nwave = iwave+1
        Q=Q[:nwave]



        ##self.data = type('', (), {})()
        ###self.data_Qinv = 1./Q


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
        self.k = 2 * pi * ff
        self.tdec=tdec
        self.finterpDEPTH=finterpDEPTH

        self.finterpdist=finterpdist
        self.Cb1 = Cb1

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

        depth= -self.finterpDEPTH((lon,lat))
        depth[numpy.isnan(depth)]=0.



        if (nature=='rcur'):
            if cdir is None: angle = coords[coords_name['angle']]
            else: angle = full((len(lon)), cdir)
            eps = 0.01 # in km, to convert the H wavelets into equivalent current wavelets
            epsx = eps * cos(angle - pi/2)
            epsy = eps * sin(angle - pi/2)


        

        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(time),len(lon)))
            else:
                result = zeros((len(coord_time)))
        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0


        iwave = -1
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                if self.wavetest[iff][P]==True:
                    if ((iwave1>=self.P_wavebounds[iff][P])&(iwave0<self.P_wavebounds[iff][P+1])):
                        distortion=self.finterpdist(self.ENSLAT[iff][P])
                        # Obs selection around point P
                        iobs = where((abs((mod(lon - self.ENSLON[iff][P]+180,360)-180) / km2deg * cos(self.ENSLAT[iff][P] * pi / 180.))/distortion < self.DX[iff]) &
                                    (abs((lat - self.ENSLAT[iff][P]) / km2deg) < self.DX[iff]))[0]
                        xx = (mod(lon[iobs] - self.ENSLON[iff][P]+180,360)-180) / km2deg * cos(self.ENSLAT[iff][P] * pi / 180.) /distortion
                        yy = (lat[iobs] - self.ENSLAT[iff][P]) / km2deg

                        # Spatial tapering shape of the wavelet and its derivative if velocity
                        facd=numpy.ones((len(iobs)))
                        facd = (depth[iobs]-self.depth1)/(self.depth2-self.depth1)
                        facd[facd>1]=1.
                        facd[facd<0]=0.
                        facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd
                        if ((compute_gc)|(compute_gceta)):
                            facs_eps = (mywindow((xx+epsx[iobs])/self.DX[iff])*mywindow((yy+epsy[iobs])/self.DX[iff])) * facd
                            fc = 2*2*pi/86164*sin(self.ENSLAT[iff][P]*pi/180.) # Coriolis parameter
                    else: iobs=numpy.empty((0))

                    enstloc = self.enst[iff][P]
                    for it in range(len(enstloc)):
                        nobs = 0
                        iiobs=[]
                        if iobs.shape[0] > 0:
                            if coordtype=='reg':
                                diff = time - enstloc[it]
                                iobs2 = where(abs(diff) < self.tdec[iff][P])[0]
                                nobs = len(iobs2)
                            else:
                                diff = time[iobs] - enstloc[it]
                                iobs2 = where(abs(diff) < self.tdec[iff][P])[0]
                                iiobs = iobs[iobs2]
                                nobs = iiobs.shape[0]
                            # diff = time[iobs] - enstloc[it]
                            # iobs2 = where(abs(diff) < 2 * self.tdec[iff][P])[0]
                            # iiobs = iobs[iobs2]
                            # nobs = iiobs.shape[0]
                            if nobs > 0:
                                tt2 = diff[iobs2]
                                #fact = exp(-2 * tt2 ** 2 / self.tdec[iff][P] ** 2)
                                fact = mywindow(tt2 / self.tdec[iff][P])

                        for itheta in range(self.ntheta):
                            kx = self.k[iff] * cos(self.theta[itheta])
                            ky = self.k[iff] * sin(self.theta[itheta])
                            for phase in [0, pi / 2]:
                                iwave += 1
                                if ((iwave >= iwave0) & (iwave <iwave1)):
                                    #if ((nobs > 0)&(self.data.Qinv[iwave]>0)):
                                    if ((nobs > 0)):
                                        if compute_gh:
                                            result[0][iwave-iwave0] = nobs
                                            try:
                                                result[1][ind_tmp:ind_tmp+nobs] = iiobs
                                            except:
                                                print()
                                            result[2][ind_tmp:ind_tmp+nobs] = sqrt(2)*cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2]*fact
                                            ind_tmp += nobs
                                        if compute_gc:
                                            result[0][iwave-iwave0] = nobs
                                            result[1][ind_tmp:ind_tmp+nobs] = iiobs
                                            result[2][ind_tmp:ind_tmp+nobs] = 10./fc*sqrt(2)*fact*(
                                                                            cos(kx*(xx[iobs2]+epsx[iiobs])+ky*(yy[iobs2]+epsy[iiobs])-phase)*facs_eps[iobs2]
                                                                            - cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2] ) / (eps*1000)
                                            ind_tmp += nobs

                                        if compute_gheta:
                                            if coordtype=='reg':
                                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * sqrt(2)* outer(fact , cos(kx*(xx)+ky*(yy)-phase)*facs)
                                            else:
                                                result[iiobs] += eta[iwave] * sqrt(2)*cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2]*fact
                                        if compute_gceta:
                                            if coordtype=='reg':
                                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * 10./fc*sqrt(2)* outer( fact,
                                                                    (cos(kx*(xx+epsx[iobs])+ky*(yy+epsy[iobs])-phase)*facs_eps
                                                                    - cos(kx*(xx)+ky*(yy)-phase)*facs )
                                                                    / (eps*1000))
                                            else:
                                                result[iiobs] += eta[iwave] * (10./fc*sqrt(2)*fact*(
                                                                            cos(kx*(xx[iobs2]+epsx[iiobs])+ky*(yy[iobs2]+epsy[iiobs])-phase)*facs_eps[iobs2]
                                                                            - cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2] )
                                                                            / (eps*1000))

        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True:           
            logging.debug('END computing G: %s %s %s gsize: %s', self.name, obs_name, label, ind_tmp )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result



class Comp_geo3ls(Comp, yaml.YAMLObject):
    # For large scales

    """
    """


    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.ens_nature = ['sla','rcur'] # Ensemble of obs nature that project on the component
        self.name='geo3ls'

    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   

    def set_basis(self, return_qinv=False):

        km2deg=1./110
        # # Definition of the wavelet basis in the domain
        # #params = config['PHYS_COMP_PROP']['GEOS']

        # # Ensemble of pseudo-frequencies for the wavelets (spatial)
        # logff = arange(
        #     log(1./self.lmin ),
        #     log(1. / self.lmax) - log(1 + self.facpsp / self.npsp),
        #     -log(1 + self.facpsp / self.npsp))[::-1]
        # ff = zeros(logff.shape[0] + 1)
        # # Last frequency set to zero (the associated wavelet is just the taper function)
        # ff[:-1] = exp(logff)
        # #k = 2 * pi * ff
        # dff = ff[1:] - ff[:-1]
        # # Ensemble of directions for the wavelets (2D plane)
        # theta = linspace(0, pi, pi * ff[0] / dff[0] * self.facpsp)[:-1]
        # ntheta = len(theta)
        # nf=len(ff)
        # logging.info('spatial normalized wavelengths: %s', 1./exp(logff))
        # logging.info('ntheta: %s', ntheta)
        # #pdb.set_trace()

        # Global time window
        deltat = self.TIME_MAX - self.TIME_MIN

        # finterpPSDS, finterpTDEC, finterpNOISEFLOOR = read_auxdata_geos(self.file_aux)
        # finterpC = read_auxdata_geosc(self.filec_aux)

        # correction factor to compensate from amplitude increase with time-superimposing
        #dd = 0.4 # spacing factor between consecutive wavelets in time
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2))
        ###ns = 4 # spacing factor between consecutive wavelets in space, for large scales


        # ENSLON = [None] # Ensemble of longitudes of the center of each wavelets
        # ENSLAT = [None] # Ensemble of latitudes of the center of each wavelets
        # enst = list() #  Ensemble of times of the center of each wavelets
        # tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.

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
        for I in range(len(ENSLAT1)):
            ENSLON1 = mod(arange(self.LON_MIN -(DX-DXG)/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    lonmax+DX/cos(ENSLAT1[I]*pi/180.)*km2deg,
                    DXG/cos(ENSLAT1[I]*pi/180.)*km2deg) , 360)
            ENSLAT=concatenate(([ENSLAT,repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON=concatenate(([ENSLON,ENSLON1]))

            # else:
            #     ENSLAT1 = arange(grid.LAT_MIN-DX[iff]*km2deg,grid.LAT_MAX+DXG[iff]*km2deg,DXG[iff]*km2deg/self.facnls)
            #     for I in range(len(ENSLAT1)):

            #         ENSLON1 = mod(arange(grid.LON_MIN-DXG[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg*(1-1./self.facnls),
            #                 lonmax+DXG[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg,
            #                 DXG[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg/self.facnls) ,360.)

            #         ENSLAT[iff]=concatenate(([ENSLAT[iff],repeat(ENSLAT1[I],len(ENSLON1))]))
            #         ENSLON[iff]=concatenate(([ENSLON[iff],ENSLON1]))

        NP = len(ENSLON)


        enst=[None]*NP
        tdec=[None]*NP
        for P in range(NP):
            tdec[P] = self.tdec_lw

            enst[P] = arange(-tdec[P]*(1-1./self.facnlt) , deltat+tdec[P]/self.facnlt , tdec[P]/self.facnlt)
            nt = len(enst[P])
            nwave += nt


        # Fill the Q diagonal matrix (expected variace for each wavelet)

        Q=zeros((nwave))
        iwave=-1
        self.P_wavebounds = [None]*(NP+1)
        varHlw = self.std_lw**2 * fcor
        for P in range(NP):
            self.P_wavebounds[P] = iwave+1
            for it in range(len(enst[P])):
                iwave += 1
                Q[iwave]=varHlw/(self.facnls*self.facnlt)
        self.P_wavebounds[P+1] = iwave +1
        if iwave+1 != nwave: pdb.set_trace()


        ###self.data_Qinv = 1./Q


        self.DX=DX
        self.ENSLON=ENSLON
        self.ENSLAT=ENSLAT
        self.NP=NP
        self.enst=enst
        self.nwave=nwave
        self.tdec=tdec


        if return_qinv==True:
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

        if (nature=='rcur'):
            if cdir is None: angle = coords[coords_name['angle']]
            else: angle = full((len(lon)), cdir)
            eps = 0.01 # in km, to convert the H wavelets into equivalent current wavelets
            epsx = eps * cos(angle - pi/2)
            epsy = eps * sin(angle - pi/2)

        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(time),len(lon)))
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
            if ((iwave1>=self.P_wavebounds[P])&(iwave0<self.P_wavebounds[P+1])):
                # Obs selection around point P
                iobs = where((abs((mod(lon - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)) < self.DX) &
                            (abs((lat - self.ENSLAT[P]) / km2deg) < self.DX))[0]
                xx = (mod(lon[iobs] - self.ENSLON[P]+180,360)-180) / km2deg * cos(self.ENSLAT[P] * pi / 180.)
                yy = (lat[iobs] - self.ENSLAT[P]) / km2deg

                # Spatial tapering shape of the wavelet and its derivative if velocity
                facs = mywindow(xx / self.DX) * mywindow(yy / self.DX)
                if ((compute_gc)|(compute_gceta)):
                    facs_eps = (mywindow((xx+epsx[iobs])/self.DX)*mywindow((yy+epsy[iobs])/self.DX))
                    fc = 2*2*pi/86164*sin(self.ENSLAT[P]*pi/180.) # Coriolis parameter
            else: iobs=numpy.empty((0))

            enstloc = self.enst[P]
            for it in range(len(enstloc)):
                nobs = 0
                iiobs=[]
                if iobs.shape[0] > 0:
                    if coordtype=='reg':
                        diff = time - enstloc[it]
                        iobs2 = where(abs(diff) < self.tdec[P])[0]
                        nobs = len(iobs2)
                    else:
                        diff = time[iobs] - enstloc[it]
                        iobs2 = where(abs(diff) < self.tdec[P])[0]
                        iiobs = iobs[iobs2]
                        nobs = iiobs.shape[0]
                    # diff = time[iobs] - enstloc[it]
                    # iobs2 = where(abs(diff) < 2 * self.tdec[iff][P])[0]
                    # iiobs = iobs[iobs2]
                    # nobs = iiobs.shape[0]
                    if nobs > 0:
                        tt2 = diff[iobs2]
                        #fact = exp(-2 * tt2 ** 2 / self.tdec[iff][P] ** 2)
                        fact = mywindow(tt2 / self.tdec[P])


                iwave += 1
                if ((iwave >= iwave0) & (iwave <iwave1)):
                    if (nobs > 0):#&(self.data.Qinv[iwave]>0)):
                        if compute_gh:
                            result[0][iwave-iwave0] = nobs
                            result[1][ind_tmp:ind_tmp+nobs] = iiobs
                            result[2][ind_tmp:ind_tmp+nobs] = (facs[iobs2]*fact)**2
                            ind_tmp += nobs
                        if compute_gc:
                            result[0][iwave-iwave0] = nobs
                            result[1][ind_tmp:ind_tmp+nobs] = iiobs
                            result[2][ind_tmp:ind_tmp+nobs] = (10./fc*fact**2* (facs_eps[iobs2]**2 - facs[iobs2]**2) / (eps*1000))
                            ind_tmp += nobs
                        if compute_gheta:
                            if coordtype=='reg':
                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * outer(fact**2 , facs**2)
                            else:
                                result[iiobs] += eta[iwave] * (facs[iobs2]*fact)**2
                        if compute_gceta:
                            if coordtype=='reg':
                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * 10./fc * outer(fact**2 , (facs_eps**2 - facs**2) / (eps*1000) )
                            else:
                                result[iiobs] += eta[iwave] * 10./fc*fact**2* (facs_eps[iobs2]**2 - facs[iobs2]**2) / (eps*1000)



        # if compute_g==True: 
        #     logging.info('END computing G: %s %s %s %s', self.name, obs_name, iwave0, iwave1 )
        #     result = result[:ind_tmp]
        # if iwave+1 != self.nwave: pdb.set_trace()

        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True: 
            logging.debug('END computing G: %s %s %s', self.name, obs_name, label )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result


class Comp_geodyn(Comp):

    """
    """



    def __init__(self,**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        ###if 'facRo' not in kwargs.keys(): kwargs['facRo']=8.

        self.ens_nature = ['sla','rcur'] # Ensemble of obs nature that project on the component
        self.name='geodyn'


    def set_domain(self, grid):

        self.TIME_MIN = grid.TIME_MIN
        self.TIME_MAX = grid.TIME_MAX
        self.LON_MIN = grid.LON_MIN
        self.LON_MAX = grid.LON_MAX
        self.LAT_MIN = grid.LAT_MIN
        self.LAT_MAX = grid.LAT_MAX   
     

    def set_basis(self, return_qinv=False):

        km2deg=1./110
        # Definition of the wavelet basis in the domain
        #params = config['PHYS_COMP_PROP']['GEOS']

        lat_tmp = numpy.arange(-90,90,0.1)
        alpha=(self.distortion_eq-1)*numpy.sin(self.lat_distortion_eq*numpy.pi/180)**self.distortion_eq_law
        finterpdist = scipy.interpolate.interp1d(lat_tmp, 1+alpha/(numpy.sin(numpy.maximum(self.lat_distortion_eq,numpy.abs(lat_tmp))*numpy.pi/180)**self.distortion_eq_law))
        #finterpdist = scipy.interpolate.interp1d(lat_tmp, 1+0.025/(numpy.sin(numpy.maximum(5,numpy.abs(lat_tmp))*numpy.pi/180)**2))

        # Ensemble of pseudo-frequencies for the wavelets (spatial)
        logff = arange(
            log(1./self.lmin ),
            log(1. / self.lmax) - log(1 + self.facpsp / self.npsp),
            -log(1 + self.facpsp / self.npsp))[::-1]
        # ff = zeros(logff.shape[0] + 1)
        # # Last frequency set to zero (the associated wavelet is just the taper function)
        # ff[:-1] = exp(logff)
        ff = exp(logff)
        #k = 2 * pi * ff
        dff = ff[1:] - ff[:-1]
        # Ensemble of directions for the wavelets (2D plane)
        theta = linspace(0, pi, int(pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf=len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./exp(logff))
        logging.info('ntheta: %s', ntheta)
        #pdb.set_trace()

        # Global time window
        deltat = self.TIME_MAX - self.TIME_MIN

        finterpPSDS, finterpTDEC, finterpNOISEFLOOR = read_auxdata_geos(self.file_aux)
        finterpC = read_auxdata_geosc(self.filec_aux)
        finterpDEPTH = read_auxdata_depth(self.filec_aux)

        # correction factor to compensate from amplitude increase with time-superimposing
        #dd = 0.4 # spacing factor between consecutive wavelets in time
        fcor = 0.5# 1./(1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2))
        ###ns = 4 # spacing factor between consecutive wavelets in space, for large scales


        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        enst = list() #  Ensemble of times of the center of each wavelets
        tdec = list() # Ensemble of equivalent decorrelation times. Used to define enst.
        Cb1 = list() # First baroclinic phase speed

        DX = 1./ff*self.npsp * 0.5 #wavelet extension
        DXG = DX / self.facns #distance (km) between the wavelets grid in space



        NP = empty(nf, dtype='int16') # Nomber of spatial wavelet locations for a given frequency
        nwave=0
        lonmax=self.LON_MAX
        if (self.LON_MAX<self.LON_MIN): lonmax=self.LON_MAX+360.
        for iff in range(nf):
            ENSLON[iff]=[]
            ENSLAT[iff]=[]
            # if iff<nf-1:
            ENSLAT1 = arange(self.LAT_MIN-(DX[iff]-DXG[iff])*km2deg,self.LAT_MAX+DX[iff]*km2deg,DXG[iff]*km2deg)
            for I in range(len(ENSLAT1)):
                ENSLON1 = mod(arange(self.LON_MIN -(DX[iff]-DXG[iff])/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        lonmax+DX[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I]),
                        DXG[iff]/cos(ENSLAT1[I]*pi/180.)*km2deg*finterpdist(ENSLAT1[I])) , 360)
                ENSLAT[iff]=concatenate(([ENSLAT[iff],repeat(ENSLAT1[I],len(ENSLON1))]))
                ENSLON[iff]=concatenate(([ENSLON[iff],ENSLON1]))


            NP[iff] = len(ENSLON[iff])


            enst.append(list())
            tdec.append(list())
            Cb1.append(list())

            for P in range(NP[iff]):
                enst[-1].append(list())
                tdec[-1].append(list())
                Cb1[-1].append(list())

                # if iff==nf-1:
                #     tdec[-1][-1] = self.tdec_lw
                # else:
                dlon=DX[iff]*km2deg/cos(ENSLAT[iff][P] * pi / 180.)*finterpdist(ENSLAT[iff][P])
                dlat=DX[iff]*km2deg
                elon=numpy.linspace(ENSLON[iff][P]-dlon,ENSLON[iff][P]+dlon,10)
                elat=numpy.linspace(ENSLAT[iff][P]-dlat,ENSLAT[iff][P]+dlat,10)
                elon2,elat2=numpy.meshgrid(elon,elat)
                tmp = finterpC((elon2.flatten(),elat2.flatten()))
                tmp = tmp[numpy.isnan(tmp)==False]
                if len(tmp)>0:
                    C = numpy.nanmean(finterpC((elon2.flatten(),elat2.flatten())))
                else: C=numpy.nan
                #tdec[-1][-1] = self.tdec_sw * 3./C
                if numpy.isnan(C): C=0.
                if C>100: pdb.set_trace()
                #if C<1: C=1.sparse_matrix
                Cb1[-1][-1] = C
                #test = abs(finterpPSDS((ff[1],ENSLAT[iff][P],ENSLON[iff][P])))
                #if numpy.isnan(test)==False: PSDLOC=test
                #print('PSDLOC',PSDLOC)
                #tdec_sw_lmin=self.tdec_sw_lmin * (PSDLOC/self.PSDR)**self.fpt
                #print('tdec_sw_lmin',tdec_sw_lmin)
                #print('Cref/C',self.Cref/C)
                #tdec[-1][-1] = (self.tdec_sw_lmax -(ff[iff]-ff[0])/(ff[-2]-ff[0])*(self.tdec_sw_lmax-tdec_sw_lmin)  ) * self.Cref/C

                fc=(2*2*pi/86164*sin(ENSLAT[iff][P]*pi/180.))
                Ro = C / numpy.abs(fc) /1000. # Rossby radius (km)
                #print('Ro',Ro)
                #print('C',C)
                if Ro>self.Romax: Ro=self.Romax
                if C>0: td1=self.factdec / (1./(self.facRo*Ro)*C/1000*86400)
                else: td1=numpy.nan
                #if C==0.: pdb.set_trace()
                PSDS = finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P]))
                if Ro>0: PSDSR = finterpPSDS((1./(self.facRo*Ro),ENSLAT[iff][P],ENSLON[iff][P]))
                else: PSDSR = numpy.nan
                if PSDS<=PSDSR: tdec[-1][-1] = td1 * (PSDS/PSDSR)**self.tssr
                else: tdec[-1][-1] = td1
                if tdec[-1][-1]>self.tdecmax: tdec[-1][-1]=self.tdecmax
                #if tdec[-1][-1]>100: pdb.set_trace()
                cp=1./(2*2*numpy.pi/86164*numpy.sin(max(10,numpy.abs(ENSLAT[iff][P]))*numpy.pi/180.))/300000
                tdecp=(1./ff[iff])*1000/cp/86400/4
                #print('tdecP', tdecp)
                if tdecp<tdec[-1][-1]: tdec[-1][-1]=tdecp

                #print('tdec',tdec[-1][-sparse_matrix1])
                #enst[-1][-1] = arange(-1.5*tdec[-1][-1],deltat+1.5*tdec[-1][-1],dd*tdec[-1][-1])
                try: enst[-1][-1] = arange(-tdec[-1][-1]*(1-1./self.facnlt) , deltat+tdec[-1][-1]/self.facnlt , tdec[-1][-1]/self.facnlt)
                except: pass
                nt = len(enst[iff][P])
                nwave += ntheta*2*nt

        # Fill the Q diagonal matrix (expected variace for each wavelet)
        self.wavetest=[None]*nf
        Q=zeros((nwave))
        iwave=-1
        ffx = outer(ff[:-1],cos(theta))
        ffy = outer(ff[:-1],sin(theta))
        self.iff_wavebounds = [None]*(nf+1)
        self.P_wavebounds = [None]*(nf+1)
        # Loop on all wavelets of given pseudo-period
        for iff in range(nf):
            self.iff_wavebounds[iff] = iwave+1
            self.P_wavebounds[iff] = [None]*(NP[iff]+1)
            self.wavetest[iff]=numpy.ones((NP[iff]), dtype=bool)
            for P in range(NP[iff]):
                self.P_wavebounds[iff][P] = iwave+1
                try: PSDLOC = abs(finterpPSDS((ff[iff],ENSLAT[iff][P],ENSLON[iff][P])))
                except: pdb.set_trace()
                C = Cb1[iff][P]
                fc=(2*2*pi/86164*sin(ENSLAT[iff][P]*pi/180.))
                if fc==0: Ro=self.Romax
                else:
                    Ro = C / numpy.abs(fc) /1000.  # Rossby radius (km)
                    if Ro>self.Romax: Ro=self.Romax
                #if ((1./ff[iff] < self.cutRo * Ro) & (1./ff[iff] <self.lminmax)): self.wavetest[iff][P]=False
                if ((1./ff[iff] < self.cutRo * Ro) ): self.wavetest[iff][P]=False
                if tdec[iff][P]<self.tdecmin: self.wavetest[iff][P]=False
                if numpy.isnan(PSDLOC): self.wavetest[iff][P]=False
                if ((numpy.isnan(Cb1[iff][P]))|(Cb1[iff][P]==0)): self.wavetest[iff][P]=False
                if PSDLOC<=0:
                    pdb.set_trace()
                if self.wavetest[iff][P]==True:
                    #print('Cb1',Cb1[iff][P])
                    #print('tdec',tdec[iff][P])
                    for it in range(len(enst[iff][P])):
                        for itheta in range(len(theta)):
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ * numpy.exp(-3*(self.cutRo * Ro*ff[iff])**3)
                            if Q[iwave]<1e-20: pdb.set_trace()
                            iwave += 1
                            Q[iwave] = PSDLOC*ff[iff]**2 * self.facQ* numpy.exp(-3*(self.cutRo * Ro*ff[iff])**3)
                            if numpy.isnan(Q[iwave]): pdb.set_trace()

            self.P_wavebounds[iff][P+1] = iwave +1
        self.iff_wavebounds[iff+1] = iwave +1


        #if iwave+1 != nwave: pdb.set_trace()
        nwave = iwave+1
        Q=Q[:nwave]



        ##self.data = type('', (), {})()
        ###self.data_Qinv = 1./Q


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
        self.k = 2 * pi * ff
        self.tdec=tdec
        self.finterpDEPTH=finterpDEPTH

        self.finterpdist=finterpdist
        self.Cb1 = Cb1

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

        depth= -self.finterpDEPTH((lon,lat))
        depth[numpy.isnan(depth)]=0.



        if (nature=='rcur'):
            if cdir is None: angle = coords[coords_name['angle']]
            else: angle = full((len(lon)), cdir)
            eps = 0.01 # in km, to convert the H wavelets into equivalent current wavelets
            epsx = eps * cos(angle - pi/2)
            epsy = eps * sin(angle - pi/2)


        

        if compute_geta:
            if coordtype=='reg':
                result = zeros((len(time),len(lon)))
            else:
                result = zeros((len(coord_time)))
        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0


        iwave = -1
        for iff in range(self.nf):
            for P in range(self.NP[iff]):
                if self.wavetest[iff][P]==True:
                    if ((iwave1>=self.P_wavebounds[iff][P])&(iwave0<self.P_wavebounds[iff][P+1])):
                        distortion=self.finterpdist(self.ENSLAT[iff][P])
                        # Obs selection around point P
                        iobs = where((abs((mod(lon - self.ENSLON[iff][P]+180,360)-180) / km2deg * cos(self.ENSLAT[iff][P] * pi / 180.))/distortion < self.DX[iff]) &
                                    (abs((lat - self.ENSLAT[iff][P]) / km2deg) < self.DX[iff]))[0]
                        xx = (mod(lon[iobs] - self.ENSLON[iff][P]+180,360)-180) / km2deg * cos(self.ENSLAT[iff][P] * pi / 180.) /distortion
                        yy = (lat[iobs] - self.ENSLAT[iff][P]) / km2deg

                        # Spatial tapering shape of the wavelet and its derivative if velocity
                        facd=numpy.ones((len(iobs)))
                        facd = (depth[iobs]-self.depth1)/(self.depth2-self.depth1)
                        facd[facd>1]=1.
                        facd[facd<0]=0.
                        facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd
                        if ((compute_gc)|(compute_gceta)):
                            facs_eps = (mywindow((xx+epsx[iobs])/self.DX[iff])*mywindow((yy+epsy[iobs])/self.DX[iff])) * facd
                            fc = 2*2*pi/86164*sin(self.ENSLAT[iff][P]*pi/180.) # Coriolis parameter
                    else: iobs=numpy.empty((0))

                    enstloc = self.enst[iff][P]
                    for it in range(len(enstloc)):
                        nobs = 0
                        iiobs=[]
                        if iobs.shape[0] > 0:
                            if coordtype=='reg':
                                diff = time - enstloc[it]
                                iobs2 = where(abs(diff) < self.tdec[iff][P])[0]
                                nobs = len(iobs2)
                            else:
                                diff = time[iobs] - enstloc[it]
                                iobs2 = where(abs(diff) < self.tdec[iff][P])[0]
                                iiobs = iobs[iobs2]
                                nobs = iiobs.shape[0]
                            # diff = time[iobs] - enstloc[it]
                            # iobs2 = where(abs(diff) < 2 * self.tdec[iff][P])[0]
                            # iiobs = iobs[iobs2]
                            # nobs = iiobs.shape[0]
                            if nobs > 0:
                                tt2 = diff[iobs2]
                                #fact = exp(-2 * tt2 ** 2 / self.tdec[iff][P] ** 2)
                                fact = mywindow(tt2 / self.tdec[iff][P])

                        for itheta in range(self.ntheta):
                            kx = self.k[iff] * cos(self.theta[itheta])
                            ky = self.k[iff] * sin(self.theta[itheta])
                            for phase in [0, pi / 2]:
                                iwave += 1
                                if ((iwave >= iwave0) & (iwave <iwave1)):
                                    #if ((nobs > 0)&(self.data.Qinv[iwave]>0)):
                                    if ((nobs > 0)):
                                        if compute_gh:
                                            result[0][iwave-iwave0] = nobs
                                            result[1][ind_tmp:ind_tmp+nobs] = iiobs
                                            result[2][ind_tmp:ind_tmp+nobs] = sqrt(2)*cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2]*fact
                                            ind_tmp += nobs
                                        if compute_gc:
                                            result[0][iwave-iwave0] = nobs
                                            result[1][ind_tmp:ind_tmp+nobs] = iiobs
                                            result[2][ind_tmp:ind_tmp+nobs] = 10./fc*sqrt(2)*fact*(
                                                                            cos(kx*(xx[iobs2]+epsx[iiobs])+ky*(yy[iobs2]+epsy[iiobs])-phase)*facs_eps[iobs2]
                                                                            - cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2] ) / (eps*1000)
                                            ind_tmp += nobs

                                        if compute_gheta:
                                            if coordtype=='reg':
                                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * sqrt(2)* outer(fact , cos(kx*(xx)+ky*(yy)-phase)*facs)
                                            else:
                                                result[iiobs] += eta[iwave] * sqrt(2)*cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2]*fact
                                        if compute_gceta:
                                            if coordtype=='reg':
                                                result[iobs2[0]:iobs2[-1]+1,iobs] += eta[iwave] * 10./fc*sqrt(2)* outer( fact,
                                                                    (cos(kx*(xx+epsx[iobs])+ky*(yy+epsy[iobs])-phase)*facs_eps
                                                                    - cos(kx*(xx)+ky*(yy)-phase)*facs )
                                                                    / (eps*1000))
                                            else:
                                                result[iiobs] += eta[iwave] * (10./fc*sqrt(2)*fact*(
                                                                            cos(kx*(xx[iobs2]+epsx[iiobs])+ky*(yy[iobs2]+epsy[iiobs])-phase)*facs_eps[iobs2]
                                                                            - cos(kx*(xx[iobs2])+ky*(yy[iobs2])-phase)*facs[iobs2] )
                                                                            / (eps*1000))

        if iwave+1 != self.nwave: pdb.set_trace()

        if compute_g==True:           
            logging.debug('END computing G: %s %s %s gsize: %s', self.name, obs_name, label, ind_tmp )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result

