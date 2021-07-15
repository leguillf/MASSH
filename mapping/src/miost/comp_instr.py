# -*- coding: utf-8 -*-
"""
"""
from numpy import pi, exp, where, arange, where, concatenate, zeros, empty, delete, inner, append, full, sum, log10, cos, sin
import numpy
from scipy.interpolate import interp1d
from allcomps import Comp
import yaml
import pdb
from tools import mywindow
from rw import read_auxdata_depth
from scipy.sparse import csc_matrix, coo_matrix
from netCDF4 import Dataset
import logging
from mpi4py import MPI

class CompNlwe(Comp, yaml.YAMLObject):

    yaml_tag = u'!CompNlwe'


    # def __setstate__(self, state):
    #     self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompNlwe,self).__init__(**kwargs)

        ###self.float_type = 'f2'
        self.ens_nature = ['sla'] # Ensemble of obs nature that project on the component

    #def set_domain(self,name, time0, time1):
    def set_domain(self,obs,obs_data,grid,comm):  
        self.name='nlwe_'+ '+'.join([obs[ko].name for ko in range(len(obs))])
        time0=0.
        time1=grid.TIME_MAX-grid.TIME_MIN
        dd=0.5
        self.fcor = 1. / (1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2))
        lwtdec = self.TDEC
        enst=arange(-1.5*lwtdec + time0, time1+1.5*lwtdec,dd*lwtdec)

        indmask = numpy.ones((len(enst)), dtype=bool)

        for ko in range(len(obs)): # note: only obs related to the component, generally 1 only
            coords_time = obs_data[ko][2][obs_data[ko][3]['time']]

            iobsr=where(((coords_time[1:] - coords_time[:-1])>2*lwtdec))[0]
            indkill=empty([0], dtype=int)
            for k in range(len(iobsr)):
                indk=where((enst[-1]-coords_time[iobsr[k]]>2*lwtdec)&(coords_time[iobsr[k]+1]-enst[-1]>2*lwtdec))[0]
                indkill=concatenate(([indkill,indk]),axis=0)
            indk = where((coords_time[0]-enst > 2*lwtdec))[0]
            indkill = concatenate(([indkill,indk]),axis=0)
            indk = where((enst - coords_time[-1] > 2*lwtdec))[0]
            indkill = concatenate(([indkill,indk]),axis=0)
            indmask[indkill]=False

        if comm is not None:
            comm.barrier()
            indmask = comm.allreduce(indmask, op=MPI.SUM)   

        self.enst = enst[indmask]




    def set_basis(self,return_qinv=False):


        self.nwave = len(self.enst)
        ###self.data_Qinv = numpy.full((nwave),1./self.STD**2)
        Q = numpy.full((self.nwave),1./self.STD**2)

        if return_qinv==True:
            return 1./Q

        
    def operg(self, coords=None, coords_name=None, cdir=None, config=None, nature=None, compute_g=False, 
                compute_geta=False, coordtype='scattered', iwave0=0, iwave1=None, obs_name=None, gsize_max=None, int_type='i8', float_type='f4', label=None):



        lon = coords[coords_name['lon']]
        lat = coords[coords_name['lat']]
        time = coords[coords_name['time']]


        if iwave1==None: iwave1=self.nwave

        if compute_g==True: logging.info('START computing G: %s %s %s ', self.name, obs_name, label )


        if compute_geta:
            result = zeros((len(time)))

        if compute_g:
            result=[None]*3
            result[0]=numpy.zeros((iwave1-iwave0), dtype=int_type)
            result[1]=numpy.empty((gsize_max), dtype=int_type)
            result[2]=numpy.empty((gsize_max), dtype=float_type)
            ind_tmp = 0


        iwave = -1

        for timeref in self.enst:
            iwave += 1
            if ((iwave >= iwave0) & (iwave <iwave1)):
                diff = time - timeref
                iobs = where(abs(diff) < 2 * self.TDEC)[0]
                nobs = len(iobs)
                if nobs>0:
                    if compute_geta:
                        result[iobs] += 1. / self.fcor * exp(-2 * diff[iobs] ** 2 / self.TDEC ** 2) * self.data_eta[iwave]
                    if compute_g:
                        result[0][iwave-iwave0] = nobs
                        result[1][ind_tmp:ind_tmp+nobs] = iobs
                        result[2][ind_tmp:ind_tmp+nobs] = (1. / self.fcor * exp(-2 * diff[iobs] ** 2 / self.TDEC ** 2))
                        ind_tmp += nobs


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g==True: 
            logging.info('END computing G: %s %s %s', self.name, obs_name, label )
            return numpy.copy(result[0]), numpy.copy(result[1][:ind_tmp]), numpy.copy(result[2][:ind_tmp])
        else:
            return result
            

class CompNbe(Comp, yaml.YAMLObject):
# b for barotrop

    yaml_tag = u'!CompNbe'


    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompNbe,self).__init__(**kwargs)

        self.ens_nature = ['sla'] # Ensemble of obs nature that project on the component
    
    def set_basis(self,obs):

        coordinates = obs.data

        dd=0.5
        self.fcor = 1. / (1 + exp(-2 * dd ** 2) + exp(-2 * (2 * dd) ** 2) + exp(-2 * (3 * dd) ** 2))


        lwtdec = self.TDEC
        enst=arange(-1.5*lwtdec + obs.data['time'].min(), obs.data['time'].max()+1.5*lwtdec,dd*lwtdec)
        iobsr=where(((coordinates['time'][1:] - coordinates['time'][:-1])>2*lwtdec))[0]
        indkill=empty([0], dtype=int)
        for k in range(len(iobsr)):
            indk=where((enst[-1]-coordinates['time'][iobsr[k]]>2*lwtdec)&(coordinates['time'][iobsr[k]+1]-enst[-1]>2*lwtdec))[0]
            indkill=concatenate(([indkill,indk]),axis=0)
        indk = where((coordinates['time'][0]-enst > 2*lwtdec))[0]
        indkill = concatenate(([indkill,indk]),axis=0)
        indk = where((enst - coordinates['time'][-1] > 2*lwtdec))[0]
        indkill = concatenate(([indkill,indk]),axis=0)
        enst=delete(enst,indkill)
        nwave = len(enst)

        finterpDEPTH = read_auxdata_depth(self.file_aux)

        self.data = empty((nwave), dtype=[('CFAC','f4'), ('Qinv','f4'), ('eta','f4'),('dJp','f4'),  
            ('b','f4'), ('x','f4'), ('ax','f4'), ('rest','f4'), ('rest_next','f4'), 
            ('p','f4'), ('Ap','f4'), ('dJo','f4'), ('dJ','f4'), ('dJ_beps','f4'), ('dir','f4'), 
            ('vec','f4'), ('vecb','f4'), ('JJ','f4') ])    
        
        self.data['Qinv'][:] = 1./self.STD**2
        self.data['eta'][:] = 0. 
        self.data['dJp'][:] = 0.

        self.Jp = 0. # at first iteration, Jp is zero

        self.enst=enst
        self.nwave=nwave
        self.name='nbe_'+obs.name

        self.finterpDEPTH=finterpDEPTH


        
    def operg(self ,coordinates, config=None, nature=None, compute_g=False, compute_geta=False, indg=None):

        if compute_g:
            result=None
            Gf = empty((int(self.gsize_max)),dtype='f4')
            indobs = empty((int(self.gsize_max)),dtype='u8')  # try u4 ?
            size = 0
            cumsize = 0   
        else :
            result = zeros((len(coordinates)))

        lon=coordinates['lon']
        lat=coordinates['lat']
        time=coordinates['time']

        depth= -self.finterpDEPTH((lon,lat))
        depth[numpy.isnan(depth)]=0.

        iwave = -1

        for timeref in self.enst:
            iwave += 1
            diff = time - timeref
            iobs = where(abs(diff) < 2 * self.TDEC)[0]
            nobs = len(iobs)

            facd=numpy.ones((len(iobs)))
            facd = 1. - (depth[iobs]-self.depth1)/(self.depth2-self.depth1)
            facd[facd>1]=1.
            facd[facd<0]=0.

            if nobs>0:
                if compute_geta:
                    result[iobs] += facd * 1. / self.fcor * exp(-2 * diff[iobs] ** 2 / self.TDEC ** 2) * self.eta[iwave]
                if compute_g:
                    Gf[size:size+nobs] = facd * 1. / self.fcor * exp(-2 * diff[iobs] ** 2 / self.TDEC ** 2)

            if compute_g:
                indobs[size:size+nobs] = iobs         
                size += nobs
                cumsize=append(cumsize,size)


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g:   
            if size>0:            
                self.G[indg] = csc_matrix((Gf[:size], indobs[:size],cumsize), shape=(len(coordinates),self.nwave))
            else: self.G[indg] = numpy.array([])
            self.gsize[indg] = size # number of non-zero terms of the sparse G matrix

        return result
            



class CompSkimYaw(Comp, yaml.YAMLObject):

    yaml_tag = u'!CompSkimYaw'

    # __slots__ = (
    #     'ffl',
    #     'PSl',
    #     'lcut',
    #     'lmin',
    #     'lmax',
    #     'slope',
    #     'sigma',
    #     'npsp',
    #     'enst',
    #     'data',
    #     'Jp',
    #     'nwave',
    #     'name',
    #     'write'
    #     )

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompSkimYaw,self).__init__(**kwargs)



        if self.ps_from_file==False:
            df = 1./self.lmax
            f = arange(df,1./self.lmin,df)
            PS=(f*self.lcut)**self.slope
            PS[f<1./self.lcut]=1.
            PS=self.sigma**2*PS/sum(PS*df)

        if self.ps_from_file==True:
            filenc = self.ps_file
            fid = Dataset(filenc)
            f = numpy.array(fid.variables['f'][:])
            PS = numpy.array(fid.variables[self.ps_varname][:])
            PS *= self.ps_fac
            self.vsat = numpy.array(fid.variables['vsat'][:])
            fid.close()

        logf=log10(f)
        logPS=log10(PS)
        fint=interp1d(logf,logPS)

        logffl = arange(log10(1./self.lmax),
                          log10(1./self.lmin),# + numpy.log10(1 + 2. / npsp),
                          log10(1 + 2. / self.npsp))

        self.ffl = 10**(logffl)
        self.PSl=10**fint(logffl)
         
        self.ffl = self.ffl[self.PSl>0.]
        self.PSl = self.PSl[self.PSl>0.]



    def set_basis(self, obs):

        time = obs.data['time'] * 86400. # in sec
        #angle = obs.data['angle_from_ac']

        enst=[None]*len(self.ffl)
        nwave=0
        for k in range(len(self.ffl)):
            dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
            # Reperer et eliminer les trous pour les multi-pass en regional
            enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            nwave=nwave+2*len(enst[k])

        Q=zeros(nwave)
        #ffl_wave=zeros(nwave)
        #tc_wave=numpy.zeros(nwave)
        iwave=-1
        for k in range(len(self.ffl)):
            dxp=0.25/self.ffl[k]
            for P in enst[k]:
                iwave +=1
                Q[iwave]=self.PSl[k]*self.ffl[k]/self.npsp
               # ffl_wave[iwave]=self.ffl[k]
               # tc_wave[iwave]=P
                iwave +=1
                Q[iwave]=self.PSl[k]*self.ffl[k]/self.npsp
               # ffl_wave[iwave]=self.ffl[k]
               # tc_wave[iwave]=P+dxp

        if iwave != nwave - 1 : pdb.set_trace()


        self.data = empty((nwave), dtype=[('CFAC','f4'), ('Qinv','f4'), ('eta','f4'),('dJp','f4'),  
                    ('b','f4'), ('x','f4'), ('ax','f4'), ('rest','f4'), ('rest_next','f4'), 
                    ('p','f4'), ('Ap','f4'), ('dJo','f4'), ('dJ','f4'), ('dJ_beps','f4'), ('dir','f4'), 
                    ('vec','f4'), ('vecb','f4'), ('JJ','f4') ])    
        
        self.data['Qinv'][:] = 1./Q
        self.data['eta'][:] = 0. 
        self.data['dJp'][:] = 0.

        self.Jp = 0. # at first iteration, Jp is zero

        self.enst=enst
        self.nwave=nwave
        self.name='SkimYaw_'+obs.name
        




    def operg(self ,coordinates, coord_time=None, cdir=None, config=None, nature=None, compute_g=False, compute_geta=False, indg=None):

        time = coordinates['time'] *86400. #in sec
        if cdir is None: angle = coordinates['angle_from_ac']
        else: angle = full((len(coordinates['time'])), cdir)

        if compute_g:
            result=None
            Gf = empty((int(self.gsize_max)),dtype='f4')
            indobs = empty((int(self.gsize_max)),dtype='u8')  # try u4 ?
            size = 0
            cumsize = 0   
        else :
            result = zeros((len(coordinates)))

        iwave = -1
        for k in range(len(self.ffl)):
            dxp = 0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
            for P in range(len(self.enst[k])):
                # Obs selection around point P
                for phase in [0, dxp]:
                    iwave +=1
                    iobs = where( (abs(time-self.enst[k][P]-phase)<self.npsp/self.ffl[k]/2))[0]
                    nobs = len(iobs)
                    if nobs > 0:
                        tloc = time[iobs]-self.enst[k][P]-phase
                        #if ((k==len(self.ffl)-1)&(P==3)): pdb.set_trace()
                        tap = mywindow(tloc*2*self.ffl[k]/self.npsp)
                        if compute_g:
                            Gf[size:size+nobs] = cos(2 * pi * self.ffl[k] * tloc ) * tap * cos(angle[iobs]) * 1.e-6 * self.vsat                    
                        if compute_geta: 
                            result[iobs] += self.data['eta'][iwave] * cos(2 * pi * self.ffl[k] * tloc ) * tap * cos(angle[iobs]) * 1.e-6 * self.vsat
                    if compute_g:
                        indobs[size:size+nobs] = iobs         
                        size += nobs
                        cumsize=append(cumsize,size)


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g:   
            if size>0:            
                self.G[indg] = csc_matrix((Gf[:size], indobs[:size],cumsize), shape=(len(coordinates),self.nwave))
            else: self.G[indg] = numpy.array([])
            self.gsize[indg] = size # number of non-zero terms of the sparse G matrix

        return result

class CompSkimYawTed(Comp, yaml.YAMLObject):

    yaml_tag = u'!CompSkimYawTed'


    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompSkimYawTed,self).__init__(**kwargs)


        df=1./self.lambda_max
        f=numpy.arange(1./self.lambda_max,1./self.lambda_min,df)

        rwidth = self.rwidth # 0.01
        fa_ref = 1./self.torb # (86400*29./412)**-1 # Hz
        ens_fa = fa_ref * (numpy.arange(1,self.npeaks+1,1))
        #ens_valS = numpy.array([1e6, 3e4, 4e1, 2e0])/4e1*8.5e2
        ens_valS = numpy.ones((self.npeaks))*1e6/4e1*8.5e2
        ens_sigma = ens_fa * rwidth * (2.355)**-1

        PS = f*0.
        for iff in range(self.npeaks):
            PSp=numpy.exp(-0.5*(f-ens_fa[iff])**2/(ens_sigma[iff])**2)
            PSp[PSp<1e-3]=0.
            PSp=PSp*ens_valS[iff]
            PS += PSp


        logf=log10(f)
        logPS=log10(PS)
        fint=interp1d(logf,logPS)

        logffl = arange(log10(1./self.lambda_max),
                          log10(1./self.lambda_min),# + numpy.log10(1 + 2. / npsp),
                          log10(1 + 2. / self.npsp))

        self.ffl = 10**(logffl)
        self.PSl=10**fint(logffl)
         
        self.ffl = self.ffl[self.PSl>0.]
        self.PSl = self.PSl[self.PSl>0.]
        self.ffl = numpy.concatenate(([0],self.ffl))

        #pdb.set_trace()



    def set_basis(self, obs):

        time = obs.data['time'] * 86400. # in sec
        #angle = obs.data['angle_from_ac']

        enst=[None]*len(self.ffl)
        nwave=0
        for k in range(len(self.ffl)):
            # dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
            # enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            if k==0:
                dxp=time.max()-time.min()
                enst[k] = [(time.max()+time.min())*0.5]
                #nwave += len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += len(enst[k]) * (2*(self.nthetaharm-1) ) * len(obs.ens_bincl)
            else:
                dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
                enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
                #nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) ) * len(obs.ens_bincl)

        Q=zeros(nwave) + 10.
        # #ffl_wave=zeros(nwave)
        # #tc_wave=numpy.zeros(nwave)
        # iwave=-1
        # for incl_id in range(len(obs.ens_bincl)):
        #     for k in range(len(self.ffl)):
        #         dxp=0.25/self.ffl[k]
        #         for P in range(len(enst[k])):
        #             for phase in [0, dxp]:
        #                 for THETAM in range(self.nthetaharm): 
        #                     if THETAM==0: ens_phasetheta=[0]
        #                     else: ens_phasetheta=[0,0.5*pi]
        #                     for phasetheta in ens_phasetheta:
        #                         iwave +=1
        #                         Q[iwave]=1000*self.PSl[k]*self.ffl[k]/self.npsp

        #if iwave != nwave - 1 : pdb.set_trace()


        self.data = empty((nwave), dtype=[('CFAC','f4'), ('Qinv','f4'), ('eta','f4'),('dJp','f4'),  
                    ('b','f4'), ('x','f4'), ('ax','f4'), ('rest','f4'), ('rest_next','f4'), 
                    ('p','f4'), ('Ap','f4'), ('dJo','f4'), ('dJ','f4'), ('dJ_beps','f4'), ('dir','f4'), 
                    ('vec','f4'), ('vecb','f4'), ('JJ','f4') ])    
        
        self.data['Qinv'][:] = 1./Q
        self.data['eta'][:] = 0. 
        self.data['dJp'][:] = 0.

        self.Jp = 0. # at first iteration, Jp is zero

        self.enst=enst
        self.nwave=nwave
        self.name='SkimYawTed_'+obs.name
        self.ens_bincl = obs.ens_bincl

        




    def operg(self ,coordinates, coord_time=None, cdir=None, config=None, nature=None, compute_g=False, compute_geta=False, indg=None):

        time = coordinates['time'] *86400. #in sec
        if cdir is None: angle = coordinates['angle_from_ac']
        else: angle = full((len(coordinates['time'])), cdir)
        beam_incl = coordinates['beam_inc']

        if compute_g:
            result=None
            Gf = empty((int(self.gsize_max)),dtype='f4')
            indobs = empty((int(self.gsize_max)),dtype='u8')  # try u4 ?
            size = 0
            cumsize = 0   
        else :
            result = zeros((len(coordinates)))

        iwave = -1
        for bincl in self.ens_bincl:
            for k in range(len(self.ffl)):
                if k==0: dxp=time.max()-time.min()
                else: dxp=0.25/self.ffl[k]
                for P in range(len(self.enst[k])):
                    if k==0: ens_phase = [0]
                    else: ens_phase = [0, dxp]
                    for phase in ens_phase:
                        iobs = where( (abs(time-self.enst[k][P]-phase)<self.npsp/self.ffl[k]/2) & (beam_incl==bincl))[0]
                        nobs = len(iobs)
                        for THETAM in range(self.nthetaharm)[1:]: 
                            if THETAM==0: ens_phasetheta=[0]
                            else: ens_phasetheta=[0,0.5*pi]
                            for phasetheta in ens_phasetheta:
                                iwave +=1
                                if nobs > 0:
                                    tloc = time[iobs]-self.enst[k][P]-phase
                                    tap = mywindow(tloc*2*self.ffl[k]/self.npsp)
                                    if compute_g:
                                        Gf[size:size+nobs] = cos(2 * pi * self.ffl[k] * tloc ) * tap * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])                  
                                    if compute_geta: 
                                        result[iobs] += self.data['eta'][iwave] * cos(2 * pi * self.ffl[k] * tloc ) * tap * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])
                                if compute_g:
                                    indobs[size:size+nobs] = iobs         
                                    size += nobs
                                    cumsize=append(cumsize,size)


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g:   
            if size>0:            
                self.G[indg] = csc_matrix((Gf[:size], indobs[:size],cumsize), shape=(len(coordinates),self.nwave))
            else: self.G[indg] = numpy.array([])
            self.gsize[indg] = size # number of non-zero terms of the sparse G matrix

        return result

class CompSkimYawTedPer(Comp, yaml.YAMLObject):

    yaml_tag = u'!CompSkimYawTedPer'


    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompSkimYawTedPer,self).__init__(**kwargs)


        # df=1./self.lambda_max
        # f=numpy.arange(1./self.lambda_max,1./self.lambda_min,df)

        # rwidth = self.rwidth # 0.01
        fa_ref = 1./self.torb # (86400*29./412)**-1 # Hz
        ens_fa = fa_ref * (numpy.arange(1,self.npeaks+1,1))
        #ens_valS = numpy.array([1e6, 3e4, 4e1, 2e0])/4e1*8.5e2
        # ens_valS = numpy.ones((self.npeaks))*1e6/4e1*8.5e2
        # ens_sigma = ens_fa * rwidth * (2.355)**-1





        self.ffl = numpy.concatenate(([0],ens_fa))

        #pdb.set_trace()



    def set_basis(self, obs):

        time = obs.data['time'] * 86400. # in sec
        #angle = obs.data['angle_from_ac']

        enst=[None]*len(self.ffl)
        nwave=0
        for k in range(len(self.ffl)):
            # dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
            # enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            if k==0:
            #dxp=time.max()-time.min()
                enst[k] = [(time.max()+time.min())*0.5]
            #nwave += len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += len(enst[k]) * (2*(self.nthetaharm-1) ) * len(obs.ens_bincl)
            else:
            #     dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
                enst[k] = [(time.max()+time.min())*0.5]
                #enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            #     #nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) ) * len(obs.ens_bincl)

        Q=zeros(nwave) + 1000.
        # #ffl_wave=zeros(nwave)
        # #tc_wave=numpy.zeros(nwave)
        # iwave=-1
        # for incl_id in range(len(obs.ens_bincl)):
        #     for k in range(len(self.ffl)):
        #         dxp=0.25/self.ffl[k]
        #         for P in range(len(enst[k])):
        #             for phase in [0, dxp]:
        #                 for THETAM in range(self.nthetaharm): 
        #                     if THETAM==0: ens_phasetheta=[0]
        #                     else: ens_phasetheta=[0,0.5*pi]
        #                     for phasetheta in ens_phasetheta:
        #                         iwave +=1
        #                         Q[iwave]=1000*self.PSl[k]*self.ffl[k]/self.npsp

        #if iwave != nwave - 1 : pdb.set_trace()


        self.data = empty((nwave), dtype=[('CFAC','f4'), ('Qinv','f4'), ('eta','f4'),('dJp','f4'),  
                    ('b','f4'), ('x','f4'), ('ax','f4'), ('rest','f4'), ('rest_next','f4'), 
                    ('p','f4'), ('Ap','f4'), ('dJo','f4'), ('dJ','f4'), ('dJ_beps','f4'), ('dir','f4'), 
                    ('vec','f4'), ('vecb','f4'), ('JJ','f4') ])    
        
        self.data['Qinv'][:] = 1./Q
        self.data['eta'][:] = 0. 
        self.data['dJp'][:] = 0.

        self.Jp = 0. # at first iteration, Jp is zero

        self.enst=enst
        self.nwave=nwave
        self.name='SkimYawTed_'+obs.name
        self.ens_bincl = obs.ens_bincl

        




    def operg(self ,coordinates, coord_time=None, cdir=None, config=None, nature=None, compute_g=False, compute_geta=False, indg=None):

        time = coordinates['time'] *86400. #in sec
        if cdir is None: angle = coordinates['angle_from_ac']
        else: angle = full((len(coordinates['time'])), cdir)
        beam_incl = coordinates['beam_inc']

        if compute_g:
            result=None
            Gf = empty((int(self.gsize_max)),dtype='f4')
            indobs = empty((int(self.gsize_max)),dtype='u8')  # try u4 ?
            size = 0
            cumsize = 0   
        else :
            result = zeros((len(coordinates)))

        iwave = -1
        for bincl in self.ens_bincl:
            for k in range(len(self.ffl)):
                if k==0: dxp=time.max()-time.min()
                else: dxp=0.25/self.ffl[k]
                for P in range(len(self.enst[k])):
                    if k==0: ens_phase = [0]
                    else: ens_phase = [0, dxp]
                    for phase in ens_phase:
                        iobs = where( (abs(time-self.enst[k][P]-phase)<self.npsp/self.ffl[k]/2) & (beam_incl==bincl))[0]
                        nobs = len(iobs)
                        for THETAM in range(self.nthetaharm)[1:]: 
                            if THETAM==0: ens_phasetheta=[0]
                            else: ens_phasetheta=[0,0.5*pi]
                            for phasetheta in ens_phasetheta:
                                iwave +=1
                                if nobs > 0:
                                    tloc = time[iobs]-self.enst[k][P]-phase
                                    #tap = mywindow(tloc*2*self.ffl[k]/self.npsp)
                                    if compute_g:
                                        Gf[size:size+nobs] = cos(2 * pi * self.ffl[k] * tloc ) * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])                  
                                    if compute_geta: 
                                        result[iobs] += self.data['eta'][iwave] * cos(2 * pi * self.ffl[k] * tloc ) * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])
                                if compute_g:
                                    indobs[size:size+nobs] = iobs         
                                    size += nobs
                                    cumsize=append(cumsize,size)


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g:   
            if size>0:            
                self.G[indg] = csc_matrix((Gf[:size], indobs[:size],cumsize), shape=(len(coordinates),self.nwave))
            else: self.G[indg] = numpy.array([])
            self.gsize[indg] = size # number of non-zero terms of the sparse G matrix

        return result


class CompSkimYawTedPerM(Comp, yaml.YAMLObject):

    #with a non-zero mean
    yaml_tag = u'!CompSkimYawTedPerM'


    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CompSkimYawTedPerM,self).__init__(**kwargs)


        # df=1./self.lambda_max
        # f=numpy.arange(1./self.lambda_max,1./self.lambda_min,df)

        # rwidth = self.rwidth # 0.01
        fa_ref = 1./self.torb # (86400*29./412)**-1 # Hz
        ens_fa = fa_ref * (numpy.arange(1,self.npeaks+1,1))
        #ens_valS = numpy.array([1e6, 3e4, 4e1, 2e0])/4e1*8.5e2
        # ens_valS = numpy.ones((self.npeaks))*1e6/4e1*8.5e2
        # ens_sigma = ens_fa * rwidth * (2.355)**-1





        self.ffl = numpy.concatenate(([0],ens_fa))

        #pdb.set_trace()



    def set_basis(self, obs):

        time = obs.data['time'] * 86400. # in sec
        #angle = obs.data['angle_from_ac']

        enst=[None]*len(self.ffl)
        nwave=0
        for k in range(len(self.ffl)):
            # dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
            # enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            if k==0:
            #dxp=time.max()-time.min()
                enst[k] = [(time.max()+time.min())*0.5]
            #nwave += len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
            else:
            #     dxp=0.25/self.ffl[k]  # for a pi/2 phase, cos to sin
                enst[k] = [(time.max()+time.min())*0.5]
                #enst[k] = arange(time.min()-dxp,time.max()+self.npsp/self.ffl[k]/2+dxp,self.npsp/self.ffl[k]/2)
            #     #nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)
                nwave += 2*len(enst[k]) * (2*(self.nthetaharm-1) +1) * len(obs.ens_bincl)

        Q=zeros(nwave) + 10.
        # #ffl_wave=zeros(nwave)
        # #tc_wave=numpy.zeros(nwave)
        # iwave=-1
        # for incl_id in range(len(obs.ens_bincl)):
        #     for k in range(len(self.ffl)):
        #         dxp=0.25/self.ffl[k]
        #         for P in range(len(enst[k])):
        #             for phase in [0, dxp]:
        #                 for THETAM in range(self.nthetaharm): 
        #                     if THETAM==0: ens_phasetheta=[0]
        #                     else: ens_phasetheta=[0,0.5*pi]
        #                     for phasetheta in ens_phasetheta:
        #                         iwave +=1
        #                         Q[iwave]=1000*self.PSl[k]*self.ffl[k]/self.npsp

        #if iwave != nwave - 1 : pdb.set_trace()


        self.data = empty((nwave), dtype=[('CFAC','f4'), ('Qinv','f4'), ('eta','f4'),('dJp','f4'),  
                    ('b','f4'), ('x','f4'), ('ax','f4'), ('rest','f4'), ('rest_next','f4'), 
                    ('p','f4'), ('Ap','f4'), ('dJo','f4'), ('dJ','f4'), ('dJ_beps','f4'), ('dir','f4'), 
                    ('vec','f4'), ('vecb','f4'), ('JJ','f4') ])    
        
        self.data['Qinv'][:] = 1./Q
        self.data['eta'][:] = 0. 
        self.data['dJp'][:] = 0.

        self.Jp = 0. # at first iteration, Jp is zero

        self.enst=enst
        self.nwave=nwave
        self.name='SkimYawTed_'+obs.name
        self.ens_bincl = obs.ens_bincl

        




    def operg(self ,coordinates, coord_time=None, cdir=None, config=None, nature=None, compute_g=False, compute_geta=False, indg=None):

        time = coordinates['time'] *86400. #in sec
        if cdir is None: angle = coordinates['angle_from_ac']
        else: angle = full((len(coordinates['time'])), cdir)
        beam_incl = coordinates['beam_inc']

        if compute_g:
            result=None
            Gf = empty((int(self.gsize_max)),dtype='f4')
            indobs = empty((int(self.gsize_max)),dtype='u8')  # try u4 ?
            size = 0
            cumsize = 0   
        else :
            result = zeros((len(coordinates)))

        iwave = -1
        for bincl in self.ens_bincl:
            for k in range(len(self.ffl)):
                if k==0: dxp=time.max()-time.min()
                else: dxp=0.25/self.ffl[k]
                for P in range(len(self.enst[k])):
                    if k==0: ens_phase = [0]
                    else: ens_phase = [0, dxp]
                    for phase in ens_phase:
                        iobs = where( (abs(time-self.enst[k][P]-phase)<self.npsp/self.ffl[k]/2) & (beam_incl==bincl))[0]
                        nobs = len(iobs)
                        for THETAM in range(self.nthetaharm)[:]: 
                            if THETAM==0: ens_phasetheta=[0]
                            else: ens_phasetheta=[0,0.5*pi]
                            for phasetheta in ens_phasetheta:
                                iwave +=1
                                if nobs > 0:
                                    tloc = time[iobs]-self.enst[k][P]-phase
                                    #tap = mywindow(tloc*2*self.ffl[k]/self.npsp)
                                    if compute_g:
                                        Gf[size:size+nobs] = cos(2 * pi * self.ffl[k] * tloc ) * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])                  
                                    if compute_geta: 
                                        result[iobs] += self.data['eta'][iwave] * cos(2 * pi * self.ffl[k] * tloc ) * 1.e-6 * self.vsat * cos(THETAM*angle[iobs]+phasetheta) *  cos(angle[iobs])
                                if compute_g:
                                    indobs[size:size+nobs] = iobs         
                                    size += nobs
                                    cumsize=append(cumsize,size)


        if iwave != self.nwave - 1 : pdb.set_trace()

        if compute_g:   
            if size>0:            
                self.G[indg] = csc_matrix((Gf[:size], indobs[:size],cumsize), shape=(len(coordinates),self.nwave))
            else: self.G[indg] = numpy.array([])
            self.gsize[indg] = size # number of non-zero terms of the sparse G matrix

        return result
