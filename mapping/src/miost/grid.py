# -*- coding: utf-8 -*-
"""
"""
from numpy import array, nan, arange, meshgrid, isnan, pi, empty, sin, cos, diff, where, tile, mod, outer, zeros
import numpy
from scipy import interpolate
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from os import path, remove
import pdb
import netCDF4 as nc
import matplotlib.pylab as plt
import csv
import pickle
from datetime import datetime
import tools
from comp_iw import jd2ap
import logging

class Grid(object):
    """
    """

    def __init__(self, **kwargs):
        pass


class Grid_3d(Grid):


    """
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        
        with Dataset(path.join(self.TEMPLATE_FILE), 'r') as fcid:
            if self.FLAG_MDT:
                MDT = array(fcid.variables[self.MDT_NAME][:])
            lat = array(fcid.variables[self.LAT_NAME][:])
            lon = array(fcid.variables[self.LON_NAME][:])
            lon = mod(lon,360.)
            if self.FLAG_MDT:
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()
            else:
                MDT = array(fcid.variables[self.MDT_NAME][0,:,:])
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()

            dlon = lon[1]-lon[0]
            dlat = lat[1]-lat[0]

        #zone = domain['ZONE']
        # lonmax=zone['LON_MAX']
        if (self.LON_MAX<self.LON_MIN): 
            ix = where((lon>=self.LON_MIN-dlon)|(lon<=self.LON_MAX+dlon))[0]
        else:
            ix = where((lon>=self.LON_MIN-dlon)&(lon<=self.LON_MAX+dlon))[0]
        iy = where((lat>=self.LAT_MIN-dlat)&(lat<=self.LAT_MAX+dlat))[0]
        self.lon = lon[ix]
        self.lat = lat[iy]
        (self.lon2, self.lat2) = meshgrid(self.lon,self.lat)

        if self.FLAG_MDT:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
            deg2m=110000.
            fc = 2*2*pi/86164*sin(self.lat2*pi/180)
            self.MDU = +self.MDT
            self.MDV = +self.MDT
            self.MDV[:-1,:-1] = 10./fc[:-1,:-1]*diff(self.MDT, axis=1)[:-1,:]/(diff(self.lon2,axis=1)[:-1,:]*deg2m*cos(self.lat2[:-1,:-1]*pi/180.))
            self.MDU[:-1,:-1] = -10./fc[:-1,:-1]*diff(self.MDT, axis=0)[:,:-1]/(diff(self.lat2,axis=0)[:,:-1]*deg2m)
        else:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]


        self.TIME_MIN= (datetime.strptime(self.DATE_MIN, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX= (datetime.strptime(self.DATE_MAX, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        # self.TIME_MIN= (datetime.date(self.DATE_MIN[0],self.DATE_MIN[1],self.DATE_MIN[2])-datetime.date(1950,1,1)).days
        # self.TIME_MAX= (datetime.date(self.DATE_MAX[0],self.DATE_MAX[1],self.DATE_MAX[2])-datetime.date(1950,1,1)).days
        deltat = self.TIME_MAX - self.TIME_MIN
        step = self.TIME_STEP
        self.time = arange(0.,deltat+step,step)
        self.timej = self.TIME_MIN + self.time
        self.nt = len(self.time)
        try:
            step = self.TIME_STEP_HF
            self.time_hf = arange(0.,deltat+step,step)
            self.timej_hf = self.TIME_MIN + self.time_hf
            self.nt_hf = len(self.time_hf)   
        except: pass   
        try:
            self.time_lf = arange(0.,deltat,self.TIME_STEP_LF)
            self.timej_lf = self.TIME_MIN + self.time_lf
            self.nt_lf = len(self.time_lf)   
        except: pass           
        self.nx = len(self.lon)
        self.ny = len(self.lat)


    def write_outputs(self, comp, config):


        address = config['PATH']['OUTPUT']
        rootname = config['RUN_NAME']

        self.grido = type('', (), {})()
        #self.grido = empty(self.nt*self.nx*self.ny,dtype=[('lon','f4'),('lat','f4'),('time','f4')])
        time, lat, lon = meshgrid(self.time, self.lat, self.lon,indexing='ij')
        self.grido.time,self.grido.lat,self.grido.lon = time.flatten(), lat.flatten(), lon.flatten()

        self.gridor = type('', (), {})()
        #self.gridor = empty(self.nx*self.ny,dtype=[('lon','f4'),('lat','f4')])
        lat, lon = meshgrid(self.lat, self.lon,indexing='ij')
        self.gridor.lat,self.gridor.lon = lat.flatten(), lon.flatten()
               


        MDT2 = tile(self.MDT,(self.nt,1,1)).flatten()
        ind_sel = where((isnan(MDT2)==False))
        ind_selr = where((isnan(self.MDT.flatten())==False))[0]


        filenc = address +'/'+ rootname + '_analysis.nc'
        if path.exists(filenc):remove(filenc)
        with Dataset(filenc,"w") as fid:
            fid.description = " Miost analysis "
            fid.createDimension('time', None)
            fid.createDimension('time_hf', None)
            fid.createDimension('time_lf', None)
            fid.createDimension('y', len(self.lat))
            fid.createDimension('x', len(self.lon))

            v=fid.createVariable('lon', 'f4', ('x'))
            v[:]=self.lon
            v=fid.createVariable('lat', 'f4', ('y'))
            v[:]=self.lat
            v=fid.createVariable('time', 'f4', ('time',))
            v[:]=self.timej
            try:
                v=fid.createVariable('time_hf', 'f4', ('time_hf',))
                v[:]=self.timej_hf      
            except: pass      
            try:
                v=fid.createVariable('time_lf', 'f4', ('time_lf',))
                v[:]=self.timej_lf      
            except: pass                  

            ######################################################################################

            if (self.OUTPUT_FORMAT=='H3'):

                for kc in range(len(comp)):
                    if comp[kc].write==True :
                        if ((comp[kc].name=='geo3ss')|(comp[kc].name=='geo3ls')) : 
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            coords=[None]*3
                            coords[0]=self.gridor.lon[ind_selr]
                            coords[1]=self.gridor.lat[ind_selr]
                            coords[2]=self.time
                            coords_name={'lon':0, 'lat':1, 'time':2}
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, coordtype='reg', nature='sla', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Hvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('geo3ss h written')


                        if comp[kc].name=='barotrop' : 
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            tmp = comp[kc].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', nature='sla', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable('Hb', 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('barotrop h written')                  

                        if comp[kc].name[:2]=='iw' :
                            #varPHI = fid.createVariable('phi_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            tmp = empty((self.ny*self.nx))
                            tmp[:] = nan
                            pvar = fid.createVariable('dphi_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            avar = fid.createVariable('a_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            var1 = fid.createVariable('Hit1_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            var2 = fid.createVariable('Hit2_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            

                            Amp_astr, Phi0_astr, dPhi_astr = jd2ap(comp[kc].file_data_astr, comp[kc].tidal_comp,self.tref_iw,self.timej_lf)
                            pvar[:] = dPhi_astr
                            avar[:] = Amp_astr


                            var1.tidal_comp=comp[kc].tidal_comp
                            var2.tidal_comp=comp[kc].tidal_comp
                            var1.Ttide=comp[kc].Ttide
                            var1.tref=self.tref_iw
                            var1.phiref = Phi0_astr

                            # Af0, v0u0 = comp[kc].wave_table.compute_nodal_corrections(numpy.array((numpy.zeros((1))+(-7305+self.tref_iw)*86400), dtype='datetime64[s]').astype('datetime64[s]').astype('float'))
                            # var1.phiref=v0u0[0]
                            # dt1_phi = -v0u0[0]/(comp[kc].wave_table.freq()*86400)
                            # dt2_phi = dt1_phi + 0.25*comp[kc].Ttide/24
                            # #tmp = comp[kc].operg(self.gridor[ind_selr],coord_time = numpy.array(-self.TIME_MIN+0.*comp[kc].Ttide/24),coordtype='reg', nature='sla', compute_geta=True)
                            # #valr[ind_selr]=tmp/Amp_astr0
                            # ##Af0, v0u0 = comp[kc].wave_table.compute_nodal_corrections(numpy.array([(-7305+15340)*86400], dtype='datetime64[s]').astype('datetime64[s]').astype('float'))
                            # ##tref=15340.
                            dt1_phi = - (dPhi_astr+Phi0_astr) / (comp[kc].freq_tide*86400)
                            dt1_ref = - numpy.mod( self.timej_lf - self.tref_iw , comp[kc].Ttide/24)

                            for itlf in range(len(self.time_lf)): 
                                #dt1 = dt1_phi - numpy.mod(self.time_lf[itlf]-self.tref_iw, comp[kc].Ttide/24)
                                time1 = numpy.array(self.time_lf[itlf] + dt1_ref[itlf] + dt1_phi[itlf] + 0.*comp[kc].Ttide )
                                coords=[None]*3
                                coords[0]=self.gridor.lon[ind_selr]
                                coords[1]=self.gridor.lat[ind_selr]
                                coords[2]=time1
                                coords_name={'lon':0, 'lat':1, 'time':2}
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True)
                                var1[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                time2 = time1 + 0.25*comp[kc].Ttide/24
                                coords[2]=time2
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True)
                                var2[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                #varPHI[itlf]=v0u0[0]


                            # #tmp = comp[kc].operg(self.gridor[ind_selr],coord_time = numpy.array(-self.TIME_MIN+0.25*comp[kc].Ttide/24),coordtype='reg', nature='sla', compute_geta=True)
                            # #valr[ind_selr]=tmp/Amp_astr0
                            # time2 = numpy.array(tref+(numpy.pi/2-v0u0[0])/(comp[kc].wave_table.freq()*86400) + 0.*comp[kc].Ttide )
                            # tmp = comp[kc].operg(self.gridor[ind_selr],coord_time = -self.TIME_MIN+time2,coordtype='reg', nature='sla', compute_geta=True)
                            # valr[ind_selr]=tmp
                            # var = fid.createVariable('IW_H2_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('y','x'))
                            # var[:] = valr.reshape(self.ny, self.nx)
                            # var.tref=time2
                            # var.Ttide=comp[kc].Ttide
                            # var.tidal_comp=comp[kc].tidal_comp
                            print('iw h written for ', comp[kc].tidal_comp , '  mode ' , str(comp[kc].mode))                  


                try:
                    var = fid.createVariable('MDH', 'f4', ('y','x'))
                    var[:]=self.MDT
                except:
                    pass                       



            if (self.OUTPUT_FORMAT=='HUV'):

                for kc in range(len(comp)):
                    if comp[kc].write==True :
                        if ((comp[kc].name=='geo3ss')|(comp[kc].name=='geo3ls')) : 
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            coords=[None]*3
                            coords[0]=self.gridor.lon[ind_selr]
                            coords[1]=self.gridor.lat[ind_selr]
                            coords[2]=self.time
                            coords_name={'lon':0, 'lat':1, 'time':2}
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, coordtype='reg', nature='sla', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Hvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('geo3 h written')
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, cdir=0., coordtype='reg', nature='rcur', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Uvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('geo3 u written')
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, cdir=numpy.pi/2, coordtype='reg', nature='rcur', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Vvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('geo3 v written')                            

                        if comp[kc].name=='barotrop' : 
                            valr = empty((self.nt,self.ny*self.nx))
                            valr[:] = nan
                            tmp = comp[kc].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', nature='sla', compute_geta=True)
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable('Hb', 'f4', ('time','y','x'))
                            var[:] = valr.reshape(self.nt, self.ny, self.nx)
                            print('barotrop h written')                  


                try:
                    var = fid.createVariable('MDH', 'f4', ('y','x'))
                    var[:]=self.MDT
                    var = fid.createVariable('MDU', 'f4', ('y','x'))
                    var[:]=self.MDU        
                    var = fid.createVariable('MDV', 'f4', ('y','x'))
                    var[:]=self.MDV              
                except:
                    pass        


            ######################################################################################
            if (self.OUTPUT_FORMAT=='HUVtests'):

                is_kc_geos = False
                is_kc_geo = False
                is_kc_cageos_rot = False
                is_kc_cageos_div = False
                is_kc_io = False
                for kc in range(len(comp)):
                    if comp[kc].write==True :
                        if ( (comp[kc].name=='geos') | (comp[kc].name=='geosf') ) : 
                            is_kc_geos = True
                            kc_geos = kc
                        if comp[kc].name=='geo' : 
                            is_kc_geo = True
                            kc_geo = kc
                        if comp[kc].name=='cageos_rot': 
                            is_kc_cageos_rot = True
                            kc_cageos_rot = kc   
                        if comp[kc].name=='cageos_div': 
                            is_kc_cageos_div = True
                            kc_cageos_div = kc        
                        if comp[kc].name=='io': 
                            is_kc_io = True
                            kc_io = kc             


                val = empty((self.nt*self.ny*self.nx))
                val[:] = nan
                valr = empty((self.nt,self.ny*self.nx))
                valr[:] = nan
                try:
                    valr_hf = empty((self.nt_hf,self.ny*self.nx))
                    valr_hf[:] = nan  
                except: pass                                 

                if is_kc_geos:
                    tmp = comp[kc_geos].operg(self.grido[ind_sel], nature='sla', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('H', 'f4', ('time','y','x'))
                    var[:] = val.reshape(self.nt, self.ny, self.nx) 
                    print('geos h written')

                if is_kc_geos:

                    tmp = comp[kc_geos].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Ug', 'f4', ('time','y','x'))
                    var[:] = val.reshape(self.nt, self.ny, self.nx) 

                    tmp = comp[kc_geos].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Vg', 'f4', ('time','y','x'))
                    var[:] = val.reshape(self.nt, self.ny, self.nx) 
                    print('geos cur written')

                if is_kc_geo:

                    tmp = comp[kc_geo].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', nature='sla', compute_geta=True)
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('H', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx) 
                    print('geo h written')

                    tmp = comp[kc_geo].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=0., nature='rcur', compute_geta=True)
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('Ug', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx) 

                    tmp = comp[kc_geo].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=pi/2, nature='rcur', compute_geta=True)
                    valr[:,ind_selr]=tmp           
                    var = fid.createVariable('Vg', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx) 
                    print('geo cur written')


                if is_kc_cageos_rot:
                    tmp = comp[kc_cageos_rot].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=0., nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)                  
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('Uagr', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx) 

                    tmp = comp[kc_cageos_rot].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=pi/2, nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_svac_yawTrue)                  
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('Vagr', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx)
             
                    print('cageos_rot written')

                if is_kc_cageos_div:
                    tmp = comp[kc_cageos_div].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=0., nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)                  
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('Uagd', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx) 

                    tmp = comp[kc_cageos_div].operg(self.gridor[ind_selr],coord_time = self.time,coordtype='reg', cdir=pi/2, nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)                  
                    valr[:,ind_selr]=tmp
                    var = fid.createVariable('Vagd', 'f4', ('time','y','x'))
                    var[:] = valr.reshape(self.nt, self.ny, self.nx)

                    print('cageos_div written')

                if is_kc_io:
                    tmp = comp[kc_io].operg(self.gridor[ind_selr],coord_time = self.time_hf,coordtype='reg', cdir=0., nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)                  
                    valr_hf[:,ind_selr]=tmp
                    var = fid.createVariable('Uio', 'f4', ('time_hf','y','x'))
                    var[:] = valr_hf.reshape(self.nt_hf, self.ny, self.nx) 

                    tmp = comp[kc_io].operg(self.gridor[ind_selr],coord_time = self.time_hf,coordtype='reg', cdir=pi/2, nature='rcur', compute_geta=True)
                    #tmp = comp[kc_cageos_rot].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)                  
                    valr_hf[:,ind_selr]=tmp
                    var = fid.createVariable('Vio', 'f4', ('time_hf','y','x'))
                    var[:] = valr_hf.reshape(self.nt_hf, self.ny, self.nx)

                    print('io written')

                try:
                    var = fid.createVariable('MDH', 'f4', ('y','x'))
                    var[:]=self.MDT
                    var = fid.createVariable('MDU', 'f4', ('y','x'))
                    var[:]=self.MDU
                    var = fid.createVariable('MDV', 'f4', ('y','x'))
                    var[:]=self.MDV        
                except:
                    pass                       
                
            ######################################################################################
            ######################################################################################    
            print('Outputs written in ' + filenc)

class Grid_msit(Grid):


    """
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        
        with Dataset(path.join(self.TEMPLATE_FILE), 'r') as fcid:
            if self.FLAG_MDT:
                MDT = array(fcid.variables[self.MDT_NAME][:,:])
            lat = array(fcid.variables[self.LAT_NAME][:])
            lon = array(fcid.variables[self.LON_NAME][:])
            lon = mod(lon,360.)
            if self.FLAG_MDT:
                MDT[numpy.isnan(MDT)]=-10000
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()
            else:
                MDT = array(fcid.variables[self.MDT_NAME][:,:])
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()

            dlon = lon[1]-lon[0]
            dlat = lat[1]-lat[0]

        #zone = domain['ZONE']
        # lonmax=zone['LON_MAX']
        if (self.LON_MAX<self.LON_MIN): 
            #ix = where((lon>=self.LON_MIN-dlon)|(lon<=self.LON_MAX+dlon))[0]
            ix = numpy.concatenate((where(lon>=self.LON_MIN-dlon)[0], where(lon<=self.LON_MAX+dlon)[0]))
        else:
            ix = where((lon>=self.LON_MIN-dlon)&(lon<=self.LON_MAX+dlon))[0]
        iy = where((lat>=self.LAT_MIN-dlat)&(lat<=self.LAT_MAX+dlat))[0]
        self.lon = lon[ix]
        self.lat = lat[iy]
        (self.lon2, self.lat2) = meshgrid(self.lon,self.lat)

        if self.FLAG_MDT:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix]
            deg2m=110000.
            fc = 2*2*pi/86164*sin(self.lat2*pi/180)
            self.MDU = +self.MDT
            self.MDV = +self.MDT
            #pdb.set_trace()
            self.MDV[:-1,:-1] = 10./fc[:-1,:-1]*diff(self.MDT, axis=1)[:-1,:]/(diff(self.lon2,axis=1)[:-1,:]*deg2m*cos(self.lat2[:-1,:-1]*pi/180.))
            self.MDU[:-1,:-1] = -10./fc[:-1,:-1]*diff(self.MDT, axis=0)[:,:-1]/(diff(self.lat2,axis=0)[:,:-1]*deg2m)
        else:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]


        self.TIME_MIN= (datetime.strptime(self.DATE_MIN, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX= (datetime.strptime(self.DATE_MAX, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        # self.TIME_MIN= (datetime.date(self.DATE_MIN[0],self.DATE_MIN[1],self.DATE_MIN[2])-datetime.date(1950,1,1)).days
        # self.TIME_MAX= (datetime.date(self.DATE_MAX[0],self.DATE_MAX[1],self.DATE_MAX[2])-datetime.date(1950,1,1)).days
        deltat = self.TIME_MAX - self.TIME_MIN
        step = self.TIME_STEP
        self.time = arange(0.,deltat+step,step)
        self.timej = self.TIME_MIN + self.time
        self.nt = len(self.time)

        self.time_lf = arange(0.,deltat,self.TIME_STEP_LF)
        self.timej_lf = self.TIME_MIN + self.time_lf
        self.nt_lf = len(self.time_lf)   
  
        self.nx = len(self.lon)
        self.ny = len(self.lat)


    def write_outputs(self, comp, data_comp, config, rank, size):


        ensit = numpy.concatenate((numpy.arange(self.nt)[::self.NSTEPS_NC], [self.nt] ))
        ntasks = len(ensit)-1
        tasks= numpy.array_split(numpy.arange(ntasks), size)
        for itask in tasks[rank]:
            address = config['PATH']['OUTPUT']
            rootname = config['RUN_NAME']

            self.grido = type('', (), {})()
            #self.grido = empty(self.nt*self.nx*self.ny,dtype=[('lon','f4'),('lat','f4'),('time','f4')])
            time, lat, lon = meshgrid(self.time[ensit[itask]:ensit[itask+1]], self.lat, self.lon,indexing='ij')
            self.grido.time,self.grido.lat,self.grido.lon = time.flatten(), lat.flatten(), lon.flatten()

            self.gridor = type('', (), {})()
            #self.gridor = empty(self.nx*self.ny,dtype=[('lon','f4'),('lat','f4')])
            lat, lon = meshgrid(self.lat, self.lon,indexing='ij')
            self.gridor.lat,self.gridor.lon = lat.flatten(), lon.flatten()
                


            # MDT2 = tile(self.MDT,(self.nt,1,1)).flatten()
            # ind_sel = where((isnan(MDT2)==False))
            ind_selr = where((isnan(self.MDT.flatten())==False))[0]


            filenc = address +'/'+ rootname + '_ms_analysis_'+ str(int(self.timej[ensit[itask]])) +'to'+ str(int(self.timej[ensit[itask+1]-1])) +'.nc'
            if path.exists(filenc):remove(filenc)
            with Dataset(filenc,"w") as fid:
                fid.description = " Miost analysis "
                fid.createDimension('time', None)
                fid.createDimension('y', len(self.lat))
                fid.createDimension('x', len(self.lon))

                v=fid.createVariable('lon', 'f4', ('x'))
                v[:]=self.lon
                v=fid.createVariable('lat', 'f4', ('y'))
                v[:]=self.lat
                v=fid.createVariable('time', 'f4', ('time',))
                v[:]=self.timej[ensit[itask]:ensit[itask+1]]



                for kc in range(len(comp))[::-1]:
                    if comp[kc].write==True :
                        if ((comp[kc].name=='geo3ss')|(comp[kc].name=='geo3ls')|(comp[kc].name=='barotrop')) : 
                            valr = empty((ensit[itask+1]-ensit[itask],self.ny*self.nx))
                            valr[:] = nan
                            coords=[None]*3
                            coords[0]=self.gridor.lon[ind_selr]
                            coords[1]=self.gridor.lat[ind_selr]
                            coords[2]=self.time[ensit[itask]:ensit[itask+1]]
                            coords_name={'lon':0, 'lat':1, 'time':2}
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Hvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(ensit[itask+1]-ensit[itask], self.ny, self.nx)
                            #logging.info('%s h written', comp[kc].Hvarname)

                var = fid.createVariable('MDH', 'f4', ('y','x'))
                var[:]=self.MDT   
            logging.info(' %s written', filenc)     

        if rank==0:

            filenc = address +'/'+ rootname + '_it_analysis.nc'
            if path.exists(filenc):remove(filenc)
            with Dataset(filenc,"w") as fid:
                fid.description = " Miost analysis "
                fid.createDimension('time', None)
                fid.createDimension('time_hf', None)
                fid.createDimension('time_lf', None)
                fid.createDimension('y', len(self.lat))
                fid.createDimension('x', len(self.lon))

                v=fid.createVariable('lon', 'f4', ('x'))
                v[:]=self.lon
                v=fid.createVariable('lat', 'f4', ('y'))
                v[:]=self.lat

                v=fid.createVariable('time_lf', 'f4', ('time_lf',))
                v[:]=self.timej_lf      


                for kc in range(len(comp)):
                    if comp[kc].write==True :

                        if comp[kc].name[:2]=='iw' :
                            tmp = empty((self.ny*self.nx))
                            tmp[:] = nan
                            pvar = fid.createVariable('dphi_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            avar = fid.createVariable('a_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            varA = fid.createVariable('HitA_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varB = fid.createVariable('HitB_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            
                            Amp_astr, Phi_astr = jd2ap(comp[kc].file_data_astr, comp[kc].tidal_comp,self.timej_lf)
                            Amp0_astr, Phi0_astr = jd2ap(comp[kc].file_data_astr, comp[kc].tidal_comp, self.tref_iw)
                            dPhi_astr = Phi_astr - Phi0_astr
                            pvar[:] = dPhi_astr
                            avar[:] = Amp_astr

                            phiref = comp[kc].freq_tide*86400*self.tref_iw +  Phi0_astr

                            varA.tidal_comp=comp[kc].tidal_comp
                            varA.Ttide=comp[kc].Ttide
                            varA.tref=self.tref_iw
                            varA.phiref = phiref

                            for itlf in range(len(self.time_lf)): 
                                dt = -numpy.mod(comp[kc].freq_tide*86400*(self.timej_lf[itlf]-self.tref_iw)+phiref+dPhi_astr[itlf], 2*numpy.pi) / (comp[kc].freq_tide*86400)
                                coords=[None]*3
                                coords[0]=self.gridor.lon[ind_selr]
                                coords[1]=self.gridor.lat[ind_selr]
                                coords[2]=numpy.array(self.time_lf[itlf]+dt)
                                coords_name={'lon':0, 'lat':1, 'time':2}
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                                varA[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                coords[2] = numpy.array(self.time_lf[itlf] + dt + 0.25*comp[kc].Ttide/24 )
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                                varB[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]

                            logging.info('iw h written for %s  mode %s', comp[kc].tidal_comp , str(comp[kc].mode))                  
            logging.info(' %s written', filenc)      


class Grid_msithuv(Grid):


    """
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        
        with Dataset(path.join(self.TEMPLATE_FILE), 'r') as fcid:
            if self.FLAG_MDT:
                MDT = array(fcid.variables[self.MDT_NAME][:,:-1])
            lat = array(fcid.variables[self.LAT_NAME][:])
            lon = array(fcid.variables[self.LON_NAME][:-1])
            lon = mod(lon,360.)
            if self.FLAG_MDT:
                MDT[numpy.isnan(MDT)]=-10000
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()
            else:
                MDT = array(fcid.variables[self.MDT_NAME][0,:,:])
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()

            dlon = lon[1]-lon[0]
            dlat = lat[1]-lat[0]

        #zone = domain['ZONE']
        # lonmax=zone['LON_MAX']
        if (self.LON_MAX<self.LON_MIN): 
            #ix = where((lon>=self.LON_MIN-dlon)|(lon<=self.LON_MAX+dlon))[0]
            ix = numpy.concatenate((where(lon>=self.LON_MIN-dlon)[0], where(lon<=self.LON_MAX+dlon)[0]))
        else:
            ix = where((lon>=self.LON_MIN-dlon)&(lon<=self.LON_MAX+dlon))[0]
        iy = where((lat>=self.LAT_MIN-dlat)&(lat<=self.LAT_MAX+dlat))[0]
        self.lon = lon[ix]
        self.lat = lat[iy]
        (self.lon2, self.lat2) = meshgrid(self.lon,self.lat)

        if self.FLAG_MDT:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix]
            deg2m=110000.
            fc = 2*2*pi/86164*sin(self.lat2*pi/180)
            self.MDU = +self.MDT
            self.MDV = +self.MDT
            #pdb.set_trace()
            self.MDV[:-1,:-1] = 10./fc[:-1,:-1]*diff(self.MDT, axis=1)[:-1,:]/(diff(self.lon2,axis=1)[:-1,:]*deg2m*cos(self.lat2[:-1,:-1]*pi/180.))
            self.MDU[:-1,:-1] = -10./fc[:-1,:-1]*diff(self.MDT, axis=0)[:,:-1]/(diff(self.lat2,axis=0)[:,:-1]*deg2m)
        else:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]


        self.TIME_MIN= (datetime.strptime(self.DATE_MIN, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX= (datetime.strptime(self.DATE_MAX, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        # self.TIME_MIN= (datetime.date(self.DATE_MIN[0],self.DATE_MIN[1],self.DATE_MIN[2])-datetime.date(1950,1,1)).days
        # self.TIME_MAX= (datetime.date(self.DATE_MAX[0],self.DATE_MAX[1],self.DATE_MAX[2])-datetime.date(1950,1,1)).days
        deltat = self.TIME_MAX - self.TIME_MIN
        step = self.TIME_STEP
        self.time = arange(0.,deltat+step,step)
        self.timej = self.TIME_MIN + self.time
        self.nt = len(self.time)

        self.time_lf = arange(0.,deltat,self.TIME_STEP_LF)
        self.timej_lf = self.TIME_MIN + self.time_lf
        self.nt_lf = len(self.time_lf)   
  
        self.nx = len(self.lon)
        self.ny = len(self.lat)


    def write_outputs(self, comp, data_comp, config, rank, size):


        ensit = numpy.concatenate((numpy.arange(self.nt)[::self.NSTEPS_NC], [self.nt] ))
        ntasks = len(ensit)-1
        tasks= numpy.array_split(numpy.arange(ntasks), size)
        for itask in tasks[rank]:
            address = config['PATH']['OUTPUT']
            rootname = config['RUN_NAME']

            self.grido = type('', (), {})()
            #self.grido = empty(self.nt*self.nx*self.ny,dtype=[('lon','f4'),('lat','f4'),('time','f4')])
            time, lat, lon = meshgrid(self.time[ensit[itask]:ensit[itask+1]], self.lat, self.lon,indexing='ij')
            self.grido.time,self.grido.lat,self.grido.lon = time.flatten(), lat.flatten(), lon.flatten()

            self.gridor = type('', (), {})()
            #self.gridor = empty(self.nx*self.ny,dtype=[('lon','f4'),('lat','f4')])
            lat, lon = meshgrid(self.lat, self.lon,indexing='ij')
            self.gridor.lat,self.gridor.lon = lat.flatten(), lon.flatten()
                


            # MDT2 = tile(self.MDT,(self.nt,1,1)).flatten()
            # ind_sel = where((isnan(MDT2)==False))
            ind_selr = where((isnan(self.MDT.flatten())==False))[0]


            filenc = address +'/'+ rootname + '_ms_analysis_'+ str(int(self.timej[ensit[itask]])) +'to'+ str(int(self.timej[ensit[itask+1]-1])) +'.nc'
            if path.exists(filenc):remove(filenc)
            with Dataset(filenc,"w") as fid:
                fid.description = " Miost analysis "
                fid.createDimension('time', None)
                fid.createDimension('y', len(self.lat))
                fid.createDimension('x', len(self.lon))

                v=fid.createVariable('lon', 'f4', ('x'))
                v[:]=self.lon
                v=fid.createVariable('lat', 'f4', ('y'))
                v[:]=self.lat
                v=fid.createVariable('time', 'f4', ('time',))
                v[:]=self.timej[ensit[itask]:ensit[itask+1]]



                for kc in range(len(comp))[::-1]:
                    if comp[kc].write==True :
                        if ((comp[kc].name=='geo3ss')|(comp[kc].name=='geo3ls')|(comp[kc].name=='barotrop')) : 
                            valr = empty((ensit[itask+1]-ensit[itask],self.ny*self.nx))
                            valr[:] = nan
                            coords=[None]*3
                            coords[0]=self.gridor.lon[ind_selr]
                            coords[1]=self.gridor.lat[ind_selr]
                            coords[2]=self.time[ensit[itask]:ensit[itask+1]]
                            coords_name={'lon':0, 'lat':1, 'time':2}
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Hvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(ensit[itask+1]-ensit[itask], self.ny, self.nx)
                            #logging.info('%s h written', comp[kc].Hvarname)

                var = fid.createVariable('MDH', 'f4', ('y','x'))
                var[:]=self.MDT   
            logging.info(' %s written', filenc)     

        if rank==0:

            filenc = address +'/'+ rootname + '_it_analysis.nc'
            if path.exists(filenc):remove(filenc)
            with Dataset(filenc,"w") as fid:
                fid.description = " Miost analysis "
                fid.createDimension('time', None)
                fid.createDimension('time_hf', None)
                fid.createDimension('time_lf', None)
                fid.createDimension('y', len(self.lat))
                fid.createDimension('x', len(self.lon))

                v=fid.createVariable('lon', 'f4', ('x'))
                v[:]=self.lon
                v=fid.createVariable('lat', 'f4', ('y'))
                v[:]=self.lat

                v=fid.createVariable('time_lf', 'f4', ('time_lf',))
                v[:]=self.timej_lf      


                for kc in range(len(comp)):
                    if comp[kc].write==True :

                        if comp[kc].name[:2]=='iw' :
                            tmp = empty((self.ny*self.nx))
                            tmp[:] = nan
                            pvar = fid.createVariable('dphi_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            avar = fid.createVariable('a_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf'))
                            varA = fid.createVariable('HitA_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varB = fid.createVariable('HitB_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varAu = fid.createVariable('UitA_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varAv = fid.createVariable('VitA_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varBu = fid.createVariable('UitB_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))
                            varBv = fid.createVariable('VitB_'+comp[kc].tidal_comp+'mode'+str(comp[kc].mode), 'f4', ('time_lf','y','x'))                            
                            
                            Amp_astr, Phi_astr = jd2ap(comp[kc].file_data_astr, comp[kc].tidal_comp,self.timej_lf)
                            Amp0_astr, Phi0_astr = jd2ap(comp[kc].file_data_astr, comp[kc].tidal_comp, self.tref_iw)
                            dPhi_astr = Phi_astr - Phi0_astr
                            pvar[:] = dPhi_astr
                            avar[:] = Amp_astr

                            phiref = comp[kc].freq_tide*86400*self.tref_iw +  Phi0_astr

                            varA.tidal_comp=comp[kc].tidal_comp
                            varA.Ttide=comp[kc].Ttide
                            varA.tref=self.tref_iw
                            varA.phiref = phiref

                            for itlf in range(len(self.time_lf)): 
                                dt = -numpy.mod(comp[kc].freq_tide*86400*(self.timej_lf[itlf]-self.tref_iw)+phiref+dPhi_astr[itlf], 2*numpy.pi) / (comp[kc].freq_tide*86400)
                                coords=[None]*3
                                coords[0]=self.gridor.lon[ind_selr]
                                coords[1]=self.gridor.lat[ind_selr]
                                coords[2]=numpy.array(self.time_lf[itlf]+dt)
                                coords_name={'lon':0, 'lat':1, 'time':2}
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                                varA[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='rcur', cdir=0., compute_geta=True, eta=data_comp[kc])
                                varAu[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='rcur', cdir=numpy.pi/2, compute_geta=True, eta=data_comp[kc])
                                varAv[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]                                
                                coords[2] = numpy.array(self.time_lf[itlf] + dt + 0.25*comp[kc].Ttide/24 )
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                                varB[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='rcur', cdir=0., compute_geta=True, eta=data_comp[kc])
                                varBu[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]
                                tmp[ind_selr] = comp[kc].operg(coords=coords,coords_name=coords_name,coordtype='reg', nature='rcur', cdir=numpy.pi/2, compute_geta=True, eta=data_comp[kc])
                                varBv[itlf,:,:] = tmp.reshape(self.ny, self.nx) / Amp_astr[itlf]    


                            logging.info('iw h written for %s  mode %s', comp[kc].tidal_comp , str(comp[kc].mode))                  
            logging.info(' %s written', filenc)      

class Grid_mshuv(Grid):


    """
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        
        with Dataset(path.join(self.TEMPLATE_FILE), 'r') as fcid:
            if self.FLAG_MDT:
                MDT = array(fcid.variables[self.MDT_NAME][:,:-1])
            lat = array(fcid.variables[self.LAT_NAME][:])
            lon = array(fcid.variables[self.LON_NAME][:-1])
            lon = mod(lon,360.)
            if self.FLAG_MDT:
                MDT[numpy.isnan(MDT)]=-10000
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()
            else:
                MDT = array(fcid.variables[self.MDT_NAME][0,:,:])
                MDT[(MDT>100)|(MDT<-100)]=nan
                MDT=MDT.squeeze()

            dlon = lon[1]-lon[0]
            dlat = lat[1]-lat[0]

        #zone = domain['ZONE']
        # lonmax=zone['LON_MAX']
        if (self.LON_MAX<self.LON_MIN): 
            #ix = where((lon>=self.LON_MIN-dlon)|(lon<=self.LON_MAX+dlon))[0]
            ix = numpy.concatenate((where(lon>=self.LON_MIN-dlon)[0], where(lon<=self.LON_MAX+dlon)[0]))
        else:
            ix = where((lon>=self.LON_MIN-dlon)&(lon<=self.LON_MAX+dlon))[0]
        iy = where((lat>=self.LAT_MIN-dlat)&(lat<=self.LAT_MAX+dlat))[0]
        self.lon = lon[ix]
        self.lat = lat[iy]
        (self.lon2, self.lat2) = meshgrid(self.lon,self.lat)

        if self.FLAG_MDT:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix]
            deg2m=110000.
            fc = 2*2*pi/86164*sin(self.lat2*pi/180)
            self.MDU = +self.MDT
            self.MDV = +self.MDT
            #pdb.set_trace()
            self.MDV[:-1,:-1] = 10./fc[:-1,:-1]*diff(self.MDT, axis=1)[:-1,:]/(diff(self.lon2,axis=1)[:-1,:]*deg2m*cos(self.lat2[:-1,:-1]*pi/180.))
            self.MDU[:-1,:-1] = -10./fc[:-1,:-1]*diff(self.MDT, axis=0)[:,:-1]/(diff(self.lat2,axis=0)[:,:-1]*deg2m)
        else:
            self.MDT = MDT[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]


        self.TIME_MIN= (datetime.strptime(self.DATE_MIN, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        self.TIME_MAX= (datetime.strptime(self.DATE_MAX, '%Y-%m-%d')-datetime.strptime('1950-1-1', '%Y-%m-%d')).days
        # self.TIME_MIN= (datetime.date(self.DATE_MIN[0],self.DATE_MIN[1],self.DATE_MIN[2])-datetime.date(1950,1,1)).days
        # self.TIME_MAX= (datetime.date(self.DATE_MAX[0],self.DATE_MAX[1],self.DATE_MAX[2])-datetime.date(1950,1,1)).days
        deltat = self.TIME_MAX - self.TIME_MIN
        step = self.TIME_STEP
        self.time = arange(0.,deltat+step,step)
        self.timej = self.TIME_MIN + self.time
        self.nt = len(self.time)
  
        self.nx = len(self.lon)
        self.ny = len(self.lat)


    def write_outputs(self, comp, data_comp, config, rank, size):


        ensit = numpy.concatenate((numpy.arange(self.nt)[::self.NSTEPS_NC], [self.nt] ))
        ntasks = len(ensit)-1
        tasks= numpy.array_split(numpy.arange(ntasks), size)
        for itask in tasks[rank]:
            address = config['PATH']['OUTPUT']
            rootname = config['RUN_NAME']

            self.grido = type('', (), {})()
            #self.grido = empty(self.nt*self.nx*self.ny,dtype=[('lon','f4'),('lat','f4'),('time','f4')])
            time, lat, lon = meshgrid(self.time[ensit[itask]:ensit[itask+1]], self.lat, self.lon,indexing='ij')
            self.grido.time,self.grido.lat,self.grido.lon = time.flatten(), lat.flatten(), lon.flatten()

            self.gridor = type('', (), {})()
            #self.gridor = empty(self.nx*self.ny,dtype=[('lon','f4'),('lat','f4')])
            lat, lon = meshgrid(self.lat, self.lon,indexing='ij')
            self.gridor.lat,self.gridor.lon = lat.flatten(), lon.flatten()
                


            # MDT2 = tile(self.MDT,(self.nt,1,1)).flatten()
            # ind_sel = where((isnan(MDT2)==False))
            ind_selr = where((isnan(self.MDT.flatten())==False))[0]


            filenc = address +'/'+ rootname + '_ms_analysis_'+ str(int(self.timej[ensit[itask]])) +'to'+ str(int(self.timej[ensit[itask+1]-1])) +'.nc'
            if path.exists(filenc):remove(filenc)
            with Dataset(filenc,"w") as fid:
                fid.description = " Miost analysis "
                fid.createDimension('time', None)
                fid.createDimension('y', len(self.lat))
                fid.createDimension('x', len(self.lon))

                v=fid.createVariable('lon', 'f4', ('x'))
                v[:]=self.lon
                v=fid.createVariable('lat', 'f4', ('y'))
                v[:]=self.lat
                v=fid.createVariable('time', 'f4', ('time',))
                v[:]=self.timej[ensit[itask]:ensit[itask+1]]



                for kc in range(len(comp))[::-1]:
                    if comp[kc].write==True :
                        if ((comp[kc].name=='geo3ss')|(comp[kc].name=='geo3ls')|(comp[kc].name=='barotrop')) : 
                            valr = empty((ensit[itask+1]-ensit[itask],self.ny*self.nx))
                            valr[:] = nan
                            coords=[None]*3
                            coords[0]=self.gridor.lon[ind_selr]
                            coords[1]=self.gridor.lat[ind_selr]
                            coords[2]=self.time[ensit[itask]:ensit[itask+1]]
                            coords_name={'lon':0, 'lat':1, 'time':2}
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, coordtype='reg', nature='sla', compute_geta=True, eta=data_comp[kc])
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Hvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(ensit[itask+1]-ensit[itask], self.ny, self.nx)
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, cdir=0, coordtype='reg', nature='rcur', compute_geta=True, eta=data_comp[kc])
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Uvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(ensit[itask+1]-ensit[itask], self.ny, self.nx)
                            tmp = comp[kc].operg(coords=coords,coords_name=coords_name, cdir=numpy.pi/2, coordtype='reg', nature='rcur', compute_geta=True, eta=data_comp[kc])
                            valr[:,ind_selr]=tmp
                            var = fid.createVariable(comp[kc].Vvarname, 'f4', ('time','y','x'))
                            var[:] = valr.reshape(ensit[itask+1]-ensit[itask], self.ny, self.nx)                            

                var = fid.createVariable('MDH', 'f4', ('y','x'))
                var[:]=self.MDT   
                var = fid.createVariable('MDU', 'f4', ('y','x'))
                var[:]=self.MDU 
                var = fid.createVariable('MDV', 'f4', ('y','x'))
                var[:]=self.MDV                 
            logging.info(' %s written', filenc)     









class Grid_swath(Grid):


    """
    """
    
    # yaml_tag = u'!Grid_swath'


    # def __setstate__(self, state):
    #     print(dir(self))
    #     self.__init__(**state)  

    def __init__(self,**kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(Grid_swath,self).__init__(**kwargs)

        template_file = self.dirname + self.rootname+'_L2C_c'+str(self.CYCLE).zfill(2)+'_p'+str(self.PASS).zfill(3)+'.nc'

        with Dataset(path.join(template_file), 'r') as fcid:

            self.al = array(fcid.variables[self.AL_NAME][:])
            self.ac = array(fcid.variables[self.AC_NAME][:])
            self.lon = array(fcid.variables[self.LON_NAME][:,:])
            self.lat = array(fcid.variables[self.LAT_NAME][:,:])            
            time_nadir = array(fcid.variables[self.TIME_NADIR_NAME][:]) 
            self.time0 = time_nadir[0]
            self.time_nadir = time_nadir #- self.time0
            MDT = array(fcid.variables[self.MDT_NAME][:,:])
            MDT[(MDT>100)|(MDT<-100)]=nan
            self.MDT=MDT.squeeze()
            self.uac_true = array(fcid.variables[self.UAC_TRUE_NAME][:,:])
            self.ual_true = array(fcid.variables[self.UAL_TRUE_NAME][:,:])
            self.uac_true[(self.uac_true<-100.)|(self.uac_true==0)]=nan
            self.ual_true[(self.ual_true<-100.)|(self.ual_true==0)]=nan

        (self.al2, self.ac2) = meshgrid(self.al,self.ac)
        self.nal = len(self.al)
        self.nac = len(self.ac)
        self.al_min = self.al.min()
        self.al_max = self.al.max()
        self.ac_min = self.ac.min()
        self.ac_max = self.ac.max()        

        self.time_hf = arange(self.time_nadir[0],self.time_nadir[-1],self.DT_HF/86400) 





    def write_outputs(self, comp, config):


        address = config['PATH']['OUTPUT']

        self.grido = empty(self.nal*self.nac,dtype=[('al','f4'),('ac','f4')])
        al, ac = meshgrid(self.al, self.ac,indexing='ij')
        self.grido['ac'],self.grido['al'] = ac.flatten(), al.flatten()
               
        ind_sel = where((isnan(self.MDT.flatten())==False))[0]

        val = empty((self.nal*self.nac))
        val[:] = nan



        filenc = address + self.rootname+'_'+config['RUN_NAME']+'_L2C_c'+str(self.CYCLE).zfill(2)+'_p'+str(self.PASS).zfill(3)+'.nc'
        if path.exists(filenc):remove(filenc)

        fid=nc.Dataset(filenc,"w")
        # fid.history = "Created " + datetime.today().strftime("%d/%m/%y")

        fid.createDimension('time', len(self.time_nadir))
        fid.createDimension('ac', len(self.ac))

        v=fid.createVariable('time', 'f4', ('time'))
        v[:]=self.time_nadir
        #v.units = 'seconds since the beginning of the sampling'
        v.long_name='time'

        v=fid.createVariable('xac', 'f4', ('ac'))
        v[:]=self.ac
        v.long_name='across-track distance'

        v=fid.createVariable('xal', 'f4', ('time'))
        v[:]=self.al
        v.long_name='across-track distance'


        v=fid.createVariable('lon', 'f4', ('time','ac'))
        v[:]=self.lon
        v.units = 'degrees east'
        v.long_name='longitude'

        v=fid.createVariable('lat', 'f4', ('time','ac'))
        v[:]=self.lat
        v.units = 'degrees north'
        v.long_name='latitude'

        var = fid.createVariable('Ual_true', 'f4', ('time','ac'))
        var[:] = self.ual_true

        var = fid.createVariable('Uac_true', 'f4', ('time','ac'))
        var[:] = self.uac_true


        Uac = zeros((self.nal, self.nac))
        Ual = zeros((self.nal, self.nac))
        count_yaw = 0

        for kc in range(len(comp)):
            if comp[kc].write==True :
                if comp[kc].name=='ps_rot':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_rot cur written')
                    

                if comp[kc].name=='ps_div':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_div cur written')  

                if type(comp[kc]).__name__=='CompSkimYaw':
                    count_yaw += 1
                    if count_yaw==1: 
                        Uac_yaw = zeros((len(self.time_nadir)))
                        Uac_yaw_hf = zeros((len(self.time_hf)))
                    coordinates = empty(len(self.time_nadir), dtype=[('time','f4')])
                    coordinates.time=self.time_nadir
                    Uac_yaw += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)
                    coordinates = empty(len(self.time_hf), dtype=[('time','f4')])
                    coordinates.time=self.time_hf
                    Uac_yaw_hf += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)

        if (count_yaw>0):
            var = fid.createVariable('Uac_yaw', 'f4', ('time'))
            var[:] = Uac_yaw
            fid.createDimension('time_hf', len(self.time_hf))
            var = fid.createVariable('time_hf', 'f4', ('time_hf'))
            var[:] = self.time_hf
            var = fid.createVariable('Uac_yaw_hf', 'f4', ('time_hf'))
            var[:] = Uac_yaw_hf

            ko=comp[kc].obs_datasets[0]
            if config['OBS'][ko].yaw_error['Flag']==True:
                fileyaw = config['OBS'][ko].yaw_error['sample']
                fidn = Dataset(fileyaw)
                time_yaw = array(fidn.variables.time[:])
                vac_yaw = array(fidn.variables['vac_yaw'][:])
                fidn.close()
                finterp_vac_yaw = interp1d(time_yaw,vac_yaw)
                Err_yaw = finterp_vac_yaw(mod(self.time_nadir*86400,time_yaw.max())) * config['OBS'][ko].yaw_error['data_fac']
                var = fid.createVariable('Uac_yaw_true', 'f4', ('time'))
                var[:] = Err_yaw

                yaw_hf= finterp_vac_yaw(mod(self.time_hf*86400,time_yaw.max())) * config['OBS'][ko].yaw_error['data_fac']
                var = fid.createVariable('Uac_yaw_true_hf', 'f4', ('time_hf'))
                var[:] = yaw_hf


                


        var = fid.createVariable('Uac', 'f4', ('time','ac'))
        Uac[isnan(self.uac_true)]=nan
        var[:] = Uac
        var = fid.createVariable('Ual', 'f4', ('time','ac'))
        Ual[isnan(self.ual_true)]=nan
        var[:] = Ual
        
        fid.close()
        print('')
        print('#############################################')
        print(filenc+' written')
        print('#############################################')
        print('')

class Grid_swath_glo(Grid):


    """
    """
    
    # yaml_tag = u'!Grid_swath_glo'


    # def __setstate__(self, state):
    #     print(dir(self))
    #     self.__init__(**state)  

    def __init__(self,**kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(Grid_swath_glo,self).__init__(**kwargs)

        self.al = []
        self.lon = []
        self.lat = []
        self.time_nadir = []
        self.MDT = []
        self.uac_true = []
        self.ual_true = []

        count = 0
        for CYCLE in numpy.arange(self.CYCLE0,self.CYCLE1+1):
            pass0=1
            pass1=self.NPASS
            if CYCLE==self.CYCLE0: pass0=self.PASS0
            if CYCLE==self.CYCLE1: pass1=self.PASS1
            for PASS in numpy.arange(pass0,pass1+1):

                template_file = self.dirname + self.rootname+'_L2C_c'+str(CYCLE).zfill(2)+'_p'+str(PASS).zfill(3)+'.nc'

                with Dataset(path.join(template_file), 'r') as fcid:

                    al = array(fcid.variables[self.AL_NAME][:])
                    self.ac = array(fcid.variables[self.AC_NAME][:])
                    lon = array(fcid.variables[self.LON_NAME][:,:])
                    lat = array(fcid.variables[self.LAT_NAME][:,:])            
                    time_nadir = array(fcid.variables[self.TIME_NADIR_NAME][:]) 
                    MDT = array(fcid.variables[self.MDT_NAME][:,:])
                    MDT[(MDT>100)|(MDT<-100)]=nan
                    MDT=MDT.squeeze()
                    if numpy.mod(PASS,2)==0:
                        uac_true = array(fcid.variables[self.UAC_TRUE_NAME][:,:])
                        ual_true = array(fcid.variables[self.UAL_TRUE_NAME][:,:])
                    if numpy.mod(PASS,2)==1:
                        uac_true = -array(fcid.variables[self.UAC_TRUE_NAME][:,:])
                        ual_true = -array(fcid.variables[self.UAL_TRUE_NAME][:,:])                    
                    uac_true[(uac_true<-100.)|(uac_true==0)]=nan
                    ual_true[(ual_true<-100.)|(ual_true==0)]=nan
                    uac_true[(uac_true>100.)|(uac_true==0)]=nan
                    ual_true[(ual_true>100.)|(ual_true==0)]=nan

                    time_hf = arange(time_nadir[0],time_nadir[-1],self.DT_HF/86400)

                    count+=1
                    if count == 1:
                        self.al = al
                        self.lon = lon
                        self.lat = lat
                        self.time_nadir = time_nadir
                        self.time_hf = time_hf
                        self.MDT = MDT
                        self.uac_true = uac_true
                        self.ual_true = ual_true
                    else:
                        self.al = numpy.concatenate((self.al, al))
                        self.lon = numpy.concatenate(([self.lon, lon]))
                        self.lat = numpy.concatenate(([self.lat, lat]))
                        self.time_nadir = numpy.concatenate((self.time_nadir, time_nadir))
                        self.time_hf = numpy.concatenate((self.time_hf, time_hf))
                        self.MDT = numpy.concatenate(([self.MDT, MDT]))
                        self.uac_true = numpy.concatenate(([self.uac_true, uac_true]))
                        self.ual_true = numpy.concatenate(([self.ual_true, ual_true]))

        # Refill time_hf because of trimmed swath in Lucile's run
        self.time_hf = arange(self.time_hf[0],self.time_hf[-1],self.DT_HF/86400)

        self.time0 = self.time_nadir[0]
        (self.al2, self.ac2) = meshgrid(self.al,self.ac)
        self.nal = len(self.al)
        self.nac = len(self.ac)
        self.al_min = self.al.min()
        self.al_max = self.al.max()
        self.ac_min = self.ac.min()
        self.ac_max = self.ac.max()        





    def write_outputs(self, comp, config):


        address = config['PATH']['OUTPUT']

        self.grido = empty(self.nal*self.nac,dtype=[('al','f4'),('ac','f4')])
        al, ac = meshgrid(self.al, self.ac,indexing='ij')
        self.grido['ac'],self.grido['al'] = ac.flatten(), al.flatten()
               
        ind_sel = where((isnan(self.MDT.flatten())==False))[0]

        val = empty((self.nal*self.nac))
        val[:] = nan



        filenc = address + self.rootname+'_'+config['RUN_NAME']+'_L2C_c'+str(self.CYCLE0).zfill(2)+'_p'+str(self.PASS0).zfill(3)+'_c'+str(self.CYCLE1).zfill(2)+'_p'+str(self.PASS1).zfill(3)+'.nc'
        if path.exists(filenc):remove(filenc)

        fid=nc.Dataset(filenc,"w")
        # fid.history = "Created " + datetime.today().strftime("%d/%m/%y")

        fid.createDimension('time', len(self.time_nadir))
        fid.createDimension('ac', len(self.ac))

        v=fid.createVariable('time', 'f4', ('time'))
        v[:]=self.time_nadir
        #v.units = 'seconds since the beginning of the sampling'
        v.long_name='time'

        v=fid.createVariable('xac', 'f4', ('ac'))
        v[:]=self.ac
        v.long_name='across-track distance'

        v=fid.createVariable('xal', 'f4', ('time'))
        v[:]=self.al
        v.long_name='across-track distance'


        v=fid.createVariable('lon', 'f4', ('time','ac'))
        v[:]=self.lon
        v.units = 'degrees east'
        v.long_name='longitude'

        v=fid.createVariable('lat', 'f4', ('time','ac'))
        v[:]=self.lat
        v.units = 'degrees north'
        v.long_name='latitude'

        var = fid.createVariable('Ual_true', 'f4', ('time','ac'))
        var[:] = self.ual_true

        var = fid.createVariable('Uac_true', 'f4', ('time','ac'))
        var[:] = self.uac_true


        Uac = zeros((self.nal, self.nac))
        Ual = zeros((self.nal, self.nac))
        count_yaw = 0

        for kc in range(len(comp)):
            if comp[kc].write==True :
                if comp[kc].name=='ps_rot':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_rot cur written')
                    

                if comp[kc].name=='ps_div':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_div cur written')  

                if type(comp[kc]).__name__=='CompSkimYaw':
                    count_yaw += 1
                    if count_yaw==1: 
                        Uac_yaw = zeros((len(self.time_nadir)))
                        Uac_yaw_hf = zeros((len(self.time_hf)))
                    coordinates = empty(len(self.time_nadir), dtype=[('time','f4')])
                    coordinates.time=self.time_nadir
                    Uac_yaw += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)
                    coordinates = empty(len(self.time_hf), dtype=[('time','f4')])
                    coordinates.time=self.time_hf
                    Uac_yaw_hf += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)

        if (count_yaw>0):
            var = fid.createVariable('Uac_yaw', 'f4', ('time'))
            var[:] = Uac_yaw
            fid.createDimension('time_hf', len(self.time_hf))
            var = fid.createVariable('time_hf', 'f4', ('time_hf'))
            var[:] = self.time_hf
            var = fid.createVariable('Uac_yaw_hf', 'f4', ('time_hf'))
            var[:] = Uac_yaw_hf

            ko=comp[kc].obs_datasets[0]
            if config['OBS'][ko].yaw_error['Flag']==True:
                fileyaw = config['OBS'][ko].yaw_error['sample']
                fidn = Dataset(fileyaw)
                time_yaw = array(fidn.variables.time[:])
                #vac_yaw = array(fidn.variables['vac_yaw'][:])
                vsat = fidn.variables['vsat'] 
                vac_yaw = numpy.array(fidn.variables['yaw_angle'][:]) * 1e-6 * vsat
                fidn.close()
                finterp_vac_yaw = interp1d(time_yaw,vac_yaw)
                Err_yaw = finterp_vac_yaw(mod(self.time_nadir*86400,time_yaw.max())) * config['OBS'][ko].yaw_error['data_fac']
                var = fid.createVariable('Uac_yaw_true', 'f4', ('time'))
                var[:] = Err_yaw

                yaw_hf= finterp_vac_yaw(mod(self.time_hf*86400,time_yaw.max())) * config['OBS'][ko].yaw_error['data_fac']
                var = fid.createVariable('Uac_yaw_true_hf', 'f4', ('time_hf'))
                var[:] = yaw_hf


                


        var = fid.createVariable('Uac', 'f4', ('time','ac'))
        Uac[isnan(self.uac_true)]=nan
        var[:] = Uac
        var = fid.createVariable('Ual', 'f4', ('time','ac'))
        Ual[isnan(self.ual_true)]=nan
        var[:] = Ual
        
        fid.close()
        print('')
        print('#############################################')
        print(filenc+' written')
        print('#############################################')
        print('')

class Grid_swath2019_glo(Grid):


    """
    """
    
    # yaml_tag = u'!Grid_swath2019_glo'


    # def __setstate__(self, state):
    #     print(dir(self))
    #     self.__init__(**state)  

    def __init__(self,**kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(Grid_swath2019_glo,self).__init__(**kwargs)

        self.al = []
        self.lon = []
        self.lat = []
        self.time_nadir = []
        self.MDT = []
        self.uac_true = []
        self.ual_true = []

        count = 0
        for CYCLE in numpy.arange(self.CYCLE0,self.CYCLE1+1):
            pass0=1
            pass1=self.NPASS
            if CYCLE==self.CYCLE0: pass0=self.PASS0
            if CYCLE==self.CYCLE1: pass1=self.PASS1
            for PASS in numpy.arange(pass0,pass1+1):

                template_file = self.dirname + self.rootname+'_l2c_c'+str(CYCLE).zfill(2)+'_p'+str(PASS).zfill(3)+'.nc'

                with Dataset(path.join(template_file), 'r') as fcid:

                    al = array(fcid.variables[self.AL_NAME][:])
                    self.ac = array(fcid.variables[self.AC_NAME][:])
                    lon = array(fcid.variables[self.LON_NAME][:,:])
                    lat = array(fcid.variables[self.LAT_NAME][:,:])            
                    time_nadir = array(fcid.variables[self.TIME_NADIR_NAME][:]) 
                    MDT = array(fcid.variables[self.MDT_NAME][:,:])
                    MDT[(MDT>100)|(MDT<-100)]=nan
                    MDT=MDT.squeeze()
                    if numpy.mod(PASS,2)==0:
                        uac_true = array(fcid.variables[self.UAC_TRUE_NAME][:,:])
                        ual_true = array(fcid.variables[self.UAL_TRUE_NAME][:,:])
                    if numpy.mod(PASS,2)==1:
                        uac_true = -array(fcid.variables[self.UAC_TRUE_NAME][:,:])
                        ual_true = -array(fcid.variables[self.UAL_TRUE_NAME][:,:])                    
                    uac_true[(uac_true<-100.)|(uac_true==0)]=nan
                    ual_true[(ual_true<-100.)|(ual_true==0)]=nan
                    uac_true[(uac_true>100.)|(uac_true==0)]=nan
                    ual_true[(ual_true>100.)|(ual_true==0)]=nan

                    time_hf = arange(time_nadir[0],time_nadir[-1],self.DT_HF/86400)

                    count+=1
                    if count == 1:
                        self.al = al
                        self.lon = lon
                        self.lat = lat
                        self.time_nadir = time_nadir
                        self.time_hf = time_hf
                        self.MDT = MDT
                        self.uac_true = uac_true
                        self.ual_true = ual_true
                    else:
                        self.al = numpy.concatenate((self.al, al))
                        self.lon = numpy.concatenate(([self.lon, lon]))
                        self.lat = numpy.concatenate(([self.lat, lat]))
                        self.time_nadir = numpy.concatenate((self.time_nadir, time_nadir))
                        self.time_hf = numpy.concatenate((self.time_hf, time_hf))
                        self.MDT = numpy.concatenate(([self.MDT, MDT]))
                        self.uac_true = numpy.concatenate(([self.uac_true, uac_true]))
                        self.ual_true = numpy.concatenate(([self.ual_true, ual_true]))

        # Refill time_hf because of trimmed swath in Lucile's run
        self.time_hf = arange(self.time_hf[0],self.time_hf[-1],self.DT_HF/86400)

        self.time0 = self.time_nadir[0]
        (self.al2, self.ac2) = meshgrid(self.al,self.ac)
        self.nal = len(self.al)
        self.nac = len(self.ac)
        self.al_min = self.al.min()
        self.al_max = self.al.max()
        self.ac_min = self.ac.min()
        self.ac_max = self.ac.max()        





    def write_outputs(self, comp, config):


        address = config['PATH']['OUTPUT']

        self.grido = empty(self.nal*self.nac,dtype=[('al','f4'),('ac','f4')])
        al, ac = meshgrid(self.al, self.ac,indexing='ij')
        self.grido['ac'],self.grido['al'] = ac.flatten(), al.flatten()
               
        ind_sel = where((isnan(self.MDT.flatten())==False))[0]

        val = empty((self.nal*self.nac))
        val[:] = nan



        filenc = address + self.rootname+'_'+config['RUN_NAME']+'_l2c_c'+str(self.CYCLE0).zfill(2)+'_p'+str(self.PASS0).zfill(3)+'_c'+str(self.CYCLE1).zfill(2)+'_p'+str(self.PASS1).zfill(3)+'.nc'
        if path.exists(filenc):remove(filenc)

        fid=nc.Dataset(filenc,"w")
        # fid.history = "Created " + datetime.today().strftime("%d/%m/%y")

        fid.createDimension('time', len(self.time_nadir))
        fid.createDimension('ac', len(self.ac))

        v=fid.createVariable('time', 'f4', ('time'))
        v[:]=self.time_nadir
        #v.units = 'seconds since the beginning of the sampling'
        v.long_name='time'

        v=fid.createVariable('xac', 'f4', ('ac'))
        v[:]=self.ac
        v.long_name='across-track distance'

        v=fid.createVariable('xal', 'f4', ('time'))
        v[:]=self.al
        v.long_name='across-track distance'


        v=fid.createVariable('lon', 'f4', ('time','ac'))
        v[:]=self.lon
        v.units = 'degrees east'
        v.long_name='longitude'

        v=fid.createVariable('lat', 'f4', ('time','ac'))
        v[:]=self.lat
        v.units = 'degrees north'
        v.long_name='latitude'

        var = fid.createVariable('Ual_true', 'f4', ('time','ac'))
        var[:] = self.ual_true

        var = fid.createVariable('Uac_true', 'f4', ('time','ac'))
        var[:] = self.uac_true


        Uac = zeros((self.nal, self.nac))
        Ual = zeros((self.nal, self.nac))
        count_yaw = 0

        for kc in range(len(comp)):
            if comp[kc].write==True :
                if comp[kc].name=='ps_rot':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_rot', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_rot cur written')
                    

                if comp[kc].name=='ps_div':
                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=0., nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp
                    var = fid.createVariable('Uac_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Uac += val.reshape(self.nal, self.nac)

                    tmp = comp[kc].operg(self.grido[ind_sel], cdir=pi/2, nature='rcur', compute_geta=True)
                    val[ind_sel]=tmp           
                    var = fid.createVariable('Ual_div', 'f4', ('time','ac'))
                    var[:] = val.reshape(self.nal, self.nac) 
                    Ual += val.reshape(self.nal, self.nac)
                    print('ps_div cur written')  

                if ((type(comp[kc]).__name__=='CompSkimYawTed')|(type(comp[kc]).__name__=='CompSkimYawTedPer')|(type(comp[kc]).__name__=='CompSkimYawTedPerM')):
                    print('compute and write yaw TED...')
                    psi = numpy.linspace(0,2*pi,120)
                    indt=numpy.where((self.time_hf<0.1))[0]
                    coordinates = empty((len(self.time_hf[indt]),len(psi)), dtype=[('time','f4'), ('angle_from_ac','f4'), ('beam_inc','f4')])
                    coordinates.time[:,:] = numpy.meshgrid(psi,self.time_hf[indt])[1]
                    coordinates['angle_from_ac'][:,:] = numpy.meshgrid(psi,self.time_hf[indt])[0]
                    coordinates['beam_inc'][:,:] = 12.
                    yaw_ted = comp[kc].operg(coordinates.flatten(), compute_geta=True).reshape(len(self.time_hf[indt]),len(psi))

                    file = '/home/cubelmann/Data/FilesErikTED_ConfA/ADS_EQ_reference_35750_10_ted_Az_arcsec_Rant.tab'
                    yaw0=numpy.empty((120,1201))
                    for ii in range(120):
                        with open (file, 'r') as f:
                            data = [row[1+ii] for row in csv.reader(f,delimiter='\t')]
                        for k in range(len(data))[:-1]:
                            yaw0[ii,k]=float(data[k+1])*4.84813681109536
                    file= '/home/cubelmann/Data/FilesErikTED_ConfA/ADS_P15_cold_35750_10_ted_Az_arcsec_Rant.tab'
                    yaw2=numpy.empty((120,1201))
                    for ii in range(120):
                        with open (file, 'r') as f:
                            data = [row[1+ii] for row in csv.reader(f,delimiter='\t')]
                        for k in range(len(data))[:-1]:
                            yaw2[ii,k]=float(data[k+1])*4.84813681109536

                    # file= '/home/cubelmann/Data/FilesErikTED_ConfA/ADS_P15_cold_35750_10_ted_Az_arcsec_Rant.tab'
                    # yaw2=numpy.empty((120,1201))
                    # for ii in range(120):
                    #     with open (file, 'r') as f:
                    #         data = [row[1+ii] for row in csv.reader(f,delimiter='\t')]
                    #     for k in range(len(data))[:-1]:
                    #         yaw2[ii,k]=float(data[k+1])*4.84813681109536 
                    # yaw2m = numpy.mean(yaw2,axis=1)
                    # yaw2a = yaw2.transpose()-yaw2m
                    # yaw2a = yaw2a.transpose()

                    time=numpy.arange(0,1201,1.)*5.
                    psi=numpy.arange(0,360,3)
                    finterp = interpolate.interp2d(time,numpy.concatenate((psi,[360])),numpy.concatenate((yaw2-yaw0,yaw2[0,:].reshape(1,1201)-yaw0[0,:].reshape(1,1201)),axis=0))
                    #finterp = interpolate.interp2d(time,numpy.concatenate((psi,[360])),numpy.concatenate((yaw2a,yaw2a[0,:].reshape(1,1201)),axis=0))
                    yaw_ted_true = numpy.zeros((len(coordinates.flatten())))
                    for ii in range(len(coordinates.flatten())):
                        yaw_ted_true[ii] = finterp(numpy.mod(coordinates.time.flatten()[ii]*86400,comp[kc].torb),coordinates['angle_from_ac'].flatten()[ii]*180/numpy.pi) * 1e-6 * comp[kc].vsat * cos(coordinates['angle_from_ac'].flatten()[ii])
                    
                    yaw_ted_true = yaw_ted_true.reshape(len(self.time_hf[indt]),len(psi))

                    with open(comp[kc].file_outputs, 'wb') as f:
                        pickle.dump([psi,self.time_hf[indt],yaw_ted,yaw_ted_true],f)

                    plt.close()
                    plt.figure()
                    plt.pcolormesh(psi,self.time_hf[indt],yaw_ted-yaw_ted_true,vmin=-0.1,vmax=0.1, cmap='seismic')
                    plt.colorbar()
                    plt.savefig('l2bres1')
                    plt.ion()
                    plt.show()
                    pdb.set_trace()

                if type(comp[kc]).__name__=='CompSkimYaw':
                    count_yaw += 1
                    if count_yaw==1: 
                        Uac_yaw = zeros((len(self.time_nadir)))
                        Uac_yaw_hf = zeros((len(self.time_hf)))
                    coordinates = empty(len(self.time_nadir), dtype=[('time','f4')])
                    coordinates.time=self.time_nadir
                    Uac_yaw += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)
                    coordinates = empty(len(self.time_hf), dtype=[('time','f4')])
                    coordinates.time=self.time_hf
                    #tmp = comp[kc].operg(coordinates)
                    Uac_yaw_hf += comp[kc].operg(coordinates, cdir=0., nature='rcur', compute_geta=True)

        if (count_yaw>0):
            var = fid.createVariable('Uac_yaw', 'f4', ('time'))
            var[:] = Uac_yaw
            fid.createDimension('time_hf', len(self.time_hf))
            var = fid.createVariable('time_hf', 'f4', ('time_hf'))
            var[:] = self.time_hf
            var = fid.createVariable('Uac_yaw_hf', 'f4', ('time_hf'))
            var[:] = Uac_yaw_hf

            ko=comp[kc].obs_datasets[0]
            if config['OBS'][ko].add_yawerr['Flag']==True:
                fileyaw = config['OBS'][ko].add_yawerr['sample']
                fidn = Dataset(fileyaw)
                time_yaw = array(fidn.variables.time[:])
                #vac_yaw = array(fidn.variables['vac_yaw'][:])
                vsat = fidn.variables['vsat'] 
                vac_yaw = numpy.array(fidn.variables['yaw_angle'][:]) * 1e-6 * vsat
                fidn.close()
                finterp_vac_yaw = interp1d(time_yaw,vac_yaw)
                Err_yaw = finterp_vac_yaw(mod(self.time_nadir*86400,time_yaw.max())) * config['OBS'][ko].add_yawerr['data_fac']
                var = fid.createVariable('Uac_yaw_true', 'f4', ('time'))
                var[:] = Err_yaw

                yaw_hf= finterp_vac_yaw(mod(self.time_hf*86400,time_yaw.max())) * config['OBS'][ko].add_yawerr['data_fac']
                var = fid.createVariable('Uac_yaw_true_hf', 'f4', ('time_hf'))
                var[:] = yaw_hf


                


        var = fid.createVariable('Uac', 'f4', ('time','ac'))
        Uac[isnan(self.uac_true)]=nan
        var[:] = Uac
        var = fid.createVariable('Ual', 'f4', ('time','ac'))
        Ual[isnan(self.ual_true)]=nan
        var[:] = Ual
        
        fid.close()
        print('')
        print('#############################################')
        print(filenc+' written')
        print('#############################################')
        print('')


     
