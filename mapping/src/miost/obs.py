# -*- coding: utf-8 -*-
"""
"""
import os 
import logging
import numpy
from numpy import empty
import scipy
import pandas as pd 

############

from glob import glob
from netCDF4 import Dataset
import xarray as xr
from datetime import datetime, timedelta
###import h5py
import pdb
import matplotlib.pylab as plt 


# import dask.array
# import dask.distributed


##import zarr

class Obs(object):

    __slots__ = (
        'noise', 'dirname','root1','root2','nature', 'substract_mdt', 'name','coords','coords_name','data_val','data_noise'
    )
    IDENT = '' 

    def __init__(self, COMP=[], substract_mdt=False,**kwargs): 
        self.substract_mdt = substract_mdt
        self.COMP = COMP


class ObsSla(Obs):
    """
    """

    __slots__ = (
        'nature'
        'data',
        'nobs',
        'sat_list', 'sad_noise', 'psd2noise', 'COMP'
        )



    def __init__(self, **kwargs):

        self.nature = 'sla'
        # self.data = empty((0),dtype=[('lon','f4'),('lat','f4'),('time','f4'),('val','f4'),('noise','f4'),('innov','f4'),
        #              ('sens','f4'),('guess','f4')])

        super(ObsSla,self).__init__(**kwargs)

    def Substract_mdt(self, grid):

        logging.debug('substract mdt')
        #mss = numpy.zeros(self.nobs)
        finterp=scipy.interpolate.RegularGridInterpolator([grid.lat,grid.lon],grid.MDT)
        for kk in numpy.arange(0,self.nobs,1000):
            iind = numpy.arange(kk,numpy.minimum(kk+1000,self.nobs))
            try:
              tmp = finterp((self.data['lat'][iind],self.data['lon'][iind]))
            except: pdb.set_trace()
            #mss[iind] = +tmp
            self.data['val'][iind] += - tmp
        logging.debug('substract mdt done')
        indkeep = numpy.where((self.data['val']>-9999)&(self.data['val']<9999))[0]
        self.data = self.data[indkeep]
        self.nobs = len(indkeep)

class ObsRcur(Obs):
    """
    """

    __slots__ = (
        'nature'
        'data',
        'nobs',
        'Jo',
        )

    def __init__(self, **kwargs):
        self.nature='rcur'
        self.data = empty((0),dtype=[('lon','f4'),('lat','f4'),('time','f4'),('val','f4'),('angle','f4'),('noise','f4'),
                    ('innov','f4'),('sens','f4'),('guess','f4')])

        super(ObsRcur,self).__init__(**kwargs)



    def Substract_mdt(self, grid):
        deg2m=110000.
        logging.debug('substract mvgeo')

        finterpx=scipy.interpolate.RegularGridInterpolator([grid.lat,grid.lon],grid.MDU)
        finterpy=scipy.interpolate.RegularGridInterpolator([grid.lat,grid.lon],grid.MDV)

        for kk in numpy.arange(0,self.nobs,1000):
            iind = numpy.arange(kk,numpy.minimum(kk+1000,self.nobs))
            try:
                tmp = numpy.cos(self.data['angle'][iind])*finterpx((self.data['lat'][iind],self.data['lon'][iind])) + numpy.sin(self.data['angle'][iind])*finterpy((self.data['lat'][iind],self.data['lon'][iind]))
            except:
                pdb.set_trace()
            self.data['val'][iind] += - tmp
        logging.debug('substract mvgeo done')
        indkeep = numpy.where((self.data['val']>-9999)&(self.data['val']<9999))[0]
        self.data = self.data[indkeep]
        self.nobs = len(indkeep)




###################################################################
###################################################################

class MASSH(ObsSla):
    """
    """
    __slots__ = ('name','dict_obs','subsampling','noise','data_lon','data_lat','data_time','nobs', 'nobs_tot')



    def __init__(self, dict_obs,noise=0.03,subsampling=1,**kwargs):
        self.noise=noise
        self.dict_obs = dict_obs
        self.subsampling = subsampling
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(MASSH,self).__init__(**kwargs)



    def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        
        time0 = datetime(2003,1,1) + timedelta(days=box[4]-19358)
        time1 = datetime(2003,1,1) + timedelta(days=box[5]-19358)

        lon= numpy.array([])
        lat= numpy.array([])
        time= numpy.array([])
        ssh= numpy.array([])
        
        for dt in self.dict_obs:
            
            if (dt<=time1) & (dt>=time0):
                
                    path_obs = self.dict_obs[dt]['obs_name']
                    sat =  self.dict_obs[dt]['satellite']
                    
                    for _sat,_path_obs in zip(sat,path_obs):
                        
                        ds = xr.open_dataset(_path_obs).squeeze() 
                        time_obs = ds[_sat.name_obs_time].values
                        time_obs = (time_obs-numpy.datetime64(time0))/numpy.timedelta64(1, 'D')
                        lon_obs = ds[_sat.name_obs_lon] % 360
                        lat_obs = ds[_sat.name_obs_lat]
                        
                        ds = ds.where((lon0<=lon_obs) & (lon1>=lon_obs) & 
                  (lat0<=lat_obs) & (lat1>=lat_obs), drop=True)
                        
                        if _sat.kind=='fullSSH':
                            if len(ds[_sat.name_obs_lon].shape)==1:
                                lon_obs = ds[_sat.name_obs_lon].values[::self.subsampling]
                                lat_obs = ds[_sat.name_obs_lat].values[::self.subsampling]
                                lon_obs,lat_obs = numpy.meshgrid(lon_obs,lat_obs)
                            else:
                                lon_obs = ds[_sat.name_obs_lon].values[::self.subsampling,::self.subsampling]
                                lat_obs = ds[_sat.name_obs_lat].values[::self.subsampling,::self.subsampling]
                            ssh_obs = ds[_sat.name_obs_var[0]].values[::self.subsampling,::self.subsampling]
                        
                        ds.close()
                        del ds
                        
                        # Flattening
                        lon1d = lon_obs.ravel()
                        lat1d = lat_obs.ravel()
                        ssh1d = ssh_obs.ravel()
                        
                        # Remove NaN pixels
                        indNoNan= ~numpy.isnan(ssh1d)
                        lon1d = lon1d[indNoNan]
                        lat1d = lat1d[indNoNan]
                        ssh1d = ssh1d[indNoNan]    
                        
                        # Append to arrays
                        time = numpy.append(time,time_obs*numpy.ones(lon1d.size))
                        lon = numpy.append(lon,lon1d)
                        lat = numpy.append(lat,lat1d)
                        ssh = numpy.append(ssh,ssh1d)
        
        ssh[numpy.isnan(ssh)] = 0
                    
        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
        values=None
        noise=None
        if len(time)>0:
            indsort = numpy.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]
                lat=lat[indsort]
                time=time[indsort]
                ssh=ssh[indsort]

            coords[coords_att['lon']] = lon
            coords[coords_att['lat']] = lat
            coords[coords_att['time']] = time      
            values =  ssh
            noise =  self.noise * numpy.ones(time.size)
                    
        return [values, noise, coords, coords_att]
                        
    

# class fullSSH(ObsSla):
#     """
#     """
#     __slots__ = ('subsampling','name_time', 'name_lon', 'name_lat','name','name_ssh','data_lon','data_lat','data_time','nobs', 'nobs_tot')



#     def __init__(self, noise=0.03,**kwargs):
#         self.noise=noise
#         #self.dirname = kwargs.get('dirname')
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         super(fullSSH,self).__init__(**kwargs)



#     def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

#         lon0 = box[0]
#         lon1 = box[1]
#         lat0 = box[2]
#         lat1 = box[3]
        
#         time0 = numpy.datetime64('2003-01-01') + numpy.timedelta64(box[4]-19358,'D')
#         time1 = numpy.datetime64('2003-01-01') + numpy.timedelta64(box[5]-19358,'D')

#         lon=[]
#         lat=[]
#         time=[]
#         sla=[]
        
#         # open dataset 
#         ds = xr.open_mfdataset(os.path.join(self.dirname,self.root1+'*.nc'))
        
#         # Selecting data
#         time_obs = ds[self.name_time]
#         lon_obs = ds[self.name_lon] % 360
#         lat_obs = ds[self.name_lat]
        
#         ds = ds.where((time_obs>=time0)&(time_obs<=time1)&(lon0<=lon_obs) & (lon1>=lon_obs) & 
#                   (lat0<=lat_obs) & (lat1>=lat_obs), drop=True) 
        
#         print(ds)
        
#         time_obs = ds[self.name_time].values
#         lon_obs = ds[self.name_lon].values[::self.subsampling]
#         lat_obs = ds[self.name_lat].values[::self.subsampling]
#         ssh_obs = ds[self.name_ssh].values[:,::self.subsampling,::self.subsampling]
        
#         time = []
#         lon = []
#         lat = []
#         sla = []
        
#         for t,_time in enumerate(time_obs):
#             for i,_lat in enumerate(lat_obs):
#                 for j,_lon in enumerate(lon_obs):
#                     time.append(_time)
#                     lon.append(_lon)
#                     lat.append(_lat)
#                     sla.append(ssh_obs[t,i,j])
        
#         time = numpy.asarray(time)
#         lon = numpy.asarray(lon)
#         lat = numpy.asarray(lat)
#         sla = numpy.asarray(sla)
        
#         sla[numpy.isnan(sla)] = 0
        
#         coords = [None]*3
#         coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
#         values=None
#         noise=None
#         if len(time)>0:
#             indsort = numpy.argsort(time)
#             if len(indsort)>0:
#                 lon=lon[indsort]
#                 lat=lat[indsort]
#                 time=time[indsort]
#                 sla=sla[indsort]

#             coords[coords_att['lon']] = numpy.array((lon), dtype=float_type) 
#             coords[coords_att['lat']] = numpy.array((lat), dtype=float_type)
#             coords[coords_att['time']] = numpy.array((time-time0)/ numpy.timedelta64(1, 'D'), dtype=float_type)       
#             values =  numpy.array((sla),dtype=float_type)  
#             noise =  self.noise * numpy.ones(time.size)
                    
#         return [values, noise, coords, coords_att]
    
   
    
class Cmems(ObsSla):

    """
    """
    __slots__ = ('creation_date', 'var2add', 'var2rem','name','data_lon','data_lat','data_time','nobs', 'nobs_tot')



    def __init__(self, noise=0.03,**kwargs):
        self.noise=noise
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(Cmems,self).__init__(**kwargs)



    def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        time0 = box[4]
        time1 = box[5]

        date0 = datetime(2003, 1, 1) + timedelta(time0-19358)
        date0=date0.strftime('%Y%m%d')
        year0=date0[:4]
        month0=date0[4:6]
        day0=date0[6:8]

        date1 = datetime(2003, 1, 1) + timedelta(time1-19358)
        date1=date1.strftime('%Y%m%d')
        year1=date1[:4]
        month1=date1[4:6]
        day1=date1[6:8]

        lon=[]
        lat=[]
        time=[]
        sla=[]

        is_first_file=numpy.full(len(self.sat_list),True)
        first_file=[None]*len(self.sat_list)
        last_file=[None]*len(self.sat_list)

        count_file=0
        
        totsize=0
        indsize=numpy.zeros((len(self.sat_list)),'i')
        for isat, sat in enumerate(self.sat_list):
            for year in numpy.arange(int(year0),int(year1)+1):
                m0=1
                m1=12
                if year==int(year0): m0=int(month0)
                if year==int(year1): m1=int(month1)
                for month in numpy.arange(m0,m1+1):
                    d0=1
                    d1=31
                    if ((year==int(year0))&(month==int(month0))): d0=int(day0)
                    if ((year==int(year1))&(month==int(month1))): d1=int(day1)
                    for day in numpy.arange(d0,d1+1):
                        #filen = glob(f'{self.dirname}/{sat}/'+ str(year) + '/'+self.root1+'_' + sat + '_'+self.root2+'_' + str(year).zfill(4) + str(month).zfill(2) + str(day).zfill(2) +'_'+self.creation_date+'.nc')
                        filen = glob(self.dirname + '/' + sat + '/'+ str(year) + '/'+self.root1+'_' + sat + '_'+self.root2+'_' + str(year).zfill(4) + str(month).zfill(2) + str(day).zfill(2) +'_'+self.creation_date+'.nc')
                        if len(filen)>1: 
                            print('file ambiguity')
                            pdb.set_trace()
                        if len(filen)==1:
                            count_file += 1
                            if numpy.mod(count_file+chunk, nchunks)==0:
                            # if is_first_file[isat]==True:
                            #     first_file[isat]=filen  
                            #     is_first_file[isat]=False
                            # last_file[isat]=filen
                            
                                fid = Dataset(filen[0])
                                timep = numpy.array(fid.variables['time'][:]) 
                                inds=numpy.where((timep>=time0) & (timep<(time1)))[0]
                                if len(inds)>0:
                                    lonp=numpy.array((fid.variables['longitude'][inds]),dtype=float_type)
                                    latp=numpy.array((fid.variables['latitude'][inds]),dtype=float_type)
                                    indss=numpy.where((lonp>lon0 ) &( lonp<lon1 ) & (latp>lat0 ) &( latp<lat1 ) )[0]
                                    if len(indss)>0:
                                        slal=numpy.zeros((len(indss)))
                                        for var in self.var2add:
                                            slal+=numpy.array((fid.variables[var][inds[indss]]),dtype=float_type)
                                        for var in self.var2rem:
                                            slal-=numpy.array((fid.variables[var][inds[indss]]),dtype=float_type)
                                        lonl=lonp[indss]
                                        latl=latp[indss]
                                        timel=timep[inds[indss]]
                                        time=numpy.concatenate([time,timel])
                                        lon=numpy.concatenate([lon,lonl])
                                        lat=numpy.concatenate([lat,latl])
                                        sla=numpy.concatenate([sla,slal])

        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
        values=None
        noise=None
        if len(time)>0:
            indsort = numpy.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]
                lat=lat[indsort]
                time=time[indsort]
                sla=sla[indsort]

            fcid = Dataset(self.sad_noise, 'r')
            glon = numpy.array(fcid.variables['lon'][:])
            glat = numpy.array(fcid.variables['lat'][:])
            NOISEFLOOR = numpy.array(fcid.variables['NOISEFLOOR'][:,:])
            finterpNOISEFLOOR = scipy.interpolate.RegularGridInterpolator((glat,glon),NOISEFLOOR,bounds_error=False,fill_value=None)
            noise=numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type)

            coords[coords_att['lon']] = numpy.array((lon), dtype=float_type) 
            coords[coords_att['lat']] = numpy.array((lat), dtype=float_type)
            coords[coords_att['time']] = numpy.array((time-time0), dtype=float_type)       
            values =  numpy.array((sla),dtype=float_type)  
            noise =  numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type) 

        return [values, noise, coords, coords_att]

class CmemsAvg(ObsSla):

    """
    """
    __slots__ = ('creation_date', 'var2add', 'var2rem','name','data_lon','data_lat','data_time','nobs', 'nobs_tot', 'q')



    def __init__(self, noise=0.03,**kwargs):
        self.noise=noise
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CmemsAvg,self).__init__(**kwargs)



    def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        time0 = box[4]
        time1 = box[5]

        date0 = datetime(2003, 1, 1) + timedelta(time0-19358)
        date0=date0.strftime('%Y%m%d')
        year0=date0[:4]
        month0=date0[4:6]
        day0=date0[6:8]

        date1 = datetime(2003, 1, 1) + timedelta(time1-19358)
        date1=date1.strftime('%Y%m%d')
        year1=date1[:4]
        month1=date1[4:6]
        day1=date1[6:8]

        lon=[]
        lat=[]
        time=[]
        sla=[]

        is_first_file=numpy.full(len(self.sat_list),True)
        first_file=[None]*len(self.sat_list)
        last_file=[None]*len(self.sat_list)

        count_file=0
        
        totsize=0
        indsize=numpy.zeros((len(self.sat_list)),'i')
        for isat, sat in enumerate(self.sat_list):
            for year in numpy.arange(int(year0),int(year1)+1):
                m0=1
                m1=12
                if year==int(year0): m0=int(month0)
                if year==int(year1): m1=int(month1)
                for month in numpy.arange(m0,m1+1):
                    d0=1
                    d1=31
                    if ((year==int(year0))&(month==int(month0))): d0=int(day0)
                    if ((year==int(year1))&(month==int(month1))): d1=int(day1)
                    for day in numpy.arange(d0,d1+1):
                        #filen = glob(f'{self.dirname}/{sat}/'+ str(year) + '/'+self.root1+'_' + sat + '_'+self.root2+'_' + str(year).zfill(4) + str(month).zfill(2) + str(day).zfill(2) +'_'+self.creation_date+'.nc')
                        filen = glob(self.dirname + '/' + sat + '/'+ str(year) + '/'+self.root1+'_' + sat + '_'+self.root2+'_' + str(year).zfill(4) + str(month).zfill(2) + str(day).zfill(2) +'_'+self.creation_date+'.nc')
                        if len(filen)>1: 
                            print('file ambiguity')
                            pdb.set_trace()
                        if len(filen)==1:
                            count_file += 1
                            if numpy.mod(count_file+chunk, nchunks)==0:
                            # if is_first_file[isat]==True:
                            #     first_file[isat]=filen  
                            #     is_first_file[isat]=False
                            # last_file[isat]=filen
                            
                                fid = Dataset(filen[0])
                                timep = numpy.array(fid.variables['time'][:]) 
                                inds=numpy.where((timep>=time0) & (timep<(time1)))[0]
                                if len(inds)>0:
                                    lonp=numpy.array((fid.variables['longitude'][inds]),dtype=float_type)
                                    latp=numpy.array((fid.variables['latitude'][inds]),dtype=float_type)
                                    if lon1>lon0: indss=numpy.where((lonp>lon0 ) &( lonp<lon1 ) & (latp>lat0 ) &( latp<lat1 ) )[0]
                                    else: indss=numpy.where(((lonp>lon0 ) | ( lonp<lon1 )) & (latp>lat0 ) &( latp<lat1 ) )[0]
                                    if len(indss)>0:
                                        slal=numpy.zeros((len(indss)))
                                        for var in self.var2add:
                                            slal+=numpy.array((fid.variables[var][inds[indss]]),dtype=float_type)
                                        for var in self.var2rem:
                                            slal-=numpy.array((fid.variables[var][inds[indss]]),dtype=float_type)
                                        lonl=lonp[indss]
                                        latl=latp[indss]
                                        timel=timep[inds[indss]]


                                        ########################
                                        #######################
                                        timelr = []
                                        slalr = []
                                        lonlr = []
                                        latlr = []
                                        iic=numpy.where((numpy.diff(timel)>1.6e-5))
                                        tmp_timel=numpy.array_split(timel,iic[0]+1)
                                        #print(timel)
                                        tmp_slal=numpy.array_split(slal,iic[0]+1)
                                        tmp_lonl=numpy.array_split(lonl,iic[0]+1)
                                        tmp_latl=numpy.array_split(latl,iic[0]+1)
                                        for k in range(len(tmp_timel)):
                                            ensi = numpy.arange(0,len(tmp_timel[k]),self.q)
                                            #print(ensi)
                                            ir=numpy.mod(len(tmp_timel[k]),self.q)

                                            #print(tmp_timelr)
                                            #print(numpy.mean(tmp_timel[k][-ir:]))
                                            if ir>0: 
                                                tmp_timelr = numpy.mean(tmp_timel[k][:-ir].reshape(-1, self.q), axis=1)
                                                tmp_slalr = numpy.mean(tmp_slal[k][:-ir].reshape(-1, self.q), axis=1)
                                                tmp_lonlr = numpy.mean(tmp_lonl[k][:-ir].reshape(-1, self.q), axis=1)
                                                tmp_latlr = numpy.mean(tmp_latl[k][:-ir].reshape(-1, self.q), axis=1)
                                                
                                                tmp_timelr = numpy.concatenate((tmp_timelr,[numpy.mean(tmp_timel[k][-ir:])]))
                                                tmp_slalr = numpy.concatenate((tmp_slalr,[numpy.mean(tmp_slal[k][-ir:])]))
                                                tmp_lonlr = numpy.concatenate((tmp_lonlr,[numpy.mean(tmp_lonl[k][-ir:])]))
                                                tmp_latlr = numpy.concatenate((tmp_latlr,[numpy.mean(tmp_latl[k][-ir:])]))
                                            else:
                                                tmp_timelr = numpy.mean(tmp_timel[k].reshape(-1, self.q), axis=1)
                                                tmp_slalr = numpy.mean(tmp_slal[k].reshape(-1, self.q), axis=1)
                                                tmp_lonlr = numpy.mean(tmp_lonl[k].reshape(-1, self.q), axis=1)
                                                tmp_latlr = numpy.mean(tmp_latl[k].reshape(-1, self.q), axis=1)

                                            timelr = numpy.concatenate((timelr,tmp_timelr))
                                            slalr = numpy.concatenate((slalr,tmp_slalr))
                                            lonlr = numpy.concatenate((lonlr,tmp_lonlr))
                                            latlr = numpy.concatenate((latlr,tmp_latlr))
                                        #######################
                                        #######################




                                        time=numpy.concatenate([time,timelr])
                                        lon=numpy.concatenate([lon,lonlr])
                                        lat=numpy.concatenate([lat,latlr])
                                        sla=numpy.concatenate([sla,slalr])

        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
        values=None
        noise=None
        if len(time)>0:
            indsort = numpy.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]
                lat=lat[indsort]
                time=time[indsort]
                sla=sla[indsort]

            fcid = Dataset(self.sad_noise, 'r')
            glon = numpy.array(fcid.variables['lon'][:])
            glat = numpy.array(fcid.variables['lat'][:])
            NOISEFLOOR = numpy.array(fcid.variables['NOISEFLOOR'][:,:])
            finterpNOISEFLOOR = scipy.interpolate.RegularGridInterpolator((glat,glon),NOISEFLOOR,bounds_error=False,fill_value=None)
            noise=numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type)

            coords[coords_att['lon']] = numpy.array((lon), dtype=float_type) 
            coords[coords_att['lat']] = numpy.array((lat), dtype=float_type)
            coords[coords_att['time']] = numpy.array((time-time0), dtype=float_type)       
            values =  numpy.array((sla),dtype=float_type)  
            noise =  numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type) / self.q**0.5

        return [values, noise, coords, coords_att]


class CmemsZarr(ObsSla):

    """
    """
    __slots__ = ('creation_date', 'varkey','name','data_lon','data_lat','data_time','nobs', 'nobs_tot')



    def __init__(self, noise=0.03,**kwargs):
        self.noise=noise
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(CmemsZarr,self).__init__(**kwargs)



    



    def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        time0 = box[4]
        time1 = box[5]


        lon=[]
        lat=[]
        time=[]
        sla=[]

        for isat, sat in enumerate(self.sat_list):
            #logging.info('load sat : %s', sat)
            alti=zarr.open(self.dirname + '/' + sat, mode='r')
            t0=alti.time[0]
            t1=alti.time[-1]
            if ((t0<(time1-7305.)*86400*1e6)&(t1>=(time0-7305.)*86400*1e6)):
                indt =  numpy.where((alti.time[:]>=(time0-7305.)*86400*1e6) & (alti.time[:]<(time1-7305.)*86400*1e6))[0]
                i0 = numpy.linspace(indt[0],indt[-1]+1,nchunks+1, dtype='int32')[chunk]
                i1 = numpy.linspace(indt[0],indt[-1]+1,nchunks+1, dtype='int32')[chunk+1]
                # i0=indt[0]
                # i1=indt[-1]+1
                inds = numpy.where((alti.latitude[i0:i1]>=lat0)&(alti.latitude[i0:i1]<lat1)&(alti.longitude[i0:i1]>=lon0)&(alti.longitude[i0:i1]<lon1))[0]
                lonl = alti.longitude[i0:i1][inds]
                latl = alti.latitude[i0:i1][inds]
                timel = alti.time[i0:i1][inds]/86400*1e-6+7305.
                if self.varkey=='sla-lwe':
                    slal = alti.sla_unfiltered[i0:i1][inds] - alti.lwe[i0:i1][inds]

                time=numpy.concatenate([time,timel])
                lon=numpy.concatenate([lon,lonl])
                lat=numpy.concatenate([lat,latl])
                sla=numpy.concatenate([sla,slal])


        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }
        values=None
        noise=None
        if len(time)>0:
            indsort = numpy.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]
                lat=lat[indsort]
                time=time[indsort]
                sla=sla[indsort]

            fcid = Dataset(self.sad_noise, 'r')
            glon = numpy.array(fcid.variables['lon'][:])
            glat = numpy.array(fcid.variables['lat'][:])
            NOISEFLOOR = numpy.array(fcid.variables['NOISEFLOOR'][:,:])
            finterpNOISEFLOOR = scipy.interpolate.RegularGridInterpolator((glat,glon),NOISEFLOOR,bounds_error=False,fill_value=None)
            noise=numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type)

            coords[coords_att['lon']] = numpy.array((lon), dtype=float_type) 
            coords[coords_att['lat']] = numpy.array((lat), dtype=float_type)
            coords[coords_att['time']] = numpy.array((time-time0), dtype=float_type)       
            values =  numpy.array((sla),dtype=float_type)  
            noise =  numpy.array((self.psd2noise*finterpNOISEFLOOR((lat,lon))), dtype=float_type) 

        return [values, noise, coords, coords_att]


class Drifters(ObsRcur):

    def __init__(self, noise=0.03,**kwargs):
        self.noise=noise
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(Drifters,self).__init__(**kwargs)



    def get_obs(self,grid, float_type='f4'):


        lon0 = grid.LON_MIN
        lon1 = grid.LON_MAX
        lat0 = grid.LAT_MIN
        lat1 = grid.LAT_MAX
        time0 = grid.TIME_MIN
        time1 = grid.TIME_MAX

        time,lon,lat,u,v=numpy.loadtxt(self.dirname+self.data_file,skiprows=0,usecols=(1,2,3,4,5),unpack=True)

        inds=numpy.where((time>=time0) & (time<(time1)))[0]
        time=time[inds]
        lon=lon[inds]
        lat=lat[inds]
        u=u[inds]
        v=v[inds]

        inds=numpy.where((lon>lon0 ) &( lon<lon1 ) & (lat>lat0 ) &( lat<lat1 ) )[0]
        time=time[inds]
        lon=lon[inds]
        lat=lat[inds]
        u=u[inds]
        v=v[inds]        

        time=numpy.concatenate((time,time))
        lon=numpy.concatenate((lon,lon))
        lat=numpy.concatenate((lat,lat))
        val=numpy.concatenate((u,v))
        angle = numpy.concatenate((u*0,v*0+numpy.pi/2))


        self.nobs=len(time)


        if self.nobs>0:
            self.coords = [None]*4
            self.coords[0] = numpy.array((lon), dtype=float_type) #dask.array.from_array(lon, chunks=(self.nobs // params_algo['NB_BLOCKS'],))
            self.coords[1] = numpy.array((lat), dtype=float_type) #dask.array.from_array(lat, chunks=(self.nobs // params_algo['NB_BLOCKS'],))
            self.coords[2] = numpy.array((time-grid.TIME_MIN), dtype=float_type) #dask.array.from_array(time - time0, chunks=(self.nobs // params_algo['NB_BLOCKS'],))
            self.coords[3] = numpy.array((angle), dtype=float_type)
            self.coords_name={'lon':0, 'lat':1, 'time':2, 'angle':3}
            self.data_val = numpy.array((val),dtype=float_type)
            self.data_noise = numpy.array((self.noise), dtype=float_type)

            #pdb.set_trace()
            if self.substract_mdt:
                self.Substract_mdt(grid)
        

class DriftersGps(ObsRcur):

    def __init__(self, noise=0.03,**kwargs):
        self.noise=noise
        #self.dirname = kwargs.get('dirname')
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(DriftersGps,self).__init__(**kwargs)



    ##def get_obs(self,grid, float_type='f4'):
    def get_obs(self,box, float_type='f4', chunk=0, nchunks=1):

        if nchunks>1: exit # To do implement lecture partagée

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        time0 = box[4]
        time1 = box[5]

        delta = datetime(1979, 1, 1) - datetime(1950, 1, 1)

        fid = Dataset(self.dirname+self.data_file)
        drogue =  numpy.array(fid.variables['DROGUE'][:])
        time = numpy.array(fid.variables['TIME'][:]) / 24. + delta.days
        lon = numpy.array(fid.variables['LON'][:]) 
        lat = numpy.array(fid.variables['LAT'][:]) 
        u = numpy.array(fid.variables['U'][:])
        v = numpy.array(fid.variables['V'][:])
        fid.close()

        inds = numpy.where((drogue==1) & (numpy.abs(u)<2.) & (numpy.abs(v)<2.) & (lon>lon0 ) &( lon<lon1 ) & (lat>lat0 ) &( lat<lat1 ) & (time>=time0)&(time<=time1) )[0]

        u = u[inds]
        v = v[inds]
        time = time[inds]
        lon = lon[inds]
        lat = lat[inds]   

        time=numpy.concatenate((time,time))
        lon=numpy.concatenate((lon,lon))
        lat=numpy.concatenate((lat,lat))
        val=numpy.concatenate((u,v))
        angle = numpy.concatenate((u*0,v*0+numpy.pi/2))


        self.nobs=len(time)
      


        coords = [None]*4
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'angle':3, 'nobs':len(time) }
        values=None
        noise=None
        if len(time)>0:
            # indsort = numpy.argsort(time)
            # if len(indsort)>0:
            #     lon=lon[indsort]
            #     lat=lat[indsort]
            #     time=time[indsort]
            #     sla=sla[indsort]

            coords[coords_att['lon']] = numpy.array((lon), dtype=float_type) 
            coords[coords_att['lat']] = numpy.array((lat), dtype=float_type)
            coords[coords_att['time']] = numpy.array((time-time0), dtype=float_type)     
            coords[coords_att['angle']] = numpy.array((angle), dtype=float_type)  
            values =  numpy.array((val),dtype=float_type)  
            noise =  numpy.array((time*0.+self.noise), dtype=float_type)

        return [values, noise, coords, coords_att]
