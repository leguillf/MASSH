#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:35:02 2021

@author: leguillou
"""

import numpy as np
import xarray as xr
import sys,os
import pandas as pd 
from copy import deepcopy
import matplotlib.pylab as plt

from . import grid

class State:
    """
    NAME
       ini
    DESCRIPTION
        Main function calling subfunctions considering the kind of init the
    user set
        Args:
            config (module): configuration module
    """
    
   # __slots__ = ('config','lon','lat','var','name_lon','name_lat','name_var','name_exp_save','path_save','ny','nx','f','g')
    
    def __init__(self,config):
        
        self.config = config
        
        # Parameters
        self.name_lon = config.name_mod_lon
        self.name_lat = config.name_mod_lat
        self.name_var = config.name_mod_var
        self.name_exp_save = config.name_exp_save
        self.path_save = config.path_save
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
        self.g = config.g
        
        # Initialize grid
        self.geo_grid = False
        if config.name_init == 'geo_grid':
             self.ini_geo_grid(config)
        elif config.name_init == 'from_file':
             self.ini_from_file(config)
        else:
            sys.exit("Initialization '" + config.name_init + "' not implemented yet")
            
        self.ny,self.nx = self.lon.shape
        self.f = 4*np.pi/86164*np.sin(self.lat*np.pi/180)
        
        # Compute cartesian grid 
        DX,DY = grid.lonlat2dxdy(self.lon,self.lat)
        dx = np.mean(DX)
        dy = np.mean(DY)
        X,Y = grid.dxdy2xy(DX,DY)
        self.DX = DX
        self.DY = DY
        self.X = X
        self.Y = Y
        self.dx = dx
        self.dy = dy
        
        #  Initialize state variables
        self.var = pd.Series(dtype=np.float64)
        if config.name_model=='QG1L':
            self.ini_var_qg1l(config)
        elif config.name_model=='SW1L':
            self.ini_var_sw1l(config)
        else:
            sys.exit("Model '" + config.name_model + "' not implemented yet")
            
        if not os.path.exists(config.tmp_DA_path):
            os.makedirs(config.tmp_DA_path)
        
    def __str__(self):
        message = ''
        for name in self.name_var:
            message += name + ':' + str(self.var[name].shape) + '\n'
        return message

    def ini_geo_grid(self,config):
        """
        NAME
            ini_geo_grid
    
        DESCRIPTION
            Create state grid, regular in (lon,lat) and save to init file
            Args:
                config (module): configuration module
        """
        self.geo_grid = True
        lon = np.arange(config.lon_min, config.lon_max + config.dx, config.dx) % 360
        lat = np.arange(config.lat_min, config.lat_max + config.dy, config.dy) 
        lon,lat = np.meshgrid(lon,lat)
        self.lon = lon % 360
        self.lat = lat
    
    def ini_from_file(self,config):
        """
        NAME
            ini_from_file
    
        DESCRIPTION
            Copy state grid from existing file and save to init file
            Args:
                config (module): configuration module
        """
        dsin = xr.open_dataset(config.name_init_grid)
        lon = dsin[config.name_init_lon].values
        lat = dsin[config.name_init_lat].values
        if len(lon.shape)==1:
            self.geo_grid = True
            lon,lat = np.meshgrid(lon,lat)
        self.lon = lon % 360
        self.lat = lat
        dsin.close()
        del dsin
        
    def ini_var_qg1l(self,config):
        """
        NAME
            ini_var_qg1l
    
        DESCRIPTION
            Initialize QG1L state variables. First one is SSH, 
            second one (optional) is Potential Voriticy 
            and third one (optional) is f/c where f is the Coriolis frequency
            and c the phase velocity of the first baroclinic Rossby Radius.
        """
        if len(self.name_var) not in [1,2,3]:
            sys.exit('For QG1L: wrong number variable names')
        for i, var in enumerate(self.name_var):
            if (config.name_init == 'from_file') and (config.name_init_var is not None) and (i==0):
                dsin = xr.open_dataset(config.name_init_grid)
                var_init = dsin[config.name_init_var]
                if len(var_init.shape)==3:
                    var_init = var_init[0,:,:]
                self.var[var] = var_init.values
                dsin.close()
                del dsin
            else:
                self.var[var] = np.zeros((self.ny,self.nx))
            

    def ini_var_sw1l(self,config):
        """
        NAME
            ini_var_qg1l
    
        DESCRIPTION
            Initialize QG1L state variables. First one is zonal velocity, 
            second one (optional) is meridional velocity 
            and third one (optional) is SSH
        """
        if len(self.name_var) != 3:
            sys.exit('For SW1L: wrong number variable names')
        for i, var in enumerate(self.name_var):
            if i==0:
                self.var[var] = np.zeros((self.ny,self.nx-1))
            elif i==1:
                self.var[var] = np.zeros((self.ny-1,self.nx))
            else:
                self.var[var] = np.zeros((self.ny,self.nx))


    def save_output(self,date):
        
        filename = os.path.join(self.path_save,self.name_exp_save\
                + '_y' + str(date.year)\
                + 'm' + str(date.month).zfill(2)\
                + 'd' + str(date.day).zfill(2)\
                + 'h' + str(date.hour).zfill(2)\
                + str(date.minute).zfill(2) + '.nc')
        
        coords = {}
        coords['time'] = (('time'), [pd.to_datetime(date)],)
        
        if self.geo_grid:
            coords['lon'] = (('lon',), self.lon[0,:])
            coords['lat'] = (('lat',), self.lat[:,0])
            var = {'ssh':(('lat','lon'),self.getvar(ind=self.get_indsave()))}
        else:
            coords['lon'] = (('y','x',), self.lon)
            coords['lat'] = (('y','x',), self.lat)
            var = {'ssh':(('y','x'),self.getvar(ind=self.get_indsave()))}
        ds = xr.Dataset(var,coords=coords)
        ds.to_netcdf(filename,engine='h5netcdf',
                     encoding={'time': {'units': 'days since 1900-01-01'}})
        
        ds.close()
        del ds
        
        return
            

    def save(self,filename=None):
        """
        NAME
            save
    
        DESCRIPTION
            Save State in a netcdf file
            Args:
                filename (str): path (dir+name) of the netcdf file.
                date (datetime): present date
                """
        
        _namey = {}
        _namex = {}
        outvars = {}
        cy,cx = 1,1
        for i, name in enumerate(self.name_var):
            outvar = self.var.values[i]
            y1,x1 = outvar.shape
            if y1 not in _namey:
                _namey[y1] = 'y'+str(cy)
                cy += 1
            if x1 not in _namex:
                _namex[x1] = 'x'+str(cx)
                cx += 1
                    
            outvars[name] = ((_namey[y1],_namex[x1],), outvar[:,:])
            
        ds = xr.Dataset(outvars)
        ds.to_netcdf(filename,engine='h5netcdf')
        ds.close()
        del ds
        
        return

    def load_output(self,date):
        filename = os.path.join(self.path_save,self.name_exp_save\
            + '_y' + str(date.year)\
            + 'm' + str(date.month).zfill(2)\
            + 'd' + str(date.day).zfill(2)\
            + 'h' + str(date.hour).zfill(2)\
            + str(date.minute).zfill(2) + '.nc')
            
        ds = xr.open_dataset(filename)
        
        ds1 = ds.copy()
        
        ds.close()
        del ds
        
        return ds1
    
    def load(self,filename):

        with xr.open_dataset(filename,engine='h5netcdf') as ds:
            for i, name in enumerate(self.name_var):
                self.var.values[i] = ds[name].values
                
                
    
    def random(self,ampl=1):
        other = self.free()
        for i, name in enumerate(self.name_var):
            other.var.values[i] = ampl * np.random.random(self.var[name].shape)
        return other
    
    def free(self):
        other = State(self.config)
        return other
    
    def copy(self):
        other = State(self.config)
        for i in range(len(self.name_var)):
            other.var.values[i] = deepcopy(self.var.values[i])
        return other
    
    def getvar(self,ind=None,vect=False):
        if ind is not None:
            res = self.var.values[ind]
            if vect:
                res = res.ravel()
        else:
            res = []
            for i in range(len(self.name_var)):
                if vect:
                    res = np.concatenate((res,self.var.values[i].ravel()))
                else:
                    res.append(self.var.values[i])
        return deepcopy(res)
    
    def setvar(self,var,ind=None):
        if ind is None:
            for i in range(len(self.name_var)):
                self.var.values[i] = deepcopy(var[i])
        else:
            self.var.values[ind] = deepcopy(var)
    
    def scalar(self,coeff):
        for i, name in enumerate(self.name_var):
            self.var.values[i] *= coeff
        
    def Sum(self,State1):
        for i, name in enumerate(self.name_var):
            self.var.values[i] += State1.var.values[i]
            
    def plot(self,title=None,cmap='RdBu_r'):
        
        nvar = len(self.name_var)
    
        fig,axs = plt.subplots(1,nvar,figsize=(nvar*7,5),sharey=True)
        
        if title is not None:
            fig.suptitle(title)
            
        if nvar==1:
            axs = [axs]
            
        for i in range(nvar):
            ax = axs[i]
            ax.set_title(self.name_var[i])
            im = ax.pcolormesh(self.lon,self.lat,self.var.values[i],cmap=cmap, shading='auto')
            plt.colorbar(im,ax=ax)
        
        plt.show()
    
    def get_indobs(self) :
        '''
        Return the indice of the observed variable, SSH
        '''
        if self.config['name_model']=='QG1L' :
            return 0
        elif self.config['name_model']=='SW1L' :
            return 2
        else :
            print('model not implemented')
            
    def get_indsave(self) :
        '''
        Return the indice of the variable to save, SSH
        '''
        if self.config['name_model']=='QG1L' :
            return 0
        elif self.config['name_model']=='SW1L' :
            return 2
        else :
            print('model not implemented')


    
    
    

    
