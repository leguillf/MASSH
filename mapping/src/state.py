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
        # Initialize grid
        if config.name_init == 'geo_grid':
             self.ini_geo_grid(config)
        elif config.name_init == 'from_file':
             self.ini_from_file(config)
        else:
            sys.exit("Initialization '" + config.name_init + "' not implemented yet")
        self.ny,self.nx = self.lon.shape
        #  Initialize state variables
        self.var = pd.Series(dtype=np.float64)
        if config.name_model=='QG1L':
            self.ini_var_qg1l()
        elif config.name_model=='SW1L':
            self.ini_var_sw1l()
        else:
            sys.exit("Model '" + config.name_model + "' not implemented yet")
        
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
        lon = np.arange(config.lon_min, config.lon_max + config.dx, config.dx) % 360
        lat = np.arange(config.lat_min, config.lat_max + config.dy, config.dy) 
        lon,lat = np.meshgrid(lon,lat)
        self.lon = lon
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
            lon,lat = np.meshgrid(lon,lat)
        self.lon = lon
        self.lat = lat
            
    def ini_var_qg1l(self):
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
            self.var[var] = np.zeros((self.ny,self.nx))

    def ini_var_sw1l(self):
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


    def save(self,filename=None,date=None):
        """
        NAME
            save
    
        DESCRIPTION
            Save the grid and variables in a netcdf file
            Args:
                filename (str): path (dir+name) of the netcdf file.
                date (datetime): present date
                """
        
        if filename is None:
            filename = self.path_save + '/' + self.name_exp_save\
                + '_y' + str(date.year)\
                + 'm' + str(date.month).zfill(2)\
                + 'd' + str(date.day).zfill(2)\
                + 'h' + str(date.hour).zfill(2)\
                + str(date.minute).zfill(2) + '.nc'
                
        dictout = {self.name_lon: (('y','x',), self.lon),
                   self.name_lat: (('y','x',), self.lat),
               }
    
        if date is not None:
            dictout['time'] = (('t'), [pd.to_datetime(date)])
            
        for i, name in enumerate(self.name_var):
            dictout[name] = (('y'+str(i), 'x'+str(i),), self.var.values[i])
            
        ds = xr.Dataset(dictout)
        ds.to_netcdf(filename)
        ds.close()
    
    def load(self,filename):
        ds = xr.open_dataset(filename)
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


    
    
    

    