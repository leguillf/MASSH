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
import matplotlib.pyplot as plt
from scipy import interpolate
import glob
from datetime import datetime
import pyinterp 
import pyinterp.fill
import pickle 


from . import grid

class State:
    """
    NAME
       State
    DESCRIPTION
        Main class handling the grid initialization, the storage of model variables and the saving of outputs 
    """

    
    def __init__(self,config,first=True, verbose=True):

        if first and verbose:
            print(config.GRID)
        
        self.config = config
        
        # Parameters
        self.name_time = config.EXP.name_time
        self.name_lon = config.EXP.name_lon
        self.name_lat = config.EXP.name_lat
        self.name_exp_save = config.EXP.name_exp_save
        self.path_save = config.EXP.path_save
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
        self.flag_plot = config.EXP.flag_plot

        #  Initialize state variables dictonary
        self.var = {}

        # Initialize controle parameters dictonary
        self.params = {}

        # Initialize grid
        if first:
            self.geo_grid = False
            self.mask = None
            if config.GRID.super == 'GRID_GEO':
                self.ini_geo_grid(config.GRID)
            elif config.GRID.super == 'GRID_CAR':
                self.ini_car_grid(config.GRID)
            elif config.GRID.super == 'GRID_FROM_FILE':
                self.ini_grid_from_file(config.GRID)
            elif config.GRID.super == 'GRID_RESTART':
                self.ini_grid_restart()
            else:
                sys.exit("Initialization '" + config.GRID.name_grid + "' not implemented yet")

            

            self.ny,self.nx = self.lon.shape

            self.lon_min = np.nanmin(self.lon)
            self.lon_max = np.nanmax(self.lon)
            self.lat_min = np.nanmin(self.lat)
            self.lat_max = np.nanmax(self.lat)

            if np.sign(self.lon_min)==-1:
                self.lon_unit = '-180_180'
            else:
                self.lon_unit = '0_360'

            # Mask
            self.ini_mask(config.GRID)

            # Compute cartesian grid 
            DX,DY = grid.lonlat2dxdy(self.lon,self.lat)
            dx = np.nanmean(DX)
            dy = np.nanmean(DY)
            DX[np.isnan(DX)] = dx # For cartesian grid
            DY[np.isnan(DY)] = dy # For cartesian grid
            X,Y = grid.dxdy2xy(DX,DY)
            self.DX = DX
            self.DY = DY
            self.X = X
            self.Y = Y
            self.dx = dx
            self.dy = dy

            # Coriolis
            self.f = 4*np.pi/86164*np.sin(self.lat*np.pi/180)
            
            # Gravity
            self.g = 9.81


    def ini_geo_grid(self,config):
        """
        NAME
            ini_geo_grid
    
        DESCRIPTION
            Create state grid, regular in (lon,lat) 
            Args:
                config (module): configuration module
        """
        self.geo_grid = True
        lon = np.arange(config.lon_min, config.lon_max + config.dlon, config.dlon) 
        lat = np.arange(config.lat_min, config.lat_max + config.dlat, config.dlat) 
        lon,lat = np.meshgrid(lon,lat)
        self.lon = lon
        self.lat = lat
        self.present_date = config.init_date
    
    def ini_car_grid(self,config):
        """
        NAME
            ini_car_grid
    
        DESCRIPTION
            Create state grid, regular in (x,y) 
            Args:
                config (module): configuration module
        """

        km2deg = 1./110

        ENSLAT = np.arange(
            config.lat_min,
            config.lat_max + config.dx*km2deg,
            config.dx*km2deg)

        ENSLON = np.arange(
                    config.lon_min,
                    config.lon_max+config.dx/np.cos(np.min(np.abs(ENSLAT))*np.pi/180.)*km2deg,
                    config.dx/np.cos(np.min(np.abs(ENSLAT))*np.pi/180.)*km2deg)

        lat2d = np.zeros((ENSLAT.size,ENSLON.size))*np.nan
        lon2d = np.zeros((ENSLAT.size,ENSLON.size))*np.nan

        for I in range(len(ENSLAT)):
            for J in range(len(ENSLON)):
                lat2d[I,J] = ENSLAT[I]
                lon2d[I,J] = ENSLON[len(ENSLON)//2] + (J-len(ENSLON)//2)*config.dx/np.cos(ENSLAT[I]*np.pi/180.)*km2deg
        
        self.lon = lon2d
        self.lat = lat2d

    def ini_grid_from_file(self,config):
        """
        NAME
            ini_from_file
    
        DESCRIPTION
            Copy state grid from existing file 
            Args:
                config (module): configuration module
        """
        
        dsin = xr.open_dataset(config.path_init_grid)

        lon = dsin[config.name_init_lon].values
        lat = dsin[config.name_init_lat].values

        if len(lon.shape)==1:
            self.geo_grid = True
            lon,lat = np.meshgrid(lon,lat)

        if config.subsampling is not None:
            lon = lon[::config.subsampling,::config.subsampling]
            lat = lat[::config.subsampling,::config.subsampling]
            
        self.lon = lon 
        self.lat = lat
        self.present_date = config.init_date
        dsin.close()
        del dsin
        
    def ini_grid_restart(self):
        # Look for last output
        files = sorted(glob.glob(os.path.join(self.path_save,self.name_exp_save+'*.nc')))
        if len(files)==0:
            sys.exit('Error: you set *name_init="restart"*, but no output files are available')
        else:
            # last output
            file = files[-1]
            # Open dataset
            dsin = xr.open_dataset(file).squeeze()
            # Read grid
            lon = dsin[self.name_lon].values
            lat = dsin[self.name_lat].values
            if len(lon.shape)==1:
                self.geo_grid = True
                lon,lat = np.meshgrid(lon,lat)
            self.lon = lon 
            self.lat = lat
            self.present_date = datetime.utcfromtimestamp(dsin['time'].values.tolist()/1e9)
            if self.first:
                print('Restarting experiment at',self.present_date)
            dsin.close()
            del dsin
            
                
    
    def ini_mask(self,config):
        
        """
        NAME
            ini_mask
    
        DESCRIPTION
            Read mask file, interpolate it to state grid, 
            and apply to state variable
        """

        # Read mask
        if config.name_init_mask is not None and os.path.exists(config.name_init_mask):
            ds = xr.open_dataset(config.name_init_mask).squeeze()
            name_lon = config.name_var_mask['lon']
            name_lat = config.name_var_mask['lat']
            name_var = config.name_var_mask['var']
        else:
            self.mask = (np.isnan(self.lon) + np.isnan(self.lat)).astype(bool)
            return

        # Convert longitudes
        if np.sign(ds[name_lon].data.min())==-1 and self.lon_unit=='0_360':
            ds = ds.assign_coords({name_lon:((name_lon, ds[name_lon].data % 360))})
        elif np.sign(ds[name_lon].data.min())>=0 and self.lon_unit=='-180_180':
            ds = ds.assign_coords({name_lon:((name_lon, (ds[name_lon].data + 180) % 360 - 180))})
        ds = ds.sortby(ds[name_lon])    

        dlon =  np.nanmax(self.lon[:,1:] - self.lon[:,:-1])
        dlat =  np.nanmax(self.lat[1:,:] - self.lat[:-1,:])
        dlon +=  np.nanmax(ds[name_lon].data[1:] - ds[name_lon].data[:-1])
        dlat +=  np.nanmax(ds[name_lat].data[1:] - ds[name_lat].data[:-1])
       
        ds = ds.sel(
            {name_lon:slice(self.lon_min-dlon,self.lon_max+dlon),
             name_lat:slice(self.lat_min-dlat,self.lat_max+dlat)})

        lon = ds[name_lon].values
        lat = ds[name_lat].values
        var = ds[name_var]
                
        if len(var.shape)==2:
            mask = var
        elif len(var.shape)==3:
            mask = var[0,:,:]
        
        # Interpolate to state grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        x_target = self.lon.T
        y_target = self.lat.T
        grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, mask.T)
        mask_interp = pyinterp.bivariate(grid_source,
                                        x_target.flatten(),
                                        y_target.flatten(),
                                        bounds_error=False).reshape(x_target.shape).T
                                        
        # Convert to bool if float type     
        if mask_interp.dtype!=bool : 
            self.mask = np.empty((self.ny,self.nx),dtype='bool')
            ind_mask = (np.isnan(mask_interp)) | (mask_interp==1) | (np.abs(mask_interp)>10)
            self.mask[ind_mask] = True
            self.mask[~ind_mask] = False
        else:
            self.mask = mask_interp.copy()
        
        self.mask += (np.isnan(self.lon) + np.isnan(self.lat)).astype(bool)
            
    def save_output(self,date,name_var=None):
        
        filename = os.path.join(self.path_save,f'{self.name_exp_save}'\
            f'_y{date.year}'\
            f'm{str(date.month).zfill(2)}'\
            f'd{str(date.day).zfill(2)}'\
            f'h{str(date.hour).zfill(2)}'\
            f'm{str(date.minute).zfill(2)}.nc')
        
        coords = {}
        coords[self.name_time] = ((self.name_time), [pd.to_datetime(date)],)

        if self.geo_grid:
                coords[self.name_lon] = ((self.name_lon,), self.lon[0,:])
                coords[self.name_lat] = ((self.name_lat,), self.lat[:,0])
                dims = (self.name_time,self.name_lat,self.name_lon)
        else:
            coords[self.name_lon] = (('y','x',), self.lon)
            coords[self.name_lat] = (('y','x',), self.lat)
            dims = ('time','y','x')

        if name_var is None:
            name_var = self.var.keys()
         
        var = {}              
        for name in name_var:

            var_to_save = +self.var[name]

            # Apply Mask
            if self.mask is not None:
                var_to_save[self.mask] = np.nan
        
            if len(var_to_save.shape)==2:
                var_to_save = var_to_save[np.newaxis,:,:]
            
            var[name] = (dims, var_to_save)
            
        ds = xr.Dataset(var, coords=coords)
        ds.to_netcdf(filename,
                     encoding={'time': {'units': 'days since 1900-01-01'}},
                     unlimited_dims={'time':True})
        
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
        
        
        # Variables
        _namey = {}
        _namex = {}
        outvars = {}
        cy,cx = 1,1
        for name,var in self.var.items():
            y1,x1 = var.shape
            if y1 not in _namey:
                _namey[y1] = 'y'+str(cy)
                cy += 1
            if x1 not in _namex:
                _namex[x1] = 'x'+str(cx)
                cx += 1
            outvars[name] = ((_namey[y1],_namex[x1],), var[:,:])
        ds = xr.Dataset(outvars)
        ds.to_netcdf(filename,group='var')
        ds.close()
        
        # Parameters
        _namey = {}
        _namex = {}
        _namez = {}
        outparams = {}
        cy,cx,cz = 1,1,1
        for name,var in self.params.items():
            if len(var.shape)==2:
                y1,x1 = var.shape
                if y1 not in _namey:
                    _namey[y1] = 'y'+str(cy)
                    cy += 1
                if x1 not in _namex:
                    _namex[x1] = 'x'+str(cx)
                    cx += 1
                outparams[name] = ((_namey[y1],_namex[x1],), var[:,:])
            else:
                z1 = var.size
                if z1 not in _namez:
                    _namez[z1] = 'z'+str(cz)
                    cz += 1
                outparams[name] = ((_namez[z1],), var.flatten())

        ds = xr.Dataset(outparams)
        ds.to_netcdf(filename,group='params',mode='a')
        ds.close()
        
        return

    def load_output(self,date,name_var=None):
        filename = os.path.join(self.path_save,f'{self.name_exp_save}'\
            f'_y{date.year}'\
            f'm{str(date.month).zfill(2)}'\
            f'd{str(date.day).zfill(2)}'\
            f'h{str(date.hour).zfill(2)}'\
            f'm{str(date.minute).zfill(2)}.nc')
            
        ds = xr.open_dataset(filename)
        
        ds1 = ds.copy().squeeze()
        
        ds.close()
        del ds
        
        if name_var is None:
            return ds1
        
        else:
            return np.array([ds1[name].values for name in name_var])
    
    def load(self,filename):

        with xr.open_dataset(filename,group='var') as ds:
            for name in self.var.keys():
                self.var[name] = ds[name].values
        
        with xr.open_dataset(filename,group='params') as ds:
            for name in self.params.keys():
                self.params[name] = ds[name].values
            
    
    def random(self,ampl=1):
        other = self.copy(free=True)
        for name in self.var.keys():
            other.var[name] = ampl * np.random.random(self.var[name].shape).astype('float64')
            other.var[name][self.mask] = np.nan
        for name in self.params.keys():
            other.params[name] = ampl * np.random.random(self.params[name].shape).astype('float64')
            other.params[name][self.mask] = np.nan
        return other
    
    
    def copy(self, free=False):

        # Create new instance
        other = State(self.config,first=False)

        # Copy all attributes
        other.ny = self.ny
        other.nx = self.nx
        other.DX = self.DX
        other.DY = self.DY
        other.X = self.X
        other.Y = self.Y
        other.dx = self.dx
        other.dy = self.dy
        other.f = self.f
        other.mask = self.mask
        other.lon = self.lon
        other.lat = self.lat
        other.geo_grid = self.geo_grid

        # (deep)Copy model variables
        for name in self.var.keys():
            if free:
                other.var[name] = np.zeros_like(self.var[name])
            else:
                other.var[name] = deepcopy(self.var[name])
        
        # (deep)Copy model parameters
        for name in self.params.keys():
            if free:
                other.params[name] = np.zeros_like(self.params[name])
            else:
                other.params[name] = deepcopy(self.params[name])

        return other
    
    def getvar(self,name_var=None,vect=False):
        if name_var is not None:
            if type(name_var) in (list,np.ndarray):
                var_to_return = []
                for name in name_var:
                    if vect:
                        var_to_return = np.concatenate((var_to_return,self.var[name].ravel()))
                    else:
                        var_to_return.append(self.var[name])
                    
            else:
                var_to_return = self.var[name_var]
                if vect:
                    var_to_return = var_to_return.ravel()
        else:
            var_to_return = []
            for name in self.var.keys():
                if vect:
                    var_to_return = np.concatenate((var_to_return,self.var[name].ravel()))
                else:
                    var_to_return.append(self.var[name])

        return deepcopy(np.asarray(var_to_return))

    def getparams(self,name_params=None,vect=False):
        if name_params is not None:
            if type(name_params) in (list,np.ndarray):
                params_to_return = []
                for name in name_params:
                    if vect:
                        params_to_return = np.concatenate((params_to_return,self.params[name].ravel()))
                    else:
                        params_to_return.append(self.params[name])
                    
            else:
                params_to_return = self.params[name_params]
                if vect:
                    params_to_return = params_to_return.ravel()
        else:
            params_to_return = []
            for name in self.params:
                if vect:
                    params_to_return = np.concatenate((params_to_return,self.params[name].ravel()))
                else:
                    params_to_return.append(self.params[name])

        return deepcopy(np.asarray(params_to_return))

    def setvar(self,var,name_var=None,add=False):

        if name_var is None:
            for i,name in enumerate(self.var):
                if add:
                    self.var[name] += var[i]
                else:
                    self.var[name] = deepcopy(var[i])
        else:
            if type(name_var) in (list,np.ndarray):
                for i,name in enumerate(name_var):
                    if add:
                        self.var[name] += var[i]
                    else:
                        self.var[name] = deepcopy(var[i])
            else:
                if add:
                    self.var[name_var] += var
                else:
                    self.var[name_var] = deepcopy(var)
    
    def scalar(self,coeff,copy=False):
        if copy:
            State1 = self.copy()
            for name in self.var.keys():
                State1.var[name] *= coeff
            for name in self.params.keys():
                State1.params[name] *= coeff
            return State1
        else:
            for name in self.var.keys():
                self.var[name] *= coeff
            for name in self.params.keys():
                self.params[name] *= coeff
        
    def Sum(self,State1,copy=False):
        if copy:
            State2 = self.copy()
            for name in self.var.keys():
                State2.var[name] += State1.var[name]
            for name in self.params.keys():
                State2.params[name] += State1.params[name]
            return State2
        else:
            for name in self.var.keys():
                self.var[name] += State1.var[name]
            for name in self.params.keys():
                self.params[name] += State1.params[name]
            
    def plot(self,title=None,cmap='RdBu_r',ind=None,params=False):
        
        if self.flag_plot<1:
            return
        
        if ind is not None:
            indvar = ind
        else:
            if not params:
                indvar = np.arange(0,len(self.var.keys()))
            else:
                indvar = np.arange(0,len(self.params.keys()))
        nvar = len(indvar)
 
        fig,axs = plt.subplots(1,nvar,figsize=(nvar*7,5))
        
        if title is not None:
            fig.suptitle(title)
            
        if nvar==1:
            axs = [axs]
        
        if not params:
            for ax,name_var in zip(axs,self.var):
                ax.set_title(name_var)
                _min = np.nanmin(self.var[name_var])
                _max = np.nanmax(self.var[name_var])
                _max_abs = np.nanmax(np.absolute(self.var[name_var]))
                if np.sign(_min)!=np.sign(_max) and ((_max-np.abs(_min))<.5*_max_abs):
                    im = ax.pcolormesh(self.var[name_var],cmap=cmap,\
                                    shading='auto', vmin = -_max_abs, vmax = _max_abs)
                else:
                    im = ax.pcolormesh(self.var[name_var], shading='auto')
                plt.colorbar(im,ax=ax)
        else:
            for ax,name_var in zip(axs,self.params):
                ax.set_title(name_var)
                if np.sign(np.nanmin(self.params[name_var]))!=np.sign(np.nanmax(self.params[name_var])):
                    cmap_range = np.nanmax(np.absolute(self.params[name_var]))
                    im = ax.pcolormesh(self.params[name_var],cmap=cmap,\
                                    shading='auto', vmin = -cmap_range, vmax = cmap_range)
                else:
                    im = ax.pcolormesh(self.params[name_var],shading='auto')
                plt.colorbar(im,ax=ax)
        
        plt.show()
        




    
    
    

    
