#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:35:02 2021

@author: leguillou
"""
from .config import USE_FLOAT64
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
        self.tmp_DA_path = config.EXP.tmp_DA_path
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
            elif config.GRID.super == 'GRID_CAR_CENTER':
                self.ini_car_grid_center(config.GRID)
            elif config.GRID.super == 'GRID_FROM_FILE':
                self.ini_grid_from_file(config.GRID)
            elif config.GRID.super == 'GRID_RESTART':
                self.ini_grid_restart()
            else:
                sys.exit("Initialization '" + config.GRID.name_grid + "' not implemented yet")
            
            #if self.lon.min()<-180:
            #    self.lon = self.lon%360

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
            if config.GRID.super != 'GRID_FROM_FILE' or config.GRID.var_name_mask is None:
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

        km2deg = 1./111.32

        if not None in [config.ny,config.nx]:
            
            ENSLAT = np.linspace(
                config.lat_min,
                config.lat_max,
                config.ny)

            ENSLON = np.linspace(
                config.lon_min,
                config.lon_max,
                config.nx)
            dx = np.cos(np.min(np.abs(ENSLAT))*np.pi/180.)/km2deg*(ENSLON[1]-ENSLON[0]) # in km
        else:
            ENSLAT = np.arange(
                config.lat_min,
                config.lat_max + config.dx*km2deg,
                config.dx*km2deg)
        
            ENSLON = np.arange(
                        config.lon_min,
                        config.lon_max+config.dx/np.cos(np.min(np.abs(ENSLAT))*np.pi/180.)*km2deg,
                        config.dx/np.cos(np.min(np.abs(ENSLAT))*np.pi/180.)*km2deg)
            dx = config.dx

        lat2d = np.zeros((ENSLAT.size,ENSLON.size))*np.nan
        lon2d = np.zeros((ENSLAT.size,ENSLON.size))*np.nan

        for I in range(len(ENSLAT)):
            for J in range(len(ENSLON)):
                lat2d[I,J] = ENSLAT[I]
                lon2d[I,J] = ENSLON[len(ENSLON)//2] + (J-len(ENSLON)//2)*dx/np.cos(ENSLAT[I]*np.pi/180.) * km2deg
        
        self.lon = lon2d
        self.lat = lat2d

    def ini_car_grid_center(self, config):
        """
        Creates a 2D geographical grid (lon, lat) with constant spacing in km.

        Parameters:
        - center_lon, center_lat: Center of the grid (degrees)
        - spacing_km: Desired spacing between points (km)
        - shape: Tuple (ny, nx), number of points in lat and lon
        """
        
        ny, nx = config.shape  # Grid size

        ny = int(ny)
        nx = int(nx)

        lon_center = config.lon_center
        lat_center = config.lat_center

        spacing_km  = config.spacing_km

        # Compute latitude spacing in degrees (constant)
        lat_spacing_deg = spacing_km / 111.32  

        # Generate latitude points centered at center_lat
        lat_points = lat_center + (np.arange(ny) - ny // 2) * lat_spacing_deg

        # Compute longitude spacing dynamically at each latitude
        lon_grid = np.zeros((ny, nx))
        lat_grid = np.zeros((ny, nx))

        for i, lat in enumerate(lat_points):

            lat_grid[i, :] = lat

            lon_spacing_deg = spacing_km / (111.32 * np.cos(np.radians(lat)))  # Adjust for latitude
            lon_points = lon_center + (np.arange(nx) - nx // 2) * lon_spacing_deg
            
            lon_grid[i, :] = lon_points

        # Correct lon format
        if lon_grid.min()<-180:
            lon_grid = lon_grid%360

        self.lon = lon_grid
        self.lat = lat_grid

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

        if config.init_var:
            for name in dsin:
                self.var[name] = +dsin[name].values
            
        self.lon = lon 
        self.lat = lat 

        # TEST
        f = 4*np.pi/86164*np.sin(self.lat*np.pi/180)
        
        self.present_date = config.init_date

        if config.var_name_mask is not None:
            if len(dsin[config.var_name_mask].shape)==3:
                self.mask = np.isnan(dsin[config.var_name_mask][0])
            elif len(dsin[config.var_name_mask].shape)==2:
                self.mask = np.isnan(dsin[config.var_name_mask])

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
            print('No mask provided')
            self.mask = (np.isnan(self.lon) + np.isnan(self.lat)).astype(bool)
            return

        # Convert longitudes
        if np.sign(ds[name_lon].data.min())==-1 and self.lon_unit=='0_360':
            ds = ds.assign_coords({name_lon:((name_lon, ds[name_lon].data % 360))})
        elif (np.sign(ds[name_lon].data.min())>=0 or ds[name_lon].data.max()>180) and self.lon_unit=='-180_180':
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
        if self.lon_unit=='-180_180':
            is_circle = True
        else:
            is_circle = False
        x_source_axis = pyinterp.Axis(lon, is_circle=is_circle)
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
            ind_mask = (np.isnan(mask_interp)) 
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

        if name_var is None:
            name_var = self.var.keys()
         
        var = {}              
        for name in name_var:

            var_to_save = +np.array(self.var[name])

            # Apply Mask
            try:
                if self.mask is not None:
                    var_to_save[self.mask] = np.nan
            except:
                var_to_save = var_to_save
        
            if len(var_to_save.shape)==2:
                var_to_save = var_to_save[np.newaxis,:,:]
            
            _dims = ['time','y','x']
            if var_to_save.shape[1]!=self.lon.shape[0]:     
                _dims[1] += name
            if var_to_save.shape[2]!=self.lon.shape[1]:
                _dims[2] += name      
            var[name] = (_dims, var_to_save)

        if os.path.exists(filename):
            ds = xr.open_dataset(filename)
            dsout = ds.copy().load()
            ds.close()
            del ds 
            for name in var.keys():
                dsout[name] = (var[name][0], var[name][1])
            dsout.to_netcdf(filename,
                         unlimited_dims={'time':True})
            
        else:
            ds = xr.Dataset(var, coords=coords)
            ds.to_netcdf(filename,
                        unlimited_dims={'time':True})
            ds.close()
            del ds
        
        return 

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

        if name_var is None:
            name_var = self.var.keys()
         
        var = {}              
        for name in name_var:

            var_to_save = +np.array(self.var[name])

            # Apply Mask
            try:
                if self.mask is not None:
                    var_to_save[self.mask] = np.nan
            except:
                var_to_save = var_to_save
        
            if len(var_to_save.shape)==2:
                var_to_save = var_to_save[np.newaxis,:,:]
            
            if self.geo_grid:
                _dims = ['time','lat','lon']
            else:
                _dims = ['time','y','x']
            if var_to_save.shape[1]!=self.lon.shape[0]:     
                _dims[1] += name
            if var_to_save.shape[2]!=self.lon.shape[1]:
                _dims[2] += name      
            var[name] = (_dims, var_to_save)

        if os.path.exists(filename):
            ds = xr.open_dataset(filename)
            dsout = ds.copy().load()
            ds.close()
            del ds 
            for name in var.keys():
                dsout[name] = (var[name][0], var[name][1])
            dsout.to_netcdf(filename,
                         unlimited_dims={'time':True})
            
        else:
            ds = xr.Dataset(var, coords=coords)
            ds.to_netcdf(filename,
                        unlimited_dims={'time':True})
            ds.close()
            del ds
        
        return 

    def load_output(self, date, name_var=None):
        

        filename = os.path.join(
            self.path_save,
            f'{self.name_exp_save}_y{date.year}m{str(date.month).zfill(2)}d{str(date.day).zfill(2)}'
            f'h{str(date.hour).zfill(2)}m{str(date.minute).zfill(2)}.nc'
        )

        with xr.open_dataset(filename) as ds:  # Ensure thread safety
            ds1 = ds.load().copy().squeeze()  # Fully load data before closing
        
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
        np.random.seed(0)
        other = self.copy(free=True) 
        for name in self.var.keys():
            other.var[name] = ampl * np.random.random(self.var[name].shape)
            if self.mask is not None:
                try:
                    other.var[name][self.mask] = 0.
                except:
                    pass
        for name in self.params.keys():
            other.params[name] = ampl * np.random.random(self.params[name].shape)
            if self.mask is not None:
                try:
                    other.params[name][self.mask] = 0.
                except:
                    pass
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
                other.var[name] = self.var[name]*0
            else:
                other.var[name] = deepcopy(self.var[name])
        
        # (deep)Copy model parameters
        for name in self.params.keys():
            if free:
                other.params[name] = self.params[name]*0
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
            
    def plot(self, title=None, cmap='RdBu_r', ind=None, name_save=None, params=False):
        
        if self.flag_plot<1:
            return
        
        if ind is not None:
            indvar = ind
        else:
            if not params:
                # Only plot 2D variables (skip per-layer 3D arrays)
                plot_keys = [k for k in self.var if np.ndim(self.var[k]) == 2]
                indvar = np.arange(0, len(plot_keys))
            else:
                plot_keys = list(self.params.keys())
                indvar = np.arange(0,len(self.params.keys()))
        nvar = len(indvar)
 
        fig,axs = plt.subplots(1,nvar,figsize=(nvar*7,5))
        
        if title is not None:
            fig.suptitle(title)
            
        if nvar==1:
            axs = [axs]
        
        if not params:
            for ax,name_var in zip(axs,plot_keys):
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
            for ax,name_var in zip(axs,plot_keys):
                ax.set_title(name_var)

                try:
                    if np.sign(np.nanmin(self.params[name_var]))!=np.sign(np.nanmax(self.params[name_var])):
                        cmap_range = np.nanmax(np.absolute(self.params[name_var]))
                        im = ax.pcolormesh(self.params[name_var],cmap=cmap,\
                                        shading='auto', vmin = -cmap_range, vmax = cmap_range)
                    else:
                        im = ax.pcolormesh(self.params[name_var],shading='auto')
                    plt.colorbar(im,ax=ax)

                except:
                    try:
                        plt.plot(self.params[name_var])
                    except:
                        print("Can't plot parameters")
        if name_save is not None:
            plt.savefig(f'{self.tmp_DA_path}/{name_save}.png', bbox_inches='tight')
        plt.show()
        




    
    
    

    
