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
    
    def __init__(self,config,first=True):
        
        self.config = config
        self.first = first
        
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
             self.ini_grid_from_file(config)
        elif config.name_init == 'restart':
             self.ini_grid_restart()
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
        if config.name_model is None or config.name_model in ['Diffusion','QG1L','JAX-QG1L']:
            self.ini_var_qg1l(config)
        elif config.name_model=='SW1L':
            self.ini_var_sw1l(config)
        elif config.name_model=='SW1LM':
            self.ini_var_sw1lm(config)
        elif hasattr(config.name_model,'__len__') and len(config.name_model)==2 :
            self.ini_var_bm_it(config)
        else:
            sys.exit("Model '" + config.name_model + "' not implemented yet")
        # Read output variable from previous run 
        if config.name_init == 'restart':
            self.ini_var_restart()
        # Add mask if provided
        if self.first:
            try: self.ini_mask(config)
            except: 
                print('Warning: unable to compute mask')
                self.mask = np.zeros((self.ny,self.nx),dtype='bool')
        else:
            self.mask = np.zeros((self.ny,self.nx),dtype='bool')   
        if not os.path.exists(config.tmp_DA_path):
            os.makedirs(config.tmp_DA_path)
            
        
        self.mdt = None
        self.depth = None
        if first:
            # MDT
            if config.path_mdt is not None and os.path.exists(config.path_mdt):
            
                ds = xr.open_dataset(config.path_mdt).squeeze()
                
                name_var_mdt = {}
                name_var_mdt['lon'] = config.name_var_mdt['lon']
                name_var_mdt['lat'] = config.name_var_mdt['lat']
                
                
                
                if 'mdt' in config.name_var_mdt and config.name_var_mdt['mdt'] in ds:
                    name_var_mdt['var'] = config.name_var_mdt['mdt']
                    self.mdt = grid.interp2d(ds,
                                             name_var_mdt,
                                             self.lon,
                                             self.lat)
            # MDT
            if config.file_depth is not None and os.path.exists(config.file_depth):
                ds = xr.open_dataset(config.file_depth).squeeze()
            
                self.depth = grid.interp2d(ds,
                                         config.name_var_depth,
                                         self.lon,
                                         self.lat)
            
        
        # Model parameters
        self.params = None
        
        
        
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
        self.present_date = config.init_date
    
    def ini_grid_from_file(self,config):
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
            self.lon = lon % 360
            self.lat = lat
            self.present_date = datetime.utcfromtimestamp(dsin['time'].values.tolist()/1e9)
            if self.first:
                print('Restarting experiment at',self.present_date)
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
            ini_var_sw1l
    
        DESCRIPTION
            Initialize SW1LM state variables. First one is zonal velocity, 
            second one is meridional velocity 
            and third one is SSH
        """
        
        if len(self.name_var) != 3:
            if self.first:
                print('Warning: For SW1L: wrong number variable names')
            self.name_var = ['u','v','h']
        if self.first:
            print(self.name_var)
            
        for i, var in enumerate(self.name_var):
            if i==0:
                self.var[var] = np.zeros((self.ny,self.nx-1))
            elif i==1:
                self.var[var] = np.zeros((self.ny-1,self.nx))
            else:
                self.var[var] = np.zeros((self.ny,self.nx))
                
    def ini_var_sw1lm(self,config):
        """
        NAME
            ini_var_sw1lm
    
        DESCRIPTION
            Initialize SW1LM state variables. As many (u,v,h) as the 
            number of modes
        """
        
        if len(self.name_var) != 3*(config.Nmodes+1):
            if self.first:
                print('Warning: For SW1LM: wrong number variable names')
            self.name_var = []
            for i in range(config.Nmodes):
                self.name_var.append('u'+str(i+1))
                self.name_var.append('v'+str(i+1))
                self.name_var.append('h'+str(i+1))
            self.name_var.append('u')
            self.name_var.append('v')
            self.name_var.append('h')
            if self.first:
                print(self.name_var)
            
        for i, var in enumerate(self.name_var):
            if i%3==0:
                self.var[var] = np.zeros((self.ny,self.nx-1))
            elif i%3==1:
                self.var[var] = np.zeros((self.ny-1,self.nx))
            else:
                self.var[var] = np.zeros((self.ny,self.nx))
        
    def ini_var_bm_it(self,config):
        if len(config.name_mod_var) != 1 + config.Nmodes*3 + 1:
            if self.first:
                print('Warning: For BM & IT: wrong number variable names')
            self.name_var = ["h_bm"]
            if config.Nmodes==1:
                self.name_var += ["u_it","v_it","h_it","h"]
            else:
                for i in range(1,config.Nmodes+1):
                    self.name_var += [f"u_it_{i}",f"v_it_{i}",f"h_it_{i}"]
                self.name_var += ["u_it","v_it","h_it","h"]   
        
            if self.first:
                print(self.name_var)
                
        for i, var in enumerate(self.name_var):
            if i%3==0 or i==len(self.name_var)-1:
                # SSH
                self.var[var] = np.zeros((self.ny,self.nx))
            elif i%3==1:
                # U
                self.var[var] = np.zeros((self.ny,self.nx-1))
            elif i%3==2:
                # V
                self.var[var] = np.zeros((self.ny-1,self.nx))
        
            
        
    def ini_var_restart(self):
        ds = self.load_output(self.present_date)
        name = self.name_var[self.get_indsave()]
        self.var[name] = ds[name].values
        ds.close()
        del ds
        
            
    
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
        elif config.path_mdt is not None and os.path.exists(config.path_mdt):
            ds = xr.open_dataset(config.path_mdt).squeeze()
            name_lon = config.name_var_mdt['lon']
            name_lat = config.name_var_mdt['lat']
            name_var = config.name_var_mdt['mdt']
        else:
            self.mask = np.zeros((self.ny,self.nx),dtype='bool')
            return
        
        dlon =  (self.lon[:,1:] - self.lon[:,:-1]).max()
        dlat =  (self.lat[1:,:] - self.lat[:-1,:]).max()
       
        ds = ds.sel(
            {name_lon:slice(self.lon.min()-dlon,self.lon.max()+dlon),
             name_lat:slice(self.lat.min()-dlat,self.lat.max()+dlat)})
        
        lon = ds[name_lon].values
        lat = ds[name_lat].values
        var = ds[name_var].values
        lon = lon % 360
        
        if len(lon.shape)==1:
            lon_mask,lat_mask = np.meshgrid(lon,lat)
        else:
            lon_mask = +lon
            lat_mask = +lat
                
        if len(var.shape)==2:
            mask = +var
        elif len(var.shape)==3:
            mask = +var[0,:,:]
        
        # Interpolate to state grid
        if np.any(lon_mask!=self.lon) or np.any(lat_mask!=self.lat):
            mask_interp = interpolate.griddata(
                (lon_mask.ravel(),lat_mask.ravel()), mask.ravel(),
                (self.lon.ravel(),self.lat.ravel())).reshape((self.ny,self.nx))
        else:
            mask_interp = mask.copy()
        
        # Convert to bool if float type     
        if mask_interp.dtype!=np.bool : 
            self.mask = np.empty((self.ny,self.nx),dtype='bool')
            ind_mask = (np.isnan(mask_interp)) | (mask_interp==1) | (np.abs(mask_interp)>10)
            self.mask[ind_mask] = True
            self.mask[~ind_mask] = False
        else:
            self.mask = mask_interp.copy()
                            
        # Apply to state variable (SSH only)
        if config.name_model=='QG1L' or (hasattr(config.name_model,'__len__') and len(config.name_model)==2):
            self.var[0][self.mask] = np.nan
            

    def save_output(self,date,mdt=None):
        
        name_lon = self.name_lon 
        name_lat = self.name_lat
        
        
        filename = os.path.join(self.path_save,self.name_exp_save\
                + '_y' + str(date.year)\
                + 'm' + str(date.month).zfill(2)\
                + 'd' + str(date.day).zfill(2)\
                + 'h' + str(date.hour).zfill(2)\
                + str(date.minute).zfill(2) + '.nc')
        
        coords = {}
        coords['time'] = (('time'), [pd.to_datetime(date)],)
        
        indsave = self.get_indsave()
        if hasattr(indsave,'__len__'):
            names_var = [self.name_var[i] for i in indsave]
            vars_to_save = self.getvar(ind=indsave)
        else:
            names_var = [self.name_var[indsave]]
            vars_to_save = [self.getvar(ind=indsave)]
         
        var = {}              
        for i,(name_var,var_to_save) in enumerate(zip(names_var,vars_to_save)):
            # Apply Mask
            if self.mask is not None:
                var_to_save[self.mask] = np.nan
                
            if len(var_to_save.shape)==2:
                var_to_save = var_to_save[np.newaxis,:,:]
                
            if self.geo_grid:
                coords[name_lon] = ((name_lon,), self.lon[0,:])
                coords[name_lat] = ((name_lat,), self.lat[:,0])
                dims = ('time','lat','lon')
                
            else:
                coords[name_lon] = (('y','x',), self.lon)
                coords[name_lat] = (('y','x',), self.lat)
                dims = ('time','y','x')
            
            var[name_var] = (dims,var_to_save)
        
        
        if mdt is not None:
            var['mdt'] = (dims[1:],mdt)
            
        ds = xr.Dataset(var,coords=coords)
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
            
        if self.params is not None:
            outvars['params'] = (('p',self.params.flatten()))
            
        ds = xr.Dataset(outvars)
        ds.to_netcdf(filename)
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
        
        return ds1.squeeze()
    
    def load(self,filename):

        with xr.open_dataset(filename) as ds:
            for i, name in enumerate(self.name_var):
                self.var.values[i] = ds[name].values
            if 'params' in ds:
                self.params = ds.params.values
    
    def random(self,ampl=1):
        other = self.free()
        for i, name in enumerate(self.name_var):
            other.var.values[i] = ampl * np.random.random(self.var[name].shape)
        return other
    
    def free(self):
        other = State(self.config,first=False)
        other.mask = self.mask
        other.params = self.params
        other.mdt = self.mdt
        other.depth = self.depth
        
        return other
    
    def copy(self):
        other = State(self.config,first=False)
        for i in range(len(self.name_var)):
            other.var.values[i] = deepcopy(self.var.values[i])
        other.mask = self.mask
        other.mdt = self.mdt
        other.depth = self.depth
        if self.params is not None:
            other.params = +self.params
        
        return other
    
    def getvar(self,ind=None,vect=False):
        if ind is not None:
            if hasattr(ind,'__len__'):
                res = []
                for i in ind:
                    res.append(self.var.values[i])
            else:
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
            if hasattr(ind,'__len__'):
                for i,_ind in enumerate(ind):
                    self.var.values[_ind] = deepcopy(var[i])
            else:
                self.var.values[ind] = deepcopy(var)
    
    def scalar(self,coeff):
        for i, name in enumerate(self.name_var):
            self.var.values[i] *= coeff
        if self.params is not None:
            self.params *= coeff
        
    def Sum(self,State1):
        for i, name in enumerate(self.name_var):
            self.var.values[i] += State1.var.values[i]
        if self.params is not None and State1.params is not None:
            self.params += State1.params
            
    def plot(self,title=None,cmap='RdBu_r',ind=None):
        
        if self.config.flag_plot==0:
            return
        
        if ind is not None:
            indvar = ind
        else:
            indvar = np.arange(0,len(self.name_var))
        nvar = len(indvar)
 
        fig,axs = plt.subplots(1,nvar,figsize=(nvar*7,5))
        
        if title is not None:
            fig.suptitle(title)
            
        if nvar==1:
            axs = [axs]
            
        for ax,i in zip(axs,indvar):
            ax.set_title(self.name_var[i])
            im = ax.pcolormesh(self.var.values[i],cmap=cmap,\
                               shading='auto')
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
        elif self.config['name_model']=='SW1LM' :
            return 2 + (self.config.Nmodes)*3
        elif hasattr(self.config['name_model'],'__len__') and len(self.config['name_model'])==2 :
            return -1
        else :
            return 0
            
    def get_indsave(self) :
        '''
        Return the indice of the variable to save, SSH
        '''
        if self.config['name_model'] is None or self.config['name_model']=='QG1L' :
            return 0
        elif self.config['name_model']=='SW1L' :
            return 2
        elif self.config['name_model']=='SW1LM' :
            return [2 + i*3 for i in range(self.config.Nmodes+1)]
        elif hasattr(self.config['name_model'],'__len__') and len(self.config['name_model'])==2 :
            ind = [i*3 for i in range(self.config.Nmodes+1)]
            ind.append(-1)
            return ind
        else :
            return 0


    
    
    

    
