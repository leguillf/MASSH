#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:36:20 2021

@author: leguillou
"""
from .config import USE_FLOAT64
from importlib.machinery import SourceFileLoader 
import sys
import xarray as xr
import numpy as np
import os
from math import pi
from datetime import timedelta
import matplotlib.pylab as plt 
import pyinterp
from copy import deepcopy
import time

import jax
import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
from jax.lax import scan, fori_loop

jax.config.update("jax_enable_x64", USE_FLOAT64)

import warnings 

try:
    import jaxparrow
except ImportError:
    warnings.warn('jaxparrow not installed, some jax functions will not be available')


from . import  grid
from .tools import gaspari_cohn
from .exp import Config as Config


def Model(config, State, verbose=True):
    """
    NAME
        Model

    DESCRIPTION
        Main function calling subclass for specific models
    """
    if config.MOD is None:
        return
    
    elif config.MOD.super is None:
        return Model_multi(config,State)

    elif config.MOD.super is not None:
        if verbose:
            print(config.MOD)
        if config.MOD.super=='MOD_Id':
            return Model_Id(config,State)
        if config.MOD.super=='MOD_DIFF':
            return Model_diffusion(config,State)
        elif config.MOD.super=='MOD_QG1L_JAX':
            return Model_qg1l_jax(config,State)
        elif config.MOD.super=='MOD_CSW1L':
            return Model_csw1l(config,State)
        elif config.MOD.super=='MOD_QGSW':
            return Model_qgsw(config,State)
        elif config.MOD.super=='MOD_BMIT':
            return Model_bmit(config,State)
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    

###############################################################################
#                            Mother Model                                     #
###############################################################################

class M:

    def __init__(self,config,State):
        
        # Time parameters
        self.dt = config.MOD.dtmodel
        if self.dt>0:
            self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        else:
            self.nt = 1
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        if self.dt>0:
            self.timestamps = [] 
            t = config.EXP.init_date
            while t<=config.EXP.final_date:
                self.timestamps.append(t)
                t += timedelta(seconds=self.dt)
            self.timestamps = np.asarray(self.timestamps)
        else:
            self.timestamps = np.array([config.EXP.init_date])

        # Model variables
        self.name_var = config.MOD.name_var
        if config.MOD.var_to_save is not None:
            self.var_to_save = config.MOD.var_to_save 
        else:
            self.var_to_save = []
            for name in self.name_var:
                self.var_to_save.append(self.name_var[name])


    def init(self, State, t0=0):

        return
    
    def set_bc(self,time_bc,var_bc):

        return

    def ano_bc(self,t,State,sign):

        return
        
    def step(self,State,nstep=1,t=None):

        return 
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        return

    def step_adj(self,adState,State,nstep=1,t=None):

        return
    
    def save_output(self,State,present_date,name_var=None,t=None):
        
        State.save_output(present_date,name_var)
    

###############################################################################
#                            Identity Model                                   #
###############################################################################

class Model_Id(M):
    
    def __init__(self,config,State):

        super().__init__(config,State)
        

        # Initialization 
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        
        # Model Parameters (Flux)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))


        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        
        # Weight map to apply BC in a smoothed way
        if config.MOD.dist_sponge_bc is not None:
            Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
        else:
            Wbc = np.zeros((State.ny,State.nx)) 
            
        self.Wbc = Wbc
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State)
            print('Adjoint test:')
            adjoint_test(self,State)

    def init(self, State, t0=0):

        for name in self.name_var: 
            if t0 in self.bc[name]:
                if self.init_from_bc:
                    State.setvar(self.bc[name][t0], self.name_var[name])
                else:
                    State.var[self.name_var[name]] = self.Wbc * self.bc[name][t0]

    def set_bc(self,time_bc,var_bc):
        
        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        self.bc[_name_var_mod][t] = var_bc[_name_var_bc][i]
        

    def _apply_bc(self,State,t0,t):

        for name in self.name_var:
            if t not in self.bc[name]:
                State.var[self.name_var[name]] +=\
                    self.Wbc * (self.bc[name][t]-self.bc[name][t0]) 

    def step(self,State, nstep=1,**kwargs):

        # Loop on model variables
        for name in self.name_var:

            # Current trajectory
            var0 = State.getvar(self.name_var[name])
            var1 = +var0
            params = State.params[self.name_var[name]]
            var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            #var1[var1>4] = 4
            var1[var1<0] = 0

            State.setvar(var1, self.name_var[name])

        
    def step_tgl(self,dState,State, nstep=1,**kwargs):

        # Loop on model variables
        for name in self.name_var:

            # Current trajectory
            var0 = State.getvar(self.name_var[name])
            var1 = +var0
            params = State.params[self.name_var[name]]
            var1 += params

            # Tangent trajectory
            dvar0 = dState.getvar(self.name_var[name])
            dvar1 = +dvar0
            dparams = dState.params[self.name_var[name]]
            dvar1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * dparams
            #dvar1[var1>4] = 0.
            dvar1[var1<0] = 0.
            
            dState.setvar(dvar1, self.name_var[name])

        
    def step_adj(self,adState,State, nstep=1,**kwargs):

        # Loop on model variables
        for name in self.name_var:

            # Current trajectory
            var0 = State.getvar(self.name_var[name])
            var1 = +var0
            params = State.params[self.name_var[name]]
            var1 += params

            # Adjoint trajectory
            advar0 = adState.getvar(self.name_var[name])
            advar1 = +advar0
            #advar1[var1>4] = 0.
            advar1[var1<0] = 0.
            adState.params[self.name_var[name]] += (1-self.Wbc)*nstep*self.dt/(3600*24) * advar1
            
            adState.setvar(advar1, self.name_var[name])


###############################################################################
#                            Diffusion Models                                 #
###############################################################################
        
class Model_diffusion(M):
    
    def __init__(self,config,State):

        super().__init__(config,State)
        
        self.Kdiffus = config.MOD.Kdiffus
        self.SIC_mod = config.MOD.SIC_mod
        self.dx = State.DX
        self.dy = State.DY

        # Initialization 
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    var_init.data[np.isnan(var_init)] = 0.
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        
        # Model Parameters (Flux)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        
        # Weight map to apply BC in a smoothed way
        if config.MOD.dist_sponge_bc is not None:
            Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
        else:
            Wbc = np.zeros((State.ny,State.nx)) 
            if State.mask is not None:
                for i,j in np.argwhere(State.mask):
                    for p1 in [-1,0,1]:
                        for p2 in [-1,0,1]:
                            itest=i+p1
                            jtest=j+p2
                            if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                                if Wbc[itest,jtest]==0:
                                    Wbc[itest,jtest] = 1
        self.Wbc = Wbc
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State)
            print('Adjoint test:')
            adjoint_test(self,State)

    

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and name in self.bc and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])

    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()
        State0.save_output(present_date, name_var=self.name_var.values())

    def set_bc(self,time_bc,var_bc):
        
        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        self.bc[_name_var_mod][t] = var_bc[_name_var_bc][i]


    def step(self,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            var0 = State.getvar(self.name_var[name])
            
            # Init
            var1 = +var0

            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    var1[1:-1,1:-1] += self.dt*self.Kdiffus*(\
                        (var1[1:-1,2:]+var1[1:-1,:-2]-2*var1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                        (var1[2:,1:-1]+var1[:-2,1:-1]-2*var1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
            
            # Update state
            if self.name_var[name] in State.params:
                params = State.params[self.name_var[name]]
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            State.setvar(var1, self.name_var[name])
        


    def step_tgl(self,dState,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            var0 = dState.getvar(self.name_var[name])
            
            # Init
            var1 = +var0
            
            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    var1[1:-1,1:-1] += self.dt*self.Kdiffus*(\
                        (var1[1:-1,2:]+var1[1:-1,:-2]-2*var1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                        (var1[2:,1:-1]+var1[:-2,1:-1]-2*var1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
            

            # Update state
            if self.name_var[name] in dState.params:
                params = dState.params[self.name_var[name]]
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params

            dState.setvar(var1,self.name_var[name])
        
    def step_adj(self,adState,State,nstep=1,t=None):

        # Loop on model variables
        for name in self.name_var:

            # Get state variable
            advar0 = adState.getvar(self.name_var[name])

            # Init
            advar1 = +advar0
            
            # Time propagation
            if self.Kdiffus>0:
                for _ in range(nstep):
                    
                    advar1[1:-1,2:] += self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,:-2] += self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,1:-1] += -2*self.dt*self.Kdiffus/(self.dx[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    
                    advar1[2:,1:-1] += self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[:-2,1:-1] += self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    advar1[1:-1,1:-1] += -2*self.dt*self.Kdiffus/(self.dy[1:-1,1:-1]**2) * advar0[1:-1,1:-1]
                    
                    advar0 = +advar1
                

            # Update state and parameters
            if self.name_var[name] in State.params:
                adState.params[self.name_var[name]] += (1-self.Wbc)*nstep*self.dt/(3600*24) * advar0 
            
            advar1[np.isnan(advar1)] = 0
            adState.setvar(advar1,self.name_var[name])


###############################################################################
#                       Quasi-Geostrophic Models                              #
###############################################################################
    
class Model_qg1l_jax(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",f'{dir_model}/jqgm.py').load_module() 
        model = getattr(qgm, config.MOD.name_class)

        # Coriolis
        if config.MOD.f0 is not None and config.MOD.constant_f:
            self.f = config.MOD.f0
        else:
            self.f = State.f
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
            
        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            self.mdt = None
        
        # Open Bathymetry if provided
        if config.MOD.file_bathy_aux is not None and os.path.exists(config.MOD.file_bathy_aux):

            ds = xr.open_dataset(config.MOD.file_bathy_aux)
            name_lon = config.MOD.name_var_bathy['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    
            
            bathymetry = grid.interp2d(ds,
                                   config.MOD.name_var_bathy,
                                   State.lon,
                                   State.lat)

            # Get H so that we have a maximum value 
            bathymetry[State.mask] = np.nan
            H =  np.nanmean(bathymetry) # Mean depth
            self.bathymetry_PV_term = - (bathymetry - H) / H
            self.bathymetry_PV_term[self.bathymetry_PV_term>config.MOD.bathy_ratio_max] = config.MOD.bathy_ratio_max
            self.bathymetry_PV_term[self.bathymetry_PV_term<-config.MOD.bathy_ratio_max] = -config.MOD.bathy_ratio_max
            self.bathymetry_PV_term[np.isnan(self.bathymetry_PV_term)] = config.MOD.bathy_ratio_max
            
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.bathymetry_PV_term)
                plt.colorbar()
                plt.title('Bathymetry PV term')
                plt.show()
                
        else:
            self.bathymetry_PV_term = np.zeros((State.ny,State.nx))
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmean(State.DX), np.nanmean(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmean(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for c=', np.nanmean(self.c))
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

            
        # Initialize model state
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            try:
                dsin = xr.open_dataset(config.GRID.path_init_grid, group='variables')
            except:
                dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
            dsin.close()
            del dsin
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
                if State.mask is not None:
                    State.var[self.name_var[name]][State.mask] = np.nan

        # Initialize model Parameters (Flux on SSH and tracers)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        self.Wbc = np.zeros((State.ny,State.nx))
        if config.MOD.dist_sponge_bc is not None and State.mask is not None:
            if config.MOD.advect_tracer and config.MOD.bc_trac=='OBC':
                bc = False # No sponge band for open boundaries
            else:
                bc = True
            self.Wbc = grid.compute_weight_map(State.lon, State.lat, deepcopy(State.mask), config.MOD.dist_sponge_bc, bc=bc)
            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.Wbc)
                plt.colorbar()
                plt.title('Wbc')
                plt.show()

        # Use boundary conditions as mean field (for 4Dvar only)
        if config.INV is not None and config.INV.super=='INV_4DVAR':
            self.anomaly_from_bc = config.INV.anomaly_from_bc
        else:
            self.anomaly_from_bc = False

        # Tracer advection flag
        self.advect_pv = config.MOD.advect_pv
        self.advect_tracer = config.MOD.advect_tracer
        self.forcing_tracer_from_bc = config.MOD.forcing_tracer_from_bc

        # Ageostrophic velocity flag
        if 'U' in self.name_var and 'V' in self.name_var:
            self.ageo_velocities = True
        else:
            self.ageo_velocities = False
        
        # Save SSH, geostrophic velocities and cyclogeostrophic velocities
        self.save_diagnosed_variables = config.MOD.save_diagnosed_variables
        # Save control parameters, i.e. corrective fluxes
        self.save_params = config.MOD.save_params

       # Masked array for model initialization
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
    
        # Model initialization
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=SSH0,
                         c=self.c,
                         upwind=config.MOD.upwind,
                         time_scheme=config.MOD.time_scheme,
                         g=State.g,
                         f=self.f,
                         Wbc=self.Wbc,
                         Kdiffus=config.MOD.Kdiffus,
                         Kdiffus_trac=config.MOD.Kdiffus_trac,
                         bc_trac=config.MOD.bc_trac,
                         advect_pv=self.advect_pv,
                         ageo_velocities=self.ageo_velocities,
                         constant_c=config.MOD.constant_c,
                         constant_f=config.MOD.constant_f,
                         solver=config.MOD.solver,
                         tile_size=config.MOD.tile_size,
                         tile_overlap=config.MOD.tile_overlap,
                         mdt=self.mdt,
                         bathymetry_PV_term=self.bathymetry_PV_term)

        # Model functions initialization
        self.qgm_step = self.qgm.step_jit
        self.qgm_step_tgl = self.qgm.step_tgl_jit
        self.qgm_step_adj = self.qgm.step_adj_jit

        self.step_jax_jit = jit(self.step_jax, static_argnums=[3])

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('QG1L_JAX Tangent test:')
            tangent_test(self,State,nstep=10)
            print('QG1L_JAX Adjoint test:')
            adjoint_test(self,State,nstep=10)
    
    def init(self, State, t0=0):

        if self.anomaly_from_bc:
            return
        elif type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])
    
    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()

        name_var = [self.name_var['SSH']]

        if self.advect_tracer:
            for name in self.name_var:
                if name!='SSH':
                    name_var += [self.name_var[name]]
        
        # Save SSH and geostrophic velocities
        name_var_diag = []
        if self.save_diagnosed_variables:

            # Add MDT to sla to get SSH
            if self.mdt is not None:
                ssh = State0.getvar(name_var=self.name_var['SSH']) + self.mdt
                State0.setvar(ssh, name_var='ssh')
                if name_var is not None:
                    name_var_diag += ['ssh']
            else:
                ssh = State0.getvar(name_var=self.name_var['SSH'])
            
            # Current velocities
            # Geostrophy
            result = jaxparrow.geostrophy(ssh, State0.lat, State0.lon, State0.mask)
            ug, vg = result[0], result[1]
            ug_t = jaxparrow.tools.operators.interpolation(ug, State0.mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
            vg_t = jaxparrow.tools.operators.interpolation(vg, State0.mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
            # Cyclogeostrophy
            result = jaxparrow.cyclogeostrophy(ssh, State0.lat, State0.lon, State0.mask)
            uc, vc = result[0], result[1]
            uc_t = jaxparrow.tools.operators.interpolation(uc, State0.mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
            vc_t = jaxparrow.tools.operators.interpolation(vc, State0.mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
            # Set geostrophic velocities
            State0.setvar(ug_t, name_var='ug')
            State0.setvar(vg_t, name_var='vg')
            # Set cyclogeostrophic velocities
            State0.setvar(uc_t, name_var='uc')
            State0.setvar(vc_t, name_var='vc')
            # Flux
            Fssh = State0.params[self.name_var['SSH']]
            State0.setvar(Fssh, name_var='Fssh')

            if name_var is not None:
                name_var_diag += ['ug', 'vg', 'uc', 'vc', 'Fssh']

        State0.save_output(present_date, name_var+name_var_diag)#, save_params=self.save_params)

    def set_bc(self,time_bc,var_bc):

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Remove nan
                        var_bc_t[np.isnan(var_bc_t)] = 0.
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_bc):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
                        self.forcing[_name_var_mod][t] = var_bc[_name_var_bc][i]
        
        # For step_jax
        self.bc_time_jax = jnp.array(list(self.bc['SSH'].keys()))
        self.bc_time = np.array(list(self.bc['SSH'].keys()))
        self.bc_values = {t: jnp.array(self.bc['SSH'][t]) for t in self.bc['SSH']}
        for name in var_bc:
            self.bc_values[name] = jnp.array(list(self.bc['SSH'].values()))
            
    def _apply_bc(self,t0,t1):
        
        Xb = jnp.zeros((self.ny,self.nx,))
        
        if 'SSH' not in self.bc:
            return Xb
        elif len(self.bc['SSH'].keys())==0:
             return Xb
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        Xb = self.bc_values[t0]

        if self.advect_tracer:
            Xb = Xb[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH' and name in self.bc and len(self.bc[name].keys())>0:
                    if t1 in self.bc[name]: 
                        Cb = self.bc[name][t1]
                    else:
                        # Find closest time
                        t_list = np.array(list(self.bc['SSH'].keys()))
                        idx_closest = np.argmin(np.abs(t_list-t1))
                        new_t1 = t_list[idx_closest]
                        Cb = self.bc[name][new_t1]
                    Xb = np.append(Xb, Cb[np.newaxis,:,:], axis=0)     
        
        return Xb

    def _apply_bc_jax(self,t,t1):

        Xb = jnp.zeros((self.ny,self.nx,))

        idt = jnp.where(self.bc_time_jax==t, size=1)[0]
        Xb = self.bc_values['SSH'][idt][0]

        if self.advect_tracer:
            for name in self.name_var:
                if name!='SSH':
                    idt = jnp.where(self.bc_time_jax==t1, size=1)[0]
                    Cb = self.bc_values[name][idt][0]
                    Xb = jnp.append(Xb[jnp.newaxis,:,:], 
                                    Cb[jnp.newaxis,:,:], axis=0)     
        return Xb
    
    def step(self,State,nstep=1,t=0):

        # Boundary feld
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable(s)
        X0 = State.getvar(name_var=self.name_var['SSH'])
        if self.advect_tracer:
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = jnp.append(X0, U, axis=0)
                X0 = jnp.append(X0, V, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    C0 = State.getvar(name_var=self.name_var[name])[jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1, Xb, nstep=nstep)
        
        # Update state
        if self.name_var['SSH'] in State.params:
            Fssh = State.params[self.name_var['SSH']]
            if self.advect_tracer: 
                State.setvar(X1[0]+ nstep*self.dt/(3600*24) * Fssh, 
                             name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = State.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            State.setvar(X1[i] + nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc * (Xb[i] - X0[i]), 
                                         name_var=self.name_var[name])
                        # Only forcing flux
                        else:
                            State.setvar(X1[i] + nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc ,
                                        name_var=self.name_var[name])
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State.setvar(X1, name_var=self.name_var['SSH'])
    
    def step_jax(self,t,State_vars,State_params,nstep=1):

        # Boundary field
        Xb = self._apply_bc_jax(t,t+nstep*self.dt)

        # Get state variable(s)
        X0 = State_vars[self.name_var['SSH']]
        if self.advect_tracer:
            X0 = X0[jnp.newaxis,:,:]
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    C0 = State_vars[self.name_var[name]][jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)

        # Update state
        if self.name_var['SSH'] in State_params:
            Fssh = State_params[self.name_var['SSH']] # Forcing term for SSH
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh 
                State_vars[self.name_var['SSH']] = X1[0]
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = State_params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Only forcing flux
                        X1[i] += nstep*self.dt/(3600*24) * (1-self.Wbc)  * Fc 
                        State_vars[self.name_var[name]] = X1[i]
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State_vars[self.name_var['SSH']] = X1
        
        for name in self.name_var:
            State_vars[self.name_var[name]] = jnp.nan_to_num(State_vars[self.name_var[name]])
            
        return State_vars

    def step_tgl(self,dState,State,nstep=1,t=0):

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))
        
        # Get state variable
        dX0 = dState.getvar(name_var=self.name_var['SSH']).astype('float64')
        X0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        if self.advect_tracer:
            dX0 = dX0[np.newaxis,:,:]
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                dU = dState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                dV = dState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                dX0 = np.append(dX0, dU, axis=0)
                dX0 = np.append(dX0, dV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    dC0 = dState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    dX0 = np.append(dX0, dC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        dX1 = +dX0.astype('float64')
        X1 = +X0.astype('float64')

        # Time propagation
        dX1 = self.qgm_step_tgl(dX1,X1,Xb,nstep=nstep)

        # Convert to numpy and reshape
        dX1 = np.array(dX1).astype('float64')

        # Update state
        if self.name_var['SSH'] in dState.params:
            dFssh = dState.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            if self.advect_tracer:
                dX1[0] +=  nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        dFc = dState.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            Fc = State.params[self.name_var[name]] 
                            dX1[i] +=  nstep*self.dt/(3600*24) * (1-self.Wbc) *\
                                  (dFc * (Xb[i] - X0[i]) - Fc * dX0[i])
                        # Only forcing flux
                        else:
                            dX1[i] +=  nstep*self.dt/(3600*24) * dFc  * (1-self.Wbc)
                        dState.setvar(dX1[i], name_var=self.name_var[name])
            else:
                dX1 += nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1, name_var=self.name_var['SSH'])

    def step_adj(self,adState,State,nstep=1,t=0):

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable
        adSSH0 = adState.getvar(name_var=self.name_var['SSH'])#.astype('float64')
        SSH0 = State.getvar(name_var=self.name_var['SSH'])#.astype('float64')
        if self.advect_tracer:
            adX0 = adSSH0[jnp.newaxis,:,:]
            X0 = SSH0[jnp.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                adU = adState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                adV = adState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                adX0 = np.append(adX0, adU, axis=0)
                adX0 = np.append(adX0, adV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    adC0 = adState.getvar(name_var=self.name_var[name])[jnp.newaxis,:,:]
                    adX0 = jnp.append(adX0, adC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        else:
            adX0 = adSSH0
            X0 = SSH0

        # Init
        adX1 = +adX0
        X1 = +X0

        # Time propagation
        adX1 = self.qgm_step_adj(adX1,X1,Xb,nstep=nstep)

        # Update state and parameters
        if self.name_var['SSH'] in adState.params:
            for i,name in enumerate(self.name_var):
                adparams = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name])#.astype('float64') 
                if name!='SSH':
                    adparams *= (1-self.Wbc)
                    if self.forcing_tracer_from_bc:
                        Fc = State.params[self.name_var[name]] 
                        adparams *=  (Xb[i] - X0[i])
                        adX1[i] += -nstep*self.dt/(3600*24) * (1-self.Wbc) * Fc * adX0[i]
                adState.params[self.name_var[name]] += adparams  
                
        if self.advect_tracer:
            adState.setvar(adX1[0],self.name_var['SSH'])
            for i,name in enumerate(self.name_var):
                if name!='SSH':
                    adState.setvar(adX1[i],self.name_var[name])
        else:
            adState.setvar(adX1,self.name_var['SSH'])


###############################################################################
#                         Shallow Water Models                                #
###############################################################################

class Model_csw1l(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        self.config = config

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = config.MOD.dir_model
        
        swm = SourceFileLoader("swm", 
                                dir_model + "/jswm.py").load_module()
        model = swm.CSWm
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx
        if config.MOD.force_constant_km_grid:
            if config.MOD.dx_km is not None:
                State.dx = config.MOD.dx_km * 1000
            if config.MOD.dy_km is not None:
                State.dy = config.MOD.dy_km * 1000
            DX = np.ones_like(State.lon)*State.dx
            DY = np.ones_like(State.lat)*State.dy
            X, Y = grid.dxdy2xy(DX,DY)
            self.DX = DX
            self.DY = DY
            print('Force constant grid: DX=',State.dx,' DY=',State.dy)
        else:
            X = State.X
            Y = State.Y
            self.DX = State.DX
            self.DY = State.DY
            

        # Gravity
        self.g = State.g

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            print('Warning: No MDT file provided. MDT set to zero. It might affect coupling terms computation.')
            self.mdt = np.zeros((State.ny,State.nx))

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))

        # Open Coriolis Frequency if provided
        if config.MOD.filef_aux is not None and os.path.exists(config.MOD.filef_aux):

            ds = xr.open_dataset(config.MOD.filef_aux)
            name_lon = config.MOD.name_var_f['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.f = grid.interp2d(ds,
                                   config.MOD.name_var_f,
                                   State.lon,
                                   State.lat)
        
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.f)
                plt.colorbar()
                plt.title('Coriolis Frequency')
                plt.show()
                
        else:
            self.f = State.f
        self.f_on_u = (self.f[:, :-1] + self.f[:, 1:]) / 2
        self.f_on_v = (self.f[:-1, :] + self.f[1:, :]) / 2
            
        # Equivalent depth
        self.Heb = self.c**2 / self.g

        # Mean depth (bathymetry)
        self.H = config.MOD.H 
        
        # Boundary angles
        if config.MOD.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
        else:
            self.bc_theta = np.array([0])
        
        # Open Boundary condition kind
        self.bc_kind = config.MOD.bc_kind
        
        # Tide frequencies
        self.omegas = np.asarray(config.MOD.w_waves)
        

        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmean(State.DX), np.nanmean(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmax(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for max(c)=', np.nanmax(self.c), 'm/s and grid spacing ~', grid_spacing, 'm: dt<=', dt, 's')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

        
        # Initialize model state
        self.name_var = config.MOD.name_var
        self.var_to_save = [self.name_var['SSH']] # ssh

        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    if name=='U':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                    elif name=='V':
                        State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                    elif name=='SSH':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                if name=='U':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                elif name=='V':
                    State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                elif name=='SSH':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))

        
        # Model Parameters (OBC, He and alpha coefficient)
        self.name_params = config.MOD.name_params
        if 'He_mean' in self.name_params:
            State.params['He_mean'] = np.zeros((self.ny, self.nx))
        if 'alpha' in self.name_params:
            State.params['alpha'] = np.zeros((self.ny, self.nx))
        if 'alpha_He' in self.name_params:
            State.params['alpha_He'] = np.zeros((self.ny, self.nx))
        if 'alpha_Uu' in self.name_params: 
            State.params['alpha_Uu'] = np.zeros((self.ny, self.nx))
        if 'alpha_Uz' in self.name_params: 
            State.params['alpha_Uz'] = np.zeros((self.ny, self.nx))
        if 'alpha_Up' in self.name_params: 
            State.params['alpha_Up'] = np.zeros((self.ny, self.nx))
        if 'hbc' in self.name_params:
            self.shapehbcx = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.nx # NX
                              ]
            self.shapehbcy = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.ny # NY
                              ]
            State.params['hbcx'] = np.zeros((self.shapehbcx))
            State.params['hbcy'] = np.zeros((self.shapehbcy))
        

        # Coupling terms from vertical modes
        self.flag_coupling_from_bm = config.MOD.flag_coupling_from_bm
        if self.flag_coupling_from_bm:
            self._compute_coupling_coefficients(config.MOD.path_vertical_modes)
        if self.flag_coupling_from_bm and config.MOD.path_bm is not None and os.path.exists(config.MOD.path_bm):
            self.is_bm = True
            self._compute_bm_fields(config.MOD.path_bm, config.MOD.name_var_bm)
        else:
            self.is_bm = False

        # Sponge layer
        self.flag_bc_sponge = config.MOD.flag_bc_sponge
        self.sponge_width = config.MOD.dist_sponge_bc
        self.sponge_coef = config.MOD.sponge_coef
        if self.flag_bc_sponge:
            self.Xh = State.X
            self.Yh = State.Y
            self.Xu = 0.5 * (State.X[:, :-1] + State.X[:, 1:])
            self.Yu = 0.5 * (State.Y[:, :-1] + State.Y[:, 1:])
            self.Xv = 0.5 * (State.X[:-1, :] + State.X[1:, :])
            self.Yv = 0.5 * (State.Y[:-1, :] + State.Y[1:, :])

            if config.MOD.periodic_x and config.MOD.periodic_y:
                sponge_h = np.zeros_like(self.Xh)
                sponge_u = np.zeros_like(self.Xu)
                sponge_v = np.zeros_like(self.Xv)
            elif config.MOD.periodic_x and not config.MOD.periodic_y:
                sponge_h = np.maximum.reduce([gaspari_cohn(self.Yh-self.Yh[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yh[-1,:][None,:]-self.Yh, self.sponge_width*1e3)])
                sponge_u = np.maximum.reduce([gaspari_cohn(self.Yu-self.Yu[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yu[-1,:][None,:]-self.Yu, self.sponge_width*1e3)])
                sponge_v = np.maximum.reduce([gaspari_cohn(self.Yv-self.Yv[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yv[-1,:][None,:]-self.Yv, self.sponge_width*1e3)])
            elif not config.MOD.periodic_x and config.MOD.periodic_y:
                sponge_h = np.maximum.reduce([gaspari_cohn(self.Xh-self.Xh[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xh[:,-1][:,None]-self.Xh, self.sponge_width*1e3)])
                sponge_u = np.maximum.reduce([gaspari_cohn(self.Xu-self.Xu[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xu[:,-1][:,None]-self.Xu, self.sponge_width*1e3)])
                sponge_v = np.maximum.reduce([gaspari_cohn(self.Xv-self.Xv[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xv[:,-1][:,None]-self.Xv, self.sponge_width*1e3)])
            else:
                sponge_h = np.maximum.reduce([gaspari_cohn(self.Xh-self.Xh[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xh[:,-1][:,None]-self.Xh, self.sponge_width*1e3), gaspari_cohn(self.Yh-self.Yh[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yh[-1,:][None,:]-self.Yh, self.sponge_width*1e3)])
                sponge_u = np.maximum.reduce([gaspari_cohn(self.Xu-self.Xu[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xu[:,-1][:,None]-self.Xu, self.sponge_width*1e3), gaspari_cohn(self.Yu-self.Yu[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yu[-1,:][None,:]-self.Yu, self.sponge_width*1e3)])
                sponge_v = np.maximum.reduce([gaspari_cohn(self.Xv-self.Xv[:,0][:,None], self.sponge_width*1e3), gaspari_cohn(self.Xv[:,-1][:,None]-self.Xv, self.sponge_width*1e3), gaspari_cohn(self.Yv-self.Yv[0,:][None,:], self.sponge_width*1e3), gaspari_cohn(self.Yv[-1,:][None,:]-self.Yv, self.sponge_width*1e3)])  

            def _normalize_sponge(sponge):
                sponge_min = jnp.min(sponge)
                sponge_max = jnp.max(sponge)
                return (sponge - sponge_min) / (sponge_max - sponge_min)

            self.sponge_h = _normalize_sponge(sponge_h)
            self.sponge_u = _normalize_sponge(sponge_u)
            self.sponge_v = _normalize_sponge(sponge_v)

            if not config.MOD.periodic_y:
                self.sponge_on_h_S = (np.abs(self.Yh-self.Yh[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_S = (np.abs(self.Yu-self.Yu[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_S = (np.abs(self.Yv-self.Yv[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_N = (np.abs(self.Yh-self.Yh[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_N = (np.abs(self.Yu-self.Yu[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_v_N = (np.abs(self.Yv-self.Yv[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
            else:
                self.sponge_on_h_S = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_S = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_S = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_N = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_N = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_N = np.zeros_like(self.Yv, dtype=bool)
            if not config.MOD.periodic_x:
                self.sponge_on_h_W = (np.abs(self.Xh-self.Xh[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_W = (np.abs(self.Xu-self.Xu[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_W = (np.abs(self.Xv-self.Xv[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_E = (np.abs(self.Xh-self.Xh[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_E = (np.abs(self.Xu-self.Xu[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_E = (np.abs(self.Xv-self.Xv[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
            else:
                self.sponge_on_h_W = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_W = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_W = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_E = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_E = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_E = np.zeros_like(self.Yv, dtype=bool)
            
            self.weight_sponge_u = np.array(self.sponge_on_u_S, dtype=float) + np.array(self.sponge_on_u_N, dtype=float) + np.array(self.sponge_on_u_W, dtype=float) + np.array(self.sponge_on_u_E, dtype=float)
            self.weight_sponge_v = np.array(self.sponge_on_v_S, dtype=float) + np.array(self.sponge_on_v_N, dtype=float) + np.array(self.sponge_on_v_W, dtype=float) + np.array(self.sponge_on_v_E, dtype=float)
            self.weight_sponge_h = np.array(self.sponge_on_h_S, dtype=float) + np.array(self.sponge_on_h_N, dtype=float) + np.array(self.sponge_on_h_W, dtype=float) + np.array(self.sponge_on_h_E, dtype=float)
            self.weight_sponge_u[self.weight_sponge_u==0] = 1
            self.weight_sponge_v[self.weight_sponge_v==0] = 1
            self.weight_sponge_h[self.weight_sponge_h==0] = 1

            # Update mask (for avoiding assimilation in sponge regions)
            State.mask[self.sponge_on_h_S] = True
            State.mask[self.sponge_on_h_N] = True
            State.mask[self.sponge_on_h_W] = True
            State.mask[self.sponge_on_h_E] = True
        
        self.mask = State.mask

        # Model initialization
        self.swm = model(X=X,
                         Y=Y,
                         dt=self.dt,
                         bc=self.bc_kind,
                         omegas=self.omegas,
                         bc_theta=self.bc_theta,
                         f=self.f,
                         Heb=self.Heb,
                         obc_north=config.MOD.obc_north,
                         obc_west=config.MOD.obc_west,
                         obc_south=config.MOD.obc_south,
                         obc_east=config.MOD.obc_east,
                         periodic_x=config.MOD.periodic_x,
                         periodic_y=config.MOD.periodic_y,
                         )
    
        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep)
        self._jstep_tgl_jit = jit(self._jstep_tgl)
        self._jstep_adj_jit = jit(self._jstep_adj)
        self._compute_advective_terms_from_bm_jit = jit(self._compute_advective_terms_from_bm)
        self._compute_He_from_bm_jit = jit(self._compute_He_from_bm)
        self._compute_w1_IT_jit = jit(self._compute_w1_IT)
        self._compute_IT_2D_jit = jit(self._compute_IT_2D)

        # Functions related to time_scheme
        if self.time_scheme=='Euler':
            self.swm_step = self.swm.step_euler_jit
            self.swm_step_tgl = self.swm.step_euler_tgl_jit
            self.swm_step_adj = self.swm.step_euler_adj_jit
        elif self.time_scheme=='rk4':
            self.swm_step = self.swm.step_rk4_jit
            self.swm_step_tgl = self.swm.step_rk4_tgl_jit
            self.swm_step_adj = self.swm.step_rk4_adj_jit
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('CSW1L Tangent test:')
            #tangent_test(self,State,nstep=10)
            print('CSW1L Adjoint test:')
            #adjoint_test(self,State,nstep=1)

    def save_output(self,State,present_date,name_var=None,t=None):

        name_var_to_save = [self.name_var['SSH'], 
                            self.name_var['U']+'_interp', 
                            self.name_var['V']+'_interp',
                            'He']
        
        u = +State.getvar(name_var=self.name_var['U'])
        v = +State.getvar(name_var=self.name_var['V'])
        u_to_save = np.zeros((State.ny,State.nx))
        v_to_save = np.zeros((State.ny,State.nx))
        u_to_save[:,1:-1] = (u[:,1:] + u[:,:-1]) * .5
        v_to_save[1:-1,:] = (v[1:,:] + v[:-1,:]) * .5
        State.var[self.name_var['U']+'_interp'] = u_to_save
        State.var[self.name_var['V']+'_interp'] = v_to_save

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
        else:
            h_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha' in self.name_params:
            alpha = +State.params['alpha'] 
        else:
            alpha = jnp.zeros((self.ny,self.nx))
        
        He2d = self._compute_He_from_bm(He_mean, alpha, h_bm)

        State.var['He'] = He2d + self.Heb

        if self.flag_coupling_from_bm:
            if 'He_mean' in self.name_params:
                State.var['He_mean'] = State.params['He_mean']
                name_var_to_save += ['He_mean']
            if self.is_bm:
                State.var['ssh_bm'] = h_bm
                name_var_to_save += ['ssh_bm']
            if 'alpha' in self.name_params:
                State.var['alpha'] = State.params['alpha']
                name_var_to_save += ['alpha']
            if 'alpha_He' in self.name_params:  
                State.var['alpha_He'] = State.params['alpha_He']
                name_var_to_save += ['alpha_He']
            if 'alpha_Uu' in self.name_params: 
                State.var['alpha_Uu'] = State.params['alpha_Uu']
                name_var_to_save += ['alpha_Uu']
            if 'alpha_Up' in self.name_params: 
                State.var['alpha_Up'] = State.params['alpha_Up']
                name_var_to_save += ['alpha_Up']
            if 'alpha_Uz' in self.name_params: 
                State.var['alpha_Uz'] = State.params['alpha_Uz']
                name_var_to_save += ['alpha_Uz']

        State.save_output(present_date,
                          name_var=name_var_to_save)

    def _compute_coupling_coefficients(self, path_vertical_modes):

        # Open dataset with vertical mode structures
        ds = xr.open_dataset(path_vertical_modes)
        phi1  = ds.phi.sel(mode=1)      # (s_rho, y_rho)
        phi1 = phi1/phi1[-1]
        phi1p = ds.dphidz.sel(mode=1)   # (s_w,  y_rho)  → given derivative!
        c1    = ds.c.sel(mode=1)        # (y_rho)
        N2    = ds.N2                   # (y_rho, s_w)
        z_r   = ds.z_r                  # (s_rho, y_rho)
        z_w = ds.z_w
        dz_r  = ds.dz                   # (s_rho, y_rho)
        s_w = ds.s_w

        #########################################################################
        # Compute U11 coupling term for mode 1
        #########################################################################
        integrand = phi1/phi1[-1] * (phi1**2) * dz_r  # multiply by layer thickness
        U11u = (1/self.H) * integrand.sum(dim="s_rho")   # integral in z
        self.U11u = jnp.array(U11u.values)  
        plt.figure()
        plt.pcolormesh(self.U11u)
        plt.colorbar()
        plt.title('U11u')
        plt.show()

        #########################################################################
        # Compute U11p coupling term for mode 1
        #########################################################################
        φ = -(c1**2 / N2)* phi1p          # (s_w, y_rho)
        phi1_w = phi1.interp(s_rho=s_w)   # now on (s_w, y_rho)
        phi1_w[0] = phi1_w[1]
        phi1_w[-1] = phi1_w[-2]

        integrand = (
            phi1_w/phi1_w[-1]
            * (N2 / c1**2)
            * φ**2
        )   # dims: (s_w, y_rho)
        # thickness between w-levels: same size as s_w except top/bottom.
        dz_w = z_w.diff("s_w")                   # (s_w_minus1, y_rho)
        # pad to same length as s_w (xarray aligns automatically)
        dz_w = dz_w.reindex(s_w=s_w, fill_value=np.nan)
        U11p = (1/self.H) * (integrand * dz_w).sum(dim="s_w")   # integral in z
        self.U11p = jnp.array(U11p.values)
        plt.figure()
        plt.pcolormesh(self.U11p)
        plt.colorbar()
        plt.title('U11p')
        plt.show()

        #########################################################################
        # Compute U11z coupling term for mode 1
        #########################################################################
        self.U11z = -self.U11p
        plt.figure()
        plt.pcolormesh(self.U11z)
        plt.colorbar()
        plt.title('U11z')
        plt.show()

        #########################################################################
        # Compute dHe coupling term for mode 1
        #########################################################################
        # forward and backward slopes
        slope_f = (phi1.shift(s_rho=-1) - phi1) / (z_r.shift(s_rho=-1) - z_r)
        slope_b = (phi1 - phi1.shift(s_rho=1)) / (z_r - z_r.shift(s_rho=1))

        # central 2nd derivative
        phi1pp = 2 * (slope_f - slope_b) / (z_r.shift(s_rho=-1) - z_r.shift(s_rho=1))

        # fill top/bottom NaNs (simple, safe)
        phi1pp = phi1pp.fillna(0)

        # vertical interpolation from s_w → s_rho
        phi1p_rho = phi1p.interp(s_w=ds.s_rho)

        N2_rho = N2.interp(s_w=ds.s_rho)

        c1_rho = c1.broadcast_like(phi1p_rho)

        A = - (c1_rho**2 / N2_rho) * phi1p_rho
        A2 = A**2

        integrand = phi1pp * A2
        dHe = (1/self.H) * (integrand * dz_r).sum("s_rho")
        self.dHe = jnp.array(dHe.values) 

        plt.figure()
        plt.pcolormesh(self.dHe)
        plt.colorbar()
        plt.title('dHe')
        plt.show()
    
    def _compute_bm_fields(self, path_bm, name_var_bm):

        dsbm = xr.open_mfdataset(path_bm)
        dsbm_interp = dsbm.interp(time=self.timestamps,
                                    method='linear').load()
        self.ssh_bm_data = {}
        self.u_bm_data = {}
        self.v_bm_data = {}
        for t, date in zip(self.T, self.timestamps):
            _h = dsbm_interp[name_var_bm['ssh_bm']].sel(time=date).values
            _u = np.zeros((self.ny,self.nx))
            _v = np.zeros((self.ny,self.nx))
            _u[1:-1,1:] = - self.g/self.f[1:-1,1:]*\
            (_h[2:,:-1]+_h[2:,1:]-_h[:-2,1:]-_h[:-2,:-1])/(4*self.DY[1:-1,1:])
                
            _v[1:,1:-1] = self.g/self.f[1:,1:-1]*\
                (_h[1:,2:]+_h[:-1,2:]-_h[:-1,:-2]-_h[1:,:-2])/(4*self.DX[1:,1:-1])

            self.ssh_bm_data[t] = _h#jnp.array(_h)
            self.u_bm_data[t] = _u#jnp.array(_u)
            self.v_bm_data[t] = _v#jnp.array(_v)
            

            
            
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,8))
        im1 = ax1.pcolormesh(self.ssh_bm_data[0])
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('SSH BM at t=0s')
        im2 = ax2.pcolormesh(self.u_bm_data[0])
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('U BM at t=0s')
        im3 = ax3.pcolormesh(self.v_bm_data[0])
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('V BM at t=0s')
        plt.show()

    def _compute_He_from_bm(self, He, alpha_He, h_bm):
        
        """ Compute equivalent depth with coupling term
        --------------
        Inputs:
        He        : IT equivalent depth control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        He_tot    : Total equivalent depth with coupling term (m)
        """

        if He is not None:
            if self.flag_coupling_from_bm and alpha_He is not None and h_bm is not None:
                He2d =  He + (.5 + alpha_He) * self.dHe * (h_bm - self.mdt)
            else:
                He2d = He
        else:
            He2d = jnp.zeros((self.ny,self.nx))
        
        He2d = jnp.where(self.mask, 0., He2d)
        
        return He2d

    def _compute_advective_terms_from_bm(self, alpha_Uu, alpha_Up, alpha_Uz, u_bm, v_bm):

        """ Compute advective terms with coupling term
        --------------
        Inputs:
        alpha     : IT advective control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        u11, v11  : advective coupling terms for u and v (m/s)
        u11p, v11p: advective coupling terms for u and v (m/s)
        """

        if self.flag_coupling_from_bm and u_bm is not None and v_bm is not None:
            if alpha_Uu is not None:
                u11u = ((.5 - alpha_Uu) + (.5 + alpha_Uu) * self.U11u) * u_bm 
                v11u = ((.5 - alpha_Uu) + (.5 + alpha_Uu) * self.U11u) * v_bm 
            else:
                u11u = None
                v11u = None
            if alpha_Uz is not None:
                u11z = (.5 + alpha_Uz) * self.U11z * u_bm 
                v11z = (.5 + alpha_Uz) * self.U11z * v_bm 
            else:
                u11z = None
                v11z = None
            if alpha_Up is not None:
                u11p = ((.5 - alpha_Up) + (.5 + alpha_Up) * self.U11p) * u_bm 
                v11p = ((.5 - alpha_Up) + (.5 + alpha_Up) * self.U11p) * v_bm 
            else:
                u11p = None
                v11p = None
        else:
            u11u = None
            v11u = None
            u11z = None
            v11z = None
            u11p = None
            v11p = None
        
        return u11u, v11u, u11p, v11p, u11z, v11z

    def _compute_w1_IT(self,t,He,h_SN,h_WE):
        """
        Compute first characteristic variable w1 for internal tides from external 
        data

        Parameters
        ----------
        t : float 
            time in seconds
        He : 2D array
        h_SN : ND array
            amplitude of SSH for southern/northern borders
        h_WE : ND array
            amplitude of SSH for western/eastern borders

        Returns
        -------
        w1ext: 1D array
            flattened  first characteristic variable (South/North/West/East)
        """

        if h_SN is None or h_WE is None:
            return None 
        
        # Adjust time for 1d bc
        if self.bc_kind=='1d':
            t += self.dt
        
        # South
        HeS = (He[0,:]+He[1,:])/2
        fS = (self.f[0,:]+self.f[1,:])/2
        w1S = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fS**2)/(self.g*HeS))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = jnp.cos(theta) * k
                kxy = kx*self.swm.Xv[0,:] + ky*self.swm.Yv[0,:]
                
                h = h_SN[j,0,0,i]* jnp.cos(w*t-kxy)  +\
                        h_SN[j,0,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fS**2)*( \
                    h_SN[j,0,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fS*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,0,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fS*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                
                w1S += v + jnp.sqrt(self.g/HeS) * h
         
        # North
        fN = (self.f[-1,:]+self.f[-2,:])/2
        HeN = (He[-1,:]+He[-2,:])/2
        w1N = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fN**2)/(self.g*HeN))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = -jnp.cos(theta) * k
                kxy = kx*self.swm.Xv[-1,:] + ky*self.swm.Yv[-1,:]
                h = h_SN[j,1,0,i]* jnp.cos(w*t-kxy)+\
                        h_SN[j,1,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fN**2)*(\
                    h_SN[j,1,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fN*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,1,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fN*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                w1N += v - jnp.sqrt(self.g/HeN) * h

        # West
        fW = (self.f[:,0]+self.f[:,1])/2
        HeW = (He[:,0]+He[:,1])/2
        w1W = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fW**2)/(self.g*HeW))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.swm.Xu[:,0] + ky*self.swm.Yu[:,0]
                h = h_WE[j,0,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,0,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fW**2)*(\
                    h_WE[j,0,0,i]*(w*kx*jnp.cos(w*t-kxy) \
                              + fW*ky*jnp.sin(w*t-kxy)
                                  ) +\
                    h_WE[j,0,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fW*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1W += u + jnp.sqrt(self.g/HeW) * h

        
        # East
        HeE = (He[:,-1]+He[:,-2])/2
        fE = (self.f[:,-1]+self.f[:,-2])/2
        w1E = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fE**2)/(self.g*HeE))
            for i,theta in enumerate(self.bc_theta):
                kx = -jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.swm.Xu[:,-1] + ky*self.swm.Yu[:,-1]
                h = h_WE[j,1,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,1,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fE**2)*(\
                    h_WE[j,1,0,i]* (w*kx*jnp.cos(w*t-kxy) \
                                + fE*ky*jnp.sin(w*t-kxy)
                                    ) +\
                    h_WE[j,1,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fE*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1E += u - jnp.sqrt(self.g/HeE) * h
        
        w1ext = (w1S,w1N,w1W,w1E)    
        
        return w1ext

    def _compute_IT_2D(self,t,He,h_SN,h_WE,flag_tangent=True):
        """
        Compute 2D plane wave IT fields 

        Parameters
        ----------
        t : float 
            time in seconds
        He : 2D array
        h_SN : ND array
            amplitude of SSH for southern/northern borders
        h_WE : ND array
            amplitude of SSH for western/eastern borders

        Returns
        -------
        u,v,h: 2D arrays
            
        """

        u_S = jnp.zeros((self.ny,self.nx-1))
        v_S = jnp.zeros((self.ny-1,self.nx))
        h_S = jnp.zeros((self.ny,self.nx))
        u_N = jnp.zeros((self.ny,self.nx-1))
        v_N = jnp.zeros((self.ny-1,self.nx))
        h_N = jnp.zeros((self.ny,self.nx))
        u_W = jnp.zeros((self.ny,self.nx-1))
        v_W = jnp.zeros((self.ny-1,self.nx))
        h_W = jnp.zeros((self.ny,self.nx))
        u_E = jnp.zeros((self.ny,self.nx-1))
        v_E = jnp.zeros((self.ny-1,self.nx))
        h_E = jnp.zeros((self.ny,self.nx))

        He_on_u = (He[:,1:] + He[:,:-1]) /2
        He_on_v = (He[1:,:] + He[:-1,:]) /2

        for j,w in enumerate(self.omegas):
            k_on_h = jnp.sqrt((w**2-self.f**2)/(self.g*He))
            k_on_u = jnp.sqrt((w**2-self.f_on_u**2)/(self.g*(He_on_u)))
            k_on_v = jnp.sqrt((w**2-self.f_on_v**2)/(self.g*(He_on_v)))

            for i,theta in enumerate(self.bc_theta):

                ####################################
                # South
                ####################################
                kx_on_h = jnp.sin(theta) * k_on_h
                ky_on_h = jnp.cos(theta) * k_on_h
                kx_on_u = jnp.sin(theta) * k_on_u
                ky_on_u = jnp.cos(theta) * k_on_u
                kx_on_v = jnp.sin(theta) * k_on_v
                ky_on_v = jnp.cos(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_S += self.sponge_on_h_S *(\
                    h_SN[j,0,0,i]* jnp.cos(w*t-kxy_on_h)  +\
                    h_SN[j,0,1,i]* jnp.sin(w*t-kxy_on_h)) 
                
                # v
                h_cos_theta_on_v_S = h_SN[j,0,0,i]
                h_sin_theta_on_v_S = h_SN[j,0,1,i]
                v_S += self.sponge_on_v_S * (self.g/(w**2-self.f_on_v**2)*( \
                    h_cos_theta_on_v_S * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                    ) +\
                    h_sin_theta_on_v_S * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                    )
                        ))
                
                # u
                if flag_tangent:
                    h_cos_theta_on_u_S = (h_cos_theta_on_v_S[1:] + h_cos_theta_on_v_S[:-1]) * 0.5
                    h_sin_theta_on_u_S = (h_sin_theta_on_v_S[1:] + h_sin_theta_on_v_S[:-1]) * 0.5
                    u_S += self.sponge_on_u_S * (self.g/(w**2-self.f_on_u**2)*( \
                        h_cos_theta_on_u_S * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                    + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                        ) +\
                        h_sin_theta_on_u_S * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                    - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                        )
                            ))
                
                ####################################
                # North
                ####################################
                kx_on_h = +jnp.sin(theta) * k_on_h
                ky_on_h = -jnp.cos(theta) * k_on_h
                kx_on_u = +jnp.sin(theta) * k_on_u
                ky_on_u = -jnp.cos(theta) * k_on_u
                kx_on_v = +jnp.sin(theta) * k_on_v
                ky_on_v = -jnp.cos(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_N += self.sponge_on_h_N *(\
                    h_SN[j,1,0,i]* jnp.cos(w*t-kxy_on_h)  +\
                    h_SN[j,1,1,i]* jnp.sin(w*t-kxy_on_h)) 
                
                # v
                h_cos_theta_on_v_N = h_SN[j,1,0,i]
                h_sin_theta_on_v_N = h_SN[j,1,1,i]
                v_N += self.sponge_on_v_N * (self.g/(w**2-self.f_on_v**2)*( \
                    h_cos_theta_on_v_N * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                    ) +\
                    h_sin_theta_on_v_N * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                    )
                        ))  
                
                # u
                if flag_tangent:
                    h_cos_theta_on_u_N = (h_cos_theta_on_v_N[1:] + h_cos_theta_on_v_N[:-1]) * 0.5
                    h_sin_theta_on_u_N = (h_sin_theta_on_v_N[1:] + h_sin_theta_on_v_N[:-1]) * 0.5
                    u_N += self.sponge_on_u_N * (self.g/(w**2-self.f_on_u**2)*( \
                        h_cos_theta_on_u_N * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                    + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                        ) +\
                        h_sin_theta_on_u_N * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                    - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                        )
                            ))
                
                ####################################
                # West
                ####################################
                kx_on_h = jnp.cos(theta) * k_on_h
                ky_on_h = jnp.sin(theta) * k_on_h
                kx_on_u = jnp.cos(theta) * k_on_u
                ky_on_u = jnp.sin(theta) * k_on_u
                kx_on_v = jnp.cos(theta) * k_on_v
                ky_on_v = jnp.sin(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_W += self.sponge_on_h_W *(\
                    h_WE[j,0,0,i][:,None]* jnp.cos(w*t-kxy_on_h)  +\
                    h_WE[j,0,1,i][:,None]* jnp.sin(w*t-kxy_on_h)) 
                
                # u
                h_cos_theta_on_u_W = h_WE[j,0,0,i][:,None]
                h_sin_theta_on_u_W = h_WE[j,0,1,i][:,None]
                u_W += self.sponge_on_u_W * (self.g/(w**2-self.f_on_u**2)*( \
                    h_cos_theta_on_u_W * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                    ) +\
                    h_sin_theta_on_u_W * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                    )
                        ))

                # v
                if flag_tangent:
                    h_cos_theta_on_v_W = (h_cos_theta_on_u_W[1:] + h_cos_theta_on_u_W[:-1]) * 0.5
                    h_sin_theta_on_v_W = (h_sin_theta_on_u_W[1:] + h_sin_theta_on_u_W[:-1]) * 0.5
                    v_W += self.sponge_on_v_W * (self.g/(w**2-self.f_on_v**2)*( \
                        h_cos_theta_on_v_W * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                    - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                        ) +\
                        h_sin_theta_on_v_W * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                    + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                        )
                            ))  
                
                
                
                ####################################
                # East
                ####################################
                kx_on_h = -jnp.cos(theta) * k_on_h
                ky_on_h = jnp.sin(theta) * k_on_h
                kx_on_u = -jnp.cos(theta) * k_on_u
                ky_on_u = jnp.sin(theta) * k_on_u
                kx_on_v = -jnp.cos(theta) * k_on_v
                ky_on_v = jnp.sin(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_E += self.sponge_on_h_E *(\
                    h_WE[j,1,0,i][:,None]* jnp.cos(w*t-kxy_on_h)  +\
                    h_WE[j,1,1,i][:,None]* jnp.sin(w*t-kxy_on_h))
                
                # u
                h_cos_theta_on_u_E = h_WE[j,1,0,i][:,None]
                h_sin_theta_on_u_E = h_WE[j,1,1,i][:,None]
                u_E += self.sponge_on_u_E * (self.g/(w**2-self.f_on_u**2)*( \
                    h_cos_theta_on_u_E * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                    ) +\
                    h_sin_theta_on_u_E * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                    )
                        ))
                
                # v
                if flag_tangent:
                    h_cos_theta_on_v_E = (h_cos_theta_on_u_E[1:] + h_cos_theta_on_u_E[:-1]) * 0.5
                    h_sin_theta_on_v_E = (h_sin_theta_on_u_E[1:] + h_sin_theta_on_u_E[:-1]) * 0.5
                    v_E += self.sponge_on_v_E * (self.g/(w**2-self.f_on_v**2)*( \
                        h_cos_theta_on_v_E * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                    - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                        ) +\
                        h_sin_theta_on_v_E * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                    + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                        )
                            ))
        
        u_it = (u_S + u_N + u_W + u_E) / self.weight_sponge_u
        v_it = (v_S + v_N + v_W + v_E) / self.weight_sponge_v
        h_it = (h_S + h_N + h_W + h_E) / self.weight_sponge_h


        return u_it, v_it, h_it
    
    def step(self,State,nstep=1,t=0):

        # Get state variable
        u = +State.getvar(name_var=self.name_var['U'])
        v = +State.getvar(name_var=self.name_var['V'])
        h = +State.getvar(name_var=self.name_var['SSH'])

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            alpha_He = +State.params['alpha']
        else:
            alpha_He = None
        if 'alpha_Uu' in self.name_params:
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = None
        if 'alpha_Up' in self.name_params:
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = None
        if 'hbc' in self.name_params:
            h_SN = +State.params['hbcx']
            h_WE = +State.params['hbcy']
        else:
            h_SN = None
            h_WE = None
            
        # Time stepping
        for it in range(nstep):
            # One time step
            u, v, h = self._jstep_jit(t+it*self.dt, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
        
        # Update state
        State.setvar([u,v,h],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def _jstep(self, t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm):

        # Compute equivalent depth anomaly from control parameters
        He2d = self._compute_He_from_bm_jit(He_mean, alpha_He, h_bm)

        # Compute advective terms from control parameters
        u11u, v11u, u11p, v11p, u11z, v11z = self._compute_advective_terms_from_bm_jit(alpha_Uu, alpha_Up, alpha_Uz, u_bm, v_bm)
        
        # Compute characteristic variable from external data
        if not self.flag_bc_sponge:
            w1ext = self._compute_w1_IT_jit(t, self.Heb+He2d, h_SN, h_WE)
        else:
            w1ext = None

        # One forward step
        u1, v1, h1 = self.swm_step(u, v, h, He2d, w1ext=w1ext, u11u=u11u, v11u=v11u, u11p=u11p, v11p=v11p, u11z=u11z, v11z=v11z)

        # Sponge Boundary conditions
        if self.flag_bc_sponge:
            _u_b, _v_b, _h_b = self._compute_IT_2D_jit(t, self.Heb+He2d, h_SN, h_WE)
            u1 += self.sponge_coef * self.sponge_u * (_u_b - u)
            v1 += self.sponge_coef * self.sponge_v * (_v_b - v)
            h1 += self.sponge_coef * self.sponge_h * (_h_b - h)
    
        return u1, v1, h1
        
    def step_tgl(self,dState,State,nstep=1,t=0):
        
        # Get state variable
        du = dState.getvar(name_var=self.name_var['U'])
        dv = dState.getvar(name_var=self.name_var['V'])
        dh = dState.getvar(name_var=self.name_var['SSH'])
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        h = State.getvar(name_var=self.name_var['SSH'])

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            dHe_mean = +dState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = dHe_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            dalpha_He = +dState.params['alpha_He']
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            dalpha_He = +dState.params['alpha']
            alpha_He = +State.params['alpha']
        else:
            alpha_He = dalpha_He = None
        if 'alpha_Uu' in self.name_params:
            dalpha_Uu = +dState.params['alpha_Uu']
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            dalpha_Uu = +dState.params['alpha']
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = dalpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            dalpha_Uz = +dState.params['alpha_Uz']
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            dalpha_Uz = +dState.params['alpha']
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = dalpha_Uz = None
        if 'alpha_Up' in self.name_params:
            dalpha_Up = +dState.params['alpha_Up']
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            dalpha_Up = +dState.params['alpha']
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = dalpha_Up = None
        if 'hbc' in self.name_params:
            dh_SN = dState.params['hbcx']
            dh_WE = dState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            dh_SN = dh_WE = None
            h_SN = h_WE = None
            
        # Time stepping
        for it in range(nstep):
            # One time step
            du, dv, dh = self._jstep_tgl_jit(t+it*self.dt, 
                                             du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE,
                                             u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
            u, v, h = self._jstep_jit(t+it*self.dt, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
        
        # Update state
        dState.setvar([du,dv,dh],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def _jstep_tgl(self, t, 
                   du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE,
                   u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm):
        
        def wrapped_jstep(x):
            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE = x
            return self._jstep(t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
        primals = ((u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE),)
        tangents = ((du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE),)

        _, dy = jax.jvp(wrapped_jstep, primals, tangents)

        return dy  # returns (du, dv, dh)
    
    def step_adj(self, adState, State, nstep=1, t=0):
        
        # Get state variable
        adu = adState.getvar(name_var=self.name_var['U'])
        adv = adState.getvar(name_var=self.name_var['V'])
        adh = adState.getvar(name_var=self.name_var['SSH'])
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        h = State.getvar(name_var=self.name_var['SSH'])    
        
        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            adHe_mean = +adState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = adHe_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            adalpha_He = +adState.params['alpha_He']
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            adalpha_He = +adState.params['alpha']
            alpha_He = +State.params['alpha']
        else:
            alpha_He = adalpha_He = None
        if 'alpha_Uu' in self.name_params:
            adalpha_Uu = +adState.params['alpha_Uu']
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            adalpha_Uu = +adState.params['alpha']
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = adalpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            adalpha_Uz = +adState.params['alpha_Uz']
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            adalpha_Uz = +adState.params['alpha']
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = adalpha_Uz = None
        if 'alpha_Up' in self.name_params:
            adalpha_Up = +adState.params['alpha_Up']
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            adalpha_Up = +adState.params['alpha']
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = adalpha_Up = None
        if 'hbc' in self.name_params:
            adh_SN = adState.params['hbcx']
            adh_WE = adState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            adh_SN = adh_WE = None
            h_SN = h_WE = None
        

        # Current trajectory
        u_list = [u]; v_list = [v]; h_list = [h]
        for it in range(nstep):
            # One time step
            u, v, h = self._jstep_jit(t+it*self.dt, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
            u_list.append(u); v_list.append(v); h_list.append(h)
            
        # Reverse-time adjoint loop
        for it in reversed(range(nstep)):
            # Non-linear trajectory at time it
            u = u_list[it]
            v = v_list[it]
            h = h_list[it]
            adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE = self._jstep_adj_jit(t+it*self.dt, 
                                                                            adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE,
                                                                            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)

        if 'He_mean' in self.name_params:
            adState.params['He_mean'] = adHe_mean
        adalpha = 0.
        if 'alpha_He' in self.name_params:
            adState.params['alpha_He'] = adalpha_He
        elif 'alpha' in self.name_params:
            adalpha += adalpha_He
        if 'alpha_Uu' in self.name_params:
            adState.params['alpha_Uu'] = adalpha_Uu
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Uu
        if 'alpha_Up' in self.name_params:
            adState.params['alpha_Up'] = adalpha_Up
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Up
        if 'alpha_Uz' in self.name_params:
            adState.params['alpha_Uz'] = adalpha_Uz
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Uz
        if 'alpha' in self.name_params:
            adState.params['alpha'] = adalpha
        if 'hbc' in self.name_params:
            adState.params['hbcx'] = adh_SN
            adState.params['hbcy'] = adh_WE
        
        # Update state and parameters
        adState.setvar(adu,self.name_var['U'])
        adState.setvar(adv,self.name_var['V'])
        adState.setvar(adh,self.name_var['SSH'])

    def _jstep_adj(self, t, 
                  adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE,
                  u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm):

        def wrapped_jstep(x):
            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE = x
            return self._jstep(t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)
        primals = ((u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE),)
        cotangents = (adu, adv, adh)  

        _, vjp_fn = jax.vjp(wrapped_jstep, *primals)
        adjoints = vjp_fn(cotangents)

        adu, adv, adh, _adHe_mean, _adalpha_He, _adalpha_Uu, _adalpha_Up, _adalpha_Uz, _adhSN, _adhWE = adjoints[0]

        if adHe_mean is not None:
            adHe_mean += _adHe_mean
        if adalpha_He is not None:
            adalpha_He += _adalpha_He
        if adalpha_Uu is not None:
            adalpha_Uu += _adalpha_Uu
        if adalpha_Up is not None:
            adalpha_Up += _adalpha_Up
        if adalpha_Uz is not None:
            adalpha_Uz += _adalpha_Uz
        if adh_SN is not None:
            adh_SN += _adhSN
        if adh_WE is not None:
            adh_WE += _adhWE

        return adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE


###############################################################################
#                             QG-SW Models                                    #
###############################################################################

class Model_qgsw(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qgsw'))
        else:
            dir_model = config.MOD.dir_model  
        SourceFileLoader("tools", dir_model+"/tools.py").load_module() 
        SourceFileLoader("helmholtz", dir_model+"/helmholtz.py").load_module() 
        SourceFileLoader("helmholtz_multigrid", dir_model+"/helmholtz_multigrid.py").load_module() 
        SourceFileLoader("masks", dir_model+"/masks.py").load_module() 
        SourceFileLoader("flux", dir_model+"/flux.py").load_module()
        SourceFileLoader("finite_diff", dir_model+"/finite_diff.py").load_module()
        SourceFileLoader("reconstruction", dir_model+"/reconstruction.py").load_module() 
        if config.MOD.name_class.lower()=='qg':
            SourceFileLoader('sw',f'{dir_model}/sw.py').load_module() 
        sw = SourceFileLoader('sw',f'{dir_model}/sw.py').load_module() 
        qg = SourceFileLoader('qg',f'{dir_model}/qg.py').load_module() 
        model_sw = getattr(sw, 'SW')
        model_qg = getattr(qg, 'QG')

        if USE_FLOAT64: 
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32

        self.nl = config.MOD.nl

        if config.MOD.name_class.lower()=='qg':
            model = model_qg
        else:
            model = model_sw

        # Coriolis
        if config.MOD.f0 is not None and config.MOD.constant_f:
            self.f = config.MOD.f0 * np.ones((State.ny,State.nx))
        else:
            self.f = State.f
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
        pad = ((1,0),(1,0))
        _f = np.pad(self.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])
        
        # Gravity 
        self.g = 9.81

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_v = 0.5*(self.dy[:,1:] + self.dy[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])
        self.dx_on_u = 0.5*(self.dx[1:,:] + self.dx[:-1,:])
        self.Xh = State.X
        self.Yh = State.Y
        self.Xu, self.Yu = grid.dxdy2xy(self.dx_on_u, self.dy_on_u)
        self.Xv, self.Yv = grid.dxdy2xy(self.dx_on_v, self.dy_on_v)

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            mdt[np.isnan(mdt)] = 0

            mdu, mdv = self.ssh2uv(mdt)

        
            if config.EXP.flag_plot>0:
                fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))

                im1 = ax1.pcolormesh(mdt)
                plt.colorbar(im1, ax=ax1)
                ax1.set_title('MDT')

                im2 = ax2.pcolormesh(mdu)
                plt.colorbar(im2, ax=ax2)
                ax2.set_title('MDU')

                im3 = ax3.pcolormesh(mdv)
                plt.colorbar(im3, ax=ax3)
                ax3.set_title('MDV')

                plt.show()

            self.mdu = jnp.expand_dims(mdu.astype(self.dtype).T, axis=(0,1))
            self.mdv = jnp.expand_dims(mdv.astype(self.dtype).T, axis=(0,1))
            self.mdt = jnp.expand_dims(mdt.astype(self.dtype).T, axis=(0,1))

        else:
            self.mdt = self.mdu = self.mdv = None

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        # Reduced gravity
        if config.MOD.nl==1:
            if config.MOD.g_prime is None:
                g_prime = np.array([[[9.81]]])
            else:
                g_prime = config.MOD.g_prime
                if not hasattr(g_prime,'shape'):
                    g_prime = np.array([[[g_prime]]])
                elif len(g_prime.shape)==2:
                    g_prime = np.expand_dims(g_prime, axis=0)
                elif len(g_prime.shape)==1:
                    g_prime = np.expand_dims(g_prime, axis=(0,1))
                    
            if config.MOD.H is None:
                H = np.expand_dims(self.c.T**2/g_prime[0,0,0], axis=0) 
                H0 = np.array([[[np.nanmean(self.c)**2/g_prime[0,0,0]]]])
                H[np.isnan(H)] = 0.
                if config.EXP.flag_plot>0:
                    plt.figure()
                    plt.pcolormesh(H[0,:,:].T)
                    plt.colorbar()
                    plt.title('Equivalent Height')
                    plt.show()
                if config.MOD.name_class.lower()=='qg':
                    H = H0 # QG needs constant H
            else:
                H = H0 = config.MOD.H
                if not hasattr(H,'shape'):
                    H = H0 = np.array([[[H]]])
                elif len(H.shape)==2:
                    H = H0 = np.expand_dims(H, axis=0)   
                elif len(H.shape)==1:
                    H = H0 = np.expand_dims(H, axis=(0,1))
        else:
            g_prime = config.MOD.g_prime
            if not hasattr(g_prime, 'shape'):
                g_prime = np.array(g_prime)
            if len(g_prime.shape) == 1:
                g_prime = g_prime.reshape(-1, 1, 1)
            H = config.MOD.H
            if not hasattr(H, 'shape'):
                H = np.array(H)
            if len(H.shape) == 1:
                H = H.reshape(-1, 1, 1)
            elif len(H.shape) == 2:
                H = np.expand_dims(H, axis=0)
        
        self.H0 = H
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmin(State.DX), np.nanmin(State.DY)) 
            # For nl>1 the fastest wave is the barotropic mode sqrt(g*sum(H))
            if self.nl > 1:
                H_arr = np.asarray(config.MOD.H)
                g_arr = np.asarray(config.MOD.g_prime)
                c_max = float(np.sqrt(g_arr[0] * H_arr.sum()))
                print(f'Barotropic wave speed c_max = {c_max:.2f} m/s')
            else:
                c_max = float(np.nanmax(self.c))
            dt = config.MOD.cfl * grid_spacing / c_max
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print(f'CFL condition for c= {c_max}')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt
    
        # Initialize model state
        if (config.GRID.super == 'GRID_FROM_FILE'):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if config.MOD.name_init_var is not None and name in config.MOD.name_init_var:
                    name_init_var = config.MOD.name_init_var[name]
                else:
                    name_init_var = config.MOD.name_var[name]
                if name_init_var in dsin:
                    var_init = dsin[name_init_var]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    var_init.data[np.isnan(var_init)] = 0.
                    State.var[self.name_var[name]] = var_init.values
                else:
                    if name=='U':
                        State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx+1), dtype=self.dtype)
                    elif name=='V':
                        State.var[self.name_var[name]] = jnp.zeros((State.ny+1,State.nx), dtype=self.dtype)
                    elif name=='SSH':
                        State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx), dtype=self.dtype)
            dsin.close()
            del dsin   
        else:
            for name in self.name_var:  
                if name=='U':
                    State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx+1), dtype=self.dtype)
                elif name=='V':
                    State.var[self.name_var[name]] = jnp.zeros((State.ny+1,State.nx), dtype=self.dtype)
                elif name=='SSH':
                    State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx), dtype=self.dtype)

        # For nl>1, also initialize per-layer storage in State
        if self.nl > 1:
            State.var['u_layers'] = jnp.zeros((self.nl, State.ny, State.nx+1), dtype=self.dtype)
            State.var['v_layers'] = jnp.zeros((self.nl, State.ny+1, State.nx), dtype=self.dtype)
            State.var['h_layers'] = jnp.zeros((self.nl, State.ny, State.nx), dtype=self.dtype)

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # Sponge layer
        self.sponge_width = config.MOD.dist_sponge_bc  # km
        self.sponge_coef = config.MOD.sponge_coef

        if self.sponge_width is not None and self.sponge_width>0:
            sponge_h = np.maximum.reduce([
                gaspari_cohn(self.Xh.T-self.Xh[:,0].T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Xh[:,-1].T-self.Xh.T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Yh-self.Yh[0,:], self.sponge_width*1e3), 
                gaspari_cohn(self.Yh[-1,:]-self.Yh, self.sponge_width*1e3)
                ])
            sponge_u = np.maximum.reduce([
                gaspari_cohn(self.Xu.T-self.Xu[:,0].T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Xu[:,-1].T-self.Xu.T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Yu-self.Yu[0,:], self.sponge_width*1e3), 
                gaspari_cohn(self.Yu[-1,:]-self.Yu, self.sponge_width*1e3)
                ])
            sponge_v = np.maximum.reduce([
                gaspari_cohn(self.Xv.T-self.Xv[:,0].T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Xv[:,-1].T-self.Xv.T, self.sponge_width*1e3).T, 
                gaspari_cohn(self.Yv-self.Yv[0,:], self.sponge_width*1e3), 
                gaspari_cohn(self.Yv[-1,:]-self.Yv, self.sponge_width*1e3)
                ]) 
        else:
            sponge_h = np.zeros((State.ny,State.nx))
            sponge_u = np.zeros((State.ny,State.nx+1))
            sponge_v = np.zeros((State.ny+1,State.nx))

        def _normalize_sponge(sponge):
            sponge_min = jnp.min(sponge)
            sponge_max = jnp.max(sponge)
            return (sponge - sponge_min) / (sponge_max - sponge_min)

        if True:
            mask_h = State.mask.copy()
            lon_h = State.lon
            lat_h = State.lat
            mask_u = np.zeros((State.ny, State.nx+1), dtype=bool)
            mask_u[:, 1:State.nx] = mask_h[:, :-1] & mask_h[:, 1:]
            mask_v = np.zeros((State.ny+1, State.nx), dtype=bool)
            mask_v[1:State.ny, :] = mask_h[:-1, :] & mask_h[1:, :]
            lon_u = np.zeros((State.ny, State.nx+1))
            lat_u = np.zeros((State.ny, State.nx+1))
            lon_u[:, 1:State.nx] = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u[:, 1:State.nx] = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_v = np.zeros((State.ny+1, State.nx))
            lat_v = np.zeros((State.ny+1, State.nx))
            lon_v[1:State.ny, :] = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v[1:State.ny, :] = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])
            # West boundary
            lon_u[:, 0] = lon_h[:, 0] - 0.5 * (lon_h[:, 1] - lon_h[:, 0])
            lat_u[:, 0] = lat_h[:, 0] - 0.5 * (lat_h[:, 1] - lat_h[:, 0])
            # East boundary
            lon_u[:, State.nx] = lon_h[:, -1] + 0.5 * (lon_h[:, -1] - lon_h[:, -2])
            lat_u[:, State.nx] = lat_h[:, -1] + 0.5 * (lat_h[:, -1] - lat_h[:, -2])
            # South boundary
            lon_v[0, :] = lon_h[0, :] - 0.5 * (lon_h[1, :] - lon_h[0, :])
            lat_v[0, :] = lat_h[0, :] - 0.5 * (lat_h[1, :] - lat_h[0, :])
            # North boundary
            lon_v[State.ny, :] = lon_h[-1, :] + 0.5 * (lon_h[-1, :] - lon_h[-2, :])
            lat_v[State.ny, :] = lat_h[-1, :] + 0.5 * (lat_h[-1, :] - lat_h[-2, :])

            sponge_h = grid.compute_weight_map(lon_h, lat_h, mask_h, config.MOD.dist_sponge_bc)
            sponge_u = grid.compute_weight_map(lon_u, lat_u, mask_u, config.MOD.dist_sponge_bc)
            sponge_v = grid.compute_weight_map(lon_v, lat_v, mask_v, config.MOD.dist_sponge_bc)

            
            fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
            im1 = ax1.pcolormesh(sponge_h)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('Sponge SSH')
            im2 = ax2.pcolormesh(sponge_u)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('Sponge U')
            im3 = ax3.pcolormesh(sponge_v)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title('Sponge V')
            plt.show()

        self.sponge_u = sponge_u
        self.sponge_v = sponge_v
        self.sponge_h = sponge_h    


        self.sponge_h = _normalize_sponge(sponge_h)
        self.sponge_u = _normalize_sponge(sponge_u)
        self.sponge_v = _normalize_sponge(sponge_v)


    
        # Model initialization
        params = {
            "nx": State.nx,
            "ny": State.ny,
            "nl": config.MOD.nl,
            "dx": State.dx,
            "dy": State.dy,
            "H": H,
            "g_prime": g_prime,
            "f": np.pad(self.f.T, ((0, 1), (0, 1)), mode='edge'),
            "taux": 0.,
            "tauy": 0.,
            "bottom_drag_coef": config.MOD.bottom_drag_coef,
            "rho_water": getattr(config.MOD, 'rho_water', 1025.0),
            "h_wind": getattr(config.MOD, 'h_wind', None),
            "dtype": self.dtype,
            "mask": (1-State.mask.astype(int).T),
            "compile": True,
            "slip_coef": config.MOD.slip_coef,
            "visc_coef": config.MOD.visc_coef,
            "diff_coef": config.MOD.visc_coef,
            "dt": self.dt,
            "barotropic_filter": False,
            'barotropic_filter_spectral': False,
            'sponge_coef': self.sponge_coef,
            'forcing_momentum': getattr(config.MOD, 'forcing_momentum', 'direct'),
        }

        self.model = model(params)

        # Sponge masks: (1, 1, nx, ny) — broadcasts across all layers.
        # Layer 0 is nudged toward surface BC; deep layers toward zero (their BC is 0).
        self.model.sponge_u = jnp.expand_dims(jnp.asarray(self.sponge_u.T.astype(self.dtype)), axis=(0,1))
        self.model.sponge_v = jnp.expand_dims(jnp.asarray(self.sponge_v.T.astype(self.dtype)), axis=(0,1))
        self.model.sponge_h = jnp.expand_dims(jnp.asarray(self.sponge_h.T.astype(self.dtype)), axis=(0,1))

        # Maximum number of model steps per JIT call (limits GPU memory)
        self.max_nstep = getattr(config.MOD, 'max_nstep', 240)

        # Model functions initialization
        self.model_step = self.model.step
        self.model_step_tgl = self.model.step_tgl
        self.model_step_adj = self.model.step_adj
        
        flag_compile = True
        if flag_compile:
            self.jstep_jit = jax.jit(self.jstep, static_argnames=['nstep'])
            self.jstep_tgl_jit = jax.jit(self.jstep_tgl, static_argnames=['nstep'])
            self.jstep_adj_jit = jax.jit(self.jstep_adj, static_argnames=['nstep'])
            self.jstep_core_jit = jax.jit(self.jstep_core, static_argnames=['nstep'])
        else:
            self.jstep_jit = self.jstep
            self.jstep_tgl_jit = self.jstep_tgl
            self.jstep_adj_jit = self.jstep_adj
            self.jstep_core_jit = self.jstep_core


        # Control parameters
        self.name_params = config.MOD.name_params if config.MOD.name_params is not None else []
        if 'H' in self.name_params:
            if (config.GRID.super == 'GRID_FROM_FILE'):
                dsin = xr.open_dataset(config.GRID.path_init_grid)
                if 'H' in dsin:
                    State.params['H'] = dsin['H'].values.squeeze()
                    State.params['H'][np.isnan(State.params['H'])] = 0.
                else:
                    State.params['H'] = np.zeros((State.ny,State.nx))
                dsin.close()
                del dsin
            else:
                State.params['H'] = np.zeros((State.ny,State.nx))
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros_like(State.var[self.name_var[name]])

        # Wind forcing (must be loaded after self.timestamps and self.dt are set)
        self._load_wind_forcing(config, State)

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            #tangent_test(self,State,nstep=10)#, ampl=1e-3)
            print('Adjoint test:')
            #self.adjoint_test_jstep(nstep=10)
            adjoint_test(self,State,nstep=10)#, ampl=1e-3)
    
    def _load_wind_forcing(self, config, State):
        """
        Read a NetCDF wind file, convert (u10, v10) to wind stress using the
        bulk formula  tau = rho_air * Cd * |U10| * U10,  and precompute
        taux / tauy at every model timestamp on the u- and v-grids.

        Config keys used (all optional):
          config.MOD.path_wind       – path to the .nc file (None → no wind)
          config.MOD.name_var_wind   – dict with keys 'lon','lat','time','u10','v10'
                                       (defaults to ERA5 conventions)
          config.MOD.rho_air         – air density in kg/m³  (default 1.25)
          config.MOD.Cd_wind         – drag coefficient      (default 1.5e-3)

        Stores:
          self.taux_wind  – ndarray (nt, ny, nx+1)  wind stress on u-grid
          self.tauy_wind  – ndarray (nt, ny+1, nx)  wind stress on v-grid
        """
        self.taux_wind = None
        self.tauy_wind = None

        path_wind = getattr(config.MOD, 'path_wind', None)
        if path_wind is None or not os.path.exists(path_wind):
            return

        rho_air = getattr(config.MOD, 'rho_air', 1.25)
        Cd      = getattr(config.MOD, 'Cd_wind', 1.5e-3)
        nv = getattr(config.MOD, 'name_var_wind',
                     {'lon': 'longitude', 'lat': 'latitude', 'time': 'time',
                      'u10': 'u10', 'v10': 'v10'})

        ds = xr.open_dataset(path_wind)

        # --- longitude convention ---
        lon_vals = ds[nv['lon']].values
        if lon_vals.min() < 0 and State.lon_unit == '0_360':
            ds = ds.assign_coords({nv['lon']: (nv['lon'], lon_vals % 360)})
        elif lon_vals.min() >= 0 and State.lon_unit == '-180_180':
            ds = ds.assign_coords({nv['lon']: (nv['lon'], (lon_vals + 180) % 360 - 180)})
        ds = ds.sortby(nv['lon'])

        # --- compute wind stress on the wind file's native grid ---
        u10 = ds[nv['u10']].values   # (ntime, nlat, nlon)
        v10 = ds[nv['v10']].values
        wspd   = np.sqrt(u10**2 + v10**2)
        taux_w = rho_air * Cd * wspd * u10
        tauy_w = rho_air * Cd * wspd * v10

        wind_lons  = ds[nv['lon']].values
        wind_lats  = ds[nv['lat']].values
        wind_times = ds[nv['time']].values.astype('datetime64[s]')
        ds.close()

        # --- wind update cadence ---
        wind_dt = getattr(config.MOD, 'wind_timestep', 3600)  # seconds
        wind_dt = max(wind_dt, self.dt)  # at least one model step
        total_seconds = (config.EXP.final_date - config.EXP.init_date).total_seconds()
        nt_wind = 1 + int(total_seconds // wind_dt)
        self.wind_dt = wind_dt

        # --- spatial interpolation to model h-grid ---
        from scipy.interpolate import RegularGridInterpolator
        taux_hgrid = np.zeros((nt_wind, State.ny, State.nx))
        tauy_hgrid = np.zeros((nt_wind, State.ny, State.nx))

        # wind sample timestamps as datetime64[s]
        wind_sample_times = np.array(
            [config.EXP.init_date + timedelta(seconds=i * wind_dt) for i in range(nt_wind)],
            dtype='datetime64[s]')

        # ensure lats are monotonically increasing for RegularGridInterpolator
        if wind_lats[0] > wind_lats[-1]:
            wind_lats  = wind_lats[::-1]
            taux_w     = taux_w[:, ::-1, :]
            tauy_w     = tauy_w[:, ::-1, :]

        for it in range(nt_wind):
            # --- temporal linear interpolation ---
            t_model = wind_sample_times[it]
            idx_r = int(np.searchsorted(wind_times, t_model))
            idx_r = np.clip(idx_r, 1, len(wind_times) - 1)
            idx_l = idx_r - 1

            dt_wind   = float((wind_times[idx_r] - wind_times[idx_l]).astype('float64'))
            dt_interp = float((t_model          - wind_times[idx_l]).astype('float64'))
            alpha = (dt_interp / dt_wind) if dt_wind > 0 else 0.
            alpha = np.clip(alpha, 0., 1.)

            taux_t = (1 - alpha) * taux_w[idx_l] + alpha * taux_w[idx_r]
            tauy_t = (1 - alpha) * tauy_w[idx_l] + alpha * tauy_w[idx_r]

            # --- spatial interpolation ---
            pts = np.stack([State.lat.ravel(), State.lon.ravel()], axis=-1)

            fi_taux = RegularGridInterpolator(
                (wind_lats, wind_lons), taux_t,
                method='linear', bounds_error=False, fill_value=0.)
            fi_tauy = RegularGridInterpolator(
                (wind_lats, wind_lons), tauy_t,
                method='linear', bounds_error=False, fill_value=0.)

            taux_hgrid[it] = fi_taux(pts).reshape(State.ny, State.nx)
            tauy_hgrid[it] = fi_tauy(pts).reshape(State.ny, State.nx)

        # --- interpolate from h-grid to u-grid (ny, nx+1) and v-grid (ny+1, nx) ---
        taux_u = np.zeros((nt_wind, State.ny,     State.nx + 1))
        tauy_v = np.zeros((nt_wind, State.ny + 1, State.nx    ))

        taux_u[:, :, 1:State.nx] = 0.5 * (taux_hgrid[:, :, :-1] + taux_hgrid[:, :, 1:])
        taux_u[:, :, 0]          = taux_hgrid[:, :,  0]
        taux_u[:, :, State.nx]   = taux_hgrid[:, :, -1]

        tauy_v[:, 1:State.ny, :] = 0.5 * (tauy_hgrid[:, :-1, :] + tauy_hgrid[:, 1:, :])
        tauy_v[:, 0,          :] = tauy_hgrid[:,  0, :]
        tauy_v[:, State.ny,   :] = tauy_hgrid[:, -1, :]

        self.taux_wind = taux_u   # (nt_wind, ny, nx+1)
        self.tauy_wind = tauy_v   # (nt_wind, ny+1, nx)
        print(f'  - Wind forcing loaded: {path_wind}  ({nt_wind} wind snapshots, wind_timestep={wind_dt}s)')
        print(f'    max |taux|={np.nanmax(np.abs(taux_u)):.3e}  '
              f'max |tauy|={np.nanmax(np.abs(tauy_v)):.3e}')

    def _get_wind_stress(self, t):
        """
        Return (taux, tauy) as jnp arrays for time t (seconds from init).
        Returns (None, None) when no wind file was loaded.
        """
        if self.taux_wind is None:
            return None, None
        it = int(round(t / self.wind_dt)) if self.wind_dt > 0 else 0
        it = int(np.clip(it, 0, len(self.taux_wind) - 1))
        taux = jnp.asarray(self.taux_wind[it], dtype=self.dtype)  # (ny, nx+1)
        tauy = jnp.asarray(self.tauy_wind[it], dtype=self.dtype)  # (ny+1, nx)
        return taux, tauy

    def _sync_layers_from_surface(self, State):
        """Project 2D surface fields into per-layer arrays (layer 0 gets the perturbation)."""
        ssh = jnp.asarray(State.getvar(name_var=self.name_var['SSH']), dtype=self.dtype)
        u   = jnp.asarray(State.getvar(name_var=self.name_var['U']),   dtype=self.dtype)
        v   = jnp.asarray(State.getvar(name_var=self.name_var['V']),   dtype=self.dtype)
        State.var['h_layers'] = jnp.zeros((self.nl, self.ny, self.nx), dtype=self.dtype).at[0].set(ssh)
        State.var['u_layers'] = jnp.zeros((self.nl, self.ny, self.nx+1), dtype=self.dtype).at[0].set(u)
        State.var['v_layers'] = jnp.zeros((self.nl, self.ny+1, self.nx), dtype=self.dtype).at[0].set(v)

    def _diagnose_surface(self, State):
        """Set 2D surface SSH/U/V from per-layer arrays."""
        State.setvar(State.var['h_layers'].sum(axis=0), name_var=self.name_var['SSH'])
        State.setvar(State.var['u_layers'][0],          name_var=self.name_var['U'])
        State.setvar(State.var['v_layers'][0],          name_var=self.name_var['V'])

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            if 'SSH' in self.init_from_bc:
                u0 = self.bc['U'][t0]
                v0 = self.bc['V'][t0]
                ssh0 = self.bc['SSH'][t0]
                State.setvar(u0, self.name_var['U'])
                State.setvar(v0, self.name_var['V'])
                State.setvar(ssh0, self.name_var['SSH'])
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])

        # For nl>1, project surface fields into per-layer arrays
        if self.nl > 1:
            self._sync_layers_from_surface(State)

    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()

        _name_var = [self.name_var['U'], self.name_var['V'], self.name_var['SSH']] if name_var is None else name_var

        if 'H' in self.name_params:
            State0.var['H'] = +State.params['H']
            _name_var += ['H']
    
        State0.save_output(present_date, name_var=_name_var)
    
    def set_bc(self,time_bc,var_bc):

        for i,t in enumerate(time_bc):
            ssh_bc_t = +var_bc['SSH'][i]
            # Remove nan
            ssh_bc_t[np.isnan(ssh_bc_t)] = 0.

            if 'U' not in var_bc or 'V' not in var_bc:
                # Compute (u,v) from ssh (geostrophy)
                u, v = self.ssh2uv(ssh_bc_t)
                # TESTTING
                u *= 0
                v *= 0
            else:
                u = +var_bc['U'][i]
                v = +var_bc['V'][i]
                # Remove nan
                u[np.isnan(u)] = 0.
                v[np.isnan(v)] = 0.

            # Fill bc dictionnary
            self.bc['U'][t] = u
            self.bc['V'][t] = v
            self.bc['SSH'][t] = ssh_bc_t

        self.bc_time = np.asarray(time_bc)

    def _apply_bc(self,t0,t1):
        
        u_b = jnp.zeros((self.ny,self.nx+1))
        v_b = jnp.zeros((self.ny+1,self.nx))
        ssh_b = jnp.zeros((self.ny,self.nx))

        if 'SSH' not in self.bc:
            return u_b,v_b,ssh_b
        elif len(self.bc['SSH'].keys())==0:
             return u_b,v_b,ssh_b
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        u_b = self.bc['U'][t0]
        v_b = self.bc['V'][t0]
        ssh_b = self.bc['SSH'][t0]

        return u_b, v_b, ssh_b
    
    def ssh2uv(self, ssh):
        """Nonlinear SSH → (u, v) model."""

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * np.diff(_ssh, axis=0) / self.dy_on_u
        _v = self.g / self.f_on_v * np.diff(_ssh, axis=1) / self.dx_on_v

        return _u, _v

    def ssh2uv_tgl(self, ssh, dssh):
        """Tangent-linear model using JAX forward-mode differentiation."""
        _, (dug, dvg) = jax.jvp(
            lambda s: self.ssh2uv(s),
            (ssh,),
            (dssh,)
        )
        return dug, dvg

    def ssh2uv_adj(self, ssh, adug, advg):
        """Adjoint model using JAX reverse-mode differentiation."""
        _, vjp_fun = jax.vjp(lambda s: self.ssh2uv(s), ssh)
        (adssh,) = vjp_fun((adug, advg))
        return adssh
    
    def h_on_u(self, h, pad=((1,0),(1,0))):
        """Interpolate h to u points."""
        _h = jnp.pad(h, pad_width=pad, mode='edge')
        h_on_u = 0.5*(_h[1:,:] + _h[:-1,:])
        return h_on_u
    
    def h_on_v(self, h, pad=((1,0),(1,0))):
        """Interpolate h to v points."""
        _h = jnp.pad(h, pad_width=pad, mode='edge')
        h_on_v = 0.5*(_h[:,1:] + _h[:,:-1])
        return h_on_v
    
    def jstep_core(self, t, u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
                taux=None, tauy=None, nstep=1):
        """
        taux: wind stress on u-grid, shape (ny, nx+1) in State convention, or None.
        tauy: wind stress on v-grid, shape (ny+1, nx) in State convention, or None.
        They are converted to SW model convention (nx-1, ny) and (nx, ny-1) internally.
        Fu/Fv/Fh: 2D forcing in State convention — converted to per-layer for nl>1.
        u_b/v_b/h_b: 2D BCs in State convention — converted to per-layer for nl>1.
        """
        u, v, h = u0, v0, h0

        # Add MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h = h.at[:, 0:1, :, :].add(self.mdt)
                u = u.at[:, 0:1, :, :].add(self.mdu)
                v = v.at[:, 0:1, :, :].add(self.mdv)
            else:
                h = h + self.mdt
                u = u + self.mdu
                v = v + self.mdv

        # Convert wind stress from State (ny, nx+1/ny+1) to SW model (nx-1/nx, ny/ny-1)
        # SW model expects taux on interior u-faces: (nx-1, ny)
        #                  tauy on interior v-faces: (nx, ny-1)
        _taux = taux[:, 1:-1].T if taux is not None else None   # (nx-1, ny)
        _tauy = tauy[1:-1, :].T if tauy is not None else None   # (nx, ny-1)

        # Convert BCs and forcing from State convention to SW convention
        if self.nl > 1:
            # Per-layer: put 2D BC/forcing in layer 0, zeros in other layers
            u_b_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(u_b.T)
            v_b_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(v_b.T)
            h_b_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(h_b.T)
            Fu_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(Fu.T) / (3600 * 24)
            Fv_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(Fv.T) / (3600 * 24)
            Fh_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(Fh.T) / (3600 * 24)
        else:
            u_b_sw = u_b.T
            v_b_sw = v_b.T
            h_b_sw = h_b.T
            Fu_sw = Fu.T / (3600 * 24)
            Fv_sw = Fv.T / (3600 * 24)
            Fh_sw = Fh.T / (3600 * 24)

        # Step
        u1, v1, h1 = self.model_step(
                u, v, h, H=H, nstep=nstep,
                u_b=u_b_sw, v_b=v_b_sw, h_b=h_b_sw,
                Fu=Fu_sw, Fv=Fv_sw, Fh=Fh_sw,
                taux=_taux, tauy=_tauy,
            )

        # Remove MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h1 = h1.at[:, 0:1, :, :].add(-self.mdt)
                u1 = u1.at[:, 0:1, :, :].add(-self.mdu)
                v1 = v1.at[:, 0:1, :, :].add(-self.mdv)
            else:
                h1 = h1 - self.mdt
                u1 = u1 - self.mdu
                v1 = v1 - self.mdv

        return u1, v1, h1
    
    def jstep(self, t, u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
             taux=None, tauy=None, nstep=1):
        return self.jstep_core(
            t, u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b,
            taux=taux, tauy=tauy,
            nstep=nstep,
        )
    
    def jstep_tgl(self, t, du0, dv0, dh0, dH, dFu, dFv, dFh, 
                u0, v0, h0, H, Fu, Fv, Fh, 
                u_b, v_b, h_b, taux=None, tauy=None, nstep=1):

        # Define a partial function that fixes constant parameters
        # taux/tauy are prescribed forcings: closed over, not differentiated
        f = lambda u, v, h, H, Fu, Fv, Fh: self.jstep_core_jit(
            t, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
            taux=taux, tauy=tauy,
            nstep=nstep,
        )

        # JVP (forward mode)
        (u1, v1, h1), (du1, dv1, dh1) = jax.jvp(
            f,
            (u0, v0, h0, H, Fu, Fv, Fh),
            (du0, dv0, dh0, dH, dFu, dFv, dFh)
        )

        return du1, dv1, dh1
    
    def jstep_adj(self, t, adu1, adv1, adh1, adH, adFu, adFv, adFh,
                u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
                taux=None, tauy=None, nstep=1):

        # taux/tauy are prescribed forcings: closed over, not differentiated
        f = lambda u, v, h, H, Fu, Fv, Fh: self.jstep_core_jit(
            t, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
            taux=taux, tauy=tauy,
            nstep=nstep,
        )

        # Build the VJP function
        (u1, v1, h1), vjp_fun = jax.vjp(f, u0, v0, h0, H, Fu, Fv, Fh)

        # Apply adjoints at output
        adu0, adv0, adh0, _adH, _adFu, _adFv, _adFh = vjp_fun((adu1, adv1, adh1))
        if adH is not None:
            adH += _adH
        adFu += _adFu
        adFv += _adFv
        adFh += _adFh
        
        return adu0, adv0, adh0, adH, adFu, adFv, adFh

    def step(self,State,nstep=1,t=0):

        if self.nl > 1:
            # Read per-layer state and convert to SW convention (1, nl, nx, ny)
            u = jnp.expand_dims(State.var['u_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(State.var['v_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(State.var['h_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            h = State.getvar(name_var=self.name_var['SSH'])
            u = State.getvar(name_var=self.name_var['U'])
            v = State.getvar(name_var=self.name_var['V'])
            u = jnp.expand_dims(u.astype(self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(v.astype(self.dtype).T, axis=(0,1))
            h = jnp.expand_dims(h.astype(self.dtype).T, axis=(0,1))
        
        # Get parameters (2D forcing from Basis)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            H = jnp.expand_dims(H.astype(self.dtype).T, axis=0)
        else:
            H = None

        # Sub-stepping loop (limits lax.scan length for memory efficiency)
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_b, v_b, h_b = self._apply_bc(t_chunk, int(t_chunk + n_chunk * self.dt))
            taux, tauy = self._get_wind_stress(t_chunk)
            u, v, h = self.jstep_jit(t_chunk, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                      taux=taux, tauy=tauy, nstep=n_chunk)
            step_done += n_chunk

        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            State.var['h_layers'] = h[0].transpose(0, 2, 1)
            State.var['u_layers'] = u[0].transpose(0, 2, 1)
            State.var['v_layers'] = v[0].transpose(0, 2, 1)
            # Diagnose 2D surface fields
            self._diagnose_surface(State)
        else:
            u = u[0,0].T
            v = v[0,0].T
            h = h[0,0].T
            State.setvar(u, name_var=self.name_var['U'])
            State.setvar(v, name_var=self.name_var['V'])
            State.setvar(h, name_var=self.name_var['SSH'])   

    def step_tgl(self,dState,State,nstep=1,t=0):

        if self.nl > 1:
            # Per-layer perturbation
            du = jnp.expand_dims(dState.var['u_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            dv = jnp.expand_dims(dState.var['v_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            dh = jnp.expand_dims(dState.var['h_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            # Per-layer forward trajectory
            u = jnp.expand_dims(State.var['u_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(State.var['v_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(State.var['h_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            dh = dState.getvar(name_var=self.name_var['SSH'])
            du = dState.getvar(name_var=self.name_var['U'])
            dv = dState.getvar(name_var=self.name_var['V'])
            h = State.getvar(name_var=self.name_var['SSH'])
            u = State.getvar(name_var=self.name_var['U'])
            v = State.getvar(name_var=self.name_var['V'])
            du = jnp.expand_dims(du.astype(self.dtype).T, axis=(0,1))
            dv = jnp.expand_dims(dv.astype(self.dtype).T, axis=(0,1))
            dh = jnp.expand_dims(dh.astype(self.dtype).T, axis=(0,1))
            u = jnp.expand_dims(u.astype(self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(v.astype(self.dtype).T, axis=(0,1))
            h = jnp.expand_dims(h.astype(self.dtype).T, axis=(0,1))

        # Get parameters (2D forcing — per-layer conversion in jstep_core via AD)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        dFu = dState.params[self.name_var['U']]
        dFv = dState.params[self.name_var['V']]
        dFh = dState.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            dH = dState.params['H']
            H = jnp.expand_dims(H.astype(self.dtype ).T, axis=0)
            dH = jnp.expand_dims(dH.astype(self.dtype).T, axis=0)
        else:
            H = dH = None
        
        # Sub-stepping loop (limits lax.scan length for memory efficiency)
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_b, v_b, h_b = self._apply_bc(t_chunk, int(t_chunk + n_chunk * self.dt))
            taux, tauy = self._get_wind_stress(t_chunk)
            # Propagate tangent (JVP includes forward internally)
            du, dv, dh = self.jstep_tgl_jit(t_chunk, du, dv, dh, dH, dFu, dFv, dFh,
                                            u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                            taux=taux, tauy=tauy, nstep=n_chunk)
            # Propagate forward state for next chunk
            u, v, h = self.jstep_jit(t_chunk, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                      taux=taux, tauy=tauy, nstep=n_chunk)
            step_done += n_chunk

        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            dState.var['h_layers'] = dh[0].transpose(0, 2, 1)
            dState.var['u_layers'] = du[0].transpose(0, 2, 1)
            dState.var['v_layers'] = dv[0].transpose(0, 2, 1)
            # Diagnose surface TLM
            dState.setvar(dState.var['h_layers'].sum(axis=0), name_var=self.name_var['SSH'])
            dState.setvar(dState.var['u_layers'][0],          name_var=self.name_var['U'])
            dState.setvar(dState.var['v_layers'][0],          name_var=self.name_var['V'])
        else:
            du = du[0,0].T
            dv = dv[0,0].T
            dh = dh[0,0].T
            dState.setvar(du, name_var=self.name_var['U'])
            dState.setvar(dv, name_var=self.name_var['V'])
            dState.setvar(dh, name_var=self.name_var['SSH'])

    def step_adj(self,adState,State,nstep=1,t=0):

        if self.nl > 1:
            # --- Combine surface adjoint (from obs) with per-layer adjoint ---
            # Adjoint of ssh = h_layers.sum(axis=0): broadcast ad_ssh to all layers
            adh_surface = jnp.asarray(adState.getvar(name_var=self.name_var['SSH']), dtype=self.dtype)
            adu_surface = jnp.asarray(adState.getvar(name_var=self.name_var['U']),   dtype=self.dtype)
            adv_surface = jnp.asarray(adState.getvar(name_var=self.name_var['V']),   dtype=self.dtype)

            adh_layers = jnp.asarray(adState.var.get('h_layers',
                            jnp.zeros((self.nl, self.ny, self.nx), dtype=self.dtype)), dtype=self.dtype)
            adu_layers = jnp.asarray(adState.var.get('u_layers',
                            jnp.zeros((self.nl, self.ny, self.nx+1), dtype=self.dtype)), dtype=self.dtype)
            adv_layers = jnp.asarray(adState.var.get('v_layers',
                            jnp.zeros((self.nl, self.ny+1, self.nx), dtype=self.dtype)), dtype=self.dtype)

            # Adjoint of sum: each layer receives the surface SSH adjoint
            adh_layers = adh_layers + adh_surface[None, :, :]
            # Adjoint of selecting layer 0 for surface u/v
            adu_layers = adu_layers.at[0].add(adu_surface)
            adv_layers = adv_layers.at[0].add(adv_surface)

            # Convert to SW convention (1, nl, nx, ny)
            adu = jnp.expand_dims(adu_layers.transpose(0, 2, 1), axis=0)
            adv = jnp.expand_dims(adv_layers.transpose(0, 2, 1), axis=0)
            adh = jnp.expand_dims(adh_layers.transpose(0, 2, 1), axis=0)

            # Forward trajectory from State (per-layer)
            u = jnp.expand_dims(State.var['u_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(State.var['v_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(State.var['h_layers'].astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            adh0 = adState.getvar(name_var=self.name_var['SSH'])
            adu0 = adState.getvar(name_var=self.name_var['U'])
            adv0 = adState.getvar(name_var=self.name_var['V'])
            h = State.getvar(name_var=self.name_var['SSH'])
            u = State.getvar(name_var=self.name_var['U'])
            v = State.getvar(name_var=self.name_var['V'])
            adu = jnp.expand_dims((+adu0).astype(self.dtype).T, axis=(0,1))
            adv = jnp.expand_dims((+adv0).astype(self.dtype).T, axis=(0,1))
            adh = jnp.expand_dims((+adh0).astype(self.dtype).T, axis=(0,1))
            u = jnp.expand_dims(u.astype(self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(v.astype(self.dtype).T, axis=(0,1))
            h = jnp.expand_dims(h.astype(self.dtype).T, axis=(0,1))

        # Get parameters (2D forcing)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            H = jnp.expand_dims(H.astype(self.dtype).T, axis=0)
        else:
            H = None
        
        # Get adjoint parameters (2D — AD traces per-layer conversion automatically)
        adFu = adState.params[self.name_var['U']]
        adFv = adState.params[self.name_var['V']]
        adFh = adState.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            adH = adState.params['H']
            adH = jnp.expand_dims(adH.astype(self.dtype).T, axis=0)
        else:
            adH = None

        # Build chunk schedule
        chunks = []
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            chunks.append((t_chunk, n_chunk))
            step_done += n_chunk

        # Forward pass: store boundary states at chunk boundaries
        fwd_states = [(u, v, h)]
        if len(chunks) > 1:
            u_fwd, v_fwd, h_fwd = u, v, h
            for t_chunk, n_chunk in chunks[:-1]:
                u_b, v_b, h_b = self._apply_bc(t_chunk, int(t_chunk + n_chunk * self.dt))
                taux, tauy = self._get_wind_stress(t_chunk)
                u_fwd, v_fwd, h_fwd = self.jstep_jit(
                    t_chunk, u_fwd, v_fwd, h_fwd, H, Fu, Fv, Fh, u_b, v_b, h_b,
                    taux=taux, tauy=tauy, nstep=n_chunk)
                fwd_states.append((u_fwd, v_fwd, h_fwd))

        # Reverse adjoint through chunks
        for i in range(len(chunks) - 1, -1, -1):
            t_chunk, n_chunk = chunks[i]
            u_i, v_i, h_i = fwd_states[i]
            u_b, v_b, h_b = self._apply_bc(t_chunk, int(t_chunk + n_chunk * self.dt))
            taux, tauy = self._get_wind_stress(t_chunk)
            adu, adv, adh, adH, adFu, adFv, adFh = self.jstep_adj_jit(
                t_chunk, adu, adv, adh, adH, adFu, adFv, adFh,
                u_i, v_i, h_i, H, Fu, Fv, Fh,
                u_b, v_b, h_b,
                taux=taux, tauy=tauy,
                nstep=n_chunk)

        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            adh_layers = adh[0].transpose(0, 2, 1)
            adu_layers = adu[0].transpose(0, 2, 1)
            adv_layers = adv[0].transpose(0, 2, 1)
            # Store per-layer adjoints
            adState.var['h_layers'] = adh_layers
            adState.var['u_layers'] = adu_layers
            adState.var['v_layers'] = adv_layers
            # Reset surface adjoints (consumed — distributed to layers above)
            adState.setvar(jnp.zeros((self.ny, self.nx),   dtype=self.dtype), self.name_var['SSH'])
            adState.setvar(jnp.zeros((self.ny, self.nx+1), dtype=self.dtype), self.name_var['U'])
            adState.setvar(jnp.zeros((self.ny+1, self.nx), dtype=self.dtype), self.name_var['V'])
        else:
            adu = adu[0,0].T
            adv = adv[0,0].T
            adh = adh[0,0].T
            adState.setvar(adu,self.name_var['U'])
            adState.setvar(adv,self.name_var['V'])
            adState.setvar(adh,self.name_var['SSH'])

        if 'H' in self.name_params:
            adH = adH[0].T
            adState.params['H'] = adH

        # Update adjoint parameters (2D, same for both nl=1 and nl>1)
        adState.params[self.name_var['U']] = adFu 
        adState.params[self.name_var['V']] = adFv 
        adState.params[self.name_var['SSH']] = adFh 

    def adjoint_test_jstep(self, nstep=1, seed=42):
        """
        Low-level adjoint test for jstep_tgl / jstep_adj.

        Checks  <M dx, y> == <dx, M* y>
        where M = jstep_tgl and M* = jstep_adj, operating on the
        differentiated variables (u, v, h, H, Fu, Fv, Fh).

        Boundary conditions (u_b, v_b, h_b) and wind stress are NOT
        differentiated — they are closed over in the lambda inside
        jstep_tgl / jstep_adj.
        """
        key = jax.random.PRNGKey(seed)
        dtype = self.dtype
        nx = self.nx
        ny = self.ny

        # --- shapes ---------------------------------------------------------
        # u0/v0/h0 enter jstep in SW-internal convention (transposed + expanded)
        nl = self.nl
        u_shape = (1, nl, nx + 1, ny)
        v_shape = (1, nl, nx, ny + 1)
        h_shape = (1, nl, nx, ny)
        H_shape = (1, nx, ny)
        # Fu/Fv/Fh stay in State convention (ny, nx+1/ny+1/nx) — always 2D
        Fu_shape = (ny, nx + 1)
        Fv_shape = (ny + 1, nx)
        Fh_shape = (ny, nx)
        # boundary conditions (State convention)
        ub_shape = (ny, nx + 1)
        vb_shape = (ny + 1, nx)
        hb_shape = (ny, nx)

        def rand(key, shape):
            key, subkey = jax.random.split(key)
            return key, jax.random.normal(subkey, shape=shape, dtype=dtype) * 1e-4

        # Masks from the SW model (squeeze batch/layer dims for masking)
        mask_u = self.model.masks.u[0, 0]   # (nx+1, ny)
        mask_v = self.model.masks.v[0, 0]   # (nx, ny+1)
        mask_h = self.model.masks.h[0, 0]   # (nx, ny)

        # --- base trajectory ------------------------------------------------
        key, u0 = rand(key, u_shape)
        key, v0 = rand(key, v_shape)
        key, h0 = rand(key, h_shape)
        u0 = u0 * mask_u
        v0 = v0 * mask_v
        h0 = h0 * mask_h

        has_H = 'H' in self.name_params
        if has_H:
            key, H = rand(key, H_shape)
        else:
            H = None

        key, Fu = rand(key, Fu_shape)
        key, Fv = rand(key, Fv_shape)
        key, Fh = rand(key, Fh_shape)

        # boundary conditions (fixed, not differentiated)
        key, u_b = rand(key, ub_shape)
        key, v_b = rand(key, vb_shape)
        key, h_b = rand(key, hb_shape)

        # --- TLM perturbation -----------------------------------------------
        key, du0 = rand(key, u_shape);  du0 = du0 * mask_u
        key, dv0 = rand(key, v_shape);  dv0 = dv0 * mask_v
        key, dh0 = rand(key, h_shape);  dh0 = dh0 * mask_h
        if has_H:
            key, dH = rand(key, H_shape)
        else:
            dH = None
        key, dFu = rand(key, Fu_shape)
        key, dFv = rand(key, Fv_shape)
        key, dFh = rand(key, Fh_shape)

        # --- ADJ cotangent (output space: u1, v1, h1) -----------------------
        key, wu = rand(key, u_shape);  wu = wu * mask_u
        key, wv = rand(key, v_shape);  wv = wv * mask_v
        key, wh = rand(key, h_shape);  wh = wh * mask_h

        # --- Run TLM --------------------------------------------------------
        du1, dv1, dh1 = self.jstep_tgl(
            0, du0, dv0, dh0, dH, dFu, dFv, dFh,
            u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b, nstep=nstep)

        # --- Run ADJ --------------------------------------------------------
        # Zero accumulators so output = pure adjoint
        adH_in  = jnp.zeros(H_shape, dtype=dtype) if has_H else None
        adFu_in = jnp.zeros(Fu_shape, dtype=dtype)
        adFv_in = jnp.zeros(Fv_shape, dtype=dtype)
        adFh_in = jnp.zeros(Fh_shape, dtype=dtype)

        adu0, adv0, adh0, adH_out, adFu_out, adFv_out, adFh_out = self.jstep_adj(
            0, wu, wv, wh, adH_in, adFu_in, adFv_in, adFh_in,
            u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b, nstep=nstep)

        # --- Check NaN ------------------------------------------------------
        has_nan = (
            jnp.any(jnp.isnan(du1)) | jnp.any(jnp.isnan(dv1)) |
            jnp.any(jnp.isnan(dh1)) | jnp.any(jnp.isnan(adu0)) |
            jnp.any(jnp.isnan(adv0)) | jnp.any(jnp.isnan(adh0)))
        if has_nan:
            print(f'  jstep adjoint test (dtype={dtype}, {nstep=}): NaN detected!')
            return float('nan')

        # --- Inner products (f64 accumulation) ------------------------------
        to64 = lambda x: x.astype(jnp.float64)

        # <M dx, y>  (output space)
        ps1 = (jnp.sum(to64(du1) * to64(wu))
             + jnp.sum(to64(dv1) * to64(wv))
             + jnp.sum(to64(dh1) * to64(wh)))

        # <dx, M* y>  (input space: u, v, h, H, Fu, Fv, Fh)
        ps2 = (jnp.sum(to64(du0) * to64(adu0))
             + jnp.sum(to64(dv0) * to64(adv0))
             + jnp.sum(to64(dh0) * to64(adh0))
             + jnp.sum(to64(dFu) * to64(adFu_out))
             + jnp.sum(to64(dFv) * to64(adFv_out))
             + jnp.sum(to64(dFh) * to64(adFh_out)))
        if has_H:
            ps2 += jnp.sum(to64(dH) * to64(adH_out))

        ratio = float(ps1 / ps2)
        print(f'  jstep adjoint test (dtype={dtype}, {nstep=}): '
              f'<Mdx,y>/<dx,M*y> = {ratio}')
        

class Model_bmit(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        self.config = config

        # BM model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",f'{dir_model}/jqgm.py').load_module() 
        model_bm = qgm.Qgm

        # IT model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = config.MOD.dir_model
        
        swm = SourceFileLoader("swm", 
                                dir_model + "/jswm.py").load_module()
        model_it = swm.CSWm

        # grid
        self.ny = State.ny
        self.nx = State.nx
        self.Xu = self._rho_on_u(State.X)
        self.Yu = self._rho_on_u(State.Y)
        self.Xv = self._rho_on_v(State.X)
        self.Yv = self._rho_on_v(State.Y)

        # Coriolis
        self.f = State.f
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = State.g

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            self.mdt = None

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        self.f_on_u = (self.f[:, :-1] + self.f[:, 1:]) / 2
        self.f_on_v = (self.f[:-1, :] + self.f[1:, :]) / 2

        # Equivalent depth
        self.Heb = self.c**2 / self.g

        # Mean depth (bathymetry)
        self.H = config.MOD.H 
        
        # Boundary angles
        if config.MOD.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
        else:
            self.bc_theta = np.array([0])
        
        # Open Boundary condition kind
        self.bc_kind = config.MOD.bc_kind

        # External boundary conditions
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        
        # Tide frequencies
        self.omegas = np.asarray(config.MOD.w_waves)
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmean(State.DX), np.nanmean(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmax(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for max(c)=', np.nanmax(self.c), 'm/s and grid spacing ~', grid_spacing, 'm: dt<=', dt, 's')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

        # Initialize model state
        self.name_var = config.MOD.name_var
        self.var_to_save = [self.name_var['SSH']] # ssh

        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = xr.open_dataset(config.GRID.path_init_grid)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    if name=='U_IT':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                    elif name=='V_IT':
                        State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                    elif name in ['SSH_IT', 'SSH_BM', 'SSH']:
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                if name=='U_IT':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                elif name=='V_IT':
                    State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                elif name in ['SSH_IT', 'SSH_BM', 'SSH', 'U_BM', 'V_BM']:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # BM Model Parameters (Flux)
        State.params[self.name_var['SSH_BM']] = np.zeros((State.ny,State.nx))

        # IT Model Parameters (OBC, He and alpha coefficient)
        self.name_params_it = config.MOD.name_params_it
        if 'He_mean' in self.name_params_it:
            State.params['He_mean'] = np.zeros((self.ny, self.nx))
        if 'alpha' in self.name_params_it:
            State.params['alpha'] = np.zeros((self.ny, self.nx))
        if 'hbc' in self.name_params_it:
            self.shapehbcx = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.nx # NX
                              ]
            self.shapehbcy = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.ny # NY
                              ]
            State.params['hbcx'] = np.zeros((self.shapehbcx))
            State.params['hbcy'] = np.zeros((self.shapehbcy))
        

        # Coupling terms from vertical modes
        self.flag_coupling_from_bm = config.MOD.flag_coupling_from_bm
        if self.flag_coupling_from_bm and os.path.isfile(config.MOD.path_vertical_modes):
            self._compute_coupling_coefficients(config.MOD.path_vertical_modes)
        else:
            self.flag_coupling_from_bm = False


        # Sponge layer
        self.flag_bc_sponge = config.MOD.flag_bc_sponge
        self.sponge_width = config.MOD.dist_sponge_bc
        self.sponge_coef = config.MOD.sponge_coef
        if self.flag_bc_sponge:
            self.Xh = State.X
            self.Yh = State.Y
            self.Xu = 0.5 * (State.X[:, :-1] + State.X[:, 1:])
            self.Yu = 0.5 * (State.Y[:, :-1] + State.Y[:, 1:])
            self.Xv = 0.5 * (State.X[:-1, :] + State.X[1:, :])
            self.Yv = 0.5 * (State.Y[:-1, :] + State.Y[1:, :])

            if config.MOD.periodic_x and config.MOD.periodic_y:
                sponge_h = np.zeros_like(self.Xh)
                sponge_u = np.zeros_like(self.Xu)
                sponge_v = np.zeros_like(self.Xv)
            elif config.MOD.periodic_x and not config.MOD.periodic_y:
                sponge_h = np.maximum.reduce([
                    gaspari_cohn(self.Yh-self.Yh[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yh[-1,:]-self.Yh, self.sponge_width*1e3)])
                sponge_u = np.maximum.reduce([
                    gaspari_cohn(self.Yu-self.Yu[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yu[-1,:]-self.Yu, self.sponge_width*1e3)
                    ])
                sponge_v = np.maximum.reduce([
                    gaspari_cohn(self.Yv-self.Yv[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yv[-1,:]-self.Yv, self.sponge_width*1e3)
                    ])
            elif not config.MOD.periodic_x and config.MOD.periodic_y:
                sponge_h = np.maximum.reduce([
                    gaspari_cohn(self.Xh.T-self.Xh[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xh[:,-1].T-self.Xh.T, self.sponge_width*1e3).T
                    ])
                sponge_u = np.maximum.reduce([
                    gaspari_cohn(self.Xu.T-self.Xu[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xu[:,-1].T-self.Xu.T, self.sponge_width*1e3).T
                    ])
                sponge_v = np.maximum.reduce([
                    gaspari_cohn(self.Xv.T-self.Xv[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xv[:,-1].T-self.Xv.T, self.sponge_width*1e3).T
                    ])
            else:
                sponge_h = np.maximum.reduce([
                    gaspari_cohn(self.Xh.T-self.Xh[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xh[:,-1].T-self.Xh.T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Yh-self.Yh[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yh[-1,:]-self.Yh, self.sponge_width*1e3)
                    ])
                sponge_u = np.maximum.reduce([
                    gaspari_cohn(self.Xu.T-self.Xu[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xu[:,-1].T-self.Xu.T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Yu-self.Yu[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yu[-1,:]-self.Yu, self.sponge_width*1e3)
                    ])
                sponge_v = np.maximum.reduce([
                    gaspari_cohn(self.Xv.T-self.Xv[:,0].T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Xv[:,-1].T-self.Xv.T, self.sponge_width*1e3).T, 
                    gaspari_cohn(self.Yv-self.Yv[0,:], self.sponge_width*1e3), 
                    gaspari_cohn(self.Yv[-1,:]-self.Yv, self.sponge_width*1e3)
                    ]) 

            def _normalize_sponge(sponge):
                sponge_min = jnp.min(sponge)
                sponge_max = jnp.max(sponge)
                return (sponge - sponge_min) / (sponge_max - sponge_min)

            self.sponge_h = _normalize_sponge(sponge_h)
            self.sponge_u = _normalize_sponge(sponge_u)
            self.sponge_v = _normalize_sponge(sponge_v)

            if not config.MOD.periodic_y:
                self.sponge_on_h_S = (np.abs(self.Yh-self.Yh[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_S = (np.abs(self.Yu-self.Yu[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_S = (np.abs(self.Yv-self.Yv[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_N = (np.abs(self.Yh-self.Yh[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_N = (np.abs(self.Yu-self.Yu[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_v_N = (np.abs(self.Yv-self.Yv[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
            else:
                self.sponge_on_h_S = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_S = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_S = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_N = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_N = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_N = np.zeros_like(self.Yv, dtype=bool)
            if not config.MOD.periodic_x:
                self.sponge_on_h_W = (np.abs(self.Xh-self.Xh[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_W = (np.abs(self.Xu-self.Xu[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_W = (np.abs(self.Xv-self.Xv[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_E = (np.abs(self.Xh-self.Xh[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_E = (np.abs(self.Xu-self.Xu[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_E = (np.abs(self.Xv-self.Xv[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
            else:
                self.sponge_on_h_W = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_W = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_W = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_E = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_E = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_E = np.zeros_like(self.Yv, dtype=bool)
            
            self.weight_sponge_u = np.array(self.sponge_on_u_S, dtype=float) + np.array(self.sponge_on_u_N, dtype=float) + np.array(self.sponge_on_u_W, dtype=float) + np.array(self.sponge_on_u_E, dtype=float)
            self.weight_sponge_v = np.array(self.sponge_on_v_S, dtype=float) + np.array(self.sponge_on_v_N, dtype=float) + np.array(self.sponge_on_v_W, dtype=float) + np.array(self.sponge_on_v_E, dtype=float)
            self.weight_sponge_h = np.array(self.sponge_on_h_S, dtype=float) + np.array(self.sponge_on_h_N, dtype=float) + np.array(self.sponge_on_h_W, dtype=float) + np.array(self.sponge_on_h_E, dtype=float)
            self.weight_sponge_u[self.weight_sponge_u==0] = 1
            self.weight_sponge_v[self.weight_sponge_v==0] = 1
            self.weight_sponge_h[self.weight_sponge_h==0] = 1

            # Update mask (for avoiding assimilation in sponge regions)
            State.mask[self.sponge_on_h_S] = True
            State.mask[self.sponge_on_h_N] = True
            State.mask[self.sponge_on_h_W] = True
            State.mask[self.sponge_on_h_E] = True
        
        self.mask = State.mask
        
        # Model initialization
        self.model_bm = model_bm(
            dx=State.DX,
            dy=State.DY,
            dt=self.dt,
            SSH=State.getvar(name_var=self.name_var['SSH_BM']),
            c=self.c,
            time_scheme=config.MOD.time_scheme_bm,
            g=config.MOD.g,
            f=self.f,
            Kdiffus=config.MOD.Kdiffus,
            mdt=self.mdt
            )
        
        self.model_it = model_it(
            X=State.X,
            Y=State.Y,
            dt=self.dt,
            bc=self.bc_kind,
            omegas=self.omegas,
            bc_theta=self.bc_theta,
            f=self.f,
            Heb=self.Heb,
            obc_north=config.MOD.obc_north,
            obc_west=config.MOD.obc_west,
            obc_south=config.MOD.obc_south,
            obc_east=config.MOD.obc_east,
            periodic_x=config.MOD.periodic_x,
            periodic_y=config.MOD.periodic_y,
            )
    
        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep)
        self._jstep_tgl_jit = jit(self._jstep_tgl)
        self._jstep_adj_jit = jit(self._jstep_adj)
        self._compute_w1_IT_jit = jit(self._compute_w1_IT)
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('BM/IT Tangent test:')
            tangent_test(self,State,nstep=10)
            print('BM/IT Adjoint test:')
            adjoint_test(self,State,nstep=10)
    
    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])
        
        State.var[self.name_var['SSH']] = State.var[self.name_var['SSH_BM']] + State.var[self.name_var['SSH_IT']]
    
    def save_output(self,State,present_date,name_var=None,t=None):

        name_var_to_save = [self.name_var['SSH_BM'], 
                            self.name_var['SSH_IT'], 
                            self.name_var['SSH'], 
                            self.name_var['U_IT']+'_interp', 
                            self.name_var['V_IT']+'_interp',
                            'He',
                            'He_mean',
                            'alpha']
        
        if self.mdt is not None:
            ssh_bm = State.getvar(name_var=self.name_var['SSH_BM']) + self.mdt
            ssh = State.getvar(name_var=self.name_var['SSH']) + self.mdt
            State.setvar(ssh_bm, name_var='ssh_bm')
            State.setvar(ssh, name_var='ssh')
            if name_var is not None:
                name_var_to_save += ['ssh_bm','ssh']
        
        u = +State.getvar(name_var=self.name_var['U_IT'])
        v = +State.getvar(name_var=self.name_var['V_IT'])
        u_to_save = np.zeros((State.ny,State.nx))
        v_to_save = np.zeros((State.ny,State.nx))
        u_to_save[:,1:-1] = (u[:,1:] + u[:,:-1]) * .5
        v_to_save[1:-1,:] = (v[1:,:] + v[:-1,:]) * .5
        State.var[self.name_var['U_IT']+'_interp'] = u_to_save
        State.var[self.name_var['V_IT']+'_interp'] = v_to_save

        if 'He_mean' in self.name_params_it:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha' in self.name_params_it:
            alpha = +State.params['alpha'] 
        else:
            alpha = jnp.zeros((self.ny,self.nx))
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        He2d = self._compute_He_from_bm(He_mean, alpha, h_bm)

        State.var['He'] = He2d + self.Heb
        State.var['He_mean'] = He_mean
        State.var['alpha'] = alpha

        State.save_output(present_date,
                          name_var=name_var_to_save)
        
    def set_bc(self,time_bc,var_bc):

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Remove nan
                        var_bc_t[np.isnan(var_bc_t)] = 0.
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_bc):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
        
        # For step_jax
        self.bc_time_jax = jnp.array(list(self.bc['SSH_BM'].keys()))
        self.bc_time = np.array(list(self.bc['SSH_BM'].keys()))
        self.bc_values = {t: jnp.array(self.bc['SSH_BM'][t]) for t in self.bc['SSH_BM']}
        for name in var_bc:
            self.bc_values[name] = jnp.array(list(self.bc['SSH_BM'].values()))
            
    def _apply_bc(self,t0):
        
        ssh_bc = jnp.zeros((self.ny,self.nx,))
        
        if 'SSH_BM' not in self.bc:
            return ssh_bc
        elif len(self.bc['SSH_BM'].keys())==0:
             return ssh_bc
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        ssh_bc = self.bc_values[t0]
        
        return ssh_bc

    def _rho_on_u(self,rho):
        
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def _rho_on_v(self,rho):
        
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def _compute_coupling_coefficients(self, path_vertical_modes):

        # Open dataset with vertical mode structures
        ds = xr.open_dataset(path_vertical_modes)
        phi1  = ds.phi.sel(mode=1)      # (s_rho, y_rho)
        phi1 = phi1/phi1[-1]
        phi1p = ds.dphidz.sel(mode=1)   # (s_w,  y_rho)  → given derivative!
        c1    = ds.c.sel(mode=1)        # (y_rho)
        N2    = ds.N2                   # (y_rho, s_w)
        z_r   = ds.z_r                  # (s_rho, y_rho)
        z_w = ds.z_w
        dz_r  = ds.dz                   # (s_rho, y_rho)
        s_w = ds.s_w

        #########################################################################
        # Compute U11 coupling term for mode 1
        #########################################################################
        integrand = phi1/phi1[-1] * (phi1**2) * dz_r  # multiply by layer thickness
        U11 = (1/self.H) * integrand.sum(dim="s_rho")   # integral in z
        self.U11 = jnp.array(U11.values)  
        plt.figure()
        plt.pcolormesh(self.U11)
        plt.colorbar()
        plt.title('U11')
        plt.show()

        #########################################################################
        # Compute U11p coupling term for mode 1
        #########################################################################
        φ = -(c1**2 / N2)* phi1p          # (s_w, y_rho)
        phi1_w = phi1.interp(s_rho=s_w)   # now on (s_w, y_rho)
        phi1_w[0] = phi1_w[1]
        phi1_w[-1] = phi1_w[-2]

        integrand = (
            phi1_w/phi1_w[-1]
            * (N2 / c1**2)
            * φ**2
        )   # dims: (s_w, y_rho)
        # thickness between w-levels: same size as s_w except top/bottom.
        dz_w = z_w.diff("s_w")                   # (s_w_minus1, y_rho)
        # pad to same length as s_w (xarray aligns automatically)
        dz_w = dz_w.reindex(s_w=s_w, fill_value=np.nan)
        U11p = (1/self.H) * (integrand * dz_w).sum(dim="s_w")   # integral in z
        self.U11p = jnp.array(U11p.values)
        plt.figure()
        plt.pcolormesh(self.U11p)
        plt.colorbar()
        plt.title('U11p')
        plt.show()

        #########################################################################
        # Compute dHe coupling term for mode 1
        #########################################################################
        # forward and backward slopes
        slope_f = (phi1.shift(s_rho=-1) - phi1) / (z_r.shift(s_rho=-1) - z_r)
        slope_b = (phi1 - phi1.shift(s_rho=1)) / (z_r - z_r.shift(s_rho=1))

        # central 2nd derivative
        phi1pp = 2 * (slope_f - slope_b) / (z_r.shift(s_rho=-1) - z_r.shift(s_rho=1))

        # fill top/bottom NaNs (simple, safe)
        phi1pp = phi1pp.fillna(0)

        # vertical interpolation from s_w → s_rho
        phi1p_rho = phi1p.interp(s_w=ds.s_rho)

        N2_rho = N2.interp(s_w=ds.s_rho)

        c1_rho = c1.broadcast_like(phi1p_rho)

        A = - (c1_rho**2 / N2_rho) * phi1p_rho
        A2 = A**2

        integrand = phi1pp * A2
        dHe = (1/self.H) * (integrand * dz_r).sum("s_rho")
        self.dHe = jnp.array(dHe.values) 

        plt.figure()
        plt.pcolormesh(self.dHe)
        plt.colorbar()
        plt.title('dHe')
        plt.show()

    def _compute_He_from_bm(self, He, alpha, h_bm):
        
        """ Compute equivalent depth with coupling term
        --------------
        Inputs:
        He        : IT equivalent depth control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        He_tot    : Total equivalent depth with coupling term (m)
        """

        if He is not None:
            if self.flag_coupling_from_bm and h_bm is not None:
                He2d =  He + (.5 + alpha) * self.dHe * h_bm 
            else:
                He2d = He
        else:
            He2d = jnp.zeros((self.ny,self.nx))
        
        He2d = jnp.where(self.mask, 0., He2d)
        
        return He2d

    def _compute_advective_terms_from_bm(self, alpha, u_bm, v_bm):

        """ Compute advective terms with coupling term
        --------------
        Inputs:
        alpha     : IT advective control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        u11, v11  : advective coupling terms for u and v (m/s)
        u11p, v11p: advective coupling terms for u and v (m/s)
        """

        if self.flag_coupling_from_bm and u_bm is not None and v_bm is not None:
            u11 = ((.5 - alpha) + (.5 + alpha) * self.U11) * u_bm 
            v11 = ((.5 - alpha) + (.5 + alpha) * self.U11) * v_bm 
            u11p = ((.5 - alpha) + (.5 + alpha) * self.U11p) * u_bm 
            v11p = ((.5 - alpha) + (.5 + alpha) * self.U11p) * v_bm 
        else:
            u11 = None
            v11 = None
            u11p = None
            v11p = None
        
        return u11, v11, u11p, v11p

    def _compute_w1_IT(self,t,He,h_SN,h_WE):
        """
        Compute first characteristic variable w1 for internal tides from external 
        data

        Parameters
        ----------
        t : float 
            time in seconds
        He : 2D array
        h_SN : ND array
            amplitude of SSH for southern/northern borders
        h_WE : ND array
            amplitude of SSH for western/eastern borders

        Returns
        -------
        w1ext: 1D array
            flattened  first characteristic variable (South/North/West/East)
        """

        if h_SN is None or h_WE is None:
            return None 
        
        # Adjust time for 1d bc
        if self.bc_kind=='1d':
            t += self.dt
        
        # South
        HeS = (He[0,:]+He[1,:])/2
        fS = (self.f[0,:]+self.f[1,:])/2
        w1S = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fS**2)/(self.g*HeS))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = jnp.cos(theta) * k
                kxy = kx*self.swm.Xv[0,:] + ky*self.swm.Yv[0,:]
                
                h = h_SN[j,0,0,i]* jnp.cos(w*t-kxy)  +\
                        h_SN[j,0,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fS**2)*( \
                    h_SN[j,0,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fS*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,0,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fS*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                
                w1S += v + jnp.sqrt(self.g/HeS) * h
         
        # North
        fN = (self.f[-1,:]+self.f[-2,:])/2
        HeN = (He[-1,:]+He[-2,:])/2
        w1N = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fN**2)/(self.g*HeN))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = -jnp.cos(theta) * k
                kxy = kx*self.swm.Xv[-1,:] + ky*self.swm.Yv[-1,:]
                h = h_SN[j,1,0,i]* jnp.cos(w*t-kxy)+\
                        h_SN[j,1,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fN**2)*(\
                    h_SN[j,1,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fN*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,1,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fN*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                w1N += v - jnp.sqrt(self.g/HeN) * h

        # West
        fW = (self.f[:,0]+self.f[:,1])/2
        HeW = (He[:,0]+He[:,1])/2
        w1W = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fW**2)/(self.g*HeW))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.swm.Xu[:,0] + ky*self.swm.Yu[:,0]
                h = h_WE[j,0,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,0,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fW**2)*(\
                    h_WE[j,0,0,i]*(w*kx*jnp.cos(w*t-kxy) \
                              + fW*ky*jnp.sin(w*t-kxy)
                                  ) +\
                    h_WE[j,0,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fW*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1W += u + jnp.sqrt(self.g/HeW) * h

        
        # East
        HeE = (He[:,-1]+He[:,-2])/2
        fE = (self.f[:,-1]+self.f[:,-2])/2
        w1E = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fE**2)/(self.g*HeE))
            for i,theta in enumerate(self.bc_theta):
                kx = -jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.swm.Xu[:,-1] + ky*self.swm.Yu[:,-1]
                h = h_WE[j,1,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,1,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fE**2)*(\
                    h_WE[j,1,0,i]* (w*kx*jnp.cos(w*t-kxy) \
                                + fE*ky*jnp.sin(w*t-kxy)
                                    ) +\
                    h_WE[j,1,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fE*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1E += u - jnp.sqrt(self.g/HeE) * h
        
        w1ext = (w1S,w1N,w1W,w1E)    
        
        return w1ext

    def _compute_IT_2D(self,t,He,h_SN,h_WE,flag_tangent=True):
        """
        Compute 2D plane wave IT fields 

        Parameters
        ----------
        t : float 
            time in seconds
        He : 2D array
        h_SN : ND array
            amplitude of SSH for southern/northern borders
        h_WE : ND array
            amplitude of SSH for western/eastern borders

        Returns
        -------
        u,v,h: 2D arrays
            
        """

        u_S = jnp.zeros((self.ny,self.nx-1))
        v_S = jnp.zeros((self.ny-1,self.nx))
        h_S = jnp.zeros((self.ny,self.nx))
        u_N = jnp.zeros((self.ny,self.nx-1))
        v_N = jnp.zeros((self.ny-1,self.nx))
        h_N = jnp.zeros((self.ny,self.nx))
        u_W = jnp.zeros((self.ny,self.nx-1))
        v_W = jnp.zeros((self.ny-1,self.nx))
        h_W = jnp.zeros((self.ny,self.nx))
        u_E = jnp.zeros((self.ny,self.nx-1))
        v_E = jnp.zeros((self.ny-1,self.nx))
        h_E = jnp.zeros((self.ny,self.nx))

        He_on_u = (He[:,1:] + He[:,:-1]) /2
        He_on_v = (He[1:,:] + He[:-1,:]) /2

        for j,w in enumerate(self.omegas):
            k_on_h = jnp.sqrt((w**2-self.f**2)/(self.g*He))
            k_on_u = jnp.sqrt((w**2-self.f_on_u**2)/(self.g*(He_on_u)))
            k_on_v = jnp.sqrt((w**2-self.f_on_v**2)/(self.g*(He_on_v)))

            for i,theta in enumerate(self.bc_theta):

                ####################################
                # South
                ####################################
                kx_on_h = jnp.sin(theta) * k_on_h
                ky_on_h = jnp.cos(theta) * k_on_h
                kx_on_u = jnp.sin(theta) * k_on_u
                ky_on_u = jnp.cos(theta) * k_on_u
                kx_on_v = jnp.sin(theta) * k_on_v
                ky_on_v = jnp.cos(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_S += self.sponge_on_h_S *(\
                    h_SN[j,0,0,i]* jnp.cos(w*t-kxy_on_h)  +\
                    h_SN[j,0,1,i]* jnp.sin(w*t-kxy_on_h)) 
                
                # v
                h_cos_theta_on_v_S = h_SN[j,0,0,i]
                h_sin_theta_on_v_S = h_SN[j,0,1,i]
                v_S += self.sponge_on_v_S * (self.g/(w**2-self.f_on_v**2)*( \
                    h_cos_theta_on_v_S * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                    ) +\
                    h_sin_theta_on_v_S * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                    )
                        ))
                
                # u
                if flag_tangent:
                    h_cos_theta_on_u_S = (h_cos_theta_on_v_S[1:] + h_cos_theta_on_v_S[:-1]) * 0.5
                    h_sin_theta_on_u_S = (h_sin_theta_on_v_S[1:] + h_sin_theta_on_v_S[:-1]) * 0.5
                    u_S += self.sponge_on_u_S * (self.g/(w**2-self.f_on_u**2)*( \
                        h_cos_theta_on_u_S * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                    + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                        ) +\
                        h_sin_theta_on_u_S * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                    - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                        )
                            ))
                
                ####################################
                # North
                ####################################
                kx_on_h = +jnp.sin(theta) * k_on_h
                ky_on_h = -jnp.cos(theta) * k_on_h
                kx_on_u = +jnp.sin(theta) * k_on_u
                ky_on_u = -jnp.cos(theta) * k_on_u
                kx_on_v = +jnp.sin(theta) * k_on_v
                ky_on_v = -jnp.cos(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_N += self.sponge_on_h_N *(\
                    h_SN[j,1,0,i]* jnp.cos(w*t-kxy_on_h)  +\
                    h_SN[j,1,1,i]* jnp.sin(w*t-kxy_on_h)) 
                
                # v
                h_cos_theta_on_v_N = h_SN[j,1,0,i]
                h_sin_theta_on_v_N = h_SN[j,1,1,i]
                v_N += self.sponge_on_v_N * (self.g/(w**2-self.f_on_v**2)*( \
                    h_cos_theta_on_v_N * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                    ) +\
                    h_sin_theta_on_v_N * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                    )
                        ))  
                
                # u
                if flag_tangent:
                    h_cos_theta_on_u_N = (h_cos_theta_on_v_N[1:] + h_cos_theta_on_v_N[:-1]) * 0.5
                    h_sin_theta_on_u_N = (h_sin_theta_on_v_N[1:] + h_sin_theta_on_v_N[:-1]) * 0.5
                    u_N += self.sponge_on_u_N * (self.g/(w**2-self.f_on_u**2)*( \
                        h_cos_theta_on_u_N * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                    + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                        ) +\
                        h_sin_theta_on_u_N * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                    - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                        )
                            ))
                
                ####################################
                # West
                ####################################
                kx_on_h = jnp.cos(theta) * k_on_h
                ky_on_h = jnp.sin(theta) * k_on_h
                kx_on_u = jnp.cos(theta) * k_on_u
                ky_on_u = jnp.sin(theta) * k_on_u
                kx_on_v = jnp.cos(theta) * k_on_v
                ky_on_v = jnp.sin(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_W += self.sponge_on_h_W *(\
                    h_WE[j,0,0,i][:,None]* jnp.cos(w*t-kxy_on_h)  +\
                    h_WE[j,0,1,i][:,None]* jnp.sin(w*t-kxy_on_h)) 
                
                # u
                h_cos_theta_on_u_W = h_WE[j,0,0,i][:,None]
                h_sin_theta_on_u_W = h_WE[j,0,1,i][:,None]
                u_W += self.sponge_on_u_W * (self.g/(w**2-self.f_on_u**2)*( \
                    h_cos_theta_on_u_W * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                    ) +\
                    h_sin_theta_on_u_W * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                    )
                        ))

                # v
                if flag_tangent:
                    h_cos_theta_on_v_W = (h_cos_theta_on_u_W[1:] + h_cos_theta_on_u_W[:-1]) * 0.5
                    h_sin_theta_on_v_W = (h_sin_theta_on_u_W[1:] + h_sin_theta_on_u_W[:-1]) * 0.5
                    v_W += self.sponge_on_v_W * (self.g/(w**2-self.f_on_v**2)*( \
                        h_cos_theta_on_v_W * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                    - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                        ) +\
                        h_sin_theta_on_v_W * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                    + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                        )
                            ))  
                
                
                
                ####################################
                # East
                ####################################
                kx_on_h = -jnp.cos(theta) * k_on_h
                ky_on_h = jnp.sin(theta) * k_on_h
                kx_on_u = -jnp.cos(theta) * k_on_u
                ky_on_u = jnp.sin(theta) * k_on_u
                kx_on_v = -jnp.cos(theta) * k_on_v
                ky_on_v = jnp.sin(theta) * k_on_v
                kxy_on_h = kx_on_h*self.Xh + ky_on_h*self.Yh
                kxy_on_u = kx_on_u*self.Xu + ky_on_u*self.Yu
                kxy_on_v = kx_on_v*self.Xv + ky_on_v*self.Yv

                # h
                h_E += self.sponge_on_h_E *(\
                    h_WE[j,1,0,i][:,None]* jnp.cos(w*t-kxy_on_h)  +\
                    h_WE[j,1,1,i][:,None]* jnp.sin(w*t-kxy_on_h))
                
                # u
                h_cos_theta_on_u_E = h_WE[j,1,0,i][:,None]
                h_sin_theta_on_u_E = h_WE[j,1,1,i][:,None]
                u_E += self.sponge_on_u_E * (self.g/(w**2-self.f_on_u**2)*( \
                    h_cos_theta_on_u_E * (w*kx_on_u*jnp.cos(w*t-kxy_on_u) \
                                + self.f_on_u*ky_on_u*jnp.sin(w*t-kxy_on_u)
                                    ) +\
                    h_sin_theta_on_u_E * (w*kx_on_u*jnp.sin(w*t-kxy_on_u) \
                                - self.f_on_u*ky_on_u*jnp.cos(w*t-kxy_on_u)
                                    )
                        ))
                
                # v
                if flag_tangent:
                    h_cos_theta_on_v_E = (h_cos_theta_on_u_E[1:] + h_cos_theta_on_u_E[:-1]) * 0.5
                    h_sin_theta_on_v_E = (h_sin_theta_on_u_E[1:] + h_sin_theta_on_u_E[:-1]) * 0.5
                    v_E += self.sponge_on_v_E * (self.g/(w**2-self.f_on_v**2)*( \
                        h_cos_theta_on_v_E * (w*ky_on_v*jnp.cos(w*t-kxy_on_v) \
                                    - self.f_on_v*kx_on_v*jnp.sin(w*t-kxy_on_v)
                                        ) +\
                        h_sin_theta_on_v_E * (w*ky_on_v*jnp.sin(w*t-kxy_on_v) \
                                    + self.f_on_v*kx_on_v*jnp.cos(w*t-kxy_on_v)
                                        )
                            ))
        
        u_it = (u_S + u_N + u_W + u_E) / self.weight_sponge_u
        v_it = (v_S + v_N + v_W + v_E) / self.weight_sponge_v
        h_it = (h_S + h_N + h_W + h_E) / self.weight_sponge_h


        return u_it, v_it, h_it
    
    def _jstep(self, t, 
            h_bm, u_it, v_it, h_it, h_tot, 
            F_bm, He_mean, alpha, h_SN, h_WE,
            h_bm_bc):

        """ One time step of the coupled BM/IT model
        --------------
        Inputs:
        t         : time (s)
        h_bm      : [dynamic] BM sea surface height (m) 
        u_it      : [dynamic] IT zonal velocity (m/s)
        v_it      : [dynamic] IT meridional velocity (m/s)
        h_it      : [dynamic] IT sea surface height (m)
        h_tot     : [dynamic] Total sea surface height (m)
        F_bm      : [dynamic] Flux correction BM (m)
        He_mean   : [dynamic] IT equivalent depth mean (m)
        alpha     : [dynamic] IT coupling coefficient (m)
        h_SN      : [dynamic] IT boundary condition North/South (m)
        h_WE      : [dynamic] IT boundary condition West/East (m)
        h_bm_bc   : [static] BM sea surface height boundary condition (m)
        --------------
        Outputs:
        h_bm      : updated BM sea surface height (m)
        u_it      : updated IT zonal velocity (m/s)
        v_it      : updated IT meridional velocity (m/s)
        h_it      : updated IT sea surface height (m)
        h_tot     : updated Total sea surface height (m)
        """

        # Compute equivalent depth anomaly from control parameters
        He2d = self._compute_He_from_bm(He_mean, alpha, h_bm)
        He_tot = self.Heb + He2d

        # Compute advective terms from control parameters
        if self.mdt is not None:
            ssh_bm = h_bm + self.mdt
        else:
            ssh_bm = h_bm
        u_bm, v_bm = self.model_bm.h2uv(ssh_bm)
        u11, v11, u11p, v11p = self._compute_advective_terms_from_bm(alpha, u_bm, v_bm)

        # Compute characteristic variable from external data
        if not self.flag_bc_sponge:
            w1ext = self._compute_w1_IT(t, He_tot, h_SN, h_WE)
        else:
            w1ext = None

        # One BM forward step
        h_bm = self.model_bm.step(h_bm, h_bm_bc)

        # One IT forward step
        u_it, v_it, h_it = self.model_it.step(u_it, v_it, h_it, He2d, w1ext=w1ext, u11=u11, v11=v11, u11p=u11p, v11p=v11p)

        # Sponge Boundary conditions
        if self.flag_bc_sponge:
            _u_b, _v_b, _h_b = self._compute_IT_2D(t, He_tot, h_SN, h_WE)
            u_it += self.sponge_coef * self.sponge_u * (_u_b - u_it)
            v_it += self.sponge_coef * self.sponge_v * (_v_b - v_it)
            h_it += self.sponge_coef * self.sponge_h * (_h_b - h_it)

        # Sum of BM/IT
        h_tot = h_bm + h_it

        # Flux correction BM
        if F_bm is not None:
            h_bm = h_bm + self.dt/(3600*24) * F_bm
    
        return h_bm, u_it, v_it, h_it, h_tot
     
    def _jstep_tgl(self, t, 
                dh_bm, du_it, dv_it, dh_it, dh_tot, 
                dF_bm, dHe_mean, dalpha, dh_SN, dh_WE,
                h_bm, u_it, v_it, h_it, h_tot, 
                F_bm, He_mean, alpha, h_SN, h_WE, 
                h_bm_bc):
        """ One time step of the coupled BM/IT model tangent linear
        --------------
        Inputs:
        t         : time (s)
        dh_bm     : [dynamic] perturbation BM sea surface height (m) 
        du_it     : [dynamic] perturbation IT zonal velocity (m/s)
        dv_it     : [dynamic] perturbation IT meridional velocity (m/s)
        dh_it     : [dynamic] perturbation IT sea surface height (m)
        dh_tot    : [dynamic] perturbation Total sea surface height (m)
        dF_bm     : [dynamic] perturbation Flux correction BM (m)
        dHe_mean  : [dynamic] perturbation IT equivalent depth mean (m)
        dalpha    : [dynamic] perturbation IT equivalent depth anomaly (m)
        dh_SN     : [dynamic] perturbation IT boundary condition North/South (m)
        dh_WE     : [dynamic] perturbation IT boundary condition West/East (m)
        h_bm      : [static] BM sea surface height (m) 
        u_it      : [static] IT zonal velocity (m/s)
        v_it      : [static] IT meridional velocity (m/s)
        h_it      : [static] IT sea surface height (m)
        h_tot     : [static] Total sea surface height (m)
        F_bm      : [static] Flux correction BM (m)
        He_mean   : [static] IT equivalent depth mean (m)
        alpha     : [static] IT equivalent depth anomaly (m)
        h_SN      : [static] IT boundary condition North/South (m)
        h_WE      : [static] IT boundary condition West/East (m)
        h_bm_bc   : [static] BM sea surface height boundary condition (m)
        --------------
        Outputs:
        du_it     : updated perturbation IT zonal velocity (m/s)
        dv_it     : updated perturbation IT meridional velocity (m/s)
        dh_it     : updated perturbation IT sea surface height (m)
        dh_bm     : updated perturbation BM sea surface height (m)
        dh_tot    : updated perturbation Total sea surface height (m)
        """
        
        def wrapped_jstep(x):
            h_bm, u_it, v_it, h_it, h_tot, \
                F_bm, He_mean, alpha, h_SN, h_WE = x
            return self._jstep(t, 
                            h_bm, u_it, v_it, h_it, h_tot, 
                            F_bm, He_mean, alpha, h_SN, h_WE,
                            h_bm_bc)

        primals = ((h_bm, u_it, v_it, h_it, h_tot, 
                    F_bm, He_mean, alpha, h_SN, h_WE),)
        tangents = ((dh_bm, du_it, dv_it, dh_it, dh_tot, 
                    dF_bm, dHe_mean, dalpha, dh_SN, dh_WE,),)

        _, dy = jax.jvp(wrapped_jstep, primals, tangents)

        return dy  # returns (dh_bm, du_it, dv_it, dh_it, dh_tot)

    def _jstep_adj(self, t, 
                adh_bm, adu_it, adv_it, adh_it, adh_tot, 
                adF_bm, adHe_mean, adalpha, adh_SN, adh_WE,
                h_bm, u_it, v_it, h_it, h_tot, 
                F_bm, He_mean, alpha, h_SN, h_WE, 
                h_bm_bc):
        """ One time step of the coupled BM/IT model adjoint
        --------------
        Inputs:
        t         : time (s)
        adh_bm    : [dynamic] adjoint BM sea surface height (m) 
        adu_it    : [dynamic] adjoint IT zonal velocity (m/s)
        adv_it    : [dynamic] adjoint IT meridional velocity (m/s)
        adh_it    : [dynamic] adjoint IT sea surface height (m)
        adh_tot   : [dynamic] adjoint Total sea surface height (m)
        adF_bm    : [dynamic] adjoint Flux correction BM (m)
        adHe      : [dynamic] adjoint IT equivalent depth (m)
        adh_SN    : [dynamic] adjoint IT boundary condition North/South (m)
        adh_WE    : [dynamic] adjoint IT boundary condition West/East (m)
        h_bm      : [static] BM sea surface height (m) 
        u_it      : [static] IT zonal velocity (m/s)
        v_it      : [static] IT meridional velocity (m/s)
        h_it      : [static] IT sea surface height (m)
        h_tot     : [static] Total sea surface height (m)
        F_bm      : [static] Flux correction BM (m)
        He        : [static] IT equivalent depth (m)
        h_SN      : [static] IT boundary condition North/South (m)
        h_WE      : [static] IT boundary condition West/East (m)
        h_bm_bc   : [static] BM sea surface height boundary condition (m)
        --------------
        Outputs:
        adh_bm    : updated adjoint BM sea surface height (m) 
        adu_it    : updated adjoint IT zonal velocity (m/s)
        adv_it    : updated adjoint IT meridional velocity (m/s)
        adh_it    : updated adjoint IT sea surface height (m)
        adh_tot   : updated adjoint Total sea surface height (m)
        adHe      : updated adjoint IT equivalent depth (m)
        adh_SN    : updated adjoint IT boundary condition North/South (m)
        adh_WE    : updated adjoint IT boundary condition West/East (m)
        """

        def wrapped_jstep(x):
            h_bm, u_it, v_it, h_it, h_tot, \
                F_bm, He_mean, alpha, h_SN, h_WE = x
            return self._jstep(t, 
                            h_bm, u_it, v_it, h_it, h_tot, 
                            F_bm, He_mean, alpha, h_SN, h_WE,
                            h_bm_bc)

        primals = ((h_bm, u_it, v_it, h_it, h_tot, 
                    F_bm, He_mean, alpha, h_SN, h_WE),)
        cotangents = (adh_bm, adu_it, adv_it, adh_it, adh_tot)  

        _, vjp_fn = jax.vjp(wrapped_jstep, *primals)
        adjoints = vjp_fn(cotangents)

        adh_bm, adu_it, adv_it, adh_it, adh_tot, \
            _adF_bm, _adHe_mean, _adalpha, _adhSN, _adhWE = adjoints[0]
        
        if adF_bm is not None:
            adF_bm += _adF_bm
        if adHe_mean is not None:
            adHe_mean += _adHe_mean
        if adalpha is not None:
            adalpha += _adalpha
        if adh_SN is not None:
            adh_SN += _adhSN
        if adh_WE is not None:
            adh_WE += _adhWE

        return adh_bm, adu_it, adv_it, adh_it, adh_tot, adF_bm, adHe_mean, adalpha, adh_SN, adh_WE
    

    def step(self,State,nstep=1,t=0):

        # Get state variable
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])

        # Static BM Boundary condition
        h_bm_bc = self._apply_bc(t)

        # Get control (dynamic) parameters
        F_bm = State.params[self.name_var['SSH_BM']] # Flux correction BM
        if 'He_mean' in self.name_params_it:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
            
            alpha = +State.params['alpha'] 
        if self.flag_coupling_from_bm and 'alpha' in self.name_params_it:
            alpha = +State.params['alpha']
        else:
            alpha = jnp.zeros((self.ny,self.nx))
        if 'hbc' in self.name_params_it:
            h_SN = +State.params['hbcx']
            h_WE = +State.params['hbcy']
        else:
            h_SN = None
            h_WE = None
            
        # Time stepping
        for it in range(nstep):
            # One time step
            h_bm, u_it, v_it, h_it, h_tot = self._jstep_jit(t+it*self.dt, 
                                                            h_bm, u_it, v_it, h_it, h_tot, 
                                                            F_bm, He_mean, alpha, h_SN, h_WE,
                                                            h_bm_bc)
        
        # Update state
        State.setvar([h_bm, u_it,v_it,h_it, h_tot],[
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )
    
    def step_tgl(self,dState,State,nstep=1,t=0):
        
        # Get state variable
        dh_bm = +dState.getvar(name_var=self.name_var['SSH_BM'])
        du_it = +dState.getvar(name_var=self.name_var['U_IT'])
        dv_it = +dState.getvar(name_var=self.name_var['V_IT'])
        dh_it = +dState.getvar(name_var=self.name_var['SSH_IT'])
        dh_tot = +dState.getvar(name_var=self.name_var['SSH'])
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])

        # Static BM Boundary condition
        h_bm_bc = self._apply_bc(t)
        
        # Get control (dynamic) parameters
        dF_bm = dState.params[self.name_var['SSH_BM']] # Flux correction BM
        F_bm = State.params[self.name_var['SSH_BM']] # Flux correction BM
        if 'He_mean' in self.name_params_it:
            dHe_mean = +dState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = dHe_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha' in self.name_params_it:
            dalpha = +dState.params['alpha']
            alpha = +State.params['alpha'] 
        if 'hbc' in self.name_params_it:
            dh_SN = dState.params['hbcx']
            dh_WE = dState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            dh_SN = dh_WE = None
            h_SN = h_WE = None
            
        # Time stepping
        for it in range(nstep):
            # One time step
            dh_bm, du_it, dv_it, dh_it, dh_tot = self._jstep_tgl_jit(
                                                        t+it*self.dt, 
                                                        dh_bm, du_it, dv_it, dh_it, dh_tot, 
                                                        dF_bm, dHe_mean, dalpha, dh_SN, dh_WE,
                                                        h_bm, u_it, v_it, h_it, h_tot, 
                                                        F_bm, He_mean, alpha, h_SN, h_WE,
                                                        h_bm_bc
                                                        )
            h_bm, u_it, v_it, h_it, h_tot = self._jstep_jit(t+it*self.dt, 
                                                            h_bm, u_it, v_it, h_it, h_tot, 
                                                            F_bm, He_mean, alpha, h_SN, h_WE,
                                                            h_bm_bc
                                                            )

        # Update state
        dState.setvar([dh_bm, du_it,dv_it,dh_it,dh_tot],[
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )
       
    def step_adj(self, adState, State, nstep=1, t=0):
        
        # Get state variable
        adh_bm = +adState.getvar(name_var=self.name_var['SSH_BM'])
        adu_it = +adState.getvar(name_var=self.name_var['U_IT'])
        adv_it = +adState.getvar(name_var=self.name_var['V_IT'])
        adh_it = +adState.getvar(name_var=self.name_var['SSH_IT'])
        adh_tot = +adState.getvar(name_var=self.name_var['SSH'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])
        
        # Get control (dynamic) parameters
        adF_bm = adState.params[self.name_var['SSH_BM']] # Flux correction BM
        F_bm = State.params[self.name_var['SSH_BM']] # Flux correction BM
        # Get parameters
        if 'He_mean' in self.name_params_it:
            adHe_mean = +adState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = adHe_mean = jnp.zeros((self.ny,self.nx))
        if self.flag_coupling_from_bm and 'alpha' in self.name_params_it:
            adalpha = +adState.params['alpha']
            alpha = +State.params['alpha'] 
        else:
            alpha = adalpha = jnp.zeros((self.ny,self.nx))
        if 'hbc' in self.name_params_it:
            adh_SN = adState.params['hbcx']
            adh_WE = adState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            adh_SN = adh_WE = None
            h_SN = h_WE = None
        
        # Static BM Boundary condition
        h_bm_bc = self._apply_bc(t)

        # Current trajectory
        h_bm_list = [h_bm]; u_it_list = [u_it]; v_it_list = [v_it]; h_it_list = [h_it]; h_tot_list = [h_tot]
        for it in range(nstep):
            # One time step
            h_bm, u_it, v_it, h_it, h_tot = self._jstep_jit(t+it*self.dt, 
                                               h_bm, u_it, v_it, h_it, h_tot, 
                                               F_bm, He_mean, alpha, h_SN, h_WE,
                                               h_bm_bc)
            h_bm_list.append(h_bm); u_it_list.append(u_it); v_it_list.append(v_it); h_it_list.append(h_it); h_tot_list.append(h_tot)
            
        # Reverse-time adjoint loop
        for it in reversed(range(nstep)):
            # Non-linear trajectory at time it
            h_bm = h_bm_list[it]
            u_it = u_it_list[it]
            v_it = v_it_list[it]
            h_it = h_it_list[it]
            h_tot = h_tot_list[it]
            adh_bm, adu_it, adv_it, adh_it, adh_tot, \
                adF_bm, adHe_mean, adalpha, adh_SN, adh_WE = self._jstep_adj_jit(t+it*self.dt, 
                                                        adh_bm, adu_it, adv_it, adh_it, adh_tot, 
                                                        adF_bm, adHe_mean, adalpha, adh_SN, adh_WE,
                                                        h_bm, u_it, v_it, h_it, h_tot, 
                                                        F_bm, He_mean, alpha, h_SN, h_WE,
                                                        h_bm_bc)
        
        # Update state and parameters
        adState.setvar([adh_bm, adu_it, adv_it, adh_it, adh_tot],[
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )

        adState.params[self.name_var['SSH_BM']] = adF_bm
        if 'He_mean' in self.name_params_it:
            adState.params['He_mean'] = adHe_mean
        if self.flag_coupling_from_bm and 'alpha' in self.name_params_it:
            adState.params['alpha'] = adalpha
        if 'hbc' in self.name_params_it:
            adState.params['hbcx'] = adh_SN
            adState.params['hbcy'] = adh_WE
    
    
###############################################################################
#                             Multi-models                                    #
###############################################################################      

class Model_multi:

    def __init__(self,config,State):

        # Initialize models
        self.Models = []
        _config = config.copy()

        for _MOD in config.MOD:
            _config.MOD = config.MOD[_MOD]
            self.Models.append(Model(_config,State))
            print()

        # Time parameters
        dts = [_Model.dt for _Model in self.Models]
        self.dt = int(np.max(dts)) # We take the longer timestep 
        if False:
            for _MOD,_Model,dt in zip(config.MOD,self.Models, dts):
                mult = int(np.ceil(self.dt / dt))
                adj_dt =  int(self.dt//mult)
                if adj_dt != dt:
                    print(f'Adjusting dt {dt} for {_MOD} -> {adj_dt} to be a divisor of dt {self.dt}')
                _Model.dt = adj_dt
        self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.timestamps = [] 
        t = config.EXP.init_date
        while t<=config.EXP.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)

        # Model variables: for each variable ('SSH', 'SST', 'Chl' etc...), 
        # we initialize a new variable for the sum of the different contributions
        self.name_var = {}
        self.name_var_tot = {}
        _name_var_tmp = []
        self.var_to_save = []
        for M in self.Models:
            self.var_to_save = np.concatenate((self.var_to_save, M.var_to_save))
            for name in M.name_var:
                if name not in _name_var_tmp:
                    _name_var_tmp.append(name)
                else:
                    # At least two component for the same variable, so we initialize a global variable
                    new_name = f'{name}_tot'
                    self.name_var[name] = new_name
                    self.name_var_tot[name] = new_name
                    # Initialize new State variable
                    if new_name in State.var:
                        State.var[new_name] += State.var[M.name_var[name]]
                    else:
                        State.var[new_name] = State.var[M.name_var[name]].copy()
                    if M.name_var[name] in M.var_to_save and new_name not in self.var_to_save:
                        self.var_to_save = np.append(self.var_to_save,new_name)
        self.var_to_save = list(self.var_to_save)

        for M in self.Models:
            for name in M.name_var:
                if name not in self.name_var_tot:
                    self.name_var[name] = M.name_var[name]

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            #for M in self.Models:
                #print('Tangent test:')
                #tangent_test(M,State,nstep=10)
                #print('Adjoint test:')
                #adjoint_test(M,State,nstep=10)
            print('MultiModel Tangent test:')
            tangent_test(self,State,nstep=1)
            print('MultiModel Adjoint test:')
            adjoint_test(self,State,nstep=1)

    def init(self,State,t0=0):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        for M in self.Models:
            M.init(State, t0=t0)
            for name in self.name_var:
                if name in M.name_var:
                    var_tot_tmp[name] += State.var[M.name_var[name]]

        # Update state
        for name in self.name_var:
            State.var[self.name_var[name]] = var_tot_tmp[name]

    def set_bc(self,time_bc,var_bc):

        for M in self.Models:
            M.set_bc(time_bc,var_bc)

    def save_output(self,State,present_date,name_var=None,t=None):

        for M in self.Models:
            M.save_output(State,present_date)
        
        State.save_output(present_date,name_var=self.var_to_save)

    def step(self,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = jnp.zeros_like(State.var[self.name_var[name]]) 
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Forward propagation
            M.step(State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and (name in self.name_var_tot):
                    var_tot_tmp[name] += +State.var[M.name_var[name]]
                    
        # Update state
        for name in self.name_var_tot:
            State.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_tgl(self,dState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Tangent propagation
            M.step_tgl(dState,State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and name in var_tot_tmp:
                    var_tot_tmp[name] += dState.var[M.name_var[name]]
        
        # Update state
        for name in self.name_var_tot:
            dState.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_adj(self,adState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = adState.var[self.name_var_tot[name]]
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Add to local variable
            for name in self.name_var:
                if name in M.name_var and name in self.name_var_tot:
                    adState.var[M.name_var[name]] += var_tot_tmp[name]  
            # Adjoint propagation
            M.step_adj(adState,State,nstep=_nstep,t=t)
        
        for name in self.name_var_tot:
            adState.var[self.name_var_tot[name]] *= 0 
                

###############################################################################
#                       Tangent and Adjoint tests                             #
###############################################################################     

def tangent_test(M,State,t0=0,nstep=1, ampl=1):

    # Boundary conditions
    var_bc = {}

    for name in M.name_var:
        var_bc[name] = {0:ampl*np.random.random(State.var[M.name_var[name]].shape).astype('float64'),
                        1:ampl*np.random.random(State.var[M.name_var[name]].shape).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)

    State0 = State.random(ampl=ampl)
    dState = State.random(ampl=ampl)
    State0_tmp = State0.copy()
    
    M.step(t=t0,State=State0_tmp,nstep=nstep)
    X2 = State0_tmp.getvar(vect=True) 
    
    for p in range(10):
        
        lambd = 10**(-p)
        
        State1 = dState.copy()
        State1.scalar(lambd)
        State1.Sum(State0)

        M.step(t=t0,State=State1,nstep=nstep)
        X1 = State1.getvar(vect=True)
        
        dState1 = dState.copy()
        dState1.scalar(lambd)
        M.step_tgl(t=t0,dState=dState1,State=State0,nstep=nstep)

        dX = dState1.getvar(vect=True)
        
        mask = np.isnan(X1+X2+dX)
        
        ps = np.linalg.norm(X1[~mask]-X2[~mask]-dX[~mask])/np.linalg.norm(dX[~mask])
    
        print('%.E' % lambd,'%.E' % ps)
        
def adjoint_test(M, State, t0=0, nstep=1, ampl=1):

    model_dtype = getattr(M, 'dtype', np.float64)
    np_dtype = np.float32 if model_dtype == jnp.float32 else np.float64

    # ------------------------------------------------------------------
    # Build per-shape ocean masks so that land points stay at zero.
    # Non-zero land values create large gradients at land-ocean boundaries
    # that amplify float32 roundoff in WENO/padding operations, breaking
    # the inner-product identity even though the operators are exact duals.
    # ------------------------------------------------------------------
    masks_by_shape = {}
    if State.mask is not None:
        ny, nx = State.mask.shape
        mask_h = np.asarray(State.mask, dtype=bool)
        masks_by_shape[(ny, nx)] = mask_h.astype(np_dtype)
        # U-grid mask: (ny, nx+1)
        mask_u = np.zeros((ny, nx + 1), dtype=np_dtype)
        mask_u[:, 1:nx] = (mask_h[:, :-1] & mask_h[:, 1:]).astype(np_dtype)
        masks_by_shape[(ny, nx + 1)] = mask_u
        # V-grid mask: (ny+1, nx)
        mask_v = np.zeros((ny + 1, nx), dtype=np_dtype)
        mask_v[1:ny, :] = (mask_h[:-1, :] & mask_h[1:, :]).astype(np_dtype)
        masks_by_shape[(ny + 1, nx)] = mask_v

    # ------------------------------------------------------------------
    # Use DIFFERENT seeds for trajectory / perturbation / adjoint vector.
    # State.random() always resets np.random.seed(0), so calling it three
    # times produces three IDENTICAL states.  With identical dState and
    # adState the inner-product test degenerates and f32 rounding errors
    # no longer cancel, giving a spurious ~0.1 % deviation from 1.0.
    # ------------------------------------------------------------------
    def _apply_mask(arr):
        m = masks_by_shape.get(arr.shape)
        return arr * m if m is not None else arr

    def make_rand_state(seed):
        np.random.seed(seed)
        s = State.copy(free=True)
        for name in s.var:
            s.var[name] = _apply_mask(
                (ampl * np.random.random(s.var[name].shape)).astype(np_dtype))
        for name in s.params:
            s.params[name] = _apply_mask(
                (ampl * np.random.random(s.params[name].shape)).astype(np_dtype))
        return s

    # Boundary conditions
    np.random.seed(99)
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {
            0: _apply_mask(ampl * np.random.random(State.var[M.name_var[name]].shape).astype(np_dtype)),
            1: _apply_mask(ampl * np.random.random(State.var[M.name_var[name]].shape).astype(np_dtype)),
        }
    M.set_bc([t0, t0 + nstep * M.dt], var_bc)

    State0  = make_rand_state(seed=10)   # trajectory
    dState  = make_rand_state(seed=20)   # TLM perturbation
    adState = make_rand_state(seed=30)   # ADJ input

    # Snapshot before TLM / ADJ
    dX0  = np.concatenate((dState.getvar(vect=True), dState.getparams(vect=True)))
    adX0 = np.concatenate((adState.getvar(vect=True), adState.getparams(vect=True)))

    # Run TLM
    M.step_tgl(t=t0, dState=dState, State=State0, nstep=nstep)
    dX1  = np.concatenate((dState.getvar(vect=True), dState.getparams(vect=True)))

    # Run ADJ
    M.step_adj(t=t0, adState=adState, State=State0, nstep=nstep)
    adX1 = np.concatenate((adState.getvar(vect=True), adState.getparams(vect=True)))

    mask = np.isnan(adX0 + dX0 + adX1 + dX1)

    # Compute inner products in f64 for accurate accumulation
    ps1 = np.inner(dX1[~mask], adX0[~mask])
    ps2 = np.inner(dX0[~mask], adX1[~mask])

    print(f'  adjoint_test ({np_dtype.__name__}):  <Mdx,y> = {ps1} <dx,M*y> = {ps2} \n<Mdx,y>/<dx,M*y> = {ps1/ps2}')

    
    

    
    
    
    
