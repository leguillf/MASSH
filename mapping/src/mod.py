#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:36:20 2021

@author: leguillou
"""

from importlib.machinery import SourceFileLoader 
import sys
import xarray as xr
import numpy as np
import os
from math import pi
from datetime import timedelta
import matplotlib.pylab as plt 

import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

from . import  grid

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
        if config.MOD.super=='MOD_DIFF':
            return Model_diffusion(config,State)
        elif config.MOD.super=='MOD_QG1L_NP':
            return Model_qg1l_np(config,State)
        elif config.MOD.super=='MOD_QG1L_JAX':
            return Model_qg1l_jax(config,State)
        elif config.MOD.super=='MOD_SW1L_NP':
            return Model_sw1l_np(config,State)
        elif config.MOD.super=='MOD_SW1L_JAX':
            return Model_sw1l_jax(config,State)
        elif config.MOD.super=='MOD_TRAC':
            return Model_trac(config,State)
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    
class M:

    def __init__(self,config,State):
        
        # Time parameters
        self.dt = config.MOD.dtmodel
        self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.EXP.init_date
        while t<=config.EXP.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        self.timestamps = np.asarray(self.timestamps)

        # Model variables
        self.name_var = config.MOD.name_var
        if config.MOD.var_to_save is not None:
            self.var_to_save = config.MOD.var_to_save 
        else:
            self.var_to_save = []
            for name in self.name_var:
                self.var_to_save.append(self.name_var[name])


    def init(self, State):
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
    

###############################################################################
#                            Diffusion Models                                 #
###############################################################################
        
class Model_diffusion(M):
    
    def __init__(self,config,State):

        super().__init__(config,State)
        
        self.Kdiffus = config.MOD.Kdiffus
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
            Wbc[0,:] = 1 # Border
            Wbc[:,0] = 1 # Border
            Wbc[-1,:] = 1 # Border
            Wbc[:,-1] = 1 # Border
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
            if t in self.bc[name]:
                State.var[self.name_var[name]] +=\
                    self.Wbc * (self.bc[name][t]-self.bc[name][t0]) 


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
        
        # Boundary conditions
        self._apply_bc(State,t,t+nstep*self.dt)

        
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
    
class Model_qg1l_np(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        if config.MOD.use_jax:
            qgm = SourceFileLoader("qgm",dir_model + "/jqgm.py").load_module() 
            model = qgm.Qgm
        else:
            SourceFileLoader("qgm",dir_model + "/qgm.py").load_module() 
            SourceFileLoader("qgm_tgl", 
                                    dir_model + "/qgm_tgl.py").load_module() 
            
            qgm = SourceFileLoader("qgm_adj", 
                                        dir_model + "/qgm_adj.py").load_module() 
            model = qgm.Qgm_adj

        # Coriolis
        self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)

        # Gravity
        self.g = config.MOD.g
        
        # Open MDT map if provided
        self.hbc = self.mdt = self.mdu = self.mdv = None
        if (config.MOD.Reynolds or config.MOD.use_mdt_on_borders) and config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt).squeeze()
            ds.load()
            
            name_var_mdt = {}
            name_var_mdt['lon'] = config.MOD.name_var_mdt['lon']
            name_var_mdt['lat'] = config.MOD.name_var_mdt['lat']
            
            
            
            if 'mdt' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdt'] in ds:
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdt']
                mdt = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                
                #self.mdt[np.isnan(self.mdt)] = 0
                if config.EXP.flag_plot>0:
                    plt.figure()
                    plt.pcolormesh(mdt)
                    plt.show()
            else:
                sys.exit('Warning: wrong variable name for mdt')
            if 'mdu' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdu'] in ds \
                and 'mdv' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdv'] in ds:
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdu']
                mdu = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdv']
                mdv = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                mdu = mdv = None
             
            if config.MOD.Reynolds:
                print('MDT is prescribed, thus the QGPV will be expressed thanks \
                to Reynolds decomposition. However, be sure that observed and boundary \
                variable are SLAs!')
                self.hbc = None
                self.mdt = mdt
                self.mdu = mdu
                self.mdv = mdv
            elif config.MOD.use_mdt_on_borders: 
                self.hbc = mdt
         
        # Open Rossby Radius if provided
        if self.mdt is not None and config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.MOD.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.cmin is not None:
                self.c[self.c<config.cmin] = config.cmin
            
            if config.cmax is not None:
                self.c[self.c>config.cmax] = config.cmax
            
            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
            
        # Initialize model state
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
                if State.mask is not None:
                    State.var[self.name_var[name]][State.mask] = np.nan

        # Observed variable
        self.name_obs_var = self.name_var['SSH'] # SSH

        # Initialize model Parameters (Flux)
        self.nparams = State.ny*State.nx
        self.sliceparams = slice(0,self.nparams)

        # Boundary conditions
        self.SSHb = {}
        self.Wbc = np.zeros((State.ny,State.nx))
        if config.MOD.dist_sponge_bc is not None:
            self.Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
        
        # Model initialization
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=State.getvar(name_var=self.name_var['SSH']),
                         c=self.c,
                         upwind=config.MOD.upwind,
                         upwind_adj=config.MOD.upwind_adj,
                         time_scheme=config.MOD.time_scheme,
                         g=config.MOD.g,
                         f=self.f,
                         hbc=self.hbc,
                         qgiter=config.MOD.qgiter,
                         qgiter_adj=config.MOD.qgiter_adj,
                         diff=config.MOD.only_diffusion,
                         Kdiffus=config.MOD.Kdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)
                         
        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State)
            print('Adjoint test:')
            adjoint_test(self,State)
        
    def set_bc(self,time_bc,var_bc):
        
        for var in var_bc:
            if var=='SSH':
                for i,t in enumerate(time_bc):
                    self.SSHb[t] = var_bc[var][i]

    def step(self,State,nstep=1,t=None):
        
        # Get state variable
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
        
        # init
        SSH1 = +SSH0

        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step(SSH1)
        
        # Update state
        if self.name_var['SSH'] in State.params:
            params = State.params[self.name_var['SSH']]
            SSH1 += nstep*self.dt/(3600*24) * params
        State.setvar(SSH1, name_var=self.name_var['SSH'])

    def step_nudging(self,State,tint,Nudging_term=None,t=None):
    
        # Read state variable
        ssh_0 = State.getvar(name_var=self.name_var['SSH'])
        
        if 'PV' in self.name_var:
            flag_pv = True
            pv_0 = State.getvar(name_var=self.name_var['PV'])
        else:
            flag_pv = False
            pv_0 = self.qgm.h2pv(ssh_0)

        # Boundary condition
        if t in self.SSHb:
            Qbc = self.qgm.h2pv(self.SSHb[t])
            ssh_0 = self.Wbc*self.SSHb[t] + (1-self.Wbc)*ssh_0
            pv_0 = self.Wbc*Qbc + (1-self.Wbc)*pv_0
        
        # Model propagation
        deltat = np.abs(tint)
        way = np.sign(tint)
        t = 0
        ssh_1 = +ssh_0
        pv_1 = +pv_0
        while t<deltat:
            ssh_1, pv_1 = self.qgm.step(h0=ssh_1, q0=pv_1, way=way)
            t += self.dt
            
        # Nudging
        if Nudging_term is not None:
            # Nudging towards relative vorticity
            if np.any(np.isfinite(Nudging_term['rv'])):
                indNoNan = (~np.isnan(Nudging_term['rv'])) & (self.qgm.mask>1) 
                pv_1[indNoNan] += (1-self.Wbc[indNoNan]) *\
                    Nudging_term['rv'][indNoNan]
            # Nudging towards ssh
            if np.any(np.isfinite(Nudging_term['ssh'])):
                indNoNan = (~np.isnan(Nudging_term['ssh'])) & (self.qgm.mask>1) 
                pv_1[indNoNan] -= (1-self.Wbc[indNoNan]) *\
                    (self.g*self.f[indNoNan])/self.c[indNoNan]**2 * \
                        Nudging_term['ssh'][indNoNan]
                # Inversion pv -> ssh
                ssh_b = +ssh_1
                ssh_1[indNoNan] = self.qgm.pv2h(pv_1,ssh_b)[indNoNan]
        
        if np.any(np.isnan(ssh_1[self.qgm.mask>1])):
            if t in self.SSHb:
                ind = (np.isnan(ssh_1)) & (self.qgm.mask>1)
                ssh_1[ind] = self.SSHb[t][ind] 
                print('Warning: Invalid value encountered in mod_qg1l, we replace by boundary values')
            else: sys.exit('Invalid value encountered in mod_qg1l')
            
        # Update state 
        State.setvar(ssh_1,name_var=self.name_var['SSH'])
        if flag_pv:
            State.setvar(pv_1,name_var=self.name_var['PV']) 

    def step_tgl(self,dState,State,nstep=1,t=None):
        
        # Get state variable
        dSSH0 = dState.getvar(name_var=self.name_var['SSH'])
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
        
        # init
        dSSH1 = +dSSH0
        SSH1 = +SSH0
        
        # Time propagation
        for i in range(nstep):
            dSSH1 = self.qgm.step_tgl(dh0=dSSH1,h0=SSH1)
            SSH1 = self.qgm.step(h0=SSH1)
        
        # Update state
        if dState.params is not None:
            dparams = dState.params[self.sliceparams].reshape((State.ny,State.nx))
            dSSH1 += nstep*self.dt/(3600*24) * dparams
        dState.setvar(dSSH1,name_var=self.name_var['SSH'])
        
    def step_adj(self,adState,State,nstep=1,t=None):
        
        # Get state variable
        adSSH0 = adState.getvar(self.name_var['SSH'])
        SSH0 = State.getvar(self.name_var['SSH'])
        
        # Init
        adSSH1 = +adSSH0
        SSH1 = +SSH0

        # Current trajectory
        traj = [SSH1]
        if nstep>1:
            for i in range(nstep):
                SSH1 = self.qgm.step(SSH1)
                traj.append(SSH1)
        
        # Time propagation
        for i in reversed(range(nstep)):
            SSH1 = traj[i]
            adSSH1 = self.qgm.step_adj(adSSH1,SSH1)

        # Convert to numpy
        if self.jax:
            adSSH1 = np.array(adSSH1)

        # Update state  and parameters
        if adState.params is not None:
            adState.params[self.sliceparams] += nstep*self.dt/(3600*24) * adSSH0.flatten()
        adSSH1[np.isnan(adSSH1)] = 0
        adState.setvar(adSSH1,self.name_var['SSH'])

class Model_qg1l_jax(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",dir_model + "/jqgm.py").load_module() 
        model = qgm.Qgm

        # Coriolis
        self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = config.MOD.g
        
        # Open MDT map if provided
        self.mdt = self.mdu = self.mdv = None
        if config.MOD.Reynolds  and config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt).squeeze()
            ds.load()
            
            name_var_mdt = {}
            name_var_mdt['lon'] = config.MOD.name_var_mdt['lon']
            name_var_mdt['lat'] = config.MOD.name_var_mdt['lat']
            
            if 'mdt' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdt'] in ds:
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdt']
                self.mdt = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                
                if config.EXP.flag_plot>0:
                    plt.figure()
                    plt.pcolormesh(self.mdt)
                    plt.show()
            else:
                sys.exit('Warning: wrong variable name for mdt')
            if 'mdu' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdu'] in ds \
                and 'mdv' in config.MOD.name_var_mdt and config.MOD.name_var_mdt['mdv'] in ds:
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdu']
                self.mdu = grid.interp2d(ds,
                                    name_var_mdt,
                                    State.lon,
                                    State.lat)
                name_var_mdt['var'] = config.MOD.name_var_mdt['mdv']
                self.mdv = grid.interp2d(ds,
                                    name_var_mdt,
                                    State.lon,
                                    State.lat)
            
        # Open Rossby Radius if provided
        if self.mdt is not None and config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.MOD.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.cmin is not None:
                self.c[self.c<config.cmin] = config.cmin
            
            if config.cmax is not None:
                self.c[self.c>config.cmax] = config.cmax
            
            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
            
        # Initialize model state
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
                if State.mask is not None:
                    State.var[self.name_var[name]][State.mask] = np.nan

        # Initialize model Parameters (Flux on SSH and tracers)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # Use boundary conditions as mean field (for 4Dvar only)
        if config.INV is not None and config.INV.super=='INV_4DVAR':
            self.anomaly_from_bc = config.INV.anomaly_from_bc
        else:
            self.anomaly_from_bc = False

        # Tracer advection flag
        self.advect_tracer = config.MOD.advect_tracer

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
                         g=config.MOD.g,
                         f=self.f,
                         diff=config.MOD.only_diffusion,
                         Kdiffus=config.MOD.Kdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)

        # Model functions initialization
        if config.INV is not None and config.INV.super in ['INV_4DVAR','INV_4DVAR_PARALLEL']:
            self.qgm_step = self.qgm.step_jit
            self.qgm_step_tgl = self.qgm.step_tgl_jit
            self.qgm_step_adj = self.qgm.step_adj_jit
        else:
            self.qgm_step = self.qgm.step
        
        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=10)
            print('Adjoint test:')
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

    def set_bc(self,time_bc,var_bc):

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        self.bc[_name_var_mod][t] = var_bc[_name_var_bc][i]

    def ano_bc(self,t,State,sign):

        if not self.anomaly_from_bc:
            return
        else:
            for name in self.name_var:
                if t in self.bc[name]:
                    State.var[self.name_var[name]] += sign * self.bc[name][t]
            

    def _apply_bc(self,t,t1):
        
        Xb = np.zeros((self.ny,self.nx,))

        if 'SSH' not in self.bc:
            return Xb
        elif t not in self.bc['SSH']:
            return Xb
        else:
            Xb = self.bc['SSH'][t]
            if self.advect_tracer:
                for name in self.name_var:
                    if name!='SSH':
                        if t1 in self.bc[name]: 
                            Cb = self.bc[name][t1]
                        else:
                            Cb = np.zeros((self.ny,self.nx,))
                        Xb = np.append(Xb[np.newaxis,:,:], 
                                       Cb[np.newaxis,:,:], axis=0)     
        return Xb
    
    def step(self,State,nstep=1,t=None):

        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable(s)
        X0 = State.getvar(name_var=self.name_var['SSH'])
        if self.advect_tracer:
            X0 = X0[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH':
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)
        t1 = t+nstep*self.dt

        # Convert to numpy array
        X1 = np.array(X1)
        
        # Update state
        if self.name_var['SSH'] in State.params:
            Fssh = State.params[self.name_var['SSH']] # Forcing term for SSH
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh
                State.setvar(X1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = State.params[self.name_var[name]] # Forcing term for tracer
                        X1[i] += nstep*self.dt/(3600*24) * Fc
                        State.setvar(X1[i], name_var=self.name_var[name])
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State.setvar(X1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t1,State,-1)
    

    def step_tgl(self,dState,State,nstep=1,t=None):

        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))
        
        # Get state variable
        dX0 = dState.getvar(name_var=self.name_var['SSH'])
        X0 = State.getvar(name_var=self.name_var['SSH'])
        if self.advect_tracer:
            dX0 = dX0[np.newaxis,:,:]
            X0 = X0[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH':
                    dC0 = dState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    dX0 = np.append(dX0, dC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        dX1 = +dX0
        X1 = +X0

        # Time propagation
        dX1 = self.qgm_step_tgl(dX1,X1,Xb=Xb,nstep=nstep)

        # Convert to numpy and reshape
        dX1 = np.array(dX1)

        # Update state
        if self.name_var['SSH'] in dState.params:
            dFssh = dState.params[self.name_var['SSH']] # Forcing term for SSH
            if self.advect_tracer:
                dX1[0] += nstep*self.dt/(3600*24) * dFssh
                dState.setvar(dX1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        dFc = dState.params[self.name_var[name]] # Forcing term for tracer
                        dX1[i] += nstep*self.dt/(3600*24) * dFc
                        dState.setvar(dX1[i], name_var=self.name_var[name])
            else:
                dX1 += nstep*self.dt/(3600*24) * dFssh
                dState.setvar(dX1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t,State,-1)
        

    def step_adj(self,adState,State,nstep=1,t=None):
        
        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable
        adSSH0 = adState.getvar(name_var=self.name_var['SSH'])
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
        if self.advect_tracer:
            adX0 = adSSH0[np.newaxis,:,:]
            X0 = SSH0[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH':
                    adC0 = adState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    adX0 = np.append(adX0, adC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        else:
            adX0 = adSSH0
            X0 = SSH0
        
        # Init
        adX1 = +adX0
        X1 = +X0

        # Time propagation
        adX1 = self.qgm_step_adj(adX1,X1,Xb,nstep=nstep)

        # Convert to numpy and reshape
        adX1 = np.array(adX1).squeeze()

        # Update state and parameters
        if self.name_var['SSH'] in adState.params:
            for name in self.name_var:
                adState.params[self.name_var[name]] += nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name])
                
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

class Model_sw1l_np(M):
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
        
        SourceFileLoader("obcs", 
                                dir_model+"/obcs.py").load_module() 
        SourceFileLoader("obcs_tgl", 
                                dir_model + "/obcs_tgl.py").load_module() 
        SourceFileLoader("obcs_adj", 
                                dir_model + "/obcs_adj.py").load_module() 
        SourceFileLoader("swm", 
                                dir_model + "/swm.py").load_module() 
        SourceFileLoader("swm_tgl", 
                                dir_model + "/swm_tgl.py").load_module() 
        
        swm = SourceFileLoader("swm_adj", 
                                dir_model + "/swm_adj.py").load_module() 
        model = swm.Swm_adj
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx
        
        # Coriolis
        self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)

        # Gravity
        self.g = config.MOD.g
             
        # Equivalent depth
        if config.MOD.He_data is not None and os.path.exists(config.MOD.He_data['path']):
            ds = xr.open_dataset(config.MOD.He_data['path'])
            self.Heb = ds[config.MOD.He_data['var']].values
        else:
            self.Heb = config.MOD.He_init
            
            
        if config.MOD.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
        else:
            self.bc_theta = np.array([0])
            
        self.omegas = np.asarray(config.MOD.w_waves)
        self.bc_kind = config.MOD.bc_kind

        
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

        
        # Model Parameters (OBC & He)
        self.shapeHe = [State.ny,State.nx]
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
        self.sliceHe = slice(0,np.prod(self.shapeHe))
        self.slicehbcx = slice(np.prod(self.shapeHe),
                               np.prod(self.shapeHe)+np.prod(self.shapehbcx))
        self.slicehbcy = slice(np.prod(self.shapeHe)+np.prod(self.shapehbcx),
                               np.prod(self.shapeHe)+np.prod(self.shapehbcx)+np.prod(self.shapehbcy))
        self.nparams = np.prod(self.shapeHe)+np.prod(self.shapehbcx)+np.prod(self.shapehbcy)
        State.params['He'] = np.zeros((self.shapeHe))
        State.params['hbcx'] = np.zeros((self.shapehbcx))
        State.params['hbcy'] = np.zeros((self.shapehbcy))
        
        # Model initialization
        self.swm = model(X=State.X,
                        Y=State.Y,
                        dt=self.dt,
                        bc=self.bc_kind,
                        omegas=self.omegas,
                        bc_theta=self.bc_theta,
                        f=self.f)
        
        # Functions related to time_scheme
        if self.time_scheme=='Euler':
            self.swm_step = self.swm.step_euler
            self.swm_step_tgl = self.swm.step_euler_tgl
            self.swm_step_adj = self.swm.step_euler_adj
        elif self.time_scheme=='rk4':
            self.swm_step = self.swm.step_rk4
            self.swm_step_tgl = self.swm.step_rk4_tgl
            self.swm_step_adj = self.swm.step_rk4_adj

        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=10)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=10)

        
    def step(self,State,nstep=1,t0=0,t=0):

        # Init
        u0 = State.getvar(self.name_var['U'])
        v0 = State.getvar(self.name_var['V'])
        h0 = State.getvar(self.name_var['SSH'])
        u = +u0
        v = +v0
        h = +h0
        
        # Get params in physical space
        if State.params is not None:
            He = State.params['He'].reshape(self.shapeHe) + self.Heb
            hbcx = State.params['hbcx'].reshape(self.shapehbcx)
            hbcy = State.params['hbcy'].reshape(self.shapehbcy)
        else:
            He = hbcx = hbcy = None
        
        # Time propagation
        for i in range(nstep):
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            u,v,h = self.swm_step(
                t+i*self.dt,
                u,v,h,He=He,hbcx=hbcx,hbcy=hbcy,first=first)
            
        State.setvar(u,self.name_var['U'])
        State.setvar(v,self.name_var['V'])
        State.setvar(h,self.name_var['SSH'])
        
    def step_tgl(self,dState,State,nstep=1,t0=0,t=0):
        
        # Get state variables and model parameters
        du0 = dState.getvar(self.name_var['U'])
        dv0 = dState.getvar(self.name_var['V'])
        dh0 = dState.getvar(self.name_var['SSH'])
        u0 = State.getvar(self.name_var['U'])
        v0 = State.getvar(self.name_var['V'])
        h0 = State.getvar(self.name_var['SSH'])
        
        if State.params is not None:
            He = State.params['He'].reshape(self.shapeHe) + self.Heb
            hbcx = State.params['hbcx'].reshape(self.shapehbcx)
            hbcy = State.params['hbcy'].reshape(self.shapehbcy)
        else:
            He = hbcx = hbcy = None
            
        if dState.params is not None:
            dHe = dState.params['He'].reshape(self.shapeHe) 
            dhbcx = dState.params['hbcx'].reshape(self.shapehbcx)
            dhbcy = dState.params['hbcy'].reshape(self.shapehbcy)
        else:
            dHe = dhbcx = dhbcy = None
    
        du = +du0
        dv = +dv0
        dh = +dh0
        u = +u0
        v = +v0
        h = +h0
        
        # Time propagation
        # Current trajectory
        traj = [(u,v,h)]
        if nstep>1:
            for i in range(nstep):
                if t+i*self.dt==t0:
                        first = True
                else: first = False
                u,v,h = self.swm_step(
                        t+i*self.dt,
                        u,v,h,He=He,hbcx=hbcx,hbcy=hbcy,first=first)
                traj.append((u,v,h))
            
        for i in range(nstep):
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            u,v,h = traj[i]
            
            du,dv,dh = self.swm_step_tgl(
                t+i*self.dt,du,dv,dh,u,v,h,
                dHe=dHe,He=He,
                dhbcx=dhbcx,dhbcy=dhbcy,hbcx=hbcx,hbcy=hbcy,first=first)
            
        dState.setvar(du,self.name_var['U'])
        dState.setvar(dv,self.name_var['V'])
        dState.setvar(dh,self.name_var['SSH'])
        
    def step_adj(self,adState, State, nstep=1, t0=0,t=0):
        
        # Get variables
        adu0 = adState.getvar(self.name_var['U'])
        adv0 = adState.getvar(self.name_var['V'])
        adh0 = adState.getvar(self.name_var['SSH'])
        u0 = State.getvar(self.name_var['U'])
        v0 = State.getvar(self.name_var['V'])
        h0 = State.getvar(self.name_var['SSH'])
        
        if State.params is not None:
            He = State.params['He'].reshape(self.shapeHe) + self.Heb
            hbcx = State.params['hbcx'].reshape(self.shapehbcx)
            hbcy = State.params['hbcy'].reshape(self.shapehbcy)
        else:
            He = hbcx = hbcy = None
        
        # Init
        adu = +adu0
        adv = +adv0
        adh = +adh0
        u = +u0
        v = +v0
        h = +h0
        adHe = He*0
        adhbcx = hbcx*0
        adhbcy = hbcy*0
        
        # Time propagation
        # Current trajectory
        traj = [(u,v,h)]
        if nstep>1:
            for i in range(nstep):
                if t+i*self.dt==t0:
                        first = True
                else: first = False
                u,v,h = self.swm_step(
                        t+i*self.dt,
                        u,v,h,He=He,hbcx=hbcx,hbcy=hbcy,first=first)
                traj.append((u,v,h))
            
        for i in reversed(range(nstep)):
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            u,v,h = traj[i]
        
            adu,adv,adh,adHe_tmp,adhbcx_tmp,adhbcy_tmp =\
                self.swm_step_adj(t+i*self.dt,adu,adv,adh,u,v,h,
                                      He,hbcx,hbcy,first=first)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
            
        # Update state
        adState.setvar(adu,self.name_var['U'])
        adState.setvar(adv,self.name_var['V'])
        adState.setvar(adh,self.name_var['SSH'])
        
        # Update parameters
        adState.params['He'] += adHe
        adState.params['hbcx'] += adhbcx
        adState.params['hbcy'] += adhbcy
    
class Model_sw1l_jax(M):
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
        model = swm.Swm
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx

        # Coriolis
        self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = config.MOD.g
             
        # Equivalent depth
        if config.MOD.He_data is not None and os.path.exists(config.MOD.He_data['path']):
            ds = xr.open_dataset(config.MOD.He_data['path'])
            self.Heb = ds[config.MOD.He_data['var']].values
        else:
            self.Heb = config.MOD.He_init
            
            
        if config.MOD.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
        else:
            self.bc_theta = np.array([0])
            
        self.omegas = np.asarray(config.MOD.w_waves)
        self.bc_kind = config.MOD.bc_kind

        
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

        
        # Model Parameters (OBC & He)
        self.shapeHe = [State.ny,State.nx]
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
        self.sliceHe = slice(0,np.prod(self.shapeHe))
        self.slicehbcx = slice(np.prod(self.shapeHe),
                               np.prod(self.shapeHe)+np.prod(self.shapehbcx))
        self.slicehbcy = slice(np.prod(self.shapeHe)+np.prod(self.shapehbcx),
                               np.prod(self.shapeHe)+np.prod(self.shapehbcx)+np.prod(self.shapehbcy))
        self.nparams = np.prod(self.shapeHe)+np.prod(self.shapehbcx)+np.prod(self.shapehbcy)
        State.params['He'] = np.zeros((self.shapeHe))
        State.params['hbcx'] = np.zeros((self.shapehbcx))
        State.params['hbcy'] = np.zeros((self.shapehbcy))
        
        # Model initialization
        self.swm = model(X=State.X,
                        Y=State.Y,
                        dt=self.dt,
                        bc=self.bc_kind,
                        omegas=self.omegas,
                        bc_theta=self.bc_theta,
                        f=self.f)
        
        
        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep)
        self._jstep_tgl_jit = jit(self._jstep_tgl)
        self._jstep_adj_jit = jit(self._jstep_adj)
        self._compute_w1_IT_jit = jit(self._compute_w1_IT)
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
            print('Tangent test:')
            tangent_test(self,State,nstep=10)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=10)

    
    def step(self,State,nstep=1,t=0):

        # Get state variable
        X0 = +State.getvar(
            [self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']],vect=True)
        
        # Remove NaN
        X0[np.isnan(X0)] = np.nan

        # Get params in physical space
        if State.params is not None:
            params = +State.getparams(['He','hbcx','hbcy'],vect=True)
            X0 = np.concatenate((X0,params))

        # Init
        X1 = +X0
        # Add time in control vector (for JAX)
        X1 = np.append(t,X1)
        # Time stepping
        for _ in range(nstep):
            # One time step
            X1 = self._jstep_jit(X1)
        
        # Remove time in control vector
        X1 = X1[1:]
        
        # Convert to numpy and reshape
        u1 = np.array(X1[self.swm.sliceu]).reshape(self.swm.shapeu)
        v1 = np.array(X1[self.swm.slicev]).reshape(self.swm.shapev)
        h1 = np.array(X1[self.swm.sliceh]).reshape(self.swm.shapeh)
        
        State.setvar([u1,v1,h1],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def _jstep(self,X0):
        
        t,X1 = X0[0],jnp.asarray(+X0[1:])
        
        # Get He,obcs parameters
        params = None
        if X1.size==self.swm.nstates+self.nparams:
            params = X1[self.swm.nstates:]
            He = +params[self.sliceHe].reshape(self.shapeHe)+self.Heb
            hbcx = +params[self.slicehbcx].reshape(self.shapehbcx)
            hbcy = +params[self.slicehbcy].reshape(self.shapehbcy)        
        
        # Time propagation
        _X1 = +X1[:self.swm.nstates]
        if params is not None:
            # First characteristic variables w1 from external data
            if self.bc_kind=='1d':
                tbc = t + self.dt
            else:
                tbc = t

            w1S,w1N,w1W,w1E = self._compute_w1_IT_jit(tbc,He,hbcx,hbcy)
            
            w1ext = jnp.concatenate((w1S,w1N,w1W,w1E))
            _X1 = jnp.concatenate((_X1, # State variables
                                   He.flatten(),w1ext)) # Model parameters 
        # One forward step
        _X1 = self.swm_step(_X1)
        
        # Retrieve inital form
        X1 = X1.at[:self.swm.nstates].set(_X1[:self.swm.nstates])
        
        if params is not None:
            X1 = X1.at[self.swm.nstates:].set(params)
        
        X1 = jnp.append(jnp.array(t+self.dt),X1)
    
        return X1
        
    def step_tgl(self,dState,State,nstep=1,t=0):
        
        # Get state variable
        dX0 = +dState.getvar(
            [self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']],vect=True)
        X0 = +State.getvar(
            [self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']],vect=True)
        
        # Get params in physical space
        if State.params is not None:
            dparams = +dState.getparams(['He','hbcx','hbcy'],vect=True)
            dX0 = np.concatenate((dX0,dparams))
            params = +State.getparams(['He','hbcx','hbcy'],vect=True)
            X0 = np.concatenate((X0,params))         

        # Init
        dX1 = +dX0
        X1 = +X0
        # Add time in control vector (for JAX)
        dX1 = np.append(t,dX1)
        X1 = np.append(t,X1)
        # Time stepping
        for i in range(nstep):
            # One timestep
            dX1 = self._jstep_tgl_jit(dX1,X1)
            if i<nstep-1:
                X1 = self._jstep_jit(X1)
                
        # Remove time in control vector
        dX1 = dX1[1:]
        
        # Reshaping
        du1 = np.array(dX1[self.swm.sliceu]).reshape(self.swm.shapeu)
        dv1 = np.array(dX1[self.swm.slicev]).reshape(self.swm.shapev)
        dh1 = np.array(dX1[self.swm.sliceh]).reshape(self.swm.shapeh)
        
        dState.setvar([du1,dv1,dh1],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def _jstep_tgl(self,dX0,X0):
        
        _,dX1 = jvp(self._jstep_jit, (X0,), (dX0,))
        
        return dX1
    
    def step_adj(self,adState, State, nstep=1,t=0):
        
        # Get state variable
        adX0 = +adState.getvar(
            [self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']],vect=True)
        X0 = +State.getvar(
            [self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']],vect=True)
        
        # Remove NaN
        X0[np.isnan(X0)] = np.nan
        adX0[np.isnan(adX0)] = np.nan
        
        # Get params in physical space
        if State.params is not None:
            adparams = +adState.getparams(['He','hbcx','hbcy'],vect=True)
            adX0 = np.concatenate((adX0,adparams))
            params = +State.getparams(['He','hbcx','hbcy'],vect=True)
            X0 = np.concatenate((X0,params))         
        
        # Init
        adX1 = +adX0
        X1 = +X0
        
        # Current trajectory
        # Add time in control vector (for JAX)
        X1 = np.append(t,X1)
        traj = [X1]
        if nstep>1:
            for i in range(nstep):
                # One timestep
                X1 = self._jstep_jit(X1)
                if i<nstep-1:
                    traj.append(+X1)
            
        # Reversed time propagation
        # Add time in control vector (for JAX)
        adX1 = np.append(traj[-1][0],adX1)
        for i in reversed(range(nstep)):
            X1 = traj[i]
            # One timestep
            adX1 = self._jstep_adj_jit(adX1,X1)
        
        # Remove time in control vector
        adX1 = adX1[1:]
        
        # Reshaping
        adu1 = np.array(adX1[self.swm.sliceu]).reshape(self.swm.shapeu)
        adv1 = np.array(adX1[self.swm.slicev]).reshape(self.swm.shapev)
        adh1 = np.array(adX1[self.swm.sliceh]).reshape(self.swm.shapeh)
        adparams = np.array(adX1[self.swm.nstates:])
        adHe = +adparams[self.sliceHe].reshape(self.shapeHe)
        adhbcx = +adparams[self.slicehbcx].reshape(self.shapehbcx)
        adhbcy = +adparams[self.slicehbcy].reshape(self.shapehbcy)        
        
        # Update state
        adu1[np.isnan(adu1)] = 0
        adv1[np.isnan(adv1)] = 0
        adh1[np.isnan(adh1)] = 0
        adState.setvar([adu1,adv1,adh1],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
        # Update parameters
        adState.params['He'] = adHe
        adState.params['hbcx'] = adhbcx
        adState.params['hbcy'] = adhbcy
    
    def _jstep_adj(self,adX0,X0):
        
        _, adf = vjp(self._jstep_jit, X0)
        
        return adf(adX0)[0]

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
        
        return w1S,w1N,w1W,w1E     
    

###############################################################################
#                           Tracer advection                                  #
###############################################################################

class Model_trac(M):
    
    def __init__(self,config,State):

        super().__init__(config,State)
        
        self.dx = State.DX
        self.dy = State.DY

        _config = config.copy()
        _config.MOD = Config(config.MOD.model_dyn)
        print('* Initializing the dynamical model *')
        self.model_dyn = Model(_config, State)
        print('* End of initializing the dynamical model *\n')

        # Variable names
        self.name_trac = {} # name of tracer variables
        for name in self.name_var:
            self.name_trac[name] = self.name_var[name]
        self.list_var_dyn = []
        for name in self.model_dyn.name_var:
            # all variable names (including dynamical varables)
            self.name_var[name] = self.model_dyn.name_var[name] 
            self.list_var_dyn.append(self.model_dyn.name_var[name])
        self.var_to_save = np.concatenate((self.var_to_save, self.model_dyn.var_to_save)) # names of variables to save
        print('Tracer variables:', self.name_trac)
        print('Dynamical variables:', self.model_dyn.name_var)

        # Time step, taken as the minimum of tracer advection scheme and dynamical model
        self.dt = min(self.dt, self.model_dyn.dt)
        self.model_dyn.dt = self.dt

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
            mask = +State.mask
            mask[:2,:] = 1 # Border
            mask[:,:2] = 1 # Border
            mask[-3:,:] = 1 # Border
            mask[:,-3:] = 1 # Border
            Wbc = grid.compute_weight_map(State.lon, State.lat, mask, config.MOD.dist_sponge_bc)
        else:
            Wbc = np.zeros((State.ny,State.nx)) 
            Wbc[:2,:] = 1 # Border
            Wbc[:,:2] = 1 # Border
            Wbc[-2:,:] = 1 # Border
            Wbc[:,-2:] = 1 # Border
            if State.mask is not None:
                for i,j in np.argwhere(State.mask):
                    for p1 in [-2,-1,0,1,2]:
                        for p2 in [-2,-1,0,1,2]:
                            itest=i+p1
                            jtest=j+p2
                            if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                                if Wbc[itest,jtest]==0:
                                    Wbc[itest,jtest] = 1
        self.Wbc = Wbc

        # Names of other state variables needed to compute total surface current
        self.name_ssh = self.name_u = self.name_v = None
        if 'SSH' in self.model_dyn.name_var:
            self.name_ssh = self.model_dyn.name_var['SSH']
        if 'U' in self.model_dyn.name_var:
            self.name_u = self.model_dyn.name_var['U']
        if 'V' in self.model_dyn.name_var:
            self.name_v = self.model_dyn.name_var['V']

        # Parameters for computing geostrophic current from SSH (if asked)
        self.compute_ugeo_from_ssh = config.MOD.compute_ugeo_from_ssh
        if self.compute_ugeo_from_ssh:
            self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
            self.g = 9.81
        
        # jax compiling functions
        self._ssh2uv_jit = jit(self._ssh2uv)
        self._ssh2uv_adj_jit = jit(self._ssh2uv_adj)
        self._adv_jit = jit(self._adv)
        self._step_jax_jit = jit(self._step_jax,static_argnums=(1,))
        self._step_jax_tgl_jit = jit(self._step_jax_tgl,static_argnums=(2,))
        self._step_jax_adj_jit = jit(self._step_jax_adj,static_argnums=(2,))
        
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=2)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=2)

    def init(self, State, t0=0):

        # Init dynamical variables
        self.model_dyn.init(State,t0=t0)

        # Init tracer variables
        for name in self.name_trac: 
            if t0 in self.bc[name]:
                if self.init_from_bc:
                    State.setvar(self.bc[name][t0], self.name_var[name])
                else:
                    State.var[self.name_var[name]] = self.Wbc * self.bc[name][t0]

    def set_bc(self,time_bc,var_bc):

        # Set BC for dynamical variables
        self.model_dyn.set_bc(time_bc,var_bc)

        # Set BC for tracer variables
        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_trac:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_bc):
                        self.bc[_name_var_mod][t] = var_bc[_name_var_bc][i]

    def _apply_bc(self,State,t0,t):

        for name in self.name_trac:
            if t in self.bc[name]:
                State.var[self.name_var[name]] +=\
                    self.Wbc * (self.bc[name][t]-self.bc[name][t0]) 
    
    def _ssh2uv(self, h):

        u = jnp.zeros((self.ny, self.nx))
        v = jnp.zeros((self.ny, self.nx))

        u = u.at[1:-1, 1:].set(- self.g / self.f[1:-1, 1:] * \
                               (h[2:, :-1] + h[2:, 1:] - h[:-2, 1:] - h[:-2, :-1]) / (4 * self.dy[1:-1, 1:]))

        v = v.at[1:, 1:-1].set(self.g / self.f[1:, 1:-1] * \
                               (h[1:, 2:] + h[:-1, 2:] - h[:-1, :-2] - h[1:, :-2]) / (4 * self.dx[1:, 1:-1]))

        u = jnp.where(jnp.isnan(u), 0, u)
        v = jnp.where(jnp.isnan(v), 0, v)

        return  jnp.array([u, v])

    def _ssh2uv_adj(self,aduv,h):

        _, adf = vjp(self._ssh2uv_jit, h)

        return adf(aduv)[0]
    
    def _get_uv_from_model_dyn(self, State):
        u = jnp.zeros((self.ny,self.nx))
        v = jnp.zeros((self.ny,self.nx))
        if self.compute_ugeo_from_ssh and self.name_ssh in State.var:
            ssh = jnp.array(State.getvar(self.name_ssh))
            uvg = self._ssh2uv_jit(ssh)
            u += uvg[0]
            v += uvg[1]
        if self.name_u is not None and self.name_u in State.var:
            u += jnp.array(State.getvar(self.name_u))
        if self.name_v is not None and self.name_v in State.var:
            v += jnp.array(State.getvar(self.name_v))
        
        return u,v

    def _get_uv_from_model_dyn_adj(self,adu,adv,adState,State):
        if self.compute_ugeo_from_ssh and self.name_ssh in State.var:
            ssh = jnp.array(State.getvar(self.name_ssh))
            aduvg = jnp.array([adu,adv])
            adssh = self._ssh2uv_adj_jit(aduvg,ssh)
            adState.setvar(np.array(adssh),self.name_ssh,add=True)
        if self.name_u is not None and self.name_u in State.var:
            adState.setvar(np.array(adu),self.name_u,add=True)
        if self.name_v is not None and self.name_v in State.var:
            adState.setvar(np.array(adv),self.name_v,add=True)

    def _adv(self,up,vp,um,vm,c):
        
        """
            3rd-order upwind scheme
        """
        
        res = \
            - up*1/(6*self.dx[2:-2,2:-2])*\
                (2*c[2:-2,3:-1]+3*c[2:-2,2:-2]-6*c[2:-2,1:-3]+c[2:-2,:-4]) \
            + um*1/(6*self.dx[2:-2,2:-2])*\
                (c[2:-2,4:]-6*c[2:-2,3:-1]+3*c[2:-2,2:-2]+2*c[2:-2,1:-3])  \
            - vp*1/(6*self.dy[2:-2,2:-2])*\
                (2*c[3:-1,2:-2]+3*c[2:-2,2:-2]-6*c[1:-3,2:-2]+c[:-4,2:-2]) \
            + vm*1/(6*self.dy[2:-2,2:-2])*\
                (c[4:,2:-2]-6*c[3:-1,2:-2]+3*c[2:-2,2:-2]+2*c[1:-3,2:-2])
        
        return res

    
    def _step_jax(self,X0,nstep=1):

        # Get variables
        var0 = X0[0]
        u = X0[1]
        v = X0[2]

        # compute velocities at u,v points
        up = 0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        um = 0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        vp = 0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        vm = 0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        
        # upwind flow
        up = jnp.where(up<0, 0, up)
        um = jnp.where(um>0, 0, um)
        vp = jnp.where(vp<0, 0, vp)
        vm = jnp.where(vm>0, 0, vm)

        var1 = +var0

        for _ in range(nstep):
            rc = jnp.zeros((self.ny,self.nx))
            rc = rc.at[2:-2,2:-2].set(
                rc[2:-2,2:-2] + self._adv_jit(up,vp,um,vm,var1))
            rc = jnp.where(jnp.isnan(rc), 0, rc)

            # Euler timestep
            var1 += self.dt * rc
        
        X1 = +X0
        X1 = X1.at[0].set(var1)
        
        return X1

    def _step_jax_tgl(self,dX0,X0,nstep=1):

        _, dX1 = jvp(partial(self._step_jax_jit, nstep=nstep), (X0,), (dX0,))

        return dX1
    
    def _step_jax_adj(self,adX0,X0,nstep=1):

        _, adf = vjp(partial(self._step_jax_jit, nstep=nstep), X0)

        return adf(adX0)[0]

    def step(self,State,nstep=1,t=None):

        # Get dynamical state trajectory
        u_trj = []
        v_trj = []
        for i in range(nstep):
            # Get current components (u,v)
            u,v = self._get_uv_from_model_dyn(State) 
            u_trj.append(u)
            v_trj.append(v)
            # Forward dynamical model
            self.model_dyn.step(State,nstep=1,t=t)

        # Loop on tracer variables
        for name in self.name_trac:

            # Get state variable
            var0 = State.getvar(self.name_trac[name])
            var0 = jnp.array(var0)

            # Time propation
            X = jnp.array([var0,jnp.zeros_like(var0),jnp.zeros_like(var0)])
            for i in range(nstep):
                # Set current components (u,v)
                X = X.at[1].set(u_trj[i]) 
                X = X.at[2].set(v_trj[i])
                # Forward advection
                X = self._step_jax_jit(X,nstep=1)
                
            # back to numpy
            var1 = np.array(X[0])

            # Update state
            if self.name_trac[name] in State.params:
                params = State.params[self.name_trac[name]]
                var1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params
            State.setvar(var1, self.name_trac[name])

        # Boundary conditions
        self._apply_bc(State,t,t+nstep*self.dt)
        
        
    def step_tgl(self,dState,State,nstep=1,t=None):

        # Get forward dynamical trajectory
        u_trj = []
        v_trj = []
        du_trj = []
        dv_trj = []
        dynvar_trj = []
        State_tmp = State.copy()
        for i in range(nstep):
            # Get current components (u,v)
            u,v = self._get_uv_from_model_dyn(State_tmp) 
            u_trj.append(u)
            v_trj.append(v)
            dynvar_trj.append(State_tmp.getvar(self.list_var_dyn))
            # Get current components (du,dv)
            du,dv = self._get_uv_from_model_dyn(dState) 
            du_trj.append(du)
            dv_trj.append(dv)
            # Forward dynamical model
            self.model_dyn.step_tgl(dState,State_tmp,nstep=1,t=t)
            self.model_dyn.step(State_tmp,nstep=1,t=t)

        # Loop on model variables
        for name in self.name_trac:

            # Get state variable
            dvar0 = dState.getvar(self.name_trac[name])
            var0 = State.getvar(self.name_trac[name])
            dvar0 = jnp.array(dvar0)
            var0 = jnp.array(var0)

            # Time propation
            X = jnp.array([var0,jnp.zeros_like(var0),jnp.zeros_like(var0)])
            dX = jnp.array([dvar0,jnp.zeros_like(var0),jnp.zeros_like(var0)])
            for i in range(nstep):
                # Set current components (u,v)
                X = X.at[1].set(u_trj[i]) 
                X = X.at[2].set(v_trj[i]) 
                # Get u,v from dynamics
                du = du_trj[i]
                dv = dv_trj[i]
                dX = dX.at[1].set(du) 
                dX = dX.at[2].set(dv) 
                # Forward advection
                dX = self._step_jax_tgl_jit(dX,X,nstep=1)
                X = self._step_jax_jit(X,nstep=1)

            # back to numpy
            dvar1 = np.array(dX[0])

            # Update state
            if self.name_trac[name] in dState.params:
                params = dState.params[self.name_trac[name]]
                dvar1 += (1-self.Wbc)*nstep*self.dt/(3600*24) * params
            dState.setvar(dvar1, self.name_trac[name])
        
    def step_adj(self,adState,State,nstep=1,t=None):

        ###############################
        # Forward dynamical trajectory
        ###############################
        u_trj = []
        v_trj = []
        dynvar_trj = []
        State_tmp = State.copy()
        for i in range(nstep):
            # Get current components (u,v)
            u,v = self._get_uv_from_model_dyn(State_tmp) 
            u_trj.append(u)
            v_trj.append(v)
            dynvar_trj.append(State_tmp.getvar(self.list_var_dyn))
            # Forward dynamical model
            #self.model_dyn.step(State_tmp,nstep=1,t=t)

        ###############################
        # Forward tracer advections
        ###############################
        X_traj = {}
        for name in self.name_trac:
            X_traj[name] = []
            # Get state variable
            var0 = State.getvar(self.name_trac[name])
            # convert to jax.numpy
            var0 = jnp.array(var0)
            # Time propation
            X = jnp.array([var0, jnp.zeros_like(var0), jnp.zeros_like(var0)])
            for i in range(nstep):
                # Set current components (u,v)
                X = X.at[1].set(u_trj[i]) 
                X = X.at[2].set(v_trj[i])
                X_traj[name].append(+X)
                # Forward advection
                X = self._step_jax_jit(X,nstep=1)
        
        ###############################
        # Adjoint trajectory
        ###############################
        adX = {}
        advar0 = {}
        for name in self.name_trac:
            _advar0 = adState.getvar(self.name_trac[name])
            advar0[name] = _advar0
            _advar0 = jnp.array(_advar0)
            adX[name] = jnp.zeros((3,self.ny,self.nx))
            adX[name] = adX[name].at[0].set(_advar0)

        for i in reversed(range(nstep)):
            # Set current components (u,v)
            State_tmp.setvar(dynvar_trj[i], self.list_var_dyn)
            # Adjoint dynamics
            #self.model_dyn.step_adj(adState,State_tmp,nstep=1,t=t)
            # Loop on tracer
            adu = jnp.zeros((self.ny,self.nx))
            adv = jnp.zeros((self.ny,self.nx))
            for name in self.name_trac:
                X = +X_traj[name][i]
                # Adjoint advection
                adX[name] = self._step_jax_adj_jit(adX[name],X,nstep=1)
                # Adjoint contribution of advection on velocities for this tracer
                adu += +adX[name][1]
                adv += +adX[name][2]
                adX[name] = adX[name].at[1].set(jnp.zeros_like(adu))
                adX[name] = adX[name].at[2].set(jnp.zeros_like(adv))
            # Adjoint contribution of advection on dynamical variables for all tracer
            self._get_uv_from_model_dyn_adj(adu,adv,adState,State_tmp)

        # Update state and parameters
        for name in self.name_trac:
            advar1 = np.array(adX[name][0])
            if self.name_trac[name] in State.params:
                adState.params[self.name_trac[name]] += (1-self.Wbc)*nstep*self.dt/(3600*24) * advar0[name]
            advar1[np.isnan(advar1)] = 0
            adState.setvar(advar1,self.name_trac[name])


###############################################################################
#                             Multi-models                                    #
###############################################################################      

class Model_multi:

    def __init__(self,config,State):

        self.Models = []
        _config = config.copy()

        for _MOD in config.MOD:
            _config.MOD = config.MOD[_MOD]
            self.Models.append(Model(_config,State))
            print()

        # Time parameters
        self.dt = int(np.max([M.dt for M in self.Models])) # We take the longer timestep 
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
                    # Initialize new State variable
                    State.var[new_name] = State.var[M.name_var[name]].copy()
                    if M.name_var[name] in M.var_to_save and new_name not in self.var_to_save:
                        self.var_to_save = np.append(self.var_to_save,new_name)
        self.var_to_save = list(self.var_to_save)

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=10)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=10)

    def set_bc(self,time_bc,var_bc):

        for M in self.Models:
            M.set_bc(time_bc,var_bc)

    def step(self,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Forward propagation
            M.step(State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var:
                    var_tot_tmp[name] += State.var[M.name_var[name]]
        
        # Update state
        for name in self.name_var:
            State.var[self.name_var[name]] = var_tot_tmp[name]

    def step_tgl(self,dState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Tangent propagation
            M.step_tgl(dState,State,nstep=_nstep,t=t)
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var:
                    var_tot_tmp[name] += dState.var[M.name_var[name]]
        
        # Update state
        for name in self.name_var:
            dState.var[self.name_var[name]] = var_tot_tmp[name]

    def step_adj(self,adState,State,nstep=1,t=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = adState.var[self.name_var[name]]
        
        # Loop over models
        for M in self.Models:
            _nstep = nstep*self.dt//M.dt
            # Add to local variable
            for name in self.name_var:
                if name in M.name_var:
                    adState.var[M.name_var[name]] += var_tot_tmp[name]  
            # Adjoint propagation
            M.step_adj(adState,State,nstep=_nstep,t=t)
        
        for name in self.name_var:
            adState.var[self.name_var[name]] *= 0 
                

        
###############################################################################
#                       Tangent and Adjoint tests                             #
###############################################################################     
    
def tangent_test(M,State,t0=0,nstep=1):

    # Boundary conditions
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {0:np.random.random((M.ny,M.nx))}
    M.set_bc([t0],var_bc)

    State0 = State.random()
    dState = State.random()
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
        
def adjoint_test(M,State,t0=0,nstep=1):

    # Boundary conditions
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {0:np.random.random((M.ny,M.nx))}
    M.set_bc([t0],var_bc)
    
    # Current trajectory
    State0 = State.random()
    
    # Perturbation
    dState = State.random()
    dX0 = np.concatenate((dState.getvar(vect=True),dState.getparams(vect=True)))
    
    # Adjoint
    adState = State.random()
    adX0 = np.concatenate((adState.getvar(vect=True),adState.getparams(vect=True)))
    
    # Run TLM
    M.step_tgl(t=t0,dState=dState,State=State0,nstep=nstep)
    dX1 = np.concatenate((dState.getvar(vect=True),dState.getparams(vect=True)))
    
    # Run ADJ
    M.step_adj(t=t0,adState=adState,State=State0,nstep=nstep)
    adX1 = np.concatenate((adState.getvar(vect=True),adState.getparams(vect=True)))
    
    mask = np.isnan(adX0+dX0)
    
    ps1 = np.inner(dX1[~mask],adX0[~mask])
    ps2 = np.inner(dX0[~mask],adX1[~mask]) 
    
    print(ps1/ps2)

    
    

    
    
    
    
