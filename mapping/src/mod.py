#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:36:20 2021

@author: leguillou
"""

import jax
jax.config.update("jax_enable_x64", True)

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
from jax.lax import scan

from functools import partial

from . import  grid

from .exp import Config as Config

import warnings 


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
        elif config.MOD.super=='MOD_QG1L_JAX_FULL':
            return Model_qg1l_jax_full(config,State)
        elif config.MOD.super=='MOD_SW1L_NP':
            return Model_sw1l_np(config,State)
        elif config.MOD.super=='MOD_SW1L_JAX':
            return Model_sw1l_jax(config,State)
        elif config.MOD.super=='MOD_TRACADV_SSH':
            return Model_tracadv_ssh(config,State)
        elif config.MOD.super=='MOD_TRACADV_VEL':
            return Model_tracadv_vel(config,State)
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    
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
        self.g = State.g
        
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
        self.init_from_bc = config.MOD.init_from_bc
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
                         f=State.f,
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
    
    def init(self, State, t0=0):

        if self.init_from_bc:
            State.setvar(self.SSHb[t0], self.name_var['SSH'])
            if 'PV' in self.name_var:
                pv_b = self.qgm.h2pv(self.SSHb[t0])
                State.setvar(pv_b, self.name_var['PV'])

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

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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
        self.f = State.f
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0
            
        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)


            self.c[np.isnan(self.c)] = np.nanmean(self.c)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
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
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        self.Wbc = np.zeros((State.ny,State.nx))
        if config.MOD.dist_sponge_bc is not None and State.mask is not None:
            self.Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
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
                         g=State.g,
                         f=self.f,
                         Wbc=self.Wbc,
                         Kdiffus=config.MOD.Kdiffus,
                         Kdiffus_trac=config.MOD.Kdiffus_trac,
                         bc_trac=config.MOD.bc_trac)

        # Model functions initialization
        if config.INV is not None and config.INV.super in ['INV_4DVAR','INV_4DVAR_PARALLEL']:
            self.qgm_step = self.qgm.step_jit
            self.qgm_step_tgl = self.qgm.step_tgl_jit
            self.qgm_step_adj = self.qgm.step_adj_jit
        else:
            self.qgm_step = self.qgm.step_jit
        
        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=100)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=100)
    
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
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Remove nan
                        var_bc_t[self.qgm.mask==0] = 0.
                        var_bc_t[np.isnan(var_bc_t)] = 0.
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_bc):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
                        self.forcing[_name_var_mod][t] = var_bc[_name_var_bc][i]

    def ano_bc(self,t,State,sign):

        if not self.anomaly_from_bc:
            return
        else:
            for name in self.name_var:
                if t in self.bc[name]:
                    State.var[self.name_var[name]] += sign * self.bc[name][t]
            
    def _apply_bc(self,t0,t1):
        
        Xb = np.zeros((self.ny,self.nx,))

        if 'SSH' not in self.bc:
            return Xb
        elif t0 not in self.bc['SSH']:
            # Find closest time
            t_list = np.array(list(self.bc['SSH'].keys()))
            idx_closest = np.argmin(np.abs(t_list-t0))
            t0 = t_list[idx_closest]

        Xb = self.bc['SSH'][t0]
        if self.advect_tracer:
            for name in self.name_var:
                if name!='SSH':
                    if t1 in self.bc[name]: 
                        Cb = self.bc[name][t1]
                    else:
                        # Find closest time
                        t_list = np.array(list(self.bc['SSH'].keys()))
                        idx_closest = np.argmin(np.abs(t_list-t1))
                        new_t1 = t_list[idx_closest]
                        Cb = self.bc[name][new_t1]
                    Xb = np.append(Xb[np.newaxis,:,:], 
                                    Cb[np.newaxis,:,:], axis=0)     
        return Xb.astype('float64')
    
    def step(self,State,nstep=1,t=0):
 
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
        X1 = +X0.astype('float64')

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)
        t1 = t + nstep*self.dt

        # Convert to numpy array
        X1 = np.array(X1).astype('float64')
        
        # Update state
        if self.name_var['SSH'] in State.params:
            Fssh = State.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh 
                State.setvar(X1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = +State.params[self.name_var[name]] # Forcing term for tracer
                        if name in self.forcing and t in self.forcing[name]:
                            Fc[1:-1,1:-1] += (self.forcing[name][t]*(3600*24))[1:-1,1:-1]
                        X1[i] += nstep*self.dt/(3600*24) * Fc  * (1-self.Wbc)
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
        dX0 = dState.getvar(name_var=self.name_var['SSH']).astype('float64')
        X0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
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
        dX1 = +dX0.astype('float64')
        X1 = +X0.astype('float64')

        # Time propagation
        dX1 = self.qgm_step_tgl(dX1,X1,Xb=Xb,nstep=nstep)

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
                        dFc = dState.params[self.name_var[name]] # Forcing term for tracer
                        dX1[i] +=  nstep*self.dt/(3600*24) * dFc  * (1-self.Wbc)
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
        adSSH0 = adState.getvar(name_var=self.name_var['SSH']).astype('float64')
        SSH0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        if self.advect_tracer:
            adX0 = adSSH0[np.newaxis,:,:].astype('float64')
            X0 = SSH0[np.newaxis,:,:].astype('float64')
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
        adX1 = np.array(adX1).squeeze().astype('float64')

        # Update state and parameters
        if self.name_var['SSH'] in adState.params:
            for name in self.name_var:
                adparams = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name]).astype('float64') 
                if name!='SSH':
                    adparams *= (1-self.Wbc)
                adState.params[self.name_var[name]] += adparams  
                
        if self.advect_tracer:
            adState.setvar(adX1[0],self.name_var['SSH'])
            for i,name in enumerate(self.name_var):
                if name!='SSH':
                    adState.setvar(adX1[i],self.name_var[name])
        else:
            adState.setvar(adX1,self.name_var['SSH'])
   
class Model_qg1l_jax_full(Model_qg1l_jax):
    def __init__(self,config,State):
        super().__init__(config,State)
        self.step_jit = jit(self.step, static_argnums=[0,3])

    def _apply_bc(self,t,t1):

        Xb = jnp.zeros((self.ny,self.nx,))

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
                            Cb = jnp.zeros((self.ny,self.nx,))
                        Xb = jnp.append(Xb[jnp.newaxis,:,:], 
                                       Cb[jnp.newaxis,:,:], axis=0)     
        return Xb

    def step(self,t,State_var,State_params,nstep=1):

        # Boundary field
        Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # Get state variable(s)
        X0 = State_var[self.name_var['SSH']]
        if self.advect_tracer:
            X0 = X0[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH':
                    C0 = State_var[self.name_var[name]][jnp.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1,Xb,nstep=nstep)
        
        # Update state
        if self.name_var['SSH'] in State_params:
            Fssh = State_params[self.name_var['SSH']] # Forcing term for SSH
            if self.advect_tracer:
                X1[0] += nstep*self.dt/(3600*24) * Fssh
                State_var[self.name_var['SSH']] = X1[0]
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        Fc = State_params[self.name_var[name]] # Forcing term for tracer
                        X1[i] += nstep*self.dt/(3600*24) * Fc
                        State_var[self.name_var[name]] = X1[i]
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State_var[self.name_var['SSH']] = X1

        return State_var


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
        self.g = State.g
             
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
        
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx

        # Coriolis
        self.f = State.f
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = State.g

        # Variables and parameters names arguments 
        self.name_var = config.MOD.name_var
        self.var_to_save = [self.name_var['SSH']] # ssh
        self.name_params = config.MOD.name_params
             
        # Equivalent depth
        if config.MOD.He_data is not None and os.path.exists(config.MOD.He_data['path']):
            ds = xr.open_dataset(config.MOD.He_data['path'])
            self.Heb = ds[config.MOD.He_data['var']].values
        else:
            self.Heb = config.MOD.He_init
        
        # Entering waves BC 
        if 'hbcx' in self.name_params and 'hbcy' in self.name_params :
            if config.MOD.Ntheta>0:
                theta_p = np.arange(0,pi/2+pi/2/config.MOD.Ntheta,pi/2/config.MOD.Ntheta)
                self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
            else:
                self.bc_theta = np.array([0])
        elif 'hbcx' in config.MOD.name_params or 'hbcy' in config.MOD.name_params :
            warnings.warn("Only partly controlling boundary conditions (either just x or y)", Warning)
            
        self.omegas = np.asarray(config.MOD.w_waves)
        self.bc_kind = config.MOD.bc_kind

        
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

        # Initializing model params
        self.init_params(State,config.MOD.name_params,config)

        # Model initialization
        self.swm = swm.Swm(Model = self,
                           State = State) 
        
        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep)
        self._jstep_tgl_jit = jit(self._jstep_tgl)
        self._jstep_adj_jit = jit(self._jstep_adj)
        #self._compute_w1_IT_jit = jit(self._compute_w1_IT)
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

    def init_params(self,State,name_params,config) :
        """
        NAME
            init_params
        
        ARGUMENT : 
            param : parameter to initialize among 
        #       - He : Equivalent Height 
        #       - hbcx : SSH boundary condition for x 
        #       - hbcy : SSH boundary condition for y 
        #       - itg : Internal Tide Generation 
    
        DESCRIPTION
            Initializes the parameter of the object State + sets parameters characteristics in Model object. 
        """
        self.shape_params = {}
        self.slice_params = {}

        # Setting the shapes of parameters
        for param in name_params : 
            if param not in ['He', 'hbcx', 'hbcy', 'itg'] : 
                sys.exit(param+" not implemented. Please choose parameters among ['He', 'hbcx', 'hbcy', 'itg'].")
            elif param =='He' : 
                self.shape_params['He'] = [State.ny,State.nx]
            elif param =='hbcx' : 
                self.shape_params['hbcx'] = [len(self.omegas), # tide frequencies
                                            2, # North/South
                                            2, # cos/sin
                                            len(self.bc_theta), # Angles
                                            State.nx # NX
                                            ]
            elif param =='hbcy' :
                self.shape_params['hbcy'] = [len(self.omegas), # tide frequencies
                                            2, # North/South
                                            2, # cos/sin
                                            len(self.bc_theta), # Angles
                                            State.ny # NY
                                            ]
            elif param =='itg' :
                self.shape_params['itg'] = [2, # A and B, coefficient in front of cos and sin 
                                            State.ny, #NY
                                            State.nx] #NX
        
        # Setting number of parameters 
        self.nparams = sum(list(map(np.prod,list(self.shape_params.values()))))

        # Setting slices of parameters
        idx = 0 
        for param in name_params : 
            self.slice_params[param] = slice(idx, idx + np.prod(self.shape_params[param]))
            idx += np.prod(self.shape_params[param])

        # Initializing the parameters of the object State
        for param in name_params :     
            State.params[param] = np.zeros((self.shape_params[param]))

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
            for param in self.name_params : 
                params = +State.getparams(param,vect=True)
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
        # One forward step
        X0 = self.swm_step(X0)
        return X0
        
        
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
            dparams = +dState.getparams(self.name_params,vect=True)
            dX0 = np.concatenate((dX0,dparams))
            params = +State.getparams(self.name_params,vect=True)
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
            adparams = +adState.getparams(self.name_params,vect=True)
            adX0 = np.concatenate((adX0,adparams))
            params = +State.getparams(self.name_params,vect=True)
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

        # Update state
        adu1[np.isnan(adu1)] = 0
        adv1[np.isnan(adv1)] = 0
        adh1[np.isnan(adh1)] = 0
        adState.setvar([adu1,adv1,adh1],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
        # Update parameters
        for param in self.name_params : 
            adState.params[param]=+adparams[self.slice_params[param]].reshape(self.shape_params[param])
    
    def _jstep_adj(self,adX0,X0):
        
        _, adf = vjp(self._jstep_jit, X0)
        
        return adf(adX0)[0]
  


###############################################################################
#                        Tracer Advection Models                              #
###############################################################################

class Model_tracadv_ssh(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        # Initialize model state variables
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
            

        # Initialize grid spacing
        self.dx = State.DX
        self.dy = State.DY
        self.ny, self.nx = State.ny, State.nx

        # Spatial scheme
        self.upwind = config.MOD.upwind

        # Time scheme
        self.time_scheme = config.MOD.time_scheme

        # Diffusion
        self.Kdiffus = config.MOD.Kdiffus

        # Initialize model parameters (Flux/Forcing)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # Initialize boundary weight map 
        self.Wbc = np.zeros((State.ny,State.nx))
        if State.mask is not None:
            if config.MOD.dist_sponge_bc:
                self.Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
                if config.EXP.flag_plot>1:
                    plt.figure()
                    plt.pcolormesh(self.Wbc)
                    plt.colorbar()
                    plt.title('Wbc')
                    plt.show()
            else: 
                self.Wbc[State.mask] = 1
                indNan = np.argwhere(State.mask)
                for i,j in indNan:
                    for p1 in range(-2,3):
                        for p2 in range(-2,3):
                            itest=i+p1
                            jtest=j+p2
                            if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                                if not State.mask[itest,jtest]:
                                    self.Wbc[itest,jtest] = 1  

        # To compute geostrophic current from SSH
        self.g = State.g
        self.f = State.f

        self._step_jax_jit = jit(self._step_jax, static_argnums=3)  
        self._step_jax_tgl_jit = jit(self._step_jax_tgl, static_argnums=4)  
        self._step_jax_adj_jit = jit(self._step_jax_adj, static_argnums=4)    

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=100)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=100)

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
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
    
    
    def _ssh2uv(self,ssh):

        u = np.zeros((self.ny,self.nx))
        v = np.zeros((self.ny,self.nx))

        u[1:-1,1:] = - self.g / self.f[1:-1,1:] * (ssh[2:, :-1] + ssh[2:, 1:] - ssh[:-2, 1:] - ssh[:-2, :-1])  / (4 * self.dy[1:-1,1:])
        v[1:,1:-1] =   self.g / self.f[1:,1:-1] * (ssh[1:, 2:] + ssh[:-1, 2:] - ssh[:-1, :-2] - ssh[1:, :-2])  / (4 * self.dx[1:,1:-1])

        u = jnp.where(jnp.isnan(u), 0, u)
        v = jnp.where(jnp.isnan(v), 0, v)

        return u, v

    def _ssh2uv_adj(self,adu,adv):

        adssh = np.zeros((self.ny,self.nx))
            
        adssh[2:,:-1] += -self.g/self.f[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adssh[2:,1:] += -self.g/self.f[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adssh[:-2,1:] += +self.g/self.f[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adssh[:-2,:-1] += +self.g/self.f[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        
        adssh[1:,2:] += self.g/self.f[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adssh[:-1,2:] += self.g/self.f[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adssh[:-1,:-2] += -self.g/self.f[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adssh[1:,:-2] += -self.g/self.f[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]

        return adssh

    def _apply_bc_ssh(self,t,ssh,tgl=False):

        if 'SSH' in self.bc and t in self.bc['SSH']:
            if tgl:
                sshb = np.zeros((self.ny,self.nx))
            else:
                sshb = +self.bc['SSH'][t]
            if self.upwind==1:
                ssh[0,:] = sshb[0,:]
                ssh[-1,:] = sshb[-1,:]
                ssh[:,0] = sshb[:,0]
                ssh[:,-1] = sshb[:,-1]
            else:
                ssh[:2,:] = sshb[:2,:]
                ssh[-2:,:] = sshb[-2:,:]
                ssh[:,:2] = sshb[:,:2]
                ssh[:,-2:] = sshb[:,-2:]

        return ssh

    def _get_tracer_bc(self, t):

        cb = np.zeros((len(self.name_var)-1,self.ny,self.nx))
        i = 0
        for name in self.name_var:
            if name!='SSH':
                if name in self.bc and t in self.bc[name]:
                    cb[i] = self.bc[name][t]
                    i += 1
        
        return cb
    
    def _adv(self,u,v,c0):

        """
            main function for upwind advection schemes
        """

        #  Interpolate velocities on T-points
        up = 0.5*(u[1:-1, 1:-1]+u[1:-1, 2:])
        um = 0.5*(u[1:-1, 1:-1]+u[1:-1, 2:])
        vp = 0.5*(v[1:-1, 1:-1]+v[2:, 1:-1])
        vm = 0.5*(v[1:-1, 1:-1]+v[2:, 1:-1])
        
        # Upwind velocities
        up = jnp.where(up < 0, 0, up)
        um = jnp.where(um > 0, 0, um)
        vp = jnp.where(vp < 0, 0, vp)
        vm = jnp.where(vm > 0, 0, vm)

        # Advection
        res = jnp.zeros_like(c0,dtype='float64')
        if self.upwind == 1:
            res = res.at[:,1:-1,1:-1].set(self._adv1(up, vp, um, vm, c0))
        elif self.upwind == 2:
            res = res.at[:,2:-2,2:-2].set(self._adv2(up, vp, um, vm, c0))
        elif self.upwind == 3:
            res = res.at[:,2:-2,2:-2].set(self._adv3(up, vp, um, vm, c0))
        
        # Diffusion 
        if self.Kdiffus is not None:
            res = res.at[:,2:-2,2:-2].set(
                res[:,2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (c0[:,2:-2,3:-1]+c0[:,2:-2,1:-3]-2*c0[:,2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (c0[:,3:-1,2:-2]+c0[:,1:-3,2:-2]-2*c0[:,2:-2,2:-2])
            )
        
        return res

    def _adv1(self, up, vp, um, vm, c0):

        """
            1st-order upwind scheme
        """
        
        
        res = \
            - up  / self.dx[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 1:-1, :-2]) \
            + um  / self.dx[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 1:-1, 2:])  \
            - vp  / self.dy[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, :-2, 1:-1]) \
            + vm  / self.dy[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 2:, 1:-1])

        return res

    def _adv2(self, up, vp, um, vm, c0):

        """
            2nd-order upwind scheme
        """

        res = \
            - up[1:-1, 1:-1] * 1 / (2 * self.dx[2:-2,2:-2]) * \
                (3 * c0[:, 2:-2, 2:-2] - 4 * c0[:, 2:-2, 1:-3] + c0[:, 2:-2, :-4]) \
            + um[1:-1, 1:-1] * 1 / (2 * self.dx[2:-2,2:-2]) * \
                (c0[:, 2:-2, 4:] - 4 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2]) \
            - vp[1:-1, 1:-1] * 1 / (2 * self.dy[2:-2,2:-2]) * \
                (3 * c0[:, 2:-2, 2:-2] - 4 * c0[:, 1:-3, 2:-2] + c0[:, :-4, 2:-2]) \
            + vm[1:-1, 1:-1] * 1 / (2 * self.dy[2:-2,2:-2]) * \
                (c0[:, 4:, 2:-2] - 4 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2])

        return res

    def _adv3(self, up, vp, um, vm, c0):
        """
            3rd-order upwind scheme
        """

        res = \
            - up[1:-1, 1:-1] * 1 / (6 * self.dx[2:-2,2:-2]) * \
            (2 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2] - 6 * c0[:, 2:-2, 1:-3] + c0[:, 2:-2, :-4]) \
            + um[1:-1, 1:-1] * 1 / (6 * self.dx[2:-2,2:-2]) * \
            (c0[:, 2:-2, 4:] - 6 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2] + 2 * c0[:, 2:-2, 1:-3]) \
            - vp[1:-1, 1:-1] * 1 / (6 * self.dy[2:-2,2:-2]) * \
            (2 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2] - 6 * c0[:, 1:-3, 2:-2] + c0[:, :-4, 2:-2]) \
            + vm[1:-1, 1:-1] * 1 / (6 * self.dy[2:-2,2:-2]) * \
            (c0[:, 4:, 2:-2] - 6 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2] + 2 * c0[:, 1:-3, 2:-2])

        return res
    
    def _bc(self,u,v,c0,c1,cb):
        
        if self.upwind==1:
            # Compute adimensional coefficients fro OBC
            r1_S = 1/2 * self.dt/self.dy[0,:] * (v[1,:]  + jnp.abs(v[1,:] ))
            r2_S = 1/2 * self.dt/self.dy[0,:] * (v[1,:]  - jnp.abs(v[1,:] ))
            r1_N = 1/2 * self.dt/self.dy[-1,:] * (v[-1,:]  + jnp.abs(v[-1,:] ))
            r2_N = 1/2 * self.dt/self.dy[-1,:] * (v[-1,:]  - jnp.abs(v[-1,:] ))
            r1_W = 1/2 * self.dt/self.dx[:,0] * (u[:,1] + jnp.abs(u[:,1]))
            r2_W = 1/2 * self.dt/self.dx[:,0] * (u[:,1] - jnp.abs(u[:,1]))
            r1_E = 1/2 * self.dt/self.dx[:,-1] * (u[:,-1] + jnp.abs(u[:,-1]))
            r2_E = 1/2 * self.dt/self.dx[:,-1] * (u[:,-1] - jnp.abs(u[:,-1]))

            # South
            c1 = c1.at[:,0,:].set(
                c0[:,0,:] - (r1_S*(c0[:,0,:]-cb[:,0,:]) + r2_S*(c0[:,1,:]-c0[:,0,:])))

            # North
            c1 = c1.at[:,-1,:].set(
                c0[:,-1,:] - (r1_N*(c0[:,-1,:]-c0[:,-2,:]) + r2_N*(cb[:,-1,:]-c0[:,-1,:])))
            
            # West
            c1 = c1.at[:,:,0].set(
                c0[:,:,0] - (r1_W*(c0[:,:,0]-cb[:,:,0]) + r2_W*(c0[:,:,1]-c0[:,:,0])))
            
            # East
            c1 = c1.at[:,:,-1].set(
                c0[:,:,-1] - (r1_E*(c0[:,:,-1]-c0[:,:,-2]) + r2_E*(cb[:,:,-1]-c0[:,:,-1])))
        else:
            # BC value on borders
            c1 = c1.at[:,0,:].set(cb[:,0,:])
            c1 = c1.at[:,-1,:].set(cb[:,-1,:])
            c1 = c1.at[:,:,0].set(cb[:,:,0])
            c1 = c1.at[:,:,-1].set(cb[:,:,-1])
            # OBC on inner pixels
            r1_S = 1/2 * self.dt/self.dy[1,:] * (v[2,:]  + jnp.abs(v[2,:] ))
            r2_S = 1/2 * self.dt/self.dy[1,:] * (v[2,:]  - jnp.abs(v[2,:] ))
            r1_N = 1/2 * self.dt/self.dy[-2,:] * (v[-2,:]  + jnp.abs(v[-2,:] ))
            r2_N = 1/2 * self.dt/self.dy[-2,:] * (v[-2,:]  - jnp.abs(v[-2,:] ))
            r1_W = 1/2 * self.dt/self.dx[:,1] * (u[:,2] + jnp.abs(u[:,2]))
            r2_W = 1/2 * self.dt/self.dx[:,1] * (u[:,2] - jnp.abs(u[:,2]))
            r1_E = 1/2 * self.dt/self.dx[:,-2] * (u[:,-2] + jnp.abs(u[:,-2]))
            r2_E = 1/2 * self.dt/self.dx[:,-2] * (u[:,-2] - jnp.abs(u[:,-2]))

            # South
            c1 = c1.at[:,1,:].set(
                c0[:,1,:] - (r1_S*(c0[:,1,:]-cb[:,1,:]) + r2_S*(c0[:,2,:]-c0[:,1,:])))

            # North
            c1 = c1.at[:,-2,:].set(
                c0[:,-2,:] - (r1_N*(c0[:,-2,:]-c0[:,-3,:]) + r2_N*(cb[:,-2,:]-c0[:,-2,:])))
            
            # West
            c1 = c1.at[:,:,1].set(
                c0[:,:,1] - (r1_W*(c0[:,:,1]-cb[:,:,1]) + r2_W*(c0[:,:,2]-c0[:,:,1])))
            
            # East
            c1 = c1.at[:,:,-2].set(
                c0[:,:,-2] - (r1_E*(c0[:,:,-2]-c0[:,:,-3]) + r2_E*(cb[:,:,-2]-c0[:,:,-2])))

        # Sponge
        c1 = self.Wbc * cb + (1 - self.Wbc) * c1
        
        return c1

    def _one_step_for_scan(self,X0,X):

        u = X0[0][0]
        v = X0[0][1]
        c0 = X0[0][2:]
        cb0 = X0[1]
        cb1 = X0[2]
        nstep = X0[3]
        step = X0[4]
        if len(c0.shape)==2:
            c0 = c0[jnp.newaxis,:,:]
        
        # Spatial scheme
        rhs = self._adv(u,v,c0)

        # Time scheme
        if self.time_scheme=='Euler':
            # Euler 
            c1 = c0 + self.dt * rhs
        elif self.time_scheme=='rk2':
            # rk2
            c12 = c0 + 0.5*rhs*self.dt
            rhs12 = self._adv(u,v,c12)
            c1 = c0 + self.dt * rhs12
        elif self.time_scheme=='rk4':
            # rk4
            ## k1
            k1 = rhs * self.dt
            ## k2
            c2 = c0 + 0.5*k1
            rhs2 = self._adv(u,v,c2)
            k2 = rhs2*self.dt
            ## k3
            c3 = c0 + 0.5*k2
            rhs3 = self._adv(u,v,c3)
            k3 = rhs3*self.dt
            ## k4
            c4 = c0 + k2
            rhs4 = self._adv(u,v,c4) 
            k4 = rhs4*self.dt
            # q increment
            c1 = c0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
        
        # Boundary condition
        cb = (1-(step+1)/nstep) * cb0 + (step+1)/nstep * cb1 # linear interpolation
        c1 = self._bc(u,v,c0,c1,cb)
            
        X = jnp.append(u[jnp.newaxis,:,:],v[jnp.newaxis,:,:],axis=0)
        X = jnp.append(X,c1,axis=0)

        X = (X,cb0,cb1,nstep,step+1)

        return X, X

    def _step_jax(self,X0,cb0,cb1,nstep):

        # Add static arguments in X dynamic variables for scan function
        step = 0
        X0 = (X0, cb0, cb1, nstep, step)

        # Run scan
        X1, _ = scan(self._one_step_for_scan, init=X0, xs=jnp.zeros(nstep))

        # Get dynamic variables 
        X1, _, _, _, _ = X1
        
        return X1
    
    def _step_jax_tgl(self,dX0,X0,cb0,cb1,nstep):

        _, dX1 = jvp(partial(self._step_jax_jit, cb0=cb0, cb1=cb1, nstep=nstep), (X0,), (dX0,))

        return dX1

    def _step_jax_adj(self,adX0,X0,cb0,cb1,nstep):
        
        _, adf = vjp(partial(self._step_jax_jit, cb0=cb0, cb1=cb1, nstep=nstep), X0)
        
        return adf(adX0)[0]
    
    def step(self,State,nstep=1,t=0):

        # Init
        X0 = np.zeros((1+len(self.name_var),State.ny,State.nx))

        # Get SSH variable
        ssh = State.getvar(name_var=self.name_var['SSH'])

        # SSH boundary conditions
        ssh = self._apply_bc_ssh(t,ssh)

        # Convert to geostrophic velocities
        u,v = self._ssh2uv(ssh)
        X0[0] = u
        X0[1] = v

        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name!='SSH':
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Convert to JAX
        X0 = jnp.array(X0)

        # Forward propagation 
        X1 = self._step_jax_jit(X0,cb0,cb1,nstep)

        # Back to numpy
        c1 = np.array(X1)[2:] # advected tracer concentrations

        # SSH boundary conditions
        ssh = self._apply_bc_ssh(t+nstep*self.dt,ssh)
        
        # Update state
        Fssh = State.params[self.name_var['SSH']] 
        ssh[2:-2,2:-2] += nstep*self.dt/(3600*24) * Fssh[2:-2,2:-2]
        State.setvar(ssh, name_var=self.name_var['SSH'])
        i = 0
        for name in self.name_var:
            if name!='SSH':
                Fc = State.params[self.name_var[name]] # Forcing term for tracer
                c1[i,2:-2,2:-2] += (nstep*self.dt/(3600*24) * Fc  * (1-self.Wbc))[2:-2,2:-2]
                State.setvar(c1[i], name_var=self.name_var[name])
                i += 1
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        # Init
        dX0 = np.zeros((1+len(self.name_var),State.ny,State.nx))
        X0 = np.zeros((1+len(self.name_var),State.ny,State.nx))

        # Get SSH variable
        dssh = dState.getvar(name_var=self.name_var['SSH'])
        ssh = State.getvar(name_var=self.name_var['SSH'])

        # SSH boundary conditions
        dssh = self._apply_bc_ssh(t,dssh,tgl=True)
        ssh = self._apply_bc_ssh(t,ssh)

        # Convert to geostrophic velocities
        du,dv = self._ssh2uv(dssh)
        u,v = self._ssh2uv(ssh)

        dX0[0] = du
        dX0[1] = dv
        X0[0] = u
        X0[1] = v
        
        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name!='SSH':
                dc0 = dState.getvar(name_var=self.name_var[name])
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                dX0[i] = dc0
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Convert to JAX
        dX0 = jnp.array(dX0)
        X0 = jnp.array(X0)

        # Time propagation
        dX1 = self._step_jax_tgl_jit(dX0,X0,cb0,cb1,nstep)

        # Back to numpy
        dc1 = np.array(dX1)[2:] # advected tracer concentrations

        # SSH boundary conditions
        dssh = self._apply_bc_ssh(t+nstep*self.dt,dssh,tgl=True)
        
        # Update state
        dFssh = dState.params[self.name_var['SSH']] 
        dssh[2:-2,2:-2] += nstep*self.dt/(3600*24) * dFssh[2:-2,2:-2]
        dState.setvar(dssh, name_var=self.name_var['SSH'])
        i = 0
        for name in self.name_var:
            if name!='SSH':
                dFc = dState.params[self.name_var[name]] # Forcing term for tracer
                dc1[i,2:-2,2:-2] += nstep*self.dt/(3600*24) * dFc[2:-2,2:-2] * (1-self.Wbc[2:-2,2:-2])
                dState.setvar(dc1[i], name_var=self.name_var[name])
                i += 1

    def step_adj(self,adState,State,nstep=1,t=None):

        # Init
        adX0 = np.zeros((1+len(self.name_var),State.ny,State.nx))
        X0 = np.zeros((1+len(self.name_var),State.ny,State.nx))

        # Get SSH variable
        ssh = State.getvar(name_var=self.name_var['SSH'])

        # SSH boundary conditions
        ssh = self._apply_bc_ssh(t,ssh)

        # Convert to geostrophic velocities
        u,v = self._ssh2uv(ssh)
        X0[0] = u
        X0[1] = v
        
        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name!='SSH':
                adc0 = adState.getvar(name_var=self.name_var[name])
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                adX0[i] = adc0
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Update parameters
        for name in self.name_var:
            adF = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name]) 
            if name!='SSH':
                adF *= (1-self.Wbc)
            adState.params[self.name_var[name]][2:-2,2:-2] += adF[2:-2,2:-2]

        # Convert to JAX
        adX0 = jnp.array(adX0)
        X0 = jnp.array(X0)

        # Time propagation
        adX1 = self._step_jax_adj_jit(adX0,X0,cb0,cb1,nstep)
        
        # Back to numpy
        adX1 = np.array(adX1) 
        adX1[np.isnan(adX1)] = 0
        adu = adX1[0]
        adv = adX1[1]
        adc1 = adX1[2:]

        # Get SSH variable
        adssh = adState.getvar(name_var=self.name_var['SSH'])

        # Convert to geostrophic velocities
        adssh += self._ssh2uv_adj(adu,adv)

        # SSH boundary conditions
        adssh = self._apply_bc_ssh(t,adssh,tgl=True)

        # Update SSH
        adState.setvar(adssh, name_var=self.name_var['SSH'])

        # Update tracers
        i = 0
        for name in self.name_var:
            if name!='SSH':
                adState.setvar(adc1[i], name_var=self.name_var[name])
                i += 1


class Model_tracadv_vel(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        # Initialize model state variables
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
            

        # Initialize grid spacing
        self.dx = State.DX
        self.dy = State.DY
        self.ny, self.nx = State.ny, State.nx

        # Spatial scheme
        self.upwind = config.MOD.upwind

        # Time scheme
        self.time_scheme = config.MOD.time_scheme

        # Diffusion
        self.Kdiffus = config.MOD.Kdiffus

        # Initialize model parameters (Flux/Forcing)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # Initialize boundary weight map 
        self.Wbc = np.zeros((State.ny,State.nx))
        if State.mask is not None:
            if config.MOD.dist_sponge_bc:
                self.Wbc = grid.compute_weight_map(State.lon, State.lat, +State.mask, config.MOD.dist_sponge_bc)
                if config.EXP.flag_plot>1:
                    plt.figure()
                    plt.pcolormesh(self.Wbc)
                    plt.colorbar()
                    plt.title('Wbc')
                    plt.show()
            else: 
                self.Wbc[State.mask] = 1
                indNan = np.argwhere(State.mask)
                for i,j in indNan:
                    for p1 in range(-2,3):
                        for p2 in range(-2,3):
                            itest=i+p1
                            jtest=j+p2
                            if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                                if not State.mask[itest,jtest]:
                                    self.Wbc[itest,jtest] = 1  
            

        self._step_jax_jit = jit(self._step_jax, static_argnums=3)  
        self._step_jax_tgl_jit = jit(self._step_jax_tgl, static_argnums=4)  
        self._step_jax_adj_jit = jit(self._step_jax_adj, static_argnums=4)    

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State,nstep=100)
            print('Adjoint test:')
            adjoint_test(self,State,nstep=100)

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
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
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_bc):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
                        self.forcing[_name_var_mod][t] = var_bc[_name_var_bc][i]
    
    def _add_geo_vel(self,t,u,v):

        utot = u.copy()
        vtot = v.copy()

        if 'U' in self.bc and t in self.bc['U']:
            utot += self.bc['U'][t]
        if 'V' in self.bc and t in self.bc['V']:
            vtot += self.bc['V'][t]

        return utot, vtot

    def _get_tracer_bc(self, t):

        cb = np.zeros((len(self.name_var)-2,self.ny,self.nx))
        i = 0
        for name in self.name_var:
            if name not in ['U','V']:
                if name in self.bc and t in self.bc[name]:
                    cb[i] = self.bc[name][t]
                    i += 1
        
        return cb
    
    def _adv(self,u,v,c0):

        """
            main function for upwind advection schemes
        """

        up = jnp.where(u < 0, 0, u)
        um = jnp.where(u > 0, 0, u)
        vp = jnp.where(v < 0, 0, v)
        vm = jnp.where(v > 0, 0, v)
    
        res = jnp.zeros_like(c0,dtype='float64')

        if self.upwind == 1:
            res = res.at[:,1:-1,1:-1].set(self._adv1(up, vp, um, vm, c0))
        else:
            res = res.at[:,1:-1,1:-1].set(self._adv1(up, vp, um, vm, c0))
            if self.upwind == 2:
                res = res.at[:,2:-2,2:-2].set(self._adv2(up, vp, um, vm, c0))
            if self.upwind == 3:
                res = res.at[:,2:-2,2:-2].set(self._adv3(up, vp, um, vm, c0))
        
        if self.Kdiffus is not None:
            res = res.at[:,2:-2,2:-2].set(
                res[:,2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (c0[:,2:-2,3:-1]+c0[:,2:-2,1:-3]-2*c0[:,2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (c0[:,3:-1,2:-2]+c0[:,1:-3,2:-2]-2*c0[:,2:-2,2:-2])
            )
        
        return res

    def _adv1(self, up, vp, um, vm, c0):

        """
            1st-order upwind scheme
        """
        
        
        res = \
            - up[1:-1, 1:-1]  / self.dx[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 1:-1, :-2]) \
            + um[1:-1, 1:-1]  / self.dx[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 1:-1, 2:])  \
            - vp[1:-1, 1:-1]  / self.dy[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, :-2, 1:-1]) \
            + vm[1:-1, 1:-1]  / self.dy[1:-1, 1:-1] * (c0[:, 1:-1, 1:-1] - c0[:, 2:, 1:-1])

        return res

    def _adv2(self, up, vp, um, vm, c0):

        """
            2nd-order upwind scheme
        """

        res = \
            - up[2:-2,2:-2] * 1 / (2 * self.dx[2:-2,2:-2]) * \
                (3 * c0[:, 2:-2, 2:-2] - 4 * c0[:, 2:-2, 1:-3] + c0[:, 2:-2, :-4]) \
            + um[2:-2,2:-2] * 1 / (2 * self.dx[2:-2,2:-2]) * \
                (c0[:, 2:-2, 4:] - 4 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2]) \
            - vp[2:-2,2:-2] * 1 / (2 * self.dy[2:-2,2:-2]) * \
                (3 * c0[:, 2:-2, 2:-2] - 4 * c0[:, 1:-3, 2:-2] + c0[:, :-4, 2:-2]) \
            + vm[2:-2,2:-2] * 1 / (2 * self.dy[2:-2,2:-2]) * \
                (c0[:, 4:, 2:-2] - 4 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2])

        return res

    def _adv3(self, up, vp, um, vm, c0):
        """
            3rd-order upwind scheme
        """

        res = \
            - up[2:-2,2:-2] * 1 / (6 * self.dx[2:-2,2:-2]) * \
            (2 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2] - 6 * c0[:, 2:-2, 1:-3] + c0[:, 2:-2, :-4]) \
            + um[2:-2,2:-2] * 1 / (6 * self.dx[2:-2,2:-2]) * \
            (c0[:, 2:-2, 4:] - 6 * c0[:, 2:-2, 3:-1] + 3 * c0[:, 2:-2, 2:-2] + 2 * c0[:, 2:-2, 1:-3]) \
            - vp[2:-2,2:-2] * 1 / (6 * self.dy[2:-2,2:-2]) * \
            (2 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2] - 6 * c0[:, 1:-3, 2:-2] + c0[:, :-4, 2:-2]) \
            + vm[2:-2,2:-2] * 1 / (6 * self.dy[2:-2,2:-2]) * \
            (c0[:, 4:, 2:-2] - 6 * c0[:, 3:-1, 2:-2] + 3 * c0[:, 2:-2, 2:-2] + 2 * c0[:, 1:-3, 2:-2])

        return res
    
    def _bc(self,u,v,c0,c1,cb):
        
        # Compute adimensional coefficients fro OBC
        r1_S = 1/2 * self.dt/self.dy[0,:] * (v[0,:]  + jnp.abs(v[0,:] ))
        r2_S = 1/2 * self.dt/self.dy[0,:] * (v[0,:]  - jnp.abs(v[0,:] ))
        r1_N = 1/2 * self.dt/self.dy[-1,:] * (v[-1,:]  + jnp.abs(v[-1,:] ))
        r2_N = 1/2 * self.dt/self.dy[-1,:] * (v[-1,:]  - jnp.abs(v[-1,:] ))
        r1_W = 1/2 * self.dt/self.dx[:,0] * (u[:,0] + jnp.abs(u[:,0]))
        r2_W = 1/2 * self.dt/self.dx[:,0] * (u[:,0] - jnp.abs(u[:,0]))
        r1_E = 1/2 * self.dt/self.dx[:,-1] * (u[:,-1] + jnp.abs(u[:,-1]))
        r2_E = 1/2 * self.dt/self.dx[:,-1] * (u[:,-1] - jnp.abs(u[:,-1]))

        ibc = 0

        # South
        c1 = c1.at[:,ibc,:].set(
            c0[:,ibc,:] - (r1_S*(c0[:,ibc,:]-c0[:,ibc,:]) + r2_S*(cb[:,ibc+1,:]-c0[:,ibc,:])))

        # North
        c1 = c1.at[:,-(ibc+1),:].set(
            c0[:,-(ibc+1),:] - (r1_N*(c0[:,-(ibc+1),:]-c0[:,-(ibc+2),:]) + r2_N*(cb[:,-(ibc+1),:]-c0[:,-(ibc+1),:])))
        
        # West
        c1 = c1.at[:,:,ibc].set(
            c0[:,:,ibc] - (r1_W*(c0[:,:,ibc]-c0[:,:,ibc]) + r2_W*(cb[:,:,ibc+1]-c0[:,:,ibc])))
        
        # East
        c1 = c1.at[:,:,-1].set(
            c0[:,:,-(ibc+1)] - (r1_E*(c0[:,:,-(ibc+1)]-c0[:,:,-(ibc+2)]) + r2_E*(cb[:,:,-(ibc+1)]-c0[:,:,-(ibc+1)])))

        # Sponge
        c1 = self.Wbc * cb + (1 - self.Wbc) * c1
        
        return c1

    def _one_step_for_scan(self,X0,X):

        u = X0[0][0]
        v = X0[0][1]
        c0 = X0[0][2:]
        cb0 = X0[1]
        cb1 = X0[2]
        nstep = X0[3]
        step = X0[4]
        if len(c0.shape)==2:
            c0 = c0[jnp.newaxis,:,:]
        
        # Spatial scheme
        rhs = self._adv(u,v,c0)

        # Time scheme
        if self.time_scheme=='Euler':
            # Euler 
            c1 = c0 + self.dt * rhs
        elif self.time_scheme=='rk2':
            # rk2
            c12 = c0 + 0.5*rhs*self.dt
            rhs12 = self._adv(u,v,c12)
            c1 = c0 + self.dt * rhs12
        elif self.time_scheme=='rk4':
            # rk4
            ## k1
            k1 = rhs * self.dt
            ## k2
            c2 = c0 + 0.5*k1
            rhs2 = self._adv(u,v,c2)
            k2 = rhs2*self.dt
            ## k3
            c3 = c0 + 0.5*k2
            rhs3 = self._adv(u,v,c3)
            k3 = rhs3*self.dt
            ## k4
            c4 = c0 + k2
            rhs4 = self._adv(u,v,c4) 
            k4 = rhs4*self.dt
            # q increment
            c1 = c0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
        
        # Boundary condition
        cb = (1-(step+1)/nstep) * cb0 + (step+1)/nstep * cb1 # linear interpolation
        c1 = self._bc(u,v,c0,c1,cb)
            
        X = jnp.append(u[jnp.newaxis,:,:],v[jnp.newaxis,:,:],axis=0)
        X = jnp.append(X,c1,axis=0)

        X = (X,cb0,cb1,nstep,step+1)

        return X, X

    def _step_jax(self,X0,cb0,cb1,nstep):

        # Add static arguments in X dynamic variables for scan function
        step = 0
        X0 = (X0, cb0, cb1, nstep, step)

        # Run scan
        X1, _ = scan(self._one_step_for_scan, init=X0, xs=jnp.zeros(nstep))

        # Get dynamic variables 
        X1, _, _, _, _ = X1
        
        return X1
    
    def _step_jax_tgl(self,dX0,X0,cb0,cb1,nstep):

        _, dX1 = jvp(partial(self._step_jax_jit, cb0=cb0, cb1=cb1, nstep=nstep), (X0,), (dX0,))

        return dX1

    def _step_jax_adj(self,adX0,X0,cb0,cb1,nstep):
        
        _, adf = vjp(partial(self._step_jax_jit, cb0=cb0, cb1=cb1, nstep=nstep), X0)
        
        return adf(adX0)[0]
    
    def step(self,State,nstep=1,t=0):

        # Init
        X0 = np.zeros((len(self.name_var),State.ny,State.nx))

        # Get velocity variable(s)
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        utot, vtot = self._add_geo_vel(t,u,v) # Add geostrophic velocities
        X0[0] = utot
        X0[1] = vtot

        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name not in ['U','V']:
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Convert to JAX
        X0 = jnp.array(X0)

        # Forward propagation 
        X1 = self._step_jax_jit(X0,cb0,cb1,nstep)

        # Back to numpy
        X1 = np.array(X1) # advected tracer concentrations
        
        # Update state
        Fu = State.params[self.name_var['U']] 
        Fv = State.params[self.name_var['V']] 
        u += nstep*self.dt/(3600*24) * Fu
        v += nstep*self.dt/(3600*24) * Fv
        State.setvar(u, name_var=self.name_var['U'])
        State.setvar(v, name_var=self.name_var['V'])
        i = 0
        for name in self.name_var:
            if name not in ['U','V']:
                Fc = +State.params[self.name_var[name]] # Forcing term for tracer
                if name in self.forcing and t in self.forcing[name]:
                    Fc[1:-1,1:-1] += (self.forcing[name][t]*(3600*24))[1:-1,1:-1]
                X1[i+2] += nstep*self.dt/(3600*24) * Fc  * (1-self.Wbc)
                State.setvar(X1[i+2], name_var=self.name_var[name])
                i += 1
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        # Init
        dX0 = np.zeros((len(self.name_var),State.ny,State.nx))
        X0 = np.zeros((len(self.name_var),State.ny,State.nx))

        # Get velocity variable(s)
        du = dState.getvar(name_var=self.name_var['U'])
        dv = dState.getvar(name_var=self.name_var['V'])
        dX0[0] = du
        dX0[1] = dv
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        utot, vtot = self._add_geo_vel(t,u,v) # Add geostrophic velocities
        X0[0] = utot
        X0[1] = vtot
        
        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name not in ['U','V']:
                dc0 = dState.getvar(name_var=self.name_var[name])
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                dX0[i] = dc0
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Convert to JAX
        dX0 = jnp.array(dX0)
        X0 = jnp.array(X0)

        # Time propagation
        dX1 = self._step_jax_tgl_jit(dX0,X0,cb0,cb1,nstep)

        # Back to numpy
        dX1 = np.array(dX1) # advected tracer concentrations
        
        # Update state
        dFu = dState.params[self.name_var['U']] 
        dFv = dState.params[self.name_var['V']] 
        du += nstep*self.dt/(3600*24) * dFu
        dv += nstep*self.dt/(3600*24) * dFv
        dState.setvar(du, name_var=self.name_var['U'])
        dState.setvar(dv, name_var=self.name_var['V'])
        i = 0
        for name in self.name_var:
            if name not in ['U','V']:
                dFc = dState.params[self.name_var[name]] # Forcing term for tracer
                dX1[i+2] += nstep*self.dt/(3600*24) * dFc * (1-self.Wbc)
                dState.setvar(dX1[i+2], name_var=self.name_var[name])
                i += 1

    def step_adj(self,adState,State,nstep=1,t=None):

        # Init
        adX0 = np.zeros((len(self.name_var),State.ny,State.nx))
        X0 = np.zeros((len(self.name_var),State.ny,State.nx))

        # Get velocity variable(s)
        adu = adState.getvar(name_var=self.name_var['U'])
        adv = adState.getvar(name_var=self.name_var['V'])
        adX0[0] = adu
        adX0[1] = adv
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        utot, vtot = self._add_geo_vel(t,u,v) # Add geostrophic velocities
        X0[0] = utot
        X0[1] = vtot
        
        # Get tracer variables
        i = 2
        for name in self.name_var:
            if name not in ['U','V']:
                adc0 = adState.getvar(name_var=self.name_var[name])
                c0 = State.getvar(name_var=self.name_var[name])
                # Fill array
                adX0[i] = adc0
                X0[i] = c0
                i += 1
        
        cb0 = self._get_tracer_bc(t)
        cb1 = self._get_tracer_bc(t+nstep*self.dt)

        # Update parameters
        for name in self.name_var:
            adF = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name]) 
            if name not in ['U','V']:
                adF *= (1-self.Wbc)
            adState.params[self.name_var[name]] += adF

        # Convert to JAX
        adX0 = jnp.array(adX0)
        X0 = jnp.array(X0)

        # Time propagation
        adX1 = self._step_jax_adj_jit(adX0,X0,cb0,cb1,nstep)
        
        # Back to numpy
        adX1 = np.array(adX1) 
        adX1[np.isnan(adX1)] = 0

        # Update variables
        adState.setvar(adX1[0], name_var=self.name_var['U'])
        adState.setvar(adX1[1], name_var=self.name_var['V'])
        i = 0
        for name in self.name_var:
            if name not in ['U','V']:
                adState.setvar(adX1[i+2], name_var=self.name_var[name])
                i += 1

            
    
    def save_output(self,State,present_date,name_var=None,t=None):

        # Add geostrophic velocities
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        utot, vtot = self._add_geo_vel(t,u,v) # Add geostrophic velocities

        State0 = State.copy()
        State0.setvar(utot,name_var=self.name_var['U'])
        State0.setvar(vtot,name_var=self.name_var['V'])

        State0.save_output(present_date,name_var)


                
    


        

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
        var_bc[name] = {0:np.random.random((M.ny,M.nx)).astype('float64'),
                        1:np.random.random((M.ny,M.nx)).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)

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
        var_bc[name] = {0:np.random.random((M.ny,M.nx)).astype('float64'),
                        1:np.random.random((M.ny,M.nx)).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)
    
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

    
    

    
    
    
    
