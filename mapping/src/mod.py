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
        
        elif config.MOD.super=='MOD_QG1L':
            return Model_qg1l(config,State)

        elif config.MOD.super=='MOD_QG1L_NP':
            return Model_qg1l_np(config,State)
        
        elif config.MOD.super=='MOD_SW1L':
            return Model_sw1l(config,State)
        
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    
class M:

    """ Parent model class """

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
#                            Diffusion Model                                 #
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
#                       Quasi-Geostrophic Model                               #
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

class Model_qg1l(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",dir_model + "/qgm.py").load_module() 
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
            
    def _apply_bc(self,t0):
        
        Xb = np.zeros((self.ny,self.nx,))

        if 'SSH' not in self.bc:
            return Xb
        elif t0 not in self.bc['SSH']:
            # Find closest time
            t_list = np.array(list(self.bc['SSH'].keys()))
            idx_closest = np.argmin(np.abs(t_list-t0))
            t0 = t_list[idx_closest]

        Xb = self.bc['SSH'][t0]
        
        return Xb.astype('float64')

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
    
    def step(self, State, Nudging_term=None, nstep=1, t=0, way=1):
 
        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t)

        # Get state variable(s)
        X0 = State.getvar(name_var=self.name_var['SSH'])
        
        # init
        X1 = +X0.astype('float64')

        # Time propagation
        X1 = self.qgm_step(X1, Xb, Nudg_term=Nudging_term, nstep=nstep, way=way)
        t1 = t + nstep*self.dt

        # Convert to numpy array
        X1 = np.array(X1).astype('float64')
        
        # Update state
        if self.name_var['SSH'] in State.params:
            Fssh = State.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            X1 += nstep*self.dt/(3600*24) * Fssh 
            State.setvar(X1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t1,State,-1)
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t)
        
        # Get state variable
        dX0 = dState.getvar(name_var=self.name_var['SSH']).astype('float64')
        X0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        
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
            dX1 += nstep*self.dt/(3600*24) * dFssh  
            dState.setvar(dX1, name_var=self.name_var['SSH'])

        # Get anomaly from full field
        self.ano_bc(t,State,-1)

    def step_adj(self,adState,State,nstep=1,t=None):
        
        # Get full field from anomaly 
        self.ano_bc(t,State,+1)

        # Boundary field
        Xb = self._apply_bc(t)

        # Get state variable
        adSSH0 = adState.getvar(name_var=self.name_var['SSH']).astype('float64')
        SSH0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
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
                adState.params[self.name_var[name]] += adparams  
                
        adState.setvar(adX1,self.name_var['SSH'])


###############################################################################
#                         Shallow Water Model                                 #
###############################################################################
   
class Model_sw1l(M):
    def __init__(self,config,State):

        super().__init__(config,State)

        self.config = config
        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models'))
        else:
            dir_model = config.MOD.dir_model
        
        swm = SourceFileLoader("swm", 
                                dir_model + "/swm.py").load_module()
        model = swm.Swm
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx

        # Coriolis
        self.f = State.f
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
#                             Multi-models                                    #
###############################################################################      

class Model_multi(M):

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

    def init(self, State, t0=0):

        for M in self.Models:
            M.init(State,t0=t0)
    
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
        
    def save_output(self,State,present_date,name_var=None,t=None):

        for M in self.Models:
            M.save_output(State,present_date,name_var=name_var,t=t)
                

        
###############################################################################
#                       Tangent and Adjoint tests                             #
###############################################################################     
    
def tangent_test(M,State,t0=0,nstep=1):

    # Boundary conditions
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {0:np.random.random((State.ny,State.nx)).astype('float64'),
                        1:np.random.random((State.ny,State.nx)).astype('float64')}
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
        var_bc[name] = {0:np.random.random((State.ny,State.nx)).astype('float64'),
                        1:np.random.random((State.ny,State.nx)).astype('float64')}
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

    
    

    
    
    
    
