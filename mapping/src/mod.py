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
from math import sqrt,pi
from datetime import timedelta
import matplotlib.pylab as plt 
from copy import deepcopy

from . import tools, grid



def Model(config,State):
    """
    NAME
        main class

    DESCRIPTION
        Main function calling subclass for specific models
    """
    print('Model:',config.name_model)
    if config.name_model is None:
        return
    elif config.name_model=='Diffusion':
        return Model_diffusion(config,State)
    elif config.name_model=='QG1L':
        return Model_qg1l(config,State)
    elif config.name_model=='JAX-QG1L':
        return Model_jaxqg1l(config,State)
    elif config.name_model=='QG1LM':
        return Model_qg1lm(config,State)
    elif config.name_model=='SW1L':
        return Model_sw1l(config,State)
    elif config.name_model=='SW1LM':
        return Model_sw1lm(config,State)
    elif hasattr(config.name_model,'__len__') and len(config.name_model)==2:
        return Model_BM_IT(config,State)
    else:
        sys.exit(config.name_model + ' not implemented yet')
        
        
class Model_diffusion:
    
    def __init__(self,config,State):
        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        
        self.snu = config.snu
        self.dx = State.DX
        self.dy = State.DY
        
        # Model Parameters (Flux)
        self.nparams = State.ny*State.nx
        self.sliceparams = slice(0,self.nparams)
        
        # Mask array
        mask = np.zeros((State.ny,State.nx))+2
        mask[:2,:] = 1
        mask[:,:2] = 1
        mask[-2:,:] = 1
        mask[:,-2:] = 1
        
        
        SSH = State.getvar(0)
        
        mdt = None
        self.mdt = mdt
        
    
        if SSH is not None and mdt is not None:
            isNAN = np.isnan(SSH) | np.isnan(mdt)
        elif SSH is not None:
            isNAN = np.isnan(SSH)
        elif mdt is not None:
            isNAN = np.isnan(mdt)
        else:
            isNAN = None
            
        if isNAN is not None: 
            mask[isNAN] = 0
            indNan = np.argwhere(isNAN)
            for i,j in indNan:
                for p1 in [-1,0,1]:
                    for p2 in [-1,0,1]:
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=State.ny-1) & (jtest>=0) & (jtest<=State.nx-1)):
                          if mask[itest,jtest]==2:
                              mask[itest,jtest] = 1
         
        self.mask = mask
        
        if config.name_analysis=='4Dvar' and config.compute_test:
            print('Tangent test:')
            tangent_test(self,State,10,config.flag_use_bc)
            print('Adjoint test:')
            adjoint_test(self,State,10,config.flag_use_bc)
            
            
    def step(self,State,nstep=1,ind=0,t=None):
        # Get state variable
        SSH0 = State.getvar(ind=ind)
        
        # init
        SSH1 = +SSH0
        
        for step in range(nstep):
            SSH1[1:-1,1:-1] += self.dt*self.snu*(\
                (SSH1[1:-1,2:]+SSH1[1:-1,:-2]-2*SSH1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                (SSH1[2:,1:-1]+SSH1[:-2,1:-1]-2*SSH1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
        
        # Update state
        State.setvar(SSH1,ind=ind)
        
    def step_tgl(self,dState,State,nstep=1,ind=0,t=None):
        # Get state variable
        SSH0 = dState.getvar(ind=ind)
        
        # init
        SSH1 = +SSH0
        
        for step in range(nstep):
            SSH1[1:-1,1:-1] += self.dt*self.snu*(\
                (SSH1[1:-1,2:]+SSH1[1:-1,:-2]-2*SSH1[1:-1,1:-1])/(self.dx[1:-1,1:-1]**2) +\
                (SSH1[2:,1:-1]+SSH1[:-2,1:-1]-2*SSH1[1:-1,1:-1])/(self.dy[1:-1,1:-1]**2))
        
        # Update state
        dState.setvar(SSH1,ind=ind)
        
    
    def step_adj(self,adState,State,nstep=1,ind=0,t=None):
        # Get state variable
        adSSH0 = adState.getvar(ind=ind)
        
        # init
        adSSH1 = +adSSH0
        
        for step in range(nstep):
            
            adSSH1[1:-1,2:] += self.dt*self.snu/(self.dx[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            adSSH1[1:-1,:-2] += self.dt*self.snu/(self.dx[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            adSSH1[1:-1,1:-1] += -2*self.dt*self.snu/(self.dx[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            
            adSSH1[2:,1:-1] += self.dt*self.snu/(self.dy[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            adSSH1[:-2,1:-1] += self.dt*self.snu/(self.dy[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            adSSH1[1:-1,1:-1] += -2*self.dt*self.snu/(self.dy[1:-1,1:-1]**2) * adSSH0[1:-1,1:-1]
            
            adSSH0 = +adSSH1
            
        # Update state
        adState.setvar(adSSH1,ind=ind)
        
    
    
    
class Model_jaxqg1l:

    def __init__(self,config,State):
        # Model specific libraries
        if config.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.dir_model  
        SourceFileLoader("qgm",dir_model + "/jqgm.py").load_module() 

        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        self.timestamps = np.asarray(self.timestamps)
        
        # Open MDT map if provided
        if config.Reynolds and config.path_mdt is not None and os.path.exists(config.path_mdt):
            print('MDT is prescribed, thus the QGPV will be expressed thanks \
to Reynolds decomposition. However, be sure that observed and boundary \
variable are SLAs!')
                      
            ds = xr.open_dataset(config.path_mdt).squeeze()
            ds.load()
            
            name_var_mdt = {}
            name_var_mdt['lon'] = config.name_var_mdt['lon']
            name_var_mdt['lat'] = config.name_var_mdt['lat']
            
            
            
            if 'mdt' in config.name_var_mdt and config.name_var_mdt['mdt'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdt']
                self.mdt = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                sys.exit('Warning: wrong variable name for mdt')
            if 'mdu' in config.name_var_mdt and config.name_var_mdt['mdu'] in ds \
                and 'mdv' in config.name_var_mdt and config.name_var_mdt['mdv'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdu']
                self.mdu = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                name_var_mdt['var'] = config.name_var_mdt['mdv']
                self.mdv = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                self.mdu = self.mdv = None
                
        else:
            self.mdt = self.mdu = self.mdv = None
    
        
        # Open Rossby Radius if provided
        if self.mdt is not None and config.filec_aux is not None and os.path.exists(config.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.cmin is not None:
                self.c[self.c<config.cmin] = config.cmin
            
            if config.cmax is not None:
                self.c[self.c>config.cmax] = config.cmax
                
        else:
            self.c = config.c0 * np.ones((State.ny,State.nx))
            
        
        if config.flag_plot>1:
            plt.figure()
            plt.pcolormesh(self.c)
            plt.colorbar()
            plt.show()
            
            
        # Model Parameters (Flux)
        self.nparams = State.ny*State.nx
        self.sliceparams = slice(0,self.nparams)
        
        
        # Model initialization
        qgm = SourceFileLoader("qgm", dir_model + "/jqgm.py").load_module() 
        model = qgm.Qgm
        
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=State.getvar(ind=0),
                         c=self.c,
                         upwind=config.upwind,
                         g=State.g,
                         f=State.f,
                         qgiter=config.qgiter,
                         diff=config.only_diffusion,
                         snu=config.cdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)
        
        if config.name_analysis=='4Dvar' and config.compute_test:
            print('Tangent test:')
            tangent_test(self,State,10,config.flag_use_bc)
            print('Adjoint test:')
            adjoint_test(self,State,10,config.flag_use_bc)
        

    def step(self,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        SSH0 = State.getvar(ind=ind)
        
        # init
        SSH1 = +SSH0
        
        # Boundary condition
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1
        
        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step_jit(SSH1,way=1)
        
        # Update state
        if State.params is not None:
            params = State.params[self.sliceparams].reshape((State.ny,State.nx))
            SSH1 += nstep*self.dt/(3600*24) * params
        State.setvar(SSH1, ind=ind)


    def step_tgl(self,dState,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        dSSH0 = dState.getvar(ind=ind)
        SSH0 = State.getvar(ind=ind)
        
        # init
        dSSH1 = +dSSH0
        SSH1 = +SSH0
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            dSSH1 = (1-Wbc)*dSSH1
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1
        
        # Time propagation
        for i in range(nstep):
            dSSH1 = self.qgm.step_tgl(dh0=dSSH1,h0=SSH1)
            SSH1 = self.qgm.step(h0=SSH1)
        
        # Update state
        if dState.params is not None:
            dparams = dState.params[self.sliceparams].reshape((State.ny,State.nx))
            dSSH1 += nstep*self.dt/(3600*24) * dparams
        dState.setvar(dSSH1,ind=ind)
        
        
    def step_adj(self,adState,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        adSSH0 = adState.getvar(ind=ind)
        SSH0 = State.getvar(ind=ind)
        
        # Init
        adSSH1 = +adSSH0
        SSH1 = +SSH0
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1

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
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            adSSH1 = (1-Wbc)*adSSH1
        
        # Update state  and parameters
        if adState.params is not None:
            adState.params[self.sliceparams] += nstep*self.dt/(3600*24) * adSSH0.flatten()
            
        adSSH1 = adSSH1.at[np.isnan(adSSH1)].set(0)
        adState.setvar(adSSH1,ind=ind)
        
        
    
        
        
class Model_qg1l:

    def __init__(self,config,State):
        # Model specific libraries
        if config.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.dir_model  
        SourceFileLoader("qgm",dir_model + "/qgm.py").load_module() 

        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        self.timestamps = np.asarray(self.timestamps)
        
        # Open MDT map if provided
        if config.Reynolds and config.path_mdt is not None and os.path.exists(config.path_mdt):
            print('MDT is prescribed, thus the QGPV will be expressed thanks \
to Reynolds decomposition. However, be sure that observed and boundary \
variable are SLAs!')
                      
            ds = xr.open_dataset(config.path_mdt).squeeze()
            ds.load()
            
            name_var_mdt = {}
            name_var_mdt['lon'] = config.name_var_mdt['lon']
            name_var_mdt['lat'] = config.name_var_mdt['lat']
            
            
            
            if 'mdt' in config.name_var_mdt and config.name_var_mdt['mdt'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdt']
                self.mdt = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                
                self.mdt[np.isnan(self.mdt)] = 0
                
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.show()
            else:
                sys.exit('Warning: wrong variable name for mdt')
            if 'mdu' in config.name_var_mdt and config.name_var_mdt['mdu'] in ds \
                and 'mdv' in config.name_var_mdt and config.name_var_mdt['mdv'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdu']
                self.mdu = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                name_var_mdt['var'] = config.name_var_mdt['mdv']
                self.mdv = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                self.mdu = self.mdv = None
                
        else:
            self.mdt = self.mdu = self.mdv = None
    
        
        # Open Rossby Radius if provided
        if self.mdt is not None and config.filec_aux is not None and os.path.exists(config.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.cmin is not None:
                self.c[self.c<config.cmin] = config.cmin
            
            if config.cmax is not None:
                self.c[self.c>config.cmax] = config.cmax
                
        else:
            self.c = config.c0 * np.ones((State.ny,State.nx))
            
        
        if config.flag_plot>1:
            plt.figure()
            plt.pcolormesh(self.c)
            plt.colorbar()
            plt.show()
            
            
        # Model Parameters (Flux)
        self.nparams = State.ny*State.nx
        self.sliceparams = slice(0,self.nparams)
        
        
        # Model initialization
        SourceFileLoader("qgm", 
                                 dir_model + "/qgm.py").load_module() 
        SourceFileLoader("qgm_tgl", 
                                 dir_model + "/qgm_tgl.py").load_module() 
        
        if config.name_analysis in ['4Dvar','incr4Dvar']:
            qgm_adj = SourceFileLoader("qgm_adj", 
                                     dir_model + "/qgm_adj.py").load_module() 
            model = qgm_adj.Qgm_adj
        else:
            qgm = SourceFileLoader("qgm", 
                                     dir_model + "/qgm.py").load_module() 
            model = qgm.Qgm
        
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=State.getvar(ind=0),
                         c=self.c,
                         upwind=config.upwind,
                         upwind_adj=config.upwind_adj,
                         g=State.g,
                         f=State.f,
                         qgiter=config.qgiter,
                         qgiter_adj=config.qgiter_adj,
                         diff=config.only_diffusion,
                         snu=config.cdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)
        
        print('qgiter:',self.qgm.qgiter)
        print('qgiter_adj:',self.qgm.qgiter_adj)
        print('upwind:',self.qgm.upwind)
        print('upwind_adj:',self.qgm.upwind_adj)
        
        
        
        if config.name_analysis=='4Dvar' and config.compute_test and config.name_model=='QG1L':
            print('Tangent test:')
            tangent_test(self,State,10,config.flag_use_boundary_conditions)
            print('Adjoint test:')
            adjoint_test(self,State,10,config.flag_use_boundary_conditions)

    def step(self,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        SSH0 = State.getvar(ind=ind)
        
        # init
        SSH1 = +SSH0
        
        # Boundary condition
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1
        
        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step(SSH1,way=1)
        
        # Update state
        if State.params is not None:
            params = State.params[self.sliceparams].reshape((State.ny,State.nx))
            SSH1 += nstep*self.dt/(3600*24) * params
        State.setvar(SSH1, ind=ind)

    
            
    def step_nudging(self,State,tint,Hbc=None,Wbc=None,Nudging_term=None,t=None):
    
        # Read state variable
        ssh_0 = State.getvar(0)
        
        if len(State.name_var)>1 and State.name_var[1] in State.var:
            flag_pv = True
            pv_0 = State.getvar(1)
        else:
            flag_pv = False
            pv_0 = self.qgm.h2pv(ssh_0)

        # Boundary condition
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            Qbc = self.qgm.h2pv(Hbc)
            ssh_0 = Wbc*Hbc + (1-Wbc)*ssh_0
            pv_0 = Wbc*Qbc + (1-Wbc)*pv_0
        
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
                pv_1[indNoNan] += (1-Wbc[indNoNan]) *\
                    Nudging_term['rv'][indNoNan]
            # Nudging towards ssh
            if np.any(np.isfinite(Nudging_term['ssh'])):
                indNoNan = (~np.isnan(Nudging_term['ssh'])) & (self.qgm.mask>1) 
                pv_1[indNoNan] -= (1-Wbc[indNoNan]) *\
                    (State.g*State.f[indNoNan])/self.c[indNoNan]**2 * \
                        Nudging_term['ssh'][indNoNan]
                # Inversion pv -> ssh
                ssh_b = +ssh_1
                ssh_1[indNoNan] = self.qgm.pv2h(pv_1,ssh_b)[indNoNan]
        
        if np.any(np.isnan(ssh_1[self.qgm.mask>1])):
            if Hbc is not None:
                ind = (np.isnan(ssh_1)) & (self.qgm.mask>1)
                ssh_1[ind] = Hbc[ind] 
                print('Warning: Invalid value encountered in mod_qg1l, we replace by boundary values')
                print(np.where(ind))
            else: sys.exit('Invalid value encountered in mod_qg1l')
            
        # Update state 
        State.setvar(ssh_1,0)
        if flag_pv:
            State.setvar(pv_1,1)
        
        
    def step_tgl(self,dState,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        dSSH0 = dState.getvar(ind=ind)
        SSH0 = State.getvar(ind=ind)
        
        # init
        dSSH1 = +dSSH0
        SSH1 = +SSH0
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            dSSH1 = (1-Wbc)*dSSH1
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1
        
        # Time propagation
        for i in range(nstep):
            dSSH1 = self.qgm.step_tgl(dh0=dSSH1,h0=SSH1)
            SSH1 = self.qgm.step(h0=SSH1)
        
        # Update state
        if dState.params is not None:
            dparams = dState.params[self.sliceparams].reshape((State.ny,State.nx))
            dSSH1 += nstep*self.dt/(3600*24) * dparams
        dState.setvar(dSSH1,ind=ind)
        
        
    def step_adj(self,adState,State,nstep=1,Hbc=None,Wbc=None,ind=0,t=None):
        
        # Get state variable
        adSSH0 = adState.getvar(ind=ind)
        SSH0 = State.getvar(ind=ind)
        
        # Init
        adSSH1 = +adSSH0
        SSH1 = +SSH0
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1

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
        
        # Boundary conditions
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            adSSH1 = (1-Wbc)*adSSH1
        
        # Update state  and parameters
        if adState.params is not None:
            adState.params[self.sliceparams] += nstep*self.dt/(3600*24) * adSSH0.flatten()
            
        adSSH1[np.isnan(adSSH1)] = 0
        adState.setvar(adSSH1,ind=ind)
        

        
        
        

class Model_qg1lm:

    def __init__(self,config,State):
        # Model specific libraries
        if config.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.dir_model 

        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        self.timestamps = np.asarray(self.timestamps)
        
        # Open MDT map if provided
        if config.Reynolds and config.path_mdt is not None and os.path.exists(config.path_mdt):
            print('MDT is prescribed, thus the QGPV will be expressed thanks \
to Reynolds decomposition. However, be sure that observed and boundary \
variable are SLAs!')
                      
            ds = xr.open_dataset(config.path_mdt).squeeze()
            ds.load()
            
            name_var_mdt = {}
            name_var_mdt['lon'] = config.name_var_mdt['lon']
            name_var_mdt['lat'] = config.name_var_mdt['lat']
            
            
            
            if 'mdt' in config.name_var_mdt and config.name_var_mdt['mdt'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdt']
                self.mdt = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                sys.exit('Warning: wrong variable name for mdt')
            if 'mdu' in config.name_var_mdt and config.name_var_mdt['mdu'] in ds \
                and 'mdv' in config.name_var_mdt and config.name_var_mdt['mdv'] in ds:
                name_var_mdt['var'] = config.name_var_mdt['mdu']
                self.mdu = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
                name_var_mdt['var'] = config.name_var_mdt['mdv']
                self.mdv = grid.interp2d(ds,
                                         name_var_mdt,
                                         State.lon,
                                         State.lat)
            else:
                self.mdu = self.mdv = None
                
        else:
            self.mdt = self.mdu = self.mdv = None
    
        
        # Open Rossby Radius if provided
        if self.mdt is not None and config.filec_aux is not None and os.path.exists(config.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.cmin is not None:
                self.c[self.c<config.cmin] = config.cmin
            
            if config.cmax is not None:
                self.c[self.c>config.cmax] = config.cmax
                
        else:
            self.c = config.c0 * np.ones((State.ny,State.nx))
            
        
        if config.flag_plot>1:
            plt.figure()
            plt.pcolormesh(self.c)
            plt.colorbar()
            plt.show()
            
            
        # Model Parameters (Flux)
        self.nparams = 2*State.ny*State.nx
        self.sliceparams_ls = slice(0,State.ny*State.nx)
        self.sliceparams_ss = slice(State.ny*State.nx,self.nparams)
        
        
        # Model initialization
        qgm = SourceFileLoader("qgm", dir_model + "/jqgm.py").load_module() 
        model = qgm.Qgm

        
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=State.getvar(ind=0),
                         c=self.c,
                         upwind=config.upwind,
                         upwind_adj=config.upwind_adj,
                         g=State.g,
                         f=State.f,
                         qgiter=config.qgiter,
                         qgiter_adj=config.qgiter_adj,
                         diff=config.only_diffusion,
                         snu=config.cdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)

        if config.name_analysis=='4Dvar' and config.compute_test:
            print('Tangent test:')
            tangent_test(self,State,0,nstep=2)
            print('Adjoint test:')
            adjoint_test(self,State,1)


    def step(self,State,nstep=1,t=None):
        
        # Get state variable
        SSH0 = State.getvar(ind=[0,1],vect=True)

        # init
        SSH1 = +SSH0
        
        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step_mean_jit(SSH1)


        # Update state
        if State.params is not None:
            SSH1 += nstep*self.dt/(3600*24) * State.params
            
            
        SSHls1 = SSH1[:self.ny*self.nx].reshape((self.ny,self.nx))
        SSHss1 = SSH1[self.ny*self.nx:].reshape((self.ny,self.nx))
        State.setvar(SSHls1, ind=0)
        State.setvar(SSHss1, ind=1)
        State.setvar(SSHls1+SSHss1,ind=2)
        
        
        
    def step_tgl(self,dState,State,nstep=1,t=None):
        
        # Get state variable
        dSSH0 = dState.getvar(ind=[0,1],vect=True)
        SSH0 = State.getvar(ind=[0,1],vect=True)
        
        # init
        dSSH1 = +dSSH0
        SSH1 = +SSH0
        
        # Time propagation
        for i in range(nstep):
            dSSH1 = self.qgm.step_mean_tgl_jit(dSSH1,SSH1)
            SSH1 = self.qgm.step_mean_jit(SSH1)

        # Update state
        if dState.params is not None:
            dSSH1 += nstep*self.dt/(3600*24) * dState.params
        
        dSSHls1 = dSSH1[:self.ny*self.nx].reshape((self.ny,self.nx))
        dSSHss1 = dSSH1[self.ny*self.nx:].reshape((self.ny,self.nx))
        dState.setvar(dSSHls1, ind=0)
        dState.setvar(dSSHss1, ind=1)
        dState.setvar(dSSHls1+dSSHss1,ind=2)
  
        
    def step_adj(self,adState,State,nstep=1,t=None):
        
        # Get state variable
        adSSHtot = adState.getvar(ind=2,vect=True) 
        adSSHls0 = adState.getvar(ind=0,vect=True) + adSSHtot
        adSSHss0 = adState.getvar(ind=1,vect=True) + adSSHtot
        adSSH0 = np.concatenate((adSSHls0,adSSHss0))
        SSH0 = State.getvar(ind=[0,1],vect=True)
        
        # init
        adSSH1 = +adSSH0 
        SSH1 = +SSH0
        
        # Current trajectory
        traj = [SSH1]
        if nstep>1:
            for i in range(nstep):
                SSH1 = self.qgm.step_mean_jit(SSH1)
                traj.append(SSH1)
        
        # Time propagation
        for i in reversed(range(nstep)):
            SSH1 = traj[i]
            adSSH1 = self.qgm.step_mean_adj_jit(adSSH1,SSH1)
        
        # Update state  and parameters
        if adState.params is not None:
            adState.params += nstep*self.dt/(3600*24) * adSSH0 
            
        adSSHls1 = adSSH1[:self.ny*self.nx].reshape((self.ny,self.nx))
        adSSHss1 = adSSH1[self.ny*self.nx:].reshape((self.ny,self.nx))

        adState.setvar(adSSHls1,ind=0)
        adState.setvar(adSSHss1,ind=1)
        adState.setvar(np.zeros((self.ny,self.nx)),ind=2)
        
        
class Model_sw1l:
    def __init__(self,config,State,
                 He_init=None,D_He=None,T_He=None,D_bc=None,T_bc=None):
        self.config = config
        # Model specific libraries
        if config.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = config.dir_model
            
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
        
        swm_adj = SourceFileLoader("swm_adj", 
                                 dir_model + "/swm_adj.py").load_module() 
        
        # Model grid 
        self.sw_in = config.sw_in # Avoding boundary pixels
        print('Length of the boundary band to ignore:',self.sw_in)
        self.nyin,self.nxin = State.ny -2*self.sw_in, State.nx -2*self.sw_in
        if self.sw_in>0:
            self.Xin = State.X[self.sw_in:-self.sw_in,self.sw_in:-self.sw_in]
            self.Yin = State.Y[self.sw_in:-self.sw_in,self.sw_in:-self.sw_in]
            self.fin = State.f[self.sw_in:-self.sw_in,self.sw_in:-self.sw_in]
        else:
            self.Xin = +State.X
            self.Yin = +State.Y
            self.fin = +State.f
            
        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.time_scheme = config.sw_time_scheme
        print('time scheme:',self.time_scheme)
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        self.timestamps = np.asarray(self.timestamps)        
        
        if config.He_data is not None and os.path.exists(config.He_data['path']):
            ds = xr.open_dataset(config.He_data['path'])
            self.Heb = ds[config.He_data['var']].values
        else:
            if He_init is None:
                self.Heb = config.He_init
            else:
                self.Heb = He_init
            print('Heb:',self.Heb)
            
        if config.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.Ntheta,pi/2/config.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
            print(self.bc_theta)
        else:
            self.bc_theta = np.array([0])
            
        self.omegas = np.asarray(config.w_igws)
        self.bc_kind = config.bc_kind
        
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
        self.sliceparams = slice(0,self.nparams)
        
        # Model initialization
        self.swm = swm_adj.Swm_adj(X=self.Xin,
                                   Y=self.Yin,
                                   dt=self.dt,
                                   bc=self.bc_kind,
                                   omegas=self.omegas,
                                   bc_theta=self.bc_theta,
                                   f=self.fin)
        
        if self.time_scheme=='Euler':
            self.swm_step = self.swm.step_euler
            self.swm_step_tgl = self.swm.step_euler_tgl
            self.swm_step_adj = self.swm.step_euler_adj
        elif self.time_scheme=='lf':
            self.swm_step = self.swm.step_lf
            self.swm_step_tgl = self.swm.step_lf_tgl
            self.swm_step_adj = self.swm.step_lf_adj
        elif self.time_scheme=='rk4':
            self.swm_step = self.swm.step_rk4
            self.swm_step_tgl = self.swm.step_rk4_tgl
            self.swm_step_adj = self.swm.step_rk4_adj
        
        self.mdt = None
        
        # Tests
        if config.name_analysis=='4Dvar' and config.compute_test and config.name_model=='SW1L':
            print('tangent test:')
            tangent_test(self,State,self.T[-1],nstep=1)
            print('adjoint test:')
            adjoint_test(self,State,self.T[-1],nstep=1)
       
            
    def step(self,State,nstep=1,t0=0,ind=[0,1,2],t=None):

        # Init
        u0,v0,h0 = State.getvar(ind)
        u = +u0
        v = +v0
        h = +h0
        
        # Get params in physical space
        if State.params is not None:
            He = State.params[self.sliceHe].reshape(self.shapeHe)+self.Heb
            hbcx = State.params[self.slicehbcx].reshape(self.shapehbcx)
            hbcy = State.params[self.slicehbcy].reshape(self.shapehbcy)
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
            
        State.setvar([u,v,h],ind=ind)
        
    
    def step_tgl(self,dState,State,nstep=1,t0=0,ind=[0,1,2],t=None):
        
        # Get state variables and model parameters
        du0,dv0,dh0 = dState.getvar(ind=ind)
        u0,v0,h0 = State.getvar(ind=ind)
        
        if State.params is not None:
            He = State.params[self.sliceHe].reshape(self.shapeHe)+self.Heb
            hbcx = State.params[self.slicehbcx].reshape(self.shapehbcx)
            hbcy = State.params[self.slicehbcy].reshape(self.shapehbcy)
        else:
            He = hbcx = hbcy = None
            
        if dState.params is not None:
            dHe = dState.params[self.sliceHe].reshape(self.shapeHe)
            dhbcx = dState.params[self.slicehbcx].reshape(self.shapehbcx)
            dhbcy = dState.params[self.slicehbcy].reshape(self.shapehbcy)
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
            
        dState.setvar([du,dv,dh],ind=ind)
        

    def step_adj(self,adState, State, nstep=1, t0=0,ind=None,t=None):
        
        # Get variables
        adu0,adv0,adh0 = adState.getvar(ind=ind)
        u0,v0,h0 = State.getvar(ind=ind)
        
        if State.params is not None:
            He = State.params[self.sliceHe].reshape(self.shapeHe)+self.Heb
            hbcx = State.params[self.slicehbcx].reshape(self.shapehbcx)
            hbcy = State.params[self.slicehbcy].reshape(self.shapehbcy)
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
        adState.setvar([adu,adv,adh],ind=ind)
        
        # Update parameters
        adState.params[self.sliceparams] += np.concatenate((adHe.flatten(), adhbcx.flatten(), adhbcy.flatten()))
    

class Model_sw1lm:
    
    def __init__(self,config,State):
        #if config.Nmodes==1:
        #    sys.exit('Error: *Nmodes has to be >1 for SW1LM model')
        self.Nmodes = config.Nmodes
        He_init = self.check_param(config,'He_init')
        D_He = self.check_param(config,'D_He')
        T_He = self.check_param(config,'T_He')
        D_bc = self.check_param(config,'D_bc')
        T_bc = self.check_param(config,'T_bc')
        
        self.Models = []
        self.nParams = 0
        self.sliceHe = np.array([],dtype=int)
        self.slicehbcx = np.array([],dtype=int)
        self.slicehbcy = np.array([],dtype=int)
        print()
        self.ind = []
        for i in range(self.Nmodes):
            print(f'* Mode {i+1}')
            M = Model_sw1l(config,State,He_init[i],D_He[i],T_He[i],D_bc[i],T_bc[i])
            self.Models.append(M)
            self.sliceHe = np.append(self.sliceHe,
                                     self.nParams+\
                                         np.arange(M.sliceHe.start,M.sliceHe.stop))
            self.slicehbcx = np.append(self.slicehbcx,
                                       self.nParams+\
                                           np.arange(M.slicehbcx.start,M.slicehbcx.stop))
            self.slicehbcy = np.append(self.slicehbcy,
                                       self.nParams+\
                                         np.arange(M.slicehbcy.start,M.slicehbcy.stop))
            self.nParams += M.nParams
            
            indi = np.arange(3*i,3*(i+1)) # Indexes for state variables relative to mode i 
            if hasattr(config['name_model'],'__len__') and len(config['name_model'])==2:
                # BM & IT separation : we exclude first variable which is for SSH BM  
                self.ind.append(1+indi)
            else:
                self.ind.append(indi)
            
            if i==0:
                self.nt = M.nt
                self.dt = M.dt
                self.T = M.T.copy()
                self.timestamps = M.timestamps.copy()
            print()
            
        inds = np.arange(3*self.Nmodes,3*(self.Nmodes+1)) # Indexes for state variables relative to the sum of all modes
        if hasattr(config['name_model'],'__len__') and len(config['name_model'])==2:
            # BM & IT separation : we exclude first variable which is for SSH BM  
            self.ind.append(1+inds)
        else:
            self.ind.append(inds)
        
        # Tests
        if config.name_analysis=='4Dvar' and config.compute_test and config.name_model=='SW1LM':
            print('tangent test:')
            tangent_test(M,State,self.T[10],nstep=config.checkpoint)
            print('adjoint test:')
            adjoint_test(M,State,self.T[10],nstep=config.checkpoint)
        
    def restart(self):
        for M in self.Models:
            M.swm.restart()
    
    def check_param(self,config,name):
        
        if hasattr(config[name],'__len__'):
            if len(config[name])!=self.Nmodes:
                print(f'Warning: len({name}) != Nmodes \
                      --> We take the first value for each mode')
                param = [config[name][0] for _ in range(self.Nmodes)]
            else:
                param = config[name]
        else:
            print(f'Warning: {name} is not a list!')
            param = [config[name] for _ in range(self.Nmodes)]
        
        return param
    
    def slice_param(self,imode):
        
        i0 = 0
        if imode>0:
            for i in range(imode):
                i0 += self.Models[i].nParams
        return slice(i0,i0+self.Models[imode].nParams)
    
    def step(self,t,State,params,nstep=1,t0=0,ind=None):
        u = 0
        v = 0
        h = 0
        for i in range(self.Nmodes):
            _params = params[self.slice_param(i)]
            self.Models[i].step(t,State,_params,nstep,t0,self.ind[i])
            _u,_v,_h = State.getvar(ind=self.ind[i])
            u += _u
            v += _v 
            h += _h
        State.setvar([u,v,h],ind=self.ind[-1])
        
    def step_tgl(self,t,dState,State,dparams,params,nstep=1,t0=0,ind=None):
        du = 0
        dv = 0
        dh = 0
        for i in range(self.Nmodes):
            _params = params[self.slice_param(i)]
            _dparams = dparams[self.slice_param(i)]
            
            self.Models[i].step_tgl(
                t,dState,State,_dparams,_params,nstep,t0,ind=self.ind[i]) 
            
            _du,_dv,_dh = dState.getvar(ind=self.ind[i])
            du += _du
            dv += _dv 
            dh += _dh
            
        dState.setvar([du,dv,dh],ind=self.ind[-1]) 
    
    def step_adj(self,t,adState, State, adparams0, params, nstep=1, t0=0,ind=None):

        adparams = +adparams0*0
        
        adu,adv,adh = adState.getvar(ind=self.ind[-1]) 
        
        for i in range(self.Nmodes):
            
            _adu,_adv,_adh = adState.getvar(ind=self.ind[i]) 
            adState.setvar([_adu+adu,_adv+adv,_adh+adh],ind=self.ind[i]) 
            
            _params = params[self.slice_param(i)]
            _adparams0 = adparams0[self.slice_param(i)]
            
            _adparams = self.Models[i].step_adj(
                t,adState,State,_adparams0,_params,nstep,t0,ind=self.ind[i])
            
            
            adparams[self.slice_param(i)] = _adparams
            
        adu,adv,adh = adState.getvar(ind=self.ind[-1])
        adState.setvar([0*adu,0*adv,0*adh],ind=self.ind[-1]) 
        
        return adparams
    
        
    def run(self,t0,tint,State,params,return_traj=False,nstep=1):
        if return_traj:
            tt = [t0]
            traj = [State.getvar()]
        t = t0
        while t <= t0+tint: 
            _nstep = nstep
            self.step(t,State,params,nstep=_nstep,t0=t0)
            t += nstep*self.dt
            if return_traj:
                traj.append(State.getvar())
                tt.append(t)
        if return_traj:
            return tt,traj

    def run_tgl(self,t0,tint,dState,State,dparams,params,nstep=1):
        State_tmp = State.copy()
        tt,traj = self.run(t0,tint,State_tmp,params,return_traj=True,nstep=nstep)
        for i,t in enumerate(tt[:-1]): 
            State_tmp.setvar(traj[i])
            _nstep = nstep
            self.step_tgl(t,dState,State_tmp,dparams,params,nstep=_nstep,t0=t0)
    
    def run_adj(self,t0,tint,adState,State,adparams,params,nstep=1):
        State_tmp = State.copy()
        tt,traj = self.run(t0,tint,State_tmp,params,return_traj=True,nstep=nstep)
        for i in reversed(range(len(tt[:-1]))):
            t = tt[i]
            State_tmp.setvar(traj[i])
            _nstep = nstep
            adparams = self.step_adj(
                t,adState,State_tmp,adparams,params,nstep=_nstep,t0=t0)
        return adparams
    
    
        
            
class Model_BM_IT:
    
    def __init__(self,config,State):
        print('\n* BM Model')
        if config.name_model[0]=='Diffusion':
            self.Model_BM = Model_diffusion(config,State)
        elif config.name_model[0]=='QG1L':
            self.Model_BM = Model_qg1l(config,State)
        elif config.name_model[0]=='JAX-QG1L':
            self.Model_BM = Model_jaxqg1l(config,State)
            
        print('\n* IT Model')
        if config.name_model[1]=='SW1L':
            self.Model_IT = Model_sw1l(config,State)
        elif config.name_model[1]=='SW1LM':
            self.Model_IT = Model_sw1lm(config,State)
        
        self.timestamps = self.Model_BM.timestamps
        self.dt = self.Model_BM.dt
        self.T = self.Model_BM.T
        
        self.mdt = self.Model_BM.mdt
    
        if config.name_model[1]=='SW1L':
            self.indit = [1,2,3]
        elif config.name_model[1]=='SW1LM':
            self.indit = None
        
        # Model parameters slices (first slices for BM, the others for IT)
        self.Model_IT.sliceHe = slice(self.Model_BM.nparams,
                                      self.Model_BM.nparams + np.prod(self.Model_IT.shapeHe))
        self.Model_IT.slicehbcx = slice(self.Model_BM.nparams + np.prod(self.Model_IT.shapeHe),
                               self.Model_BM.nparams+ np.prod(self.Model_IT.shapeHe)+np.prod(self.Model_IT.shapehbcx))
        self.Model_IT.slicehbcy = slice(self.Model_BM.nparams + np.prod(self.Model_IT.shapeHe)+np.prod(self.Model_IT.shapehbcx),
                               self.Model_BM.nparams + np.prod(self.Model_IT.shapeHe)+np.prod(self.Model_IT.shapehbcx)+np.prod(self.Model_IT.shapehbcy))
        self.Model_IT.sliceparams = slice(self.Model_BM.nparams,
                                          self.Model_BM.nparams + self.Model_IT.nparams)
        self.nparams = self.Model_BM.nparams + self.Model_IT.nparams
        
        if config.compute_test:
            print('tangent test:')
            tangent_test(self,State,self.Model_BM.T[10],nstep=config.checkpoint)
            print('adjoint test:')
            adjoint_test(self,State,self.Model_BM.T[10],nstep=config.checkpoint)
        
        
    def step(self,t,State,Hbc=None,Wbc=None,nstep=1,t0=0):
        
        h = 0

        self.Model_BM.step(t=t,State=State,nstep=nstep,Hbc=Hbc,Wbc=Wbc,ind=0)
        h += State.getvar(ind=0)
        
        self.Model_IT.step(t=t,State=State,nstep=nstep,t0=t0,ind=self.indit)
        h += State.getvar(ind=-2)
        
        State.setvar(h,ind=-1)
        
        
    def step_tgl(self,t,dState,State,Hbc=None,Wbc=None,nstep=1,t0=0):
        
        dh = 0
          
        self.Model_BM.step_tgl(t=t,State=State,dState=dState,nstep=nstep,ind=0)
        dh += dState.getvar(ind=0)
        
        self.Model_IT.step_tgl(t=t,State=State,dState=dState,nstep=nstep, 
                               t0=t0,ind=self.indit)
        dh += dState.getvar(ind=-2)

        dState.setvar(dh,ind=-1)
    
    def step_adj(self,t,adState,State,Hbc=None,Wbc=None,nstep=1,t0=0):
        
        adh = adState.getvar(ind=-1) 
        
        _adh = adState.getvar(ind=0) 
        adState.setvar(adh+_adh,ind=0)
        self.Model_BM.step_adj(t=t,adState=adState,State=State,nstep=nstep,ind=0,Hbc=Hbc,Wbc=Wbc)
    
        _adh = adState.getvar(ind=-2) 
        adState.setvar(adh+_adh,ind=-2)
        self.Model_IT.step_adj(t=t,adState=adState,State=State,nstep=nstep,t0=t0,ind=self.indit)
        adh += adState.getvar(ind=-2)

        adState.setvar(0*adh,ind=-1)

    
     
    
    
    
    
    
def tangent_test(M,State,tint,t0=0,nstep=1):

    State0 = State.random()
    dState = State.random()
    
    State0.params =  np.random.random((M.nparams,))
    dState.params =  np.random.random((M.nparams,))
    
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
        
    
def adjoint_test(M,State,tint,t0=0,nstep=1):
    
    # Current trajectory
    State0 = State.random()
    State0.params =  np.random.random((M.nparams,))
    
    # Perturbation
    dState = State.random()
    dX = dState.getvar(vect=True)
    dState.params = np.random.random((M.nparams,))
    
    # Adjoint
    adState = State.random()
    adState.params = np.zeros((M.nparams,))
    adX = adState.getvar(vect=True)
    
    # Run TLM
    M.step_tgl(t=t0,dState=dState,State=State0,nstep=nstep)
    TLM = dState.getvar(vect=True)
    
    # Run ADJ
    M.step_adj(t=t0,adState=adState,State=State0,nstep=nstep)
    ADM = adState.getvar(vect=True)
    
    mask = np.isnan(TLM+adX+dX)
    
    ps1 = np.inner(TLM[~mask],adX[~mask]) + np.inner(dState.params,adState.params*0) 
    ps2 = np.inner(dX[~mask],ADM[~mask]) + np.inner(dState.params,adState.params) 
    
    print(ps1/ps2)

    
    
    
    
    
    
