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
from scipy.interpolate import griddata

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
    elif config.name_model=='QG1L':
        return Model_qg1l(config,State)
    elif config.name_model=='SW1L':
        return Model_sw1l(config,State)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        
        
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

        # Model timestep
        self.dt = config.dtmodel
        
        # Open MDT map if provided
        if config.path_mdt is not None and os.path.exists(config.path_mdt):
            print('MDT is prescribed, thus the QGPV will be expressed thanks \
to Reynolds decomposition. However, be sure that observed and boundary \
variable are SLAs!')
                      
            ds = xr.open_dataset(config.path_mdt)
            
            self.mdt = grid.interp2d(ds,
                                     config.name_var_mdt,
                                     State.lon,
                                     State.lat)
        else:
            self.mdt = None
        
        # Open Rossby Radius if provided
        if config.filec_aux is not None and os.path.exists(config.filec_aux):
            
            print('Rossby Radius is prescribed, be sure to have provided MDT as well')

            ds = xr.open_dataset(config.filec_aux)
            
            self.c = grid.interp2d(ds,
                                   config.name_var_c,
                                   State.lon,
                                   State.lat)
            self.c[self.c>3.5] = 3.5
            self.c[self.c<2.2] = 2.2
        else:
            self.c = config.c0 * np.ones((State.ny,State.nx))
        
        
        # Model initialization
        SourceFileLoader("qgm", 
                                 dir_model + "/qgm.py").load_module() 
        SourceFileLoader("qgm_tgl", 
                                 dir_model + "/qgm_tgl.py").load_module() 
        
        qwm_adj = SourceFileLoader("qgm_adj", 
                                 dir_model + "/qgm_adj.py").load_module() 
        model = qwm_adj.Qgm_adj
            
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=State.getvar(ind=0),
                         c=config.c0,
                         g=State.g,
                         f=State.f,
                         qgiter=config.qgiter,
                         diff=config.only_diffusion,
                         snu=config.cdiffus,
                         mdt=self.mdt)
        self.State = State
        

        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)

        if config.name_analysis=='4Dvar':
            print('Tangent test:')
            self.tangent_test(State,10)
            
            print('Adjoint test:')
            self.adjoint_test(State,10)

    def step(self,State,nstep=1):
        
        # Get state variable
        SSH0 = State.getvar(ind=0)
        
        # init
        SSH1 = +SSH0
        
        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step(SSH1,way=1)
            
        # Update state
        State.setvar(SSH1,ind=0)
    
    def step_id(self,State,nstep=1) :
        
        # get the ssh
        SSH0 = State.getvar(ind=0)
        
        # init
        SSH1 = +SSH0
        
        # update state
        State.setvar(SSH1,ind=0)
        
    def step_adj_id(self,adState,State,nstep=1):
        
        # Get state variable
        adSSH0 = adState.getvar(ind=0)
        SSH0 = State.getvar(ind=0)
        
        # Init
        adSSH1 = +adSSH0

        # Update state  and parameters
        adState.setvar(adSSH1,ind=0)
        
        return
            
    def step_nudging(self,State,tint,Hbc=None,Wbc=None,Nudging_term=None):
    
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
                    (self.State.g*self.State.f[indNoNan])/self.c[indNoNan]**2 * \
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
        
    def step_tgl(self,dState,State,nstep=1):
        
        # Get state variable
        dSSH0 = dState.getvar(ind=0)
        SSH0 = State.getvar(ind=0)
        
        # init
        dSSH1 = +dSSH0
        SSH1 = +SSH0
        
        # Time propagation
        for i in range(nstep):
            
            dSSH1 = self.qgm.step_tgl(dh0=dSSH1,h0=SSH1)
            SSH1 = self.qgm.step(h0=SSH1)
        
        # Update state
        dState.setvar(dSSH1,ind=0)
        
    def step_adj(self,adState,State,nstep=1):
        
        # Get state variable
        adSSH0 = adState.getvar(ind=0)
        SSH0 = State.getvar(ind=0)
        
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
        
        # Update state  and parameters
        adState.setvar(adSSH1,ind=0)
        
        return
    
    
    def tangent_test(self,State,nstep):
    
        State0 = State.random(1e-2)
        dState = State.random(1e-2)
        
        State0_tmp = State0.copy()
        self.step(State0_tmp,nstep)
        X2 = State0_tmp.getvar(vect=True)

        for p in range(10):
            
            lambd = 10**(-p)
            
            State1 = dState.copy()
            State1.scalar(lambd)
            State1.Sum(State0)
            self.step(State1,nstep)
            X1 = State1.getvar(vect=True)
            
            dState1 = dState.copy()
            dState1.scalar(lambd)
            self.step_tgl(dState1,State0,nstep)
            dX = dState1.getvar(vect=True)
            
            ps = np.linalg.norm(X1-X2-dX)/np.linalg.norm(dX)

            print('%.E' % lambd,'%.E' % ps)
         
            
    def adjoint_test(self,State,nstep):
        
        # Current trajectory
        State0 = State.random(1e-2)
        
        # Perturbation
        dState = State.random()
        dX = dState.getvar(vect=True)

        # Adjoint
        adState = State.random()
        adY = adState.getvar(vect=True)
        
        # Run TLM
        self.step_tgl(dState,State0,nstep=nstep)
        dY = dState.getvar(vect=True)
        
        # Run ADJ
        self.step_adj(adState,State0,nstep=nstep)
        adX = adState.getvar(vect=True)
       
        ps1 = np.inner(dX,adX)
        ps2 = np.inner(dY,adY)
        
        print(ps1/ps2)
        
class Model_sw1l:
    def __init__(self,config,State):
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
                
        
        # State variable dimensions
        self.shapeu = State.var[0].shape
        self.shapev = State.var[1].shape
        self.shapeh = State.var[2].shape 
        self.nu = np.prod(self.shapeu)
        self.nv = np.prod(self.shapev)
        self.nh = np.prod(self.shapeh)

        # Get background value for He
        if config.He_data is not None and os.path.exists(config.He_data['path']):
            ds = xr.open_dataset(config.He_data['path'])
            self.Heb = ds[config.He_data['var']].values
        else:
            self.Heb = config.He_init
            
        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        cfl = min(State.dx, State.dy) / np.max(np.sqrt(State.g * self.Heb))
        print('CFL:',cfl)
        if self.dt>=cfl:
            print('WARNING: timestep>=CFL')
        self.T = np.arange(self.nt) * self.dt
        self.time_scheme = config.sw_time_scheme
        print('time scheme:',self.time_scheme)
        
        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)
        
        # He gaussian components  
        self.shapeHe = [State.ny,State.nx] 
        self.nHe = np.prod(self.shapeHe)
        self.He_gauss = 0
        ## In Space
        if config.D_He is not None:
            self.He_gauss = 1
            He_xy_gauss = []
            isub_He = int(config.D_He/State.dy)  
            jsub_He = int(config.D_He/State.dx)  
            for i in range(-2*isub_He,State.ny+3*isub_He,isub_He):
                y = i*State.dy
                for j in range(-2*jsub_He,State.nx+3*jsub_He,jsub_He):
                    x = j*State.dx
                    mat = np.ones((self.shapeHe))
                    for ii in range(State.ny):
                        for jj in range(State.nx):
                            dist = sqrt((State.Y[ii,jj]-y)**2+(State.X[ii,jj]-x)**2)
                            mat[ii,jj] = tools.gaspari_cohn(dist,7*config.D_He/2)
                    He_xy_gauss.append(mat)
            self.He_xy_gauss = np.asarray(He_xy_gauss)
            self.nHe = len(He_xy_gauss)        
            self.shapeHe = [self.nHe]
            ## In time 
            if config.T_He is not None:
                self.He_gauss = 2
                He_t_gauss = []
                ksub_He = int(config.T_He/self.dt)  
                for k in range(-2*ksub_He,self.nt+3*ksub_He,ksub_He):
                    He_t_gauss.append(tools.gaspari_cohn(self.T-k*self.dt,7*config.T_He/2))
                self.He_t_gauss = np.asarray(He_t_gauss)
                self.shapeHe = [len(self.He_t_gauss),self.nHe]
                self.nHe = np.prod(self.shapeHe)
                
            print('Gaussian He:',self.shapeHe)
        
        # Boundary conditions 
        self.omegas = np.asarray(config.w_igws)
        self.bc_kind = config.bc_kind
        # Angles for incoming wavenumbers
        if config.Ntheta>0:
            theta_p = np.arange(0,pi/2+pi/2/config.Ntheta,pi/2/config.Ntheta)
            self.bc_theta = np.append(theta_p-pi/2,theta_p[1:]) 
        else:
            self.bc_theta = np.array([0])
    
        # gaussian components
        self.shapehbcx = [self.omegas.size,2,2,self.bc_theta.size,State.nx]
        self.shapehbcy = [self.omegas.size,2,2,self.bc_theta.size,State.ny]
        self.bc_gauss = 0
    
        ## In Space
        if config.D_bc is not None:
            self.bc_gauss = 1
            bc_x_gauss = []
            bc_y_gauss = []
            isub_bc = int(config.D_bc//State.dy)  
            jsub_bc = int(config.D_bc//State.dx)  
            self.bcy = np.arange(-2*isub_bc*State.dy,
                                 (State.ny+3*isub_bc)*State.dy,
                                 isub_bc*State.dy)
            self.bcx = np.arange(-2*jsub_bc*State.dx,
                                 (State.nx+3*jsub_bc)*State.dx,
                                 jsub_bc*State.dx)
            
            for xj in self.bcx:
                bc_x_gauss.append(tools.gaspari_cohn(State.X[State.ny//2,:]-xj,
                                                     7*config.D_bc/2))
            for yi in self.bcy:
                bc_y_gauss.append(tools.gaspari_cohn(State.Y[:,State.nx//2]-yi,
                                                     7*config.D_bc/2))   
            self.bc_x_gauss = np.asarray(bc_x_gauss)
            self.bc_y_gauss = np.asarray(bc_y_gauss)
            self.shapehbcx[-1] = len(bc_x_gauss)
            self.shapehbcy[-1] = len(bc_y_gauss)
            ## In time 
            if config.T_bc is not None:
                self.bc_gauss = 2
                bc_t_gauss = []
                ksub_bc = int(config.T_bc//self.dt)  
                self.bct = np.arange(-2*ksub_bc*self.dt,(self.nt+3*ksub_bc)*self.dt,ksub_bc*self.dt)
                for kt in self.bct:
                    bc_t_gauss.append(tools.gaspari_cohn(self.T-kt,7*config.T_bc/2))
                self.bc_t_gauss = np.asarray(bc_t_gauss)
                self.shapehbcx.insert(-1,len(bc_t_gauss))
                self.shapehbcy.insert(-1,len(bc_t_gauss))         
        self.nbcx = np.prod(self.shapehbcx)
        self.nbcy = np.prod(self.shapehbcy)
        self.nbc = self.nbcx + self.nbcy
        
        print('BC:',self.bc_kind)
        print('Omegas:',self.omegas)
        print('Thetas:',self.bc_theta)
        print('Shape BC x:',self.shapehbcx)
        print('Shape BC y:',self.shapehbcy)
        
        # Slices for model parametersparameters
        self.nParams = self.nHe + self.nbc
        self.sliceHe = slice(0,self.nHe)
        self.slicehbcx = slice(self.nHe,self.nHe+self.nbcx)
        self.slicehbcy = slice(self.nHe+self.nbcx,self.nParams)
        
        # Model initialization
        self.swm = swm_adj.Swm_adj(X=State.X,
                                   Y=State.Y,
                                   dt=self.dt,
                                   bc=self.bc_kind,
                                   omegas=self.omegas,
                                   bc_theta=self.bc_theta,
                                   f=State.f)
        
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
        
        # Tests
        print('tangent test:')
       # self.tangent_test(State,self.T[-2],nstep=config.checkpoint)
        print('adjoint test:')
       # self.adjoint_test(State,self.T[-2],nstep=config.checkpoint)
       

            
    def restart(self):
        
        self.swm.restart()
    
    def reshapeParams(self,params):
        
        He = +params[self.sliceHe].reshape(self.shapeHe)
        hbcx = +params[self.slicehbcx].reshape(self.shapehbcx)
        hbcy = +params[self.slicehbcy].reshape(self.shapehbcy)
        
        return He,hbcx,hbcy
    
    def vectorizeParams(self,He,hbcx,hbcy):
        
        params = np.zeros(self.nParams)
        params[self.sliceHe] = He.ravel()
        params[self.slicehbcx] = hbcx.ravel()
        params[self.slicehbcy] = hbcy.ravel()
        
        return params
        
    def get_He2d(self,t=None,He=None,He_mean=None):
        
        if He_mean is None:
            He_mean = self.Heb
        
        if He is not None:
            if self.He_gauss==2:
                if len(np.shape(He))!=2:
                    raise SystemExit('Error: He need to be 2D in space-time \
                                     gaussian mode')
                He3d = np.tensordot(He,self.He_xy_gauss,(1,0))
                indt = int(t//self.dt)
                He2d = He_mean + \
                    np.tensordot(He3d,self.He_t_gauss[:,indt],(0,0))
                    
            elif self.He_gauss==1:
                if len(np.shape(He))!=1:
                    raise SystemExit('Error: He need to be 1D in space \
                                     gaussian mode')
                He2d = He_mean +\
                    np.sum(He[:,np.newaxis,np.newaxis]*self.He_xy_gauss,axis=0)
            else:
                He2d = He_mean + He
                
        else:
            He2d = He_mean 
            
        return He2d
    
    def get_hbc1d(self,t=None,hbcx=None,hbcy=None):
        
        # South/North
        if hbcx is not None:
            if self.bc_gauss==2:
                if len(np.shape(hbcx))!=6:
                    raise SystemExit('Error: hbcx need to be 6D in space-time \
                                     gaussian mode')
                hbcx_1d = np.tensordot(hbcx,self.bc_x_gauss,(-1,0))
                indt = int(t//self.dt)
                hbcx_1d = np.tensordot(hbcx_1d,self.bc_t_gauss[:,indt],(-2,0))
            
            elif self.bc_gauss==1:
                hbcx_1d = np.tensordot(hbcx,self.bc_x_gauss,(-1,0))
            else:
                hbcx_1d = hbcx
        else:
            hbcx_1d = np.zeros([self.omegas.size,2,2,self.bc_theta.size,self.shapeh[1]])
        
        # West/East
        if hbcy is not None:
            if self.bc_gauss==2:
                if len(np.shape(hbcy))!=6:
                    raise SystemExit('Error: hbcy need to be 6D in space-time \
                                     gaussian mode')
                hbcy_1d = np.tensordot(hbcy,self.bc_y_gauss,(-1,0))
                indt = int(t//self.dt)
                hbcy_1d = np.tensordot(hbcy_1d,self.bc_t_gauss[:,indt],(-2,0))
            
            elif self.bc_gauss==1:
                hbcy_1d = np.tensordot(hbcy,self.bc_y_gauss,(-1,0))
            else:
                hbcy_1d = hbcy
        else:
            hbcy_1d = np.zeros([self.omegas.size,2,2,self.bc_theta.size,self.shapeh[0]])
    
        return hbcx_1d,hbcy_1d
    
    def reduced_shape_He(self,t,adHe2d_incr):
        
        adHe_incr = np.zeros(self.shapeHe)
        
        if self.He_gauss==2:
            indt = int(t//self.dt)
            adHe3d = adHe2d_incr[:,:,np.newaxis]*self.He_t_gauss[:,indt]
            adHe_incr += np.tensordot(adHe3d,
                                      self.He_xy_gauss[:,:,:],([0,1],[1,2])) 
        elif self.He_gauss==1:
            for p in range(self.nHe):
                adHe_incr[p] += np.sum(
                    adHe2d_incr*self.He_xy_gauss[p,:,:])
        else:
            adHe_incr = adHe2d_incr
        
        return adHe_incr
    
    def reduced_shape_hbc(self,t,adhbcx1d_incr,adhbcy1d_incr):
        
        adhbcx_incr = np.zeros(self.shapehbcx)
        adhbcy_incr = np.zeros(self.shapehbcy)
        
        if self.bc_gauss==2:
           indt = int(t/self.dt)
           adhbcx2d_incr = adhbcx1d_incr[:,:,:,:,:,np.newaxis]*\
               self.bc_t_gauss[:,indt]
           adhbcx_incr += np.tensordot(adhbcx2d_incr,
                     self.bc_x_gauss,(-2,-1))
           
           adhbcy2d_incr = adhbcy1d_incr[:,:,:,:,:,np.newaxis]*\
               self.bc_t_gauss[:,indt]
           adhbcy_incr += np.tensordot(adhbcy2d_incr,
                     self.bc_y_gauss,(-2,-1))
           
        elif self.bc_gauss==1:
            for p in range(len(self.bc_x_gauss)):
                adhbcx_incr[:,:,:,:,p] +=  np.sum(
                    adhbcx1d_incr*self.bc_x_gauss[p],axis=-1)
            for p in range(len(self.bc_y_gauss)):
                adhbcy_incr[:,:,:,:,p] +=  np.sum(
                    adhbcy1d_incr*self.bc_y_gauss[p],axis=-1)
        else:
            adhbcx_incr += adhbcx1d_incr
            adhbcy_incr += adhbcy1d_incr
            
        return adhbcx_incr,adhbcy_incr
    
        
    def step(self,t,State,params,nstep=1,t0=0):
        
        # Get state variables and model parameters
        u0,v0,h0 = State.getvar()
        He,hbcx,hbcy = self.reshapeParams(params)

        # Model parameters: switch to physical space
        He2d = self.get_He2d(t,He,He_mean=self.Heb)
        tbc = t
        if self.bc_kind=='1d':
            tbc += self.dt       
        hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
        
        # init
        u = +u0
        v = +v0
        h = +h0
        
        # Time propagation
        for i in range(nstep):
            
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            #first = False
            
            u,v,h = self.swm_step(
                t+i*self.dt,
                u,v,h,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,first=first)
            
        # Update state
        State.setvar([u,v,h])
        
    
    def step_tgl(self,t,dState,State,dparams,params,nstep=1,t0=0):
        
        # Get state variables and model parameters
        du0,dv0,dh0 = dState.getvar()
        u0,v0,h0 = State.getvar()

        He,hbcx,hbcy = self.reshapeParams(params)
        dHe,dhbcx,dhbcy = self.reshapeParams(dparams)
        
        # Model parameters: switch to physical space
        He2d = self.get_He2d(t,He,He_mean=self.Heb)
        dHe2d = self.get_He2d(t,dHe,He_mean=0.)
        tbc = t
        if self.bc_kind=='1d' :
            tbc += self.dt
        hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
        dhbcx1d,dhbcy1d = self.get_hbc1d(tbc,dhbcx,dhbcy)
        
        # init
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
                        u,v,h,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,first=first)
                traj.append((u,v,h))
            
        for i in range(nstep):
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            u,v,h = traj[i]
            
            du,dv,dh = self.swm_step_tgl(
                t+i*self.dt,du,dv,dh,u,v,h,
                dHe=dHe2d,He=He2d,
                dhbcx=dhbcx1d,dhbcy=dhbcy1d,hbcx=hbcx1d,hbcy=hbcy1d,first=first)
      
        # Update state 
        dState.setvar([du,dv,dh])
        
    
    
    def step_adj(self,t,adState, State, adparams0, params, nstep=1, t0=0):
        
        # Get variables
        adu0,adv0,adh0 = adState.getvar()
        u0,v0,h0 = State.getvar()
        He,hbcx,hbcy = self.reshapeParams(params)
        adHe0,adhbcx0,adhbcy0 = self.reshapeParams(adparams0)
        
        # Model parameters: switch to physical space
        He2d = self.get_He2d(t,He)
        tbc = t
        if self.bc_kind=='1d':
            tbc += self.dt
        hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
        
        # Init
        adu = +adu0
        adv = +adv0
        adh = +adh0
        adHe2d = 0
        adhbcx1d = 0
        adhbcy1d = 0
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
                        u,v,h,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,first=first)
                traj.append((u,v,h))
            
        for i in reversed(range(nstep)):
            if t+i*self.dt==t0:
                    first = True
            else: first = False
            u,v,h = traj[i]
        
            adu,adv,adh,adHe2d_tmp,adhbcx1d_tmp,adhbcy1d_tmp =\
                self.swm_step_adj(t+i*self.dt,adu,adv,adh,u,v,h,
                                      He2d,hbcx1d,hbcy1d,first=first)
            adHe2d += adHe2d_tmp
            adhbcx1d += adhbcx1d_tmp
            adhbcy1d += adhbcy1d_tmp
            
        # Back to reduced form
        adHe = self.reduced_shape_He(t,adHe2d)
        adHe += adHe0 
        adhbcx,adhbcy = self.reduced_shape_hbc(tbc,adhbcx1d,adhbcy1d)
        adhbcx += adhbcx0 
        adhbcy += adhbcy0 
        
        # Update state  and parameters
        adState.setvar([adu,adv,adh])
        adparams = self.vectorizeParams(adHe,adhbcx,adhbcy)
        
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
    
    def tangent_test(self,State,tint,t0=0,nstep=1):
    
        State0 = State.random()
        dState = State.random()
        
        params = np.random.random(self.nParams)
        dparams = np.random.random(self.nParams)
        
        State0_tmp = State0.copy()
        self.run(t0,tint,State0_tmp,params,nstep=nstep)
        #self.step(t0,State0_tmp,params,nstep=nstep)
        X2 = State0_tmp.getvar(vect=True)

        for p in range(10):
            
            lambd = 10**(-p)
            
            State1 = dState.copy()
            State1.scalar(lambd)
            State1.Sum(State0)
            self.run(t0,tint,State1,params+lambd*dparams,nstep=nstep)
            #self.step(t0,State1,params+lambd*dparams,nstep=nstep)
            X1 = State1.getvar(vect=True)
            
            dState1 = dState.copy()
            dState1.scalar(lambd)
            self.run_tgl(t0,tint,dState1,State0,lambd*dparams,params,nstep=nstep)
            #self.step_tgl(t0,dState1,State0,lambd*dparams,params,nstep=nstep)
            dX = dState1.getvar(vect=True)
            
            ps = np.linalg.norm(X1-X2-dX)/np.linalg.norm(dX)

            print('%.E' % lambd,'%.E' % ps)
            
        
    def adjoint_test(self,State,tint,t0=0,nstep=1):
        
        # Current trajectory
        State0 = State.random()
        params = np.random.random(self.nParams)
        
        # Perturbation
        dState = State.random()
        dX = dState.getvar(vect=True)
        dparams = np.random.random(self.nParams)
        dX = np.concatenate((dX,dparams))
        
        # Adjoint
        adState = State.random()
        adX = adState.getvar(vect=True)
        adparams = np.random.random(self.nParams)
        adX = np.concatenate((adX,adparams))
        
        # Run TLM
        self.run_tgl(t0,tint,dState,State0,dparams,params,nstep=nstep)
        #self.step_tgl(t0,dState,State0,dparams,params,nstep=nstep)
        TLM = dState.getvar(vect=True)
        
        TLM = np.concatenate((TLM,dparams))
        
        # Run ADJ
        adparams = self.run_adj(
            t0,tint,adState,State0,adparams,params,nstep=nstep)
        #adparams = self.step_adj(
        #    t0,adState,State0,adparams,params,nstep=nstep)
        ADM = adState.getvar(vect=True)
        ADM = np.concatenate((ADM,adparams))
        
        ps1 = np.inner(TLM,adX)
        ps2 = np.inner(dX,ADM)
        
        print(ps1/ps2)

    
    
