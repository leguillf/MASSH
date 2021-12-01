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
from copy import deepcopy
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
    elif config.name_model=='SW1LM':
        return Model_sw1lm(config,State)
    elif config.name_model=='QG1L_SW1L':
        return Model_qg1l_sw1l(config,State)
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

        # Time parameters
        self.dt = config.dtmodel
        self.nt = 1 + int((config.final_date - config.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        
        # Open MDT map if provided
        if config.path_mdt is not None and os.path.exists(config.path_mdt):
            print('MDT is prescribed, thus the QGPV will be expressed thanks \
to Reynolds decomposition. However, be sure that observed and boundary \
variable are SLAs!')
                      
            ds = xr.open_dataset(config.path_mdt).squeeze()
            
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
        if config.c0 is None and config.filec_aux is not None and os.path.exists(config.filec_aux):
            
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
        
        if config.name_analysis=='4Dvar':
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
                         c=config.c0,
                         g=State.g,
                         f=State.f,
                         qgiter=config.qgiter,
                         diff=config.only_diffusion,
                         snu=config.cdiffus,
                         mdt=self.mdt,
                         mdv=self.mdv,
                         mdu=self.mdu)
        self.State = State
        

        # Construct timestamps
        self.timestamps = [] 
        t = config.init_date
        while t<=config.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)

        if config.name_analysis=='4Dvar' and config.compute_test and config.name_model=='QG1L':
            print('Tangent test:')
            self.tangent_test(State,10,config.flag_use_boundary_conditions)

            print('Adjoint test:')
            self.adjoint_test(State,10,config.flag_use_boundary_conditions)

    def step(self,State,nstep=1,Hbc=None,Wbc=None,ind=0):
        
        # Get state variable
        SSH0 = State.getvar(ind=ind)
        
        # init
        SSH1 = +SSH0
        
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            SSH1 = Wbc*Hbc + (1-Wbc)*SSH1
        
        # Time propagation
        for i in range(nstep):
            SSH1 = self.qgm.step(SSH1,way=1)

        # Update state
        State.setvar(SSH1,ind=ind)
    
            
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
        
    def step_tgl(self,dState,State,nstep=1,Hbc=None,Wbc=None,ind=0):
        
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
        dState.setvar(dSSH1,ind=ind)
        
    def step_adj(self,adState,State,nstep=1,Hbc=None,Wbc=None,ind=0):
        
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
        adState.setvar(adSSH1,ind=ind)
        
    def tangent_test(self,State,nstep,bc=False):
    
        State0 = State.random(1e-2)
        dState = State.random(1e-2)
        
        if bc:
            Hbc = np.random.random((State.ny,State.nx))
            Wbc = np.random.random((State.ny,State.nx))
        else:
            Hbc = Wbc = None
        
        State0_tmp = State0.copy()
        self.step(State0_tmp,nstep=nstep,Hbc=Hbc,Wbc=Wbc)
        X2 = State0_tmp.getvar(vect=True)

        for p in range(10):
            
            lambd = 10**(-p)
            
            State1 = dState.copy()
            State1.scalar(lambd)
            State1.Sum(State0)
            self.step(State1,nstep=nstep,Hbc=Hbc,Wbc=Wbc)
            X1 = State1.getvar(vect=True)
            
            dState1 = dState.copy()
            dState1.scalar(lambd)
            self.step_tgl(dState1,State0,nstep=nstep,Hbc=Hbc,Wbc=Wbc)
            dX = dState1.getvar(vect=True)
            
            mask = np.isnan(X1+X2+dX)
            ps = np.linalg.norm(X1[~mask]-X2[~mask]-dX[~mask])/np.linalg.norm(dX[~mask])

            print('%.E' % lambd,'%.E' % ps)
         
            
    def adjoint_test(self,State,nstep,bc=False):
        
        # Current trajectory
        State0 = State.random(1e-2)
        
        # Perturbation
        dState = State.random()
        dX = dState.getvar(vect=True)
        
        # Adjoint
        adState = State.random()
        adY = adState.getvar(vect=True)
        
        if bc:
            Hbc = np.random.random((State.ny,State.nx))
            Wbc = np.random.random((State.ny,State.nx))
        else:
            Hbc = Wbc = None
        
        # Run TLM
        self.step_tgl(dState,State0,nstep=nstep,Hbc=Hbc,Wbc=Wbc)
        dY = dState.getvar(vect=True)
        
        # Run ADJ
        self.step_adj(adState,State0,nstep=nstep,Hbc=Hbc,Wbc=Wbc)
        adX = adState.getvar(vect=True)
           
        mask = np.isnan(dX + adX + dY + adY)
        ps1 = np.inner(dX[~mask],adX[~mask])
        ps2 = np.inner(dY[~mask],adY[~mask])
        
        print(ps1/ps2)

        
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
                
        
        # State variable dimensions
        self.shapeu = State.var[0].shape
        self.shapev = State.var[1].shape
        self.shapeh = State.var[2].shape 
        self.nu = np.prod(self.shapeu)
        self.nv = np.prod(self.shapev)
        self.nh = np.prod(self.shapeh)

        
            
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
            
        #########################
        # He 
        #########################
        
        # Get background value for He
        if config.He_data is not None and os.path.exists(config.He_data['path']):
            ds = xr.open_dataset(config.He_data['path'])
            self.Heb = ds[config.He_data['var']].values
        else:
            if He_init is None:
                self.Heb = config.He_init
            else:
                self.Heb = He_init
            print('Heb:',self.Heb)
            
        # Gaussian components
        self.shapeHe = [State.ny,State.nx] 
        self.nHe = np.prod(self.shapeHe)
        self.He_gauss = 0
        if D_He is None:
            D_He = config.D_He
        if T_He is None:
            T_He = config.T_He
        ## In Space
        if D_He is not None:
            self.He_gauss = 1
            He_xy_gauss = []
            isub_He = int(D_He/State.dy)  
            jsub_He = int(D_He/State.dx)  
            for i in range(-2*isub_He,State.ny+3*isub_He,isub_He):
                y = i*State.dy
                for j in range(-2*jsub_He,State.nx+3*jsub_He,jsub_He):
                    x = j*State.dx
                    mat = np.ones((self.shapeHe))
                    for ii in range(State.ny):
                        for jj in range(State.nx):
                            dist = sqrt((State.Y[ii,jj]-y)**2+(State.X[ii,jj]-x)**2)
                            mat[ii,jj] = tools.gaspari_cohn(dist,7*D_He/2)
                    He_xy_gauss.append(mat)
            self.He_xy_gauss = np.asarray(He_xy_gauss)
            self.nHe = len(He_xy_gauss)        
            self.shapeHe = [self.nHe]
            ## In time 
            if T_He is not None:
                self.He_gauss = 2
                He_t_gauss = []
                ksub_He = int(T_He/self.dt)  
                for k in range(-2*ksub_He,self.nt+3*ksub_He,ksub_He):
                    He_t_gauss.append(tools.gaspari_cohn(self.T-k*self.dt,7*T_He/2))
                self.He_t_gauss = np.asarray(He_t_gauss)
                self.shapeHe = [len(self.He_t_gauss),self.nHe]
                self.nHe = np.prod(self.shapeHe)
                
            print('Gaussian He:',self.shapeHe)
        
        
        #########################
        # Boundary conditions 
        #########################
        
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
        if D_bc is None:
            D_bc = config.D_bc
        if T_bc is None:
            T_bc = config.T_bc
        ## In Space
        if D_bc is not None:
            self.bc_gauss = 1
            bc_x_gauss = []
            bc_y_gauss = []
            isub_bc = int(D_bc//State.dy)  
            jsub_bc = int(D_bc//State.dx)  
            self.bcy = np.arange(-2*isub_bc*State.dy,
                                 (State.ny+3*isub_bc)*State.dy,
                                 isub_bc*State.dy)
            self.bcx = np.arange(-2*jsub_bc*State.dx,
                                 (State.nx+3*jsub_bc)*State.dx,
                                 jsub_bc*State.dx)
            
            for xj in self.bcx:
                bc_x_gauss.append(tools.gaspari_cohn(State.X[State.ny//2,:]-xj,
                                                     7*D_bc/2))
            for yi in self.bcy:
                bc_y_gauss.append(tools.gaspari_cohn(State.Y[:,State.nx//2]-yi,
                                                     7*D_bc/2))   
            self.bc_x_gauss = np.asarray(bc_x_gauss)
            self.bc_y_gauss = np.asarray(bc_y_gauss)
            self.shapehbcx[-1] = len(bc_x_gauss)
            self.shapehbcy[-1] = len(bc_y_gauss)
            ## In time 
            if T_bc is not None:
                self.bc_gauss = 2
                bc_t_gauss = []
                ksub_bc = int(T_bc//self.dt)  
                self.bct = np.arange(-2*ksub_bc*self.dt,(self.nt+3*ksub_bc)*self.dt,ksub_bc*self.dt)
                for kt in self.bct:
                    bc_t_gauss.append(tools.gaspari_cohn(self.T-kt,7*T_bc/2))
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
        if config.name_analysis=='4Dvar' and config.compute_test and config.name_model=='SW1L':
            print('tangent test:')
            self.tangent_test(State,self.T[-2],nstep=config.checkpoint)
            print('adjoint test:')
            self.adjoint_test(State,self.T[-2],nstep=config.checkpoint)
       

            
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
    
        
    def step(self,t,State,params,nstep=1,t0=0,ind=[0,1,2]):
        
        # Get state variables and model parameters
        u0,v0,h0 = State.getvar(ind=ind)
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
        State.setvar([u,v,h],ind=ind)
        
    
    def step_tgl(self,t,dState,State,dparams,params,nstep=1,t0=0,ind=[0,1,2]):
        
        # Get state variables and model parameters
        du0,dv0,dh0 = dState.getvar(ind=ind)
        u0,v0,h0 = State.getvar(ind=ind)

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
        dState.setvar([du,dv,dh],ind=ind)
        
    
    
    def step_adj(self,t,adState, State, adparams0, params, nstep=1, t0=0,ind=None):
        
        # Get variables
        adu0,adv0,adh0 = adState.getvar(ind=ind)
        u0,v0,h0 = State.getvar(ind=ind)
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
        
        # Update parameters
        adparams = self.vectorizeParams(adHe,adhbcx,adhbcy)
        
        # Update state
        adState.setvar([adu,adv,adh],ind=ind)
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
            
            if i==0:
                self.nt = M.nt
                self.dt = M.dt
                self.T = M.T.copy()
                self.timestamps = M.timestamps.copy()
            print()
        
        # Tests
        if config.compute_test:
            print('tangent test:')
            self.tangent_test(State,self.T[10],nstep=config.checkpoint)
            print('adjoint test:')
            self.adjoint_test(State,self.T[10],nstep=config.checkpoint)
        
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
    
    def step(self,t,State,params,nstep=1,t0=0):
        u = 0
        v = 0
        h = 0
        for i in range(self.Nmodes):
            ind = np.arange(3*i,3*(i+1))
            _params = params[self.slice_param(i)]
            self.Models[i].step(t,State,_params,nstep,t0,ind=ind)
            _u,_v,_h = State.getvar(ind=ind)
            u += _u
            v += _v 
            h += _h
        ind = np.arange(3*self.Nmodes,3*(self.Nmodes+1))
        State.setvar([u,v,h],ind=ind)
        
    def step_tgl(self,t,dState,State,dparams,params,nstep=1,t0=0):
        du = 0
        dv = 0
        dh = 0
        for i in range(self.Nmodes):
            ind = np.arange(3*i,3*(i+1))
            _params = params[self.slice_param(i)]
            _dparams = dparams[self.slice_param(i)]
            
            self.Models[i].step_tgl(
                t,dState,State,_dparams,_params,nstep,t0,ind=ind) 
            
            _du,_dv,_dh = dState.getvar(ind=ind)
            du += _du
            dv += _dv 
            dh += _dh
            
        ind = np.arange(3*self.Nmodes,3*(self.Nmodes+1))
        dState.setvar([du,dv,dh],ind=ind) 
    
    def step_adj(self,t,adState, State, adparams0, params, nstep=1, t0=0):

        adparams = +adparams0*0
        
        ind = np.arange(3*self.Nmodes,3*(self.Nmodes+1))
        adu,adv,adh = adState.getvar(ind=ind) 
        
        for i in range(self.Nmodes):
            ind = np.arange(3*i,3*(i+1))
            
            _adu,_adv,_adh = adState.getvar(ind=ind) 
            adState.setvar([_adu+adu,_adv+adv,_adh+adh],ind=ind) 
            
            _params = params[self.slice_param(i)]
            _adparams0 = adparams0[self.slice_param(i)]
            
            _adparams = self.Models[i].step_adj(
                t,adState,State,_adparams0,_params,nstep,t0,ind=ind)
            
            
            
            adparams[self.slice_param(i)] = _adparams
            
        ind = np.arange(3*self.Nmodes,3*(self.Nmodes+1))
        adu,adv,adh = adState.getvar(ind=ind)
        adState.setvar([0*adu,0*adv,0*adh],ind=ind) 
        
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
        #self.run_tgl(t0,tint,dState,State0,dparams,params,nstep=nstep)
        self.step_tgl(t0,dState,State0,dparams,params,nstep=nstep)
        TLM = dState.getvar(vect=True)
        
        TLM = np.concatenate((TLM,dparams))
        
        # Run ADJ
        #adparams = self.run_adj(
        #    t0,tint,adState,State0,adparams,params,nstep=nstep)
        adparams = self.step_adj(
            t0,adState,State0,adparams,params,nstep=nstep)
        ADM = adState.getvar(vect=True)
        ADM = np.concatenate((ADM,adparams))
        
        ps1 = np.inner(TLM,adX)
        ps2 = np.inner(dX,ADM)
        
        print(ps1/ps2)
        
            
class Model_qg1l_sw1l:
    
    def __init__(self,config,State):
        print('\n* QG Model')
        self.Model_QG = Model_qg1l(config,State)
        print('\n* SW Model')
        self.Model_SW = Model_sw1l(config,State)
        
        self.nt = self.Model_SW.nt
        self.dt = self.Model_SW.dt
        self.T = self.Model_SW.T.copy()
        self.timestamps = self.Model_SW.timestamps.copy()
        self.nParams = self.Model_SW.nParams
        self.sliceHe = self.Model_SW.sliceHe
        self.slicehbcx = self.Model_SW.slicehbcx
        self.slicehbcy = self.Model_SW.slicehbcy
        self.mdt = self.Model_QG.mdt
        
        if config.compute_test:
            print('tangent test:')
            self.tangent_test(State,self.T[10],nstep=config.checkpoint)
            print('adjoint test:')
            self.adjoint_test(State,self.T[10],nstep=config.checkpoint)
        
        
    def step(self,t,State,params,nstep=1,t0=0):
        
        h = 0
        
        self.Model_QG.step(State,nstep=nstep,ind=0)
        h += State.getvar(ind=0)
        
        self.Model_SW.step(t,State,params,nstep=nstep,t0=t0,ind=[1,2,3])
        h += State.getvar(ind=3)
        
        State.setvar(h,ind=4)
        
        
    def step_tgl(self,t,dState,State,dparams,params,nstep=1,t0=0):
        
        dh = 0
        
        self.Model_QG.step_tgl(dState,State,nstep=nstep,ind=0)
        dh += dState.getvar(ind=0)
        
        self.Model_SW.step_tgl(t,dState,State,dparams,params,nstep=nstep,t0=t0,ind=[1,2,3])
        dh += dState.getvar(ind=3)

        dState.setvar(dh,ind=4)
    
    def step_adj(self,t,adState,State,adparams,params,nstep=1,t0=0):
        
        adh = adState.getvar(ind=4) 
        
        _adh = adState.getvar(ind=0) 
        adState.setvar(adh+_adh,ind=0)
        self.Model_QG.step_adj(adState,State,nstep=nstep,ind=0)
    
        _adh = adState.getvar(ind=3) 
        adState.setvar(adh+_adh,ind=3)
        adparams = self.Model_SW.step_adj(t,adState,State,adparams,params,nstep=nstep,t0=t0,ind=[1,2,3])
        adh += adState.getvar(ind=3)

        adState.setvar(0*adh,ind=4)
        
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
        #self.run_tgl(t0,tint,dState,State0,dparams,params,nstep=nstep)
        self.step_tgl(t0,dState,State0,dparams,params,nstep=nstep)
        TLM = dState.getvar(vect=True)
        
        TLM = np.concatenate((TLM,dparams))
        
        # Run ADJ
        #adparams = self.run_adj(
        #    t0,tint,adState,State0,adparams,params,nstep=nstep)
        adparams = self.step_adj(
            t0,adState,State0,adparams,params,nstep=nstep)
        ADM = adState.getvar(vect=True)
        ADM = np.concatenate((ADM,adparams))
        
        ps1 = np.inner(TLM,adX)
        ps2 = np.inner(dX,ADM)
        
        print(ps1/ps2)
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
