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
from . import switchvar, tools, grid


def Model(config,State):
    """
    NAME
        main class

    DESCRIPTION
        Main function calling subclass for specific models
    """
    if config.name_model=='QG1L':
        return Model_qg1l(config,State)
    elif config.name_model=='SW1L':
        return Model_sw1l(config,State)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        
        
class Model_qg1l:
    
    def __init__(self,config,State):
        # Model specific libraries
        SourceFileLoader("modgrid", 
                         "../models/model_qg1l/modgrid.py").load_module() 
        SourceFileLoader("moddyn", 
                         "../models/model_qg1l/moddyn.py").load_module()     
        SourceFileLoader("modelliptic", 
                         "../models/model_qg1l/modelliptic.py").load_module()     
        self.qgsw = SourceFileLoader("qgsw", 
                                 "../models/model_qg1l/qgsw.py").load_module() 

        # Model parameters
        self.name_grd = config.name_grd
        self.dt = config.dtmodel
        self.g = 9.81
        self.c = config.c
        self.f = 4*np.pi/86164*np.sin(State.lat*np.pi/180)
        self.only_diffusion = config.only_diffusion
        self.cdiffus = config.cdiffus
        self.qgiter = config.qgiter
        
        State.ini_var_qg1l()
            
        
        
    def step(self,State,tint,Hbc=None,Wbc=None,Nudging_term=None):
        # Read state variable
        ssh_0 = State.getvar(0)
        if len(State.name_var)>1 and State.name_var[1] in State.var:
            flag_pv = True
            pv_0 = State.getvar(1)
        else:
            pv_0 = switchvar.ssh2pv(ssh_0,State.lon,State.lat,self.c,
                                    name_grd=self.name_grd)
        # Get model parameter from model state or use default one
        flag_K = False
        if len(State.name_var)>2 and State.name_var[2] in State.var:
            flag_K = True
            K = State.getvar(2)
            c = np.mean(State.f/np.sqrt(K))
        else:
            c = self.c
            K = (self.f/c)**2
        # Boundary condition
        if Wbc is None:
            Wbc = np.zeros((State.ny,State.nx))
        if Hbc is not None:
            Qbc = switchvar.ssh2pv(Hbc, State.lon, State.lat, self.c, 
                                   name_grd=self.name_grd)
            ssh_0 = Wbc*Hbc + (1-Wbc)*ssh_0
            pv_0 = Wbc*Qbc + (1-Wbc)*pv_0
            
        # Model propagation
        deltat = np.abs(tint)
        ssh_1, pv_1, trash = self.qgsw.qgsw(Hi=ssh_0, PVi=pv_0, c=c,
                                            lon=State.lon, lat=State.lat,
                                            tint=tint,
                                            dtout=deltat,
                                            dt=self.dt,
                                            name_grd=self.name_grd,
                                            diff=self.only_diffusion,
                                            snu=self.cdiffus)
        
        # Nudging
        if Nudging_term is not None:
            # Nudging towards relative vorticity
            if np.any(np.isfinite(Nudging_term['rv'])):
                indNoNan = ~np.isnan(Nudging_term['rv'])
                pv_1[-1][indNoNan] += (1-Wbc[indNoNan]) *\
                    Nudging_term['rv'][indNoNan]
            # Nudging towards ssh
            if np.any(np.isfinite(Nudging_term['ssh'])):
                indNoNan = ~np.isnan(Nudging_term['ssh'])
                pv_1[-1][indNoNan] -= (1-Wbc[indNoNan]) * (self.g/self.f[indNoNan]) *\
                    K[indNoNan] * Nudging_term['ssh'][indNoNan]
    
    
                # Inversion pv -> ssh
                ssh_b = ssh_1[-1].copy()
                ssh_1[-1] = switchvar.pv2ssh(
                    State.lon,State.lat,pv_1[-1],ssh_b,c,nitr=self.qgiter,
                    name_grd=self.name_grd)
        
        if np.any(np.isnan(ssh_1)):
            sys.exit('Invalid value encountered in mod_qg1l')
        
        # Update state 
        State.setvar(ssh_1[-1],0)
        if flag_pv:
            State.setvar(pv_1[-1],1)
        if flag_K:
            State.setvar(K,2)
            
        
    def step_adj():
        return
    
        
class Model_sw1l:
    def __init__(self,config,State):
        self.config = config
        # Model specific libraries
        SourceFileLoader("obcs", 
                                 "../models/model_sw1l/obcs.py").load_module() 
        SourceFileLoader("obcs_tgl", 
                                 "../models/model_sw1l/obcs_tgl.py").load_module() 
        SourceFileLoader("obcs_adj", 
                                 "../models/model_sw1l/obcs_adj.py").load_module() 
        SourceFileLoader("swm", 
                                 "../models/model_sw1l/swm.py").load_module() 
        SourceFileLoader("swm_tgl", 
                                 "../models/model_sw1l/swm_tgl.py").load_module() 
        
        swm_adj = SourceFileLoader("swm_adj", 
                                 "../models/model_sw1l/swm_adj.py").load_module() 
                
        # Compute cartesian grid 
        DX,DY = grid.lonlat2dxdy(State.lon,State.lat)
        dx = np.mean(DX)
        dy = np.mean(DY)
        X,Y = grid.dxdy2xy(DX,DY)
        
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
        cfl = min(dx, dy) / np.max(np.sqrt(State.g * self.Heb))
        if self.dt>=cfl:
            print('WARNING: timestep>=CFL')
        self.T = np.arange(self.nt) * self.dt
        self.time_scheme = config.sw_time_scheme
        
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
            isub_He = int(config.D_He/dy)  
            jsub_He = int(config.D_He/dx)  
            for i in range(-2*isub_He,State.ny+3*isub_He,isub_He):
                y = i*dy
                for j in range(-2*jsub_He,State.nx+3*jsub_He,jsub_He):
                    x = j*dx
                    mat = np.ones((self.shapeHe))
                    for ii in range(State.ny):
                        for jj in range(State.nx):
                            dist = sqrt((Y[ii,jj]-y)**2+(X[ii,jj]-x)**2)
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
            isub_bc = int(config.D_bc//dy)  
            jsub_bc = int(config.D_bc//dx)  
            self.bcy = np.arange(-2*isub_bc*dy,(State.ny+3*isub_bc)*dy,isub_bc*dy)
            self.bcx = np.arange(-2*jsub_bc*dx,(State.nx+3*isub_bc)*dx,jsub_bc*dx)
            
            for xj in self.bcx:
                bc_x_gauss.append(tools.gaspari_cohn(X[State.ny//2,:],7*config.D_bc/2))
            for yi in self.bcy:
                bc_y_gauss.append(tools.gaspari_cohn(Y[:,State.nx//2]-yi,7*config.D_bc/2))   
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
        self.swm = swm_adj.Swm_adj(X=X,
                                   Y=Y,
                                   dt=self.dt,
                                   bc=self.bc_kind,
                                   omegas=self.omegas,
                                   bc_theta=self.bc_theta,
                                   f=State.f)
        
        
        
        # Tests
        print('tangent test:')
        self.tangent_test(State,10*self.dt)
        print('adjoint test:')
        self.adjoint_test(State,10*self.dt)
        
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
            hbcx_1d = np.zeros([self.omegas.size,2,2,self.bc_theta.size,self.nx])
        
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
            hbcy_1d = np.zeros([self.omegas.size,2,2,self.bc_theta.size,self.ny])
        
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
    
        
    def step(self,t,State,params,step=0):
        
        # Get state variables and model parameters
        u0,v0,h0 = State.getvar()
        
        He,hbcx,hbcy = self.reshapeParams(params)

        # Model parameters: switch to physical space
        He2d = self.get_He2d(t,He,He_mean=self.Heb)
        tbc = t
        if self.bc_kind=='1d':
            tbc += self.dt       
        hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
        
        # Time propagation
        if self.time_scheme=='Euler':
            u,v,h = self.swm.step_euler(t,u0,v0,h0,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,step=step)
        
        if self.time_scheme=='lf':
            u,v,h = self.swm.step_lf(t,u0,v0,h0,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,step=step)
        
        # Update state
        State.setvar([u,v,h])
        
        

        
    
    def step_tgl(self,t,dState,State,dparams,params,step=0):
        
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
        
        # Time propagation
        if self.time_scheme=='Euler':
            du,dv,dh = self.swm.step_euler_tgl(
                t,du0,dv0,dh0,u0,v0,h0,
                dHe=dHe2d,He=He2d,
                dhbcx=dhbcx1d,dhbcy=dhbcy1d,hbcx=hbcx1d,hbcy=hbcy1d,step=step)
    
        if self.time_scheme=='lf':
            du,dv,dh = self.swm.step_lf_tgl(
                t,du0,dv0,dh0,u0,v0,h0,
                dHe=dHe2d,He=He2d,
                dhbcx=dhbcx1d,dhbcy=dhbcy1d,hbcx=hbcx1d,hbcy=hbcy1d,step=step)
        
        # Update state 
        dState.setvar([du,dv,dh])
        
    
    
    def step_adj(self,t,adState, State, adparams0, params, step=0):
        
        # Get state variables
        u0,v0,h0 = State.getvar()
        adu0,adv0,adh0 = adState.getvar()
        He,hbcx,hbcy = self.reshapeParams(params)
        adHe0,adhbcx0,adhbcy0 = self.reshapeParams(adparams0)
        
        # Model parameters: switch to physical space
        He2d = self.get_He2d(t,He)
        tbc = t
        if self.bc_kind=='1d':
            tbc += self.dt
        hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
        
        # Time propagation
        if self.time_scheme=='Euler':
            adu,adv,adh,adHe2d,adhbcx1d,adhbcy1d =\
                self.swm.step_adj_euler(t,adu0,adv0,adh0,u0,v0,h0,
                                      He2d,hbcx1d,hbcy1d,step)
                
        if self.time_scheme=='lf':
            adu,adv,adh,adHe2d,adhbcx1d,adhbcy1d =\
                self.swm.step_adj_lf(t,adu0,adv0,adh0,u0,v0,h0,
                                      He2d,hbcx1d,hbcy1d,step)
                
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
        
        
    def run(self,t0,tint,State,params,return_traj=False):
        if return_traj:
            traj = [State.getvar()]
        t,step = t0,0
        while t <= t0+tint: 
            self.step(t,State,params,step=step)
            if return_traj:
                traj.append(State.getvar())
            step +=1
            t += self.dt
        if return_traj:
            return traj

    def run_tgl(self,t0,tint,dState,State,dparams,params):
        
        State_tmp = State.copy()
        traj = self.run(t0,tint,State_tmp,params,return_traj=True)
        t,step = t0,0
        while t <= t0+tint: 
            State_tmp.setvar(traj[step])
            self.step_tgl(t,dState,State_tmp,dparams,params,step)
            step +=1
            t += self.dt

    
    def run_adj(self,t0,tint,adState,State,adparams,params):

        State_tmp = State.copy()
        traj = self.run(t0,tint,State_tmp,params,return_traj=True)
        t,step = t0+tint, len(traj)-2
        while t >= t0: 
            State_tmp.setvar(traj[step])
            adparams = self.step_adj(
                t,adState,State_tmp,adparams,params,step=step)
            step -=1
            t -= self.dt
            
        return adparams
    
    # def tangent_test(self,State):
    
    #     t0 = 0
    #     tint = 12*3600
    #     State0 = State.random()
    #     dState = State.random()
    
    #     params = np.random.random(self.nParams)
    #     dparams = np.random.random(self.nParams)
        
    #     State0_tmp = State0.copy()
    #     self.run(t0,tint,State0_tmp,params)
    #     X2 = State0_tmp.getvar(vect=True)
        
    #     for p in range(10):
            
    #         lambd = 10**(-p)
            
    #         State1 = State0.free()
    #         State1.setvar(lambd*dState.var.values+State0.var.values)
    #         self.run(t0,tint,State1,params+lambd*dparams)
    #         X1 = State1.getvar(vect=True)
            
    #         dState1 = State0.free()
    #         dState1.setvar(lambd*dState.var.values)
    #         self.run_tgl(t0,tint,dState1,State0,lambd*dparams,params)
    #         dX = dState1.getvar(vect=True)
            
    #         ps = np.linalg.norm(X1-X2-dX)/np.linalg.norm(dX)
            
    #         print('%.E' % lambd,'%.E' % ps)
    
    # def tangent_test(self,State):
    #     t0 = 0
        
    #     State0 = State.random()
    #     u,v,h = State0.getvar()
    #     _u0 = deepcopy(u)
        
    #     params = np.random.random(self.nParams)
        
    #     dState = State.random()
    #     du,dv,dh = dState.getvar()
    #     dparams = np.random.random(self.nParams)
        
    #     He,hbcx,hbcy = self.reshapeParams(params)
    #     dHe,dhbcx,dhbcy = self.reshapeParams(dparams)
        
    #     # Model parameters: switch to physical space
    #     He2d = self.get_He2d(t0,He,He_mean=self.Heb)
    #     dHe2d = self.get_He2d(t0,He,He_mean=0)
    #     tbc = t0
    #     if self.bc_kind=='1d':
    #         tbc += self.dt       
    #     hbcx1d,hbcy1d = self.get_hbc1d(tbc,hbcx,hbcy)
    #     dhbcx1d,dhbcy1d = self.get_hbc1d(tbc,dhbcx,dhbcy)
            
        
    #     #u2,v2,h2 = self.swm.step_lf(t0,u,v,h,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,step=0)
    #     State0_tmp = self.step(t0,State0,params)
    #     u2,v2,h2 = State0_tmp.getvar()
    #     print(np.array_equal(u,_u0))
    #     #u0,v0,h0 = State0.getvar()
        
    #     #u2,v2,h2 = self.swm.step_lf(t0,deepcopy(u),deepcopy(v),deepcopy(h),He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,step=0)
    #     _u2,_v2,_h2 = self.swm.step_lf(t0,u,v,h,He=He2d,hbcx=hbcx1d,hbcy=hbcy1d,step=0)
    #     print(np.array_equal(u,_u0))

        
    #     for p in range(10):
    #         lambd = 10**(-p)
        
    #         u1,v1,h1 = self.swm.step_lf(t0,u+lambd*du,v+lambd*dv,h+lambd*dh,
    #                               He=He2d+lambd*dHe2d,
    #                               hbcx=hbcx1d+lambd*dhbcx1d,hbcy=hbcy1d+lambd*dhbcy1d,step=0)
            
            
 
            
    #         du1,dv1,dh1 = self.swm.step_lf_tgl(t0,lambd*du,lambd*dv,lambd*dh,u,v,h,
    #                                     dHe=lambd*dHe2d,He=He2d,
    #                                     dhbcx=lambd*dhbcx1d,hbcx=hbcx1d,dhbcy=lambd*dhbcy1d,hbcy=hbcy1d,step=0)

            
            
    #         psu = np.linalg.norm((u1-u2)-du1)/np.linalg.norm(du1)
    #         psv = np.linalg.norm((v1-v2)-dv1)/np.linalg.norm(dv1)
    #         psh = np.linalg.norm((h1-h2)-dh1)/np.linalg.norm(dh1)
            
    #         print('%.E' % lambd,'%.E' % psu,'%.E' % psv,'%.E' % psh)
        
    # def tangent_test(self,State):
    
    #     t0 = 0
    #     tint = 12*3600
    #     State0 = State.random()
    #     dState = State.random()
        
    #     params = np.random.random(self.nParams)
    #     dparams = np.random.random(self.nParams)
        
    #     State0_tmp = self.step(t0,State0,params)
    #     u2,v2,h2 = State0_tmp.getvar()

    #     for p in [5,6]:#range(10):
            
    #         lambd = 10**(-p)
            
    #         State1 = dState.copy()
    #         #State1.setvar(lambd*dState.var.values+State0.var.values)
    #         State1.scalar(lambd)
    #         State1.Sum(State0)
    #         State1_tmp = self.step(t0,State1,params+lambd*dparams)
            
    #         u1,v1,h1 = State1_tmp.getvar()
            
    #         dState1 = dState.copy()
    #         dState1.scalar(lambd)
    #         dState1_tmp = self.step_tgl(t0,dState1,State0,lambd*dparams,params)
    #         du1,dv1,dh1 = dState1_tmp.getvar()
            
            
            
            
    #         psu = np.linalg.norm((u1-u2)-du1)/np.linalg.norm(du1)
    #         psv = np.linalg.norm((v1-v2)-dv1)/np.linalg.norm(dv1)
    #         psh = np.linalg.norm((h1-h2)-dh1)/np.linalg.norm(dh1)
            
    #         print('%.E' % lambd,'%.E' % psu,'%.E' % psv,'%.E' % psh)

    def tangent_test(self,State,tint,t0=0):
    
        State0 = State.random()
        dState = State.random()
        
        params = np.random.random(self.nParams)
        dparams = np.random.random(self.nParams)
        
        State0_tmp = State0.copy()
        self.run(t0,tint,State0_tmp,params)
        X2 = State0_tmp.getvar(vect=True)

        for p in range(10):
            
            lambd = 10**(-p)
            
            State1 = dState.copy()
            State1.scalar(lambd)
            State1.Sum(State0)
            self.run(t0,tint,State1,params+lambd*dparams)
            X1 = State1.getvar(vect=True)
            
            dState1 = dState.copy()
            dState1.scalar(lambd)
            self.run_tgl(t0,tint,dState1,State0,lambd*dparams,params)
            dX = dState1.getvar(vect=True)
            
            ps = np.linalg.norm(X1-X2-dX)/np.linalg.norm(dX)

            print('%.E' % lambd,'%.E' % ps)
            
        
    def adjoint_test(self,State,tint,t0=0):
        
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
        self.run_tgl(t0,tint,dState,State0,dparams,params)
        TLM = dState.getvar(vect=True)
        
        TLM = np.concatenate((TLM,dparams))
        
        # Run ADJ
        adparams = self.run_adj(
            t0,tint,adState,State0,adparams,params)
        ADM = adState.getvar(vect=True)
        ADM = np.concatenate((ADM,adparams))
        
        ps1 = np.inner(TLM,adX)
        ps2 = np.inner(dX,ADM)
        
        print(ps1/ps2)

    
    