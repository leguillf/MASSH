#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""

import time


import os
import xarray as xr 
import numpy as np 
import pandas as pd 
from datetime import timedelta
from src import grid as grid
import datetime
import jax.numpy as jnp 
import jax.lax as lax
import jax
from jax.lax import scan
jax.config.update("jax_enable_x64", True)


class Cov :
    # case of a simple diagonal covariance matrix
    def __init__(self,sigma=None):
        
        if sigma is None:
            sigma = 1
            
        self.sigma = sigma
        
    def inv(self,X):
        return 1/self.sigma**2 * X    
    
    def sqr(self,X):
        return self.sigma * X
    
    def invsqr(self,X):
        return 1/self.sigma * X
    
    
class Variational:
    
    def __init__(self, 
                 config=None, M=None, H=None, State=None, R=None,B=None, Basis=None, Xb=None, checkpoints=None):
        
        # Objects
        self.M = M # model
        self.H = H # observational operator
        self.State = State # state variables
    
        # Covariance matrixes
        self.B = B
        self.R = R
        
        # Background state
        self.Xb = Xb
        
        # Temporary path where to save model trajectories
        self.tmp_DA_path = config.EXP.tmp_DA_path

        # checkpoint 
        self.checkpoints = checkpoints
        
        # preconditioning
        self.prec = config.INV.prec
        
        # Wavelet reduced basis
        self.dtbasis = int(config.INV.timestep_checkpoint.total_seconds()//M.dt)
        self.basis = Basis 
        
        # Save cost function and its gradient at each iteration 
        self.save_minimization = config.INV.save_minimization
        if self.save_minimization:
            self.J = []
            self.dJ = [] # For incremental 4Dvar only
            self.G = []
        
        # For incremental 4Dvar only
        self.X0 = self.Xb*0
        
        # Grad test
        if config.INV.compute_test:
            print('Gradient test:')
            if self.prec:
                X = (np.random.random(self.basis.nbasis)-0.5)
            else:
                X = self.B.sqr(np.random.random(self.basis.nbasis)-0.5) + self.Xb
            grad_test(self.cost,self.grad,X)

        
    def cost(self,X0):
                
        # Initial state
        State = self.State.copy()
        State.plot(title='State variables at the start of cost function evaluation')
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
                Jb = np.dot(X0,self.B.inv(X0)) # cost of background term
        else:
            X  = X0 - self.Xb
            Jb = 0
    
        # Observational cost function evaluation
        Jo = 0.
        for i in range(len(self.checkpoints)-1):
            
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            
            # 1. Misfit
            if self.H.is_obs(timestamp):
                misfit = self.H.misfit(timestamp,State) # d=Hx-xobs   
                Jo += misfit.dot(self.R.inv(misfit))
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                self.basis.operg(t/3600/24, X, State=State)
            
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoints[i]) + '.nc'))

            # 3. Run forward model
            self.M.step(t=t,State=State,nstep=nstep)

        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if self.H.is_obs(timestamp):
            misfit = self.H.misfit(timestamp,State) # d=Hx-xobsx
            Jo += misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
         
        State.plot(title='State variables at the end of cost function evaluation')
        State.plot(title='Parameters at the end of cost function evaluation',params=True)
        
        if self.save_minimization:
            self.J.append(J)

        return J
    
    def grad(self,X0): 
                
        X = +X0 
        
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                gb = X0      # gradient of background term
            else:
                X  = X0 + self.Xb
                gb = self.B.inv(X0) # gradient of background term
        else:
            X  = X0 + self.Xb
            gb = 0
            
        # Current trajectory
        State = self.State.copy()
        
        # Ajoint initialization   
        adState = self.State.copy(free=True)
        adX = X*0

        # Last timestamp
        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if self.H.is_obs(timestamp):
            self.H.adj(timestamp,adState,self.R)

        # Time loop
        for i in reversed(range(0,len(self.checkpoints)-1)):
            
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.checkpoints[i]) + '.nc'))
            
            # 3. Run adjoint model 
            self.M.step_adj(t=t, adState=adState, State=State, nstep=nstep) # i+1 --> i
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                adX += self.basis.operg_transpose(t=t/3600/24,adState=adState)
            
            # 1. Misfit 
            if self.H.is_obs(timestamp):
                self.H.adj(timestamp,adState,self.R)

        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient

        adState.plot(title='adjoint variables at the end of gradient function evaluation')
        State.plot(title='adjoint parameters at the end of gradient function evaluation',params=True)
        
        if self.save_minimization:
            self.G.append(np.max(np.abs(g)))

        return g     
    
class Variational_jax:
    
    def __init__(self, 
                 config=None, M=None, H=None, State=None, R=None,B=None, Basis=None, Xb=None, checkpoints=None, nstep=None):
        
        # Objects
        self.M = M # model
        self.H = H # observational operator
        self.State = State # state variables
    
        # Covariance matrixes
        self.B = B
        self.R = R
        
        # Background state
        self.Xb = jnp.array(Xb)
        
        # Temporary path where to save model trajectories
        self.tmp_DA_path = config.EXP.tmp_DA_path

        # checkpoint 
        self.checkpoints = jnp.asarray(checkpoints)

        # Timetamps
        self.T = jnp.asarray(M.T)

        self.nstep = nstep
                                    
        # preconditioning
        self.prec = config.INV.prec
        
        # Wavelet reduced basis
        self.dtbasis = int(config.INV.timestep_checkpoint.total_seconds()//M.dt)
        self.basis = Basis 
        
        # Save cost function and its gradient at each iteration 
        self.save_minimization = config.INV.save_minimization
        if self.save_minimization:
            self.J = []
            self.dJ = [] # For incremental 4Dvar only
            self.G = []
        
        # For incremental 4Dvar only
        self.X0 = self.Xb*0
        
        # Grad test
        if config.INV.compute_test:
            print('Gradient test:')
            if self.prec:
                X = jnp.array(np.random.random(self.basis.nbasis)-0.5)
            else:
                X = jnp.array(self.B.sqr(np.random.random(self.basis.nbasis)-0.5) + self.Xb)
            grad_test(self.cost,self.grad,X)
            
    
    def misfit_update(self, args):
        Jo, State_var, t = args
        misfit = self.H.misfit(State_var, t)  # d=Hx-xobs
        return Jo + misfit.dot(self.R.inv(misfit)), State_var, t
    
    def no_update(self,args):
        return args
    
    def body_fn(self, i, val):
        t, Jo, State_var, State_params, X = val

        # 1. Misfit
        Jo, State_var, t = lax.cond(
            self.H.is_obs_time(t),
            self.misfit_update,
            self.no_update,
            (Jo, State_var, t)
        )

        # 2. Basis
        self.basis.operg(t / 3600 / 24, X, State_params)

        # 3. Run forward model
        State_var = self.M.step(t, State_var, State_params, nstep=self.nstep)

        # Update time
        t += self.nstep * self.M.dt

        return t, Jo, State_var, State_params, X 
    
    def cost(self, X0):
        # Initial state
        State = self.State.copy()
        
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec:
                X = self.B.sqr(X0) + self.Xb
                Jb = X0.dot(X0)  # cost of background term
            else:
                X = X0 + self.Xb
                Jb = jnp.dot(X0, self.B.inv(X0))  # cost of background term
        else:
            X = X0 - self.Xb
            Jb = 0

        # Observational cost function evaluation
        Jo = 0.0
        State_var = State.var
        State_params = State.params
    
        # Initial values
        t_init = 0.0
        Jo_init = Jo
        val_init = (t_init, Jo_init, State_var, State_params, X)

        # Use fori_loop to iterate
        t_final = self.M.T[-1]
        n_iters = int(t_final // (self.nstep * self.M.dt)) + 1
        t, Jo, State_var, State_params, X = lax.fori_loop(0, n_iters, self.body_fn, val_init)

        # Handle the final timestamp after the loop
        Jo, State_var, t = lax.cond(
            self.H.is_obs_time(t),
            self.misfit_update,
            self.no_update,
            (Jo, State_var, t)
        )

        # Cost function
        J = 0.5 * (Jo + Jb)

        if self.save_minimization:
            self.J.append(J)

        return J
    
    
    def grad(self,X0): 

        grad_fun = jax.grad(self.cost)

        return grad_fun(X0)


def grad_test(J, G, X):
    h = np.random.random(X.size)
    h /= np.linalg.norm(h)
    JX = J(X)
    GX = G(X)
    Gh = h.dot(np.where(np.isnan(GX),0,GX))
    for p in range(10):
        lambd = 10**(-p)
        test = np.abs(1. - (J(X+lambd*h) - JX)/(lambd*Gh))
        
        print(f'{lambd:.1E} , {test:.2E}')

def plot_grad_test(L) :
    '''
    plots the result of a gradient test, L is a list containing
    the test results
    '''
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(L[0],L[1],'o','red')
    ax.plot(L[0],L[1],'orange')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('gradient test')
    ax.set_xlabel('order')
    ax.invert_xaxis()
    plt.show()


def background(config,State):
    '''
    if prescribe background files exist: read and return them
    else create them using an the 4Dvar with identity model (diffusion with Kdiffus=0) on the large scale basis components
    '''
      
    
    if config.path_background is not None and os.path.exists(config.path_background): 
        
        print('Background available at path_background')
        
        ds = xr.open_dataset(config.path_background)
         
        
        Xb = ds[config.name_bkg_var].values 
        
        ds.close()
         
        
        
    else: 
        print('Background not available, creating one with 4Dvar and Diffusion model')
        
        original_name_model = config.name_model
        original_name_mod_var = config.name_mod_var
        original_maxiter = config.maxiter
        original_maxiter_inner = config.maxiter_inner
        original_largescale_error_ratio = config.largescale_error_ratio
        original_Kdiffus = config.Kdiffus
        original_satellite = config.satellite
        
        # Modify appropriate config params to perform 4Dvar-Diffusion
        config.name_model = 'Diffusion'
        config.name_mod_var = ['ssh']
        config.maxiter = config.bkg_maxiter
        config.maxiter_inner = config.bkg_maxiter_inner
        config.largescale_error_ratio = 1.
        config.Kdiffus = config.bkg_Kdiffus
        if config.bkg_satellite is not None:
            config.satellite = config.bkg_satellite
        
        # Perform 4Dvar-Identity
        from src import state as state
        State = state.State(config) 
        from src import mod as mod
        Model = mod.Model(config,State) 
        from src import obs as obs
        dict_obs = obs.obs(config,State) 
        from src import ana as ana
        ana.ana(config,State,Model,dict_obs=dict_obs)
         
        
        
        # Reset original config params 
        config.name_model = original_name_model
        config.name_mod_var = original_name_mod_var
        config.maxiter = original_maxiter
        config.maxiter_inner = original_maxiter_inner
        config.largescale_error_ratio = original_largescale_error_ratio
        config.Kdiffus = original_Kdiffus
        if config.bkg_satellite is not None:
            config.satellite = original_satellite
        
        
        # Open background state 
        if config.path_background is None:
            path_save = f'{config.tmp_DA_path}/Xini.nc'
        else:
            path_save = config.path_background
            os.system(f'cp {config.tmp_DA_path}/Xini.nc {path_save}')
        
        ds = xr.open_mfdataset(path_save)
        Xb = ds[config.name_bkg_var].values 
        ds.close()
        
        # Delete temporary file 
        if config.path_background is None:
            os.system(f'rm {config.tmp_DA_path}/Xini.nc')
        
    return Xb
        
        
        
        
        
        
        
        
        
        
        