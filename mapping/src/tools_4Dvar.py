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
from datetime import timedelta,datetime
from src import grid as grid
import jax.numpy as jnp 
import jax.lax as lax
import jax
from jax.lax import scan
jax.config.update("jax_enable_x64", True)
from sys import getsizeof
import matplotlib.pyplot as plt 


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

        # multiplication coefficient 
        self.cost_function_coeff = config.INV.cost_function_coeff
        
        # Wavelet reduced basis
        self.dtbasis = int(config.INV.timestep_checkpoint.total_seconds()//M.dt)
        self.basis = Basis 

        # Basis params information 
        # Commented because it doesn't work for multi model 
        # self.slice_params = Basis.slice_params
        
        # Save cost function and its gradient at each iteration 
        self.save_minimization = config.INV.save_minimization
        if self.save_minimization:
            self.J = []
            self.dJ = [] # For incremental 4Dvar only
            self.G = []
            # background cost term 
            self.Jb = {}
            # Commented because it doesn't work for multi model
            # for param in Basis.name_params : 
            #     self.Jb[param] = []
            # observational cost term 
            self.Jo = []
            # background grad norm
            self.Gb = {}
            # Commented because it doesn't work for multi model
            # for param in Basis.name_params : 
            #     self.Gb[param] = []
            # observational grad norm 
            self.Go = []
            
        
        # For incremental 4Dvar only
        self.X0 = self.Xb*0

        # Dictionnary to save misfits 
        self.misfits = {}

        # Dictionnary to save States (to avoid storing them with nc)
        self.States = {}

        # Computing RMSE at each iteration (for OSSE framework)
        self.save_rmse = config.INV.save_rmse 
        self.tmp_DA_path = config.EXP.tmp_DA_path
        if self.save_rmse :
            self.path_reference = config.INV.path_reference 
            self.var_to_compare = config.INV.var_to_compare 
            self.name_var_reference = config.INV.name_var_reference

            self.ds_reference = xr.open_mfdataset(self.path_reference)
            self.ds_reference = self.ds_reference.sel(time=slice(np.datetime64(config.EXP.init_date),np.datetime64(config.EXP.final_date)+np.timedelta64(1,"h")))
            self.ds_reference = self.ds_reference.interp({self.name_var_reference["lon"]:State.lon[0,:],
                                                        self.name_var_reference["lat"]:State.lat[:,0]})
            self.n_compute_cost=0 # variable of number of cost funtion computation
            # Dictionnary to save cross differences  
            self.cross_diff = {}
            for _var in self.var_to_compare: 
                self.cross_diff[_var]=np.empty((len(self.checkpoints)-1,
                                                self.ds_reference.dims[self.name_var_reference["lon"]],
                                                self.ds_reference.dims[self.name_var_reference["lat"]]))

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

        # Measuring computation times 
        #t_misfit = []
        #t_basis = []
        #t_model = []

        for i in range(len(self.checkpoints)-1):
            
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            nstep = self.checkpoints[i+1] - self.checkpoints[i]

            # Measuring computation times
            #t0 = datetime.now()

            # 1. Misfit
            if self.H.is_obs(timestamp):
                misfit, self.misfits[timestamp] = self.H.misfit(timestamp,State) # d=Hx-xobs   
                Jo += misfit.dot(self.R.inv(misfit))
                if self.save_rmse:
                    # ds_rmse = xr.Dataset()
                    for _var in self.var_to_compare.keys():
                        self.cross_diff[_var][i,:,:] = (self.ds_reference[_var][t//3600,:,:]-State.var[self.var_to_compare[_var]])**2

                        # _var_rmse = xr.DataArray(np.sqrt(np.mean((self.ds_reference[_var][t//3600,:,:]-State.var[self.var_to_compare[_var]])**2)), 
                        #                          dims=('time','lon', 'lat'), 
                        #                          coords={'time': self.ds_reference.time[0]+np.timedelta64(t//3600,'h'),
                        #                                  'lon': self.ds_reference[self.name_var_reference["lon"]], 
                        #                                  'lat': self.ds_reference[self.name_var_reference["lat"]]},
                        #                          name=_var)
                        # ds_rmse[_var]=_var_rmse
                    


            # Measuring computation times
            #t_misfit.append(datetime.now()-t0)
            #t0 = datetime.now()

            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                self.basis.operg(t/3600/24, X, State=State)

            self.States[self.checkpoints[i]] = State.copy()

            # Measuring computation times
            #t_basis.append(datetime.now()-t0)
            #t0 = datetime.now()

            # 3. Run forward model
            self.M.step(t=t,State=State,nstep=nstep)

            # Measuring computation times
            #t_model.append(datetime.now()-t0)
        

        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if self.H.is_obs(timestamp):
            misfit, self.misfits[timestamp] = self.H.misfit(timestamp,State) # d=Hx-xobsx
            Jo += misfit.dot(self.R.inv(misfit))
        
        # Cost function 
        J = 1/2 * (Jo + Jb)

        # print("Jb = ",Jb)
        # print("Jo = ",Jo)
        
        State.plot(title='State variables at the end of cost function evaluation')
        ### TO DO : HOW TO PLOT THE PARAMETERS ### 
        #State.plot(title='Parameters at the end of cost function evaluation',params=True)
        
        if self.save_minimization: #saving cost function terms 
            self.J.append(J) # total cost
            self.Jo.append(Jo) # observational cost 
            # Commented because it doesn't work for multi model
            # for param in self.Jb.keys() : # background cost
            #     ## defining X0_specific - the part of vector X for specific parameter && B_specific - the part of Cov matrix B for specific parameter##
            #     if param == "hbcx":
            #         X0_specific = np.concatenate((X0[self.slice_params["hbcS"]],X0[self.slice_params["hbcN"]]))
            #         B_specific = Cov(np.concatenate((self.B.sigma[self.slice_params["hbcS"]],self.B.sigma[self.slice_params["hbcN"]])))
            #     elif param == "hbcy":
            #         X0_specific = np.concatenate((X0[self.slice_params["hbcE"]],X0[self.slice_params["hbcW"]]))
            #         B_specific = Cov(np.concatenate((self.B.sigma[self.slice_params["hbcE"]],self.B.sigma[self.slice_params["hbcW"]])))
            #     else :
            #         X0_specific = X0[self.slice_params[param]]
            #         B_specific = Cov(self.B.sigma[self.slice_params[param]])
            #     if self.B is not None:
            #         if self.prec :
            #             self.Jb[param].append(X0_specific.dot(X0_specific)) # cost of background term
            #         else:
            #             self.Jb[param].append(np.dot(X0_specific,B_specific.inv(X0_specific))) # cost of background term
            #     else:
            #         self.Jb[param].append(0)
        
        if self.save_rmse:
            self.n_compute_cost+=1
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H%M%S")
            ds = xr.Dataset()
            for _var in self.var_to_compare.keys():
                ds["rmse_"+_var] = xr.DataArray([np.sqrt(np.nansum(self.cross_diff[_var])/self.cross_diff[_var].size)], 
                                        dims=["i"], coords={"i": [self.n_compute_cost]})
            ds.to_netcdf(os.path.join(self.tmp_DA_path,'rmse-'+current_time+'.nc'))
            ds.close()

        # Measuring computation times
        #print(f"MEAN COMPUTATION TIME FOR COST FUNCTION : \n - MISFIT : {np.mean(np.array(t_misfit))} \n - BASIS : {np.mean(np.array(t_basis))} \n - MODEL : {np.mean(np.array(t_model))} \n ")
        
        return J*self.cost_function_coeff
    
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
            self.H.adj(timestamp,adState,self.misfits[timestamp],self.R)

        # Measuring computation times 
        #t_misfit = []
        #t_basis = []
        #t_model = []

        # Time loop
        for i in reversed(range(0,len(self.checkpoints)-1)):
            
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            
            # Measuring computation times
            #t0 = datetime.now()

            State = self.States[self.checkpoints[i]]
            
            # 3. Run adjoint model 
            self.M.step_adj(t=t, adState=adState, State=State, nstep=nstep) # i+1 --> i

            # Measuring computation times
            #t_model.append(datetime.now()-t0)
            #t0 = datetime.now()

            adState.plot(title=f'adjoint variables at i = {i}')
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                adX += self.basis.operg_transpose(t=t/3600/24,adState=adState)

            # Measuring computation times
            #t_basis.append(datetime.now()-t0)
            #t0 = datetime.now()
            
            # 1. Misfit 
            if self.H.is_obs(timestamp):
                self.H.adj(timestamp,adState,self.misfits[timestamp],self.R)

            # Measuring computation times
            #t_misfit.append(datetime.now()-t0)


        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient

        adState.plot(title='adjoint variables at the end of gradient function evaluation')
        self.basis.operg(t/3600/24,adX,State=State)

        ### TO DO : HOW TO PLOT THE PARAMETERS ### 
        #State.plot(title='adjoint parameters at the end of gradient function evaluation',params=True)
        
        if self.save_minimization:
            self.G.append(np.max(np.abs(g))) # gradient of all parameters 
            self.Go.append(np.max(np.abs(adX))) # observational cost 
            # Commented because it doesn't work for multi model
            # for param in self.Gb.keys() : # background cost
            #     ## defining X0_specific - the part of vector X for specific parameter && B_specific - the part of Cov matrix B for specific parameter##
            #     if param == "hbcx":
            #         X0_specific = np.concatenate((X0[self.slice_params["hbcS"]],X0[self.slice_params["hbcN"]]))
            #         B_specific = Cov(np.concatenate((self.B.sigma[self.slice_params["hbcS"]],self.B.sigma[self.slice_params["hbcN"]])))
            #     elif param == "hbcy":
            #         X0_specific = np.concatenate((X0[self.slice_params["hbcE"]],X0[self.slice_params["hbcW"]]))
            #         B_specific = Cov(np.concatenate((self.B.sigma[self.slice_params["hbcE"]],self.B.sigma[self.slice_params["hbcW"]])))
            #     else :
            #         X0_specific = X0[self.slice_params[param]]
            #         B_specific = Cov(self.B.sigma[self.slice_params[param]])
            #     if self.B is not None:
            #         if self.prec :
            #             self.Gb[param].append(np.max(np.abs(X0_specific))) # cost of background term
            #         else:
            #             self.Gb[param].append(np.max(np.abs(self.B_specific.inv(X0_specific)))) # cost of background term
            #     else:
            #         self.Gb[param].append(0)

        # Measuring computation times
        #print(f"MEAN COMPUTATION TIME FOR COST FUNCTION : \n - MISFIT : {np.mean(np.array(t_misfit))} \n - BASIS : {np.mean(np.array(t_basis))} \n - MODEL : {np.mean(np.array(t_model))} \n ")

        return g
    

class Variational_jax:
    
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
        self.Xb = jnp.array(Xb)
        
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
                X = jnp.array(np.random.random(self.basis.nbasis)-0.5)
            else:
                X = jnp.array(self.B.sqr(np.random.random(self.basis.nbasis)-0.5) + self.Xb)
            grad_test(self.cost,self.grad,X)
            
            
        
    def cost(self,X0):
                
        # Initial state
        State = self.State.copy()
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
                Jb = jnp.dot(X0,self.B.inv(X0)) # cost of background term
        else:
            X  = X0 - self.Xb
            Jb = 0
    
        # Observational cost function evaluation
        Jo = 0.
        State_var = State.var
        State_params = State.params

        #for i in range(len(self.checkpoints)-1):
        i = 0

        timestamp = self.M.timestamps[self.checkpoints[i]]
        t = self.M.T[self.checkpoints[i]]
        nstep = self.checkpoints[i+1] - self.checkpoints[i]
        
        # 1. Misfit
        if timestamp in self.H.date_obs:
            misfit = self.H.misfit_jit(State_var,timestamp) # d=Hx-xobs   
            Jo += misfit.dot(self.R.inv(misfit))
        
        # 2. Reduced basis
        if self.checkpoints[i]%self.dtbasis==0:
            State_params = self.basis.operg_jit(t/3600/24, X, State_params)

        # 3. Run forward model
        State_var = self.M.step_jit(t,State_var, State_params,nstep=nstep)

        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if timestamp in self.H.date_obs:
            misfit = self.H.misfit_jit(State_var,timestamp) # d=Hx-xobsx
            Jo += misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
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
    #print("GX",GX)
    Gh = h.dot(np.where(np.isnan(GX),0,GX))
    #print("Gh",Gh)
    L = [[],[]]
    for p in range(10):
        lambd = 10**(-p)
        JXlh = J(X+lambd*h)
        test = np.abs(1. - (JXlh - JX)/(lambd*Gh))
        L[0].append(lambd)
        L[1].append(test)
        print(f'{lambd:.1E} , {test:.2E}')
    plot_grad_test(L)

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
        
        
        
        
        
        
        
        
        
        
        