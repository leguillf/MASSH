#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""

import xarray as xr 
import numpy as np 

class Obsopt:

    def __init__(self,npix,dict_obs,dt):
        
        self.npix = npix
        self.dt = dt
        self.dict_obs = dict_obs        # observation dictionnary
            
    def isobserved(self,t):
        
        delta_t = [(t - tobs).total_seconds() for tobs in self.dict_obs.keys()]
        if len(delta_t)>0:
            is_obs = np.min(np.abs(delta_t))<=self.dt/2
        else: is_obs = False
        
        return is_obs


    def misfit(self,t,State):
        
         if self.isobserved(t): 
             
            Hx = State.getvar(2).ravel()
            
            delta_t = [(t - tobs).total_seconds() 
                       for tobs in self.dict_obs.keys()]
            t_obs = [tobs for tobs in self.dict_obs.keys()]
            
            ind_obs = np.argmin(np.abs(delta_t))
            tobs = t_obs[ind_obs]
            
            sat_info_list = self.dict_obs[tobs]['satellite']
            obs_file_list = self.dict_obs[tobs]['obs_name']
            # For now we consider only one complete SSH obs
            with xr.open_dataset(obs_file_list[0]) as ncin:
                yobs = ncin[sat_info_list[0].name_obs_var[0]].values.ravel()
            
            return Hx - yobs
        
    def adj(self,adState,misfit):
        adState.var[2] += misfit.reshape(adState.var[2].shape)
        
        
class Cov:
    
    def __init__(self,sigma):
        self.sigma = sigma
        
    def inv(self,X):
        return 1/self.sigma**2 * X    
        

class Variational:
    
    def __init__(self, 
                 M=None, H=None, State=None, R=None,B=None, Xb=None, 
                 tmp_DA_path=None):
        
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
        self.tmp_DA_path = tmp_DA_path
        
        
    def cost(self,X0):

        X = +X0
        
        # initial state
        State = self.State.free()
        
        # Background cost function evaluation 
        if self.B is not None:
            Jb = (X-self.Xb).dot(self.B.inv(X-self.Xb))
        else:
            Jb = 0
        
        # Observational cost function evaluation
        Jo = 0.
        State.save(self.tmp_DA_path + '/model_state_0.nc')    
        
        for i,time in enumerate(self.M.timestamps[:-1]):
            if self.H.isobserved(time):
                misfit = self.H.misfit(time,State) # d=Hx-xobs                
                Jo += misfit.dot(self.R.inv(misfit))
            self.M.step(self.M.T[i],State,X,step=i)
            State.save(self.tmp_DA_path + '/model_state_' + str(i+1) + '.nc')

        if self.H.isobserved(self.M.timestamps[-1]):
            misfit = self.H.misfit(self.M.timestamps[-1],State) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
        return J
    
        
    def grad(self,X0): 

        X = +X0 
        
        # Background cost function grandient
        if self.B is not None:
            gb = self.B.inv(X-self.Xb)
        else:
            gb = 0
        
        # Ajoint initialization   
        adState = self.State.free()
        adX = np.zeros_like(X0)
        
        # Current trajectory
        State = self.State.free()
        
        # Last timestamp
        if self.H.isobserved(self.M.timestamps[-1]):
            State.load(self.tmp_DA_path + '/model_state_' + str(self.M.nt-1) + '.nc')
            misfit = self.H.misfit(self.M.timestamps[-1],State) # d=Hx-yobs
            self.H.adj(adState,self.R.inv(misfit))
            
        # Time loop
        self.M.restart()        
        for i in reversed(range(0,self.M.nt-1)):
            
            t = self.M.timestamps[i]
            
            # Read model state
            State.load(self.tmp_DA_path + '/model_state_' + str(i) + '.nc')
            
            # One backward step
            adX = self.M.step_adj(self.M.T[i], adState, State, adX, X, step=i)
            
            # Calculation of adjoint forcing
            if self.H.isobserved(t):
                misfit = self.H.misfit(t,State) # d=Hx-yobs     
                self.H.adj(adState,self.R.inv(misfit))
        
        g = adX + gb
        
        return g 
            
    
def grad_test(J, G, X):
        h = np.random.random(X.size)
        h /= np.linalg.norm(h)
        JX = J(X)
        Gh = np.inner(h,G(X))
        for p in range(10):
            lambd = 10**(-p)
            test = (J(X+lambd*h) - JX)/(lambd*Gh)
            
            print(p,test)



