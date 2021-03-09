#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import os
import xarray as xr 
import numpy as np 

class Obsopt:

    def __init__(self,npix,dict_obs,timestamps,dt):
        
        self.npix = npix
        self.dict_obs = dict_obs        # observation dictionnary
        self.dt = dt
        self.ind_obs = []
        for i,t in enumerate(timestamps):
            if self.isobserved(t):
                self.ind_obs.append(i)
        self.ind_obs = np.asarray(self.ind_obs)
        
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
    
    def sqr(self,X):
        return self.sigma**0.5 * X   
        

class Variational:
    
    def __init__(self, 
                 M=None, H=None, State=None, R=None,B=None, Xb=None, 
                 tmp_DA_path=None, checkpoint=1, prec=False):
        
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
        
        # Compute checkpoints
        self.checkpoint = [0]
        if H.isobserved(M.timestamps[0]):
            self.isobs = [True]
        else:
            self.isobs = [False]
        check = 0
        for i,t in enumerate(M.timestamps[:-1]):
            if i>0 and (H.isobserved(t) or check==checkpoint):
                self.checkpoint.append(i)
                check = 0
                if H.isobserved(t):
                    self.isobs.append(True)
                else:
                    self.isobs.append(False)
            check += 1
        if H.isobserved(M.timestamps[-1]):
            self.isobs.append(True)
        else:
            self.isobs.append(False)
            
        self.checkpoint.append(len(M.timestamps)-1) # last timestep
        
        print('checkpoint:')
        for i,check in enumerate(self.checkpoint):
            print(M.timestamps[check],end='')
            if self.isobs[i]:
                print(': obs',end='')
            print()
        
        # preconditioning
        self.prec = prec
        
        # Grad test
        X = np.random.random()
        if self.B is not None:
            X *= self.B.sigma 
        print('gradient test:')
        #grad_test(self.cost, self.grad, X)
        
        
    def cost(self,X0):
        
        # initial state
        State = self.State.free()
        
        # Background cost function evaluation 
        # if self.B is not None:
        #     Jb = (X-self.Xb).dot(self.B.inv(X-self.Xb))
        # else:
        #     Jb = 0
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                #gb = v        # gradient of background term
                Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
                #gb = np.dot(self.B.inv,v) # gradient of background term
                Jb = np.dot(X0,self.B.inv(X0))        # cost of background term
        else:
            X  = X0 + self.Xb
            Jb = 0
        
        # Observational cost function evaluation
        Jo = 0.
        State.save(os.path.join(self.tmp_DA_path,
                    'model_state_' + str(self.checkpoint[0]) + '.nc'),
                    grd=False)
        
        for i in range(len(self.checkpoint)-1):
            
            timestamp = self.M.timestamps[self.checkpoint[i]]
            t = self.M.T[self.checkpoint[i]]
            
            # Misfit
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State) # d=Hx-xobs                
                Jo += misfit.dot(self.R.inv(misfit))
                
            # Run forward model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step(t,State,X,nstep=nstep)
            
            # Save state for adj computation 
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i+1]) + '.nc'),
                        grd=False)
            

        if self.isobs[-1]:
            misfit = self.H.misfit(self.M.timestamps[self.checkpoint[-1]],State) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
        return J
    
        
    def grad(self,X0): 
                
        X = +X0 
        
        # Background cost function grandient
        # if self.B is not None:
        #     gb = self.B.inv(X-self.Xb)
        # else:
        #     gb = 0
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                gb = X0      # gradient of background term
                #Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
                gb = self.B.inv(X0) # gradient of background term
                #Jb = np.dot(X0,np.dot(self.B.inv,X0))         # cost of background term
        else:
            X  = X0 + self.Xb
            gb = 0
        
        # Ajoint initialization   
        adState = self.State.free()
        adX = np.zeros_like(X0)
        
        # Current trajectory
        State = self.State.free()
        
        # Last timestamp
        if self.isobs[-1]:
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[-1]) + '.nc'))
            misfit = self.H.misfit(self.M.timestamps[self.checkpoint[-1]],State) # d=Hx-yobs
            self.H.adj(adState,self.R.inv(misfit))
            
        # Time loop
        self.M.restart()  
        for i in reversed(range(0,len(self.checkpoint)-1)):
            
            timestamp = self.M.timestamps[self.checkpoint[i]]
            t = self.M.T[self.checkpoint[i]]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i]) + '.nc'))
            
            # Run adjoint model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            adX = self.M.step_adj(t, adState, State, adX, X, nstep=nstep)
            
            # Misfit 
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State) # d=Hx-yobs     
                self.H.adj(adState,self.R.inv(misfit))
                
        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient
        
        return g 
            
    
def grad_test(J, G, X):
        h = np.random.random(X.size)
        h /= np.linalg.norm(h)
        JX = J(X)
        Gh = np.inner(h,G(X))
        for p in range(10):
            lambd = 10**(-p)
            test = np.abs(1. - (J(X+lambd*h) - JX)/(lambd*Gh))
            
            print('%.E' % lambd,'%.E' % test)



