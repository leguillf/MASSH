#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import os,sys
import xarray as xr 
import numpy as np 
from scipy.spatial.distance import cdist


from . import grid

class Obsopt:

    def __init__(self,State,dict_obs,Model):
        
        self.npix = State.lon.size
        self.dict_obs = dict_obs        # observation dictionnary
        self.dt = Model.dt
        self.date_obs = {}
        for t in Model.timestamps:
            if self.isobserved(t):
                delta_t = [(t - tobs).total_seconds() 
                           for tobs in self.dict_obs.keys()]
                t_obs = [tobs for tobs in self.dict_obs.keys()] 
                
                ind_obs = np.argmin(np.abs(delta_t))
                self.date_obs[t] = t_obs[ind_obs]
                
        self.obs_sparse = {}
        # For grid interpolation:
        coords_geo = np.column_stack((State.lon.ravel(), State.lat.ravel()))
        self.coords_car = grid.geo2cart(coords_geo)
        self.indexes = {}
        self.weights = {}
        self.maskobs = {}
        
        for t in self.date_obs:
            # Read obs
            sat_info_list = self.dict_obs[self.date_obs[t]]['satellite']
            obs_file_list = self.dict_obs[self.date_obs[t]]['obs_name']
            
            obs_sparse = False   
            lon_obs = np.array([])
            lat_obs = np.array([])
            for sat_info,obs_file in zip(sat_info_list,obs_file_list):
                if sat_info.kind=='fullSSH':
                    if obs_sparse:
                        sys.exit("Error: in Obsopt: \
                                 can't handle 'fullSSH' and 'swot_simulator'\
                                 observations at the same time, sorry")
                    elif len(sat_info_list)>1:
                        sys.exit("Error: in Obsopt: \
                                 can't handle several 'fullSSH'\
                                 observations at the same time, sorry")
            
                elif sat_info.kind=='swot_simulator':
                    obs_sparse = True
                    with xr.open_dataset(obs_file) as ncin:
                        lon = ncin[sat_info.name_obs_lon].values.ravel()
                        lat = ncin[sat_info.name_obs_lat].values.ravel()
                    lon_obs = np.concatenate((lon_obs,lon))
                    lat_obs = np.concatenate((lat_obs,lat))
                                    
            self.obs_sparse[t] = obs_sparse
            
            if obs_sparse :
                # Compute indexes and weights of neighbour grid pixels
                print(t,'Compute obs interpolator')
                indexes,weights = self.interpolator(lon_obs,lat_obs)
                self.indexes[t] = indexes
                self.weights[t] = weights
                self.maskobs[t] = np.isnan(lon_obs)*np.isnan(lat_obs)
                    
        
    def isobserved(self,t):
        
        delta_t = [(t - tobs).total_seconds() for tobs in self.dict_obs.keys()]
        if len(delta_t)>0:
            is_obs = np.min(np.abs(delta_t))<=self.dt/2
        else: is_obs = False
        
        return is_obs
    
    def interpolator(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)
        dist = cdist(coords_car_obs,self.coords_car, metric="euclidean") 
        
        indexes = []
        weights = []
        for i in range(lon_obs.size):
            _dist = dist[i,:]
            # 4 closest
            ind4 = np.argsort(_dist)[:4]
            indexes.append(ind4)
            weights.append(1/_dist[ind4])   
            
        return indexes,weights

    def misfit(self,t,State):
        
        # Read obs
        sat_info_list = self.dict_obs[self.date_obs[t]]['satellite']
        obs_file_list = self.dict_obs[self.date_obs[t]]['obs_name']
        
        Yobs = np.array([])        
        for sat_info,obs_file in zip(sat_info_list,obs_file_list):
            with xr.open_dataset(obs_file) as ncin:
                yobs = ncin[sat_info.name_obs_var[0]].values.ravel() # SSH_obs
            Yobs = np.concatenate((Yobs,yobs))
        
        X = State.getvar(2).ravel() # SSH from state
        if self.obs_sparse[t] :
            # Get indexes and weights of neighbour grid pixels
            indexes = self.indexes[t]
            weights = self.weights[t]
            
            # Compute inerpolation of state to obs space
            HX = np.zeros_like(Yobs)
            
            for i,(ind,w) in enumerate(zip(indexes,weights)):
                if not self.maskobs[t][i]:
                    # Average
                    HX[i] = np.average(X[ind],weights=w)
        else:
            HX = X # H==Id
            
        res = HX - Yobs
        res[np.isnan(res)] = 0
        
        return res
    
    
    def adj(self,t,adState,misfit):
        
        if self.obs_sparse[t]:
            adHssh = np.zeros(self.npix)
            for i,(ind,w) in enumerate(zip(self.indexes[t],self.weights[t])):
                if not self.maskobs[t][i]:
                    # Average
                    for _ind,_w in zip(ind,w):
                        if w.sum()!=0:
                            adHssh[_ind] += _w*misfit[i]/(w.sum())
            
            adState.var[2] += adHssh.reshape(adState.var[2].shape)
        else:
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
        grad_test(self.cost, self.grad, X)
        
        
    def cost(self,X0):
        
        # initial state
        State = self.State.free()
        
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
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
            self.H.adj(self.M.timestamps[self.checkpoint[-1]],adState,self.R.inv(misfit))
            
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
                self.H.adj(timestamp,adState,self.R.inv(misfit))
                
        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient
        
        return g 
            
    
def grad_test(J, G, X):
        h = np.random.random(X.size)
        h /= np.linalg.norm(h)
        JX = J(X)
        GX = G(X)
        Gh = h.dot(np.where(np.isnan(GX),0,GX))
        for p in range(10):
            lambd = 10**(-p)
            test = np.abs(1. - (J(X+lambd*h) - JX)/(lambd*Gh))
            
            print('%.E' % lambd,'%.E' % test)



