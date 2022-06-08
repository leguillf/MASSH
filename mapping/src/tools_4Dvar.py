#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import os,sys
import xarray as xr 
import numpy as np 
import pandas as pd 
from datetime import timedelta,datetime
from src import grad_tool as grad_tool
from src import grid as grid
from copy import deepcopy
import pickle


import matplotlib.pylab as plt 

class Obsopt:

    def __init__(self,config,State,dict_obs,Model):
        
        self.npix = State.lon.size
        self.dict_obs = dict_obs        # observation dictionnary
        self.dt = config.dtmodel
        self.date_obs = {}
        self.obs_data = {}
        
        self.save_obs_tmp = False
        
        self.compute_H = config.compute_H
        
        if config.time_obs_min is not None:
            time_obs_min = config.time_obs_min
        else:
            time_obs_min = config.init_date
        
        if config.time_obs_max is not None:
            time_obs_max = config.time_obs_max
        else:
            time_obs_max = config.final_date
            
        date1 = time_obs_min.strftime('%Y%m%d')
        date2 = time_obs_max.strftime('%Y%m%d')
        
        box = f'{int(State.lon.min())}_{int(State.lon.max())}_{int(State.lat.min())}_{int(State.lat.max())}'
        self.name_H = f'H_{"_".join(config.satellite)}_{date1}_{date2}_{box}_{int(State.dx)}_{int(State.dy)}_{config.Npix_H}'
        print(self.name_H)
        
        if State.config['name_model'] in ['Diffusion','SW1L','SW1LM','QG1L','JAX-QG1L'] or \
             hasattr(config.name_model,'__len__') and len(config.name_model)==2:
            for t in Model.timestamps:
                if self.isobserved(t):
                    delta_t = [(t - tobs).total_seconds() 
                               for tobs in self.dict_obs.keys()]
                    t_obs = [tobs for tobs in self.dict_obs.keys()] 
                    
                    ind_obs = np.argmin(np.abs(delta_t))
                    self.date_obs[t] = t_obs[ind_obs]
        
        # Temporary path where to save H operators
        self.tmp_DA_path = config.tmp_DA_path
        if config.path_H is not None:
            # We'll save to *path_H* or read in *path_H* from previous run
            self.path_H = config.path_H
            self.read_H = True
            if not os.path.exists(self.path_H):
                os.makedirs(self.path_H)
        else:
            # We'll use temporary directory to save the files
            self.path_H = self.tmp_DA_path
            self.read_H = False
        self.obs_sparse = {}
        
        # For grid interpolation:
        self.Npix = config.Npix_H
        coords_geo = np.column_stack((State.lon.ravel(), State.lat.ravel()))
        self.coords_car = grid.geo2cart(coords_geo)

        if State.config['name_model'] in ['SW1L','SW1LM']:
            coords_geo_bc = np.column_stack((
                np.concatenate((State.lon[0,:],State.lon[1:-1,-1],State.lon[-1,:],State.lon[:,0])),
                np.concatenate((State.lat[0,:],State.lat[1:-1,-1],State.lat[-1,:],State.lat[:,0]))
                ))
            
        elif State.config['name_model'] in ['QG1L','JAX-QG1L'] or hasattr(config.name_model,'__len__') and len(config.name_model)==2:
            if State.config['name_model'] in ['QG1L','JAX-QG1L'] : mask = Model.qgm.mask
            else: mask = Model.Model_BM.qgm.mask
            coords_geo_bc = np.column_stack((State.lon[np.where(mask<2)].ravel(),
                                             State.lat[np.where(mask<2)].ravel()))
            
        elif State.config['name_model'] in ['Diffusion'] or hasattr(config.name_model,'__len__') and len(config.name_model)==2:
            mask = Model.mask
            coords_geo_bc = np.column_stack((State.lon[np.where(mask<2)].ravel(),
                                             State.lat[np.where(mask<2)].ravel()))
            
        self.ind_bc = []
        for i in range(coords_geo.shape[0]):
            if np.any(np.all(np.isclose(coords_geo_bc,coords_geo[i]), axis=1)):
                self.ind_bc.append(i)
        
        # Mask coast pixels
        self.dist_coast = config.dist_coast
        if config.mask_coast and self.dist_coast is not None and State.mask is not None and np.any(State.mask):
            self.flag_mask_coast = True
            lon_land = State.lon[State.mask].ravel()
            lat_land = State.lat[State.mask].ravel()
            coords_geo_land = np.column_stack((lon_land,lat_land))
            self.coords_car_land = grid.geo2cart(coords_geo_land)
            
        else: self.flag_mask_coast = False
        
        for t in self.date_obs:
            self.process_obs(t)
            
        # Compute checkpoints
        self.checkpoint = [0]
        if self.isobserved(Model.timestamps[0]):
            self.isobs = [True]
        else:
            self.isobs = [False]
        check = 0
        for i,t in enumerate(Model.timestamps[:-1]):
            if i>0 and (self.isobserved(t) or check==config.checkpoint):
                self.checkpoint.append(i)
                if check==config.checkpoint:
                    check = 0
                if self.isobserved(t):
                    self.isobs.append(True)
                else:
                    self.isobs.append(False)
            check += 1
        if self.isobserved(Model.timestamps[-1]):
            self.isobs.append(True)
        else:
            self.isobs.append(False)   
        self.checkpoint.append(len(Model.timestamps)-1) # last timestep
        self.checkpoint = np.asarray(self.checkpoint)
            
    def process_obs(self,t):

        if self.read_H and not self.compute_H:
            file_H = os.path.join(
                self.path_H,self.name_H+t.strftime('_%Y%m%d_%H%M.nc'))
            if os.path.exists(file_H):
                new_file_H = os.path.join(
                    self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc'))
                os.system(f"cp {file_H} {new_file_H}")
                self.obs_sparse[t] = True
                return t
        else:
            file_H = os.path.join(
                self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc'))
        
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
                             observations at the same time, sorry.\
                             Hint: reduce *assimiliation_time_step* parameter")
                
                
            elif sat_info.kind in ['swot_simulator','CMEMS']:
                obs_sparse = True
            with xr.open_dataset(obs_file) as ncin:
                lon = ncin[sat_info.name_obs_lon].values
                lat = ncin[sat_info.name_obs_lat].values
            if len(lon.shape)==1  and len(ncin[sat_info.name_obs_var[0]].shape)>1 :
                lon,lat = np.meshgrid(lon,lat)
            lon = lon.ravel()
            lat = lat.ravel()
            lon_obs = np.concatenate((lon_obs,lon))
            lat_obs = np.concatenate((lat_obs,lat))
        
        if obs_sparse:
            # Compute indexes and weights of neighbour grid pixels
            indexes,weights = self.interpolator(lon_obs,lat_obs)
        
            # Compute mask 
            maskobs = np.isnan(lon_obs)*np.isnan(lat_obs)
            if self.flag_mask_coast:
                coords_geo_obs = np.column_stack((lon_obs,lat_obs))
                coords_car_obs = grid.geo2cart(coords_geo_obs)
                for i in range(lon_obs.size):
                    _dist = np.min(np.sqrt(np.sum(np.square(coords_car_obs[i]-self.coords_car_land),axis=1)))
                    if _dist<self.dist_coast:
                        maskobs[i] = True
            
            # save in netcdf
            dsout = xr.Dataset({"indexes": (("Nobs","Npix"), indexes),
                                "weights": (("Nobs","Npix"), weights),
                                "maskobs": (("Nobs"), maskobs)},                
                               )
            dsout.to_netcdf(file_H,
                encoding={'indexes': {'dtype': 'int16'}})
            
            if self.read_H:
                    new_file_H = os.path.join(
                        self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc'))
                    if file_H!=new_file_H:
                        os.system(f"cp {file_H} {new_file_H}")
                        
        self.obs_sparse[t] = obs_sparse
        
        return t
                
    
            
    def isobserved(self,t):
        
        delta_t = [(t - tobs).total_seconds() for tobs in self.dict_obs.keys()]
        if len(delta_t)>0:
            is_obs = np.min(np.abs(delta_t))<=self.dt/2
        else: is_obs = False
        
        return is_obs
    
        
    
    def interpolator(self,lon_obs,lat_obs):
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        indexes = []
        weights = []
        for iobs in range(lon_obs.size):
            _dist = np.sqrt(np.sum(np.square(coords_car_obs[iobs]-self.coords_car),axis=1))
            # Npix closest
            ind_closest = np.argsort(_dist)
            # Get Npix closest pixels (ignoring boundary pixels)
            n = 0
            i = 0
            ind = []
            w = []
            while n<self.Npix:
                if ind_closest[i] in self.ind_bc:
                    #Ignoring boundary pixels 
                    w.append(0.)
                else:
                    w.append(1/_dist[ind_closest[i]])
                ind.append(ind_closest[i])
                n += 1
                i +=1 
            indexes.append(ind)
            weights.append(w)   
            
        return np.asarray(indexes),np.asarray(weights)
    

    
    def H(self,t,X):
        
        if not self.obs_sparse[t] :
            return X
        # Get indexes and weights of neighbour grid pixels
        ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc')))
        indexes = ds['indexes'].values
        weights = ds['weights'].values
        maskobs = ds['maskobs'].values
        
        # Compute inerpolation of X to obs space
        HX = np.zeros(indexes.shape[0])
        
        for i,(mask,ind,w) in enumerate(zip(maskobs,indexes,weights)):
            if not mask:
                # Average
                if ind.size>1:
                    try:
                        HX[i] = np.average(X[ind],weights=w)
                    except:
                        HX[i] = np.nan
                else:
                    HX[i] = X[ind[0]]
            else:
                HX[i] = np.nan
        
        return HX

    def misfit(self,t,State,dState=None):
        
        if self.save_obs_tmp and t in self.obs_data:
            Yobs,noise = self.obs_data[t]
        else:
            # Read obs
            sat_info_list = self.dict_obs[self.date_obs[t]]['satellite']
            obs_file_list = self.dict_obs[self.date_obs[t]]['obs_name']
            
            Yobs = np.array([]) 
            noise = np.array([]) 
            for sat_info,obs_file in zip(sat_info_list,obs_file_list):
                with xr.open_dataset(obs_file) as ncin:
                    yobs = ncin[sat_info.name_obs_var[0]].values.ravel() # SSH_obs
                Yobs = np.concatenate((Yobs,yobs))
                if sat_info.sigma_noise is not None:
                    _noise = sat_info.sigma_noise
                else:
                    _noise = 1
                noise = np.concatenate((noise,np.ones_like(yobs)*_noise))
            if self.save_obs_tmp:
                self.obs_data[t] = (Yobs,noise)
            
        X = State.getvar(ind=State.get_indobs()).ravel() # SSH from state
        
        # Only for incremental 4Dvar
        if dState is not None:
            dX = dState.getvar(ind=State.get_indobs()).ravel() # SSH from state
            X += dX
            
        HX = self.H(t,X)
        
        res = (HX - Yobs)
            
        res[np.isnan(res)] = 0
        
        inv_noise2 = noise**(-2)
        
        # Save misfit
        with open(os.path.join(self.tmp_DA_path,f'misfit_{t}.pic'),'wb') as f:
            pickle.dump((res,inv_noise2),f)

        return res,inv_noise2
    
    def load_misfit(self,t):
        
        with open(os.path.join(self.tmp_DA_path,f'misfit_{t}.pic'),'rb') as f:
            res,inv_noise2 = pickle.load(f)
        
        return res,inv_noise2

        

    def adj(self,t,adState,misfit):
        
        if not self.obs_sparse[t] :
            ind = adState.get_indobs()
            adState.var[ind] += misfit.reshape(adState.var[ind].shape)
        
        else:
        
            adH = np.zeros(self.npix)
            ds = xr.open_dataset(os.path.join(
                    self.tmp_DA_path,self.name_H+t.strftime('_%Y%m%d_%H%M.nc')))
            indexes = ds['indexes'].values
            weights = ds['weights'].values
            maskobs = ds['maskobs'].values
            Nobs,Npix = indexes.shape
            
            for i in range(Nobs):
                if not maskobs[i]:
                    # Average
                    for j in range(Npix):
                        if weights[i].sum()!=0:
                            adH[indexes[i,j]] += weights[i,j]*misfit[i]/(weights[i].sum())
        
            ind = adState.get_indobs()
            
            adState.var[ind] += adH.reshape(adState.var[ind].shape)

        
            
class Cov :
    # case of a simple diagonal covariance matrix
    def __init__(self,sigma=None):
        
        if sigma is None:
            sigma = 1
            
        self.sigma = sigma
        
    def inv(self,X):
        return 1/self.sigma**2 * X    
    
    def sqr(self,X):
        return self.sigma**0.5 * X
    
    

class Variational:
    
    def __init__(self, 
                 config=None, M=None, H=None, State=None, R=None,B=None, basis=None, Xb=None):
        
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
        self.tmp_DA_path = config.tmp_DA_path
        
        # preconditioning
        self.prec = config.prec
        
        # Wavelet reduced basis
        self.dtbasis = config.checkpoint
        self.basis = basis 
        
        # Save cost function and its gradient at each iteration 
        self.save_minimization = config.save_minimization
        if self.save_minimization:
            self.J = []
            self.dJ = [] # For incremental 4Dvar only
            self.G = []
        
        # For incremental 4Dvar only
        self.X0 = self.Xb*0
        
        # Grad test
        if config.compute_test:
            print('Gradient test:')
            X = 1e-2*(np.random.random(self.basis.nbasis)-0.5)*self.B.sigma 
            grad_test(self.cost,self.grad,X)
            
            
        
    def cost(self,X0):
                
        # initial state
        State = self.State.free()
        State.params = np.zeros((self.M.nparams,))
        
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
        for i in range(len(self.H.checkpoint)-1):

            timestamp = self.M.timestamps[self.H.checkpoint[i]]
            t = self.M.T[self.H.checkpoint[i]]
            nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
            
            # 1. Misfit
            if self.H.isobs[i]:
                misfit, inv_noise2 = self.H.misfit(timestamp,State) # d=Hx-xobs   
                Jo += (misfit*inv_noise2).dot(self.R.inv(misfit))
            
            # 2. Reduced basis
            if self.H.checkpoint[i]%self.dtbasis==0:
                self.basis.operg(X,t/3600/24,State=State)
        
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.H.checkpoint[i]) + '.nc'))

            # 3. Run forward model
            self.M.step(t=t,State=State,nstep=nstep)
            
        if self.H.isobs[-1]:
            timestamp = self.M.timestamps[self.H.checkpoint[-1]]
            misfit, inv_noise2 = self.H.misfit(timestamp,State) # d=Hx-xobsx
            Jo += (misfit*inv_noise2).dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
        State.plot(title='end of cost function evaluation')
        
        if self.save_minimization:
            self.J.append(J)
        
        return J
    
    
    def dcost(self,dX0):
                
        # initial state
        State = self.State.free()
        dState = self.State.free()
        State.params = np.zeros((self.M.nparams,))
        dState.params = np.zeros((self.M.nparams,))
        
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                dX  = self.B.sqr(dX0)
                dJb = (dX0+self.X0).dot(dX0+self.X0) # cost of background term
            else:
                dX  = +dX0 
                dJb = np.dot(dX0+self.X0,self.B.inv(dX0+self.X0)) # cost of background term
        else:
            dX  = +dX0 
            dJb = 0
    
        # Observational cost function evaluation
        dJo = 0.
        for i in range(len(self.H.checkpoint)-1):

            timestamp = self.M.timestamps[self.H.checkpoint[i]]
            t = self.M.T[self.H.checkpoint[i]]
            nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.H.checkpoint[i]) + '.nc'))
            
            # 1. Misfit
            if self.H.isobs[i]:
                misfit,inv_noise2 = self.H.misfit(timestamp,State,dState=dState) # d=Hx-xobs
                dJo += (misfit*inv_noise2).dot(self.R.inv(misfit))
            
            # 2. Reduced basis
            if self.H.checkpoint[i]%self.dtbasis==0:
                self.basis.operg(dX,t/3600/24,State=dState)
                
            # 3. Run forward model
            self.M.step_tgl(t=t, dState=dState, State=State, nstep=nstep)
            
        if self.H.isobs[-1]:
            timestamp = self.M.timestamps[self.H.checkpoint[-1]]
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.H.checkpoint[-1]) + '.nc'))
            misfit,inv_noise2 = self.H.misfit(timestamp,State,dState=dState) # d=Hx-xobs
            dJo += (misfit*inv_noise2).dot(self.R.inv(misfit))  
        
        # Cost function 
        dJ = 1/2 * (dJo + dJb)
        
        dState.plot(title='end of quadratic cost function evaluation')
        
        if self.save_minimization:
            self.dJ.append(dJ)
                
        return dJ
    
        
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
        State = self.State.free()
        
        # Ajoint initialization   
        adState = self.State.free()
        adState.params = np.zeros((self.M.nparams,))
        adX = X*0
        
        # Last timestamp
        if self.H.isobs[-1]:
            timestamp = self.M.timestamps[self.H.checkpoint[-1]]
            misfit,inv_noise2 = self.H.load_misfit(timestamp) # d=Hx-yobs
            self.H.adj(timestamp,adState,self.R.inv(misfit*inv_noise2))

        # Time loop
        for i in reversed(range(0,len(self.H.checkpoint)-1)):
            
            nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
            timestamp = self.M.timestamps[self.H.checkpoint[i]]
            t = self.M.T[self.H.checkpoint[i]]
 
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.H.checkpoint[i]) + '.nc'))
            
            # 3. Run adjoint model 
            self.M.step_adj(t=t, adState=adState, State=State, nstep=nstep) # i+1 --> i
            
            # 2. Reduced basis
            if self.H.checkpoint[i]%self.dtbasis==0:
                self.basis.operg_transpose(adState,adX,t/3600/24)
                adState.params *= 0
                
            # 1. Misfit 
            if self.H.isobs[i]:
                misfit,inv_noise2 = self.H.load_misfit(timestamp)
                self.H.adj(timestamp,adState,self.R.inv(misfit*inv_noise2))
        
        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient
        
        adState.plot()
        
        if self.save_minimization:
            self.G.append(np.max(np.abs(g)))
        
        return g 
    

    def dgrad(self,dX0): 
                
        dX = +dX0 
        
        if self.B is not None:
            if self.prec :
                dX  = self.B.sqr(dX0) 
                dgb = dX0      # gradient of background term
            else:
                dX  = dX0 
                dgb = self.B.inv(dX0) # gradient of background term
        else:
            dX  = dX0 
            dgb = 0
            
        # Current trajectory
        State = self.State.free()
        dState = self.State.free()
        
        # Ajoint initialization   
        adState = self.State.free()
        adState.params = np.zeros((self.M.nparams,))
        adX = dX*0
        
        # Last timestamp
        if self.H.isobs[-1]:
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.H.checkpoint[-1]) + '.nc'))
            dState.load(os.path.join(self.tmp_DA_path,
                       'model_dstate_' + str(self.H.checkpoint[-1]) + '.nc'))
            timestamp = self.M.timestamps[self.H.checkpoint[-1]]
            misfit = self.H.misfit(timestamp,dState.sum(State.scalar(-1)),square=True) # d=Hx-yobs
            self.H.adj(timestamp,adState,self.R.inv(misfit))

        # Time loop
        for i in reversed(range(0,len(self.H.checkpoint)-1)):
            
            nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
            timestamp = self.M.timestamps[self.H.checkpoint[i]]
            t = self.M.T[self.H.checkpoint[i]]
 
            # 4. Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.H.checkpoint[i]) + '.nc'))
            
            # 3. Run adjoint model 
            self.M.step_adj(t, adState, State, nstep=nstep) # i+1 --> i
            
            # 2. Reduced basis
            if self.H.checkpoint[i]%self.dtbasis==0:
                self.basis.operg_transpose(adState,adX,t/3600/24)
                adState.params *= 0
                
            # 1. Misfit 
            if self.H.isobs[i]:
                dState.load(os.path.join(self.tmp_DA_path,
                       'model_dstate_' + str(self.H.checkpoint[i]) + '.nc'))
                misfit = self.H.misfit(timestamp,dState.sum(State.scalar(-1)),square=True) # d=Hx-yobs
                HdX = self.H.H(timestamp,dState.getvar(ind=State.get_indobs()).flatten())
                self.H.adj(timestamp,adState,self.R.inv(misfit+HdX))
        
        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        dg = adX + dgb  # total gradient
        
        adState.plot()
        
        return dg 
    
    

# class incrVariational:
    
#     def __init__(self, 
#                  config=None, M=None, H=None, State=None, R=None,B=None, basis=None, Xb=None):
        
#         # Objects
#         self.M = M # model
#         self.H = H # observational operator
#         self.State = State # state variables
    
#         # Covariance matrixes
#         self.B = B
#         self.R = R
        
#         # Background state
#         self.Xb = Xb
        
#         # Temporary path where to save model trajectories
#         self.tmp_DA_path = config.tmp_DA_path
        
#         # preconditioning
#         self.prec = config.prec
        
#         # Wavelet reduced basis
#         self.dtbasis = config.checkpoint
#         self.basis = basis 
        
#         # Save cost function and its gradient at each iteration 
#         self.save_minimization = config.save_minimization
#         if self.save_minimization:
#             self.J = []
#             self.G = []
        
#         # Grad test
#         if config.compute_test:
#             print('Gradient test:')
#             X = (np.random.random(self.basis.nbasis)-0.5)*self.B.sigma 
#             grad_test(self.cost,self.grad,X)
            
            
        
#     def cost(self,X0):
                
#         # initial state
#         State = self.State.free()
#         State.params = np.zeros((self.M.nparams,))
        
#         # Background cost function evaluation 
#         if self.B is not None:
#             if self.prec :
#                 X  = self.B.sqr(X0) + self.Xb
#                 Jb = X0.dot(X0) # cost of background term
#             else:
#                 X  = X0 + self.Xb
#                 Jb = np.dot(X0,self.B.inv(X0)) # cost of background term
#         else:
#             X  = X0 + self.Xb
#             Jb = 0
    
#         # Observational cost function evaluation
#         Jo = 0.
#         for i in range(len(self.H.checkpoint)-1):

#             timestamp = self.M.timestamps[self.H.checkpoint[i]]
#             t = self.M.T[self.H.checkpoint[i]]
#             nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
            
#             # 1. Misfit
#             if self.H.isobs[i]:
#                 misfit = self.H.misfit(timestamp,State,square=False) # d=Hx-xobs   
#                 Jo += self.H.misfit(timestamp,State).dot(self.R.inv(misfit))
            
#             # 2. Reduced basis
#             if self.H.checkpoint[i]%self.dtbasis==0:
#                 self.basis.operg(X,t/3600/24,State=State)
        
#             State.save(os.path.join(self.tmp_DA_path,
#                         'model_state_' + str(self.H.checkpoint[i]) + '.nc'))

#             # 3. Run forward model
#             self.M.step_tgl(t, dState, State, nstep=nstep)
            
#         if self.H.isobs[-1]:
#             State.save(os.path.join(self.tmp_DA_path,
#                         'model_state_' + str(self.H.checkpoint[-1]) + '.nc'))
#             misfit = self.H.misfit(self.M.timestamps[self.H.checkpoint[-1]],State,square=False) # d=Hx-xobsx
#             Jo += misfit.dot(self.R.inv(misfit))  
        
#         # Cost function 
#         J = 1/2 * (Jo + Jb)
        
#         State.plot()
        
#         if self.save_minimization:
#             self.J.append(J)
        
#         return J
    
        
#     def grad(self,X0): 
                
#         X = +X0 
        
#         if self.B is not None:
#             if self.prec :
#                 X  = self.B.sqr(X0) + self.Xb
#                 gb = X0      # gradient of background term
#             else:
#                 X  = X0 + self.Xb
#                 gb = self.B.inv(X0) # gradient of background term
#         else:
#             X  = X0 + self.Xb
#             gb = 0
            
#         # Current trajectory
#         State = self.State.free()
        
#         # Ajoint initialization   
#         adState = self.State.free()
#         adState.params = np.zeros((self.M.nparams,))
#         adX = X*0
        
#         # Last timestamp
#         if self.H.isobs[-1]:
#             State.load(os.path.join(self.tmp_DA_path,
#                        'model_state_' + str(self.H.checkpoint[-1]) + '.nc'))
#             timestamp = self.M.timestamps[self.H.checkpoint[-1]]
#             misfit = self.H.misfit(timestamp,State,square=True) # d=Hx-yobs
#             self.H.adj(timestamp,adState,self.R.inv(misfit))

#         # Time loop
#         for i in reversed(range(0,len(self.H.checkpoint)-1)):
            
#             nstep = self.H.checkpoint[i+1] - self.H.checkpoint[i]
#             timestamp = self.M.timestamps[self.H.checkpoint[i]]
#             t = self.M.T[self.H.checkpoint[i]]
 
#             # 4. Read model state
#             State.load(os.path.join(self.tmp_DA_path,
#                        'model_state_' + str(self.H.checkpoint[i]) + '.nc'))
            
#             # 3. Run adjoint model 
#             self.M.step_adj(t, adState, State, nstep=nstep) # i+1 --> i
            
#             # 2. Reduced basis
#             if self.H.checkpoint[i]%self.dtbasis==0:
#                 self.basis.operg_transpose(adState,adX,t/3600/24)
#                 adState.params *= 0
                
#             # 1. Misfit 
#             if self.H.isobs[i]:
#                 misfit = self.H.misfit(timestamp,State,square=True) # d=Hx-yobs
#                 self.H.adj(timestamp,adState,self.R.inv(misfit))
        
#         if self.prec :
#             adX = np.transpose(self.B.sqr(adX)) 
        
#         g = adX + gb  # total gradient
        
#         adState.plot()
        
#         if self.save_minimization:
#             self.G.append(np.max(np.abs(g)))
        
#         return g 
    
class Variational_QG_init :
    
    def __init__(self,config=None,M=None, H=None, State=None, R=None,B=None, Xb=None, tmp_DA_path=None,
                 date_ini=None, date_final=None) :
        
         # Objects
        self.M = M # model
        self.H = H # observational operator
        self.State = State.free() # state variables
        self.adState = State.free()
    
        # Covariance matrixes
        self.B = B
        self.R = R
        
        # Background state
        Xb[np.isnan(Xb)] = 0
        self.Xb = Xb
        
        # Temporary path where to save model trajectories
        self.tmp_DA_path = tmp_DA_path
        
        # Covariance matrix building using gradient condition
        self.grad_term = config.grad_term
        if self.grad_term :
            self.gradop = grad_tool.grad_op(self.State)
            self.B_grad = Cov(self.State.config.sigma_B_grad)
        
        # preconditioning
        self.prec = config.prec
        
        # initial and final date of the assimilation window
        self.date_ini = date_ini
        self.date_final = date_final
        
        # model time step
        self.dt = timedelta(seconds=State.config.dtmodel)
        
        # checkpoint indicate the iteration where the algorithm stop
        self.checkpoint = [0]
        self.timestamps = [date_ini]
        # isobs has the same length as checkpoint and indicates when obs are available at a checkpoint
        if self.H.isobserved(self.date_ini) :
            self.isobs = [True]
        else :
            self.isobs = [False]
        
        t,i = date_ini+self.dt, 1
        while t < self.date_final :
            
            if self.H.isobserved(t) :
                self.checkpoint.append(i)
                self.timestamps.append(t)
                self.isobs.append(True)
            i += 1
            t += self.dt
        
        if self.H.isobserved(self.date_final) :
            self.isobs.append(True)
        else :
            self.isobs.append(False)
            
        self.n_iter = i
        self.checkpoint.append(self.n_iter)
        self.timestamps.append(self.date_final)
    
                
        # Boundary conditions
        if config.flag_use_boundary_conditions:
            timestamps_bc = np.array(
                [pd.Timestamp(timestamp) for timestamp in self.timestamps])
            self.bc_field, self.bc_weight = grid.boundary_conditions(
                config.file_boundary_conditions,
                config.lenght_bc,
                config.name_var_bc,
                timestamps_bc,
                State.lon,
                State.lat,
                config.flag_plot,
                mask=np.copy(State.mask))
        else: 
            self.bc_field = np.array([None,]*len(self.timestamps))
            self.bc_weight = None
            
        print("\n ** gradient test ** \n")
        self.grad_test(10,config.flag_plot>=1)
        
        
    
    def cost(self,X0) :
        '''
        Compute the 4Dvar cost function for the SSH field var represented by the 
        1D vector X0
        '''
        
        # initial state
        State = self.State
        
        if self.prec :
            X = self.B.sqr(X0) + self.Xb
            X_var = X.reshape((State.ny,State.nx))
        else :
            X_var = X0.reshape((State.ny,State.nx))
        State.setvar(X_var,0) # initiate the State with X0
        
        X0[np.isnan(X0)] = 0
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                Jb = X0.dot(X0) # cost of background term with change of variable
            else:
                dx = X0-self.Xb
                Jb = np.dot(dx,self.B.inv(dx))  # cost of background term
                if self.grad_term :
                    # Jgrad represent a condition on the gradient of X0
                    Jgrad = np.dot(dx,self.gradop.T_grad(self.B_grad.inv(self.gradop.grad(dx))))
                    Jb += Jgrad
        else:
            Jb = 0.
        
        # Observational cost function evaluation
        Jo = 0.
        State.save(os.path.join(self.tmp_DA_path,
                    'model_state_' + str(self.checkpoint[0]) + '.nc'))
        
        for i in range(len(self.checkpoint)-1):
            
            # time corresponding to the checkpoint
            timestamp = self.timestamps[i]
            
            # Misfit
            if self.isobs[i] :
                misfit = self.H.misfit(timestamp,State,square=False) # d=Hx-xobs                
                Jo += misfit.dot(self.R.inv(misfit))
            
            # Run forward model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step(State, nstep=nstep,
                        Hbc=self.bc_field[i], Wbc=self.bc_weight)
            
            # Save state for adj computation 
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i+1]) + '.nc'))
        
        if self.isobs[-1]:
            misfit = self.H.misfit(self.timestamps[-1],State,square=False) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
        return J
    
    def grad(self,X0) :
        
        if self.B is not None:
            if self.prec :
                gb = X0 # gradient of the background term
            else:
                dx = X0 - self.Xb
                gb = self.B.inv(dx) # gradient of background term
                if self.grad_term :
                    g_grad = self.gradop.T_grad(self.B_grad.inv(self.gradop.grad(dx)))
                    gb += g_grad
        else:
            gb = 0
        
        # Ajoint initialization   
        adState = self.adState
        adState.setvar(np.zeros((adState.ny,adState.nx)),0) # initiate the State with X0
        
        # Current trajectory
        State = self.State
        
        
        # Last timestamp
        if self.isobs[-1]:
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[-1]) + '.nc'))
            misfit = self.H.misfit(self.timestamps[-1],State,square=True) # d=Hx-yobs
            self.H.adj(self.timestamps[-1],adState,self.R.inv(misfit))
            
        # Time loop
        for i in reversed(range(0,len(self.checkpoint)-1)):
                
            timestamp = self.timestamps[i]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i]) + '.nc'))
            
            # Run adjoint model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step_adj(adState, State, nstep=nstep, 
                            Hbc=self.bc_field[i], Wbc=self.bc_weight)
            
            # Misfit 
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State,square=True) # d=Hx-yobs
                self.H.adj(timestamp,adState,self.R.inv(misfit))
        adX = adState.getvar(0).ravel()
        if self.prec :
            # express the gradient of the cost function related to the preconditionned variable
            # from the one related to the state variable
            adX = self.B.prec_filter(adX,State)
        
        g = adX + gb  # total gradient
            
        return g
    
    def grad_test(self,deg=5,plot=True) :
        '''
        performs a gradient test
         - deg : degree of precision of the test
        '''
        n = len(self.State.getvar(0).ravel())
        X = np.random.random(n)
        dX = np.ones(n)
        Jx = self.cost(X) # cost in X
        g = self.grad(X) # grad of cost in X
        L_result = [[],[]]
        for i in range(deg) :
            Jxdx = self.cost(X+dX)
            test = abs(1 - np.dot(g,dX)/(Jxdx-Jx))
            print(f'{10**(-i):.1E} , {test:.1E}')
            dX = 0.1*dX
            L_result[0].append(10**-i)
            L_result[1].append(test)
        if plot :
            plot_grad_test(L_result)
        
        
class Variational_SW:
    
    def __init__(self, config=None, M=None, H=None, State=None, R=None,B=None, basis=None, Xb=None):
        
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
        self.tmp_DA_path = config.tmp_DA_path
        
        # Compute checkpoints
        self.checkpoint = [0]
        if H.isobserved(M.timestamps[0]):
            self.isobs = [True]
        else:
            self.isobs = [False]
        check = 0
        for i,t in enumerate(M.timestamps[:-1]):
            if i>0 and (H.isobserved(t) or check==config.checkpoint):
                self.checkpoint.append(i)
                if check==config.checkpoint:
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
        
        # Reduced basis components
        self.basis = basis
        self.dt_basis = config.checkpoint # Number of timesteps between two successive reduced-basis components update
        self.indt_basis = np.arange(0,self.checkpoint[-1],config.checkpoint) 
        
        # preconditioning
        self.prec = config.prec

        # Grad test
        if config.compute_test:
            X = np.random.random((self.basis.nbasis))-0.5
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
                    'model_state_' + str(self.checkpoint[0]) + '.nc'))
        
        params = self.basis.operg(X,0)
        for i in range(len(self.checkpoint)-1):
            
            t = self.M.T[self.checkpoint[i]]
            timestamp = self.M.timestamps[self.checkpoint[i]]
            
            # Misfit
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State,square=False) # d=Hx-xobs   
                Jo += misfit.dot(self.R.inv(misfit))
            
            # Reduced basis
            if i>0 and self.checkpoint[i] in self.indt_basis:
                params = self.basis.operg(X,int(self.checkpoint[i]//self.dt_basis))
            
            # Run forward model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step(t,State,params,nstep=nstep)
                
            # Save state for adj computation 
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i+1]) + '.nc'))
            
        if self.isobs[-1]:
            misfit = self.H.misfit(self.M.timestamps[self.checkpoint[-1]],State,square=False) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)

        
        State.plot()
        
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
        State = self.State.free()
        
        # Ajoint initialization   
        adState = self.State.free()
        adX = X0*0
        adparams = np.zeros(self.basis.nphys)
        
        # Last timestamp
        if self.isobs[-1]:
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[-1]) + '.nc'))
            timestamp = self.M.timestamps[self.checkpoint[-1]]
            misfit = self.H.misfit(timestamp,State,square=True) # d=Hx-yobs
            
            self.H.adj(timestamp,adState,self.R.inv(misfit))
            
        # Time loop
        self.M.restart()  
        params = self.basis.operg(X,-1)
        
        for i in reversed(range(0,len(self.checkpoint)-1)):
            timestamp = self.M.timestamps[self.checkpoint[i]]
            t = self.M.T[self.checkpoint[i]]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i]) + '.nc'))
            
            # Run adjoint model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            adparams += self.M.step_adj(t, adState, State, params, nstep=nstep)
            
            # Reduced basis
            if self.checkpoint[i] in self.indt_basis:
                adX += self.basis.operg_transpose(adparams,int(self.checkpoint[i]//self.dt_basis))
                params = self.basis.operg(X,int(self.checkpoint[i]//self.dt_basis)-1)
                adparams = np.zeros(self.basis.nphys)

            # Misfit 
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State,square=True) # d=Hx-yobs
                self.H.adj(timestamp,adState,self.R.inv(misfit))
        
        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient
        
        adState.plot()
        
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


