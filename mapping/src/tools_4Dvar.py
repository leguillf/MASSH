#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import os,sys
import xarray as xr 
import numpy as np 
from datetime import timedelta

from dask import delayed,compute

#import matplotlib.pylab as plt 

from . import grid

class Obsopt:

    def __init__(self,config,State,dict_obs,Model):
        
        self.npix = State.lon.size
        self.dict_obs = dict_obs        # observation dictionnary
        self.dt = Model.dt
        self.date_obs = {}
        
        if State.config['name_model']=='SW1L' or State.config['name_model']=='QG1L' :
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
            if not os.path.exists(self.path_H):
                os.makedirs(self.path_H)
        else:
            # We'll use temporary directory to save the files
            self.path_H = None
        
        self.obs_sparse = {}
        
        # For grid interpolation:
        coords_geo = np.column_stack((State.lon.ravel(), State.lat.ravel()))
        self.coords_car = grid.geo2cart(coords_geo)

        
        # Mask coast pixels
        self.dist_coast = config.dist_coast
        if config.mask_coast and self.dist_coast is not None and State.mask is not None and np.any(State.mask):
            self.flag_mask_coast = True
            lon_land = State.lon[State.mask].ravel()
            lat_land = State.lat[State.mask].ravel()
            coords_geo_land = np.column_stack((lon_land,lat_land))
            self.coords_car_land = grid.geo2cart(coords_geo_land)
            
        else: self.flag_mask_coast = False
        
        
        delayed_results = []
        for t in self.date_obs:
            self.process_obs(t)
            #res = delayed(self.process_obs)(t)
            #delayed_results.append(res)
        #compute(*delayed_results, scheduler="processes")
            
    def process_obs(self,t):
        
        if self.path_H is not None:
            file_H = os.path.join(
                self.path_H,'H_'+t.strftime('%Y%m%d_%H%M.nc'))
            if os.path.exists(file_H):
                new_file_H = os.path.join(
                    self.tmp_DA_path,'H_'+t.strftime('%Y%m%d_%H%M.nc'))
                os.system(f"cp {file_H} {new_file_H}")
                return t
        else:
            file_H = os.path.join(
                self.tmp_DA_path,'H_'+t.strftime('%Y%m%d_%H%M.nc'))
        
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
                
                
            elif sat_info.kind=='swot_simulator':
                obs_sparse = True
            with xr.open_dataset(obs_file) as ncin:
                lon = ncin[sat_info.name_obs_lon].values.ravel()
                lat = ncin[sat_info.name_obs_lat].values.ravel()
            if lon.size!=lat.size:
                lon,lat = np.meshgrid(lon,lat)
                lon = lon.ravel()
                lat = lat.ravel()
            lon_obs = np.concatenate((lon_obs,lon))
            lat_obs = np.concatenate((lat_obs,lat))
                                        
    
        if obs_sparse:
            # Compute indexes and weights of neighbour grid pixels
            indexes,weights = self.interpolator(lon_obs,lat_obs)
        else:
            indexes,weights = np.arange(lon_obs.size),np.ones((lon_obs.size,))
            indexes = indexes[:,np.newaxis]
            weights = weights[:,np.newaxis]
        
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
        
        if self.path_H is not None:
                new_file_H = os.path.join(
                    self.tmp_DA_path,'H_'+t.strftime('%Y%m%d_%H%M.nc'))
                os.system(f"cp {file_H} {new_file_H}")
        
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
        for i in range(lon_obs.size):
            _dist = np.sqrt(np.sum(np.square(coords_car_obs[i]-self.coords_car),axis=1))
            # 4 closest
            ind4 = np.argsort(_dist)[:4]
            indexes.append(ind4)
            weights.append(1/_dist[ind4])   
            
        return np.asarray(indexes),np.asarray(weights)
    

    
    def H(self,t,X):
        
        #if self.obs_sparse[t] :
        # Get indexes and weights of neighbour grid pixels
        ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,'H_'+t.strftime('%Y%m%d_%H%M.nc')))
        indexes = ds['indexes'].values
        weights = ds['weights'].values
        maskobs = ds['maskobs'].values
        
        # Compute inerpolation of X to obs space
        HX = np.zeros(indexes.shape[0])
        
        for i,(mask,ind,w) in enumerate(zip(maskobs,indexes,weights)):
            if not mask:
                # Average
                if ind.size>1:
                    HX[i] = np.average(X[ind],weights=w)
                else:
                    HX[i] = X[ind[0]]
            else:
                HX[i] = np.nan
        
        return HX

    def misfit(self,t,State):
        
        # Read obs
        sat_info_list = self.dict_obs[self.date_obs[t]]['satellite']
        obs_file_list = self.dict_obs[self.date_obs[t]]['obs_name']
        
        Yobs = np.array([])        
        for sat_info,obs_file in zip(sat_info_list,obs_file_list):
            with xr.open_dataset(obs_file) as ncin:
                yobs = ncin[sat_info.name_obs_var[0]].values.ravel() # SSH_obs
            Yobs = np.concatenate((Yobs,yobs))
        
        X = State.getvar(State.get_indobs()).ravel() # SSH from state
        
        HX = self.H(t,X)
        res = HX - Yobs
        res[np.isnan(res)] = 0
        
        return res
    
    
    def adj(self,t,adState,misfit):
        
        ind = adState.get_indobs()
        
        #if self.obs_sparse[t]:
        adHssh = np.zeros(self.npix)
        ds = xr.open_dataset(os.path.join(
                self.tmp_DA_path,'H_'+t.strftime('%Y%m%d_%H%M.nc')))
        indexes = ds['indexes'].values
        weights = ds['weights'].values
        maskobs = ds['maskobs'].values
        Nobs,Npix = indexes.shape
        
        for i in range(Nobs):
            if not maskobs[i]:
                # Average
                for j in range(Npix):
                    if weights[i].sum()!=0:
                        adHssh[indexes[i,j]] += weights[i,j]*misfit[i]/(weights[i].sum())
        
        adState.var[ind] += adHssh.reshape(adState.var[ind].shape)

        
            
class Cov :
    # case of a simple diagonal covariance matrix
    def __init__(self,sigma):
        self.sigma = sigma
        
    def inv(self,X):
        return 1/self.sigma**2 * X    
    
    def sqr(self,X):
        return self.sigma**0.5 * X


class Grad_op :
    
    def __init__(self,State) :
        self.shape = State.var[0].shape # shape of the grid
        self.nx = self.shape[0] # number of grid points in the x-direction
        self.ny = self.shape[1] # number of grid points in the y-direction
        self.dx = State.dx # mean spatial step in x-direction
        self.dy = State.dy # mean spatial step in y-direction


    def T_grad(self,g) :
        '''
        transposed operator of grad
        g is a 1D array with a length of 2*nx*ny
        returns a 1D array with a length of nx*ny
        '''
        n_vec = self.nx*self.ny
        gx, gy = g[:n_vec].reshape(self.shape), g[n_vec:].reshape(self.shape)
        Xout = self.T_gradx(gx) + self.T_grady(gy)
        Xout = Xout.ravel()
        return Xout

        self.nx = self.shape[0] # number of grid points in the x-direction
        self.ny = self.shape[1] # number of grid points in the y-direction
        self.dx = State.dx # mean spatial step in x-direction
        self.dy = State.dy # mean spatial step in y-direction
        
    def gradx(self,X) :
        '''
        X is a 2D array of shape (nx,ny)
        the function returns the gradient in the x-direction of X
        '''
        Xsup = np.roll(X,-1,axis=0)
        Xout = self.dx*(Xsup-X)
        Xout[self.nx-1,:] = np.zeros(self.ny) 
        return Xout

    def grady(self,X) :
        '''
        X is a 2D array of shape (nx,ny)
        the function returns the gradient in the y-direction of X
        '''
        Xsup = np.roll(X,-1,axis=1)
        Xout = self.dy*(Xsup-X)
        Xout[:,self.ny-1] = np.zeros(self.nx)
        return Xout
    
    def grad(self,X) :
        '''
        X is a one dimensionnal array with a length of nx*ny
        returns a 1D array of size 2*nx*ny, the gradient of X
        '''
        Xvar = X.reshape(self.shape)
        gx = self.gradx(Xvar)
        gy = self.grady(Xvar)
        Xout = np.concatenate((gx.ravel(),gy.ravel()))
        return Xout
    
    def T_gradx(self,gx) :
        '''
        transposed operator of gradx
        gx is a 2D array of shape (nx,ny)
        returns a 2D array of shape (nx,ny)
        '''
        gx_inf = np.roll(gx,1,axis=0)
        Xout = gx_inf-gx
        Xout[0,:] = -gx[0,:]
        Xout[self.nx-1,:] = gx[self.nx-2,:]
        Xout = self.dx*Xout
        return Xout

    def T_grady(self,gy) :
        '''
        transposed operator of grady
        gy is a 2D array of shape (nx,ny)
        returns a 2D array of shape (nx,ny)
        '''
        gy_inf = np.roll(gy,1,axis=1)
        Xout = gy_inf-gy
        Xout[:,0] = -gy[:,0]
        Xout[:,self.ny-1] = gy[:,self.ny-2]
        Xout = self.dy*Xout
        return Xout
    
    def T_grad(self,g) :
        '''
        transposed operator of grad
        g is a 1D array with a length of 2*nx*ny
        returns a 1D array with a length of nx*ny
        '''
        n_vec = self.nx*self.ny
        gx, gy = g[:n_vec].reshape(self.shape), g[n_vec:].reshape(self.shape)
        Xout = self.T_gradx(gx) + self.T_grady(gy)
        Xout = Xout.ravel()
        return Xout


class Variational_QG :
    
    def __init__(self,M=None, H=None, State=None, R=None,B=None, Xb=None, tmp_DA_path=None,
                 date_ini=None, date_final=None, prec=False, grad_term=False) :
        
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
        
        # Covariance matrix building using gradient condition
        self.grad_term = grad_term
        if self.grad_term :
            self.grad_op = Grad_op(self.State)
            self.B_grad = Cov(self.State.config.sigma_B_grad)
        
        # preconditioning
        self.prec = prec
        
        # initial and final date of the assimilation window
        self.date_ini = date_ini
        self.date_final = date_final
        
        # model time step
        self.dt = timedelta(seconds=State.config.dtmodel)
        
        # checkpoint indicate the iteration where the algorithm stop
        self.checkpoint = [0]
        # isobs has the same length as checkpoint and indicates when obs are available at a checkpoint
        if self.H.isobserved(self.date_ini) :
            self.isobs = [True]
        else :
            self.isobs = [False]
        
        t,i = date_ini+self.dt, 1
        while t < self.date_final :
            
            if self.H.isobserved(t) :
                self.checkpoint.append(i)
                self.isobs.append(True)
            i += 1
            t += self.dt
        
        if self.H.isobserved(self.date_final) :
            self.isobs.append(True)
        else :
            self.isobs.append(False)
            
        self.n_iter = i
        self.checkpoint.append(self.n_iter)
        
        # indicates the corresponding iteration of the timestamps
        self.start_iter = int((self.date_ini - State.config.init_date).total_seconds()/self.dt.total_seconds())
        
        print("\n ** gradient test ** \n")
        self.grad_test(deg=8)
        
        
    
    def cost(self,X0) :
        '''
        Compute the 4Dvar cost function for the SSH field var represented by the 
        1D vector X0
        '''
        print('\n cost use \n')
        
        # initial state
        State = self.State.free() # create new state
        if self.prec :
            X = self.B.sqr(X0) + self.Xb
            X_var = X.reshape(self.State.var.ssh.shape)
        else :
            X_var = X0.reshape(self.State.var.ssh.shape)
        State.setvar(X_var,0) # initiate the State with X0
        
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                Jb = X0.dot(X0) # cost of background term with change of variable
            else:
                dx = X0-self.Xb
                Jb = np.dot(dx,self.B.inv(dx))  # cost of background term
                if self.grad_term :
                    # Jgrad represent a condition on the gradient of X0
                    Jgrad = np.dot(dx,self.grad_op.T_grad(self.B_grad.inv(self.grad_op.grad(dx))))
                    Jb += Jgrad
        else:
            Jb = 0.
        
        # Observational cost function evaluation
        Jo = 0.
        State.save(os.path.join(self.tmp_DA_path,
                    'model_state_' + str(self.checkpoint[0]) + '.nc'))
        
        for i in range(len(self.checkpoint)-1):
            
            # time corresponding to the checkpoint
            timestamp = self.M.timestamps[self.checkpoint[i]+self.start_iter]
            
            # Misfit
            if self.isobs[i] :
                misfit = self.H.misfit(timestamp,State) # d=Hx-xobs                
                Jo += misfit.dot(self.R.inv(misfit))
            
            # Run forward model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step(State,nstep=nstep)
            # Save state for adj computation 
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i+1]) + '.nc'))
        
        if self.isobs[-1]:
            misfit = self.H.misfit(self.M.timestamps[self.checkpoint[-1]],State) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.R.inv(misfit))  
        
        # Cost function 
        J = 1/2 * (Jo + Jb)
        
        print(f'\n cost eval : {J}, Jobs={0.5*Jo}')
        
        return J
    
    def grad(self,X0) :
        
        if self.B is not None:
            if self.prec :
                gb = X0 # gradient of the background term
            else:
                dx = X0 - self.Xb
                gb = self.B.inv(dx) # gradient of background term
                if self.grad_term :
                    g_grad = self.grad_op.T_grad(self.B_grad.inv(self.grad_op.grad(dx)))
                    gb += g_grad
        else:
            gb = 0
        
        # Ajoint initialization   
        adState = self.State.free()
        
        # Current trajectory
        State = self.State.free()
        
        
        # Last timestamp
        if self.isobs[-1]:
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[-1]) + '.nc'))
            misfit = self.H.misfit(self.M.timestamps[self.checkpoint[-1]],State) # d=Hx-yobs
            self.H.adj(self.M.timestamps[self.checkpoint[-1]],adState,self.R.inv(misfit))
            
        # Time loop
        for i in reversed(range(0,len(self.checkpoint)-1)):
            
            timestamp = self.M.timestamps[self.checkpoint[i]+self.start_iter]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoint[i]) + '.nc'))
            
            # Run adjoint model
            nstep = self.checkpoint[i+1] - self.checkpoint[i]
            self.M.step_adj(adState, State, nstep=nstep)
            
            # Misfit 
            if self.isobs[i]:
                misfit = self.H.misfit(timestamp,State) # d=Hx-yobs
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
        print(L_result)
        if plot :
            plot_grad_test(L_result)
        
        
        
        

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
        if False:
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
                    'model_state_' + str(self.checkpoint[0]) + '.nc'))
        
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
                        'model_state_' + str(self.checkpoint[i+1]) + '.nc'))
            

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
            timestamp = self.M.timestamps[self.checkpoint[-1]]
            misfit = self.H.misfit(timestamp,State) # d=Hx-yobs
            
            self.H.adj(timestamp,adState,self.R.inv(misfit))
            
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


