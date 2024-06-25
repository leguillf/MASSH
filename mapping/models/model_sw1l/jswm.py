import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
from jax import debug
#from jax.config import config
#config.update("jax_enable_x64", True)
from jax.lax import scan
from jax import vmap

import matplotlib.pylab as plt
import numpy as np

from functools import partial

class Swm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,Model,State):
        
        ###############
        # COORDINATES #
        ###############
        
        self.X = State.X # X coordinates
        self.Y = State.Y # Y coordinates
        self.Xu = self.rho_on_u(self.X) # X coordinates on the u grid 
        self.Yu = self.rho_on_u(self.Y) # Y coordinates on the u grid
        self.Xv = self.rho_on_v(self.X) # X coordinates on the v grid
        self.Yv = self.rho_on_v(self.Y) # Y coordinates on the v grid

        ########
        # DATA #
        ########

        # Bathymetry gradient # 

        self.grad_bathymetry_x = State.grad_bathymetry_x
        self.grad_bathymetry_y = State.grad_bathymetry_y

        # Tidal Velocity # 

        self.tidal_U = State.tidal_U
        self.tidal_V = State.tidal_U

        ##############
        # PARAMETERS #
        ##############
        
        self.g = Model.g
        self.dt = Model.dt 
        self.omegas = Model.omegas
        self.omega_names = Model.omega_names

        self.ny,self.nx = self.X.shape
 
        self.nu = self.Xu.size
        self.nv = self.Xv.size
        self.nh = self.X.size

        self.nHe = self.nh
        self.nBc = 2*(self.ny + self.nx)
        self.nparams = Model.nparams # Number of parameters of the model 

        if "hbcx" in Model.name_params and "hbcy" in Model.name_params : 
            self.bc_theta = Model.bc_theta

        if hasattr(Model.f, "__len__") and Model.f.shape==self.X.shape:
            self.f = Model.f
        else: 
            self.f = Model.f * jnp.ones_like(self.X)
        
        if hasattr(Model.Heb, "__len__") and Model.f.shape==self.X.shape:
            self.Heb = Model.Heb
        else: 
            self.Heb = Model.Heb * jnp.ones_like(self.X)
        
        # - Internal Tide Generation - # 
        self.anisotropic_itg = Model.anisotropic_itg # if True, expression of ITG is anisotropic 

        #########################
        # FUNCTIONAL PARAMETERS #
        ######################### 

        self.bc_kind = Model.bc_kind
        self.time_scheme = Model.time_scheme 
        self.bc_island = Model.bc_island # Boundary condition type on the island coast 

        #############
        # VARIABLES # 
        #############
        
        # -- Coastal indexes #
        self.idxcoast = Model.idxcoast 
        
        # -- Coastal auxiliary variables #
        interval = 0
        for slicevar,size in zip(["sliceu","slicev","sliceh",
                        "slicevN","slicehN",
                        "slicevS","slicehS",
                        "sliceuW","slicehW",
                        "sliceuE","slicehE"],
                        [self.nu,self.nv,self.nh,
                        self.idxcoast["vN"][0].size,self.idxcoast["hN"][0].size,
                        self.idxcoast["vS"][0].size,self.idxcoast["hS"][0].size,
                        self.idxcoast["uW"][0].size,self.idxcoast["hW"][0].size,
                        self.idxcoast["uE"][0].size,self.idxcoast["hE"][0].size]):
            setattr(self,slicevar,slice(interval,interval+size))
            interval+=size

        self.nstates = self.nu + self.nv + self.nh + \
                       self.idxcoast["vN"][0].size + self.idxcoast["hN"][0].size + \
                       self.idxcoast["vS"][0].size + self.idxcoast["hS"][0].size + \
                       self.idxcoast["uW"][0].size + self.idxcoast["hW"][0].size + \
                       self.idxcoast["uE"][0].size + self.idxcoast["hE"][0].size

        self.name_var = Model.name_var 
        self.name_params = Model.name_params # list of all parameters name
        self.slice_params = Model.slice_params # dictionary containing slices of all parameters 
        self.shape_params = Model.shape_params # dictionary containing shapes of all parameters
        
        self.shapeu = self.Xu.shape
        self.shapev = self.Xv.shape
        self.shapeh = self.X.shape

        self.mask = Model.mask # mask for integration 
        
        #################
        # JAX FUNCTIONS #
        #################
         
        self.u_on_v_jit = jit(self.u_on_v)
        self.v_on_u_jit = jit(self.v_on_u)
        self.rhs_u_jit = jit(self.rhs_u)
        self.rhs_v_jit = jit(self.rhs_v)
        self.rhs_h_jit = jit(self.rhs_h)
        self.obcs_jit = jit(self.obcs)
        self._compute_w1_IT_jit = jit(self._compute_w1_IT)

        self.coastbcs_jit = jit(self.coastbcs)
        self.coastN_jit = jit(self.coastN,static_argnums=[0,1])
        self.coastS_jit = jit(self.coastS,static_argnums=[0,1])
        self.coastW_jit = jit(self.coastW,static_argnums=[0,1])
        self.coastE_jit = jit(self.coastE,static_argnums=[0,1])

        self.euler_jit = jit(self.euler)
        self.rk4_jit = jit(self.rk4)
        self.step_jit = jit(self.step, static_argnums=1)
        self.one_step_jit = jit(self.one_step)
        self.one_step_for_scan_jit = jit(self.one_step_for_scan)
        self.step_tgl_jit = jit(self.step_tgl, static_argnums=2)
        self.step_adj_jit = jit(self.step_adj, static_argnums=2)
        
    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
    
    def rho_on_u(self,rho):
        """
        DESCRIPTION 
        Returns the expression of density rho on the u grid. 
        """
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def rho_on_v(self,rho):
        """
        DESCRIPTION 
        Returns the expression of density rho on the v grid. 
        """
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def u_on_v(self,u):
        """
        DESCRIPTION 
        Returns the expression of velocity u on the v grid. 
        """
        #um = 0.25 * (u[2:-1,:-1] + u[2:-1,1:] + u[1:-2,:-1] + u[1:-2,1:])

        # -- 1st trial -- #

        #u1 = jnp.expand_dims(u[2:-1,:-1],axis =2)
        #u2 = jnp.expand_dims(u[2:-1,1:],axis =2)
        #u3 = jnp.expand_dims(u[1:-2,:-1],axis =2)
        #u4 = jnp.expand_dims(u[1:-2,1:],axis =2)
        
        #um = jnp.nanmean(jnp.concatenate((u1,u2,u3,u4),axis=2),axis=2)

        # -- 2nd trial -- #

        um = jnp.mean( jnp.array([ u[2:-1,:-1], u[2:-1,1:], u[1:-2,:-1], u[1:-2,1:]]), axis=0 )
        
        return um
    
    def v_on_u(self,v):
        """
        DESCRIPTION 
        Returns the expression of velocity v on the u grid. 
        """
        #vm = 0.25 * (v[:-1,2:-1] + v[:-1,1:-2] + v[1:,2:-1] + v[1:,1:-2])

        # -- 1st trial -- #

        #v1 = jnp.expand_dims(v[:-1,2:-1],axis =2)
        #v2 = jnp.expand_dims(v[:-1,1:-2],axis =2)
        #v3 = jnp.expand_dims(v[1:,2:-1],axis =2)
        #v4 = jnp.expand_dims(v[1:,1:-2],axis =2)
        
        #vm = jnp.nanmean(jnp.concatenate((v1,v2,v3,v4),axis=2),axis=2)
        
        # -- 2nd trial -- #

        vm = jnp.mean( jnp.array([ v[:-1,2:-1], v[:-1,1:-2], v[1:,2:-1], v[1:,1:-2]]), axis=0 )

        return vm
    
    ###########################################################################
    #                          Right hand sides                               #
    ###########################################################################
    
    def rhs_u(self,vm,h,hW,hE):
        
        rhs_u = jnp.zeros(self.Xu.shape)

        rhs_u = rhs_u.at[1:-1,1:-1].set((self.f[1:-1,2:-1]+self.f[1:-1,1:-2])/2 * vm -\
            self.g * (h[1:-1,2:-1] - h[1:-1,1:-2]) / ((self.X[1:-1,2:-1]-self.X[1:-1,1:-2])))

        ### Test with coast ##
        '''
        h_right = h.copy() # right side of SSH 
        h_left = h.copy() # left side of SSH
        h_right = h_right.at[self.idxcoast["hE"]].set(hE) 
        h_left = h_left.at[self.idxcoast["hW"]].set(hW)
        
        rhs_u = rhs_u.at[1:-1,1:-1].set((self.f[1:-1,2:-1]+self.f[1:-1,1:-2])/2 * vm -\
            self.g * (h_right[1:-1,2:-1] - h_left[1:-1,1:-2]) / ((self.X[1:-1,2:-1]-self.X[1:-1,1:-2])))
        '''

        return rhs_u
        

    def rhs_v(self,um,h,hN,hS):
        
        rhs_v = jnp.zeros_like(self.Xv)
        
        rhs_v = rhs_v.at[1:-1,1:-1].set(-(self.f[2:-1,1:-1]+self.f[1:-2,1:-1])/2 * um -\
            self.g * (h[2:-1,1:-1] - h[1:-2,1:-1]) / ((self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1])))

        ### Test with coast ##
        '''
        h_up = h.copy() # upper side of SSH 
        h_down = h.copy() # down side of SSH 
        h_up = h_up.at[self.idxcoast["hN"]].set(hN)
        h_down = h_down.at[self.idxcoast["hS"]].set(hS)

        rhs_v = rhs_v.at[1:-1,1:-1].set(-(self.f[2:-1,1:-1]+self.f[1:-2,1:-1])/2 * um -\
            self.g * (h_up[2:-1,1:-1] - h_down[1:-2,1:-1]) / ((self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1])))
        '''

        return rhs_v
        
    
    def rhs_h(self,u,v,He,uE,uW,vN,vS,rhs_itg):
        rhs_h = jnp.zeros_like(self.X)

        rhs_h = rhs_h.at[1:-1,1:-1].set(- He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) + \
                (v[1:,1:-1] - v[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))+ \
                rhs_itg[1:-1,1:-1])
        
        ### Test with coast ### 
        '''
        u_right = u.copy() # right side of u 
        u_left = u.copy() # left side of u
        v_up = v.copy() # upper side of v
        v_down = v.copy() # dow side of v
        
        u_right = u_right.at[self.idxcoast["uE"]].set(uE)
        u_left = u_left.at[self.idxcoast["uW"]].set(uW)
        v_up = v_up.at[self.idxcoast["vN"]].set(vN)
        v_down = v_down.at[self.idxcoast["vS"]].set(vS)

        rhs_h = rhs_h.at[1:-1,1:-1].set(- He[1:-1,1:-1] * (\
                (u_right[1:-1,1:] - u_left[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) + \
                (v_up[1:,1:-1] - v_down[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))+ \
               rhs_itg[1:-1,1:-1])
        '''
        
        return rhs_h
        
    
    
    ###########################################################################
    #                      Open Boundary Conditions                           #
    ###########################################################################

    def _compute_w1_IT(self,t,He,h_SN,h_WE):
        """
        Compute first characteristic variable w1 for internal tides from external 
        data

        Parameters
        ----------
        t : float 
            time in seconds
        He : 2D array
        h_SN : ND array
            amplitude of SSH for southern/northern borders
        h_WE : ND array
            amplitude of SSH for western/eastern borders

        Returns
        -------
        w1ext: 1D array
            flattened  first characteristic variable (South/North/West/East)
        """
        
        # South
        HeS = (He[0,:]+He[1,:])/2
        fS = (self.f[0,:]+self.f[1,:])/2
        w1S = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fS**2)/(self.g*HeS))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = jnp.cos(theta) * k
                kxy = kx*self.Xv[0,:] + ky*self.Yv[0,:]
                
                h = h_SN[j,0,0,i]* jnp.cos(w*t-kxy)  +\
                        h_SN[j,0,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fS**2)*( \
                    h_SN[j,0,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fS*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,0,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fS*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                
                w1S += v + jnp.sqrt(self.g/HeS) * h
        
        # North
        fN = (self.f[-1,:]+self.f[-2,:])/2
        HeN = (He[-1,:]+He[-2,:])/2
        w1N = jnp.zeros(self.nx)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fN**2)/(self.g*HeN))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.sin(theta) * k
                ky = -jnp.cos(theta) * k
                kxy = kx*self.Xv[-1,:] + ky*self.Yv[-1,:]
                h = h_SN[j,1,0,i]* jnp.cos(w*t-kxy)+\
                        h_SN[j,1,1,i]* jnp.sin(w*t-kxy) 
                v = self.g/(w**2-fN**2)*(\
                    h_SN[j,1,0,i]* (w*ky*jnp.cos(w*t-kxy) \
                                - fN*kx*jnp.sin(w*t-kxy)
                                    ) +\
                    h_SN[j,1,1,i]* (w*ky*jnp.sin(w*t-kxy) \
                                + fN*kx*jnp.cos(w*t-kxy)
                                    )
                        )
                w1N += v - jnp.sqrt(self.g/HeN) * h

        # West
        fW = (self.f[:,0]+self.f[:,1])/2
        HeW = (He[:,0]+He[:,1])/2
        w1W = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fW**2)/(self.g*HeW))
            for i,theta in enumerate(self.bc_theta):
                kx = jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.Xu[:,0] + ky*self.Yu[:,0]
                h = h_WE[j,0,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,0,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fW**2)*(\
                    h_WE[j,0,0,i]*(w*kx*jnp.cos(w*t-kxy) \
                              + fW*ky*jnp.sin(w*t-kxy)
                                  ) +\
                    h_WE[j,0,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fW*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1W += u + jnp.sqrt(self.g/HeW) * h

        
        # East
        HeE = (He[:,-1]+He[:,-2])/2
        fE = (self.f[:,-1]+self.f[:,-2])/2
        w1E = jnp.zeros(self.ny)
        for j,w in enumerate(self.omegas):
            k = jnp.sqrt((w**2-fE**2)/(self.g*HeE))
            for i,theta in enumerate(self.bc_theta):
                kx = -jnp.cos(theta)* k
                ky = jnp.sin(theta)* k
                kxy = kx*self.Xu[:,-1] + ky*self.Yu[:,-1]
                h = h_WE[j,1,0,i]*jnp.cos(w*t-kxy) +\
                        h_WE[j,1,1,i]*jnp.sin(w*t-kxy)
                u = self.g/(w**2-fE**2)*(\
                    h_WE[j,1,0,i]* (w*kx*jnp.cos(w*t-kxy) \
                                + fE*ky*jnp.sin(w*t-kxy)
                                    ) +\
                    h_WE[j,1,1,i]*(w*kx*jnp.sin(w*t-kxy) \
                              - fE*ky*jnp.cos(w*t-kxy)
                                  )
                        )
                w1E += u - jnp.sqrt(self.g/HeE) * h
        
        return w1S,w1N,w1W,w1E     

    def obcs(self,u,v,h,u0,v0,h0,He,w1ext):
        
        g = self.g
                
        #######################################################################
        # South
        #######################################################################
        HeS = (He[0,:]+He[1,:])/2
        cS = jnp.sqrt(g*HeS)
        if self.bc_kind=='1d':
            cS *= self.dt/(self.Y[1,:]-self.Y[0,:])

        # 1. w1
        w1extS = +w1ext[0]
        
        if self.bc_kind=='1d':
            w1S = w1extS
        elif self.bc_kind=='2d':
            # dw1dy0
            w10  = v0[0,:] + jnp.sqrt(g/HeS)* (h0[0,:]+h0[1,:])/2
            w10_ = (v0[0,:]+v0[1,:])/2 + jnp.sqrt(g/HeS)* h0[1,:]
            _w10 = w1extS
            dw1dy0 = (w10_ - _w10)/self.dy
            # dudx0
            dudx0 = jnp.zeros(self.nx)
            dudx0[1:-1] = ((u0[0,1:] + u0[1,1:] - u0[0,:-1] - u0[1,:-1])/2)/self.dx
            dudx0[0] = dudx0[1]
            dudx0[-1] = dudx0[-2]
            # w1S
            w1S = w10 - self.dt*cS* (dw1dy0 + dudx0)
        
        # 2. w2
        w20 = (u0[0,:] + u0[1,:])/2
        if self.bc_kind=='1d':
            w2S = w20
        elif self.bc_kind=='2d':
            dhdx0 = ((h0[0,1:]+h0[1,1:]-h0[0,:-1]-h0[1,:-1])/2)/self.dx
            w2S = w20 - self.dt*g* dhdx0 
                
        # 3. w3
        if self.bc_kind=='1d':
            _vS = (1-3/2*cS)* v0[0,:] + cS/2* (4*v0[1,:] - v0[2,:])
            _hS = (1/2+cS)* h0[1,:] + (1/2-cS)* h0[0,:]
            w3S = _vS - jnp.sqrt(g/HeS) * _hS
        elif self.bc_kind=='2d':
            w30   = v0[0,:] - jnp.sqrt(g/HeS)* (h0[0,:]+h0[1,:])/2
            w30_  = (v0[0,:]+v0[1,:])/2  - jnp.sqrt(g/HeS)* h0[1,:]
            w30__ = v0[1,:] - jnp.sqrt(g/HeS)* (h0[1,:]+h0[2,:])/2
            dw3dy0 =  -(3*w30 - 4*w30_ + w30__)/(self.dy/2)
            w3S = w30 + self.dt*cS* (dw3dy0 + dudx0) 

        # 4. Values on BC
        uS = w2S
        vS = (w1S + w3S)/2
        hS = jnp.sqrt(HeS/g) *(w1S - w3S)/2
        
        #######################################################################
        # North
        #######################################################################
        HeN = (He[-1,:]+He[-2,:])/2
        cN = jnp.sqrt(g*HeN)
        if self.bc_kind=='1d':
            cN *= self.dt/(self.Y[-1,:]-self.Y[-2,:])

        # 1. w1
        w1extN = +w1ext[1]
        
        if self.bc_kind=='1d':
            w1N = w1extN
        elif self.bc_kind=='2d':
            w10  = v0[-1,:] - jnp.sqrt(g/HeN)* (h0[-1,:]+h0[-2,:])/2
            w10_ = (v0[-1,:]+v0[-2,:])/2 - jnp.sqrt(g/HeN)* h0[-2,:]
            _w10 = w1extN
            dw1dy0 = (_w10 - w10_)/self.dy
            dudx0 = jnp.zeros(self.nx)
            dudx0[1:-1] = ((u0[-1,1:] + u0[-2,1:] - u0[-1,:-1] - u0[-2,:-1])/2)/self.dx
            dudx0[0] = dudx0[1]
            dudx0[-1] = dudx0[-2]
            w1N = w10 + self.dt*cN* (dw1dy0 + dudx0) 
            
        # 2. w2
        w20 = (u0[-1,:] + u0[-2,:])/2
        if self.bc_kind=='1d':   
            w2N = w20
        elif self.bc_kind=='2d':
            dhdx0 = ((h0[-1,1:]+h0[-2,1:]-h0[-1,:-1]-h0[-2,:-1])/2)/self.dx
            w2N = w20 - self.dt*g*dhdx0 
        # 3. w3
        if self.bc_kind=='1d':   
            _vN = (1-3/2*cN)* v0[-1,:] + cN/2* (4*v0[-2,:] - v0[-3,:])
            _hN = (1/2+cN)* h0[-2,:] + (1/2-cN)* h0[-1,:]
            w3N = _vN + jnp.sqrt(g/HeN) * _hN
        elif self.bc_kind=='2d':
            w30   = v0[-1,:] + jnp.sqrt(g/HeN)* (h0[-1,:]+h0[-2,:])/2
            w30_  = (v0[-1,:]+v0[-2,:])/2 + jnp.sqrt(g/HeN)* h0[-2,:]
            w30__ = v0[-2,:] + jnp.sqrt(g/HeN)* (h0[-2,:]+h0[-3,:])/2
            dw3dy0 =  (3*w30 - 4*w30_ + w30__)/(self.dy/2)
            w3N = w30 - self.dt*cN* (dw3dy0 + dudx0) 
        
        # 4. Values on BC
        uN = w2N
        vN = (w1N + w3N)/2 
        hN = jnp.sqrt(HeN/g) *(w3N - w1N)/2
        
        #######################################################################
        # West
        #######################################################################
        HeW = (He[:,0]+He[:,1])/2
        cW = jnp.sqrt(g*HeW)
        if self.bc_kind=='1d':
            cW *= self.dt/(self.X[:,1]-self.X[:,0])
        
        # 1. w1
        w1extW = +w1ext[2]
        
        if self.bc_kind=='1d':   
            w1W = w1extW
        elif self.bc_kind=='2d':
            w10  = u0[:,0] + jnp.sqrt(g/HeW)* (h0[:,0]+h0[:,1])/2
            w10_ = (u0[:,0]+u0[:,1])/2 + jnp.sqrt(g/HeW)* h0[:,1]
            _w10 = w1extW
            dw1dx0 = (w10_ - _w10)/self.dx
            dvdy0 = jnp.zeros(self.ny)
            dvdy0[1:-1] = ((v0[1:,0] + v0[1:,1] - v0[:-1,0] - v0[:-1,1])/2)/self.dy
            dvdy0[0] = dvdy0[1]
            dvdy0[-1] = dvdy0[-2]
            w1W = w10 - self.dt*cW* (dw1dx0 + dvdy0) 
            
        # 2. w2
        w20 = (v0[:,0] + v0[:,1])/2
        if self.bc_kind=='1d':   
            w2W = w20
        elif self.bc_kind=='2d':
            dhdy0 = ((h0[1:,0]+h0[1:,1]-h0[:-1,0]-h0[:-1,1])/2)/self.dy
            w2W = w20 - self.dt*g * dhdy0 
                
        # 3. w3
        if self.bc_kind=='1d':   
            _uW = (1-3/2*cW)* u0[:,0] + cW/2* (4*u0[:,1]-u0[:,2]) 
            _hW = (1/2+cW)*h0[:,1] + (1/2-cW)*h0[:,0]
            w3W = _uW - jnp.sqrt(g/HeW)* _hW
        elif self.bc_kind=='2d':
            w30   = u0[:,0] - jnp.sqrt(g/HeW)* (h0[:,0]+h0[:,1])/2
            w30_  = (u0[:,0]+u0[:,1])/2 - jnp.sqrt(g/HeW)* h0[:,1]
            w30__ = u0[:,1] - jnp.sqrt(g/HeW)* (h0[:,1]+h0[:,2])/2
            dw3dx0 = -(3*w30 - 4*w30_ + w30__)/(self.dx/2)
            w3W = w30 + self.dt*cW* (dw3dx0 + dvdy0)
            
        # 4. Values on BC
        uW = (w1W + w3W)/2 
        vW = w2W
        hW = jnp.sqrt(HeW/g)*(w1W - w3W)/2
        
        #######################################################################
        # East
        #######################################################################
        HeE = (He[:,-1]+He[:,-2])/2
        cE = jnp.sqrt(g*HeE)
        if self.bc_kind=='1d':
            cE *= self.dt/(self.X[:,-1]-self.X[:,-2])
        
        # 1. w1
        w1extE = +w1ext[3]
        
        if self.bc_kind=='1d':   
            w1E = w1extE
        elif self.bc_kind=='2d':
            w10  = u0[:,-1] - jnp.sqrt(g/HeE)* (h0[:,-1]+h0[:,-2])/2
            w10_ = (u0[:,-1]+u0[:,-2])/2 - jnp.sqrt(g/HeE)* h0[:,-2]
            _w10 = w1extE
            dw1dx0 = (_w10 - w10_)/self.dx
            dvdy0 = jnp.zeros(self.ny)
            dvdy0[1:-1] = ((v0[1:,-1] + v0[1:,-2] - v0[:-1,-1] - v0[:-1,-2])/2)/self.dy
            dvdy0[0] = dvdy0[1]
            dvdy0[-1] = dvdy0[-2]
            w1E = w10 + self.dt*cE* (dw1dx0 + dvdy0) 
        # 2. w2
        w20 = (v0[:,-1] + v0[:,-2])/2
        if  self.bc_kind=='1d':   
            w2E = w20
        elif self.bc_kind=='2d':
            w20 = (v0[:,-1] + v0[:,-2])/2
            dhdy0 = ((h0[1:,-1]+h0[1:,-2]-h0[:-1,-1]-h0[:-1,-2])/2)/self.dy
            w2E = w20 - self.dt*g * dhdy0 
        # 3. w3
        if self.bc_kind=='1d':   
            _uE = (1-3/2*cE)* u0[:,-1] + cE/2* (4*u0[:,-2]-u0[:,-3])
            _hE = ((1/2+cE)*h0[:,-2] + (1/2-cE)*h0[:,-1])
            w3E = _uE + jnp.sqrt(g/HeE)* _hE 
        elif self.bc_kind=='2d':
            w30   = u0[:,-1] + jnp.sqrt(g/HeE)* (h0[:,-1]+h0[:,-2])/2
            w30_  = (u0[:,-1]+u0[:,-2])/2 + jnp.sqrt(g/HeE)* h0[:,-2]
            w30__ = u0[:,-2] + jnp.sqrt(g/HeE)* (h0[:,-2]+h0[:,-3])/2
            dw3dx0 =  (3*w30 - 4*w30_ + w30__)/(self.dx/2)
            w3E = w30 - self.dt*cE* (dw3dx0 + dvdy0) 
            
        # 4. Values on BC
        uE = (w1E + w3E)/2 
        vE = w2E
        hE = jnp.sqrt(HeE/g)*(w3E - w1E)/2
        
        #######################################################################
        # Update border pixels 
        #######################################################################
        # South
        u = u.at[0,1:-1].set(2* uS[1:-1] - u[1,1:-1])
        v = v.at[0,1:-1].set(vS[1:-1])
        h = h.at[0,1:-1].set(2* hS[1:-1] - h[1,1:-1])
        # North
        u = u.at[-1,1:-1].set(2* uN[1:-1] - u[-2,1:-1])
        v = v.at[-1,1:-1].set(vN[1:-1])
        h = h.at[-1,1:-1].set(2* hN[1:-1] - h[-2,1:-1])
        # West
        u = u.at[1:-1,0].set(uW[1:-1])
        v = v.at[1:-1,0].set(2* vW[1:-1] - v[1:-1,1])
        h = h.at[1:-1,0].set(2* hW[1:-1] - h[1:-1,1])
        # East
        u = u.at[1:-1,-1].set(uE[1:-1])
        v = v.at[1:-1,-1].set(2* vE[1:-1] - v[1:-1,-2])
        h = h.at[1:-1,-1].set(2* hE[1:-1] - h[1:-1,-2])
        # South-West
        u = u.at[0,0].set((uS[0] + uW[0])/2)
        v = v.at[0,0].set((vS[0] + vW[0])/2)
        h = h.at[0,0].set((hS[0] + hW[0])/2)
        # South-East
        u = u.at[0,-1].set((uS[-1] + uE[0])/2)
        v = v.at[0,-1].set((vS[-1] + vE[0])/2)
        h = h.at[0,-1].set((hS[-1] + hE[0])/2)
        # North-West
        u = u.at[-1,0].set((uN[0] + uW[-1])/2)
        v = v.at[-1,0].set((vN[0] + vW[-1])/2)
        h = h.at[-1,0].set((hN[0] + hW[-1])/2)
        # North-East
        u = u.at[-1,-1].set((uN[-1] + uE[-1])/2)
        v = v.at[-1,-1].set((vN[-1] + vE[-1])/2)
        h = h.at[-1,-1].set((hN[-1] + hE[-1])/2)

        return u,v,h

    def coastN(self,i_h,j_h,i_v,j_v,vN,hN,h,v0,h0,He):
        
        HeN = He[i_h-1,j_h]
        cN = jnp.sqrt(self.g*HeN)
        if self.bc_kind=='1d':
            cN *= self.dt/(self.Y[i_h,j_h]-self.Y[i_h-1,j_h])

        # 1. w1 
        w1extN = 0 # because no entering wave at the coast

        if self.bc_kind == '1d' : 
            w1N = w1extN 
        elif self.bc_kind == '2d' :
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")

        # 2. w2 
        # w2 isn't calculated because there is no interest if bc_kind=='1d'
        
        # 3. w3 
        if self.bc_kind == '1d' : 
            _vN = (1-3/2*cN)*vN+cN/2*(4*v0[i_v-1,j_v]-v0[i_v-2,j_v])
            _hN = (1/2+cN)*h0[i_h-1,j_h]+(1/2-cN)*hN
            w3N = _vN + jnp.sqrt(self.g/HeN) * _hN
        if self.bc_kind =='2d': 
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")
        
        # 4. Values on BC 
        _vN = (w1N + w3N)/2
        _hN = jnp.sqrt(HeN/self.g) *(w3N - w1N)/2

        # 5. Assigning values 
        vN = _vN
        hN = 2*_hN - h[i_h-1,j_h]

        return vN, hN

    def coastS(self,i_h,j_h,i_v,j_v,vS,hS,h,v0,h0,He):

        HeS = He[i_h+1,j_h] # = equivalent height of the upper pixel 
        cS = jnp.sqrt(self.g*HeS)
        if self.bc_kind=='1d':
            cS *= self.dt/(self.Y[i_h+1,j_h]-self.Y[i_h,j_h])

        # 1. w1 
        w1extS = 0 # because no entering wave at the coast
        
        if self.bc_kind=='1d': 
            w1S = w1extS 
        elif self.bc_kind=='2d':
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")

        # 2. w2 
        # w2 isn't calculated because there is no interest if bc_kind=='1d'

        # 3. w3 
        if self.bc_kind=='1d': 
            _vS = (1-3/2*cS)*vS + cS/2*(4*v0[i_v+1,j_v]-v0[i_v+2,j_v])
            _hS = (1/2+cS)*h0[i_h+1,j_h]+(1/2-cS)*hS
            w3S = _vS - jnp.sqrt(self.g/HeS) * _hS 
        elif self.bc_kind=='2d':    
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")

        # 4. Values on BC
        _vS = (w1S + w3S)/2 
        _hS = jnp.sqrt(HeS/self.g) *(w1S - w3S)/2

        # 5. Assigning values 
        vS = _vS
        hS = 2*_hS - h[i_h+1,j_h]

        return vS, hS
    
    def coastW(self,i_h,j_h,i_u,j_u,uW,hW,h,u0,h0,He):

        HeW = He[i_h,j_h+1]
        cW = jnp.sqrt(self.g*HeW)
        if self.bc_kind == '1d' : 
            cW *= self.dt/(self.X[i_h,j_h+1]-self.X[i_h,j_h])
        
        # 1. w1 
        w1extW = 0 # because no entering wave at the coast

        if self.bc_kind == '1d' : 
            w1W = w1extW 
        elif self.bc_kind == '2d' : 
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")

        # 2. w2 
        # w2 isn't calculated because there is no interest if bc_kind=='1d'
        
        # 3. w3 
        if self.bc_kind == '1d' : 
            _uW = (1-3/2*cW)*uW+cW/2*(4*u0[i_u,j_u+1]-u0[i_u,j_u+2])
            _hW = (1/2+cW)*h0[i_h,j_h+1] + (1/2-cW)*hW
            w3W = _uW - jnp.sqrt(self.g/HeW)* _hW 
        elif self.bc_kind == '2d' : 
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")
        
        # 4. Values on BC 
        _uW = (w1W + w3W)/2
        _hW = jnp.sqrt(HeW/self.g)*(w1W-w3W)/2

        # 5. Assigning values
        uW = _uW
        hW = 2*_hW - h[i_h,j_h+1]

        return uW, hW
    
    def coastE(self,i_h,j_h,i_u,j_u,uE,hE,h,u0,h0,He):

        HeE = He[i_h,j_h-1]
        cE = jnp.sqrt(self.g*HeE)
        if self.bc_kind == '1d' :
            cE *= self.dt/(self.X[i_h,j_h]-self.X[i_h,j_h-1])

        # 1. w1
        w1extE = 0 # because no entering wave at the coast

        if self.bc_kind == '1d' : 
            w1E = w1extE 
        elif self.bc_kind == '2d' : 
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")
        
        # 2. w2 
        # w2 isn't calculated because there is no interest if bc_kind=='1d'
        
        # 3. w3 
        if self.bc_kind == '1d' : 
            _uE = (1-3/2*cE)*uE + cE/2* (4*u0[i_u,j_u-1]-u0[i_u,j_u-2])
            _hE = ((1/2 + cE)*h0[i_h,j_h-1]+(1/2 - cE)*hE)
            w3E = _uE +jnp.sqrt(self.g/HeE)* _hE
        elif self.bc_kind == '2d' : 
            raise Exception("bc_kind==\'2d\' not implemented yet. Please change config.")

        # 4. Values on BC
        _uE = (w1E + w3E)/2 
        _hE = jnp.sqrt(HeE/self.g)*(w3E - w1E)/2

        # 5. Assigning values
        uE = _uE
        hE = 2*_hE - h[i_h,j_h-1]

        return uE, hE
    
    def coastbcs(self,h,u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,He):
        _vN, _hN, _vS, _hS, _uW, _hW, _uE, _hE = np.empty((8,0), dtype='float64')
        # -- NORTH -- #
        if np.any(self.idxcoast["hN"][0])==True : 
            _vN, _hN = jnp.array([self.coastN_jit(i_h,j_h,i_v,j_v,vN,hN,h,v0,h0,He) for i_h,j_h,i_v,j_v,vN,hN in zip(self.idxcoast["hN"][0],self.idxcoast["hN"][1],
                                                                                                                     self.idxcoast["vN"][0],self.idxcoast["vN"][1],
                                                                                                                     vN,hN)]).T
        # -- SOUTH -- #
        if np.any(self.idxcoast["hS"][0])==True :
            _vS, _hS = jnp.array([self.coastS_jit(i_h,j_h,i_v,j_v,vS,hS,h,v0,h0,He) for i_h,j_h,i_v,j_v,vS,hS in zip(self.idxcoast["hS"][0],self.idxcoast["hS"][1],
                                                                                                                     self.idxcoast["vS"][0],self.idxcoast["vS"][1],
                                                                                                                     vS,hS)]).T
        # -- WEST -- #
        if np.any(self.idxcoast["hW"][0])==True :
            _uW, _hW = jnp.array([self.coastW_jit(i_h,j_h,i_u,j_u,uW,hW,h,u0,h0,He) for i_h,j_h,i_u,j_u,uW,hW in zip(self.idxcoast["hW"][0],self.idxcoast["hW"][1],
                                                                                                                     self.idxcoast["uW"][0],self.idxcoast["uW"][1],
                                                                                                                     uW,hW)]).T
        # -- EAST -- #
        if np.any(self.idxcoast["hE"][0])==True :
            _uE, _hE = jnp.array([self.coastE_jit(i_h,j_h,i_u,j_u,uE,hE,h,u0,h0,He) for i_h,j_h,i_u,j_u,uE,hE in zip(self.idxcoast["hE"][0],self.idxcoast["hE"][1],
                                                                                                                     self.idxcoast["uE"][0],self.idxcoast["uE"][1],
                                                                                                                     uE,hE)]).T
        return _vN, _hN, _vS, _hS, _uW, _hW, _uE, _hE

    def one_step(self, X0):
        
        ########################## 
        ###   INITIALIZATION   ###
        ##########################

        t,X1 = X0[0],jnp.asarray(+X0[1:])

        #########################
        # -- State variables -- #
        #########################
        u0 = X1[self.sliceu].reshape(self.shapeu)
        v0 = X1[self.slicev].reshape(self.shapev)
        h0 = X1[self.sliceh].reshape(self.shapeh)

        #############################
        # -- Auxiliary variables -- #
        #############################

        #vN = hN = vS = hS = uW = hW = uE = hE = 0.

        #if np.any(self.idxcoast["hN"]) == True or np.any(self.idxcoast["hS"]) == True \
        #    or np.any(self.idxcoast["hW"]) == True or np.any(self.idxcoast["hE"]) == True :

        vN,hN = X1[self.slicevN],X1[self.slicehN]
        vS,hS = X1[self.slicevS],X1[self.slicehS]
        uW,hW = X1[self.sliceuW],X1[self.slicehW]
        uE,hE = X1[self.sliceuE],X1[self.slicehE]

        ############################
        # -- Control Parameters -- #
        ############################

        params = X1[self.nstates:]

        #print("shape He : ",self.shape_params['He'])
        #print("size param : ",params.size)
        #print("slice He : ",self.slice_params['He'])



        # - Equivalent height - #
        if 'He' in self.name_params:
            He = params[self.slice_params['He']].reshape(self.shape_params['He'])+self.Heb
        else :
            He = self.Heb # value of He by default 

        # - ITG : Internal Tide Generation - # 
        if 'itg' in self.name_params:
            itg = params[self.slice_params['itg']].reshape(self.shape_params['itg']) # parameters for itg forcing 
            rhs_itg = np.zeros_like(self.X) # term on the right hand side of the equation, for itg forcing 
            if not self.anisotropic_itg : 
                for i,_omega in enumerate(self.omegas) : 
                    rhs_itg += itg[i,0,:]*jnp.cos(_omega*jnp.array(t))+ itg[i,1,:]*jnp.sin(_omega*jnp.array(t))
            elif self.anisotropic_itg :
                for (_w_name,(i,_omega)) in zip(self.omega_names,enumerate(self.omegas)) : 
                    rhs_itg+=self.grad_bathymetry_x*self.tidal_U[_w_name]*(itg[i,0,:]*jnp.cos(_omega*jnp.array(t))+itg[i,1,:]*jnp.sin(_omega*jnp.array(t))) # component for x gradient
                    rhs_itg+=self.grad_bathymetry_y*self.tidal_V[_w_name]*(itg[i,2,:]*jnp.cos(_omega*jnp.array(t))+itg[i,3,:]*jnp.sin(_omega*jnp.array(t))) # component for y gradient
        
        else : 
            rhs_itg = jnp.zeros((self.ny, self.nx))

        # - SSH Boundary Condition - # 
        if 'hbcx' in self.name_params and 'hbcy' in self.name_params: 
            hbcx = params[self.slice_params['hbcx']].reshape(self.shape_params['hbcx'])
            hbcy = params[self.slice_params['hbcy']].reshape(self.shape_params['hbcy'])

            if self.bc_kind=='1d':
                tbc = t + self.dt
            else:
                tbc = t

            w1S,w1N,w1W,w1E = self._compute_w1_IT_jit(tbc,He,hbcx,hbcy)

        #######################
        ###   INTEGRATION   ###
        #######################

        # -- Euler -- #
        if self.time_scheme == 'Euler':
            u,v,h = self.euler_jit(u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,rhs_itg,He)

        # -- RK4 -- #
        elif self.time_scheme == 'rk4':
            u,v,h = self.rk4_jit(u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,rhs_itg,He)

        #############################
        ###   BOUNDARY CONDITIONS ###
        #############################

        # -- External boarder -- # 
        # 1. if external boundary conditions are controled 
        if 'hbcx' in self.name_params and 'hbcy' in self.name_params: 
            u,v,h = self.obcs_jit(u,v,h,u0,v0,h0,He,w1ext=(w1S,w1N,w1W,w1E))
        # 2. if external boundary conditions aren't controled, but internal tide generation yes, entering wave in set to zero to enable generated waves exiting the domain 
        elif 'itg' in self.name_params :  
            w1S,w1N,w1W,w1E = jnp.zeros(self.nx),jnp.zeros(self.nx),jnp.zeros(self.ny),jnp.zeros(self.ny)
            u,v,h = self.obcs_jit(u,v,h,u0,v0,h0,He,w1ext=(w1S,w1N,w1W,w1E))

        #debug.print("u {x} : ",x=u)
        #debug.print("v {x} : ",x=v)
        #debug.print("h {x} : ",x=h)

        # -- Coastal values -- # 
        #if np.any(self.idxcoast["hN"]) == True or np.any(self.idxcoast["hS"]) == True \
        #    or np.any(self.idxcoast["hW"]) == True or np.any(self.idxcoast["hE"]) == True :
        if self.bc_island == "radiative" : 
            vN, hN, vS, hS, uW, hW, uE, hE = self.coastbcs_jit(h,u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,He)

        ########################
        ###   OUTPUT ARRAY   ###
        ########################

        _X1 = jnp.concatenate((u.flatten(),v.flatten(),h.flatten(),vN,hN,vS,hS,uW,hW,uE,hE))

        if X1.size==(self.nstates+self.nparams):
            _X1 = jnp.concatenate((_X1,params))
        
        _X1 = jnp.append(jnp.array(t+self.dt),_X1)
        
        return _X1
    
    def one_step_for_scan(self,X0,X1):

        X1 = self.one_step_jit(X0)

        return X1,X1
    
    def step(self, X0, nstep=1):

        # Time propagation
        X1, _ = scan(self.one_step_for_scan_jit, init=X0, xs=jnp.zeros(nstep))
        # for _ in range(nstep):
        #     # One time step
        #    X1 = self.one_step_jit(X0)
        
        return X1
    
    def euler(self,u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,rhs_itg,He):
        
        ########################
        ###   EULER METHOD   ###
        ########################

        # -- RHS calculation -- # 
        ku = self.rhs_u_jit(self.v_on_u_jit(v0),h0,hW,hE)
        kv = self.rhs_v_jit(self.u_on_v_jit(u0),h0,hN,hS)
        kh = self.rhs_h_jit(u0,v0,He,uE,uW,vN,vS,rhs_itg)

        # -- Time Propagation -- # 
        u = u0 + self.dt*ku 
        v = v0 + self.dt*kv
        h = h0 + self.dt*kh

        return u,v,h

    def rk4(self,u0,v0,h0,vN,hN,vS,hS,uW,hW,uE,hE,rhs_itg,He):

        ######################
        ###   RK4 METHOD   ###
        ######################

        # -- k1 -- #
        ku1 = self.rhs_u_jit(self.v_on_u_jit(v0),h0,hW,hE)*self.dt
        kv1 = self.rhs_v_jit(self.u_on_v_jit(u0),h0,hN,hS)*self.dt
        kh1 = self.rhs_h_jit(u0,v0,He,uE,uW,vN,vS,rhs_itg)*self.dt

        # -- k2 -- #
        ku2 = self.rhs_u_jit(self.v_on_u_jit(v0+0.5*kv1),h0+0.5*kh1,hW,hE)*self.dt
        kv2 = self.rhs_v_jit(self.u_on_v_jit(u0+0.5*ku1),h0+0.5*kh1,hN,hS)*self.dt
        kh2 = self.rhs_h_jit(u0+0.5*ku1,v0+0.5*kv1,He,uE,uW,vN,vS,rhs_itg)*self.dt

        # -- k3 -- #
        ku3 = self.rhs_u_jit(self.v_on_u_jit(v0+0.5*kv2),h0+0.5*kh2,hW,hE)*self.dt
        kv3 = self.rhs_v_jit(self.u_on_v_jit(u0+0.5*ku2),h0+0.5*kh2,hN,hS)*self.dt
        kh3 = self.rhs_h_jit(u0+0.5*ku2,v0+0.5*kv2,He,uE,uW,vN,vS,rhs_itg)*self.dt

        # -- k4 -- #
        ku4 = self.rhs_u_jit(self.v_on_u_jit(v0+kv3),h0+kh3,hW,hE)*self.dt
        kv4 = self.rhs_v_jit(self.u_on_v_jit(u0+ku3),h0+kh3,hN,hS)*self.dt
        kh4 = self.rhs_h_jit(u0+ku3,v0+kv3,He,uE,uW,vN,vS,rhs_itg)*self.dt

        # -- Time Propagation -- # 
        u = u0 + 1/6*(ku1+2*ku2+2*ku3+ku4)
        v = v0 + 1/6*(kv1+2*kv2+2*kv3+kv4)
        h = h0 + 1/6*(kh1+2*kh2+2*kh3+kh4)
        
        return u,v,h

    def step_tgl(self, dX0, X0, nstep=1):

        _, dX1 = jvp(partial(self.step_jit, nstep=nstep), (X0,), (dX0,))

        return dX1
    
    def step_adj(self,adX0,X0,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,nstep=nstep), X0)
        
        return adf(adX0)[0]        


