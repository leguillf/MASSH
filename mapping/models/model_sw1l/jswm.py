import numpy as np 

import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
import jax
from jax.lax import scan, dynamic_index_in_dim
from functools import partial
jax.config.update("jax_enable_x64", True)

import matplotlib.pylab as plt

    
class CSWm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,X=None,Y=None,dt=None,time_scheme='rk4',bc_kind='1d',g=9.81,f=1e-4,Heb=0.7,obc_south=True, obc_north=True, obc_west=True, obc_east=True, periodic_x=False, periodic_y=False, **arr_kwargs):
        
        self.X = X
        self.Y = Y
        self.Xu = self.rho_on_u(X)
        self.Yu = self.rho_on_u(Y)
        self.Xv = self.rho_on_v(X)
        self.Yv = self.rho_on_v(Y)
        self.dt = dt
        self.bc_kind = bc_kind
        self.g = g
        if hasattr(f, "__len__") and f.shape==self.X.shape:
            self.f = f
        else: 
            self.f = f * jnp.ones_like(self.X)
        self.f_on_u = self.rho_on_u(self.f)
        self.f_on_v = self.rho_on_v(self.f)

        self.DX = self.X[:,1:]-self.X[:,:-1]
        self.DY = self.Y[1:,:]-self.Y[:-1,:]
        self.DXu = self.Xu[:,1:]-self.Xu[:,:-1]
        self.DYv = self.Yv[1:,:]-self.Yv[:-1,:]
        
        if hasattr(Heb, "__len__") and Heb.shape==self.X.shape:
            self.Heb = Heb
        else: 
            self.Heb = Heb * jnp.ones_like(self.X)
        
        self.ny,self.nx = self.X.shape
                
        self.nu = self.Xu.size
        self.nv = self.Xv.size
        self.nh = self.X.size
        self.nstates = self.nu + self.nv + self.nh
        self.nHe = self.nh
        self.nBc = 2*(self.ny + self.nx)
        self.nparams = self.nHe + self.nBc
        
        self.sliceu = slice(0,
                            self.nu)
        self.slicev = slice(self.nu,
                            self.nu+self.nv)
        self.sliceh = slice(self.nu+self.nv,
                            self.nu+self.nv+self.nh)
        self.sliceHe = slice(self.nu+self.nv+self.nh,
                             self.nu+self.nv+self.nh+self.nHe)
        self.sliceBc = slice(self.nu+self.nv+self.nh+self.nHe,
                             self.nu+self.nv+self.nh+self.nHe+self.nBc)
                             
        
        self.shapeu = self.Xu.shape
        self.shapev = self.Xv.shape
        self.shapeh = self.X.shape
        self.shapeHe = self.X.shape
        
        # Open Boundary Conditions
        self.obc_south = obc_south
        self.obc_north = obc_north
        self.obc_west = obc_west
        self.obc_east = obc_east
        
        # Periodic Boundary Conditions
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        
        # JAX compiling
        self.u_on_v_jit = jit(self.u_on_v)
        self.v_on_u_jit = jit(self.v_on_u)
        self.rhs_u_jit = jit(self.rhs_u)
        self.rhs_v_jit = jit(self.rhs_v)
        self.rhs_h_jit = jit(self.rhs_h)
        self.obcs_jit = jit(self.obcs)
        self.step_euler_jit = jit(self.step_euler)
        self.step_euler_tgl_jit = jit(self.step_euler_tgl)
        self.step_euler_adj_jit = jit(self.step_euler_adj)
        self.step_rk4_jit = jit(self.step_rk4)
        self.step_rk4_tgl_jit = jit(self.step_rk4_tgl)
        self.step_rk4_adj_jit = jit(self.step_rk4_adj)
        self.step_leapfrog_jit = jit(self.step_leapfrog)

        if time_scheme=='rk4':
            self.step = self.step_rk4_jit
            self.step_tgl = self.step_rk4_tgl_jit
            self.step_adj = self.step_rk4_adj_jit
        elif time_scheme=='Euler':
            self.step = self.step_euler_jit
            self.step_tgl = self.step_euler_tgl_jit
            self.step_adj = self.step_euler_adj_jit

    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
    
    def rho_on_u(self,rho):
        
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def rho_on_v(self,rho):
        
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def u_on_rho(self,u):
        
        um = 0.5 * (u[:,1:] + u[:,:-1]) #(ny,nx-2)
        um = jnp.pad(um, ((0,0),(1,1)), mode='edge')  # (ny,nx)
        
        return um
    
    def v_on_rho(self,v):
        
        vm = 0.5 * (v[1:,:] + v[:-1,:]) #(ny-2,nx)
        vm = jnp.pad(vm, ((1,1),(0,0)), mode='edge')  # (ny,nx)
        
        return vm
    
    def u_on_v(self,u):
        
        um = 0.25 * (u[1:,:-1] + u[1:,1:] + u[:-1,:-1] + u[:-1,1:])
        
        return um

    
    def v_on_u(self,v):
        
        vm = 0.25 * (v[:-1,1:] + v[:-1,:-1] + v[1:,1:] + v[1:,:-1])
        
        return vm
    
    def adv(self, up, vp, um, vm, q0):

        """
            3rd-order upwind scheme
        """

        dx = self.Xu[1:-1,1:] - self.Xu[1:-1,:-1] # shape (ny-2, nx-2)
        dy = self.Yv[1:,1:-1] - self.Yv[:-1,1:-1] # shape (ny-2, nx-2)

        _adv =  + up[2:-2,2:-2] * 1 / (6 * dx[1:-1,1:-1]) * (2 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] - 6 * q0[2:-2, 1:-3] + q0[2:-2, :-4]) \
                - um[2:-2,2:-2] * 1 / (6 * dx[1:-1,1:-1]) * (q0[2:-2, 4:] - 6 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] + 2 * q0[2:-2, 1:-3]) \
                + vp[2:-2,2:-2] * 1 / (6 * dy[1:-1,1:-1]) * (2 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] - 6 * q0[1:-3, 2:-2] + q0[:-4, 2:-2]) \
                - vm[2:-2,2:-2] * 1 / (6 * dy[1:-1,1:-1]) * (q0[4:, 2:-2] - 6 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] + 2 * q0[1:-3, 2:-2])
    

        return _adv 
    

    ###########################################################################
    #                          Right hand sides                               #
    ###########################################################################

    def rhs_u(self,u,v,h, u11u=None, v11u=None, u11z=None, v11z=None):
        
        rhs_u = jnp.zeros_like(u)
        
        # --- Pressure gradient + Coriolis ---
        rhs_u = rhs_u.at[1:-1,:].set(
            self.f_on_u[1:-1,:] * self.v_on_u(v) -\
            self.g * (h[1:-1,1:] - h[1:-1,:-1]) / self.DX[1:-1,:]
            )

        # -----------------------------------------
        # Mean-flow advection (u11u, v11u)
        # -----------------------------------------
        if u11u is not None and v11u is not None:
            # Shape u = (ny, nx-1)
            u_on_T = self.u_on_rho(u) # (ny, nx)
            
            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11u < 0, 0, u11u) # (ny, nx)
            um = jnp.where(u11u > 0, 0, u11u)
            vp = jnp.where(v11u < 0, 0, v11u)
            vm = jnp.where(v11u > 0, 0, v11u)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, u_on_T) # shape (ny-4, nx-4)

            # --- interpolate advection term back to u points ---
            adv_term_on_u = self.rho_on_u(adv_term_on_T) # shape (ny-4, nx-5)

            # --- add advection term ---
            rhs_u = rhs_u.at[2:-2,2:-2].set(rhs_u[2:-2,2:-2] - adv_term_on_u) 
        
        # -----------------------------------------
        # Mean-flow vertical shear (u11z, v11z)
        # -----------------------------------------
        if u11z is not None and v11z is not None:
            div = (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                  (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]
            shear_term_on_rho = u11z[1:-1,1:-1] * div # shape (ny-2, nx-2)
            shear_term_on_u = self.rho_on_u(shear_term_on_rho) # shape (ny-2, nx-3)
            rhs_u = rhs_u.at[1:-1,1:-1].set(rhs_u[1:-1,1:-1] + shear_term_on_u)
            
        return rhs_u

    def rhs_v(self,u,v,h, u11u=None, v11u=None, u11z=None, v11z=None):
        
        rhs_v = jnp.zeros_like(v)
        
        # --- Pressure gradient + Coriolis ---
        rhs_v = rhs_v.at[:,1:-1].set(
            -self.f_on_v[:,1:-1] * self.u_on_v(u) -\
            self.g * (h[1:,1:-1] - h[:-1,1:-1]) / self.DY[:,1:-1]
            )

        # -----------------------------------------
        # Mean-flow advection (u11u, v11u)
        # -----------------------------------------
        if u11u is not None and v11u is not None:

            # Shape v = (ny-1, nx)
            v_on_T = self.v_on_rho(v) # (ny, nx)
            
            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11u < 0, 0, u11u) # (ny, nx) 
            um = jnp.where(u11u > 0, 0, u11u)
            vp = jnp.where(v11u < 0, 0, v11u)
            vm = jnp.where(v11u > 0, 0, v11u)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, v_on_T) # shape (ny-4, nx-4)

            # --- interpolate advection term back to u points ---
            adv_term_on_v = self.rho_on_v(adv_term_on_T) # shape (ny-5, nx-4)

            # --- add advection term ---
            rhs_v = rhs_v.at[2:-2,2:-2].set(rhs_v[2:-2,2:-2] - adv_term_on_v) 
        
        # -----------------------------------------
        # Mean-flow vertical shear (u11z, v11z)
        # -----------------------------------------
        if u11z is not None and v11z is not None:
            div = (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]
            shear_term_on_T = v11z[1:-1,1:-1] * div # shape (ny-2, nx-2)
            shear_term_on_v = self.rho_on_v(shear_term_on_T) # shape (ny-3, nx-2)
            rhs_v = rhs_v.at[1:-1,1:-1].set(rhs_v[1:-1,1:-1] + shear_term_on_v) 
            
        return rhs_v
    
    def rhs_h(self,u,v,h, He, u11p=None, v11p=None):

        rhs_h = jnp.zeros_like(h)

        # --- Continuity equation ---
        rhs_h = rhs_h.at[1:-1,1:-1].set(- He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]))

        # -----------------------------------------
        # Mean-flow divergence (u11p, v11p)
        # -----------------------------------------
        if u11p is not None and v11p is not None:

            # --- interpolate mean flow to h points ---
            u11p_on_h = u11p
            v11p_on_h = v11p

            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11p_on_h < 0, 0, u11p_on_h)
            um = jnp.where(u11p_on_h > 0, 0, u11p_on_h)
            vp = jnp.where(v11p_on_h < 0, 0, v11p_on_h)
            vm = jnp.where(v11p_on_h > 0, 0, v11p_on_h)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, h) # shape (ny-4, nx-4)

            # --- add advection term ---
            rhs_h = rhs_h.at[2:-2,2:-2].set(rhs_h[2:-2,2:-2] - adv_term_on_T)
          
        return rhs_h
    
    
    ###########################################################################
    #                      Open Boundary Conditions                           #
    ###########################################################################
    
    def obcs(self,u,v,h,u0,v0,h0,He,w1ext):
        
        g = self.g
                
        #######################################################################
        # South
        #######################################################################
        if self.obc_south:
            HeS = (He[0,:]+He[1,:])/2
            cS = jnp.sqrt(g*HeS)
            if self.bc_kind=='1d':
                cS *= self.dt/(self.Y[1,:]-self.Y[0,:])
            fS = (self.f[0,:] + self.f[1,:])/2
            aS = jnp.sqrt(HeS/g)
        
            # 1. w1
            w1extS = +w1ext[0]
            
            if self.bc_kind=='1d':
                w1S = w1extS
            elif self.bc_kind=='2d':
                # dw1dy0
                w10  = v0[0,:] + (1/aS)* (h0[0,:]+h0[1,:])/2
                w10_ = (v0[0,:]+v0[1,:])/2 + (1/aS)* h0[1,:]
                _w10 = w1extS
                dw1dy0 = (w10_ - _w10)/(self.Y[1,:]-self.Y[0,:])
                # dudx0
                dudx0 = jnp.zeros(self.nx)
                dudx0 = dudx0.at[1:-1].set(((u0[0,1:] + u0[1,1:] - u0[0,:-1] - u0[1,:-1])/2)/(self.Xu[0,1:]-self.Xu[0,:-1]))
                dudx0 = dudx0.at[0].set(dudx0[1])
                dudx0 = dudx0.at[-1].set(dudx0[-2])
                # w1S
                w1S = w10 - self.dt*cS* (dw1dy0 + dudx0)
            
            # 2. w2
            w20 = (u0[0,:] + u0[1,:])/2
            if self.bc_kind=='1d':
                w2S = w20
            elif self.bc_kind=='2d':
                dhdx0 = ((h0[0,1:]+h0[1,1:]-h0[0,:-1]-h0[1,:-1])/2)/(self.X[0,1:]-self.X[0,:-1])
                w2S = w20 - self.dt*g* dhdx0 
                    
            # 3. w3
            if self.bc_kind=='1d':
                _vS = (1-3/2*cS)* v0[0,:] + cS/2* (4*v0[1,:] - v0[2,:])
                _hS = (1/2+cS)* h0[1,:] + (1/2-cS)* h0[0,:]
                w3S = _vS - (1/aS) * _hS
            elif self.bc_kind=='2d':
                w30   = v0[0,:] - (1/aS) * (h0[0,:]+h0[1,:])/2
                w30_  = (v0[0,:]+v0[1,:])/2  - (1/aS) * h0[1,:]
                w30__ = v0[1,:] - (1/aS) * (h0[1,:]+h0[2,:])/2
                dw3dy0 =  -(3*w30 - 4*w30_ + w30__)/((self.Y[1,:]-self.Y[0,:])/2)
                w3S = w30 + self.dt*cS* (dw3dy0 + dudx0) 

            # 4. Values on BC
            uS = w2S
            vS = (w1S + w3S)/2 
            hS = aS * (w1S - w3S)/2
            
        #######################################################################
        # North
        #######################################################################
        if self.obc_north:
            HeN = (He[-1,:]+He[-2,:])/2
            cN = jnp.sqrt(g*HeN)
            if self.bc_kind=='1d':
                cN *= self.dt/(self.Y[-1,:]-self.Y[-2,:])
            aN = jnp.sqrt(HeN/g)

            # 1. w1
            w1extN = +w1ext[1]
            
            if self.bc_kind=='1d':
                w1N = w1extN
            elif self.bc_kind=='2d':
                w10  = v0[-1,:] - (1/aN) * (h0[-1,:]+h0[-2,:])/2
                w10_ = (v0[-1,:]+v0[-2,:])/2 - (1/aN) * h0[-2,:]
                _w10 = w1extN
                dw1dy0 = (_w10 - w10_)/(self.Y[-1,:]-self.Y[-2,:])
                dudx0 = jnp.zeros(self.nx)
                dudx0 = dudx0.at[1:-1].set(((u0[-1,1:] + u0[-2,1:] - u0[-1,:-1] - u0[-2,:-1])/2)/(self.Xu[-1,1:]-self.Xu[-1,:-1]))
                dudx0 = dudx0.at[0].set(dudx0[1])
                dudx0 = dudx0.at[-1].set(dudx0[-2])
                w1N = w10 + self.dt*cN* (dw1dy0 + dudx0)

            # 2. w2
            w20 = (u0[-1,:] + u0[-2,:])/2
            if self.bc_kind=='1d':   
                w2N = w20
            elif self.bc_kind=='2d':
                dhdx0 = ((h0[-1,1:]+h0[-2,1:]-h0[-1,:-1]-h0[-2,:-1])/2)/(self.X[-1,1:]-self.X[-1,:-1])
                w2N = w20 - self.dt*g*dhdx0 
            # 3. w3
            if self.bc_kind=='1d':   
                _vN = (1-3/2*cN)* v0[-1,:] + cN/2* (4*v0[-2,:] - v0[-3,:])
                _hN = (1/2+cN)* h0[-2,:] + (1/2-cN)* h0[-1,:]
                w3N = _vN + (1/aN) * _hN
            elif self.bc_kind=='2d':
                w30   = v0[-1,:] + (1/aN) * (h0[-1,:]+h0[-2,:])/2
                w30_  = (v0[-1,:]+v0[-2,:])/2 + (1/aN) * h0[-2,:]
                w30__ = v0[-2,:] + (1/aN) * (h0[-2,:]+h0[-3,:])/2
                dw3dy0 = (3*w30 - 4*w30_ + w30__)/((self.Y[1,:]-self.Y[0,:])/2)
                w3N = w30 - self.dt*cN* (dw3dy0 + dudx0) 
            
            # 4. Values on BC
            uN = w2N
            vN = (w1N + w3N)/2 
            hN = aN * (w3N - w1N)/2
        
        #######################################################################
        # West
        #######################################################################
        if self.obc_west:
            HeW = (He[:,0]+He[:,1])/2
            cW = jnp.sqrt(g*HeW)
            if self.bc_kind=='1d':
                cW *= self.dt/(self.X[:,1]-self.X[:,0])
            aW = jnp.sqrt(HeW/g)
            
            # 1. w1
            w1extW = +w1ext[2]
            
            if self.bc_kind=='1d':   
                w1W = w1extW
            elif self.bc_kind=='2d':
                w10  = u0[:,0] + (1/aW) * (h0[:,0]+h0[:,1])/2
                w10_ = (u0[:,0]+u0[:,1])/2 + (1/aW) * h0[:,1]
                _w10 = w1extW
                dw1dx0 = (w10_ - _w10)/(self.X[:,1]-self.X[:,0])
                dvdy0 = jnp.zeros(self.ny)
                dvdy0 = dvdy0.at[1:-1].set(((v0[1:,0] + v0[1:,1] - v0[:-1,0] - v0[:-1,1])/2)/(self.Yv[1:,0]-self.Yv[:-1,0]))
                dvdy0 = dvdy0.at[0].set(dvdy0[1])
                dvdy0 = dvdy0.at[-1].set(dvdy0[-2])
                w1W = w10 - self.dt*cW* (dw1dx0 + dvdy0) 
                
            # 2. w2
            w20 = (v0[:,0] + v0[:,1])/2
            if self.bc_kind=='1d':   
                w2W = w20
            elif self.bc_kind=='2d':
                dhdy0 = ((h0[1:,0]+h0[1:,1]-h0[:-1,0]-h0[:-1,1])/2)/(self.Y[1:,0]-self.Y[:-1,0])
                w2W = w20 - self.dt*g * dhdy0 
                    
            # 3. w3
            if self.bc_kind=='1d':   
                _uW = (1-3/2*cW)* u0[:,0] + cW/2* (4*u0[:,1]-u0[:,2]) 
                _hW = (1/2+cW)*h0[:,1] + (1/2-cW)*h0[:,0]
                w3W = _uW - (1/aW) * _hW
            elif self.bc_kind=='2d':
                w30   = u0[:,0] - (1/aW) * (h0[:,0]+h0[:,1])/2
                w30_  = (u0[:,0]+u0[:,1])/2 - (1/aW) * h0[:,1]
                w30__ = u0[:,1] - (1/aW) * (h0[:,1]+h0[:,2])/2
                dw3dx0 = -(3*w30 - 4*w30_ + w30__)/((self.Xu[:,1]-self.Xu[:,0])/2)
                w3W = w30 + self.dt*cW* (dw3dx0 + dvdy0)
                
            # 4. Values on BC
            uW = (w1W + w3W)/2 
            vW = w2W
            hW = aW * (w1W - w3W)/2
        
        #######################################################################
        # East
        #######################################################################
        if self.obc_east:
            HeE = (He[:,-1]+He[:,-2])/2
            cE = jnp.sqrt(g*HeE)
            if self.bc_kind=='1d':
                cE *= self.dt/(self.X[:,-1]-self.X[:,-2])
            aE = jnp.sqrt(HeE/g)
            
            # 1. w1
            w1extE = +w1ext[3]
            
            if self.bc_kind=='1d':   
                w1E = w1extE
            elif self.bc_kind=='2d':
                w10  = u0[:,-1] - (1/aE) * (h0[:,-1]+h0[:,-2])/2
                w10_ = (u0[:,-1]+u0[:,-2])/2 - (1/aE) * h0[:,-2]
                _w10 = w1extE
                dw1dx0 = (_w10 - w10_)/(self.X[:,-1]-self.X[:,-2])
                dvdy0 = jnp.zeros(self.ny)
                dvdy0 = dvdy0.at[1:-1].set(((v0[1:,-1] + v0[1:,-2] - v0[:-1,-1] - v0[:-1,-2])/2)/(self.Yv[1:,-1]-self.Yv[:-1,-1]))
                dvdy0 = dvdy0.at[0].set(dvdy0[1])
                dvdy0 = dvdy0.at[-1].set(dvdy0[-2])
                w1E = w10 + self.dt*cE* (dw1dx0 + dvdy0) 
            # 2. w2
            w20 = (v0[:,-1] + v0[:,-2])/2
            if  self.bc_kind=='1d':   
                w2E = w20
            elif self.bc_kind=='2d':
                w20 = (v0[:,-1] + v0[:,-2])/2
                dhdy0 = ((h0[1:,-1]+h0[1:,-2]-h0[:-1,-1]-h0[:-1,-2])/2)/(self.Y[1:,-1]-self.Y[:-1,-1])
                w2E = w20 - self.dt*g * dhdy0 
            # 3. w3
            if self.bc_kind=='1d':   
                _uE = (1-3/2*cE)* u0[:,-1] + cE/2* (4*u0[:,-2]-u0[:,-3])
                _hE = ((1/2+cE)*h0[:,-2] + (1/2-cE)*h0[:,-1])
                w3E = _uE + (1/aE) * _hE 
            elif self.bc_kind=='2d':
                w30   = u0[:,-1] + (1/aE) * (h0[:,-1]+h0[:,-2])/2
                w30_  = (u0[:,-1]+u0[:,-2])/2 + (1/aE) * h0[:,-2]
                w30__ = u0[:,-2] + (1/aE) * (h0[:,-2]+h0[:,-3])/2
                dw3dx0 =  (3*w30 - 4*w30_ + w30__)/((self.Xu[:,-1]-self.Xu[:,-2])/2)
                w3E = w30 - self.dt*cE* (dw3dx0 + dvdy0) 
                
            # 4. Values on BC
            uE = (w1E + w3E)/2 
            vE = w2E
            hE = aE * (w3E - w1E)/2
        
        #######################################################################
        # Update border pixels 
        #######################################################################
        # South
        if self.obc_south:
            u = u.at[0,:].set(uS)
            v = v.at[0,:].set(vS)
            h = h.at[0,:].set(hS)
        # North
        if self.obc_north:
            u = u.at[-1,:].set(uN)
            v = v.at[-1,:].set(vN)
            h = h.at[-1,:].set(hN)
        # West
        if self.obc_west:
            u = u.at[:,0].set(uW)
            v = v.at[:,0].set(vW)
            h = h.at[:,0].set(hW)   
        # East
        if self.obc_east:
            u = u.at[:,-1].set(uE)
            v = v.at[:,-1].set(vE)
            h = h.at[:,-1].set(hE)
        # South-West
        if self.obc_south and self.obc_west:
            u = u.at[0,0].set((uS[0] + uW[0])/2)
            v = v.at[0,0].set((vS[0] + vW[0])/2)
            h = h.at[0,0].set((hS[0] + hW[0])/2)
        # South-East
        if self.obc_south and self.obc_east:
            u = u.at[0,-1].set((uS[-1] + uE[0])/2)
            v = v.at[0,-1].set((vS[-1] + vE[0])/2)
            h = h.at[0,-1].set((hS[-1] + hE[0])/2)
        # North-West
        if self.obc_north and self.obc_west:
            u = u.at[-1,0].set((uN[0] + uW[-1])/2)
            v = v.at[-1,0].set((vN[0] + vW[-1])/2)
            h = h.at[-1,0].set((hN[0] + hW[-1])/2)
        # North-East
        if self.obc_north and self.obc_east:
            u = u.at[-1,-1].set((uN[-1] + uE[-1])/2)
            v = v.at[-1,-1].set((vN[-1] + vE[-1])/2)
            h = h.at[-1,-1].set((hN[-1] + hE[-1])/2)
    
        return u,v,h
    
    def boundary_conditions(self,u,v,h,u0,v0,h0,w1ext):

        if w1ext is not None:
            u,v,h = self.obcs_jit(u,v,h,u0,v0,h0,self.Heb,w1ext)
        
        return u,v,h
    
    def periodic_boundary_conditions(self, u, v, h):
        """
        Periodic boundary conditions for staggered C-grid with ghost cells.

        h : (ny+2, nx+2)  cell centers (ghosts in x and y)
        u : (ny+2, nx+1)  x-faces     (ghosts in y only)
        v : (ny+1, nx+2)  y-faces     (ghosts in x only)
        """

        # --------------------
        # Periodic in x
        # --------------------
        if self.periodic_x:
            # h (cell centers)
            h = h.at[:, 0].set(h[:, -2])
            h = h.at[:, -1].set(h[:, 1])

            # v (y-faces) — has x-ghosts
            v = v.at[:, 0].set(v[:, -2])
            v = v.at[:, -1].set(v[:, 1])

            # u has NO x-ghosts → nothing to do

        # --------------------
        # Periodic in y
        # --------------------
        if self.periodic_y:
            # h (cell centers)
            h = h.at[0, :].set(h[-2, :])
            h = h.at[-1, :].set(h[1, :])

            # u (x-faces) — has y-ghosts
            u = u.at[0, :].set(u[-2, :])
            u = u.at[-1, :].set(u[1, :])

            # v has NO y-ghosts → nothing to do

        return u, v, h
        
    ###########################################################################
    #                            One time step                                #
    ###########################################################################

    def asselin_filter(self, q_nm1, q_n, q_np1, nu=0.1):
        """
        Asselin time filter to remove the leapfrog computational mode.
        """
        return q_n + nu * (q_nm1 - 2.0*q_n + q_np1)
            
    def step_euler(self,u0, v0, h0, He=None, w1ext=None, u11u=None, v11u=None, u11p=None, v11p=None, dc2=None):
        
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)
        
        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(u1,v1,h1)
        kv = self.rhs_v(u1,v1,h1)
        kh = self.rhs_h(u1,v1,h1,He)
        
        #######################
        #  Time propagation   #
        #######################
        u = u1 + self.dt*ku 
        v = v1 + self.dt*kv
        h = h1 + self.dt*kh
        
        #######################
        # Boundary conditions #
        #######################
        u,v,h = self.boundary_conditions(u,v,h,u0,v0,h0,w1ext)
        
        return u, v, h
    
    def step_rk4(self, u0, v0, h0, He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None):
        
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)
        
        #######################
        #  Right hand sides   #
        #######################
        # k1
        ku1 = self.rhs_u_jit(u1,v1,h1, u11u, v11u, u11z, v11z)*self.dt
        kv1 = self.rhs_v_jit(u1,v1,h1, u11u, v11u, u11z, v11z)*self.dt
        kh1 = self.rhs_h_jit(u1,v1,h1,He, u11p, v11p)*self.dt
        # k2
        ku2 = self.rhs_u_jit(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1, u11u, v11u, u11z, v11z)*self.dt
        kv2 = self.rhs_v_jit(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1, u11u, v11u, u11z, v11z)*self.dt
        kh2 = self.rhs_h_jit(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1,He, u11p, v11p)*self.dt
        # k3
        ku3 = self.rhs_u_jit(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2, u11u, v11u, u11z, v11z)*self.dt
        kv3 = self.rhs_v_jit(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2, u11u, v11u, u11z, v11z)*self.dt
        kh3 = self.rhs_h_jit(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2,He, u11p, v11p)*self.dt
        # k4
        ku4 = self.rhs_u_jit(u1+ku3,v1+kv3,h1+kh3, u11u, v11u, u11z, v11z)*self.dt
        kv4 = self.rhs_v_jit(u1+ku3,v1+kv3,h1+kh3, u11u, v11u, u11z, v11z)*self.dt
        kh4 = self.rhs_h_jit(u1+ku3,v1+kv3,h1+kh3,He, u11p, v11p)*self.dt
        
        #######################
        #   Time propagation  #
        #######################
        u = u1 + 1/6*(ku1+2*ku2+2*ku3+ku4)
        v = v1 + 1/6*(kv1+2*kv2+2*kv3+kv4)
        h = h1 + 1/6*(kh1+2*kh2+2*kh3+kh4)
        
        #######################
        # Boundary conditions #
        #######################
        u,v,h = self.boundary_conditions(u,v,h,u0,v0,h0,w1ext)
        
        return u, v, h

    def step_leapfrog(self, u_nm1, v_nm1, h_nm1,
                        u_n,   v_n,   h_n,
                        He=None, w1ext=None):

        #######################
        #   Init local state  #
        #######################
        u1 = +u_n
        v1 = +v_n
        h1 = +h_n
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)

        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(u1, v1, h1)
        kv = self.rhs_v(u1, v1, h1)
        kh = self.rhs_h(u1, v1, h1, He)

        #######################
        #  Time propagation   #
        #######################
        u_np1 = u_nm1 + 2.0 * self.dt * ku
        v_np1 = v_nm1 + 2.0 * self.dt * kv
        h_np1 = h_nm1 + 2.0 * self.dt * kh

        #######################
        # Boundary conditions #
        #######################
        u_np1, v_np1, h_np1 = self.boundary_conditions(
            u_np1, v_np1, h_np1,
            u_n, v_n, h_n,
            w1ext
        )

        return u_np1, v_np1, h_np1
      
    def step_euler_tgl(self,
                       du0, dv0, dh0, u0, v0, h0, 
                       dHe=None, dw1ext=None, du11=None, dv11=None, du11p=None, dv11p=None, ddc2=None,
                       He=None, w1ext=None, u11u=None, v11u=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2 = x
            return self.step_euler(u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2)

        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2),)
        tangents = ((du0, dv0, dh0, dHe, dw1ext, du11, dv11, du11p, dv11p, ddc2),)

        _, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)
     
    def step_rk4_tgl(self,
                     du0, dv0, dh0, u0, v0, h0, 
                     dHe=None, dw1ext=None, du11u=None, dv11u=None, du11z=None, dv11z=None, du11p=None, dv11p=None, ddc2=None,
                     He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk4(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)

        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        tangents = ((du0, dv0, dh0, dHe, dw1ext, du11u, dv11u, du11z, dv11z, du11p, dv11p, ddc2),)
        _, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)
      
    def step_euler_adj(self,
                       adu0, adv0, adh0, u0, v0, h0,
                       He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_euler(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)
        
        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        cotangents = (adu0, adv0, adh0)  

        _, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)

        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_He, adj_w1ext, adj_u11, adj_v11, adj_u11p, adj_v11p, adj_dc2)

    def step_rk4_adj(self,
                     adu0, adv0, adh0, u0, v0, h0,
                    He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk4(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)
        
        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        cotangents = (adu0, adv0, adh0)  

        _, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)

        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_He, adj_w1ext, adj_u11, adj_v11, adj_u11z, adj_v11z, adj_u11p, adj_v11p, adj_dc2)
   

if __name__ == "__main__":
    
    import numpy 
    
    x = numpy.arange(0,1e6,10e3)
    y = numpy.arange(0,1e6,10e3)
    ny,nx = y.size,x.size
    X,Y = numpy.meshgrid(x,y)
    dt = 900
    
    swm = Swm(X=X,Y=Y,dt=dt)

    N = swm.nstates + swm.nparams
    
    X0 = numpy.zeros((N,))
    
    X0[swm.sliceHe] = 0.7
    X0[swm.sliceBc][:swm.nx] = 0.02
    
    for i in range(100):
        X0 = swm.step_rk4(X0)
    
    X0 = numpy.random.random((N,))
    dX0 = numpy.random.random((N,))
    adX0 = numpy.random.random((N,))
    
    print('Tangent test:')
    X2 = swm.step_rk4_jit(X0)
    for p in range(10):
        
        lambd = 10**(-p)
        
        X1 = swm.step_rk4_jit(X0+lambd*dX0)
        
        dX1 = swm.step_rk4_tgl_jit(dX0=lambd*dX0,X0=X0)
        
        ps = numpy.linalg.norm(X1-X2-dX1)/jnp.linalg.norm(dX1)

        print('%.E' % lambd,'%.E' % ps)
    
    print('\nAdjoint test:')
    dX1 = swm.step_rk4_tgl_jit(dX0=dX0,X0=X0)
    adX1= swm.step_rk4_adj_jit(adX0,X0)
    
    ps1 = numpy.inner(dX1,adX0)
    ps2 = numpy.inner(dX0,adX1)
    
    print(ps1/ps2)
    
   