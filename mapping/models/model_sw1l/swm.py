from obcs import init_bc,obcs
import numpy as np 
from copy import deepcopy

class Swm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,X=None,Y=None,dt=None,bc=None,omegas=None,bc_theta=None,g=9.81,f=1e-4):
        
        self.X = X
        self.Y = Y
        self.Xu = self.rho_on_u(X)
        self.Yu = self.rho_on_u(Y)
        self.Xv = self.rho_on_v(X)
        self.Yv = self.rho_on_v(Y)
        self.dt = dt
        self.bc = bc
        self.omegas = omegas
        self.bc_theta = bc_theta
        self.g = g
        if hasattr(f, "__len__") and f.shape==self.X.shape:
            self.f = f
        else: 
            self.f = f * np.ones_like(self.X)
        
        self.ny,self.nx = self.X.shape
        
        #For 'ab3' timescheme
        self._pku = 0
        self._ppku = 0
        self._pkv = 0
        self._ppkv = 0
        self._pkh = 0
        self._ppkh = 0
        
        # For leap-frog timescheme
        self._pu = 0
        self._pv = 0
        self._ph = 0
        
        
    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
    
    def rho_on_u(self,rho):
        
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def rho_on_v(self,rho):
        
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def u_on_v(self,u):
        
        um = 0.25 * (u[2:-1,:-1] + u[2:-1,1:] + u[1:-2,:-1] + u[1:-2,1:])
        
        return um
    
    def v_on_u(self,v):
        
        vm = 0.25 * (v[:-1,2:-1] + v[:-1,1:-2] + v[1:,2:-1] + v[1:,1:-2])
        
        return vm
    
    ###########################################################################
    #                          Right hand sides                               #
    ###########################################################################
    
    def rhs_u(self,vm,h):
        
        rhs_u = np.zeros_like(self.Xu)
        
        rhs_u[1:-1,1:-1] = +(self.f[1:-1,2:-1]+self.f[1:-1,1:-2])/2 * vm -\
            self.g * (h[1:-1,2:-1] - h[1:-1,1:-2]) / ((self.X[1:-1,2:-1]-self.X[1:-1,1:-2]))
        
        return rhs_u

    def rhs_v(self,um,h):
        
        rhs_v = np.zeros_like(self.Xv)
        
        rhs_v[1:-1,1:-1] = -(self.f[2:-1,1:-1]+self.f[1:-2,1:-1])/2 * um -\
            self.g * (h[2:-1,1:-1] - h[1:-2,1:-1]) / ((self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1]))
            
        return rhs_v
    
    def rhs_h(self,u,v,He):
        rhs_h = np.zeros_like(self.X)
        rhs_h[1:-1,1:-1] = - He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) + \
                (v[1:,1:-1] - v[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))
          
        return rhs_h

    ###########################################################################
    #                            One time step                                #
    ###########################################################################
            
    def step_euler(self,t,u0,v0,h0,He=None,hbcx=None,hbcy=None,step=None):
        
        #######################
        #   Init local state  #
        #######################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        
        #######################
        #       Init bc       #
        #######################
        if step==0:
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
            
        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(self.v_on_u(v1),h1)
        kv = self.rhs_v(self.u_on_v(u1),h1)
        kh = self.rhs_h(u1,v1,He)
        
        #######################
        #  Time propagation   #
        #######################
        u = u1 + self.dt*ku 
        v = v1 + self.dt*kv
        h = h1 + self.dt*kh 
        
        #######################
        # Boundary conditions #
        #######################
        if hbcx is not None and hbcy is not None:
            obcs(self,t,u,v,h,u1,v1,h1,He,hbcx,hbcy)
        
        return u,v,h
    
    def step_lf(self,t,u0,v0,h0,He=None,hbcx=None,hbcy=None,step=0):
        
        #######################
        #   Init local state  #
        #######################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        
        #######################
        #       Init bc       #
        #######################
        if step==0:
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
            
        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(self.v_on_u(v1),h1)
        kv = self.rhs_v(self.u_on_v(u1),h1)
        kh = self.rhs_h(u1,v1,He)
        
        #######################
        #   Time propagation  #
        #######################
        if step==0:
            # forward euler
            u = u1 + self.dt*ku 
            v = v1 + self.dt*kv
            h = h1 + self.dt*kh
        else:
            # leap-frog
            u = self._pu + 2*self.dt*ku 
            v = self._pv + 2*self.dt*kv
            h = self._ph + 2*self.dt*kh
        # For next timestep
        self._pu = u1
        self._pv = v1
        self._ph = h1
        
        #######################
        # Boundary conditions #
        #######################
        if hbcx is not None and hbcy is not None:
            obcs(self,t,u,v,h,u1,v1,h1,He,hbcx,hbcy)
            
        return u,v,h

    
            