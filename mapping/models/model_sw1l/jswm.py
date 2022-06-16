import jax.numpy as np 
from jax import jit
from jax import jvp,vjp
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pylab as plt

class Swm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,X=None,Y=None,dt=None,bc_kind='1d',g=9.81,f=1e-4,Heb=0.7):
        
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
            self.f = f * np.ones_like(self.X)
        
        if hasattr(Heb, "__len__") and f.shape==self.X.shape:
            self.Heb = Heb
        else: 
            self.Heb = Heb * np.ones_like(self.X)
        
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
        
        
        # JAX compiling
        self.u_on_v_jit = jit(self.u_on_v)
        self.v_on_u_jit = jit(self.v_on_u)
        self.rhs_u_jit = jit(self.rhs_u)
        self.rhs_v_jit = jit(self.rhs_v)
        self.rhs_h_jit = jit(self.rhs_h)
        self.obcs_jit = jit(self.obcs)
        self.step_rk4_jit = jit(self.step_rk4)
        self.step_euler_tgl_jit = jit(self.step_euler_tgl)
        self.step_euler_adj_jit = jit(self.step_euler_adj)
        self.step_rk4_tgl_jit = jit(self.step_rk4_tgl)
        self.step_rk4_adj_jit = jit(self.step_rk4_adj)
        
        
        
        
        
        
        
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
        
        rhs_u = rhs_u.at[1:-1,1:-1].set((self.f[1:-1,2:-1]+self.f[1:-1,1:-2])/2 * vm -\
            self.g * (h[1:-1,2:-1] - h[1:-1,1:-2]) / ((self.X[1:-1,2:-1]-self.X[1:-1,1:-2])))
        
        return rhs_u

    def rhs_v(self,um,h):
        
        rhs_v = np.zeros_like(self.Xv)
        
        rhs_v = rhs_v.at[1:-1,1:-1].set(-(self.f[2:-1,1:-1]+self.f[1:-2,1:-1])/2 * um -\
            self.g * (h[2:-1,1:-1] - h[1:-2,1:-1]) / ((self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1])))
            
        return rhs_v
    
    def rhs_h(self,u,v,He):
        rhs_h = np.zeros_like(self.X)
        rhs_h = rhs_h.at[1:-1,1:-1].set(- He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) + \
                (v[1:,1:-1] - v[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1])))
          
        return rhs_h
    
    
    ###########################################################################
    #                      Open Boundary Conditions                           #
    ###########################################################################
    
    def obcs(self,u,v,h,u0,v0,h0,He,w1ext):
        
        g = self.g
                
        #######################################################################
        # South
        #######################################################################
        HeS = (He[0,:]+He[1,:])/2
        cS = np.sqrt(g*HeS)
        if self.bc_kind=='1d':
            cS *= self.dt/(self.Y[1,:]-self.Y[0,:])
      
        # 1. w1
        w1extS = +w1ext[0]
        
        if self.bc_kind=='1d':
            w1S = w1extS
        elif self.bc_kind=='2d':
            # dw1dy0
            w10  = v0[0,:] + np.sqrt(g/HeS)* (h0[0,:]+h0[1,:])/2
            w10_ = (v0[0,:]+v0[1,:])/2 + np.sqrt(g/HeS)* h0[1,:]
            _w10 = w1extS
            dw1dy0 = (w10_ - _w10)/self.dy
            # dudx0
            dudx0 = np.zeros(self.nx)
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
            w3S = _vS - np.sqrt(g/HeS) * _hS
        elif self.bc_kind=='2d':
            w30   = v0[0,:] - np.sqrt(g/HeS)* (h0[0,:]+h0[1,:])/2
            w30_  = (v0[0,:]+v0[1,:])/2  - np.sqrt(g/HeS)* h0[1,:]
            w30__ = v0[1,:] - np.sqrt(g/HeS)* (h0[1,:]+h0[2,:])/2
            dw3dy0 =  -(3*w30 - 4*w30_ + w30__)/(self.dy/2)
            w3S = w30 + self.dt*cS* (dw3dy0 + dudx0) 

        # 4. Values on BC
        uS = w2S
        vS = (w1S + w3S)/2 
        hS = np.sqrt(HeS/g) *(w1S - w3S)/2
        
        #######################################################################
        # North
        #######################################################################
        HeN = (He[-1,:]+He[-2,:])/2
        cN = np.sqrt(g*HeN)
        if self.bc_kind=='1d':
            cN *= self.dt/(self.Y[-1,:]-self.Y[-2,:])

        # 1. w1
        w1extN = +w1ext[1]
        
        if self.bc_kind=='1d':
            w1N = w1extN
        elif self.bc_kind=='2d':
            w10  = v0[-1,:] - np.sqrt(g/HeN)* (h0[-1,:]+h0[-2,:])/2
            w10_ = (v0[-1,:]+v0[-2,:])/2 - np.sqrt(g/HeN)* h0[-2,:]
            _w10 = w1extN
            dw1dy0 = (_w10 - w10_)/self.dy
            dudx0 = np.zeros(self.nx)
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
            w3N = _vN + np.sqrt(g/HeN) * _hN
        elif self.bc_kind=='2d':
            w30   = v0[-1,:] + np.sqrt(g/HeN)* (h0[-1,:]+h0[-2,:])/2
            w30_  = (v0[-1,:]+v0[-2,:])/2 + np.sqrt(g/HeN)* h0[-2,:]
            w30__ = v0[-2,:] + np.sqrt(g/HeN)* (h0[-2,:]+h0[-3,:])/2
            dw3dy0 =  (3*w30 - 4*w30_ + w30__)/(self.dy/2)
            w3N = w30 - self.dt*cN* (dw3dy0 + dudx0) 
        
        # 4. Values on BC
        uN = w2N
        vN = (w1N + w3N)/2 
        hN = np.sqrt(HeN/g) *(w3N - w1N)/2
        
        #######################################################################
        # West
        #######################################################################
        HeW = (He[:,0]+He[:,1])/2
        cW = np.sqrt(g*HeW)
        if self.bc_kind=='1d':
            cW *= self.dt/(self.X[:,1]-self.X[:,0])
        
        # 1. w1
        w1extW = +w1ext[2]
        
        if self.bc_kind=='1d':   
            w1W = w1extW
        elif self.bc_kind=='2d':
            w10  = u0[:,0] + np.sqrt(g/HeW)* (h0[:,0]+h0[:,1])/2
            w10_ = (u0[:,0]+u0[:,1])/2 + np.sqrt(g/HeW)* h0[:,1]
            _w10 = w1extW
            dw1dx0 = (w10_ - _w10)/self.dx
            dvdy0 = np.zeros(self.ny)
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
            w3W = _uW - np.sqrt(g/HeW)* _hW
        elif self.bc_kind=='2d':
            w30   = u0[:,0] - np.sqrt(g/HeW)* (h0[:,0]+h0[:,1])/2
            w30_  = (u0[:,0]+u0[:,1])/2 - np.sqrt(g/HeW)* h0[:,1]
            w30__ = u0[:,1] - np.sqrt(g/HeW)* (h0[:,1]+h0[:,2])/2
            dw3dx0 = -(3*w30 - 4*w30_ + w30__)/(self.dx/2)
            w3W = w30 + self.dt*cW* (dw3dx0 + dvdy0)
            
        # 4. Values on BC
        uW = (w1W + w3W)/2 
        vW = w2W
        hW = np.sqrt(HeW/g)*(w1W - w3W)/2
        
        #######################################################################
        # East
        #######################################################################
        HeE = (He[:,-1]+He[:,-2])/2
        cE = np.sqrt(g*HeE)
        if self.bc_kind=='1d':
            cE *= self.dt/(self.X[:,-1]-self.X[:,-2])
        
        # 1. w1
        w1extE = +w1ext[3]
        
        if self.bc_kind=='1d':   
            w1E = w1extE
        elif self.bc_kind=='2d':
            w10  = u0[:,-1] - np.sqrt(g/HeE)* (h0[:,-1]+h0[:,-2])/2
            w10_ = (u0[:,-1]+u0[:,-2])/2 - np.sqrt(g/HeE)* h0[:,-2]
            _w10 = w1extE
            dw1dx0 = (_w10 - w10_)/self.dx
            dvdy0 = np.zeros(self.ny)
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
            w3E = _uE + np.sqrt(g/HeE)* _hE 
        elif self.bc_kind=='2d':
            w30   = u0[:,-1] + np.sqrt(g/HeE)* (h0[:,-1]+h0[:,-2])/2
            w30_  = (u0[:,-1]+u0[:,-2])/2 + np.sqrt(g/HeE)* h0[:,-2]
            w30__ = u0[:,-2] + np.sqrt(g/HeE)* (h0[:,-2]+h0[:,-3])/2
            dw3dx0 =  (3*w30 - 4*w30_ + w30__)/(self.dx/2)
            w3E = w30 - self.dt*cE* (dw3dx0 + dvdy0) 
            
        # 4. Values on BC
        uE = (w1E + w3E)/2 
        vE = w2E
        hE = np.sqrt(HeE/g)*(w3E - w1E)/2
        
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
    
    
    
    ###########################################################################
    #                            One time step                                #
    ###########################################################################
            
    def step_euler(self,X0):
        
        #######################
        #       Reshaping     #
        #######################
        u0 = X0[self.sliceu].reshape(self.shapeu)
        v0 = X0[self.slicev].reshape(self.shapev)
        h0 = X0[self.sliceh].reshape(self.shapeh)
        
        if X0.size==(self.nstates+self.nparams):
            He = X0[self.sliceHe].reshape(self.shapeHe)
            Bc = X0[self.sliceBc]
            w1ext = (Bc[:self.nx],
                     Bc[self.nx:2*self.nx],
                     Bc[2*self.nx:2*self.nx+self.ny],
                     Bc[2*self.nx+self.ny:2*self.nx+2*self.ny])
        else:
            He = self.Heb
            w1ext = None
            
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
        
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
        if w1ext is not None:
            u,v,h = self.obcs_jit(u,v,h,u1,v1,h1,He,w1ext)
        
        #######################
        #      Flattening     #
        #######################
        X1 = np.concatenate((u.flatten(),v.flatten(),h.flatten()))
        
        if X0.size==(self.nstates+self.nparams):
            X1 = np.concatenate((X1,He.flatten(),Bc))
        
        return X1
    

    def step_rk4(self,X0):
        
        #######################
        #       Reshaping     #
        #######################
        u0 = X0[self.sliceu].reshape(self.shapeu)
        v0 = X0[self.slicev].reshape(self.shapev)
        h0 = X0[self.sliceh].reshape(self.shapeh)
        
        if X0.size==(self.nstates+self.nparams):
            He = X0[self.sliceHe].reshape(self.shapeHe)
            Bc = X0[self.sliceBc]
            w1ext = (Bc[:self.nx],
                     Bc[self.nx:2*self.nx],
                     Bc[2*self.nx:2*self.nx+self.ny],
                     Bc[2*self.nx+self.ny:2*self.nx+2*self.ny])
        else:
            He = self.Heb
            w1ext = None
        
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
                    
        #######################
        #  Right hand sides   #
        #######################
        # k1
        ku1 = self.rhs_u_jit(self.v_on_u_jit(v1),h1)*self.dt
        kv1 = self.rhs_v_jit(self.u_on_v_jit(u1),h1)*self.dt
        kh1 = self.rhs_h_jit(u1,v1,He)*self.dt
        # k2
        ku2 = self.rhs_u_jit(self.v_on_u_jit(v1+0.5*kv1),h1+0.5*kh1)*self.dt
        kv2 = self.rhs_v_jit(self.u_on_v_jit(u1+0.5*ku1),h1+0.5*kh1)*self.dt
        kh2 = self.rhs_h_jit(u1+0.5*ku1,v1+0.5*kv1,He)*self.dt
        # k3
        ku3 = self.rhs_u_jit(self.v_on_u_jit(v1+0.5*kv2),h1+0.5*kh2)*self.dt
        kv3 = self.rhs_v_jit(self.u_on_v_jit(u1+0.5*ku2),h1+0.5*kh2)*self.dt
        kh3 = self.rhs_h_jit(u1+0.5*ku2,v1+0.5*kv2,He)*self.dt
        # k4
        ku4 = self.rhs_u_jit(self.v_on_u_jit(v1+kv3),h1+kh3)*self.dt
        kv4 = self.rhs_v_jit(self.u_on_v_jit(u1+ku3),h1+kh3)*self.dt
        kh4 = self.rhs_h_jit(u1+ku3,v1+kv3,He)*self.dt
        
        
        #######################
        #   Time propagation  #
        #######################
        u = u1 + 1/6*(ku1+2*ku2+2*ku3+ku4)
        v = v1 + 1/6*(kv1+2*kv2+2*kv3+kv4)
        h = h1 + 1/6*(kh1+2*kh2+2*kh3+kh4)
        
        #######################
        # Boundary conditions #
        #######################
        if w1ext is not None:
            u,v,h = self.obcs_jit(u,v,h,u1,v1,h1,He,w1ext)
        
        #######################
        #      Flattening     #
        #######################
        X1 = np.concatenate((u.flatten(),v.flatten(),h.flatten()))
        
        if X0.size==(self.nstates+self.nparams):
            X1 = np.concatenate((X1,He.flatten(),Bc))
        
        return X1
    
    
    def step_euler_tgl(self,dX0,X0):
        
        _,dX1 = jvp(self.step_euler_jit, (X0,), (dX0,))
        
        return dX1
    
    def step_rk4_tgl(self,dX0,X0):
        
        _,dX1 = jvp(self.step_rk4_jit, (X0,), (dX0,))
        
        return dX1
    

    def step_euler_adj(self,adX0,X0):
        
        _, adf = vjp(self.step_euler_jit, X0)
        
        return adf(adX0)[0]

    def step_rk4_adj(self,adX0,X0):
        
        _, adf = vjp(self.step_rk4_jit, X0)
        
        return adf(adX0)[0]
     
        
     
        
     
if __name__ == "__main__":  
    
    import numpy 
    
    x = numpy.arange(0,1e6,10e3)
    y = numpy.arange(0,1e6,10e3)
    ny,nx = y.size,x.size
    X,Y = numpy.meshgrid(x,y)
    dt = 900
    
    swm = Swm(X=X,Y=Y,dt=dt)

    N = swm.nstates + swm.nparams
    
    X0 = numpy.random.random((N,))
    dX0 = numpy.random.random((N,))
    adX0 = numpy.random.random((N,))
    
    print('Tangent test:')
    X2 = swm.step_rk4_jit(X0)
    for p in range(10):
        
        lambd = 10**(-p)
        
        X1 = swm.step_rk4_jit(X0+lambd*dX0)
        
        dX1 = swm.step_rk4_tgl_jit(dX0=lambd*dX0,X0=X0)
        
        ps = numpy.linalg.norm(X1-X2-dX1)/np.linalg.norm(dX1)

        print('%.E' % lambd,'%.E' % ps)
    
    print('\nAdjoint test:')
    dX1 = swm.step_rk4_tgl_jit(dX0=dX0,X0=X0)
    adX1= swm.step_rk4_adj_jit(adX0,X0)
    
    ps1 = numpy.inner(dX1,adX0)
    ps2 = numpy.inner(dX0,adX1)
    
    print(ps1/ps2)
    
   