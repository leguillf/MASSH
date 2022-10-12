from swm import Swm
import numpy as np
from copy import deepcopy

from obcs_tgl import init_bc_tgl,obcs_tgl,update_borders_tgl
from obcs import init_bc

class Swm_tgl(Swm):
    
    def __init__(self,X=None,Y=None,dt=None,bc=None,omegas=None,bc_theta=None,g=9.81,f=1e-4):
        
        super().__init__(X,Y,dt,bc,omegas,bc_theta,g,f)
        
        self._dpku = 0
        self._dppku = 0
        self._dpkv = 0
        self._dppkv = 0
        self._dpkh = 0
        self._dppkh = 0
        
        self._dpu = 0
        self._dpv = 0
        self._dph = 0

    def rhs_h_tgl(self,du,dv,dHe,u,v,He):
        
        drhs_h = np.zeros_like(self.X)
        drhs_h[1:-1,1:-1] += \
            - dHe[1:-1,1:-1] * \
                ((u[1:-1,1:] - u[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) +   \
                 (v[1:,1:-1] - v[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))    \
            - He[1:-1,1:-1] * \
                ((du[1:-1,1:] - du[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) +   \
                 (dv[1:,1:-1] - dv[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))    

        return drhs_h  
    
    
    
    ###########################################################################
    #                            One time step                                #
    ###########################################################################
    
    def step_euler_tgl(self,t,du0,dv0,dh0,u0,v0,h0,
                     dHe=None,He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None,first=False):
          
        #######################
        #   Init local state  #
        #######################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        du1 = deepcopy(du0)
        dv1 = deepcopy(dv0)
        dh1 = deepcopy(dh0)
        
        #######################
        #       Init bc       #
        #######################
        if first:
            init_bc_tgl(self,du1,dv1,dh1,dHe,dhbcx,dhbcy,He,hbcx,hbcy,t0=t)
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
        
        #######################
        # Boundary conditions #
        #######################
        duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE = \
            obcs_tgl(self,t,du1,dv1,dh1,dHe,dhbcx,dhbcy,
                          u1,v1,h1,He,hbcx,hbcy)
        
        #######################
        #  Right hand sides   #
        #######################
        dku = self.rhs_u(self.v_on_u(dv1),dh1)
        dkv = self.rhs_v(self.u_on_v(du1),dh1)
        dkh = self.rhs_h_tgl(du1,dv1,dHe,u1,v1,He)
        
        #######################
        #  Time propagation   #
        #######################
        du = du1 + self.dt*dku 
        dv = dv1 + self.dt*dkv
        dh = dh1 + self.dt*dkh
    
        ########################
        # Update border pixels #
        ########################
        update_borders_tgl(self,du,dv,dh,
                                duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE)
        
        return du,dv,dh
    

    def step_lf_tgl(self,t,du0,dv0,dh0,u0,v0,h0,
                     dHe=None,He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None,first=False):
        
        #######################
        #   Init local state  #
        #######################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        du1 = deepcopy(du0)
        dv1 = deepcopy(dv0)
        dh1 = deepcopy(dh0)
        
        #######################
        #       Init bc       #
        #######################
        if first:
            init_bc_tgl(self,du1,dv1,dh1,dHe,dhbcx,dhbcy,He,hbcx,hbcy,t0=t)
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
            
        #######################
        # Boundary conditions #
        #######################
        duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE = \
            obcs_tgl(self,t,du1,dv1,dh1,dHe,dhbcx,dhbcy,
                          u1,v1,h1,He,hbcx,hbcy)
                
        #######################
        #  Right hand sides   #
        #######################
        dku = self.rhs_u(self.v_on_u(dv1),dh1)
        dkv = self.rhs_v(self.u_on_v(du1),dh1)
        dkh = self.rhs_h_tgl(du1,dv1,dHe,u1,v1,He)
        
        #######################
        #  Time propagation   #
        #######################
        if first:
            # forward euler
            du = du1 + self.dt*dku 
            dv = dv1 + self.dt*dkv
            dh = dh1 + self.dt*dkh
        else:
            # leap-frog
            du = self._dpu + 2*self.dt*dku 
            dv = self._dpv + 2*self.dt*dkv
            dh = self._dph + 2*self.dt*dkh
        
        self._dpu = du1
        self._dpv = dv1
        self._dph = dh1
    
        ########################
        # Update border pixels #
        ########################
        update_borders_tgl(self,du,dv,dh,
                               duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE)
        
        return du,dv,dh
    
    
    def step_rk4_tgl(self,t,du0,dv0,dh0,u0,v0,h0,
                 dHe=None,He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None,first=False):
        
        #######################
        #   Init local state  #
        #######################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        du1 = deepcopy(du0)
        dv1 = deepcopy(dv0)
        dh1 = deepcopy(dh0)
        
        #######################
        #       Init bc       #
        #######################
        if first:
            init_bc_tgl(self,du1,dv1,dh1,dHe,dhbcx,dhbcy,He,hbcx,hbcy,t0=t)
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
            
        #######################
        # Boundary conditions #
        #######################
        duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE = \
            obcs_tgl(self,t,du1,dv1,dh1,dHe,dhbcx,dhbcy,
                          u1,v1,h1,He,hbcx,hbcy)
            
        #######################
        # Current trajectory  #
        #######################
        # k1
        ku1 = self.rhs_u(self.v_on_u(v1),h1)*self.dt
        kv1 = self.rhs_v(self.u_on_v(u1),h1)*self.dt
        kh1 = self.rhs_h(u1,v1,He)*self.dt
        # k2
        ku2 = self.rhs_u(self.v_on_u(v1+0.5*kv1),h1+0.5*kh1)*self.dt
        kv2 = self.rhs_v(self.u_on_v(u1+0.5*ku1),h1+0.5*kh1)*self.dt
        kh2 = self.rhs_h(u1+0.5*ku1,v1+0.5*kv1,He)*self.dt
        # k3
        ku3 = self.rhs_u(self.v_on_u(v1+0.5*kv2),h1+0.5*kh2)*self.dt
        kv3 = self.rhs_v(self.u_on_v(u1+0.5*ku2),h1+0.5*kh2)*self.dt
        
        
        #######################
        #  Right hand sides   #
        #######################
        # k1_p
        ku1_p = self.rhs_u(self.v_on_u(dv1),dh1)*self.dt
        kv1_p = self.rhs_v(self.u_on_v(du1),dh1)*self.dt
        kh1_p = self.rhs_h_tgl(du1,dv1,dHe,u1,v1,He)*self.dt
        # k2_p
        ku2_p = self.rhs_u(self.v_on_u(dv1+0.5*kv1_p),dh1+0.5*kh1_p)*self.dt
        kv2_p = self.rhs_v(self.u_on_v(du1+0.5*ku1_p),dh1+0.5*kh1_p)*self.dt
        kh2_p = self.rhs_h_tgl(
            du1+0.5*ku1_p,dv1+0.5*kv1_p,dHe,u1+0.5*ku1,v1+0.5*kv1,He)*self.dt
        # k3_p
        ku3_p = self.rhs_u(self.v_on_u(dv1+0.5*kv2_p),dh1+0.5*kh2_p)*self.dt
        kv3_p = self.rhs_v(self.u_on_v(du1+0.5*ku2_p),dh1+0.5*kh2_p)*self.dt
        kh3_p = self.rhs_h_tgl(
            du1+0.5*ku2_p,dv1+0.5*kv2_p,dHe,u1+0.5*ku2,v1+0.5*kv2,He)*self.dt
        # k4_p
        ku4_p = self.rhs_u(self.v_on_u(dv1+kv3_p),dh1+kh3_p)*self.dt
        kv4_p = self.rhs_v(self.u_on_v(du1+ku3_p),dh1+kh3_p)*self.dt
        kh4_p = self.rhs_h_tgl(
            du1+ku3_p,dv1+kv3_p,dHe,u1+ku3,v1+kv3,He)*self.dt
        
        #######################
        #  Time propagation   #
        #######################
        du = du1 + 1/6*(ku1_p+2*ku2_p+2*ku3_p+ku4_p)
        dv = dv1 + 1/6*(kv1_p+2*kv2_p+2*kv3_p+kv4_p)
        dh = dh1 + 1/6*(kh1_p+2*kh2_p+2*kh3_p+kh4_p)

        ########################
        # Update border pixels #
        ########################
        update_borders_tgl(self,du,dv,dh,
                               duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE)
            
        return du,dv,dh
 
    
    
