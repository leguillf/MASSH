#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from swm_tgl import Swm_tgl
import numpy as np
from copy import deepcopy

from obcs_adj import init_bc_adj,obcs_adj
from obcs import init_bc

class Swm_adj(Swm_tgl):
    
    def __init__(self,X=None,Y=None,dt=None,bc=None,omegas=None,bc_theta=None,g=9.81,f=1e-4):
        
        super().__init__(X,Y,dt,bc,omegas,bc_theta,g,f)
        
        self.restart()
            
    def restart(self):
        self._adpku = 0
        self._adppku = 0
        self._adpkv = 0
        self._adppkv = 0
        self._adpkh = 0
        self._adppkh = 0
        self._adpu = 0
        self._adpv = 0
        self._adph = 0
            
    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
                
    def u_on_v_adj(self,adum):
        
        adu = np.zeros_like(self.Xu)
        adu[2:-1,:-1] += 0.25 * adum
        adu[2:-1,1:]  += 0.25 * adum
        adu[1:-2,:-1] += 0.25 * adum
        adu[1:-2,1:]  += 0.25 * adum
        
        return adu
    
    def v_on_u_adj(self,advm):
        
        adv = np.zeros_like(self.Xv)
        adv[:-1,2:-1] += 0.25 * advm
        adv[:-1,1:-2] += 0.25 * advm
        adv[1:,2:-1]  += 0.25 * advm
        adv[1:,1:-2]  += 0.25 * advm
        
        return adv
    
    ###########################################################################
    #                  Right hand side for u equation                         #
    ###########################################################################
    
    def rhs_u_adj(self,adrhs_u):
        
        adh = np.zeros_like(self.X)
        
        advm = (self.f[1:-1,2:-1]+self.f[1:-1,1:-2])/2 * adrhs_u[1:-1,1:-1]
        
        adh[1:-1,2:-1] += - self.g * adrhs_u[1:-1,1:-1] / (self.X[1:-1,2:-1]-self.X[1:-1,1:-2])
        adh[1:-1,1:-2] += + self.g * adrhs_u[1:-1,1:-1] / (self.X[1:-1,2:-1]-self.X[1:-1,1:-2])
        
        return advm,adh
    
    def rhs_v_adj(self,advrhs):
        
        adh = np.zeros_like(self.X)
        
        adum = - (self.f[2:-1,1:-1]+self.f[1:-2,1:-1])/2 * advrhs[1:-1,1:-1]
        
        adh[2:-1,1:-1] += - self.g * advrhs[1:-1,1:-1] / (self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1])
        adh[1:-2,1:-1] += + self.g * advrhs[1:-1,1:-1] / (self.Y[2:-1,1:-1]-self.Y[1:-2,1:-1])
        
        return adum,adh
    
    def rhs_h_adj(self,t,adrhs_h,u,v,He):
        
        adu = np.zeros_like(self.Xu)
        adv = np.zeros_like(self.Xv)
        adHe2d = np.zeros_like(self.X)
        
        adu[1:-1,1:] += - He[1:-1,1:-1] * adrhs_h[1:-1,1:-1] / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1])
        adu[1:-1,:-1] += + He[1:-1,1:-1] * adrhs_h[1:-1,1:-1] / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1])
        adv[1:,1:-1] += - He[1:-1,1:-1] * adrhs_h[1:-1,1:-1] / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1])
        adv[:-1,1:-1] += + He[1:-1,1:-1] * adrhs_h[1:-1, 1:-1] / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1])
        adHe2d[1:-1,1:-1] = - adrhs_h[1:-1, 1:-1] * \
                ((u[1:-1,1:] - u[1:-1,:-1]) / (self.Xu[1:-1,1:] - self.Xu[1:-1,:-1]) +   \
                 (v[1:,1:-1] - v[:-1,1:-1]) / (self.Yv[1:,1:-1] - self.Yv[:-1,1:-1]))  
            
        return adu,adv,adHe2d
    
    ###########################################################################
    #                            One time step                                #
    ###########################################################################

    def step_adj_euler(self,t,adu0,adv0,adh0, u0,v0,h0, 
                       He=None,hbcx=None,hbcy=None,step=None):
        
        ########################
        #         Init         #
        ########################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        adu = 0
        adv = 0
        adh = 0
        adHe = 0
        adhbcx = 0
        adhbcy = 0
        
        if step==0:
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
        
        #######################
        # Boundary conditions #
        #######################
        adu_tmp,adv_tmp,adh_tmp,adHe_tmp,adhbcx_tmp,adhbcy_tmp = \
            obcs_adj(self,t,adu0,adv0,adh0,u1,v1,h1,He,hbcx,hbcy)
        adu += adu_tmp
        adv += adv_tmp
        adh += adh_tmp
        adHe += adHe_tmp
        adhbcx += adhbcx_tmp
        adhbcy += adhbcy_tmp
        
        #######################
        #  Time propagation   #
        #######################
        adku = self.dt*adu0
        adkv = self.dt*adv0
        adkh = self.dt*adh0
        
        #######################
        #  Right hand sides   #
        #######################
        adu_tmp,adh_tmp = self.rhs_v_adj(adkv)
        adu_tmp = self.u_on_v_adj(adu_tmp)
        adu += adu_tmp
        adh += adh_tmp
        adv_tmp,adh_tmp = self.rhs_u_adj(adku)
        adv_tmp = self.v_on_u_adj(adv_tmp)
        adv += adv_tmp
        adh += adh_tmp
        adu_tmp,adv_tmp,adHe_tmp = self.rhs_h_adj(t,adkh,u1,v1,He)
        adu += adu_tmp
        adv += adv_tmp
        adHe += adHe_tmp
        
        #######################
        #       Update        #
        #######################
        adu += adu0 
        adv += adv0 
        adh += adh0 
        
        #######################
        #       Init bc       #
        #######################
        if step==0:
            adHe_tmp,adhbcx_tmp,adhbcy_tmp = init_bc_adj(
                self,adu,adv,adh,He,hbcx,hbcy,t0=t)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
        
        
        return adu,adv,adh,adHe,adhbcx,adhbcy
    
        
    def step_adj_lf(self,t,adu0,adv0,adh0, u0,v0,h0, 
                       He=None,hbcx=None,hbcy=None,step=None):
        
        ########################
        #         Init         #
        ########################
        u1 = deepcopy(u0)
        v1 = deepcopy(v0)
        h1 = deepcopy(h0)
        adu = 0
        adv = 0
        adh = 0
        adHe = 0
        adhbcx = 0
        adhbcy = 0
        
        if step==0:
            init_bc(self,u1,v1,h1,He,hbcx,hbcy,t0=t)
        
        #######################
        # Boundary conditions #
        #######################
        adu_tmp,adv_tmp,adh_tmp,adHe_tmp,adhbcx_tmp,adhbcy_tmp = \
            obcs_adj(self,t,adu0,adv0,adh0,u1,v1,h1,He,hbcx,hbcy)
        adu += adu_tmp
        adv += adv_tmp
        adh += adh_tmp
        adHe += adHe_tmp
        adhbcx += adhbcx_tmp
        adhbcy += adhbcy_tmp
            
        #######################
        #   Time propagation  #
        #######################
        if step==0:
            # forward euler
            adku = self.dt*adu0
            adkv = self.dt*adv0
            adkh = self.dt*adh0
        else:
            # leap-frog
            adku = 2*self.dt*adu0
            adkv = 2*self.dt*adv0
            adkh = 2*self.dt*adh0
            
        #######################
        #  Right hand sides   #
        #######################
        adu_tmp,adh_tmp = self.rhs_v_adj(adkv)
        adu_tmp = self.u_on_v_adj(adu_tmp)
        adu += adu_tmp
        adh += adh_tmp
        adv_tmp,adh_tmp = self.rhs_u_adj(adku)
        adv_tmp = self.v_on_u_adj(adv_tmp)
        adv += adv_tmp
        adh += adh_tmp
        adu_tmp,adv_tmp,adHe_tmp = self.rhs_h_adj(t,adkh,u1,v1,He)
        adu += adu_tmp
        adv += adv_tmp
        adHe += adHe_tmp
    
        #######################
        #       Update        #
        #######################
        adu += self._adpu
        adv += self._adpv
        adh += self._adph
        
        #######################
        #       Init bc       #
        #######################
        if step==0:
            # Init
            adHe_tmp,adhbcx_tmp,adhbcy_tmp = init_bc_adj(
                self,adu,adv,adh,He,hbcx,hbcy,t0=t)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
        
        
        if step==0:                    
            adu += adu0 
            adv += adv0 
            adh += adh0 
            
        self._adpu = adu0
        self._adpv = adv0
        self._adph = adh0
    
        return adu,adv,adh,adHe,adhbcx,adhbcy
    
   
    
