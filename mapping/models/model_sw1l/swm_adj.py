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
    #                 Auxillary functions for step_adj_rk*                    #
    ###########################################################################
    
    def kv_adj(self,kv2,adu,adh,ku1=None,kh1=None,c=1):
    
        adu_tmp,adh_tmp = self.rhs_v_adj(kv2)
        adu_tmp = self.u_on_v_adj(adu_tmp)
        adu += adu_tmp*self.dt
        adh += adh_tmp*self.dt
        kv2 = 0.
        res = [kv2,adu,adh]
        if ku1 is not None:
            ku1 += c*adu_tmp*self.dt
            res.append(ku1)
        if kh1 is not None:
            kh1 += c*adh_tmp*self.dt
            res.append(kh1)
        
        return res
    
    def ku_adj(self,ku2,adv,adh,kv1=None,kh1=None,c=1):
    
        adv_tmp,adh_tmp = self.rhs_u_adj(ku2)
        adv_tmp = self.v_on_u_adj(adv_tmp)
        adv += adv_tmp*self.dt
        adh += adh_tmp*self.dt
        ku2 = 0.
        res = [ku2,adv,adh]
        if kv1 is not None:
            kv1 += c*adv_tmp*self.dt
            res.append(kv1)
        if kh1 is not None:
            kh1 += c*adh_tmp*self.dt
            res.append(kh1)
        
        return res
    
    def kh_adj(self,t,kh2,adu,adv,adHe,u,v,He,ku1=None,kv1=None,c=1):
        adu_tmp,adv_tmp,adHe_tmp = self.rhs_h_adj(t,kh2,u,v,He)
        adu += adu_tmp*self.dt
        adv += adv_tmp*self.dt
        adHe += adHe_tmp*self.dt
        kh2 = 0
        res = [kh2,adu,adv,adHe]
        if ku1 is not None:
            ku1 += c*adu_tmp*self.dt
            res.append(ku1)
        if kv1 is not None:
            kv1 += c*adv_tmp*self.dt
            res.append(kv1)
        
        return res
    
    
    ###########################################################################
    #                            One time step                                #
    ###########################################################################

    def step_euler_adj(self,t,adu0,adv0,adh0, u0,v0,h0, 
                       He=None,hbcx=None,hbcy=None,first=False):
        
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
        
        if first:
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
        if first:
            adHe_tmp,adhbcx_tmp,adhbcy_tmp = init_bc_adj(
                self,adu,adv,adh,He,hbcx,hbcy,t0=t)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
        
        
        return adu,adv,adh,adHe,adhbcx,adhbcy
    
        
    def step_lf_adj(self,t,adu0,adv0,adh0, u0,v0,h0, 
                       He=None,hbcx=None,hbcy=None,first=False):
        
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
        
        if first:
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
        if first:
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
        if first:
            # Init
            adHe_tmp,adhbcx_tmp,adhbcy_tmp = init_bc_adj(
                self,adu,adv,adh,He,hbcx,hbcy,t0=t)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
        
        
        if first:                    
            adu += adu0 
            adv += adv0 
            adh += adh0 
            
        self._adpu = adu0
        self._adpv = adv0
        self._adph = adh0
    
        return adu,adv,adh,adHe,adhbcx,adhbcy
    
   
    def step_rk4_adj(self,t,adu0,adv0,adh0, u0,v0,h0, 
                       He=None,hbcx=None,hbcy=None,first=False):
        
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
        
        if first:
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
        #   Time propagation  #
        #######################
        # Update
        kh1_ad = 1/6 * adh0
        kh2_ad = 1/3 * adh0
        kh3_ad = 1/3 * adh0
        kh4_ad = 1/6 * adh0
        #adh0 = 0
        kv1_ad = 1/6 * adv0
        kv2_ad = 1/3 * adv0
        kv3_ad = 1/3 * adv0
        kv4_ad = 1/6 * adv0
        #adv0 = 0
        ku1_ad = 1/6 * adu0
        ku2_ad = 1/3 * adu0
        ku3_ad = 1/3 * adu0
        ku4_ad = 1/6 * adu0
        #adu0 = 0
        
        #######################
        #  Right hand sides   #
        #######################
        # kh4_ad
        kh4_ad,adu_incr,adv_incr,adHe,ku3_ad,kv3_ad = self.kh_adj(
            t,kh4_ad,adu,adv,adHe,u1+ku3,v1+kv3,He,ku3_ad,kv3_ad)
        
        # kv4_ad
        kv4_ad,adu,adh,ku3_ad,kh3_ad = self.kv_adj(
            kv4_ad,adu,adh,ku3_ad,kh3_ad)
        
        # ku4_ad
        ku4_ad,adv,adh,kv3_ad,kh3_ad = self.ku_adj(
            ku4_ad,adv,adh,kv3_ad,kh3_ad)
        
        # kh3_ad
        kh3_ad,adu,adv,adHe,ku2_ad,kv2_ad = self.kh_adj(
            t,kh3_ad,adu,adv,adHe,u1+0.5*ku2,v1+0.5*kv2,He,ku2_ad,kv2_ad,1/2)
        
        # kv3_ad
        kv3_ad,adu,adh,ku2_ad,kh2_ad = self.kv_adj(
            kv3_ad,adu,adh,ku2_ad,kh2_ad,1/2)
        
        # ku3_ad
        ku3_ad,adv,adh,kv2_ad,kh2_ad = self.ku_adj(
            ku3_ad,adv,adh,kv2_ad,kh2_ad,1/2)
        
        # kh2_ad
        kh2_ad,adu,adv,adHe,ku1_ad,kv1_ad = self.kh_adj(
            t,kh2_ad,adu,adv,adHe,u1+0.5*ku1,v1+0.5*kv1,He,ku1_ad,kv1_ad,1/2)
        
        # kv2_ad
        kv2_ad,adu,adh,ku1_ad,kh1_ad = self.kv_adj(
            kv2_ad,adu,adh,ku1_ad,kh1_ad,1/2)
        
        # ku2_ad
        ku2_ad,adv,adh,kv1_ad,kh1_ad = self.ku_adj(
            ku2_ad,adv,adh,kv1_ad,kh1_ad,1/2)
        
        # kh1_ad
        kh1_ad,adu,adv,adHe = self.kh_adj(
            t,kh1_ad,adu,adv,adHe,u1,v1,He)
        
        # kv1_ad
        kv1_ad,adu,adh = self.kv_adj(
            kv1_ad,adu,adh,None,None)
        
        # ku1_ad
        ku1_ad,adv,adh = self.ku_adj(
            ku1_ad,adv,adh,None,None)
    
        #######################
        #       Update        #
        #######################
        adu += adu0 
        adv += adv0 
        adh += adh0 
        
        #######################
        #       Init bc       #
        #######################
        if first:
            adHe_tmp,adhbcx_tmp,adhbcy_tmp = init_bc_adj(
                self,adu,adv,adh,He,hbcx,hbcy,t0=t)
            adHe += adHe_tmp
            adhbcx += adhbcx_tmp
            adhbcy += adhbcy_tmp
            
        return adu,adv,adh,adHe,adhbcx,adhbcy
