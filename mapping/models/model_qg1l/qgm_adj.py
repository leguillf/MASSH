#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:22:53 2021

@author: leguillou
"""
from qgm_tgl import Qgm_tgl
import numpy as np

class Qgm_adj(Qgm_tgl):
    
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,upwind=3,upwind_adj=None,
                 g=9.81,f=1e-4,qgiter=1,qgiter_adj=None,diff=False,snu=None,
                 mdt=None,mdu=None,mdv=None):
        super().__init__(dx,dy,dt,SSH,c,upwind,upwind_adj,g,f,qgiter,qgiter_adj,diff,snu,mdt,mdu,mdv)
    
    def qrhs_adj(self,adrq,u,v,q,way):

        adrq[np.isnan(adrq)]=0.
        adrq[np.where((self.mask<=1))]=0
        
        adu=np.zeros((self.ny,self.nx))
        adv=np.zeros((self.ny,self.nx))
        adq=np.zeros((self.ny,self.nx))
        
        if not self.diff:

            uplus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            uminus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            vplus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            vminus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        
            induplus = np.where((uplus<0))
            uplus[induplus]=0
            induminus = np.where((uminus>=0))
            uminus[induminus]=0
            indvplus = np.where((vplus<0))
            vplus[indvplus]=0
            indvminus = np.where((vminus>=0))
            vminus[indvminus]=0
            
            
            aduplus,advplus,aduminus,advminus,adq = self._rq_adj(adrq,adq,
                                                 uplus,vplus,uminus,vminus,q)
            
            if self.mdt is not None:
                
                uplusbar = way*self.uplusbar
                uplusbar[np.where((uplusbar<0))] = 0
                vplusbar = way*self.vplusbar
                vplusbar[np.where((vplusbar<0))] = 0
                uminusbar = way*self.uminusbar
                uminusbar[np.where((uminusbar>0))] = 0
                vminusbar = way*self.vminusbar
                vminusbar[np.where((vminusbar>0))] = 0
                    
                _aduplus,_advplus,_aduminus,_advminus,adq = self._rq_adj(adrq,adq,
                         uplusbar,vplusbar,uminusbar,vminusbar,self.qbar)
                aduplus += _aduplus
                advplus += _advplus
                aduminus += _aduminus
                advminus += _advminus
            
            aduplus[induplus]=0
            aduminus[induminus]=0
            advplus[indvplus]=0
            advminus[indvminus]=0
            
            adu[2:-2,2:-2]=adu[2:-2,2:-2] + way*0.5*aduplus
            adu[2:-2,3:-1]=adu[2:-2,3:-1] + way*0.5*aduplus
            adu[2:-2,2:-2]=adu[2:-2,2:-2] + way*0.5*aduminus
            adu[2:-2,3:-1]=adu[2:-2,3:-1] + way*0.5*aduminus
            
            adv[2:-2,2:-2]=adv[2:-2,2:-2] + way*0.5*advplus
            adv[3:-1,2:-2]=adv[3:-1,2:-2] + way*0.5*advplus
            adv[2:-2,2:-2]=adv[2:-2,2:-2] + way*0.5*advminus
            adv[3:-1,2:-2]=adv[3:-1,2:-2] + way*0.5*advminus
        
            adv[2:-2,2:-2]=adv[2:-2,2:-2]-(self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/(2*self.dy[2:-2,2:-2])*way*0.5*(adrq[2:-2,2:-2])
            adv[3:-1,2:-2]=adv[3:-1,2:-2]-(self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/(2*self.dy[2:-2,2:-2])*way*0.5*(adrq[2:-2,2:-2])
        
        #diffusion
        if self.snu is not None:
            adq[2:-2,3:-1] += self.snu/(self.dx[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            adq[2:-2,1:-3] += self.snu/(self.dx[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            adq[2:-2,2:-2] += -2*self.snu/(self.dx[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            adq[3:-1,2:-2] += self.snu/(self.dy[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            adq[1:-3,2:-2] += self.snu/(self.dy[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            adq[2:-2,2:-2] += -2*self.snu/(self.dy[2:-2,2:-2]**2)*adrq[2:-2,2:-2]
            
        return adu, adv, adq
    
    def _rq_adj(self,adrq,adq,uplus,vplus,uminus,vminus,q):
        
        """
            main function for upwind schemes
        """
        
        if self.upwind_adj==1:
            return self._rq_adj1(adrq,adq,uplus,vplus,uminus,vminus,q)
        elif self.upwind_adj==2:
            return self._rq_adj2(adrq,adq,uplus,vplus,uminus,vminus,q)
        elif self.upwind_adj==3:
            return self._rq_adj3(adrq,adq,uplus,vplus,uminus,vminus,q)
    
    
    def _rq_adj1(self,adrq,adq,uplus,vplus,uminus,vminus,q):
        
        """
            1st-order upwind scheme
        """
        
        aduplus= -adrq[2:-2,2:-2]*1/(self.dx[2:-2,2:-2])*\
            (q[2:-2,2:-2]-q[2:-2,1:-3])
        
        aduminus= adrq[2:-2,2:-2]*1/(self.dx[2:-2,2:-2])*\
            (q[2:-2,2:-2]-q[2:-2,3:-1])
            
        advplus = -adrq[2:-2,2:-2]*1/(self.dy[2:-2,2:-2])*\
            (q[2:-2,2:-2]-q[1:-3,2:-2])
            
        advminus = adrq[2:-2,2:-2]*1/(self.dy[2:-2,2:-2])*\
            (q[2:-2,2:-2]-q[3:-1,2:-2])
            
        adq[2:-2,2:-2] = adq[2:-2,2:-2] \
            - uplus *1/(self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] \
            + uminus*1/(self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] \
            - vplus *1/(self.dy[2:-2,2:-2])* adrq[2:-2,2:-2] \
            + vminus*1/(self.dy[2:-2,2:-2])* adrq[2:-2,2:-2]
            
        adq[2:-2,1:-3] = adq[2:-2,1:-3] \
            + uplus *1/(self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] 
        adq[2:-2,3:-1] = adq[2:-2,3:-1] \
            - uminus *1/(self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] 
        adq[1:-3,2:-2] = adq[1:-3,2:-2] \
            + vplus *1/(self.dy[2:-2,2:-2])* adrq[2:-2,2:-2] 
        adq[3:-1,2:-2] = adq[3:-1,2:-2] \
            - vminus *1/(self.dy[2:-2,2:-2])* adrq[2:-2,2:-2] 
            
        return aduplus,advplus,aduminus,advminus,adq
    
    
    def _rq_adj2(self,adrq,adq,uplus,vplus,uminus,vminus,q):
        
        """
            2nd-order upwind scheme
        """
        
        aduplus= -adrq[2:-2,2:-2]*1/(2*self.dx[2:-2,2:-2])*\
        (3*q[2:-2,2:-2]-4*q[2:-2,1:-3]+q[2:-2,:-4])
        
        aduminus= adrq[2:-2,2:-2]*1/(2*self.dx[2:-2,2:-2])*\
        (q[2:-2,4:]-4*q[2:-2,3:-1]+3*q[2:-2,2:-2])
            
        advplus = -adrq[2:-2,2:-2]*1/(2*self.dy[2:-2,2:-2])*\
        (3*q[2:-2,2:-2]-4*q[1:-3,2:-2]+q[:-4,2:-2])
            
        advminus = adrq[2:-2,2:-2]*1/(2*self.dy[2:-2,2:-2])*\
        (q[4:,2:-2]-4*q[3:-1,2:-2]+3*q[2:-2,2:-2])
            
        adq[2:-2,2:-2] = adq[2:-2,2:-2] \
            - uplus *1/(2*self.dx[2:-2,2:-2])* 3*adrq[2:-2,2:-2] \
            + uminus*1/(2*self.dx[2:-2,2:-2])* 3*adrq[2:-2,2:-2] \
            - vplus *1/(2*self.dy[2:-2,2:-2])* 3*adrq[2:-2,2:-2] \
            + vminus*1/(2*self.dy[2:-2,2:-2])* 3*adrq[2:-2,2:-2]
            
        adq[2:-2,1:-3] = adq[2:-2,1:-3] \
            + uplus *1/(2*self.dx[2:-2,2:-2])* 4*adrq[2:-2,2:-2] 
            
        adq[2:-2,:-4] = adq[2:-2,:-4] \
            - uplus *1/(2*self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] 
        
        adq[2:-2,4:] = adq[2:-2,4:] \
            + uminus *1/(2*self.dx[2:-2,2:-2])* adrq[2:-2,2:-2] 
            
        adq[2:-2,3:-1] = adq[2:-2,3:-1] \
            - uminus *1/(2*self.dx[2:-2,2:-2])* 4*adrq[2:-2,2:-2] 
        
        adq[1:-3,2:-2] = adq[1:-3,2:-2] \
            + vplus *1/(2*self.dy[2:-2,2:-2])* 4*adrq[2:-2,2:-2] 
            
        adq[:-4,2:-2] = adq[:-4,2:-2] \
            - vplus *1/(2*self.dy[2:-2,2:-2])* adrq[2:-2,2:-2] 
            
        adq[4:,2:-2] = adq[4:,2:-2] \
            + vminus *1/(2*self.dy[2:-2,2:-2])* adrq[2:-2,2:-2] 
        
        adq[3:-1,2:-2] = adq[3:-1,2:-2] \
            - vminus *1/(2*self.dy[2:-2,2:-2])* 4*adrq[2:-2,2:-2] 
    
        return aduplus,advplus,aduminus,advminus,adq
    
    
    def _rq_adj3(self,adrq,adq,uplus,vplus,uminus,vminus,q):
        
        """
            3rd-order upwind scheme
        """
        
        aduplus= -adrq[2:-2,2:-2]*1/(6*self.dx[2:-2,2:-2])*\
        (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- \
         6*q[2:-2,1:-3]+q[2:-2,:-4] )
        
        aduminus= adrq[2:-2,2:-2]*1/(6*self.dx[2:-2,2:-2])*\
        (q[2:-2,4:]-6*q[2:-2,3:-1]+ \
         3*q[2:-2,2:-2]+2*q[2:-2,1:-3] ) 
            
        advplus = -adrq[2:-2,2:-2]*1/(6*self.dy[2:-2,2:-2])*\
        (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-  \
         6*q[1:-3,2:-2]+q[:-4,2:-2])
            
        advminus = adrq[2:-2,2:-2]*1/(6*self.dy[2:-2,2:-2])*\
        (q[4:,2:-2]-6*q[3:-1,2:-2]+ \
         3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
            
        adq[2:-2,2:-2]=adq[2:-2,2:-2] - uplus*1/(6*self.dx[2:-2,2:-2])*\
        (3*adrq[2:-2,2:-2])\
        + uminus*1/(6*self.dx[2:-2,2:-2])*\
        (3*adrq[2:-2,2:-2])\
        - vplus*1/(6*self.dy[2:-2,2:-2])*(3*adrq[2:-2,2:-2])\
        + vminus*1/(6*self.dy[2:-2,2:-2])*(3*adrq[2:-2,2:-2])
    
        adq[2:-2,3:-1]=adq[2:-2,3:-1] - uplus*1/(6*self.dx[2:-2,2:-2])*\
        (2*adrq[2:-2,2:-2])\
        + uminus*1/(6*self.dx[2:-2,2:-2])*\
        (-6*adrq[2:-2,2:-2])
    
        adq[3:-1,2:-2]=adq[3:-1,2:-2] \
        - vplus*1/(6*self.dy[2:-2,2:-2])*(2*adrq[2:-2,2:-2])\
        + vminus*1/(6*self.dy[2:-2,2:-2])*(-6*adrq[2:-2,2:-2])
    
        adq[2:-2,1:-3]=adq[2:-2,1:-3] - uplus*1/(6*self.dx[2:-2,2:-2])*\
        (-6*adrq[2:-2,2:-2])\
        + uminus*1/(6*self.dx[2:-2,2:-2])*\
        (2*adrq[2:-2,2:-2])
        adq[1:-3,2:-2]=adq[1:-3,2:-2] \
        - vplus*1/(6*self.dy[2:-2,2:-2])*(-6*adrq[2:-2,2:-2])\
        + vminus*1/(6*self.dy[2:-2,2:-2])*(2*adrq[2:-2,2:-2])
    
        adq[2:-2,4:]=adq[2:-2,4:] \
        + uminus*1/(6*self.dx[2:-2,2:-2])*\
        (adrq[2:-2,2:-2])
    
        adq[4:,2:-2]=adq[4:,2:-2] \
        + vminus*1/(6*self.dy[2:-2,2:-2])*(adrq[2:-2,2:-2])
    
        adq[2:-2,:-4]=adq[2:-2,:-4] - uplus*1/(6*self.dx[2:-2,2:-2])*\
        (adrq[2:-2,2:-2])
    
        adq[:-4,2:-2]=adq[:-4,2:-2] \
        - vplus*1/(6*self.dy[2:-2,2:-2])*(adrq[2:-2,2:-2])
            
        return aduplus,advplus,aduminus,advminus,adq
    
    
    def h2uv_adj(self,adu,adv):

        adh = np.zeros((self.ny,self.nx))
        
        adu[self.mask<=1] = 0
        adv[self.mask<=1] = 0
    
        adh[2:,:-1] += -self.g/self.f0[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adh[2:,1:] += -self.g/self.f0[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adh[:-2,1:] += +self.g/self.f0[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        adh[:-2,:-1] += +self.g/self.f0[1:-1,1:]/(4*self.dy[1:-1,1:]) * adu[1:-1,1:]
        
        adh[1:,2:] += self.g/self.f0[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adh[:-1,2:] += self.g/self.f0[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adh[:-1,:-2] += -self.g/self.f0[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        adh[1:,:-2] += -self.g/self.f0[1:,1:-1]/(4*self.dx[1:,1:-1]) * adv[1:,1:-1]
        
        return adh
    
    
    
    def h2pv_adj(self,adq):
        
        adh = np.zeros((self.ny,self.nx))
        adq_tmp = +adq
        
        ind = np.where((self.mask==1))
        adh[ind] = -self.g*self.f0[ind]/(self.c[ind]**2) * adq_tmp[ind]
        adq_tmp[ind] = 0
        
        ind = np.where((self.mask==0))
        adh[ind] = 0
        adq_tmp[ind] = 0

        adh[2:,1:-1] += self.g/self.f0[1:-1,1:-1]/self.dy[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[:-2,1:-1] += self.g/self.f0[1:-1,1:-1]/self.dy[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[1:-1,1:-1] += -2*self.g/self.f0[1:-1,1:-1]/self.dy[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[1:-1,2:] += self.g/self.f0[1:-1,1:-1]/self.dx[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[1:-1,:-2] += self.g/self.f0[1:-1,1:-1]/self.dx[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[1:-1,1:-1] += -2*self.g/self.f0[1:-1,1:-1]/self.dx[1:-1,1:-1]**2 * adq_tmp[1:-1,1:-1]
        adh[1:-1,1:-1] += -self.g*self.f0[1:-1,1:-1]/(self.c[1:-1,1:-1]**2) * adq_tmp[1:-1,1:-1]
                
        return adh
    
    def h2pv_1d_adj(self,adq1d,a,b):
            
        adh1d = np.zeros((self.np,))
        adq1d_tmp = +adq1d
        # q1d[self.vp1] = h1d[self.vp1]
        adh1d[self.vp1] += adq1d_tmp[self.vp1]
        adq1d_tmp[self.vp1] = 0
        
        # q1d[self.vp2] = a[self.vp2]*\
        #     ((h1d[self.vp2e]+h1d[self.vp2w]-2*h1d[self.vp2])/self.dx1d[self.vp2]**2 +\
        #      (h1d[self.vp2n]+h1d[self.vp2s]-2*h1d[self.vp2])/self.dy1d[self.vp2]**2) +\
        #         b[self.vp2] * h1d[self.vp2]
        adh1d[self.vp2e] += a[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2w] += a[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += -2*a[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2n] += a[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2s] += a[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += -2*a[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += b[self.vp2] * adq1d_tmp[self.vp2]
        
        adq1d_tmp[self.vp2] = 0
    
        return adh1d
    
    
    def alpha_adj(self,adalpha,p,gg,aaa,bbb):
        
        tmp = np.dot(p,self.h2pv_1d(p,aaa,bbb))
        
        if tmp!=0. : 
            # dalpha = -((np.dot(dp,gg)+np.dot(p,dgg))*tmp - dtmp*np.dot(p,gg))/tmp**2
            adp = -gg/tmp * adalpha
            adgg = -p/tmp * adalpha
            adtmp = +np.dot(p,gg)/tmp**2 * adalpha
        else:
            adp = adgg = np.zeros((self.np,))
            adtmp = 0.
        
        # dtmp = np.dot(dp,self.h2pv_1d(p,aaa,bbb)) + np.dot(p,self.h2pv_1d(dp,aaa,bbb))
        adp += self.h2pv_1d(p,aaa,bbb) * adtmp
        adp += self.h2pv_1d_adj(adtmp*p,aaa,bbb)
        
        return adp,adgg
            
    
    def beta_adj(self,adbeta,r,rnew):
        
        norm_rnew = self.norm(rnew)
        norm_r = self.norm(r)
        adr = -2*(norm_rnew/norm_r**2)**2 * r * adbeta
        adrnew = 2/norm_r**2 * rnew * adbeta
        
        return adr,adrnew
    
    
    def pv2h_adj(self,adh,q,hg):
        ######################
        # Forward iterations
        ######################
        x = +hg[self.indi,self.indj]
        q1d = q[self.indi,self.indj]
        
        aaa = self.g/self.f01d
        bbb = - self.g*self.f01d / self.c1d**2
        ccc = +q1d
        aaa[self.vp1] = 0
        bbb[self.vp1] = 1
        ccc[self.vp1] = x[self.vp1]  ##boundary condition
        
        gg = self.h2pv_1d(x,aaa,bbb) - ccc
        p = -gg
        
        gg_list = [gg]
        p_list = [p]
        alpha_list = []
        a1_list = []
        a2_list = []
        beta_list = []
        for itr in range(self.qgiter-1): 
            a1 = np.dot(gg,gg)
            alpha = self.alpha(p,gg,aaa,bbb)
            x = x + alpha*p
            gg = self.h2pv_1d(x,aaa,bbb) - ccc
            a2 = np.dot(gg,gg)
            if a1!=0:
                beta = a2/a1
            else: 
                beta = 1.
            p = -gg + beta*p
            
            alpha_list.append(alpha)
            gg_list.append(gg)
            p_list.append(p)
            a1_list.append(a1)
            a2_list.append(a2)
            beta_list.append(beta)
        val1 = -np.dot(p,gg)
        val2 = np.dot(p,self.h2pv_1d(p,aaa,bbb))
        if (val2==0.): 
            s=1.
        else: 
            s=val1/val2
            
        ######################
        # Adjoint iterations
        ######################
        # Initialize adjoint variable
        adq = np.zeros((self.ny,self.nx))
        adhg = np.zeros((self.ny,self.nx))
        
        adh_tmp = +adh
        
        # back to 2D
        #dh[self.indi,self.indj] = dx1[:]
        adx1 = +adh_tmp[self.indi,self.indj]
        adh_tmp[self.indi,self.indj] = 0.
        
        # dx1 = dx + ds*p + s*dp
        adx = +adx1
        ads = p*adx1
        adp = s*adx1
        adx1 *= 0.
        
        if (val2==0.): 
            # ds = 0.
            ads = 0.
            adval2 = adval1 = 0
        else: 
            #ds = (dval1*val2 - val1*dval2)/val2**2.
            adval1 = np.sum(val2/val2**2. * ads)
            adval2 = np.sum(-(val1/val2**2.) * ads)
            ads = 0.
        
        # dval2 = np.dot(dp,self.h2pv_1d(p,aaa,bbb)) + np.dot(p,self.h2pv_1d(dp,aaa,bbb))
        adp += adval2 * self.h2pv_1d(p,aaa,bbb)
        adp += adval2 * self.h2pv_1d_adj(p,aaa,bbb)
        adval2 = 0.
        
        # dval1 = -np.dot(dp,gg)-np.dot(p,dgg)
        adp += -gg*adval1
        adgg = -p*adval1
        adval1 = 0.
        
        adbeta = 0.
        ada1 = 0.
        ada2 = 0.
        adccc = 0.
        adalpha = 0.
        for itr in reversed(range(self.qgiter-1)): 
            # dgg = +dggnew
            adggnew = +adgg
            adgg *= 0.
            #dp = +dpnew
            adpnew = +adp
            adp *= 0
            #dx = +dxnew 
            adxnew = +adx
            adx *= 0
            
            # dpnew = -dggnew + dbeta*p_list[itr] + beta_list[itr]*dp
            adggnew += -adpnew
            adbeta += np.sum(p_list[itr]*adpnew)
            adp += beta_list[itr]*adpnew
            adpnew *= 0
            
            if a1_list[itr]!=0:
                # dbeta = (da2*a1_list[itr]-a2_list[itr]*da1)/a1_list[itr]**2
                ada2 += a1_list[itr]/a1_list[itr]**2. * adbeta
                ada1 += -a2_list[itr]/a1_list[itr]**2. * adbeta
            adbeta = 0.
            
            # da2 = 2*np.dot(dggnew,gg_list[itr+1])
            adggnew += 2.*gg_list[itr+1] * ada2 
            ada2 = 0.
            
            # dggnew = self.h2pv_1d(dxnew,aaa,bbb) - dccc
            adxnew += self.h2pv_1d_adj(adggnew,aaa,bbb)
            adccc += -adggnew
            adggnew *= 0.
            
            # dxnew = dx + dalpha*p_list[itr] + alpha_list[itr]*dp
            adalpha += np.sum(p_list[itr] * adxnew)
            adp += alpha_list[itr] * adxnew
            adx += adxnew
            adxnew *= 0
            
            # dalpha = self.alpha_tgl(dp,dgg,p_list[itr],gg_list[itr],aaa,bbb)
            adp_tmp,adgg_tmp = self.alpha_adj(adalpha,p_list[itr],gg_list[itr],aaa,bbb)
            adp += adp_tmp
            adgg += adgg_tmp
            adalpha = 0.
            
            # da1 = 2*np.dot(dgg,gg_list[itr]) 
            adgg += 2.*gg_list[itr] * ada1
            ada1 = 0
        
        # dp = -dgg
        adgg += -adp
        adp *= 0.
        
        # dgg = self.h2pv_1d(dx,aaa,bbb) - dccc
        adx += self.h2pv_1d_adj(adgg,aaa,bbb)
        adccc += -adgg
        adgg *= 0.
        
        # dccc[self.vp1] = dx[self.vp1]  ## boundary condition  
        adx[self.vp1] += adccc[self.vp1]
        adccc[self.vp1] *= 0.
        
        # dccc = +dq1d
        adq1d = +adccc
        
        # dq1d = dq[self.indi,self.indj]
        adq[self.indi,self.indj] = +adq1d
        adq1d *= 0.
        
        # dx = +dhg[self.indi,self.indj]
        adhg[self.indi,self.indj] = +adx
        adx *= 0.
        
        return adq,adhg
    
    
    
        
    def pv2h_adj_2(self,adh,q,hg):
        
        #####################
        # Current trajectory
        #####################
        q1d = +q[self.indi,self.indj]
        hg1d = +hg[self.indi,self.indj]
        a = self.g/self.f01d
        b = -self.g*self.f01d/(self.c1d)**2
        a[self.vp1] = 0
        b[self.vp1] = 1
        q1d[self.vp1] = hg1d[self.vp1] # Boundary conditions
        r = +q1d - self.h2pv_1d(hg1d,a,b)
        d = +r
        alpha = self.alpha(r,d,a,b)
        alpha_list = [alpha]
        beta_list = []
        r_list = [r]
        d_list = [d]
        if self.qgiter_adj>1:
            # Loop
            for itr in range(self.qgiter):
                # Update direction
                r[self.vp1] = hg1d[self.vp1] # Boundary conditions
                rnew = r - alpha * self.h2pv_1d(d,a,b)
                beta = self.beta(r,rnew)
                r = +rnew
                d = r + beta * d
                alpha = self.alpha(r,d,a,b)
                # Append to lists
                alpha_list.append(alpha)
                r_list.append(r)
                d_list.append(d)
                beta_list.append(beta)
        
        #####################
        # Adjoint computation 
        #####################
        
        # Initialize adjoint variable
        adalpha = 0
        adbeta = 0
        adhg1d = np.zeros((self.np,))
        adr = np.zeros((self.np,))
        adrnew = np.zeros((self.np,))
        add = np.zeros((self.np,))
        addnew = np.zeros((self.np,))
        
        # back to 2D
        #dh[self.indi,self.indj] = dh1d[:]
        adh1d = +adh[self.indi,self.indj]
        #adh[self.indi,self.indj] = 0
    
        if self.qgiter_adj>1:
            # Adjoint Loop
            for itr in reversed(range(self.qgiter_adj)):
                # Update SSH
                # dh1d = dhg1d + dalpha*d_list[itr+1] + alpha_list[itr+1]*dd
                adhg1d += adh1d
                adalpha += np.sum(d_list[itr+1]*adh1d)
                add += alpha_list[itr+1]*adh1d
                adh1d = np.zeros((self.np,))
                
                # Compute new direction
                
                # dd = +ddnew
                addnew += add
                add = np.zeros((self.np,))
                
                # dalpha = self.alpha_tgl(dr,ddnew,r_list[itr+1],d_list[itr+1],a,b)
                adr_tmp,addnew_tmp = self.alpha_adj(adalpha,r_list[itr+1],d_list[itr+1],a,b)
                adr += adr_tmp
                addnew += addnew_tmp
                adalpha = 0
                
                # ddnew = dr + dbeta * d_list[itr] + beta_list[itr] * dd
                adr += addnew
                adbeta += np.sum(d_list[itr]*addnew)
                add += beta_list[itr] * addnew
                addnew = np.zeros((self.np,))
                
                # Compute beta
                
                # dr = +drnew
                adrnew += adr
                adr = np.zeros((self.np,))
                
                # dbeta = self.beta_tgl(dr,drnew,r_list[itr],r_list[itr+1])
                adr_tmp,adrnew_tmp = self.beta_adj(adbeta,r_list[itr],r_list[itr+1])
                adr += adr_tmp
                adrnew += adrnew_tmp
                adbeta = 0
                
                # drnew = dr - (dalpha*self.h2pv_1d(d_list[itr],a,b) +
                #               alpha_list[itr]*self.h2pv_1d(dd,a,b))
                adr += adrnew
                adalpha += -np.sum(self.h2pv_1d(d_list[itr],a,b)*adrnew)
                add += -self.h2pv_1d_adj(alpha_list[itr]*adrnew,a,b)
                adrnew = np.zeros((self.np,))
                
                #dr[self.vp1] = dhg1d[self.vp1] # Boundary conditions
                adhg1d[self.vp1] += adr[self.vp1]
                adr[self.vp1] = 0
                
                # Update guess value
                # dhg1d = +dh1d    
                adh1d += adhg1d
                adhg1d = np.zeros((self.np,))
                
                
        # dh1d = dhg1d + dalpha*d_list[0] + alpha_list[0]*dd
        adhg1d += adh1d
        adalpha += np.sum(d_list[0]*adh1d)
        add += alpha_list[0]*adh1d
        adh1d = np.zeros((self.np,))
            
        # dalpha = self.alpha_tgl(dr,dd,r,d,a,b)
        adr_tmp,add_tmp = self.alpha_adj(adalpha,r_list[0],d_list[0],a,b)
        adr += adr_tmp
        add += add_tmp
        adalpha = 0
    
        # dd = +dr
        adr += add
        add = np.zeros((self.np,))
        
        # dr = +dq1d-self.h2pv_1d(dhg1d,a,b)
        adq1d = +adr
        adhg1d += -self.h2pv_1d_adj(adr,a,b)
        adr = np.zeros((self.np,))
        
        # dq1d[self.vp1] = dhg1d[self.vp1] # Boundary conditions
        adhg1d[self.vp1] += adq1d[self.vp1]
        adq1d[self.vp1] = 0
        
        #dhg1d = +dhg[self.indi,self.indj]
        adhg = np.zeros((self.ny,self.nx))
        adhg[:,:] = np.NAN
        adhg[self.indi,self.indj] = adhg1d[:]
        adhg1d = np.zeros((self.np,))
        
        # dq1d = +dq[self.indi,self.indj]
        adq = np.zeros((self.ny,self.nx))
        adq[:,:] = np.NAN
        adq[self.indi,self.indj] = adq1d[:]
        adq1d = np.zeros((self.np,))
        
        
        return adq,adhg

    
    
    def step_adj(self,adh1,h0,addphidt=None,dphidt=None,way=1):
        
        azeros = +adh1*0
        
        if np.all(h0[self.mask>=1]==0):
            return adh1
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq
        if dphidt is not None:
            q1 += self.dt*dphidt
            
        # 5/ q-->h
        adq1,adh0 = self.pv2h_adj(adh1,q1,+h0)
        adh1 = +azeros
        
        # 4/ Time increment
        adq0 = +adq1
        adrq = self.dt * adq1
        if addphidt is not None:
            addphidt += self.dt * adq1
        adq1 = +azeros
         
        # 3/ (u,v,q)-->rq
        adu0,adv0,adq_tmp = self.qrhs_adj(adrq,u,v,qb0,way)
        adq0 += adq_tmp
        adrq = +azeros
        
        # 2/ h-->(u,v)
        adh0 += self.h2uv_adj(adu0,adv0)
        adu0 = +azeros
        adv0 = +azeros
            
        # 1/ h-->q
        adh0 += self.h2pv_adj(adq0)
        
        return adh0
        
        
    def adjtest_h2pv_1d(self):
        a = np.random.random((self.np,))
        b = np.random.random((self.np,))
        
        h1d = np.random.random((self.np,))
        adq1d = np.random.random((self.np,))
        
        q1d = self.h2pv_1d(h1d,a,b)
        adh1d = self.h2pv_1d_adj(adq1d,a,b)
        
        
        ps1 = np.inner(q1d,adq1d)
        ps2 = np.inner(h1d,adh1d)
        
        print('test h2pv_1d:', ps1/ps2)
        
    def adjtest_pv2h(self):
        
        q = np.random.random((self.ny,self.nx))
        hg = np.random.random((self.ny,self.nx))
        
        dq = np.random.random((self.ny,self.nx))
        dhg = np.random.random((self.ny,self.nx))
        adh = np.random.random((self.ny,self.nx))
        
        dh = self.pv2h_tgl(dq,dhg,q,hg)
        adq,adhg = self.pv2h_adj(adh,q,hg)
    
        
        ps1 = np.inner(dh.ravel(),adh.ravel())
        ps2 = np.inner(np.concatenate((dq.ravel(),dhg.ravel())),np.concatenate((adq.ravel(),adhg.ravel())))

        print('test pv2h:', ps1/ps2)
        
        
        
    def adjtest_pv2h_2(self):
        
        h = np.random.random((self.ny,self.nx))
        adh = np.random.random((self.ny,self.nx))
        
        
        adqguess = - self.c**2./(self.g*self.f0) * h
        
        adq = self.pv2h_2(adh,adqguess) 
        
        
        q = self.h2pv_2(h)
        
        ps1 = np.inner(q.ravel(),adq.ravel())
        ps2 = np.inner(h.ravel(),adh.ravel())
        
        print('test pv2h_2:', ps1/ps2)
        
        
    def adjtest_alpha(self):
        a = np.random.random((self.np,))
        b = np.random.random((self.np,))
        
        r = np.random.random((self.np,))
        d = np.random.random((self.np,))
        
        dr = np.random.random((self.np,))
        dd = np.random.random((self.np,))
        
        adalpha = np.random.random()
        
        
        dalpha = self.alpha_tgl(dr,dd,r,d,a,b)
        
        adr,add = self.alpha_adj(adalpha,r,d,a,b)
    
        
        ps1 = np.inner(dalpha,adalpha)
        ps2 = np.inner(np.concatenate((dr,dd)),np.concatenate((adr,add)))

        print('test alpha:', ps1/ps2)
        


if __name__ == "__main__":    
    
    ny,nx = 100,150
    dx = 10e3 * np.ones((ny,nx))
    dy = 12e3 * np.ones((ny,nx))
    dt = 300
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm_adj(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=5)

    qgm.adjtest_pv2h()
    qgm.adjtest_h2pv_1d()
    qgm.adjtest_alpha()
    # Current trajectory
    SSH0 = 1e-2*np.random.random((ny,nx))

    # Perturbation
    dSSH0 = 1e-2*np.random.random((ny,nx))

    # Adjoint
    adSSH0 = 1e-2*np.random.random((ny,nx))
    
    nstep = 3
    
    SSH1 = +SSH0
    traj = [SSH1]
    if nstep>1:
        for i in range(nstep):
            SSH1 = qgm.step(SSH1)
            traj.append(SSH1)
            
    # Run TLM
    print('tlm')
    dSSH1 = +dSSH0
    for i in range(nstep):
        dSSH1 = qgm.step_tgl(dSSH1,traj[i])

    # Run ADJ
    print('\nadj')
    adSSH1 = +adSSH0 
    for i in reversed(range(nstep)):
        adSSH1 = qgm.step_adj(adSSH1,traj[i])
    
    
    ps1 = np.inner(dSSH1.ravel(),adSSH0.ravel())
    ps2 = np.inner(dSSH0.ravel(),adSSH1.ravel())
        
    print('\ntest:',ps1/ps2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    