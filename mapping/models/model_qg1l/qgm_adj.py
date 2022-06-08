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
    
    def h2pv_1d_adj(self,adq1d):
            
        adh1d = np.zeros((self.np,))
        adq1d_tmp = +adq1d
        # q1d[self.vp1] = h1d[self.vp1]
        adh1d[self.vp1] += adq1d_tmp[self.vp1]
        adq1d_tmp[self.vp1] = 0
        
        adh1d[self.vp2e] += self.aaa[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2w] += self.aaa[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += -2*self.aaa[self.vp2]/self.dx1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2n] += self.aaa[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2s] += self.aaa[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += -2*self.aaa[self.vp2]/self.dy1d[self.vp2]**2 * adq1d_tmp[self.vp2]
        adh1d[self.vp2] += self.bbb[self.vp2] * adq1d_tmp[self.vp2]
        
        adq1d_tmp[self.vp2] = 0
    
        return adh1d
    
    
    def alpha_adj(self,adalpha,p,gg):
        
        tmp = np.dot(p,self.h2pv_1d(p))
        
        if tmp!=0. : 
            # dalpha = -((np.dot(dp,gg)+np.dot(p,dgg))*tmp - dtmp*np.dot(p,gg))/tmp**2
            adp = -gg/tmp * adalpha
            adgg = -p/tmp * adalpha
            adtmp = +np.dot(p,gg)/tmp**2 * adalpha
        else:
            adp = adgg = np.zeros((self.np,))
            adtmp = 0.
        
        # dtmp = np.dot(dp,self.h2pv_1d(p,aaa,bbb)) + np.dot(p,self.h2pv_1d(dp,aaa,bbb))
        adp += self.h2pv_1d(p) * adtmp
        adp += self.h2pv_1d_adj(adtmp*p)
        
        return adp,adgg
            

    def pv2h_adj(self,adh,q,hg):
        ######################
        # Forward iterations
        ######################
        x = +hg[self.indi,self.indj]
        q1d = q[self.indi,self.indj]

        ccc = +q1d
        
        r = self.h2pv_1d(x) - ccc
        r[self.vp1] = 0 ## boundary condition   
        d = -r
        alpha = self.alpha(d,r)
        xnew = x + alpha*d
        
        alpha_list = [alpha]
        a1_list = []
        a2_list = []
        beta_list = []
        r_list = [r]
        d_list = [d]
        if self.qgiter_adj>1:
            for itr in range(self.qgiter_adj): 
                # Update guess value
                x = +xnew
                
                # Compute beta
                rnew = self.h2pv_1d(xnew) - ccc
                rnew[self.vp1] = 0 ## boundary condition   
                a1 = np.dot(r,r)
                a2 = np.dot(rnew,rnew)
                if a1!=0:
                    beta = a2/a1
                else: 
                    beta = 1.
                r = +rnew
                
                # Compute new direction
                dnew = -rnew + beta*d
                d = +dnew
                
                # Update state
                alpha = self.alpha(d,r)
                xnew = x + alpha*d
                
                alpha_list.append(alpha)
                r_list.append(r)
                d_list.append(d)
                beta_list.append(beta)
                a1_list.append(a1)
                a2_list.append(a2)
                    
            
        ######################
        # Adjoint iterations
        ######################
        # Initialize adjoint variable
        adq = np.zeros((self.ny,self.nx))
        adhg = np.zeros((self.ny,self.nx))
        adalpha = 0
        adbeta = 0
        ada1 = 0
        ada2 = 0
        adx = np.zeros((self.np,))
        adxnew = np.zeros((self.np,))
        adr = np.zeros((self.np,))
        adrnew = np.zeros((self.np,))
        add = np.zeros((self.np,))
        addnew = np.zeros((self.np,))
        adccc = np.zeros((self.np,))
        adq1d = np.zeros((self.np,))
        

        # back to 2D
        #dh[self.indi,self.indj] = +dxnew[:]
        adxnew += +adh[self.indi,self.indj]
        
        if self.qgiter_adj>1:        
            for itr in reversed(range(self.qgiter_adj)): 
                # 4. Update state
                # dxnew = dx + dalpha*d_list[itr+1] + alpha_list[itr+1]*dd
                adalpha += np.sum(d_list[itr+1] * adxnew)
                add += alpha_list[itr+1] * adxnew
                adx += adxnew
                adxnew *= 0
                # dalpha = self.alpha_tgl(dd,dr,d_list[itr+1],r_list[itr+1])
                add_tmp,adr_tmp = self.alpha_adj(adalpha,d_list[itr+1],r_list[itr+1])
                adr += adr_tmp
                add += add_tmp
                adalpha *= 0
                
                # 3. Compute new direction
                # dd = +ddnew
                addnew += add
                add *= 0
                # ddnew = -drnew + dbeta*d_list[itr] + beta_list[itr]*dd
                adrnew += -addnew
                adbeta += np.sum(d_list[itr]*addnew)
                add += beta_list[itr]*addnew
                addnew *= 0
                
                # 2. Compute beta
                # dr = +drnew
                adrnew += adr
                adr *= 0
                if a1_list[itr]!=0:
                    # dbeta = (da2*a1_list[itr]-a2_list[itr]*da1)/a1_list[itr]**2.
                    ada2 += a1_list[itr]/a1_list[itr]**2. * adbeta
                    ada1 += -a2_list[itr]/a1_list[itr]**2. * adbeta
                adbeta = 0.
                # da2 = 2.*np.dot(drnew,r_list[itr+1])
                adrnew += 2.*r_list[itr+1] * ada2 
                ada2 = 0.
                # da1 = 2.*np.dot(dr,r_list[itr]) 
                adr += 2.*r_list[itr] * ada1
                ada1 = 0
                # drnew[self.vp1] = 0 ## boundary condition     
                adrnew[self.vp1] = 0
                # drnew = self.h2pv_1d(dxnew) - dccc
                adxnew += self.h2pv_1d_adj(+adrnew)
                adccc += -adrnew
                adrnew *= 0.
                
                # 1. Update guess value
                # dx = +dxnew
                adxnew += adx
                adx *= 0
                
        # dxnew = dx + dalpha*d_list[0] + alpha_list[0]*dd
        adalpha += np.sum(d_list[0] * adxnew)
        add += alpha_list[0] * adxnew
        adx += adxnew
        adxnew *= 0
        
        # dalpha = self.alpha_tgl(dd,dr,d_list[0],r_list[0])
        add_tmp,adr_tmp = self.alpha_adj(adalpha,d_list[0],r_list[0])
        adr += adr_tmp
        add += add_tmp
        adalpha *= 0
        
        # dd = -dr
        adr += -add
        add *= 0
        
        # dr[self.vp1] = 0 ## boundary condition     
        adr[self.vp1] = 0
        
        # dr = self.h2pv_1d(dx) - dccc
        adx += self.h2pv_1d_adj(+adr)
        adccc += -adr
        adr *= 0.
        
        # dccc[self.vp2] = +dq1d[self.vp2]
        adq1d[self.vp2] += +adccc[self.vp2]
        adccc[self.vp2] *= 0
        
        # dq1d = dq[self.indi,self.indj]
        adq[self.indi,self.indj] += +adq1d
        adq1d *= 0.
        
        # dx = +dhg[self.indi,self.indj]
        adhg[self.indi,self.indj] += +adx
        adx *= 0.
    
        return adq,adhg
    
    
    def testadj_h2uv(self):
        dh = np.random.random((self.ny,self.nx))
        adu = np.random.random((self.ny,self.nx))
        adv = np.random.random((self.ny,self.nx))
        
        du,dv = self.h2uv(dh)
        adh = self.h2uv_adj(adu,adv)
        
        ps1 = np.inner(dh.ravel(),adh.ravel())
        ps2 = np.inner(np.concatenate((du.ravel(),dv.ravel())),
                       np.concatenate((adu.ravel(),adv.ravel())))
        print('testadj_h2uv:',ps1/ps2)
    
    def testadj_h2pv(self):
        dh = np.random.random((self.ny,self.nx))
        adq = np.random.random((self.ny,self.nx))
        
        dq = self.h2pv(dh)
        adh = self.h2pv_adj(adq)
        
        ps1 = np.inner(dh.ravel(),adh.ravel())
        ps2 = np.inner(dq.ravel(),adq.ravel())
        
        print('testadj_h2pv:',ps1/ps2)
        
    def testadj_qrhs(self):
        
        u = np.random.random((self.ny,self.nx))
        v = np.random.random((self.ny,self.nx))
        q = np.random.random((self.ny,self.nx))
        
        du = np.random.random((self.ny,self.nx))
        dv = np.random.random((self.ny,self.nx))
        dq = np.random.random((self.ny,self.nx))
        
        adrq = np.random.random((self.ny,self.nx))
        
        drq = self.qrhs_tgl(du,dv,dq,u,v,q,1)
        
        adu,adv,adq = self.qrhs_adj(adrq,u,v,q,1)
        

        ps1 = np.inner(drq.ravel(),adrq.ravel())
        ps2 = np.inner(np.concatenate((du.ravel(),dv.ravel(),dq.ravel())),
                       np.concatenate((adu.ravel(),adv.ravel(),adq.ravel())))
        print('testadj_qrhs:',ps1/ps2)
        
    def testadj_pv2h(self):
        
        q = np.random.random((self.ny,self.nx))
        hg = np.random.random((self.ny,self.nx))
        
        
        dq = np.random.random((self.ny,self.nx))
        dhg = np.random.random((self.ny,self.nx))
        
        
        adh = np.random.random((self.ny,self.nx))
        
        dh = self.pv2h_tgl(dq,dhg,q,hg)
        
        adq,adhg = self.pv2h_adj(adh,q,hg)
        
        
        ps1 = np.inner(dh.ravel(),adh.ravel())
        ps2 = np.inner(np.concatenate((dq[self.mask>0].ravel(),dhg[self.mask>0].ravel())),
                       np.concatenate((adq[self.mask>0].ravel(),adhg[self.mask>0].ravel())))
        print('testadj_pv2h:',ps1/ps2)
        
        
        
    def step_adj(self,adh1,h0,addphidt=None,dphidt=None,way=1):
        
        azeros = +adh1*0
        
        if addphidt is None and np.all(h0[self.mask>=1]==0):
            return adh1
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        indNan_u = np.isnan(u)
        indNan_v = np.isnan(v)
        u[indNan_u] = 0
        v[indNan_v] = 0
        
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
        adu0[indNan_u] = 0
        adv0[indNan_v] = 0
        adh0 += self.h2uv_adj(adu0,adv0)
        adu0 = +azeros
        adv0 = +azeros
            
        # 1/ h-->q
        adh0 += self.h2pv_adj(adq0)
        
        
        return adh0
    

        
    
        


if __name__ == "__main__":    
    import matplotlib.pylab as plt
    
    ny,nx = 100,150
    dx = 10e3 * np.ones((ny,nx))
    dy = 12e3 * np.ones((ny,nx))
    dt = 300
    SSH = np.zeros((ny,nx))
    
    SSH[:10,:10] = np.nan
    
    c = 2.5
    
    qgm = Qgm_adj(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,mdt=np.random.random((ny,nx)),qgiter=10)
    
    plt.figure()
    plt.pcolormesh(qgm.mask)
    plt.colorbar()
    plt.show()
    
    qgm.testadj_h2uv()
    qgm.testadj_pv2h()
    
    # Current trajectory
    SSH0 = 1e-2*np.random.random((ny,nx))

    # Perturbation
    dSSH0 = 1e-2*np.random.random((ny,nx))

    # Adjoint
    adSSH0 = 1e-2*np.random.random((ny,nx))
    
    nstep = 100
    
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

    

    ps1 = np.inner(dSSH1[qgm.mask>0].ravel(),adSSH0[qgm.mask>0].ravel())
    ps2 = np.inner(dSSH0[qgm.mask>0].ravel(),adSSH1[qgm.mask>0].ravel())
        
    print('\ntest:',ps1/ps2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    