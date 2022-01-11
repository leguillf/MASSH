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
    

    def step_adj(self,adh1,h0,adq1=None,q0=None,addphidt=None,dphidt=None,way=1):
        
        azeros = +adh1*0
        
        if np.all(h0[self.mask>=1]==0):
            if adq1 is None:
                return adh1
            else:
                return adh1,adq1
    
        
        # Tangent trajectory
        if q0 is None:
            qb0 = self.h2pv(h0)
        else:
            qb0 = +q0
        u,v = self.h2uv(h0)
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq
        if dphidt is not None:
            q1 += self.dt*dphidt
            
        # 5/ q-->h
        fguess = -self.c**2./(self.g*self.f0)*adh1 
        if adq1 is not None:
            adqguess = 2*adq1 - fguess
            adqguess[self.mask==1] = fguess[self.mask==1]
        else:
            adqguess = fguess
        adq1_tmp = self.pv2h(adh1,adqguess)
        if adq1 is None:
            adq1b = adq1_tmp
        else:
            adq1b = adq1 + adq1_tmp
        adh1 = +azeros
        
        # 4/ Time increment
        adq0 = +adq1b
        adrq = self.dt * adq1b
        if addphidt is not None:
            addphidt += self.dt * adq1b
        adq1b = +azeros
         
        # 3/ (u,v,q)-->rq
        adu0,adv0,adq_tmp = self.qrhs_adj(adrq,u,v,qb0,way)
        adq0 += adq_tmp
        adrq = +azeros
        
        # 2/ h-->(u,v)
        adh0 = +adh1
        adh0 += self.h2uv_adj(adu0,adv0)
        adu0 = +azeros
        adv0 = +azeros
            
        # 1/ h-->q
        #adh0 += self.h2pv_adj(adq0)
        
        if adq1 is None:
            return adh0
        else:
            return adh0,adq0
        
        
    

if __name__ == "__main__":    
    
    ny,nx = 100,150
    dx = 10e3 * np.ones((ny,nx))
    dy = 12e3 * np.ones((ny,nx))
    dt = 600
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm_adj(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=100)
    
    qgm.adjtest_pv2h()
    qgm.adjtest_pv2h_2()
    qgm.adjtest_h2pv_1d()
    qgm.adjtest_alpha()
    # Current trajectory
    SSH0 = 1e-2*np.random.random((ny,nx))

    # Perturbation
    dSSH0 = 1e-2*np.random.random((ny,nx))

    # Adjoint
    adSSH0 = 1e-2*np.random.random((ny,nx))
    
    nstep = 1
    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    