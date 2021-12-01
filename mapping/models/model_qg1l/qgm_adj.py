#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:22:53 2021

@author: leguillou
"""

import matplotlib.pylab as plt

from qgm_tgl import Qgm_tgl
import numpy as np

class Qgm_adj(Qgm_tgl):
    
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,
                 g=9.81,f=1e-4,qgiter=1,diff=False,snu=None,
                 mdt=None,mdu=None,mdv=None):
        super().__init__(dx,dy,dt,SSH,c,g,f,qgiter,diff,snu,mdt,mdu,mdv)
    
    def qrhs_adj(self,adrq,u,v,q,way):

        adrq[np.isnan(adrq)]=0.
        adrq[np.where((self.mask<=1))]=0
    
        adu=np.zeros((self.ny,self.nx))
        adv=np.zeros((self.ny,self.nx))
        adq=np.zeros((self.ny,self.nx))
    
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
            _aduplus,_advplus,_aduminus,_advminus,adq = self._rq_adj(adrq,adq,
                     way*self.uplusbar,way*self.vplusbar,way*self.uminusbar,way*self.vminusbar,self.qbar)
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
            adq[2:-2,2:-2]=adq[2:-2,2:-2]+self.snu/(self.dx[2:-2,2:-2]**2)*(adrq[2:-2,3:-1]+adrq[2:-2,1:-3]-2*adrq[2:-2,2:-2]) \
            +self.snu/(self.dy[2:-2,2:-2]**2)*(adrq[3:-1,2:-2]+adrq[1:-3,2:-2]-2*adrq[2:-2,2:-2]) 
            
        
        return adu, adv, adq
    
    
    def _rq_adj(self,adrq,adq,uplus,vplus,uminus,vminus,q):
        
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
    
    def alpha_adj(self,adalpha,r,d):
        norm_r = self.norm(r)
        dAd = d.ravel().dot(self.h2pv(d).ravel())
        
        adr = 2*r/dAd * adalpha
        add = -(norm_r/dAd)**2 *  (self.h2pv_adj(d) + self.h2pv(d)) * adalpha         
        
        return adr,add
    
    def beta_adj(self,adbeta,r,rnew):
        
        norm_rnew = self.norm(rnew)
        norm_r = self.norm(r)
        adr = -2*(norm_rnew/norm_r**2)**2 * r * adbeta
        adrnew = 2/norm_r**2 * rnew * adbeta
        
        return adr,adrnew
    
    def pv2h_adj(self,adh,q,hg):
        
        adh[self.mask==0] = 0
        
        # Current trajectory
        q_tmp = +q
        q_tmp[self.mask==0] = 0
        hg[self.mask==0] = 0
        r = +q_tmp - self.h2pv(hg)
        d = +r
        alpha = self.alpha(r,d)
        alpha_list = [alpha]
        beta_list = []
        r_list = [r]
        d_list = [d]
        if self.qgiter>1:
            # Loop
            for itr in range(self.qgiter):
                # Update direction
                rnew = r - alpha * self.h2pv(d)
                beta = self.beta(r,rnew)
                r = +rnew
                d = r + beta * d
                alpha = self.alpha(r,d)
                # Append to lists
                alpha_list.append(alpha)
                r_list.append(r)
                d_list.append(d)
                beta_list.append(beta)
                
        
        # Initialize adjoint variable
        adalpha = 0
        adbeta = 0
        adhg = np.zeros((self.ny,self.nx))
        adhb = np.zeros((self.ny,self.nx))
        adr = np.zeros((self.ny,self.nx))
        adrnew = np.zeros((self.ny,self.nx))
        add = np.zeros((self.ny,self.nx))
        addnew = np.zeros((self.ny,self.nx))
        if self.qgiter>1:
            # Adjoint Loop
            for itr in reversed(range(self.qgiter)):
                # Update SSH
                # dh = dhg + dalpha*d_list[itr+1] + alpha_list[itr+1]*dd
                adhg += adh
                adalpha += np.sum(d_list[itr+1]*adh)
                add += alpha_list[itr+1]*adh
                adh = np.zeros((self.ny,self.nx))
                
                # Compute new direction
                # dd = +ddnew
                addnew += add
                add = np.zeros((self.ny,self.nx))
                # dalpha = self.alpha_tgl(dr,ddnew,r_list[itr+1],d_list[itr+1])
                adr_tmp,addnew_tmp = self.alpha_adj(adalpha,r_list[itr+1],d_list[itr+1])
                adr += adr_tmp
                addnew += addnew_tmp
                adalpha = 0
                # ddnew = dr + dbeta * d_list[itr] + beta_list[itr] * dd
                adr += addnew
                adbeta += np.sum(d_list[itr]*addnew)
                add += beta_list[itr] * addnew
                addnew = np.zeros((self.ny,self.nx))
                
                # Compute beta
                # dr = +drnew
                adrnew += adr
                adr = np.zeros((self.ny,self.nx))
                # dbeta = self.beta_tgl(dr,drnew,r_list[itr],r_list[itr+1])
                adr_tmp,adrnew_tmp = self.beta_adj(adbeta,r_list[itr],r_list[itr+1])
                adr += adr_tmp
                adrnew += adrnew_tmp
                adbeta = 0
                # drnew = dr - (dalpha*self.h2pv(d_list[itr]) +
                #               alpha_list[itr]*self.h2pv(dd))
                adr += adrnew
                adalpha += -np.sum(self.h2pv(d_list[itr])*adrnew)
                add += self.h2pv_adj(-alpha_list[itr]*adrnew)
                adrnew = np.zeros((self.ny,self.nx))
                
                # Update guess value
                # dhg = +dh
                adh += adhg
                adhg = np.zeros((self.ny,self.nx))
                
                
        # dh = dhg + dalpha*d_list[0] + alpha_list[0]*dd
        adhg += adh
        adalpha += np.sum(d_list[0]*adh)
        add += alpha_list[0]*adh
        adh = np.zeros((self.ny,self.nx))
            
        # dalpha = self.alpha_tgl(dr,dd,r,d)
        adr_tmp,add_tmp = self.alpha_adj(adalpha,r_list[0],d_list[0])
        adr += adr_tmp
        add += add_tmp
        adalpha = 0
    
        # dd = +dr
        adr += add
        add = np.zeros((self.ny,self.nx))
        
        # dr = +dq-self.h2pv(dhg)
        adq = +adr
        adhg += -self.h2pv_adj(adr)
        adr = np.zeros((self.ny,self.nx))
        
        adq[self.mask==0] = np.nan
        adhg[self.mask==0] = np.nan
        
        
        return adq,adhg

        
    def step_adj(self,adh1,h0,way=1):
        
        azeros = +adh1*0
        
        if np.all(h0[self.mask>=1]==0):
            return adh1
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq
        
        # 5/ q-->h
        adq1,adh0 = self.pv2h_adj(adh1,q1,h0)
        adh1 = +azeros
        
        # 4/ Time increment
        adq0 = +adq1
        adrq = self.dt * adq1
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
        
        



if __name__ == "__main__":    
    
    ny,nx = 100,150
    dx = 10e3 * np.ones((ny,nx))
    dy = 12e3 * np.ones((ny,nx))
    dt = 600
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm_adj(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=20)
        
    # Current trajectory
    SSH0 = 1e-2*np.random.random((ny,nx))

    # Perturbation
    dSSH0 = 1e-2*np.random.random((ny,nx))

    # Adjoint
    adSSH0 = 1e-2*np.random.random((ny,nx))
    
    nstep = 10
    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    