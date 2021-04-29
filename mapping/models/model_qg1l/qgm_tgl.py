#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:31:17 2021

@author: leguillou
"""

from qgm import Qgm
import numpy as np

class Qgm_tgl(Qgm):
    
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,
                 g=9.81,f=1e-4,qgiter=1,diff=False,snu=None):
        super().__init__(dx,dy,dt,SSH,c,g,f,qgiter,diff,snu)
    
    
    def qrhs_tgl(self,du,dv,dq,u,v,q,way):

        drq = np.zeros((self.ny,self.nx))
    
        uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        duplus = way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
        duplus[np.where((uplus<0))] = 0
        uplus[np.where((uplus<0))] = 0
        uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        duminus = way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
        duminus[np.where((uminus>=0))] = 0
        uminus[np.where((uminus>=0))] = 0
        vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        dvplus = way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
        dvplus[np.where((vplus<0))] = 0
        vplus[np.where((vplus<0))] = 0
        vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        dvminus = way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
        dvminus[np.where((vminus>=0))] = 0
        vminus[np.where((vminus>=0))] = 0
    
        drq[2:-2,2:-2] = drq[2:-2,2:-2] \
            - uplus*1/(6*self.dx[2:-2,2:-2])*(2*dq[2:-2,3:-1]+3*dq[2:-2,2:-2]-\
                                            6*dq[2:-2,1:-3]+dq[2:-2,:-4]) \
            + uminus*1/(6*self.dx[2:-2,2:-2])*(dq[2:-2,4:]-6*dq[2:-2,3:-1]+\
                                               3*dq[2:-2,2:-2]+2*dq[2:-2,1:-3])\
            - vplus*1/(6*self.dy[2:-2,2:-2])*(2*dq[3:-1,2:-2]+3*dq[2:-2,2:-2]-\
                                              6*dq[1:-3,2:-2]+dq[:-4,2:-2])\
            + vminus*1/(6*self.dy[2:-2,2:-2])*(dq[4:,2:-2]-6*dq[3:-1,2:-2]+ \
                                               3*dq[2:-2,2:-2]+2*dq[1:-3,2:-2])
    
        drq[2:-2,2:-2] = drq[2:-2,2:-2] \
            - duplus*1/(6*self.dx[2:-2,2:-2])*(2*q[2:-2,3:-1]+3*q[2:-2,2:-2]-\
                                               6*q[2:-2,1:-3]+q[2:-2,:-4] ) \
            + duminus*1/(6*self.dx[2:-2,2:-2])*(q[2:-2,4:]-6*q[2:-2,3:-1]+ \
                                                3*q[2:-2,2:-2]+2*q[2:-2,1:-3])\
            - dvplus*1/(6*self.dy[2:-2,2:-2])*(2*q[3:-1,2:-2]+3*q[2:-2,2:-2]- \
                                               6*q[1:-3,2:-2]+q[:-4,2:-2])  \
            + dvminus*1/(6*self.dy[2:-2,2:-2])*(q[4:,2:-2]-6*q[3:-1,2:-2]+ \
                                                3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
    
        drq[2:-2,2:-2] = drq[2:-2,2:-2]-\
            (self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/\
                (2*self.dy[2:-2,2:-2])*way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2]) 
    
        #diffusion
        if self.snu is not None:
            drq[2:-2,2:-2] = drq[2:-2,2:-2] +\
                self.snu/(self.dx[2:-2,2:-2]**2)*\
                    (dq[2:-2,3:-1]+dq[2:-2,1:-3]-2*dq[2:-2,2:-2]) \
                    +self.snu/(self.dy[2:-2,2:-2]**2)*\
                        (dq[3:-1,2:-2]+dq[1:-3,2:-2]-2*dq[2:-2,2:-2])
    
            drq[np.where((self.mask<=1))]=0
    
        return drq
    
    
    def alpha_tgl(self,dr,dd,r,d):
        norm_r = self.norm(r)
        dAd = d.ravel().dot(self.h2pv(d).ravel())

        dalpha = 2*np.inner(dr.ravel(),r.ravel()) / dAd - (norm_r/dAd)**2*\
            (d.ravel().dot(self.h2pv(dd).ravel())+\
             dd.ravel().dot(self.h2pv(d).ravel())) 

        return dalpha
    
    def beta_tgl(self,dr,drnew,r,rnew):
        dbeta = (2*np.inner(drnew.ravel(),rnew.ravel())*self.norm(r)**2-\
                2*np.inner(dr.ravel(),r.ravel())*self.norm(rnew)**2)/\
                self.norm(r)**4
            
        return dbeta
    
    def pv2h_tgl(self,dq,dhg,q,hg):
        

        # Current trajectory
        r = +q - self.h2pv(hg)
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
        
        dr = +dq - self.h2pv(dhg)
        dd = +dr        
        dalpha = self.alpha_tgl(dr,dd,r_list[0],d_list[0])
        dh = dhg + dalpha*d_list[0] + alpha_list[0]*dd
        if self.qgiter>1:
            for itr in range(self.qgiter):
                # Update guess value
                dhg = +dh    
                # Compute beta
                drnew = dr - (dalpha*self.h2pv(d_list[itr]) +
                              alpha_list[itr]*self.h2pv(dd))
                dbeta = self.beta_tgl(dr,drnew,r_list[itr],r_list[itr+1])
                dr = +drnew
                # Compute new direction
                ddnew = dr + dbeta * d_list[itr] + beta_list[itr] * dd
                dalpha = self.alpha_tgl(dr,ddnew,r_list[itr+1],d_list[itr+1])
                dd = +ddnew
                # Update SSH
                dh = dhg + dalpha*d_list[itr+1] + alpha_list[itr+1]*dd
                  
        return dh
    
    
    def step_tgl(self,dh0,h0,way=1):
        
        if np.all(h0==0):
            return dh0
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq
        
        # 1/ h-->q
        dq0 = self.h2pv(dh0)
        
        # 2/ h-->(u,v)
        du,dv = self.h2uv(dh0)
        
        # 3/ (u,v,q)-->rq
        drq = self.qrhs_tgl(du,dv,dq0,u,v,qb0,way)
        
        # 4/ Time increment
        dq1 = +dq0
        dq1 += self.dt * drq
        
        # 5/ q-->h
        dh1 = self.pv2h_tgl(dq1,dh0,q1,h0)
        
        return dh1

        
        
        


if __name__ == "__main__":
    
    ny,nx = 100,100
    dx = dy = 10e3 * np.ones((ny,nx))
    dt = 1200
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm_tgl(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=10)
    
    # Tangent test    
    SSH0 = np.random.random((ny,nx))
    dSSH = np.random.random((ny,nx))
    
    SSH2 = qgm.step(SSH0)
    
    for p in range(10):
        
        lambd = 10**(-p)
        
        SSH1 = qgm.step(SSH0+lambd*dSSH)
        
        dSSH1 = qgm.step_tgl(dh0=lambd*dSSH,h0=SSH0)
        
        ps = np.linalg.norm(SSH1-SSH2-dSSH1)/np.linalg.norm(dSSH1)

        print('%.E' % lambd,'%.E' % ps)
    
    