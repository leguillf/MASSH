#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:31:17 2021

@author: leguillou
"""

from qgm import Qgm
import numpy as np

class Qgm_tgl(Qgm):
    
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,upwind=3,upwind_adj=None,
                 g=9.81,f=1e-4,qgiter=1,qgiter_adj=None,diff=False,Kdiffus=None,
                 mdt=None,mdu=None,mdv=None):
        super().__init__(dx,dy,dt,SSH,c,upwind,upwind_adj,g,f,qgiter,qgiter_adj,diff,Kdiffus,mdt,mdu,mdv)
    
    
    def qrhs_tgl(self,du,dv,dq,u,v,q,way):

        drq = np.zeros((self.ny,self.nx))
        
        if not self.diff:
            uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            duplus = way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
            uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            duminus = way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
            vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            dvplus = way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
            vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            dvminus = way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
            
            duplus[np.where((uplus<0))] = 0
            duminus[np.where((uminus>0))] = 0
            dvplus[np.where((vplus<0))] = 0
            dvminus[np.where((vminus>=0))] = 0
            
            uplus[np.where((uplus<0))] = 0
            uminus[np.where((uminus>0))] = 0
            vplus[np.where((vplus<0))] = 0
            vminus[np.where((vminus>=0))] = 0
            
            
        
            drq[2:-2,2:-2] = drq[2:-2,2:-2] + self._rq_tgl(+duplus,+dvplus,+duminus,+dvminus,+dq,
                                                           +uplus,+vplus,+uminus,+vminus,+q)
            
                
            if self.mdt is not None:
                
                uplusbar = way*self.uplusbar
                uplusbar[np.where((uplusbar<0))] = 0
                vplusbar = way*self.vplusbar
                vplusbar[np.where((vplusbar<0))] = 0
                uminusbar = way*self.uminusbar
                uminusbar[np.where((uminusbar>0))] = 0
                vminusbar = way*self.vminusbar
                vminusbar[np.where((vminusbar>0))] = 0
                   
                drq[2:-2,2:-2] = drq[2:-2,2:-2] + self._rq_tgl(
                    duplus,dvplus,duminus,dvminus,dq,
                    uplusbar,vplusbar,uminusbar,vminusbar,self.qbar)
    
                    
            drq[2:-2,2:-2] = drq[2:-2,2:-2]-\
                (self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/\
                    (2*self.dy[2:-2,2:-2])*way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2]) 
    
        #diffusion
        if self.Kdiffus is not None:
            drq[2:-2,2:-2] = drq[2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (dq[2:-2,3:-1]+dq[2:-2,1:-3]-2*dq[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (dq[3:-1,2:-2]+dq[1:-3,2:-2]-2*dq[2:-2,2:-2])
    
        drq[np.where((self.mask<=1))] = 0
        drq[np.isnan(drq)] = 0
        
        return drq
    
    
    def _rq_tgl(self,duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q):
        
        """
            main function for upwind schemes
        """
        
        if self.upwind==1:
            return self._rq1_tgl(duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q)
        elif self.upwind==2:
            return self._rq2_tgl(duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q)
        elif self.upwind==3:
            return self._rq3_tgl(duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q)
        
        
    def _rq1_tgl(self,duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q):
        
        """
            1st-order upwind scheme
        """
        
        res = \
            - uplus*1/(self.dx[2:-2,2:-2]) * (dq[2:-2,2:-2]-dq[2:-2,1:-3]) \
            + uminus*1/(self.dx[2:-2,2:-2])* (dq[2:-2,2:-2]-dq[2:-2,3:-1]) \
            - vplus*1/(self.dy[2:-2,2:-2]) * (dq[2:-2,2:-2]-dq[1:-3,2:-2]) \
            + vminus*1/(self.dy[2:-2,2:-2])* (dq[2:-2,2:-2]-dq[3:-1,2:-2]) \
    \
            - duplus*1/(self.dx[2:-2,2:-2]) * (q[2:-2,2:-2]-q[2:-2,1:-3]) \
            + duminus*1/(self.dx[2:-2,2:-2])* (q[2:-2,2:-2]-q[2:-2,3:-1]) \
            - dvplus*1/(self.dy[2:-2,2:-2]) * (q[2:-2,2:-2]-q[1:-3,2:-2]) \
            + dvminus*1/(self.dy[2:-2,2:-2])* (q[2:-2,2:-2]-q[3:-1,2:-2])
            
        return res
    
    def _rq2_tgl(self,duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q):
        
        """
            2nd-order upwind scheme
        """
        
        res = \
            - uplus*1/(2*self.dx[2:-2,2:-2])*\
                (3*dq[2:-2,2:-2]-4*dq[2:-2,1:-3]+dq[2:-2,:-4]) \
            + uminus*1/(2*self.dx[2:-2,2:-2])*\
                (dq[2:-2,4:]-4*dq[2:-2,3:-1]+3*dq[2:-2,2:-2])  \
            - vplus*1/(2*self.dy[2:-2,2:-2])*\
                (3*dq[2:-2,2:-2]-4*dq[1:-3,2:-2]+dq[:-4,2:-2]) \
            + vminus*1/(2*self.dy[2:-2,2:-2])*\
                (dq[4:,2:-2]-4*dq[3:-1,2:-2]+3*dq[2:-2,2:-2])  \
    \
            - duplus*1/(2*self.dx[2:-2,2:-2])*\
                (3*q[2:-2,2:-2]-4*q[2:-2,1:-3]+q[2:-2,:-4]) \
            + duminus*1/(2*self.dx[2:-2,2:-2])*\
                (q[2:-2,4:]-4*q[2:-2,3:-1]+3*q[2:-2,2:-2])  \
            - dvplus*1/(2*self.dy[2:-2,2:-2])*\
                (3*q[2:-2,2:-2]-4*q[1:-3,2:-2]+q[:-4,2:-2]) \
            + dvminus*1/(2*self.dy[2:-2,2:-2])*\
                (q[4:,2:-2]-4*q[3:-1,2:-2]+3*q[2:-2,2:-2])
            
        return res
    
    
    def _rq3_tgl(self,duplus,dvplus,duminus,dvminus,dq,uplus,vplus,uminus,vminus,q):
        
        """
            3rd-order upwind scheme
        """
        
        res = \
            - uplus*1/(6*self.dx[2:-2,2:-2])*\
                (2*dq[2:-2,3:-1]+3*dq[2:-2,2:-2]-6*dq[2:-2,1:-3]+dq[2:-2,:-4])\
            + uminus*1/(6*self.dx[2:-2,2:-2])*\
                (dq[2:-2,4:]-6*dq[2:-2,3:-1]+3*dq[2:-2,2:-2]+2*dq[2:-2,1:-3]) \
            - vplus*1/(6*self.dy[2:-2,2:-2])*\
                (2*dq[3:-1,2:-2]+3*dq[2:-2,2:-2]-6*dq[1:-3,2:-2]+dq[:-4,2:-2]) \
            + vminus*1/(6*self.dy[2:-2,2:-2])*\
                (dq[4:,2:-2]-6*dq[3:-1,2:-2]+3*dq[2:-2,2:-2]+2*dq[1:-3,2:-2]) \
    \
            - duplus*1/(6*self.dx[2:-2,2:-2])*\
                (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]-6*q[2:-2,1:-3]+q[2:-2,:-4])\
            + duminus*1/(6*self.dx[2:-2,2:-2])*\
                (q[2:-2,4:]-6*q[2:-2,3:-1]+3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
            - dvplus*1/(6*self.dy[2:-2,2:-2])*\
                (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-6*q[1:-3,2:-2]+q[:-4,2:-2])  \
            + dvminus*1/(6*self.dy[2:-2,2:-2])*\
                (q[4:,2:-2]-6*q[3:-1,2:-2]+3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
            
        return res

    
    
    def alpha_tgl(self,dd,dr,d,r):
        
        tmp = np.dot(d,self.h2pv_1d(d))
        dtmp = np.dot(dd,self.h2pv_1d(d)) + np.dot(d,self.h2pv_1d(dd))
        
        if tmp!=0. : 
            return -((np.dot(dd,r)+np.dot(d,dr))*tmp - dtmp*np.dot(d,r))/tmp**2
        else: 
            return 0.
    
    
    def pv2h_tgl(self,dq,dhg,q,hg):
        """ Q to SSH
        
        This code solve a linear system of equations using Conjugate Gradient method
    
        Args:
            q (2D array): Potential Vorticity field
            hg (2D array): SSH guess
            grd (Grid() object): check modgrid.py
    
        Returns:
            h (2D array): SSH field. 
        """
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
                    
        
        
        #####################
        # Tangent iterations
        ######################
        dx = +dhg[self.indi,self.indj]
        dq1d = +dq[self.indi,self.indj]

        dccc = +dq1d
        
        dr = self.h2pv_1d(dx) - dccc
        dr[self.vp1] = 0 ## boundary condition     
        
        dd = -dr
        
        dalpha = self.alpha_tgl(dd,dr,d_list[0],r_list[0])
        dxnew = dx + dalpha*d_list[0] + alpha_list[0]*dd
        
        if self.qgiter_adj>1:
            for itr in range(self.qgiter_adj): 
                
                # 1. Update guess value
                dx = +dxnew
                
                # 2. Compute beta
                drnew = self.h2pv_1d(dxnew) - dccc
                drnew[self.vp1] = 0 ## boundary condition     
                da1 = 2.*np.dot(dr,r_list[itr]) 
                da2 = 2.*np.dot(drnew,r_list[itr+1])
                if a1_list[itr]!=0:
                    dbeta = (da2*a1_list[itr]-a2_list[itr]*da1)/a1_list[itr]**2.
                else: 
                    dbeta = 0.  
                dr = +drnew
                
                # 3. Compute new direction
                ddnew = -drnew + dbeta*d_list[itr] + beta_list[itr]*dd
                dd = +ddnew
                
                # 4. Update state
                dalpha = self.alpha_tgl(dd,dr,d_list[itr+1],r_list[itr+1])
                dxnew = dx + dalpha*d_list[itr+1] + alpha_list[itr+1]*dd
            

        # back to 2D
        dh = np.empty((self.ny,self.nx))
        dh[:,:] = np.NAN
        dh[self.indi,self.indj] = +dxnew[:]
    
        return dh
    
    
    def step_tgl(self,dh0,h0,way=1):
        
        if np.all(h0==0):
            return dh0
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        indNan_u = np.isnan(u)
        indNan_v = np.isnan(v)
        u[indNan_u] = 0
        v[indNan_v] = 0
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq

        # 1/ h-->q
        dq0 = self.h2pv(dh0)

        # 2/ h-->(u,v)
        du,dv = self.h2uv(dh0)
        du[indNan_u] = 0
        dv[indNan_v] = 0
        
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
    
    qgm = Qgm_tgl(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=100)
    
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
    
    