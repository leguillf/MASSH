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
                 g=9.81,f=1e-4,qgiter=1,qgiter_adj=None,diff=False,snu=None,
                 mdt=None,mdu=None,mdv=None):
        super().__init__(dx,dy,dt,SSH,c,upwind,upwind_adj,g,f,qgiter,qgiter_adj,diff,snu,mdt,mdu,mdv)
    
    
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
        if self.snu is not None:
            drq[2:-2,2:-2] = drq[2:-2,2:-2] +\
                self.snu/(self.dx[2:-2,2:-2]**2)*\
                    (dq[2:-2,3:-1]+dq[2:-2,1:-3]-2*dq[2:-2,2:-2]) +\
                self.snu/(self.dy[2:-2,2:-2]**2)*\
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

    
    
    def alpha_tgl(self,dp,dgg,p,gg,aaa,bbb):
        
        tmp = np.dot(p,self.h2pv_1d(p,aaa,bbb))
        dtmp = np.dot(dp,self.h2pv_1d(p,aaa,bbb)) + np.dot(p,self.h2pv_1d(dp,aaa,bbb))
        
        if tmp!=0. : 
            return -((np.dot(dp,gg)+np.dot(p,dgg))*tmp - dtmp*np.dot(p,gg))/tmp**2
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
        # Tangent iterations
        ######################
        dx = +dhg[self.indi,self.indj]
        dq1d = +dq[self.indi,self.indj]
        dccc = +dq1d
    
        dccc[self.vp1] = dx[self.vp1]  ## boundary condition        
        dgg = self.h2pv_1d(dx,aaa,bbb) - dccc
        dp = -dgg
        
        for itr in range(self.qgiter-1): 
            da1 = 2.*np.dot(dgg,gg_list[itr]) 
            dalpha = self.alpha_tgl(dp,dgg,p_list[itr],gg_list[itr],aaa,bbb)
            dxnew = dx + dalpha*p_list[itr] + alpha_list[itr]*dp
            dggnew = self.h2pv_1d(dxnew,aaa,bbb) - dccc
            da2 = 2.*np.dot(dggnew,gg_list[itr+1])
            if a1_list[itr]!=0:
                dbeta = (da2*a1_list[itr]-a2_list[itr]*da1)/a1_list[itr]**2.
            else: 
                dbeta = 0.                
            dpnew = -dggnew + dbeta*p_list[itr] + beta_list[itr]*dp
            
            dgg = +dggnew
            dp = +dpnew
            dx = +dxnew 
            

        dval1 = -np.dot(dp,gg)-np.dot(p,dgg)
        dval2 = np.dot(dp,self.h2pv_1d(p,aaa,bbb)) + np.dot(p,self.h2pv_1d(dp,aaa,bbb))
        if (val2==0.): 
            ds = 0.
        else: 
            ds = (dval1*val2 - val1*dval2)/val2**2.
            
        dx1 = dx + s*dp + ds*p 
    
        # back to 2D
        dh = np.empty((self.ny,self.nx))
        dh[:,:] = np.NAN
        dh[self.indi,self.indj] = +dx1[:]
    
        return dh
    
    
    def step_tgl(self,dh0,h0,ddphidt=None,dphidt=None,way=1):
        
        if np.all(h0==0):
            return dh0
        
        # Tangent trajectory
        qb0 = self.h2pv(h0)
        u,v = self.h2uv(h0)
        rq = self.qrhs(u,v,qb0,way)
        q1 = qb0 + self.dt*rq
        if dphidt is not None:
            q1 += self.dt*dphidt
        # 1/ h-->q
        dq0 = self.h2pv(dh0)

        # 2/ h-->(u,v)
        du,dv = self.h2uv(dh0)
        
        # 3/ (u,v,q)-->rq
        drq = self.qrhs_tgl(du,dv,dq0,u,v,qb0,way)
        
        # 4/ Time increment
        dq1 = +dq0
        dq1 += self.dt * drq
        if ddphidt is not None:
            dq1 += self.dt*ddphidt
        
        # 5/ q-->h
        dh1 = self.pv2h_tgl(dq1,dh0,q1,h0)
        
        return dh1

        
        
        


if __name__ == "__main__":
    
    ny,nx = 100,100
    dx = dy = 10e3 * np.ones((ny,nx))
    dt = 1200
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm_tgl(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH,qgiter=3)
    
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
    
    