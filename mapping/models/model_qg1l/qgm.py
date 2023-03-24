import numpy as np
import matplotlib.pylab as plt 




class Qgm:
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,upwind=3,upwind_adj=None,
                 g=9.81,f=1e-4,qgiter=1,qgiter_adj=None,diff=False,Kdiffus=None,
                 mdt=None,mdu=None,mdv=None):
        
        # Grid spacing
        self.dx = dx
        self.dy = dy
        
        # Grid shape
        ny,nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny
        
        # Time step
        self.dt = dt
        
        # Gravity
        self.g = g
        
        # Coriolis
        if hasattr(f, "__len__") and f.shape==self.dx.shape:
            self.f0 = f
        else: 
            self.f0 = f * np.ones_like(self.dx)
            
        # Rossby radius  
        if hasattr(c, "__len__") and c.shape==self.dx.shape:
            self.c = c
        else: 
            self.c = c * np.ones_like(self.dx)
        
        # Mask array
        mask = np.zeros((ny,nx))+2
        mask[:2,:] = 1
        mask[:,:2] = 1
        mask[-2:,:] = 1
        mask[:,-2:] = 1
    
        if SSH is not None and mdt is not None:
            isNAN = np.isnan(SSH) | np.isnan(mdt)
        elif SSH is not None:
            isNAN = np.isnan(SSH)
        elif mdt is not None:
            isNAN = np.isnan(mdt)
        else:
            isNAN = None
            
        if isNAN is not None: 
            mask[isNAN] = 0
            indNan = np.argwhere(isNAN)
            for i,j in indNan:
                for p1 in [-1,0,1]:
                    for p2 in [-1,0,1]:
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                          if mask[itest,jtest]==2:
                              mask[itest,jtest] = 1
        self.mask = mask

        # Parameter for the gradient conjugate descent
        self.qgiter = qgiter # Nb of iterations 
        if qgiter_adj is not None:
            self.qgiter_adj = qgiter_adj
        else:
            self.qgiter_adj = qgiter
        _np = np.shape(np.where(mask>=1))[1]
        _np2 = np.shape(np.where(mask==2))[1]
        _np1 = np.shape(np.where(mask==1))[1]
        self.np = _np
        self.np2 = _np2
        self.mask1d=np.zeros((_np))
        self.H=np.zeros((_np))
        self.c1d=np.zeros((_np))
        self.f01d=np.zeros((_np))
        self.dx1d=np.zeros((_np))
        self.dy1d=np.zeros((_np))
        self.indi=np.zeros((_np), dtype=int)
        self.indj=np.zeros((_np), dtype=int)
        self.vp1=np.zeros((_np1), dtype=int)
        self.vp2=np.zeros((_np2), dtype=int)
        self.vp2=np.zeros((_np2), dtype=int)
        self.vp2n=np.zeros((_np2), dtype=int)
        self.vp2nn=np.zeros((_np2), dtype=int)
        self.vp2s=np.zeros((_np2), dtype=int)
        self.vp2ss=np.zeros((_np2), dtype=int)
        self.vp2e=np.zeros((_np2), dtype=int)
        self.vp2ee=np.zeros((_np2), dtype=int)
        self.vp2w=np.zeros((_np2), dtype=int)
        self.vp2ww=np.zeros((_np2), dtype=int)
        self.vp2nw=np.zeros((_np2), dtype=int)
        self.vp2ne=np.zeros((_np2), dtype=int)
        self.vp2se=np.zeros((_np2), dtype=int)
        self.vp2sw=np.zeros((_np2), dtype=int)
        self.indp=np.zeros((ny,nx), dtype=int)
        
        p=-1
        for i in range(ny):
            for j in range(nx):
                if (mask[i,j]>=1):
                    p=p+1
                    self.mask1d[p]=mask[i,j]
                    self.H[p]=SSH[i,j]
                    self.dx1d[p]=dx[i,j]
                    self.dy1d[p]=dy[i,j]
                    self.f01d[p]=self.f0[i,j]
                    self.c1d[p]=self.c[i,j]
                    self.indi[p]=i
                    self.indj[p]=j
                    self.indp[i,j]=p
    
        p2=-1
        p1=-1
        for p in range(_np):
            if (self.mask1d[p]==2):
                p2=p2+1
                i=self.indi[p]
                j=self.indj[p]
                self.vp2[p2]=p
                self.vp2n[p2]=self.indp[i+1,j]
                self.vp2nn[p2]=self.indp[i+2,j]
                self.vp2s[p2]=self.indp[i-1,j]
                self.vp2ss[p2]=self.indp[i-2,j]
                self.vp2e[p2]=self.indp[i,j+1]
                self.vp2ee[p2]=self.indp[i,j+2]
                self.vp2w[p2]=self.indp[i,j-1]
                self.vp2ww[p2]=self.indp[i,j-2]
                self.vp2nw[p2]=self.indp[i+1,j-1]
                self.vp2ne[p2]=self.indp[i+1,j+1]
                self.vp2se[p2]=self.indp[i-1,j+1]
                self.vp2sw[p2]=self.indp[i-1,j-1]
            if (self.mask1d[p]==1):
                p1=p1+1
                i=self.indi[p]
                j=self.indj[p]
                self.vp1[p1]=p
        
        self.aaa = self.g/self.f01d
        self.bbb = - self.g*self.f01d / self.c1d**2
        self.aaa[self.vp1] = 0
        self.bbb[self.vp1] = 1
        
        
        # Spatial scheme
        self.upwind = upwind
        if upwind_adj is None:
            self.upwind_adj = upwind
        else:
            self.upwind_adj = upwind_adj
        
        # Diffusion 
        self.diff = diff
        self.Kdiffus = Kdiffus
        if Kdiffus is not None and Kdiffus==0:
            self.Kdiffus = None
        

        # MDT
        self.mdt = mdt
        if self.mdt is not None:
            if mdu is  None or mdv is  None:
                self.ubar,self.vbar = self.h2uv(self.mdt)
                self.qbar = self.h2pv(self.mdt,c=np.nanmean(self.c)*np.ones_like(self.dx))
            else:
                self.ubar = mdu
                self.vbar = mdv
                self.qbar = self.huv2pv(mdt,mdu,mdv,c=np.nanmean(self.c)*np.ones_like(self.dx))
                self.mdt = self.pv2h(self.qbar,+mdt)
            #self.ubar,self.vbar = self.h2uv(self.mdt,ubc=mdu,vbc=mdv)
            #self.qbar = self.h2pv(self.mdt)
            #self.qbar = self.huv2pv(self.ubar,self.vbar,self.mdt,c=np.nanmean(self.c)*np.ones_like(self.dx))
            #self.mdt = self.pv2h(self.qbar,+mdt)
            # For qrhs
            self.uplusbar  = 0.5*(self.ubar[2:-2,2:-2]+self.ubar[2:-2,3:-1])
            self.uminusbar = 0.5*(self.ubar[2:-2,2:-2]+self.ubar[2:-2,3:-1])
            self.vplusbar  = 0.5*(self.vbar[2:-2,2:-2]+self.vbar[3:-1,2:-2])
            self.vminusbar = 0.5*(self.vbar[2:-2,2:-2]+self.vbar[3:-1,2:-2])
        
            
    
    def h2uv(self,h,ubc=None,vbc=None):
        """ SSH to U,V
    
        Args:
            h (2D array): SSH field.
    
        Returns:
            u (2D array): Zonal velocity  
            v (2D array): Meridional velocity
    
        """
        u = np.zeros((self.ny,self.nx))
        v = np.zeros((self.ny,self.nx))
    
        u[1:-1,1:] = - self.g/self.f0[1:-1,1:]*\
            (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy[1:-1,1:])
             
        v[1:,1:-1] = + self.g/self.f0[1:,1:-1]*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx[1:,1:-1])
        
        if ubc is not None and vbc is not None:
            u[self.mask==1] = ubc[self.mask==1]
            v[self.mask==1] = vbc[self.mask==1]
        
        return u,v


    def h2pv(self,h,c=None):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius 
    
        Returns:
            q: Potential Vorticity field  
        """
        
        if c is None:
            c = self.c
            
        q = np.zeros((self.ny,self.nx))
        
        q[1:-1,1:-1] = self.g/self.f0[1:-1,1:-1]*\
            ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 +\
              (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) -\
                self.g*self.f0[1:-1,1:-1]/(c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        
        ind = np.where((self.mask==1))
        q[ind] = -self.g*self.f0[ind]/(c[ind]**2) * h[ind]
            
        ind = np.where((self.mask==0))
        q[ind] = 0
    
        return q
    
    
    def huv2pv(self,h,u,v,c=None):
        
        if c is None:
            c = self.c
            
        q = np.zeros((self.ny,self.nx))
        
        q[1:-1,1:-1] = \
            0.5*(v[1:-1,2:] - v[1:-1,:-2]) / self.dx[1:-1,1:-1]-\
            0.5*(u[2:,1:-1] - u[:-2,1:-1]) / self.dy[1:-1,1:-1]  -\
                self.g*self.f0[1:-1,1:-1]/(c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        
        ind = np.where((self.mask==1))
        q[ind] = -self.g*self.f0[ind]/(c[ind]**2) * h[ind]
            
        ind = np.where((self.mask==0))
        q[ind] = 0
        
        return q
    
    def h2pv_1d(self,h1d):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius 
    
        Returns:
            q: Potential Vorticity field  
        """
    
            
        q1d = np.zeros(self.np,) 
        
        q1d[self.vp2] = self.aaa[self.vp2]*\
            ((h1d[self.vp2e]+h1d[self.vp2w]-2*h1d[self.vp2])/self.dx1d[self.vp2]**2 +\
             (h1d[self.vp2n]+h1d[self.vp2s]-2*h1d[self.vp2])/self.dy1d[self.vp2]**2) +\
                self.bbb[self.vp2] * h1d[self.vp2]
        q1d[self.vp1] = self.bbb[self.vp1] * h1d[self.vp1]
    
        return q1d

    def alpha(self,d,r):
        tmp = np.dot(d,self.h2pv_1d(d))
        if tmp!=0. : 
            return -np.dot(d,r)/tmp
        else: 
            return 1.


    def pv2h_old(self,q,hg):
        ny,nx,=np.shape(hg)
        g=self.g


        x=hg[self.indi,self.indj]
        q1d=q[self.indi,self.indj]

        #aaa=g/grd.f01d
        #bbb=-g*grd.f01d/grd.c1d**2
        #ccc=q1d-grd.f01d
        aaa=g/self.f0.ravel() 
        bbb = - g*self.f0.ravel() / (self.c.ravel())**2
        ccc=+q1d
        #ccc=+q1d-grd.f01d  

        aaa[self.vp1]=0
        bbb[self.vp1]=1
        ccc[self.vp1]=x[self.vp1]  ##boundary condition

        vec=+x

        avec,=self.compute_avec(vec,aaa,bbb)
        gg=avec-ccc
        p=-gg
        #print 'test1', numpy.var(gg)

        for itr in range(self.qgiter-1): 
            vec=+p
            avec,=self.compute_avec(vec,aaa,bbb)
            tmp=np.dot(p,avec)
            
            if tmp!=0. : s=-np.dot(p,gg)/tmp
            else: s=1.
            
            a1=np.dot(gg,gg)
            x=x+s*p
            vec=+x
            avec,=self.compute_avec(vec,aaa,bbb)
            gg=avec-ccc
            #print 'test', numpy.var(gg)
            a2=np.dot(gg,gg)
            
            if a1!=0: beta=a2/a1
            else: beta=1.
            
            p=-gg+beta*p
        
        vec=+p
        avec,=self.compute_avec(vec,aaa,bbb)
        val1=-np.dot(p,gg)
        val2=np.dot(p,avec)
        if (val2==0.): 
            s=1.
        else: 
            s=val1/val2
        
        #pdb.set_trace()
        a1=np.dot(gg,gg)
        x=x+s*p

        # back to 2D
        h=np.empty((ny,nx))
        h[:,:]=np.NAN
        h[self.indi,self.indj]=x[:]


        return h

    def compute_avec(self,vec,aaa,bbb):

        avec=np.empty(self.np,) 
        #avec[grd.vp2]=aaa[grd.vp2]*((vec[grd.vp2e]+vec[grd.vp2w]-2*vec[grd.vp2])/(grd.dx1d[grd.vp2]**2)+(vec[grd.vp2n]+vec[grd.vp2s]-2*vec[grd.vp2])/(grd.dy1d[grd.vp2]**2)) + bbb[grd.vp2]*vec[grd.vp2]
        avec[self.vp2]=aaa[self.vp2]*((vec[self.vp2e]+vec[self.vp2w]-2*vec[self.vp2])/(self.dx1d[self.vp2]**2)+(vec[self.vp2n]+vec[self.vp2s]-2*vec[self.vp2])/(self.dy1d[self.vp2]**2)) + bbb[self.vp2]*vec[self.vp2]
        avec[self.vp1]=vec[self.vp1]
        
        return avec,



    def pv2h(self,q,hg):
        """ Q to SSH
        
        This code solve a linear system of equations using Conjugate Gradient method
    
        Args:
            q (2D array): Potential Vorticity field
            hg (2D array): SSH guess
            grd (Grid() object): check modgrid.py
    
        Returns:
            h (2D array): SSH field. 
        """    

        x = +hg[self.indi,self.indj]
        q1d = q[self.indi,self.indj]
    
        ccc = +q1d
        ccc[self.vp1] = x[self.vp1]

        r = self.h2pv_1d(x) - ccc
        
        
        d = -r
        
        alpha = self.alpha(d,r)
        xnew = x + alpha*d
        
        if self.qgiter>1:
            for itr in range(self.qgiter): 
                
                # Update guess value
                x = +xnew
                
                # Compute beta
                rnew = self.h2pv_1d(xnew) - ccc
                
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
    
        # back to 2D
        h = np.empty((self.ny,self.nx))
        h[:,:] = np.NAN
        h[self.indi,self.indj] = xnew[:]
    
        return h

    def qrhs(self,u,v,q,way):

        """ PV increment, upwind scheme
    
        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)
    
        Returns:
            rq (2D array): PV increment  
    
        """
        rq = np.zeros((self.ny,self.nx))
          
        if not self.diff:
            uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            
            uplus[np.where((uplus<0))] = 0
            uminus[np.where((uminus>0))] = 0
            vplus[np.where((vplus<0))] = 0
            vminus[np.where((vminus>=0))] = 0
            
            rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq(uplus,vplus,uminus,vminus,q)
            
            if self.mdt is not None:
                
                uplusbar = way*self.uplusbar
                uplusbar[np.where((uplusbar<0))] = 0
                vplusbar = way*self.vplusbar
                vplusbar[np.where((vplusbar<0))] = 0
                uminusbar = way*self.uminusbar
                uminusbar[np.where((uminusbar>0))] = 0
                vminusbar = way*self.vminusbar
                vminusbar[np.where((vminusbar>0))] = 0
                
                rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq(
                    uplusbar,vplusbar,uminusbar,vminusbar,q)
                rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq(uplus,vplus,uminus,vminus,self.qbar)
                
            rq[2:-2,2:-2] = rq[2:-2,2:-2] - way*\
                (self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/(2*self.dy[2:-2,2:-2])\
                    *0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    
        #diffusion
        if self.Kdiffus is not None:
            rq[2:-2,2:-2] = rq[2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (q[2:-2,3:-1]+q[2:-2,1:-3]-2*q[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (q[3:-1,2:-2]+q[1:-3,2:-2]-2*q[2:-2,2:-2])
            
        rq[np.where((self.mask<=1))] = 0
        rq[np.isnan(rq)] = 0

        return rq
    
    def _rq(self,uplus,vplus,uminus,vminus,q):
        
        """
            main function for upwind schemes
        """
        
        if self.upwind==1:
            return self._rq1(uplus,vplus,uminus,vminus,q)
        elif self.upwind==2:
            return self._rq2(uplus,vplus,uminus,vminus,q)
        elif self.upwind==3:
            return self._rq3(uplus,vplus,uminus,vminus,q)
        
    def _rq1(self,uplus,vplus,uminus,vminus,q):
        
        """
            1st-order upwind scheme
        """
        
        res = \
            - uplus*1/(self.dx[2:-2,2:-2]) * (q[2:-2,2:-2]-q[2:-2,1:-3]) \
            + uminus*1/(self.dx[2:-2,2:-2])* (q[2:-2,2:-2]-q[2:-2,3:-1]) \
            - vplus*1/(self.dy[2:-2,2:-2]) * (q[2:-2,2:-2]-q[1:-3,2:-2]) \
            + vminus*1/(self.dy[2:-2,2:-2])* (q[2:-2,2:-2]-q[3:-1,2:-2])
        
        return res
    
    def _rq2(self,uplus,vplus,uminus,vminus,q):
        
        """
            2nd-order upwind scheme
        """
        
        res = \
            - uplus*1/(2*self.dx[2:-2,2:-2])*\
                (3*q[2:-2,2:-2]-4*q[2:-2,1:-3]+q[2:-2,:-4]) \
            + uminus*1/(2*self.dx[2:-2,2:-2])*\
                (q[2:-2,4:]-4*q[2:-2,3:-1]+3*q[2:-2,2:-2])  \
            - vplus*1/(2*self.dy[2:-2,2:-2])*\
                (3*q[2:-2,2:-2]-4*q[1:-3,2:-2]+q[:-4,2:-2]) \
            + vminus*1/(2*self.dy[2:-2,2:-2])*\
                (q[4:,2:-2]-4*q[3:-1,2:-2]+3*q[2:-2,2:-2])
        
        return res

    def _rq3(self,uplus,vplus,uminus,vminus,q):
        
        """
            3rd-order upwind scheme
        """
        
        res = \
            - uplus*1/(6*self.dx[2:-2,2:-2])*\
                (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]-6*q[2:-2,1:-3]+q[2:-2,:-4]) \
            + uminus*1/(6*self.dx[2:-2,2:-2])*\
                (q[2:-2,4:]-6*q[2:-2,3:-1]+3*q[2:-2,2:-2]+2*q[2:-2,1:-3])  \
            - vplus*1/(6*self.dy[2:-2,2:-2])*\
                (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-6*q[1:-3,2:-2]+q[:-4,2:-2]) \
            + vminus*1/(6*self.dy[2:-2,2:-2])*\
                (q[4:,2:-2]-6*q[3:-1,2:-2]+3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
        
        return res
            
    
    def step(self,h0,q0=None,way=1):
        
        """ Propagation 
    
        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)
    
        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)
    
        """
        
        if np.all(h0==0):
            if q0 is None:
                return h0
            else:
                return h0,q0
   
        # 1/ h-->q
        if q0 is None:
            qb0 = self.h2pv(h0)
        else:
            qb0 = +q0

        # 2/ h-->(u,v)
        u,v = self.h2uv(h0)
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        
        # 3/ (u,v,q)-->rq
        rq = self.qrhs(u,v,qb0,way)
        
        # 4/ increment integration 
        q1 = qb0 + self.dt*rq
        
        # 5/ q-->h
        h1 = self.pv2h(q1,+h0)
        
        if q0 is None:
            return h1
        else:
            return h1,q1
             

