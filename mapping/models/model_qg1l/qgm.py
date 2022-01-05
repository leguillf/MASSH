import numpy as np
import matplotlib.pylab as plt 



class Qgm:
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,
                 g=9.81,f=1e-4,qgiter=1,diff=False,snu=None,
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
        
        # Diffusion 
        self.diff = diff
        self.snu = snu
        if snu is not None and snu==0:
            self.snu = None
        
        # Nb of iterations for elliptical inversion
        self.qgiter = qgiter
        
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
        else:
            u[self.mask==1] = 0
            v[self.mask==1] = 0
        u[self.mask==0] = 0
        v[self.mask==0] = 0
    
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
        
    def norm(self,r):
        return np.linalg.norm(r)
    
    def alpha(self,r,d):
        return self.norm(r)**2/(d.ravel().dot(self.h2pv(d).ravel()))
    
    def beta(self,r,rnew):
        return self.norm(rnew)**2 / self.norm(r)**2
    
    def pv2h(self,q,hg):
        
        """ compute SSH from PV
    
        Args:
            q (2D array): PV
            hg (2D array): background SSH

    
        Returns:
            h (2D array): SSH
    
        """
        if np.all(q==0):
            return hg
        
        q_tmp = +q
        
        q_tmp[self.mask==0] = 0
        hg[self.mask==0] = 0
        
        r = +q_tmp - self.h2pv(hg)
        d = +r
        alpha = self.alpha(r,d)
        h = hg + alpha*d
        if self.qgiter>1:
            for itr in range(self.qgiter): 
                # Update guess value
                hg = +h
                # Compute beta
                rnew = r - alpha * self.h2pv(d)
                beta = self.beta(r,rnew)
                r = +rnew
                # Compute new direction
                dnew = r + beta * d
                alpha = self.alpha(r,dnew)
                d = +dnew
                # Update SSH
                h = hg + alpha*d 
                
        h[self.mask==0] = np.nan
        
        return h
    
    
    def qrhs(self,u,v,q,way):

        """ Q increment
    
        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)
    
        Returns:
            rq (2D array): Q increment  
    
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
        if self.snu is not None:
            rq[2:-2,2:-2] = rq[2:-2,2:-2] +\
                self.snu/(self.dx[2:-2,2:-2]**2)*\
                    (q[2:-2,3:-1]+q[2:-2,1:-3]-2*q[2:-2,2:-2]) +\
                self.snu/(self.dy[2:-2,2:-2]**2)*\
                    (q[3:-1,2:-2]+q[1:-3,2:-2]-2*q[2:-2,2:-2])
            
        rq[np.where((self.mask<=1))] = 0
        rq[np.isnan(rq)] = 0

        return rq
    
    
    def _rq(self,uplus,vplus,uminus,vminus,q):
        
        res = \
            - uplus*1/(6*self.dx[2:-2,2:-2])*\
                (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]-6*q[2:-2,1:-3]+q[2:-2,:-4])\
            + uminus*1/(6*self.dx[2:-2,2:-2])*\
                (q[2:-2,4:]-6*q[2:-2,3:-1]+3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
            - vplus*1/(6*self.dy[2:-2,2:-2])*\
                (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-6*q[1:-3,2:-2]+q[:-4,2:-2])  \
            + vminus*1/(6*self.dy[2:-2,2:-2])*\
                (q[4:,2:-2]-6*q[3:-1,2:-2]+3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
        
        return res
            
        

    
    def step(self,h0,q0=None,dphidt=None,way=1):
        
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
        
        
        # 3/ (u,v,q)-->rq
        rq = self.qrhs(u,v,qb0,way)
        
        # 4/ increment integration 
        q1 = qb0 + self.dt*rq
        if dphidt is not None:
            q1 += self.dt*dphidt
        
        # 5/ q-->h
        h1 = self.pv2h(q1,+h0)
        
        if q0 is None:
            return h1
        else:
            return h1,q1
             

