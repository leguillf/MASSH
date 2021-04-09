import numpy as np




class Qgm:
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,
                 g=9.81,f=1e-4,qgiter=1,diff=False,snu=None):
        
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
        mask[:2,:]=1
        mask[:,:2]=1
        mask[-3:,:]=1
        mask[:,-3:]=1
        
        if SSH is not None:
            mask[np.where((np.isnan(SSH)))]=0
            
            indNan = np.argwhere(np.isnan(SSH))
            for i,j in indNan:
                for p1 in range(-2,3):
                    for p2 in range(-2,3):
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                        mask[itest,jtest] = 1
        self.mask = mask
        
        # Diffusion 
        self.diff = diff
        self.snu = snu
        
        # Nb of iterations for elliptical inversion
        self.qgiter = qgiter
        
        # Internal variables
        np0=np.shape(np.where(mask>=1))[1]
        np2=np.shape(np.where(mask==2))[1]
        np1=np.shape(np.where(mask==1))[1]
        self.mask1d=np.zeros((np0))
        self.H=np.zeros((np0))
        self.c1d=np.zeros((np0))
        self.f01d=np.zeros((np0))
        self.dx1d=np.zeros((np0))
        self.dy1d=np.zeros((np0))
        self.indi=np.zeros((np0), dtype=np.int)
        self.indj=np.zeros((np0), dtype=np.int)
        self.vp1=np.zeros((np1), dtype=np.int)
        self.vp2=np.zeros((np2), dtype=np.int)
        self.vp2=np.zeros((np2), dtype=np.int)
        self.vp2n=np.zeros((np2), dtype=np.int)
        self.vp2nn=np.zeros((np2), dtype=np.int)
        self.vp2s=np.zeros((np2), dtype=np.int)
        self.vp2ss=np.zeros((np2), dtype=np.int)
        self.vp2e=np.zeros((np2), dtype=np.int)
        self.vp2ee=np.zeros((np2), dtype=np.int)
        self.vp2w=np.zeros((np2), dtype=np.int)
        self.vp2ww=np.zeros((np2), dtype=np.int)
        self.vp2nw=np.zeros((np2), dtype=np.int)
        self.vp2ne=np.zeros((np2), dtype=np.int)
        self.vp2se=np.zeros((np2), dtype=np.int)
        self.vp2sw=np.zeros((np2), dtype=np.int)
        self.indp=np.zeros((ny,nx), dtype=np.int) 
    
        p=-1
        for i in range(ny):
          for j in range(nx):
            if (mask[i,j]>=1):
              p=p+1
              self.mask1d[p]=mask[i,j]
              self.dx1d[p]=dx[i,j]
              self.dy1d[p]=dy[i,j]
              self.f01d[p]=self.f0[i,j]
              self.indi[p]=i
              self.indj[p]=j
              self.indp[i,j]=p
     
     
        p2=-1
        p1=-1
        for p in range(np0):
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
        self.np0 = np0
        self.np2 = np2
        
        
    
    def h2uv(self,h):
        """ SSH to U,V
    
        Args:
            h (2D array): SSH field.
            grd (Grid() object): check modgrid.py
    
        Returns:
            u (2D array): Zonal velocity  
            v (2D array): Meridional velocity
    
        """
        u = np.zeros((self.ny,self.nx))
        v = np.zeros((self.ny,self.nx))
    
        u[1:-1,1:] = - self.g/self.f0[1:-1,1:]*\
            ( 0.25*(h[1:-1,1:]+h[1:-1,:-1]+h[2:,:-1]+h[2:,1:]) -\
             0.25*(h[1:-1,1:]+h[:-2,1:]+h[:-2,:-1]+h[1:-1,:-1]) ) /\
                self.dy[1:-1,1:]
        v[1:,1:-1] = + self.g/self.f0[1:,1:-1]*\
            ( 0.25*(h[1:,1:-1]+h[1:,2:]+h[:-1,2:]+h[:-1,1:-1]) -\
             0.25*(h[1:,1:-1]+h[:-1,1:-1]+h[:-1,:-2]+h[1:,:-2]) ) /\
                self.dx[1:,1:-1]
    
        u[np.where((self.mask<=1))]=0
        v[np.where((self.mask<=1))]=0
    
        return u,v




    def uv2rv(self,u,v):
        """ U,V to relative vorticity Qr
    
        Args:
            u (2D array): Zonal velocity  
            v (2D array): Meridional velocity
    
        Returns:
            xi (2D array): Relative vorticity
        """
        
        ny,nx, = np.shape((self.ny,self.nx))
        
        gradX_V = np.zeros((self.ny,self.nx))
        gradY_U = np.zeros((self.ny,self.nx))
        
        xi = np.zeros((self.ny,self.nx))
    
        gradY_U[1:-1,1:-1] = 0.5*(u[2:,1:-1] - u[:-2,1:-1]) / self.dy[1:-1,1:-1]
        gradX_V[1:-1,1:-1] = 0.5*(v[1:-1,2:] - v[1:-1,:-2]) / self.dx[1:-1,1:-1]
        
        xi[1:-1,1:-1] = gradX_V[1:-1,1:-1] - gradY_U[1:-1,1:-1]
        
        ind = np.where((self.mask==1))
        xi[ind] = 0
    
        return xi
    
    def h2pv(self,h):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            grd (Grid() object): check modgrid.py
    
        Returns:
            q: Potential Vorticity field  
        """
    
        q = -self.g*self.f0/(self.c**2) * h 
    
        q[1:-1,1:-1] = self.g/self.f0[1:-1,1:-1]*\
            ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 +\
             (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) -\
                self.g*self.f0[1:-1,1:-1]/(self.c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        
        ind = np.where((self.mask==1))
        q[ind]= -self.g*self.f0[ind]/(self.c[ind]**2) * h[ind]
    
        ind = np.where((np.isnan(q)))
        qtemp =- self.g*self.f0/(self.c**2) * h
        q[ind] = qtemp[ind]
        
        ind = np.where((self.mask==0))
        q[ind] = 0
    
        return q
    
    def h2rv(self,h):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            grd (Grid() object): check modgrid.py
    
        Returns:
            q: Potential Vorticity field  
        """
    
        q = np.zeros((self.ny,self.nx))
    
        q[1:-1,1:-1] = self.g/self.f0[1:-1,1:-1]*\
            ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2  +\
             (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) 
        
        ind = np.where((self.mask==1))
        q[ind] = 0
        
        ind = np.where((self.mask==0))
        q[ind]=0
    
        return q
    
    
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
    
    
        x = hg[self.indi,self.indj]
        q1d = q[self.indi,self.indj]
    
        aaa = self.g/self.f01d
        if type(self.c) is float or type(self.c) is int:
            bbb = - self.g*self.f01d / self.c**2
        else:
            bbb = - self.g*self.f01d / (self.c.ravel())**2
        ccc=+q1d
    
        aaa[self.vp1] = 0
        bbb[self.vp1] = 1
        ccc[self.vp1] = x[self.vp1]  # boundary condition
    
        vec = +x
    
        avec = self.compute_avec(vec,aaa,bbb)
        gg = avec-ccc
        p = -gg
        
        for itr in range(self.qgiter-1): 
            vec = +p
            avec = self.compute_avec(vec,aaa,bbb)
            tmp = np.dot(p,avec)
            
            if tmp!=0. : s = -np.dot(p,gg)/tmp
            else: s=1.
            
            a1 = np.dot(gg,gg)
            x = x+s*p
            vec = +x
            avec = self.compute_avec(vec,aaa,bbb)
            gg = avec-ccc
            a2 = np.dot(gg,gg)
            
            if a1!=0: beta = a2/a1
            else: beta=1.
            
            p = -gg + beta*p
        
        vec = +p
        avec = self.compute_avec(vec,aaa,bbb)
        val1 = -np.dot(p,gg)
        val2 = np.dot(p,avec)
        if (val2==0.): 
            s=1.
        else: 
            s=val1/val2
        
        a1 = np.dot(gg,gg)
        x = x+s*p
    
        # back to 2D
        h = np.empty((self.ny,self.nx))
        h[:,:] = np.NAN
        h[self.indi,self.indj] = x[:]
    
    
        return h
    
    def compute_avec(self,vec,aaa,bbb):
        
        avec = np.empty(self.np0,) 
        avec[self.vp2] = aaa[self.vp2]*\
            ((vec[self.vp2e]+vec[self.vp2w]-2*vec[self.vp2])/(self.dx1d[self.vp2]**2)+\
             (vec[self.vp2n]+vec[self.vp2s]-2*vec[self.vp2])/(self.dy1d[self.vp2]**2))+\
                bbb[self.vp2]*vec[self.vp2]
        avec[self.vp1] = vec[self.vp1]
     
        return avec
    
    def qrhs(self,u,v,q,way):

        """ Q increment
    
        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : Q start
            grd (Grid() object): check modgrid.py
            way: forward (+1) or backward (-1)
    
        Returns:
            rq (2D array): Q increment  
    
        """
        rq = np.zeros((self.ny,self.nx))
          
        if not self.diff:
            uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            uplus[np.where((uplus<0))] = 0
            uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            uminus[np.where((uminus>0))] = 0
            vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            vplus[np.where((vplus<0))] = 0
            vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            vminus[np.where((vminus>=0))] = 0
        
            rq[2:-2,2:-2] = rq[2:-2,2:-2]\
                - uplus*1/(6*self.dx[2:-2,2:-2])*\
                    (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]-6*q[2:-2,1:-3]+q[2:-2,:-4])\
                + uminus*1/(6*self.dx[2:-2,2:-2])*\
                    (q[2:-2,4:]-6*q[2:-2,3:-1]+3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
                - vplus*1/(6*self.dy[2:-2,2:-2])*\
                    (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-6*q[1:-3,2:-2]+q[:-4,2:-2])  \
                + vminus*1/(6*self.dy[2:-2,2:-2])*\
                    (q[4:,2:-2]-6*q[3:-1,2:-2]+3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
        
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
    
        return rq


    
    def step(self,Hi=None,PVi=None,way=1):
        

        ############################
        # Active variable initializations
        ############################
        h = +Hi    
        if PVi is not None:
            q = PVi
        else:
            q, = self.h2pv(h)
    
        qb = +q 
        hguess = +h
        
        ########################
        # Main routines
        ########################

        # 1/ 
        if self.diff:
            u = np.zeros((self.ny,self.nx))
            v = np.zeros((self.ny,self.nx))
        else:
            u,v = self.h2uv(h)
        
        # 2/ Advection
        rq = self.qrhs(u,v,qb,way)
        
        # 3/ Time integration 
        q = qb + self.dt*rq
        
        # 4/
        h = self.pv2h(q,hguess)

              
        ############################
        #Returning variables 
        ############################      
        if PVi is not None:
            return h,q
        else:
            return h





