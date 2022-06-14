import jax.numpy as np
from jax import jit
from jax import jvp,vjp
import matplotlib.pylab as plt 
import numpy
from jax.config import config
config.update("jax_enable_x64", True)


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
        
        mask = mask.at[:2,:].set(1)
        mask = mask.at[:,:2].set(1)
        mask = mask.at[-2:,:].set(1)
        mask = mask.at[:,-2:].set(1)
        
    
        if SSH is not None and mdt is not None:
            isNAN = np.isnan(SSH) | np.isnan(mdt)
        elif SSH is not None:
            isNAN = np.isnan(SSH)
        elif mdt is not None:
            isNAN = np.isnan(mdt)
        else:
            isNAN = None
            
        if isNAN is not None: 
            mask = mask.at[isNAN].set(0)
            indNan = np.argwhere(isNAN)
            for i,j in indNan:
                for p1 in [-1,0,1]:
                    for p2 in [-1,0,1]:
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                          if mask[itest,jtest]==2:
                              mask = mask.at[itest,jtest].set(1)
        
                              

        self.mask = mask
        self.ind1 = np.where((mask==1))
        self.ind0 = np.where((mask==0))

        
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
        self.indi=np.zeros((_np), dtype='int')
        self.indj=np.zeros((_np), dtype='int')
        self.vp1=np.zeros((_np1), dtype='int')
        self.vp2=np.zeros((_np2), dtype='int')
        self.vp2=np.zeros((_np2), dtype='int')
        self.vp2n=np.zeros((_np2), dtype='int')
        self.vp2nn=np.zeros((_np2), dtype='int')
        self.vp2s=np.zeros((_np2), dtype='int')
        self.vp2ss=np.zeros((_np2), dtype='int')
        self.vp2e=np.zeros((_np2), dtype='int')
        self.vp2ee=np.zeros((_np2), dtype='int')
        self.vp2w=np.zeros((_np2), dtype='int')
        self.vp2ww=np.zeros((_np2), dtype='int')
        self.vp2nw=np.zeros((_np2), dtype='int')
        self.vp2ne=np.zeros((_np2), dtype='int')
        self.vp2se=np.zeros((_np2), dtype='int')
        self.vp2sw=np.zeros((_np2), dtype='int')
        self.indp=np.zeros((ny,nx), dtype='int')
        
        p=-1
        for i in range(ny):
          for j in range(nx):
            if (mask[i,j]>=1):
              p=p+1
              #self.mask1d[p]= mask[i,j]
              self.mask1d = self.mask1d.at[p].set(mask[i,j])
              #self.H[p]=SSH[i,j]
              self.H = self.H.at[p].set(SSH[i,j])
              #self.dx1d[p]=dx[i,j]
              self.dx1d = self.dx1d.at[p].set(dx[i,j])
              #self.dy1d[p]=dy[i,j]
              self.dy1d = self.dy1d.at[p].set(dy[i,j])
              #self.f01d[p]=self.f0[i,j]
              self.f01d = self.f01d.at[p].set(self.f0[i,j])
              #self.c1d[p]=self.c[i,j]
              self.c1d = self.c1d.at[p].set(self.c[i,j])
              #self.indi[p]=i
              self.indi = self.indi.at[p].set(i)
              #self.indj[p]=j
              self.indj = self.indj.at[p].set(j)
              #self.indp[i,j]=p
              self.indp = self.indp.at[i,j].set(p)
     
        
        
        p2=-1
        p1=-1
        for p in range(_np):
          if (self.mask1d[p]==2):
            p2=p2+1
            i=self.indi[p]
            j=self.indj[p]
            #self.vp2[p2]=p
            self.vp2 = self.vp2.at[p2].set(p)
            #self.vp2n[p2]=self.indp[i+1,j]
            self.vp2n = self.vp2n.at[p2].set(self.indp[i+1,j])
            #self.vp2nn[p2]=self.indp[i+2,j]
            self.vp2nn = self.vp2nn.at[p2].set(self.indp[i+2,j])
            #self.vp2s[p2]=self.indp[i-1,j]
            self.vp2s = self.vp2s.at[p2].set(self.indp[i-1,j])
            #self.vp2ss[p2]=self.indp[i-2,j]
            self.vp2ss = self.vp2ss.at[p2].set(self.indp[i-2,j])
            #self.vp2e[p2]=self.indp[i,j+1]
            self.vp2e = self.vp2e.at[p2].set(self.indp[i,j+1])
            #self.vp2ee[p2]=self.indp[i,j+2]
            self.vp2ee = self.vp2ee.at[p2].set(self.indp[i,j+2])
            #self.vp2w[p2]=self.indp[i,j-1]
            self.vp2w = self.vp2w.at[p2].set(self.indp[i,j-1])
            #self.vp2ww[p2]=self.indp[i,j-2]
            self.vp2ww = self.vp2ww.at[p2].set(self.indp[i,j-2])
            #self.vp2nw[p2]=self.indp[i+1,j-1]
            self.vp2bnw = self.vp2nw.at[p2].set(self.indp[i+1,j-1])
            #self.vp2ne[p2]=self.indp[i+1,j+1]
            self.vp2ne = self.vp2ne.at[p2].set(self.indp[i+1,j+1])
            #self.vp2se[p2]=self.indp[i-1,j+1]
            self.vp2se = self.vp2se.at[p2].set(self.indp[i-1,j+1])
            #self.vp2sw[p2]=self.indp[i-1,j-1]
            self.vp2nw = self.vp2nw.at[p2].set(self.indp[i-1,j-1])
          if (self.mask1d[p]==1):
            p1=p1+1
            #self.vp1[p1]=p
            self.vp1 = self.vp1.at[p1].set(p)
        
        self.aaa = self.g/self.f01d
        self.bbb = - self.g*self.f01d / self.c1d**2
        self.aaa = self.aaa.at[self.vp1].set(0)
        self.bbb = self.bbb.at[self.vp1].set(1)
        
        # Spatial scheme
        self.upwind = upwind

        # Diffusion 
        self.diff = diff
        self.Kdiffus = Kdiffus
        if Kdiffus is not None and Kdiffus==0:
            self.Kdiffus = None
        
        # Nb of iterations for elliptical inversion
        self.qgiter = qgiter
            
        
        self.hbc = np.zeros((self.ny,self.nx))
            
        
        # JIT compiling functions
        self.h2uv_jit = jit(self.h2uv)
        self.h2pv_jit = jit(self.h2pv)
        self.h2pv_1d_jit = jit(self.h2pv_1d)
        self.alpha_jit = jit(self.alpha)
        self.pv2h_jit = jit(self.pv2h)
        self.qrhs_jit = jit(self.qrhs)
        self._rq_jit = jit(self._rq)
        self._rq1_jit = jit(self._rq1)
        self._rq2_jit = jit(self._rq2)
        self._rq3_jit = jit(self._rq3)
        self.step_jit = jit(self.step)
        self.step_tgl_jit = jit(self.step_tgl)
        self.step_adj_jit = jit(self.step_adj)
        self.step_multiscales_jit = jit(self.step_multiscales)
        self.step_multiscales_tgl_jit = jit(self.step_multiscales_tgl)
        self.step_multiscales_adj_jit = jit(self.step_multiscales_adj)
        
        
        # MDT
        self.mdt = mdt
        if self.mdt is not None:
            if mdu is  None or mdv is  None:
                self.ubar,self.vbar = self.h2uv_jit(self.mdt)
                self.qbar = self.h2pv_jit(self.mdt,c=np.nanmean(self.c)*np.ones_like(self.dx))
            else:
                self.ubar = mdu
                self.vbar = mdv
                self.qbar = self.huv2pv(mdt,mdu,mdv,c=np.nanmean(self.c)*np.ones_like(self.dx))
                self.mdt = self.pv2h_jit(self.qbar,+mdt)
            #self.ubar,self.vbar = self.h2uv_jit(self.mdt,ubc=mdu,vbc=mdv)
            #self.qbar = self.h2pv_jit(self.mdt)
            #self.qbar = self.huv2pv(self.ubar,self.vbar,self.mdt,c=np.nanmean(self.c)*np.ones_like(self.dx))
            #self.mdt = self.pv2h_jit(self.qbar,+mdt)
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
    
        # u[1:-1,1:] = - self.g/self.f0[1:-1,1:]*\
        #     (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy[1:-1,1:])
        u = u.at[1:-1,1:].set(- self.g/self.f0[1:-1,1:]*\
         (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy[1:-1,1:]))
             
        # v[1:,1:-1] = + self.g/self.f0[1:,1:-1]*\
        #     (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx[1:,1:-1])
        v = v.at[1:,1:-1].set(self.g/self.f0[1:,1:-1]*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx[1:,1:-1]))
        
        if ubc is not None and vbc is not None:
            #u[self.mask==1] = ubc[self.mask==1]
            u = u.at[self.mask==1].set(ubc[self.mask==1])
            #v[self.mask==1] = vbc[self.mask==1]
            v = v.at[self.mask==1].set(vbc[self.mask==1])
            
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
        
        # q[1:-1,1:-1] = self.g/self.f0[1:-1,1:-1]*\
        #     ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 +\
        #       (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) -\
        #         self.g*self.f0[1:-1,1:-1]/(c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        q = q.at[1:-1,1:-1].set(
            self.g/self.f0[1:-1,1:-1]*\
            ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 +\
              (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) -\
                self.g*self.f0[1:-1,1:-1]/(c[1:-1,1:-1]**2) *h[1:-1,1:-1])
        
        #ind = np.where((self.mask==1))
        #q[ind] = -self.g*self.f0[ind]/(c[ind]**2) * h[ind]#self.hbc[ind]
        q = q.at[self.ind1].set(
            -self.g*self.f0[self.ind1]/(c[self.ind1]**2) * h[self.ind1])
        #ind = np.where((self.mask==0))
        #q[ind] = 0
        q = q.at[self.ind0].set(0)
    
        return q
    
    def h2pv_2(self,h):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            grd (Grid() object): check modgrid.py
    
        Returns:
            q: Potential Vorticity field  
        """
        g=self.g
        
        q=- g*self.f0/(self.c**2) *h 
    
        q[1:-1,1:-1] = g/self.f0[1:-1,1:-1]*((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 \
                                          + (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) \
                                          - g*self.f0[1:-1,1:-1]/(self.c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        
        ind=np.where((self.mask==1))
        
        q[ind]=- g*self.f0[ind]/(self.c[ind]**2) *h[ind]
    
        qtemp=- g*self.f0/(self.c**2) *h
        q[np.where((np.isnan(q)))]=qtemp[np.where((np.isnan(q)))]
        ind=np.where((self.mask==0))
        
        q[ind]=0
    
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
        
        # q1d[self.vp2] = self.aaa[self.vp2]*\
        #     ((h1d[self.vp2e]+h1d[self.vp2w]-2*h1d[self.vp2])/self.dx1d[self.vp2]**2 +\
        #      (h1d[self.vp2n]+h1d[self.vp2s]-2*h1d[self.vp2])/self.dy1d[self.vp2]**2) +\
        #         self.bbb[self.vp2] * h1d[self.vp2]
        q1d = q1d.at[self.vp2].set(
            self.aaa[self.vp2]*\
                ((h1d[self.vp2e]+h1d[self.vp2w]-2*h1d[self.vp2])/self.dx1d[self.vp2]**2 +\
                 (h1d[self.vp2n]+h1d[self.vp2s]-2*h1d[self.vp2])/self.dy1d[self.vp2]**2) +\
                    self.bbb[self.vp2] * h1d[self.vp2])
            
        #q1d[self.vp1] = +h1d[self.vp1]
        q1d = q1d.at[self.vp1].set(+h1d[self.vp1])
    
        return q1d

    def alpha(self,d,r):
        # tmp = np.dot(d,self.h2pv_1d(d))
        # if tmp!=0. : 
        #     return -np.dot(d,r)/tmp
        # else: 
        #     return 1.
        return np.where(np.dot(d,self.h2pv_1d_jit(d))==0., 1, -np.dot(d,r)/np.dot(d,self.h2pv_1d_jit(d)))
    

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
    
        r = self.h2pv_1d_jit(x) - ccc
        #r[self.vp1] = 0 ## boundary condition     
        r = r.at[self.vp1].set(0)
        
        d = -r
        
        alpha = self.alpha_jit(d,r)
        xnew = x + alpha*d
        
        if self.qgiter>1:
            for itr in range(self.qgiter): 
                
                # Update guess value
                x = +xnew
                
                # Compute beta
                rnew = self.h2pv_1d_jit(xnew) - ccc
                #rnew[self.vp1] = 0 ## boundary condition  
                rnew = rnew.at[self.vp1].set(0)
                #a1 = np.dot(r,r)
                #a2 = np.dot(rnew,rnew)
                # if a1!=0:
                #     beta = a2/a1
                # else: 
                #     beta = 1.
                beta = np.where(np.dot(r,r)!=0,
                                np.dot(rnew,rnew)/np.dot(r,r),
                                1)
                r = +rnew
                
                # Compute new direction
                dnew = -rnew + beta*d
                d = +dnew
                
                # Update state
                alpha = self.alpha_jit(d,r)
                xnew = x + alpha*d
        
    
        # back to 2D
        h = np.empty((self.ny,self.nx))
        #h[:,:] = np.NAN
        h = h.at[:,:].set(numpy.NAN)
        #h[self.indi,self.indj] = xnew[:]
        h = h.at[self.indi,self.indj].set(xnew[:])
    
        return h

    
    def qrhs(self,u,v,q,uls=None,vls=None,qls=None,way=1):

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
            
            #uplus[np.where((uplus<0))] = 0
            uplus = np.where(uplus<0, 0, uplus)
            #uminus[np.where((uminus>0))] = 0
            uminus = np.where(uminus>0, 0, uminus)
            #vplus[np.where((vplus<0))] = 0
            vplus = np.where(vplus<0, 0, vplus)
            #vminus[np.where((vminus>=0))] = 0
            vminus = np.where(vminus>0, 0, vminus)
            
            #rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,q)
            rq = rq.at[2:-2,2:-2].set(
                rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,q))
            
            if self.mdt is not None:
                
                uplusbar = way*self.uplusbar
                #uplusbar[np.where((uplusbar<0))] = 0
                uplusbar = np.where(uplusbar<0, 0, uplusbar)
                vplusbar = way*self.vplusbar
                #vplusbar[np.where((vplusbar<0))] = 0
                vplusbar = np.where(vplusbar<0, 0, vplusbar)
                uminusbar = way*self.uminusbar
                #uminusbar[np.where((uminusbar>0))] = 0
                uminusbar = np.where(uminusbar>0, 0, uminusbar)
                vminusbar = way*self.vminusbar
                #vminusbar[np.where((vminusbar>0))] = 0
                vminusbar = np.where(vminusbar>0, 0, vminusbar)
                
                # rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq_jit(
                #     uplusbar,vplusbar,uminusbar,vminusbar,q)
                rq = rq.at[2:-2,2:-2].set(
                    rq[2:-2,2:-2] + self._rq_jit(uplusbar,vplusbar,uminusbar,vminusbar,q))
                # rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,self.qbar)
                rq = rq.at[2:-2,2:-2].set(
                    rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,self.qbar))
                
            if uls is not None:

                uplusls  = way * 0.5*(uls[2:-2,2:-2]+uls[2:-2,3:-1])
                uminusls = way * 0.5*(uls[2:-2,2:-2]+uls[2:-2,3:-1])
                vplusls  = way * 0.5*(vls[2:-2,2:-2]+vls[3:-1,2:-2])
                vminusls = way * 0.5*(vls[2:-2,2:-2]+vls[3:-1,2:-2])
            
                # uplusls[np.where((uplusls<0))] = 0
                uplusls = np.where(uplusls<0, 0, uplusls)
                # vplusls[np.where((vplusls<0))] = 0
                vplusls = np.where(vplusls<0, 0, vplusls)
                # uminusls[np.where((uminusls>0))] = 0
                uminusls = np.where(uminusls>0, 0, uminusls)
                # vminusls[np.where((vminusls>0))] = 0
                vminusls = np.where(vminusls>0, 0, vminusls)
                
                # rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq(
                #     uplusls,vplusls,uminusls,vminusls,q)
                rq = rq.at[2:-2,2:-2].set(
                    rq[2:-2,2:-2] + self._rq_jit(uplusls,vplusls,uminusls,vminusls,q))
                # rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq(
                #     uplus,vplus,uminus,vminus,qls)
                rq = rq.at[2:-2,2:-2].set(
                    rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,qls))
                
            # rq[2:-2,2:-2] = rq[2:-2,2:-2] - way*\
            #     (self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/(2*self.dy[2:-2,2:-2])\
            #         *0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            rq = rq.at[2:-2,2:-2].set(
                rq[2:-2,2:-2] - way*\
                     (self.f0[3:-1,2:-2]-self.f0[1:-3,2:-2])/(2*self.dy[2:-2,2:-2])\
                         *0.5*(v[2:-2,2:-2]+v[3:-1,2:-2]))
    
        #diffusion
        if self.Kdiffus is not None:
            rq[2:-2,2:-2] = rq[2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (q[2:-2,3:-1]+q[2:-2,1:-3]-2*q[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (q[3:-1,2:-2]+q[1:-3,2:-2]-2*q[2:-2,2:-2])
            
        #rq[np.where((self.mask<=1))] = 0
        rq = np.where(self.mask<=1, 0, rq)
        #rq[np.isnan(rq)] = 0
        rq = np.where(np.isnan(rq), 0, rq)
        
        return rq
    
    def _rq(self,uplus,vplus,uminus,vminus,q):
        
        """
            main function for upwind schemes
        """
        
        if self.upwind==1:
            return self._rq1_jit(uplus,vplus,uminus,vminus,q)
        elif self.upwind==2:
            return self._rq2_jit(uplus,vplus,uminus,vminus,q)
        elif self.upwind==3:
            return self._rq3_jit(uplus,vplus,uminus,vminus,q)
        
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
            
    
    def step(self,h0,way=1):
        
        """ Propagation 
    
        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)
    
        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)
    
        """
        qb0 = self.h2pv_jit(h0)
        
        # 2/ h-->(u,v)
        u,v = self.h2uv_jit(h0)
        #u[np.isnan(u)] = 0
        u = np.where(np.isnan(u),0,u)
        #v[np.isnan(v)] = 0
        v = np.where(np.isnan(v),0,v)

        # 3/ (u,v,q)-->rq
        rq = self.qrhs_jit(u,v,qb0,way=way)
        
        # 4/ increment integration 
        q1 = qb0 + self.dt*rq

        # 5/ q-->h
        h1 = self.pv2h_jit(q1,h0)
        
        return h1
    
    def step_multiscales(self,h0,way=1):
        
        """ Propagation 
    
        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)
    
        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)
    
        """
        hls = +h0[:self.ny*self.nx].reshape((self.ny,self.nx))
        h0 = +h0[self.ny*self.nx:].reshape((self.ny,self.nx))
            
   
        qb0 = self.h2pv_jit(h0)
        
        # 2/ h-->(u,v)
        u,v = self.h2uv_jit(h0)
        #u[np.isnan(u)] = 0
        u = np.where(np.isnan(u),0,u)
        #v[np.isnan(v)] = 0
        v = np.where(np.isnan(v),0,v)
        
        qls = self.h2pv(hls)
        uls,vls = self.h2uv(hls)
        uls = np.where(np.isnan(uls),0,uls)
        vls = np.where(np.isnan(vls),0,vls)

        # 3/ (u,v,q)-->rq
        rq = self.qrhs_jit(u,v,qb0,uls=uls,vls=vls,qls=qls,way=way)
        
        # 4/ increment integration 
        q1 = qb0 + self.dt*rq

        # 5/ q-->h
        h1 = self.pv2h_jit(q1,h0)
        
        return np.concatenate((hls.flatten(),h1.flatten()))
    
        
    def step_tgl(self,dh0,h0):
        
        _,dh1 = jvp(self.step_jit, (h0,), (dh0,))
        
        return dh1
    
    def step_adj(self,adh0,h0,adhls=None,hls=None):
        
        _, adf = vjp(self.step_jit, h0)
        
        return adf(adh0)[0]
    
    def step_multiscales_tgl(self,dh0,h0):
        
        _,dh1 = jvp(self.step_multiscales_jit, (h0,), (dh0,))
        
        return dh1
    
    def step_multiscales_adj(self,adh0,h0):
        
        _, adf = vjp(self.step_multiscales_jit, h0,)
        
        adh1 = adf(adh0)[0]
        adh1 = np.where(np.isnan(adh1),0,adh1)
        
        return adh1
        

if __name__ == "__main__":  
    
    ny,nx = 10,10
    dx = 10e3 * np.ones((ny,nx))
    dy = 12e3 * np.ones((ny,nx))
    dt = 300
    
    SSH0 = numpy.random.random((ny,nx))#random.uniform(key,shape=(ny,nx))    
    MDT = numpy.random.random((ny,nx))
    c = 2.5
    
    qgm = Qgm(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH0,qgiter=1,mdt=MDT)
    
    
    
    # Current trajectory
    SSH0 = np.array(1e-2*numpy.random.random((ny,nx)))
    
    # Perturbation
    dSSH = np.array(1e-2*numpy.random.random((ny,nx)))

    # Adjoint
    adSSH0 = np.array(1e-2*numpy.random.random((ny,nx)))

    # Tangent test        
    SSH2 = qgm.step_jit(SSH0)
    print('Tangent test:')
    for p in range(10):
        
        lambd = 10**(-p)
        
        SSH1 = qgm.step_jit(SSH0+lambd*dSSH)
        
        dSSH1 = qgm.step_tgl_jit(dh0=lambd*dSSH,h0=SSH0)
        
        mask = np.isnan(SSH1-SSH2-dSSH1)
        ps = np.linalg.norm((SSH1-SSH2-dSSH1)[~mask].flatten())/np.linalg.norm(dSSH1[~mask])

        print('%.E' % lambd,'%.E' % ps)
    
    # Adjoint test
    dSSH1 = qgm.step_tgl_jit(dh0=dSSH,h0=SSH0)
    adSSH1 = qgm.step_adj_jit(adSSH0,SSH0)
    mask = np.isnan(dSSH1+adSSH1+SSH0+dSSH)
    
    ps1 = np.inner(dSSH1[~mask].flatten(),adSSH0[~mask].flatten())
    ps2 = np.inner(dSSH[~mask].flatten(),adSSH1[~mask].flatten())
        
    print('\nAdjoint test:',ps1/ps2)





    # Current trajectory
    SSH0 = np.array(1e-2*numpy.random.random((2*ny*nx)))
    
    # Perturbation
    dSSH = np.array(1e-2*numpy.random.random((2*ny*nx)))

    # Adjoint
    adSSH0 = np.array(1e-2*numpy.random.random((2*ny*nx)))

    
    # # Tangent test        
    SSH2 = qgm.step_multiscales_jit(SSH0)
    print('Tangent test:')
    for p in range(10):
        
        lambd = 10**(-p)
        
        SSH1 = qgm.step_multiscales_jit(SSH0+lambd*dSSH)
        
        dSSH1 = qgm.step_multiscales_tgl_jit(dh0=lambd*dSSH,h0=SSH0)
        
        ps = np.linalg.norm(SSH1-SSH2-dSSH1)/np.linalg.norm(dSSH1)

        print('%.E' % lambd,'%.E' % ps)
    
    # Adjoint test
    dSSH1 = qgm.step_multiscales_tgl_jit(dh0=dSSH,h0=SSH0)
    adSSH1 = qgm.step_multiscales_adj_jit(adSSH0,SSH0)
    
    ps1 = np.inner(dSSH1,adSSH0)
    ps2 = np.inner(dSSH,adSSH1)
        
    print('\nAdjoint test:',ps1/ps2)
