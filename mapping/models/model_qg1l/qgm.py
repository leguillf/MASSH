import numpy as np
import matplotlib.pylab as plt 



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
        mask[:1,:] = 1
        mask[:,:1] = 1
        mask[-1:,:] = 1
        mask[:,-1:] = 1
        
        if SSH is not None:
            mask[np.isnan(SSH)]=0
            
            indNan = np.argwhere(np.isnan(SSH))
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
            (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy[1:-1,1:])
             
        v[1:,1:-1] = + self.g/self.f0[1:,1:-1]*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx[1:,1:-1])
    
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
        
        q = np.zeros((self.ny,self.nx))
    
        q[1:-1,1:-1] = self.g/self.f0[1:-1,1:-1]*\
            ((h[2:,1:-1]+h[:-2,1:-1]-2*h[1:-1,1:-1])/self.dy[1:-1,1:-1]**2 +\
             (h[1:-1,2:]+h[1:-1,:-2]-2*h[1:-1,1:-1])/self.dx[1:-1,1:-1]**2) -\
                self.g*self.f0[1:-1,1:-1]/(self.c[1:-1,1:-1]**2) *h[1:-1,1:-1]
        
        ind = np.where((self.mask==1))
        q[ind]= -self.g*self.f0[ind]/(self.c[ind]**2) * h[ind]
    
        ind = np.where((np.isnan(q)))
        q[ind] = 0
        
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
    
    def norm(self,r):
        return np.linalg.norm(r)
    
    def alpha(self,r,d):
        return self.norm(r)**2/(d.ravel().dot(self.h2pv(d).ravel()))
    
    def beta(self,r,rnew):
        return self.norm(rnew)**2 / self.norm(r)**2
    
    def pv2h(self,q,hg):
        q_tmp = +q
        
        q_tmp[self.mask==0] = 0
        hg[self.mask==0] = 0
        hg[np.isnan(hg)] = 0
        q_tmp[np.isnan(q_tmp)] = 0
        
        # plt.figure()
        # plt.pcolormesh(q_tmp)
        # plt.show()
        # plt.figure()
        # plt.pcolormesh(hg)
        # plt.show()
        
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


    
    def step(self,h0,q0=None,way=1):
        
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
        
        # 5/ q-->h
        h1 = self.pv2h(q1,h0)
        
        if q0 is None:
            return h1
        else:
            return h1,q1
             

if __name__ == "__main__":
    
    
    
    import xarray as xr
    from scipy.ndimage import gaussian_filter
    import matplotlib.pylab as plt
    
    
    ds = xr.open_dataset('~/WORK/Developpement/Studies/MASSH/data_Example1/init.nc')
    print(ds)
    SSH_true = ds.sossheig.data
    SSH_true[500:,500:] = np.nan
    
    plt.figure()
    plt.pcolormesh(SSH_true)
    plt.show()
    
    ny,nx = SSH_true.shape
    dx = dy = 1e3 * np.ones((ny,nx))
    dt = 300
    SSH = np.zeros((ny,nx))
    c = 2.5
    
    qgm = Qgm(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH_true,qgiter=100)
    
    ssh = +SSH_true
    t = 0
    for i in range(100000):
        if i%10000==0:
            ssh = qgm.step(ssh)
            plt.figure()
            plt.pcolormesh(ssh)
            plt.show()
        
