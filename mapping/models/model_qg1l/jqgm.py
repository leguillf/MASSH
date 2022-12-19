import jax.numpy as jnp
import numpy as np 
from jax import jit
from jax import jvp,vjp
import matplotlib.pylab as plt 
import numpy
import jax 
from functools import partial
jax.config.update("jax_enable_x64", True)


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return jnp.fft.irfft(-1j*jnp.pad(x, (1,1)), axis=-1, norm=norm)[1:x.shape[0]+1,1:x.shape[1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T



def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f.astype(jnp.float64)) / operator_dst)

def inverse_elliptic_dst_tgl(self,dh0,h0):
        
    _,dh1 = jvp(self.inverse_elliptic_dst, (h0,), (dh0,))
    
    return dh1

def inverse_elliptic_dst_adj(self,adh0,h0):
    
    _, adf = vjp(self.inverse_elliptic_dst, h0)
    
    return adf(adh0)[0]


class Qgm:
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self,dx=None,dy=None,dt=None,SSH=None,c=None,upwind=3,
                 g=9.81,f=1e-4,diff=False,Kdiffus=None, hbc=None,
                 mdt=None,mdu=None,mdv=None,time_scheme='Euler', *args, **kwargs):
        
        # Grid shape
        ny,nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (np.nanmean(dx)+np.nanmean(dy))/2

        self.dx = dx * np.ones((ny,nx))
        self.dy = dy * np.ones((ny,nx))
        
        # Time step
        self.dt = dt
        
        # Gravity
        self.g = g
        
        # Coriolis
        if hasattr(f, "__len__") and f.shape==self.dx.shape:
            self.f0 = np.nanmean(f) * np.ones((ny,nx))
        else: 
            self.f0 = f * np.ones((ny,nx))
            
        # Rossby radius  
        if hasattr(c, "__len__") and c.shape==self.dx.shape:
            self.c = c
        else: 
            self.c = np.nanmean(f) * np.ones_like(self.dx)
        
        # Mask array
        mask = np.zeros((ny,nx))+2
        
        mask[:2,:] = 1
        mask[:,:2] = 1
        mask[-3:,:] = 1
        mask[:,-3:] = 1
        
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
                for p1 in range(-2,3):
                    for p2 in range(-2,3):
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                          if mask[itest,jtest]==2:
                              mask[itest,jtest] = 1
        
        self.mask = mask
        self.ind1 = np.where((mask==1))
        self.ind0 = np.where((mask==0))

        # Spatial scheme
        self.upwind = upwind
        
        # Time scheme 
        self.time_scheme = time_scheme

        # Diffusion 
        self.diff = diff
        self.Kdiffus = Kdiffus
        if Kdiffus is not None and Kdiffus==0:
            self.Kdiffus = None
        
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
            
        # Boundary conditions
        if hbc is not None:
            self.hbc = hbc.copy().astype('float64')
        else:
            self.hbc = np.zeros((self.ny,self.nx)).astype('float64')

        # Elliptical inversion
        x, y = np.meshgrid(np.arange(1,nx-1,dtype='float64'),
                       np.arange(1,ny-1,dtype='float64'))
        laplace_dst = 2*(np.cos(np.pi/(nx-1)*x) - 1)/np.mean(dx)**2 +\
             2*(np.cos(np.pi/(ny-1)*y) - 1)/np.mean(dy)**2
        self.helmoltz_dst = self.g/self.f0.mean() * laplace_dst  - self.g*self.f0.mean()/self.c.mean()**2
            
        
        # JIT compiling functions
        self.h2uv_jit = jit(self.h2uv)
        self.h2pv_jit = jit(self.h2pv)
        self.inverse_elliptic_dst_jit = jit(inverse_elliptic_dst)
        self.pv2h_jit = jit(self.pv2h)
        self.qrhs_jit = jit(self.qrhs)
        self.crhs_jit = jit(self.crhs)
        self._rq_jit = jit(self._rq)
        self._rq1_jit = jit(self._rq1)
        self._rq2_jit = jit(self._rq2)
        self._rq3_jit = jit(self._rq3)
        self.euler_jit = jit(self.euler)
        self.rk2_jit = jit(self.rk2)
        self.rk4_jit = jit(self.rk4)
        self.one_step_jit = jit(self.one_step)
        self.one_step_tracer_jit = jit(self.one_step_tracer)
        self.step_jit = jit(self.step,static_argnums=(2,3,))
        self.step_tgl_jit = jit(self.step_tgl,static_argnums=(3,4,))
        self.step_adj_jit = jit(self.step_adj,static_argnums=(3,4,))
        self.step_tracer_jit = jit(self.step_tracer,static_argnums=(2,3,))
        self.step_tracer_tgl_jit = jit(self.step_tracer_tgl,static_argnums=(3,4,))
        self.step_tracer_adj_jit = jit(self.step_tracer_adj,static_argnums=(3,4,))
        self.step_multiscales_jit = jit(self.step_multiscales)
        self.step_multiscales_tgl_jit = jit(self.step_multiscales_tgl)
        self.step_multiscales_adj_jit = jit(self.step_multiscales_adj)
        
        
    def h2uv(self,h,ubc=None,vbc=None):
        """ SSH to U,V
    
        Args:
            h (2D array): SSH field.
    
        Returns:
            u (2D array): Zonal velocity  
            v (2D array): Meridional velocity
    
        """
        u = jnp.zeros((self.ny,self.nx))
        v = jnp.zeros((self.ny,self.nx))
    
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
            u = u.at[self.ind1].set(ubc[self.ind1])
            #v[self.mask==1] = vbc[self.mask==1]
            v = v.at[self.ind1].set(vbc[self.ind1])
        
        u = jnp.where(jnp.isnan(u),0,u)
        v = jnp.where(jnp.isnan(v),0,v)
            
        return u,v

    def h2pv(self,h,hbc,c=None):
        """ SSH to Q
    
        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius 
    
        Returns:
            q: Potential Vorticity field  
        """
        
        if c is None:
            c = self.c
            
        q = jnp.zeros((self.ny,self.nx))
        
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
                -self.g*self.f0[self.ind1]/(c[self.ind1]**2) * hbc[self.ind1])
        #ind = np.where((self.mask==0))
        #q[ind] = 0
        q = q.at[self.ind0].set(0)
    
        return q
    
    def qrhs(self,u,v,q,hb,uls=None,vls=None,qls=None,way=1):

        """ PV increment, upwind scheme
    
        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)
    
        Returns:
            rq (2D array): PV increment  
    
        """
        rq = jnp.zeros((self.ny,self.nx))
          
        if not self.diff:
            uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
            vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
            
            #uplus[np.where((uplus<0))] = 0
            uplus = jnp.where(uplus<0, 0, uplus)
            #uminus[np.where((uminus>0))] = 0
            uminus = jnp.where(uminus>0, 0, uminus)
            #vplus[np.where((vplus<0))] = 0
            vplus = jnp.where(vplus<0, 0, vplus)
            #vminus[np.where((vminus>=0))] = 0
            vminus = jnp.where(vminus>0, 0, vminus)
            
            #rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,q)
            rq = rq.at[2:-2,2:-2].set(
                rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,q))
            
            if self.mdt is not None:
                
                uplusbar = way*self.uplusbar
                #uplusbar[np.where((uplusbar<0))] = 0
                uplusbar = jnp.where(uplusbar<0, 0, uplusbar)
                vplusbar = way*self.vplusbar
                #vplusbar[np.where((vplusbar<0))] = 0
                vplusbar = jnp.where(vplusbar<0, 0, vplusbar)
                uminusbar = way*self.uminusbar
                #uminusbar[np.where((uminusbar>0))] = 0
                uminusbar = jnp.where(uminusbar>0, 0, uminusbar)
                vminusbar = way*self.vminusbar
                #vminusbar[np.where((vminusbar>0))] = 0
                vminusbar = jnp.where(vminusbar>0, 0, vminusbar)
                
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
                uplusls = jnp.where(uplusls<0, 0, uplusls)
                # vplusls[np.where((vplusls<0))] = 0
                vplusls = jnp.where(vplusls<0, 0, vplusls)
                # uminusls[np.where((uminusls>0))] = 0
                uminusls = jnp.where(uminusls>0, 0, uminusls)
                # vminusls[np.where((vminusls>0))] = 0
                vminusls = jnp.where(vminusls>0, 0, vminusls)
                
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
            rq = rq.at[2:-2,2:-2].set(rq[2:-2,2:-2] +\
                self.Kdiffus/(self.dx[2:-2,2:-2]**2)*\
                    (q[2:-2,3:-1]+q[2:-2,1:-3]-2*q[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy[2:-2,2:-2]**2)*\
                    (q[3:-1,2:-2]+q[1:-3,2:-2]-2*q[2:-2,2:-2]))
            
        #rq[np.where((self.mask<=1))] = 0
        rq = jnp.where(self.mask<=1, 0, rq)
        #rq[np.isnan(rq)] = 0
        rq = jnp.where(jnp.isnan(rq), 0, rq)
        
        return rq
    
    def crhs(self,u,v,c,way=1):

        """ Tracer advection increment, upwind scheme
    
        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            c : tracer concentration
            way: forward (+1) or backward (-1)
    
        Returns:
            rc (2D array): Tracer advection increment
    
        """
        rc = jnp.zeros((self.ny,self.nx))
          
        uplus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        uminus = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        vplus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        vminus = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        
        #uplus[np.where((uplus<0))] = 0
        uplus = jnp.where(uplus<0, 0, uplus)
        #uminus[np.where((uminus>0))] = 0
        uminus = jnp.where(uminus>0, 0, uminus)
        #vplus[np.where((vplus<0))] = 0
        vplus = jnp.where(vplus<0, 0, vplus)
        #vminus[np.where((vminus>=0))] = 0
        vminus = jnp.where(vminus>0, 0, vminus)
        
        #rq[2:-2,2:-2] = rq[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,q)
        rc = rc.at[2:-2,2:-2].set(
            rc[2:-2,2:-2] + self._rq_jit(uplus,vplus,uminus,vminus,c))

        rc = jnp.where(self.mask<=1, 0, rc)
        rc = jnp.where(jnp.isnan(rc), 0, rc)
        
        return rc
    
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
            
    def pv2h(self,q,hbc):
        
        # Interior pv
        qbc = self.h2pv_jit(hbc,hbc).astype('float64')
        qin = q[1:-1,1:-1] - qbc[1:-1,1:-1]
    
        # Inverse sine tranfrom to get reconstructed ssh
        hrec = jnp.zeros_like(q).astype('float64')
        inv = self.inverse_elliptic_dst_jit(qin,self.helmoltz_dst)
        hrec = hrec.at[1:-1,1:-1].set(inv)

        # add the boundary value
        hrec += hbc

        return hrec

    def euler(self,q0,rq,way):

        return q0 + way * self.dt*rq

    def rk2(self,q0,rq,hb,way):

        # k2
        q12 = q0 + 0.5*rq*self.dt
        h12 = self.pv2h_jit(q12,hb)
        u12,v12 = self.h2uv_jit(h12)
        u12 = jnp.where(jnp.isnan(u12),0,u12)
        v12 = jnp.where(jnp.isnan(v12),0,v12)
        rq12 = self.qrhs_jit(u12,v12,q12,hb,way=way)

        q1 = q0 + self.dt*rq12

        return q1
    
    def rk4(self,q0,rq,hb,way):

        # k1
        k1 = rq*self.dt
        # k2
        q2 = q0 + 0.5*k1
        h2 = self.pv2h_jit(q2,hb)
        u2,v2 = self.h2uv_jit(h2)
        u2 = jnp.where(jnp.isnan(u2),0,u2)
        v2 = jnp.where(jnp.isnan(v2),0,v2)
        rq2 = self.qrhs_jit(u2,v2,q2,hb,way=way)
        k2 = rq2*self.dt
        # k3
        q3 = q0 + 0.5*k2
        h3 = self.pv2h_jit(q3,hb)
        u3,v3 = self.h2uv_jit(h3)
        u3 = jnp.where(jnp.isnan(u3),0,u3)
        v3 = jnp.where(jnp.isnan(v3),0,v3)
        rq3 = self.qrhs_jit(u3,v3,q3,hb,way=way)
        k3 = rq3*self.dt
        # k4
        q4 = q0 + k2
        h4 = self.pv2h_jit(q4,hb)
        u4,v4 = self.h2uv_jit(h4)
        u4 = jnp.where(jnp.isnan(u4),0,u4)
        v4 = jnp.where(jnp.isnan(v4),0,v4)
        rq4 = self.qrhs_jit(u4,v4,q4,hb,way=way)
        k4 = rq4*self.dt
        # q increment
        q1 = q0 + (k1+2*k2+2*k3+k4)/6.

        return q1

    def one_step(self,h0,q0,hb,way=1):

        #  h-->(u,v)
        u,v = self.h2uv_jit(h0)

        # (u,v,q)-->rq
        rq = self.qrhs_jit(u,v,q0,hb,way=way)
        
        # 4/ increment integration 
        if self.time_scheme == 'Euler':
            q1 = self.euler_jit(q0,rq,way)
        elif self.time_scheme == 'rk2':
            q1 = self.rk2_jit(q0,rq,hb,way)
        elif self.time_scheme == 'rk4':
            q1 = self.rk4_jit(q0,rq,hb,way)
            
        # q-->h
        h1 = self.pv2h_jit(q1,hb)

        return h1,q1

    def one_step_tracer(self,h0,q0,c0,hb,way=1):

        #  h-->(u,v)
        u,v = self.h2uv_jit(h0)

        # advection term for q
        rq = self.qrhs_jit(u,v,q0,hb,way=way)
        
        # advection term(s) for tracer(s)
        rc = jnp.zeros_like(c0)
        for i in range(c0.shape[0]):
            rc = rc.at[i,:,:].set(
                self.crhs_jit(u,v,c0[i],way=way)
                )
        
        # 4/ increment integration 
        if self.time_scheme == 'Euler':
            q1 = self.euler_jit(q0,rq,way)
        elif self.time_scheme == 'rk2':
            q1 = self.rk2_jit(q0,rq,hb,way)
        elif self.time_scheme == 'rk4':
            q1 = self.rk4_jit(q0,rq,hb,way)
        c1 = self.euler_jit(c0,rc,way)
            
        # q-->h
        h1 = self.pv2h_jit(q1,hb)

        return h1,q1,c1


    def step(self,h0,hb,way=1,nstep=1):
        
        """ Propagation 
    
        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)
    
        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)
    
        """

        # h-->q
        q0 = self.h2pv_jit(h0,hb)

        # Init
        q1 = +q0
        h1 = +h0

        # Time propagation
        for _ in range(nstep):
            h1,q1 = self.one_step_jit(h1,q1,hb,way=way)
            
        # Mask
        h1 = h1.at[self.ind0].set(np.nan)
        
        return h1

    def step_tracer(self,X0,hb,way=1,nstep=1):

        # Get SSH
        h0 = +X0[0]

        # Get tracers
        c0 = X0[1:]

        # h-->q
        q0 = self.h2pv_jit(h0,hb)

        # Init
        q1 = +q0
        h1 = +h0
        c1 = +c0

        # Time propagation
        for _ in range(nstep):
            h1,q1,c1 = self.one_step_tracer_jit(h1,q1,c1,hb,way=way)
            
        # Mask
        h1 = h1.at[self.ind0].set(np.nan)
        c1 = c1.at[:,self.ind0[0],self.ind0[1]].set(np.nan)

        # Concatenate
        X1 = jnp.append(h1[jnp.newaxis,:,:],c1,axis=0)
    
        return X1



    
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
        hb = +h0[:self.ny*self.nx].reshape((self.ny,self.nx))
        hls = +h0[self.ny*self.nx:2*self.ny*self.nx].reshape((self.ny,self.nx))
        h0 = +h0[2*self.ny*self.nx:].reshape((self.ny,self.nx))
            
   
        qb0 = self.h2pv_jit(h0)
        
        # 2/ h-->(u,v)
        u,v = self.h2uv_jit(h0)
        #u[np.isnan(u)] = 0
        u = jnp.where(jnp.isnan(u),0,u)
        #v[np.isnan(v)] = 0
        v = jnp.where(jnp.isnan(v),0,v)
        
        qls = self.h2pv(hls)
        uls,vls = self.h2uv(hls)
        uls = jnp.where(jnp.isnan(uls),0,uls)
        vls = jnp.where(jnp.isnan(vls),0,vls)

        # 3/ (u,v,q)-->rq
        rq = self.qrhs_jit(u,v,qb0,uls=uls,vls=vls,qls=qls,way=way)
        
        # 4/ increment integration 
        q1 = qb0 + self.dt*rq

        # 5/ q-->h
        h1 = self.pv2h_jit(q1,hb)
        
        return jnp.concatenate((hb.flatten(),hls.flatten(),h1.flatten()))
    
        
    def step_tgl(self,dh0,h0,hb,way=1,nstep=1):
        
        _,dh1 = jvp(partial(self.step_jit,hb=hb,nstep=nstep,way=way), (h0,), (dh0,))
        
        return dh1

    def step_tracer_tgl(self,dX0,X0,hb,way=1,nstep=1):
        
        _,dh1 = jvp(partial(self.step_tracer_jit,hb=hb,nstep=nstep,way=way), (X0,), (dX0,))
        
        return dh1
    
    def step_adj(self,adh0,h0,hb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,hb=hb,nstep=nstep,way=way), h0)
        
        return adf(adh0)[0]

    def step_tracer_adj(self,adX0,X0,hb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_tracer_jit,hb=hb,nstep=nstep,way=way), X0)
        
        return adf(adX0)[0]
    
    def step_multiscales_tgl(self,dh0,h0):
        
        _,dh1 = jvp(self.step_multiscales_jit, (h0,), (dh0,))
        
        return dh1
    
    def step_multiscales_adj(self,adh0,h0):
        
        _, adf = vjp(self.step_multiscales_jit, h0,)
        
        adh1 = adf(adh0)[0]
        adh1 = jnp.where(jnp.isnan(adh1),0,adh1)
        
        return adh1
        

if __name__ == "__main__":  
    
    ny,nx = 10,10
    dx = 10e3 * jnp.ones((ny,nx))
    dy = 12e3 * jnp.ones((ny,nx))
    dt = 300
    
    SSH0 = numpy.random.random((ny,nx))#random.uniform(key,shape=(ny,nx))    
    MDT = numpy.random.random((ny,nx))
    c = 2.5

    qgm = Qgm(dx=dx,dy=dy,dt=dt,c=c,SSH=SSH0,qgiter=1,mdt=MDT)
    
    # Current trajectory
    SSH0 = jnp.array(1e-2*numpy.random.random((ny,nx)))
    
    # Perturbation
    dSSH = jnp.array(1e-2*numpy.random.random((ny,nx)))

    # Adjoint
    adSSH0 = jnp.array(1e-2*numpy.random.random((ny,nx)))

    # Tangent test        
    SSH2 = qgm.step_jit(SSH0)
    print('Tangent test:')
    for p in range(10):
        
        lambd = 10**(-p)
        
        SSH1 = qgm.step_jit(SSH0+lambd*dSSH)
        
        dSSH1 = qgm.step_tgl_jit(dh0=lambd*dSSH,h0=SSH0)
        
        mask = jnp.isnan(SSH1-SSH2-dSSH1)
        ps = jnp.linalg.norm((SSH1-SSH2-dSSH1)[~mask].flatten())/jnp.linalg.norm(dSSH1[~mask])

        print('%.E' % lambd,'%.E' % ps)
    
    # Adjoint test
    dSSH1 = qgm.step_tgl_jit(dh0=dSSH,h0=SSH0)
    adSSH1 = qgm.step_adj_jit(adSSH0,SSH0)
    mask = jnp.isnan(dSSH1+adSSH1+SSH0+dSSH)
    
    ps1 = jnp.inner(dSSH1[~mask].flatten(),adSSH0[~mask].flatten())
    ps2 = jnp.inner(dSSH[~mask].flatten(),adSSH1[~mask].flatten())
        
    print('\nAdjoint test:',ps1/ps2)


   