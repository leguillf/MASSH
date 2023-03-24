import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import jvp, vjp
import matplotlib.pylab as plt
import numpy
import jax
from functools import partial

jax.config.update("jax_enable_x64", True)

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return jnp.fft.irfft(-1j * jnp.pad(x, (1, 1)), axis=-1, norm=norm)[1:x.shape[0] + 1, 1:x.shape[1] + 1]

def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T

@jit
def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f.astype(jnp.float64)) / operator_dst)
@jit
def inverse_elliptic_dst_tgl(dh0, h0):
    _, dh1 = jvp(inverse_elliptic_dst, (h0,), (dh0,))

    return dh1

@jit
def inverse_elliptic_dst_adj(adh0, h0):
    _, adf = vjp(inverse_elliptic_dst, h0)

    return adf(adh0)[0]


class Qgm:

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, upwind=3,
                 g=9.81, f=1e-4, diff=False, Kdiffus=None, hbc=None,
                 mdt=None, mdu=None, mdv=None, time_scheme='Euler', *args, **kwargs):

        # Grid shape
        ny, nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (np.nanmean(dx) + np.nanmean(dy)) / 2
        self.dx = dx 
        self.dy = dy 

        # Time step
        self.dt = dt

        # Gravity
        self.g = g

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = np.nanmean(f) * np.ones((self.ny,self.nx))
        else:
            self.f = f * np.ones((self.ny,self.nx))


        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = c
        else:
            self.c = c * np.ones((self.ny,self.nx))

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
                for p1 in range(-2,3):
                    for p2 in range(-2,3):
                      itest=i+p1
                      jtest=j+p2
                      if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                          if mask[itest,jtest]==2:
                              mask[itest,jtest] = 1
        
        self.mask = mask
        self.ind1 = np.where((mask == 1))
        self.ind0 = np.where((mask == 0))

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Elliptical inversion
        x, y = np.meshgrid(np.arange(1, nx - 1, dtype='float64'),
                           np.arange(1, ny - 1, dtype='float64'))
        laplace_dst = 2 * (np.cos(np.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (np.cos(np.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # JIT compiling functions
        self.h2uv_jit = jit(self.h2uv)
        self.h2pv_jit = jit(self.h2pv)
        self.pv2h_jit = jit(self.pv2h)
        self.rhs_jit = jit(self.rhs)
        self._adv_jit = jit(self._adv)
        self._adv1_jit = jit(self._adv1)
        self._adv2_jit = jit(self._adv2)
        self._adv3_jit = jit(self._adv3)
        self.euler_jit = jit(self.euler)
        self.rk2_jit = jit(self.rk2)
        self.rk4_jit = jit(self.rk4)
        self.bc_jit = jit(self.bc)
        self.one_step_jit = jit(self.one_step)
        self.step_jit = jit(self.step, static_argnums=3)
        self.step_tgl_jit = jit(self.step_tgl, static_argnums=4)
        self.step_adj_jit = jit(self.step_adj, static_argnums=4)



    def h2uv(self, h):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity

        """
    
        u = - self.g / self.f[1:-1,1:] * (h[2:, :-1] + h[2:, 1:] - h[:-2, 1:] - h[:-2, :-1])  / (4 * self.dy)
        v =   self.g / self.f[1:,1:-1] * (h[1:, 2:] + h[:-1, 2:] - h[:-1, :-2] - h[1:, :-2])  / (4 * self.dx)

        u = jnp.where(jnp.isnan(u), 0, u)
        v = jnp.where(jnp.isnan(v), 0, v)

        return u, v

    def h2pv(self, h, hbc, c=None):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """

        if c is None:
            c = self.c

        q = jnp.zeros((self.ny, self.nx))

        q = q.at[1:-1, 1:-1].set(
            self.g / self.f[1:-1, 1:-1] * \
            ((h[2:, 1:-1] + h[:-2, 1:-1] - 2 * h[1:-1, 1:-1]) / self.dy ** 2 + \
             (h[1:-1, 2:] + h[1:-1, :-2] - 2 * h[1:-1, 1:-1]) / self.dx ** 2) - \
            self.g * self.f[1:-1, 1:-1] / (c[1:-1, 1:-1] ** 2) * h[1:-1, 1:-1])

        q = q.at[self.ind1].set(
                -self.g*self.f[self.ind1]/(c[self.ind1]**2) * hbc[self.ind1])
        q = q.at[self.ind0].set(0)

        return q

    def rhs(self,u,v,var0,way=1):

        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment

        """

        if len(var0.shape)==3:
            q0 = var0[0]
            c0 = var0[1:]
        else:
            q0 = var0
            c0 = None

        incr = jnp.zeros_like(var0)

        #######################
        # Upwind current
        #######################
        u_on_T = way*(u[:,1:] + u[:,:-1])/2
        v_on_T = way*(v[1:,:] + v[:-1,:])/2
        up = jnp.where(u_on_T < 0, 0, u_on_T)
        um = jnp.where(u_on_T > 0, 0, u_on_T)
        vp = jnp.where(v_on_T < 0, 0, v_on_T)
        vm = jnp.where(v_on_T > 0, 0, v_on_T)

        #######################
        # PV advection
        #######################
        rhs_q = self._adv_jit(up, vp, um, vm, q0)
        rhs_q = rhs_q.at[1:-1,1:-1].set(
            rhs_q[1:-1,1:-1] - way*\
                (self.f[2:,1:-1]-self.f[:-2,1:-1])/(2*self.dy)\
                    *0.5*(v[1:,:]+v[:-1,:]))
        rhs_q = jnp.where(jnp.isnan(rhs_q), 0, rhs_q)

        #######################
        # Tracer advection
        #######################
        incr = +rhs_q
        if c0 is not None:
            incr = incr[jnp.newaxis,:,:] # add new axis to concatenate with advected tracer
            for i in range(c0.shape[0]):
                rhs_c = jnp.zeros((self.ny,self.nx))
                rhs_c = self._adv_jit(up, vp, um, vm, c0[i])
                rhs_c = jnp.where(self.mask == 0, np.nan, rhs_c)
                rhs_c = jnp.where(jnp.isnan(rhs_c), 0, rhs_c)
                incr = jnp.append(incr[:,:],rhs_c[jnp.newaxis,:,:],axis=0)
    
        return incr
    
    def _adv(self,up, vp, um, vm,var0):
        """
            main function for upwind schemes
        """
        
        res = jnp.zeros_like(var0)

        if self.upwind == 1:
            res = res.at[1:-1,1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
        elif self.upwind == 2:
            res = res.at[2:-2,2:-2].set(self._adv2_jit(up, vp, um, vm, var0))
        elif self.upwind == 3:
            res = res.at[2:-2,2:-2].set(self._adv3_jit(up, vp, um, vm, var0))
        
        # Use first order scheme for boundary pixels
        if self.upwind>1:
            res_tmp = jnp.zeros_like(var0)
            res_tmp = res_tmp.at[1:-1, 1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
        models/model_qg1l/jqgm.py     res = res.at[self.ind1].set(res_tmp[self.ind1])
        
        return res

    def _adv1(self, up, vp, um, vm, var0):

        """
            1st-order upwind scheme
        """

        res = \
            - up  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, :-2]) \
            + um  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, 2:]) \
            - vp  / self.dy * (var0[1:-1, 1:-1] - var0[:-2, 1:-1]) \
            + vm / self.dy * (var0[1:-1, 1:-1] - var0[2:, 1:-1])

        return res

    def _adv2(self, up, vp, um, vm, var0):

        """
            2nd-order upwind scheme
        """

        res = \
            - up[1:-1,1:-1] * 1 / (2 * self.dx) * \
            (3 * var0[2:-2, 2:-2] - 4 * var0[2:-2, 1:-3] + var0[2:-2, :-4]) \
            + um[1:-1,1:-1] * 1 / (2 * self.dx) * \
            (var0[2:-2, 4:] - 4 * var0[2:-2, 3:-1] + 3 * var0[2:-2, 2:-2]) \
            - vp[1:-1,1:-1] * 1 / (2 * self.dy) * \
            (3 * var0[2:-2, 2:-2] - 4 * var0[1:-3, 2:-2] + var0[:-4, 2:-2]) \
            + vm[1:-1,1:-1] * 1 / (2 * self.dy) * \
            (var0[4:, 2:-2] - 4 * var0[3:-1, 2:-2] + 3 * var0[2:-2, 2:-2])

        return res

    def _adv3(self, up, vp, um, vm, var0):

        """
            3rd-order upwind scheme
        """

        res = \
            - up[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (2 * var0[2:-2, 3:-1] + 3 * var0[2:-2, 2:-2] - 6 * var0[2:-2, 1:-3] + var0[2:-2, :-4]) \
            + um[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (var0[2:-2, 4:] - 6 * var0[2:-2, 3:-1] + 3 * var0[2:-2, 2:-2] + 2 * var0[2:-2, 1:-3]) \
            - vp[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (2 * var0[3:-1, 2:-2] + 3 * var0[2:-2, 2:-2] - 6 * var0[1:-3, 2:-2] + var0[:-4, 2:-2]) \
            + vm[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (var0[4:, 2:-2] - 6 * var0[3:-1, 2:-2] + 3 * var0[2:-2, 2:-2] + 2 * var0[1:-3, 2:-2])

        return res

    def pv2h(self, q, hbc):

        # Interior pv
        qbc = self.h2pv_jit(hbc,hbc).astype('float64')
        qin = q[1:-1,1:-1] - qbc[1:-1,1:-1]
    
        # Inverse sine tranfrom to get reconstructed ssh
        hrec = jnp.zeros_like(q).astype('float64')
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec = hrec.at[1:-1, 1:-1].set(inv)

        # add the boundary value
        hrec += hbc

        return hrec

    def euler(self, var0, incr, way):

        return var0 + way * self.dt * incr

    def rk2(self, var0, incr, hb, way):

        # k2
        var12 = var0 + 0.5*incr*self.dt
        if len(incr.shape)==3:
            q12 = var12[0]
            c12 = var12[1:]
        else:
            q12 = +var12
        h12 = self.pv2h_jit(q12,hb)
        u12,v12 = self.h2uv_jit(h12)
        u12 = jnp.where(jnp.isnan(u12),0,u12)
        v12 = jnp.where(jnp.isnan(v12),0,v12)
        if len(incr.shape)==3:
            var12 = jnp.append(q12[jnp.newaxis,:,:],c12,axis=0)
        else:
            var12 = +q12
        incr12 = self.rhs_jit(u12,v12,var12,way=way)

        var1 = var0 + self.dt * incr12

        return var1

    def rk4(self, q0, rq, hb, way):

        # k1
        k1 = rq * self.dt
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
        q1 = q0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

        return q1



    def bc(self,var1,var0,u,v,varb):

        """
        Open Boundary Conditions for tracers, following Mellor (1996)
        """

        if len(varb.shape)==3 and varb.shape[0]>1:
            # Compute adimensional coefficients
            r1_S = 1/2 * self.dt/self.dy * (v[0,:]  + jnp.abs(v[0,:] ))
            r2_S = 1/2 * self.dt/self.dy * (v[0,:]  - jnp.abs(v[0,:] ))
            r1_N = 1/2 * self.dt/self.dy * (v[-1,:]  + jnp.abs(v[-1,:] ))
            r2_N = 1/2 * self.dt/self.dy * (v[-1,:]  - jnp.abs(v[-1,:] ))
            r1_W = 1/2 * self.dt/self.dx * (u[:,0] + jnp.abs(u[:,0]))
            r2_W = 1/2 * self.dt/self.dx* (u[:,0] - jnp.abs(u[:,0]))
            r1_E = 1/2 * self.dt/self.dx * (u[:,-1] + jnp.abs(u[:,-1]))
            r2_E = 1/2 * self.dt/self.dx * (u[:,-1] - jnp.abs(u[:,-1]))

            for i in range(1, varb.shape[0]):

                # South
                var1 = var1.at[i,0,1:-1].set(
                    var0[i,0,1:-1] - (r1_S*(var0[i,0,1:-1]-varb[i,0,1:-1]) + r2_S*(var0[i,1,1:-1]-var0[i,0,1:-1])))

                # North
                var1 = var1.at[i,-1,1:-1].set(
                    var0[i,-1,1:-1] - (r1_N*(var0[i,-1,1:-1]-var0[i,-2,1:-1]) + r2_N*(varb[i,-1,1:-1]-var0[i,-1,1:-1])))
                
                # West
                var1 = var1.at[i,1:-1,0].set(
                    var0[i,1:-1,0] - (r1_W*(var0[i,1:-1,0]-varb[i,1:-1,0]) + r2_W*(var0[i,1:-1,1]-var0[i,1:-1,0])))
                
                # East
                var1 = var1.at[i,1:-1,-1].set(
                    var0[i,1:-1,-1] - (r1_E*(var0[i,1:-1,-1]-var0[i,1:-1,-2]) + r2_E*(varb[i,1:-1,-1]-var0[i,1:-1,-1])))
            
        return var1
        
    def one_step(self, h0, var0, Xb, way=1):

        if len(Xb.shape)==3:
            hb = +Xb[0]
        else:
            hb = +Xb

        # compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # compute increment
        incr = self.rhs_jit(u,v,var0,way=way)
        
        # time integration 
        if self.time_scheme == 'Euler':
            var1 = self.euler_jit(var0, incr, way)
        elif self.time_scheme == 'rk2':
            var1 = self.rk2_jit(var0, incr, hb, way)
        elif self.time_scheme == 'rk4':
            var1 = self.rk4_jit(var0, incr, hb, way)

        # elliptical inversion 
        if len(var1.shape)==3:
            q1 = +var1[0]
        else:
            q1 = +var1
        h1 = self.pv2h_jit(q1, hb)

        # tracer boundary conditions
        var1 = self.bc_jit(var1,var0,u,v,Xb)

        return h1, var1

    
    def step(self, X0, Xb, way=1, nstep=1):

        """ Propagation

        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)

        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)

        """
        
        # Get SSH and tracers
        if len(X0.shape)==3:
            h0 = +X0[0]
            c0 = +X0[1:]
            hb = +Xb[0]
        else:
            h0 = +X0
            c0 = None
            hb = +Xb

        # h-->q
        q0 = self.h2pv_jit(h0, hb)

        # Init
        h1 = +h0
        var1 = +q0
        if c0 is not None:
            var1 = jnp.append(var1[jnp.newaxis,:,:],c0,axis=0)

        # Time propagation
        for _ in range(nstep):
            h1, var1 = self.one_step_jit(h1, var1, Xb, way=way)

        # Mask
        h1 = h1.at[self.ind0].set(np.nan)

        # Concatenate
        if len(var1.shape)==3:
            X1 = jnp.append(h1[jnp.newaxis,:,:],var1[1:],axis=0)
        else:
            X1 = +h1

        return X1


    def step_multiscales(self, h0, way=1):

        """ Propagation

        Args:
            h0 (2D array): initial SSH
            q0 (2D array, optional): initial PV
            way: forward (+1) or backward (-1)

        Returns:
            h1 (2D array): propagated SSH
            q1 (2D array): propagated PV (if q0 is provided)

        """
        hb = +h0[:self.ny * self.nx].reshape((self.ny, self.nx))
        hls = +h0[self.ny * self.nx:2 * self.ny * self.nx].reshape((self.ny, self.nx))
        h0 = +h0[2 * self.ny * self.nx:].reshape((self.ny, self.nx))

        qb0 = self.h2pv_jit(h0)

        # 2/ h-->(u,v)
        u, v = self.h2uv_jit(h0)
        # u[np.isnan(u)] = 0
        u = jnp.where(jnp.isnan(u), 0, u)
        # v[np.isnan(v)] = 0
        v = jnp.where(jnp.isnan(v), 0, v)

        qls = self.h2pv(hls)
        uls, vls = self.h2uv(hls)
        uls = jnp.where(jnp.isnan(uls), 0, uls)
        vls = jnp.where(jnp.isnan(vls), 0, vls)

        # 3/ (u,v,q)-->rq
        rq = self.qrhs_jit(u, v, qb0, uls=uls, vls=vls, qls=qls, way=way)

        # 4/ increment integration
        q1 = qb0 + self.dt * rq

        # 5/ q-->h
        h1 = self.pv2h_jit(q1, hb)

        return jnp.concatenate((hb.flatten(), hls.flatten(), h1.flatten()))

    def step_tgl(self, dX0, X0, Xb, way=1, nstep=1):

        _, dX1 = jvp(partial(self.step_jit, Xb=Xb, nstep=nstep, way=way), (X0,), (dX0,))

        return dX1
    
    def step_adj(self,adX0,X0,Xb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,Xb=Xb,nstep=nstep,way=way), X0)
        
        return adf(adX0)[0]
    
    def step_multiscales_tgl(self,dh0,h0):
        
        _,dh1 = jvp(self.step_multiscales_jit, (h0,), (dh0,))
        
        return dh1

    def step_multiscales_adj(self, adh0, h0):

        _, adf = vjp(self.step_multiscales_jit, h0, )

        adh1 = adf(adh0)[0]
        adh1 = jnp.where(jnp.isnan(adh1), 0, adh1)

        return adh1


if __name__ == "__main__":

    ny, nx = 10, 10
    dx = 10e3 * jnp.ones((ny, nx))
    dy = 12e3 * jnp.ones((ny, nx))
    dt = 300

    SSH0 = numpy.random.random((ny, nx))  # random.uniform(key,shape=(ny,nx))
    MDT = numpy.random.random((ny, nx))
    hbc = np.zeros((ny, nx)).astype('float64')
    c = 2.5

    qgm = Qgm(dx=dx, dy=dy, dt=dt, c=c, SSH=SSH0, qgiter=1, mdt=MDT)

    # Current trajectory
    SSH0 = jnp.array(1e-2 * numpy.random.random((ny, nx)))

    # Perturbation
    dSSH = jnp.array(1e-2 * numpy.random.random((ny, nx)))

    # Adjoint
    adSSH0 = jnp.array(1e-2 * numpy.random.random((ny, nx)))

    # Tangent test
    SSH2 = qgm.step_jit(h0=SSH0, hb=hbc)
    # SSH2 = qgm.step(h0=SSH0, hb=hbc)
    print('Tangent test:')
    for p in range(10):
        lambd = 10 ** (-p)

        SSH1 = qgm.step_jit(h0=SSH0 + lambd * dSSH, hb=hbc)
        dSSH1 = qgm.step_tgl_jit(dh0=lambd * dSSH, h0=SSH0, hb=hbc)

        # SSH1 = qgm.step(h0=SSH0 + lambd * dSSH, hb=hbc)
        # dSSH1 = qgm.step_tgl(dh0=lambd * dSSH, h0=SSH0, hb=hbc)

        mask = jnp.isnan(SSH1 - SSH2 - dSSH1)
        ps = jnp.linalg.norm((SSH1 - SSH2 - dSSH1)[~mask].flatten()) / jnp.linalg.norm(dSSH1[~mask])

        print('%.E' % lambd, '%.E' % ps)

    # Adjoint test
    dSSH1 = qgm.step_tgl_jit(dh0=dSSH, h0=SSH0, hb=hbc)
    adSSH1 = qgm.step_adj_jit(adh0=SSH0, h0=SSH0, hb=hbc)
    # dSSH1 = qgm.step_tgl(dh0=dSSH, h0=SSH0, hb=hbc)
    # adSSH1 = qgm.step_adj(adh0=SSH0, h0=SSH0, hb=hbc)
    mask = jnp.isnan(dSSH1 + adSSH1 + SSH0 + dSSH)

    ps1 = jnp.inner(dSSH1[~mask].flatten(), adSSH0[~mask].flatten())
    ps2 = jnp.inner(dSSH[~mask].flatten(), adSSH1[~mask].flatten())

    print('\nAdjoint test:', ps1 / ps2)