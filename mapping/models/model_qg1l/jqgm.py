
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import jvp, vjp
import matplotlib.pylab as plt
import numpy as np
from jax.lax import scan, fori_loop, dynamic_slice, dynamic_update_slice
from jax import vmap
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)



def gaspari_cohn(r,c):
    """
    NAME 
        bfn_gaspari_cohn

    DESCRIPTION 
        Gaspari-Cohn function. Inspired from E.Cosmes.
        
        Args: 
            r : array of value whose the Gaspari-Cohn function will be applied
            c : Distance above which the return values are zeros


        Returns:  smoothed values 
            
    """ 
    if type(r) is float or type(r) is int:
        ra = np.array([r])
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp

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
                 g=9.81, f=1e-4, time_scheme='Euler', Wbc=None, Kdiffus=None, Kdiffus_trac=None,bc_trac='OBC',
                 split_in_bins=False,lenght_bins=None,facbin=None):

        # Grid shape
        ny, nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (np.nanmean(dx) + np.nanmean(dy)) / 2
        self.dx = dx.astype('float64')
        self.dy = dy.astype('float64') 

        # Time step
        self.dt = dt

        # Gravity
        self.g = g

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = (np.nanmean(f) * np.ones((self.ny,self.nx))).astype('float64')
        else:
            self.f = (f * np.ones((self.ny,self.nx))).astype('float64')


        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = (np.nanmean(c) * np.ones((self.ny,self.nx))).astype('float64')
        else:
            self.c = c * np.ones((self.ny,self.nx)).astype('float64')

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Elliptical inversion operator
        x, y = np.meshgrid(np.arange(1, nx - 1, dtype='float64'),
                           np.arange(1, ny - 1, dtype='float64'))
        laplace_dst = 2 * (np.cos(np.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (np.cos(np.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # get land pixels
        if SSH is not None:
            isNAN = np.isnan(SSH) 
        else:
            isNAN = None

        ################
        # Mask array
        ################

        # mask=3 away from the coasts
        mask = np.zeros((ny,nx),dtype='int64')+3

        # mask=1 for borders of the domain 
        mask[0,:] = 1
        mask[:,0] = 1
        mask[-1,:] = 1
        mask[:,-1] = 1

        # mask=2 for pixels adjacent to the borders 
        mask[1,1:-1] = 2
        mask[1:-1,1] = 2
        mask[-2,1:-1] = 2
        mask[-3,1:-1] = 2
        mask[1:-1,-2] = 2
        mask[1:-1,-3] = 2

        # mask=0 on land 
        if isNAN is not None:
            mask[isNAN] = 0
            indNan = np.argwhere(isNAN)
            for i,j in indNan:
                for p1 in range(-2,3):
                    for p2 in range(-2,3):
                        itest=i+p1
                        jtest=j+p2
                        if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                            # mask=1 for coast pixels
                            if (mask[itest,jtest]>=2) and (p1 in [-1,0,1] and p2 in [-1,0,1]):
                                mask[itest,jtest] = 1   
                            # mask=1 for pixels adjacent to the coast
                            elif (mask[itest,jtest]==3):
                                mask[itest,jtest] = 2     
        
        self.mask = mask
        self.ind0 = mask==0
        self.ind1 = mask==1
        self.ind2 = mask==2
        self.ind12 = self.ind1 + self.ind2

        # Weight map to apply boundary conditions 
        if Wbc is None or np.all(Wbc==0.):
            self.Wbc = np.zeros((ny,nx),dtype='float64')
            self.Wbc[self.ind1] = 1.
        else:
            self.Wbc = Wbc.astype('float64')

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus
        self.Kdiffus_trac = Kdiffus_trac

        # BC
        self.bc_trac = bc_trac

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
        self.one_step_for_scan_jit = jit(self.one_step_for_scan)
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
    
        u = jnp.zeros((self.ny,self.nx))
        v = jnp.zeros((self.ny,self.nx))

        u = u.at[1:-1,1:].set(- self.g/self.f[1:-1,1:]*\
         (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy))
             
        v = v.at[1:,1:-1].set(self.g/self.f[1:,1:-1]*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx))
        
        u = jnp.where(jnp.isnan(u),0,u)
        v = jnp.where(jnp.isnan(v),0,v)
            
        return u,v

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

        q = jnp.zeros((self.ny, self.nx),dtype='float64')

        q = q.at[1:-1, 1:-1].set(
            self.g / self.f[1:-1, 1:-1] * \
            ((h[2:, 1:-1] + h[:-2, 1:-1] - 2 * h[1:-1, 1:-1]) / self.dy ** 2 + \
             (h[1:-1, 2:] + h[1:-1, :-2] - 2 * h[1:-1, 1:-1]) / self.dx ** 2) - \
            self.g * self.f[1:-1, 1:-1] / (c[1:-1, 1:-1] ** 2) * h[1:-1, 1:-1])

        q = jnp.where(jnp.isnan(q),0,q)

        q = q.at[self.ind12].set(- \
            self.g * self.f[self.ind12] / (c[self.ind12] ** 2) * hbc[self.ind12])
        
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

        incr = jnp.zeros_like(var0,dtype='float64')

        #######################
        # Upwind current
        #######################
        u_on_T = way*0.5*(u[1:-1,1:-1]+u[1:-1,2:])
        v_on_T = way*0.5*(v[1:-1,1:-1]+v[2:,1:-1])
        up = jnp.where(u_on_T < 0, 0, u_on_T)
        um = jnp.where(u_on_T > 0, 0, u_on_T)
        vp = jnp.where(v_on_T < 0, 0, v_on_T)
        vm = jnp.where(v_on_T > 0, 0, v_on_T)

        #######################
        # PV advection
        #######################
        rhs_q = self._adv_jit(up, vp, um, vm, q0)
        rhs_q = rhs_q.at[2:-2,2:-2].set(
                rhs_q[2:-2,2:-2] - way*\
                    (self.f[3:-1,2:-2]-self.f[1:-3,2:-2])/(2*self.dy)\
                        *0.5*(v[2:-2,2:-2]+v[3:-1,2:-2]))
        # Diffusion
        if self.Kdiffus is not None:
            rhs_q = rhs_q.at[2:-2,2:-2].set(
                rhs_q[2:-2,2:-2] +\
                self.Kdiffus/(self.dx**2)*\
                    (q0[2:-2,3:-1]+q0[2:-2,1:-3]-2*q0[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy**2)*\
                    (q0[3:-1,2:-2]+q0[1:-3,2:-2]-2*q0[2:-2,2:-2])
            )
        rhs_q = jnp.where(jnp.isnan(rhs_q), 0, rhs_q)
        rhs_q = rhs_q.at[self.ind12].set(0)
        rhs_q = rhs_q.at[self.ind0].set(0)

        #######################
        # Tracer advection
        #######################
        incr = +rhs_q
        if c0 is not None:
            incr = incr[jnp.newaxis,:,:] # add new axis to concatenate with advected tracer
            for i in range(c0.shape[0]):
                rhs_c = jnp.zeros((self.ny,self.nx),dtype='float64')
                # Advection
                rhs_c = self._adv_jit(up, vp, um, vm, c0[i])
                # Diffusion
                if self.Kdiffus_trac is not None:
                    rhs_c = rhs_c.at[2:-2,2:-2].set(
                        rhs_c[2:-2,2:-2] +\
                        self.Kdiffus_trac/(self.dx**2)*\
                            (c0[i,2:-2,3:-1]+c0[i,2:-2,1:-3]-2*c0[i,2:-2,2:-2]) +\
                        self.Kdiffus_trac/(self.dy**2)*\
                            (c0[i,3:-1,2:-2]+c0[i,1:-3,2:-2]-2*c0[i,2:-2,2:-2])
                    )
                rhs_c = jnp.where(jnp.isnan(rhs_c), 0, rhs_c)
                rhs_c = rhs_c.at[self.ind0].set(0)
                incr = jnp.append(incr[:,:],rhs_c[jnp.newaxis,:,:],axis=0)
    
        return incr
    
    def _adv(self,up, vp, um, vm, var0):
        """
            main function for upwind schemes
        """
        
        res = jnp.zeros_like(var0,dtype='float64')

        if self.upwind == 1:
            res = res.at[1:-1,1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
        elif self.upwind == 2:
            res = res.at[2:-2,2:-2].set(self._adv2_jit(up, vp, um, vm, var0))
        elif self.upwind == 3:
            res = res.at[2:-2,2:-2].set(self._adv3_jit(up, vp, um, vm, var0))

        # Use first order scheme for boundary pixels
        if self.upwind>1 and self.bc_trac=='OBC':
            res_tmp = jnp.zeros_like(var0,dtype='float64')
            res_tmp = res_tmp.at[1:-1, 1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
            res = res.at[self.ind2].set(res_tmp[self.ind2])
        
        return res

    def _adv1(self, up, vp, um, vm, var0):

        """
            1st-order upwind scheme
        """

        res = \
            - up  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, :-2]) \
            + um  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, 2:])  \
            - vp  / self.dy * (var0[1:-1, 1:-1] - var0[:-2, 1:-1]) \
            + vm  / self.dy * (var0[1:-1, 1:-1] - var0[2:, 1:-1])

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

    def pv2h(self, q, hb, qb):

        # Interior pv
        qin = q[1:-1,1:-1] - qb[1:-1,1:-1]
        
        # Inverse sine tranfrom to get reconstructed ssh
        hrec = jnp.zeros_like(q,dtype='float64')
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec = hrec.at[1:-1, 1:-1].set(inv)

        # add the boundary value
        hrec += hb

        return hrec

    def euler(self, var0, incr, way):

        return var0 + way * self.dt * incr

    def rk2(self, var0, incr, hb, qb, way):

        # k2
        var12 = var0 + 0.5*incr*self.dt
        if len(incr.shape)==3:
            q12 = var12[0]
            c12 = var12[1:]
        else:
            q12 = +var12
        h12 = self.pv2h_jit(q12,hb,qb)
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

    def rk4(self, q0, rq, hb, qb, way):

        # k1
        k1 = rq * self.dt
        # k2
        q2 = q0 + 0.5*k1
        h2 = self.pv2h_jit(q2,hb,qb)
        u2,v2 = self.h2uv_jit(h2)
        u2 = jnp.where(jnp.isnan(u2),0,u2)
        v2 = jnp.where(jnp.isnan(v2),0,v2)
        rq2 = self.qrhs_jit(u2,v2,q2,hb,way=way)
        k2 = rq2*self.dt
        # k3
        q3 = q0 + 0.5*k2
        h3 = self.pv2h_jit(q3,hb,qb)
        u3,v3 = self.h2uv_jit(h3)
        u3 = jnp.where(jnp.isnan(u3),0,u3)
        v3 = jnp.where(jnp.isnan(v3),0,v3)
        rq3 = self.qrhs_jit(u3,v3,q3,hb,way=way)
        k3 = rq3*self.dt
        # k4
        q4 = q0 + k2
        h4 = self.pv2h_jit(q4,hb,qb)
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

            # Compute adimensional coefficients fro OBC
            r1_S = 1/2 * self.dt/self.dy * (v[1,1:-1]  + jnp.abs(v[1,1:-1] ))
            r2_S = 1/2 * self.dt/self.dy * (v[1,1:-1]  - jnp.abs(v[1,1:-1] ))
            r1_N = 1/2 * self.dt/self.dy * (v[-1,1:-1]  + jnp.abs(v[-1,1:-1] ))
            r2_N = 1/2 * self.dt/self.dy * (v[-1,1:-1]  - jnp.abs(v[-1,1:-1] ))
            r1_W = 1/2 * self.dt/self.dx * (u[1:-1,1] + jnp.abs(u[1:-1,1]))
            r2_W = 1/2 * self.dt/self.dx * (u[1:-1,1] - jnp.abs(u[1:-1,1]))
            r1_E = 1/2 * self.dt/self.dx * (u[1:-1,-1] + jnp.abs(u[1:-1,-1]))
            r2_E = 1/2 * self.dt/self.dx * (u[1:-1,-1] - jnp.abs(u[1:-1,-1]))

            for i in range(1, varb.shape[0]):

                if self.bc_trac=='OBC':

                    #######################
                    # Tracer Open BC
                    #######################
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

                else:
                    var1 = var1.at[i,self.ind12].set(varb[i,self.ind12])
            
                #######################
                # Tracer Relaxation BC
                #######################
                var1 = var1.at[i,1:-1,1:-1].set(self.Wbc[1:-1,1:-1] * varb[i,1:-1,1:-1] + (1-self.Wbc[1:-1,1:-1]) * var1[i,1:-1,1:-1])
            
        return var1

    def one_step(self, h0, var0, hb, varb, way=1):

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # Boundary field for PV
        if len(varb.shape)==3:
            qb = +varb[0]
        else:
            qb = +varb
        
        # Compute increment
        incr = self.rhs_jit(u,v,var0,way=way)
        
        # Time integration 
        if self.time_scheme == 'Euler':
            var1 = self.euler_jit(var0, incr, way)
        elif self.time_scheme == 'rk2':
            var1 = self.rk2_jit(var0, incr, hb, qb, way)
        elif self.time_scheme == 'rk4':
            var1 = self.rk4_jit(var0, incr, hb, qb, way)

        # Elliptical inversion 
        if len(var1.shape)==3:
            q1 = +var1[0]
        else:
            q1 = +var1
        h1 = self.pv2h_jit(q1, hb, qb)

        # Tracer boundary conditions
        var1 = self.bc_jit(var1,var0,u,v,varb)

        return h1, var1

    def one_step_for_scan(self,X0,X):

        h1, var1, hb, varb = X0
        h1, var1 = self.one_step_jit(h1, var1, hb, varb)
        X = (h1, var1, hb, varb)

        return X,X

    
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
            h0 = +X0[0] # SSH field
            c0 = +X0[1:] # Tracer concentrations
            hb = +Xb[0] # Boundary field for SSH
            cb = +Xb[1:] # Boundary fields for tracers
        else:
            h0 = +X0
            c0 = None
            hb = +Xb
            cb = None

        # Tracer mask
        if c0 is not None:
            c0 = c0.at[:,self.ind0].set(0)
            cb = cb.at[:,self.ind0].set(0)
        # h-->q
        q0 = self.h2pv_jit(h0, hb)
        qb = self.h2pv_jit(hb, hb)

        # Init
        h1 = +h0
        var1 = +q0
        varb = +qb
        if c0 is not None:
            var1 = jnp.append(var1[jnp.newaxis,:,:],c0,axis=0)
            varb = jnp.append(varb[jnp.newaxis,:,:],cb,axis=0)

        # Time propagation
        X1, _ = scan(self.one_step_for_scan_jit, init=(h1, var1, hb, varb), xs=jnp.zeros(nstep))
        h1, var1, hb, varb = X1

        # Mask
        h1 = h1.at[self.ind0].set(jnp.nan)
        if len(var1.shape)==3:
            var1 = var1.at[1:,self.ind0].set(np.nan)

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
    
class Qgm_bins:

    #num_devices = multiprocessing.cpu_count()
    #os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, upwind=3,
                 g=9.81, f=1e-4, time_scheme='Euler', Wbc=None, Kdiffus=None, Kdiffus_trac=None,bc_trac='OBC',
                 split_in_bins=False,lenght_bins=None,facbin=None):

        # Grid shape
        ny, nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (np.nanmean(dx) + np.nanmean(dy)) / 2
        self.dx = dx.astype('float64')
        self.dy = dy.astype('float64') 

        # Time step
        self.dt = dt

        # Gravity
        self.g = np.float64(g)

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus
        self.Kdiffus_trac = Kdiffus_trac

        # BC
        self.bc_trac = bc_trac

        # Spatial bins 
        if split_in_bins and lenght_bins is not None:
            # Get dimension of a bin
            nbinx = int(2*round(lenght_bins / (2*dx *1e-3)))
            nbiny = int(2*round(lenght_bins / (2*dy *1e-3)))
        else:
            nbinx = nx
            nbiny = ny

        # Get centers of bins
        binx = np.arange(nbinx/2, nx, nbinx/facbin, dtype=int)
        biny = np.arange(nbiny/2, ny, nbiny/facbin, dtype=int)
        
        # Compute parameters on each bin
        self.start_indices = []
        self.slice_sizes = []
        self.shape = []
        self.f = []
        self.c = []
        self.Wbc = []
        self.ind0 = []
        self.ind1 = []
        self.ind2 = []
        self.ind12 = []
        self.helmoltz_dst = []
        self.Wbin = []
        self.Wbin_sum = np.zeros((ny,nx))

        for jb,_binx in enumerate(binx):
            for ib,_biny in enumerate(biny):
                
                if jb==0:
                    xmin = 0
                else:
                    xmin = _binx-int(nbinx/2)

                if ib==0:
                    ymin = 0
                else:
                    ymin = _biny-int(nbiny/2)
            
                if jb==len(binx)-1:
                    xmax = nx
                    xmin = nx - nbinx 
                else:
                    xmax = _binx+int(nbinx/2)

                if ib==len(biny)-1:
                    ymax = ny
                    ymin = ny - nbiny 
                else:
                    ymax = _biny+int(nbiny/2)

                _nx = int(xmax - xmin)
                _ny = int(ymax - ymin)

                ## mask=3 away from the coasts
                mask = np.zeros((_ny,_nx),dtype='int64')+3

                ## mask=1 for borders of the domain 
                mask[0,:] = 1
                mask[:,0] = 1
                mask[-1,:] = 1
                mask[:,-1] = 1

                ## mask=2 for pixels adjacent to the borders 
                mask[1,1:-1] = 2
                mask[1:-1,1] = 2
                mask[-2,1:-1] = 2
                mask[-3,1:-1] = 2
                mask[1:-1,-2] = 2
                mask[1:-1,-3] = 2

                # get land pixels
                if SSH is not None:
                    isNAN = np.isnan(SSH[ymin:ymax,xmin:xmax]) 
                else:
                    isNAN = None

                ## mask=0 on land 
                if isNAN is not None:
                    mask[isNAN] = 0
                    indNan = np.argwhere(isNAN)
                    for i,j in indNan:
                        for p1 in range(-2,3):
                            for p2 in range(-2,3):
                                itest=i+p1
                                jtest=j+p2
                                if ((itest>=0) & (itest<=_ny-1) & (jtest>=0) & (jtest<=_nx-1)):
                                        # mask=1 for coast pixels
                                        if (mask[itest,jtest]>=2) and (p1 in [-1,0,1] and p2 in [-1,0,1]):
                                            mask[itest,jtest] = 1   
                                        # mask=1 for pixels adjacent to the coast
                                        elif (mask[itest,jtest]==3):
                                            mask[itest,jtest] = 2     
                ind0 = mask==0
                ind1 = mask==1
                ind2 = mask==2
                ind12 = ind1+ind2

                if not np.all(ind0):

                    # Weight map to apply boundary conditions 
                    if Wbc is None or np.all(Wbc==0.):
                        _Wbc = np.zeros((_ny,_nx),dtype='float64')
                        _Wbc[ind1] = 1.
                    else:
                        _Wbc = Wbc[ymin:ymax,xmin:xmax].astype('float64')

                    # Coriolis
                    if hasattr(f, "__len__"):
                        _f = np.nanmean(f[ymin:ymax,xmin:xmax])
                    else:
                        _f = f
                    
                    # Rossby radius
                    if hasattr(c, "__len__"):
                        _c = np.nanmean(c[ymin:ymax,xmin:xmax])
                    else:
                        _c = c
                    
                    # Elliptical inversion operator
                    x, y = np.meshgrid(np.arange(1, _nx - 1, dtype='float64'),
                                    np.arange(1, _ny - 1, dtype='float64'))
                    laplace_dst = 2 * (np.cos(np.pi / (_nx - 1) * x) - 1) / self.dx ** 2 + \
                                2 * (np.cos(np.pi / (_ny - 1) * y) - 1) / self.dy ** 2
                    _helmoltz_dst = self.g / _f * np.float64(laplace_dst) - self.g * _f / _c ** 2

                    # Spatial weights for merging bins
                    winy = np.ones(_ny)
                    winx = np.ones(_nx)
                    if ib>0:
                        winy[:int(_ny-_ny/facbin)] = np.linspace(0,1,int(_ny-_ny/facbin)) #gaspari_cohn(np.arange(0,_ny),_ny/2)[:int(_ny/2)][::-1]
                    if jb>0:
                        winx[:int(_nx-_nx/facbin)] = np.linspace(0,1,int(_nx-_nx/facbin))#gaspari_cohn(np.arange(0,_nx),_nx/2)[:int(_nx/2)][::-1]
                    if ib<len(biny)-1:
                        winy[int(_ny/facbin):] = np.linspace(1,0,int(_ny-_ny/facbin))#gaspari_cohn(np.arange(0,_ny),_ny/2)[:_ny-int(_ny/2)]
                    if jb<len(binx)-1:
                        winx[int(_nx/facbin):] = np.linspace(1,0,int(_nx-_nx/facbin))#gaspari_cohn(np.arange(0,_nx),_nx/2)[:_nx-int(_nx/2)]
                    _Wbin = winy[:,np.newaxis] * winx[np.newaxis,:]

                    _Wbin_tmp = np.zeros((ny,nx))
                    _Wbin_tmp[ymin:ymax,xmin:xmax] = _Wbin
                    
                    # Append to list
                    self.start_indices.append([ymin,xmin])
                    self.ind0.append(ind0)
                    self.ind1.append(ind1)
                    self.ind2.append(ind2)
                    self.ind12.append(ind12)
                    self.Wbc.append(_Wbc)
                    self.c.append(_c)
                    self.f.append(_f)
                    self.helmoltz_dst.append(_helmoltz_dst)
                    self.Wbin.append(_Wbin)
                    self.Wbin_sum[ymin:ymax,xmin:xmax] += _Wbin

        self.start_indices = jnp.array(self.start_indices)
        self.ind0 = jnp.array(self.ind0)
        self.ind1 = jnp.array(self.ind1)
        self.ind2 = jnp.array(self.ind2)
        self.ind12 = jnp.array(self.ind12)
        self.Wbc = jnp.array(self.Wbc)
        self.c = jnp.array(self.c)
        self.f = jnp.array(self.f)
        self.helmoltz_dst = jnp.array(self.helmoltz_dst)
        self.Wbin = jnp.array(self.Wbin)
        self.slice_sizes = [nbiny,nbinx]
                

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
        self.one_step_for_scan_jit = jit(self.one_step_for_scan)
        self.step_bin_jit = jit(self.step_bin, static_argnums=(2))
        self.step_bin_vmap_jit = jit(self.step_bin_vmap, static_argnums=(1))
        self.step_jit = jit(self.step, static_argnums=3)
        self.step_tgl_jit = jit(self.step_tgl, static_argnums=4)
        self.step_adj_jit = jit(self.step_adj, static_argnums=4)



    def h2uv(self, h, ib):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity

        """
    
        u = jnp.zeros_like(h)
        v = jnp.zeros_like(h)

        u = u.at[1:-1,1:].set(- self.g/self.f[ib]*\
         (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy))
             
        v = v.at[1:,1:-1].set(self.g/self.f[ib]*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx))
        
        u = jnp.where(jnp.isnan(u),0,u)
        v = jnp.where(jnp.isnan(v),0,v)
            
        return u,v

    def h2pv(self, h, hbc, ib):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """

        q = jnp.zeros_like(h).astype('float64')

        q = q.at[1:-1, 1:-1].set(
            self.g / self.f[ib] * \
            ((h[2:, 1:-1] + h[:-2, 1:-1] - 2 * h[1:-1, 1:-1]) / self.dy ** 2 + \
             (h[1:-1, 2:] + h[1:-1, :-2] - 2 * h[1:-1, 1:-1]) / self.dx ** 2) - \
            self.g * self.f[ib] / (self.c[ib] ** 2) * h[1:-1, 1:-1])

        q = jnp.where(jnp.isnan(q),0.,q)
        q = jnp.where(self.ind12[ib], 
                      - self.g * self.f[ib] / (self.c[ib] ** 2) * hbc, q) 
        q = jnp.where(self.ind0[ib], 0., q) 

        return q

    def pv2h(self, q, hb, qb, ib):

        # Interior pv
        qin = q[1:-1,1:-1] - qb[1:-1,1:-1]
        
        # Inverse sine tranfrom to get reconstructed ssh
        hrec = jnp.zeros_like(q, dtype='float64')
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst[ib])
        hrec = hrec.at[1:-1, 1:-1].set(inv.astype('float64'))

        # add the boundary value
        hrec += hb

        return hrec

    def rhs(self,u,v,q0,c0,ib,way=1):

        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment

        """

        #######################
        # Upwind current
        #######################
        u_on_T = way*0.5*(u[1:-1,1:-1]+u[1:-1,2:])
        v_on_T = way*0.5*(v[1:-1,1:-1]+v[2:,1:-1])
        up = jnp.where(u_on_T < 0, 0, u_on_T)
        um = jnp.where(u_on_T > 0, 0, u_on_T)
        vp = jnp.where(v_on_T < 0, 0, v_on_T)
        vm = jnp.where(v_on_T > 0, 0, v_on_T)

        #######################
        # PV advection
        #######################
        rhs_q = self._adv_jit(up, vp, um, vm, q0, ib)

        # Diffusion
        if self.Kdiffus is not None:
            rhs_q = rhs_q.at[2:-2,2:-2].set(
                rhs_q[2:-2,2:-2] +\
                self.Kdiffus/(self.dx**2)*\
                    (q0[2:-2,3:-1]+q0[2:-2,1:-3]-2*q0[2:-2,2:-2]) +\
                self.Kdiffus/(self.dy**2)*\
                    (q0[3:-1,2:-2]+q0[1:-3,2:-2]-2*q0[2:-2,2:-2])
            )
        
        rhs_q = jnp.where(jnp.isnan(rhs_q), 0, rhs_q)
        rhs_q = jnp.where(self.ind12[ib], 0, rhs_q)
        rhs_q = jnp.where(self.ind0[ib], 0, rhs_q)

        #######################
        # Tracer advection
        #######################
        rhs_c = None
        if c0 is not None:
            rhs_c = jnp.zeros_like(c0)
            for i in range(c0.shape[0]):
                _rhs_c = jnp.zeros((self.ny,self.nx), dtype='float64')
                # Advection
                _rhs_c = self._adv_jit(up, vp, um, vm, c0[i], ib)
                # Diffusion
                if self.Kdiffus_trac is not None:
                    _rhs_c = _rhs_c.at[2:-2,2:-2].set(
                        _rhs_c[2:-2,2:-2] +\
                        self.Kdiffus_trac/(self.dx**2)*\
                            (c0[i,2:-2,3:-1]+c0[i,2:-2,1:-3]-2*c0[i,2:-2,2:-2]) +\
                        self.Kdiffus_trac/(self.dy**2)*\
                            (c0[i,3:-1,2:-2]+c0[i,1:-3,2:-2]-2*c0[i,2:-2,2:-2])
                    )
                _rhs_c = jnp.where(jnp.isnan(_rhs_c), 0, _rhs_c)
                _rhs_c = jnp.where(self.ind0[ib], 0, _rhs_c)
                rhs_c = rhs_c.at[i].set(_rhs_c)
    
        return rhs_q, rhs_c
    
    def _adv(self,up, vp, um, vm, var0, ib):
        """
            main function for upwind schemes
        """
        
        res = jnp.zeros_like(var0,dtype='float64')

        if self.upwind == 1:
            res = res.at[1:-1,1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
        elif self.upwind == 2:
            res = res.at[2:-2,2:-2].set(self._adv2_jit(up, vp, um, vm, var0))
        elif self.upwind == 3:
            res = res.at[2:-2,2:-2].set(self._adv3_jit(up, vp, um, vm, var0))

        # Use first order scheme for boundary pixels
        if self.upwind>1 and self.bc_trac=='OBC':
            res_tmp = jnp.zeros_like(var0,dtype='float64')
            res_tmp = res_tmp.at[1:-1, 1:-1].set(self._adv1_jit(up, vp, um, vm, var0))
            res = jnp.where(self.ind2[ib], res_tmp, res)
        
        return res

    def _adv1(self, up, vp, um, vm, var0):

        """
            1st-order upwind scheme
        """

        res = \
            - up  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, :-2]) \
            + um  / self.dx * (var0[1:-1, 1:-1] - var0[1:-1, 2:])  \
            - vp  / self.dy * (var0[1:-1, 1:-1] - var0[:-2, 1:-1]) \
            + vm  / self.dy * (var0[1:-1, 1:-1] - var0[2:, 1:-1])

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

    def euler(self, q0, c0, rhs):

        q1 = q0 + self.dt * rhs[0]
        if c0 is not None:
            c1 = c0 + self.dt * rhs[1]
        else:
            c1 = None
        
        return q1, c1

    def rk2(self, q0, c0, rhs, hb, qb, ib):

        # k2
        q12 = q0 + 0.5*self.dt * rhs[0]
        if c0 is not None:
            c12 = c0 + 0.5*self.dt * rhs[1]
        else:
            c12 = None
        h12 = self.pv2h_jit(q12,hb,qb,ib)
        u12,v12 = self.h2uv_jit(h12,ib)
        u12 = jnp.where(jnp.isnan(u12),0,u12)
        v12 = jnp.where(jnp.isnan(v12),0,v12)

        rhs12 = self.rhs_jit(u12,v12,q12,c12,ib)
        q1 = q0 + self.dt * rhs12[0]
        if c0 is not None:
            c1 = c0 + self.dt * rhs12[1]
        else:
            c1 = None

        return q1,c1

    def rk4(self, q0, rq, hb, qb, way):

        # k1
        k1 = rq * self.dt
        # k2
        q2 = q0 + 0.5*k1
        h2 = self.pv2h_jit(q2,hb,qb)
        u2,v2 = self.h2uv_jit(h2)
        u2 = jnp.where(jnp.isnan(u2),0,u2)
        v2 = jnp.where(jnp.isnan(v2),0,v2)
        rq2 = self.qrhs_jit(u2,v2,q2,hb,way=way)
        k2 = rq2*self.dt
        # k3
        q3 = q0 + 0.5*k2
        h3 = self.pv2h_jit(q3,hb,qb)
        u3,v3 = self.h2uv_jit(h3)
        u3 = jnp.where(jnp.isnan(u3),0,u3)
        v3 = jnp.where(jnp.isnan(v3),0,v3)
        rq3 = self.qrhs_jit(u3,v3,q3,hb,way=way)
        k3 = rq3*self.dt
        # k4
        q4 = q0 + k2
        h4 = self.pv2h_jit(q4,hb,qb)
        u4,v4 = self.h2uv_jit(h4)
        u4 = jnp.where(jnp.isnan(u4),0,u4)
        v4 = jnp.where(jnp.isnan(v4),0,v4)
        rq4 = self.qrhs_jit(u4,v4,q4,hb,way=way)
        k4 = rq4*self.dt
        # q increment
        q1 = q0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

        return q1
    
    def bc(self,c1,c0,u,v,cb):

        """
        Open Boundary Conditions for tracers, following Mellor (1996)
        """

        # Compute adimensional coefficients fro OBC
        r1_S = 1/2 * self.dt/self.dy * (v[1,1:-1]  + jnp.abs(v[1,1:-1] ))
        r2_S = 1/2 * self.dt/self.dy * (v[1,1:-1]  - jnp.abs(v[1,1:-1] ))
        r1_N = 1/2 * self.dt/self.dy * (v[-1,1:-1]  + jnp.abs(v[-1,1:-1] ))
        r2_N = 1/2 * self.dt/self.dy * (v[-1,1:-1]  - jnp.abs(v[-1,1:-1] ))
        r1_W = 1/2 * self.dt/self.dx * (u[1:-1,1] + jnp.abs(u[1:-1,1]))
        r2_W = 1/2 * self.dt/self.dx * (u[1:-1,1] - jnp.abs(u[1:-1,1]))
        r1_E = 1/2 * self.dt/self.dx * (u[1:-1,-1] + jnp.abs(u[1:-1,-1]))
        r2_E = 1/2 * self.dt/self.dx * (u[1:-1,-1] - jnp.abs(u[1:-1,-1]))

        for i in range(c1.shape[0]):

            if self.bc_trac=='OBC':

                #######################
                # Tracer Open BC
                #######################
                # South
                c1 = c1.at[i,0,1:-1].set(
                    c0[i,0,1:-1] - (r1_S*(c0[i,0,1:-1]-cb[i,0,1:-1]) + r2_S*(c0[i,1,1:-1]-c0[i,0,1:-1])))

                # North
                c1 = c1.at[i,-1,1:-1].set(
                    c0[i,-1,1:-1] - (r1_N*(c0[i,-1,1:-1]-c0[i,-2,1:-1]) + r2_N*(cb[i,-1,1:-1]-c0[i,-1,1:-1])))
                
                # West
                c1 = c1.at[i,1:-1,0].set(
                    c0[i,1:-1,0] - (r1_W*(c0[i,1:-1,0]-cb[i,1:-1,0]) + r2_W*(c0[i,1:-1,1]-c0[i,1:-1,0])))
                
                # East
                c1 = c1.at[i,1:-1,-1].set(
                    c0[i,1:-1,-1] - (r1_E*(c0[i,1:-1,-1]-c0[i,1:-1,-2]) + r2_E*(cb[i,1:-1,-1]-c0[i,1:-1,-1])))

            else:
                c1 = c1.at[i,self.ind12].set(cb[i,self.ind12])
        
            #######################
            # Tracer Relaxation BC
            #######################
            c1 = c1.at[i,1:-1,1:-1].set(self.Wbc[1:-1,1:-1] * cb[i,1:-1,1:-1] + (1-self.Wbc[1:-1,1:-1]) * c1[i,1:-1,1:-1])
            
        return c1

    def one_step(self, h0, c0, hb, cb, qb, ib):

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0, ib)

        # Compute PV
        q0 = self.h2pv_jit(h0, hb, ib)
        
        # Compute increment
        rhs = self.rhs_jit(u, v, q0, c0, ib)
        
        # Time integration 
        if self.time_scheme == 'Euler':
            q1, c1 = self.euler_jit(q0, c0, rhs)
        elif self.time_scheme == 'rk2':
            q1, c1 = self.rk2_jit(q0, c0, rhs, hb, qb, ib)
        elif self.time_scheme == 'rk4':
            q1, c1 = self.rk4_jit(q0, c0, rhs, hb, qb, ib)

        # Elliptical inversion 
        h1 = self.pv2h_jit(q1, hb, qb, ib)

        # Tracer boundary conditions
        if c0 is not None:
            c1 = self.bc_jit(c1, c0, u, v, cb)

        return h1, c1

    def one_step_for_scan(self, X0, X, ib=0):

        h1, c1, hb, cb, qb = X0
        h1, c1 = self.one_step_jit(h1, c1, hb, cb, qb, ib)
        X = (h1, c1, hb, cb, qb)

        return X, X

    def step_bin(self, ib, X0, nstep):

        h0, c0, hb, cb = X0

        # Select bin
        _h0 = dynamic_slice(h0, self.start_indices[ib], self.slice_sizes) 
        _hb = dynamic_slice(hb, self.start_indices[ib], self.slice_sizes)

        if c0 is not None:
            _c0 = c0[:,self.slicey[ib,0]:self.slicey[ib,1],self.slicex[ib,0]:self.slicex[ib,1]]
            _cb = cb[:,self.slicey[ib,0]:self.slicey[ib,1],self.slicex[ib,0]:self.slicex[ib,1]]
        else:
            _c0 = None
            _cb = None
        
        # Mask
        if _c0 is not None:
            _c0 = _c0.at[:,self.ind0[ib]].set(0)
            _cb = _cb.at[:,self.ind0[ib]].set(0)

        # Compute boundary PV field
        _qb = self.h2pv_jit(_hb, _hb, ib)
        
        # Time stepping
        X1, _ = scan(partial(self.one_step_for_scan_jit, ib=ib), 
                        init=(_h0, _c0, _hb, _cb, _qb), xs=jnp.zeros(nstep))
        _h1 = X1[0]
        _c1 = X1[1]
        
        # Spatial smoothing
        _h1 *= self.Wbin[ib]
        if _c1 is not None:
            _c1 *= self.Wbin[ib]

        # Mask
        _h1 = jnp.where(self.ind0[ib], jnp.nan, _h1)
        if _c1 is not None:
            _c1 = jnp.where(self.ind0[ib], jnp.nan, _c1)

        # Fill new array
        h1 = jnp.zeros((self.ny,self.nx))
        h1 = dynamic_update_slice(h1, _h1, self.start_indices[ib])
        if c0 is not None:
            c1 = jnp.zeros((self.ny,self.nx))
            c1 = dynamic_update_slice(c1, _c1, self.start_indices[ib])
        else:
            c1 = None
        
        return h1, c1
    
    def step_bin_vmap(self,X0,nstep):

        return vmap(partial(self.step_bin_jit, nstep=nstep), in_axes=(0,None))(jnp.arange(0,len(self.Wbin)),X0)
    
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
            h0 = +X0[0] # SSH field
            c0 = +X0[1:] # Tracer concentrations
            hb = +Xb[0] # Boundary field for SSH
            cb = +Xb[1:] # Boundary fields for tracers
        else:
            h0 = +X0
            c0 = None
            hb = +Xb
            cb = None
         
        h1, c1 = self.step_bin_vmap_jit((h0, c0, hb, cb), nstep)

        # Smooth bins
        h1 = h1.sum(axis=0)/self.Wbin_sum
        if c1 is not None:
            c1 = c1.sum(axis=0)/self.Wbin_sum

        # Concatenate
        if c0 is not None:
            X1 = jnp.append(h1[jnp.newaxis,:,:], c1, axis=0)
        else:
            X1 = +h1

        return X1

    def step_tgl(self, dX0, X0, Xb, way=1, nstep=1):

        _, dX1 = jvp(partial(self.step_jit, Xb=Xb, nstep=nstep, way=way), (X0,), (dX0,))

        return dX1
    
    def step_adj(self,adX0,X0,Xb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_jit, Xb=Xb, nstep=nstep, way=way), X0)
        
        return adf(adX0)[0]
    
if __name__ == "__main__":

    import timeit
    

    ny, nx = 100, 100
    dx = 10e3 * jnp.ones((ny, nx))
    dy = 12e3 * jnp.ones((ny, nx))
    dt = 300

    SSH0 = np.random.random((ny, nx)).astype('float64')
    c = 2.7*np.ones((ny, nx),dtype='float64')
    f = 1e-4*np.ones((ny, nx),dtype='float64')

    qgm = Qgm(dx=dx, dy=dy, dt=dt, c=c, f=f, SSH=SSH0)

    SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

    ####################
    # h2pv
    ####################
    if False:
        print('*** h2pv ***')
        # Current trajectory
        SSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Perturbation
        dSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Adjoint
        adPV = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        def h2pv_tgl(dh0, h0, hb):

            _, dh1 = jvp(partial(qgm.h2pv, hbc=hb, ib=0), (h0,), (dh0,))

            return dh1
        
        def h2pv_adj(adq0,h0,hb):
            
            _, adf = vjp(partial(qgm.h2pv, hbc=hb, ib=0), h0)
            
            return adf(adq0)[0]

        # Forward
        PV0 = qgm.h2pv(SSH, SSHb, ib=0).astype('float64')
    
        print('Tangent test:')
        for p in range(10):
            lambd = 10 ** (-p)

            PV1 = qgm.h2pv(SSH + lambd * dSSH, SSHb, ib=0).astype('float64')
            dPV = h2pv_tgl(lambd * dSSH, SSH, SSHb).astype('float64')

            ps = jnp.linalg.norm((PV1 - PV0 - dPV).flatten()) / jnp.linalg.norm(dPV)

            print('%.E' % lambd, '%.E' % ps)
        
        # Adjoint test
        dPV = h2pv_tgl(dSSH, SSH, SSHb).astype('float64')
        adSSH = h2pv_adj(adPV, SSH, SSHb).astype('float64')

        ps1 = jnp.inner(dPV.flatten(), adPV.flatten())
        ps2 = jnp.inner(dSSH.flatten(), adSSH.flatten())

        print('\nAdjoint test:', ps1 / ps2)

    ####################
    # pv2h
    ####################
    if False:
        print('*** pv2h ***')
        # Current trajectory
        PV = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')
        PVb = qgm.h2pv(SSHb, SSHb, ib=0).astype('float64')

        # Perturbation
        dPV = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Adjoint
        adSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        def pv2h_tgl(dq, q, hb, qb):

            _, dh = jvp(partial(qgm.pv2h, hb=hb, qb=qb, ib=0), (q,), (dq,))

            return dh
        
        def pv2h_adj(adh, q, hb, qb):
            
            _, adf = vjp(partial(qgm.pv2h, hb=hb, qb=qb, ib=0), q)
            
            return adf(adh)[0]

        # Forward
        SSH = qgm.pv2h(PV, SSHb, PVb, ib=0).astype('float64')
        
        print('Tangent test:')
        for p in range(10):
            lambd = 10 ** (-p)

            SSH1 = qgm.pv2h(PV + lambd * dPV, SSHb, PVb, ib=0).astype('float64')
            dSSH = pv2h_tgl(lambd * dPV, PV, SSHb, PVb).astype('float64')

            ps = jnp.linalg.norm((SSH1 - SSH - dSSH).flatten()) / jnp.linalg.norm(dSSH)

            print('%.E' % lambd, '%.E' % ps)
        
        # Adjoint test
        dSSH = pv2h_tgl(dPV, PV, SSHb, PVb).astype('float64')
        adPV = pv2h_adj(adSSH, PV, SSHb, PVb).astype('float64')

        ps1 = jnp.inner(dPV.flatten(), adPV.flatten())
        ps2 = jnp.inner(dSSH.flatten(), adSSH.flatten())

        print('\nAdjoint test:', ps1 / ps2)
    
    
    ####################
    # step
    ####################
    if False:
        print('*** step ***')
        # Current trajectory
        SSH0 = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Perturbation
        dSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Adjoint
        adSSH0 = jnp.array(1e-2 * np.random.random((ny, nx))).astype('float64')

        # Tangent test
        SSH2 = qgm.step_jit(X0=SSH0, Xb=SSHb).astype('float64')
        print('Tangent test:')
        for p in range(10):
            lambd = 10 ** (-p)

            SSH1 = qgm.step_jit(X0=SSH0 + lambd * dSSH, Xb=SSHb).astype('float64')
            dSSH1 = qgm.step_tgl_jit(dX0=lambd * dSSH, X0=SSH0, Xb=SSHb).astype('float64')

            mask = jnp.isnan(SSH1 - SSH2 - dSSH1)
            ps = jnp.linalg.norm((SSH1 - SSH2 - dSSH1)[~mask].flatten()) / jnp.linalg.norm(dSSH1[~mask])

            print('%.E' % lambd, '%.E' % ps)

        # Adjoint test
        dSSH1 = qgm.step_tgl_jit(dX0=dSSH, X0=SSH0, Xb=SSHb).astype('float64')
        adSSH1 = qgm.step_adj_jit(adX0=adSSH0, X0=SSH0, Xb=SSHb).astype('float64')

        mask = jnp.isnan(dSSH1 + adSSH1 + SSH0 + dSSH)

        ps1 = jnp.inner(dSSH1[~mask].flatten(), adSSH0[~mask].flatten())
        ps2 = jnp.inner(dSSH[~mask].flatten(), adSSH1[~mask].flatten())

        print('\nAdjoint test:', ps1 / ps2)