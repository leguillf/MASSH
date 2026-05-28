
import sys 
sys.path.insert(0, '../../src') # add src to path to import modules
from src.config import USE_FLOAT64
import numpy as np
from jax import jit
from jax import jvp, vjp
import matplotlib.pylab as plt
import numpy as np
from jax.lax import scan, dynamic_update_slice
from jax import vmap
from jax.scipy.sparse.linalg import cg as jcg
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", USE_FLOAT64)



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

def dynamic_slice_tile(array, y_start, x_start, tile_height, tile_width):
    """
    Dynamically slice a tile from the array.
    
    Args:
        array (jax.numpy.ndarray): Input array to slice from.
        y_start (int): Starting y-index.
        x_start (int): Starting x-index.
        tile_height (int): Height of the tile.
        tile_width (int): Width of the tile.
    
    Returns:
        jax.numpy.ndarray: The sliced tile.
    """
    return jax.lax.dynamic_slice(
        array,
        (y_start, x_start),
        (tile_height, tile_width)
    )

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
    if USE_FLOAT64:
        _f = f.astype('float64')
    else:
        _f = f.astype('float32')
    return dstI2D(dstI2D(_f) / operator_dst)
@jit
def inverse_elliptic_dst_tgl(dh0, h0):
    _, dh1 = jvp(inverse_elliptic_dst, (h0,), (dh0,))

    return dh1

@jit
def inverse_elliptic_dst_adj(adh0, h0):
    _, adf = vjp(inverse_elliptic_dst, h0)

    return adf(adh0)[0]

def cg(q_flat, h2pv_operator, tol=1e-5, maxiter=1000):
    return jcg(h2pv_operator, q_flat, tol=tol, maxiter=maxiter)


class Qgm:

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, Kdiffus=None, upwind=3, g=9.81, f=1e-4, time_scheme='Euler', compile=True, mdt=None, bathymetry_PV_term=None, formulation='ssh', ** kwargs):

        if USE_FLOAT64:
            self.dtype = 'float64'
        else:
            self.dtype = 'float32'

        # Formulation: 'ssh' (work in SSH space) or 'sf' (work in streamfunction space)
        self.formulation = formulation

        # Grid shape
        ny, nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        if hasattr(dx, "__len__"):
            self.dx = dx[int(ny/2),int(nx/2)].astype(self.dtype)
            self.dy = dy[int(ny/2),int(nx/2)].astype(self.dtype) 
        else:
            self.dx = dx
            self.dy = dy

        # Time step
        self.dt = dt

        # Gravity
        self.g = g

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = f.astype(self.dtype)            # full 2D array (needed for sf h<->phi conversion)
            self.f0 = np.nanmean(f).astype(self.dtype)  # scalar reference value for QG operators
            # Beta plane
            self.beta = (f[2:,:] - f[:-2,:]) / (2*self.dy)
        else:
            self.f = np.array(f, dtype=self.dtype)
            self.f0 = np.array(f, dtype=self.dtype)
            self.beta = None


        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = np.nanmean(c).astype(self.dtype)
        else:
            self.c = c.astype(self.dtype)

        # MDT
        self.mdt = mdt

        # Bathymetry
        self.bathymetry_PV_term = bathymetry_PV_term

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Elliptical inversion operator
        x, y = np.meshgrid(np.arange(1, nx - 1, dtype=self.dtype),
                           np.arange(1, ny - 1, dtype=self.dtype))
        laplace_dst = 2 * (np.cos(np.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (np.cos(np.pi / (ny - 1) * y) - 1) / self.dy ** 2
        if self.formulation == 'sf':
            self.helmoltz_dst = laplace_dst - (self.f0 / self.c) ** 2
        else:
            self.helmoltz_dst = self.g / self.f0 * laplace_dst - self.g * self.f0 / self.c ** 2
            

        ################
        # Mask array
        ################
        # mask=3 away from the coasts
        mask = 3 * np.ones((ny,nx),dtype='int64')

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
        if SSH is not None:
            isNAN = np.isnan(SSH) # get land pixels
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

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus

        # JIT compiling functions
        if compile:
            self.h2uv_jit = jit(self.h2uv)
            self.h2pv_jit = jit(self.h2pv)
            self.pv2h_jit = jit(self.pv2h)
            self.rhs_jit = jit(self.rhs)
            self.adv_jit = jit(self.adv)
            self.euler_jit = jit(self.euler)
            self.rk2_jit = jit(self.rk2)
            self.rk3_jit = jit(self.rk3)
            self.one_step_jit = jit(self.one_step)
            self.one_step_for_scan_jit = jit(self.one_step_for_scan)
            self.step_jit = jit(self.step, static_argnums=2)
            self.step_tgl_jit = jit(self.step_tgl, static_argnums=3)
            self.step_adj_jit = jit(self.step_adj, static_argnums=3)

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

        if self.formulation == 'sf':
            u = u.at[1:-1,1:].set(- self.g/self.f[1:-1,1:]*\
             (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy))
            v = v.at[1:,1:-1].set(self.g/self.f[1:,1:-1]*\
                (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx))
        else:
            u = u.at[1:-1,1:].set(- self.g/self.f0*\
             (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy))
            v = v.at[1:,1:-1].set(self.g/self.f0*\
                (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx))
        
        u = jnp.where(jnp.isnan(u),0,u)
        v = jnp.where(jnp.isnan(v),0,v)
            
        return u,v

    def h2pv(self, h, hb, c=None):
        """ SSH to PV

        Args:
            h (2D array): SSH field.
            hb (2D array): Background SSH field

        Returns:
            q: Potential Vorticity field
        """

        if c is None:
            c = self.c

        q = jnp.zeros((self.ny, self.nx),dtype=self.dtype)

        if self.formulation == 'sf':
            phi  = (self.g / self.f) * h
            phib = (self.g / self.f) * hb
            q = q.at[1:-1, 1:-1].set(
                ((phi[2:, 1:-1] + phi[:-2, 1:-1] - 2 * phi[1:-1, 1:-1]) / self.dy ** 2 +
                 (phi[1:-1, 2:] + phi[1:-1, :-2] - 2 * phi[1:-1, 1:-1]) / self.dx ** 2) -
                (self.f0 / c) ** 2 * phi[1:-1, 1:-1])
            q = jnp.where(jnp.isnan(q), 0, q)
            q = q.at[self.ind12].set(-(self.f0 / c) ** 2 * phib[self.ind12])
        else:
            q = q.at[1:-1, 1:-1].set(
                self.g / self.f0 * \
                ((h[2:, 1:-1] + h[:-2, 1:-1] - 2 * h[1:-1, 1:-1]) / self.dy ** 2 + \
                 (h[1:-1, 2:] + h[1:-1, :-2] - 2 * h[1:-1, 1:-1]) / self.dx ** 2) - \
                self.g * self.f0 / (c ** 2) * h[1:-1, 1:-1])
            q = jnp.where(jnp.isnan(q), 0, q)
            q = q.at[self.ind12].set(-self.g * self.f0 / (c ** 2) * hb[self.ind12])

        q = q.at[self.ind0].set(0)

        return q
    
    def pv2h(self, q, hb, qb):

        """ PV to SSH 

        Args:
            q (2D array): SSH field.
            hb (2D array): Background SSH field
            qb (2D array): Background PV field

        Returns:
            h: SSH field
        """

        # Interior pv
        qin = q[1:-1,1:-1] - qb[1:-1,1:-1]

        if self.formulation == 'sf':
            # Inversion gives streamfunction phi; convert back to SSH
            phib = (self.g / self.f) * hb
            phi = jnp.zeros_like(q, dtype=self.dtype)
            inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
            phi = phi.at[1:-1, 1:-1].set(inv)
            phi += phib
            h = (self.f / self.g) * phi
        else:
            # Inversion gives SSH directly
            h = jnp.zeros_like(q, dtype=self.dtype)
            inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
            h = h.at[1:-1, 1:-1].set(inv)
            h += hb

        return h

    def rhs(self,u,v,q,way=1):

        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment

        """

        q0 = +q


        #######################
        # Upwind current
        #######################
        u_on_T = way*0.5*(u[1:-1,1:-1]+u[1:-1,2:])
        v_on_T = way*0.5*(v[1:-1,1:-1]+v[2:,1:-1])
        up = jnp.where(u_on_T < 0, 0, u_on_T)
        um = jnp.where(u_on_T > 0, 0, u_on_T)
        vp = jnp.where(v_on_T < 0, 0, v_on_T)
        vm = jnp.where(v_on_T > 0, 0, v_on_T)

        # PV advection
        rhs_q = self.adv(up, vp, um, vm, q0)

        # Bathymetry
        if self.bathymetry_PV_term is not None:
            rhs_q += self.adv(up, vp, um, vm, self.f0 * self.bathymetry_PV_term)

        # Beta plane
        if self.beta is not None:
            rhs_q = rhs_q.at[2:-2,2:-2].set(
                    rhs_q[2:-2,2:-2] - way * self.beta[1:-1,2:-2] * (v[2:-2,2:-2]+v[3:-1,2:-2])/2)
            
        # PV Diffusion
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
            
        return rhs_q

    def adv(self, up, vp, um, vm, q0):

        """
            3rd-order upwind scheme
        """

        ugradq = jnp.zeros_like(q0,dtype=self.dtype)

        ugradq = ugradq.at[2:-2,2:-2].set(
            - up[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (2 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] - 6 * q0[2:-2, 1:-3] + q0[2:-2, :-4]) \
            + um[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (q0[2:-2, 4:] - 6 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] + 2 * q0[2:-2, 1:-3]) \
            - vp[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (2 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] - 6 * q0[1:-3, 2:-2] + q0[:-4, 2:-2]) \
            + vm[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (q0[4:, 2:-2] - 6 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] + 2 * q0[1:-3, 2:-2])
            )

        return ugradq
    
    def euler(self, var0, incr, way):

        """
            Euler time scheme
        """

        return var0 + way * self.dt * incr

    def rk2(self, var0, incr, hb, qb, way):

        """
            2rd-order Runge-Kutta time scheme
        """

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

    def rk3(self, var0, incr, hb, qb, way):

        """
            3rd-order Runge-Kutta time scheme
        """
        # k1
        var1 = var0 + self.dt * incr
        
        # k2
        var12 = var0 + 0.5 * self.dt * incr
        if len(incr.shape) == 3:
            q12 = var12[0]
            c12 = var12[1:]
        else:
            q12 = +var12
        h12 = self.pv2h_jit(q12, hb, qb)
        u12, v12 = self.h2uv_jit(h12)
        u12 = jnp.where(jnp.isnan(u12), 0, u12)
        v12 = jnp.where(jnp.isnan(v12), 0, v12)
        if len(incr.shape) == 3:
            var12 = jnp.append(q12[jnp.newaxis, :, :], c12, axis=0)
        else:
            var12 = +q12
        incr12 = self.rhs_jit(u12, v12, var12, way=way)
        
        # k3
        var13 = var0 - self.dt * incr + 2 * self.dt * incr12
        if len(incr.shape) == 3:
            q13 = var13[0]
            c13 = var13[1:]
        else:
            q13 = +var13
        h13 = self.pv2h_jit(q13, hb, qb)
        u13, v13 = self.h2uv_jit(h13)
        u13 = jnp.where(jnp.isnan(u13), 0, u13)
        v13 = jnp.where(jnp.isnan(v13), 0, v13)
        if len(incr.shape) == 3:
            var13 = jnp.append(q13[jnp.newaxis, :, :], c13, axis=0)
        else:
            var13 = +q13
        incr13 = self.rhs_jit(u13, v13, var13, way=way)
        
        var_final = var0 + (self.dt / 6) * (incr + 4 * incr12 + incr13)
        
        return var_final

    
    def one_step(self, h0, q0, hb, qb, way=1):

        """
            One step forward
        """

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # Compute increment
        incr = self.rhs_jit(u,v,q0,way=way)
        
        # Time integration 
        if self.time_scheme == 'Euler':
            q1 = self.euler_jit(q0, incr, way)
        elif self.time_scheme == 'rk2':
            q1 = self.rk2_jit(q0, incr, hb, qb, way)
        elif self.time_scheme == 'rk3':
            q1 = self.rk3_jit(q0, incr, hb, qb, way)

        # Elliptical inversion 
        h1 = self.pv2h_jit(q1, hb, qb)

        return h1, q1

    def one_step_for_scan(self,X0,X):

        """
            One step forward for scan
        """

        h1, q1, hb, qb = X0
        h1, q1 = self.one_step_jit(h1, q1, hb, qb)
        X = (h1, q1, hb, qb)

        return X,X

    def step(self, h0, hb, nstep=1):

        """ Propagation

        Args:
            h0 (2D array): initial SSH
            hb (2D array): background SSH
            nstep (int): number of time-step

        Returns:
            h1 (2D array): propagated SSH

        """

        # Add MDT
        if self.mdt is not None:
            h0 += self.mdt
            hb += self.mdt

        # Compute potential voriticy
        q0 = self.h2pv_jit(h0, hb)
        qb = self.h2pv_jit(hb, hb)

        # Init
        h1 = +h0
        q1 = +q0
    
        # Time propagation
        X1, _ = scan(self.one_step_for_scan_jit, init=(h1, q1, hb, qb), xs=jnp.zeros(nstep))
        h1, q1, hb, qb = X1
    
        # Mask
        h1 = h1.at[self.ind0].set(jnp.nan)

        # Back to sla
        if self.mdt is not None:
            h1 -= self.mdt

        return h1

    def step_tgl(self, dh0, h0, hb, nstep=1):

        _, dh1 = jvp(partial(self.step_jit, hb=hb, nstep=nstep), (h0,), (dh0,))

        return dh1
    
    def step_adj(self,adh0,h0,hb,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,hb=hb,nstep=nstep), h0)
        
        return adf(adh0)[0]
    
class Qgm_trac:

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, upwind=3,
                 g=9.81, f=1e-4, time_scheme='Euler', Wbc=None, Kdiffus=None, Kdiffus_trac=None,bc_trac='OBC',
                 mdt=None,
                 ageo_velocities=False,advect_pv=True,
                 sponge_coef=0.,
                 ** kwargs):

        # Grid shape
        ny, nx, = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        if hasattr(dx, "__len__"):
            self.dx = dx[int(ny/2),int(nx/2)].astype('float64')
            self.dy = dy[int(ny/2),int(nx/2)].astype('float64') 
        else:
            self.dx = dx
            self.dy = dy

        # Time step
        self.dt = dt

        # Gravity
        self.g = g

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = np.nanmean(f).astype('float64')
            # Beta plane
            self.beta = (f[2:,:] - f[:-2,:]) / (2*self.dy)
        else:
            self.f = f.astype('float64')
            self.beta = None

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = np.nanmean(c).astype('float64')
        else:
            self.c = c.astype('float64')

        # MDT
        self.mdt = mdt

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Elliptical inversion operator
        x, y = np.meshgrid(np.arange(1, nx - 1, dtype='float64'),
                           np.arange(1, ny - 1, dtype='float64'))
        laplace_dst = 2 * (np.cos(np.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (np.cos(np.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f * laplace_dst - self.g * self.f / self.c ** 2

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

        # Tracer sponge Rayleigh-damping coefficient (per substep).
        # Applied as a tendency: var1 += sponge_coef * Wbc * ocean * (varb - var1).
        # Typical values 0.01-0.1; 0 disables the sponge entirely.
        self.sponge_coef = float(sponge_coef)

        # Ocean mask (1 in ocean, 0 on land) used to disable the tracer sponge on
        # land cells.  Mirrors SW's `_c_b = c_b * masks.h`: in SW the prognostic
        # tracer and the BC tracer are both forced to 0 over land via masks.h, so
        # the sponge term `gamma * (c_b - c)` vanishes there.  In QG the
        # prognostic tracer is NOT mask-aware and `varb` has NaN->0 fills on
        # land (see Model_qg1l_jax.set_bc), so without this gate the sponge
        # would relax coastal-interior cells toward those land zeros and
        # produce spurious freshwater/cold intrusions near the coast.
        self.ocean_h = (self.ind0 == False).astype('float64')

        # Adect PV flag
        self.advect_pv = advect_pv

        # Ageostrophic velocities
        self.ageo_velocities = ageo_velocities

        # JIT compiling functions
        self.h2uv_jit = jit(self.h2uv)
        self.h2pv_jit = jit(self.h2pv)
        self.cg_jit = jit(cg, static_argnums=(1,2,3))
        self.pv2h_jit = jit(self.pv2h)
        self.rhs_jit = jit(self.rhs)
        self.adv_jit = jit(self.adv)
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

        u = u.at[1:-1,1:].set(- self.g/self.f*\
         (h[2:,:-1]+h[2:,1:]-h[:-2,1:]-h[:-2,:-1])/(4*self.dy))
             
        v = v.at[1:,1:-1].set(self.g/self.f*\
            (h[1:,2:]+h[:-1,2:]-h[:-1,:-2]-h[1:,:-2])/(4*self.dx))
        
        u = jnp.where(jnp.isnan(u),0,u)
        v = jnp.where(jnp.isnan(v),0,v)
            
        return u,v

    def h2pv(self, h, hb, c=None):
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
            self.g / self.f * \
            ((h[2:, 1:-1] + h[:-2, 1:-1] - 2 * h[1:-1, 1:-1]) / self.dy ** 2 + \
             (h[1:-1, 2:] + h[1:-1, :-2] - 2 * h[1:-1, 1:-1]) / self.dx ** 2) - \
            self.g * self.f / (c ** 2) * h[1:-1, 1:-1])

        q = jnp.where(jnp.isnan(q),0,q)

        q = q.at[self.ind12].set(- \
            self.g * self.f / (c ** 2) * hb[self.ind12])
        
        q = q.at[self.ind0].set(0)

        return q
    
    def pv2h(self, q, hb, qb):

        # Interior pv
        qin = q[1:-1,1:-1] - qb[1:-1,1:-1]
        
        # Inverse sine tranfrom to get reconstructed SSH
        h = jnp.zeros_like(q,dtype='float64')
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        h = h.at[1:-1, 1:-1].set(inv)

        # add the boundary value
        h += hb

        return h

    def rhs(self,u,v,ua,va,var0,way=1):

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
        if self.advect_pv:
            # advection
            rhs_q = self.adv(up, vp, um, vm, q0)
            # Beta plane
            if self.beta is not None:
                rhs_q = rhs_q.at[2:-2,2:-2].set(
                        rhs_q[2:-2,2:-2] - way * self.beta[1:-1,2:-2] * (v[2:-2,2:-2]+v[3:-1,2:-2])/2)
        else:
            rhs_q = jnp.zeros_like(q0,dtype='float64')
        # PV Diffusion
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
        if c0 is not None:
            incr = incr.at[0].set(rhs_q)

            #######################
            # Ageostrophic upwind current
            #######################
            if self.ageo_velocities:
                ua_on_T = way*0.5*(ua[1:-1,1:-1]+ua[1:-1,2:])
                va_on_T = way*0.5*(va[1:-1,1:-1]+va[2:,1:-1])
                uap = jnp.where(ua_on_T < 0, 0, ua_on_T)
                uam = jnp.where(ua_on_T > 0, 0, ua_on_T)
                vap = jnp.where(va_on_T < 0, 0, va_on_T)
                vam = jnp.where(va_on_T > 0, 0, va_on_T)
                up += uap
                um += uam
                vp += vap
                vm += vam

            # Force advective velocities to zero at land/coast T-points so
            # tracers are not transported into or out of dry cells.  Without
            # this, geostrophic u,v from h2uv near the coast push the (NN-
            # filled) values around and produce spurious coastal patches.
            u_mask = self.ocean_h[1:-1,1:-1]
            up = up * u_mask
            um = um * u_mask
            vp = vp * u_mask
            vm = vm * u_mask
        
            #######################
            # Advection
            #######################
            for i in range(c0.shape[0]):
                rhs_c = jnp.zeros((self.ny,self.nx),dtype='float64')
                # Advection
                rhs_c = self.adv(up, vp, um, vm, c0[i])
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
                incr = incr.at[i+1].set(rhs_c)
        else:
            incr = rhs_q
            
        return incr
    
    def adv(self, up, vp, um, vm, q0):

        """
            3rd-order upwind scheme
        """

        ugradq = jnp.zeros_like(q0,dtype='float64')

        ugradq = ugradq.at[2:-2,2:-2].set(
            - up[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (2 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] - 6 * q0[2:-2, 1:-3] + q0[2:-2, :-4]) \
            + um[1:-1,1:-1] * 1 / (6 * self.dx) * \
            (q0[2:-2, 4:] - 6 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] + 2 * q0[2:-2, 1:-3]) \
            - vp[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (2 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] - 6 * q0[1:-3, 2:-2] + q0[:-4, 2:-2]) \
            + vm[1:-1,1:-1] * 1 / (6 * self.dy) * \
            (q0[4:, 2:-2] - 6 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] + 2 * q0[1:-3, 2:-2])
            )

        return ugradq

    def euler(self, var0, incr, way):

        return var0 + way * self.dt * incr

    def rk2(self, var0, incr, ua, va, hb, qb, way):

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
        incr12 = self.rhs_jit(u12,v12,ua,va,var12,way=way)

        var1 = var0 + self.dt * incr12

        return var1

    def rk4(self, q0, rq, ua, va, hb, qb, way):

        # k1
        k1 = rq * self.dt
        # k2
        q2 = q0 + 0.5*k1
        h2 = self.pv2h_jit(q2,hb,qb)
        u2,v2 = self.h2uv_jit(h2)
        u2 = jnp.where(jnp.isnan(u2),0,u2)
        v2 = jnp.where(jnp.isnan(v2),0,v2)
        rq2 = self.qrhs_jit(u2,v2,ua,vaq2,hb,way=way)
        k2 = rq2*self.dt
        # k3
        q3 = q0 + 0.5*k2
        h3 = self.pv2h_jit(q3,hb,qb)
        u3,v3 = self.h2uv_jit(h3)
        u3 = jnp.where(jnp.isnan(u3),0,u3)
        v3 = jnp.where(jnp.isnan(v3),0,v3)
        rq3 = self.qrhs_jit(u3,v3,ua,vaq3,hb,way=way)
        k3 = rq3*self.dt
        # k4
        q4 = q0 + k2
        h4 = self.pv2h_jit(q4,hb,qb)
        u4,v4 = self.h2uv_jit(h4)
        u4 = jnp.where(jnp.isnan(u4),0,u4)
        v4 = jnp.where(jnp.isnan(v4),0,v4)
        rq4 = self.qrhs_jit(u4,v4,ua,va,q4,hb,way=way)
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
                # Tracer Sponge (Rayleigh damping toward BC field)
                #######################
                # Per-substep tendency nudging:
                #   var1 <- var1 + sponge_coef * Wbc * (varb - var1)
                # Replaces the previous hard blend (equivalent to sponge_coef == 1),
                # which clamped the interior to the BC field within ~one timestep and
                # propagated land/edge extrapolation artefacts (e.g. coastal SSS
                # intrusions) into the domain. sponge_coef=0 disables the sponge.
                if self.sponge_coef > 0.:
                    var1 = var1.at[i,1:-1,1:-1].add(
                        self.sponge_coef * self.Wbc[1:-1,1:-1] *
                        self.ocean_h[1:-1,1:-1] *
                        (varb[i,1:-1,1:-1] - var1[i,1:-1,1:-1]))

                #######################
                # Tracer land-cell clamp
                #######################
                # Force prognostic tracer values at land cells to the (now
                # nearest-neighbour-filled) BC value. QG tracer advection is
                # not mask-aware: without this clamp, land cells would drift
                # under advection/diffusion and leak unphysical values into
                # adjacent coastal ocean cells the next step. Mirrors SW's
                # behaviour where the prognostic is kept consistent on land
                # via masks.h.
                var1 = var1.at[i, self.ind0].set(varb[i, self.ind0])
            
        return var1

    def one_step(self, h0, ua, va, var0, hb, varb, way=1):

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # Boundary field for PV
        if len(varb.shape)==3:
            qb = +varb[0]
        else:
            qb = +varb
        
        # Compute increment
        incr = self.rhs_jit(u,v,ua,va,var0,way=way)
        
        # Time integration 
        if self.time_scheme == 'Euler':
            var1 = self.euler_jit(var0, incr, way)
        elif self.time_scheme == 'rk2':
            var1 = self.rk2_jit(var0, incr, ua, va, hb, qb, way)
        elif self.time_scheme == 'rk4':
            var1 = self.rk4_jit(var0, incr, ua, va, hb, qb, way)

        # Elliptical inversion 
        if len(var1.shape)==3:
            q1 = +var1[0]
        else:
            q1 = +var1
        h1 = self.pv2h_jit(q1, hb, qb)

        # Tracer boundary conditions
        if ua is not None and va is not None:
            ubc = u + ua
            vbc = v + va
        else:
            ubc = u
            vbc = v 
        var1 = self.bc_jit(var1,var0,ubc,vbc,varb)

        return h1, var1

    def one_step_for_scan(self,X0,X):

        h1, ua, va, var1, hb, varb = X0
        h1, var1 = self.one_step_jit(h1, ua, va, var1, hb, varb)
        X = (h1, ua, va, var1, hb, varb)

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
            if self.ageo_velocities:
                ua0 = +X0[1]
                va0 = +X0[2]
                c0 = +X0[3:] # Tracer concentrations
            else:
                c0 = +X0[1:] # Tracer concentrations
            hb = +Xb[0] # Boundary field for SSH
            cb = +Xb[1:] # Boundary fields for tracers
        else:
            h0 = +X0
            c0 = None
            hb = +Xb
            cb = None

        # Tracer mask
        # NOTE: previously zeroed c0/cb at land cells (self.ind0).  That
        # undid the nearest-neighbour fill applied to tracer BCs in
        # Model_qg1l_jax.set_bc / _apply_bc, so coastal/land cells became
        # 0 K / 0 psu at the start of every step.  Stencils and tracer
        # diffusion then pulled adjacent ocean cells towards 0 — the
        # source of the spurious coastal cold/fresh patches that grew in
        # after the first timestep.  Force the prognostic at land to
        # match the (NN-filled) BC and leave the BC itself untouched.
        if c0 is not None:
            c0 = c0.at[:,self.ind0].set(cb[:,self.ind0])
        
        # Add MDT
        if self.mdt is not None:
            h0 += self.mdt
            hb += self.mdt

        # Compute potential voriticy
        q0 = self.h2pv_jit(h0, hb)
        qb = self.h2pv_jit(hb, hb)

        # Init
        h1 = +h0
        var1 = +q0
        varb = +qb
        if self.ageo_velocities:
            ua = +ua0
            va = +va0
        else:
            ua = va = None

        if c0 is not None:
            var1 = jnp.append(var1[jnp.newaxis,:,:],c0,axis=0)
            varb = jnp.append(varb[jnp.newaxis,:,:],cb,axis=0)

        # Time propagation
        for _ in range(nstep):
            h1, var1 = self.one_step_jit(h1, ua, va, var1, hb, varb)

        # Mask
        h1 = h1.at[self.ind0].set(jnp.nan)
        if len(var1.shape)==3:
            var1 = var1.at[1:,self.ind0].set(np.nan)
        
        # Back to sla
        if self.mdt is not None:
            h1 -= self.mdt

        # Concatenate
        if len(var1.shape)==3:
            if self.ageo_velocities:
                X1 = jnp.concatenate((h1[jnp.newaxis,:,:],ua0[jnp.newaxis,:,:],va0[jnp.newaxis,:,:],var1[1:]),axis=0)
            else:
                X1 = jnp.append(h1[jnp.newaxis,:,:],var1[1:],axis=0)
        else:
            X1 = h1

        return X1

    def step_tgl(self, dX0, X0, Xb, way=1, nstep=1):

        _, dX1 = jvp(partial(self.step_jit, Xb=Xb, nstep=nstep, way=way), (X0,), (dX0,))

        return dX1
    
    def step_adj(self,adX0,X0,Xb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,Xb=Xb,nstep=nstep,way=way), X0)
        
        return adf(adX0)[0]

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, upwind=3,
                 g=9.81, f=1e-4, time_scheme='Euler', Wbc=None, Kdiffus=None, Kdiffus_trac=None,bc_trac='OBC',
                 mdt=None, sponge_coef=0.):

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
            self.f0 = (np.nanmean(f) * np.ones((self.ny,self.nx))).astype('float64')
            self.f = f
        else:
            self.f0 = f
            self.f = (f * np.ones((self.ny,self.nx))).astype('float64')


        # Rossby radius
        if hasattr(c, "__len__"):
            self.c0 = (np.nanmean(c) * np.ones((self.ny,self.nx))).astype('float64')
            self.c = c
        else:
            self.c0 = c
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
        self.helmoltz_dst = self.g / self.f[1:ny-1,1:nx-1] * laplace_dst - self.g * self.f[1:ny-1,1:nx-1] / self.c[1:ny-1,1:nx-1] ** 2

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

        # Tracer sponge Rayleigh-damping coefficient (per substep).
        # Applied as a tendency: var1 += sponge_coef * Wbc * ocean * (varb - var1).
        # See Qgm_trac.__init__ for the rationale of the ocean mask.
        self.sponge_coef = float(sponge_coef)
        self.ocean_h = (self.ind0 == False).astype('float64')

        self.ageo_velocities = None
        self.advect_pv = True

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

    def rhs(self,u,v,ua,va,var0,way=1):

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
        if self.advect_pv:
            rhs_q = self._adv_jit(up, vp, um, vm, q0)
            rhs_q = rhs_q.at[2:-2,2:-2].set(
                    rhs_q[2:-2,2:-2] - way*\
                        (self.f[3:-1,2:-2]-self.f[1:-3,2:-2])/(2*self.dy)\
                            *0.5*(v[2:-2,2:-2]+v[3:-1,2:-2]))
        else:
            rhs_q = jnp.zeros_like(q0,dtype='float64')
        # PV Diffusion
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
        if c0 is not None:
            incr = incr.at[0].set(rhs_q)

            #######################
            # Ageostrophic upwind current
            #######################
            ua_on_T = way*0.5*(ua[1:-1,1:-1]+ua[1:-1,2:])
            va_on_T = way*0.5*(va[1:-1,1:-1]+va[2:,1:-1])
            uap = jnp.where(ua_on_T < 0, 0, ua_on_T)
            uam = jnp.where(ua_on_T > 0, 0, ua_on_T)
            vap = jnp.where(va_on_T < 0, 0, va_on_T)
            vam = jnp.where(va_on_T > 0, 0, va_on_T)

            # Force advective velocities to zero at land/coast T-points so
            # tracers are not transported into or out of dry cells.  See
            # Qgm_trac.rhs for rationale.
            u_mask = self.ocean_h[1:-1,1:-1]
            up = up * u_mask
            um = um * u_mask
            vp = vp * u_mask
            vm = vm * u_mask
            uap = uap * u_mask
            uam = uam * u_mask
            vap = vap * u_mask
            vam = vam * u_mask
            #######################
            # Advection
            #######################
            for i in range(c0.shape[0]):
                rhs_c = jnp.zeros((self.ny,self.nx),dtype='float64')
                # Advection
                rhs_c = self._adv_jit(up+uap, vp+vap, um+uam, vm+vam, c0[i])
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
                incr = incr.at[i+1].set(rhs_c)
        else:
            incr = rhs_q
            
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

    def rk2(self, var0, incr, ua, va, hb, qb, way):

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
        incr12 = self.rhs_jit(u12,v12,ua,va,var12,way=way)

        var1 = var0 + self.dt * incr12

        return var1

    def rk4(self, q0, rq, ua, va, hb, qb, way):

        # k1
        k1 = rq * self.dt
        # k2
        q2 = q0 + 0.5*k1
        h2 = self.pv2h_jit(q2,hb,qb)
        u2,v2 = self.h2uv_jit(h2)
        u2 = jnp.where(jnp.isnan(u2),0,u2)
        v2 = jnp.where(jnp.isnan(v2),0,v2)
        rq2 = self.qrhs_jit(u2,v2,ua,vaq2,hb,way=way)
        k2 = rq2*self.dt
        # k3
        q3 = q0 + 0.5*k2
        h3 = self.pv2h_jit(q3,hb,qb)
        u3,v3 = self.h2uv_jit(h3)
        u3 = jnp.where(jnp.isnan(u3),0,u3)
        v3 = jnp.where(jnp.isnan(v3),0,v3)
        rq3 = self.qrhs_jit(u3,v3,ua,vaq3,hb,way=way)
        k3 = rq3*self.dt
        # k4
        q4 = q0 + k2
        h4 = self.pv2h_jit(q4,hb,qb)
        u4,v4 = self.h2uv_jit(h4)
        u4 = jnp.where(jnp.isnan(u4),0,u4)
        v4 = jnp.where(jnp.isnan(v4),0,v4)
        rq4 = self.qrhs_jit(u4,v4,ua,va,q4,hb,way=way)
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
                # Tracer Sponge (Rayleigh damping toward BC field)
                #######################
                # Per-substep tendency nudging:
                #   var1 <- var1 + sponge_coef * Wbc * (varb - var1)
                # Replaces the previous hard blend (equivalent to sponge_coef == 1),
                # which clamped the interior to the BC field within ~one timestep and
                # propagated land/edge extrapolation artefacts (e.g. coastal SSS
                # intrusions) into the domain. sponge_coef=0 disables the sponge.
                if self.sponge_coef > 0.:
                    var1 = var1.at[i,1:-1,1:-1].add(
                        self.sponge_coef * self.Wbc[1:-1,1:-1] *
                        self.ocean_h[1:-1,1:-1] *
                        (varb[i,1:-1,1:-1] - var1[i,1:-1,1:-1]))

                #######################
                # Tracer land-cell clamp
                #######################
                # Force prognostic tracer values at land cells to the (now
                # nearest-neighbour-filled) BC value. QG tracer advection is
                # not mask-aware: without this clamp, land cells would drift
                # under advection/diffusion and leak unphysical values into
                # adjacent coastal ocean cells the next step. Mirrors SW's
                # behaviour where the prognostic is kept consistent on land
                # via masks.h.
                var1 = var1.at[i, self.ind0].set(varb[i, self.ind0])
            
        return var1

    def one_step(self, h0, ua, va, var0, hb, varb, way=1):

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # Boundary field for PV
        if len(varb.shape)==3:
            qb = +varb[0]
        else:
            qb = +varb
        
        # Compute increment
        incr = self.rhs_jit(u,v,ua,va,var0,way=way)
        
        # Time integration 
        if self.time_scheme == 'Euler':
            var1 = self.euler_jit(var0, incr, way)
        elif self.time_scheme == 'rk2':
            var1 = self.rk2_jit(var0, incr, ua, va, hb, qb, way)
        elif self.time_scheme == 'rk4':
            var1 = self.rk4_jit(var0, incr, ua, va, hb, qb, way)

        # Elliptical inversion 
        if len(var1.shape)==3:
            q1 = +var1[0]
        else:
            q1 = +var1
        h1 = self.pv2h_jit(q1, hb, qb)

        # Tracer boundary conditions
        var1 = self.bc_jit(var1,var0,u+ua,v+va,varb)

        return h1, var1

    def one_step_for_scan(self,X0,X):

        h1, ua, va, var1, hb, varb = X0
        h1, var1 = self.one_step_jit(h1, ua, va, var1, hb, varb)
        X = (h1, ua, va, var1, hb, varb)

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
            if self.ageo_velocities:
                ua0 = +X0[1]
                va0 = +X0[2]
                c0 = +X0[3:] # Tracer concentrations
            else:
                c0 = +X0[1:] # Tracer concentrations
            hb = +Xb[0] # Boundary field for SSH
            cb = +Xb[1:] # Boundary fields for tracers
        else:
            h0 = +X0
            c0 = None
            hb = +Xb
            cb = None

        # Tracer mask
        # See Qgm_trac.step for rationale: zeroing tracers on land
        # destroys the NN-filled BCs and produces coastal 0 patches
        # after the first step.  Force the prognostic at land to the
        # (NN-filled) BC value and leave the BC untouched.
        if c0 is not None:
            c0 = c0.at[:,self.ind0].set(cb[:,self.ind0])
        # h-->q
        q0 = self.h2pv_jit(h0, hb)
        qb = self.h2pv_jit(hb, hb)

        # Init
        h1 = +h0
        var1 = +q0
        varb = +qb
        if self.ageo_velocities:
            ua = +ua0
            va = +va0
        else:
            ua = jnp.zeros_like(h0)
            va = jnp.zeros_like(h0)
        if c0 is not None:
            var1 = jnp.append(var1[jnp.newaxis,:,:],c0,axis=0)
            varb = jnp.append(varb[jnp.newaxis,:,:],cb,axis=0)

        # Time propagation
        X1, _ = scan(self.one_step_for_scan_jit, init=(h1, ua, va, var1, hb, varb), xs=jnp.zeros(nstep))
        h1, ua, va, var1, hb, varb = X1

        # Mask
        h1 = h1.at[self.ind0].set(jnp.nan)
        if len(var1.shape)==3:
            var1 = var1.at[1:,self.ind0].set(np.nan)

        # Concatenate
        if len(var1.shape)==3:
            if self.ageo_velocities:
                X1 = jnp.concatenate((h1[jnp.newaxis,:,:],ua0[jnp.newaxis,:,:],va0[jnp.newaxis,:,:],var1[1:]),axis=0)
            else:
                X1 = jnp.append(h1[jnp.newaxis,:,:],var1[1:],axis=0)
        else:
            X1 = +h1

        return X1

    def step_tgl(self, dX0, X0, Xb, way=1, nstep=1):

        _, dX1 = jvp(partial(self.step_jit, Xb=Xb, nstep=nstep, way=way), (X0,), (dX0,))

        return dX1
    
    def step_adj(self,adX0,X0,Xb,way=1,nstep=1):
        
        _, adf = vjp(partial(self.step_jit,Xb=Xb,nstep=nstep,way=way), X0)
        
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