import numpy
from math import cos,sin,pi,isnan
import matplotlib.pylab as plt

def h2uv(h,grd):
    """ SSH to U,V

    Args:
        h (2D array): SSH field.
        grd (Grid() object): check modgrid.py

    Returns:
        u (2D array): Zonal velocity  
        v (2D array): Meridional velocity

    """
    ny,nx,=numpy.shape(grd.mask)
    u=numpy.zeros((ny,nx))
    v=numpy.zeros((ny,nx))

    g=grd.g
    u[1:-1,1:] = - g/grd.f0[1:-1,1:]*( 0.25*(h[1:-1,1:]+h[1:-1,:-1]+h[2:,:-1]+h[2:,1:]) - 0.25*(h[1:-1,1:]+h[:-2,1:]+h[:-2,:-1]+h[1:-1,:-1]) ) / grd.dy[1:-1,1:]
    v[1:,1:-1] = + g/grd.f0[1:,1:-1]*( 0.25*(h[1:,1:-1]+h[1:,2:]+h[:-1,2:]+h[:-1,1:-1]) - 0.25*(h[1:,1:-1]+h[:-1,1:-1]+h[:-1,:-2]+h[1:,:-2]) ) / grd.dx[1:,1:-1]

    #  u[numpy.where((grd.mask<=1))]=0
    #  v[numpy.where((grd.mask<=1))]=0
    u[numpy.where((numpy.isnan(u)))]=0
    v[numpy.where((numpy.isnan(v)))]=0

    return u,v




def uv2rv(u,v,grd):
    """ U,V to relative vorticity Qr

    Args:
        u (2D array): Zonal velocity  
        v (2D array): Meridional velocity

    Returns:
        xi (2D array): Relative vorticity
    """
    
    ny,nx, = numpy.shape(grd.mask)
    
    gradX_V = numpy.zeros((ny,nx))
    gradY_U = numpy.zeros((ny,nx))
    
    xi = numpy.zeros((ny,nx))

    gradY_U[1:-1,1:-1] = 0.5*(u[2:,1:-1] - u[:-2,1:-1]) / grd.dy[1:-1,1:-1]
    gradX_V[1:-1,1:-1] = 0.5*(v[1:-1,2:] - v[1:-1,:-2]) / grd.dx[1:-1,1:-1]
    
    xi[1:-1,1:-1] = gradX_V[1:-1,1:-1] - gradY_U[1:-1,1:-1]
    
    ind=numpy.where((grd.mask==1))
    xi[ind] = 0

    return xi,





def aduv2adh(adu,adv,grd):

    adh=numpy.zeros((grd.ny,grd.nx))

    g=grd.g

    adh[1:-1,1:]=adh[1:-1,1:]  -g/grd.f0[1:-1,1:]*( 0.25*(adu[1:-1,1:]) - 0.25*(adu[1:-1,1:]) )/grd.dy[1:-1,1:] 
    adh[1:-1,:-1]=adh[1:-1,:-1] -g/grd.f0[1:-1,1:]*( 0.25*(adu[1:-1,1:]) - 0.25*(adu[1:-1,1:]) )/grd.dy[1:-1,1:]
    adh[2:,:-1]=adh[2:,:-1] -g/grd.f0[1:-1,1:]*( 0.25*(adu[1:-1,1:])  )/grd.dy[1:-1,1:]
    adh[2:,1:]=adh[2:,1:] -g/grd.f0[1:-1,1:]*( 0.25*(adu[1:-1,1:]) )/grd.dy[1:-1,1:]
    adh[:-2,1:]=adh[:-2,1:] -g/grd.f0[1:-1,1:]*( - 0.25*(adu[1:-1,1:]) )/grd.dy[1:-1,1:]
    adh[:-2,:-1]=adh[:-2,:-1] -g/grd.f0[1:-1,1:]*( - 0.25*(adu[1:-1,1:]) )/grd.dy[1:-1,1:]

    adh[1:,1:-1]=adh[1:,1:-1] + g/grd.f0[1:,1:-1]*( 0.25*(adv[1:,1:-1]) - 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]
    adh[1:,2:]=adh[1:,2:] + g/grd.f0[1:,1:-1]*( 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]
    adh[:-1,2:]=adh[:-1,2:] + g/grd.f0[1:,1:-1]*( 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]
    adh[:-1,1:-1]=adh[:-1,1:-1] + g/grd.f0[1:,1:-1]*( 0.25*(adv[1:,1:-1]) - 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]
    adh[:-1,:-2]=adh[:-1,:-2] + g/grd.f0[1:,1:-1]*(- 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]
    adh[1:,:-2]=adh[1:,:-2] + g/grd.f0[1:,1:-1]*(- 0.25*(adv[1:,1:-1]) )/grd.dx[1:,1:-1]

    adh[numpy.where((numpy.isnan(adh)))]=0

    return adh

def aduv2adh2(adu,adv,grd):
    adh=numpy.zeros((grd.ny,grd.nx))

    g=grd.g
    adh[1:-1,1:-1]=g/grd.f0[1:-1,1:-1]*( 0.25*(adu[1:-1,1:-1]+adu[1:-1,2:]+adu[2:,1:-1]+adu[2:,2:]) - 0.25*(adu[1:-1,1:-1]+adu[1:-1,2:]+adu[:-2,1:-1]+adu[:-2,2:]) )/grd.dy[1:-1,1:-1] \
    -g/grd.f0[1:-1,1:-1]*( 0.25*(adv[1:-1,1:-1]+adv[1:-1,2:]+adv[2:,1:-1]+adv[2:,2:]) - 0.25*(adv[1:-1,1:-1]+adv[2:,1:-1]+adv[1:-1,:-2]+adv[2:,:-2]) )/grd.dx[1:-1,1:-1]

    #  adh[numpy.where((grd.mask<=1))]=0
    adh[numpy.where((numpy.isnan(adh)))]=0

    return adh


def qrhs(u,v,q,grd,way,diff,snu=None):

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
    rq=numpy.zeros((grd.ny,grd.nx))
      
    if not diff:
        uplus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        uplus[numpy.where((uplus<0))]=0
        uminus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
        uminus[numpy.where((uminus>0))]=0
        vplus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        vplus[numpy.where((vplus<0))]=0
        vminus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
        vminus[numpy.where((vminus>=0))]=0
    
        rq[2:-2,2:-2] =rq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
        (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- \
         6*q[2:-2,1:-3]+q[2:-2,:-4] ) \
        + uminus*1/(6*grd.dx[2:-2,2:-2])*\
        (q[2:-2,4:]-6*q[2:-2,3:-1]+ \
         3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
        - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-  \
                                         6*q[1:-3,2:-2]+q[:-4,2:-2])  \
        + vminus*1/(6*grd.dy[2:-2,2:-2])*(q[4:,2:-2]-6*q[3:-1,2:-2]+ \
                                          3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
    
        #  rq[2:-2,2:-2] =rq[2:-2,2:-2] -way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])*(q[2:-2,3:-1]-q[2:-2,1:-3])*1/(2*grd.dx[2:-2,2:-2]) \
        #                               -way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])*(q[3:-1,2:-2]-q[1:-3,2:-2])*1/(2*grd.dy[2:-2,2:-2])
    
        rq[2:-2,2:-2]=rq[2:-2,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2])*way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])

    #diffusion
    if snu is not None:
        rq[2:-2,2:-2]=rq[2:-2,2:-2]+snu/(grd.dx[2:-2,2:-2]**2)*(q[2:-2,3:-1]+q[2:-2,1:-3]-2*q[2:-2,2:-2]) \
        +snu/(grd.dy[2:-2,2:-2]**2)*(q[3:-1,2:-2]+q[1:-3,2:-2]-2*q[2:-2,2:-2])
         

        rq[numpy.where((grd.mask<=1))]=0

    return rq,


def qrhs_tgl(du,dv,dq,u,v,q,grd,way,snu=None):

    drq=numpy.zeros((grd.ny,grd.nx))

    uplus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    duplus=way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
    duplus[numpy.where((uplus<0))]=0
    uplus[numpy.where((uplus<0))]=0
    uminus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    duminus=way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
    duminus[numpy.where((uminus>=0))]=0
    uminus[numpy.where((uminus>=0))]=0
    vplus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    dvplus=way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
    dvplus[numpy.where((vplus<0))]=0
    vplus[numpy.where((vplus<0))]=0
    vminus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    dvminus=way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
    dvminus[numpy.where((vminus>=0))]=0
    vminus[numpy.where((vminus>=0))]=0

    drq[2:-2,2:-2] =drq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (2*dq[2:-2,3:-1]+3*dq[2:-2,2:-2]- \
     6*dq[2:-2,1:-3]+dq[2:-2,:-4] ) \
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (dq[2:-2,4:]-6*dq[2:-2,3:-1]+ \
     3*dq[2:-2,2:-2]+2*dq[2:-2,1:-3]) \
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*dq[3:-1,2:-2]+3*dq[2:-2,2:-2]-  \
                                     6*dq[1:-3,2:-2]+dq[:-4,2:-2])  \
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(dq[4:,2:-2]-6*dq[3:-1,2:-2]+ \
                                      3*dq[2:-2,2:-2]+2*dq[1:-3,2:-2])

    drq[2:-2,2:-2] =drq[2:-2,2:-2] - duplus*1/(6*grd.dx[2:-2,2:-2])*\
    (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- \
     6*q[2:-2,1:-3]+q[2:-2,:-4] ) \
    + duminus*1/(6*grd.dx[2:-2,2:-2])*\
    (q[2:-2,4:]-6*q[2:-2,3:-1]+ \
     3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
    - dvplus*1/(6*grd.dy[2:-2,2:-2])*(2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-  \
                                      6*q[1:-3,2:-2]+q[:-4,2:-2])  \
    + dvminus*1/(6*grd.dy[2:-2,2:-2])*(q[4:,2:-2]-6*q[3:-1,2:-2]+ \
                                       3*q[2:-2,2:-2]+2*q[1:-3,2:-2])


    ######drq[2:-2,2:-2]=drq[2:-2,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[2:-2,2:-2])/grd.dy[2:-2,2:-2]*way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
    drq[2:-2,2:-2]=drq[2:-2,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2])*way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2]) 

    #diffusion
    if snu is not None:
        drq[2:-2,2:-2]=drq[2:-2,2:-2]+snu/(grd.dx[2:-2,2:-2]**2)*(dq[2:-2,3:-1]+dq[2:-2,1:-3]-2*dq[2:-2,2:-2]) \
        +snu/(grd.dy[2:-2,2:-2]**2)*(dq[3:-1,2:-2]+dq[1:-3,2:-2]-2*dq[2:-2,2:-2])

        drq[numpy.where((grd.mask<=1))]=0

    return drq,

def qrhs_adj(adrq,u,v,q,grd,way,snu=None):

    adrq[numpy.isnan(adrq)]=0.

    adu=numpy.zeros((grd.ny,grd.nx))
    adv=numpy.zeros((grd.ny,grd.nx))
    adq=numpy.zeros((grd.ny,grd.nx))

    uplus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    # uplus[numpy.where((uplus<0))]=0
    uminus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    # uminus[numpy.where((uminus>=0))]=0
    vplus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    # vplus[numpy.where((vplus<0))]=0
    vminus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    # vminus[numpy.where((vminus>=0))]=0

    aduplus= -adrq[2:-2,2:-2]*1/(6*grd.dx[2:-2,2:-2])*\
    (2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- \
     6*q[2:-2,1:-3]+q[2:-2,:-4] )
    aduplus[numpy.where((uplus<0))]=0

    aduminus= adrq[2:-2,2:-2]*1/(6*grd.dx[2:-2,2:-2])*\
    (q[2:-2,4:]-6*q[2:-2,3:-1]+ \
     3*q[2:-2,2:-2]+2*q[2:-2,1:-3] ) 
    aduminus[numpy.where((uminus>=0))]=0

    adu[2:-2,2:-2]=adu[2:-2,2:-2] + way*0.5*aduplus
    adu[2:-2,3:-1]=adu[2:-2,3:-1] + way*0.5*aduplus
    adu[2:-2,2:-2]=adu[2:-2,2:-2] + way*0.5*aduminus
    adu[2:-2,3:-1]=adu[2:-2,3:-1] + way*0.5*aduminus


    advplus = -adrq[2:-2,2:-2]*1/(6*grd.dy[2:-2,2:-2])*\
    (2*q[3:-1,2:-2]+3*q[2:-2,2:-2]-  \
     6*q[1:-3,2:-2]+q[:-4,2:-2])
    advplus[numpy.where((vplus<0))]=0

    advminus = adrq[2:-2,2:-2]*1/(6*grd.dy[2:-2,2:-2])*\
    (q[4:,2:-2]-6*q[3:-1,2:-2]+ \
     3*q[2:-2,2:-2]+2*q[1:-3,2:-2])
    advminus[numpy.where((vminus>0))]=0

    adv[2:-2,2:-2]=adv[2:-2,2:-2] + way*0.5*advplus
    adv[3:-1,2:-2]=adv[3:-1,2:-2] + way*0.5*advplus
    adv[2:-2,2:-2]=adv[2:-2,2:-2] + way*0.5*advminus
    adv[3:-1,2:-2]=adv[3:-1,2:-2] + way*0.5*advminus

    adv[2:-2,2:-2]=adv[2:-2,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2])*way*0.5*(adrq[2:-2,2:-2])
    adv[3:-1,2:-2]=adv[3:-1,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2])*way*0.5*(adrq[2:-2,2:-2])


    uplus[numpy.where((uplus<0))]=0
    uminus[numpy.where((uminus>=0))]=0
    vplus[numpy.where((vplus<0))]=0
    vminus[numpy.where((vminus>=0))]=0

    adq[2:-2,2:-2]=adq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (3*adrq[2:-2,2:-2])\
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (3*adrq[2:-2,2:-2])\
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(3*adrq[2:-2,2:-2])\
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(3*adrq[2:-2,2:-2])

    adq[2:-2,3:-1]=adq[2:-2,3:-1] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (2*adrq[2:-2,2:-2])\
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (-6*adrq[2:-2,2:-2])

    adq[3:-1,2:-2]=adq[3:-1,2:-2] \
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*adrq[2:-2,2:-2])\
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(-6*adrq[2:-2,2:-2])

    adq[2:-2,1:-3]=adq[2:-2,1:-3] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (-6*adrq[2:-2,2:-2])\
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (2*adrq[2:-2,2:-2])
    adq[1:-3,2:-2]=adq[1:-3,2:-2] \
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(-6*adrq[2:-2,2:-2])\
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(2*adrq[2:-2,2:-2])

    adq[2:-2,4:]=adq[2:-2,4:] \
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (adrq[2:-2,2:-2])

    adq[4:,2:-2]=adq[4:,2:-2] \
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(adrq[2:-2,2:-2])

    adq[2:-2,:-4]=adq[2:-2,:-4] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (adrq[2:-2,2:-2])

    adq[:-4,2:-2]=adq[:-4,2:-2] \
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(adrq[2:-2,2:-2])


    #diffusion
    if snu is not None:
        adq[2:-2,2:-2]=adq[2:-2,2:-2]+snu/(grd.dx[2:-2,2:-2]**2)*(adrq[2:-2,3:-1]+adrq[2:-2,1:-3]-2*adrq[2:-2,2:-2]) \
        +snu/(grd.dy[2:-2,2:-2]**2)*(adrq[3:-1,2:-2]+adrq[1:-3,2:-2]-2*adrq[2:-2,2:-2]) 
        


        adq[numpy.where((grd.mask<=1))]=0

    return adu, adv, adq,

def qrhs_adj2(adrq,u,v,q,grd,way):

    #way=-way

    adq=numpy.zeros((grd.ny,grd.nx))
    adu=numpy.zeros((grd.ny,grd.nx))
    adv=numpy.zeros((grd.ny,grd.nx))

    uplus=-way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    #duplus=way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
    #duplus[numpy.where((uplus<0))]=0
    uplus[numpy.where((uplus<0))]=0
    uminus=-way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    #duminus=way*0.5*(du[2:-2,2:-2]+du[2:-2,3:-1])
    #duminus[numpy.where((uminus>0))]=0
    uminus[numpy.where((uminus>0))]=0
    vplus=-way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    #dvplus=way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
    #dvplus[numpy.where((vplus<0))]=0
    vplus[numpy.where((vplus<0))]=0
    vminus=-way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    #dvminus=way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])
    #dvminus[numpy.where((vminus>=0))]=0
    vminus[numpy.where((vminus>=0))]=0

    adq[2:-2,2:-2] =adq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*\
    (2*adrq[2:-2,3:-1]+3*adrq[2:-2,2:-2]- \
     6*adrq[2:-2,1:-3]+adrq[2:-2,:-4] ) \
    + uminus*1/(6*grd.dx[2:-2,2:-2])*\
    (adrq[2:-2,4:]-6*adrq[2:-2,3:-1]+ \
     3*adrq[2:-2,2:-2]+2*adrq[2:-2,1:-3]) \
    - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*adrq[3:-1,2:-2]+3*adrq[2:-2,2:-2]-  \
                                     6*adrq[1:-3,2:-2]+adrq[:-4,2:-2])  \
    + vminus*1/(6*grd.dy[2:-2,2:-2])*(adrq[4:,2:-2]-6*adrq[3:-1,2:-2]+ \
                                      3*adrq[2:-2,2:-2]+2*adrq[1:-3,2:-2])

    adu[2:-2,2:-2] =adu[2:-2,2:-2] - way*0.5*(adrq[2:-2,2:-2]+adrq[2:-2,1:-3])*1./(grd.dx[2:-2,2:-2])*(q[2:-2,2:-2]-q[2:-2,1:-3])
    adv[2:-2,2:-2] =adv[2:-2,2:-2] - way*0.5*(adrq[2:-2,2:-2]+adrq[1:-3,2:-2])*(1./(grd.dy[2:-2,2:-2])*(q[2:-2,2:-2]-q[1:-3,2:-2]) \
                                                                                -(grd.f0[3:-1,2:-2]-grd.f0[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2]) )


    #drq[2:-2,2:-2]=drq[2:-2,2:-2]-(grd.f0[3:-1,2:-2]-grd.f0[2:-2,2:-2])/grd.dy[2:-2,2:-2]*way*0.5*(dv[2:-2,2:-2]+dv[3:-1,2:-2])

    adq[numpy.where((grd.mask<=1))]=0
    adu[numpy.where((grd.mask<=1))]=0
    adv[numpy.where((grd.mask<=1))]=0

    return adu,adv,adq,
