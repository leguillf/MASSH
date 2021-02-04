import numpy as np
from scipy.interpolate import griddata
from scipy import signal
import modgrid
import moddyn
import modelliptic
import pickle
import os


def qgsw(Hi=None, PVi=None, c=None, lon=None, lat=None, tint=None, dtout=None, dt=None,obsspace=None, scheme='Euler', diff=False, snu=None, name_grd=None, qgiter=1):
     
    """ QG Shallow Water model

    Args:
        Hi (2D array): Initial SSH field.
        c (same size as Hi): Rossby first baroclinic phase speed
        lon (2D array): longitudes
        lat (2D array): latitudes
        tint (scalar): Time of propagator integration in seconds. Can be positive (forward integration) or negative (backward integration)
        dtout (scalar): Time period of outputs
        dt (scalar): Propagator time step

    Returns:
        SSH: 3D array with dimensions (timesteps, height, width), SSH forecast  
    """
    way=np.sign(tint)

  ##############
  # Setups
  ##############
    if name_grd is not None:
        name_grd += '_QGSW'
        if not os.path.isfile(name_grd):    
           grd=modgrid.grid(Hi,lon,lat)
           if os.path.exists(os.path.dirname(name_grd)) is False:
                os.makedirs(os.path.dirname(name_grd))
           with open(name_grd, 'wb') as grd_file:
               pickle.dump(grd, grd_file)
               grd_file.close()
        else:
           with open(name_grd, 'rb') as grd_file:
               grd = pickle.load(grd_file)
               grd_file.close()
    else:
        grd = modgrid.grid(Hi,lon,lat)

    time_abs=0.
    index_time=0  
    if obsspace is not None:
        hg=np.empty((np.shape(obsspace)[0]))
        hg[:]=np.NAN
        iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
        if np.size(iobs)>0:
            hg[iobs]=griddata((lon.ravel(), lat.ravel()), Hi.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))
    else:
        hg=None 
    
    nindex_time=np.int(abs(tint)/dtout + 1) 
    grdnx=np.int(grd.nx)
    grdny=np.int(grd.ny)
    SSH=np.empty((nindex_time,grdny,grdnx)) 
    SSH[index_time,:,:]=Hi  
    PV=np.empty((nindex_time,grdny,grdnx)) 
    PV[index_time,:,:]=PVi  

    nstep=int(abs(tint)/dt)
    stepout=int(dtout/dt)
    
  ############################
  # Active variable initializations
  ############################
    h=+Hi    
    if PVi is not None:
        q = PVi
    else:
        q,=modelliptic.h2pv(h,grd,c)

    hb = +h # just for hguess   
            

  ############################
  # Time loop
  ############################

    for step in range(nstep): 
        time_abs=(step+1)*dt
        if (np.mod(step+1,stepout)==0):
            index_time += 1

        ############################
        #Initialization of previous fields
        ############################

        hguess=2*h-hb
                
        hb=+h
        qb=+q    

        ########################
        # Main routines
        ########################

        # 1/ 
        if diff:
            u = np.zeros((grd.ny,grd.nx))
            v = np.zeros((grd.ny,grd.nx))
        else:
            u,v, = moddyn.h2uv(h,grd)
        
        # 2/ Advection
        rq, = moddyn.qrhs(u,v,qb,grd,way,diff,snu)
        
        # 3/ Time integration 
        if scheme=='Euler':
            q = qb + dt*rq
            
        elif scheme=='Runge-Kutta':
            # k1
            k1 = rq*dt
            # k2
            q2 = qb + 0.5*k1
            h2, = modelliptic.pv2h(q2,hguess,grd,qgiter)
            u2,v2, = moddyn.h2uv(h2,grd)
            rq2, = moddyn.qrhs(u2,v2,q2,grd,way)
            k2 = rq2*dt
            # k3
            q3 = qb + 0.5*k2
            h3, = modelliptic.pv2h(q3,hguess,grd,qgiter)
            u3,v3, = moddyn.h2uv(h3,grd)
            rq3, = moddyn.qrhs(u3,v3,q3,grd,way)
            k3 = rq3*dt
            # k4
            q4 = qb + k2
            h4, = modelliptic.pv2h(q4,hguess,grd,qgiter)
            u4,v4, = moddyn.h2uv(h4,grd)
            rq4, = moddyn.qrhs(u4,v4,q4,grd,way)
            k4 = rq4*dt
            # q increment
            q = qb + (k1+2*k2+2*k3+k4)/6.
            
        # 4/
        h,=modelliptic.pv2h(q,hguess,grd,c,qgiter)


        ############################
        #Saving outputs
        ############################

        if (np.mod(step+1,stepout)==0): 
            SSH[index_time,:,:]=h
            PV[index_time,:,:]=q

        if obsspace is not None:
            iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
            if np.size(iobs)>0:
                hg[iobs]=griddata((lon.ravel(), lat.ravel()), h.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))
      
    ############################
    #Returning variables 
    ############################      
    if PVi is not None:
        return SSH,PV,hg
    else:
        return SSH, hg





def qgsw_stochuv(Hi=None, c=None, lon=None, lat=None, tint=None, dtout=None, dt=None,obsspace=None,snu=None,qgiter=1):
     
    """ QG Shallow Water model

    Args:
        Hi (2D array): Initial SSH field.
        c (same size as Hi): Rossby first baroclinic phase speed
        lon (2D array): longitudes
        lat (2D array): latitudes
        tint (scalar): Time of propagator integration in seconds. Can be positive (forward integration) or negative (backward integration)
        dtout (scalar): Time period of outputs
        dt (scalar): Propagator time step

    Returns:
        SSH: 3D array with dimensions (timesteps, height, width), SSH forecast  
    """
    way=np.sign(tint)
    
  ##############
  # Setups
  ##############
    grd=modgrid.grid(Hi,c,snu,lon,lat) 
    #plt.figure()
    #plt.pcolor(grd.mask)
    #plt.show()
    time_abs=0.
    index_time=0  
    if obsspace is not None:
        hg=np.empty((np.shape(obsspace)[0]))
        hg[:]=np.NAN
        iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
        if np.size(iobs)>0:
            hg[iobs]=griddata((lon.ravel(), lat.ravel()), h.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))
    else:
        hg=None 
    nindex_time=np.int(abs(tint)/dtout + 1) 
    grdnx=np.int(grd.nx)
    grdny=np.int(grd.ny)
    SSH=np.empty((nindex_time,grdny,grdnx)) 
    SSH[index_time,:,:]=Hi  

    nstep=int(abs(tint)/dt)
    stepout=int(dtout/dt)

  ############################
  # Passive variable initializations
  ############################
 


  ############################
  # Active variable initializations
  ############################


    h=+Hi
    
    q,=modelliptic.h2pv(h,grd)

    hb=+h # just for hguess

  ############################
  # Time loop
  ############################

    for step in range(nstep): 
        #print step
        time_abs=(step+1)*dt
        if (np.mod(step+1,stepout)==0):
            index_time += 1

        ############################
        #Initialization of previous fields
        ############################

        hguess=2*h-hb
        hb=+h
        qb=+q    

        ########################
        # Main routines
        ########################

        # 1/ 
        u,v, = moddyn.h2uv(h,grd) 
        
        # 1b/ 
        # Generate spatially correlated noise # Check http://www2.geog.ucl.ac.uk/~plewis/geogg122-2011-12/dem1.html
        sizefilter = 30
        max_noise = 0.02
        nx = np.shape(u)[0]+2*sizefilter
        ny = np.shape(u)[1]+2*sizefilter
        u_uniform = np.random.rand(nx,ny)
        v_uniform = np.random.rand(nx,ny)
        x, y = np.mgrid[-sizefilter:sizefilter+1, -sizefilter:sizefilter+1]
        gauss = np.exp(-0.333*(x**2/float(sizefilter)+y**2/float(sizefilter)))
        gauss_filter = gauss/gauss.sum()
        u_corr_noise = signal.convolve(u_uniform,gauss_filter,mode='valid')
        v_corr_noise = signal.convolve(v_uniform,gauss_filter,mode='valid')
        # rescale so it lies between -max_noise and max_noise
        u_corr_noise = ((u_corr_noise - u_corr_noise.min())/(u_corr_noise.max() - u_corr_noise.min())-0.5)*max_noise*2
        v_corr_noise = ((v_corr_noise - v_corr_noise.min())/(v_corr_noise.max() - v_corr_noise.min())-0.5)*max_noise*2
        
        # Apply noises to u and v    
        u = u + u_corr_noise  
        v = v + v_corr_noise  

        # 2/
        rq, = moddyn.qrhs(u,v,qb,grd,way)

        # 3/    
        q =qb + dt*rq

        # 4/
        h,=modelliptic.pv2h(q,hguess,grd,qgiter)


        ############################
        #Saving outputs
        ############################

        if (np.mod(step+1,stepout)==0): 
            SSH[index_time,:,:]=h

        if obsspace is not None:
            iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
            if np.size(iobs)>0:
                hg[iobs]=griddata((lon.ravel(), lat.ravel()), h.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))


    return SSH,hg
