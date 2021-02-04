import numpy as np
from math import cos,sin,pi,isnan
from scipy.interpolate import griddata
import time
import numpy.matlib as matlib
import modgrid
import moddyn
import modelliptic
import matplotlib.pylab as plt
import pdb

def qgsw_tgl(Htraj=None, dHi=None, c=None, lon=None, lat=None, tint=None, dtout=None, dt=None,obsspace=None,rappel=None,snu=None):
 
  way=np.sign(tint)

  ##############
  # Setups
  ##############
  dHi=dHi+Htraj[0,:,:]*0.

  grd=modgrid.grid(dHi,c,snu,lon,lat)
  time_abs=0.
  index_time=0  
  if obsspace is not None:
    dhg=np.empty((np.shape(obsspace)[0]))
    dhg[:]=np.NAN
    iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
    if np.size(iobs)>0:
      dhg[iobs]=griddata((lon.ravel(), lat.ravel()), h.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))
  else:
    dhg=None

  nindex_time=np.abs(tint)/dtout + 1
  dSSH=np.empty((nindex_time,grd.ny,grd.nx))
  dSSH[index_time,:,:]=dHi  

  nstep=int(abs(tint)/dt)
  stepout=int(dtout/dt)


 
  ############################
  # Active variable initializations
  ############################

  dh=+dHi
  dq,=modelliptic.h2pv(dh,grd)

  dhb=+dh # just for hguess
 
  ############################
  # Time loop
  ############################

  for step in range(nstep): 
    #print step
    time_abs=(step+1)*dt
    if (np.mod(step+1,stepout)==0):
      index_time += 1

    ############################
    #Tangent update on current trajectory
    ############################

    h=Htraj[index_time,:,:].squeeze()
    q,=modelliptic.h2pv(h,grd)
    u,v, = moddyn.h2uv(h,grd)

    ############################
    # Initializations
    ############################

    dhguess=2*dh-dhb
    dhb=+dh
    dqb=+dq

    ########################
    # Main routines
    ########################

    # 1/
    du,dv, = moddyn.h2uv(dhb,grd)

    # 2/
    drq, = moddyn.qrhs_tgl(du,dv,dqb,u,v,q,grd,way)

    # 3/
    if rappel is not None:    
      dq = dqb + dt*(drq-rappel*(dqb))
    else:
      dq =dqb + dt*drq

    # 4/ From new q, we update h
    dh,=modelliptic.pv2h(dq,dhguess,grd)

    ############################
    #Saving outputs
    ############################

    if (np.mod(step+1,stepout)==0): 
      dSSH[index_time,:,:]=dh

    if obsspace is not None:
      iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
      if np.size(iobs)>0:
        dhg[iobs]=griddata((lon.ravel(), lat.ravel()), dh.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))

  return dSSH,dhg




