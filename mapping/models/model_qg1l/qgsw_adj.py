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

def qgsw_adj(Htraj=None, c=None, lon=None, lat=None, tint=None, dtout=None, dt=None,obsspace=None, sens=None, rappel=None,snu=None):

  adHf=Htraj[0,:,:]*0.
 
  way=np.sign(tint)

  ##############
  # Setups
  ##############

  grd=modgrid.grid(adHf,c,snu,lon,lat)
  time_abs=0.
  index_time=-1 
  nindex_time=np.abs(tint)/dtout + 1

  adSSH=np.empty((nindex_time,grd.ny,grd.nx))

  nstep=int(abs(tint)/dt)
  stepout=int(dtout/dt)

  deltat=1*dt
 
  ############################
  # Active variable initializations
  ############################

  azeros=adHf*0.
  adh=+azeros
  adhb=+azeros
  adq=+azeros
  adqb=+azeros
  adrq=+azeros
  adu=+azeros
  adv=+azeros

  adqguess=+azeros

  #adqguess=- 1./grd.g*1./grd.f0*(grd.c**2) *adh 


  ############################
  # Time loop
  ############################

  Jdp=+azeros

  for step in range(nstep): 
    #print step
    time_abs=step*dt
    if (np.mod(step,stepout)==0):
      index_time += 1

    ############################
    #Tangent update on current trajectory
    ############################

    h=Htraj[-index_time-1,:,:].squeeze()
    q,=modelliptic.h2pv(h,grd)
    u,v, = moddyn.h2uv(h,grd)


    ########################
    # Adjoint forcing
    ########################

    iobs=np.where((way*obsspace[:,2]>=np.abs(tint)-time_abs-dt/2) & (way*obsspace[:,2]<np.abs(tint)-time_abs+dt/2))[0]
    if np.shape(iobs)[0]>0: 
      #print 'times: ', min(obsspace[iobs,2]), max(obsspace[iobs,2])
      Jd=sensongrid(obsspace,sens,iobs,grd)
      #print 'some forcing at step ', step
      #adq_forcing_guess=-1./grd.g*1./grd.f0*(grd.c**2) *Jd
      #adq_forcing, = modelliptic.pv2h(Jd,adq_forcing_guess,grd,nitr=10)
      #adq=adq+adq_forcing
      nitr=10
    else:
      Jd= +azeros
      nitr=10

    #adh=adh+Jd
    adh=adh+Jd
    #Jdp=Jd

    ########################
    # Main routines
    ########################
    
    # 4/
    fguess=- grd.c**2./(grd.g*grd.f0) *Jd 
    adqguess=adqguess+fguess
    adqguess[grd.mask==1]=+fguess[grd.mask==1]
    adq_tmp, = modelliptic.pv2h(adh,adqguess,grd,nitr=nitr)
    if step==0: adq_tmpb=+adq_tmp
    adqguess=+2*adq_tmp-adq_tmpb
    #adqguess=adq_tmp*0.
    adq_tmpb=+adq_tmp
    adq=adq+adq_tmp
    adh=+azeros
    ## Local adj test
    #if step==5:
    #  Madqguess=- grd.c**2/(grd.g*grd.f0) *adq
    #  Madq, = modelliptic.pv2h(adq,Madqguess,grd,nitr=50)
    #  #Madq, = modelliptic.h2pv(adq,grd)
    #  MtMadqguess= (grd.c**2/(grd.g*grd.f0))**2 *adq
    #  MtMadq, = modelliptic.pv2h(Madq,MtMadqguess,grd,nitr=50)
    #  #MtMadq, = modelliptic.h2pv(Madq,grd)
    #  ind=np.where((grd.mask>=1))
    #  print np.inner(Madq[ind],Madq[ind])
    #  print np.inner(adq[ind],MtMadq[ind])
    #  pdb.set_trace()

    # 3/
    if rappel is not None:
      adqb = adqb+(1-deltat*rappel)*adq
      adrq = adrq+deltat*adq
    else:
      adqb = adqb + adq
      adrq = adrq + deltat*adq
    adq=+azeros

    # 2/
    adu_tmp,adv_tmp,adqb_tmp, = moddyn.qrhs_adj(adrq,u,v,q,grd,way)
    adu=adu+adu_tmp
    adv=adv+adv_tmp
    adqb=adqb+adqb_tmp
    adrq=+azeros
    ## local adjoint test : Exact success
    #if step==100:
    #  #adv=adv*0.
    #  #adu=adu*0.
    ##  #u=u*0.
    ##  #v=v*0.
    #  adqb=adqb*0.
    #  Mdrq, = moddyn.qrhs_tgl(adu,adv,adqb,u,v,q,grd,way)
    #  MtMadu,MtMadv,MtMadqb, = moddyn.qrhs_adj(Mdrq,u,v,q,grd,way)
    #  ind=np.where((grd.mask>=1))
    #  print np.inner(Mdrq[ind],Mdrq[ind])
    #  print np.inner(MtMadu[ind],adu[ind])
    #  print np.inner(MtMadv[ind],adv[ind])
    #  print np.inner(MtMadqb[ind],adqb[ind])
    #  pdb.set_trace()

    # 1/
    adhb_tmp = moddyn.aduv2adh(adu,adv,grd)
    adhb=adhb+adhb_tmp
    #if step==200: pdb.set_trace()
    adu=+azeros
    adv=+azeros
    ## local adjoint test : Exact success
    #Madu,Madv, =moddyn.h2uv(adhb,grd)
    #MtMadhb =moddyn.aduv2adh(Madu,Madv,grd)
    #ind=np.where((grd.mask>=1))
    #print np.inner(Madu[ind],Madu[ind])
    #print np.inner(Madv[ind],Madv[ind])
    #print np.inner(adhb[ind],MtMadhb[ind])
    #pdb.set_trace()

    ############################
    #Saving outputs
    ############################
    adhout,=modelliptic.h2pv(adqb,grd)
    adhout[grd.mask==0]=np.nan
    if (np.mod(step,stepout)==0):
      adSSH[index_time,:,:]=+adhout

    ############################
    # Update previous fields
    ############################

    adq=adq+adqb
    adh=adh+adhb 
    adqb=+azeros
    adhb=+azeros   

    #if step==100: pdb.set_trace()


  ############################
  #Saving final outputs
  ############################
  index_time += 1
  time_abs=nstep*dt
  adhout,=modelliptic.h2pv(adq,grd)
  adhout[grd.mask==0]=np.nan
  iobs=np.where((way*obsspace[:,2]>=np.abs(tint)-time_abs-dt/2) & (way*obsspace[:,2]<np.abs(tint)-time_abs+dt/2))[0]
  if np.shape(iobs)[0]>0:
    #print 'times: ', min(obsspace[iobs,2]), max(obsspace[iobs,2])
    Jd=sensongrid(obsspace,sens,iobs,grd)
  else:
    Jd= +azeros
  if (np.mod(step,stepout)==0):
    adSSH[index_time,:,:]=+adhout+Jd 
  
 
 # pdb.set_trace()
  return adSSH,



def sensongrid(obsspace,sens,iobs,grd):
  lon=grd.lon
  lat=grd.lat
  Jd=np.zeros((grd.ny,grd.nx))
  for iiobs in iobs:
    dist=obsspace[iiobs,0]-lon[0,:]
    j1=np.where((dist>=0))[0][-1]
    if dist[j1]==0.: cj1=1 
    else: cj1=1-dist[j1]/(lon[0,j1+1]-lon[0,j1] )  # For regular grid only
    dist=obsspace[iiobs,1]-lat[:,0]
    i1=np.where((dist>=0))[0][-1]
    if dist[i1]==0.: ci1=1
    else: ci1=1-dist[i1]/(lat[i1+1,0]-lat[i1,0] )  # For regular grid only
    Jd[i1,j1]=Jd[i1,j1]+sens[iiobs]*ci1*cj1
    if j1+1<grd.nx: Jd[i1,j1+1]=Jd[i1,j1+1]+sens[iiobs]*ci1*(1-cj1)
    if i1+1<grd.ny: Jd[i1+1,j1]=Jd[i1+1,j1]+sens[iiobs]*(1-ci1)*cj1
    if ((j1+1<grd.nx)&(i1+1<grd.ny)): Jd[i1+1,j1+1]=Jd[i1+1,j1+1]+sens[iiobs]*(1-ci1)*(1-cj1)
  #Jd[grd.mask==1]=0.
  #pdb.set_trace()
  return Jd
