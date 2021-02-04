import numpy as numpy
from math import pi

class grid():

  def __init__(self,SSH,lon,lat):

    ny,nx,=numpy.shape(SSH)

    mask=numpy.zeros((ny,nx))+2
    mask[:2,:]=1
    mask[:,:2]=1
    mask[-3:,:]=1
    mask[:,-3:]=1
    dx=numpy.zeros((ny,nx))
    dy=numpy.zeros((ny,nx))
    #f0=numpy.zeros((ny,nx))

    dlon = numpy.gradient(lon)
    dlat = numpy.gradient(lat)
    dx = numpy.sqrt((dlon[1]*111000*numpy.cos(numpy.deg2rad(lat)))**2
                 + (dlat[1]*111000)**2)
    dy = numpy.sqrt((dlon[0]*111000*numpy.cos(numpy.deg2rad(lat)))**2
                 + (dlat[0]*111000)**2)
    
    dx[0,:]=dx[1,:]
    dx[-1,:]=dx[-2,:] 
    dx[:,0]=dx[:,1]
    dx[:,-1]=dx[:,-2]
    dy[0,:]=dy[1,:]
    dy[-1,:]=dy[-2,:] 
    dy[:,0]=dy[:,1]
    dy[:,-1]=dy[:,-2]
    mask[numpy.where((numpy.isnan(SSH)))]=0

    f0=2*2*pi/86164*numpy.sin(lat*pi/180) 
    indNan = numpy.argwhere(numpy.isnan(SSH))
    for i,j in indNan:
        for p1 in range(-2,3):
            for p2 in range(-2,3):
              itest=i+p1
              jtest=j+p2
              if ((itest>=0) & (itest<=ny-1) & (jtest>=0) & (jtest<=nx-1)):
                mask[itest,jtest] = 1


    

    #myplt.pcolor(mask)
    #stop

    np=numpy.shape(numpy.where(mask>=1))[1]
    np2=numpy.shape(numpy.where(mask==2))[1]
    np1=numpy.shape(numpy.where(mask==1))[1]
    self.mask1d=numpy.zeros((np))
    self.H=numpy.zeros((np))
    self.c1d=numpy.zeros((np))
    self.f01d=numpy.zeros((np))
    self.dx1d=numpy.zeros((np))
    self.dy1d=numpy.zeros((np))
    self.indi=numpy.zeros((np), dtype=numpy.int)
    self.indj=numpy.zeros((np), dtype=numpy.int)
    self.vp1=numpy.zeros((np1), dtype=numpy.int)
    self.vp2=numpy.zeros((np2), dtype=numpy.int)
    self.vp2=numpy.zeros((np2), dtype=numpy.int)
    self.vp2n=numpy.zeros((np2), dtype=numpy.int)
    self.vp2nn=numpy.zeros((np2), dtype=numpy.int)
    self.vp2s=numpy.zeros((np2), dtype=numpy.int)
    self.vp2ss=numpy.zeros((np2), dtype=numpy.int)
    self.vp2e=numpy.zeros((np2), dtype=numpy.int)
    self.vp2ee=numpy.zeros((np2), dtype=numpy.int)
    self.vp2w=numpy.zeros((np2), dtype=numpy.int)
    self.vp2ww=numpy.zeros((np2), dtype=numpy.int)
    self.vp2nw=numpy.zeros((np2), dtype=numpy.int)
    self.vp2ne=numpy.zeros((np2), dtype=numpy.int)
    self.vp2se=numpy.zeros((np2), dtype=numpy.int)
    self.vp2sw=numpy.zeros((np2), dtype=numpy.int)
    self.indp=numpy.zeros((ny,nx), dtype=numpy.int) 

    p=-1
    for i in range(ny):
      for j in range(nx):
        if (mask[i,j]>=1):
          p=p+1
          self.mask1d[p]=mask[i,j]
          self.H[p]=SSH[i,j]
          self.dx1d[p]=dx[i,j]
          self.dy1d[p]=dy[i,j]
          self.f01d[p]=f0[i,j]
          self.indi[p]=i
          self.indj[p]=j
          self.indp[i,j]=p
 
 
    p2=-1
    p1=-1
    for p in range(np):
      if (self.mask1d[p]==2):
        p2=p2+1
        i=self.indi[p]
        j=self.indj[p]
        self.vp2[p2]=p
        self.vp2n[p2]=self.indp[i+1,j]
        self.vp2nn[p2]=self.indp[i+2,j]
        self.vp2s[p2]=self.indp[i-1,j]
        self.vp2ss[p2]=self.indp[i-2,j]
        self.vp2e[p2]=self.indp[i,j+1]
        self.vp2ee[p2]=self.indp[i,j+2]
        self.vp2w[p2]=self.indp[i,j-1]
        self.vp2ww[p2]=self.indp[i,j-2]
        self.vp2nw[p2]=self.indp[i+1,j-1]
        self.vp2ne[p2]=self.indp[i+1,j+1]
        self.vp2se[p2]=self.indp[i-1,j+1]
        self.vp2sw[p2]=self.indp[i-1,j-1]
      if (self.mask1d[p]==1):
        p1=p1+1
        i=self.indi[p]
        j=self.indj[p]
        self.vp1[p1]=p
    self.mask=mask
    self.f0=f0
    self.dx=dx
    self.dy=dy
    self.np=np
    self.np2=np2
    self.nx=nx
    self.ny=ny 
    self.lon=lon
    self.lat=lat
    self.g=9.81

    return None




