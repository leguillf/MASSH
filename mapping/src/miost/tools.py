import sys,os,shutil
import numpy
from numpy import pi, zeros
from scipy import interpolate
import os.path
import scipy.io
import netCDF4 as nc
import pdb
import matplotlib.pylab as plt
import time
import shutil
from scipy.fftpack import ifft, ifft2, fft, fft2

#import pytide 
import datetime



def PSD1toPSD2R(ff,PSD):
  nf=len(ff)
  df=ff[1]-ff[0]
  ffx=ff
  ffy=ff
  ffxx,ffyy=numpy.meshgrid(ffx,ffy)
  ffrr=numpy.sqrt(ffxx**2+ffyy**2)
  PSD2=numpy.zeros((nf,nf))
  PSD2r=numpy.zeros((nf))


  for iff in numpy.arange(nf-2,-1,-1):
    f1=ff[iff+1]
    f2=ff[iff]
    iy=numpy.where((ffrr[iff,:]<f1)&(ffrr[iff,:]>=f2))
    indxy=numpy.where((ffrr<f1)&(ffrr>=f2))
    PSD2r[iff] = ( PSD[iff]/df - numpy.sum(PSD2[iff,:]) - numpy.sum(ffrr[iff,iy]-ff[iff])*PSD2r[iff+1]/(ff[iff+1]-ff[iff]) ) / (numpy.sum(-ffrr[iff,iy]+ff[iff+1])/(ff[iff+1]-ff[iff]))
    vec=PSD2r[iff+1]*(ffrr[indxy]-f2)/(f1-f2) + PSD2r[iff]*(-ffrr[indxy]+f1)/(f1-f2)
    vec[numpy.where((vec<0.))]=0.
    PSD2[indxy]=vec

  return PSD2r



def mywindow(x): #xloc must be between -1 and 1
     y=numpy.cos(x*0.5*numpy.pi)**2
     return y



def mywindow3(x): #xloc must be between -1 and 1
    y = zeros(x.shape)
    y[(x >= 0) & (x <= 0.5)] = 1 - 4 * x[(x >= 0) & (x <= 0.5)] ** 3
    y[(-x >= 0) & (-x <= 0.5)] = 1 + 4 * x[(-x >= 0) & (-x <= 0.5)] ** 3
    y[x >= 0.5] = -4 * (x[x >= 0.5] - 1) ** 3
    y[-x >= 0.5] = 4 * (x[-x >= 0.5] + 1) ** 3
    y[abs(x) >= 1] = 0.
    y = y ** 0.5
    return y




