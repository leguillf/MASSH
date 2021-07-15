# -*- coding: utf-8 -*-
"""
"""
import logging
import numpy, scipy
from netCDF4 import Dataset
import pdb

def read_auxdata_geos(file_aux):

    # Read spectrum database
    fcid = Dataset(file_aux, 'r')
    ff1 = numpy.array(fcid.variables['f'][:])
    lon = numpy.array(fcid.variables['lon'][:])
    lat = numpy.array(fcid.variables['lat'][:])
    NOISEFLOOR = numpy.array(fcid.variables['NOISEFLOOR'][:,:])
    PSDS = numpy.array(fcid.variables['PSDS'][:,:,:])
    tdec = numpy.array(fcid.variables['tdec'][:,:,:])

    finterpPSDS = scipy.interpolate.RegularGridInterpolator((ff1,lat,lon),PSDS,bounds_error=False,fill_value=None)
    finterpTDEC = scipy.interpolate.RegularGridInterpolator((ff1,lat,lon),tdec,bounds_error=False,fill_value=None)
    #finterpTDEC = []
    finterpNOISEFLOOR = scipy.interpolate.RegularGridInterpolator((lat,lon),NOISEFLOOR,bounds_error=False,fill_value=None)

    return finterpPSDS,finterpTDEC,  finterpNOISEFLOOR


def read_auxdata_geosc(filec_aux):

    # Read spectrum database
    fcid = Dataset(filec_aux, 'r')
    lon = numpy.array(fcid.variables['lon'][:])
    lat = numpy.array(fcid.variables['lat'][:])
    C1 = numpy.array(fcid.variables['c1'][:,:])
    finterpC = scipy.interpolate.RegularGridInterpolator((lon,lat),C1,bounds_error=False,fill_value=None)
    return finterpC

def read_auxdata_depth(filec_aux):

    # Read spectrum database
    fcid = Dataset(filec_aux, 'r')
    lon = numpy.array(fcid.variables['lon'][:])
    lat = numpy.array(fcid.variables['lat'][:])
    DEPTH=numpy.array(fcid.variables['H'][:,:])
    finterpDEPTH = scipy.interpolate.RegularGridInterpolator((lon,lat),DEPTH,bounds_error=False,fill_value=None)
    return finterpDEPTH    

def read_auxdata_varcit(file_aux):

    # Read spectrum database
    fcid = Dataset(file_aux, 'r')
    lon = numpy.array(fcid.variables['lon'][:])
    lat = numpy.array(fcid.variables['lat'][:])
    VARIANCE=numpy.array(fcid.variables['variance'][:,:])
    finterpVARIANCE = scipy.interpolate.RegularGridInterpolator((lon,lat),VARIANCE.T,bounds_error=False,fill_value=None)
    return finterpVARIANCE   








