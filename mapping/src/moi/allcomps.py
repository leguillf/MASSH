# -*- coding: utf-8 -*-
"""
"""
import sys,os,shutil
import logging
import numpy
from numpy import pi, where, empty, zeros, inner, int_
from scipy import interpolate
import os.path
import scipy.io
import netCDF4 as nc
import pdb
import matplotlib.pylab as plt
import time
import shutil
from scipy.fftpack import ifft, ifft2, fft, fft2
#from .tools_cython import index_generator, compute_hg_
#from .sparse_matrix import Sparse_dask_glo, Scipy_csc_glo


# import h5py
# import glob

class Comp:
    """Partie commune de toute les ondes
    """

    __slots__ = (
        'nwave',
        'data_Qinv',
        'data_eta',
        'obs_datasets',
        'name',
        'id'
            )
    # IDENT = ''

    def __init__(self, **kwargs):
        pass





                                                                               
# Function to flag the datasets that are affected by the component
    def flag_obs_datasets(self,obs):
        self.obs_datasets=[]
        for ko in range(len(obs)):
            if obs[ko].nature in self.ens_nature:
                self.obs_datasets.append(ko)



        






