#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2020a_4DVARQG_13'

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = name_experiment, # name of output files

    path_save = f'../outputs/2020a_4DVARQG/{name_experiment}', # path of output files

    tmp_DA_path = f"../scratch/2020a_4DVARQG/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,4,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=6),  

    saveoutput_time_step = timedelta(hours=6),  # time step at which the states are saved 

    flag_plot = 0,

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_CAR',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dx = 10.,                                              # grid spacinng in km

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'


myMOD = dict(

    super = 'MOD_QG1L_JAX',

    name_var = {'SSH':'ssh'},

    dtmodel = 600, # model timestep

    time_scheme = 'rk2',

    c0 = 2.7,
    
)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = 'myBC' # For now, only BC_EXT is available

myBC = dict(

    super = 'BC_EXT',

    file = '../../data/2020a_SSH_mapping_NATL60/2020a_SSH_mapping_NATL60_DUACS_swot_en_j1_tpn_g2.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    name_var = {'SSH':'gssh'}, # name of the boundary conditions variable

    name_mod_var = {'SSH':'ssh'},

)

#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = 'myOBSOP'

myOBSOP = dict(

    super = 'OBSOP_INTERP',

    path_save = None, # Directory where to save observational operator

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_coast = False,

    mask_borders = False,

)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################

NAME_BASIS = 'myBASIS'

myBASIS = dict(

    super = 'BASIS_BM',

    flux = False,

    save_wave_basis = False, # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

    wavelet_init = True, # Estimate the initial state 

    name_mod_var = 'ssh',

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp= 3.5, # Defines the wavelet shape

    facpsp= 1.5, # factor to fix df between wavelets

    lmin= 80, # minimal wavelength (in km)

    lmax= 970., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 10, # Largest mesoscale time of decorrelation 

    sloptdec = -.5, # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = .5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 0., # minimum time of decorrelation 

    tdecmax = 20., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = .03 , # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -2 # Slope such as Q = lambda^slope where lamda is the wavelength

)


#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',

    compute_test = False, # TLM, ADJ & GRAD tests

    gtol = 1e-3, # Gradient norm must be less than gtol before successful termination.

    maxiter = 2000, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['J1','EN','TPN','G2','SWOT']

J1 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_jason1.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

EN = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_envisat.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

TPN = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

G2 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_geosat2.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_karin_swot.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_xac = 'x_ac',

    name_var = {'SSH':'ssh_model'},

)

#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSSE',

    name_ref = '../../data/2020a_SSH_mapping_NATL60/dc_ref/NATL60-CJM165_GULFSTREAM*.nc',

    time_min = datetime(2012,10,22,0),

    time_max = datetime(2012,12,2,0),

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'sossheig',

    options_ref = {'combine':'nested', 'concat_dim':'time', 'parallel':True},

    name_exp_var = 'ssh'

)
