#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2020a_4DVARId'

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

    path_save = f'outputs/{name_experiment}', # path of output files

    tmp_DA_path = f"scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,15,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=3),  

    saveoutput_time_step = timedelta(hours=6),  # time step at which the states are saved 

    flag_plot = 1,

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dlon = 1/10.,                                            # zonal grid spatial step (in degree)

    dlat = 1/10.,                                            # meridional grid spatial step (in degree)

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'


myMOD = dict(

    super = 'MOD_DIFF',

    name_var = {'SSH':"ssh"},

    dtmodel = 3600, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

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

    flux = True,

    wavelet_init = False, # Estimate the initial state 

    name_mod_var = 'ssh',

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 10, # Largest mesoscale time of decorrelation 

    sloptdec = -.5, # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = 1, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 0., # minimum time of decorrelation 

    tdecmax = 15., # maximum time of decorrelation 

    facQ = 1, # factor to be multiplied to the estimated Q

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

    gtol = 1e-5, # Gradient norm must be less than gtol before successful termination.

    maxiter = 200, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['J1','EN','TPN','G2']

J1 = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dc_obs/2020a_SSH_mapping_NATL60_jason1.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

EN = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dc_obs/2020a_SSH_mapping_NATL60_envisat.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

TPN = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dc_obs/2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

G2 = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dc_obs/2020a_SSH_mapping_NATL60_geosat2.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},

)

SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = 'data/dc_obs/2020a_SSH_mapping_NATL60_karin_swot.nc',

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

    dir_output = f'diags/{name_experiment}',

    time_min = datetime(2012,10,22,0),

    time_max = datetime(2012,12,4,0),

    name_ref = 'data/dc_ref/NATL60-CJM165_GULFSTREAM*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'sossheig',

    options_ref = {'combine':'nested', 'concat_dim':'time', 'parallel':True},

    name_exp_var = 'ssh',

    compare_to_baseline = True,

    name_bas = 'data/2020a_SSH_mapping_NATL60_DUACS_en_j1_tpn_g2.nc',

    name_bas_time = 'time',

    name_bas_lon = 'lon',

    name_bas_lat = 'lat',

    name_bas_var = 'gssh'

)

