#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2022a_4DVARQG'

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

    path_save = f'../outputs/{name_experiment}', # path of output files

    tmp_DA_path = f"../scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2012,2,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,3,1,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  

    saveoutput_time_step = timedelta(hours=12),  # time step at which the states are saved 

    flag_plot = 1,

    write_obs = True, # the observation files are very low to process, so we decide to save the extracted informations in *path_obs* to read it for several experiments

    path_obs = f'../obs/{name_experiment}'

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_CAR',

    lon_min = 230.,                                         # domain min longitude

    lon_max = 239.5,                                         # domain max longitude

    lat_min = 30.,                                          # domain min latitude

    lat_max = 39.5,                                          # domain max latitude

    dx = 25.,                                           # zonal grid spatial step (in degree)

    name_init_mask = '../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mask = {'lon':'longitude','lat':'latitude','var':'mdt'}

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD2'

myMOD1 = dict(

    super = 'MOD_QG1L_JAX',

    name_var = {'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'Euler',

    c0 = 2.7,
    
)

myMOD2 = dict(

    super = 'MOD_SW1L',

    name_var = {'SSH':"ssh"},

    name_init_var = {},

    dtmodel = 3600, # model timestep

    Kdiffus = 0 # coefficient of diffusion. Set to 0 for Identity model

)


#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = 'myBC'

myBC = dict(

    super = 'BC_EXT',

    file = '../../data/2022a_mapping_HFdynamic/mdt.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = None,

    name_var = {'SSH':'ssh'}, # name of the boundary conditions variable

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

NAME_BASIS = 'myBASIS1'

myBASIS1 = dict(

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

    sloptdec = -1., # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = .5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 0., # minimum time of decorrelation 

    tdecmax = 20., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = .03 , # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -2 # Slope such as Q = lambda^slope where lamda is the wavelength

)

myBASIS2 = dict(

    super = 'BASIS_BMaux',

    flux = True,

    save_wave_basis = False, # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

    wavelet_init = True, # Estimate the initial state 

    name_mod_var = 'ssh', # Name of the related model variable (only useful if wavelet_init==True)

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp= 3.5, # Defines the wavelet shape

    facpsp= 1.5, # factor to fix df between wavelets

    lmin= 80, # minimal wavelength (in km)

    lmax= 970., # maximal wavelength (in km)

    factdec = 15, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 1, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    distortion_eq = 2.,

    lat_distortion_eq = 5.,

    distortion_eq_law = 2.,

    file_aux = '../aux/aux_reduced_basis_BM.nc',

    filec_aux = '../aux/aux_first_baroclinic_speed.nc',

    tssr = 0.5,

    facRo = 8.,

    Romax = 150.,

    cutRo =  1.6

)

myBASIS3 = dict(

    super = 'BASIS_LS',

    flux = True,

    wavelet_init = True,

    name_mod_var = 'ssh',

    facnls= 3., #factor for large-scale wavelet spacing
        
    facnlt= 3.,
        
    tdec_lw= 25.,
        
    std_lw= 0.04,
        
    lambda_lw= 970,

    fcor = .5

)

#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',

    compute_test = False, # TLM, ADJ & GRAD tests

    gtol = 1e-3, # Gradient norm must be less than gtol before successful termination.

    maxiter = 0, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning,

    restart_4Dvar = True
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['SWOT','ALG','C2','H2G','J2G','J2N','J3','S3A']


SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_swot/2022a_SSH_mapping_CalXover_swot.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',

    name_var = {'SSH':'ssh_model'},

)

ALG = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/alg/dt_global_alg_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

C2 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/c2/dt_global_c2_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

H2G = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/h2g/dt_global_h2g_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

J2G = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/j2g/dt_global_j2g_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

J2N = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/j2n/dt_global_j2n_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

J3 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/j3/dt_global_j3_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

S3A = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2022a_mapping_HFdynamic/dc_obs_nadirs/s3a/dt_global_s3a_phy_l3*.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSSE',

    name_ref = '../../data/2022a_mapping_HFdynamic/dc_ref_eval/2022a_SSH_mapping_CalXover_eval*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'ssh',

    options_ref = {'parallel':True},

    name_exp_var = 'ssh'

)
