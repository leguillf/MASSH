#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2022a_4DVARSW'

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
from math import pi
 
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

    final_date = datetime(2012,2,10,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  

    saveoutput_time_step = timedelta(hours=1),  # time step at which the states are saved 

    flag_plot = 0,

    write_obs = True, # the observation files are very low to process, so we decide to save the extracted informations in *path_obs* to read it for several experiments

    path_obs = f'../obs/2022a_4DVARQG'

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO',

    lon_min = 230.,                                         # domain min longitude

    lon_max = 239.5,                                         # domain max longitude

    lat_min = 30.,                                          # domain min latitude

    lat_max = 39.5,                                          # domain max latitude

    dlon = 1/4,                                           # zonal grid spatial step (in degree)

    dlat = 1/4,

    name_init_mask = '../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mask = {'lon':'longitude','lat':'latitude','var':'mdt'}

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD1'

myMOD1 = dict(

    super = 'MOD_SW1L_NP',

    name_var = {'U':'u', 'V':'v', 'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*pi/12.14/3600], # wave frequencies (in seconds)

    He_init = 0.7, # Mean height (in m)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)


myMOD2 = dict(

    super = 'MOD_SW1L_JAX',

    name_var = {'U':'u', 'V':'v', 'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*pi/12.14/3600], # igw frequencies (in seconds)

    He_init = 0.7, # Mean height (in m)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)



#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = None


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

    super = 'BASIS_IT',

    Nwaves = 1, # number of wave component 

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    sigma_B_He = 0.2, # Background variance for He

    sigma_B_bc = 1e-2, # Background variance for bc

    facgauss = 3.5,  # factor for gaussian spacing= both space/time

    D_He = 200, # Space scale of gaussian decomposition for He (in km)

    T_He = 20, # Time scale of gaussian decomposition for He (in days)

    D_bc = 200, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

)




#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',

    compute_test = True, # TLM, ADJ & GRAD tests

    gtol = 1e-3, # Gradient norm must be less than gtol before successful termination.

    maxiter = 10, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=24), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning,
 
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
