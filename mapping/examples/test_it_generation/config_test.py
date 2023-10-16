#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: vbellemin
"""

name_experiment = 'config_test'

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

    init_date = datetime(2012,6,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,6,2,0),  # final date (yyyy,mm,dd,hh) 

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

    lon_min = 0.,                                         # domain min longitude

    lon_max = 11.,                                         # domain max longitude

    lat_min = 0.,                                          # domain min latitude

    lat_max = 11.,                                          # domain max latitude

    dlon = 1,                                           # zonal grid spatial step (in degree)

    dlat = 1,

    name_init_mask = "./mask/mask_test.nc",

    name_var_mask = {'lon':'longitude','lat':'latitude','var':'mask'}

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD1'

myMOD1 = dict(

    super = 'MOD_SW1L_JAX',

    name_var = {'U':'u', 'V':'v', 'SSH':'ssh'},

    name_params = ['itg','He'],

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_island = "dirichlet",

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

    super = 'OBSOP_INTERP_L4',

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    mask_borders = False,

    interp_method = 'linear' # either 'nearest', 'linear', 'cubic' (use only 'cubic' when data is full of non-NaN)

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

    prec = True, # preconditioning,
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['IT']

IT = dict(

    super = "OBS_L4",

    path = '/bettik/bellemva/MITgcm/MITgcm_it/by_mode/*', # path of observation netcdf file(s)

    name_time = 'time', # name of time coordinate
    
    name_lon = 'longitude', # name of longitude coordinate

    name_lat = 'latitude', # name of latitude coordinate
    
    name_var = {"SSH":"ssh_it1"}, # dictionnary of observed variables (keys: variable types [SSH,SST etc...]; values: name of observed variables)

    name_err = {}, # dictionnary of measurement error variables (keys: variable types [SSH,SST etc...]; values: name of error variables)

    subsampling = None, # Subsampling in time (in number of model time step). Set to None for no subsampling

    sigma_noise = None  # Value of (constant) measurement error (will be used if *name_err* is not provided)

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
