#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2023b_4DVARSW'

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

    path_save = f'outputs/{name_experiment}', # path of output files

    tmp_DA_path = f"scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2012,5,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,5,10,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=3),  

    saveoutput_time_step = timedelta(hours=3),  # time step at which the states are saved 

    flag_plot = 1,

    write_obs = True, # the observation files are very low to process, so we decide to save the extracted informations in *path_obs* to read it for several experiments

    path_obs = f'obs',

    compute_obs = False

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_CAR',

    lon_min = 230.,                                         # domain min longitude

    lon_max = 240,                                         # domain max longitude

    lat_min = 30.,                                          # domain min latitude

    lat_max = 40,                                          # domain max latitude

    dx = 10.,                                               # grid spacing in km

    name_init_mask = 'data_2023b/dc_ref_eval/2023b_SSHmapping_HF_California_eval_2012-05-01.nc',

    name_var_mask = {'lon':'longitude','lat':'latitude','var':'ssh'}

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD2'

myMOD1 = dict(

    super = 'MOD_QG1L',

    name_var = {'SSH':'ssh_bm'},

    dtmodel = 600, # model timestep

    time_scheme = 'rk2',

    filec_aux = '../aux/aux_first_baroclinic_speed.nc', # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'lon','lat':'lat','var':'c1'}, # Variable names for the phase velocity auxilliary file 

    init_from_bc = True,

    Kdiffus = 150
    
)

myMOD2 = dict(

    super = 'MOD_SW1L',

    name_var = {'U':'u_it', 'V':'v_it', 'SSH':'ssh_it'},

    dtmodel = 600, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*pi/12.14/3600], # IT frequencies (in seconds)

    He_init = 0.7, # Mean height (in m)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = None#'myBC' # For now, only BC_EXT is available

myBC = dict(

    super = 'BC_EXT',

    file = 'outputs/2023b_OI/2023b_OI*.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    name_var = {'SSH':'ssh'}, # name of the boundary conditions variable

    name_mod_var = {'SSH':'ssh'},

)


#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = 'myOBSOP'

myOBSOP = dict(

    super = 'OBSOP_INTERP_L3',

    path_save = 'H/', # Directory where to save observational operator

    write_op = True,

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_coast = False,

    mask_borders = False,

)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################

NAME_BASIS = 'myBASIS2'#['myBASIS1','myBASIS2']

myBASIS1 = dict(

    super = 'BASIS_BMaux',

    name_mod_var = 'ssh_bm',

    file_aux = '../aux/aux_reduced_basis_BM.nc', # Name of auxilliary file in which are stored the std and tdec for each locations at different wavelengths.

    lmin = 80, # minimal wavelength (in km)

    lmax = 900., # maximal wavelength (in km)

    factdec = 8., # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2., # minimum time of decorrelation 

    tdecmax = 20., # maximum time of decorrelation 

)

myBASIS2 = dict(

    super = 'BASIS_IT',

    Nwaves = 1, # number of wave component 

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    sigma_B_He = 0.2, # Background variance for He

    sigma_B_bc = 1e-3, # Background variance for bc

    facgauss = 3.5,  # factor for gaussian spacing= both space/time

    D_He = 300, # Space scale of gaussian decomposition for He (in km)

    T_He = 20, # Time scale of gaussian decomposition for He (in days)

    D_bc = 300, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

)

#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',#_PARALLEL',

    nprocs = 8,

    space_window_size_proc = None, # Space window size of one process (in °). Set to None for no split in space.

    time_window_size_proc = 40, # Time window size of one process (days). Set to None for no split in time.

    time_overlap_frac = .5, # Overlap fraction of two succesive time windows 

    restart_4Dvar = False,

    compute_test = True, # TLM, ADJ & GRAD tests

    gtol = 1e-3, # Gradient norm must be less than gtol before successful termination.

    maxiter = 200, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = True, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=3), #  timesteps separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

    prec = True, # preconditoning,
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['SWOT','ALG','C2','J3','S3A','S3B']

SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = 'data_2023b/dc_obs_swot/SSH_SWOT*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',

    name_var = {'SSH':'ssh'},

    sigma_noise = 1e-1,

)

ALG = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data_2023b/dc_obs_nadirs/alg/SSH_NADIR*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    sigma_noise = 3e-2,

)

C2 = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data_2023b/dc_obs_nadirs/c2/SSH_NADIR*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    sigma_noise = 3e-2,

)

J3 = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data_2023b/dc_obs_nadirs/j3/SSH_NADIR*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    sigma_noise = 3e-2,

)

S3A = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data_2023b/dc_obs_nadirs/s3a/SSH_NADIR*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    sigma_noise = 3e-2,

)

S3B = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data_2023b/dc_obs_nadirs/s3b/SSH_NADIR*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    sigma_noise = 3e-2,

)

#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = ['myDIAG1', 'myDIAG2']

myDIAG1 = dict(

    super = 'DIAG_OSSE',

    time_min = datetime(2012,5,15,0),

    time_max = datetime(2012,7,15,0),

    name_ref = 'data_2023b/dc_ref_eval/2023b_SSHmapping_HF_California_eval_*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'longitude',

    name_ref_lat = 'latitude',

    name_ref_var = 'ssh',

    name_exp_var = 'SSH_tot',

    compare_to_baseline = True,

    name_bas = 'outputs/2023b_OI/2023b_OI*.nc',

    name_bas_time = 'time',

    name_bas_lon = 'lon',

    name_bas_lat = 'lat',

    name_bas_var = 'ssh'

)

myDIAG2 = dict(

    super = 'DIAG_OSSE',

    time_min = datetime(2012,5,15,0),

    time_max = datetime(2012,7,15,0),

    name_ref = 'data_2023b/dc_ref_eval/2023b_SSHmapping_HF_California_eval_*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'longitude',

    name_ref_lat = 'latitude',

    name_ref_var = 'ssh_bm',

    name_exp_var = 'ssh_bm',

    compare_to_baseline = True,

    name_bas = 'outputs/2023b_OI/2023b_OI*.nc',

    name_bas_time = 'time',

    name_bas_lon = 'lon',

    name_bas_lat = 'lat',

    name_bas_var = 'ssh'

)
