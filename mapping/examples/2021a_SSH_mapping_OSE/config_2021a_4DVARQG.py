#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2021a_4DVARQG_5km'

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

    init_date = datetime(2017,7,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2017,12,31,0),  # final date (yyyy,mm,dd,hh) 

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

    lon_min = 295examples/2021a_SSH_mapping_OSE/config_2021a_4DVARQG.py,                                        # domain min longitude

    lon_max = 305,                                        # domain max longitude

    lat_min = 33,                                         # domain min latitude

    lat_max = 43,                                         # domain max latitude

    dx = 5.,                                              # grid spacinng in km

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'


myMOD = dict(

    super = 'MOD_QG1L_JAX',

    name_var = {'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'rk2',

    c0 = 2.7,

    init_from_bc = True
    
)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = 'myBC' # For now, only BC_EXT is available

myBC = dict(

    super = 'BC_EXT',

    file = 'data/OSE_ssh_mapping_DUACS.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

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

    super = 'OBSOP_INTERP',

    path_save = None, # Directory where to save observational operator

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################

NAME_BASIS = 'myBASIS'

myBASIS = dict(

    super = 'BASIS_BM',

    flux = False,

    wavelet_init = False, # Estimate the initial state 

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

    Qmax = .01 , # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

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

    maxiter = 1000, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['ALG','H2G','J2G','J2N','J3','S3A']

ALG = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)

H2G = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)

J2G = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)

J2N = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)

J3 = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)

S3A = dict(

    super = 'OBS_SSH_NADIR',

    path = 'data/dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'sla_filtered'},

    add_mdt = True, 

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

)


#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSE',

    dir_output = None,

    time_min = datetime(2017,10,15,0),

    time_max = datetime(2017,12,15,0),

    name_ref = 'data/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc',

    name_ref_time = 'time',

    name_ref_lon = 'longitude',

    name_ref_lat = 'latitude',

    name_ref_var = 'sla_unfiltered',

    delta_t_ref = 0.9434,

    velocity_ref = 6.77,

    add_mdt_to_ref = True, 

    lenght_scale = 1000,

    path_mdt = '../../aux/aux_mdt_cnes_cls18_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

    name_exp_var = 'ssh',


)

