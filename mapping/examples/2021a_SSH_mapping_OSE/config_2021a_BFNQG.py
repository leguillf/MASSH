#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2021a_BFNQG' # name of the experiment

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime, timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = name_experiment, # name of output files

    path_save = f'outputs/{name_experiment}', # path of output files

    tmp_DA_path = f"scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2016,12,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2018,1,31,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=3),  

    saveoutput_time_step = timedelta(hours=3),  # time step at which the states are saved 

    flag_plot = 0,

    write_obs = True

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO',

    lon_min = 295.25,                                        # domain min longitude

    lon_max = 304.75,                                        # domain max longitude

    lat_min = 33.25,                                         # domain min latitude

    lat_max = 42.75,                                         # domain max latitude

    dx = 1/20.,                                            # zonal grid spatial step (in degree)

    dy = 1/20.,                                            # meridional grid spatial step (in degree)

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'

myMOD = dict(

    super = 'MOD_QG1L_NP', # 1.5-layer Qusi-Geostrophic model written in Numpy (only for BFN, use MOD_QG1L_JAX for 4Dvar). 

    name_var = {'SSH':"ssh", "PV":"pv"}, # Adding PV enables to store it at every time step for the computation of nudging term

    dtmodel = 600, # model timestep

    c0 = 2.7, # 1st baroclinic phase velocity (m/s), assumed constant all over the domain

)

#################################################################################################################################
# Boundary conditions
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

    dist_sponge = 50 # Peripherical band width (km) on which the boundary conditions are applied

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

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

    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

)

#################################################################################################################################
# INVERSION
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(
    
    super = 'INV_BFN',

    window_size = timedelta(days=7), # length of the bfn time window

    window_output = timedelta(days=3), # length of the output time window, in the middle of the bfn window. (need to be smaller than *bfn_window_size*)

    propagation_timestep = timedelta(hours=3), # propagation time step of the BFN, corresponding to the time step at which the nudging term is computed

    max_iteration = 20, # maximal number of iterations if *bfn_criterion* is not met

    criterion = 1e-3 # convergence criterion 

)


#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = None

