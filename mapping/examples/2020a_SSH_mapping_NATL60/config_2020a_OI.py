#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2020a_OI' # name of the experiment

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

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,15,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=3),  

    saveoutput_time_step = timedelta(days=1),  # time step at which the states are saved 

    flag_plot = 0,

    write_obs = True

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

    dx = 1/4.,                                            # zonal grid spatial step (in degree)

    dy = 1/4.,                                            # meridional grid spatial step (in degree)

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


#################################################################################################################################
# INVERSION
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_OI',

    name_var = {'SSH':'ssh'},
    
    Lt = 7, # days

    Lx = 1, # degreee

    Ly = 1, # degree

    sigma_R = 5e-2 # meters

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
