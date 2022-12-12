#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = '2020a_BFNQG', # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = '2020a_BFNQG', # name of output files

    path_save = '../outputs/2020a_BFNQG', # path of output files

    tmp_DA_path = "../scratch/2020a_BFNQG", # temporary data assimilation directory path,

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,9,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=6),  

    saveoutput_time_step = timedelta(hours=6),  # time step at which the states are saved 

    flag_plot = 0,

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

    dx = 1/10.,                                            # zonal grid spatial step (in degree)

    dy = 1/10.,                                            # meridional grid spatial step (in degree)

)



#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'

myMOD = dict(

    super = 'MOD_QG1L_NP',

    name_var = {'SSH':"ssh", "PV":"pv"},

    dtmodel = 1200, # model timestep

    c0 = 2.7,

)

#########################################
###   External boundary conditions    ### 
#########################################

NAME_BC = 'myBC' # For now, only BC_EXT is available

myBC = dict(

    super = 'BC_EXT',

    file = '../../data/2020a_SSH_mapping_NATL60/2020a_SSH_mapping_NATL60_DUACS_swot_en_j1_tpn_g2.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    name_var = {'SSH':'gssh'}, # name of the boundary conditions variable

    name_mod_var = {'SSH':'ssh'},

    dist_sponge = 50 # Peripherical band width (km) on which the boundary conditions are applied

)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['J1','EN','TPN','G2']

SWOT_KARIN = dict(

    super = 'OBS_SSH_SWATH',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_karin_swot.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_xac = 'x_ac',

    name_var = {'SSH':'ssh_model'},
    
    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

    nudging_params_relvort = {'sigma':0,'K':0.05,'Tau':timedelta(hours=12)}

)

SWOT_NAD = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_nadir_swot.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},
    
    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

)

J1 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_jason1.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},
    
    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

)

EN = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_envisat.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},
    
    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

)

TPN = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},
    
    nudging_params_ssh = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},

)

G2 = dict(

    super = 'OBS_SSH_NADIR',

    path = '../../data/2020a_SSH_mapping_NATL60/dc_obs/2020a_SSH_mapping_NATL60_geosat2.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh_model'},
    
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

    max_iteration = 10, # maximal number of iterations if *bfn_criterion* is not met

)


#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSSE',

    dir_output = '../diags/2020a_BFNQG',

    time_min = datetime(2012,10,22,0),

    time_max = datetime(2012,12,2,0),

    name_ref = '../../data/2020a_SSH_mapping_NATL60/dc_ref/NATL60-CJM165_GULFSTREAM*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'sossheig',

    options_ref = {'combine':'nested', 'concat_dim':'time', 'parallel':True},

    name_exp_var = 'ssh'

)
