#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2020a_ForwardQG_JAX'

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

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,10,10,0),  # final date (yyyy,mm,dd,hh) 

    saveoutput_time_step = timedelta(hours=12),  # time step at which the states are saved 

    flag_plot = 4,

)

#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID' 

myGRID = dict(

    super = 'GRID_FROM_FILE',

    path_init_grid = '../../data/2020a_SSH_mapping_NATL60/dc_ref/NATL60-CJM165_GULFSTREAM_y2012m10d01.1h_SSH.nc', 

    name_init_lon = 'lon',

    name_init_lat = 'lat',

    subsampling = 6

)
   

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'

myMOD = dict(

    super = 'MOD_QG1L_JAX',

    name_var = {'SSH':'ssh'},

    name_init_var = {'SSH':'sossheig'},

    dtmodel = 900, # model timestep

    time_scheme = 'Euler',

    c0 = 2.7,

)

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
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSSE',

    name_ref = '../../data/2020a_SSH_mapping_NATL60/dc_ref/NATL60-CJM165_GULFSTREAM*.nc',

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'sossheig',

    options_ref = {'combine':'nested', 'concat_dim':'time', 'parallel':True},

    name_exp_var = 'ssh'

)
 





   




