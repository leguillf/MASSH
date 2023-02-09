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

    name_experiment = '2022a_ForwardQG', # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = '2022a_ForwardQG', # name of output files

    path_save = '../outputs/2022a_ForwardQG', # path of output files

    tmp_DA_path = "../scratch/2022a_ForwardQG", # temporary data assimilation directory path,

    init_date = datetime(2012,2,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,2,10,0),  # final date (yyyy,mm,dd,hh) 

    saveoutput_time_step = timedelta(days=1),  # time step at which the states are saved 

    flag_plot = 4,

)

#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID' 

myGRID = dict(

    super = 'GRID_FROM_FILE',

    path_init_grid = '../../data/2022a_mapping_HFdynamic/dc_ref_eval/2022a_SSH_mapping_CalXover_eval_2012-02-01.nc', 

    name_init_lon = 'lon',

    name_init_lat = 'lat',

    subsampling = None

)
   

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'

myMOD = dict(

    super = 'MOD_QG1L_JAX',

    use_jax = True,

    name_var = {'SSH':'ssh'},

    name_init_var = {'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'Euler',

    c0 = 2.7,

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
 





   




