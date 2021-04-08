#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

##########################################
##########################################
##                                      ##
##      Example 4 - Parameters          ##
##                                      ##
##              SWOT DA                 ##
##                                      ## 
##            with model QG             ## 
##    and a Back and Forth Nudging      ## 
##          in OSMOSIS region           ## 
##     from 10/01/2012 to 10/10/2012    ## 
##                                      ## 
##########################################
# Settings:                              #
# - experimental parameters              #
# - global libraries                     #
# - initialization parameters            #
# - time parameters                      #
# - model parameters                     #
# - analysis parameters                  #
# - observation parameters               #
# - outputs parameters                   #
# - temporary DA parameters              #
##########################################
##########################################
 

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
# - name_experiment: name of the experiment
# - name_domain: name of the study domain 
#################################################################################################################################

name_experiment = 'Example0' 
name_domain = 'GULFSTREAM'

#################################################################################################################################
# Global libraries     
#################################################################################################################################
# - datetime
# - timedelta
#################################################################################################################################
from datetime import datetime,timedelta
    
#################################################################################################################################
# Initialization parameters
#################################################################################################################################
# - name_init: 'geo_grid' computes a spherical regular grid. You can also set 'from_file'
# - file_name_init_SSH_field: name of the file used for initialization
# - path_init_SSH_field: path (directory+file) used for initialization
# - name_init_lon: name of longitude field stored in *file_name_init_SSH_field*
# - name_init_lat: name of latitude field stored in *file_name_init_SSH_field*   
#################################################################################################################################    

name_init = 'from_file'                                # either 'geo_grid' or 'from_file'

# - parameters specific to 'geo_grid'  

name_init_grid = '../../data_Example1/dc_ref/NATL60-CJM165_GULFSTREAM_y2012m10d01.1h_SSH.nc'

name_init_lon = 'lon'

name_init_lat = 'lat'

name_init_var = 'sossheig'

#################################################################################################################################
# Time parameters
#################################################################################################################################
# - init_date: initial date (yyyy,mm,dd,hh) 
# - final_date: final date (yyyy,mm,dd,hh)
# - assimilation_time_step: assimilation time step (corresponding to observation update timestep)
# - savepoutput_time_step: time step plot at which the states are saved 
# - plot_time_step: time step plot at which the states are plotted (for debugging)
#################################################################################################################################
   
init_date = datetime(2012,10,1,0)     

final_date = datetime(2012,10,2,0)

saveoutput_time_step = timedelta(hours=1) 

#################################################################################################################################
# Model parameters
#################################################################################################################################
# - name_model: model function. Here we use the QGSW_Nudging function, allowing to nudge the 1.5 layer quasi-geostrophic (QG) model.
# - n_mod_var: number of physical variables in the state vector 
# - name_mod_var: name of these variables. For the BFN along with the QG model, two variables are needed: Sea Surface Height (SSH) 
#   and Potential Vorticity (PV)
# - name_mod_lon: name of the model longitude
# - name_mod_lat: name of the model latitude
# Both name_mod_lon and name_mod_lat are used in the output files.
#################################################################################################################################
       
name_model = 'QG1L'           
    
name_mod_var = ["ssh"]  

n_mod_var = len(name_mod_var)             

name_mod_lon = "nav_lon"

name_mod_lat = "nav_lat"

####################################
### Function-specific parameters ### 
#################################### 
# - parameters specific to QG model
#    * qgiter: number of iterations to perform the gradient conjugate algorithm (to inverse SSH from PV)
#    * c: first baroclinic gravity-wave phase speed (in m/s) related to Rossby Radius of deformation
#    * dtmodel: timestep of the model (in seconds). Typical values: between 200s and 1000s. If the model crashes, reduce its value.
#################################################################################################################################

qgiter = 20

c = 2.7

dtmodel = 60   

#################################################################################################################################
# Analysis parameters
#################################################################################################################################

name_analysis = None


satellite = None

#################################################################################################################################
# Outputs parameters
#################################################################################################################################
# - saveoutputs: save outputs flag (True or False)
# - name_exp_save: name of output files
# - path_save: path of output files
# - flag_plot: between 0 and 4. 0 for none plot, 4 for full plot
#################################################################################################################################

saveoutputs = True         

name_exp_save = name_experiment 

path_save = '../outputs/' + name_exp_save + '/'

flag_plot = 0
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
# - name_grd: name used for saving the QG grid to avoid calculating it every time.
#################################################################################################################################
        
tmp_DA_path = "../scratch/" +  name_exp_save + '/'
 
name_grd = tmp_DA_path + 'QGgrid'





   




