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

name_experiment = 'wp6_BM'
name_domain = 'wp6_16km'

#################################################################################################################################
# Global libraries     
#################################################################################################################################
# - datetime
# - timedelta
#################################################################################################################################
import os
from datetime import datetime,timedelta
    
#################################################################################################################################
# Initialization parameters
#################################################################################################################################
# - name_assim_init: name of the initialization function. Here, we use a steady state, meaning that all pixel values are set to 0.
# - file_name_init_SSH_field: name of the file used for initialization
# - path_init_SSH_field: path (directory+file) used for initialization
# - name_init_lon: name of longitude field stored in *file_name_init_SSH_field*
# - name_init_lat: name of latitude field stored in *file_name_init_SSH_field*   
#################################################################################################################################    

name_init = 'from_file'

name_init_grid = '../../data/wp6/wp6_16km_ref.nc'

name_init_lon = 'lon'

name_init_lat = 'lat'

name_init_file = 'init_state.nc'

#################################################################################################################################
# Time parameters
#################################################################################################################################
# - init_date: initial date (yyyy,mm,dd,hh) 
# - final_date: final date (yyyy,mm,dd,hh)
# - assimilation_time_step: assimilation time step (corresponding to observation update timestep)
# - savepoutput_time_step: time step plot at which the states are saved 
# - plot_time_step: time step plot at which the states are plotted (for debugging)
#################################################################################################################################
   
init_date = datetime(2010,2,1,0)

final_date = datetime(2010,3,1,0)

assimilation_time_step = timedelta(hours=1)  

saveoutput_time_step = timedelta(hours=1) 

plot_time_step = timedelta(days=1)  

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
    
name_mod_var = ["ssh","pv"]  

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

dtmodel = 300   

c = 2.7

#################################################################################################################################
# Analysis parameters
#################################################################################################################################
# - name_analysis: analysis function. Here, of course, we set BFN
# - parameters specific to BFN:
#    * bfn_window_size: length of the bfn time window
#    * bfn_window_output: length of the output time window, in the middle of the bfn window. (need to be smaller than *bfn_window_size*)
#    * bfn_propation_timestep: propagation time step of the BFN, corresponding to the time step at which the nudging term is computed
#    * bfn_window_overlap: 
#    * bfn_criterion: convergence criterion. typical value: 0.01
#    * bfn_max_iteration: maximal number of iterations if *bfn_criterion* is not met
#    * save_bfn_trajectory: save or not the back and forth iterations (for debugging)
#    * dist_scale: distance for which observations are 'spatially spread'
#    * save_obs_proj: save or not the projected observation as pickle format. Set to True to maximize the speed of the algorithm. 
#    * flag_use_boundary_conditions: set or not boundary conditions from file
#    * file_boundary_conditions: file of the boundary conditions (can be 2D or 3D. If 3D, it will be interpolated every *bfn_propation_timestep*).
#      The boundary conditions have to be prescribed on the same grid as *file_name_init_SSH_field*
#      If no file is specified, or the file does not exist, the boundary conditions are set to 0. 
#    * lenght_bc: lenght of the peripherical band for which the boundary conditions are applied
#    * name_time_bc: name of the boundary conditions time
#    * name_var_bc: name of the boundary conditions variable
#################################################################################################################################

name_analysis = 'BFN'

####################################
### Function-specific parameters ### 
#################################### 

bfn_window_size = timedelta(days=15)

bfn_window_output = timedelta(days=7)

bfn_max_iteration = 5

save_obs_proj = True

flag_use_boundary_conditions = True

file_boundary_conditions = None

lenght_bc = 5

#################################################################################################################################
# Observation parameters
#################################################################################################################################
# - satellite: list of satellite names 

satellite = ["nr"]
write_obs = True

# - For each *satellite*:
#    * kind_sat: "swathSSH" for SWOT, "nadir" for nadirs  
#    * obs_path_sat: directory where the observations are stored
#    * obs_prefixe_sat: prefixe in observation files
#    * name_obs_var_sat: name of the observed variables
#    * name_obs_lon_sat: name of the observation longitude
#    * name_obs_lat_sat: name of the observation latitude
#    * name_obs_time_sat: name of the observation time
#    * name_obs_xac_sat: name of the observation across track distance (only for swathSSH satellites).
#    * use_invobs_file_sat: use existing projected observations computed from SOSIE for instance (True or False). For BFN, always set False.
#    * swath_width_swot_sat: swath width in km (only for swathSSH satellites)
#    * gap_width_swot_sat: gap width in km (only for swathSSH satellites)
#    * nudging_params_stretching_sat: nudging parameters relative to the stretching part of the PV. 
#    * nudging_params_relvort_sat: nudging parameters relative to the relative part of the PV (for nadir, set to None).
#            * sigma: if sigma>0, the model states are spatially filtered by a gaussian kernel (sigma is the variance) before being nudged towards observations
#            * K: nominal nudging coefficient (0<K<1)
#            * Tau: half of the time window for which an obervation is assimilated. 
#################################################################################################################################
kind_nr = "fullSSH"
obs_path_nr = '../../data/wp6/'
obs_name_nr = "wp6_16km_obs_75hrs"
name_obs_var_nr = ["ssh_meso"]
name_obs_lon_nr = "lon"
name_obs_lat_nr = "lat"
name_obs_time_nr = "time_obs"
nudging_params_stretching_nr = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_nr = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}

#################################################################################################################################
# Outputs parameters
#################################################################################################################################
# - saveoutputs: save outputs flag (True or False)
# - name_exp_save: name of output files
# - path_save: path of output files
# - flag_plot: between 0 and 4. 0 for none plot, 4 for full plot
#################################################################################################################################

saveoutputs = True         

name_exp_save = name_experiment + '_' + name_domain

path_save = '../outputs/' + name_exp_save + '/'

flag_plot = 1
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
# - name_grd: name used for saving the QG grid to avoid calculating it every time.
#################################################################################################################################
        
tmp_DA_path = "../scratch/" +  name_exp_save + '/'
 
name_grd = tmp_DA_path + 'QGgrid'





   




