#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

##########################################
##########################################
##                                      ##
##      Example 2 - Parameters          ##
##                                      ##
##               MASSH                  ##
##                                      ## 
##            with model SW             ##
##         and a 4D variational         ##
##       in a idealized region          ##
##                                      ## 
##########################################
##########################################
 

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
# - name_experiment: name of the experiment
# - name_domain: name of the study domain 
#################################################################################################################################

name_experiment = 'Example2'

#################################################################################################################################
# Global libraries     
#################################################################################################################################
# - datetime
# - timedelta
#################################################################################################################################
from datetime import datetime,timedelta
from math import pi

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

name_init_grid = '../../data_Example2/data_BM-IT_idealized/ref.nc'

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
   
init_date = datetime(2010,5,1,0)

final_date = datetime(2010,5,15,0)

assimilation_time_step = timedelta(hours=1)  

saveoutput_time_step = timedelta(hours=3)

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
       
name_model = 'SW1L'

dir_model =  '../models/model_sw1l/'
    
name_mod_var = ["u","v","ssh"]

n_mod_var = len(name_mod_var)             

name_mod_lon = "nav_lon"

name_mod_lat = "nav_lat"

####################################
### Function-specific parameters ### 
#################################### 
# - parameters specific to SW model

dtmodel = 1200

sw_time_scheme = 'rk4' # Time scheme of the model (e.g. Euler,rk4,lf)

bc_kind = '1d'

w_igws = [2*pi/12/3600] # igw frequencies (in seconds)

He_init = 0.9 # Mean height (in m)

Ntheta = 1 # Number of angles (computed from the normal of the border) of incoming waves

D_He = None #150e3 # Space scale of gaussian decomposition for He (in m)

T_He = timedelta(days=20).total_seconds() # Time scale of gaussian decomposition for He (in m)

D_bc = 150e3 # Space scale of gaussian decomposition for boundary conditions (in m)

T_bc = timedelta(days=20).total_seconds() # Time scale of gaussian decomposition for boundary conditions (in m)

He_data = None # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

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

name_analysis = '4Dvar'

####################################
### Function-specific parameters ### 
#################################### 

path_init_4Dvar = None 

checkpoint = 100 # Number of model timesteps separating two consecutive analysis 

sigma_R = 1e-2 # Observational standard deviation

sigma_B_He = 0.2 # Background variance for He

sigma_B_bc = 1e-3 # Background variance for bc

prec = False # preconditoning

gtol = 1e-2 # Gradient norm must be less than gtol before successful termination.

maxiter = 3 # Maximal number of iterations for the minimization process

#################################################################################################################################
# Observation parameters
#################################################################################################################################
# - satellite: list of satellite names 

satellite = ["nr"]
write_obs = False
# - For each *satellite*:
#    * kind_sat: "swathSSH" for SWOT, "nadir" for nadirs  
#    * obs_path_sat: directory where the observations are stored
#    * obs_prefixe_sat: prefixe in observation files
#    * name_obs_var_sat: name of the observed variables
#    * name_obs_lon_sat: name of the observation longitude
#    * name_obs_lat_sat: name of the observation latitude
#    * name_obs_time_sat: name of the observation time
#    * name_obs_xac_sat: name of the observation across track distance (only for swathSSH satellites)
#################################################################################################################################
kind_nr = "fullSSH"
obs_path_nr = '../../data_Example2/data_BM-IT_idealized/'
obs_name_nr = "obs"
name_obs_var_nr = ["ssh_obs"]
name_obs_lon_nr = "lon"
name_obs_lat_nr = "lat"
name_obs_time_nr = "time_obs"
nudging_params_stretching_nr = None
nudging_params_relvort_nr = None

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
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
# - name_grd: name used for saving the QG grid to avoid calculating it every time.
#################################################################################################################################
        
tmp_DA_path = "../scratch/" +  name_exp_save + '/'






   




