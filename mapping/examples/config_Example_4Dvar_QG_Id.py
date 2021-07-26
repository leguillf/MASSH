#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:02:10 2021

@author: renamatt
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

name_experiment = '4Dvar_QG_SWOT' 
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

name_init = 'from_file'

name_init_grid = '../../data_Example1/data_BM-IT_idealized/ref.nc'

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
# - window_time_step : time step corresponding to the size of an assimilation window
#################################################################################################################################
   
init_date = datetime(2016,1,2,0)     

final_date = datetime(2016,1,20,0)

assimilation_time_step = timedelta(hours=3)  

saveoutput_time_step = timedelta(hours=3) 

window_time_step = timedelta(days=6)

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
#################################################################################################################################

qgiter = 20

c = 2.7

dtmodel = 300

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
#    * name_var_bc: name of the boundary conditions variable
#################################################################################################################################

name_analysis = '4Dvar'

####################################
### 4Dvar-specific parameters ### 
#################################### 

#################################################################################################################################
# - name_analysis: analysis function. Here, of course, we set BFN
# - parameters specific to BFN:
#    * sigma_B : standard deviation of the background error covariance matrix
#    * sigma_R : standard deviation of the observation error covariance matrix 
#    * maxiter : maximal number of iteration allowwed in the minimization process
#    * background_state : path to ncdf file containing background ssh field
#    *
#################################################################################################################################

path_init_4Dvar = None

sigma_B = None

sigma_R = 0.1

maxiter = 15

gtol = 1e-5

prec = False

filter_use = False

filter_name = None

filter_order = None

gradient term = False

sigma_B_grad = None

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
#    * name_obs_xac_sat: name of the observation across track distance (only for swathSSH satellites).
#    * use_invobs_file_sat: use existing projected observations computed from SOSIE for instance (True or False). For BFN, always set False.
#    * swath_width_swot_sat: swath width in km (only for swathSSH satellites)
#    * gap_width_swot_sat: gap width in km (only for swathSSH satellites)
#################################################################################################################################


kind_nr = "fullSSH"
obs_path_nr = '../../data_Example2/data_BM-IT_idealized/'
obs_name_nr = "obs"
name_obs_var_nr = ["ssh_meso"]
name_obs_lon_nr = "lon"
name_obs_lat_nr = "lat"
name_obs_time_nr = "time_obs"

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

flag_plot = 1
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
# - name_grd: name used for saving the QG grid to avoid calculating it every time.
#################################################################################################################################
        
tmp_DA_path = "../scratch/" +  name_exp_save + '/'
 
name_grd = tmp_DA_path + 'QGgrid'





   




