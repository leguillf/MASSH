#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:40:30 2021

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

name_experiment = 'test_4Dvar_QG_swot' 
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

name_init = 'geo_grid'                                # either 'geo_grid' or 'from_file'

# - parameters specific to 'geo_grid'  

lon_min = 295.                                        # domain min longitude

lon_max = 305.                                        # domain max longitude

lat_min = 33.                                         # domain min latitude

lat_max = 41.                                         # domain max latitude

dx = 1/5.                                            # zonal grid spatial step (in degree)

dy = 1/5.                                            # meridional grid spatial step (in degree)

#################################################################################################################################
# Time parameters
#################################################################################################################################
# - init_date: initial date (yyyy,mm,dd,hh) 
# - final_date: final date (yyyy,mm,dd,hh)
# - assimilation_time_step: assimilation time step (corresponding to observation update timestep)
# - savepoutput_time_step: time step plot at which the states are saved 
# - plot_time_step: time step plot at which the states are plotted (for debugging)
#################################################################################################################################
   
init_date = datetime(2013,1,2,0)     

final_date = datetime(2013,1,20,0)

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
#    *
#    *
#    *
#################################################################################################################################

path_init_4Dvar = None

sigma_B = 1.

sigma_R = 0.1

maxiter = 15

gtol = 1e-5

prec = False

filter_name = None

filter_order = None

#################################################################################################################################
# Observation parameters
#################################################################################################################################
# - satellite: list of satellite names 

satellite = ["swot"]
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

kind_swot = "swot_simulator"
obs_path_swot = '../../data_Example1/dc_obs/'
obs_name_swot = "2020a_SSH_mapping_NATL60_karin_swot.nc" 
name_obs_var_swot = ["ssh_model"]     
name_obs_lon_swot = "lon"
name_obs_lat_swot = "lat"
name_obs_time_swot = "time"
name_obs_xac_swot = "x_ac"

kind_nadir_swot = "swot_simulator"
obs_path_nadir_swot = '../../data_Example1/dc_obs/'
obs_name_nadir_swot = "2020a_SSH_mapping_NATL60_nadir_swot.nc" 
name_obs_var_nadir_swot = ["ssh_model"]     
name_obs_lon_nadir_swot = "lon"
name_obs_lat_nadir_swot = "lat"
name_obs_time_nadir_swot = "time"
name_obs_xac_nadir_swot = None

kind_jason1 = "swot_simulator"
obs_path_jason1 = '../../data_Example1/dc_obs/'
obs_name_jason1 = "2020a_SSH_mapping_NATL60_jason1.nc" 
name_obs_var_jason1 = ["ssh_model"]     
name_obs_lon_jason1 = "lon"
name_obs_lat_jason1 = "lat"
name_obs_time_jason1 = "time"
name_obs_xac_jason1 = None

kind_geosat2 = "swot_simulator"
obs_path_geosat2 = '../../data_Example1/dc_obs/'
obs_name_geosat2 = "2020a_SSH_mapping_NATL60_geosat2.nc" 
name_obs_var_geosat2 = ["ssh_model"]     
name_obs_lon_geosat2 = "lon"
name_obs_lat_geosat2 = "lat"
name_obs_time_geosat2 = "time"
name_obs_xac_geosat2 = None

kind_envisat = "swot_simulator"
obs_path_envisat = '../../data_Example1/dc_obs/'
obs_name_envisat = "2020a_SSH_mapping_NATL60_envisat.nc" 
name_obs_var_envisat = ["ssh_model"]     
name_obs_lon_envisat = "lon"
name_obs_lat_envisat = "lat"
name_obs_time_envisat = "time"
name_obs_xac_envisat = None

kind_topex = "swot_simulator"
obs_path_topex = '../../data_Example1/dc_obs/'
obs_name_topex = "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc" 
name_obs_var_topex = ["ssh_model"]     
name_obs_lon_topex = "lon"
name_obs_lat_topex = "lat"
name_obs_time_topex = "time"
name_obs_xac_topex = None

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





   




