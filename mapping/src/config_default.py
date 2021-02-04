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

name_experiment = 'my_exp' 
name_domain = 'my_domain'

#################################################################################################################################
# Global libraries     
#################################################################################################################################
# - datetime
# - timedelta
#################################################################################################################################
import os
from datetime import datetime,timedelta
from math import pi
#################################################################################################################################
# Initialization parameters
#################################################################################################################################
name_init = 'geo_grid' # Either 'geo_grid' or 'from_file'

name_init_file = 'init_state.nc' # Name of init file, which will be used by other functions

# For name_init=='from_file'

name_init_grid = '' 

name_init_lon = ''

name_init_lat = ''

# For name_init=='geo_grid'

lon_min = 294.                                        # domain min longitude

lon_max = 306.                                        # domain max longitude

lat_min = 32.                                         # domain min latitude

lat_max = 44.                                         # domain max latitude

dx = 1/10.                                            # zonal grid spatial step (in degree)

dy = 1/10.                                            # meridional grid spatial step (in degree)


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

final_date = datetime(2012,12,2,0)  

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

dir_model = os.path.dirname(os.path.abspath(__file__)) + '../models/model_qgsw/'

dtmodel = 300   

qgiter = 20

c = 2.7

cdiffus = 0. 

only_diffusion = False

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
### BFN-specific parameters ### 
#################################### 

bfn_window_size = timedelta(days=7)

bfn_window_output = timedelta(days=3)

bfn_propation_timestep = timedelta(hours=1)

bfn_window_overlap = True

bfn_criterion = 0.01

bfn_max_iteration = 5

save_bfn_trajectory = False

dist_scale = 10 # in km

save_obs_proj = False

path_save_proj = None

flag_use_boundary_conditions = True

file_boundary_conditions = None

lenght_bc = 20

name_var_bc = None

scalenudg = None

####################################
### 4Dvar-specific parameters ### 
#################################### 

path_init_4Dvar = None 

sw_time_scheme = 'lf' # Time scheme of the model (e.g. Euler,rk4,lf,ab3)

bc_kind = '1d'

w_igws = [2*pi/12/3600] # igw frequencies (in seconds)

He_init = 0.9 # Mean height (in m)

Ntheta = 1 # Number of angles (computed from the normal of the border) of incoming waves

D_He = 200e3 # Space scale of gaussian decomposition for He (in m)

T_He = timedelta(days=10).total_seconds() # Time scale of gaussian decomposition for He (in m)

D_bc = 200e3 # Space scale of gaussian decomposition for boundary conditions (in m)

T_bc = timedelta(days=10).total_seconds() # Time scale of gaussian decomposition for boundary conditions (in m)

sigma_R = 1e-2 # Observational standard deviation

sigma_B_He = 0.2 # Background variance for He

sigma_B_bc = 1e-2 # Background variance for bc

He_data = None#{'path':'/Users/leguillou/WORK/Developpement/Studies/DA_IGWs/data/wp6_He_16km_mean.nc',
           #'time':None,'lon':'lon','lat':'lat','var':'He'}  # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

gtol = 1e-5 # Gradient norm must be less than gtol before successful termination.

maxiter = 20 # Maximal number of iterations for the minimization process

#################################################################################################################################
# Observation parameters
#################################################################################################################################
# - satellite: list of satellite names 

satellite = ["swot","nadir_swot","jason1","geosat2","envisat","topex"]
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

kind_swot = "swot_simulator"
obs_path_swot = 'dc_obs/'
obs_name_swot = "2020a_SSH_mapping_NATL60_karin_swot.nc" 
name_obs_var_swot = ["ssh_model"]     
name_obs_lon_swot = "lon"
name_obs_lat_swot = "lat"
name_obs_time_swot = "time"
name_obs_xac_swot = "x_ac"
nudging_params_stretching_swot = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_swot = {'sigma':0,'K':0.05,'Tau':timedelta(hours=12)}

kind_nadir_swot = "swot_simulator"
obs_path_nadir_swot = 'dc_obs/'
obs_name_nadir_swot = "2020a_SSH_mapping_NATL60_nadir_swot.nc" 
name_obs_var_nadir_swot = ["ssh_model"]     
name_obs_lon_nadir_swot = "lon"
name_obs_lat_nadir_swot = "lat"
name_obs_time_nadir_swot = "time"
name_obs_xac_nadir_swot = None
nudging_params_stretching_nadir_swot = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_nadir_swot = None

kind_jason1 = "swot_simulator"
obs_path_jason1 = 'dc_obs/'
obs_name_jason1 = "2020a_SSH_mapping_NATL60_jason1.nc" 
name_obs_var_jason1 = ["ssh_model"]     
name_obs_lon_jason1 = "lon"
name_obs_lat_jason1 = "lat"
name_obs_time_jason1 = "time"
name_obs_xac_jason1 = None
nudging_params_stretching_jason1 = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_jason1 = None

kind_geosat2 = "swot_simulator"
obs_path_geosat2 = 'dc_obs/'
obs_name_geosat2 = "2020a_SSH_mapping_NATL60_geosat2.nc" 
name_obs_var_geosat2 = ["ssh_model"]     
name_obs_lon_geosat2 = "lon"
name_obs_lat_geosat2 = "lat"
name_obs_time_geosat2 = "time"
name_obs_xac_geosat2 = None
nudging_params_stretching_geosat2 = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_geosat2 = None

kind_envisat = "swot_simulator"
obs_path_envisat = 'dc_obs/'
obs_name_envisat = "2020a_SSH_mapping_NATL60_envisat.nc" 
name_obs_var_envisat = ["ssh_model"]     
name_obs_lon_envisat = "lon"
name_obs_lat_envisat = "lat"
name_obs_time_envisat = "time"
name_obs_xac_envisat = None
nudging_params_stretching_envisat = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_envisat = None

kind_topex = "swot_simulator"
obs_path_topex = 'dc_obs/'
obs_name_topex = "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc" 
name_obs_var_topex = ["ssh_model"]     
name_obs_lon_topex = "lon"
name_obs_lat_topex = "lat"
name_obs_time_topex = "time"
name_obs_xac_topex = None
nudging_params_stretching_topex = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)}
nudging_params_relvort_topex = None

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

path_save = 'outputs/' + name_exp_save + '/'

flag_plot = 1
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
# - name_grd: name used for saving the QG grid to avoid calculating it every time.
#################################################################################################################################
        
tmp_DA_path = "scratch/" +  name_exp_save + '/'
 
name_grd = tmp_DA_path + 'QGgrid'





   




