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

flag_plot = 0
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
# - tmp_DA_path: temporary data assimilation directory path
#################################################################################################################################
        
tmp_DA_path = "scratch/" +  name_exp_save + '/'
 

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
name_init = 'geo_grid' # Either 'geo_grid' or 'from_file'

name_init_file = 'init_state.nc' # Name of init file, which will be used by other functions

# For name_init=='from_file'

name_init_grid = '' 

name_init_lon = ''

name_init_lat = ''

name_init_var = None

# For name_init=='geo_grid'

lon_min = 294.                                        # domain min longitude

lon_max = 306.                                        # domain max longitude

lat_min = 32.                                         # domain min latitude

lat_max = 44.                                         # domain max latitude

dx = 1/10.                                            # zonal grid spatial step (in degree)

dy = 1/10.                                            # meridional grid spatial step (in degree)

# Mask 

name_init_mask = None

name_var_mask = {'lon':'','lat':'','var':''}

# Gravity

g = 9.81

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
    
name_mod_var = ["ssh"]  

n_mod_var = len(name_mod_var)             

name_mod_lon = "lon"

name_mod_lat = "lat"

# - parameters specific to QG model
#    * qgiter: number of iterations to perform the gradient conjugate algorithm (to inverse SSH from PV)
#    * c: first baroclinic gravity-wave phase speed (in m/s) related to Rossby Radius of deformation
#    * dtmodel: timestep of the model (in seconds). Typical values: between 200s and 1000s. If the model crashes, reduce its value.
#    * path_mdt: (string) full path of the netcdf file containing MDT data
#    _ name_var_mdt: (dict) {'lon':name_lon_mdt,'lat':name_lon_mdt,'var':name_var_mdt}

dir_model =  None

dtmodel = 300   

qgiter = 20

c0 = 2.7

cdiffus = 0. 

only_diffusion = False

path_mdt = None # If provided, QGPV will be expressed thanks to the Reynolds decompositon

name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''} 

# - parameters specific to SW model

sw_time_scheme = 'lf' # Time scheme of the model (e.g. Euler,rk4,lf)

bc_kind = '1d'

w_igws = [2*pi/12/3600] # igw frequencies (in seconds)

Nmodes = 1

He_init = 0.9 # Mean height (in m)

Ntheta = 1 # Number of angles (computed from the normal of the border) of incoming waves

D_He = 200e3 # Space scale of gaussian decomposition for He (in m)

T_He = timedelta(days=10).total_seconds() # Time scale of gaussian decomposition for He (in m)

D_bc = 200e3 # Space scale of gaussian decomposition for boundary conditions (in m)

T_bc = timedelta(days=10).total_seconds() # Time scale of gaussian decomposition for boundary conditions (in m)

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

name_analysis = 'BFN'

flag_use_boundary_conditions = True

lenght_bc = 50

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

file_boundary_conditions = None

name_var_bc = {'time':'','lon':'','lat':'','var':''}

scalenudg = None

Knudg = None

####################################
### 4Dvar-specific parameters ### 
#################################################################################################################################
# - path_ini_4Dvar : path to file used to initialize the SSH field
# - checkpoint :  Number of model timesteps separating two consecutive analysis 
# - sigma_R : standard deviation error of the observation error
# - sigma_B_He : Background variance for He
# - sigma_B_bc : Background variance for bc
# - prec : use of a preconditionning
# - filter_name : name of the filter used in case of preconditionning
# - filter_order : order of the filter used
# - gtol : Gradient norm must be less than gtol before successful termination
# - maxiter : Maximal number of iterations for the minimization process
# - eps_bc : Damping ratio of the R^{-1} matrix at border pixels
#################################################################################################################################

reduced_basis = False # whether to compute wavelet basis for 4DvarQG system

compute_test = False # TLM,ADJ & GRAD tests

path_init_4Dvar = None 

path_H = None 

Npix_H = 4 # Number of pixels to perform projection y=Hx

checkpoint = 1 # Number of model timesteps separating two consecutive analysis 

window_length = timedelta(days=3) # Length of the 4Dvar time window

window_save = timedelta(days=1) # Length of the saving 4Dvar time window

window_overlap = True # If True, smooth output trajectory in time 

sigma_R = 1e-2 # Observational standard deviation

sigma_B_He = 0.2 # Background variance for He

sigma_B_bc = 1e-2 # Background variance for bc

sigma_B_grad = 1 # Background variance for regularization term (proportional to grad(X))

scalemodes = None # Only for SW1LM model, 

scalew_igws = None 

prec = False # preconditoning

grad_term = False # Add a term that minimizes the gradient of SSH in the cost function 

filter_name = None # name of filter used in preconditionning

filter_order = None # order of the filter

gtol = 1e-5 # Gradient norm must be less than gtol before successful termination.

maxiter = 20 # Maximal number of iterations for the minimization process

mask_coast = True

dist_coast = 100 #km

####################################
### MIOST-specific parameters ### 
#################################### 

miost_window_size = timedelta(days=15)

miost_window_output = timedelta(days=15)

miost_window_overlap = True

dir_miost = None

obs_subsampling = 1


#########################################
### Wavelet basis specific parameters ### 
#########################################

file_aux = ''

filec_aux = ''

name_var_c = {'lon':'lon','lat':'lat','var':'c1'}

facns= 1. #factor for wavelet spacing= space

facnlt= 2.

npsp= 3.5 # Defines the wavelet shape

facpsp= 1.5 #1.5 # factor to fix df between wavelets

lmin= 80 

lmax= 970.

cutRo= 1.6

factdec= 15.

tdecmin= 2.5

tdecmax= 30.

tssr= 0.5

facRo= 8.

Romax= 150.

facQ= 1,

depth1= 200.  

depth2= 2000.   

distortion_eq= 2.

lat_distortion_eq= 5.

distortion_eq_law= 2.

gsize_max = 500000000

#################################################################################################################################
# Observation parameters
#################################################################################################################################
# - satellite: list of satellite names 
# - write_obs: (bool) save observation dictionary in *path_obs*
# - path_obs: (string) if set to None, observations are saved in *tmp_DA_path*
# - detrend: (bool) apply a 2D detrending on observations


satellite = []

path_obs = None

detrend = False

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







   




