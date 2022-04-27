#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################

name_experiment = 'my_exp' # name of the experiment

#################################################################################################################################
# Outputs parameters
#################################################################################################################################

saveoutputs = True # save outputs flag (True or False)

name_exp_save = 'my_output_name' # name of output files

path_save = 'outputs/' + name_exp_save + '/' # path of output files

flag_plot = 0 # between 0 and 4. 0 for none plot, 4 for full plot
    
#################################################################################################################################
# Temporary DA parameters
#################################################################################################################################
        
tmp_DA_path = "scratch/" +  name_exp_save + '/' # temporary data assimilation directory path
 

#################################################################################################################################
# Global libraries     
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
   
init_date = datetime(2012,10,1,0) # initial date (yyyy,mm,dd,hh) 

final_date = datetime(2012,12,2,0)  # final date (yyyy,mm,dd,hh) 

assimilation_time_step = timedelta(hours=1)  # assimilation time step (corresponding to observation update timestep)

saveoutput_time_step = timedelta(hours=1)  # time step at which the states are saved 

plot_time_step = timedelta(days=1)  #  time step at which the states are plotted (for debugging)

#################################################################################################################################
# Model parameters
#################################################################################################################################
       
name_model = 'QG1L'     
    
name_mod_var = ["ssh"]  

n_mod_var = len(name_mod_var)             

name_mod_lon = "lon"

name_mod_lat = "lat"

dtmodel = 300   # model timestep


# - parameters specific to QG model

upwind = 3 # Order of the upwind scheme for PV advection (either 1,2 or 3)

upwind_adj = None # idem but for the adjoint loop

Reynolds = False # If True, Reynolds decomposition will be applied. Be sure to have provided MDT and that obs are SLAs!

qgiter = 20 # number of iterations to perform the gradient conjugate algorithm (to inverse SSH from PV)

qgiter_adj = None # idem for the adjoint loop

c0 = 2.7 # If not None, fixed value for phase velocity 

filec_aux = '' # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

name_var_c = {'lon':'','lat':'','var':''} # Variable names for the phase velocity auxilliary file 

only_diffusion = False # If True, use only diffusion in the QG propagation

cdiffus = 0. # Coefficient for the diffusion 

path_mdt = None # If provided, QGPV will be expressed thanks to the Reynolds decompositon

name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''} 


# - parameters specific to SW model

sw_time_scheme = 'lf' # Time scheme of the model (e.g. Euler,rk4,lf)

sw_in = 0 # Length of the boundary band to ignore  

bc_kind = '1d' # Either 1d or 2d

w_igws = [2*pi/12/3600] # igw frequencies (in seconds)

Nmodes = 1

He_init = 0.9 # Mean height (in m)

He_data = None # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used


#################################################################################################################################
# Analysis parameters
#################################################################################################################################

name_analysis = 'BFN'

###################################
###   BFN-specific parameters   ### 
################################### 

bfn_window_size = timedelta(days=7) # length of the bfn time window

bfn_window_output = timedelta(days=3) # length of the output time window, in the middle of the bfn window. (need to be smaller than *bfn_window_size*)

bfn_propation_timestep = timedelta(hours=1) # propagation time step of the BFN, corresponding to the time step at which the nudging term is computed

bfn_window_overlap = True # overlap the BFN windows

bfn_criterion = 0.01 # convergence criterion. typical value: 0.01

bfn_max_iteration = 5 # maximal number of iterations if *bfn_criterion* is not met

save_bfn_trajectory = False # save or not the back and forth iterations (for debugging)

dist_scale = 10 # distance (in km) for which observations are spatially spread

save_obs_proj = False # save or not the projected observation as pickle format. Set to True to maximize the speed of the algorithm.

path_save_proj = None # path to save projected observations

flag_use_bc = False # set or not boundary conditions

lenght_bc = 50 # lenght (in km) of the peripherical band for which the boundary conditions are applied

file_bc = None # netcdf file(s) in whihch the boundary conditions fields are stored

name_var_bc = {'time':'','lon':'','lat':'','var':''} # name of the boundary conditions variable

add_mdt_bc = False # Add mdt to boundary conditions. Useful only if boundary conditions are on sla

use_bc_on_coast = True # use boundary conditions on coast. Useful only if MDT or a mask is provided 

bc_mindepth = None # minimal depth below which boundary conditions are used  

file_depth = None # netcdf file(s) in which the topography is stored

name_var_depth = {'time':'','lon':'','lat':'','var':''} # name of the topography variable

scalenudg = None 

Knudg = None


####################################
###   4Dvar-specific parameters  ### 
####################################

compute_test = False # TLM, ADJ & GRAD tests

path_init_4Dvar = None # To restart the minimization process from a specified control vector

restart_4Dvar = False # To restart the minimization process from the last control vector

gtol = None # Gradient norm must be less than gtol before successful termination.

maxiter = 20 # Maximal number of iterations for the minimization process

maxiter_inner = 3 # Maximal number of iterations for the outer loop (only for incr4Dvar)

maxiter_outer = 3 # Maximal number of iterations for the inner loop (only for incr4Dvar)

opt_method = 'L-BFGS-B' # method for scipy.optimize.minimize

save_minimization = False # save cost function and its gradient at each iteration 

path_H = None # Directory where to save observational operator

compute_H = False # Force computing H 

Npix_H = 4 # Number of pixels to perform projection y=Hx

checkpoint = 1 # Number of model timesteps separating two consecutive analysis 

window_length = timedelta(days=3) # Length of the 4Dvar time window

window_save = timedelta(days=1) # Length of the saving 4Dvar time window

window_overlap = True # If True, smooth output trajectory in time 

sigma_R = 1e-2 # Observational standard deviation

sigma_B_He = 0.2 # Background variance for He

sigma_B_bc = 1e-2 # Background variance for bc

facB_bc_coast = 1 # Factor for sigma_B_bc located at coast. Useful only if mask is provided

facB_He_coast = 1  # Factor for sigma_B_He located at coast. Useful only if mask is provided

sigma_B_grad = 1 # Background variance for regularization term (proportional to grad(X))

scalemodes = None # Only for SW1LM model, 

scalew_igws = None 

prec = False # preconditoning

grad_term = False # Add a term that minimizes the gradient of SSH in the cost function 

filter_name = None # name of filter used in preconditionning

filter_order = None # order of the filter

mask_coast = False

dist_coast = 100 # km


####################################
###   MIOST-specific parameters  ### 
#################################### 

miost_window_size = timedelta(days=15)

miost_window_output = timedelta(days=15)

miost_window_overlap = True

dir_miost = None

obs_subsampling = 1


#########################################
### Reduced basis specific parameters ### 
#########################################

# - For BM

save_wave_basis = False # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

wavelet_init = True # Estimate the initial state 

facns = 1. #factor for wavelet spacing= space

facnlt = 2. #factor for wavelet spacing= time

npsp= 3.5 # Defines the wavelet shape

facpsp= 1.5 # factor to fix df between wavelets

lmin= 80 # minimal wavelength (in km)

lmax= 970. # maximal wavelength (in km)

lmeso = 300 # Largest mesoscale wavelenght 

tmeso = 20 # Largest mesoscale time of decorrelation 

sloptdec = -1.28 # Slope such as tdec = lambda^slope where lamda is the wavelength

factdec = 0.5 # factor to be multiplied to the computed time of decorrelation 

tdecmin = 2.5 # minimum time of decorrelation 

tdecmax = 40. # maximum time of decorrelation 

facQ= 1 # factor to be multiplied to the estimated Q

Qmax = 100 # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

slopQ = -5 # Slope such as Q = lambda^slope where lamda is the wavelength

# - For IT 

Ntheta = 1 # Number of angles (computed from the normal of the border) of incoming waves

facgauss = 3.5  # factor for gaussian spacing= both space/time

D_He = 200 # Space scale of gaussian decomposition for He (in km)

T_He = 20 # Time scale of gaussian decomposition for He (in days)

D_bc = 200 # Space scale of gaussian decomposition for boundary conditions (in km)

T_bc = 20 # Time scale of gaussian decomposition for boundary conditions (in days)



#################################################################################################################################
# Observation parameters
#################################################################################################################################

satellite = ['swot']

time_obs_min = None 

time_obs_max = None

write_obs = False # save observation dictionary in *path_obs*

compute_obs = False # force computing observations 

path_obs = None # if set to None, observations are saved in *tmp_DA_path*

detrend = False # apply a 2D detrending on observations

substract_mdt = False

# - For each *satellite*:

kind_swot = "swot_simulator"   # "swathSSH" for SWOT, "nadir" for nadirs  
obs_path_swot = '../../data_Example1/dc_obs/' # directory where the observations are stored
obs_name_swot = "2020a_SSH_mapping_NATL60_karin_swot.nc" # full name (or prefixe) in observation file(s)
name_obs_var_swot = ["ssh_model"] # name of the observed variables 
name_obs_lon_swot = "lon" # name of longitude variable
name_obs_lat_swot = "lat" # name of latitude variable
name_obs_time_swot = "time" # name of time variable
name_obs_xac_swot = "x_ac" # name of the across track distance variable (only for swathSSH satellites)
sigma_noise_swot = 1e-2 # Noise of observations
add_mdt_swot = False # Whether to add mdt to observation variables or not
substract_mdt_swot = False # Whether to substract mdt to observation variables or not
nudging_params_stretching_swot = {'sigma':0,'K':0.1,'Tau':timedelta(days=1)} # nudging parameters relative to the stretching part of the PV. 
nudging_params_relvort_swot = None # nudging parameters relative to the relative part of the PV (for nadir, set to None).



   




