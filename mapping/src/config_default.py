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

    name_experiment = 'my_exp', # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = 'my_output_name', # name of output files

    path_save = 'outputs', # path of output files

    tmp_DA_path = "scratch/", # temporary data assimilation directory path

    flag_plot = 0, # between 0 and 4. 0 for none plot, 4 for full plot

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,2,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  # assimilation time step (corresponding to observation update timestep)

    saveoutput_time_step = timedelta(hours=1),  # time step at which the states are saved 

    plot_time_step = timedelta(days=1),  #  time step at which the states are plotted (for debugging),

    time_obs_min = None, 

    time_obs_max = None,

    write_obs = False, # save observation dictionary in *path_obs*

    compute_obs = False, # force computing observations 

    path_obs = None # if set to None, observations are saved in *tmp_DA_path*

)


#################################################################################################################################
# GRID 
#################################################################################################################################
NAME_GRID = 'GRID_GEO'

# From file
GRID_FROM_FILE = dict(

    path_init_grid = '', 

    name_init_lon = '',

    name_init_lat = '',

    name_init_var = [None], # List of variables with same dimension of the model state variables. If one variable is None, it will be set to zero.

    subsampling = None,

)

# Regular geographical grid
GRID_GEO = dict(

    lon_min = 294.,                                        # domain min longitude

    lon_max = 306.,                                        # domain max longitude

    lat_min = 32.,                                         # domain min latitude

    lat_max = 44.,                                         # domain max latitude

    dlon = 1/10.,                                            # zonal grid spatial step (in degree)

    dlat = 1/10.,                                            # meridional grid spatial step (in degree)

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

GRID_CAR = dict(

    super = 'GRID_CAR',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dx = 25.,                                              # grid spacinng in km

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

# Restart from previous run 
GRID_RESTART = dict(

    name_grid = 'restart',

)


#################################################################################################################################
# OBSERVATIONS 
#################################################################################################################################
NAME_OBS = None

# Nadir altimetry
OBS_SSH_NADIR = dict(

    path = '',

    name_time = '',
    
    name_lon = '',

    name_lat = '',
    
    name_var = {'SSH':''},

    sigma_noise = None,

    add_mdt = None,

    substract_mdt = None,

    path_mdt = None,

    name_var_mdt = None,
    
    nudging_params_ssh = None,

    detrend = False

)

# Swath altimetry
OBS_SSH_SWATH = dict(

    path = '',

    name_time = '',
    
    name_lon = '',

    name_lat = '',

    name_xac = None,
    
    name_var = {'SSH':''},

    sigma_noise = None,

    add_mdt = None,

    substract_mdt = None,

    path_mdt = None,

    name_var_mdt = None,
    
    nudging_params_ssh = None,

    nudging_params_relvort = None,

    detrend = False
    
)

#################################################################################################################################
# MODELS
#################################################################################################################################
NAME_MOD = None # Either DIFF, QG1L, QG1LM, SW1L, SW1LM    

# Diffusion model
MOD_DIFF = dict(

    name_var = {'SSH':"ssh"},

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0 # coefficient of diffusion. Set to 0 for Identity model

)

# 1.5-layer Quasi-Geostrophic model
MOD_QG1L_NP = dict(

    name_var = {'SSH':"ssh"},

    name_init_var = {},

    dir_model = None,

    dtmodel = 300, # model timestep

    upwind = 3, # Order of the upwind scheme for PV advection (either 1,2 or 3)

    upwind_adj = None, # idem but for the adjoint loop

    Reynolds = False, # If True, Reynolds decomposition will be applied. Be sure to have provided MDT and that obs are SLAs!

    qgiter = 20, # number of iterations to perform the gradient conjugate algorithm (to inverse SSH from PV)

    qgiter_adj = None, # idem for the adjoint loop

    c0 = 2.7, # If not None, fixed value for phase velocity 

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None,

    cmax = None,

    only_diffusion = False, # If True, use only diffusion in the QG propagation

    path_mdt = None, # If provided, QGPV will be expressed thanks to the Reynolds decompositon

    name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''},

    g = 9.81 

)

MOD_QG1L_JAX = dict(

    name_var = {'SSH':"ssh"},

    name_init_var = {},

    dir_model = None,

    multiscale = False,

    dtmodel = 300, # model timestep

    time_scheme = 'Euler', # Time scheme of the model (e.g. Euler,rk4)

    upwind = 3, # Order of the upwind scheme for PV advection (either 1,2 or 3)

    upwind_adj = None, # idem but for the adjoint loop

    Reynolds = False, # If True, Reynolds decomposition will be applied. Be sure to have provided MDT and that obs are SLAs!

    c0 = 2.7, # If not None, fixed value for phase velocity 

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None,

    cmax = None,

    only_diffusion = False, # If True, use only diffusion in the QG propagation

    path_mdt = None, # If provided, QGPV will be expressed thanks to the Reynolds decompositon

    name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''},

    g = 9.81 

)

# 1.5-layer Shallow-Water model
MOD_SW1L = dict(

    name_var = ["u","v","ssh"],

    name_init_var = [],

    dir_model = None,

    use_jax = False,

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_igws = [2*3.14/12/3600], # igw frequencies (in seconds)

    He_init = 0.9, # Mean height (in m)

    He_data = None, # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

    Ntheta = 1 # Number of angles (computed from the normal of the border) of incoming waves

)


#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = None # For now, only BC_EXT is available

# External boundary conditions
BC_NUD = dict(

    lenght_bc = 50, # lenght (in km) of the peripherical band for which the boundary conditions are applied

    file_bc = None, # netcdf file(s) in whihch the boundary conditions fields are stored

    name_var_bc = {'time':'','lon':'','lat':'','var':''}, # name of the boundary conditions variable

    add_mdt_bc = False, # Add mdt to boundary conditions. Useful only if boundary conditions are on sla

    use_bc_on_coast = True, # use boundary conditions on coast. Useful only if MDT or a mask is provided 

    file_depth = None, # netcdf file(s) in which the topography is stored

    name_var_depth = {'time':'','lon':'','lat':'','var':''}, # name of the topography variable

    bc_mindepth = None # minimal depth below which boundary conditions are used  

)

BC_EXT = dict(

    file = None, # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = '',

    name_lat = '',

    name_time = '',

    name_var = {},

    name_mod_var = {},

    dist_sponge = None  # distance (in km) for which boundary fields are spatially spread close to the borders (useful for MOD_QG1L_NP model)

)


#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = None

OBSOP_INTERP = dict(

    path_save = None, # Directory where to save observational operator

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_coast = False,

    dist_coast = 100, # km

    mask_borders = False,

)

#################################################################################################################################
# INVERSION METHODS
#################################################################################################################################
NAME_INV = None

# Optimal Interpolation
INV_OI = dict(

    Lt = 7, # days

    Lx = 1, # degreee

    Ly = 1, # degree

    noise = 5e-2 # meters

)

# Back and Forth Nudging
INV_BFN = dict(

    window_size = timedelta(days=7), # length of the bfn time window

    window_output = timedelta(days=3), # length of the output time window, in the middle of the bfn window. (need to be smaller than *bfn_window_size*)

    propagation_timestep = timedelta(hours=1), # propagation time step of the BFN, corresponding to the time step at which the nudging term is computed

    window_overlap = True, # overlap the BFN windows

    criterion = 0.01, # convergence criterion. typical value: 0.01

    max_iteration = 5, # maximal number of iterations if *bfn_criterion* is not met

    save_trajectory = False, # save or not the back and forth iterations (for debugging)

    dist_scale = 10, #

    save_obs_proj = False, # save or not the projected observation as pickle format. Set to True to maximize the speed of the algorithm.

    path_save_proj = None, # path to save projected observations

    use_bc_as_init = False, # Whether to use boundary conditions as initialization for the first temporal window

    scalenudg = None 

)

# 4-Dimensional Variational 
INV_4DVAR = dict(

    compute_test = False, # TLM, ADJ & GRAD tests

    path_init_4Dvar = None, # To restart the minimization process from a specified control vector

    restart_4Dvar = False, # To restart the minimization process from the last control vector

    gtol = None, # Gradient norm must be less than gtol before successful termination.

    maxiter = 10, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    sigma_B = None,

    prec = False, # preconditoning
    
    prescribe_background = False, # To prescribe a background on BM basis or compute it from a 4Dvar-Identity model (eq. to MIOST)

    bkg_satellite = None, # satellite constellation for 4Dvar-Identity model background if prescribe_background == True

    path_background = None, # Path to the precribed background on BM basis
    
    bkg_Kdiffus = 0., # 0 diffusion to perform the 4Dvar-Identity model 

    name_bkg_var = 'res' ,# Default name of the BM basis variable the prescribed or computed background 

    bkg_maxiter = 30, # 4Dvar-Identity model maximal number of iterations for the minimization process

    bkg_maxiter_inner = 10, # 4Dvar-Identity model maximal number of iterations for the outer loop (only for incr4Dvar)

    largescale_error_ratio = 1, # Ratio to reduce BM basis background error over lmeso wavelenghts

    only_largescale = False # Flag to prescribe only BM basis background error over lmeso wavelenghts
 
)

# Multi-scale Optimal Interpolation 
INV_MIOST = dict(

    window_size = timedelta(days=15),

    window_output = timedelta(days=15),

    window_overlap = True,

    dir = None,

    obs_subsampling = 1

)

#################################################################################################################################
# REDUCED BASIS
#################################################################################################################################

NAME_BASIS = None

# Balanced Motions
BASIS_BM = dict(

    name_mod_var = None, # Name of the related model variable 
    
    flux = True,

    save_wave_basis = False, # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

    wavelet_init = True, # Estimate the initial state 

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp= 3.5, # Defines the wavelet shape

    facpsp= 1.5, # factor to fix df between wavelets

    lmin= 80, # minimal wavelength (in km)

    lmax= 970., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 20, # Largest mesoscale time of decorrelation 

    sloptdec = -1.28, # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -5 # Slope such as Q = lambda^slope where lamda is the wavelength

)

# Balanced Motions with auxilliary data 
BASIS_BMaux = dict(

    name_mod_var = None, # Name of the related model variable
    
    flux = True,

    save_wave_basis = False, # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

    wavelet_init = True, # Estimate the initial state 

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp= 3.5, # Defines the wavelet shape

    facpsp= 1.5, # factor to fix df between wavelets

    lmin= 80, # minimal wavelength (in km)

    lmax= 970., # maximal wavelength (in km)

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    distortion_eq = 2.,

    lat_distortion_eq = 5.,

    distortion_eq_law = 2.,

    file_aux = None,

    filec_aux = None,

    tssr = 0.5,

    facRo = 8.,

    Romax = 150.,

    cutRo =  1.6

)

# Large scales 
BASIS_LS = dict(

    flux = True,

    name_mod_var = None, # Name of the related model variable

    wavelet_init = True,

    facnls= 3., #factor for large-scale wavelet spacing
        
    facnlt= 3.,
        
    tdec_lw= 25.,
        
    std_lw= 0.04,
        
    lambda_lw= 970,

    fcor = .5

)

# Internal Tides
BASIS_IT = dict(

    sigma_B_He = 0.2, # Background variance for He

    sigma_B_bc = 1e-2, # Background variance for bc

    facgauss = 3.5,  # factor for gaussian spacing= both space/time

    D_He = 200, # Space scale of gaussian decomposition for He (in km)

    T_He = 20, # Time scale of gaussian decomposition for He (in days)

    D_bc = 200, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

    facB_bc_coast = 1, # Factor for sigma_B_bc located at coast. Useful only if mask is provided

    facB_He_coast = 1,  # Factor for sigma_B_He located at coast. Useful only if mask is provided

    scalemodes = None, # Only for SW1LM model, 

    scalew_igws = None 

)


#################################################################################################################################
# DIAGNOSTICS
#################################################################################################################################
NAME_DIAG = None

# Observatory System Simulation Experiment 
DIAG_OSSE = dict(

    dir_output = None,

    time_min = None,

    time_max = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,

    name_ref = '',

    name_ref_time = '',

    name_ref_lon = '',

    name_ref_lat = '',

    name_ref_var = '',

    options_ref =  None,

    name_exp_var = ''

)




