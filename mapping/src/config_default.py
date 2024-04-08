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

    path_obs = None, # if set to None, observations are saved in *tmp_DA_path*

    path_bathymetry = None, # path to read bathymetry netcdf file.   

    name_var_bathy = {'lon':'','lat':'','var':''},

    coriolis_force = True, # if set to False, coriolis force is set to 0 (for idealized case for instance)

    n_workers = 1 # number of workers to parallelize experiment preparation (like Obsop, ...)

)


#################################################################################################################################
# GRID 
#################################################################################################################################
NAME_GRID = 'GRID_GEO'

# Read grid from file
GRID_FROM_FILE = dict(

    path_init_grid = '', 

    name_init_lon = '',

    name_init_lat = '',

    subsampling = None,

)

# Regular geodetic grid
GRID_GEO = dict(

    lon_min = 294.,                                        # domain min longitude

    lon_max = 306.,                                        # domain max longitude

    lat_min = 32.,                                         # domain min latitude

    lat_max = 44.,                                         # domain max latitude

    dlon = 1/10.,                                            # zonal grid spatial step (in degree)

    dlat = 1/10.,                                            # meridional grid spatial step (in degree)

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''},

    interp_method_mask = "nearest", # mask interpolation method between "bivariate", "nearest" and "inverse_distance_weighting" 

)

# Regular cartesian grid 
GRID_CAR = dict(

    super = 'GRID_CAR',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dx = 25.,                                              # grid spacing in km

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''},

    interp_method_mask = "nearest", # mask interpolation method between "bivariate", "nearest" and "inverse_distance_weighting" 

)

# Restart from previous run 
GRID_RESTART = dict(

    name_grid = 'restart',

)


#################################################################################################################################
# OBSERVATIONS 
#################################################################################################################################
NAME_OBS = None

# L4 products (has to be on 2D latitude x longitude grids)
OBS_L4 = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {}, # dictionnary of observed variables (keys: variable types [SSH,SST etc...]; values: name of observed variables)

    name_err = {}, # dictionnary of measurement error variables (keys: variable types [SSH,SST etc...]; values: name of error variables)

    subsampling = None, # Subsampling in time (in number of model time step). Set to None for no subsampling

    sigma_noise = None  # Value of (constant) measurement error (will be used if *name_err* is not provided)

)

# Nadir altimetry
OBS_SSH_NADIR = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {'SSH':''}, # dictionnary of observed variables (keys: only SSH because altimetry; values: name of observed SSH)

    varmax = 1e2, # Maximal value of observations considered 

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    nudging_params_ssh = None, # dictionary of nudging parameters on SSH {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that 'sigma' parameter is useless now, and will be removed soon,

    delta_t = None, # Sampling period of the satellite (in s), used for computing geostrophic current 

    velocity = None # Velocity of the satellite (in m/s), used for computing geostrophic current 

)

# Swath altimetry
OBS_SSH_SWATH = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate

    name_xac = None, # name of across track coordinate (like in SWOTsimulator output files)
    
    name_var = {'SSH':''}, # dictionnary of observed variables (keys: only SSH because altimetry; values: name of observed SSH)

    varmax = 1e2, # Maximal value of observations considered 

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    nudging_params_ssh = None, # dictionary of nudging parameters on SSH {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon

    nudging_params_relvort = None, # dictionary of nudging parameters on Relative Vorticity {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon
    
    concat_dim = "num_lines" # dimension name along which to concatenate 
    
)

#################################################################################################################################
# MODELS
#################################################################################################################################
NAME_MOD = None # Either DIFF, QG1L, QG1LM, SW1L, SW1LM    

# Diffusion model
MOD_DIFF = dict(

    name_var = {'SSH':"ssh"},

    var_to_save = None,

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

    init_from_bc = False,

    dist_sponge_bc = None  # distance (in km) for which boundary fields are spatially spread close to the borders
)

# 1.5-layer Quasi-Geostrophic models
MOD_QG1L_NP = dict(

    name_var = {'SSH':"ssh"},

    init_from_bc = False,

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    name_init_var = {},

    dir_model = None,

    var_to_save = None,

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

    name_var = {'SSH':"ssh"}, # Dictionnary of variable name (need to be at least SSH, and optionaly tracer variables SST, SSS etc.)

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file 

    dir_model = None, # directory of the model (if other than mapping/models/model_qg1l)

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save

    upwind = 3, # Order of the upwind scheme for PV advection (either 1,2 or 3)

    advect_tracer = False, # Whether or not to advect tracers. If True, need to add tracer variables (e.g. SST) in *name_var*

    dtmodel = 300, # model timestep

    time_scheme = 'Euler', # Time scheme of the model (e.g. Euler,rk2,rk4)

    c0 = 2.7, # If not None, fixed value for phase velocity 

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None, # Minimum value of phase velocity to consider

    cmax = None, # Maximum value of phase velocity to consider

    init_from_bc = False, # Whether or not to initialize the model with boundary fields.

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    Kdiffus = None,

    Kdiffus_trac = None,

    bc_trac = 'OBC', # Either OBC or fixed

)

MOD_QG1L_JAX_FULL = dict(

    name_var = {'SSH':"ssh"},

    init_from_bc = False,

    name_init_var = {},

    dir_model = None,

    var_to_save = None,

    multiscale = False,

    advect_tracer = False,

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
MOD_SW1L_NP = dict(

    name_var = {'U':'u','V':'v','SSH':'ssh'},

    name_init_var = [],

    dir_model = None,

    var_to_save = None,

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*3.14/12/3600], # igw frequencies (in seconds)

    He_init = 0.9, # Mean height (in m)

    He_data = None, # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)

MOD_SW1L_JAX = dict(

    name_var = {'U':'u','V':'v','SSH':'ssh'},

    name_init_var = [],

    name_params = ['He', 'hbcx', 'hbcy', 'itg'], # list of parameters to control (among 'He', 'hbcx', 'hbcy', 'itg')

    dir_model = None,

    var_to_save = None, # Variables to save in output netcdf files 

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    bc_island = "dirichlet", # Either "dirichlet" (orthogonal velocity forced to zero) or "radiative" (dissipative boundaries)

    w_waves = [2*3.14/(12*60+25)/60], # igw frequencies (in seconds)

    He_init = 0.9, # Mean height (in m)

    He_data = None, # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)

# Linear Shallow-Water model
MOD_SW1L = dict(

    name_var = {'U':'u','V':'v','SSH':'ssh'},

    name_init_var = [],

    name_params = ['He', 'hbcx', 'hbcy'], # list of parameters to control (among 'He', 'hbcx', 'hbcy', 'itg')

    dir_model = None,

    var_to_save = None,

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*3.14/12/3600], # igw frequencies (in seconds)

    He_init = 0.9, # Mean height (in m)

    He_data = None, # He external data that will be used as apriori for the inversion. If path is None, *He_init* will be used

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81

)

# Tracer advection

MOD_TRACADV_SSH = dict(

    name_var = {'SST':'sst','SSH':'ssh'}, # Dictionnary of variable names (need to be at least one tracer e.g SST and SSH variables.)

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file 

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save

    upwind = 3, # Order of the upwind scheme for tracer advection (either 1,2 or 3)

    time_scheme = 'Euler', # Either Euler, rk2 or rk4

    dtmodel = 300, # model timestep

    init_from_bc = False, # Whether or not to initialize the model with boundary fields.

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    Kdiffus = None

)

MOD_TRACADV_VEL = dict(

    name_var = {'SST':'sst','U':'u','V':'v'}, # Dictionnary of variable name (need to be at least one tracer e.g SST and optionaly velocity variables.)

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file 

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save

    upwind = 3, # Order of the upwind scheme for tracer advection (either 1,2 or 3)

    time_scheme = 'Euler', # Either Euler, rk2 or rk4

    dtmodel = 300, # model timestep

    init_from_bc = False, # Whether or not to initialize the model with boundary fields.

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    Kdiffus = None

)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = None # For now, only BC_EXT is available

# External boundary conditions
BC_EXT = dict(

    file = None, # netcdf file(s) in which the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = None,

    name_var = {},

)


#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = None

OBSOP_INTERP_L3 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_borders = False,

)

OBSOP_INTERP_L3_GEOCUR = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_borders = False,

)

OBSOP_INTERP_L4 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    mask_borders = False,

    interp_method = 'linear' # either 'nearest', 'linear', 'cubic' (use only 'cubic' when data is full of non-NaN)

)

#################################################################################################################################
# INVERSION METHODS
#################################################################################################################################
NAME_INV = None

# Optimal Interpolation
INV_OI = dict(

    name_var = {'SSH':'ssh'},

    Lt = 7, # days

    Lx = 1, # degreee

    Ly = 1, # degree

    sigma_R = 5e-2 # meters

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

    path_save_control_vectors = None, # Path where to save the control vector at each 4Dvar iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

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

    only_largescale = False, # Flag to prescribe only BM basis background error over lmeso wavelenghts

    anomaly_from_bc = False # Whether to perform the minimization with anomalies from boundary condition field(s)
 
)

INV_4DVAR_JAX = dict(

    compute_test = False, # TLM, ADJ & GRAD tests

    path_init_4Dvar = None, # To restart the minimization process from a specified control vector

    restart_4Dvar = False, # To restart the minimization process from the last control vector

    gtol = None, # Gradient norm must be less than gtol before successful termination.

    maxiter = 10, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    path_save_control_vectors = None, # Path where to save the control vector at each 4Dvar iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

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

    only_largescale = False, # Flag to prescribe only BM basis background error over lmeso wavelenghts

    anomaly_from_bc = False # Whether to perform the minimization with anomalies from boundary condition field(s)
 
)

INV_4DVAR_PARALLEL = dict(

    nprocs = 1,

    overlap_frac = .5,

    window_size_proc = timedelta(days=30),

    compute_test = False, # TLM, ADJ & GRAD tests

    path_init_4Dvar = None, # To restart the minimization process from a specified control vector

    restart_4Dvar = False, # To restart the minimization process from the last control vector

    gtol = None, # Gradient norm must be less than gtol before successful termination.

    maxiter = 10, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    path_save_control_vectors = None, # Path where to save the control vector at each 4Dvar iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

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

    only_largescale = False, # Flag to prescribe only BM basis background error over lmeso wavelenghts

    anomaly_from_bc = False # Whether to perform the minimization with anomalies from boundary condition field(s)
 
)

# Multi-scale Optimal Interpolation (Ubelmann et al. 2021) 
INV_MOI = dict(

    dir = None, # Directory of .py scripts

    name_var = False,

    path_mdt = None, # path of Mean Dynamic Topography (MDT) netcdf file.  

    name_var_mdt = {'lon':'','lat':'','mdt':''}, # name of coordinates and variable of the MDT file
    
    window_size = timedelta(days=15),

    window_output = timedelta(days=15),

    window_overlap = True,

    sigma_R = 1e-2, 

    set_geo3ss6d = True, # Estimate small scales balanced motion component

    set_geo3ls = True, # Estimate large scales balanced motion component

    lmin= 80, # minimal wavelength (in km)

    lmax= 970., # maximal wavelength (in km)

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    file_aux = None,

    filec_aux = None,

)

#################################################################################################################################
# REDUCED BASIS
#################################################################################################################################

NAME_BASIS = None

# Balanced Motions
BASIS_BM = dict(

    name_mod_var = None, # Name of the related model variable 
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 20, # Largest mesoscale time of decorrelation 

    sloptdec = -1.28, # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -5, # Slope such as Q = lambda^slope where lamda is the wavelength,

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)

BASIS_GEOCUR = dict(

    name_mod_u = None, # Name of the related model variable 

    name_mod_v = None, # Name of the related model variable 
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 20, # Largest mesoscale time of decorrelation 

    sloptdec = -1.28, # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -5, # Slope such as Q = lambda^slope where lamda is the wavelength,

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)

BASIS_GAUSS3D = dict(

    name_mod_var = '', # Name of the related model variable 

    flux = False,

    facns = 1., # Factor for gaussian spacing in space

    facnlt = 2., # Factor for gaussian spacing in time

    sigma_D = 300, # Spatial scale (km)

    sigma_T = 20, # Time scale (days)

    sigma_Q = 0.01, # Standard deviation for matrix Q 

)

BASIS_BM_JAX = dict(

    name_mod_var = None, # Name of the related model variable 

    wavelet_init = True, # Estimate the initial state 
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    save_wave_basis = 'inline', # 'inline' for saving in RAM, 'offline' for saving in tmp_DA_path, False for computing basis component at each time

    facns = 1., #factor for wavelet spacing in space

    facnlt = 2., #factor for wavelet spacing in time

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

    slopQ = -5, # Slope such as Q = lambda^slope where lamda is the wavelength,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)

# Constant basis 
BASIS_CONSTANT = dict(

    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    facnlt = 2., #factor for wavelet spacing in time

    name_mod_var = None, # Name of the related model variable 

    tdec = 20 # Decorrelation time of the constant wavelet 

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

    cutRo =  1.6,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

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

    fcor = .5,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector
)

# Internal Tides
BASIS_IT = dict(

    ### COMMON PARAMETER ###

    scalemodes = None, # Only for SW1LM model, 

    scalew_igws = None,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None, # name of the variable of the basis vector 

    facgauss = 3.5,  # factor for gaussian spacing= both space/time

    ### - HBC PARAMETER ### 

    Nwaves = 1, # number of wave component 

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    sigma_B_bc = 1e-2, # Background variance for bc

    D_bc = 200, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

    facB_bc_coast = 1, # Factor for sigma_B_bc located at coast. Useful only if mask is provided

    facB_He_coast = 1,  # Factor for sigma_B_He located at coast. Useful only if mask is provided

    ### - ITG PARAMETER - ### 

    sigma_B_itg = 1e-2, # Background variance for itg

    sigma_B_itg_bathy_modulated = False, # True if sigma_B_itg is modulated by bathymetry gradient 

    bathymetry_gradient_smooth = False, # True if bathymetry graident needs to be smoothened by a gaussian kernel 

    itg_time_dependant = False, # True if internal tide generation parameter changes in time  

    reduced_basis_itg = False, # True if internal tide generation parameter needs to be decomposed on a reduced basis 

    itg_bathymetry_selected = False, # True if itg sources are located on highest bathymetry gradients

    itg_single_bathy = False, # for twin exp 

    bathymetry_gradient_threshold = None, # the percentage of highest gradient pixels to select for itg parameter 

    D_itg = 100, # Space scale of gaussian decomposition for internal tide generation (in km), if None any decomposition basis is created

    T_itg = 20, # Time scale of gaussian decomposition for internal tide generation (in days)

    # - Anisotropic ITG - #

    anisotropic_itg = False,

    ### - HE PARAMETER - ### 

    sigma_B_He = 0.2, # Background variance for He

    He_time_dependant = True, # True if equivalent height parameter changes in time  

    He_uniform = True, # True if He is uniform over space AND time, useful for twin experiment 

    D_He = 200, # Space scale of gaussian decomposition for He (in km)

    T_He = 20, # Time scale of gaussian decomposition for He (in days)

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

    options_ref =  {},

    name_exp_var = '',

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var = None,

    name_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

DIAG_OSSE_UV = dict(

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

    name_ref_var_u = '',

    name_ref_var_v = '',

    options_ref =  {},

    name_exp_var_ssh = '',

    name_exp_var_u = None,

    name_exp_var_v = None,

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var_ssh = None,

    name_bas_var_u = None,

    name_bas_var_v = None,

    name_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''},

)

# Observatory System Experiment (e.g. validation with real data)
DIAG_OSE = dict(

    dir_output = None,

    time_min = None,

    time_max = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,

    bin_lon_step = 1,

    bin_lat_step = 1,

    bin_time_step = '1D',

    name_ref = '',

    name_ref_time = '',

    name_ref_lon = '',

    name_ref_lat = '',

    name_ref_var = '',

    options_ref =  {},

    add_mdt_to_ref = False,

    path_mdt = None,

    name_var_mdt = None,
    
    delta_t_ref = 0.9434, # s

    velocity_ref = 6.77, # km/s

    lenght_scale = 1000, # km

    nb_min_obs = 10,

    name_exp_var = '',

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var = None

)




