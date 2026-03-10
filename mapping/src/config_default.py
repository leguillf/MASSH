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

    lon_obs_min = None,

    lon_obs_max = None,

    lat_obs_min = None,

    lat_obs_max = None,

    write_obs = False, # save observation dictionary in *path_obs*

    compute_obs = False, # force computing observations 

    path_obs = None # if set to None, observations are saved in *tmp_DA_path*

)


#################################################################################################################################
# GRID 
#################################################################################################################################
NAME_GRID = 'GRID_GEO'

# Read grid from file
GRID_FROM_FILE = dict(

    path_init_grid = '', 

    name_init_lon = 'lon',

    name_init_lat = 'lat',

    subsampling = None,

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''},

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

    name_var_mask = {'lon':'','lat':'','var':''}

)

# Regular cartesian grid 
GRID_CAR = dict(

    super = 'GRID_CAR',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dx = 25.,                                              # grid spacing in km

    nx = None,                                             # If not None, use nx to compute dx 

    ny = None,                                             #

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

GRID_CAR_CENTER = dict(

    super = 'GRID_CAR',

    lon_center = 295.,                                        # Center of the grid (degrees)                                   # domain max longitude

    lat_center = 33.,                                         # Center of the grid (degrees)

    spacing_km = 43.,                                         # Desired spacing between points (km)

    shape = [128,128],                                        # number of points in lat and lon

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

# L4 products (has to be on 2D latitude x longitude grids)
OBS_L4 = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {}, # dictionnary of observed variables (keys: variable types [SSH,SST etc...]; values: name of observed variables)

    name_err = {}, # dictionnary of measurement error variables (keys: variable types [SSH,SST etc...]; values: name of error variables)

    subsampling = None, # Subsampling in time (in number of model time step). Set to None for no subsampling

    sigma_noise = None,  # Value of (constant) measurement error (will be used if *name_err* is not provided)

    offset = None, # Value to add to observations

)

# Nadir altimetry
OBS_SSH_NADIR = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {'SSH':''}, # dictionnary of observed variables (keys: only SSH because altimetry; values: name of observed SSH, can be a lost of variables to combine, see *combine_var* parameter below)

    combine_var = None, # If not None, dictionnary of variable to combine to get the observed variable (keys: same as name_var; values: list of -1 or +1 to indicate how to combine variables in name_var, e.g. {'SSH':[-1,1]} to compute SSH as the difference between the second and the first variable in name_var['SSH'] list)
    
    synthetic_noise = None, # Std of synthetic noise (std in meters) to artificially add to the data

    varmax = 1e2, # Maximal value of observations considered 

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    path_err = None, # path of error file 

    name_var_err = None, # dictionary of error coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
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
    
    subsampling = None,
    
    synthetic_noise = None, # Std of synthetic noise (std in meters) to artificially add to the data

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    path_err = None, # path of error file 

    name_var_err = None, # dictionary of error coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    nudging_params_ssh = None, # dictionary of nudging parameters on SSH {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon

    nudging_params_relvort = None, # dictionary of nudging parameters on Relative Vorticity {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon
    
)

#################################################################################################################################
# MODELS
#################################################################################################################################
NAME_MOD = None # Either DIFF, QG1L, QG1LM, SW1L, SW1LM    

# Diffusion model
MOD_Id = dict(

    name_var = {'SSH':"ssh"},

    var_to_save = None,

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

    SIC_mod = False, # flag to activate variable limits [0,100] (i.e., for sea ice concentration)

    init_from_bc = False,

    dist_sponge_bc = None  # distance (in km) for which boundary fields are spatially spread close to the borders
)

# Diffusion model
MOD_DIFF = dict(

    name_var = {'SSH':"ssh"},

    var_to_save = None,

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

    SIC_mod = False, # flag to activate variable limits [0,100] (i.e., for sea ice concentration)

    init_from_bc = False,

    dist_sponge_bc = None  # distance (in km) for which boundary fields are spatially spread close to the borders
)

MOD_DIFF_JAX = dict(

    name_var = {'SSH':"ssh"},

    var_to_save = None,

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

    init_from_bc = False,

    dist_sponge_bc = None  # distance (in km) for which boundary fields are spatially spread close to the borders
)

# 1.5-layer Quasi-Geostrophic models
MOD_QG1L_JAX = dict(

    name_class = 'Qgm', # Name of the model class in jqgm.py

    name_var = {'SSH':"ssh"}, # Dictionnary of variable name (need to be at least SSH, and optionaly tracer variables SST, SSS etc. and/or ageostrophic velocities U, V)

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file 

    dir_model = None, # directory of the model (if other than mapping/models/model_qg1l)

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save

    save_diagnosed_variables = False, # Whether to save diagnosed variables (e.g. SSH, geostrophic velocies and cyclogeostrophic velocities) in the output netcdf files

    save_params = False, # Whether to save control parameters (e.g. corrective fluxes) in the output netcdf files

    upwind = 3, # Order of the upwind scheme for PV advection (either 1,2 or 3) 

    advect_pv = True, # Whether or not to advect PV. 

    advect_tracer = False, # Whether or not to advect tracers. If True, need to add tracer variables (e.g. SST) in *name_var*

    dtmodel = 1200, # model timestep

    cfl = None, # If not None, dtmodel is set such as dtmodel=cfl*dx/c

    time_scheme = 'Euler', # Time scheme of the model (e.g. Euler,rk2,rk4)

    c0 = 2.7, # If not None, fixed value for phase velocity 

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None, # Minimum value of phase velocity to consider

    cmax = None, # Maximum value of phase velocity to consider

    file_bathy_aux = None, # Name of netcdf file for ocean bathymetry field. If prescribed, bathymetry will be taken into account in the model

    name_var_bathy = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of bathymetry netcdf file

    bathy_ratio_max = None, # Maximum value of bathymetry-related PV term

    solver = 'spectral', # Solver for Elliptical Equation inversion (either spectral or cg - for Conjugate Gradient)

    init_from_bc = False, # Whether or not to initialize the model with boundary fields.

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    Kdiffus = None,

    Kdiffus_trac = None,

    bc_trac = 'OBC', # Either OBC or fixed

    forcing_tracer_from_bc = False, # Whether to use BC fields to force tracer advection,

    constant_c = True,

    constant_f = True,

    f0 = None,

    tile_size = 32, # Only for name_class=='QgmWithTiles'
            
    tile_overlap = 16,  # Only for name_class=='QgmWithTiles'

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

)

# 1.5-layer Shallow-Water model
MOD_CSW1L = dict(

    name_var = {'U':'u','V':'v','SSH':'ssh'},

    name_init_var = {},

    dir_model = None,

    var_to_save = None,

    name_params = ['He_mean', 'He_ano', 'hbc'], # list of parameters to control (among 'He_mean', 'He_ano', 'hbc')

    dtmodel = 300, # model timestep

    force_constant_km_grid = False, # Whether to force constant km grid spacing (True) or use the grid spacing from the grid (False)

    dx_km = None, # If force_constant_km_grid is True, zonal grid spacing (in km)

    dy_km = None, # If force_constant_km_grid is True, meridional grid spacing (in km)

    path_mdt = None, # If provided, QGPV will be expressed thanks to the Reynolds decompositon

    name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''},

    filec_aux = None, # auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    filef_aux = None, # auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_f = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    c0 = 2.7, # If filec_aux is None, fixed value for phase velocity (m/s)

    H = 4e3, # Mean depth (in m)

    cfl = None, # If not None, dtmodel is set such as dtmodel=cfl*dx/sqrt(gHe)

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*3.14/12/3600], # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81,

    flag_coupling_from_bm = False, # Whether to compute He corrections from the balanced motion field

    path_vertical_modes = None, # Path of the vertical modes netcdf file

    path_bm = None, # Path of the balanced motion netcdf file

    name_var_bm = {'time':'','lon':'','lat':'','ssh_bm':''},

    obc_north = True,

    obc_west = True,

    obc_south = True,

    obc_east = True,

    periodic_x = False,

    periodic_y = False,

    flag_bc_sponge = False,

    dist_sponge_bc = None,

    sponge_coef = 0.05,

)

# QG-SW Models

MOD_QGSW = dict(

    name_class = 'qg', # Name of the model class (either qg or sw)

    name_var = {'U':'u', 'V':'v', 'SSH':'ssh'},

    name_init_var = {}, 

    var_to_save = None,

    name_params = None,#['H', 'hbcx', 'hbcy', 'itg'], # list of parameters to control (among 'H', 'hbc', 'hbcy', 'itg')

    nl = 1, # number of layers in the model (for nl>1, set H and g_prime as lists/arrays)

    dtmodel = 1200, # model timestep

    f0 = None, # Coriolis parameter (in s^-1). If None, f0 will be computed from the grid

    c0 = 2.7,

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None, # Minimum value of phase velocity to consider

    cmax = None, # Maximum value of phase velocity to consider

    H = None, # mean layer depth(s) in meters.  Scalar or list, e.g. H=[500., 2500.] for nl=2

    g_prime = None, # reduced gravity(ies).  Scalar or list, e.g. g_prime=[9.81, 0.02] for nl=2

    init_from_bc = True,

    cfl = .25,

    bottom_drag_coef = 0.,

    slip_coef = 0., # slip coefficient for the bottom drag (in m/s)

    taux = 0., # wind stress in N/m^2

    tauy = 0., # wind stress in N/m^2

    path_mdt = None, # path of MDT

    name_var_mdt = {'lon':'','lat':'','var':''}, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    dist_sponge_bc = None,

    sponge_coef = 0.,

    visc_coef = 0., # viscosity coefficient

    path_wind = None, # path to NetCDF wind file containing u10/v10 (if None, no wind forcing)

    name_var_wind = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time',
                     'u10': 'u10', 'v10': 'v10'}, # variable names in the wind NetCDF file

    rho_air = 1.25, # air density (kg/m³) used in the bulk wind-stress formula

    Cd_wind = 1.5e-3, # drag coefficient used in the bulk wind-stress formula tau = rho_air * Cd * |U10| * U10

    rho_water = 1025.0, # ocean water density (kg/m³) used to convert wind stress [Pa] to acceleration [m²/s²]: tau/(rho_water*H)*dx

    # Physical layer depth (m) for the wind-stress denominator:  tau / (rho_water * h_wind) * dx
    # IMPORTANT for 1-layer QG/SW models: the model equivalent depth H = c²/g ≈ 0.4–1 m is
    # NOT the physical mixed-layer depth (~50–200 m).  Without setting h_wind, wind forcing
    # is 100–500× too large.  Set h_wind to the actual mixed-layer depth, e.g.:
    #   h_wind = 100.     # 100 m mixed layer
    # Leave None to use the model's reference layer thickness (correct only for multi-layer
    # models where H represent the true physical layer depths).
    h_wind = None,

    wind_timestep = 3600, # wind update interval in seconds (default: 1 hour). Wind stress is
                          # precomputed at this cadence and held constant between updates.
                          # Reduces memory when the model timestep is very small.

    max_nstep = 240, # maximum number of model steps per JIT call. Large nstep values are
                     # split into chunks of max_nstep to limit GPU memory usage.
                     # Decrease if running out of GPU memory.

    # Momentum forcing mode for external forcing (Fu, Fv, Fh).
    # 'direct'          : use Fu, Fv as provided (default).
    # 'mass_consistent' : derive Fu, Fv from Fh so that velocity is conserved
    #                     when mass is added:  Fu = -u/h * Fh,  Fv = -v/h * Fh.
    forcing_momentum = 'direct',

)

MOD_BMIT = dict(

    # Common parameters for BM and IT components

    name_var = {'U_IT':'u_it','V_IT':'v_it','SSH_IT':'ssh_it', 'SSH_BM':'ssh_bm', 'SSH':'ssh'},

    name_init_var = [],

    dir_model = None,

    var_to_save = None,

    dtmodel = 300, # model timestep

    filec_aux = None, # auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    c0 = 2.7, # If filec_aux is None, fixed value for phase velocity (m/s)

    cfl = None, # If not None, dtmodel is set such as dtmodel=cfl*dx/sqrt(gHe)

    init_from_bc = False,

    # BM parameters

    time_scheme_bm = 'Euler', # Time scheme of the model (e.g. Euler,rk2,rk4)

    Kdiffus = 0, # Coefficient of diffusion for the BM component

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    # IT parameters

    time_scheme_it = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    name_params_it = ['He', 'hbc'], # list of parameters to control (among 'He', 'hbc')

    H = 4e3, # Mean depth (in m)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*3.14/12/3600], # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81,

    flag_coupling_from_bm = False, # Whether to compute He corrections from the balanced motion field

    path_vertical_modes = None, # Path of the vertical modes netcdf file

    obc_north = True,

    obc_west = True,

    obc_south = True,

    obc_east = True,

    periodic_x = False,

    periodic_y = False,

    flag_bc_sponge = False,

    dist_sponge_bc = None,

    sponge_coef = 0.05,

)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = None # For now, only BC_EXT is available

# External boundary conditions
BC_EXT = dict(

    file = None, # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = None,

    name_var = {},

    c_grid = False, # whether the grid is a C-grid (True) or A-grid (False)

)


#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = None

OBSOP_INTERP_L3 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    name_var = 'SSH',

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_borders = False,

)

OBSOP_INTERP_L3_JAX = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    name_var = 'SSH',

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_borders = False,

)

OBSOP_INTERP_L4 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    name_var = 'SSH',

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    mask_borders = False,

    interp_method = 'linear', # either 'nearest', 'linear', 'cubic' (use only 'cubic' when data is full of non-NaN)

    gradients = False

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

    flag_full_jax = False,
    
    compute_test = False, # TLM, ADJ & GRAD tests

    freq_it_plot = 10, # Frequency of iteration to plot the cost function and its gradient  

    JAX_mem_fraction = None,

    path_init_4Dvar = None, # To restart the minimization process from a specified control vector

    restart_4Dvar = False, # To restart the minimization process from the last control vector

    ftol = None, # The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.

    gtol = None, # Gradient norm must be less than gtol*g0 (g0 being the gradient at first iteration) before successful termination.

    maxiter = 10, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    path_save_control_vectors = None, # Path where to save the control vector at each 4Dvar iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

    sigma_B = None,

    prec = False, # preconditoning

    path_background = None, # Path of a control vector from another experiment to use as the background 

    anomaly_from_bc = False # Whether to perform the minimization with anomalies from boundary condition field(s)
 
)

#################################################################################################################################
# REDUCED BASIS
#################################################################################################################################

NAME_BASIS = None

# Balanced Motions 
BASIS_BM = dict(

    name_mod_var = None, # Name of the related model variable 

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)
    
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

    facQ = 1, # factor to be multiplied to the estimated Q

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -5, # Slope such as Q = lambda^slope where lamda is the wavelength,

    norm_time = True, # Whether to normalize the time component of the basis vectors (set True for dynamical forcings)

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)
 
BASIS_BM_JAX = dict(

    name_mod_var = 'ssh', # Name of the related model variable 

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)

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

    facQ = 1, # factor to be multiplied to the estimated Q

    Qmax = 1e-3, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -5, # Slope such as Q = lambda^slope where lamda is the wavelength,

    norm_time = True, # Whether to normalize the time component of the basis vectors (set True for dynamical forcings)

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)

BASIS_GAUSSV2 = dict( 

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

    facQ = 1, # factor to be multiplied to the estimated Q

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

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)

    flux = False,

    facns = 2., # Factor for gaussian spacing in space

    facnlt = 1., # Factor for gaussian spacing in time

    sigma_D = 300, # Spatial scale (km)

    sigma_T = 20, # Time scale (days)

    sigma_Q = 0.01, # Standard deviation for matrix Q 

    fcor = .5,

    normalize_fact = True,

    time_spinup = None, # days

    flag_variable_Q = False,

    path_sad = None,

    name_var_sad = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)

BASIS_GAUSS3D_JAX = dict(

    name_mod_var = '', # Name of the related model variable 

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)

    flux = False,

    facns = 2., # Factor for gaussian spacing in space

    facnlt = 1., # Factor for gaussian spacing in time

    sigma_D = 300, # Spatial scale (km)

    sigma_T = 20, # Time scale (days)

    sigma_Q = 0.01, # Standard deviation for matrix Q 

    fcor = .5,

    normalize_fact = True,

    time_spinup = None, # days

    flag_variable_Q = False,

    path_sad = None,

    name_var_sad = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

) 

# Wavelet 3D
BASIS_WAVELET3D = dict(

    name_mod_var = None, # Name of the related model variable 

    flux = False,

    facnst = 1., #factor for wavelet spacing in space and time 

    npsp = 3.5, # Defines the wavelet shape, both in space and time 

    facpsp = 1.5, # factor to fix df between wavelets, both in space and time 

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    tmin = 2, # minimum time of decorrelation 

    tmax = 20., # maximum time of decorrelation 

    sigma_Q = 1e-1, # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

)
 

# Balanced Motions with auxilliary data 
BASIS_BMaux = dict(

    name_mod_var = None, # Name of the related model variable 

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space 

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    file_aux = '', # Name of auxilliary file in which are stored the std and tdec for each locations at different wavelengths.

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ = 1, # factor to be multiplied to the estimated Q

    l_largescale = 500, # factor to be multiplied to the estimated Q

    facQ_largescale = 1, # factor to be multiplied to the estimated Q

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None, # name of the variable of the basis vector

    norm_time = True,

    file_facQaux = None,

    name_var_facQaux = {'lon':'', 'lat':'', 'var':''}

)

BASIS_BMaux_JAX = dict(

    name_mod_var = None, # Name of the related model variable 

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space 

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    file_aux = '', # Name of auxilliary file in which are stored the std and tdec for each locations at different wavelengths.

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ = 1, # factor to be multiplied to the estimated Q

    facQ_aux_path = None,

    l_largescale = 500, # factor to be multiplied to the estimated Q

    facQ_largescale = 1, # factor to be multiplied to the estimated Q

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None, # name of the variable of the basis vector

    norm_time = True,

    file_facQaux = None,

    name_var_facQaux = {'wavenumber':'', 'lon':'', 'lat':'', 'var':''}

)


# Internal Tides

BASIS_HBC_JAX = dict(

    name_params = ['hbcx', 'hbcy'], # list of parameters to control (among 'He', 'hbcx', 'hbcy', 'itg')

    ### COMMON PARAMETER ### 

    # facgauss = 3.5,  # factor for gaussian spacing= both space/time

    facns = 3.5, # factor for gaussian spacing in space

    facnlt = 2.5, # factor for gaussian spacing in time 

    time_dependant = True, # True if gaussian basis is time dependant

    ### - HBC PARAMETER ### 

    sigma_B_bc = 1e-2, # Background variance for bc

    D_bc = 200, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

    Nwaves = 1, # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

)

BASIS_HBC_CST_JAX = dict(

    sigma_B_bc = 1e-2, # Background variance for bc

    Nwaves = 1, # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

)

BASIS_OFFSET = dict(

    name_mod_var = None,

    sigma_B = None, 

)

BASIS_OFFSET_JAX = dict(

    name_mod_var = None,

    sigma_B = None, 

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

    time_step = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,
    
    path_images2mp4 = None,

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
    
    delta_t_ref = None, # s

    velocity_ref = None, # km/s

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




