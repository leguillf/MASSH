#
import os
import grid
import comp_geo3 
import comp_iw 
import comp_instr
import obs 

OBS_DIR = os.path.dirname(__file__)
SAD_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.dirname(__file__)

config=dict(
    
RUN_NAME = '', # Set automatically with filename
PATH = dict(OUTPUT= OUTPUTS_DIR),

ALGO = dict(
    USE_MPI= False,
    store_gtranspose= False, # only if USE_MPI
    INV_METHOD= 'PCG_INV',
    NITER= 800  , # Maximum number of iterations in the variational loop
    EPSPILLON_REST= 1.e-7,
    gsize_max = 5000000000 ,
    float_type= 'f8',
    int_type= 'i8'),

GRID = grid.Grid_msit(
    TEMPLATE_FILE= SAD_DIR+'/mdt13_interpolated.nc',
    LON_NAME= 'lon',
    LAT_NAME= 'lat',
    MDT_NAME= 'MDT',
    FLAG_MDT= False,
    DATE_MIN= '2012-7-1',
    DATE_MAX= '2012-7-20',
    TIME_STEP= 1.,
    NSTEPS_NC= 1,
    TIME_STEP_LF= 10000., # For internal tides with seasons
    LON_MIN= 230.,
    LON_MAX= 240.,
    LAT_MIN= 30.,
    LAT_MAX= 40.,
    tref_iw= 15340.),

PHYS_COMP=[

    comp_geo3.Comp_geo3ss6d(
        facns= 1., #factor for wavelet spacing= space
        facnlt= 2.,
        npsp= 3.5, # Defines the wavelet shape
        facpsp= 1.5, #1.5 # factor to fix df between wavelets
        lmin= 80 ,
        lmax= 970.,
        cutRo= 1.6,
        factdec= 15.,
        tdecmin= 2.5,
        tdecmax= 40.,
        tssr= 0.5,
        facRo= 8.,
        Romax= 150.,
        facQ= 1.,
        depth1= 0.,
        depth2= 30.,
        distortion_eq= 2.,
        lat_distortion_eq= 5.,
        distortion_eq_law= 2.,
        file_aux= SAD_DIR+'/aux_data_global_v3.nc',
        filec_aux= SAD_DIR+'/Rossby_radius.nc',
        write= True,
        Hvarname= 'Hss'),


    comp_geo3.Comp_geo3ls(
        facnls= 3., #factor for large-scale wavelet spacing
        facnlt= 3.,
        tdec_lw= 25.,
        std_lw= 0.04,
        lambda_lw= 970, #768.05127036
        file_aux= SAD_DIR+'/aux_data_global_v3.nc',
        filec_aux= SAD_DIR+'/Rossby_radius.nc',
        write= True,
        Hvarname= 'Hls'),

    ],

OBS_COMP=[

    ],


OBS=[

    obs.fullSSH(
        name= 'fullSSH',
        dirname= OBS_DIR,
        root1= '',
        name_time='time',
        name_lon='lon',
        name_lat='lat',
        name_ssh='ssh',
        subsampling=1
        ),
    
        ]

)


