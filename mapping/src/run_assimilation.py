import os
import sys
import glob
import pickle
import numpy as np
import multiprocessing as mp
from scipy.interpolate import griddata, LinearNDInterpolator, RegularGridInterpolator
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from datetime import timedelta
from functools import partial
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon
import xarray as xr

from . import exp, grid, state, mod, inv, diag
from .tools import gaspari_cohn

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def prepare_process(config, config_eq, State, 
                    init_date, final_date,
                    grid_type=None, grid_type_eq=None,
                    nx_proc=None, ny_proc=None, dx=None, dy=None,
                    dlon=None, dlat=None,
                    time_window_size_proc=None, space_window_size_proc_x=None, space_window_size_proc_y=None, 
                    space_window_size_proc_x_eq=None, space_window_size_proc_y_eq=None,
                    nx_proc_eq=None, ny_proc_eq=None,
                    time_overlap=5, space_overlap_x=2, space_overlap_y=2,
                    flag_init_from_previous=True, flag_init=False, flag_background=False,
                    flag_assim=True, flag_assim_restart=False,
                    name_exp_init=None, name_exp_background=None,
                    gpu_devices=['0'],
                    dir_save_pickle=None):
    """
    Prepare subprocesses for assimilation in subwindows in time and space.
    The subprocesses can then be run in parallel using multiprocessing. 
    The outputs of the subprocesses can then be merged using merge_output_date().

    Parameters
    ----------
    config : Config
        Main configuration object (contains GRID, EXP, MOD, INV, etc.).
    config_eq : Config
        Configuration used for subwindows that cross the equator (lat0 < 0 and lat1 > 0).
    State : State
        Global state object (used for grid info and output file naming).
    init_date : datetime
        Start date of the full assimilation period.
    final_date : datetime
        End date of the full assimilation period.
    grid_type : str, optional
        Grid type for subwindows: 'GRID_CAR' or 'GRID_GEO'.
        If None, read from config.GRID.super (default: 'GRID_CAR').
    grid_type_eq : str, optional
        Grid type for equatorial subwindows: 'GRID_CAR' or 'GRID_GEO'.
        If None, uses grid_type.
    nx_proc : int, optional
        Number of grid points in x for each subwindow (GRID_CAR only).
        If None and grid_type is 'GRID_CAR', defaults to 128.
    ny_proc : int, optional
        Number of grid points in y for each subwindow (GRID_CAR only).
        If None and grid_type is 'GRID_CAR', defaults to 128.
    dx : float, optional
        Grid spacing in x in km for each subwindow (GRID_CAR only).
        If None and grid_type is 'GRID_CAR', defaults to 10 km.
    dy : float, optional
        Grid spacing in y in km for each subwindow (GRID_CAR only).
        If None and grid_type is 'GRID_CAR', defaults to 10 km.
    dlon : float, optional
        Grid spacing in longitude in degrees for each subwindow (GRID_GEO only).
        If None and grid_type is 'GRID_GEO', read from config.GRID.dlon.
    dlat : float, optional
        Grid spacing in latitude in degrees for each subwindow (GRID_GEO only).
        If None and grid_type is 'GRID_GEO', read from config.GRID.dlat.
    time_window_size_proc : float, optional
        Size of each temporal subwindow in days. If None, the full time period is used as a single window.
    space_window_size_proc_x : float, optional
        Size of each spatial subwindow in the x/longitude direction (degrees for GRID_GEO, degrees for GRID_CAR).
        If None, the full longitude range is used.
    space_window_size_proc_y : float, optional
        Size of each spatial subwindow in the y/latitude direction (degrees).
        If None, the full latitude range is used.
    space_window_size_proc_x_eq : float, optional
        Size of equatorial subwindows in x/longitude direction (degrees).
        If None, uses space_window_size_proc_x.
    space_window_size_proc_y_eq : float, optional
        Size of equatorial subwindows in y/latitude direction (degrees).
        The equatorial tile is always centered at latitude 0, spanning
        [-space_window_size_proc_y_eq/2, +space_window_size_proc_y_eq/2].
        Other tiles are positioned outward from the equatorial tile edges.
        If None, no special equatorial handling (standard tiling from lat_min).
    nx_proc_eq : int, optional
        Number of grid points in x for equatorial subwindows (GRID_CAR only).
        If None, uses nx_proc.
    ny_proc_eq : int, optional
        Number of grid points in y for equatorial subwindows (GRID_CAR only).
        If None, uses ny_proc.
    time_overlap : float, optional
        Overlap between consecutive time windows in days (default: 5).
    space_overlap_x : float, optional
        Overlap between consecutive space windows in the x/longitude direction in degrees (default: 2).
    space_overlap_y : float, optional
        Overlap between consecutive space windows in the y/latitude direction in degrees (default: 2).
    flag_init_from_previous : bool, optional
        If True, initialize each time window from the output of the previous one (default: True).
    flag_init : bool, optional
        If True, initialize the control vector from a previous experiment given by name_exp_init (default: False).
    flag_background : bool, optional
        If True, use a background field from another experiment given by name_exp_background (default: False).
    flag_assim : bool, optional
        If True, create and launch assimilation subprocesses (default: True).
    flag_assim_restart : bool, optional
        If True, re-run assimilation even if a converged control vector already exists (default: False).
    name_exp_init : str, optional
        Name of a previous experiment to initialize control vectors from (used when flag_init is True).
    name_exp_background : str, optional
        Name of a previous experiment to use as background (used when flag_background is True).
    gpu_devices : list of str, optional
        List of GPU device IDs to distribute subprocesses across (default: ['0']).
    dir_save_pickle : str, optional
        If provided, save pickle files (config and state) for each subwindow
        to this directory. The directory structure will mirror the subwindow layout.
        If None, no pickle files are saved (default: None).

    Returns
    -------
    list_processes : list of list of Process
        Assimilation subprocesses grouped by time window.
    list_config : list of list of Config
        Subwindow configurations grouped by time window.
    list_State : list of list of State
        Subwindow states grouped by time window.
    list_date_start : list of datetime
        Start date of each time window.
    list_date_end : list of datetime
        End date of each time window.
    list_date_middle : list of datetime
        Middle date of each time window.
    list_lonlat : list of tuple
        Center (lon, lat) of each spatial subwindow (from the first time window).
    weights_space : list of 2D arrays
        Weight maps for each subwindow, interpolated onto the target grid.
    weights_space_sum : 2D array
        Sum of all weight maps (for normalization).
    interpolators : list of callable
        Precomputed interpolation operators mapping each subwindow grid to the target grid.
    """

    # Split full experimental time window in sub windows
    list_processes = []
    list_config = []
    list_State = []
    list_date_start = []
    list_date_end = []
    list_date_middle = []
    list_lonlat = []
    iproc = 0
    n_wt = 0 

    id_gpu = 0

    # Determine grid type
    if grid_type is None:
        grid_type = getattr(config.GRID, 'super', 'GRID_CAR')
    if grid_type_eq is None:
        grid_type_eq = grid_type
    if grid_type == 'GRID_GEO':
        if dlon is None:
            dlon = config.GRID.dlon
        if dlat is None:
            dlat = config.GRID.dlat
    elif grid_type == 'GRID_CAR':
        if nx_proc is None:
            nx_proc = 128
        if ny_proc is None:
            ny_proc = 128
        if dx is None:
            dx = 10
        if dy is None:
            dy = 10
    # Also resolve dlon/dlat from config_eq when grid_type_eq is GRID_GEO
    if grid_type_eq == 'GRID_GEO' and dlon is None:
        dlon = config_eq.GRID.dlon
    if grid_type_eq == 'GRID_GEO' and dlat is None:
        dlat = config_eq.GRID.dlat

    # Pre-compute latitude bands: (lat0, lat1, _ny, _nx_proc_band, _space_x_band, is_eq)
    lat_bands = []
    if space_window_size_proc_y is not None:
        has_eq_band = (space_window_size_proc_y_eq is not None
                       and config.GRID.lat_min < 0 and config.GRID.lat_max > 0)

        if has_eq_band:
            _nx_eq = nx_proc_eq if nx_proc_eq is not None else nx_proc
            _ny_eq = ny_proc_eq if ny_proc_eq is not None else ny_proc
            _space_x_eq = space_window_size_proc_x_eq if space_window_size_proc_x_eq is not None else space_window_size_proc_x

            eq_south = -space_window_size_proc_y_eq / 2
            eq_north = space_window_size_proc_y_eq / 2
            lat_bands.append((eq_south, eq_north, _ny_eq, _nx_eq, _space_x_eq, True))

            # Southern bands: from equatorial south edge downward
            _lat1 = eq_south + space_overlap_y
            while _lat1 > config.GRID.lat_min:
                _lat0 = _lat1 - space_window_size_proc_y
                _ny_band = ny_proc
                if _lat0 < config.GRID.lat_min:
                    _lat0 = config.GRID.lat_min
                    _ny_band = max(1, int(ny_proc * (_lat1 - _lat0) / space_window_size_proc_y))
                # Skip if band is entirely within the equatorial band
                if _lat0 >= eq_south:
                    break
                lat_bands.append((_lat0, _lat1, _ny_band, nx_proc, space_window_size_proc_x, False))
                if _lat0 <= config.GRID.lat_min:
                    break
                _lat1 = _lat0 + space_overlap_y

            # Northern bands: from equatorial north edge upward
            _lat0 = eq_north - space_overlap_y
            while _lat0 < config.GRID.lat_max:
                _lat1 = _lat0 + space_window_size_proc_y
                _ny_band = ny_proc
                if _lat1 > config.GRID.lat_max:
                    _lat1 = config.GRID.lat_max
                    _ny_band = max(1, int(ny_proc * (_lat1 - _lat0) / space_window_size_proc_y))
                # Skip if band is entirely within the equatorial band
                if _lat1 <= eq_north:
                    break
                lat_bands.append((_lat0, _lat1, _ny_band, nx_proc, space_window_size_proc_x, False))
                if _lat1 >= config.GRID.lat_max:
                    break
                _lat0 = _lat1 - space_overlap_y
        else:
            # Standard: from lat_min upward
            _n = 0
            while True:
                _lat0 = config.GRID.lat_min + _n * (space_window_size_proc_y - space_overlap_y)
                if _lat0 >= config.GRID.lat_max:
                    break
                _lat1 = _lat0 + space_window_size_proc_y
                _ny_band = ny_proc
                if _lat1 > config.GRID.lat_max:
                    _lat1 = config.GRID.lat_max
                    _ny_band = max(1, int(ny_proc * (_lat1 - _lat0) / space_window_size_proc_y))
                lat_bands.append((_lat0, _lat1, _ny_band, nx_proc, space_window_size_proc_x, False))
                _n += 1
    else:
        lat_bands = [(config.GRID.lat_min, config.GRID.lat_max, ny_proc, nx_proc, space_window_size_proc_x, False)]

    lat_bands.sort(key=lambda x: x[0])

    date1 = init_date
    i = -1
    while date1<final_date:

        i += 1
        if flag_init_from_previous or i==0:
            list_processes.append([])
        list_config.append([])
        list_State.append([])

        # compute subwindow time period
        if time_window_size_proc is not None:
            time_delta = timedelta(days=time_window_size_proc)
            date0 = init_date + n_wt * (time_delta - timedelta(days=time_overlap))
            delta_t = (date0 - init_date) %  config.EXP.saveoutput_time_step
            date0 += delta_t
            date1 = min(date0 + time_delta, final_date)
            n_wt += 1
        else:
            date0 = init_date
            date1 = final_date
        list_date_start.append(date0)
        list_date_end.append(date1)
        list_date_middle.append(date0 + (date1-date0)/2)
        print(f'*** Time window: {date0} -> {date1}')

        iproc_tw = 0
        _prev_lat_band = None
        for lat0, lat1, _ny, _nx_proc_band, _space_x_band, is_eq in lat_bands:
            if lat1 < config.GRID.lat_min:
                continue
            _grid_type_config = grid_type_eq if is_eq else grid_type
            flag_avoid_next_window = False
            n_wx = 0
            lon1 = config.GRID.lon_min
            while lon1<config.GRID.lon_max and not flag_avoid_next_window:
                # compute subwindow longitude borders
                _nx = _nx_proc_band
                if _space_x_band is not None:
                    if _grid_type_config == 'GRID_GEO':
                        # For GRID_GEO, longitude spacing is uniform in degrees
                        lon0 = config.GRID.lon_min + n_wx * (_space_x_band - space_overlap_x)
                        lon1 = lon0 + _space_x_band
                        if lon0 + _space_x_band/2 > config.GRID.lon_max:
                            lon1 = lon0 + _space_x_band/2
                        n_wx += 1
                    else:
                        # For GRID_CAR, longitude spacing depends on latitude
                        if n_wx==0:
                            lon0 = config.GRID.lon_min
                        else:
                            if lat0>0:
                                lon0 = lon_prev[0,-1] - space_overlap_x
                            else:
                                lon0 = lon_prev[-1,-1] - space_overlap_x
                        lon1 = lon0 + _space_x_band
                        _nx = _nx_proc_band
                        if lon0 + _space_x_band/2 > config.GRID.lon_max:
                            lon1 = lon0 + _space_x_band/2
                            if _nx_proc_band is not None:
                                _nx = int(_nx_proc_band/2)
                        n_wx += 1
                else:
                    lon0 = config.GRID.lon_min
                    lon1 = config.GRID.lon_max
                    _nx = _nx_proc_band
                    
                # create config for the subwindow
                if is_eq or (lat0<0 and lat1>0):
                    _base_config = config_eq.copy()
                    _config = config_eq.copy()
                else:
                    _base_config = config.copy()
                    _config = config.copy()
                _config.EXP = _config.EXP.copy()
                _config.GRID = _config.GRID.copy()
                _config.MOD = _config.MOD.copy()
                _config.INV = _config.INV.copy()
                _config.EXP.init_date = date0
                _config.EXP.final_date = date1
                _config.GRID.lon_min = lon0
                _config.GRID.lon_max = lon1
                _config.GRID.lat_min = lat0
                _config.GRID.lat_max = lat1
                if _grid_type_config == 'GRID_GEO':
                    _config.GRID.super = 'GRID_GEO'
                    _config.GRID.dlon = dlon
                    _config.GRID.dlat = dlat
                else:
                    _config.GRID.super = 'GRID_CAR'
                    _config.GRID.nx = _nx
                    _config.GRID.ny = _ny
                    _config.GRID.dx = dx 
                    _config.GRID.dy = dy 
                 
                name_subwindow = f'subwindow_{str(list_date_middle[-1])[:10]}/subwindow_{round((lon1+lon0)/2)}_{round((lat1+lat0)/2)}'
                _config.EXP.tmp_DA_path += f'/{name_subwindow}'
                _config.EXP.path_save += f'/{name_subwindow}'
                if _config.INV.path_save_control_vectors is not None:
                    _config.INV.path_save_control_vectors += f'/{name_subwindow}'
                if _config.INV.path_background is not None:
                    _config.INV.path_background += f'/{name_subwindow}'

                # initialize State 
                _State = state.State(_config, verbose=0)

                if np.any(_State.lon.max()>config.GRID.lon_max):
                    if ((lat0+lat1)/2<0 and np.any(_State.lon[-1]>config.GRID.lon_max)) or ((lat0+lat1)/2>0 and np.any(_State.lon[0]>config.GRID.lon_max)):  
                        flag_avoid_next_window = True

                lon_prev = +_State.lon
                if _State.lon.min()<-180 or _State.lon.max()>180:
                    _State.lon = _State.lon % 360
                    _State.lon_min = _State.lon.min()
                    _State.lon_max = _State.lon.max()
                    _State.lon_unit = '0_360'
                
                if np.where(_State.mask)[0].size>.9*_State.mask.size:
                    continue

                # Init from file from previous window
                if n_wt>1 and flag_init_from_previous:
                    name_prev_subwindow = f'subwindow_{str(list_date_middle[-2])[:10]}/subwindow_{round((lon1+lon0)/2)}_{round((lat1+lat0)/2)}'
                    path_output = config.EXP.path_save + f'/{name_prev_subwindow}/'
                    filename = os.path.join(path_output,f'{State.name_exp_save}'\
                            f'_y{date0.year}'\
                            f'm{str(date0.month).zfill(2)}'\
                            f'd{str(date0.day).zfill(2)}'\
                            f'h{str(date0.hour).zfill(2)}'\
                            f'm{str(date0.minute).zfill(2)}.nc')
                    _config.GRID = exp.Config({'super': 'GRID_FROM_FILE', 'path_init_grid': filename, 'name_init_lon': 'lon', 'name_init_lat': 'lat', 
                                                'name_init_mask': _base_config.GRID.name_init_mask, 'name_var_mask': _base_config.GRID.name_var_mask, 'subsampling': None})
                    if 'super' not in _config.MOD:
                        for NAME_MOD in _config.MOD:
                            _config.MOD[NAME_MOD] = _base_config.MOD[NAME_MOD].copy()
                            _config.MOD[NAME_MOD].init_from_bc = False
                    else:
                        _config.MOD.init_from_bc = False
                
                # Start from converged state vector
                if flag_init and name_exp_init is not None:
                    path_control_init = _config.INV.path_save_control_vectors.replace(config.EXP.name_experiment, name_exp_init)
                    _config.INV.path_init_4Dvar = os.path.join(path_control_init, 'Xres.nc')
                
                # Use background from another experiment
                if flag_background and name_exp_background is not None:
                    path_background = _config.INV.path_background.replace(config.EXP.name_experiment, name_exp_background)
                    _config.INV.path_background = os.path.join(path_background, 'Xres.nc')
                
                
                # append to list
                list_config[i].append(_config)
                list_State[i].append(_State)   
                if i==0:
                    if (lat0, lat1) != _prev_lat_band:
                        eq_tag = ' [EQ]' if is_eq else ''
                        print(f'\t ** Latitudes from {lat0:.2f} to {lat1:.2f} [{_ny}x{_nx}]{eq_tag}')
                        _prev_lat_band = (lat0, lat1)
                    print(f'\t\t * Longitudes from {lon0:.2f} to {lon1:.2f}')
                    list_lonlat.append(((lon1+lon0)/2,(lat1+lat0)/2))
                iproc_tw += 1

                # create directories
                if not os.path.exists(_config.EXP.tmp_DA_path):
                    os.makedirs(_config.EXP.tmp_DA_path)
                if not os.path.exists(_config.EXP.path_save):
                    os.makedirs(_config.EXP.path_save)

                # Save pickle files for each subwindow
                if path_save_pickle is not None:
                    path_pickle = f'{path_save_pickle}/{name_subwindow}'
                    if not os.path.exists(path_pickle):
                        os.makedirs(path_pickle)
                    with open(f'{path_pickle}/state.pkl', "wb") as f:
                        pickle.dump(_State, f)
                    with open(f'{path_pickle}/config.pkl', "wb") as f:
                        pickle.dump(_config, f)

                if flag_assim and (flag_assim_restart or not os.path.exists(f'{_config.INV.path_save_control_vectors}/Xres.nc')):
                    worker = partial(inv.Inv_4Dvar, config=_config, State=_State, verbose=0, gpu_device=gpu_devices[id_gpu])
                    p = mp.get_context("spawn").Process(target=worker)
                    if flag_init_from_previous:
                        list_processes[i].append(p)
                    else:
                        list_processes[0].append(p)
                    id_gpu += 1
                    if id_gpu==len(gpu_devices):
                        id_gpu = 0
                elif i==0:
                    if not flag_assim_restart:
                        print(f'Assimilation already done for this subwindow, skipping (use flag_assim_restart=True to re-run)')
                    if not flag_assim:
                        print(f'Assimilation not requested for this subwindow, skipping (use flag_assim=True to run)')

        iproc += iproc_tw

        if i == 0:
            lonlat_grid = [(_State.lon, _State.lat) for _State in list_State[0]]
            plot_subdomains(lonlat_grid)
            # Compute weights and interpolators after first time window
            weights_space, weights_space_sum, interpolators = compute_weights_map(
                State, list_State, path_save_pickle=path_save_pickle,
                space_overlap_x=space_overlap_x, space_overlap_y=space_overlap_y,
                lon_min=config.GRID.lon_min, lon_max=config.GRID.lon_max,
                lat_min=config.GRID.lat_min, lat_max=config.GRID.lat_max)
            plot_weights(State, weights_space_sum)

    print(f'Number of tiles: {iproc} ({iproc_tw} per time window)')

    # Save global pickles
    if dir_save_pickle is not None:
        path_save_pickle = f'{dir_save_pickle}/{config.EXP.name_experiment}'
        if not os.path.exists(path_save_pickle):
            os.makedirs(path_save_pickle)
        with open(f'{path_save_pickle}/config.pkl', 'wb') as f:
            pickle.dump(config, f)
        with open(f'{path_save_pickle}/State.pkl', 'wb') as f:
            pickle.dump(State, f)
        with open(f'{path_save_pickle}/dates.pkl', 'wb') as f:
            pickle.dump((list_date_start, list_date_middle, list_date_end), f)
        with open(f'{path_save_pickle}/list_State.pkl', 'wb') as f:
            pickle.dump(list_State, f)

    return list_processes, list_config, list_State, list_date_start, list_date_end, list_date_middle, list_lonlat, \
           weights_space, weights_space_sum, interpolators

def compute_weights_map(State, list_State, path_save_pickle=None,
                        space_overlap_x=2, space_overlap_y=2,
                        lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                        taper_factor=1.0):

    """Compute weights maps and precomputed interpolation operators for merging outputs from subprocesses.
    
    Weights use smootherstep tapering based on per-row (longitude) and per-column
    (latitude) distance from tile edges. This correctly handles GRID_CAR tiles whose 
    shape is trapezoidal in lon/lat space. No tapering is applied on sides that lie
    at the domain boundary. After interpolation, weights are normalized so that the
    sum is exactly 1.0 everywhere, compensating for varying overlap widths.
    
    Interpolation operators are precomputed once per subwindow and reused for all dates,
    avoiding the expensive Delaunay triangulation of griddata at every time step.
    
    Parameters
    ----------
    State : State
        Global state object (target grid).
    list_State : list of list of State
        Subwindow states grouped by time window.
    path_save_pickle : str, optional
        If provided, save weights_space, weights_space_sum and interpolators
        to '{path_save_pickle}/weights.pkl'. If None, no pickle is saved.
    space_overlap_x : float, optional
        Overlap in x direction in degrees (default: 2).
    space_overlap_y : float, optional
        Overlap in y direction in degrees (default: 2).
    lon_min, lon_max : float, optional
        Domain boundaries in longitude. Tiles touching these boundaries 
        are not tapered on that side. If None, all sides are tapered.
    lat_min, lat_max : float, optional
        Domain boundaries in latitude. Same behavior as lon_min/lon_max.
    taper_factor : float, optional
        Multiplier for the taper width relative to the overlap distance (default: 1.0).
        A value of 1.0 ensures the taper zone exactly matches the overlap,
        giving a uniform weight sum of 1.0 via smootherstep symmetry S(t)+S(1-t)=1.
    
    Returns
    -------
    weights_space : list of 2D arrays
        Weight maps for each subwindow, interpolated onto the target grid.
    weights_space_sum : 2D array
        Sum of all weight maps (for normalization).
    interpolators : list of callable or None
        Precomputed interpolation operators mapping each subwindow grid to the target grid.
        For regular grids (GRID_GEO), uses RegularGridInterpolator.
        For irregular grids (GRID_CAR), uses LinearNDInterpolator.
        Each callable takes a 2D array on the subwindow grid and returns a 2D array on the target grid.
    """

    weights_space = [] 
    weights_space_sum = np.zeros((State.ny, State.nx))
    interpolators = []

    single_subwindow = (len(list_State[0]) == 1)
    
    lon_out = State.lon
    lat_out = State.lat

    for _State in list_State[0]:

        if single_subwindow:
            # Single subwindow: uniform weights (no tapering needed)
            _weights_space = np.ones((_State.ny, _State.nx))
        else:
            _lon = _State.lon
            _lat = _State.lat

            # Per-row distances for longitude (handles GRID_CAR trapezoidal tiles)
            # Each row has its own western/eastern edge
            dist_west = _lon - _lon[:, 0:1]
            dist_east = _lon[:, -1:] - _lon
            # Per-column distances for latitude
            dist_south = _lat - _lat[0:1, :]
            dist_north = _lat[-1:, :] - _lat

            # Don't taper at domain boundaries (per-row for lon, per-column for lat)
            tol = 0.5  # tolerance in degrees for boundary detection
            at_west = np.zeros((_State.ny, 1), dtype=bool)
            at_east = np.zeros((_State.ny, 1), dtype=bool)
            at_south = np.zeros((1, _State.nx), dtype=bool)
            at_north = np.zeros((1, _State.nx), dtype=bool)
            if lon_min is not None:
                at_west[:, 0] = _lon[:, 0] <= lon_min + tol
            if lon_max is not None:
                at_east[:, 0] = _lon[:, -1] >= lon_max - tol
            if lat_min is not None:
                at_south[0, :] = _lat[0, :] <= lat_min + tol
            if lat_max is not None:
                at_north[0, :] = _lat[-1, :] >= lat_max - tol

            # Normalized distance: 0 at edge, 1 at taper_factor * overlap distance
            taper_x = taper_factor * space_overlap_x
            taper_y = taper_factor * space_overlap_y
            wx_west = np.where(at_west, 1.0, np.clip(dist_west / taper_x, 0, 1))
            wx_east = np.where(at_east, 1.0, np.clip(dist_east / taper_x, 0, 1))
            wy_south = np.where(at_south, 1.0, np.clip(dist_south / taper_y, 0, 1))
            wy_north = np.where(at_north, 1.0, np.clip(dist_north / taper_y, 0, 1))

            # Smootherstep (C² Hermite): 6t⁵ - 15t⁴ + 10t³
            def _smootherstep(t):
                return t * t * t * (t * (t * 6 - 15) + 10)

            _weights_space = _smootherstep(wx_west) * _smootherstep(wx_east) * _smootherstep(wy_south) * _smootherstep(wy_north)
        
        lon_in = _State.lon
        lat_in = _State.lat

        # Build interpolation operator (precomputed, reused for all dates)
        _interp_func, _weights_space_interp = _build_interpolator(
            lon_in, lat_in, _weights_space, lon_out, lat_out,
            _State.lon_unit, State.lon_unit, State.ny, State.nx,
            _State.geo_grid)

        interpolators.append(_interp_func)

        ind = ~np.isnan(_weights_space_interp)
        weights_space.append(_weights_space_interp)
        weights_space_sum[ind] += _weights_space_interp[ind]

    if path_save_pickle is not None:
        if not os.path.exists(path_save_pickle):
            os.makedirs(path_save_pickle)
        with open(f'{path_save_pickle}/weights.pkl', 'wb') as f:
            pickle.dump({'weights_space': weights_space,
                         'weights_space_sum': weights_space_sum,
                         'interpolators': interpolators}, f)

    return weights_space, weights_space_sum, interpolators


class _RegularInterpolator:
    """Picklable regular grid interpolator."""
    def __init__(self, lat_1d, lon_1d, pts, ny_out, nx_out):
        self.lat_1d = lat_1d
        self.lon_1d = lon_1d
        self.pts = pts
        self.ny_out = ny_out
        self.nx_out = nx_out

    def __call__(self, var_2d):
        rgi = RegularGridInterpolator((self.lat_1d, self.lon_1d), var_2d,
                                      method='linear', bounds_error=False, fill_value=np.nan)
        return rgi(self.pts).reshape(self.ny_out, self.nx_out)


class _IrregularInterpolator:
    """Picklable irregular grid interpolator using precomputed Delaunay triangulation."""
    def __init__(self, tri, pts_out, ny_out, nx_out):
        self.tri = tri
        self.pts_out = pts_out
        self.ny_out = ny_out
        self.nx_out = nx_out

    def __call__(self, var_2d):
        lndi = LinearNDInterpolator(self.tri, var_2d.ravel())
        return lndi(self.pts_out).reshape(self.ny_out, self.nx_out)


class _SplitInterpolator:
    """Picklable interpolator for longitude-wrapping grids."""
    def __init__(self, ind_0, ind_1, tri_0, tri_1, pts_out, ny_out, nx_out):
        self.ind_0 = ind_0
        self.ind_1 = ind_1
        self.tri_0 = tri_0
        self.tri_1 = tri_1
        self.pts_out = pts_out
        self.ny_out = ny_out
        self.nx_out = nx_out

    def __call__(self, var_2d):
        f0 = LinearNDInterpolator(self.tri_0, var_2d[self.ind_0].ravel())
        f1 = LinearNDInterpolator(self.tri_1, var_2d[self.ind_1].ravel())
        r0 = f0(self.pts_out).reshape(self.ny_out, self.nx_out)
        r1 = f1(self.pts_out).reshape(self.ny_out, self.nx_out)
        return np.where(np.isnan(r0), r1, r0)


def _build_interpolator(lon_in, lat_in, values, lon_out, lat_out, 
                        lon_unit_in, lon_unit_out, ny_out, nx_out, geo_grid):
    """Build a reusable interpolation operator from a subwindow grid to the target grid.
    
    For regular (GRID_GEO) grids, uses RegularGridInterpolator (fast structured interpolation).
    For irregular (GRID_CAR) grids, precomputes a LinearNDInterpolator (Delaunay triangulation done once).
    
    Returns a callable interp_func(values_2d) -> interpolated_2d and the interpolated input values.
    """
    needs_lon_split = (lon_unit_in != lon_unit_out and lon_unit_out == '-180_180' 
                       and (lon_in.max() > 180 or lon_in.min() < -180))

    if not needs_lon_split:
        if geo_grid:
            # Regular grid: use fast RegularGridInterpolator
            lon_1d = lon_in[0, :]
            lat_1d = lat_in[:, 0]
            pts = np.column_stack([lat_out.ravel(), lon_out.ravel()])
            interp_func = _RegularInterpolator(lat_1d, lon_1d, pts, ny_out, nx_out)
            values_interp = interp_func(values)
        else:
            # Irregular grid: precompute Delaunay triangulation once
            points = np.column_stack([lon_in.ravel(), lat_in.ravel()])
            pts_out = np.column_stack([lon_out.ravel(), lat_out.ravel()])
            lndi = LinearNDInterpolator(points, values.ravel())
            values_interp = lndi(pts_out).reshape(ny_out, nx_out)
            interp_func = _IrregularInterpolator(lndi.tri, pts_out, ny_out, nx_out)
    else:
        # Longitude wrapping: split into two halves
        pts_out = np.column_stack([lon_out.ravel(), lat_out.ravel()])
        
        ind_0 = lon_in <= 180
        points_0 = np.column_stack([lon_in[ind_0].ravel(), lat_in[ind_0].ravel()])
        
        ind_1 = lon_in > 180
        lon_in_1 = (lon_in[ind_1] + 180) % 360 - 180
        lat_in_1 = lat_in[ind_1]
        points_1 = np.column_stack([lon_in_1.ravel(), lat_in_1.ravel()])

        lndi_0 = LinearNDInterpolator(points_0, values[ind_0].ravel())
        lndi_1 = LinearNDInterpolator(points_1, values[ind_1].ravel())

        v0 = lndi_0(pts_out).reshape(ny_out, nx_out)
        v1 = lndi_1(pts_out).reshape(ny_out, nx_out)
        values_interp = np.where(np.isnan(v0), v1, v0)
        values_interp = np.where(np.isnan(values_interp), np.nan, values_interp)

        interp_func = _SplitInterpolator(ind_0, ind_1, lndi_0.tri, lndi_1.tri, pts_out, ny_out, nx_out)

    return interp_func, values_interp

def plot_weights(State, weights_space_sum):
    """Plot the weights_space_sum on a map with coastlines."""

    fig, ax = plt.subplots(figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    im = ax.pcolormesh(State.lon, State.lat, weights_space_sum, transform=ccrs.PlateCarree(), cmap=cm.viridis)
    plt.colorbar(im, ax=ax, label="Weight sum")

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    plt.show()

def plot_subdomains(lonlat_grid):

    dxdy_grid = [grid.lonlat2dxdy(lon,lat) for (lon,lat) in lonlat_grid]
    dxdy = [(np.mean(_dx)*1e-3, np.mean(_dy)*1e-3) for _dx,_dy in dxdy_grid]

    def plot_lonlat(ax, lonlat_grid, dx=10, alpha_value=.3, norm=None, cmap=cm.viridis):

        lon_grid,lat_grid = lonlat_grid 

        color = cmap(norm(dx))  # Get normalized color
        color_with_alpha = (*color[:3], alpha_value)  # Convert to RGBA

        vertices = np.array([
            [lon_grid[0,0], lat_grid[0,0]], 
            [lon_grid[0,-1], lat_grid[0,-1]], 
            [lon_grid[-1,-1], lat_grid[-1,-1]], 
            [lon_grid[-1,0], lat_grid[-1,0]]
            ])

        poly_shape = Polygon(vertices)

        # Add the polygon to GeoAxes
        ax.add_feature(ShapelyFeature([poly_shape], ccrs.PlateCarree(), edgecolor='black', facecolor=color_with_alpha, linewidth=2))

    # Create a figure and an axis with PlateCarree projection
    fig, ax = plt.subplots(figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    norm = mcolors.Normalize(vmin=5, vmax=10)
    cmap = cm.viridis  # Choose colormap (e.g., 'viridis', 'jet', 'plasma', etc.)

    for _dx, lonlat in zip(dxdy,lonlat_grid): plot_lonlat(ax, lonlat, dx=_dx[0], norm=norm, cmap=cmap) 

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)  # Create ScalarMappable
    sm.set_array([])  # Dummy array for colorbar
    cbar = plt.colorbar(sm, ax=ax, label="Spatial resolution (km)")  # Add colorbar

    # Add features
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
    gl.top_labels = False  # Remove top labels
    gl.right_labels = False  # Remove right labels

    # Show the plot
    plt.show()

def merge_output_date(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, interpolators, plot=False, save=True):

    """Merge outputs from subprocesses for a given date.
    
    Uses precomputed interpolation operators (from compute_weights_map) to avoid
    recomputing Delaunay triangulations at every time step.
    """

    State0 = State.copy()
    ny, nx = State0.ny, State0.nx

    dict_var = {name: np.zeros((ny, nx)) for name in name_var_save}
        
    for _State, _weights_space, _interp_func in zip(list_State, weights_space, interpolators):

        try:
            # Load output
            _ds = _State.load_output(date)
            lon = _ds.lon.values
            lat = _ds.lat.values
            if len(lon.shape) == 1:
                lon, lat = np.meshgrid(lon, lat)

            for name in name_var_save:

                _var = _ds[name].values
                
                # Handle C-grid staggering: average U/V-grid variables to H-grid
                if _var.shape == (_State.ny, _State.nx + 1):
                    # U-grid → H-grid: average adjacent columns
                    _var = 0.5 * (_var[:, :-1] + _var[:, 1:])
                elif _var.shape == (_State.ny + 1, _State.nx):
                    # V-grid → H-grid: average adjacent rows
                    _var = 0.5 * (_var[:-1, :] + _var[1:, :])
                
                # Fill NaN gaps near coasts before interpolation
                if np.any(np.isnan(_var)) and kernel is not None:
                    _var = interpolate_replace_nans(_var, kernel)
                
                # Interpolate using precomputed operator
                if _interp_func is not None:
                    _var_interp = _interp_func(_var)
                else:
                    # Single subwindow, no interpolation needed (grids match)
                    _var_interp = _var

                # Merge
                ind = ~np.isnan(_var_interp)
                dict_var[name][ind] += (_weights_space * _var_interp / weights_space_sum)[ind]
            
            _ds.close()
            del _ds
        except Exception as e:
            print(f'[merge_output_date] Warning: failed to merge subwindow for date {date}: {e}')
            continue

    for name in name_var_save:
        # Mask
        if State0.mask is not None and np.any(State0.mask):
            dict_var[name][State0.mask] = np.nan
            if plot:
                plt.figure()
                plt.pcolormesh(State0.lon, State0.lat, dict_var[name])
                cbar = plt.colorbar()
                cbar.ax.set_ylabel(name)
                plt.title(date)
                plt.show()
        State0.setvar(dict_var[name], name)
    
    if save:
        State0.save_output(date, name_var=name_var_save)
    
def generate_dates(start_date, end_date, delta):
    """Generate a list of dates between start_date and end_date with a given timedelta."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta
    return dates

def parallel_merge(dates, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, interpolators, num_workers=4):
    """Merge outputs from subprocesses in parallel for a list of dates."""
    
    if num_workers<=1:
        for date in dates:
            merge_output_date(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, interpolators)    
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(
                merge_output_date,
                [(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, interpolators) for date in dates]
            )

def run_assimilation_time_window(config, date_start, date_middle, date_end, list_State, processes, 
                                 weights_space=None, weights_space_sum=None, interpolators=None,
                                 name_var_save=['sla'], 
                                 flag_assim=True, flag_merge_outputs=True, flag_diag=True, flag_overwrite_outputs=True,
                                 nprocs=4, nprocs_output=None,
                                 path_pickle=None):
    
    """
    Run assimilation in a given time window using subprocesses.
    The subprocesses are run in parallel and the outputs are merged.
    Diagnostics are then computed on the merged outputs.
    
    If weights_space, weights_space_sum, and interpolators are None,
    they are loaded from '{path_pickle}/weights.pkl'.
    """

    # Load weights and interpolators from pickle if not provided
    if weights_space is None and path_pickle is not None:
        with open(f'{path_pickle}/weights.pkl', 'rb') as f:
            data = pickle.load(f)
        weights_space = data['weights_space']
        weights_space_sum = data['weights_space_sum']
        interpolators = data['interpolators']

    ############################
    # Run subprocesses
    ############################
    if flag_assim:
        print('Run subprocesses')
        try:
            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w") # prevent printoing outputs
            active_processes = set()
                
            for process in processes[:nprocs]:  # Start initial nprocs processes
                process.start()
                active_processes.add(process)

            for process in processes[nprocs:]:  # Start remaining processes dynamically
                while len(active_processes) >= nprocs:
                    for p in list(active_processes):
                        if not p.is_alive():
                            p.join()
                            active_processes.remove(p)
                            break  # Start a new process immediately after one finishes
                    
                process.start()
                active_processes.add(process)

            # Wait for remaining processes to finish
            for process in active_processes:
                process.join()
            sys.stdout = old_stdout
        except:
            sys.stdout = old_stdout
            print('Unable to run subprocesses')

    
    ############################
    # Create merged config
    ############################
    config0 = config.copy()
    config0.EXP = config0.EXP.copy()
    config0.EXP.init_date = date_start
    config0.EXP.final_date = date_end
    config0.EXP.tmp_DA_path += f'/subwindow_{str(date_middle)[:10]}'
    config0.EXP.path_save += f'/subwindow_{str(date_middle)[:10]}'
    if flag_diag and config.DIAG is not None:
        config0.DIAG = config.DIAG.copy()
        if 'super' not in config0.DIAG:
            for NAME_DIAG in config0.DIAG:
                config0.DIAG[NAME_DIAG] = config.DIAG[NAME_DIAG].copy()
                config0.DIAG[NAME_DIAG].dir_output += f'/subwindow_{str(date_middle)[:10]}'
                config0.DIAG[NAME_DIAG].time_min = date_start.strftime('%Y-%m-%d')
                config0.DIAG[NAME_DIAG].time_max = date_end.strftime('%Y-%m-%d')
        else:
            config0.DIAG.dir_output += f'/subwindow_{str(date_middle)[:10]}'
            config0.DIAG.time_min = date_start.strftime('%Y-%m-%d')
            config0.DIAG.time_max = date_end.strftime('%Y-%m-%d')
    
    State0 = state.State(config0, verbose=0)
    
    ############################
    # Merge outputs
    ############################
    if flag_merge_outputs and ((flag_overwrite_outputs) or (len(glob.glob(f'{config0.EXP.path_save}/*.nc'))==0) or (flag_assim and len(processes)>0)): 
        try:
            print('Merge outputs')
            kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)  # Kernel to convolve output maps to replace NaN pixels close to the coast for interpolation
            list_dates = generate_dates(date_start, date_end, config.EXP.saveoutput_time_step)
            num_workers = nprocs_output if nprocs_output is not None else nprocs
            parallel_merge(list_dates, State0, list_State, name_var_save, kernel, weights_space, weights_space_sum, interpolators, num_workers=num_workers)

        except:
            print('Unable to merge outputs')
    
    ############################
    # Diagnostics
    ############################
    if flag_diag:
        try:
            print('Run Diagnostics')
            Diag = diag.Diag(config0,State0)
            Diag.regrid_exp()
            Diag.rmse_based_scores(plot=True)
            Diag.psd_based_scores(plot=True)
            #Diag.movie()
            Diag.Leaderboard()
        except:
            print('Unable to compute diags')
        
        del State0, config0

def merge_time_windows_outputs(config, list_date_start, list_date_middle, list_date_end, time_overlap):
    
    """
    Merge outputs from different time windows.
    
    In overlap regions between consecutive windows, a raised-cosine (Hann) blending 
    is used for a smooth transition with zero-derivative at the boundaries,
    avoiding artifacts from a linear ramp.

    Parameters
    ----------
    config : Config
        Main configuration object.
    list_date_start : list of datetime
        Start date of each time window.
    list_date_middle : list of datetime
        Middle date of each time window.
    list_date_end : list of datetime
        End date of each time window.
    time_overlap : float
        Overlap between consecutive time windows in days.
    """

    n_windows = len(list_date_start)

    def _build_path(subwindow_middle, date):
        return (f'{config.EXP.path_save}/subwindow_{str(subwindow_middle)[:10]}/'
                f'{config.EXP.name_experiment}'
                f'_y{date.year}'
                f'm{str(date.month).zfill(2)}'
                f'd{str(date.day).zfill(2)}'
                f'h{str(date.hour).zfill(2)}'
                f'm{str(date.minute).zfill(2)}.nc')
    
    def _build_output_path(date):
        return (f'{config.EXP.path_save}/'
                f'{config.EXP.name_experiment}'
                f'_y{date.year}'
                f'm{str(date.month).zfill(2)}'
                f'd{str(date.day).zfill(2)}'
                f'h{str(date.hour).zfill(2)}'
                f'm{str(date.minute).zfill(2)}.nc')

    # Collect all unique dates across all windows
    all_dates = set()
    for i in range(n_windows):
        date = list_date_start[i]
        while date <= list_date_end[i]:
            all_dates.add(date)
            date += config.EXP.saveoutput_time_step
    all_dates = sorted(all_dates)

    for date in all_dates:
        # Find which windows contain this date
        active = [i for i in range(n_windows)
                  if list_date_start[i] <= date <= list_date_end[i]]

        if len(active) == 1:
            # No overlap: use the single window directly
            i = active[0]
            dsout = xr.open_dataset(_build_path(list_date_middle[i], date)).load()

        elif len(active) >= 2:
            # Overlap region: blend the two closest consecutive windows
            i, j = active[0], active[1]
            overlap_start = list_date_start[j]
            overlap_end = list_date_end[i]
            overlap_duration = (overlap_end - overlap_start).total_seconds()

            ds1 = xr.open_dataset(_build_path(list_date_middle[i], date)).load()
            ds2 = xr.open_dataset(_build_path(list_date_middle[j], date)).load()

            if overlap_duration > 0:
                # alpha goes from 0 (at overlap_start) to 1 (at overlap_end)
                alpha = (date - overlap_start).total_seconds() / overlap_duration
                alpha = min(max(alpha, 0.0), 1.0)
                # Raised-cosine (Hann) blending: smooth S-curve with zero derivative at boundaries
                W2 = 0.5 * (1.0 - np.cos(np.pi * alpha))
                W1 = 1.0 - W2
            else:
                W1, W2 = 0.5, 0.5

            dsout = ds1.copy()
            for var in ds1.data_vars:
                if var in ds2.data_vars:
                    dsout[var] = W1 * ds1[var] + W2 * ds2[var]
                else:
                    dsout[var] = ds1[var]
            ds1.close()
            ds2.close()
        else:
            continue

        dsout.to_netcdf(_build_output_path(date))
        dsout.close()
