import os
import sys
import glob
import numpy as np
import multiprocessing as mp
from scipy.interpolate import griddata
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
                    nx_proc, ny_proc, dx, dy,
                    time_window_size_proc=None, space_window_size_proc_x=None, space_window_size_proc_y=None, 
                    time_overlap_frac=0.2, space_overlap_frac=0.2,
                    flag_init_from_previous=True, flag_init=False, flag_background=False,
                    flag_assim=True, flag_assim_restart=False,
                    name_exp_init=None, name_exp_background=None,
                    gpu_devices=['0']):
    """
    Prepare subprocesses for assimilation in subwindows in time and space.
    The subprocesses can then be run in parallel using multiprocessing. 
    The outputs of the subprocesses can then be merged using merge_output_date().
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
    n_wx = 0
    n_wy = 0

    id_gpu = 0

    date1 = init_date
    lat_min = config.GRID.lat_min
    lon_min = config.GRID.lon_min
    lat1 = lat_min
    lon1 = lon_min
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
            date0 = init_date + n_wt * time_delta * (1-time_overlap_frac)
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

        j = -1
        while lat1<config.GRID.lat_max:
            # compute subwindow latitude borders
            if space_window_size_proc_y is not None:
                lat0 = lat_min + n_wy * space_window_size_proc_y * (1-space_overlap_frac)
                lat1 = lat0 + space_window_size_proc_y
                _ny = ny_proc
                if lat0 + space_window_size_proc_y/2 > config.GRID.lat_max:
                    lat1 = lat0 + space_window_size_proc_y/2  
                    _ny = int(ny_proc/2)                  
                n_wy += 1
            else:
                lat0 = config.GRID.lat_min
                lat1 = config.GRID.lat_max
            if lat1<config.GRID.lat_min:
                continue
            j += 1
            flag_avoid_next_window = False
            while lon1<config.GRID.lon_max and not flag_avoid_next_window:
                # compute subwindow longitude borders
                if space_window_size_proc_x is not None:
                    if n_wx==0:
                        lon0 = config.GRID.lon_min
                    else:
                        if lat0>0:
                            lon0 = lon_prev[0,-1] - space_overlap_frac * space_window_size_proc_x#- (space_overlap_frac + .5) * space_window_size_proc_x + nx_proc/2 * dx / np.cos(np.radians(lat0)) /111.32
                        else:
                            lon0 = lon_prev[-1,-1] - space_overlap_frac * space_window_size_proc_x#- (space_overlap_frac + .5) * space_window_size_proc_x + nx_proc/2 * dx / np.cos(np.radians(lat1)) /111.32
                    #lon0 = config.GRID.lon_min + n_wx * space_window_size_proc * (1-space_overlap_frac)
                    lon1 = lon0 + space_window_size_proc_x#min(lon0 + space_window_size_proc, config.GRID.lon_max)
                    _nx = nx_proc
                    if lon0 + space_window_size_proc_x/2 > config.GRID.lon_max:
                        lon1 = lon0 + space_window_size_proc_x/2
                        _nx = int(nx_proc/2)
                    n_wx += 1
                else:
                    lon0 = config.GRID.lon_min
                    lon1 = config.GRID.lon_max
                    
                # create config for the subwindow
                if lat0<0 and lat1>0:
                    _config = config_eq.copy()
                else:
                    _config = config.copy()
                _config.EXP = _config.EXP.copy()
                _config.GRID = _config.GRID.copy()
                _config.MOD = _config.MOD.copy()
                _config.INV = _config.INV.copy()
                _config.EXP.init_date = date0
                _config.EXP.final_date = date1
                _config.GRID.super = 'GRID_CAR'
                _config.GRID.lon_min = lon0
                _config.GRID.lon_max = lon1
                _config.GRID.lat_min = lat0
                _config.GRID.lat_max = lat1
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
                                                'name_init_mask': config.GRID.name_init_mask, 'name_var_mask': config.GRID.name_var_mask, 'subsampling': None})
                    if 'super' not in _config.MOD:
                        for NAME_MOD in _config.MOD:
                            _config.MOD[NAME_MOD] = config.MOD[NAME_MOD].copy()
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
                    list_lonlat.append(((lon1+lon0)/2,(lat1+lat0)/2))

                # create directories
                if not os.path.exists(_config.EXP.tmp_DA_path):
                    os.makedirs(_config.EXP.tmp_DA_path)
                if not os.path.exists(_config.EXP.path_save):
                    os.makedirs(_config.EXP.path_save)

                if flag_assim and (flag_assim_restart or not os.path.exists(f'{_config.INV.path_save_control_vectors}/Xres.nc')):
                    print(iproc, name_subwindow)
                    worker = partial(inv.Inv_4Dvar, config=_config, State=_State, verbose=0, gpu_device=gpu_devices[id_gpu])
                    p = mp.get_context("spawn").Process(target=worker)
                    if flag_init_from_previous:
                        list_processes[i].append(p)
                    else:
                        list_processes[0].append(p)
                    id_gpu += 1
                    iproc += 1
                    if id_gpu==len(gpu_devices):
                        id_gpu = 0
            n_wx = 0
            lon1 = config.GRID.lon_min
        n_wy = 0
        lat1 = config.GRID.lat_min
        j = 0

    return list_processes, list_config, list_State, list_date_start, list_date_end, list_date_middle, list_lonlat

def compute_weights_map(State, list_State):

    """Compute weights maps for merging outputs from subprocesses.
    The weights maps are based on Gaspari-Cohn functions in space and a linear function in depth.
    The weights maps are then interpolated onto the target grid.
    """

    weights_space = [] 
    weights_space_sum = np.zeros((State.ny, State.nx))

    if len(list_State[0])==1:
        weights_space.append(np.ones((State.ny, State.nx)))
        weights_space_sum += np.ones((State.ny, State.nx))
        return weights_space, weights_space_sum
    
    for _State in list_State[0]:

        winy = np.ones(_State.ny)
        winx = np.ones(_State.nx)
        winy[:int(_State.ny-_State.ny/2)] = gaspari_cohn(np.arange(0,_State.ny),_State.ny/2)[:int(_State.ny/2)][::-1]
        winx[:int(_State.nx-_State.nx/2)] = gaspari_cohn(np.arange(0,_State.nx),_State.nx/2)[:int(_State.nx/2)][::-1]
        winy[int(_State.ny/2):] = gaspari_cohn(np.arange(0,_State.ny),_State.ny/2)[:_State.ny-int(_State.ny/2)]
        winx[int(_State.nx/2):] = gaspari_cohn(np.arange(0,_State.nx),_State.nx/2)[:_State.nx-int(_State.nx/2)]

        win_dy = (1 - (_State.DY-_State.DY.min())/(_State.DY.max() - _State.DY.min()))**2

        _weights_space = winy[:,np.newaxis] * winx[np.newaxis,:] * win_dy
        
        lon_in = _State.lon
        lat_in = _State.lat
        lon_out = State.lon
        lat_out = State.lat

        if _State.lon_unit != State.lon_unit and State.lon_unit=='-180_180' and (lon_in.max()>180 or lon_in.min()<-180):
            _weights_space_interp = np.zeros((State.ny,State.nx)) * np.nan
            ind_0 = lon_in<=180
            _weights_space_interp_0 = griddata((lon_in[ind_0].ravel(),lat_in[ind_0].ravel()), _weights_space[ind_0].ravel(), (lon_out.ravel(),lat_out.ravel()), method='linear').reshape((State.ny,State.nx))
            _weights_space_interp = np.where(np.isnan(_weights_space_interp_0), _weights_space_interp, _weights_space_interp_0)
            ind_1 = lon_in>180
            lon_in_1 = lon_in[ind_1] 
            # Convert to [-180, 180]
            lon_in_1 = (lon_in_1 + 180) % 360 - 180 
            lat_in_1 = lat_in[ind_1]
            _weights_space_1 = _weights_space[ind_1]
            _weights_space_interp_1 = griddata((lon_in_1.ravel(),lat_in_1.ravel()), _weights_space_1.ravel(), (lon_out.ravel(),lat_out.ravel()), method='linear').reshape((State.ny,State.nx))
            _weights_space_interp = np.where(np.isnan(_weights_space_interp_1), _weights_space_interp, _weights_space_interp_1)
        else:
            _weights_space_interp = griddata((lon_in.ravel(),lat_in.ravel()), _weights_space.ravel(), (lon_out.ravel(),lat_out.ravel()), method='linear').reshape((State.ny,State.nx))

        ind = ~np.isnan(_weights_space_interp)
        weights_space.append(_weights_space_interp)
        weights_space_sum[ind] += _weights_space_interp[ind]

    return weights_space, weights_space_sum

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

def merge_output_date(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, plot=False, save=True):

    """Merge outputs from subprocesses for a given date.
    The outputs from the subprocesses are interpolated onto the target grid and merged using the weights maps.
    The merged outputs are then saved to the target grid.
    """

    State0 = State.copy()

    lon_out = State0.lon.ravel()
    lat_out = State0.lat.ravel()
    ny,nx = State0.ny, State0.nx

    dict_var = {name: np.zeros((ny,nx)) for name in name_var_save}
        
    for _State, _weights_space in zip(list_State, weights_space):

        try:
            # Load output
            _ds = _State.load_output(date)
            lon = _ds.lon.values
            lat = _ds.lat.values
            if len(lon.shape) == 1:
                lon, lat = np.meshgrid(lon, lat)

            for name in name_var_save:

                _var = _ds[name].values
                
                # Interp
                if np.any(np.isnan(_var)) and kernel is not None:
                    _var = interpolate_replace_nans(_var, kernel)
                
                if _State.lon_unit != State.lon_unit and State.lon_unit=='-180_180' and lon.max()>180:

                    _var_interp = np.zeros((State.ny,State.nx)) * np.nan

                    ind_0 = lon<=180.
                    _var_interp_0 = griddata(
                            (lon[ind_0].ravel(), lat[ind_0].ravel()), _var[ind_0].ravel(),
                            (lon_out, lat_out),
                            method='linear'
                        ).reshape((ny,nx))
                    _var_interp = np.where(np.isnan(_var_interp_0), _var_interp, _var_interp_0)
                    
                    ind_1 = lon>180.
                    lon_in_1 = lon[ind_1] 
                    # Convert to [-180, 180]
                    lon_in_1 = (lon_in_1 + 180) % 360 - 180 
                    lat_in_1 = lat[ind_1]
                    _var_interp_1 = griddata(
                            (lon_in_1.ravel(), lat_in_1.ravel()),  _var[ind_1].ravel(),
                            (lon_out, lat_out),
                            method='linear'
                        ).reshape((ny,nx))                    
                    _var_interp = np.where(np.isnan(_var_interp_1), _var_interp, _var_interp_1)

                else:
                    _var_interp = griddata(
                        (lon.ravel(), lat.ravel()), _var.ravel(),
                        (lon_out, lat_out),
                        method='linear'
                    ).reshape((ny,nx))

                # Merge
                ind = ~np.isnan(_var_interp)
                dict_var[name][ind] += (_weights_space * _var_interp / weights_space_sum)[ind]
            
            _ds.close()
            del _ds
        except:
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

def parallel_merge(dates, State, list_State, name_var_save, kernel, weights_space, weights_space_sum, num_workers=4):
    """Merge outputs from subprocesses in parallel for a list of dates."""
    
    if num_workers<=1:
        for date in dates:
            merge_output_date(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum)    
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(
                merge_output_date,
                [(date, State, list_State, name_var_save, kernel, weights_space, weights_space_sum) for date in dates]
            )

def run_assimilation_time_window(config, date_start, date_middle, date_end, list_State, processes, 
                                 weights_space, weights_space_sum,
                                 name_var_save=['sla'], 
                                 flag_assim=True, flag_merge_outputs=True, flag_diag=True, flag_overwrite_outputs=True,
                                 nprocs=4, nprocs_output=None):
    
    """
    Run assimilation in a given time window using subprocesses.
    The subprocesses are run in parallel and the outputs are merged.
    Diagnostics are then computed on the merged outputs.
    """

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
            parallel_merge(list_dates, State0, list_State, name_var_save, kernel, weights_space, weights_space_sum, num_workers=num_workers)

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

def merge_time_windows_outputs(config, list_date_start, list_date_middle, list_date_end, time_window_size_proc, time_overlap_frac):
    
    """
    Merge outputs from different time windows.
    The outputs from the different time windows are merged using a linear function in time.
    The merged outputs are then saved to the target grid.
    """

    for i in range(len(list_date_start)):

        date = list_date_start[i]
        while date <= list_date_end[i]:
        
            path_output = f'{config.EXP.path_save}/subwindow_{str(list_date_middle[i])[:10]}/{config.EXP.name_experiment}_y{date.year}'\
                                                                                                    f'm{str(date.month).zfill(2)}'\
                                                                                                    f'd{str(date.day).zfill(2)}'\
                                                                                                    f'h{str(date.hour).zfill(2)}'\
                                                                                                    f'm{str(date.minute).zfill(2)}.nc'
            ds1 = xr.open_dataset(path_output).load()

            if i<len(list_date_start)-1 and date>=list_date_start[i+1]:
                path_output = f'{config.EXP.path_save}/subwindow_{str(list_date_middle[i+1])[:10]}/{config.EXP.name_experiment}_y{date.year}'\
                                                                                                    f'm{str(date.month).zfill(2)}'\
                                                                                                    f'd{str(date.day).zfill(2)}'\
                                                                                                    f'h{str(date.hour).zfill(2)}'\
                                                                                                    f'm{str(date.minute).zfill(2)}.nc'
                ds2 = xr.open_dataset(path_output).load()

                dsout = ds1.copy()
                W1 = (list_date_end[i] - date).total_seconds()  / (24*3600*time_window_size_proc * time_overlap_frac)
                W2 = (date - list_date_start[i+1]).total_seconds() / (24*3600*time_window_size_proc * time_overlap_frac)
                for var in ds1.data_vars:
                    if var in ds2.data_vars:
                        denom = (W1 + W2)
                        if denom == 0:
                            continue
                        dsout[var] = (W1 * ds1[var] + W2 * ds2[var]) / denom
                    else:
                        dsout[var] = ds1[var]
            
            else:
                dsout = ds1.copy()
            
            path_output = f'{config.EXP.path_save}/{config.EXP.name_experiment}_y{date.year}'\
                                                                            f'm{str(date.month).zfill(2)}'\
                                                                            f'd{str(date.day).zfill(2)}'\
                                                                            f'h{str(date.hour).zfill(2)}'\
                                                                            f'm{str(date.minute).zfill(2)}.nc'
            dsout.to_netcdf(path_output)
            dsout.close()
            date += config.EXP.saveoutput_time_step
