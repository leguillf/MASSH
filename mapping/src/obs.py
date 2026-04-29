#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:15:09 2021

@author: leguillou
"""
import os, sys
import xarray as xr
import numpy as np

import datetime 
import re
from scipy import signal
import matplotlib.pylab as plt
import glob 

from .tools import detrendn, read_auxdata
from .exp import Config


# Date patterns recognized in obs file names, in order of preference.
# Each entry is (regex, "fmt") with named groups y, m, d (d optional).
_FILENAME_DATE_PATTERNS = [
    # 8-digit YYYYMMDD (most specific) — accept either standalone or
    # surrounded by non-digits, but require y >= 1900 to avoid matching
    # arbitrary 8-digit run numbers.
    re.compile(r'(?<!\d)(?P<y>(?:19|20)\d{2})(?P<m>\d{2})(?P<d>\d{2})(?!\d)'),
    # YYYY-MM-DD or YYYY_MM_DD
    re.compile(r'(?<!\d)(?P<y>(?:19|20)\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})(?!\d)'),
    # 6-digit YYYYMM (used by some monthly archives)
    re.compile(r'(?<!\d)(?P<y>(?:19|20)\d{2})(?P<m>\d{2})(?!\d)'),
]


def _extract_file_date_range(fname):
    """Extract a (start, end) datetime pair from a file basename.

    Looks for a YYYYMMDD / YYYY-MM-DD / YYYYMM pattern in the filename.
    Returns (None, None) when no recognizable date is present (so the file
    is *kept* — we never drop a file based on absence of a date).
    """
    base = os.path.basename(fname)
    for pat in _FILENAME_DATE_PATTERNS:
        m = pat.search(base)
        if m is None:
            continue
        try:
            y = int(m.group('y'))
            mo = int(m.group('m'))
            try:
                d = int(m.group('d'))
                start = datetime.datetime(y, mo, d)
                end = start + datetime.timedelta(days=1)
            except (IndexError, KeyError):
                # Monthly pattern: span the whole month
                start = datetime.datetime(y, mo, 1)
                if mo == 12:
                    end = datetime.datetime(y + 1, 1, 1)
                else:
                    end = datetime.datetime(y, mo + 1, 1)
            return start, end
        except ValueError:
            continue
    return None, None


def _filter_files_by_date(files, date_start, date_end):
    """Keep only files whose filename-encoded date overlaps [date_start, date_end].

    Files with no recognizable date in the name are kept (conservative).
    """
    if date_start is None and date_end is None:
        return files
    t0 = None if date_start is None else (
        date_start if isinstance(date_start, datetime.datetime)
        else datetime.datetime.fromisoformat(str(date_start)))
    t1 = None if date_end is None else (
        date_end if isinstance(date_end, datetime.datetime)
        else datetime.datetime.fromisoformat(str(date_end)))
    out = []
    for f in files:
        f_start, f_end = _extract_file_date_range(f)
        if f_start is None:
            out.append(f)
            continue
        if t1 is not None and f_start > t1:
            continue
        if t0 is not None and f_end <= t0:
            continue
        out.append(f)
    return out


def _open_obs_dataset(name_obs, OBS, date_start=None, date_end=None):
    """Open the multi-file dataset for one OBS block.

    Tries the fast path first (combine='nested' + parallel=True with preprocess),
    then falls back to no-preprocess, and finally to combine='by_coords'.
    Returns the opened dataset or None if all attempts failed.

    When date_start/date_end are provided, files whose filename contains a
    date pattern outside that range are skipped before opening.
    """

    def preprocess(ds):
        name_var = [OBS.name_time, OBS.name_lon, OBS.name_lat]
        for key in OBS.name_var:
            if isinstance(OBS.name_var[key], list):
                for name in OBS.name_var[key]:
                    name_var.append(name)
            else:
                name_var.append(OBS.name_var[key])
        ds = ds[name_var]
        return ds

    if '.nc' in OBS.path and '*' not in OBS.path:
        try:
            return xr.open_dataset(OBS.path)
        except Exception:
            print(f'[{name_obs}] Error: unable to open {OBS.path}')
            return None

    if '*' in OBS.path:
        path = OBS.path
    else:
        path = f'{OBS.path}*.nc'

    files = sorted(glob.glob(path))
    if len(files) == 0:
        print(f'[{name_obs}] Warning: no files matching {path}')
        return None

    # Filter by filename date pattern (e.g. "20250603" or "202506") so we
    # don't open files that fall entirely outside the experiment period.
    n_total = len(files)
    files = _filter_files_by_date(files, date_start, date_end)
    if len(files) == 0:
        print(f'[{name_obs}] Warning: no files in date range '
              f'[{date_start}, {date_end}] (out of {n_total} candidates)')
        return None
    if len(files) < n_total:
        print(f'[{name_obs}] Date filter: {len(files)}/{n_total} files kept')

    # Get time dim name from first file (cheap)
    try:
        _ds0 = xr.open_dataset(files[0])
        name_time_dim = _ds0[OBS.name_time].dims[0]
        _ds0.close()
    except Exception:
        name_time_dim = None

    # Try combine='nested' with preprocess + parallel (fast path)
    if name_time_dim is not None:
        try:
            return xr.open_mfdataset(
                files, combine='nested', concat_dim=name_time_dim,
                preprocess=preprocess, compat='override', coords='minimal',
                parallel=True)
        except Exception:
            pass

        # Try combine='nested' without preprocess
        try:
            return xr.open_mfdataset(
                files, combine='nested', concat_dim=name_time_dim,
                compat='override', coords='minimal', parallel=True)
        except Exception:
            pass

    # Last resort: by_coords
    try:
        return xr.open_mfdataset(
            files, preprocess=preprocess, compat='override', coords='minimal')
    except Exception:
        try:
            return xr.open_mfdataset(files, compat='override', coords='minimal')
        except Exception:
            print(f'[{name_obs}] Error: unable to open multiple netcdf files')
            return None


def open_obs_datasets(config, date_start=None, date_end=None):
    """Open all OBS datasets declared in config once, lazily.

    Eagerly loads only the time/lon/lat coordinate arrays so that bbox masks
    can be built per tile without re-reading them from disk.
    Returns a dict {name_obs: xarray.Dataset} (or empty dict if config.OBS is None).

    When date_start/date_end are provided, only files whose filename date
    pattern overlaps that range are opened. This avoids paying the cost of
    opening (potentially) thousands of out-of-period files for a global archive.
    """
    if config.OBS is None:
        return {}
    datasets = {}
    for name_obs, OBS in config.OBS.items():
        print(f'Opening obs dataset: {name_obs}')
        ds = _open_obs_dataset(name_obs, OBS,
                               date_start=date_start, date_end=date_end)
        if ds is None:
            continue
        # Eagerly load 1D coordinate arrays used to build per-tile masks.
        # Skip 2D coords (e.g. SWOT swath lon/lat) — keeping them as dask
        # arrays lets the per-tile mask be computed lazily without
        # materializing the global per-pixel field.
        for cname in (OBS.name_time, OBS.name_lon, OBS.name_lat):
            try:
                if ds[cname].ndim <= 1:
                    ds[cname].load()
            except Exception:
                pass
        datasets[name_obs] = ds
    return datasets


def select_obs_datasets_time(obs_datasets, config, date_start, date_end):
    """Restrict each obs dataset to [date_start, date_end] along its time dim.

    Operates on the (already-loaded) 1D time coordinate to build a boolean
    mask, then uses isel on the underlying time *dimension* — this keeps
    SWOT swath data lazy (no broadcast against per-pixel arrays).
    Returns a new dict {name_obs: ds_subset}.
    """
    if obs_datasets is None or len(obs_datasets) == 0:
        return {}
    t0 = np.datetime64(date_start)
    t1 = np.datetime64(date_end)
    out = {}
    for name_obs, ds in obs_datasets.items():
        OBS = config.OBS[name_obs]
        time_arr = ds[OBS.name_time]
        # Identify the underlying time dimension (handles cases where
        # name_time is itself a coord on a differently-named dim)
        time_dim = time_arr.dims[0] if time_arr.ndim >= 1 else OBS.name_time
        mask = ((time_arr >= t0) & (time_arr <= t1)).values
        if mask.ndim != 1:
            # fallback: any() over non-time-dim axes (rare)
            axes = tuple(i for i, d in enumerate(time_arr.dims) if d != time_dim)
            mask = mask.any(axis=axes) if axes else mask
        if not mask.any():
            continue
        out[name_obs] = ds.isel({time_dim: mask})
    return out


def _lon_convert_mode(ds, OBS, lon_unit):
    """Decide whether to convert longitudes to 0..360 or -180..180.

    The lon min/max are computed lazily once per ds and cached on ds.attrs
    (so SWOT 2D lon is streamed only the first time).
    Returns one of '0_360', '-180_180', or None.
    """
    if '_lon_min_cached' not in ds.attrs:
        try:
            ds.attrs['_lon_min_cached'] = float(ds[OBS.name_lon].min().compute())
            ds.attrs['_lon_max_cached'] = float(ds[OBS.name_lon].max().compute())
        except Exception:
            ds.attrs['_lon_min_cached'] = 0.0
            ds.attrs['_lon_max_cached'] = 0.0
    lmin = ds.attrs['_lon_min_cached']
    lmax = ds.attrs['_lon_max_cached']
    if np.sign(lmin) == -1 and lon_unit == '0_360':
        return '0_360'
    if (np.sign(lmin) >= 0 or lmax > 180) and lon_unit == '-180_180':
        return '-180_180'
    return None


def select_obs_datasets_space(obs_datasets, config, bbox, lon_unit='0_360'):
    """Restrict each obs dataset to a lon/lat bbox, lazily.

    For 2D coords (e.g. SWOT swath), the bbox mask is reduced to the time
    dim and applied via isel — this keeps the per-pixel arrays lazy and
    avoids the where(drop=True) deep-copy that caused 50 GiB OOMs.
    For 1D coords, the standard where(drop=True) is used.
    Returns a new dict {name_obs: ds_subset}.
    """
    if obs_datasets is None or len(obs_datasets) == 0:
        return {}
    out = {}
    for name_obs, ds in obs_datasets.items():
        OBS = config.OBS[name_obs]
        # Pick a lon view that matches the requested unit (lazy, no full read)
        convert = _lon_convert_mode(ds, OBS, lon_unit)
        lon_raw = ds[OBS.name_lon]
        if convert == '0_360':
            lon_for_mask = lon_raw % 360
        elif convert == '-180_180':
            lon_for_mask = (lon_raw + 180) % 360 - 180
        else:
            lon_for_mask = lon_raw
        lat = ds[OBS.name_lat]
        bbox_mask = ((bbox[0] <= lon_for_mask) & (bbox[1] >= lon_for_mask) &
                     (bbox[2] <= lat) & (bbox[3] >= lat))
        # Time dim
        time_arr = ds[OBS.name_time]
        time_dim = time_arr.dims[0] if time_arr.ndim >= 1 else OBS.name_time
        reduce_dims = [d for d in bbox_mask.dims if d != time_dim]
        if reduce_dims:
            time_mask = bbox_mask.any(dim=reduce_dims).compute()
            ds_sel = ds.isel({time_dim: time_mask.values})
        else:
            ds_sel = ds.where(bbox_mask.compute(), drop=True)
        out[name_obs] = ds_sel
    return out


def compute_bbox(config, State):
    """Replicate the bbox computation done inside Obs() (with 2*d{lon,lat} pad).

    Used by callers that want to pre-select obs datasets over a tile before
    invoking Obs().
    """
    if config.EXP.lon_obs_min is not None:
        lon_obs_min = config.EXP.lon_obs_min
    else:
        lon_obs_min = State.lon_min
    if config.EXP.lon_obs_max is not None:
        lon_obs_max = config.EXP.lon_obs_max
    else:
        lon_obs_max = State.lon_max
    if config.EXP.lat_obs_min is not None:
        lat_obs_min = config.EXP.lat_obs_min
    else:
        lat_obs_min = State.lat_min
    if config.EXP.lat_obs_max is not None:
        lat_obs_max = config.EXP.lat_obs_max
    else:
        lat_obs_max = State.lat_max
    dlon = np.nanmax(State.lon[:, 1:] - State.lon[:, :-1])
    dlat = np.nanmax(State.lat[1:, :] - State.lat[:-1, :])
    return [lon_obs_min - 2 * dlon, lon_obs_max + 2 * dlon,
            lat_obs_min - 2 * dlat, lat_obs_max + 2 * dlat]


def Obs(config, State, obs_datasets=None, *args, **kwargs):
    """
    NAME
        obs

    DESCRIPTION
        Main function calling subfunctions considering the kind of satellite observations
        Args:
            config (module): configuration module
            State (class): class of model state

        Param:

        Returns:
            dict_obs (dictionary): the keys are the dates of observations, and the values are dictionaries gathering all information
            needed to assimilate these observations
    """
    
    if config.OBS is None:
        print('None observation has been provided')
        return {}
    
    if config.EXP.time_obs_min is not None:
        time_obs_min = config.EXP.time_obs_min
    else:
        time_obs_min = config.EXP.init_date
    
    if config.EXP.time_obs_max is not None:
        time_obs_max = config.EXP.time_obs_max
    else:
        time_obs_max = config.EXP.final_date
    
    if config.EXP.lon_obs_max is not None:
        lon_obs_max = config.EXP.lon_obs_max
    else:
        lon_obs_max = State.lon_max

    if config.EXP.lon_obs_min is not None:
        lon_obs_min = config.EXP.lon_obs_min
    else:
        lon_obs_min = State.lon_min
    
    if config.EXP.lat_obs_max is not None:
        lat_obs_max = config.EXP.lat_obs_max
    else:
        lat_obs_max = State.lat_max

    if config.EXP.lat_obs_min is not None:
        lat_obs_min = config.EXP.lat_obs_min
    else:
        lat_obs_min = State.lat_min
        
    date1 = time_obs_min.strftime('%Y%m%d')
    date2 = time_obs_max.strftime('%Y%m%d')
    box = f'{int(lon_obs_min)}_{int(lon_obs_max)}_{int(lat_obs_min)}_{int(lat_obs_max)}'
    
    name_dict_obs = f'dict_obs_{"_".join(config.OBS.keys())}_{date1}_{date2}_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}.txt'
    print('Observation information will be saved in',name_dict_obs)
    
    # Check if previous *dict_obs* has been computed
    if config.EXP.path_obs is None:
        path_save_obs = config.EXP.tmp_DA_path
    else:
        path_save_obs = config.EXP.path_obs
    if config.EXP.write_obs and os.path.exists(os.path.join(path_save_obs,name_dict_obs)) and not config.EXP.compute_obs:
        print(f'Reading {name_dict_obs} from previous run')
        with open(os.path.join(path_save_obs,name_dict_obs), 'rb') as f:
            dict_obs = eval(f.read())
            return _new_dict_obs(dict_obs,config.EXP.tmp_DA_path)
        
    # Read grid
    dlon = np.nanmax(State.lon[:,1:] - State.lon[:,:-1])
    dlat = np.nanmax(State.lat[1:,:] - State.lat[:-1,:])
    bbox = [lon_obs_min-2*dlon,lon_obs_max+2*dlon,lat_obs_min-2*dlat,lat_obs_max+2*dlat]
    
    # Compute output observation dictionnary
    dict_obs = {}
    assim_dates = []
    date = time_obs_min
    while date<=time_obs_max:
        assim_dates.append(date)
        date += config.EXP.assimilation_time_step
        
    for name_obs, OBS in config.OBS.items():

        print(f'\n{name_obs}:\n{OBS}')

        # Reuse pre-opened dataset if provided, otherwise open here
        _close_after = False
        if obs_datasets is not None and name_obs in obs_datasets:
            _ds = obs_datasets[name_obs]
        else:
            _ds = _open_obs_dataset(name_obs, OBS,
                                    date_start=time_obs_min,
                                    date_end=time_obs_max)
            _close_after = True
            if _ds is None:
                continue

        # Shallow copy (do NOT load() — let _obs_alti / _obs_l4 select the bbox first)
        ds = _ds.copy()
        if _close_after:
            _ds.close()
        
        # Name of obs files
        out_name = f'obs_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}'

        # Run subfunction specific to the kind of satellite
        if OBS.super in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
            _obs_alti(ds, assim_dates, dict_obs, name_obs, OBS, 
                                config.EXP.assimilation_time_step, 
                                config.EXP.tmp_DA_path,out_name,State.lon_unit,bbox)
        elif OBS.super=='OBS_L4':
            _obs_l4(ds, assim_dates, dict_obs, name_obs, OBS, 
                                config.EXP.assimilation_time_step, 
                                config.EXP.tmp_DA_path,out_name,State.lon_unit,bbox)
    
    # Write *dict_obs* for next experiment
    if config.EXP.write_obs:
        if not os.path.exists(path_save_obs):
            os.makedirs(path_save_obs)
        new_dict_obs = _new_dict_obs(dict_obs,path_save_obs)
        with open(os.path.join(path_save_obs,name_dict_obs), 'w') as f:
            f.write(str(new_dict_obs))
            
    return dict_obs

def _obs_alti(ds, dt_list, dict_obs, obs_name, obs_attr, dt_timestep, out_path, out_name, lon_unit='0_360', bbox=None):
    """
    NAME
        _obs_alti

    DESCRIPTION
        Subfunction handling observations generated from altimetric observations
        
    """

    
    ds = ds.assign_coords({obs_attr.name_time:ds[obs_attr.name_time]})
    ds = ds.swap_dims({ds[obs_attr.name_time].dims[0]:obs_attr.name_time})

    # Determine longitude conversion to apply (deferred until after bbox selection
    # to avoid materializing global SWOT arrays — e.g. (98M, 69) float64 = 50.9 GiB).
    lon_min_raw = float(ds[obs_attr.name_lon].min())
    lon_max_raw = float(ds[obs_attr.name_lon].max())
    if np.sign(lon_min_raw) == -1 and lon_unit == '0_360':
        _lon_convert = '0_360'
    elif (np.sign(lon_min_raw) >= 0 or lon_max_raw > 180) and lon_unit == '-180_180':
        _lon_convert = '-180_180'
    else:
        _lon_convert = None

    # Build a (lazy) lon view in the requested unit for the bbox mask only
    lon_obs_raw = ds[obs_attr.name_lon]
    if _lon_convert == '0_360':
        lon_obs_for_mask = lon_obs_raw % 360
    elif _lon_convert == '-180_180':
        lon_obs_for_mask = (lon_obs_raw + 180) % 360 - 180
    else:
        lon_obs_for_mask = lon_obs_raw
    lat_obs = ds[obs_attr.name_lat]
    bbox_mask = ((bbox[0] <= lon_obs_for_mask) & (bbox[1] >= lon_obs_for_mask) &
                 (bbox[2] <= lat_obs) & (bbox[3] >= lat_obs))
    # For swath data (2D lon/lat), reduce mask to the time dim and use isel,
    # to avoid xr.where(drop=True) deep-copying the full per-pixel arrays
    # (which can require tens of GiB for global SWOT).
    _reduce_dims = [d for d in bbox_mask.dims if d != obs_attr.name_time]
    if _reduce_dims:
        time_mask = bbox_mask.any(dim=_reduce_dims).compute()
        ds = ds.isel({obs_attr.name_time: time_mask.values})
    else:
        ds = ds.where(bbox_mask.compute(), drop=True)
    ds = ds.load()

    # Apply longitude conversion now (on the small per-tile subset)
    if _lon_convert == '0_360':
        ds[obs_attr.name_lon].data = ds[obs_attr.name_lon].data % 360
    elif _lon_convert == '-180_180':
        ds[obs_attr.name_lon].data = (ds[obs_attr.name_lon].data + 180) % 360 - 180
    # MDT 
    if True in [obs_attr.add_mdt, obs_attr.substract_mdt]:
        finterpmdt = read_auxdata(obs_attr.path_mdt, obs_attr.name_var_mdt, lon_unit)
    else:
        finterpmdt = None
    
    # Error file
    if obs_attr.path_err is not None:
        print('READ AUX DATA FOR OBS ERROR VARIANCES ================================',obs_name)
        finterperr = read_auxdata(obs_attr.path_err, obs_attr.name_var_err, lon_unit)
    else:
        finterperr = None

    # Time loop
    count = 0
    for dt_curr in dt_list:
    
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)

        try:
            _ds = ds.sel({obs_attr.name_time:slice(dt1,dt2)})
        except:
            try:
                _ds = ds.where((ds[obs_attr.name_time]<dt2) &\
                        (ds[obs_attr.name_time]>=dt1),drop=True)
            except:
                print(dt_curr,': Warning: impossible to select data for this time')
                continue


        lon = _ds[obs_attr.name_lon].values
        lat = _ds[obs_attr.name_lat].values

        is_obs = np.any(~np.isnan(lon.ravel()*lat.ravel())) * (lon.size>0)

        if is_obs:
            # Save the selected dataset in a new nc file
            varobs = {}
            for name in obs_attr.name_var:
                
                if isinstance(obs_attr.name_var[name], list):
                    _var = 0
                    for i,name_var in enumerate(obs_attr.name_var[name]):
                        if obs_attr.combine_var is not None and name in obs_attr.combine_var:
                            sign = obs_attr.combine_var[name][i]
                        _var += sign * _ds[name_var]
                    varobs[name] = _var
                else:
                    varobs[name] = _ds[obs_attr.name_var[name]]

                # Add/Remove MDT
                if finterpmdt is not None:
                    mdt_on_obs = finterpmdt((lon,lat))
                    if obs_attr.add_mdt:
                        sign = 1
                    else:
                        sign = -1
                    varobs[name].data = varobs[name].data + sign*mdt_on_obs
                # Add synthetic noise to the data
                if 'synthetic_noise' in obs_attr and obs_attr.synthetic_noise is not None:
                    varobs[name].data = varobs[name].data + np.random.normal(0,obs_attr.synthetic_noise,varobs[name].size).reshape(varobs[name].shape) 
                # Remove high values
                if 'varmax' in obs_attr and obs_attr.varmax is not None:
                    varobs[name] = varobs[name].where(np.abs(varobs[name]) <= obs_attr.varmax, np.nan)
                # Subsampling
                if obs_attr.subsampling is not None:
                    d = {}
                    for dim in varobs[name].dims:
                        d[dim] = obs_attr.subsampling
                    varobs[name] = varobs[name].coarsen(d,boundary='trim').mean()
                # Error
                if finterperr is not None:
                    err_on_obs = finterperr((lon,lat))
                    varobs[name + '_err'] = varobs[name].copy()
                    varobs[name + '_err'].data = err_on_obs

                    if obs_attr.facR is not None: 
                        print('FacR implemented', obs_attr.facR)
                        varobs[name + '_err'].data *= obs_attr.facR 

            # Build netcdf
            coords = {}
            name_coords = [obs_attr.name_time,obs_attr.name_lon,obs_attr.name_lat]
            if obs_attr.super=='OBS_SSH_SWATH' and obs_attr.name_xac is not None:
                name_coords.append(obs_attr.name_xac)
            for name in name_coords:
                if obs_attr.subsampling is not None:
                    d = {}
                    for dim in _ds[name].dims:
                        d[dim] = obs_attr.subsampling
                    coords[name] = _ds[name].coarsen(d,boundary='trim').mean()
                else:
                    coords[name] = _ds[name]
            dsout = xr.Dataset(varobs,
                                coords=coords
                                )

            # Write netcdf
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = f"{out_path}/{out_name}_{obs_name}_{'_'.join(obs_attr.name_var)}_{date}"
            if finterpmdt is not None:
                if obs_attr.add_mdt:
                    path += '_addmdt'
                elif obs_attr.substract_mdt:
                    path += '_submdt'
            path += '.nc'
            dsout.to_netcdf(path, encoding={obs_attr.name_time: {'_FillValue': None},
                                            obs_attr.name_lon: {'_FillValue': None},
                                            obs_attr.name_lat: {'_FillValue': None}})
            dsout.close()
            _ds.close()
            del dsout,_ds
            
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                dict_obs[dt_curr]['obs_name'].append(obs_name)
                dict_obs[dt_curr]['obs_path'].append(path)
                dict_obs[dt_curr]['attributes'].append(obs_attr)
            else:
                dict_obs[dt_curr] = Config({})
                dict_obs[dt_curr]['obs_name'] = [obs_name]
                dict_obs[dt_curr]['obs_path'] = [path]
                dict_obs[dt_curr]['attributes'] = [obs_attr]
                
            count +=1

        
    print(f'--> {count} tracks selected')
    
def _obs_l4(ds, dt_list, dict_obs, obs_name, obs_attr, dt_timestep, out_path, out_name, lon_unit='0_360', bbox=None):
    
    ds = ds.assign_coords({obs_attr.name_time:ds[obs_attr.name_time]})
    ds = ds.swap_dims({ds[obs_attr.name_time].dims[0]:obs_attr.name_time})

    # Subsampling
    if obs_attr.subsampling is not None:
        ds = ds.isel({obs_attr.name_time:slice(None,None,obs_attr.subsampling)})
    
    # Convert longitude
    if np.sign(ds[obs_attr.name_lon].data.min())==-1 and lon_unit=='0_360':
            ds = ds.assign_coords({obs_attr.name_lon:((obs_attr.name_lon, ds[obs_attr.name_lon].data % 360))})
    elif np.sign(ds[obs_attr.name_lon].data.min())>=0 and lon_unit=='-180_180':
        ds = ds.assign_coords({obs_attr.name_lon:((obs_attr.name_lon, (ds[obs_attr.name_lon].data + 180) % 360 - 180))})
    
    
    # Select sub area
    lon_obs = ds[obs_attr.name_lon] 
    lat_obs = ds[obs_attr.name_lat]
    ds = ds.where(((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs)).compute(), drop=True)
    ds = ds.load()

    lon_obs = ds[obs_attr.name_lon].values
    lat_obs = ds[obs_attr.name_lat].values
    if len(lon_obs.shape)==1:
        lon_obs,lat_obs = np.meshgrid(lon_obs,lat_obs)

    # Time loop
    count = 0
    for dt_curr in dt_list:
        
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)
       
        try:
            _ds = ds.sel({obs_attr.name_time:slice(dt1,dt2)})
        except:
            try:
                _ds = ds.where(((ds[obs_attr.name_time]<dt2) &\
                        (ds[obs_attr.name_time]>=dt1)).compute(),drop=True)
            except:
                print(dt_curr,': Warning: impossible to select data for this time')
                continue
        
        if _ds[obs_attr.name_time].size>0:
            # Time mean if several timestep selected
            if _ds[obs_attr.name_time].size>1:
                _ds = _ds.mean(dim=obs_attr.name_time)
            # Read variables
            varobs = {}
            for name in obs_attr.name_var:
                # Observed variable
                varobs[name] = (('y','x'), _ds[obs_attr.name_var[name]].data.squeeze())
                # Error variable
                if obs_attr.name_err is not None and name in obs_attr.name_err:
                    varobs[name+'_err'] = (('y','x'), _ds[obs_attr.name_err[name]].data.squeeze())
            # Coords
            varobs[obs_attr.name_lon] = (('y','x'), lon_obs)
            varobs[obs_attr.name_lat] = (('y','x'), lat_obs)

            # Offset
            if 'offset' in obs_attr and obs_attr.offset is not None:
                for name in obs_attr.name_var:
                    if isinstance(obs_attr.offset,dict) and name in obs_attr.offset:
                        varobs[name] = (('y','x'), varobs[name][1] + obs_attr.offset[name])
                    elif isinstance(obs_attr.offset,(int,float)):
                        varobs[name] = (('y','x'), varobs[name][1] + obs_attr.offset)
                    else:
                        print('Warning: offset should be a number or a dictionary with variable names as keys')
            
            # Save to netcdf
            dsout = xr.Dataset(varobs)
            
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = f"{out_path}/{out_name}_{obs_name}_{'_'.join(obs_attr.name_var)}_{date}.nc"
            dsout.to_netcdf(path, encoding={obs_attr.name_lon: {'_FillValue': None},
                                            obs_attr.name_lat: {'_FillValue': None}})
            dsout.close()
            _ds.close()
            del dsout,_ds
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                dict_obs[dt_curr]['obs_name'].append(obs_name)
                dict_obs[dt_curr]['obs_path'].append(path)
                dict_obs[dt_curr]['attributes'].append(obs_attr)
            else:
                dict_obs[dt_curr] = Config({})
                dict_obs[dt_curr]['obs_name'] = [obs_name]
                dict_obs[dt_curr]['obs_path'] = [path]
                dict_obs[dt_curr]['attributes'] = [obs_attr]
            
            count +=1
    
    print(f'--> {count} fields selected')

def _new_dict_obs(dict_obs, new_dir, date_min=None, date_max=None):
    """
    NAME
        _new_dict_obs

    DESCRIPTION
        Subfunction creating a new dict_obs, similar as dict_obs, except that 
        the obs files are stored in *new_dir*
        
        Args: 
            dict_obs(dict): initial dictionary
            new_dir(str): new directory where the obs will be copied
        Returns:
            new_dict_obs (dictionary)
    """
    
    new_dict_obs = {}
    for date in dict_obs:
        if date_min is not None and date<date_min:
            continue
        if date_max is not None and date>date_max:
            continue
        # Create new dict_obs by copying the obs files in *new_dir* directory 
        new_dict_obs[date] = {'obs_name':[],'obs_path':[],'attributes':[]}
        for obs_name,obs_path,attributes in zip(dict_obs[date]['obs_name'],dict_obs[date]['obs_path'],dict_obs[date]['attributes']):
            file_obs = os.path.basename(obs_path)
            new_obs_path = os.path.join(new_dir,file_obs)
            # Copy to *tmp_DA_path* directory
            if os.path.normpath(obs_path)!=os.path.normpath(new_obs_path): 
                os.system(f'cp {obs_path} {new_obs_path}')
            # Update new dictionary 
            new_dict_obs[date]['obs_name'].append(obs_name)
            new_dict_obs[date]['obs_path'].append(new_obs_path)
            new_dict_obs[date]['attributes'].append(Config(attributes))
            
    return new_dict_obs

def detrend_obs(dict_obs):

    sys.exit('obs.detrend is depreciated')
    
    for t in dict_obs:
        # Read obs
        sat_info_list = dict_obs[t]['satellite']
        obs_file_list = dict_obs[t]['obs_name']
        for sat_info,obs_file in zip(sat_info_list,obs_file_list):
            # Read obs file
            ncin = xr.open_dataset(obs_file)
            ncout = ncin.copy().load()
            ncin.close()
            del ncin
            # Load ssh
            ssh = ncout[sat_info.name_obs_var[0]].squeeze().values
            # Fill Masked pixels 
            mask = np.isnan(ssh)
            ssh[mask] = 0
            # Detrend data in all directions
            if len(ssh.shape)==0:
                ssh_detrended = +ssh
            elif len(ssh.shape)==1:
                ssh_detrended = signal.detrend(ssh)
            else:
                ssh_detrended = detrendn(ssh)
            # Re-mask
            if mask.size>1:
                ssh_detrended[mask] = np.nan
            # Write detrended observation
            ncout[sat_info.name_obs_var[0]].data = ssh_detrended.reshape(ncout[sat_info.name_obs_var[0]].shape)
            ncout.to_netcdf(obs_file)
            ncout.close()
            del ncout
                      
def get_obs(dict_obs,box,time_init,name_var='SSH'):

        lon0 = box[0]
        lon1 = box[1]
        lat0 = box[2]
        lat1 = box[3]
        
        time0 = box[4]
        time1 = box[5]

        lon = np.array([])
        lat = np.array([])
        time = np.array([])
        var = np.array([])
        
        for dt in dict_obs:
            
            if (dt<=time1) & (dt>=time0):
                
                    path_obs = dict_obs[dt]['obs_path']
                    attrs =  dict_obs[dt]['attributes']

                    for _attrs,_path_obs in zip(attrs,path_obs):
                        
                        ds = xr.open_dataset(_path_obs).squeeze() 
                        
                        if name_var not in ds.variables:
                            print(f'Warning: {name_var} not in {_path_obs}')
                            continue

                        lon_obs = ds[_attrs.name_lon] % 360
                        lat_obs = ds[_attrs.name_lat]
                        
                        ds = ds.where(((lon0<=lon_obs) & (lon1>=lon_obs) & 
                  (lat0<=lat_obs) & (lat1>=lat_obs)).compute(), drop=True)
                        time_obs = ds[_attrs.name_time].values
                        time_obs = (time_obs-np.datetime64(time_init))/np.timedelta64(1, 'D')

                        if _attrs.super=='OBS_L4':
                            if len(ds[_attrs.name_lon].shape)==1:
                                lon_obs = ds[_attrs.name_lon].values
                                lat_obs = ds[_attrs.name_lat].values
                                lon_obs,lat_obs = np.meshgrid(lon_obs,lat_obs)
                            else:
                                lon_obs = ds[_attrs.name_lon].values
                                lat_obs = ds[_attrs.name_lat].values
                            var_obs = ds[name_var].values
                            time_obs = time_obs * np.ones_like(var_obs)
                        
                        elif _attrs.super in ['OBS_SSH_NADIR','OBS_SSH_SWATH']:
                            lon_obs = ds[_attrs.name_lon].values
                            lat_obs = ds[_attrs.name_lat].values
                            var_obs = ds[name_var].values
                            if len(var_obs.shape)==2:
                                # SWATH data
                                if var_obs.shape[0]==time_obs.size:
                                    dim = 1
                                else:
                                    dim = 0
                                time_obs = time_obs.repeat(var_obs.shape[dim],axis=0)
                        ds.close()
                        del ds
                        
                        # Flattening
                        time1d = time_obs.ravel()
                        lon1d = lon_obs.ravel()
                        lat1d = lat_obs.ravel()
                        var1d = var_obs.ravel()

                        # Remove NaN pixels
                        indNoNan= ~np.isnan(var1d)
                        time1d = time1d[indNoNan]
                        lon1d = lon1d[indNoNan]
                        lat1d = lat1d[indNoNan]
                        var1d = var1d[indNoNan]    
                        
                        # Append to arrays
                        time = np.append(time,time1d)
                        lon = np.append(lon,lon1d)
                        lat = np.append(lat,lat1d)
                        var = np.append(var,var1d)
        
        coords = [None]*3
        coords_att = { 'lon':0, 'lat':1, 'time':2, 'nobs':len(time) }

        if len(time)>0:
            indsort = np.argsort(time)
            if len(indsort)>0:
                lon=lon[indsort]   
                lat=lat[indsort]
                time=time[indsort]
                var=var[indsort]

            coords[coords_att['lon']] = lon
            coords[coords_att['lat']] = lat
            coords[coords_att['time']] = time      
        
        return [var, coords, coords_att]
