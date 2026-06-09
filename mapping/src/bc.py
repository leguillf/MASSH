#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:49:01 2020

@author: leguillou
"""
import sys
import xarray as xr 
import numpy as np 
import pyinterp 
import pyinterp.fill
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd
from . import grid

import matplotlib.pylab as plt

def Bc(config, State=None, verbose=1, *args, **kwargs):
    """
    NAME
        Bc

    DESCRIPTION
        Main function for specific boundary conditions
    """
    
    if config.BC is None:
        return 
    
    elif config.BC.super is None:
        return Bc_multi(config,State=State,verbose=verbose)
    
    if verbose:
        print(config.BC)
    
    if config.BC.super=='BC_EXT':
        return Bc_ext(config,State=State)
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')


class Bc_ext:

    def __init__(self,config, State=None):

        if State is None:
            from . import state
            State = state.State(config, verbose=False)

        # Get grid coordinates
        self.lon = State.lon
        self.lat = State.lat
        self.mask = State.mask

        # Study domain borders
        lon_min = np.nanmin(self.lon)
        lon_max = np.nanmax(self.lon)
        lat_min = np.nanmin(self.lat)
        lat_max = np.nanmax(self.lat)

        # Grid spacing
        dlon = np.nanmean(self.lon[:,1:]-self.lon[:,:-1])
        dlat = np.nanmean(self.lat[1:,:]-self.lat[:-1,:])

        # Read netcdf
        _ds = xr.open_mfdataset(config.BC.file)

        # Copy dataset
        ds = _ds.copy()#.load()
        _ds.close()

        # Convert longitude 
        try:
            if np.sign(ds[config.BC.name_lon].data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, ds[config.BC.name_lon].data % 360))})
            elif np.sign(ds[config.BC.name_lon].data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({config.BC.name_lon:((config.BC.name_lon, (ds[config.BC.name_lon].data + 180) % 360 - 180 ))})
            ds = ds.sortby(ds[config.BC.name_lon])    
        except:
            print("Warning: can't convert longitude coordinates in " + config.BC.file)
        
        # Select study domain
        time_bc = ds[config.BC.name_time].values
        self.name_time_bc = config.BC.name_time
        lon_bc = ds[config.BC.name_lon].values
        lat_bc = ds[config.BC.name_lat].values
        dlon += np.nanmean(lon_bc[1:]-lon_bc[:-1])
        dlat += np.nanmean(lat_bc[1:]-lat_bc[:-1])
        dtime = time_bc[1]-time_bc[0]
        try:
            ds = ds.sel({
                config.BC.name_time:slice(np.datetime64(config.EXP.init_date)-dtime,np.datetime64(config.EXP.final_date)+dtime),
                })
        except:
            ds = ds.where(
                (ds[config.BC.name_time] >= np.datetime64(config.EXP.init_date)-dtime) & (ds[config.BC.name_time] <= np.datetime64(config.EXP.final_date)+dtime),
                drop=True
            )
        if len(ds[config.BC.name_lon].shape)==1 and len(ds[config.BC.name_lat].shape)==1:
            ds = ds.sel({
                config.BC.name_lon:slice(lon_min-2*dlon,lon_max+2*dlon),
                config.BC.name_lat:slice(lat_min-2*dlat,lat_max+2*dlat),
                })
        
        self.c_grid = config.BC.c_grid if 'c_grid' in config.BC else False
        
        # Get BC coordinates
        self.lon_bc = ds[config.BC.name_lon].values
        self.lat_bc = ds[config.BC.name_lat].values
        if config.BC.name_time is not None:
            self.time_bc = ds[config.BC.name_time].values
        else:
            self.time_bc = None

        # Get BC variables
        self.var = {}
        for name in config.BC.name_var:
            self.var[name] = ds[config.BC.name_var[name]].load()
        ds.close()        
    
    def interp(self, time):
        """
        Interpolate boundary conditions on the model grid
        """
        
        # Check time dimension
        if self.time_bc is not None and self.time_bc.size>1:
            if len(time.shape)==0:
                time = np.array([time])
            elif len(time.shape)==1:
                time = np.ascontiguousarray(time)
            else:
                sys.exit('Time dimension must be 1D or 0D')
        
        # Convert time to np.datetime64
        if time.size>1 and type(time[0])!=np.datetime64:
            time = np.array([np.datetime64(dt) for dt in time])
        elif time.size==1 and type(time)!=np.datetime64:
            time = np.array([np.datetime64(time)])
        
        # Interpolate
        if len(self.lon_bc.shape)==1 and len(self.lat_bc.shape)==1:
            var_interp = self._interp_3D(time)
        elif len(self.lon_bc.shape)==2 and len(self.lat_bc.shape)==2 and np.all(self.lon_bc==self.lon) and np.all(self.lat_bc==self.lat):
            var_interp = self._interp_1D(time)
        else:
            sys.exit('Boundary conditions coordinates must be 1D, 2D. If 2D, they must match the model grid coordinates')

        return var_interp
            
    def _interp_3D(self,time):
        """
        Interpolate boundary conditions on the model grid. It works only if the boundary conditions grid is regular.

        Strategy for time-varying sources (large global datasets): interpolate
        linearly in time first (cheap weighted sum of the two bracketing source
        slices, with edge clipping), then run a 2D bivariate spatial interpolation
        per output timestep. This avoids building (nx,ny,nt) target arrays and
        avoids running gauss_seidel on the full 3D source cube.
        """

        has_time = self.time_bc is not None and self.time_bc.size > 1

        # Source spatial axes (shared by all variables)
        x_source_axis = pyinterp.Axis(self.lon_bc, is_circle=True)
        y_source_axis = pyinterp.Axis(self.lat_bc)

        # Precompute time-interpolation weights (linear, with edge clipping)
        if has_time:
            tbc = self.time_bc
            # Bracketing indices i0, i0+1 such that tbc[i0] <= time <= tbc[i0+1]
            i0 = np.clip(np.searchsorted(tbc, time, side='right') - 1, 0, len(tbc) - 2)
            i1 = i0 + 1
            t0f = tbc[i0].astype('datetime64[ns]').astype(np.float64)
            t1f = tbc[i1].astype('datetime64[ns]').astype(np.float64)
            ttf = time.astype('datetime64[ns]').astype(np.float64)
            denom = np.where(t1f > t0f, t1f - t0f, 1.0)
            w1 = (ttf - t0f) / denom
            # Edge clipping: replicate nearest source slice
            below = time < tbc[0]
            above = time > tbc[-1]
            w1 = np.where(below, 0.0, w1)
            w1 = np.where(above, 1.0, w1)
            w0 = 1.0 - w1
            needed = np.unique(np.concatenate([i0, i1]))
            pos = {int(k): n for n, k in enumerate(needed)}

        # Interpolation
        var_interp = {}
        for name in self.var:
            # Build target (lon_t, lat_t, mask_t) for this variable
            if not self.c_grid or name not in ('U', 'V'):
                lon_t = self.lon
                lat_t = self.lat
                mask_t = self.mask
            elif name == 'U':
                lon_t = np.zeros((self.lon.shape[0], self.lon.shape[1] + 1))
                lat_t = np.zeros((self.lat.shape[0], self.lat.shape[1] + 1))
                lon_t[:, 1:-1] = (self.lon[:, 1:] + self.lon[:, :-1]) / 2.
                lon_t[:, 0]    = self.lon[:, 0]  - (self.lon[:, 1]  - self.lon[:, 0])  / 2.
                lon_t[:, -1]   = self.lon[:, -1] + (self.lon[:, -1] - self.lon[:, -2]) / 2.
                lat_t[:, 1:-1] = (self.lat[:, 1:] + self.lat[:, :-1]) / 2.
                lat_t[:, 0]    = self.lat[:, 0]  - (self.lat[:, 1]  - self.lat[:, 0])  / 2.
                lat_t[:, -1]   = self.lat[:, -1] + (self.lat[:, -1] - self.lat[:, -2]) / 2.
                mask_t = np.zeros((self.mask.shape[0], self.mask.shape[1] + 1), dtype=bool)
                mask_t[:, 1:-1] = self.mask[:, :-1] | self.mask[:, 1:]
                mask_t[:, 0]    = self.mask[:, 0]
                mask_t[:, -1]   = self.mask[:, -1]
            else:  # 'V'
                lon_t = np.zeros((self.lat.shape[0] + 1, self.lat.shape[1]))
                lat_t = np.zeros((self.lat.shape[0] + 1, self.lat.shape[1]))
                lon_t[1:-1, :] = (self.lon[1:, :] + self.lon[:-1, :]) / 2.
                lon_t[0, :]    = self.lon[0, :]  - (self.lon[1, :]  - self.lon[0, :])  / 2.
                lon_t[-1, :]   = self.lon[-1, :] + (self.lon[-1, :] - self.lon[-2, :]) / 2.
                lat_t[1:-1, :] = (self.lat[1:, :] + self.lat[:-1, :]) / 2.
                lat_t[0, :]    = self.lat[0, :]  - (self.lat[1, :]  - self.lat[0, :])  / 2.
                lat_t[-1, :]   = self.lat[-1, :] + (self.lat[-1, :] - self.lat[-2, :]) / 2.
                mask_t = np.zeros((self.mask.shape[0] + 1, self.mask.shape[1]), dtype=bool)
                mask_t[1:-1, :] = self.mask[:-1, :] | self.mask[1:, :]
                mask_t[0, :]    = self.mask[0, :]
                mask_t[-1, :]   = self.mask[-1, :]

            # Flat target coords (computed once per variable, shared across all t)
            out_shape = lon_t.shape  # (ny_t, nx_t)
            x_flat = lon_t.T.ravel()
            y_flat = lat_t.T.ravel()

            if not has_time:
                grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis,
                                              np.asarray(self.var[name]).T)
                _slice = pyinterp.bivariate(grid_source, x_flat, y_flat,
                                            bounds_error=False).reshape(out_shape[::-1]).T
                _slice[mask_t] = np.nan
                var_interp[name] = np.repeat(_slice[np.newaxis, :, :], len(time), axis=0)
                continue

            # Time-varying: load only the source slices we actually need
            da = self.var[name].squeeze()
            src = np.asarray(da.isel({self.name_time_bc: needed}).values, dtype=np.float64)
            if src.ndim == 2:
                src = src[np.newaxis]
            # src shape: (n_needed, ny_src, nx_src)

            # Fill NaN once per unique source slice (cheap 2D gauss_seidel)
            filled = np.empty_like(src)
            for k in range(src.shape[0]):
                sl = src[k]
                if np.isnan(sl).any():
                    g2 = pyinterp.Grid2D(x_source_axis, y_source_axis, sl.T.copy())
                    _, sl_filled = pyinterp.fill.gauss_seidel(g2)
                    filled[k] = sl_filled.T
                else:
                    filled[k] = sl

            nt_out = len(time)
            n_needed = src.shape[0]
            result = np.empty((nt_out,) + out_shape, dtype=np.float64)

            if nt_out > n_needed:
                # Cheaper: spatially interpolate each unique source slice ONCE,
                # then linearly blend in time on the target grid (pure NumPy).
                spatial = np.empty((n_needed,) + out_shape, dtype=np.float64)
                for k in range(n_needed):
                    g2 = pyinterp.Grid2D(x_source_axis, y_source_axis,
                                         np.ascontiguousarray(filled[k].T))
                    spatial[k] = pyinterp.bivariate(
                        g2, x_flat, y_flat, bounds_error=False
                    ).reshape(out_shape[::-1]).T
                for t in range(nt_out):
                    k0 = pos[int(i0[t])]
                    k1 = pos[int(i1[t])]
                    if k0 == k1 or w1[t] == 0.0:
                        result[t] = spatial[k0]
                    elif w1[t] == 1.0:
                        result[t] = spatial[k1]
                    else:
                        result[t] = w0[t] * spatial[k0] + w1[t] * spatial[k1]
                    result[t][mask_t] = np.nan
            else:
                # Fewer output timesteps than unique source slices: blend in time
                # first (smaller source grid), then one bivariate per output step.
                for t in range(nt_out):
                    k0 = pos[int(i0[t])]
                    k1 = pos[int(i1[t])]
                    if k0 == k1 or w1[t] == 0.0:
                        blend = filled[k0]
                    elif w1[t] == 1.0:
                        blend = filled[k1]
                    else:
                        blend = w0[t] * filled[k0] + w1[t] * filled[k1]
                    g2 = pyinterp.Grid2D(x_source_axis, y_source_axis,
                                         np.ascontiguousarray(blend.T))
                    interp = pyinterp.bivariate(
                        g2, x_flat, y_flat, bounds_error=False
                    ).reshape(out_shape[::-1]).T
                    interp[mask_t] = np.nan
                    result[t] = interp

            var_interp[name] = result

        return var_interp
    
    def _interp_1D(self, time):
        """
        Interpolate boundary conditions in time only (spatial grid matches model grid)
        """
        var_interp = {}
        # Ensure time is array
        if len(time.shape) == 0:
            time = np.array([time])
        elif len(time.shape) == 1:
            time = np.ascontiguousarray(time)
        else:
            sys.exit('Time dimension must be 1D or 0D')

        for name in self.var:
            var_data = self.var[name]
            time_bc = self.time_bc

            nt = len(time)
            out_shape = (nt,) + var_data.shape[1:]

            if time_bc is None or np.size(time_bc) <= 1:
                # Repeat the same array for each requested time
                interp_data = np.repeat(var_data.values[np.newaxis, ...], nt, axis=0)
            else:
                # Use xarray's interp method for time interpolation
                interp_result = var_data.interp({self.name_time_bc: xr.DataArray(time, dims=[self.name_time_bc])}, method="linear")
                interp_data = interp_result.values

            # Mask if needed
            if hasattr(self, 'mask'):
                mask = self.mask
                if mask.shape == interp_data.shape[1:]:
                    interp_data[:, mask] = np.nan

            var_interp[name] = interp_data

        return var_interp


class Bc_multi:

    def __init__(self,config,State=None,verbose=1):

        self.Bc = []
        _config = config.copy()

        for name_bc in config.BC:
            _config.BC = config.BC[name_bc]
            _Bc = Bc(_config,State=State,verbose=verbose)
            self.Bc.append(_Bc)

    
    def interp(self,time):

        var_interp = {}

        for _Bc in self.Bc:

            _var_interp = _Bc.interp(time)

            for name in _var_interp:
                var_interp[name] = _var_interp[name]

        return var_interp

