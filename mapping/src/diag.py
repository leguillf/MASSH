import os, sys
import glob
import numpy as np
import xarray as xr
import pyinterp 
import pyinterp.fill
from scipy.interpolate import griddata
import logging
import scipy
import xrft
import netCDF4
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from IPython.display import Video
from matplotlib.ticker import ScalarFormatter
import gc
import pandas as pd 
import subprocess
from .tools import read_auxdata
from . import grid
from . import switchvar
import cmocean


def Diag(config,State):

    """
    NAME
        Diag

    DESCRIPTION
        Main function calling subclass for specific diagnostics
    """

    if config.DIAG is None:
        return
    
    elif config.DIAG.super is None:
        return Diag_multi(config,State)
    
    print(config.DIAG)


    if config.DIAG.super=='DIAG_OSSE':
        return Diag_osse(config,State)
    if config.DIAG.super=='DIAG_OSSE_UV':
        return Diag_osse_uv(config,State)
    elif config.DIAG.super=='DIAG_OSE':
        return Diag_ose(config,State)
    else:
        sys.exit(config.DIAG.super + ' not implemented yet')

class Diag_osse():

    """
    NAME
        Diag_osse

    DESCRIPTION
        Class to compute OSSE diagnostics
    """

    def __init__(self,config,State):
        
        # name_experiment
        self.name_experiment = config.EXP.name_experiment
        # dir_output
        if config.DIAG.dir_output is None:
            self.dir_output = f'{config.EXP.path_save}/diags/'
        else:
            self.dir_output = config.DIAG.dir_output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        # time_min
        if config.DIAG.time_min is None:
            self.time_min = config.EXP.init_date
        else:
            self.time_min = config.DIAG.time_min
        # time_max
        if config.DIAG.time_max is None:
            self.time_max = config.EXP.final_date
        else:
            self.time_max = config.DIAG.time_max
        # time_step
        if config.DIAG.time_step is None:
            self.time_step = config.EXP.saveoutput_time_step
        else:
            self.time_step = config.DIAG.time_step
        # lon_min
        if config.DIAG.lon_min is None:
            self.lon_min = config.GRID.lon_min
        else:
            self.lon_min = config.DIAG.lon_min
        # lon_max
        if config.DIAG.lon_max is None:
            self.lon_max = config.GRID.lon_max
        else:
            self.lon_max = config.DIAG.lon_max
        # lat_min
        if config.DIAG.lat_min is None:
            self.lat_min = config.GRID.lat_min
        else:
            self.lat_min = config.DIAG.lat_min
        # lat_max
        if config.DIAG.lat_max is None:
            self.lat_max = config.GRID.lat_max
        else:
            self.lat_max = config.DIAG.lat_max


        # Reference data
        self.name_ref_time = config.DIAG.name_ref_time
        self.name_ref_lon = config.DIAG.name_ref_lon
        self.name_ref_lat = config.DIAG.name_ref_lat
        self.name_ref_var = config.DIAG.name_ref_var
        ref = xr.open_mfdataset(config.DIAG.name_ref,**config.DIAG.options_ref, preprocess=lambda ds: ds[[self.name_ref_var]]).squeeze()
        if np.sign(ref[self.name_ref_lon].data.min())==-1 and State.lon_unit=='0_360':
            ref = ref.assign_coords({self.name_ref_lon:((self.name_ref_lon, ref[self.name_ref_lon].data % 360))})
        elif np.sign(ref[self.name_ref_lon].data.min())==1 and State.lon_unit=='-180_180':
            ref = ref.assign_coords({self.name_ref_lon:((self.name_ref_lon, (ref[self.name_ref_lon].data + 180) % 360 - 180))})
        ref = ref.sortby(ref[self.name_ref_lon])    
        dt = (ref[self.name_ref_time][1]-ref[self.name_ref_time][0]).values/np.timedelta64(1,'s')
        idt = max(int(self.time_step.total_seconds()//dt), 1)
        self.ref = ref.sel(
            {self.name_ref_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max),idt)}
             )
        try:
            self.ref = self.ref.sel(
                {self.name_ref_lon:slice(self.lon_min,self.lon_max),
                self.name_ref_lat:slice(self.lat_min,self.lat_max)}
            )
        except:
            print('Warning: unable to select study region in the reference fields.\
That could be due to non regular grid or bad written netcdf file')
        ref.close()
        self.ref = self.ref.load()

        # Experimental data
        self.geo_grid = State.geo_grid
        self.name_exp_time = config.EXP.name_time
        self.name_exp_lon = config.EXP.name_lon
        self.name_exp_lat = config.EXP.name_lat
        self.name_exp_var = config.DIAG.name_exp_var
        exp = xr.open_mfdataset(f'{config.EXP.path_save}/{config.EXP.name_exp_save}*nc',preprocess=lambda ds: ds[[self.name_exp_var]])
        exp = exp.assign_coords({self.name_exp_lon:exp[self.name_exp_lon]})
        dt = (exp[self.name_exp_time][1]-exp[self.name_exp_time][0]).values
        self.exp = exp.sel(
            {self.name_exp_time:slice(np.datetime64(self.time_min)-dt,np.datetime64(self.time_max)+dt)},
             )
        exp.close()
        self.exp = self.exp.load()

        # Baseline data
        self.compare_to_baseline = config.DIAG.compare_to_baseline 
        if self.compare_to_baseline:
            self.name_bas_time = config.DIAG.name_bas_time
            self.name_bas_lon = config.DIAG.name_bas_lon
            self.name_bas_lat = config.DIAG.name_bas_lat
            self.name_bas_var = config.DIAG.name_bas_var
            try:
                bas = xr.open_mfdataset(config.DIAG.name_bas,preprocess=lambda ds: ds[[self.name_bas_var]])
            except:
                bas = xr.open_mfdataset(config.DIAG.name_bas)
            if np.sign(bas[self.name_bas_lon].data.min())==-1 and State.lon_unit=='0_360':
                bas = bas.assign_coords({self.name_bas_lon:((self.name_bas_lon, bas[self.name_bas_lon].data % 360))})
            elif np.sign(bas[self.name_bas_lon].data.min())==1 and State.lon_unit=='-180_180':
                bas = bas.assign_coords({self.name_bas_lon:((self.name_bas_lon, (bas[self.name_bas_lon].data + 180) % 360 - 180))})
            try:
                bas = bas.sortby(bas[self.name_bas_lon])    
            except:
                print(end='\r')
            self.bas = bas.sel(
                {self.name_bas_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max))},
                )
            try:
                dlon = self.bas[self.name_bas_lon[1:]] - self.bas[self.name_bas_lon[:-1]]
                dlat = self.bas[self.name_bas_lat[1:]] - self.bas[self.name_bas_lat[:-1]]
                self.bas = self.bas.sel(
                    {self.name_bas_lon:slice(self.lon_min-dlon,self.lon_max+dlon),
                    self.name_bas_lat:slice(self.lat_min-dlat,self.lat_max+dlat)}
                )
            except:
                print('Warning: unable to select study region in the baseline fields.')
            bas.close()
            self.bas = self.bas.load()

            if len(self.bas[self.name_bas_lon].shape)==1:
                self.geo_grid_bas = True
            else:
                self.geo_grid_bas = False
        
        # Mask
        if config.DIAG.name_mask is not None:
            name_lon = config.DIAG.name_var_mask['lon']
            name_lat = config.DIAG.name_var_mask['lat']
            name_var = config.DIAG.name_var_mask['var']
            mask = xr.open_mfdataset(config.DIAG.name_mask)
            if np.sign(mask[name_lon].data.min())==-1 and State.lon_unit=='0_360':
                mask = mask.assign_coords({name_lon:((name_lon, mask[name_lon].data % 360))})
            elif np.sign(mask[name_lon].data.min())==1 and State.lon_unit=='-180_180':
                mask = mask.assign_coords({name_lon:((name_lon, (mask[name_lon].data + 180) % 360 - 180))})
            mask = mask.sortby(mask[name_lon])  
            mask_interp = self._regrid_geo(
                mask[name_lon].values,
                mask[name_lat].values, 
                None, 
                mask[name_var],
                remove_NaN=False)   
            if mask_interp.dtype!=bool : 
                self.mask = np.empty(mask_interp.shape,dtype='bool')
                ind_mask = (np.isnan(mask_interp)) | (mask_interp==1) | (np.abs(mask_interp)>10)
                self.mask[ind_mask] = True
                self.mask[~ind_mask] = False
            else:
                self.mask = mask_interp.copy()       
        else:
            self.mask = None
    
    def regrid_exp(self):
        
        if self.geo_grid:
            self.exp_regridded =  self._regrid_geo(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp[self.name_exp_var])
        else:
            self.exp_regridded = self._regrid_unstructured(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp[self.name_exp_var])
        if self.mask is not None:
            self.exp_regridded.data[:,self.mask] = np.nan
        else:
            self.exp_regridded.data[np.isnan(self.ref[self.name_ref_var].data)] = np.nan
        
        if self.compare_to_baseline:
            if self.geo_grid_bas:
                self.bas_regridded = self._regrid_geo(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas[self.name_bas_var])    
            else:
                self.bas_regridded = self._regrid_unstructured(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas[self.name_bas_var])    
            if self.mask is not None:
                self.bas_regridded.data[:,self.mask] = np.nan
            else:
                self.bas_regridded.data[np.isnan(self.ref[self.name_ref_var].data)] = np.nan
    
    def _regrid_geo(self, lon, lat, time, var, remove_NaN=True):

        # Define source grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        if time is not None:
            z_source_axis = pyinterp.TemporalAxis(time)
        ssh_source = var.T
        if time is not None:    
            grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
        else:
            grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, ssh_source.data)
        
        # Define target grid
        if time is not None:    
            mx_target, my_target, mz_target = np.meshgrid(
                self.ref[self.name_ref_lon][:].values.flatten(),
                self.ref[self.name_ref_lat][:].values.flatten(),
                z_source_axis.safe_cast(np.ascontiguousarray(self.ref[self.name_ref_time][:].values)),
                indexing="ij")
        else:
            mx_target, my_target = np.meshgrid(
                self.ref[self.name_ref_lon][:].values.flatten(),
                self.ref[self.name_ref_lat][:].values.flatten(),
                indexing="ij")

        # Spatio-temporal Interpolation
        if time is not None:  
            ssh_interp = pyinterp.trivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            mz_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        else:
            ssh_interp = pyinterp.bivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        
        # MB add extrapolation in NaN values if needed
        #if remove_NaN and np.isnan(ssh_interp).any():
        #    x_source_axis = pyinterp.Axis(self.ref[self.name_ref_lon].values, is_circle=False)
        #    y_source_axis = pyinterp.Axis(self.ref[self.name_ref_lat].values)
        #    z_source_axis = pyinterp.TemporalAxis(np.ascontiguousarray(self.ref[self.name_ref_time][:].values))
        #    grid = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis,  ssh_interp.T)
        #    _, filled = pyinterp.fill.gauss_seidel(grid)
        #else:
        filled = ssh_interp.T
        
        # Save to dataset
        if time is not None:  
            return xr.DataArray(
                data=filled.T,
                coords={self.name_ref_time: self.ref[self.name_ref_time].values,
                        self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                        self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                        },
                dims=[self.name_ref_time, self.name_ref_lat, self.name_ref_lon]
                )
        else:
            return xr.DataArray(
                data=filled.T,
                coords={self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                        self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                        },
                dims=[self.name_ref_lat, self.name_ref_lon]
                )
    
    def _regrid_unstructured(self, lon, lat, time, var):
        
        # Spatial interpolation 
        mesh = pyinterp.RTree()
        lon_target = self.ref[self.name_ref_lon].values
        lat_target = self.ref[self.name_ref_lat].values
        if len(lon_target.shape)==1:
            lon_target, lat_target = np.meshgrid(lon_target, lat_target)
        lons = lon.ravel()
        lats = lat.ravel()
        var_regridded = np.zeros((time.size,lat_target.shape[0],lon_target.shape[1]))
        for i in range(time.size):
            data = var[i].data.ravel()
            mask = np.isnan(lons) + np.isnan(lats) + np.isnan(data)
            data = data[~mask]
            mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
            idw, _ = mesh.window_function(
                np.vstack((lon_target.ravel(), lat_target.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,
                wf='parzen',
                num_threads=0)
            var_regridded[i,:,:] = idw.reshape(lon_target.shape)

        # Save to dataset
        var_regridded = xr.DataArray(
            data=var_regridded,
            coords={self.name_ref_time: time,
                    self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                    self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                    },
            dims=[self.name_ref_time, self.name_ref_lat, self.name_ref_lon]
            )

        # Time interpolation
        var_regridded = var_regridded.interp({self.name_ref_time:self.ref[self.name_ref_time]})

        return var_regridded
        
    def rmse_based_scores(self,plot=False):
        
        logging.info('     Compute RMSE-based scores...')

        # RMSE(t) based score
        rmse_t = 1.0 - (((self.exp_regridded - self.ref[self.name_ref_var])**2).mean(
            dim=(self.name_ref_lon, self.name_ref_lat)))**0.5/(((self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_lon, self.name_ref_lat)))**0.5
        if self.compare_to_baseline:
            rmse_t_bas = 1.0 - (((self.bas_regridded - self.ref[self.name_ref_var])**2).mean(
                dim=(self.name_ref_lon, self.name_ref_lat)))**0.5/(((self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_lon, self.name_ref_lat)))**0.5
            rmse_t = xr.concat((rmse_t, rmse_t_bas), dim='run')
            rmse_t['run'] = ['experiment','baseline']
        # RMSE(x, y) based score
        rmse_xy = (((self.exp_regridded - self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_time)))**0.5
        if self.compare_to_baseline:
            rmse_xy_bas = (((self.bas_regridded - self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_time)))**0.5
            rmse_xy = xr.concat((rmse_xy, rmse_xy_bas), dim='run')
            rmse_xy['run'] = ['experiment','baseline']


        rmse_t = rmse_t.rename('rmse_t')
        rmse_xy = rmse_xy.rename('rmse_xy')

        rmse_t.to_netcdf(f'{self.dir_output}/rmse_t.nc')
        rmse_xy.to_netcdf(f'{self.dir_output}/rmse_xy.nc')

        self.rmse_t = rmse_t
        self.rmse_xy = rmse_xy

        if plot:
            if not self.compare_to_baseline:
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
                rmse_t.plot(ax=ax1)
                rmse_xy.plot(ax=ax2,cmap='Reds')
            else:
                fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,10))
                
                
                rmse_xy.sel(run='experiment').plot(ax=ax1,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax1.set_title('Experiment')

                rmse_xy.sel(run='baseline').plot(ax=ax2,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax2.set_title('Baseline')

                rmse_score = 100*(rmse_xy.sel(run='experiment')-rmse_xy.sel(run='baseline'))/rmse_xy.sel(run='baseline')
                rmse_score.plot(ax=ax3,cmap='RdBu_r',vmin=-100,vmax=100)
                ax3.set_title('Improvement Experiment VS Baseline')


                rmse_t.sel(run='experiment').plot(ax=ax4,label='Experiment')
                rmse_t.sel(run='baseline').plot(ax=ax4,label='Baseline')
                ax4.set_title('Time evolution RMSE-based scores')
                ax4.legend()

            plt.show()
            fig.savefig(f'{self.dir_output}/rmse.png',dpi=100)


        # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
        self.leaderboard_rmse = (1.0 - (((self.exp_regridded - self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5 / (
            ((self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5).values
        
        if self.compare_to_baseline:
            self.leaderboard_rmse_bas = (1.0 - (((self.bas_regridded - self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5 / (
                ((self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5).values
            self.leaderboard_rmse_std_bas = rmse_t.sel(run='baseline').std().values
            self.leaderboard_rmse_std = rmse_t.sel(run='experiment').std().values
        else:
            self.leaderboard_rmse_std = rmse_t.std().values

    def psd_based_scores(self,threshold=0.5, plot=False):
        
        # Remove nan 
        ref_noNans = self.ref[self.name_ref_var].copy()
        exp_noNans = self.exp_regridded.copy()
        mask = np.isnan(ref_noNans) + np.isnan(exp_noNans)
        ref_noNans.data[mask] = 0.
        exp_noNans.data[mask] = 0.
        if self.compare_to_baseline:
            bas_noNans = self.bas_regridded.copy()
            bas_noNans.data[mask] = 0.

        
        mean_psd_signal = self._psd(ref_noNans,dim_iso=[self.name_ref_lon,self.name_ref_lat])
        
        mean_psd_err = self._psd((exp_noNans - ref_noNans),dim_iso=[self.name_ref_lon,self.name_ref_lat])

        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        if self.compare_to_baseline:
            mean_psd_err_bas = self._psd((bas_noNans - ref_noNans),dim_iso=[self.name_ref_lon,self.name_ref_lat])
            mean_psd_err = xr.concat((mean_psd_err, mean_psd_err_bas), dim='run')
            mean_psd_err['run'] = ['experiment','baseline']
            psd_based_score_bas = (1.0 - mean_psd_err_bas/mean_psd_signal)
            psd_based_score = xr.concat((psd_based_score, psd_based_score_bas), dim='run')
            psd_based_score['run'] = ['experiment','baseline']

        mean_psd_signal.to_netcdf(f'{self.dir_output}/mean_psd_signal.nc')
        mean_psd_err.to_netcdf(f'{self.dir_output}/mean_psd_err.nc')
        
        self.mean_psd_signal = mean_psd_signal
        self.mean_psd_err = mean_psd_err
        self.psd_based_score = psd_based_score

        # Plot
        if plot:
            fig = self._plot_psd_score_v0(psd_based_score)
            fig.savefig(f'{self.dir_output}/psd.png',dpi=100)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
        level = [threshold]

        if self.compare_to_baseline:
            # Experiment
            cs = plt.contour(1./psd_based_score[f'freq_r'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='experiment'), level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score = np.min(x05)
            self.leaderboard_psdt_score = np.min(y05)/3600/24 # in days
            # Baseline
            cs = plt.contour(1./psd_based_score[f'freq_r'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='baseline'), level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score_bas = np.min(x05)
            self.leaderboard_psdt_score_bas = np.min(y05)/3600/24 # in days
        else:
            cs = plt.contour(1./psd_based_score[f'freq_r'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score, level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score = np.min(x05)
            self.leaderboard_psdt_score = np.min(y05)/3600/24 # in days

    def _plot_psd_score_v0(self, ds_psd):
            
        try:
            nb_run = len(ds_psd.run)
        except:
            nb_run = 1
        
        fig, ax0 =  plt.subplots(1, nb_run, sharey=True, figsize=(nb_run*10, 5))

        if nb_run==1:
            ax0 = [ax0]

        for run in range(nb_run):

            ax = ax0[run]
            try:
                ctitle = ds_psd.run.values[run]
            except:
                ctitle = ''
            if nb_run > 1:
                data = (ds_psd.isel(run=run).values)
            else:
                data = (ds_psd.values)
            #ax.invert_yaxis()
            ax.invert_xaxis()
            c1 = ax.contourf(1./(ds_psd.freq_r), 1./ds_psd.freq_time, data,
                            levels=np.arange(0,1.1, 0.1), cmap='RdYlGn', extend='both')
            ax.set_xlabel('spatial wavelength (degree_lon)', fontweight='bold', fontsize=18)
            ax0[0].set_ylabel('temporal wavelength (days)', fontweight='bold', fontsize=18)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(linestyle='--', lw=1, color='w')
            ax.tick_params(axis='both', labelsize=18)
            ax.set_title(f'PSD-based score ({ctitle})', fontweight='bold', fontsize=18)
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            c2 = ax.contour(1./(ds_psd.freq_r), 1./ds_psd.freq_time, data, levels=[0.5], linewidths=2, colors='k')
            
            cbar = fig.colorbar(c1, ax=ax, pad=0.01)
            cbar.add_lines(c2)

        bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
        ax0[-1].annotate('Resolved scales',
                        xy=(1.2, 0.8),
                        xycoords='axes fraction',
                        xytext=(1.2, 0.55),
                        bbox=bbox_props,
                        arrowprops=
                            dict(facecolor='black', shrink=0.05),
                            horizontalalignment='left',
                            verticalalignment='center')

        ax0[-1].annotate('UN-resolved scales',
                        xy=(1.2, 0.2),
                        xycoords='axes fraction',
                        xytext=(1.2, 0.45),
                        bbox=bbox_props,
                        arrowprops=
                        dict(facecolor='black', shrink=0.05),
                            horizontalalignment='left',
                            verticalalignment='center')
        
        plt.show()
        return fig
    
    def _psd(self,da,dim=None,dim_iso=None,dim_mean=None,detrend='constant'):

            # Rechunk 
            if dim is not None:
                chunks = {}
                if type(dim)!=list:
                    dim = [dim]
                for d in dim:
                    chunks[d] = da[d].size
                da = da.chunk(chunks)
        
            # Compute PSD
            psd = xrft.power_spectrum(
                da, 
                dim=dim, 
                detrend=detrend, 
                window=True).compute()
            
            # Isotropize
            if dim_iso is not None:
                psd = xrft.isotropize(psd,fftdim=[f'freq_{dim_iso[0]}',f'freq_{dim_iso[1]}'])

            # Positive coordinates
            ispos = True
            for d in psd.dims:
                ispos = ispos & (psd[d] > 0.)
            psd = psd.where(ispos.compute(), drop=True)
            
            # Average
            if dim_mean is not None:
                psd = psd.mean(dim=dim_mean)
            
            psd.name = f'PSD_{da.name}'
            
            return psd

    def movie(self,framerate=24,Display=True,clim=None,range_err=None,cmap='Spectral'):

        # For memory leak when saving multiple png files...
        import matplotlib
        matplotlib.use('Agg')

        # Create merged dataset
        if self.compare_to_baseline:
            name_dim_rmse = ('run', self.name_ref_time)
        else:
            name_dim_rmse = (self.name_ref_time,)
        coords = (self.name_ref_time,self.name_ref_lat,self.name_ref_lon)
        ds = xr.Dataset(
            {'ref':(coords,self.ref[self.name_ref_var].data),
            'exp':(coords,self.exp_regridded.data),
            'err':(coords,self.ref[self.name_ref_var].data-self.exp_regridded.data),
            'rmse_score':(name_dim_rmse,self.rmse_t.data)},
            coords=(
                {self.name_ref_time:self.ref[self.name_ref_time],
                self.name_ref_lat:self.ref[self.name_ref_lat],
                self.name_ref_lon:self.ref[self.name_ref_lon]})
        )
        if self.compare_to_baseline:
            ds['bas'] = (coords,self.bas_regridded.data)
            ds['err_bas'] = (coords,self.ref[self.name_ref_var].data-self.bas_regridded.data)
            ds = ds.assign_coords({'run': ['experiment','baseline']})
        
        ds = ds.chunk({self.name_ref_time:1})

        # Plotting parameters
        xlim = (ds[self.name_ref_time][0].values,ds[self.name_ref_time][-1].values)
        ylim = (ds.rmse_score.min().values,ds.rmse_score.max().values)
        if clim is None:
            clim = (ds.ref.to_dataset().apply(np.nanmin).ref.values,ds.ref.to_dataset().apply(np.nanmax).ref.values)
        if range_err is None:
            range_err = ds.err.to_dataset().apply(np.abs).apply(np.nanmax).err.values
        
        # Plotting function
        def _save_single_frame(ds, tt, xlim=xlim, ylim=ylim,clim=clim,range_err=range_err,cmap=cmap):

            if tt==0:
                return

            if self.compare_to_baseline:
                fig = plt.figure(figsize=(18,15))
                gs = gridspec.GridSpec(3,5,width_ratios=(1,1,0.05,1,0.05))
            else:
                fig = plt.figure(figsize=(18,10))
                gs = gridspec.GridSpec(2,5,width_ratios=(1,1,0.05,1,0.05))

            date = str(ds[self.name_ref_time][tt].values)[:13]

            ids = ds.isel(time=tt)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ids.ref.plot(ax=ax1,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax1.set_title('Reference')

            ax2 = fig.add_subplot(gs[0, 1])
            im = ids.exp.plot(ax=ax2,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax2.set_ylabel('')
            ax2.set_yticks([])
            ax2.set_title('Experiment')
        
            ax3 = fig.add_subplot(gs[0, 2])
            plt.colorbar(im,cax=ax3)

            ax4 = fig.add_subplot(gs[0, 3])
            im = ids.err.plot(ax=ax4,cmap='RdBu_r',vmin=-range_err,vmax=range_err,add_colorbar=False)
            ax4.set_ylabel('')
            ax4.set_yticks([])
            ax4.set_title('Difference')

            ax5 = fig.add_subplot(gs[0, 4])
            plt.colorbar(im,cax=ax5)

            if self.compare_to_baseline:

                ax2 = fig.add_subplot(gs[1, 1])
                im = ids.bas.plot(ax=ax2,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
                ax2.set_ylabel('')
                ax2.set_yticks([])
                ax2.set_title('Baseline')
        
                ax4 = fig.add_subplot(gs[1, 3])
                im = ids.err_bas.plot(ax=ax4,cmap='RdBu_r',vmin=-range_err,vmax=range_err,add_colorbar=False)
                ax4.set_ylabel('')
                ax4.set_yticks([])
                ax4.set_title('Difference')

            ids = ds.isel(time=slice(0,tt+1))
            ax = fig.add_subplot(gs[-1, :])
            if self.compare_to_baseline:
                ids.rmse_score.sel(run='experiment').plot(ax=ax,label='experiment',xlim=xlim,ylim=ylim)
                ids.rmse_score.sel(run='baseline').plot(ax=ax,label='baseline',xlim=xlim,ylim=ylim)
                ax.legend()
            else:
                ids.rmse_score.plot.line(ax=ax,xlim=xlim,ylim=ylim)
            ax.set_title(date)

            fig.savefig(f'{self.dir_output}/frame_{str(tt).zfill(5)}.png',dpi=100)

            plt.close(fig)
            del fig
            gc.collect(2)

        
        # Compute and save frames 
        for tt in range(ds[self.name_ref_time].size):
            _save_single_frame(ds, tt)

        # Create movie
        sourcefolder = self.dir_output
        moviename = 'movie.mp4'
        frame_pattern = 'frame_*.png'
        ffmpeg_options="-c:v libx264 -preset veryslow -crf 15 -pix_fmt yuv420p"

        command = 'ffmpeg -f image2 -r %i -pattern_type glob -i %s -y %s -r %i %s' % (
                framerate,
                os.path.join(sourcefolder, frame_pattern),
                ffmpeg_options,
                framerate,
                os.path.join(self.dir_output, moviename),
            )
        print(command)

        _ = subprocess.run(command.split(' '),stdout=subprocess.PIPE)

        # Delete frames
        os.system(f'rm {os.path.join(sourcefolder, frame_pattern)}')
        
        # Display movie
        if Display:
            Video(os.path.join(self.dir_output, moviename),embed=True)

    def Leaderboard(self):

        data = [[self.name_experiment, 
            np.round(self.leaderboard_rmse,2), 
            np.round(self.leaderboard_rmse_std,2), 
            np.round(self.leaderboard_psds_score,2), 
            np.round(self.leaderboard_psdt_score,2),]]

        if self.compare_to_baseline:
            data.append(['baseline', 
                np.round(self.leaderboard_rmse_bas,2), 
                np.round(self.leaderboard_rmse_std_bas,2), 
                np.round(self.leaderboard_psds_score_bas,2), 
                np.round(self.leaderboard_psdt_score_bas,2),])

         
        Leaderboard = pd.DataFrame(data, 
                                columns=['Method', 
                                            "µ(RMSE) ", 
                                            "σ(RMSE)", 
                                            'λx (degree)', 
                                            'λt (days)'])

        with open(f'{self.dir_output}/metrics.txt', 'w') as f:
            dfAsString = Leaderboard.to_string()
            f.write(dfAsString)
        
        return Leaderboard

class Diag_osse_uv():

    """
    NAME
        Diag_osse

    DESCRIPTION
        Class to compute OSSE diagnostics
    """

    def __init__(self,config,State):
        
        # name_experiment
        self.name_experiment = config.EXP.name_experiment
        # dir_output
        if config.DIAG.dir_output is None:
            self.dir_output = f'{config.EXP.path_save}/diags/'
        else:
            self.dir_output = config.DIAG.dir_output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        # time_min
        if config.DIAG.time_min is None:
            self.time_min = config.EXP.init_date
        else:
            self.time_min = config.DIAG.time_min
        # time_max
        if config.DIAG.time_max is None:
            self.time_max = config.EXP.final_date
        else:
            self.time_max = config.DIAG.time_max
        # time_step
        if config.DIAG.time_step is None:
            self.time_step = config.EXP.saveoutput_time_step
        else:
            self.time_step = config.DIAG.time_step
        # lon_min
        if config.DIAG.lon_min is None:
            self.lon_min = config.GRID.lon_min
        else:
            self.lon_min = config.DIAG.lon_min
        # lon_max
        if config.DIAG.lon_max is None:
            self.lon_max = config.GRID.lon_max
        else:
            self.lon_max = config.DIAG.lon_max
        # lat_min
        if config.DIAG.lat_min is None:
            self.lat_min = config.GRID.lat_min
        else:
            self.lat_min = config.DIAG.lat_min
        # lat_max
        if config.DIAG.lat_max is None:
            self.lat_max = config.GRID.lat_max
        else:
            self.lat_max = config.DIAG.lat_max


        # Reference data
        self.name_ref_time = config.DIAG.name_ref_time
        self.name_ref_lon = config.DIAG.name_ref_lon
        self.name_ref_lat = config.DIAG.name_ref_lat
        self.name_ref_var_u = config.DIAG.name_ref_var_u
        self.name_ref_var_v = config.DIAG.name_ref_var_v
        ref = xr.open_mfdataset(config.DIAG.name_ref,**config.DIAG.options_ref)
        if np.sign(ref[self.name_ref_lon].data.min())==-1 and State.lon_unit=='0_360':
            ref = ref.assign_coords({self.name_ref_lon:((self.name_ref_lon, ref[self.name_ref_lon].data % 360))})
        elif np.sign(ref[self.name_ref_lon].data.min())==1 and State.lon_unit=='-180_180':
            ref = ref.assign_coords({self.name_ref_lon:((self.name_ref_lon, (ref[self.name_ref_lon].data + 180) % 360 - 180))})
        ref = ref.sortby(ref[self.name_ref_lon])    
        dt = (ref[self.name_ref_time][1]-ref[self.name_ref_time][0]).values/np.timedelta64(1,'s')
        idt = max(int(self.time_step.total_seconds()//dt), 1)
        self.ref = ref.sel(
            {self.name_ref_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max),idt)}
             )
        try:
            self.ref = self.ref.sel(
                {self.name_ref_lon:slice(self.lon_min,self.lon_max),
                self.name_ref_lat:slice(self.lat_min,self.lat_max)}
            )
        except:
            print('Warning: unable to select study region in the reference fields.\
That could be due to non regular grid or bad written netcdf file')
        ref.close()
        self.ref = self.ref.load()
            
        # Experimental data
        self.exp_geo_grid = State.geo_grid
        self.name_exp_time = config.EXP.name_time
        self.name_exp_lon = config.EXP.name_lon
        self.name_exp_lat = config.EXP.name_lat
        self.name_exp_var_ssh = config.DIAG.name_exp_var_ssh
        self.name_exp_var_u = config.DIAG.name_exp_var_u
        self.name_exp_var_v = config.DIAG.name_exp_var_v
        exp = xr.open_mfdataset(f'{config.EXP.path_save}/{config.EXP.name_exp_save}*nc')
        exp = exp.assign_coords({self.name_exp_lon:exp[self.name_exp_lon]})
        dt = (exp[self.name_exp_time][1]-exp[self.name_exp_time][0]).values
        self.exp = exp.sel(
            {self.name_exp_time:slice(np.datetime64(self.time_min)-dt,np.datetime64(self.time_max)+dt)},
             )
        # Compute geostrophic current velocities
        g = 9.81
        if self.name_exp_var_u is not None and self.name_exp_var_u in exp:
            if self.name_exp_var_u != 'u':
                u = exp[self.name_exp_var_u]
                self.exp['u'] = u.rename('u')
            if self.name_exp_var_v != 'v':
                v = exp[self.name_exp_var_v]
                self.exp['v'] = v.rename('v')
        else:
            ssh = self.exp[self.name_exp_var_ssh].copy()
            u = ssh.differentiate(ssh.dims[1])
            u.data *= -g/State.f/State.DY
            v = ssh.differentiate(ssh.dims[2])
            v.data *= g/State.f/State.DX
            if self.exp_geo_grid:
                u.data *= State.lat[1,0] - State.lat[0,0]
                v.data *= State.lon[0,1] - State.lon[0,0]
            self.exp['u'] = u.rename('u')
            self.exp['v'] = v.rename('v')
        
        exp.close()
        self.exp = self.exp.load()

        # Baseline data
        self.compare_to_baseline = config.DIAG.compare_to_baseline 
        if self.compare_to_baseline:
            self.name_bas_time = config.DIAG.name_bas_time
            self.name_bas_lon = config.DIAG.name_bas_lon
            self.name_bas_lat = config.DIAG.name_bas_lat
            self.name_bas_var_ssh = config.DIAG.name_bas_var_ssh
            bas = xr.open_mfdataset(config.DIAG.name_bas)
            if np.sign(bas[self.name_bas_lon].data.min())==-1 and State.lon_unit=='0_360':
                bas = bas.assign_coords({self.name_bas_lon:((self.name_bas_lon, bas[self.name_bas_lon].data % 360))})
            elif np.sign(bas[self.name_bas_lon].data.min())==1 and State.lon_unit=='-180_180':
                bas = bas.assign_coords({self.name_bas_lon:((self.name_bas_lon, (bas[self.name_bas_lon].data + 180) % 360 - 180))})
            dt = (bas[self.name_bas_time][1]-bas[self.name_bas_time][0]).values
            self.bas = bas.sel(
                {self.name_bas_time:slice(np.datetime64(self.time_min)-dt, np.datetime64(self.time_max)+dt)},
                )
            try:
                self.bas = self.bas.sel(
                    {self.name_bas_lon:slice(self.lon_min,self.lon_max),
                    self.name_bas_lat:slice(self.lat_min,self.lat_max)}
                )
            except:
                print('Warning: unable to select study region in the baseline fields.')
            
            # Compute geostrophic current velocities
            ssh = self.bas[self.name_bas_var_ssh].copy()
            lon = self.bas[self.name_bas_lon].values
            lat = self.bas[self.name_bas_lat].values
            if len(lon.shape)==1:
                self.bas_geo_grid = True
                lon, lat = np.meshgrid(lon,lat)
            else:
                self.bas_geo_grid = False
            DX,DY = grid.lonlat2dxdy(lon, lat)
            f = 4*np.pi/86164*np.sin(lat*np.pi/180)
            u = ssh.differentiate(ssh.dims[1])
            u.data *= -g/f/DY
            v = ssh.differentiate(ssh.dims[2])
            v.data *= g/f/DX
            if self.bas_geo_grid:
                u.data *= lon[0,1] - lon[0,0]
                v.data *= lat[1,0] - lat[0,0]

            self.bas['u'] = u.rename('u')
            self.bas['v'] = v.rename('v')
            bas.close()
            self.bas = self.bas.load()

        
        # Mask
        if config.DIAG.name_mask is not None:
            name_lon = config.DIAG.name_var_mask['lon']
            name_lat = config.DIAG.name_var_mask['lat']
            name_var = config.DIAG.name_var_mask['var']
            mask = xr.open_mfdataset(config.DIAG.name_mask)
            if np.sign(mask[name_lon].data.min())==-1 and State.lon_unit=='0_360':
                mask = mask.assign_coords({name_lon:((name_lon, mask[name_lon].data % 360))})
            elif np.sign(mask[name_lon].data.min())==1 and State.lon_unit=='-180_180':
                mask = mask.assign_coords({name_lon:((name_lon, (mask[name_lon].data + 180) % 360 - 180))})
            mask = mask.sortby(mask[name_lon])  
            mask_interp = self._regrid_geo(
                mask[name_lon].values,
                mask[name_lat].values, 
                None, 
                mask[name_var],
                remove_NaN=False)   
            if mask_interp.dtype!=bool : 
                self.mask = np.empty(mask_interp.shape,dtype='bool')
                ind_mask = (np.isnan(mask_interp)) | (mask_interp==1) | (np.abs(mask_interp)>10)
                self.mask[ind_mask] = True
                self.mask[~ind_mask] = False
            else:
                self.mask = mask_interp.copy()       
        else:
            self.mask = None
    
    def regrid_exp(self):
        
        if self.exp_geo_grid:
            self.exp_regridded_u =  self._regrid_geo(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp['u'])
            self.exp_regridded_v =  self._regrid_geo(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp['v'])
        else:
            self.exp_regridded_u = self._regrid_unstructured(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp['u'])
            self.exp_regridded_v = self._regrid_unstructured(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp['v'])
        
        if self.compare_to_baseline:
            if self.bas_geo_grid:
                self.bas_regridded_u = self._regrid_geo(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas['u'])        
                self.bas_regridded_v = self._regrid_geo(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas['v'])        
            else:
                self.bas_regridded_u = self._regrid_unstructured(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas['u'])        
                self.bas_regridded_v = self._regrid_unstructured(
                    self.bas[self.name_bas_lon].values,
                    self.bas[self.name_bas_lat].values, 
                    self.bas[self.name_bas_time].values, 
                    self.bas['v'])   
        else:
            self.bas_regridded_u = None
            self.bas_regridded_v = None     
                
        if self.mask is not None:
            if self.compare_to_baseline:
                self.bas_regridded_u.data[:,self.mask] = np.nan
                self.bas_regridded_v.data[:,self.mask] = np.nan
            self.exp_regridded_u.data[:,self.mask] = np.nan
            self.exp_regridded_v.data[:,self.mask] = np.nan
        else:
            if self.compare_to_baseline:
                self.bas_regridded_u.data[np.isnan(self.ref[self.name_ref_var_u].data)] = np.nan
                self.bas_regridded_v.data[np.isnan(self.ref[self.name_ref_var_v].data)] = np.nan
            self.exp_regridded_u.data[np.isnan(self.ref[self.name_ref_var_u].data)] = np.nan
            self.exp_regridded_v.data[np.isnan(self.ref[self.name_ref_var_v].data)] = np.nan
    
    def _regrid_geo(self, lon, lat, time, var, remove_NaN=False):

        # Define source grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        if time is not None:
            z_source_axis = pyinterp.TemporalAxis(time)
        ssh_source = var.T
        if time is not None:    
            grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
        else:
            grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, ssh_source.data)
        
        # Define target grid
        if time is not None:    
            mx_target, my_target, mz_target = np.meshgrid(
                self.ref[self.name_ref_lon][:].values.flatten(),
                self.ref[self.name_ref_lat][:].values.flatten(),
                z_source_axis.safe_cast(np.ascontiguousarray(self.ref[self.name_ref_time][:].values)),
                indexing="ij")
        else:
            mx_target, my_target = np.meshgrid(
                self.ref[self.name_ref_lon][:].values.flatten(),
                self.ref[self.name_ref_lat][:].values.flatten(),
                indexing="ij")

        # Spatio-temporal Interpolation
        if time is not None:  
            ssh_interp = pyinterp.trivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            mz_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        else:
            ssh_interp = pyinterp.bivariate(grid_source,
                                            mx_target.flatten(),
                                            my_target.flatten(),
                                            bounds_error=False).reshape(mx_target.shape).T
        
        # MB add extrapolation in NaN values if needed
        #if remove_NaN and np.isnan(ssh_interp).any():
        #    x_source_axis = pyinterp.Axis(self.ref[self.name_ref_lon].values, is_circle=False)
        #    y_source_axis = pyinterp.Axis(self.ref[self.name_ref_lat].values)
        #    if time is not None:  
        #        z_source_axis = pyinterp.TemporalAxis(np.ascontiguousarray(self.ref[self.name_ref_time][:].values))
        #        grid = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis,  ssh_interp.T)
        #    else:
        #        grid = pyinterp.Grid2D(x_source_axis, y_source_axis, ssh_interp.data)
        #    _, filled = pyinterp.fill.gauss_seidel(grid)
        #else:
        filled = ssh_interp.T
        
        # Save to dataset
        if time is not None:  
            return xr.DataArray(
                data=filled.T,
                coords={self.name_ref_time: self.ref[self.name_ref_time].values,
                        self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                        self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                        },
                dims=[self.name_ref_time, self.name_ref_lat, self.name_ref_lon]
                )
        else:
            return xr.DataArray(
                data=filled.T,
                coords={self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                        self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                        },
                dims=[self.name_ref_lat, self.name_ref_lon]
                )
    
    def _regrid_unstructured(self, lon, lat, time, var):
        
        # Spatial interpolation 
        mesh = pyinterp.RTree()
        lon_target = self.ref[self.name_ref_lon].values
        lat_target = self.ref[self.name_ref_lat].values
        if len(lon_target.shape)==1:
            lon_target, lat_target = np.meshgrid(lon_target, lat_target)
        lons = lon.ravel()
        lats = lat.ravel()
        var_regridded = np.zeros((time.size,lat_target.shape[0],lon_target.shape[1]))
        for i in range(time.size):
            data = var[i].data.ravel()
            mask = np.isnan(lons) + np.isnan(lats) + np.isnan(data)
            data = data[~mask]
            mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
            idw, _ = mesh.inverse_distance_weighting(
                np.vstack((lon_target.ravel(), lat_target.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,  # We are looking for at most 11 neighbors
                radius=600000,
                num_threads=0)
            var_regridded[i,:,:] = idw.reshape(lon_target.shape)

        # Save to dataset
        var_regridded = xr.DataArray(
            data=var_regridded,
            coords={self.name_ref_time: time,
                    self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                    self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                    },
            dims=[self.name_ref_time, self.name_ref_lat, self.name_ref_lon]
            )

        # Time interpolation
        var_regridded = var_regridded.interp({self.name_ref_time:self.ref[self.name_ref_time]})

        return var_regridded
    
    def rmse_based_scores(self,plot=False):

        self.rmse_t = {}
        self.rmse_xy = {}
        self.leaderboard_rmse = {}
        self.leaderboard_rmse_bas = {}
        self.leaderboard_rmse_std = {}
        self.leaderboard_rmse_std_bas = {}

        # u component
        self._rmse_based_scores(self.exp_regridded_u,self.bas_regridded_u,self.ref[self.name_ref_var_u],'u',plot=plot)

        # v component
        self._rmse_based_scores(self.exp_regridded_v,self.bas_regridded_v,self.ref[self.name_ref_var_v],'v',plot=plot)


    def _rmse_based_scores(self,exp,bas,ref,mode,plot=False):

        # RMSE(t) based score
        rmse_t = 1.0 - (((exp - ref)**2).mean(
            dim=(self.name_ref_lon, self.name_ref_lat)))**0.5/(((ref)**2).mean(dim=(self.name_ref_lon, self.name_ref_lat)))**0.5
        if self.compare_to_baseline:
            rmse_t_bas = 1.0 - (((bas - ref)**2).mean(
                dim=(self.name_ref_lon, self.name_ref_lat)))**0.5/(((ref)**2).mean(dim=(self.name_ref_lon, self.name_ref_lat)))**0.5
            rmse_t = xr.concat((rmse_t, rmse_t_bas), dim='run')
            rmse_t['run'] = ['experiment','baseline']
        # RMSE(x, y) based score
        rmse_xy = (((exp - ref)**2).mean(dim=(self.name_ref_time)))**0.5
        if self.compare_to_baseline:
            rmse_xy_bas = (((bas - ref)**2).mean(dim=(self.name_ref_time)))**0.5
            rmse_xy = xr.concat((rmse_xy, rmse_xy_bas), dim='run')
            rmse_xy['run'] = ['experiment','baseline']


        rmse_t = rmse_t.rename('rmse_t')
        rmse_xy = rmse_xy.rename('rmse_xy')

        rmse_t.to_netcdf(f'{self.dir_output}/rmse_t_{mode}.nc')
        rmse_xy.to_netcdf(f'{self.dir_output}/rmse_xy_{mode}.nc')

        self.rmse_t[mode] = rmse_t
        self.rmse_xy[mode] = rmse_xy

        if plot:
            if not self.compare_to_baseline:
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
                rmse_t.plot(ax=ax1)
                rmse_xy.plot(ax=ax2,cmap='Reds')
            else:
                fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,10))
                
                
                rmse_xy.sel(run='experiment').plot(ax=ax1,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax1.set_title('Experiment')

                rmse_xy.sel(run='baseline').plot(ax=ax2,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax2.set_title('Baseline')

                rmse_score = 100*(rmse_xy.sel(run='experiment')-rmse_xy.sel(run='baseline'))/rmse_xy.sel(run='baseline')
                rmse_score.plot(ax=ax3,cmap='RdBu_r',vmin=-100,vmax=100)
                ax3.set_title('Improvement Experiment VS Baseline')


                rmse_t.sel(run='experiment').plot(ax=ax4,label='Experiment')
                rmse_t.sel(run='baseline').plot(ax=ax4,label='Baseline')
                ax4.set_title('Time evolution RMSE-based scores')
                ax4.legend()

            plt.show()
            fig.savefig(f'{self.dir_output}/rmse_{mode}.png',dpi=100)


        # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
        self.leaderboard_rmse[mode] = (1.0 - (((exp - ref) ** 2).mean()) ** 0.5 / (
            ((ref) ** 2).mean()) ** 0.5).values
        self.leaderboard_rmse_std[mode] = rmse_t.std().values
        
        if self.compare_to_baseline:
            self.leaderboard_rmse_bas[mode] = (1.0 - (((bas - ref) ** 2).mean()) ** 0.5 / (
                ((ref) ** 2).mean()) ** 0.5).values
            self.leaderboard_rmse_std_bas[mode] = rmse_t.sel(run='baseline').std().values


    def psd_based_scores(self,threshold=0.5, plot=False):

        self.leaderboard_psds_score = {}
        self.leaderboard_psdt_score = {}
        self.leaderboard_psds_score_bas = {}
        self.leaderboard_psdt_score_bas = {}

        # u component
        self._psd_based_scores(self.exp_regridded_u,self.bas_regridded_u,self.ref[self.name_ref_var_u],'u',threshold=threshold,plot=plot)

        # v component
        self._psd_based_scores(self.exp_regridded_v,self.bas_regridded_v,self.ref[self.name_ref_var_v],'v',threshold=threshold,plot=plot)


    def _psd_based_scores(self,exp,bas,ref,mode,threshold=0.5, plot=False):
        
        # Remove nan 
        ref_noNans = ref.copy()
        exp_noNans = exp.copy()
        mask = np.isnan(ref_noNans) + np.isnan(exp_noNans)
        ref_noNans.data[mask] = 0.
        exp_noNans.data[mask] = 0.
        if self.compare_to_baseline:
            bas_noNans = bas.copy()
            bas_noNans.data[mask] = 0.

        
        mean_psd_signal = self._psd(
            ref_noNans,
            dim_iso=[self.name_ref_lon,self.name_ref_lat])
        
        mean_psd_err = self._psd(
            (exp_noNans - ref_noNans),
            dim_iso=[self.name_ref_lon,self.name_ref_lat])

        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        if self.compare_to_baseline:
            mean_psd_err_bas = self._psd(
                (bas_noNans - ref_noNans),
                dim_iso=[self.name_ref_lon,self.name_ref_lat])
            mean_psd_err = xr.concat((mean_psd_err, mean_psd_err_bas), dim='run')
            mean_psd_err['run'] = ['experiment','baseline']
            psd_based_score_bas = (1.0 - mean_psd_err_bas/mean_psd_signal)
            psd_based_score = xr.concat((psd_based_score, psd_based_score_bas), dim='run')
            psd_based_score['run'] = ['experiment','baseline']

        mean_psd_signal.to_netcdf(f'{self.dir_output}/mean_psd_signal.nc')
        mean_psd_err.to_netcdf(f'{self.dir_output}/mean_psd_err.nc')

        # Plot
        if plot:
            fig = self._plot_psd_score_v0(psd_based_score)
            fig.savefig(f'{self.dir_output}/psd_{mode}.png',dpi=100)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
        level = [threshold]

        # Experiment
        cs = plt.contour(1./psd_based_score[f'freq_r'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='experiment'), level)
        x05, y05 = cs.collections[0].get_paths()[0].vertices.T
        plt.close()
        self.leaderboard_psds_score[mode] = np.min(x05)
        self.leaderboard_psdt_score[mode] = np.min(y05)/3600/24 # in days
        if self.compare_to_baseline:
            # Baseline
            cs = plt.contour(1./psd_based_score[f'freq_r'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='baseline'), level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score_bas[mode] = np.min(x05)
            self.leaderboard_psdt_score_bas[mode] = np.min(y05)/3600/24 # in days
        

    def _plot_psd_score_v0(self, ds_psd):
            
        try:
            nb_run = len(ds_psd.run)
        except:
            nb_run = 1
        
        fig, ax0 =  plt.subplots(1, nb_run, sharey=True, figsize=(nb_run*10, 5))

        if nb_run==1:
            ax0 = [ax0]

        for run in range(nb_run):

            ax = ax0[run]
            try:
                ctitle = ds_psd.run.values[run]
            except:
                ctitle = ''
            if nb_run > 1:
                data = (ds_psd.isel(run=run).values)
            else:
                data = (ds_psd.values)
            ax.invert_yaxis()
            ax.invert_xaxis()
            c1 = ax.contourf(1./(ds_psd.freq_r), 1./ds_psd.freq_time, data,
                            levels=np.arange(0,1.1, 0.1), cmap='RdYlGn', extend='both')
            ax.set_xlabel('spatial wavelength (degree_lon)', fontweight='bold', fontsize=18)
            ax0[0].set_ylabel('temporal wavelength (days)', fontweight='bold', fontsize=18)
            #plt.xscale('log')
            ax.set_yscale('log')
            ax.grid(linestyle='--', lw=1, color='w')
            ax.tick_params(axis='both', labelsize=18)
            ax.set_title(f'PSD-based score ({ctitle})', fontweight='bold', fontsize=18)
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            c2 = ax.contour(1./(ds_psd.freq_r), 1./ds_psd.freq_time, data, levels=[0.5], linewidths=2, colors='k')
            
            cbar = fig.colorbar(c1, ax=ax, pad=0.01)
            cbar.add_lines(c2)

        bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
        ax0[-1].annotate('Resolved scales',
                        xy=(1.2, 0.8),
                        xycoords='axes fraction',
                        xytext=(1.2, 0.55),
                        bbox=bbox_props,
                        arrowprops=
                            dict(facecolor='black', shrink=0.05),
                            horizontalalignment='left',
                            verticalalignment='center')

        ax0[-1].annotate('UN-resolved scales',
                        xy=(1.2, 0.2),
                        xycoords='axes fraction',
                        xytext=(1.2, 0.45),
                        bbox=bbox_props,
                        arrowprops=
                        dict(facecolor='black', shrink=0.05),
                            horizontalalignment='left',
                            verticalalignment='center')
        
        plt.show()
        return fig
    
    def _psd(self,da,dim=None,dim_iso=None,dim_mean=None,detrend='constant'):

            # Rechunk 
            if dim is not None:
                chunks = {}
                if type(dim)!=list:
                    dim = [dim]
                for d in dim:
                    chunks[d] = da[d].size
                da = da.chunk(chunks)
        
            # Compute PSD
            psd = xrft.power_spectrum(
                da, 
                dim=dim, 
                detrend=detrend, 
                window=True).compute()
            
            # Isotropize
            if dim_iso is not None:
                psd = xrft.isotropize(psd,fftdim=[f'freq_{dim_iso[0]}',f'freq_{dim_iso[1]}'])

            # Positive coordinates
            ispos = True
            for d in psd.dims:
                ispos = ispos & (psd[d] > 0.)
            psd = psd.where(ispos.compute(), drop=True)
            
            # Average
            if dim_mean is not None:
                psd = psd.mean(dim=dim_mean)
            
            psd.name = f'PSD_{da.name}'
            
            return psd

    def movie(self,framerate=24,Display=True,clim=None,range_err=None,cmap='RdBu_r'):
        self._movie(self.exp_regridded_u,self.bas_regridded_u,self.ref[self.name_ref_var_u],'u',
                    framerate=framerate,Display=Display,clim=clim,range_err=range_err,cmap=cmap)
        self._movie(self.exp_regridded_v,self.bas_regridded_v,self.ref[self.name_ref_var_v],'v',
                    framerate=framerate,Display=Display,clim=clim,range_err=range_err,cmap=cmap)       
         
    def _movie(self,exp,bas,ref,mode,framerate=24,Display=True,clim=None,range_err=None,cmap='RdBu_r'):

        # For memory leak when saving multiple png files...
        import matplotlib
        matplotlib.use('Agg')

        # Create merged dataset
        if self.compare_to_baseline:
            name_dim_rmse = ('run', self.name_ref_time)
        else:
            name_dim_rmse = (self.name_ref_time,)
        coords = (self.name_ref_time,self.name_ref_lat,self.name_ref_lon)
        ds = xr.Dataset(
            {'ref':(coords,ref.data),
            'exp':(coords,exp.data),
            'err':(coords,ref.data-exp.data),
            'rmse_score':(name_dim_rmse,self.rmse_t[mode].data)},
            coords=(
                {self.name_ref_time:self.ref[self.name_ref_time],
                self.name_ref_lat:self.ref[self.name_ref_lat],
                self.name_ref_lon:self.ref[self.name_ref_lon]})
        )
        if self.compare_to_baseline:
            ds['bas'] = (coords,bas.data)
            ds['err_bas'] = (coords,ref.data-bas.data)
            ds = ds.assign_coords({'run': ['experiment','baseline']})
        
        ds = ds.chunk({self.name_ref_time:1})

        # Plotting parameters
        xlim = (ds[self.name_ref_time][0].values,ds[self.name_ref_time][-1].values)
        ylim = (ds.rmse_score.min().values,ds.rmse_score.max().values)
        if clim is None:
            clim = (ds.ref.to_dataset().apply(np.nanmin).ref.values,ds.ref.to_dataset().apply(np.nanmax).ref.values)
        if range_err is None:
            range_err = ds.err.to_dataset().apply(np.abs).apply(np.nanmax).err.values
        
        # Plotting function
        def _save_single_frame(ds, tt, xlim=xlim, ylim=ylim,clim=clim,range_err=range_err,cmap=cmap):

            if tt==0:
                return

            if self.compare_to_baseline:
                fig = plt.figure(figsize=(18,15))
                gs = gridspec.GridSpec(3,5,width_ratios=(1,1,0.05,1,0.05))
            else:
                fig = plt.figure(figsize=(18,10))
                gs = gridspec.GridSpec(2,5,width_ratios=(1,1,0.05,1,0.05))

            date = str(ds[self.name_ref_time][tt].values)[:13]

            ids = ds.isel(time=tt)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ids.ref.plot(ax=ax1,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax1.set_title('Reference')

            ax2 = fig.add_subplot(gs[0, 1])
            im = ids.exp.plot(ax=ax2,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax2.set_ylabel('')
            ax2.set_yticks([])
            ax2.set_title('Experiment')
        
            ax3 = fig.add_subplot(gs[0, 2])
            plt.colorbar(im,cax=ax3)

            ax4 = fig.add_subplot(gs[0, 3])
            im = ids.err.plot(ax=ax4,cmap='RdBu_r',vmin=-range_err,vmax=range_err,add_colorbar=False)
            ax4.set_ylabel('')
            ax4.set_yticks([])
            ax4.set_title('Difference')

            ax5 = fig.add_subplot(gs[0, 4])
            plt.colorbar(im,cax=ax5)

            if self.compare_to_baseline:

                ax2 = fig.add_subplot(gs[1, 1])
                im = ids.bas.plot(ax=ax2,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
                ax2.set_ylabel('')
                ax2.set_yticks([])
                ax2.set_title('Baseline')
        
                ax4 = fig.add_subplot(gs[1, 3])
                im = ids.err_bas.plot(ax=ax4,cmap='RdBu_r',vmin=-range_err,vmax=range_err,add_colorbar=False)
                ax4.set_ylabel('')
                ax4.set_yticks([])
                ax4.set_title('Difference')

            ids = ds.isel(time=slice(0,tt+1))
            ax = fig.add_subplot(gs[-1, :])
            if self.compare_to_baseline:
                ids.rmse_score.sel(run='experiment').plot(ax=ax,label='experiment',xlim=xlim,ylim=ylim)
                ids.rmse_score.sel(run='baseline').plot(ax=ax,label='baseline',xlim=xlim,ylim=ylim)
                ax.legend()
            else:
                ids.rmse_score.plot.line(ax=ax,xlim=xlim,ylim=ylim)
            ax.set_title(date)

            fig.savefig(f'{self.dir_output}/frame_{str(tt).zfill(5)}.png',dpi=100)

            plt.close(fig)
            del fig
            gc.collect(2)

        
        # Compute and save frames 
        for tt in range(ds[self.name_ref_time].size):
            _save_single_frame(ds, tt)

        # Create movie
        sourcefolder = self.dir_output
        moviename = f'movie_{mode}.mp4'
        frame_pattern = 'frame_*.png'
        ffmpeg_options="-c:v libx264 -preset veryslow -crf 15 -pix_fmt yuv420p"

        command = 'ffmpeg -f image2 -r %i -pattern_type glob -i %s -y %s -r %i %s' % (
                framerate,
                os.path.join(sourcefolder, frame_pattern),
                ffmpeg_options,
                framerate,
                os.path.join(self.dir_output, moviename),
            )
        print(command)

        _ = subprocess.run(command.split(' '),stdout=subprocess.PIPE)

        # Delete frames
        os.system(f'rm {os.path.join(sourcefolder, frame_pattern)}')
        
        # Display movie
        if Display:
            Video(os.path.join(self.dir_output, moviename),embed=True)
        
    def Leaderboard(self):

        df = []
        names = []
        
        for mode in ['u','v']:

            data = [[self.name_experiment, 
                np.round(self.leaderboard_rmse[mode],2), 
                np.round(self.leaderboard_rmse_std[mode],2), 
                np.round(self.leaderboard_psds_score[mode],2), 
                np.round(self.leaderboard_psdt_score[mode],2),]]

            if self.compare_to_baseline:
                data.append(['baseline', 
                    np.round(self.leaderboard_rmse_bas[mode],2), 
                    np.round(self.leaderboard_rmse_std_bas[mode],2), 
                    np.round(self.leaderboard_psds_score_bas[mode],2), 
                    np.round(self.leaderboard_psdt_score_bas[mode],2),])

            
            _df = pd.DataFrame(data, 
                            columns=['Method', 
                                        "µ(RMSE) ", 
                                        "σ(RMSE)", 
                                        'λx (degree)', 
                                        'λt (days)'])
            df.append(_df)
            names += [mode,]*_df.shape[0]

        Leaderboard = pd.concat(df)
        Leaderboard.insert(0,'U component',names)

        with open(f'{self.dir_output}/metrics.txt', 'w') as f:
            dfAsString = Leaderboard.to_string()
            f.write(dfAsString)
        
        return Leaderboard

class Diag_ose():

    """
    NAME
        Diag_ose

    DESCRIPTION
        Class to compute OSE diagnostics
    """

    def __init__(self,config,State):
        
        # name_experiment
        self.name_experiment = config.EXP.name_experiment
        # dir_output
        if config.DIAG.dir_output is None:
            self.dir_output = f'{config.EXP.path_save}/diags/'
        else:
            self.dir_output = config.DIAG.dir_output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        # Save config
        cmd = f"cp {config.name_file} {self.dir_output}/config.py"
        os.system(cmd)

        # time_min
        if config.DIAG.time_min is None:
            self.time_min = config.EXP.init_date
        else:
            self.time_min = config.DIAG.time_min
        # time_max
        if config.DIAG.time_max is None:
            self.time_max = config.EXP.final_date
        else:
            self.time_max = config.DIAG.time_max
        # time_step
        if config.DIAG.time_step is None:
            self.time_step = config.EXP.saveoutput_time_step
        else:
            self.time_step = config.DIAG.time_step
        # lon_min
        if config.DIAG.lon_min is None:
            self.lon_min = config.GRID.lon_min
        else:
            self.lon_min = config.DIAG.lon_min
        # lon_max
        if config.DIAG.lon_max is None:
            self.lon_max = config.GRID.lon_max
        else:
            self.lon_max = config.DIAG.lon_max
        # lat_min
        if config.DIAG.lat_min is None:
            self.lat_min = config.GRID.lat_min
        else:
            self.lat_min = config.DIAG.lat_min
        # lat_max
        if config.DIAG.lat_max is None:
            self.lat_max = config.GRID.lat_max
        else:
            self.lat_max = config.DIAG.lat_max
        # Stats parameters
        self.bin_lon_step = config.DIAG.bin_lon_step
        self.bin_lat_step = config.DIAG.bin_lat_step
        self.bin_time_step = config.DIAG.bin_time_step
        self.lenght_scale = config.DIAG.lenght_scale
        self.nb_min_obs = config.DIAG.nb_min_obs

        # Reference data
        self.name_ref_time = config.DIAG.name_ref_time
        self.name_ref_lon = config.DIAG.name_ref_lon
        self.name_ref_lat = config.DIAG.name_ref_lat
        self.name_ref_var = config.DIAG.name_ref_var

        def preprocess(ds):
            name_var = [self.name_ref_time, self.name_ref_lon, self.name_ref_lat, self.name_ref_var]
            ds = ds[name_var]
            return ds
        
        if type(config.DIAG.name_ref) is not list:
            config.DIAG.name_ref = [config.DIAG.name_ref]
        ref = []
        delta_x = []
        for name_ref in config.DIAG.name_ref:
            try:
                _ref = xr.open_mfdataset(name_ref,**config.DIAG.options_ref,preprocess=preprocess,compat='override',coords='minimal').load()
            except:
                files = glob.glob(name_ref)
                # Get time dimension to concatenate
                _ds0 = xr.open_dataset(files[0])
                name_time_dim = _ds0[self.name_ref_time].dims[0]
                _ds0.close()
                # Open nested files
                _ref = xr.open_mfdataset(name_ref,combine='nested',concat_dim=name_time_dim,**config.DIAG.options_ref,preprocess=preprocess,compat='override',coords='minimal').load()
                
            if np.sign(_ref[self.name_ref_lon].data.min())==-1 and State.lon_unit=='0_360':
                _ref = _ref.assign_coords({self.name_ref_lon:((_ref[self.name_ref_lon].dims, _ref[self.name_ref_lon].data % 360))})
                _ref = _ref.sortby(self.name_ref_lon)
            elif np.sign(_ref[self.name_ref_lon].data.min())>=0 and State.lon_unit=='-180_180':
                _ref = _ref.assign_coords({self.name_ref_lon:((_ref[self.name_ref_lon].dims, (_ref[self.name_ref_lon].data + 180) % 360 - 180))})
                _ref = _ref.sortby(self.name_ref_lon)
            _ref = _ref.swap_dims({_ref[self.name_ref_time].dims[0]:self.name_ref_time})
            lon_ref = _ref[self.name_ref_lon] 
            lat_ref = _ref[self.name_ref_lat]
            _ref = _ref.where((lat_ref >= self.lat_min) & (lat_ref <= self.lat_max), drop=True)
            _ref = _ref.where((lon_ref >= self.lon_min) & (lon_ref <= self.lon_max), drop=True)
            try:
                _ref = _ref.sel(
                    {self.name_ref_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max))}, drop=True
                    )
            except:
                _ref = _ref.where(((_ref[self.name_ref_time]<=np.datetime64(self.time_max)) &\
                            (_ref[self.name_ref_time]>=np.datetime64(self.time_min))).compute(),drop=True)
            # Mean spatial resolution of alongtrack data
            if len(_ref[self.name_ref_lat].shape)==2:
                _lon = _ref[self.name_ref_lon][:,0].values
                _lat = _ref[self.name_ref_lat][:,0].values
            else:
                _lon = _ref[self.name_ref_lon][:].values
                _lat = _ref[self.name_ref_lat][:].values
            delta_x.append(0.001*np.median(pyinterp.geodetic.coordinate_distances(_lon[:-1],
                                                                _lat[:-1],
                                                                _lon[1:],
                                                                _lat[1:])))

            _ref.close()

            # Add MDT to reference data
            if config.DIAG.add_mdt_to_ref:
                finterpmdt = read_auxdata(config.DIAG.path_mdt,config.DIAG.name_var_mdt,State.lon_unit)
                mdt_on_ref = finterpmdt((_ref[self.name_ref_lon] , _ref[self.name_ref_lat]))
                _ref[self.name_ref_var].data += mdt_on_ref

            # Append to list
            ref.append(_ref[self.name_ref_var])
        self.ref = ref
        self.delta_x = delta_x

                                      
        # Experimental data
        self.geo_grid = State.geo_grid
        self.name_exp_time = config.EXP.name_time
        self.name_exp_lon = config.EXP.name_lon
        self.name_exp_lat = config.EXP.name_lat
        self.name_exp_var = config.DIAG.name_exp_var
        exp = xr.open_mfdataset(f'{config.EXP.path_save}/{config.EXP.name_exp_save}*nc',preprocess=lambda ds: ds[[self.name_exp_var]])[self.name_exp_var].load()
        exp = exp.assign_coords({self.name_exp_lon:exp[self.name_exp_lon]})
        dt = (exp[self.name_exp_time][1]-exp[self.name_exp_time][0]).values
        self.exp = exp.sel(
            {self.name_exp_time:slice(np.datetime64(self.time_min)-dt,np.datetime64(self.time_max)+dt)},
             )
        try:
            self.exp = self.exp.sel(
                {self.name_exp_lon:slice(self.lon_min,self.lon_max),
                self.name_exp_lat:slice(self.lat_min,self.lat_max)}
            )
        except:
            print('Warning: unable to select study region in the experiment fields.\
That could be due to non regular grid or bad written netcdf file')
            self.exp = self.exp.where(((self.exp.lon>=self.lon_min) & 
                                      (self.exp.lon<=self.lon_max) & 
                                      (self.exp.lat>=self.lat_min) & 
                                      (self.exp.lat<=self.lat_max)).compute(),
                                      drop=True)
        exp.close()

        # Baseline data
        self.compare_to_baseline = config.DIAG.compare_to_baseline 
        if self.compare_to_baseline:
            self.name_bas_time = config.DIAG.name_bas_time
            self.name_bas_lon = config.DIAG.name_bas_lon
            self.name_bas_lat = config.DIAG.name_bas_lat
            self.name_bas_var = config.DIAG.name_bas_var
            bas = xr.open_mfdataset(config.DIAG.name_bas)[self.name_bas_var]
            if np.sign(bas[self.name_bas_lon].data.min())==-1 and State.lon_unit=='0_360':
                bas = bas.assign_coords({self.name_bas_lon:((bas[self.name_bas_lon].dims, bas[self.name_bas_lon].data % 360))})
            elif np.sign(bas[self.name_bas_lon].data.min())>=0 and State.lon_unit=='-180_180':
                bas = bas.assign_coords({self.name_bas_lon:((bas[self.name_bas_lon].dims, (bas[self.name_bas_lon].data + 180) % 360 - 180))})
            bas = bas.sortby(bas[self.name_bas_lon])
            self.bas = bas.sel(
                {self.name_bas_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max))},
                )
            
            try:
                self.bas = self.bas.sel(
                    {self.name_bas_lon:slice(self.lon_min,self.lon_max),
                    self.name_bas_lat:slice(self.lat_min,self.lat_max)}
                ).load()
            except:
                print('Warning: unable to select study region in the baseline fields.')
            bas.close()
            
        # Ratio of pannel size for plotting functions
        self.ratio_fig = (self.lat_max-self.lat_min)/(self.lon_max-self.lon_min)

    def regrid_exp(self):
        
        if self.geo_grid:
            self.exp_regridded =  self._regrid_geo(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp,
                )
        else:
            self.exp_regridded = self._regrid_unstructured(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp,
                )
        
        
        if self.compare_to_baseline:
            self.bas_regridded = self._regrid_geo(
                self.bas[self.name_bas_lon].values,
                self.bas[self.name_bas_lat].values, 
                self.bas[self.name_bas_time].values, 
                self.bas,
                )        
    
    def _regrid_geo(self, lon, lat, time, var):

        # Define source grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        z_source_axis = pyinterp.TemporalAxis(time)
        var_source = var.transpose(var.dims[2], var.dims[1], var.dims[0])
        grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, var_source.data)

        exp_regridded = []
        for _ref in self.ref:
            
            # Spatio-temporal Interpolation
            if len(_ref[self.name_ref_lon].shape)==2:
                # Swath data
                lon_ref = _ref[self.name_ref_lon].values.ravel()
                lat_ref = _ref[self.name_ref_lat].values.ravel()
                time_ref = np.repeat(_ref[self.name_ref_time].values,_ref[self.name_ref_lon].shape[1])
            else:
                # Nadir data
                lon_ref = _ref[self.name_ref_lon].values
                lat_ref = _ref[self.name_ref_lat].values
                time_ref = _ref[self.name_ref_time].values

            var_interp = pyinterp.trivariate(grid_source,
                                            lon_ref, 
                                            lat_ref,
                                            z_source_axis.safe_cast(time_ref),
                                            bounds_error=False).reshape(_ref[self.name_ref_lon].shape)

            # Save to dataset
            exp_regridded.append( xr.DataArray(
                name=var.name,
                data=var_interp,
                coords={self.name_ref_time: (_ref[self.name_ref_time].dims, _ref[self.name_ref_time].values),
                        self.name_ref_lon: (_ref[self.name_ref_lon].dims, _ref[self.name_ref_lon].values), 
                        self.name_ref_lat: (_ref[self.name_ref_lat].dims, _ref[self.name_ref_lat].values), 
                        },
                dims=_ref.dims
                )
            )
        return exp_regridded

    def _regrid_unstructured(self, lon, lat, time, var):

        # Define regular grid 
        dlon = np.nanmean(lon[:,1:]-lon[:,:-1])
        dlat = np.nanmean(lat[1:,:]-lat[:-1,:])
        lon1d = np.arange(np.nanmin(lon),np.nanmax(lon)+dlon,dlon)
        lat1d = np.arange(np.nanmin(lat),np.nanmax(lat)+dlat,dlat)
        lon_target, lat_target = np.meshgrid(lon1d, lat1d)
        
        # Spatial interpolation 
        mesh = pyinterp.RTree()
        lons = lon.ravel()
        lats = lat.ravel()
        var_regridded = np.zeros((time.size,lat_target.shape[0],lon_target.shape[1]))
        for i in range(time.size):
            data = var[i].data.ravel()
            mask = np.isnan(lons) + np.isnan(lats) + np.isnan(data)
            data = data[~mask]
            mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
            idw, _ = mesh.window_function(
                np.vstack((lon_target.ravel(), lat_target.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,
                wf='parzen',
                num_threads=0)
            var_regridded[i,:,:] = idw.reshape(lon_target.shape)
        
        # Mask
        lon2d,lat2d = np.meshgrid(lon1d,lat1d)
        mask_interp = griddata((lons,lats), var[0].data.ravel(), (lon2d.ravel(),lat2d.ravel()),method='nearest').reshape(lon2d.shape)
        var_regridded[:,np.isnan(mask_interp)] = np.nan

        # Save to dataset
        var_regridded = xr.DataArray(
            name=var.name,
            data=var_regridded,
            coords={self.name_exp_time: time,
                    self.name_exp_lon: lon1d, 
                    self.name_exp_lat: lat1d, 
                    },
            dims=[self.name_exp_time, self.name_exp_lat, self.name_exp_lon]
            )
        
        dsout = xr.Dataset({self.name_exp_var:var_regridded})
        dsout = dsout.sel({self.name_exp_lon:slice(self.lon_min,self.lon_max),
                           self.name_exp_lat:slice(self.lat_min,self.lat_max)})
        dsout.to_netcdf(f'{self.dir_output}/exp_regridded.nc')

        # return regrid_geo output
        return self._regrid_geo(
                    lon1d,
                    lat1d, 
                    time, 
                    var_regridded,
                    )
          
    def rmse_based_scores(self,plot=False):

        # Merging
        _ref = xr.merge(self.ref)
        _exp = xr.merge(self.exp_regridded)
        if self.compare_to_baseline:
            _bas = xr.merge(self.bas_regridded)

        ##########################
        # get data
        ##########################

        time_alongtrack = _ref[self.name_ref_time].values
        lon_alongtrack = _ref[self.name_ref_lon].values
        lat_alongtrack = _ref[self.name_ref_lat].values
        var_alongtrack = _ref[self.name_ref_var].values
        var_exp_interp = _exp[self.name_exp_var].values
        if self.compare_to_baseline:
            var_bas_interp = _bas[self.name_bas_var].values


        ##########################
        # write spatial statistics
        ##########################
            
        binning = pyinterp.Binning2D(
        pyinterp.Axis(np.arange(self.lon_min, self.lon_max, self.bin_lon_step), is_circle=True),
        pyinterp.Axis(np.arange(self.lat_min, self.lat_max + self.bin_lat_step, self.bin_lat_step)))

        output_filename_xy = f'{self.dir_output}/rmse_xy.nc'
        ncfile = netCDF4.Dataset(output_filename_xy,'w')

        # binning alongtrack
        binning.push(lon_alongtrack, lat_alongtrack, var_alongtrack, simple=True)
        self._write_stat(ncfile, 'alongtrack', binning)
        binning.clear()

        # binning map interp
        binning.push(lon_alongtrack, lat_alongtrack, var_exp_interp, simple=True)
        self._write_stat(ncfile, 'experiment', binning)
        binning.clear()

        if self.compare_to_baseline:
            binning.push(lon_alongtrack, lat_alongtrack, var_bas_interp, simple=True)
            self._write_stat(ncfile, 'baseline', binning)
            binning.clear()

        # binning diff sla-msla
        binning.push(lon_alongtrack, lat_alongtrack, var_alongtrack - var_exp_interp, simple=True)
        self._write_stat(ncfile, 'diff_exp', binning)
        binning.clear()

        if self.compare_to_baseline:
            binning.push(lon_alongtrack, lat_alongtrack, var_alongtrack - var_bas_interp, simple=True)
            self._write_stat(ncfile, 'diff_bas', binning)
            binning.clear()

        # add rmse
        diff2 = (var_alongtrack - var_exp_interp)**2
        binning.push(lon_alongtrack, lat_alongtrack, diff2, simple=True)
        var = ncfile.groups['diff_exp'].createVariable('rmse', binning.variable('mean').dtype, ('lat','lon'), zlib=True)
        var[:, :] = np.sqrt(binning.variable('mean')).T  
        rmse_xy_exp = np.sqrt(binning.variable('mean')).T

        if self.compare_to_baseline:
            diff2 = (var_alongtrack - var_bas_interp)**2
            binning.push(lon_alongtrack, lat_alongtrack, diff2, simple=True)
            var = ncfile.groups['diff_bas'].createVariable('rmse', binning.variable('mean').dtype, ('lat','lon'), zlib=True)
            var[:, :] = np.sqrt(binning.variable('mean')).T  
            rmse_xy_bas = np.sqrt(binning.variable('mean')).T


        ncfile.close()

        ##############################
        # write time series statistics
        ##############################

        output_filename_t = f'{self.dir_output}/rmse_t.nc'

        # alongtrack
        ##############################

        # convert data vector and time vector into xarray.Dataarray
        da = xr.DataArray(var_alongtrack, coords=_ref.coords, dims=_ref.dims)

        # resample 
        da_resample = da.resample(time=self.bin_time_step)

        # compute stats
        vmean = da_resample.mean()
        vminimum = da_resample.min()
        vmaximum = da_resample.max()
        vcount_alongtrack = da_resample.count()
        vvariance = da_resample.var()
        vmedian = da_resample.median()
        vrms = np.sqrt(np.square(da).resample(time=self.bin_time_step).mean())

        rms_alongtrack = np.copy(vrms)

        dims = vmean.dims
        coords = vmean.coords
        # save stat to dataset
        ds = xr.Dataset(
            {
                "mean": (dims, vmean.values),
                "min": (dims, vminimum.values),
                "max": (dims, vmaximum.values),
                "count": (dims, vcount_alongtrack.values),
                "variance": (dims, vvariance.values),
                "median": (dims, vmedian.values),
                "rms": (dims, vrms.values),            
            },
            coords,
        )

        ds.to_netcdf(output_filename_t, group='alongtrack')


        # experiment
        ##############################

        # convert data vector and time vector into xarray.Dataarray
        da = xr.DataArray(var_exp_interp, coords=_ref.coords, dims=_ref.dims)

        # resample 
        da_resample = da.resample(time=self.bin_time_step)

        # compute stats
        vmean = da_resample.mean()
        vminimum = da_resample.min()
        vmaximum = da_resample.max()
        vcount = da_resample.count()
        vvariance = da_resample.var()
        vmedian = da_resample.median()
        vrms = np.sqrt(np.square(da).resample(time=self.bin_time_step).mean())

        # save stat to dataset
        ds = xr.Dataset(
            {
                "mean": (dims, vmean.values),
                "min": (dims, vminimum.values),
                "max": (dims, vmaximum.values),
                "count": (dims, vcount_alongtrack.values),
                "variance": (dims, vvariance.values),
                "median": (dims, vmedian.values),
                "rms": (dims, vrms.values),            
            },
            coords,
        )

        ds.to_netcdf(output_filename_t, group='experiment', mode='a')

        # baseline
        ##############################
        if self.compare_to_baseline:
            # convert data vector and time vector into xarray.Dataarray
            da = xr.DataArray(var_bas_interp, coords=_ref.coords, dims=_ref.dims)

            # resample 
            da_resample = da.resample(time=self.bin_time_step)

            # compute stats
            vmean = da_resample.mean()
            vminimum = da_resample.min()
            vmaximum = da_resample.max()
            vcount = da_resample.count()
            vvariance = da_resample.var()
            vmedian = da_resample.median()
            vrms = np.sqrt(np.square(da).resample(time=self.bin_time_step).mean())

            # save stat to dataset
            ds = xr.Dataset(
                {
                    "mean": (dims, vmean.values),
                "min": (dims, vminimum.values),
                "max": (dims, vmaximum.values),
                "count": (dims, vcount_alongtrack.values),
                "variance": (dims, vvariance.values),
                "median": (dims, vmedian.values),
                "rms": (dims, vrms.values),            
            },
            coords,
            )

            ds.to_netcdf(output_filename_t, group='baseline', mode='a')


        # diff_exp
        ##############################

        # convert data vector and time vector into xarray.Dataarray
        da = xr.DataArray(var_alongtrack - var_exp_interp, coords=_ref.coords, dims=_ref.dims)

        # resample 
        da_resample = da.resample(time=self.bin_time_step)

        # compute stats
        vmean = da_resample.mean()
        vminimum = da_resample.min()
        vmaximum = da_resample.max()
        vcount = da_resample.count()
        vvariance = da_resample.var()
        vmedian = da_resample.median()
        vrms = np.sqrt(np.square(da).resample(time=self.bin_time_step).mean())

        # rmse
        rmse_exp_t = np.copy(vrms)

        # mask rmse if nb obs < nb_min_obs
        rmse_exp_t = np.ma.masked_where(vcount_alongtrack.values < self.nb_min_obs, rmse_exp_t)

        # save stat to dataset
        ds = xr.Dataset(
            {
                "mean": (dims, vmean.values),
                "min": (dims, vminimum.values),
                "max": (dims, vmaximum.values),
                "count": (dims, vcount_alongtrack.values),
                "variance": (dims, vvariance.values),
                "median": (dims, vmedian.values),
                "rms": (dims, vrms.values),            
            },
            coords,
        )

        ds.to_netcdf(output_filename_t, group='diff_exp', mode='a')


        # diff_bas
        ##############################

        if self.compare_to_baseline:
            # convert data vector and time vector into xarray.Dataarray
            da = xr.DataArray(var_alongtrack - var_bas_interp, coords=_ref.coords, dims=_ref.dims)

            # resample 
            da_resample = da.resample(time=self.bin_time_step)

            # compute stats
            vmean = da_resample.mean()
            vminimum = da_resample.min()
            vmaximum = da_resample.max()
            vcount = da_resample.count()
            vvariance = da_resample.var()
            vmedian = da_resample.median()
            vrms = np.sqrt(np.square(da).resample(time=self.bin_time_step).mean())

            # rmse
            rmse_bas_t = np.copy(vrms)

            # mask rmse if nb obs < nb_min_obs
            rmse_bas_t = np.ma.masked_where(vcount_alongtrack.values < self.nb_min_obs, rmse_bas_t)

            # save stat to dataset
            ds = xr.Dataset(
                {
                    "mean": (dims, vmean.values),
                "min": (dims, vminimum.values),
                "max": (dims, vmaximum.values),
                "count": (dims, vcount_alongtrack.values),
                "variance": (dims, vvariance.values),
                "median": (dims, vmedian.values),
                "rms": (dims, vrms.values),            
            },
            coords,
            )

            ds.to_netcdf(output_filename_t, group='diff_bas', mode='a')

        ds.close()



        ##############################
        # RMSE score
        ##############################

        rmse_score_exp_t = 1. - rmse_exp_t/rms_alongtrack
        if len(rmse_score_exp_t.shape)==2:
            rmse_score_exp_t = rmse_score_exp_t.mean(axis=1)
        self.leaderboard_rmse_exp = np.ma.mean(np.ma.masked_invalid(rmse_score_exp_t))
        self.leaderboard_rmse_std_exp = np.ma.std(np.ma.masked_invalid(rmse_score_exp_t))

        if self.compare_to_baseline:
            rmse_score_bas_t = 1. - rmse_bas_t/rms_alongtrack
            if len(rmse_score_bas_t.shape)==2:
                rmse_score_bas_t = rmse_score_bas_t.mean(axis=1)
            self.leaderboard_rmse_bas = np.ma.mean(np.ma.masked_invalid(rmse_score_bas_t))
            self.leaderboard_rmse_std_bas = np.ma.std(np.ma.masked_invalid(rmse_score_bas_t))



        ##############################
        # plotting
        ##############################

        if plot:
            if not self.compare_to_baseline:
                fig, (ax1, ax2) = plt.subplots(1,2,figsize=(2*(100/self.ratio_fig)**.5,1*(self.ratio_fig*100)**.5))
            else:
                fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(4*(100/self.ratio_fig)**.5,1*(self.ratio_fig*100)**.5))

            ax1.plot(vmean['time'], rmse_score_exp_t, label='experiment', c='b')
            if self.compare_to_baseline:
                ax1.plot(vmean['time'], rmse_score_bas_t, label='baseline', c='r')
            ax1.legend()
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

            im2 = ax2.pcolormesh(binning.x, binning.y, rmse_xy_exp,cmap='Reds')

            if self.compare_to_baseline:
                ax3.pcolormesh(binning.x, binning.y, rmse_xy_bas,cmap='Reds',vmin=np.nanmin(rmse_xy_exp),vmax=np.nanmax(rmse_xy_exp))
                plt.colorbar(im2,ax=[ax2,ax3])
                ax3.set_title('baseline')
                ax2.set_title('experiment')
                
                im = ax4.pcolormesh(binning.x, binning.y, 100*(rmse_xy_exp**2-rmse_xy_bas**2)/rmse_xy_bas**2,vmin=-50,vmax=50,cmap='RdBu_r')
                ax4.set_title('Improvement experiment/baseline')
                plt.colorbar(im,ax=ax4)
            else:
                plt.colorbar(im2,ax=ax2)
            
            plt.show()
            fig.savefig(f'{self.dir_output}/rmse.png',dpi=200)
            

    def _write_stat(self, nc, group_name, binning):
    
        grp = nc.createGroup(group_name)
        grp.createDimension('lon', len(binning.x))
        grp.createDimension('lat', len(binning.y))
        
        longitude = grp.createVariable('lon', 'f4', 'lon', zlib=True)
        longitude[:] = binning.x
        latitude = grp.createVariable('lat', 'f4', 'lat', zlib=True)
        latitude[:] = binning.y
        
        stats = ['min', 'max', 'sum', 'sum_of_weights', 'variance', 'mean', 'count', 'kurtosis', 'skewness']
        for variable in stats:
            
            var = grp.createVariable(variable, binning.variable(variable).dtype, ('lat','lon'), zlib=True)
            var[:, :] = binning.variable(variable).T 
        
    def psd_based_scores(self, threshold=0.5, plot=True):

        wavenumber = []
        psd_ref = []
        psd_exp = []
        psd_diff_exp = []
        if self.compare_to_baseline:
            psd_bas = []
            psd_diff_bas = []

        for i,_ref in enumerate(self.ref):

            ##########################
            # get data
            ##########################
            time_alongtrack = _ref[self.name_ref_time].values
            lon_alongtrack = _ref[self.name_ref_lon].values
            lat_alongtrack = _ref[self.name_ref_lat].values
            var_alongtrack = _ref.values
            var_exp_interp = self.exp_regridded[i].values
            if self.compare_to_baseline:
                var_bas_interp = self.bas_regridded[i].values
            if len(var_alongtrack.shape)==1:
                lon_alongtrack = lon_alongtrack[:,np.newaxis]
                lat_alongtrack = lat_alongtrack[:,np.newaxis]
                var_alongtrack = var_alongtrack[:,np.newaxis]
                var_exp_interp = var_exp_interp[:,np.newaxis]
                if self.compare_to_baseline:
                    var_bas_interp = var_bas_interp[:,np.newaxis]


                    
            for ac in range(lon_alongtrack.shape[1]):

                # Mask
                msk1 = np.ma.masked_invalid(var_alongtrack[:,ac]).mask
                msk2 = np.ma.masked_invalid(var_exp_interp[:,ac]).mask
                msk = msk1 + msk2
                if self.compare_to_baseline:
                    msk += np.ma.masked_invalid(var_bas_interp[:,ac]).mask
                _var_alongtrack = np.ma.masked_where(msk, var_alongtrack[:,ac]).compressed()
                _lon_alongtrack = np.ma.masked_where(msk, lon_alongtrack[:,ac]).compressed()
                _lat_alongtrack = np.ma.masked_where(msk, lat_alongtrack[:,ac]).compressed()
                _time_alongtrack = np.ma.masked_where(msk, time_alongtrack).compressed()
                _var_exp_interp = np.ma.masked_where(msk, var_exp_interp[:,ac]).compressed()
                if self.compare_to_baseline:
                    _var_bas_interp = np.ma.masked_where(msk, var_bas_interp[:,ac]).compressed()

                if _time_alongtrack.size==0:
                    continue

                ##########################
                # compute segments
                ##########################
                _, _, ref_segment, exp_segment, npt  = self._compute_segment_alongtrack(_time_alongtrack, 
                                                                                        _lat_alongtrack, 
                                                                                        _lon_alongtrack, 
                                                                                        _var_alongtrack, 
                                                                                        _var_exp_interp, 
                                                                                        self.lenght_scale,
                                                                                        self.delta_x[i])
                if self.compare_to_baseline:
                    _, _, _, bas_segment, _  = self._compute_segment_alongtrack(_time_alongtrack, 
                                                                                _lat_alongtrack, 
                                                                                _lon_alongtrack, 
                                                                                _var_alongtrack, 
                                                                                _var_bas_interp, 
                                                                                self.lenght_scale,
                                                                                self.delta_x[i])

                ##########################
                # spectral analysis
                ##########################
                # Power spectrum density reference field
                _wavenumber, _psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                                    fs=1.0 / self.delta_x[i],
                                                                    nperseg=npt,
                                                                    scaling='density',
                                                                    noverlap=0)

                # Power spectrum density experimental field
                _, _psd_exp = scipy.signal.welch(np.asarray(exp_segment).flatten(),
                                                        fs=1.0 / self.delta_x[i],
                                                        nperseg=npt,
                                                        scaling='density',
                                                        noverlap=0)
                if self.compare_to_baseline:
                    _, _psd_bas = scipy.signal.welch(np.asarray(bas_segment).flatten(),
                                                        fs=1.0 / self.delta_x[i],
                                                        nperseg=npt,
                                                        scaling='density',
                                                        noverlap=0)

                # Power spectrum density difference 
                _, _psd_diff_exp = scipy.signal.welch(np.asarray(exp_segment).flatten()-np.asarray(ref_segment).flatten(),
                                                        fs=1.0 / self.delta_x[i],
                                                        nperseg=npt,
                                                        scaling='density',
                                                        noverlap=0)
                if self.compare_to_baseline:
                    _, _psd_diff_bas = scipy.signal.welch(np.asarray(bas_segment).flatten()-np.asarray(ref_segment).flatten(),
                                                        fs=1.0 / self.delta_x[i],
                                                        nperseg=npt,
                                                        scaling='density',
                                                        noverlap=0)

                # Append to lists
                if _psd_ref.size>0:
                    wavenumber.append(_wavenumber)
                    psd_ref.append(_psd_ref)
                    psd_exp.append(_psd_exp)
                    psd_diff_exp.append(_psd_diff_exp)
                    if self.compare_to_baseline:
                        psd_bas.append(_psd_bas)
                        psd_diff_bas.append(_psd_diff_bas)

        # Interpolate to same wavenumbers
        argmin = np.argmin([len(_wavenumber) for _wavenumber in wavenumber])
        wavenumber0 = wavenumber[argmin]
        psd_interp = []
        for i in range(len(wavenumber)):
            psd_ref[i] = np.interp(wavenumber0,wavenumber[i],psd_ref[i])
            psd_exp[i] = np.interp(wavenumber0,wavenumber[i],psd_exp[i])
            psd_diff_exp[i] = np.interp(wavenumber0,wavenumber[i],psd_diff_exp[i])
            if self.compare_to_baseline:
                psd_bas[i] = np.interp(wavenumber0,wavenumber[i],psd_bas[i])
                psd_diff_bas[i] = np.interp(wavenumber0,wavenumber[i],psd_diff_bas[i])

        # Average along across track direction
        psd_ref = np.asarray(psd_ref).mean(axis=0)
        psd_exp = np.asarray(psd_exp).mean(axis=0)
        psd_diff_exp = np.asarray(psd_diff_exp).mean(axis=0)
        if self.compare_to_baseline:
            psd_bas = np.asarray(psd_bas).mean(axis=0)
            psd_diff_bas = np.asarray(psd_diff_bas).mean(axis=0)

        # Save psd in netcdf file
        ds = xr.Dataset({"psd_ref": (["wavenumber"], psd_ref),
                        "psd_exp": (["wavenumber"], psd_exp),
                        "psd_diff_exp": (["wavenumber"], psd_diff_exp),
                        },
                        coords={"wavenumber": (["wavenumber"], wavenumber0)},
                    )
        if self.compare_to_baseline:
            ds["psd_bas"] = (["wavenumber"], psd_bas)
            ds["psd_diff_bas"] = (["wavenumber"], psd_diff_bas)

        output_filename = f'{self.dir_output}/psd.nc'
        ds.to_netcdf(output_filename)

        # Resolved scales
        y = 1./wavenumber0
        x = (1. - psd_diff_exp/psd_ref)
        f = scipy.interpolate.interp1d(x, y)
        res_exp = f(threshold)
        self.leaderboard_psd_score_exp = res_exp
        if self.compare_to_baseline:
            x = (1. - psd_diff_bas/psd_ref)
            f = scipy.interpolate.interp1d(x, y)
            res_bas = f(threshold)
            self.leaderboard_psd_score_bas = res_bas

        ##########################
        # Plotting
        ##########################
        if plot:
            plt.figure(figsize=(10, 5))
            ax = plt.subplot(121)
            ax.invert_xaxis()
            plt.plot((1./ds.wavenumber), ds.psd_ref, label='reference', color='k', lw=2)
            plt.plot((1./ds.wavenumber), ds.psd_exp, label='experiment', color='b', lw=2)
            if self.compare_to_baseline:
                plt.plot((1./ds.wavenumber), ds.psd_bas, label='baseline', color='r', lw=2)
            plt.xlabel('wavelength [km]')
            plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.grid(which='both')
            
            ax = plt.subplot(122)
            ax.invert_xaxis()
            plt.plot((1./ds.wavenumber), (1. - ds.psd_diff_exp/ds.psd_ref), color='b', lw=2)
            if self.compare_to_baseline:
                plt.plot((1./ds.wavenumber), (1. - ds.psd_diff_bas/ds.psd_ref), color='r', lw=2)
            plt.xlabel('wavelength [km]')
            plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
            plt.xscale('log')
            plt.hlines(y=0.5, 
                    xmin=np.ma.min(np.ma.masked_invalid(1./ds.wavenumber)), 
                    xmax=np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                    color='r',
                    lw=0.5,
                    ls='--')
            plt.vlines(x=res_exp, ymin=0, ymax=1, lw=0.5, color='g')
            ax.fill_betweenx((1. - ds.psd_diff_exp/ds.psd_ref), 
                            res_exp, 
                            np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                            color='b',
                            alpha=0.3, 
                            label=f'experiment: $\lambda$ > {int(res_exp)}km')
            if self.compare_to_baseline:
                ax.fill_betweenx((1. - ds.psd_diff_bas/ds.psd_ref), 
                            res_bas, 
                            np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                            color='r',
                            alpha=0.3, 
                            label=f'baseline: $\lambda$ > {int(res_bas)}km')
            ax.set_ylim(0,1)
            plt.legend(loc='best', title="resolved scales")
            plt.grid(which='both')
            
            
            plt.show()
            plt.savefig(f'{self.dir_output}/psd.png', dpi=100)


    def _compute_segment_alongtrack(self,time_alongtrack, 
                                lat_alongtrack, 
                                lon_alongtrack, 
                                ssh_alongtrack, 
                                ssh_map_interp, 
                                lenght_scale,
                                delta_x,
                                ):

        segment_overlapping = 0.25
        max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

        list_lat_segment = []
        list_lon_segment = []
        list_ssh_alongtrack_segment = []
        list_ssh_map_interp_segment = []

        # Get number of point to consider for resolution = lenghtscale in km
        npt = int(lenght_scale / delta_x)

        # cut track when diff time longer than 4*delta_t
        indi = np.where(((np.diff(time_alongtrack) > max_delta_t_gap)))[0]
        if len(indi)>0:
            track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])
        else:
            track_segment_lenght = time_alongtrack.size

        # Long track >= npt
        selected_track_segment = np.where(track_segment_lenght >= npt)[0]

        if selected_track_segment.size > 0:

            for track in selected_track_segment:

                if track-1 >= 0:
                    index_start_selected_track = indi[track-1]
                    index_end_selected_track = indi[track]
                else:
                    index_start_selected_track = 0
                    index_end_selected_track = indi[track]

                start_point = index_start_selected_track
                end_point = index_end_selected_track

                for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                    # Near Greenwhich case
                    if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                        and (lon_alongtrack[sub_segment_point] > 320.)) \
                            or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                                and (lon_alongtrack[sub_segment_point] < 50.)):

                        tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                        lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                        lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                        mean_lon_sub_segment = np.median(tmp_lon)

                        if mean_lon_sub_segment < 0:
                            mean_lon_sub_segment = mean_lon_sub_segment + 360.
                    else:

                        mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                    mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                    ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                    ssh_map_interp_segment = []
                    ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                    if np.ma.is_masked(ssh_map_interp_segment):
                        ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                        ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                    if ssh_alongtrack_segment.size > 0:
                        list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                        list_lon_segment.append(mean_lon_sub_segment)
                        list_lat_segment.append(mean_lat_sub_segment)
                        list_ssh_map_interp_segment.append(ssh_map_interp_segment)


        return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 
        
    def movie(self,framerate=24,Display=True,clim=None,range_err=None,cmap='Spectral',delete_frames=False):
        
        # For memory leak when saving multiple png files...
        import matplotlib
        matplotlib.use('Agg')
        
        # Experimental data
        ssh_exp = self.exp.values
        lon_exp = self.exp[self.name_exp_lon].values
        lat_exp = self.exp[self.name_exp_lat].values
        if len(lon_exp.shape)==1:
            lon_exp,lat_exp = np.meshgrid(lon_exp,lat_exp)
        u_exp,v_exp = switchvar.ssh2uv(ssh_exp,lon=lon_exp,lat=lat_exp)
        U_exp = np.sqrt(u_exp**2+v_exp**2)
        rv_exp = switchvar.ssh2rv(ssh_exp,lon=lon_exp,lat=lat_exp,norm=True)

        # Ranges
        ssh_min = np.nanmin(ssh_exp)
        ssh_max = np.nanmax(ssh_exp)
        U_min = 0
        U_max = np.nanmax(U_exp)
        rv_min = -.5
        rv_max = .5

        # Baseline data
        if self.compare_to_baseline:
            bas_interp = self.bas.interp({self.name_bas_time:self.exp[self.name_bas_time]})    # time interpolation of baseline
            ssh_bas = bas_interp.values
            lon_bas = self.bas[self.name_bas_lon].values
            lat_bas = self.bas[self.name_bas_lat].values
            if len(lon_bas.shape)==1:
                lon_bas,lat_bas = np.meshgrid(lon_bas,lat_bas)
            u_bas,v_bas = switchvar.ssh2uv(ssh_bas,lon=lon_bas,lat=lat_bas)
            U_bas = np.sqrt(u_bas**2+v_bas**2)
            rv_bas = switchvar.ssh2rv(ssh_bas,lon=lon_bas,lat=lat_bas,norm=True)

        # Compute frames
        for t in range(self.exp[self.name_exp_time].size):

            if self.compare_to_baseline:
                fig, axs = plt.subplots(2,3,figsize=(3*(100/self.ratio_fig)**.5,2*(self.ratio_fig*100)**.5))
            else: 
                fig, axs = plt.subplots(1,3,figsize=(3*(100/self.ratio_fig)**.5,1*(self.ratio_fig*100)**.5))
                axs = axs[np.newaxis,:]

            fig.suptitle(str(self.exp[self.name_exp_time][t].values)[:13])

            im0 = axs[0,0].pcolormesh(lon_exp,lat_exp,ssh_exp[t],cmap=cmocean.cm.deep_r,vmin=ssh_min,vmax=ssh_max)
            axs[0,0].set_xlim(self.lon_min,self.lon_max)
            axs[0,0].set_ylim(self.lat_min,self.lat_max)
            axs[0,0].set_title(r'SSH$_{exp}$ [m]')
            plt.colorbar(im0,ax=axs[0,0])

            im1 = axs[0,1].pcolormesh(lon_exp,lat_exp,U_exp[t],cmap=cmocean.cm.speed,vmin=U_min,vmax=U_max)
            axs[0,1].set_xlim(self.lon_min,self.lon_max)
            axs[0,1].set_ylim(self.lat_min,self.lat_max)
            axs[0,1].set_title(r'U$_{exp}$ [m/s]')
            plt.colorbar(im1,ax=axs[0,1])

            im2 = axs[0,2].pcolormesh(lon_exp,lat_exp,rv_exp[t],cmap=cmocean.cm.diff,vmin=rv_min,vmax=rv_max)
            axs[0,2].set_xlim(self.lon_min,self.lon_max)
            axs[0,2].set_ylim(self.lat_min,self.lat_max)
            axs[0,2].set_title(r'$\xi_{exp}/f$')
            plt.colorbar(im2,ax=axs[0,2])

            # Baseline variables 
            if self.compare_to_baseline:
                im0 = axs[1,0].pcolormesh(lon_bas,lat_bas,ssh_bas[t],cmap=cmocean.cm.deep_r,vmin=ssh_min,vmax=ssh_max)
                plt.colorbar(im0,ax=axs[1,0])
                axs[1,0].set_xlim(self.lon_min,self.lon_max)
                axs[1,0].set_ylim(self.lat_min,self.lat_max)
                axs[1,0].set_title(r'SSH$_{baseline}$ [m]')
                im1 = axs[1,1].pcolormesh(lon_bas,lat_bas,U_bas[t],cmap=cmocean.cm.speed,vmin=U_min,vmax=U_max)
                axs[1,1].set_xlim(self.lon_min,self.lon_max)
                axs[1,1].set_ylim(self.lat_min,self.lat_max)
                axs[1,1].set_title(r'U$_{baseline}$ [m/s]')
                plt.colorbar(im1,ax=axs[1,1])
                im2 = axs[1,2].pcolormesh(lon_bas,lat_bas,rv_bas[t],cmap=cmocean.cm.diff,vmin=rv_min,vmax=rv_max)
                axs[1,2].set_xlim(self.lon_min,self.lon_max)
                axs[1,2].set_ylim(self.lat_min,self.lat_max)
                axs[1,2].set_title(r'$\xi_{baseline}/f$')
                plt.colorbar(im2,ax=axs[1,2])

            fig.savefig(f'{self.dir_output}/frame_{str(t).zfill(5)}.png',dpi=200)

            plt.close(fig)
            del fig
            gc.collect(2)

        # Create movie
        sourcefolder = self.dir_output
        moviename = 'movie.mp4'
        frame_pattern = 'frame_*.png'
        ffmpeg_options="-c:v libx264 -preset veryslow -crf 15 -pix_fmt yuv420p"

        command = 'ffmpeg -f image2 -r %i -pattern_type glob -i %s -y %s -r %i %s' % (
                framerate,
                os.path.join(sourcefolder, frame_pattern),
                ffmpeg_options,
                framerate,
                os.path.join(self.dir_output, moviename),
            )
        print(command)

        _ = subprocess.run(command.split(' '),stdout=subprocess.PIPE)

        # Delete frames
        if delete_frames:
            os.system(f'rm {os.path.join(sourcefolder, frame_pattern)}')

        # Display movie
        if Display:
            Video(os.path.join(self.dir_output, moviename),embed=True)
        return 
    
    def Leaderboard(self):

        data = [[self.name_experiment, 
            np.round(self.leaderboard_rmse_exp,2), 
            np.round(self.leaderboard_rmse_std_exp,2), 
            np.round(self.leaderboard_psd_score_exp,2)]]

        if self.compare_to_baseline:
            data.append(['baseline', 
                np.round(self.leaderboard_rmse_bas,2), 
                np.round(self.leaderboard_rmse_std_bas,2), 
                np.round(self.leaderboard_psd_score_bas,2),])

         
        Leaderboard = pd.DataFrame(data, 
                                columns=['Method', 
                                            "µ(RMSE) ", 
                                            "σ(RMSE)", 
                                            'λx (km)'])

        with open(f'{self.dir_output}/metrics.txt', 'w') as f:
            dfAsString = Leaderboard.to_string()
            f.write(dfAsString)
        
        return Leaderboard

class Diag_multi:

    def __init__(self,config,State):
        
        self.dir_output = f'{config.EXP.path_save}/diags/'
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)


        self.name_diag = config.DIAG
        self.Diag = []
        _config = config.copy()

        for _DIAG in config.DIAG:
            _config.DIAG = config.DIAG[_DIAG]
            _Diag = Diag(_config,State)
            print()
            _Diag.dir_output = os.path.join(_Diag.dir_output,_DIAG)
            if not os.path.exists(_Diag.dir_output):
                os.makedirs(_Diag.dir_output)
            self.Diag.append(_Diag)

    def regrid_exp(self):

        for _Diag in self.Diag:
            _Diag.regrid_exp()
    
    def rmse_based_scores(self,plot=False):

        for _Diag in self.Diag:
            _Diag.rmse_based_scores(plot=plot)
        
    def psd_based_scores(self,plot=False,threshold=0.5):

        for _Diag in self.Diag:
            _Diag.psd_based_scores(plot=plot,threshold=threshold)
    
    def movie(self,framerate=24,Display=True,clim=None,range_err=None,cmap='Spectral'):

        for _Diag in self.Diag:
            _Diag.movie(framerate=framerate,Display=Display,clim=clim,range_err=range_err,cmap=cmap)
        
    def Leaderboard(self):

        df = []
        names = []
        for (_Diag, name) in zip(self.Diag,self.name_diag):
            _df = _Diag.Leaderboard()
            df.append(_df)
            names += [name,]*_df.shape[0]
        
        Leaderboard = pd.concat(df)
        Leaderboard.insert(0,'Diags',names)

        with open(f'{self.dir_output}/metrics.txt', 'w') as f:
            dfAsString = Leaderboard.to_string()
            f.write(dfAsString)

        return Leaderboard