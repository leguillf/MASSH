import os, sys
import numpy as np
import xarray as xr
import pyinterp 
import pyinterp.fill
import logging
import xrft
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from IPython.display import Video
from matplotlib.ticker import ScalarFormatter
from dask import delayed,compute
import gc
import pandas as pd 
import subprocess


def Diag(config,State):

    """
    NAME
        Diag

    DESCRIPTION
        Main function calling subclass for specific diagnostics
    """

    if config.DIAG is None:
        return
    
    print(config.DIAG)


    if config.DIAG.super=='DIAG_OSSE':
        return Diag_osse(config,State)
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
            self.lon_min = State.lon_min
        else:
            self.lon_min = config.DIAG.lon_min
        # lon_max
        if config.DIAG.lon_max is None:
            self.lon_max = State.lon_max
        else:
            self.lon_max = config.DIAG.lon_max
        # lat_min
        if config.DIAG.lat_min is None:
            self.lat_min = State.lat_min
        else:
            self.lat_min = config.DIAG.lat_min
        # lat_max
        if config.DIAG.lat_max is None:
            self.lat_max = State.lat_max
        else:
            self.lat_max = config.DIAG.lat_max


        # Reference data
        self.name_ref_time = config.DIAG.name_ref_time
        self.name_ref_lon = config.DIAG.name_ref_lon
        self.name_ref_lat = config.DIAG.name_ref_lat
        self.name_ref_var = config.DIAG.name_ref_var
        ref = xr.open_mfdataset(config.DIAG.name_ref,**config.DIAG.options_ref)
        ref = ref.assign_coords({self.name_ref_lon:ref[self.name_ref_lon]%360})
        dt = (ref[self.name_ref_time][1]-ref[self.name_ref_time][0]).values/np.timedelta64(1,'s')
        idt = int(self.time_step.total_seconds()//dt)
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

        # Experimental data
        self.geo_grid = State.geo_grid
        self.name_exp_time = config.EXP.name_time
        self.name_exp_lon = config.EXP.name_lon
        self.name_exp_lat = config.EXP.name_lat
        self.name_exp_var = config.DIAG.name_exp_var
        exp = xr.open_mfdataset(f'{config.EXP.path_save}/{config.EXP.name_exp_save}*nc')
        exp = exp.assign_coords({self.name_ref_lon:exp[self.name_ref_lon]%360})
        self.exp = exp.sel(
            {self.name_exp_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max))},
             )
        try:
            self.exp = self.exp.sel(
                {self.name_exp_lon:slice(self.lon_min,self.lon_max),
                self.name_exp_lat:slice(self.lat_min,self.lat_max)}
            )
        except:
            print('Warning: unable to select study region in the experiment fields.\
That could be due to non regular grid or bad written netcdf file')
        exp.close()

        # Baseline data
        self.compare_to_baseline = config.DIAG.compare_to_baseline 
        if self.compare_to_baseline:
            self.name_bas_time = config.DIAG.name_bas_time
            self.name_bas_lon = config.DIAG.name_bas_lon
            self.name_bas_lat = config.DIAG.name_bas_lat
            self.name_bas_var = config.DIAG.name_bas_var
            bas = xr.open_mfdataset(config.DIAG.name_bas)
            bas = bas.assign_coords({self.name_bas_lon:bas[self.name_bas_lon]%360})
            self.bas = bas.sel(
                {self.name_bas_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max))},
                )
            try:
                self.bas = self.bas.sel(
                    {self.name_bas_lon:slice(self.lon_min,self.lon_max),
                    self.name_bas_lat:slice(self.lat_min,self.lat_max)}
                )
            except:
                print('Warning: unable to select study region in the baseline fields.')
            bas.close()


    def regrid_exp(self):
        
        if self.geo_grid:
            self.exp_regridded =  self._regrid_geo(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp[self.name_exp_var],
                self.name_exp_var)
        else:
            self.exp_regridded = self._regrid_unstructured(
                self.exp[self.name_exp_lon].values,
                self.exp[self.name_exp_lat].values, 
                self.exp[self.name_exp_time].values, 
                self.exp[self.name_exp_var],
                self.name_exp_var)
        
        if self.compare_to_baseline:
            self.bas_regridded = self._regrid_geo(
                self.bas[self.name_bas_lon].values,
                self.bas[self.name_bas_lat].values, 
                self.bas[self.name_bas_time].values, 
                self.bas[self.name_bas_var],
                self.name_bas_var)        
    
    def _regrid_geo(self, lon, lat, time, var, name_var):

        # Define source grid
        x_source_axis = pyinterp.Axis(lon, is_circle=False)
        y_source_axis = pyinterp.Axis(lat)
        z_source_axis = pyinterp.TemporalAxis(time)
        ssh_source = var.T
        grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
        
        # Define target grid
        mx_target, my_target, mz_target = np.meshgrid(
            self.ref[self.name_ref_lon][:].values.flatten(),
            self.ref[self.name_ref_lat][:].values.flatten(),
            z_source_axis.safe_cast(np.ascontiguousarray(self.ref[self.name_ref_time][:].values)),
            indexing="ij")

        # Spatio-temporal Interpolation
        ssh_interp = pyinterp.trivariate(grid_source,
                                        mx_target.flatten(),
                                        my_target.flatten(),
                                        mz_target.flatten(),
                                        bounds_error=False).reshape(mx_target.shape).T
        
        # MB add extrapolation in NaN values if needed
        if np.isnan(ssh_interp).any():
            x_source_axis = pyinterp.Axis(self.ref[self.name_ref_lon].values, is_circle=False)
            y_source_axis = pyinterp.Axis(self.ref[self.name_ref_lat].values)
            z_source_axis = pyinterp.TemporalAxis(np.ascontiguousarray(self.ref[self.name_ref_time][:].values))
            grid = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis,  ssh_interp.T)
            _, filled = pyinterp.fill.gauss_seidel(grid)
        else:
            filled = ssh_interp.T
        
        # Save to dataset
        return xr.DataArray(
            data=filled.T,
            coords={self.name_ref_time: self.ref[self.name_ref_time].values,
                    self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                    self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                    },
            dims=[self.name_ref_time, self.name_ref_lat, self.name_ref_lon]
            )
    
    def _regrid_unstructured(self, lon, lat, time, var, name_var):
        
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

        # Mask
        var_regridded.data[np.isnan(self.ref[self.name_ref_var])] = np.nan

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
                fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,4))
                rmse_t.sel(run='experiment').plot(ax=ax1,label='experiment')
                rmse_t.sel(run='baseline').plot(ax=ax1,label='baseline')
                ax1.set_title(None)
                ax1.legend()
                rmse_xy.sel(run='experiment').plot(ax=ax2,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax2.set_title('experiment')
                rmse_xy.sel(run='baseline').plot(ax=ax3,cmap='Reds',vmin=0,vmax=rmse_xy.max().values)
                ax3.set_title('baseline')

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
        
        logging.info('     Compute PSD-based scores...')
        
        mean_psd_signal = psd(
            self.ref[self.name_ref_var],
            dim=[self.name_ref_time,self.name_ref_lon],
            dim_mean=self.name_ref_lat)
        
        mean_psd_err = psd(
            (self.exp_regridded - self.ref[self.name_ref_var]),
            dim=[self.name_ref_time,self.name_ref_lon],
            dim_mean=self.name_ref_lat)

        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        if self.compare_to_baseline:
            mean_psd_err_bas = psd(
                (self.bas_regridded - self.ref[self.name_ref_var]),
                dim=[self.name_ref_time,self.name_ref_lon],
                dim_mean=self.name_ref_lat)
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
            fig = plot_psd_score_v0(psd_based_score)
            fig.savefig(f'{self.dir_output}/psd.png',dpi=100)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
        level = [threshold]

        if self.compare_to_baseline:
            # Experiment
            cs = plt.contour(1./psd_based_score[f'freq_{self.name_ref_lon}'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='experiment'), level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score = np.min(x05)
            self.leaderboard_psdt_score = np.min(y05)/3600/24 # in days
            # Baseline
            cs = plt.contour(1./psd_based_score[f'freq_{self.name_ref_lon}'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score.sel(run='baseline'), level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score_bas = np.min(x05)
            self.leaderboard_psdt_score_bas = np.min(y05)/3600/24 # in days
        else:
            cs = plt.contour(1./psd_based_score[f'freq_{self.name_ref_lon}'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score, level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.leaderboard_psds_score = np.min(x05)
            self.leaderboard_psdt_score = np.min(y05)/3600/24 # in days


    def movie(self,framerate=24,Display=True,clim=None,range_err=None,cmap='Spectral'):

    
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

    
            ids = ds.isel(time=slice(0,tt))
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
        delayed_results = []
        for tt in range(ds[self.name_ref_time].size):
            res = delayed(_save_single_frame)(ds, tt)
            delayed_results.append(res)
        results = compute(*delayed_results, scheduler="threads")

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
        
        return Leaderboard

def plot_psd_score_v0(ds_psd):
        
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
        c1 = ax.contourf(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data,
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
        c2 = ax.contour(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data, levels=[0.5], linewidths=2, colors='k')
        
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
    

def psd(da,dim,dim_mean=None,detrend='constant'):

        # Rechunk 
        chunks = {}
        if type(dim)!=list:
            dim = [dim]
        for d in dim:
            chunks[d] = da[d].size
        signal = da.chunk(chunks)

        # Compute PSD
        psd = xrft.power_spectrum(
            signal, 
            dim=dim, 
            detrend=detrend, 
            window=True).compute()
        
        # Averaged 
        if dim_mean is not None:
            ispos = True
            for d in dim:
                ispos = ispos & (psd[f'freq_{d}'] > 0.)
            psd = psd.mean(dim=dim_mean).where(ispos, drop=True)
        
        psd.name = f'PSD_{da.name}'
        
        return psd
    
