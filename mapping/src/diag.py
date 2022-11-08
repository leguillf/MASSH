import os
import numpy as np
import xarray as xr
import pyinterp 
import pyinterp.fill
import logging
import xrft
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from dask import delayed,compute
import gc


def Diag(config,State):

    """
    NAME
        Diag

    DESCRIPTION
        Main function calling subclass for specific diagnostics
    """

    print('Diags:',config.name_diag)
    if config.name_diag is None:
        return
    elif config.name_diag=='OSSE':
        return Diag_osse(config,State)


class Diag_osse():

    """
    NAME
        Diag_osse

    DESCRIPTION
        Class to compute OSSE diagnostics
    """

    def __init__(self,config,State):
        
        self.dir_output = f'{config.path_save}/diags/'
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        # time_min
        if config.OSSE['time_min'] is None:
            self.time_min = config.init_date
        else:
            self.time_min = config.OSSE['time_min']
        # time_max
        if config.OSSE['time_max'] is None:
            self.time_max = config.final_date
        else:
            self.time_max = config.OSSE['time_max']
        # time_step
        if config.OSSE['time_step'] is None:
            self.time_step = config.saveoutput_time_step
        else:
            self.time_step = config.OSSE['time_step']
        # lon_min
        if config.OSSE['lon_min'] is None:
            self.lon_min = State.lon.min()
        else:
            self.lon_min = config.OSSE['lon_min']
        # lon_max
        if config.OSSE['lon_max'] is None:
            self.lon_max = State.lon.max()
        else:
            self.lon_max = config.OSSE['lon_max']
        # lat_min
        if config.OSSE['lat_min'] is None:
            self.lat_min = State.lat.min()
        else:
            self.lat_min = config.OSSE['lat_min']
        # lat_max
        if config.OSSE['lat_max'] is None:
            self.lat_max = State.lat.max()
        else:
            self.lat_max = config.OSSE['lat_max']

        # Reference data
        self.name_ref_time = config.OSSE['name_ref_time']
        self.name_ref_lon = config.OSSE['name_ref_lon']
        self.name_ref_lat = config.OSSE['name_ref_lat']
        self.name_ref_var = config.OSSE['name_ref_var']
        ref = xr.open_mfdataset(config.OSSE['name_ref'],**config.OSSE['options_ref'])
        ref = ref.assign_coords({self.name_ref_lon:ref[self.name_ref_lon]%360})
        dt = (ref[self.name_ref_time][1]-ref[self.name_ref_time][0]).values/np.timedelta64(1,'s')
        idt = int(self.time_step.total_seconds()//dt)
        self.ref = ref.sel(
            {self.name_ref_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max),idt),
             self.name_ref_lon:slice(self.lon_min,self.lon_max),
             self.name_ref_lat:slice(self.lat_min,self.lat_max)}
             )
        ref.close()

        # Experimental data
        self.name_exp_time = State.name_time
        self.name_exp_lon = State.name_lon
        self.name_exp_lat = State.name_lat
        self.name_exp_var = config.name_mod_var[0]
        exp = xr.open_mfdataset(f'{config.path_save}/{config.name_exp_save}*nc')
        exp = exp.assign_coords({self.name_ref_lon:exp[self.name_ref_lon]%360})
        self.exp = exp.sel(
            {self.name_exp_time:slice(np.datetime64(self.time_min),np.datetime64(self.time_max)),
             self.name_exp_lon:slice(self.lon_min,self.lon_max),
             self.name_exp_lat:slice(self.lat_min,self.lat_max)}
             )
        exp.close()

    def regrid_exp(self):
        
        # Define source grid
        x_source_axis = pyinterp.Axis(self.exp[self.name_exp_lon][:].values, is_circle=False)
        y_source_axis = pyinterp.Axis(self.exp[self.name_exp_lat][:].values)
        z_source_axis = pyinterp.TemporalAxis(self.exp[self.name_exp_time][:].values)
        ssh_source = self.exp[self.name_exp_var].T
        grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
        
        # Define target grid
        mx_target, my_target, mz_target = np.meshgrid(
            self.ref[self.name_ref_lon][:].values,
            self.ref[self.name_ref_lat][:].values,
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
            has_converged, filled = pyinterp.fill.gauss_seidel(grid)
        else:
            filled = ssh_interp.T
        
        # Save to dataset
        self.exp_regridded = xr.Dataset({self.name_exp_var : ((self.name_ref_time, self.name_ref_lat, self.name_ref_lon), filled.T)},
                                coords={self.name_ref_time: self.ref[self.name_ref_time].values,
                                        self.name_ref_lon: self.ref[self.name_ref_lon].values, 
                                        self.name_ref_lat: self.ref[self.name_ref_lat].values, 
                                        })
    
    
    def rmse_based_scores(self):
        
        logging.info('     Compute RMSE-based scores...')

        # RMSE(t) based score
        rmse_t = 1.0 - (((self.exp_regridded[self.name_exp_var] - self.ref[self.name_ref_var])**2).mean(
            dim=(self.name_ref_lon, self.name_ref_lat)))**0.5/(((self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_lon, self.name_ref_lat)))**0.5
        # RMSE(x, y) based score
        rmse_xy = (((self.exp_regridded[self.name_exp_var] - self.ref[self.name_ref_var])**2).mean(dim=(self.name_ref_time)))**0.5
        
        rmse_t = rmse_t.rename('rmse_t')
        rmse_xy = rmse_xy.rename('rmse_xy')

        # Temporal stability of the error
        reconstruction_error_stability_metric = rmse_t.std().values

        # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
        leaderboard_rmse = 1.0 - (((self.exp_regridded[self.name_exp_var] - self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5 / (
            ((self.ref[self.name_ref_var]) ** 2).mean()) ** 0.5

        logging.info('          => Leaderboard SSH RMSE score = %s', np.round(leaderboard_rmse.values, 2))
        logging.info('          Error variability = %s (temporal stability of the mapping error)', np.round(reconstruction_error_stability_metric, 2))

        rmse_t.to_netcdf(f'{self.dir_output}/rmse_t.nc')
        rmse_xy.to_netcdf(f'{self.dir_output}/rmse_xy.nc')

        self.rmse_t = rmse_t
        self.rmse_xy = rmse_xy

        return np.round(leaderboard_rmse.values, 2), np.round(reconstruction_error_stability_metric, 2)

    def psd_based_scores(self,threshold=0.5):
        
        logging.info('     Compute PSD-based scores...')
        
        with ProgressBar():

            mean_psd_signal = psd(
                self.ref[self.name_ref_var],
                dim=[self.name_ref_time,self.name_ref_lon],
                dim_mean=self.name_ref_lat)
            
            mean_psd_err = psd(
                (self.exp_regridded[self.name_exp_var] - self.ref[self.name_ref_var]),
                dim=[self.name_ref_time,self.name_ref_lon],
                dim_mean=self.name_ref_lat)
            
            # return PSD-based score
            psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)
            plot_psd_score_v0(psd_based_score)

            # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
            level = [threshold]
            cs = plt.contour(1./psd_based_score[f'freq_{self.name_ref_lon}'].values,1./psd_based_score[f'freq_{self.name_ref_time}'].values, psd_based_score, level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            
            shortest_spatial_wavelength_resolved = np.min(x05)
            shortest_temporal_wavelength_resolved = np.min(y05)/3600/24 # in days

            logging.info('          => Leaderboard Spectral score = %s (degree lon)',
                        np.round(shortest_spatial_wavelength_resolved, 2))
            logging.info('          => shortest temporal wavelength resolved = %s (days)',
                        np.round(shortest_temporal_wavelength_resolved, 2))

            mean_psd_signal.to_netcdf(f'{self.dir_output}/mean_psd_signal.nc')
            mean_psd_err.to_netcdf(f'{self.dir_output}/mean_psd_err.nc')

            self.mean_psd_signal = mean_psd_signal
            self.mean_psd_err = mean_psd_err
            self.psd_based_score = psd_based_score

            return  np.round(shortest_spatial_wavelength_resolved, 2), np.round(shortest_temporal_wavelength_resolved, 2)

    def movie(self,n_workers=1,framerate=24):

        # Create merged dataset
        coords = (self.name_ref_time,self.name_ref_lat,self.name_ref_lon)
        ds = xr.Dataset(
            {'ref':(coords,self.ref.sossheig.data),
            'exp':(coords,self.exp_regridded.ssh.data),
            'rmse_score':(self.name_ref_time,self.rmse_t.data)},
            coords=(
                {self.name_ref_time:self.ref[self.name_ref_time],
                self.name_ref_lat:self.ref[self.name_ref_lat],
                self.name_ref_lon:self.ref[self.name_ref_lon]})
        )
        ds = ds.chunk({self.name_ref_time:1})

        # Plotting parameters
        xlim = (ds[self.name_ref_time][0].values,ds[self.name_ref_time][-1].values)
        ylim = (ds.rmse_score.min().values,ds.rmse_score.max().values)
        clim = (ds.ref.min().values,ds.ref.max().values)
        cmap = 'Spectral'

        # Plotting function
        def _save_single_frame(ds, tt, xlim=xlim, ylim=ylim,clim=clim,cmap=cmap):

            if tt==0:
                return 
            
            fig = plt.figure(figsize=(10,10))

            date = str(ds[self.name_ref_time][tt].values)[:13]

            gs = gridspec.GridSpec(2,3,width_ratios=(1,1,0.1))

            ids = ds.isel(time=tt)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ids.ref.plot(ax=ax1,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax1.set_title('Reference')

            ax2 = fig.add_subplot(gs[0, 1])
            im = ids.exp.plot(ax=ax2,cmap=cmap,vmin=clim[0],vmax=clim[1],add_colorbar=False)
            ax2.set_ylabel('')
            ax2.set_title('Mapping')

            ax3 = fig.add_subplot(gs[0, 2])
            plt.colorbar(im,cax=ax3)

            ids = ds.isel(time=slice(0,tt))
            ax = fig.add_subplot(gs[1, :])
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

        command = 'ffmpeg -f image2 -r %i -pattern_type glob -i "%s" -y %s -r %i "%s"' % (
                framerate,
                os.path.join(sourcefolder, frame_pattern),
                ffmpeg_options,
                framerate,
                os.path.join(self.dir_output, moviename),
            )
        os.system(command)

        # Delete frames
        os.system(f'rm {os.path.join(sourcefolder, frame_pattern)}')
        


def plot_psd_score_v0(ds_psd):
        
    try:
        nb_experiment = len(ds_psd.experiment)
    except:
        nb_experiment = 1
    
    fig, ax0 =  plt.subplots(1, nb_experiment, sharey=True, figsize=(nb_experiment*10, 5))

    if nb_experiment==1:
        ax0 = [ax0]

    for exp in range(nb_experiment):

        ax = ax0[exp]
        try:
            ctitle = ds_psd.experiment.values[exp]
        except:
            ctitle = ''
        if nb_experiment > 1:
            data = (ds_psd.isel(experiment=exp).values)
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
    
