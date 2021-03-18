#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:23:11 2019

@author: leguillou
"""

import sys
import numpy as np
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt
import pickle
import os.path
from scipy.ndimage.filters import gaussian_filter
import glob

from . import switchvar, grid
from .tools import gaspari_cohn, hat_function, L2_scalar_prod



def bfn(config,dt_start,dt_end,one_time_step,lon,lat):
    
    # Use temp_DA_path to save the projections
    if config.save_obs_proj:
        if config.path_save_proj is None:
            pathsaveproj = config.tmp_DA_path
        else:
            pathsaveproj = config.path_save_proj
    else:
        pathsaveproj = None


    if config.name_model=='QG1L':
        return bfn_qg1l(dt_start,
                        dt_end,
                        config.assimilation_time_step,
                        one_time_step,
                        lon,
                        lat,
                        config.name_mod_var,
                        config.name_grd,
                        config.dist_scale,
                        pathsaveproj,
                        'projections_' + config.name_domain + '_' +\
                            '_'.join(config.satellite),
                        config.c,
                        config.flag_plot,
                        config.scalenudg)
                
    elif config.name_model == 'QGML':
            
            return bfn_qgml(dt_start,
                            dt_end,
                            config.assimilation_time_step,
                            one_time_step,
                            lon,
                            lat,
                            config.name_mod_var,
                            config.name_grd,
                            config.dist_scale,
                            config.Rom,
                            config.Fr,
                            config.dh,
                            config.N,
                            config.L0,
                            pathsaveproj,
                            'projections_' + config.name_domain + '_' +\
                                '_'.join(config.satellite),
                            config.flag_plot,
                            config.scalenudg)
    else:
        sys.exit('Error: No BFN class implemented for',config.name_model,'model')
        
class bfn_qg1l(object):
    def __init__(self,
                 dt_start,
                 dt_end,
                 assim_time_step,
                 model_time_step,
                 lon,
                 lat,
                 name_mod_var,
                 name_grd,
                 dist_scale,
                 path_save=None,
                 name_save=None,
                 c = 2.5,
                 flag_plot=0,
                 scalenudg=(1, 1),
                 qgiter=1):

        # Time parameters
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.assim_time_step = assim_time_step
        self.model_time_step = model_time_step
        # Grid coordinates
        if len(lon.shape)==1:
            lon,lat = np.meshgrid(lon,lat)
        self.lon2d = lon
        self.lat2d = lat
        self.ny, self.nx = self.lon2d.shape
        self.f = 4*np.pi/86164*np.sin(self.lat2d*np.pi/180)
        self.g = 9.81
        self.name_grd = name_grd
        # Model variables
        self.name_mod_var = name_mod_var # Must be [SSH, PV, K] or [SSH, PV]
        self.n_mod_var = len(name_mod_var)
        # Observation variables
        self.dist_scale = dist_scale
        self.dict_obs_ssh = {}
        self.dict_obs_rv = {}
        # Projection variables
        self.path_save = path_save
        self.name_save = name_save
        self.dict_proj_ssh = {}
        self.dict_proj_rv = {}
        # Model parameters
        self.c = c
        self.K = (self.f/self.c)**2
        # Plotting parameter
        self.flag_plot = flag_plot
        # Sponge
        self.sponge = 'gaspari-cohn'
        #re-scaling coefficitent for nudging
        if scalenudg is None:
            self.scalenudg = (1, 1)
        else:
            self.scalenudg = scalenudg

    def select_obs(self, dict_obs):

        self.dict_obs_ssh = bfn_select_obs_temporal_window(
            dict_obs,
            self.dt_start,
            self.dt_end,
            self.assim_time_step,
            'nudging_params_stretching'
            )

        self.dict_obs_rv = bfn_select_obs_temporal_window(
            dict_obs,
            self.dt_start,
            self.dt_end,
            self.assim_time_step,
            'nudging_params_relvort'
            )


    def do_projections(self):

        self.dict_proj_ssh = bfn_projections(
            'ssh',
            self.dict_obs_ssh,
            self.lon2d, self.lat2d,
            self.dist_scale,
            self.flag_plot,
            self.path_save,
            self.name_save)

        self.dict_proj_rv = bfn_projections(
            'relvort',
            self.dict_obs_rv,
            self.lon2d, self.lat2d,
            self.dist_scale,
            self.flag_plot,
            self.path_save,
            self.name_save)

        return

    def compute_nudging_term(self, date, model_state):

        # Get model SSH state
        ssh = model_state.getvar(0) # 1st variable (ssh)

        # Get observations and nudging parameters
        obs_ssh, nudging_coeff_ssh, sigma_ssh =\
            bfn_get_data_at_t(date,
                              self.dict_proj_ssh)
        obs_rv, nudging_coeff_rv, sigma_rv =\
            bfn_get_data_at_t(date,
                              self.dict_proj_rv)



        # Compute nudging term
        N = {'ssh':np.zeros_like(ssh), 'rv':np.zeros_like(ssh)}

        if obs_rv is not None and np.any(np.isfinite(obs_rv)):
            # Nudging towards relative vorticity
            rv = switchvar.ssh2rv(
                ssh, self.lon2d, self.lat2d, name_grd=self.name_grd)
            nobs = len(obs_rv)
            for iobs in range(nobs):
                indNoNan = ~np.isnan(obs_rv[iobs])
                if np.any(indNoNan):
                    # Filter model state for spectral nudging
                    rv_ls = rv.copy()
                    if sigma_rv[iobs] is not None and sigma_rv[iobs]>0:
                        rv_ls = gaussian_filter(rv_ls,sigma=sigma_rv[iobs])
                    N['rv'][indNoNan] += nudging_coeff_rv[iobs,indNoNan] *\
                        (obs_rv[iobs,indNoNan]-rv_ls[indNoNan])

        if obs_ssh is not None and np.any(np.isfinite(obs_ssh)):
            # Nudging towards ssh
            nobs = len(obs_ssh)
            for iobs in range(nobs):
                indNoNan = ~np.isnan(obs_ssh[iobs])
                if np.any(indNoNan):
                    # Filter model state for spectral nudging
                    ssh_ls = ssh.copy()
                    if sigma_ssh[iobs] is not None and sigma_ssh[iobs]>0:
                        ssh_ls = gaussian_filter(ssh_ls,sigma=sigma_ssh[iobs])
                    N['ssh'][indNoNan] += nudging_coeff_ssh[iobs,indNoNan] *\
                         (obs_ssh[iobs,indNoNan]-ssh_ls[indNoNan])

        # Mask pixels that are not influenced by observations
        N['ssh'] = N['ssh'] * self.scalenudg[0]
        N['rv'] = N['rv'] * self.scalenudg[1]
        N['ssh'][N['ssh']==0] = np.nan
        N['rv'][N['rv']==0] = np.nan

        if self.flag_plot>3:
            plt.figure()
            plt.suptitle('Nudging coefficient')
            plt.pcolormesh(self.lon2d,self.lat2d,N['ssh'])
            plt.colorbar()
            plt.show()

        return N

    def update_parameter(self, model_state, Nold, N, Wbc, way=1):

        if self.n_mod_var<3 or None in [Nold, N]:
            return model_state

        ssh = model_state.getvar(0)
        psi =  self.g/self.f * ssh # 1st variable ssh
        K = model_state.getvar(2)

        dt = np.abs(self.model_time_step.total_seconds())

        if psi.max()==0:
            return model_state

        coeff = np.min(K)/psi.max()

        if Wbc is None:
            Wbc = np.zeros((self.ny,self.nx))

        K_incr =  way * coeff * L2_scalar_prod(
                        (1-Wbc)*ssh,
                        (1-Wbc)*(self.g/self.f)*(N['ssh'] - Nold['ssh'])/dt
                        )

        if np.abs(K_incr)<0.1*np.mean(K):
            K += K_incr

        model_state.setvar(2,K)


    def convergence(self, path_forth, path_back):
        
        err = 0
        
        files_forth = sorted(glob.glob(path_forth))
        files_back = sorted(glob.glob(path_back))
        
        for (ff,fb) in zip(files_forth,files_back):
            dsf = xr.open_dataset(ff,engine='h5netcdf')
            dsb = xr.open_dataset(fb,engine='h5netcdf')
            for name_var in self.name_mod_var:
                varf = dsf[name_var].values
                varb = dsb[name_var].values
                if varf.size != 0 and np.std(varf)>0:
                    err += np.sum(np.abs(varf**2-varb**2))/np.std(varf)/varf.size
            dsf.close()
            dsb.close()
        
        return err


class bfn_qgml(object):
    def __init__(self,
                 dt_start,
                 dt_end,
                 assim_time_step,
                 model_time_step,
                 lon2d,
                 lat2d,
                 name_mod_var,
                 name_grd,
                 dist_scale,
                 Rom,
                 Fr,
                 dh,
                 N,
                 L0,
                 path_save=None,
                 name_save=None,
                 flag_plot=0,
                 scalenudg=(1, 1, 1)):

        # Time parameters
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.assim_time_step = assim_time_step
        self.model_time_step = model_time_step
        # Grid coordinates
        self.lon2d = lon2d
        self.lat2d = lat2d
        self.ny, self.nx = self.lon2d.shape
        self.f = 4*np.pi/86164*np.sin(self.lat2d*np.pi/180)
        self.g = 9.81
        self.name_grd = name_grd
        # Model variables
        self.name_mod_var = name_mod_var # Must be [SSH, PV2, .., PVn]
        self.n_mod_var = len(name_mod_var)
        # Observation variables
        self.dist_scale = dist_scale
        self.dict_obs_ssh = {}
        # Projection variables
        self.path_save = path_save
        self.name_save = name_save
        self.dict_proj_ssh = {}
        # Model parameters
        self.dh12 = 0.5 * (dh[0]+dh[1])
        self.Fr = Fr
        self.Rom = Rom
        self.dx = L0/N
        # Plotting parameter
        self.flag_plot = flag_plot
        #re-scaling coefficitent for nudging
        if scalenudg is None:
            self.scalenudg = (1, 1, 1)
        else:
            self.scalenudg = scalenudg
        # Sponge
        self.sponge = 'linear'

    def select_obs(self, dict_obs):

        self.dict_obs_ssh = bfn_select_obs_temporal_window(
            dict_obs,
            self.dt_start,
            self.dt_end,
            self.assim_time_step,
            'nudging_params_stretching'
            )

        self.dict_obs_rv = bfn_select_obs_temporal_window(
            dict_obs,
            self.dt_start,
            self.dt_end,
            self.assim_time_step,
            'nudging_params_relvort'
            )

    def do_projections(self):

        self.dict_proj_ssh = bfn_projections(
            'ssh',
            self.dict_obs_ssh,
            self.lon2d, self.lat2d,
            self.dist_scale,
            self.flag_plot,
            self.path_save,
            self.name_save)

        self.dict_proj_rv = bfn_projections(
            'relvort',
            self.dict_obs_rv,
            self.lon2d, self.lat2d,
            self.dist_scale,
            self.flag_plot,
            self.path_save,
            self.name_save)

    def compute_nudging_term(self, date, model_state):

        # Reshaping
        model_state = model_state.reshape(self.n_mod_var, self.ny, self.nx)

        # Get model SSH state
        ssh = model_state[0]  # 1st variable (ssh)

        # Get observations and nudging parameters
        obs_rv, nudging_coeff_rv, sigma_rv =\
            bfn_get_data_at_t(date,
                              self.dict_proj_rv)
        obs_ssh, nudging_coeff_ssh, sigma_ssh =\
            bfn_get_data_at_t(date,
                              self.dict_proj_ssh)

        # Compute nudging term
        N = {'psi1_1': np.zeros_like(ssh), 'psi1_2': np.zeros_like(ssh),
             'rv': np.zeros_like(ssh), 'ssh': np.zeros_like(ssh)}

        if obs_rv is not None and np.any(np.isfinite(obs_rv)):
            # Nudging towards relative vorticity
            rv = switchvar.ssh2rv(
                ssh, self.lon2d, self.lat2d, name_grd=self.name_grd)

            nobs = len(obs_rv)
            for iobs in range(nobs):
                indNoNan = ~np.isnan(obs_rv[iobs])
                if np.any(indNoNan):
                    # Filter model state for spectral nudging
                    rv_ls = rv.copy()
                    if sigma_rv[iobs] is not None and sigma_rv[iobs] > 0:
                        rv_ls = gaussian_filter(rv_ls, sigma=sigma_rv[iobs])

                    N['rv'][indNoNan] += nudging_coeff_rv[iobs, indNoNan]\
                      * (obs_rv[iobs, indNoNan]-rv_ls[indNoNan])

        if obs_ssh is not None and np.any(np.isfinite(obs_ssh)):
            # Nudging towards ssh
            nobs = len(obs_ssh)
            for iobs in range(nobs):
                indNoNan = ~np.isnan(obs_ssh[iobs])
                if np.any(indNoNan):
                    # Filter model state for spectral nudging
                    ssh_ls = ssh.copy()
                    if sigma_ssh[iobs] is not None and sigma_ssh[iobs] > 0:
                        ssh_ls = gaussian_filter(ssh_ls, sigma=sigma_ssh[iobs])
                    N['ssh'][indNoNan] += nudging_coeff_ssh[iobs, indNoNan]\
                      * (obs_ssh[iobs, indNoNan]-ssh_ls[indNoNan])

        N['ssh'] = N['ssh'] * self.scalenudg[0]
        N['psi1_1'] = self.g/self.f * N['ssh'] * self.scalenudg[0]
        N['psi1_2'] = self.g/self.f * N['ssh'] * self.scalenudg[1]
        N['rv'] = N['rv'] * self.scalenudg[2]

        # Mask pixels that are not influenced by observations
        N['ssh'][N['ssh'] == 0] = np.nan
        N['psi1_1'][N['psi1_1'] == 0] = np.nan
        N['psi1_2'][N['psi1_2'] == 0] = np.nan
        N['rv'][N['rv'] == 0] = np.nan

        if self.flag_plot > 3:
            plt.figure()
            plt.suptitle('Nudging coefficient')
            plt.pcolormesh(self.lon2d, self.lat2d, self.f/self.g * N['ssh'])
            plt.colorbar()
            plt.show()

        return N

    def convergence(self, path_forth, path_back):

        with xr.open_mfdataset(path_forth + '*', combine='nested',
                               concat_dim='member') as ds:
            ssh_forth = ds[self.name_mod_var[0]][:-1].values
            psi2_forth = ds[self.name_mod_var[1]][:-1].values

        with xr.open_mfdataset(path_back + '*', combine='nested',
                               concat_dim='member') as ds:
            ssh_back = ds[self.name_mod_var[0]][1:].values
            psi2_back = ds[self.name_mod_var[1]][1:].values

        psi1_forth = self.g/self.f[None, :, :] * ssh_forth
        ape = 0.5*(self.Fr[0]/self.Rom/self.dh12)**2\
              * np.sum((psi1_forth-psi2_forth)**2, axis=(1, 2))*self.dx**2
        ape = np.mean(ape)
        print("error on this loop for ssh: {:.4e}".format(
              np.mean(np.abs(ssh_forth - ssh_back))))
        print("error on this loop for psi2: {:.4e}".format( 
              np.mean(np.abs(psi2_forth - psi2_back))))
        print("mean ape on this loop: {:.4e}".format(ape))
        return ape


def bfn_select_obs_temporal_window(dict_obs, dt_start, dt_end,
                                   time_step, nudging_name):

    dict_obs_sel = {}

    date = dt_start

    while date < dt_end:
        if pd.to_datetime(date) in dict_obs:
            sat_info_list = dict_obs[date]['satellite']
            obs_file_list = dict_obs[date]['obs_name']
            # Loop on each satellite
            for sat_info, obs_file in zip(sat_info_list,obs_file_list):
                nudging_params = getattr(sat_info, nudging_name)
                if nudging_params is not None and nudging_params['K']>0:
                    # Get nudging parameters relative to stretching
                    K = nudging_params['K']
                    Tau = nudging_params['Tau']
                    sigma = nudging_params['sigma']
                    if date in dict_obs_sel:
                        if (sigma,Tau) in dict_obs_sel[date]:
                            dict_obs_sel[date][(sigma,Tau)]['sat_info'].append(sat_info)
                            dict_obs_sel[date][(sigma,Tau)]['obs_name'].append(obs_file)
                            dict_obs_sel[date][(sigma,Tau)]['K'].append(K)
                        else:
                            dict_obs_sel[date][(sigma,Tau)] = {'sat_info':[sat_info],
                                                               'obs_name':[obs_file],
                                                               'K':[K]
                                                               }
                    else:
                        dict_obs_sel[date] = {}
                        dict_obs_sel[date][(sigma,Tau)] = {'sat_info':[sat_info],
                                                           'obs_name':[obs_file],
                                                           'K':[K]
                                                           }
        date += time_step

    return dict_obs_sel




def bfn_select_observations_in_temporal_window(dict_obs, dt_start,
                                               dt_end, time_step):
    """
    NAME
        bfn_select_observations_in_temporal_window

    DESCRIPTION
        Select the observations that fall in a given BFN time window.
        Gather observations according to the nudging parameters (sigma,Tau)
        Create and fill dictionary for each nudging variable.

        Args:
            dict_obs (dictionary): dictionary listing observations
            dt_start (datetime): begining of the BFN time window
            dt_end (datetime): end of the BFN time window
            propagation_time_step (timedelta): propagation time step

        Param:

        Returns:
             obs_projected (masked array) : observations projected on the model grid
             nudging_coeff_projected (masked array) : nudging coefficient projected on the model grid
    """
    dict_obs_ssh = {}
    dict_obs_relvort = {}

    present_date0 = dt_start
    
    while present_date0 < dt_end:
        if pd.to_datetime(present_date0) in dict_obs:
            sat_info_list = dict_obs[present_date0]['satellite']
            obs_file_list = dict_obs[present_date0]['obs_name']
            # Loop on each satellite
            for sat_info, obs_file in zip(sat_info_list,obs_file_list):
                if sat_info.nudging_params_stretching is not None and\
                sat_info.nudging_params_stretching['K']>0:
                    # Get nudging parameters relative to stretching
                    K = sat_info.nudging_params_stretching['K']
                    Tau = sat_info.nudging_params_stretching['Tau']
                    sigma = sat_info.nudging_params_stretching['sigma']
                    if present_date0 in dict_obs_ssh:
                        if (sigma,Tau) in dict_obs_ssh[present_date0]:
                            dict_obs_ssh[present_date0][(sigma,Tau)]['sat_info'].append(sat_info)
                            dict_obs_ssh[present_date0][(sigma,Tau)]['obs_name'].append(obs_file)
                            dict_obs_ssh[present_date0][(sigma,Tau)]['K'].append(K)
                        else:
                            dict_obs_ssh[present_date0][(sigma,Tau)] = {'sat_info':[sat_info],'obs_name':[obs_file],'K':[K]}
                    else:
                        dict_obs_ssh[present_date0] = {}
                        dict_obs_ssh[present_date0][(sigma,Tau)] = {'sat_info':[sat_info],'obs_name':[obs_file],'K':[K]}
                if sat_info.nudging_params_relvort is not None and\
                sat_info.nudging_params_relvort['K']>0:
                    # Get nudging parameters relative to relative vorticity
                    K = sat_info.nudging_params_relvort['K']
                    Tau = sat_info.nudging_params_relvort['Tau']
                    sigma = sat_info.nudging_params_relvort['sigma']
                    if present_date0 in dict_obs_relvort:
                        if (sigma,Tau) in dict_obs_relvort[present_date0]:
                            dict_obs_relvort[present_date0][(sigma,Tau)]['sat_info'].append(sat_info)
                            dict_obs_relvort[present_date0][(sigma,Tau)]['obs_name'].append(obs_file)
                            dict_obs_relvort[present_date0][(sigma,Tau)]['K'].append(K)
                        else:
                            dict_obs_relvort[present_date0][(sigma,Tau)] = {'sat_info':[sat_info],'obs_name':[obs_file],'K':[K]}
                    else:
                        dict_obs_relvort[present_date0] = {}
                        dict_obs_relvort[present_date0][(sigma,Tau)] = {'sat_info':[sat_info],'obs_name':[obs_file],'K':[K]}

        present_date0 += time_step

    return dict_obs_ssh, dict_obs_relvort



#########################################################################
# Funcs related to the projection of observations on state grid
#########################################################################

def bfn_projections(varname, dict_obs_var, lon2d, lat2d, dist_scale,
                    flag_plot, path_save=None, name_save=None):

    """
    NAME
        bfn_projections

    DESCRIPTION
        main function that project all observations selected by the function "bfn_select_observations_in_temporal_window"

        Args:
            varnam (string): name of the nudging variable ('relvort' or 'ssh')
            dict_obs_var (dictionary): observations selected by the function "bfn_select_observations_in_temporal_window"
            lon2d (2D numpy array): longitudes of the model pixels
            lat2d (2D numpy array): latitudes of the model pixels
            n_neighbours (int): number of model pixels influenced by each observation
            dist_scale (float): scaling factor (in km) for tapering
            flag_plot (bool): for plotting projections or not
            path_save (string): path where to save the projections for future runs (default is None, meaning that the projections are not saved)
            name_save (string): file name to use for saving the projections for future runs (default is None, meaning that the projections are not saved)

        Param:

        Returns:
             dict_projections (dictionary) : observations &  associated nudging coefficients projected on the model grid
    """

    dict_projections = {}

    for date in dict_obs_var:
        dict_projections[date] = {}
        # Loop on nudging windows...
        for key in dict_obs_var[date]:
            # listes of satellites that are associated with
            # this nudging window at this date
            sat_info_list = dict_obs_var[date][key]['sat_info']
            obs_name_list = dict_obs_var[date][key]['obs_name']
            nudging_coeff_list = dict_obs_var[date][key]['K']
            if np.all(np.array(nudging_coeff_list) == 0):
                continue
            # Check if the projections have been saved in a previous run
            if path_save is not None and name_save is not None:
                file_obs_save = os.path.join(path_save,name_save + '_' + varname + '_sigma' +\
                                str(key[0]) + '_K' +\
                                '-'.join(map(str,nudging_coeff_list)) +\
                                '_window' + str(key[1]).replace(" ", "") +\
                                '_d' + str(dist_scale) + '_y' + str(date.year)\
                                + 'm' + str(date.month).zfill(2) + 'd' +\
                                str(date.day).zfill(2) + 'h' + str(date.hour).zfill(2) +\
                                str(date.minute).zfill(2) + '.pic')

                if os.path.isfile(file_obs_save):
                    # If yes, read the file
                    with open(file_obs_save, 'rb') as f:
                        obs_projected, nudging_coeff_projected = pickle.load(f)
                        f.close()

                    # Debug
                    if flag_plot > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        plt.suptitle('Nudging on ' + varname + ' at ' +
                                     str(date) + ' (Tau = ' + str(key[1]) +
                                     ' & sigma = ' + str(key[0]) + ')')
                        im1 = ax1.pcolormesh(lon2d, lat2d, obs_projected,shading='auto')
                        plt.colorbar(im1, ax=ax1)
                        ax1.set_title('Projected observations')
                        im2 = ax2.pcolormesh(lon2d, lat2d,
                                             nudging_coeff_projected,shading='auto')
                        plt.colorbar(im2, ax=ax2)
                        ax2.set_title('Projected nudging coeff')
                        plt.show()
                else:
                    # If no, compute the projections and save them
                    obs_projected, nudging_coeff_projected =\
                          bfn_merge_projections(varname, sat_info_list,
                                                obs_name_list,
                                                lon2d, lat2d,
                                                flag_plot,
                                                nudging_coeff_list,
                                                dist_scale)
                    with open(file_obs_save, 'wb') as f:
                        pickle.dump((obs_projected, nudging_coeff_projected),
                                    f)
                        f.close()

            else:
                obs_projected, nudging_coeff_projected =\
                       bfn_merge_projections(varname, sat_info_list,
                                             obs_name_list,
                                             lon2d, lat2d,
                                             flag_plot,
                                             nudging_coeff_list,
                                             dist_scale)

            # Update dictionary
            dict_projections[date][key] =\
                  {'obs':obs_projected, 'K':nudging_coeff_projected}

    return dict_projections


def bfn_merge_projections(varname, sat_info_list, obs_file_list,
                          lon2d, lat2d,
                          flag_plot=None,
                          nudging_coeff_list=None, dist_scale=None):


    if len(sat_info_list)==1 and sat_info_list[0].kind in ['fullSSH','fullRV']:
        # Full fields is provided, no need to compute tapering
        with xr.open_dataset(obs_file_list[0]) as ncin:
            lonobs = ncin[sat_info_list[0].name_obs_lon].values % 360
            latobs = ncin[sat_info_list[0].name_obs_lat].values
            varobs = ncin[sat_info_list[0].name_obs_var[0]].values
            if len(varobs.shape)==3:
                if varobs.shape[0]>1:
                    print('Warning: the full field provided has several\
                          timestep, we take the first one')
                varobs = varobs[0]
            if varname == 'relvort' and sat_info_list[0].kind=='fullSSH':
                proj_var = switchvar.ssh2rv(varobs, lonobs, latobs)
            else:
                proj_var = varobs

        if np.any(lonobs!=lon2d) or np.any(latobs!=lat2d):
            print('ERROR: When providing ' + sat_info_list[0].kind +\
' observations, grid has to be the same as the model one')
            sys.exit()
        proj_nudging_coeff = nudging_coeff_list[0] * np.ones_like(proj_var)


    else:
        # Construct KD tree for projection
        grnd_pix_tree, dist_threshold =\
               bfn_construct_ground_pixel_tree(lon2d, lat2d)
        if nudging_coeff_list is None:
            nudging_coeff_list = [1 for _ in range(len(sat_info_list))]
            dist_scale = 2*dist_threshold

        # Initialization
        lonobs, latobs, varobs, nudging_coeff = [np.array([]) for _ in range(4)]

        # Merge observations
        for iobs, (sat_info, obs_file, K) in\
          enumerate(zip(sat_info_list, obs_file_list, nudging_coeff_list)):
            # Open observation file
            with xr.open_dataset(obs_file) as ncin:
                lon = ncin[sat_info.name_obs_lon].values
                lat = ncin[sat_info.name_obs_lat].values
                var = [ncin[var_].values for var_ in sat_info.name_obs_var]
            K = K * np.ones_like(lon)
            # Merging
            lonobs = np.append(lonobs, lon.ravel())
            latobs = np.append(latobs, lat.ravel())
            nudging_coeff = np.append(nudging_coeff, K)
            # Check if we need to compute relative vorticity
            if varname == 'relvort' and sat_info.name_obs_xac is not None:
                # Only for 2D data (need 'xac' variable)
                xac = ncin[sat_info.name_obs_xac].values
                #rv = switchvar.ssh2rv(var[0], lon, lat, xac=xac)
                try:
                    rv = switchvar.ssh2rv(var[0], lon, lat, xac=xac)
                except: 
                    print('Warning: for ' + obs_file +\
                          ' impossible to convert ssh to relatve vorticity,\
                          we skip this date')
                    continue
                varobs = np.append(varobs, rv.ravel())
            elif varname == 'ssh':
                if sat_info.kind == 'CMEMS':
                    var = var[0] + var[1]  # SLA + MDT
                else:
                    var = var[0]  # SSH
                varobs = np.append(varobs, var.ravel())
            else:
                print('Warning: name of nudging variable not recongnized!!')

        # Create mask
        mask = varobs.copy()
        mask[np.isnan(mask)] = 1e19
        varobs = np.ma.masked_where(np.abs(mask) > 50, varobs)

        # Clean memory
        del var, mask, lon, lat

        # Perform projection
        proj_var, proj_nudging_coeff =\
               bfn_project_obsvar_to_state_grid(varobs, nudging_coeff,
                                                lonobs, latobs,
                                                grnd_pix_tree,
                                                dist_threshold,
                                                lon2d.shape[0],
                                                lon2d.shape[1],
                                                dist_scale)

    # Debug
    if flag_plot is not None and flag_plot > 1:

        params = {
            'font.size': 20,
            'axes.labelsize': 15,
            'axes.titlesize': 20,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 20,
            'legend.handlelength': 2,
            'lines.linewidth': 4
            }

        plt.rcParams.update(params)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7))
        if len(lonobs.shape) == 2:
            im0 = ax0.pcolormesh(lonobs, latobs, varobs,shading='auto')
        else:
            im0 = ax0.scatter(lonobs, latobs, c=varobs)
            ax0.set_xlim(lon2d.min(), lon2d.max())
            ax0.set_ylim(lat2d.min(), lat2d.max())
        cbar = plt.colorbar(im0, ax=ax0)
        cbar.ax.set_title("m")
        ax0.set_title('Available observations')
        im1 = ax1.pcolormesh(lon2d, lat2d, proj_var,shading='auto')
        cbar = plt.colorbar(im1, ax=ax1)
        cbar.ax.set_title("m")
        ax1.set_title('Projected observations')
        im2 = ax2.pcolormesh(lon2d, lat2d, proj_nudging_coeff,
                             cmap='Spectral_r',shading='auto')
        cbar = plt.colorbar(im2, ax=ax2)
        ax2.set_title('Nudging term')
        plt.show()

    return proj_var, proj_nudging_coeff





def bfn_project_obsvar_to_state_grid(var, nudging_coeff, lon, lat,
                                     ground_pixel_tree, dist_threshold,
                                     ny, nx, dist_scale):
    """
    NAME
        bfn_project_obsvar_to_state_grid

    DESCRIPTION
        Project the observations to the state grid by seeking the nearest grid point

        Args:
            var (string): 'relvort' or 'ssh'
            nudging_coeff (float): value of the nominal nudging coefficient
            lon (array): longitudes of observations
            lat (array): latitudes of observations
            ground_pixel_tree (KDtree): KDtree representing the coordinates of the model grid
            dist_threshold (float): distance (km) giving the size of a model pixel
            ny (int): meridional dimension of model grid
            nx (int): zonal dimension of model grid
            dist_scale (float): scaling factor (in km) for tapering

        Param:

        Returns:
             obs_projected (masked array) : observations projected on the model grid
             nudging_coeff_projected (masked array) : nudging coefficient projected on the model grid
    """

    ###########
    # Reshaping
    ###########
    var = var.ravel()

    ###############################################
    # Projecting the observations on the model grid
    ###############################################

    obs_projected = np.empty(ny*nx)
    obs_projected[:] = np.nan
    nudging_coeff_projected = np.empty(ny*nx)
    nudging_coeff_projected[:] = np.nan

    if var.size > 0:
        coords_obs_geo = np.column_stack((lon.ravel(), lat.ravel()))

        # Clean memory
        del lon, lat

        obs_tree = spatial.cKDTree(grid.geo2cart(coords_obs_geo))

        # Compute distances
        dist_mx = ground_pixel_tree.sparse_distance_matrix(obs_tree,
                                                           2*dist_scale)
        # Clean memory
        del obs_tree

        # Substracting distance values
        keys = np.array(list(dist_mx.keys()))

        if (keys.shape[0] > 0):
            ind_mod = keys[:, 0]
            ind_obs = keys[:, 1]
            varnan = np.isnan(var[ind_obs])
            dist = np.array(list(dist_mx.values()))[~varnan]
            dist = np.maximum(dist-0.5*dist_threshold, 0)

            # Dataframe initialized without nan values in var
            df = pd.DataFrame({'ind_mod': ind_mod[~varnan],
                               'var': var[ind_obs][~varnan],
                               'nudge': nudging_coeff[ind_obs][~varnan],
                               'dist': dist})

            # Clean memory
            del keys, varnan, ind_mod, ind_obs, dist, dist_mx

            # Remove external values in the pixels we have observations
            ind_dist = (df.dist == 0)
            df = df[np.logical_or(ind_dist,
                                  np.isin(df.ind_mod,
                                          df[ind_dist].ind_mod,
                                          invert=True))]

            # Compute weights
            df['weights'] = np.exp(-(df['dist']**2/(2*(0.5*dist_scale)**2)))

            # Nudge values out of pixels
            df.loc[df.dist > 0, "nudge"] *= df.loc[df.dist > 0, "weights"]

            # Clean memory
            del ind_dist, df['dist']

            # Compute weight average and save it
            df['weights'] = df['weights']**10
            wa = lambda x: np.average(x, weights=df.loc[x.index, "weights"])
            dfg = df.groupby('ind_mod')

            obs_pro = dfg['var'].apply(wa)
            obs_projected[np.array(obs_pro.index)] = np.array(obs_pro)

            nud_pro = dfg['nudge'].apply(wa)
            nudging_coeff_projected[np.array(nud_pro.index)] = np.array(nud_pro)

            #maxnud = np.nanmax(nudging_coeff_projected)
            #indnud = nudging_coeff_projected.copy()
            #indnud[np.isnan(indnud)] = maxnud
            #inud = indnud < maxnud
            #nudging_coeff_projected[inud] *= maxnud/np.max(indnud[inud])
            # Clean memory
            del obs_pro, nud_pro, wa, df, dfg

    # Normalize between 0 and the max value:
    max_ = np.max(nudging_coeff)
    min_ = 0
    nudging_coeff_projected = max_*(nudging_coeff_projected-min_)/(max_-min_)

    # Mask the useless pixels
    mask = obs_projected.copy()
    mask[np.isnan(mask)] = 1e19
    obs_projected =\
      np.ma.masked_where(np.abs(mask) > 50,
                         obs_projected).reshape(ny, nx)
    mask = nudging_coeff_projected.copy()
    mask[np.isnan(mask)] = 1e19
    nudging_coeff_projected =\
      np.ma.masked_where(np.abs(mask) > 50,
                         nudging_coeff_projected).reshape(ny, nx)

    return obs_projected, nudging_coeff_projected


def bfn_construct_ground_pixel_tree(lon, lat):
    coords = np.column_stack((lon.ravel(), lat.ravel()))
    # construct KD-tree
    ground_pixel_tree = spatial.cKDTree(grid.geo2cart(coords))
    subdomain = grid.geo2cart(coords)[0:100]
    eucl_dist = cdist(subdomain, subdomain, metric="euclidean")
    dist_threshold = np.min(eucl_dist[np.nonzero(eucl_dist)])

    return ground_pixel_tree, dist_threshold


#########################################################################
# Funcs related to the smoothing coefficients computation for Nudging/BFN
#########################################################################


def bfn_gather_smoothing_coeff_from_date(date_obs_list, date_obs_list_var,
                                         nudging_halfwindow, one_time_step,
                                         middle_date, window_size,
                                         nudging_smooth_function, flag_plot):
    smoothing_coeff = []
    timestamps = np.arange(-(window_size/2).total_seconds(),
                           (window_size/2).total_seconds()
                           + one_time_step.total_seconds(),
                           one_time_step.total_seconds())
    if flag_plot:
        plt.figure(figsize=(10, 5))
    for i, date_ in enumerate(date_obs_list):
        if date_ in date_obs_list_var:
            timestamps_ = (middle_date-date_).total_seconds()+timestamps
            smoothing_coeff_ =\
                  bfn_nudge_smoothing(timestamps_,
                                      nudging_smooth_function,
                                      nudging_halfwindow.total_seconds())
        else:
            smoothing_coeff_ = 0 * timestamps
        smoothing_coeff.append(smoothing_coeff_)
        if flag_plot:
            plt.plot(smoothing_coeff_, label=str(date_))
    if flag_plot:
        plt.xlabel('Dates')
        plt.ylabel('Smoothing coefficient')
        plt.legend()
        plt.show()
    # Converting to numpy array
    smoothing_coeff = np.asarray(smoothing_coeff)

    return smoothing_coeff


def bfn_nudge_smoothing(timestamps, smooth_function, halfwindow):
    if smooth_function == 'hat':
        return hat_function(timestamps, halfwindow)
    if smooth_function == 'gaspari-cohn':
        return gaspari_cohn(timestamps, halfwindow)

#########################################################################
# Func to group all the data needed to perform one Nudging/BFN forecast
#########################################################################


def bfn_get_data_at_t(date_t, dict_projections):
    obs_t = []
    nudging_coeff_t = []
    sigma_t = []
    for date in dict_projections:
        for sigma, Tau in dict_projections[date]:
            if date-Tau < date_t < date+Tau:
                # Compute smoothing coeff at this time considering
                # the date of observation and the nudging window
                smoothing_t = gaspari_cohn(abs((date_t-date).total_seconds()),
                                           Tau.total_seconds())
                # Append to list
                obs_t.append(dict_projections[date][sigma, Tau]['obs'])
                nudging_coeff_t.append(smoothing_t *
                                       dict_projections[date][sigma, Tau]['K'])
                sigma_t.append(sigma)
    # Convert to numpy array
    obs_t = np.asarray(obs_t)
    nudging_coeff_t = np.asarray(nudging_coeff_t)
    sigma_t = np.asarray(sigma_t)

    return obs_t, nudging_coeff_t, sigma_t
