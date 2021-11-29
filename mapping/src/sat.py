#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:17:32 2021

@author: leguillou
"""

class Satellite:

    def __init__(
        self, satellite, kind, path, name,
        name_obs_var, name_obs_lon, name_obs_lat, name_obs_time, name_obs_xac,
        name_obs_grd,
        nudging_params_stretching, nudging_params_relvort,
        sigma_noise
    ):
        self.satellite = satellite
        self.kind = kind
        self.path = path
        self.name = name
        self.name_obs_var = name_obs_var
        self.name_obs_lon = name_obs_lon
        self.name_obs_lat = name_obs_lat
        self.name_obs_time = name_obs_time
        self.name_obs_xac = name_obs_xac
        self.name_obs_grd = name_obs_grd
        self.nudging_params_stretching = nudging_params_stretching
        self.nudging_params_relvort = nudging_params_relvort
        self.sigma_noise = sigma_noise


    def __str__(self):
        return " satellite : " + str(self.satellite) + " \n kind : "+ str(self.kind) + " \n path : " + str(self.path) + " \n name : " + str(self.name)



def read_satellite_info(config,sat):
    
    kind = getattr(config, 'kind_' + sat)
    path = getattr(config, 'obs_path_' + sat)
    name = getattr(config, 'obs_name_' + sat)

    # Names of variables
    name_obs_var = getattr(config, 'name_obs_var_' + sat)
    name_obs_lon = getattr(config, 'name_obs_lon_' + sat)
    name_obs_lat = getattr(config, 'name_obs_lat_' + sat)
    name_obs_time = getattr(config, 'name_obs_time_' + sat)
    
    try: sigma_noise = getattr(config, 'sigma_noise_' + sat)
    except: sigma_noise = None
    
    name_obs_grd = [name_obs_lon,name_obs_lat,name_obs_time]
    
    # For BFN
    if (config.name_analysis == 'BFN'):
        nudging_params_stretching = getattr(config, 'nudging_params_stretching_' + sat)
        nudging_params_relvort = getattr(config, 'nudging_params_relvort_' + sat)
        if nudging_params_stretching is not None:
            if nudging_params_stretching['K'] is None:
                nudging_params_stretching['K'] = config.Knudg
            if nudging_params_stretching['Tau'] is None:
                nudging_params_stretching['Tau'] = \
                    (7/2)*config.bfn_propation_timestep/nudging_params_stretching['K']
        if nudging_params_relvort is not None:
            if nudging_params_relvort['K'] is None:
                nudging_params_relvort['K'] = config.Knudg
            if nudging_params_relvort['Tau'] is None:
                nudging_params_relvort['Tau'] = \
                    (7/2)*config.bfn_propation_timestep/nudging_params_relvort['K']
                
        if (kind == 'swot_simulator'):
            name_obs_xac = getattr(config, 'name_obs_xac_' + sat)
            if name_obs_xac is not None:
                name_obs_grd += [name_obs_xac]
        else:
            name_obs_xac = None
    else:
        name_obs_xac = None
        nudging_params_stretching = nudging_params_relvort = None

    return Satellite(
        satellite=sat, kind=kind, path=path, name=name,
        name_obs_var=name_obs_var, name_obs_lon=name_obs_lon, name_obs_lat=name_obs_lat, name_obs_time=name_obs_time,
        name_obs_xac=name_obs_xac, name_obs_grd=name_obs_grd, 
        nudging_params_relvort=nudging_params_relvort, nudging_params_stretching=nudging_params_stretching,
        sigma_noise=sigma_noise)
