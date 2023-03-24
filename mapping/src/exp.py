#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:26:07 2021

@author: leguillou
"""



import os, sys
from copy import deepcopy


class Config(dict):

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
        
    def __str__(self):
        res = []
        for key,val in self.items():
            res.append(f"{key}: {val}")
        res.append('')
        return '\n'.join(res)

    def copy(self):
        other = {}
        for key,val in self.items():
            if type(val)==dict:
                other[key] = deepcopy(val)
            else:
                other[key] = val
        return Config(other)


def _merge_configs(config_exp,config_def,NAME):

    dict_exp = getattr(config_exp,NAME)
    dict_def = getattr(config_def,dict_exp['super'])

    _config = {}
    _config['super'] = dict_exp['super']
    for key in dict_def:
        if key in dict_exp:
            # Check if we have encapsulated super class 
            if type(dict_exp[key])==dict and 'super' in dict_exp[key]:
                # If yes, we merge with default config
                _dict_def = getattr(config_def, dict_exp[key]['super'])
                for _key in _dict_def:
                    if _key not in dict_exp[key]:
                        dict_exp[key][_key] = _dict_def[_key]
            _config[key] = dict_exp[key]
        else:
            _config[key] = dict_def[key]

    return Config(_config)

def merge_configs(config_exp,config_def,name):

    try:
        NAME = getattr(config_exp,name)

        if type(NAME)==list:
            config = {}
            for _NAME in NAME:
                config[_NAME] = _merge_configs(config_exp,config_def,_NAME)
            config = Config(config)
        else:
            config = _merge_configs(config_exp,config_def,NAME)
        
        return config

    except:
        print(f"{name} is not set in the configuration file")
        return None


def Exp(path_config):
    
    """
        NAME
            exp
    
        DESCRIPTION
            Read experimental parameters
            Args:
                path_config (str): path (dir+name) of the configuration file.`
            Returns:
                configuration module
    """
    
    _dir,_config = os.path.split(path_config)
    sys.path.append(_dir)
    if _config[-3:]=='.py':
        _config = _config[:-3]
    else:
        path_config += '.py'
    config_exp = __import__(_config)
    
    # Merge with default config file
    from . import config_default as config_def
    config = Config({})

    # EXP
    config.EXP = {}
    for key,val in config_exp.EXP.items():
        config.EXP[key] = val
    for key,val in config_def.EXP.items():
        if key not in config.EXP:
            config.EXP[key] = val
    config.EXP = Config(config.EXP)
    print(config.EXP)

    config.GRID = merge_configs(config_exp,config_def,'NAME_GRID')
    config.OBS = merge_configs(config_exp,config_def,'NAME_OBS')
    config.MOD = merge_configs(config_exp,config_def,'NAME_MOD')
    config.BC = merge_configs(config_exp,config_def,'NAME_BC')
    config.OBSOP = merge_configs(config_exp,config_def,'NAME_OBSOP')
    config.BASIS = merge_configs(config_exp,config_def,'NAME_BASIS')
    config.INV = merge_configs(config_exp,config_def,'NAME_INV')
    config.DIAG = merge_configs(config_exp,config_def,'NAME_DIAG')

    # temporary directory
    if not os.path.exists(config.EXP.tmp_DA_path):
        os.makedirs(config.EXP.tmp_DA_path)
    cmd = f"cp {path_config} {config.EXP.tmp_DA_path}/config.py"
    os.system(cmd)

    # outptut directory
    if not os.path.exists(config.EXP.path_save):
        os.makedirs(config.EXP.path_save)
    cmd = f"cp {path_config} {config.EXP.path_save}/config.py"
    os.system(cmd)

    return config