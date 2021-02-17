#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:26:07 2021

@author: leguillou
"""

import os, sys

def exp(path_config):
    
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
    config_exp = __import__(_config)
    # Merge with default config file
    from . import config_default as config
    config.__dict__.update(config_exp.__dict__)
    # Clean temporary direcectory
    if os.path.exists(config.tmp_DA_path):
        cmd = 'rm ' + config.tmp_DA_path + '*'
        os.system(cmd)
    else:
        os.makedirs(config.tmp_DA_path)
    cmd = 'cp ' + path_config + '.py ' + config.tmp_DA_path + '/config.py'
    print(cmd)
    os.system(cmd)
    
    return config