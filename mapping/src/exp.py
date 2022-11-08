#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:26:07 2021

@author: leguillou
"""



import os, sys


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
    from . import config_default as config_def
    config = {}
    
    for p in dir(config_exp):
        config[p] = getattr(config_exp, p)
        
    for p in dir(config_def):
        # you can write your filter here
        if p not in dir(config_exp):
            config[p] = getattr(config_def, p)
    
    config = dotdict(config)
    
    # temp directory
    if not os.path.exists(config.tmp_DA_path):
        os.makedirs(config.tmp_DA_path)
    cmd = f'cp {path_config} {config.tmp_DA_path}/config.py'
    os.system(cmd)

    # outptut directory
    if not os.path.exists(config.path_save):
        os.makedirs(config.path_save)
    cmd = f'cp {path_config} {config.path_save}/config.py'
    os.system(cmd)

    return config