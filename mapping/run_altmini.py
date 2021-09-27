#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:02:31 2021

@author: leguillou
"""

import os
import argparse
import numpy as np
import subprocess
from datetime import datetime
import re

def create_new_config_file(src_file,out_file,list_pattern,list_subst):
    line_added = []
    with open(out_file, 'w') as out:
        with open(src_file, 'r') as src:
            lines = src.readlines()
            for line in lines:
                found = False
                for pattern,subst in zip(list_pattern,list_subst):
                    if re.search(pattern, line) and line[:len(pattern)]==pattern\
                        and line[len(pattern)] in [' ','=']:
                        line_added.append(pattern)
                        new_line = subst + '\n'
                        out.write(new_line)
                        found = True
                if not found:
                    out.write(line)
        for pattern,subst in zip(list_pattern,list_subst):
            if pattern not in line_added:
                new_line = subst + '\n'
                out.write(new_line)

if __name__ == "__main__":
    
    pwd = os.path.dirname(os.path.abspath(__file__))
    
    # Parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--p', 
                        default='Exp_joint', 
                        type=str) # Path of experiment (useful to restart an existing experiment)
    parser.add_argument('--c1', # Path of 1st config file
                        default=os.path.join('examples','config_Example2_BM.py'),
                        type=str)   
    parser.add_argument('--c2', # Path of 2nd config file
                        default=os.path.join('examples','config_Example2_IT.py'),
                        type=str)   
    parser.add_argument('--params1', type=str, default=None) # names of iterable parameters for 1st experiment
    parser.add_argument('--params2', type=str, default=None) # names of iterable parameters for 2nd experiment
    parser.add_argument('--i0', default=0, type=int) # First iteration number (useful if the experiment is restarted)
    parser.add_argument('--imax', default=None, type=int) # Maximum number of iterations
    parser.add_argument('--Kmin', default=1e-3, type=float) # Value of K below which the iterations are stopped
    opts = parser.parse_args()
    print("* Parsing:")
    i0 = opts.i0
    imax = opts.imax
    if imax is None:
        imax = np.inf
    path_exp = opts.p 
    if i0==0:
        path_exp += '_'+datetime.now().strftime('%Y-%m-%d_%H%M')
    exp_config_file_1 = opts.c1
    exp_config_file_2 = opts.c2
    params1 = opts.params1
    params2 = opts.params2
    Kmin = opts.Kmin
    print('path_exp:',path_exp)
    print('config1:',exp_config_file_1)
    print('config2:',exp_config_file_2)
    print('iterable parameters for config1:',params1)
    print('iterable parameters for congig2:',params2)
    print('Startint at iteration n°',i0)
    print('Stopping algorithm when K <',Kmin)
    
    
    # Convergence file
    path_K = os.path.join(path_exp,'K.txt')
    print('Convergence trajectory is written in:',path_K)
    
    # Create experimental directory
    if not os.path.exists(path_exp):
        print('creating',path_exp)
        os.makedirs(path_exp)
        
    # Create new config files
    path_exp_config_file_1 = os.path.join(path_exp,'config1.py')
    path_exp_config_file_2 = os.path.join(path_exp,'config2.py')
    
    create_new_config_file(exp_config_file_1,
                path_exp_config_file_1,
                ['tmp_DA_path ','path_save ','path_obs '],
                ['tmp_DA_path = "' + os.path.join(path_exp,'scratch/Exp1/iteration_0"'),
                 'path_save = "' + os.path.join(path_exp,'outputs/Exp1/iteration_0"'),
                 'path_obs = "' + os.path.join(path_exp,'obs/Exp1"')]
                )
    
    create_new_config_file(exp_config_file_2,
                path_exp_config_file_2,
                ['tmp_DA_path ','path_save ','path_obs '],
                ['tmp_DA_path = "' + os.path.join(path_exp,'scratch/Exp2/iteration_0"'),
                 'path_save = "' + os.path.join(path_exp,'outputs/Exp2/iteration_0"'),
                 'path_obs = "' + os.path.join(path_exp,'obs/Exp2"')]
                )
    
    K = np.inf
    i = i0
        
    while K>Kmin and i<=imax:
        
        time0 = datetime.now()
        print('\n*** Iteration n°'+str(i) + ' ***')
        
        # Run iteration
        print('1. Run Mapping experiments')
        run_iteration = os.path.join(pwd,'run_iteration.py')
        cmd = ['python3', run_iteration, 
               '--c1',path_exp_config_file_1,
               '--c2',path_exp_config_file_2,
               '--i',str(i),
               '--K',path_K]
        if params1 is not None:
            cmd.append('--params1')
            cmd.append(params1)
        if params2 is not None:
            cmd.append('--params2')
            cmd.append(params2)
            
        
        print(' '.join(cmd))
        
        out = open(os.path.join(path_exp,'logout_' + str(i) + '.txt'), "w")
        err = open(os.path.join(path_exp,'logerr_' + str(i) + '.txt'), "w")
        subprocess.call(cmd,stdout=out,stderr=err)
        
        # Convergence criteria 
        if i>0:
            print('2. Convergence criteria:')
            with open(path_K,'r') as f:
                last_line = f.readlines()[i-1]
                K1,K2,K = last_line.split(' ')
                print('\tK1',K1)
                print('\tK2',K2)
                print('\tK',K)
                K = float(K)
        i += 1
        time1 = datetime.now()
        print('computation time:',time1-time0)
        
        
    
