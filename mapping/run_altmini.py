#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:02:31 2021

@author: leguillou
"""

import sys,os
import numpy as np
import subprocess
from datetime import datetime
import re

K_MIN = 1e-3

def create_new_config_file(src_file,out_file,list_pattern,list_subst):
    with open(out_file, 'w') as out:
        with open(src_file, 'r') as src:
            lines = src.readlines()
            for line in lines:
                found = False
                for pattern,subst in zip(list_pattern,list_subst):
                    if re.search(pattern, line) and line[:len(pattern)]==pattern:
                        new_line = subst + '\n'
                        out.write(new_line)
                        found = True
                if not found:
                    out.write(line)

if __name__ == "__main__":
    
    
    # check number of arguments
    if  len(sys.argv)!=4:
        sys.exit('Wrong number of argument')
    # Experiment config file
    print("* Experimental configuration files")
    path_exp = sys.argv[1]
    exp_config_file_1 = sys.argv[2]
    exp_config_file_2 = sys.argv[3]
    print('path_exp:',path_exp)
    print('config1:',exp_config_file_1)
    print('config2:',exp_config_file_2)
    
    # Convergence file
    print('Convergence trajectory is written in:')
    path_K = path_exp+'/K.txt'
    
    # Create experimental directory
    if not os.path.exists(path_exp):
        print('creating',path_exp)
        os.makedirs(path_exp)
        
    # Create new config files
    path_exp_config_file_1 = path_exp + '/config1.py'
    path_exp_config_file_2 = path_exp + '/config2.py'
    create_new_config_file(exp_config_file_1,
                path_exp_config_file_1,
                ['tmp_DA_path','path_save'],
                ['tmp_DA_path = "' + path_exp + '/scratch/Exp1/iteration_0/"',
                 'path_save = "' + path_exp + '/outputs/Exp1/iteration_0/"']
                )
    create_new_config_file(exp_config_file_2,
                path_exp_config_file_2,
                ['tmp_DA_path','path_save'],
                ['tmp_DA_path = "' + path_exp + '/scratch/Exp2/iteration_0/"',
                 'path_save = "' + path_exp + '/outputs/Exp2/iteration_0/"']
                )
    
    K = np.inf
    i0 = 0
    i = i0
    while K>K_MIN:
        
        time0 = datetime.now()
        print('\n*** Iteration n°'+str(i) + ' ***')
        # Run iteration
        print('1. Run Mapping experiments')
        cmd = ['python3', os.path.dirname(os.path.abspath(__file__)) + '/run_iteration.py', 
               path_exp_config_file_1,path_exp_config_file_2,str(i),path_K]
        out = open(path_exp + '/logout_' + str(i) + '.txt', "w")
        err = open(path_exp + '/logerr_' + str(i) + '.txt', "w")
        subprocess.call(cmd,stdout=out,stderr=err)
        # Convergence criteria 
        if i>0:
            print('2. Convergence criteria:')
            with open(path_K,'r') as f:
                last_line = f.readlines()[-1]
                K1,K2,K = last_line.split(' ')
                print('\tK1',K1)
                print('\tK2',K2)
                print('\tK',K)
                K = float(K)
        i += 1
        time1 = datetime.now()
        print('computation time:',time1-time0)
        
        
    