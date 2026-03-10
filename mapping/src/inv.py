#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""
from .config import USE_FLOAT64
import sys, os 
import numpy as np 
import os
import matplotlib.pylab as plt
from datetime import datetime,timedelta
import scipy.optimize as opt
import gc
import xarray as xr

import glob
from importlib.machinery import SourceFileLoader 

import jax 
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", USE_FLOAT64)

from . import grid



def Inv(config, State=None, Model=None, dict_obs=None, Obsop=None, Basis=None, Bc=None, *args, **kwargs):

    """
    NAME
        Inv

    DESCRIPTION
        Main function calling subfunctions for specific Inversion algorithms
    """
    
    if config.INV is None:
        return Inv_forward(config, State=State, Model=Model, Bc=Bc)
    
    print(config.INV)
    
    if config.INV.super=='INV_OI':
        return Inv_oi(config, State=State, dict_obs=dict_obs)
    
    elif config.INV.super=='INV_BFN':
        return Inv_bfn(config, State=State, Model=Model, dict_obs=dict_obs, Bc=Bc)
    
    elif config.INV.super=='INV_4DVAR':
        return Inv_4Dvar(config, State=State, Model=Model, dict_obs=dict_obs, Obsop=Obsop, Basis=Basis, Bc=Bc)
    
    else:
        sys.exit(config.INV.super + ' not implemented yet')
        
def Inv_forward(config,State,Model,Bc=None):
    
    """
    NAME
        Inv_forward

    DESCRIPTION
        Run a model forward integration  
    
    """

    if 'JAX' in config.MOD.super:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    present_date = config.EXP.init_date
    if config.EXP.saveoutputs:
        State.save_output(present_date,name_var=Model.var_to_save)
        
    nstep = int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt)

    if Bc is not None:
        time_bc = np.array([np.datetime64(time) for time in Model.timestamps[::nstep]])
        t_bc = np.array([t for t in Model.T[::nstep]])
        var_bc = Bc.interp(time_bc)
        Model.set_bc(t_bc,var_bc)

    t = 0
    Model.init(State,t)
    Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)
    State.plot(title='Start of forward integration')

    while present_date + timedelta(seconds=nstep*Model.dt) <= config.EXP.final_date :
        
        # Propagation
        Model.step(State,nstep,t=t)

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

        # Save
        if config.EXP.saveoutputs:
            Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)    
        State.plot(present_date)
    
    State.plot(title='End of forward integration')
        
    return
       
def Inv_oi(config,State,dict_obs):
    
    """
    NAME
        Inv_oi

    DESCRIPTION
        Run an optimal interpolation experiment 
    
    """
    
    from . import obs
    
    # Initialize variables (normally this step is done in mod.py, but here no Model object is provided)
    for name in config.INV.name_var:
        State.var[config.INV.name_var[name]] = np.zeros((State.ny,State.nx))

    # Boundary box
    box = [State.lon.min(),State.lon.max(),State.lat.min(),State.lat.max(),
           None, None]
    
    # Time parameters
    ndays = (config.EXP.final_date-config.EXP.init_date).total_seconds()/3600/24
    dt = config.EXP.saveoutput_time_step.total_seconds()/3600/24
    times = np.arange(0, ndays + dt, dt)

    # Coordinates
    lon1d = State.lon.flatten()
    lat1d = State.lat.flatten()
    
    # Time loop
    for t in times:

        for name in config.INV.name_var:
            
            # Time boundary
            box[4] = config.EXP.init_date + timedelta(days=t-config.INV.Lt)
            box[5] = config.EXP.init_date + timedelta(days=t+config.INV.Lt)
            
            # Get obs in (time x lon x lat) cube
            obs_val, obs_coords, _ = obs.get_obs(dict_obs, box, config.EXP.init_date, name)
            obs_lon = obs_coords[0]
            obs_lat = obs_coords[1]
            obs_time = obs_coords[2]
            
            # Perform the optimal interpolation 
            BHt = np.exp(-((t - obs_time[np.newaxis,:])/config.INV.Lt)**2 - 
                        ((lon1d[:,np.newaxis] - obs_lon[np.newaxis,:])/config.INV.Lx)**2 - 
                        ((lat1d[:,np.newaxis] - obs_lat[np.newaxis,:])/config.INV.Ly)**2)
            HBHt = np.exp(-((obs_time[np.newaxis,:] - obs_time[:,np.newaxis])/config.INV.Lt)**2 -
                        ((obs_lon[np.newaxis,:] - obs_lon[:,np.newaxis])/config.INV.Lx)**2 -
                        ((obs_lat[np.newaxis,:] - obs_lat[:,np.newaxis])/config.INV.Ly)**2) 
            R = np.diag(np.full((len(obs_val)), config.INV.sigma_R**2))
            Coo = HBHt + R
            Mi = np.linalg.inv(Coo)
            sol = np.dot(np.dot(BHt, Mi), obs_val).reshape((State.ny,State.nx))
            
            # Set estimated variable
            State.setvar(sol,name_var=config.INV.name_var[name])

        # Save estimated fields for date t
        date = config.EXP.init_date + timedelta(days=t)
        State.save_output(date)

    return 

def Inv_bfn(config,State,Model,dict_obs=None,Bc=None,*args, **kwargs):
    """
    NAME
        Inv_bfn

    DESCRIPTION
        Run a Back and Forth Nudging experiment 
    
    """
    
    from . import tools_bfn as bfn

    # Flag initialization
    if config.GRID.super=='GRID_RESTART':
        restart = True
        bfn_first_window = False
    else:
        restart = False
        bfn_first_window = True
    bfn_last_window = False
    if dict_obs is None:
        call_obs_func = True
        from . import obs
    else:
        call_obs_func = False
    # BFN middle date initialization
    if State.present_date is not None:
        middle_bfn_date = State.present_date
    else:
        middle_bfn_date = config.EXP.init_date
    # In the case of Nudging (i.e. bfn_max_iteration=1), set the bfn window length as the entire experimental time period
    if config.bfn_max_iteration==1:
        new_bfn_window_size = config.EXP.final_date - config.EXP.init_date
    else:
        new_bfn_window_size = config.INV.window_size

    # propagation timestep
    one_time_step = config.INV.propagation_timestep
        
    # Main time loop
    while (middle_bfn_date <= config.EXP.final_date) and not bfn_last_window :
        #############
        # 1. SET-UP #
        #############
        # BFN period
        init_bfn_date = max(config.EXP.init_date, middle_bfn_date - new_bfn_window_size/2)
        init_bfn_date += timedelta(seconds=(init_bfn_date - config.EXP.init_date).total_seconds()\
                         / one_time_step.total_seconds()%1)
        middle_bfn_date = max(middle_bfn_date, config.EXP.init_date + new_bfn_window_size/2)
        if ((middle_bfn_date + new_bfn_window_size/2) >= config.EXP.final_date):
            bfn_last_window = True
            final_bfn_date = config.EXP.final_date
        else:
            final_bfn_date = init_bfn_date + new_bfn_window_size
            
        if bfn_first_window or restart:
            present_date_forward0 = init_bfn_date
            
        ########################
        # 2. Create BFN object #
        ########################
        bfn_obj = bfn.bfn(
            config,init_bfn_date,final_bfn_date,one_time_step,State)

        ##########################
        # 3. Boundary conditions #
        ##########################
        if Bc is not None:
            time0 = np.datetime64(init_bfn_date)
            tsec0 = (init_bfn_date - config.EXP.init_date).total_seconds()
            time_bc = []
            tsec_bc = []
            while time0<=np.datetime64(final_bfn_date):
                time_bc.append(time0)
                tsec_bc.append(tsec0)
                time0 += np.timedelta64(one_time_step)
                tsec0 += one_time_step.total_seconds()
                time_bc.append(time0)
                tsec_bc.append(tsec0)
            var_bc = Bc.interp(time_bc)
            Model.set_bc(tsec_bc,var_bc)
        
        # Initial model state
        if bfn_first_window:
            Model.init(State)
            State.plot(title='Init State')
            

        ###################
        # 4. Observations #
        ###################
        # Selection        
        if call_obs_func:
            dict_obs_it = obs.obs(config)
            bfn_obj.select_obs(dict_obs_it)
            dict_obs_it.clear()
            del dict_obs_it
        else:
            bfn_obj.select_obs(dict_obs)

        # Projection
        bfn_obj.do_projections()

        ###############
        # 5. BFN LOOP #
        ###############
        err_bfn0 = 0
        err_bfn1 = 0
        bfn_iter = 0
        Nold_t = None
        
        time0 = datetime.now()
        while bfn_iter==0 or\
              (bfn_iter < config.INV.max_iteration
              and abs(err_bfn0-err_bfn1)/err_bfn1 > config.INV.criterion):

            if bfn_iter>0:
                present_date_forward0 = init_bfn_date

            err_bfn0 = err_bfn1
            bfn_iter += 1
            
            ###################
            # 5.1. FORTH LOOP #
            ###################

            # Save state at first timestep              
            filename_forward = os.path.join(config.EXP.tmp_DA_path,'BFN_forward'\
                + '_y' + str(present_date_forward0.year)\
                + 'm' + str(present_date_forward0.month).zfill(2)\
                + 'd' + str(present_date_forward0.day).zfill(2)\
                + 'h' + str(present_date_forward0.hour).zfill(2)\
                + str(present_date_forward0.minute).zfill(2) + '.nc')
            State.save(filename_forward)
            
            while present_date_forward0 < final_bfn_date :
                
                # Time
                t = (present_date_forward0 - config.EXP.init_date).total_seconds()

                # Model propagation and apply Nudging
                Model.step_nudging(State,
                       one_time_step.total_seconds(),
                       Nudging_term=Nold_t,
                       t=t)

                # Time increment 
                present_date_forward = present_date_forward0 + one_time_step
                
                # Compute Nudging term (for next time step)
                N_t = bfn_obj.compute_nudging_term(
                        present_date_forward, State
                        )

                # Save current state     
                name_save = 'BFN_forward'\
                    + '_y' + str(present_date_forward.year)\
                    + 'm' + str(present_date_forward.month).zfill(2)\
                    + 'd' + str(present_date_forward.day).zfill(2)\
                    + 'h' + str(present_date_forward.hour).zfill(2)\
                    + str(present_date_forward.minute).zfill(2) + '.nc'
                filename_forward = os.path.join(config.EXP.tmp_DA_path,name_save)
                State.save(filename_forward)
                if config.INV.save_trajectory:
                    filename_traj = os.path.join(config.EXP.path_save,'BFN_' + str(middle_bfn_date)[:10]\
                               + '_forward_' + str(bfn_iter),name_save)

                    if not os.path.exists(os.path.dirname(filename_traj)):
                        os.makedirs(os.path.dirname(filename_traj))
                    State.save(filename_traj)
                              
                # Time update
                present_date_forward0 = present_date_forward
                Nold_t = N_t
    
            
            # Plot for debugging
            if config.EXP.flag_plot > 0:
                State.plot(title=str(present_date_forward) + ': End of forward loop n°' + str(bfn_iter))

            ##################
            # 5.2. BACK LOOP #
            ##################
            if  bfn_iter < config.INV.max_iteration:
                present_date_backward0 = final_bfn_date
                # Save state at first timestep          
                filename_backward = os.path.join(config.EXP.tmp_DA_path,'BFN_backward'\
                    + '_y' + str(present_date_backward0.year)\
                    + 'm' + str(present_date_backward0.month).zfill(2)\
                    + 'd' + str(present_date_backward0.day).zfill(2)\
                    + 'h' + str(present_date_backward0.hour).zfill(2)\
                    + str(present_date_backward0.minute).zfill(2) + '.nc')
                State.save(filename_backward)
                
                while present_date_backward0 > init_bfn_date :
                    
                    # Time
                    t = (present_date_backward0 - config.EXP.init_date).total_seconds()

                    # Propagate the state by nudging the model vorticity towards the 2D observations
                    Model.step_nudging(State,
                       -one_time_step.total_seconds(),
                       Nudging_term=Nold_t,
                       t=t)
                    
                    # Time increment
                    present_date_backward = present_date_backward0 - one_time_step

                    # Nudging term (next time step)
                    N_t = bfn_obj.compute_nudging_term(
                            present_date_backward,
                            State
                            )
                    
                    # Save current state   
                    name_save = 'BFN_backward'\
                    + '_y' + str(present_date_backward.year)\
                    + 'm' + str(present_date_backward.month).zfill(2)\
                    + 'd' + str(present_date_backward.day).zfill(2)\
                    + 'h' + str(present_date_backward.hour).zfill(2)\
                    + str(present_date_backward.minute).zfill(2) + '.nc'         
                    filename_backward = os.path.join(config.EXP.tmp_DA_path,name_save)
                    State.save(filename_backward)
                    if config.INV.save_trajectory:
                        filename_traj = os.path.join(config.EXP.path_save,'BFN_' + str(middle_bfn_date)[:10]\
                                   + '_backward_' + str(bfn_iter),name_save)

                        if not os.path.exists(os.path.dirname(filename_traj)):
                            os.makedirs(os.path.dirname(filename_traj))
                        State.save(filename_traj)

                    # Time update
                    present_date_backward0 = present_date_backward
                    Nold_t = N_t

                if config.EXP.flag_plot > 0:
                    State.plot(title=str(present_date_backward) + ': End of backward loop n°' + str(bfn_iter))

            #########################
            # 5.3. CONVERGENCE TEST #
            #########################
            if bfn_iter < config.INV.max_iteration:
                err_bfn1 = bfn_obj.convergence(
                                        path_forth=os.path.join(config.EXP.tmp_DA_path,'BFN_forward_*.nc'),
                                        path_back=os.path.join(config.EXP.tmp_DA_path,'BFN_backward_*.nc')
                                        )
            
        time1 = datetime.now()
                
        print('Loop from',init_bfn_date.strftime("%Y-%m-%d"),'to',final_bfn_date.strftime("%Y-%m-%d :"),bfn_iter,'iterations in',time1-time0,'seconds')
        
        #####################
        # 6. SAVING OUTPUTS #
        #####################
        # Set the saving temporal windowx
        if config.INV.max_iteration==1:
            write_date_min = init_bfn_date
            write_date_max = final_bfn_date
        elif bfn_first_window:
            write_date_min = init_bfn_date
            write_date_max = init_bfn_date + new_bfn_window_size/2 + config.INV.window_output/2
        elif bfn_last_window:
            write_date_min = middle_bfn_date - config.INV.window_output/2
            write_date_max = final_bfn_date
        else:
            write_date_min = middle_bfn_date - config.INV.window_output/2
            write_date_max = middle_bfn_date + config.INV.window_output/2

        # Write outputs in the saving temporal window
        present_date = init_bfn_date
        # Save first timestep
        if present_date==config.EXP.init_date:

            current_file = os.path.join(config.EXP.tmp_DA_path,'BFN_forward'\
                + '_y' + str(present_date.year)\
                + 'm' + str(present_date.month).zfill(2)\
                + 'd' + str(present_date.day).zfill(2)\
                + 'h' + str(present_date.hour).zfill(2)\
                + str(present_date.minute).zfill(2) + '.nc')
            State.load(current_file)

            if config.EXP.saveoutputs:
                State.save_output(present_date,name_var=Model.var_to_save)

        while present_date < final_bfn_date :
            present_date += one_time_step
            if (present_date > write_date_min) & (present_date <= write_date_max) :
                # Save output every *saveoutput_time_step*
                if (((present_date - config.EXP.init_date).total_seconds()
                   /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                   & (present_date>config.EXP.init_date)\
                   & (present_date<=config.EXP.final_date) :
                    # Read current converged state
                    current_file = os.path.join(config.EXP.tmp_DA_path,'BFN_forward'\
                        + '_y' + str(present_date.year)\
                        + 'm' + str(present_date.month).zfill(2)\
                        + 'd' + str(present_date.day).zfill(2)\
                        + 'h' + str(present_date.hour).zfill(2)\
                        + str(present_date.minute).zfill(2) + '.nc')
                    State.load(current_file)
                    
                    # Smooth with previous BFN window
                    if config.INV.window_overlap and not bfn_first_window and\
                        present_date<=middle_bfn_date:
                        # weight coefficients
                        W1 = max((middle_bfn_date - present_date)
                                 / (config.INV.window_output/2), 0)
                        W2 = min((present_date - write_date_min)
                                 / (config.INV.window_output/2), 1)
                        # Read variables of previous output at this timestamp
                        var1 = State.load_output(present_date,name_var=Model.var_to_save)
                        # Update state by weight averaging
                        var2 = State.getvar(name_var=Model.var_to_save)
                        State.setvar(W1*var1+W2*var2,name_var=Model.var_to_save)
                
                    # Save output
                    if config.EXP.saveoutputs:
                        State.save_output(present_date,name_var=Model.var_to_save)
        
        ########################
        # 8. PARAMETERS UPDATE #
        ########################
        if config.INV.window_overlap:
            window_lag = config.INV.window_output/2
        else:
            window_lag = config.INV.window_output

        if bfn_first_window:
            middle_bfn_date = config.EXP.init_date + new_bfn_window_size/2 + window_lag
            bfn_first_window = False
        else:
            middle_bfn_date += window_lag
        if restart:
            restart = False
    print()

    return

def Inv_4Dvar(config=None,State=None,Model=None,dict_obs=None,Obsop=None,Basis=None,Bc=None,verbose=True,gpu_device=None) :

    
    '''
    Run a 4Dvar analysis
    '''

    #if 'JAX' in config.MOD.super:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
        
    
    # Module initializations
    if Model is None:
        # initialize Model operator 
        from . import mod
        Model = mod.Model(config, State, verbose=verbose)
    if Bc is None:
        # initialize Bc 
        from . import bc
        Bc = bc.Bc(config, verbose=verbose)
    if dict_obs is None:
        # initialize Obs
        from . import obs
        dict_obs = obs.Obs(config, State)
    if Obsop is None:
        # initialize Obsop
        from . import obsop
        Obsop = obsop.Obsop(config, State, dict_obs, Model, verbose=verbose)
    if Basis is None:
        # initialize Basis
        from . import basis
        Basis = basis.Basis(config, State, verbose=verbose)

    # Process observations
    print('process observation operators')
    Obsop.process_obs()
    
    # Compute checkpoints when the cost function will be evaluated 
    nstep_check = int(config.INV.timestep_checkpoint.total_seconds()//Model.dt)
    checkpoints = [0]
    time_checkpoints = [np.datetime64(Model.timestamps[0])]
    t_checkpoints = [Model.T[0]]
    check = 0
    for i,t in enumerate(Model.timestamps[:-1]):
        if i>0 and (Obsop.is_obs(t) or check==nstep_check):
            checkpoints.append(i)
            time_checkpoints.append(np.datetime64(t))
            t_checkpoints.append(Model.T[i])
            if check==nstep_check:
                check = 0
        check += 1 
    checkpoints.append(len(Model.timestamps)-1) # last timestep
    time_checkpoints.append(np.datetime64(Model.timestamps[-1]))
    t_checkpoints.append(Model.T[-1])
    checkpoints = np.asarray(checkpoints)
    time_checkpoints = np.asarray(time_checkpoints)
    print(f'--> {checkpoints.size} checkpoints to evaluate the cost function')

    # Boundary conditions
    if Bc is not None:
        var_bc = Bc.interp(time_checkpoints)
        Model.set_bc(t_checkpoints,var_bc)
    
    # Observations operator 
    if config.INV.anomaly_from_bc: # Remove boundary fields if anomaly mode is chosen
        time_obs = [np.datetime64(date) for date in Obsop.date_obs]
        var_bc = Bc.interp(time_obs)
    else:
        var_bc = None
    
    # Initial model state
    Model.init(State)
    State.plot(title='Init State')
    State.plot(title='Init params', params=True)

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in days) for which the basis components will be compute (at each timestep_checkpoint)
        Xb, Q = Basis.set_basis(time_basis, return_q=True, State=State) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only work with reduced basis!!')
    
    # Covariance matrix
    from .tools_4Dvar import Cov
    if config.INV.sigma_B is not None:     
        print('Warning: sigma_B is prescribed --> ignore Q of the reduced basis')
        # Least squares
        B = Cov(config.INV.sigma_B)
        R = Cov(config.INV.sigma_R)
    else:
        B = Cov(Q)
        R = Cov(config.INV.sigma_R)
    
    # Read Background vector 
    if config.INV.path_background is not None: 
        # Read previous minimum 
        print('Read background basis:',config.INV.path_background)
        ds = xr.open_dataset(config.INV.path_background)
        Xb[:len(ds.res.values)] = ds.res.values   
        ds.close()

    # Variational object initialization
    if config.INV.flag_full_jax:
        from .tools_4Dvar import Variational_jax as Variational
    else:
        from .tools_4Dvar import Variational as Variational

    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints, nstep=nstep_check, freq_it_plot=config.INV.freq_it_plot)
    
    # Initial Control vector 
    if config.INV.path_init_4Dvar is None:
        if config.INV.flag_full_jax:
            Xopt = jnp.zeros((Xb.size,))
        else:
            Xopt = np.zeros((Xb.size,))
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.INV.path_init_4Dvar)
        ds = xr.open_dataset(config.INV.path_init_4Dvar)
        Xopt = var.Xb*0
        Xopt[:ds.res.size] = ds.res.values
        ds.close()
        if config.INV.prec:
            Xopt = B.invsqr(Xopt - var.Xb)
    
    # Path where to save the control vector at each 4Dvar iteration 
    # (carefull, depending on the number of control variables, these files may use large disk space)
    if config.INV.path_save_control_vectors is not None:
        path_save_control_vectors = config.INV.path_save_control_vectors
    else:
        path_save_control_vectors = config.EXP.tmp_DA_path
    if not os.path.exists(path_save_control_vectors):
        os.makedirs(path_save_control_vectors)

    # Restart mode
    maxiter = config.INV.maxiter
    if config.INV.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(path_save_control_vectors,'X_it*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            try:
                ds = xr.open_dataset(tmp_files[-1])
            except:
                if len(tmp_files)>1:
                    ds = xr.open_dataset(tmp_files[-2])
            try:
                Xopt = ds.res.values
                maxiter = max(config.INV.maxiter - len(tmp_files), 0)
                ds.close()
            except:
                Xopt = +Xopt

    if not ((config.INV.restart_4Dvar or config.INV.path_init_4Dvar is not None) and maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Main function
        if config.INV.flag_full_jax:
            from jax import jit, value_and_grad
            fun = jit(value_and_grad(var.cost))
        else:
            fun = var.cost_and_grad

        # Callback function called at every minimization iterations
        def callback(XX):
            if config.INV.save_minimization:
                ds = xr.Dataset({'res':(('x',),XX)})
                ds.to_netcdf(os.path.join(path_save_control_vectors,'X_it.nc'))
                ds.close()
                
        # Minimization options
        options = {}
        if verbose:
            options['disp'] = True
        else:
            options['disp'] = False
        options['maxiter'] = maxiter

        if config.INV.ftol is not None:
            options['ftol'] = config.INV.ftol

        if config.INV.gtol is not None:
            _, g0 = fun(Xopt*0.)
            projg0 = np.max(np.abs(g0))
            options['gtol'] = config.INV.gtol*projg0
        
        # Run minimization 
        from decimal import Decimal
        import time
        class Wrapper:
            def __init__(self):
                J0, G0 = fun(Xopt)
                self.cache = {
                    'cost':J0,
                    'grad':G0
                }
                self.J_list = []
                self.G_list = []
                self.time = time.time()
                self.it = 1
                if 'gtol' in options:
                    self.gtol = options['gtol']
                else:
                    self.gtol = None
                self.filename_out = os.path.join(path_save_control_vectors, 'iterations.txt')
                with open(self.filename_out, "w") as f:
                    f.write("Minimization\n")  # Header
                

            def __call__(self, x, *args):
                cost, grad = fun(x)
                ftol = (cost - self.cache['cost']) / max(cost, self.cache['cost'], 1)
                mean_grad = np.mean(np.abs(grad))
                mean_grad_previous = np.mean(np.abs(self.cache['grad']))
                gtol = (mean_grad - mean_grad_previous) / max(mean_grad, mean_grad_previous, 1)
                self.cache['cost'] = cost
                self.cache['grad'] = grad
                time0 = time.time()
                text = "computed in %.2E second:" % (time0 - self.time) + ', x=%.2E' % Decimal(float(x.mean()))  + ', J=%.2E' % Decimal(float(cost)) + ', G=%.2E' % Decimal(float(mean_grad))  + ', ftol=%.2E' % Decimal(abs(float(ftol)))  + ', gtol=%.2E' % Decimal(abs(float(gtol))) 
                print(f"* iteration {self.it}", text)
                with open(self.filename_out, "a") as f:
                    f.write(f"iteration {self.it}, {text}\n")
                self.time = time0
                self.it += 1

                self.J_list.append(float(cost))
                self.G_list.append(float(mean_grad))

                return cost

            def jac(self, x, *args):
                return self.cache['grad']
        
        wrapper = Wrapper()
        res = opt.minimize(wrapper, Xopt,
                        method=config.INV.opt_method,
                        jac=wrapper.jac,
                        options=options,
                        callback=callback)
        

        print ('\nIs the minimization successful? {}'.format(res.success))
        print ('\nFinal cost function value: {}'.format(res.fun))
        print ('\nNumber of iterations: {}'.format(res.nit))
        
        # Save minimization trajectory
        if config.INV.save_minimization:
            ds = xr.Dataset({'cost':(('i'),np.array(wrapper.J_list)),
                             'grad':(('i'),np.array(wrapper.G_list))
                             })
            ds.to_netcdf(os.path.join(path_save_control_vectors,'minimization_trajectory.nc'))
            ds.close()

        Xres = res.x
    else:
        print('You ask for restart_4Dvar and maxiter==0, so we save directly the trajectory')
        Xres = +Xopt
        
    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    if config.INV.prec:
        Xa = var.Xb + B.sqr(Xres)
    else:
        Xa = var.Xb + Xres
        
    # Save minimum for next experiments
    ds = xr.Dataset({'res':(('x',), Xa)})
    ds.to_netcdf(os.path.join(path_save_control_vectors, 'Xres.nc'))
    ds.close()

    # Init
    State0 = State.copy()
    Model.init(State0)
    date = config.EXP.init_date
    Model.save_output(State0, date, name_var=Model.var_to_save, t=0) 
    
    nstep = min(nstep_check, int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt))
    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        if t%int(config.INV.timestep_checkpoint.total_seconds())==0:
            Basis.operg(t/3600/24, Xa, State=State0.params)

        # Forward propagation
        Model.step(t=t, State=State0, nstep=nstep)
        date += timedelta(seconds=nstep*Model.dt)

        # Save output
        if (((date - config.EXP.init_date).total_seconds()
            /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
            Model.save_output(State0, date, name_var=Model.var_to_save, t=t) 
        
    del State, State0, Xa, dict_obs, B, R, Model, Basis, var, Xopt, Xres, checkpoints, time_checkpoints, t_checkpoints
    gc.collect()
    print()
