#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""

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
    
    elif config.INV.super=='INV_4DVAR_PARALLEL':
        return Inv_4Dvar_parallel(config, State=State)
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
        time_bc = [np.datetime64(time) for time in Model.timestamps[::nstep]]
        t_bc = [t for t in Model.T[::nstep]]
        var_bc = Bc.interp(time_bc)
        Model.set_bc(t_bc,var_bc)

    t = 0
    Model.init(State,t)
    State.plot(title='Start of forward integration')

    while present_date + timedelta(seconds=nstep*Model.dt) <= config.EXP.final_date :
        
        State.plot(present_date)
        
        # Propagation
        Model.step(State,nstep,t=t)

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

        # Save
        if config.EXP.saveoutputs:
            Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)
            
            #State.plot(present_date)
    
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

def Inv_4Dvar(config,State,Model=None,dict_obs=None,Obsop=None,Basis=None,Bc=None,verbose=True) :

    
    '''
    Run a 4Dvar analysis
    '''

    if 'JAX' in config.MOD.super:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        
    
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
    Obsop.process_obs(var_bc)
    
    # Initial model state
    Model.init(State)
    State.plot(title='Init State')

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
        
    # Variational object initialization
    from .tools_4Dvar import Variational as Variational
    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints)
    
    # Initial Control vector 
    if config.INV.path_init_4Dvar is None:
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
        tmp_files = sorted(glob.glob(os.path.join(path_save_control_vectors,'X_it-*.nc')))
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
            
    if not (config.INV.restart_4Dvar and maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Callback function called at every minimization iterations
        def callback(XX):
            if config.INV.save_minimization:
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d_%H%M%S")
                ds = xr.Dataset({'res':(('x',),XX)})
                ds.to_netcdf(os.path.join(path_save_control_vectors,'X_it-'+current_time+'.nc'))
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
            _ = var.cost(Xopt*0)
            g0 = var.grad(Xopt*0)
            projg0 = np.max(np.abs(g0))
            options['gtol'] = config.INV.gtol*projg0
        
            
        # Run minimization 
        res = opt.minimize(var.cost, Xopt,
                        method=config.INV.opt_method,
                        jac=var.grad,
                        options=options,
                        callback=callback)

        print ('\nIs the minimization successful? {}'.format(res.success))
        print ('\nFinal cost function value: {}'.format(res.fun))
        print ('\nNumber of iterations: {}'.format(res.nit))
        
        # Save minimization trajectory
        if config.INV.save_minimization:
            ds = xr.Dataset({'cost':(('j'),var.J),'grad':(('g'),var.G)})
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
    ds = xr.Dataset({'res':(('x',),Xa)})
    ds.to_netcdf(os.path.join(path_save_control_vectors,'Xres.nc'))
    ds.close()

    # Init
    State0 = State.copy()
    date = config.EXP.init_date
    Model.save_output(State0,date,name_var=Model.var_to_save,t=0) 
    
    nstep = min(nstep_check, int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt))
    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        Basis.operg(t/3600/24,Xa,State=State0)

        # Forward propagation
        Model.step(t=t,State=State0,nstep=nstep)
        date += timedelta(seconds=nstep_check*Model.dt)

        # Save output
        if (((date - config.EXP.init_date).total_seconds()
            /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
            Model.save_output(State0,date,name_var=Model.var_to_save,t=t) 

        if False:
            # Forward
            for j in range(nstep_check):
                
                if (((date - config.EXP.init_date).total_seconds()
                    /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                    & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
                    Model.save_output(State0,date,name_var=Model.var_to_save,t=t+j*Model.dt) # Save output

                # Forward propagation
                Model.step(t=t+j*Model.dt,State=State0,nstep=1)
                date += timedelta(seconds=Model.dt)
        

        
        State0.plot(date)

    # Last timestep
    if False:
        if (((date - config.EXP.init_date).total_seconds()
                    /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                    & (date>config.EXP.init_date) & (date<=config.EXP.final_date) :
            Model.save_output(State0,date,name_var=Model.var_to_save,t=t+j*Model.dt) # Save output
        
        
    del State, State0, Xa, dict_obs, B, R, Model, Basis, var, Xopt, Xres, checkpoints, time_checkpoints, t_checkpoints
    gc.collect()
    print()
    
def Inv_4Dvar_parallel(config, State=None) :  
    
    from . import state
    from .tools import gaspari_cohn
    import concurrent.futures
    from scipy.interpolate import griddata

    # Split full experimental time window in sub windows
    list_config = []
    list_State = []
    list_config_1d = []
    list_State_1d = []
    weights_space = [] 
    weights_space_sum = np.zeros((State.ny, State.nx))
    processes = []
    list_date = []
    list_lonlat = []
    iproc = 0
    n_wt = 0 
    n_wx = 0
    n_wy = 0

    date1 = config.EXP.init_date
    lat1 = config.GRID.lat_min
    lon1 = config.GRID.lon_min
    i = -1
    while date1<config.EXP.final_date:
        list_config.append([])
        list_State.append([])
        i += 1
        # compute subwindow time period
        if config.INV.time_window_size_proc is not None:
            time_delta = timedelta(days=config.INV.time_window_size_proc)
            date0 = config.EXP.init_date + n_wt * time_delta * (1-config.INV.time_overlap_frac)
            delta_t = (date0 - config.EXP.init_date) %  config.EXP.saveoutput_time_step
            date0 += delta_t
            date1 = min(date0 + time_delta, config.EXP.final_date)
            n_wt += 1
        else:
            date0 = config.EXP.init_date
            date1 = config.EXP.final_date
        list_date.append(date0 + (date1-date0)/2)

        while lat1<config.GRID.lat_max:
            # compute subwindow latitude borders
            if config.INV.space_window_size_proc is not None:
                lat0 = config.GRID.lat_min + n_wy * config.INV.space_window_size_proc * (1-config.INV.space_overlap_frac)
                lat1 = min(lat0 + config.INV.space_window_size_proc, config.GRID.lat_max)
                n_wy += 1
            else:
                lat0 = config.GRID.lat_min
                lat1 = config.GRID.lat_max
            while lon1<config.GRID.lon_max:
                # compute subwindow longitude borders
                if config.INV.space_window_size_proc is not None:
                    lon0 = config.GRID.lon_min + n_wx * config.INV.space_window_size_proc * (1-config.INV.space_overlap_frac)
                    lon1 = min(lon0 + config.INV.space_window_size_proc, config.GRID.lon_max)
                    n_wx += 1
                else:
                    lon0 = config.GRID.lon_min
                    lon1 = config.GRID.lon_max
                if i==0:
                    list_lonlat.append(((lon1+lon0)/2,(lat1+lat0)/2))
                # create config for the subwindow
                print(f'*** subwindow {iproc+1} from {date0} to {date1}, latitude from {lat0}° to {lat1}°, longitude from {lon0}° to {lon1}° ***')
                iproc += 1
                _config = config.copy()
                _config.EXP = config.EXP.copy()
                _config.GRID = config.GRID.copy()
                _config.INV = config.INV.copy()
                _config.EXP.init_date = date0
                _config.EXP.final_date = date1
                _config.GRID.lon_min = lon0
                _config.GRID.lon_max = lon1
                _config.GRID.lat_min = lat0
                _config.GRID.lat_max = lat1
        
                name_subwindow = f'{str(list_date[-1])[:10]}_{round((lon1+lon0)/2)}_{round((lat1+lat0)/2)}'
                _config.EXP.tmp_DA_path += f'/subwindow_{name_subwindow}'
                _config.EXP.path_save += f'/subwindow_{name_subwindow}'
                if _config.INV.path_save_control_vectors is not None:
                    _config.INV.path_save_control_vectors += f'/subwindow_{name_subwindow}' 
                # append to list
                list_config[i].append(_config)
                list_config_1d.append(_config)
                # create directories
                if not os.path.exists(_config.EXP.tmp_DA_path):
                    os.makedirs(_config.EXP.tmp_DA_path)
                if not os.path.exists(_config.EXP.path_save):
                    os.makedirs(_config.EXP.path_save)

                # initialize State 
                _State = state.State(_config, verbose=0)
                list_State[i].append(_State)   
                list_State_1d.append(_State)   
                
                # Compute spatial window tappering for merging outputs after inversion
                if i==0: # Only for first time window (useless to compute it for the others, because is identical)
                    winy = np.ones(_State.ny)
                    winx = np.ones(_State.nx)
                    winy[:int(_State.ny-_State.ny/2)] = gaspari_cohn(np.arange(0,_State.ny),_State.ny/2)[:int(_State.ny/2)][::-1]
                    winx[:int(_State.nx-_State.nx/2)] = gaspari_cohn(np.arange(0,_State.nx),_State.nx/2)[:int(_State.nx/2)][::-1]
                    winy[int(_State.ny/2):] = gaspari_cohn(np.arange(0,_State.ny),_State.ny/2)[:_State.ny-int(_State.ny/2)]
                    winx[int(_State.nx/2):] = gaspari_cohn(np.arange(0,_State.nx),_State.nx/2)[:_State.nx-int(_State.nx/2)]
                    _weights_space = winy[:,np.newaxis] * winx[np.newaxis,:]
                    _weights_space_interp = griddata((_State.lon.ravel(),_State.lat.ravel()), _weights_space.ravel(), (State.lon.ravel(),State.lat.ravel()), method='linear').reshape(State.lon.shape)
                    ind = ~np.isnan(_weights_space_interp)
                    weights_space.append(_weights_space_interp)
                    weights_space_sum[ind] += _weights_space_interp[ind]
            n_wx = 0
            lon1 = config.GRID.lon_min
        n_wy = 0
        lat1 = config.GRID.lat_min
    
    # Run the subprocesses
    if not config.INV.merge_outputs_only:
        if 'JAX' in config.MOD.super: # Avoid preallocating GPU memory for multi JAX processes
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            if config.INV.JAX_mem_fraction is not None and config.INV.JAX_mem_fraction>0:
                os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(min(config.INV.JAX_mem_fraction,1))
            elif len(processes)>0:
                os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(.9/config.INV.nprocs)

        # Run tasks in parallel with a maximum of config.INV.nprocs processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.INV.nprocs) as executor:
            futures = [executor.submit(Inv_4Dvar, _config, _State) for (_config, _State) in zip(list_config_1d, list_State_1d)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(f"Processed and saved: {result}")
                except Exception as exc:
                    print(f"An error occurred: {exc}")
                
    # Merge output trajectories 
    from . import mod
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    import warnings
    warnings.filterwarnings('ignore')

    print('*** Merge output trajectories ***')
    Model = mod.Model(config, State, verbose=0)
    Model.init(State)
    date = config.EXP.init_date
    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)  # Kernel to convolve output maps to replace NaN pixels close to the coast for interpolation
    # Merge in time and space
    for i in range(len(list_date)):
        _config = list_config[i][0]
        while date<=_config.EXP.final_date:
            for name in State.var:
                # Space smoothing for first time window
                _var1 = np.zeros((State.ny, State.nx)) 
                for j in range(len(list_lonlat)):
                    _State1 = list_State[i][j]
                    _ds1 = _State1.load_output(date)
                    lon = _ds1.lon.values
                    lat = _ds1.lat.values
                    if len(lon.shape)==1:
                        lon,lat = np.meshgrid(lon,lat)
                    _var = _ds1[name].values
                    #if np.any(np.isnan(_var)):
                    #    _var = interpolate_replace_nans(_var, kernel)
                    _var_interp = griddata((lon.ravel(),lat.ravel()),_var.ravel(),
                                        (State.lon.ravel(),State.lat.ravel()), 
                                        method='linear').reshape(State.lon.shape)
                    ind = ~np.isnan(_var_interp)
                    _var1[ind] += (weights_space[j]*_var_interp/weights_space_sum)[ind]
                    #except:
                        #print(f'Warning: no output for tile {list_lonlat[j]} at date {date}, we take 0')
                if State.mask is not None and np.any(State.mask):
                    _var1[State.mask] = np.nan
                # Time smoothing (except last window)
                if len(list_date)>1 and i<len(list_date)-1 and date>=list_config[i+1][0].EXP.init_date and config.INV.time_overlap_frac>0 and date>=list_config[i+1][0].EXP.init_date:
                    # Space smoothing for second time window
                    _var2 = np.zeros((State.ny, State.nx))
                    for j in range(len(list_lonlat)):
                        _State2 = list_State[i+1][j]
                        #try:
                        _ds2 = _State2.load_output(date)
                        lon = _ds2.lon.values
                        lat = _ds2.lat.values
                        if len(lon.shape)==1:
                            lon,lat = np.meshgrid(lon,lat)
                        _var = _ds2[name].values
                        if np.any(np.isnan(_var)):
                            _var = interpolate_replace_nans(_var, kernel)
                        _var_interp = griddata((lon.ravel(),lat.ravel()),_var.ravel(),
                                            (State.lon.ravel(),State.lat.ravel()), 
                                            method='linear').reshape(State.lon.shape)
                        ind = ~np.isnan(_var_interp)
                        _var2[ind] += (weights_space[j]*_var_interp/weights_space_sum)[ind]
                        #except:
                        #    print(f'Warning: no output for tile {list_lonlat[j]} at date {date}, we take 0')
                    if State.mask is not None and np.any(State.mask):
                        _var2[State.mask] = np.nan
                    # Weight coefficients
                    W1 = (list_config[i][0].EXP.final_date - date).total_seconds()  / (24*3600*config.INV.time_window_size_proc * config.INV.time_overlap_frac)
                    W2 = (date - list_config[i+1][0].EXP.init_date).total_seconds() / (24*3600*config.INV.time_window_size_proc * config.INV.time_overlap_frac)
                    # Interpolation 
                    State.setvar((W1*_var1 + W2*_var2)/(W1 + W2), name)
                else:
                    State.setvar(_var1, name)
                # Save output
                State.save_output(date)
                State.plot(date)

            date += config.EXP.saveoutput_time_step
                    
    return 
