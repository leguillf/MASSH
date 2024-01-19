#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""

import sys
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
    
    elif config.INV.super=='INV_4DVAR_JAX':
        return Inv_4Dvar_jax(config, State=State, Model=Model, dict_obs=dict_obs, Obsop=Obsop, Basis=Basis, Bc=Bc)
    
    elif config.INV.super=='INV_4DVAR_PARALLEL':
        return Inv_4Dvar_parallel(config, State=State, dict_obs=dict_obs)
    
    elif config.INV.super=='INV_MOI':
        return Inv_moi(config, State, dict_obs)
    
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
    while present_date < config.EXP.final_date :
        
        State.plot(present_date)
        
        # Propagation
        Model.step(State,nstep,t=t)

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

        # Save
        if config.EXP.saveoutputs:
            Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)
        
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


def Inv_4Dvar(config,State,Model,dict_obs=None,Obsop=None,Basis=None,Bc=None,verbose=True) :

    
    '''
    Run a 4Dvar analysis
    '''

    if 'JAX' in config.MOD.super:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
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
        Xb, Q = Basis.set_basis(time_basis,return_q=True) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only works with reduced basis!!')
    
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
        Xopt = ds.res.values
        ds.close()
    
    # Path where to save the control vector at each 4Dvar iteration 
    # (carefull, depending on the number of control variables, these files may use large disk space)
    if config.INV.path_save_control_vectors is not None:
        path_save_control_vectors = config.INV.path_save_control_vectors
    else:
        path_save_control_vectors = config.EXP.tmp_DA_path
    if not os.path.exists(path_save_control_vectors):
        os.makedirs(path_save_control_vectors)

    # Restart mode
    if config.INV.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(config.EXP.tmp_DA_path,'X_it-*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            ds = xr.open_dataset(tmp_files[-1])
            Xopt = ds.res.values
            ds.close()
        else:
            Xopt = var.Xb*0
            
    if not (config.INV.restart_4Dvar and config.INV.maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Callback function called at every minimization iterations
        def callback(XX):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H%M%S")
            ds = xr.Dataset({'res':(('x',),XX)})
            ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'X_it-'+current_time+'.nc'))
            ds.close()
                
        # Minimization options
        options = {}
        if verbose:
            options['disp'] = True
        else:
            options['disp'] = False
        options['maxiter'] = config.INV.maxiter

        if config.INV.gtol is not None:
            _ = var.cost(Xopt)
            g0 = var.grad(Xopt)
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

    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        Basis.operg(t/3600/24,Xa,State=State0)

        # Forward
        for j in range(nstep_check):
            
            if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
                Model.save_output(State0,date,name_var=Model.var_to_save,t=t+j*Model.dt) # Save output
                State0.plot(date)

            # Forward propagation
            Model.step(t=t+j*Model.dt,State=State0,nstep=1)
            date += timedelta(seconds=Model.dt)

    # Last timestep
    if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>config.EXP.init_date) & (date<=config.EXP.final_date) :
        Model.save_output(State0,date,name_var=Model.var_to_save,t=t+j*Model.dt) # Save output
    
        
    del State, State0, Xa, dict_obs, B, R
    gc.collect()
    print()

def Inv_4Dvar_jax(config,State,Model,dict_obs=None,Obsop=None,Basis=None,Bc=None,verbose=True) :
    
    '''
    Run a 4Dvar analysis
    '''
    
    # Compute checkpoints
    nstep_check = int(config.INV.timestep_checkpoint.total_seconds()//Model.dt)
    checkpoints = [0]
    time_checkpoints = [np.datetime64(Model.timestamps[0])]
    t_checkpoints = [Model.T[0]]
    check = 0
    for i,t in enumerate(Model.timestamps[:-1]):
        if i>0 and (t in Obsop.date_obs or check==nstep_check):
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
    
    # Process observations
    if config.INV.anomaly_from_bc: # Remove boundary fields if anomaly mode is chosen
        time_obs = [np.datetime64(date) for date in Obsop.date_obs]
        var_bc = Bc.interp(time_obs)
    else:
        var_bc = None
    Obsop.process_obs(var_bc)
    
    # Initial model state
    Model.init(State)
    State.plot(title='Init State')

    print("Model init : done")

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in seconds) for which the basis components will be compute (at each timestep_checkpoint)
        Xb, Q = Basis.set_basis(time_basis,return_q=True) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only work with reduced basis!!')

    print("Basis set : done")
    
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
    from .tools_4Dvar import Variational_jax as Variational
    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints)
    
    # Initial Control vector 
    if config.INV.path_init_4Dvar is None:
        Xopt = np.zeros((Xb.size,))
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.INV.path_init_4Dvar)
        ds = xr.open_dataset(config.INV.path_init_4Dvar)
        Xopt = ds.res.values
        ds.close()
    
    # Path where to save the control vector at each 4Dvar iteration 
    # (carefull, depending on the number of control variables, these files can be big)
    if config.INV.path_save_control_vectors is not None:
        path_save_control_vectors = config.INV.path_save_control_vectors
    else:
        path_save_control_vectors = config.EXP.tmp_DA_path
    if not os.path.exists(path_save_control_vectors):
        os.makedirs(path_save_control_vectors)

    # Restart mode
    if config.INV.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(config.EXP.tmp_DA_path,'X_it-*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            ds = xr.open_dataset(tmp_files[-1])
            Xopt = ds.res.values
            ds.close()
        else:
            Xopt = var.Xb*0
            
    if not (config.INV.restart_4Dvar and config.INV.maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Callback function called at every minimization iterations
        def callback(XX):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H%M%S")
            ds = xr.Dataset({'res':(('x',),XX)})
            ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'X_it-'+current_time+'.nc'))
            ds.close()
                
        # Minimization options
        options = {}
        if verbose:
            options['disp'] = True
        else:
            options['disp'] = False
        options['maxiter'] = config.INV.maxiter
        if config.INV.gtol is not None:
            _ = var.cost(Xopt)
            g0 = var.grad(Xopt)
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
    ds.to_netcdf(os.path.join(path_save_control_vectors,'Xini.nc'))
    ds.close()

    # Init
    State0 = State.copy()
    State_var = State0.var
    State_params = State0.params
    date = config.EXP.init_date

    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        State_params = Basis.operg_jit(t/3600/24, Xa, State_params)

        # Boundary conditions
        if Bc is not None:
            times = np.asarray([
                np.datetime64(date) + np.timedelta64(step*int(Model.dt),'s')\
                     for step in range(nstep_check)
                     ])
            var_bc = Bc.interp(times)
            Model.set_bc([t+step*int(Model.dt) for step in range(nstep_check)],var_bc)

        # Forward
        for j in range(nstep_check):
            
            if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
                Model.ano_bc(t+j*Model.dt,State0,+1) # Get full field from anomaly 
                State0.save_output(date,name_var=Model.var_to_save) # Save output
                Model.ano_bc(t+j*Model.dt,State0,-1) # Get anomaly from full field 

            # Forward propagation
            State_var = Model.step_jit(t+j*Model.dt,State_var, State_params,nstep=1)
            date += timedelta(seconds=Model.dt)
            for name in State_var:
                State0.var[name] = np.array(State_var[name]) # back to numpy

    # Last timestep
    if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>config.EXP.init_date) & (date<=config.EXP.final_date) :
        Model.ano_bc(t+nstep_check*Model.dt,State0,+1) # Get full field from anomaly 
        State0.save_output(date,name_var=Model.var_to_save) # Save output
        Model.ano_bc(t+nstep_check*Model.dt,State0,-1) # Get anomaly from full field 
        
    del State, State0, Xa, dict_obs, B, R
    gc.collect()
    print()

def Inv_4Dvar_parallel(config, State=None, dict_obs=None) :   

    from . import mod, state, obs, obsop, bc, basis
    from multiprocessing import Process

    # Avoid preallocating GPU memory for multi JAX processes
    if 'JAX' in config.MOD.super:
        #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(.9/config.INV.nprocs)

    # Split full experimental time window in sub windows
    list_config = []
    list_State = []
    proc = []
    iproc = 0
    date1 = config.EXP.init_date
    while iproc<config.INV.nprocs and date1<config.EXP.final_date:
        # compute subwindow
        date0 = config.EXP.init_date + iproc * config.INV.window_size_proc * (1-config.INV.overlap_frac)
        delta_t = (date0 - config.EXP.init_date) %  config.EXP.saveoutput_time_step
        date0 += delta_t
        date1 = min(date0 + config.INV.window_size_proc, config.EXP.final_date)
        if iproc==config.INV.nprocs-1 :
            date1 = config.EXP.final_date
        print(f'*** subwindow {iproc+1} from {date0} to {date1} ***')
        # create config for the subwindow
        _config = config.copy()
        _config.EXP = config.EXP.copy()
        _config.INV = config.INV.copy()
        _config.EXP.init_date = date0
        _config.EXP.final_date = date1
        _config.EXP.tmp_DA_path += f'/subwindow_{iproc+1}'
        _config.EXP.path_save += f'/subwindow_{iproc+1}'
        if _config.INV.path_save_control_vectors is not None:
            _config.INV.path_save_control_vectors += f'/subwindow_{iproc+1}' 
        # append to list
        list_config.append(_config)
        iproc += 1
        # create directories
        if not os.path.exists(_config.EXP.tmp_DA_path):
            os.makedirs(_config.EXP.tmp_DA_path)
        if not os.path.exists(_config.EXP.path_save):
            os.makedirs(_config.EXP.path_save)
        # initialize State 
        _State = state.State(_config, verbose=0)
        list_State.append(_State)
        # initialize Model operator 
        _Model = mod.Model(_config, _State, verbose=0)
        # initialize Bc 
        _Bc = bc.Bc(_config, verbose=0)
        # initialize Obs
        if dict_obs is not None:
            _dict_obs = obs._new_dict_obs(dict_obs, _config.EXP.tmp_DA_path, date_min=date0, date_max=date1)
        else:
            _dict_obs = obs.Obs(_config, _State)
        # initialize Obsop
        _Obsop = obsop.Obsop(_config, _State, _dict_obs, _Model, verbose=0)
        # initialize Basis
        _Basis = basis.Basis(config,_State, verbose=0)
        # create subprocess instance
        p = Process(target=Inv_4Dvar, args=(_config, _State, _Model, _dict_obs, _Obsop, _Basis, _Bc, 0))
        proc.append(p)
        
    # Run the subprocesses
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w") # prevent printoing outputs
    # start the processes
    for p in proc:
        p.start()
    # join the processes
    for p in proc:
        p.join()
    sys.stdout = old_stdout # reset old stdout

    # Smooth output trajectories 
    print('*** Smooth output trajectories ***')
    date = config.EXP.init_date
    for i in range(len(list_config)-1):
        config1 = list_config[i]
        config2 = list_config[i+1]
        State1 = list_State[i]
        State2 = list_State[i+1]
        while date<=config1.EXP.final_date:
            ds1 = State1.load_output(date)
            if date>=config2.EXP.init_date:
                ds2 = State2.load_output(date)
                # Weight coefficients
                W1 = (config1.EXP.final_date - date) / (config.INV.window_size_proc * config.INV.overlap_frac)
                W2 = (date - config2.EXP.init_date)  / (config.INV.window_size_proc * config.INV.overlap_frac)
                # Interpolation 
                for name in ds1.keys():
                    State.setvar(W1*ds1[name].values + W2*ds2[name].values, name)
            else:
                for name in ds1.keys():
                    State.setvar(ds1[name].values, name)
            # Save output
            State.save_output(date)
            date += config.EXP.saveoutput_time_step
    # Last subwindow 
    while date<=config2.EXP.final_date:
        ds2 = State2.load_output(date)
        for name in ds2.keys():
            State.setvar(ds2[name].values, name)
        State.save_output(date)
        date += config.EXP.saveoutput_time_step

            
    return 


def Inv_moi(config,State,dict_obs=None):
    
    
    if config.INV.dir is None:
        dir_moi = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),'moi'))
    else:
        dir_moi = config.INV.dir  
    SourceFileLoader("sparse_inversion",dir_moi + "/sparse_inversion.py").load_module() 
    SourceFileLoader("allcomps",dir_moi + "/allcomps.py").load_module() 
    SourceFileLoader("tools",dir_moi + "/tools.py").load_module() 
    SourceFileLoader("rw",dir_moi + "/rw.py").load_module() 
    SourceFileLoader("comp_iw",dir_moi + "/comp_iw.py").load_module() 
    moi = SourceFileLoader("miost",dir_moi + "/miost.py").load_module() 
    grid_moi = SourceFileLoader("miost",dir_moi + "/grid.py").load_module()
    comp_geo3 = SourceFileLoader("miost",dir_moi + "/comp_geo3.py").load_module()    
    obs = SourceFileLoader("miost",dir_moi + "/obs.py").load_module()    
    
    
    # Save grid 
    if config.INV.path_mdt is not None and os.path.exists(config.INV.path_mdt):                      
        ds = xr.open_dataset(config.INV.path_mdt).squeeze()
        
        mdt = grid.interp2d(ds,
                            config.INV.name_var_mdt,
                            State.lon,
                            State.lat)
        flag_mdt = True
    else:
        mdt = np.zeros_like(State.lon)
        flag_mdt = False
    grd = xr.Dataset({'lon':(('lon',),State.lon[State.ny//2,:]),
                      'lat':(('lat',),State.lat[:,State.nx//2]),
                      'mdt':(('lat','lon'),mdt)})
    name_grd = os.path.join(config.EXP.tmp_DA_path,'grd.nc')
    grd.to_netcdf(name_grd)
    
    dlon =  (State.lon[:,1:] - State.lon[:,:-1]).max()
    dlat =  (State.lat[1:,:] - State.lat[:-1,:]).max()
    
     # Flag initialization
    moi_first_window = True
    moi_last_window = False
    
    # MOI middle date initialization
    middle_moi_date = config.EXP.init_date

    # Loop on variables to map
    for name in config.INV.name_var:

        print(f'Mapping {name}...')

        # Initialization (normally this step is done in mod.py, but here no Model object is provided)
        State.var[config.INV.name_var[name]] = np.zeros((State.ny,State.nx))
    
        # Main time loop
        while (middle_moi_date <= config.EXP.final_date) and not moi_last_window :
            time0 = datetime.now()
            # MIOST time period
            init_moi_date = max(config.EXP.init_date, middle_moi_date - config.INV.window_size/2)
            init_moi_date += timedelta(seconds=(init_moi_date - config.EXP.init_date).total_seconds()\
                            / config.EXP.saveoutput_time_step.total_seconds()%1)
            middle_moi_date = max(middle_moi_date, config.EXP.init_date + config.INV.window_size/2)
            if ((middle_moi_date + config.INV.window_size/2) >= config.EXP.final_date):
                moi_last_window = True
                final_moi_date = config.EXP.final_date
            else:
                final_moi_date = init_moi_date + config.INV.window_size
            
                
            # CONFIG MIOST
            PHYS_COMP = []
            if config.INV.set_geo3ss6d:
                PHYS_COMP.append(
                    comp_geo3.Comp_geo3ss6d(
                        facns= 1., #factor for wavelet spacing= space
                        facnlt= 2.,
                        npsp= 3.5, # Defines the wavelet shape
                        facpsp= 1.5, #1.5 # factor to fix df between wavelets
                        lmin= config.INV.lmin, #
                        lmax= config.INV.lmax,
                        cutRo= 1.6,
                        factdec= 15,
                        tdecmin= config.INV.tdecmin, # !!!
                        tdecmax= config.INV.tdecmax,
                        tssr= 0.5,
                        facRo= 8.,
                        Romax= 150.,
                        facQ= config.INV.facQ, # TO INCREASE ENERGY
                        depth1= 0.,
                        depth2= 30.,
                        distortion_eq= 2.,
                        lat_distortion_eq= 5.,
                        distortion_eq_law= 2.,
                        file_aux= config.INV.file_aux,
                        filec_aux= config.INV.filec_aux,
                        write= True,
                        Hvarname= 'Hss')
                )
            
            if config.INV.set_geo3ls:
                PHYS_COMP.append(
                    comp_geo3.Comp_geo3ls(
                        facnls= 3., #factor for large-scale wavelet spacing
                        facnlt= 3.,
                        tdec_lw= 25.,
                        std_lw= 0.04,
                        lambda_lw= 970, #768.05127036
                        file_aux= config.INV.file_aux,
                        filec_aux= config.INV.filec_aux,
                        write= True,
                        Hvarname= 'Hls')
                )

            config_moi = dict(
            
                RUN_NAME = '', # Set automatically with filename
                PATH = dict(OUTPUT= config.EXP.tmp_DA_path),
                
                ALGO = dict(
                    USE_MPI= False,
                    store_gtranspose= False, # only if USE_MPI
                    INV_METHOD= 'PCG_INV',
                    NITER= 800  , # Maximum number of iterations in the variational loop
                    EPSPILLON_REST= 1.e-7,
                    gsize_max = 5000000000 ,
                    float_type= 'f8',
                    int_type= 'i8'),
                
                GRID = grid_moi.Grid_msit(
                    TEMPLATE_FILE= name_grd,
                    LON_NAME= 'lon',
                    LAT_NAME= 'lat',
                    MDT_NAME= 'mdt',
                    FLAG_MDT= flag_mdt,
                    DATE_MIN= init_moi_date.strftime("%Y-%m-%d"),
                    DATE_MAX= final_moi_date.strftime("%Y-%m-%d"),
                    TIME_STEP= config.EXP.saveoutput_time_step.total_seconds()/(24*3600),
                    NSTEPS_NC= int((24*3600)//config.EXP.saveoutput_time_step.total_seconds()),
                    TIME_STEP_LF= 10000., # For internal tides with seasons
                    LON_MIN= State.lon.min()-dlon,
                    LON_MAX= State.lon.max()+dlon,
                    LAT_MIN= State.lat.min()-dlat,
                    LAT_MAX= State.lat.max()+dlat,
                    tref_iw= 15340.),
                
                PHYS_COMP=PHYS_COMP,
                
                OBS_COMP=[
                
                    ],
                
                
                OBS=[
                
                    obs.MASSH(
                        name=config.EXP.name_experiment,
                        dict_obs= dict_obs,
                        name_var=name,
                        noise=config.INV.sigma_R
                        ),
                    
                        ]
                
                )
            
            # RUN MOI
            moi.run_miost(config_moi)
            
            # READ OUTPUTS AND REFORMAT
            ds = xr.open_mfdataset(os.path.join(config.EXP.tmp_DA_path,'_ms_analysis*.nc'),
                                combine='by_coords')
            if config.INV.set_geo3ss6d and config.INV.set_geo3ls:              
                ssh = ds['Hss'] + ds['Hls']
            elif config.INV.set_geo3ss6d:
                ssh = ds['Hss']
            else:
                ssh = ds['Hls']
                
            # SAVE OUTPUTS 
            # Set the saving temporal window
            if moi_first_window:
                write_date_min = init_moi_date
                write_date_max = init_moi_date + config.INV.window_size/2 + config.INV.window_output/2
            elif moi_last_window:
                write_date_min = middle_moi_date - config.INV.window_output/2
                write_date_max = final_moi_date
            else:
                write_date_min = middle_moi_date - config.INV.window_output/2
                write_date_max = middle_moi_date + config.INV.window_output/2
                
            State_tmp = State.copy(free=True)
            date = init_moi_date
            i = 0
            while date<=final_moi_date:
                if (date >= write_date_min) & (date <= write_date_max) :
                    
                    if config.INV.window_overlap and not moi_first_window and\
                            date<=middle_moi_date:
                        # weight coefficients
                        W1 = max((middle_moi_date - date)
                                / (config.INV.window_output/2), 0)
                        W2 = min((date - write_date_min)
                                / (config.INV.window_output/2), 1)
                        # Read previous output at this timestamp
                        ds1 = State.load_output(date)
                        ssh1 = ds1[config.INV.name_var[name]].data
                        ds1.close()
                        del ds1
                        # Update state
                        State_tmp.setvar(W1*ssh1 + W2*ssh[i].values + mdt, config.INV.name_var[name])
                    else:
                        State_tmp.setvar(ssh[i].values + mdt, config.INV.name_var[name])
                        
                    # Save output
                    if config.EXP.saveoutputs:
                        State_tmp.save_output(date)
                    
                date += config.EXP.saveoutput_time_step
                i += 1
            
            if config.INV.window_overlap:
                window_lag = config.INV.window_output/2
            else:
                window_lag = config.INV.window_output

            if moi_first_window:
                middle_moi_date = config.EXP.init_date + config.INV.window_size/2 + window_lag
                moi_first_window = False
            else:
                middle_moi_date += window_lag
            
            ds.close()
            del ds
            
            cmd = 'rm ' + os.path.join(config.EXP.tmp_DA_path,'_ms_analysis*.nc')
            os.system(cmd)
            
            time1 = datetime.now()
            print('Loop from',init_moi_date.strftime("%Y-%m-%d"),'to',
                final_moi_date.strftime("%Y-%m-%d : in"),time1-time0,'seconds')
         

        
