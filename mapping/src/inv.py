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
import pickle
from datetime import datetime,timedelta
import scipy.optimize as opt
import gc
import pandas as pd
import xarray as xr
import glob
from importlib.machinery import SourceFileLoader 

from . import grid


def Inv(config, State, Model, dict_obs=None, Obsop=None, Basis=None, Bc=None, test=False, *args, **kwargs):
    """
    NAME
        Inv

    DESCRIPTION
        Main function calling subfunctions for specific Inversion algorithms
    """
    
    if config.INV is None:
        return Inv_forward(config,State,Model,Bc=Bc)
    
    print(config.INV)
    
    if config.INV.super=='INV_OI':
        return Inv_oi(config,State,dict_obs)
    
    elif config.INV.super=='INV_BFN':
        return Inv_bfn(config,State,Model,dict_obs,Bc=Bc)
    
    elif config.INV.super=='INV_4DVAR':
        print(test)
        return Inv_4Dvar(config,State,Model,dict_obs=dict_obs,Obsop=Obsop,Basis=Basis,Bc=Bc,test=test)
    
    elif config.INV.super=='INV_MIOST':
        return Inv_miost(config,State,dict_obs)
    
    
    else:
        sys.exit(config.INV.super + ' not implemented yet')
        

def Inv_forward(config,State,Model,Bc=None):
    
    """
    NAME
        Inv_forward

    DESCRIPTION
        Run a model forward integration  
    
    """
    
    present_date = config.EXP.init_date
    if config.EXP.saveoutputs:
        State.save_output(present_date,name_var=Model.var_to_save)
        
    nstep = int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt)

    if Bc is not None:
        time_bc = [np.datetime64(time) for time in Model.timestamps[::nstep]]
        t_bc = [t for t in Model.T[::nstep]]
        var_bc = Bc.interp(time_bc,State.lon,State.lat)
        Model.set_bc(t_bc,var_bc)

    t = 0
    while present_date < config.EXP.final_date :
        
        State.plot(present_date)
        
        # Propagation
        Model.step(State,nstep,t=t)

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

        # Save
        if config.EXP.saveoutputs:
            State.save_output(present_date,name_var=Model.var_to_save)
        
    return

         
def Inv_oi(config,State,dict_obs):
    
    """
    NAME
        Inv_oi

    DESCRIPTION
        Run an optimal interpolation experiment 
    
    """
    
    from . import obs
    
    
    
    box = [State.lon.min(),State.lon.max(),State.lat.min(),State.lat.max(),
           None, None]
    
    ndays = (config.final_date-config.init_date).total_seconds()/3600/24
    
    dt = config.saveoutput_time_step.total_seconds()/3600/24
    times = np.arange(0, ndays + dt, dt)
    lon1d = State.lon.flatten()
    lat1d = State.lat.flatten()
    
    State0 = State.copy(free=True)
    
    # Time loop
    for i,t in enumerate(times):
        
        box[4] = config.init_date + timedelta(days=t-config.oi_Lt)
        box[5] = config.init_date + timedelta(days=t+config.oi_Lt)
        
        obs_val, obs_coords, obs_coords_att = obs.get_obs(dict_obs,box,config.init_date)
        
        obs_lon = obs_coords[0]
        obs_lat = obs_coords[1]
        obs_time = obs_coords[2]
        
        BHt = np.exp(-((t - obs_time[np.newaxis,:])/config.oi_Lt)**2 - 
                    ((lon1d[:,np.newaxis] - obs_lon[np.newaxis,:])/config.oi_Lx)**2 - 
                    ((lat1d[:,np.newaxis] - obs_lat[np.newaxis,:])/config.oi_Ly)**2)
        
        
        HBHt = np.exp(-((obs_time[np.newaxis,:] - obs_time[:,np.newaxis])/config.oi_Lt)**2 -
                    ((obs_lon[np.newaxis,:] - obs_lon[:,np.newaxis])/config.oi_Lx)**2 -
                    ((obs_lat[np.newaxis,:] - obs_lat[:,np.newaxis])/config.oi_Ly)**2)
            
            
            
        R = np.diag(np.full((len(obs_val)), config.oi_noise**2))

        Coo = HBHt + R
        Mi = np.linalg.inv(Coo)
    
        sol = np.dot(np.dot(BHt, Mi), obs_val).reshape((State.ny,State.nx))
        
        # Save output
        State0.setvar(sol,ind=0)
        date = config.init_date + timedelta(days=t)
        State0.save_output(date)

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
        
        ######################################
        # 3. BOUNDARY AND INITIAL CONDITIONS #
        ######################################
        # Boundary conditions
        Wbc = None
        if Bc is not None:
            periods = int((final_bfn_date-init_bfn_date).total_seconds()//\
                one_time_step.total_seconds() + 1)
            time_bc = [np.datetime64(time) for time in Model.timestamps[::periods]]
            t_bc = [t for t in Model.T[::periods]]
            var_bc = Bc.interp(time_bc,State.lon,State.lat)
            Wbc = Bc.compute_weight_map(State.lon,State.lat,State.mask)
            Model.set_bc(t_bc,var_bc,Wbc=Wbc)
            

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
                t = (present_date_forward0 - init_bfn_date).total_seconds()

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
                    t = (present_date_backward0 - init_bfn_date).total_seconds()

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



def Inv_4Dvar(config,State,Model,dict_obs=None,Obsop=None,Basis=None,Bc=None,test=False) :
    
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
        var_bc = Bc.interp(time_checkpoints,State.lon,State.lat)
        Model.set_bc(t_checkpoints,var_bc)

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in seconds) for which the basis components will be compute (at each timestep_checkpoint)
        Q = Basis.set_basis(time_basis,return_q=True) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only works with reduced basis!!')

    # Backgroud state 
    Xb = np.zeros((Q.size,))
    
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
        
    # Cost and Grad functions
    from .tools_4Dvar import Variational
    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints)
    
    if test : 
        return var
    
    # Initial State 
    if config.INV.path_init_4Dvar is None:
        Xopt = np.zeros((Xb.size,))
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.INV.path_init_4Dvar)
        ds = xr.open_dataset(config.INV.path_init_4Dvar)
        Xopt = ds.res.values
        ds.close()
    
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
        options = {'disp': True, 'maxiter': config.INV.maxiter}
        if config.INV.gtol is not None:
            J0 = var.cost(Xopt)
            g0 = var.grad(Xopt)
            projg0 = np.max(np.abs(g0))
            options['gtol'] = config.INV.gtol*projg0
            
        # Run minimization 
        res = opt.minimize(var.cost,Xopt,
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
            ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'minimization_trajectory.nc'))
            ds.close()

        Xres = res.x
    else:
        print('You ask for restart_4Dvar and maxiter==0, so we move directly to the saving of the trajectory')
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
    ds.to_netcdf(os.path.join(config.EXP.tmp_DA_path,'Xini.nc'))
    ds.close()

    # Init
    State0 = State.copy(free=True)
    date = config.EXP.init_date

    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        Basis.operg(t/3600/24,Xa,State=State0)

        # Boundary conditions
        if Bc is not None:
            times = np.asarray([
                np.datetime64(date) + np.timedelta64(step*int(Model.dt),'s')\
                     for step in range(nstep_check)
                     ])
            var_bc = Bc.interp(times,State.lon,State.lat)
            Model.set_bc([t+step*int(Model.dt) for step in range(nstep_check)],var_bc)

        # Forward
        for j in range(nstep_check):
            
            if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
                # Save output
                State0.save_output(date,name_var=Model.var_to_save)

            # Forward propagation
            Model.step(t=t+j*Model.dt,State=State0,nstep=1)
            date += timedelta(seconds=Model.dt)

    # Last timestep
    if (((date - config.EXP.init_date).total_seconds()
                 /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>config.EXP.init_date) & (date<=config.EXP.final_date) :
        State0.save_output(date,name_var=Model.var_to_save)
        
    del State, State0, Xa, dict_obs, B, R
    gc.collect()
    print()
    

def Inv_incr4Dvar(config,State,Model,dict_obs=None) :
    
    '''
    Run a 4Dvar analysis
    '''

    print('\n*** Observation operator ***\n')
    from .tools_4Dvar import Obsopt
    H = Obsopt(config,State,dict_obs,Model)
    
    print('\n*** Reduced basis ***\n')
    
    if config.name_model=='QG1L':
        from .tools_reduced_basis import RedBasis_BM as RedBasis
    elif config.name_model=='SW1L':
        from .tools_reduced_basis import RedBasis_IT as RedBasis
    elif hasattr(config.name_model,'__len__') and len(config.name_model)==2:
        from .tools_reduced_basis import RedBasis_BM_IT as RedBasis
        
    basis = RedBasis(config)
    time_basis = H.checkpoint * Model.dt / (24*3600)
    Q = basis.set_basis(time_basis,State.lon,State.lat,return_q=True)

    print('\n*** Covariances ***\n')
    from .tools_4Dvar import Cov
    # Covariance matrix
    if config.sigma_B is not None:          
        # Least squares
        B = Cov(config.sigma_B)
        R = Cov(config.sigma_R)
    else:
        B = Cov(Q)
        R = Cov(config.sigma_R)
        
    print('\n*** Variational ***\n')
    from .tools_4Dvar import Variational
    # backgroud state 
    Xb = np.zeros((basis.nbasis,))
    # Cost and Grad functions
    var = Variational(
        config=config, M=Model, H=H, State=State, B=B, R=R, basis=basis, Xb=Xb)

    # Initial State of the outer loop
    if config.path_init_4Dvar is None:
        Xa = var.Xb*0
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.path_init_4Dvar)
        ds = xr.open_dataset(config.path_init_4Dvar)
        Xa = ds.res.values
        ds.close()

    
    # Restart mode
    if config.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(config.tmp_DA_path,'X_it-*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            ds = xr.open_dataset(tmp_files[-1])
            Xa = ds.res.values
            ds.close()
        else:
            Xa = var.Xb*0
    
    
        
    ###################
    #  Outer loop     #
    ###################
    for i in range(config.maxiter_outer):
        
        print(f'*** Outer loop {i} ***')
        
        ######################
        #  Model integration #    
        ######################
        J = var.cost(Xa)
        print(f'\ncost: {J}\n')

        ###################
        #  Inner loop     #
        ###################
        # Initial State of the inner loop
        dX = Xa*0
        var.X0 = +Xa
    
        # Callback function called at every minimization iterations
        def callback(XX):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H%M%S")
            ds = xr.Dataset({'res':(('x',),XX)})
            ds.to_netcdf(os.path.join(config.tmp_DA_path,f'X{i}_'+current_time+'.nc'))
            ds.close()
        # Minimization options
        options = {'disp': True, 'maxiter': config.maxiter_inner}
                
        # Run minimization 
        res = opt.minimize(var.dcost,dX,
                        method=config.opt_method,
                        jac=var.grad,
                        options=options,
                        callback=callback)
        
        dXa = res.x
        
        ###################
        #    Update       #
        ###################
        Xa += dXa
    
    
    # Save minimization trajectory
    if config.save_minimization:
        ds = xr.Dataset({'dcost':(('dj'),var.dJ),'cost':(('j'),var.J),'grad':(('g'),var.G)})
        ds.to_netcdf(os.path.join(config.tmp_DA_path,'minimization_trajectory.nc'))
        ds.close()
        
    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    if config.prec:
        Xa = var.Xb + B.sqr(Xa)
    else:
        Xa = var.Xb + Xa
        
    # Save minimum for next experiments
    ds = xr.Dataset({'res':(('x',),Xa)})
    ds.to_netcdf(os.path.join(config.tmp_DA_path,'Xini.nc'))
    ds.close()

    # Init
    State0 = State.copy(free=True)
    State0.params = np.zeros((Model.nparams,))
    date = config.init_date

    # Forward propagation
    while date<config.final_date:
        # current time in secondes
        t = (date - config.init_date).total_seconds()
        
        # Reduced basis
        basis.operg(Xa,t/3600/24,State=State0)
        
        # Forward
        for j in range(config.checkpoint):
            
            Model.step(t+j*config.dtmodel,State0,nstep=1)
    
            date += timedelta(seconds=config.dtmodel)
            
            if (((date - config.init_date).total_seconds()
                 /config.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>config.init_date) & (date<=config.final_date) :
                
                # Save output
                State0.save_output(date,mdt=Model.mdt)

        
    del State, State0, res, Xa, dict_obs, B, R
    gc.collect()
    print()
    

def Inv_miost(config,State,dict_obs=None):
    
    
    if config.dir_miost is None:
        dir_miost = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),'miost'))
    else:
        dir_miost = config.dir_miost  
    SourceFileLoader("sparse_inversion",dir_miost + "/sparse_inversion.py").load_module() 
    SourceFileLoader("allcomps",dir_miost + "/allcomps.py").load_module() 
    SourceFileLoader("tools",dir_miost + "/tools.py").load_module() 
    SourceFileLoader("rw",dir_miost + "/rw.py").load_module() 
    SourceFileLoader("comp_iw",dir_miost + "/comp_iw.py").load_module() 
    miost = SourceFileLoader("miost",dir_miost + "/miost.py").load_module() 
    grid_miost = SourceFileLoader("miost",dir_miost + "/grid.py").load_module()
    comp_geo3 = SourceFileLoader("miost",dir_miost + "/comp_geo3.py").load_module()    
    obs = SourceFileLoader("miost",dir_miost + "/obs.py").load_module()    
    
    
    # Save grid 
    if config.path_mdt is not None and os.path.exists(config.path_mdt):                      
        ds = xr.open_dataset(config.path_mdt).squeeze()
        
        mdt = grid.interp2d(ds,
                            config.name_var_mdt,
                            State.lon,
                            State.lat)
        flag_mdt = True
    else:
        mdt = np.zeros_like(State.lon)
        flag_mdt = False
    grd = xr.Dataset({'lon':(('lon',),State.lon[State.ny//2,:]),
                      'lat':(('lat',),State.lat[:,State.nx//2]),
                      'mdt':(('lat','lon'),mdt)})
    name_grd = os.path.join(config.tmp_DA_path,'grd.nc')
    grd.to_netcdf(name_grd)
    
    dlon =  (State.lon[:,1:] - State.lon[:,:-1]).max()
    dlat =  (State.lat[1:,:] - State.lat[:-1,:]).max()
    
     # Flag initialization
    miost_first_window = True
    miost_last_window = False
    
    # MIOST middle date initialization
    middle_miost_date = config.init_date
    
    # Main time loop
    while (middle_miost_date <= config.final_date) and not miost_last_window :
        time0 = datetime.now()
        #############
        # 1. SET-UP #
        #############
        # MIOST time period
        init_miost_date = max(config.init_date, middle_miost_date - config.miost_window_size/2)
        init_miost_date += timedelta(seconds=(init_miost_date - config.init_date).total_seconds()\
                         / config.saveoutput_time_step.total_seconds()%1)
        middle_miost_date = max(middle_miost_date, config.init_date + config.miost_window_size/2)
        if ((middle_miost_date + config.miost_window_size/2) >= config.final_date):
            miost_last_window = True
            final_miost_date = config.final_date
        else:
            final_miost_date = init_miost_date + config.miost_window_size
        
            
        # CONFIG MIOST
        PHYS_COMP = []
        if config.miost_geo3ss6d:
            PHYS_COMP.append(
                comp_geo3.Comp_geo3ss6d(
                    facns= 1., #factor for wavelet spacing= space
                    facnlt= 2.,
                    npsp= 3.5, # Defines the wavelet shape
                    facpsp= 1.5, #1.5 # factor to fix df between wavelets
                    lmin= config.lmin, # !!!
                    lmax= config.lmax,
                    cutRo= 1.6,
                    factdec= 15,
                    tdecmin= config.tdecmin, # !!!
                    tdecmax= config.tdecmax,
                    tssr= 0.5,
                    facRo= 8.,
                    Romax= 150.,
                    facQ= config.facQ, # TO INCREASE ENERGY
                    depth1= 0.,
                    depth2= 30.,
                    distortion_eq= 2.,
                    lat_distortion_eq= 5.,
                    distortion_eq_law= 2.,
                    file_aux= config.file_aux,
                    filec_aux= config.filec_aux,
                    write= True,
                    Hvarname= 'Hss')
            )
        
        if config.miost_geo3ls:
            PHYS_COMP.append(
                comp_geo3.Comp_geo3ls(
                    facnls= 3., #factor for large-scale wavelet spacing
                    facnlt= 3.,
                    tdec_lw= 25.,
                    std_lw= 0.04,
                    lambda_lw= 970, #768.05127036
                    file_aux= config.file_aux,
                    filec_aux= config.filec_aux,
                    write= True,
                    Hvarname= 'Hls')
            )

        config_miost = dict(
        
            RUN_NAME = '', # Set automatically with filename
            PATH = dict(OUTPUT= config.tmp_DA_path),
            
            ALGO = dict(
                USE_MPI= False,
                store_gtranspose= False, # only if USE_MPI
                INV_METHOD= 'PCG_INV',
                NITER= 800  , # Maximum number of iterations in the variational loop
                EPSPILLON_REST= 1.e-7,
                gsize_max = 5000000000 ,
                float_type= 'f8',
                int_type= 'i8'),
            
            GRID = grid_miost.Grid_msit(
                TEMPLATE_FILE= name_grd,
                LON_NAME= 'lon',
                LAT_NAME= 'lat',
                MDT_NAME= 'mdt',
                FLAG_MDT= flag_mdt,
                DATE_MIN= init_miost_date.strftime("%Y-%m-%d"),
                DATE_MAX= final_miost_date.strftime("%Y-%m-%d"),
                TIME_STEP= config.saveoutput_time_step.total_seconds()/(24*3600),
                NSTEPS_NC= int((24*3600)//config.saveoutput_time_step.total_seconds()),
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
                    name=config.name_experiment,
                    dict_obs= dict_obs,
                    subsampling= config.subsampling,
                    noise=config.sigma_R
                    ),
                
                    ]
            
            )
        
        # RUN MIOST
        miost.run_miost(config_miost)
        
        # READ OUTPUTS AND REFORMAT
        ds = xr.open_mfdataset(os.path.join(config.tmp_DA_path,'_ms_analysis*.nc'),
                               combine='by_coords')
        if config.miost_geo3ss6d and config.miost_geo3ls:              
            ssh = ds['Hss'] + ds['Hls']
        elif config.miost_geo3ss6d:
            ssh = ds['Hss']
        else:
            ssh = ds['Hls']
            
        # SAVE OUTPUTS 
        # Set the saving temporal window
        if miost_first_window:
            write_date_min = init_miost_date
            write_date_max = init_miost_date + config.miost_window_size/2 + config.miost_window_output/2
        elif miost_last_window:
            write_date_min = middle_miost_date - config.miost_window_output/2
            write_date_max = final_miost_date
        else:
            write_date_min = middle_miost_date - config.miost_window_output/2
            write_date_max = middle_miost_date + config.miost_window_output/2
            
        State_tmp = State.copy(free=True)
        date = init_miost_date
        i = 0
        while date<=final_miost_date:
            if (date >= write_date_min) & (date <= write_date_max) :
                
                if config.miost_window_overlap and not miost_first_window and\
                        date<=middle_miost_date:
                    # weight coefficients
                    W1 = max((middle_miost_date - date)
                             / (config.miost_window_output/2), 0)
                    W2 = min((date - write_date_min)
                             / (config.miost_window_output/2), 1)
                    # Read previous output at this timestamp
                    ds1 = State.load_output(date)
                    ssh1 = ds1[config.name_mod_var[State.get_indsave()]].data
                    ds1.close()
                    del ds1
                    # Update state
                    State_tmp.setvar(W1*ssh1 + W2*ssh[i].values,ind=0)
                else:
                    State_tmp.setvar(ssh[i].values,ind=0)
                    
                # Save output
                if config.saveoutputs:
                    if flag_mdt:
                        State_tmp.save_output(date,mdt=mdt)
                    else:
                        State_tmp.save_output(date)
                
            date += config.saveoutput_time_step
            i += 1
        
        if config.miost_window_overlap:
            window_lag = config.miost_window_output/2
        else:
            window_lag = config.miost_window_output

        if miost_first_window:
            middle_miost_date = config.init_date + config.miost_window_size/2 + window_lag
            miost_first_window = False
        else:
            middle_miost_date += window_lag
        
        ds.close()
        del ds
        
        cmd = 'rm ' + os.path.join(config.tmp_DA_path,'_ms_analysis*.nc')
        os.system(cmd)
        
        time1 = datetime.now()
        print('Loop from',init_miost_date.strftime("%Y-%m-%d"),'to',
              final_miost_date.strftime("%Y-%m-%d : in"),time1-time0,'seconds')
         

def Inv_harm(config,State,dict_obs=None):
    
    if config.detrend:
        from . import obs
        obs.detrend_obs(dict_obs)
    
    time = []
    ssh = np.array([])
    for date in dict_obs:
        time.append((date - config.init_date).total_seconds())
        path = dict_obs[date]['obs_name'][0]
        sat = dict_obs[date]['satellite'][0]
        ds = xr.open_dataset(path).squeeze()
        ssh = np.append(ssh,ds[sat.name_obs_var[0]].values)
        
    time = np.asarray(time)
    ssh = np.asarray(ssh)

    # Harmonic analysis
    nt,ny,nx = ssh.shape
    G = np.empty((nt,2))
    eta1 = np.empty((2, ny,nx))
    
    w = config.w_igws[0]
    G[:,0] = np.cos(w*time)
    G[:,1] = np.sin(w*time)
    
    M = np.dot(np.linalg.inv(np.dot(G.T,G)) , G.T)
    
    for ix in range(nx):
        for iy in range(ny):
            eta1[:, iy, ix] = np.dot(M, ssh[:,iy,ix])
    
    
    
    # Save outputs
    State0 = State.copy(free=True)
    date = config.init_date
    while date <= config.final_date:
        t = (date - config.init_date).total_seconds()
        ssh0 = eta1[0,:,:] * np.cos(w*t) + eta1[1,:,:] * np.sin(w*t)
        State0.setvar(ssh0,ind=0)
        State0.save_output(date)
        date += config.saveoutput_time_step
    
    
def solve_pcg(G, comp_Qinv, obs_invnois2,eps_rest=1.e-7,niter=800):

    comp_CFAC = comp_Qinv**-0.5
    
    comp_rest = G.T.dot(obs_invnois2) * comp_CFAC

    comp_p = 1*comp_rest
    comp_x = np.zeros_like((comp_rest))

    rest = np.inner(comp_rest, comp_rest)
    rest0 = +rest

    itr = int(-1)
    rest2 = 0
    while ((rest / rest0 > eps_rest) & (itr < niter)):
        itr += 1
        ###########################################
        # Compute A*p
        ###########################################
        cvec = G.T.dot(G.dot(comp_p * comp_CFAC) * obs_invnois2)
        comp_Ap =  comp_p * comp_Qinv * comp_CFAC**2 + cvec * comp_CFAC

        if itr >0: rest = +rest2

        tmp = np.inner(comp_p, comp_Ap)
        alphak = rest / tmp

        ###########################################
        # New state
        ###########################################
        comp_x += alphak * comp_p

        # ###########################################
        # New direction of descent
        ###########################################
        rest2 = np.inner(comp_rest - alphak * comp_Ap, comp_rest - alphak * comp_Ap)
        betak = rest2 / rest

        # Loop updates
        comp_p *= betak
        comp_p += comp_rest - alphak * comp_Ap 
        comp_rest += -alphak * comp_Ap      
            
    return comp_x * comp_CFAC

    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
