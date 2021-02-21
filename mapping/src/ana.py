#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""
from datetime import timedelta
import sys
import numpy as np 
import calendar
import os
import matplotlib.pylab as plt
import pickle
from datetime import datetime
import scipy.optimize as opt

from . import mod,tools,grid



def ana(config, State, Model, dict_obs=None, *args, **kwargs):
    """
    NAME
        ana

    DESCRIPTION
        Main function calling subfunctions for specific Data Assimilation algorithms
    """
    
    if config.name_analysis=='BFN':
        return ana_bfn(config,State,Model,dict_obs)
    elif config.name_analysis=='4Dvar':
        return ana_4Dvar(config,State,Model,dict_obs)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        
        
    
def ana_bfn(config,State,Model,dict_obs=None, *args, **kwargs):
    """
    NAME
        ana_bfn

    DESCRIPTION
        Perform a BFN experiment on altimetric data and save the results on output directory
    
    """
    
    from . import tools_bfn as bfn

    # Flag initialization
    bfn_first_window = True
    bfn_last_window = False
    if dict_obs is None:
        call_obs_func = True
        from . import obs
    else:
        call_obs_func = False
    # BFN middle date initialization
    middle_bfn_date = config.init_date
    # In the case of Nudging (i.e. bfn_max_iteration=1), set the bfn window length as the entire period of the experience
    if config.bfn_max_iteration==1:
        print('bfn_max_iteration has been set to 1 --> '
              + 'Only one forth loop will be done on the entiere period of the experience')
        new_bfn_window_size = config.final_date - config.init_date
    else:
        new_bfn_window_size = config.bfn_window_size
    name_init = ""
    
    # Main time loop
    while (middle_bfn_date <= config.final_date) and not bfn_last_window :
        print('\n*** BFN window ***')
        #############
        # 1. SET-UP #
        #############
        # BFN period
        init_bfn_date = max(config.init_date, middle_bfn_date - new_bfn_window_size/2)
        init_bfn_date += timedelta(seconds=(init_bfn_date - config.init_date).total_seconds()\
                         / config.bfn_propation_timestep.total_seconds()%1)
        middle_bfn_date = max(middle_bfn_date, config.init_date + new_bfn_window_size/2)
        if ((middle_bfn_date + new_bfn_window_size/2) >= config.final_date):
            bfn_last_window = True
            final_bfn_date = config.final_date
        else:
            final_bfn_date = init_bfn_date + new_bfn_window_size
        print('\nfrom ', init_bfn_date, ' to ', final_bfn_date)
        # propagation timestep
        one_time_step = config.bfn_propation_timestep
        if bfn_first_window:
            present_date_forward0 = init_bfn_date
        
        ########################
        # 2. Create BFN object #
        ########################
        print('\n* Initialize BFN *')
        bfn_obj = bfn.bfn(
            config,init_bfn_date,final_bfn_date,one_time_step,State.lon,State.lat)
        
        ######################################
        # 3. BOUNDARY AND INITIAL CONDITIONS #
        ######################################
        print("\n* Boundary and initial conditions *")
        # Boundary condition
        if config.flag_use_boundary_conditions:
            timestamps = np.arange(calendar.timegm(init_bfn_date.timetuple()),
                                   calendar.timegm(final_bfn_date.timetuple()),
                                   one_time_step.total_seconds())

            bc_field, bc_weight = grid.boundary_conditions(config.file_boundary_conditions,
                                                           config.lenght_bc,
                                                           config.name_var_bc,
                                                           timestamps,
                                                           State.lon,
                                                           State.lat,
                                                           config.flag_plot,
                                                           bfn_obj.sponge)

        else:
            bc_field = bc_weight = bc_field_t = None
        # Initial condition
        if bfn_first_window:
            # Use previous state as initialization
            init_file = config.path_save + config.name_exp_save\
                        + '_y' + str(init_bfn_date.year)\
                        + 'm' + str(init_bfn_date.month).zfill(2)\
                        + 'd' + str(init_bfn_date.day).zfill(2)\
                        + 'h' + str(init_bfn_date.hour).zfill(2)\
                        + str(init_bfn_date.minute).zfill(2) + '.nc'
            if not os.path.isfile(init_file) :
                restart = False
            else:
                restart = True
                print(init_file, " is used as initialization")
                State.load(init_file)
                
        elif config.bfn_window_overlap:
            # Use last state from the last forward loop as initialization
            print(name_init)
            State.load(name_init)
        
        ###################
        # 4. Observations #
        ###################
        # Selection
        print('\n* Select observations *')
        
        if call_obs_func:
            print('Calling obs_all_observationcheck function...')
            dict_obs_it = obs.obs(config)
            bfn_obj.select_obs(dict_obs_it)
            dict_obs_it.clear()
            del dict_obs_it
        else:
            bfn_obj.select_obs(dict_obs)

        # Projection
        print('\n* Project observations *')
        bfn_obj.do_projections()

        ###############
        # 5. BFN LOOP #
        ###############
        err_bfn0 = 0
        err_bfn1 = 0
        bfn_iter = 0
        Nold_t = None

        while bfn_iter==0 or\
             (bfn_iter < config.bfn_max_iteration
              and abs(err_bfn0-err_bfn1)/err_bfn1 > config.bfn_criterion):
            if bfn_iter>0:
                present_date_forward0 = init_bfn_date

            err_bfn0 = err_bfn1
            bfn_iter += 1
            if bfn_iter == config.bfn_max_iteration:
                print('\nMaximum number of iterations achieved ('
                      + str(config.bfn_max_iteration)
                      + ') --> last Forth loop !!')

            ###################
            # 5.1. FORTH LOOP #
            ###################
            print("\n* Forward loop " + str(bfn_iter) + " *")
            while present_date_forward0 < final_bfn_date :
                
                # Retrieve corresponding time index for the forward loop
                iforward = int((present_date_forward0 - init_bfn_date)/one_time_step)

                # Get BC field
                if bc_field is not None:
                    bc_field_t = bc_field[iforward]

                # Model propagation and apply Nudging
                Model.step(State,
                       one_time_step.total_seconds(),
                       Hbc=bc_field_t,
                       Wbc=bc_weight,
                       Nudging_term=Nold_t)
                
                # Time increment 
                present_date_forward = present_date_forward0 + one_time_step
                
                # Compute Nudging term (for next time step)
                N_t = bfn_obj.compute_nudging_term(
                        present_date_forward, State
                        )

                # Update model parameter
                bfn_obj.update_parameter(State, Nold_t, N_t, bc_weight, way=1)

                # Save current state                 
                name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                filename_forward = config.tmp_DA_path + '/BFN_forth_' + name_save
                State.save(filename_forward,present_date_forward)
                if config.save_bfn_trajectory:
                    filename_traj = config.path_save + 'BFN_' + str(middle_bfn_date)[:10]\
                               + '_forth_' + str(bfn_iter) + '/' + name_save

                    if not os.path.exists(os.path.dirname(filename_traj)):
                        os.makedirs(os.path.dirname(filename_traj))
                    State.save(filename_traj,present_date_forward)
                            
                            
                # Time update
                present_date_forward0 = present_date_forward
                Nold_t = N_t
            
            # Init file for next loop
            name_init = filename_forward
            
            # Plot for debugging
            if config.flag_plot > 0:
                ssh = State.getvar(0)
                pv = State.getvar(1)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((10, 5)))
                p1 = ax1.pcolormesh(State.lon, State.lat, pv, shading='auto')
                p2 = ax2.pcolormesh(State.lon, State.lat, ssh, shading='auto')
                plt.colorbar(p1, ax=ax1)
                plt.colorbar(p2, ax=ax2)
                ax1.set_title('Potential vorticity')
                ax2.set_title('SSH')
                plt.show()

            ##################
            # 5.2. BACK LOOP #
            ##################
            if  bfn_iter < config.bfn_max_iteration:
                print("\n* Backward loop " + str(bfn_iter) + " *")
                present_date_backward0 = final_bfn_date
    
                while present_date_backward0 > init_bfn_date :
                    
                    # Retrieve corresponding time index for the backward loop
                    ibackward = int((present_date_backward0 - init_bfn_date)/one_time_step)

                    # Get BC field
                    if bc_field is not None:
                        bc_field_t = bc_field[ibackward-1]

                    # Propagate the state by nudging the model vorticity towards the 2D observations
                    Model.step(State,
                       -one_time_step.total_seconds(),
                       Hbc=bc_field_t,
                       Wbc=bc_weight,
                       Nudging_term=Nold_t)
                    
                    # Time increment
                    present_date_backward = present_date_backward0 - one_time_step

                    # Nudging term (next time step)
                    N_t = bfn_obj.compute_nudging_term(
                            present_date_backward,
                            State
                            )

                    # Update model parameter
                    bfn_obj.update_parameter(State, Nold_t, N_t, bc_weight, way=-1)
                    
                    # Save current state            
                    name_save = config.name_exp_save + '_' + str(ibackward).zfill(5) + '.nc'
                    filename_backward = config.tmp_DA_path + '/BFN_back_' + name_save
                    State.save(filename_backward,present_date_backward)
                    if config.save_bfn_trajectory:
                        filename_traj = config.path_save + 'BFN_' + str(middle_bfn_date)[:10]\
                                   + '_back_' + str(bfn_iter) + '/' + name_save

                        if not os.path.exists(os.path.dirname(filename_traj)):
                            os.makedirs(os.path.dirname(filename_traj))
                        State.save(filename_traj,present_date_backward)

                    # Time update
                    present_date_backward0 = present_date_backward
                    Nold_t = N_t

                if config.flag_plot > 0:
                    SSH = State.getvar(0)
                    PV = State.getvar(1)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((10, 5)))
                    p1 = ax1.pcolormesh(State.lon, State.lat, PV,shading='auto')
                    p2 = ax2.pcolormesh(State.lon, State.lat, SSH,shading='auto')
                    plt.colorbar(p1, ax=ax1)
                    plt.colorbar(p2, ax=ax2)
                    ax1.set_title('Potential vorticity')
                    ax2.set_title('SSH')
                    plt.show()

            #########################
            # 5.3. CONVERGENCE TEST #
            #########################
            if bfn_iter < config.bfn_max_iteration:
                print('\n* Convergence test *')
                err_bfn1 = bfn_obj.convergence(
                                        path_forth=config.tmp_DA_path + '/BFN_forth_',
                                        path_back=config.tmp_DA_path + '/BFN_back_'
                                        )


        print("\n* End of the BFN loop after " + str(bfn_iter) + " iterations *")

        #####################
        # 6. SAVING OUTPUTS #
        #####################
        print('\n* Saving last forth loop as outputs for the following dates : *')
        # Set the saving temporal window
        if config.bfn_max_iteration==1:
            write_date_min = init_bfn_date
            write_date_max = final_bfn_date
        elif bfn_first_window:
            write_date_min = init_bfn_date
            write_date_max = init_bfn_date + new_bfn_window_size/2 + config.bfn_window_output/2
        elif bfn_last_window:
            write_date_min = middle_bfn_date - config.bfn_window_output/2
            write_date_max = final_bfn_date
        else:
            write_date_min = middle_bfn_date - config.bfn_window_output/2
            write_date_max = middle_bfn_date + config.bfn_window_output/2
        # Write outputs in the saving temporal window

        present_date = init_bfn_date
        State_current = State.free()
        State_previous = State.free()
        while present_date < final_bfn_date :
            present_date += one_time_step
            if (present_date > write_date_min) & (present_date <= write_date_max) :
                # Save output every *saveoutput_time_step*
                if (((present_date - config.init_date).total_seconds()
                   /config.saveoutput_time_step.total_seconds())%1 == 0)\
                   & (present_date>config.init_date)\
                   & (present_date<=config.final_date) :
                    print(present_date, end=' / ')                
                    # Read current converged state
                    iforward = int((present_date - init_bfn_date)/one_time_step) - 1
                    name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                    current_file = config.tmp_DA_path + '/BFN_forth_' + name_save
                    State_current.load(current_file)
                    
                    # Smooth with previous BFN window
                    if config.bfn_window_overlap and (not bfn_first_window or restart):
                        # Read previous output at this timestamp
                        previous_file = config.path_save + config.name_exp_save\
                                        + '_y'+str(present_date.year)\
                                        + 'm'+str(present_date.month).zfill(2)\
                                        + 'd'+str(present_date.day).zfill(2)\
                                        + 'h'+str(present_date.hour).zfill(2)\
                                        + str(present_date.minute).zfill(2) + \
                                            '.nc'
                        if os.path.isfile(previous_file):
                            State_previous.load(previous_file)
                            
                            # weight coefficients
                            W1 = max((middle_bfn_date - present_date)
                                     / (config.bfn_window_output/2), 0)
                            W2 = min((present_date - write_date_min)
                                     / (config.bfn_window_output/2), 1)
                            State_current.scalar(W1)
                            State_previous.scalar(W2)
                            State_current.Sum(State_previous)
            
                    # Save output
                    if config.saveoutputs:
                        State_current.save(date=present_date)
        print()
        ########################
        # 8. PARAMETERS UPDATE #
        ########################
        if config.bfn_window_overlap:
            window_lag = config.bfn_window_output/2
        else:
            window_lag = config.bfn_window_output

        if bfn_first_window:
            middle_bfn_date = config.init_date + new_bfn_window_size/2 + window_lag
            bfn_first_window = False
        else:
            middle_bfn_date += window_lag

    
    
def ana_4Dvar(config,State,Model,dict_obs=None, *args, **kwargs):
    
    from .tools_4Dvar import Obsopt, Cov, Variational

    #################
    # 1. Obs op     #
    #################
    print('\n*** Obs op ***\n')
    H = Obsopt(State.lon.size,dict_obs,Model.timestamps,Model.dt)
    
    ###################
    # 2. Variationnal #
    ###################
    print('\n*** Variational ***\n')
    # Covariance matrixes
    if None in [config.sigma_R,config.sigma_B_He,config.sigma_B_bc]:          
            # Least squares
            B = None
            R = Cov(1)
    else:
        _sigma_B = np.zeros((Model.nParams)) 
        _sigma_B[Model.sliceHe] = config.sigma_B_He
        _sigma_B[Model.slicehbcx] = config.sigma_B_bc
        _sigma_B[Model.slicehbcy] = config.sigma_B_bc
        B = Cov(_sigma_B)
        R = Cov(config.sigma_R)
    # backgroud state 
    Xb = np.zeros((Model.nParams,))
    # Cost and Grad functions
    var = Variational(
        M=Model, H=H, State=State, B=B, R=R, Xb=Xb, 
        tmp_DA_path=config.tmp_DA_path, checkpoint=config.checkpoint,
        prec=config.prec)
    # Initial State
    if config.path_init_4Dvar is None:
        Xopt = np.zeros_like(var.Xb)
    else:
        # Read previous minimum
        with open(config.path_init_4Dvar, 'rb') as f:
            print('Read previous minimum:',config.path_init_4Dvar)
            Xopt = pickle.load(f)

    
    ###################
    # 3. Minimization #
    ###################
    print('\n*** Minimization ***\n')
    
    J0 = var.cost(Xopt)
    g0 = var.grad(Xopt)
    projg0 = np.max(np.abs(g0))
    print('J0=',"{:e}".format(J0))
    print('projg0',"{:e}".format(projg0))
    
    
    def callback(XX):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H%M%S")
        with open(config.tmp_DA_path + '/X_it-'+current_time+'.pic','wb') as f:
            pickle.dump(XX,f)

    res = opt.minimize(var.cost,Xopt,
                    method='L-BFGS-B',
                    jac=var.grad,
                    options={'disp': True, 'gtol': config.gtol*projg0, 'maxiter': config.maxiter},
                    callback=callback)

    print ('\nIs the minimization successful? {}'.format(res.success))
    print ('\nFinal cost function value: {}'.format(res.fun))
    print ('\nNumber of iterations: {}'.format(res.nit))

    ########################
    # 4. Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    if config.prec:
        Xa = var.Xb + B.sqr.dot(res.x)
    else:
        Xa = var.Xb + res.x
        
    # Save minimum for next experiments
    with open(config.tmp_DA_path + '/Xini.pic', 'wb') as f:
        pickle.dump(Xa,f)
    
    # Steady initial state
    State0 = State.free()
    State0.save(date=config.init_date)
    for i in range(Model.nt-1):
        t = Model.T[i] # seconds
        date = Model.timestamps[i+1] # date
        
        Model.step(t,State0,Xa)
        
        if (((date - config.init_date).total_seconds()
             /config.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>config.init_date) & (date<=config.final_date) :
            print(date, end=' / ')    
            # Save State
            State0.save(date=date)
    print()
        

    
   
    


