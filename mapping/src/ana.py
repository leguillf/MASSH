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
from datetime import datetime,timedelta
import scipy.optimize as opt
import gc
import pandas as pd

from . import grid



def ana(config, State, Model, dict_obs=None, *args, **kwargs):
    """
    NAME
        ana

    DESCRIPTION
        Main function calling subfunctions for specific Data Assimilation algorithms
    """
    
    if config.name_analysis is None: 
        return ana_forward(config,State,Model)
    if config.name_analysis=='BFN':
        return ana_bfn(config,State,Model,dict_obs)
    elif config.name_analysis=='4Dvar':
        return ana_4Dvar(config,State,Model,dict_obs)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        

def ana_forward(config,State,Model):
    
    present_date = config.init_date
    State.save_output(present_date)
    nstep = int(config.saveoutput_time_step.total_seconds()//Model.dt)
    while present_date < config.final_date :
        print(present_date)
        # Propagation
        Model.step(State,nstep)
        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        # Save
        if config.saveoutputs:
            State.save_output(present_date)
        if config.flag_plot>0:
            State.plot(present_date)
        
        
    return


def ana_4Dvar_QG(config,State,Model,dict_obs=None) :
    '''
    Run a 4Dvar analysis
    '''
    from .tools_4Dvar import Obsopt
    
    print('\n*** create and initialize State ***\n')
    State_ana = State.free()
    if config.path_init_4Dvar is not None :
        import xarray as xr
        with xr.open_dataset(config.path_init_4Dvar) as ds :
            ssh_b = np.copy(ds.ssh)
            State_ana.setvar(ssh_b,State_ana.get_indobs())
    
    print('\n*** Observation operator ***\n')
    H = Obsopt(State,dict_obs,Model,tmp_DA_path=config.tmp_DA_path)
    
    print('\n*** 4Dvar analysis ***\n')
    
     # date
    init_date = config.init_date
    final_date = config.final_date
    
    date = init_date # current date
    dt_window = config.window_time_step # size of the assimilation window
    
    n_window = int((final_date-init_date).total_seconds()/dt_window.total_seconds())
    
    n_iter = 1 + 2 * (n_window - 1) # number of minimization to perform
    
    for i in range(n_iter) :
        date_end = date + dt_window # date at end of the assimilation window
        print(f'\n*** window {i}, initial date = {date.year}:{date.month}:{date.day}\
              final date = {date_end.year}:{date_end.month}:{date_end.day} ***\n')
        if i==0 :
            first_assimilation = True
        else :
            first_assimilation = False
        # minimization
        res = window_4D(config,State_ana,Model,dict_obs=dict_obs,H=H,date_ini=date,\
                        date_final=date_end,first=first_assimilation)
        ssh0 = res.reshape(State.var.ssh.shape) # initial ssh field for the window from the 4Dvar analysis
        State_ana.setvar(ssh0,State_ana.get_indobs())
        
        if config.saveoutputs :
            
            print('\n*** Saving trajectory ***\n')
            
            date_save = date
            dt_save = config.saveoutput_time_step # time step for saving
            
            last_window = date_end == config.final_date
            if last_window :
                n_save = int((date_end - date).total_seconds()/dt_save.total_seconds()) # number of save
            else :
                n_save = int((dt_window//2).total_seconds()/dt_save.total_seconds()) # number of save
            
            n_step = int(dt_save.total_seconds()/config.dtmodel) # number of model step between two save
            
            for i in range(n_save) :
                State_ana.save(date=date_save) # save state
                Model.step(State_ana,n_step) # run forward the model
                date_save += dt_save # update time
                if last_window :
                    State_ana.save(date=date_save)
        
        else :
            # model propagation until next assimilation window
            Model.step(State_ana,nstep=n_window//2)
        
        date += dt_window//2 # time update
        print('\n*** final analysed state ***\n')
        
        
        
    
    
    

def window_4D(config,State,Model,dict_obs=None,H=None,date_ini=None,date_final=None,first=False) :
    '''
    run one assimilation on the window considered which start at time date_ini
    '''
    from .tools_4Dvar import Obsopt, Cov
    from .tools_4Dvar import Variational_QG as Variational
    
    # create obsopt if not filled out
    if H==None :
        H = Obsopt(State,dict_obs,Model,tmp_DA_path=config.tmp_DA_path)
    
    # Covariance matrixes
    if None in [config.sigma_R,config.sigma_B]:          
            # Least squares
            B = None
            R = Cov(1.)
    else:
        sigma_B = config.sigma_B
        sigma_R = config.sigma_R
        R = Cov(sigma_R)
        if first :
            B = Cov(1.)
        else :
            B = Cov(sigma_B)
    # background state
    Xb = State.getvar(State.get_indobs()).ravel() # background term for analysis
    
    
    # Cost and Grad functions
    var = Variational(
        M=Model, H=H, State=State, B=B, R=R, Xb=Xb, 
        tmp_DA_path=config.tmp_DA_path, date_ini=date_ini, date_final=date_final,
        prec=config.prec)
   
    Xopt = np.copy(Xb)
    if config.prec :
        Xopt = np.zeros_like(Xb)
    J0 = var.cost(Xopt)
    g0 = var.grad(Xopt)
    projg0 = np.max(np.abs(g0))
    
    State_callback = State.free()
    def callback(XX) :
        '''
        function called at each iteration of the minimization process
        '''
        var_ssh = XX.reshape(State_callback.var[0].shape)
        State_callback.setvar(var_ssh,0)
        State_callback.plot()
    
    res = opt.minimize(var.cost,Xopt,
                    method='L-BFGS-B',
                    jac=var.grad,
                    options={'disp': True, 'gtol': config.gtol*projg0, 'maxiter': config.maxiter},callback=callback)
    Xout = res.x
    if config.prec :
        Xout = B.sqr(Xout) + Xb
    
    return Xout
       
    
    
    
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
        new_bfn_window_size = config.final_date - config.init_date
    else:
        new_bfn_window_size = config.bfn_window_size
    name_init = ""
    
    # Main time loop
    while (middle_bfn_date <= config.final_date) and not bfn_last_window :
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
        # propagation timestep
        one_time_step = config.bfn_propation_timestep
        if bfn_first_window:
            present_date_forward0 = init_bfn_date
        
        ########################
        # 2. Create BFN object #
        ########################
        bfn_obj = bfn.bfn(
            config,init_bfn_date,final_bfn_date,one_time_step,State)
        
        ######################################
        # 3. BOUNDARY AND INITIAL CONDITIONS #
        ######################################
        # Boundary condition
        if config.flag_use_boundary_conditions:
            periods = (final_bfn_date-init_bfn_date).total_seconds()//\
                one_time_step.total_seconds() + 1
            timestamps = pd.date_range(init_bfn_date,
                                       final_bfn_date,
                                       periods=periods
                                      )
                                   #  np.arange(calendar.timegm(init_bfn_date.timetuple()),
                                   # calendar.timegm(final_bfn_date.timetuple())+\
                                   #     one_time_step.total_seconds(),
                                   # one_time_step.total_seconds())
                

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

        # Use last state from the last forward loop as initialization
        if not bfn_first_window and config.bfn_window_overlap:
            State.load(name_init)
        
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
              (bfn_iter < config.bfn_max_iteration
              and abs(err_bfn0-err_bfn1)/err_bfn1 > config.bfn_criterion):

            if bfn_iter>0:
                present_date_forward0 = init_bfn_date

            err_bfn0 = err_bfn1
            bfn_iter += 1
            
            ###################
            # 5.1. FORTH LOOP #
            ###################

            # Save state at first timestep              
            name_save = config.name_exp_save + '_' + str(0).zfill(5) + '.nc'
            filename_forward = os.path.join(config.tmp_DA_path,'BFN_forth_' + name_save)
            State.save(filename_forward)
            
            while present_date_forward0 < final_bfn_date :
                
                # Retrieve corresponding time index for the forward loop
                iforward0 = int((present_date_forward0 - init_bfn_date)/one_time_step)
                
                # Get BC field
                if bc_field is not None:
                    bc_field_t = bc_field[iforward0]

                # Model propagation and apply Nudging
                Model.step_nudging(State,
                       one_time_step.total_seconds(),
                       Hbc=bc_field_t,
                       Wbc=bc_weight,
                       Nudging_term=Nold_t)

                # Time increment 
                present_date_forward = present_date_forward0 + one_time_step
                iforward = iforward0 + 1
                
                # Compute Nudging term (for next time step)
                N_t = bfn_obj.compute_nudging_term(
                        present_date_forward, State
                        )
                
                # Update model parameter
                bfn_obj.update_parameter(State, Nold_t, N_t, bc_weight, way=1)

                # Save current state                 
                name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                filename_forward = os.path.join(config.tmp_DA_path,'BFN_forth_' + name_save)
                State.save(filename_forward)
                if config.save_bfn_trajectory:
                    filename_traj = os.path.join(config.path_save,'BFN_' + str(middle_bfn_date)[:10]\
                               + '_forth_' + str(bfn_iter),name_save)

                    if not os.path.exists(os.path.dirname(filename_traj)):
                        os.makedirs(os.path.dirname(filename_traj))
                    State.save(filename_traj)
                            
                            
                # Time update
                present_date_forward0 = present_date_forward
                Nold_t = N_t
            
            # Init file for next loop
            name_init = filename_forward
            
            # Plot for debugging
            if config.flag_plot > 0:
                State.plot(title=str(present_date_forward) + ': End of forward loop n°' + str(bfn_iter))

            ##################
            # 5.2. BACK LOOP #
            ##################
            if  bfn_iter < config.bfn_max_iteration:
                present_date_backward0 = final_bfn_date
                # Save state at first timestep          
                ibackward = int((present_date_backward0 - init_bfn_date)/one_time_step)
                name_save = config.name_exp_save + '_' + str(ibackward).zfill(5) + '.nc'
                filename_backward = os.path.join(config.tmp_DA_path,'BFN_back_' + name_save)
                State.save(filename_backward)
                
                while present_date_backward0 > init_bfn_date :
                    
                    # Retrieve corresponding time index for the backward loop
                    ibackward = int((present_date_backward0 - init_bfn_date)/one_time_step)

                    # Get BC field
                    if bc_field is not None:
                        bc_field_t = bc_field[ibackward]

                    # Propagate the state by nudging the model vorticity towards the 2D observations
                    Model.step_nudging(State,
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
                    name_save = config.name_exp_save + '_' + str(ibackward-1).zfill(5) + '.nc'
                    filename_backward = os.path.join(config.tmp_DA_path,'BFN_back_' + name_save)
                    State.save(filename_backward)
                    if config.save_bfn_trajectory:
                        filename_traj = os.path.join(config.path_save,'BFN_' + str(middle_bfn_date)[:10]\
                                   + '_back_' + str(bfn_iter),name_save)

                        if not os.path.exists(os.path.dirname(filename_traj)):
                            os.makedirs(os.path.dirname(filename_traj))
                        State.save(filename_traj)

                    # Time update
                    present_date_backward0 = present_date_backward
                    Nold_t = N_t

                if config.flag_plot > 0:
                    State.plot(title=str(present_date_backward) + ': End of backward loop n°' + str(bfn_iter))
            #########################
            # 5.3. CONVERGENCE TEST #
            #########################
            if bfn_iter < config.bfn_max_iteration:
                err_bfn1 = bfn_obj.convergence(
                                        path_forth=os.path.join(config.tmp_DA_path,'BFN_forth_*.nc'),
                                        path_back=os.path.join(config.tmp_DA_path,'BFN_back_*.nc')
                                        )
            
        time1 = datetime.now()
                
        print('Loop from',init_bfn_date.strftime("%Y-%m-%d"),'to',final_bfn_date.strftime("%Y-%m-%d :"),bfn_iter,'iterations in',time1-time0,'seconds')
        
        #####################
        # 6. SAVING OUTPUTS #
        #####################
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
        # Save first timestep
        if present_date==config.init_date:
            iforward = 0
            name_save = config.name_exp_save + '_' + str(0).zfill(5) + '.nc'
            current_file = os.path.join(config.tmp_DA_path,'BFN_forth_' + name_save)
            State_current.load(current_file)
            if config.saveoutputs:
                State_current.save_output(present_date)
        while present_date < final_bfn_date :
            present_date += one_time_step
            if (present_date > write_date_min) & (present_date <= write_date_max) :
                # Save output every *saveoutput_time_step*
                if (((present_date - config.init_date).total_seconds()
                   /config.saveoutput_time_step.total_seconds())%1 == 0)\
                   & (present_date>config.init_date)\
                   & (present_date<=config.final_date) :
                    # Read current converged state
                    iforward = int((present_date - init_bfn_date)/one_time_step) 
                    name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                    current_file = os.path.join(config.tmp_DA_path,'BFN_forth_' + name_save)
                    State_current.load(current_file)
                    
                    # Smooth with previous BFN window
                    if config.bfn_window_overlap and not bfn_first_window and\
                        present_date<=middle_bfn_date:
                        # weight coefficients
                        W1 = max((middle_bfn_date - present_date)
                                 / (config.bfn_window_output/2), 0)
                        W2 = min((present_date - write_date_min)
                                 / (config.bfn_window_output/2), 1)
                        # Read previous output at this timestamp
                        ds = State.load_output(present_date)
                        ssh1 = ds.ssh.data
                        ds.close()
                        del ds
                        # Update state
                        ssh2 = State_current.getvar(ind=State_current.get_indsave())
                        State_current.setvar(W1*ssh1+W2*ssh2,ind=0)
        
                    # Save output
                    if config.saveoutputs:
                        State_current.save_output(present_date)
        
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
    print()

    return
    
    
def ana_4Dvar(config,State,Model,dict_obs=None, *args, **kwargs):
    
    from .tools_4Dvar import Obsopt, Cov, Variational
            
    #################
    # 1. Obs op     #
    #################
    print('\n*** Obs op ***\n')
    H = Obsopt(State,dict_obs,Model,tmp_DA_path=config.tmp_DA_path)
    
    if config.detrend:
        print('\n*** Detrend obs ***\n')
        from . import obs
        obs.detrend_obs(dict_obs)
    
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
    # Tapering border pixel influence
    if config.flag_use_boundary_conditions:
        eps_bc = config.eps_bc
        dist_scale = config.lenght_bc # tapering window length scale
    else:
        eps_bc = 0
        dist_scale = None
        
    # Cost and Grad functions
    var = Variational(
        M=Model, H=H, State=State, B=B, R=R, Xb=Xb, 
        tmp_DA_path=config.tmp_DA_path, checkpoint=config.checkpoint,
        prec=config.prec,eps_bc=eps_bc,dist_scale=dist_scale)
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
    
    
    def callback(XX,projg0=projg0):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H%M%S")
        with open(os.path.join(config.tmp_DA_path,'X_it-'+current_time+'.pic'),'wb') as f:
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
        Xa = var.Xb + B.sqr(res.x)
    else:
        Xa = var.Xb + res.x
        
    # Save minimum for next experiments
    with open(os.path.join(config.tmp_DA_path,'Xini.pic'), 'wb') as f:
        pickle.dump(Xa,f)
    
    # Steady initial state
    State0 = State.free()
    State0.save_output(config.init_date)
    for i in range(Model.nt-1):
        t = Model.T[i] # seconds
        date = Model.timestamps[i+1] # date
        
        Model.step(t,State0,Xa)
        
        if (((date - config.init_date).total_seconds()
             /config.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>config.init_date) & (date<=config.final_date) :
            # Save State
            State0.save_output(date)
    
    del State, State0, res, Xa, dict_obs,J0,g0,projg0,B,R
    gc.collect()
    print()
        

    