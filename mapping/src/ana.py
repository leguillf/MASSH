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



def ana(config, State, Model, dict_obs=None, *args, **kwargs):
    """
    NAME
        ana

    DESCRIPTION
        Main function calling subfunctions for specific Data Assimilation algorithms
    """
    
    if config.name_analysis is None: 
        return ana_forward(config,State,Model)
    
    elif config.name_analysis=='BFN':
        return ana_bfn(config,State,Model,dict_obs)
    
    elif config.name_analysis=='4Dvar':
        return ana_4Dvar(config,State,Model,dict_obs)
    
    elif config.name_analysis=='MIOST':
        return ana_miost(config,State,dict_obs)
    
    elif config.name_analysis=='HARM':
        return ana_harm(config,State,dict_obs)
    
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

         

        
def ana_bfn(config,State,Model,dict_obs=None, *args, **kwargs):
    """
    NAME
        ana_bfn

    DESCRIPTION
        Perform a BFN experiment on altimetric data and save the results on output directory
    
    """
    
    from . import tools_bfn as bfn

    # Flag initialization
    if config.name_init=='restart':
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
    middle_bfn_date = State.present_date
    # In the case of Nudging (i.e. bfn_max_iteration=1), set the bfn window length as the entire experimental time period
    if config.bfn_max_iteration==1:
        new_bfn_window_size = config.final_date - config.init_date
    else:
        new_bfn_window_size = config.bfn_window_size
    name_init = ""
    # propagation timestep
    one_time_step = config.bfn_propation_timestep
        
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
        # Boundary condition
        if config.flag_use_boundary_conditions:
            periods = (final_bfn_date-init_bfn_date).total_seconds()//\
                one_time_step.total_seconds() + 1
            timestamps = pd.date_range(init_bfn_date,
                                       final_bfn_date,
                                       periods=periods
                                      )
                
            bc_field, bc_weight = grid.boundary_conditions(config.file_boundary_conditions,
                                                           config.lenght_bc,
                                                           config.name_var_bc,
                                                           timestamps,
                                                           State.lon,
                                                           State.lat,
                                                           config.flag_plot,
                                                           bfn_obj.sponge,
                                                           mask=+State.mask)
            # Add mdt if provided 
            if config.add_mdt_bc:
                try: 
                    mdt = +State.mdt[np.newaxis,:,:]
                    mdt[np.isnan(mdt)] = 0
                    bc_field += mdt
                except : print('Warning : unable to add MDT to boundary field')
        else:
            bc_field = bc_weight = bc_field_t = None

        # Use last state from the last forward loop as initialization
        if not (bfn_first_window or restart) and config.bfn_window_overlap and os.path.exists(name_init):
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
        # Set the saving temporal windowx
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
        #State_current = State.free()
        # Save first timestep
        if present_date==config.init_date:
            iforward = 0
            name_save = config.name_exp_save + '_' + str(0).zfill(5) + '.nc'
            current_file = os.path.join(config.tmp_DA_path,'BFN_forth_' + name_save)
            State.load(current_file)
            if config.saveoutputs:
                State.save_output(present_date,mdt=Model.mdt)
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
                    State.load(current_file)
                    
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
                        ssh1 = ds[config.name_mod_var[State.get_indsave()]].data
                        ds.close()
                        del ds
                        # Update state
                        ssh2 = State.getvar(ind=State.get_indsave())
                        State.setvar(W1*ssh1+W2*ssh2,ind=0)
                
                    # Save output
                    if config.saveoutputs:
                        State.save_output(present_date,mdt=Model.mdt)
        
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
        if restart:
            restart = False
    print()

    return



def ana_4Dvar(config,State,Model,dict_obs=None) :
    
    '''
    Run a 4Dvar analysis
    '''

    print('\n*** Observation operator ***\n')
    from .tools_4Dvar import Obsopt
    H = Obsopt(config,State,dict_obs,Model)
    
    print('\n*** Wavelet reduced basis ***\n')
    
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
    
    # Initial State 
    if config.path_init_4Dvar is None:
        Xopt = var.Xb*0
    else:
        # Read previous minimum 
        with open(config.path_init_4Dvar, 'rb') as f:
            print('Read previous minimum:',config.path_init_4Dvar)
            Xopt = pickle.load(f)
    
    # Restart mode
    if config.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(config.tmp_DA_path,'X_it-*')))
        if len(tmp_files)>0:
            with open(tmp_files[-1], 'rb') as f:
                print('Restart at:',tmp_files[-1])
                Xopt = pickle.load(f)
        else:
            Xopt = var.Xb*0
            
    ###################
    # Minimization    #
    ###################
    print('\n*** Minimization ***\n')
    J0 = var.cost(Xopt)
    g0 = var.grad(Xopt)
    projg0 = np.max(np.abs(g0))
    print('J0=',"{:e}".format(J0))
    print('projg0=',"{:e}".format(projg0))
    
    
    def callback(XX,projg0=projg0):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H%M%S")
        with open(os.path.join(config.tmp_DA_path,'X_it-'+current_time+'.pic'),'wb') as f:
            pickle.dump(XX,f)
    
    options = {'disp': True, 'maxiter': config.maxiter}
    if config.gtol is not None:
        options['gtol'] = config.gtol*projg0
        
    res = opt.minimize(var.cost,Xopt,
                    method='L-BFGS-B',
                    jac=var.grad,
                    options=options,
                    callback=callback)

    print ('\nIs the minimization successful? {}'.format(res.success))
    print ('\nFinal cost function value: {}'.format(res.fun))
    print ('\nNumber of iterations: {}'.format(res.nit))

    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    if config.prec:
        Xa = var.Xb + B.sqr(res.x)
    else:
        Xa = var.Xb + res.x
        
    # Save minimum for next experiments
    with open(os.path.join(config.tmp_DA_path,'Xini.pic'), 'wb') as f:
        pickle.dump(Xa,f)
    # Init
    State0 = State.free()
    State0.params = np.zeros((Model.nparams,))
    date = config.init_date

    # Forward propagation
    while date<config.final_date:
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
                    
                State0.save_output(date,mdt=Model.mdt)

        
    del State, State0, res, Xa, dict_obs,J0,g0,projg0,B,R
    gc.collect()
    print()
    
    
def ana_4Dvar_BM_IT(config,State,Model,dict_obs=None) :
    '''
    Run a 4Dvar analysis
    '''
    from .tools_4Dvar import Obsopt, Cov, Variational_BM_IT
    from .tools_reduced_basis import RedBasis_BM

    print('\n*** Observation operator ***\n')
    H = Obsopt(config,State,dict_obs,Model)
    
    print('\n*** Wavelet reduced basis ***\n')
    comp_bm = RedBasis_BM(config,State)
    q_bm = comp_bm.set_basis(return_q=True)
    
    ###################
    # Variational    #
    ###################
    print('\n*** Variational ***\n')
    
    # Covariance matrix
    R = Cov(config.sigma_R)
    _sigma_B = np.zeros((Model.Model_IT.nParams)) 
    # error on He
    _sigma_B[Model.Model_IT.sliceHe] = config.sigma_B_He 
    # error on OBCs
    
    if hasattr(config.sigma_B_bc, '__len__'):
        # Specific value for each tidal component
        if len(config.sigma_B_bc) == len(config.w_igws):
            N_one_component_x = Model.Model_IT.nbcx//len(config.w_igws)
            N_one_component_y = Model.Model_IT.nbcy//len(config.w_igws)
            for i,_sigma in enumerate(config.sigma_B_bc):
                _sigma_B[Model.Model_IT.slicehbcx][
                    i*N_one_component_x:(i+1)*N_one_component_x] = _sigma
                _sigma_B[Model.Model_IT.slicehbcy][
                    i*N_one_component_y:(i+1)*N_one_component_y] = _sigma
        else:
            _sigma_B[Model.Model_IT.slicehbcx] = config.sigma_B_bc[0]
            _sigma_B[Model.Model_IT.slicehbcy] = config.sigma_B_bc[0]
    else:
        _sigma_B[Model.Model_IT.slicehbcx] = config.sigma_B_bc
        _sigma_B[Model.Model_IT.slicehbcy] = config.sigma_B_bc
            
    if np.any(State.mask) and config.Nmodes==1:
        
        # Reduced apriori for land pixels
        land_coeff_bcx = np.ones(Model.Model_IT.shapehbcx)
        land_coeff_bcy = np.ones(Model.Model_IT.shapehbcy)
        # Loop over OBC coordinates 
        for j,x in enumerate(Model.Model_IT.bcx): # South/North
            # look for the closest grid pixel 
            jS = np.argmin(np.abs(State.X[0,:]-x)) # South
            jN = np.argmin(np.abs(State.X[-1,:]-x)) # North
            if State.mask[0,jS]:
                land_coeff_bcx[:,0,:,:,:,j] = config.facB_bc_coast
            if State.mask[-1,jN]:
                land_coeff_bcx[:,1,:,:,:,j] = config.facB_bc_coast
        for i,y in enumerate(Model.Model_IT.bcy): # West/East
            # look for the closest grid pixel 
            iW = np.argmin(np.abs(State.Y[:,0]-y)) # West
            iE = np.argmin(np.abs(State.Y[:,-1]-y)) # East
            if State.mask[iW,0]:
                land_coeff_bcy[:,0,:,:,:,i] = config.facB_bc_coast
            if State.mask[iE,-1]:
                land_coeff_bcy[:,1,:,:,:,i] = config.facB_bc_coast
        
        _sigma_B[Model.Model_IT.slicehbcx] *= land_coeff_bcx.ravel()
        _sigma_B[Model.Model_IT.slicehbcy] *= land_coeff_bcy.ravel()
        
        # Loop over He coordinates 
        land_coeff_He = np.ones(Model.Model_IT.shapeHe)
        p = -1
        for i,y in enumerate(Model.Model_IT.Hey):
            for j,x in enumerate(Model.Model_IT.Hex):
                p += 1
                dist = np.sqrt((State.Y-y)**2+(State.X-x)**2)
                i0,j0 = np.unravel_index(dist.argmin(),dist.shape)
                if State.mask[i0,j0]:
                    land_coeff_He[:,p] = config.facB_He_coast
        _sigma_B[Model.Model_IT.sliceHe] *= land_coeff_He.ravel()

    B = Cov(np.concatenate((np.sqrt(q_bm),_sigma_B)))
    
    # backgroud state 
    Xb = np.zeros((comp_bm.nwave+Model.Model_IT.nParams,))
    
    # Cost and Grad functions
    var = Variational_BM_IT(
        config=config, M=Model, H=H, State=State, B=B, R=R, comp=comp_bm, Xb=Xb)
    
    # Initial State 
    if config.path_init_4Dvar is None:
        Xopt = var.Xb*0
    else:
        # Read previous minimum 
        with open(config.path_init_4Dvar, 'rb') as f:
            print('Read previous minimum:',config.path_init_4Dvar)
            Xopt = pickle.load(f)
    
    # Restart mode
    if config.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(config.tmp_DA_path,'X_it-*')))
        if len(tmp_files)>0:
            with open(tmp_files[-1], 'rb') as f:
                print('Restart at:',tmp_files[-1])
                Xopt = pickle.load(f)
        else:
            Xopt = var.Xb*0
            
    ###################
    # Minimization    #
    ###################
    print('\n*** Minimization ***\n')
    J0 = var.cost(Xopt)
    g0 = var.grad(Xopt)
    projg0 = np.max(np.abs(g0))
    print('J0=',"{:e}".format(J0))
    print('projg0=',"{:e}".format(projg0))
    
    
    def callback(XX,projg0=projg0):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H%M%S")
        with open(os.path.join(config.tmp_DA_path,'X_it-'+current_time+'.pic'),'wb') as f:
            pickle.dump(XX,f)
    
    options = {'disp': True, 'maxiter': config.maxiter}
    if config.gtol is not None:
        options['gtol'] = config.gtol*projg0
        
    res = opt.minimize(var.cost,Xopt,
                    method='L-BFGS-B',
                    jac=var.grad,
                    options=options,
                    callback=callback)

    print ('\nIs the minimization successful? {}'.format(res.success))
    print ('\nFinal cost function value: {}'.format(res.fun))
    print ('\nNumber of iterations: {}'.format(res.nit))

    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    if config.prec:
        Xa = var.Xb + B.sqr(res.x)
    else:
        Xa = var.Xb + res.x
    
    Xbm = Xa[:comp_bm.nwave]
    Xit = Xa[comp_bm.nwave:]
        
    # Save minimum for next experiments
    with open(os.path.join(config.tmp_DA_path,'Xini.pic'), 'wb') as f:
        pickle.dump(Xa,f)
        
    # init
    State0 = State.free()
    coords = [State.lon.flatten(),State.lat.flatten(),0]
    ssh0 = comp_bm.operg(coords=coords,coords_name=var.coords_name, coordtype='reg', 
                        compute_geta=True,eta=Xbm,mode=None,
                        save_wave_basis=var.save_wave_basis).reshape(
                            (State.ny,State.nx))
    State0.setvar(ssh0,ind=0)
    date = config.init_date
    State0.save_output(date,mdt=Model.Model_BM.mdt)
    # Forward propagation
    i = 0
    while date<config.final_date:
        t = (date - config.init_date).total_seconds()
        # Forward
        for j in range(config.checkpoint):
            Model.step(t+j*Model.dt,State0,Xit,nstep=1, 
                       Hbc=var.bc_field[i],Wbc=var.bc_weight)
            date += timedelta(seconds=config.dtmodel)
            if (((date - config.init_date).total_seconds()
                 /config.saveoutput_time_step.total_seconds())%1 == 0)\
                & (date>config.init_date) & (date<=config.final_date) :
                State0.save_output(date,mdt=Model.Model_BM.mdt)
        # add Flux
        coords = [State.lon.flatten(),State.lat.flatten(),t/3600/24]
        F = comp_bm.operg(coords=coords,coords_name=var.coords_name, coordtype='reg', 
                          compute_geta=True,eta=Xbm,mode='flux',
                          save_wave_basis=config.save_wave_basis).reshape((State.ny,State.nx))  
        _var = State0.getvar(ind=0)
        State0.setvar(_var + config.checkpoint * config.dtmodel/(3600*24) * F,ind=0)
        i += 1
    
    del State, State0, res, Xa, dict_obs,J0,g0,projg0,B,R
    gc.collect()
    print()
        
    
def ana_4Dvar_QG_init(config,State,Model,dict_obs=None) :
    '''
    Run a 4Dvar analysis
    '''
    from .tools_4Dvar import Obsopt
    
    print('\n*** create and initialize State ***\n')
    if config.path_init_4Dvar is not None :
        with xr.open_dataset(config.path_init_4Dvar) as ds :
            ssh_b = np.copy(ds.ssh)
            State.setvar(ssh_b,State.get_indobs())
    
    print('\n*** Observation operator ***\n')
    H = Obsopt(config,State,dict_obs,Model)
    
    print('\n*** 4Dvar analysis ***\n')
    
    date_ini = config.init_date # current date
    window_length = config.window_length # size of the assimilation window
    
    i = 0
    while date_ini<config.final_date :
        date_end = date_ini + window_length # date at end of the assimilation window
        print(f'\n*** window {i} from {date_ini} to {date_end} ***\n')
        if i==0 :
            first_assimilation = True
        else :
            first_assimilation = False
        # minimization
        res = window_4D(config,State,Model,dict_obs=dict_obs,H=H,date_ini=date_ini,\
                        date_final=date_end,first=first_assimilation)
        ssh0 = res.reshape(State.var.ssh.shape) # initial ssh field for the window from the 4Dvar analysis
        State.setvar(ssh0,State.get_indobs())
        
        if config.flag_plot>=1:
            State.plot(date_ini)
        
        print('\n*** Saving trajectory ***\n')
        
        date_save = date_ini
        date_end_save = date_ini + config.window_save
        if config.window_overlap:
            date_end_save += config.window_save//2
        
        
        dt_save = config.saveoutput_time_step # time step for saving
        n_save = int((date_end_save - date_ini).total_seconds()/dt_save.total_seconds())+1 # number of save            
        n_step = int(dt_save.total_seconds()/config.dtmodel) # number of model step between two save
        
        # Boundary conditiond
        if config.flag_use_boundary_conditions:
            timestamps = pd.date_range(date_ini,
                                       date_end_save,
                                       periods=n_save
                                      )
            bc_field, bc_weight = grid.boundary_conditions(config.file_boundary_conditions,
                                                           config.lenght_bc,
                                                           config.name_var_bc,
                                                           timestamps,
                                                           State.lon,
                                                           State.lat,
                                                           config.flag_plot,
                                                           mask=np.copy(State.mask))

        else: 
            bc_field = np.array([None,]*n_save)
            bc_weight = None
        
        
        
        t = 0
        State_save = State.copy()
        while date_save<=date_end_save:
            
            ssh_current = State.getvar(ind=State.get_indsave())
            if config.window_overlap and not first_assimilation and\
                date_save<=(date_ini+config.window_save//2):
                    ds1 = State.load_output(date_save)
                    ssh_prev = ds1[config.name_mod_var[State.get_indsave()]].data
                    ds1.close()
                    del ds1
                    # weight coefficients
                    W1 = max((date_ini + config.window_save//2 - date_save)  /\
                             (config.window_save//2), 0)
                    W2 = min((date_save - date_ini) / (config.window_save//2), 1)
                    # Update state
                    State_save.setvar(W1*ssh_prev+W2*ssh_current,ind=0)
            else:
                State_save.setvar(ssh_current,ind=0)

            if date_save==date_ini + config.window_save:
                ssh_next = State.getvar(ind=State.get_indsave())
                
            State_save.save_output(date_save)
            Model.step(State,n_step,bc_field[t],bc_weight) # run forward the model
            date_save += dt_save # update time
            t += 1
        
        # Time update for next window
        State.setvar(ssh_next,ind=0)
        date_ini += config.window_save
        i += 1
                    
        
        print('\n*** final analysed state ***\n')
        

def window_4D(config,State,Model,dict_obs=None,H=None,date_ini=None,date_final=None,first=False) :
    '''
    run one assimilation on the window considered which start at time date_ini
    '''
    from .tools_4Dvar import Obsopt, Cov
    from .tools_4Dvar import Variational_QG_init as Variational
    
    # create obsopt if not filled out
    if H==None :
        H = Obsopt(State,dict_obs,Model)
    
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
        config=config, M=Model, H=H, State=State, B=B, R=R, Xb=Xb, 
        tmp_DA_path=config.tmp_DA_path, date_ini=date_ini, date_final=date_final)
   
    Xopt = np.copy(Xb)
    if config.prec :
        Xopt = np.zeros_like(Xb)
    
    J0 = var.cost(Xopt)
    g0 = var.grad(Xopt)
    projg0 = np.max(np.abs(g0))
        
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
    Xout = res.x
    if config.prec :
        Xout = B.sqr(Xout) + Xb
    
    return Xout
    
    


def ana_miost(config,State,dict_obs=None):
    
    
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
        
        PHYS_COMP=[
        
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
                Hvarname= 'Hss'),
        
        
            comp_geo3.Comp_geo3ls(
                facnls= 3., #factor for large-scale wavelet spacing
                facnlt= 3.,
                tdec_lw= 25.,
                std_lw= 0.04,
                lambda_lw= 970, #768.05127036
                file_aux= config.file_aux,
                filec_aux= config.filec_aux,
                write= True,
                Hvarname= 'Hls'),
            ],
        
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
        ssh = ds['Hss'] + ds['Hls']
        
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
            
        State_tmp = State.free()
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
         

def ana_harm(config,State,dict_obs=None):
    
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
    State0 = State.free()
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

    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
