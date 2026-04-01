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



class ConvergenceReached(Exception):
    pass

class CrazyGradient(Exception):
    pass


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
    
    if config.INV.super=='INV_4DVAR':
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
        
        # Sustained convergence: override scipy stopping criteria
        convergence_nit = getattr(config.INV, 'convergence_nit', None)
        ftol_threshold = options.get('ftol', None)
        gtol_threshold = options.get('gtol', None)
        if convergence_nit is not None:
            options['ftol'] = 0
            options['gtol'] = 0

        gradient_max_norm = getattr(config.INV, 'gradient_max_norm', None)
        
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
                self.convergence_count = 0
                self.last_x = None
                # For crazy gradient recovery
                self.best_cost = float(J0)
                self.best_x = np.array(Xopt).copy()
                self.best_grad_norm = float(np.max(np.abs(G0)))

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

                # Track best state
                grad_norm = np.max(np.abs(grad))
                if gradient_max_norm is not None and np.isfinite(cost) and np.isfinite(grad_norm) and grad_norm < gradient_max_norm:
                    if float(cost) < self.best_cost:
                        self.best_cost = float(cost)
                        self.best_x = np.array(x).copy()
                        self.best_grad_norm = grad_norm

                # Check for crazy gradient
                if gradient_max_norm is not None and ((not np.isfinite(cost)) or (not np.isfinite(grad_norm)) or grad_norm > gradient_max_norm):
                    print(f"\nCrazy gradient detected (cost={cost}, grad_norm={grad_norm}). Will restart from best state.")
                    raise CrazyGradient()

                # Check sustained convergence
                if convergence_nit is not None:
                    converged = False
                    if ftol_threshold is not None and abs(float(ftol)) <= ftol_threshold:
                        converged = True
                    if gtol_threshold is not None and np.max(np.abs(grad)) <= gtol_threshold:
                        converged = True
                    if converged:
                        self.convergence_count += 1
                    else:
                        self.convergence_count = 0
                    self.last_x = np.array(x).copy()
                    if self.convergence_count >= convergence_nit:
                        print(f'\nConvergence criteria met for {convergence_nit} consecutive iterations. Stopping.')
                        raise ConvergenceReached()

                return cost

            def jac(self, x, *args):
                return self.cache['grad']
        
        wrapper = Wrapper()
        max_retries = getattr(config.INV, 'max_retries', 3)
        retry = 0
        while True:
            try:
                res = opt.minimize(wrapper, Xopt,
                                method=config.INV.opt_method,
                                jac=wrapper.jac,
                                options=options,
                                callback=callback)
                print ('\nIs the minimization successful? {}'.format(res.success))
                print ('\nFinal cost function value: {}'.format(res.fun))
                print ('\nNumber of iterations: {}'.format(res.nit))
                Xres = res.x
                break
            except ConvergenceReached:
                print(f'\nNumber of iterations: {wrapper.it - 1}')
                Xres = wrapper.last_x
                break
            except CrazyGradient:
                retry += 1
                if retry > max_retries:
                    print(f"Maximum retries ({max_retries}) reached. Stopping minimization.")
                    Xres = wrapper.best_x
                    break
                print(f"Restarting minimization from best state (retry {retry}/{max_retries})...")
                Xopt = wrapper.best_x
                wrapper = Wrapper()
        
        # Save minimization trajectory
        if config.INV.save_minimization:
            ds = xr.Dataset({'cost':(('i'),np.array(wrapper.J_list)),
                             'grad':(('i'),np.array(wrapper.G_list))
                             })
            ds.to_netcdf(os.path.join(path_save_control_vectors,'minimization_trajectory.nc'))
            ds.close()
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
