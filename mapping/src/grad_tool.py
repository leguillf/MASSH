#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:49:17 2021

@author: renamatt
"""

import numpy as np

class grad_op :
    
    def __init__(self,State) :
        self.shape = State.var[0].shape # shape of the grid
        self.nx = self.shape[0] # number of grid points in the x-direction
        self.ny = self.shape[1] # number of grid points in the y-direction
        self.dx = State.dx # mean spatial step in x-direction
        self.dy = State.dy # mean spatial step in y-direction
        
    def gradx(self,X) :
        '''
        X is a 2D array of shape (nx,ny)
        the function returns the gradient in the x-direction of X
        '''
        Xsup = np.roll(X,-1,axis=0)
        Xout = self.dx*(Xsup-X)
        Xout[self.nx-1,:] = np.zeros(self.ny) 
        return Xout

    def grady(self,X) :
        '''
        X is a 2D array of shape (nx,ny)
        the function returns the gradient in the y-direction of X
        '''
        Xsup = np.roll(X,-1,axis=1)
        Xout = self.dy*(Xsup-X)
        Xout[:,self.ny-1] = np.zeros(self.nx)
        return Xout
    
    def grad(self,X) :
        '''
        X is a one dimensionnal array with a length of nx*ny
        returns a 1D array of size 2*nx*ny, the gradient of X
        '''
        Xvar = X.reshape(self.shape)
        gx = self.gradx(Xvar)
        gy = self.grady(Xvar)
        Xout = np.concatenate((gx.ravel(),gy.ravel()))
        return Xout
    
    def T_gradx(self,gx) :
        '''
        transposed operator of gradx
        gx is a 2D array of shape (nx,ny)
        returns a 2D array of shape (nx,ny)
        '''
        gx_inf = np.roll(gx,1,axis=0)
        Xout = gx_inf-gx
        Xout[0,:] = -gx[0,:]
        Xout[self.nx-1,:] = gx[self.nx-2,:]
        Xout = self.dx*Xout
        return Xout

    def T_grady(self,gy) :
        '''
        transposed operator of grady
        gy is a 2D array of shape (nx,ny)
        returns a 2D array of shape (nx,ny)
        '''
        gy_inf = np.roll(gy,1,axis=1)
        Xout = gy_inf-gy
        Xout[:,0] = -gy[:,0]
        Xout[:,self.ny-1] = gy[:,self.ny-2]
        Xout = self.dy*Xout
        return Xout
    
    def T_grad(self,g) :
        '''
        transposed operator of grad
        g is a 1D array with a length of 2*nx*ny
        returns a 1D array with a length of nx*ny
        '''
        n_vec = self.nx*self.ny
        gx, gy = g[:n_vec].reshape(self.shape), g[n_vec:].reshape(self.shape)
        Xout = self.T_gradx(gx) + self.T_grady(gy)
        Xout = Xout.ravel()
        return Xout