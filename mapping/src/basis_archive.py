class Basis_hbc: 

    def __init__(self,config, State):

        ##################
        ### - COMMON - ###
        ##################

        # Grid specs
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.ny = State.ny
        self.nx = State.nx
        self.lonS = State.lon[0,:]
        self.lonN = State.lon[-1,:]
        self.latE = State.lat[:,0]
        self.latW = State.lat[:,-1]
        self.km2deg =1./110 # Kilometer to deg factor 

        # Name of controlled parameters
        self.name_params = config.BASIS.name_params 

        # Basis reduction factor
        self.facns = config.BASIS.facns # Factor for gaussian spacing in space
        self.facnlt = config.BASIS.facnlt # Factor for gaussian spacing in time

        # Tidal frequencies 
        self.Nwaves = config.BASIS.Nwaves # Number of tidal components

        # Time dependancy 
        self.time_dependant = config.BASIS.time_dependant

        ##########################################
        ### - HEIGHT BOUNDARY CONDITIONS hbc - ###
        ##########################################

        self.D_bc = config.BASIS.D_bc # Space scale of gaussian decomposition for hbc (in km)
        self.T_bc = config.BASIS.T_bc # Time scale of gaussian decomposition for hbc (in days)

        # Number of angles (computed from the normal of the border) of incoming waves
        if config.BASIS.Ntheta>0: 
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°

        self.sigma_B_bc = config.BASIS.sigma_B_bc # Covariance sigma for hbc parameter

        self.window = mywindow

        # JIT
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

    def set_basis(self,time,return_q=False):

        """
        Set the basis for the controlled parameters of the model and calculate reduced basis functions.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        return_q : bool, optional
            If True, returns the covariance matrix Q and the background vector array Xb, by default False.

        Returns:
        --------
        tuple of np.ndarray
            If return_q is True, returns a tuple containing:
                - Xb : np.ndarray
                    Background vector array Xb.
                - Q : np.ndarray or None
                    Covariance matrix Q.
        
        """
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time 
        self.vect_time = jnp.eye(time.size)

        self.Gxy = {} # Dictionary containing gaussian basis elements for each parameters. 
        self.shape_params = {} # Dictionary containing the shapes in the reduced space of each of the parameters.
        self.shape_params_phys = {} # Dictionary containing the shapes in the physical space of each of the parameters.
        
        #############################################
        ### SETTING UP THE REDUCED BASIS ELEMENTS ###
        #############################################

        # - In Time - #
        if self.time_dependant:
            self.set_bc_gauss_t(time, TIME_MIN, TIME_MAX) 

        # - In Space - # 
        for name in self.name_params : 

            # - X height boundary conditions - #
            if name == "hbcx":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_bc_gauss_hbcx(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            
            # - Y height boundary conditions - #
            if name == "hbcy": 
                self.shape_params["hbcE"], self.shape_params["hbcW"], self.shape_params_phys["hbcE"], self.shape_params_phys["hbcW"] = self.set_bc_gauss_hbcy(LAT_MIN, LAT_MAX)

        ############################################
        ### REDUCED BASIS INFORMATION ATTRIBUTES ###
        ############################################

        # Dictionary with the number of parameters in reduced space 
        self.n_params = {}
        for param in self.shape_params.keys():
            if self.shape_params[param] == []:
                self.n_params[param] = 0
            else :
                self.n_params[param] = np.prod(self.shape_params[param])

        # Dictionary with the number of parameters in pysical space
        self.n_params_phys = dict(zip(self.shape_params_phys.keys(), map(np.prod, self.shape_params_phys.values()))) 
        # Total number of parameters in the reduced space
        self.nbasis = sum(self.n_params.values()) 
        # Total number of parameters in the physical space
        self.nphys = sum(self.n_params_phys.values()) 
        # Total number of parameters in the physical space (including time dimension)
        self.nphystot = 0 
        for param in self.n_params_phys.keys():
            self.nphystot += self.n_params_phys[param]*time.size
        # Setting up slice information for parameters 
        interval = 0 ; interval_phys = 0 
        self.slice_params = {} # Dictionary with the slices of parameters in the reduced space
        self.slice_params_phys = {} # Dictionary with the slices of parameters in the physical space
        for name in self.shape_params.keys():
            self.slice_params[name]=slice(interval,interval+self.n_params[name])
            self.slice_params_phys[name]=slice(interval_phys,interval_phys+self.n_params_phys[name])
            interval += self.n_params[name]; interval_phys += self.n_params_phys[name]
        # PRINTING REDUCED ORDER : #     
        print(f'reduced order: {self.nphystot} --> {self.nbasis}\nreduced factor: {int(self.nphystot/self.nbasis)}')

        #########################################
        ### COMPUTING THE COVARIANCE MATRIX Q ###
        #########################################        

        if return_q :
            if self.sigma_B_bc is not None:
                Q = np.zeros((self.nbasis,)) # Initializing
                for name in self.slice_params.keys() :

                    if hasattr(self.sigma_B_bc,'__len__'):
                        if len(self.sigma_B_bc)==self.Nwaves:
                            # Different background values for each frequency
                            nw = self.nbc//self.Nwaves
                            for iw in range(self.Nwaves):
                                slicew = slice(iw*nw,(iw+1)*nw)
                                Q[self.slice_params[name]][slicew]=self.sigma_B_bc[iw]
                        else:
                            # Not the right number of frequency prescribed in the config file 
                            # --> we use only the first one
                            Q[self.slice_params[name]]=self.sigma_B_bc[0]
                    else:
                        # Q[self.slice_params[name]]=self.sigma_B_bc
                        Q[self.slice_params[name]]=self.sigma_B_bc/(self.facnlt*self.facns)

            else:
                Q = None

            Xb = np.zeros_like(Q)

            return Xb, Q
    
    def set_bc_gauss_hbcx(self, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):

        """
        Set the height boundary conditions hbcx parameter recuced basis elements for both the South and North boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcS : list
                    Shape of the South boundary hbcx parameter in the reduced space.
                - shapehbcN : list
                    Shape of the North boundary hbcx parameter in the reduced space.
                - shapehbcS_phys : list
                    Shape of the South boundary hbcx parameter in the physical space.
                - shapehbcN_phys : list
                    Shape of the North boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcx parameters in the reduced space.
        """

        ###############################
        ###   - SPACE DIMENSION -   ###
        ###############################

        # - SOUTH - # 
        # Ensemble of reduced basis longitudes
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # - NORTH - #
        # Ensemble of reduced basis longitudes
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 

        # Saving gaussian reduced basis elements 
        self.Gxy["hbcS"] = sparse.CSR.fromdense(jnp.array(bc_S_gauss.T)) # For South boundary 
        self.Gxy["hbcN"] = sparse.CSR.fromdense(jnp.array(bc_N_gauss.T)) # For South boundary

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################

        # - Shapes of the hbcy parameters in the reduced space.

        if self.time_dependant : # the parameters include the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,     # - Number of basis timesteps
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,          # - Number of basis timesteps
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 

        # - Shapes of the hbcy parameters in the physical space.
        shapehbcS_phys = shapehbcN_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.nx]         # - Number of gridpoints along x axis
        
        print('nbcx:',np.prod(shapehbcS)+np.prod(shapehbcN))

        return shapehbcS, shapehbcN, shapehbcS_phys, shapehbcN_phys

    def set_bc_gauss_hbcy(self,LAT_MIN, LAT_MAX): 

        """
        Set the height boundary conditions hbcy parameter recuced basis elements for both the East and West boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcE : list
                    Shape of the East boundary hbcx parameter in the reduced space.
                - shapehbcW : list
                    Shape of the West boundary hbcx parameter in the reduced space.
                - shapehbcE_phys : list
                    Shape of the East boundary hbcx parameter in the physical space.
                - shapehbcW_phys : list
                    Shape of the West boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcy parameters in the reduced space.
        """
        
        #########################################
        ###   - COMPUTING SPACE DIMENSION -   ###
        #########################################

        # Ensemble of reduced basis latitudes (common for each boundaries)
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)

        # - EAST - #
        # Computing reduced basis elements gaussian supports 
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # - WEST - # 
        # Computing reduced basis elements gaussian supports 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # Gaussian reduced basis elements
        self.Gxy["hbcE"] = sparse.CSR.fromdense(jnp.array(bc_E_gauss.T)) # For East boundary 
        self.Gxy["hbcW"] = sparse.CSR.fromdense(jnp.array(bc_W_gauss.T)) # For West boundary 

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################
        
        # Shapes of the hbcx parameters in the reduced space.
        if self.time_dependant : # the parameters include the time dependency

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements

        # Shapes of the hbcx parameters in the physical space.
        shapehbcE_phys = shapehbcW_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.ny]         # - Number of gridpoints along x axis

        print('nbcy:',np.prod(shapehbcE)+np.prod(shapehbcW))

        return shapehbcE, shapehbcW, shapehbcE_phys, shapehbcW_phys
    
    def set_bc_gauss_t(self,time, TIME_MIN, TIME_MAX):
        # Ensemble of reduced basis timesteps
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        # bc_t_gauss = np.zeros((time.size,ENST_bc.size))
        # for i,time0 in enumerate(ENST_bc):
        #     iobs = np.where(abs(time-time0) < self.T_bc)
        #     bc_t_gauss[iobs,i] = mywindow(abs(time-time0)[iobs]/self.T_bc)

        # self.bc_t_gauss = bc_t_gauss

        self.ENST_bc = ENST_bc

        Gt = np.zeros((time.size,self.ENST_bc.size))

        for i,t in enumerate(time) :
            for it in range(len(self.ENST_bc)):
                dt = t - self.ENST_bc[it]
                if abs(dt) < self.T_bc:
                    fact = self.window(dt / self.T_bc) 
                    if fact!=0:   
                        Gt[i,it] = fact
        
        self.Gt = sparse.csr_fromdense(jnp.array(Gt).T)

    def get_bc_t_gauss_value(self,t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt @ self.vect_time[idt[0]] # Get corresponding value

    def _operg(self,t,X):

        """
        Perform the basis projection operation for a given time and parameter vector.

        This method projects the given parameter vector X from reduced basis onto the model grid, at the provided time t. The results can be stored in the provided State object.
                
        operg : | REDUCED SPACE >>>>>> PHYSICAL SPACE | (Model Grid) 

        Parameters:
        ----------
        t : float
            The time at which the projection is performed.
        X : ndarray
            The parameter vector to be projected.
        State : object, optional
            State object to store the parameters after projection. If not provided, the method returns the projected vector onto physical space.

        Returns:
        -------
        phi : ndarray
            The projected parameter vector if State is not provided. Otherwise, updates the State object in place.

        """

        ##############################
        ###   - INITIALIZATION -   ###
        ##############################

        # Time gaussian function
        if self.time_dependant:
            _Gt = self.get_bc_t_gauss_value(t) 

        # Variable to return  
        phi = jnp.zeros((self.nphys,))

        ##########################################
        ###   - BASIS PROJECTION OPERATION -   ###
        ##########################################

        for name in self.slice_params_phys.keys():

            _X = X[self.slice_params[name]]
            _X = _X.reshape(self.shape_params[name])

            if self.time_dependant:
                _X = (_Gt[None,None,None,:, None]*_X).sum(axis=3, keepdims=False)

            _X_t = _X.T

            Gxy_X_t = sparse.csr_matmat(self.Gxy[name],_X_t.reshape(_X_t.shape[0],-1))

            Gxy_X_t = Gxy_X_t.reshape((Gxy_X_t.shape[0],)+_X_t.shape[1:])

            Gxy_X = Gxy_X_t.T

            phi = phi.at[self.slice_params_phys[name]].set(Gxy_X.flatten())#.reshape(self.shape_params_phys[name]))
        
        return phi

    def _operg_reduced(self, t, phi_2d):
        """
        Project a 2D physical space field back to the reduced space.

        Parameters:
            t: Current time
            phi_2d: 2D physical space field to project back.

        Returns:
            Reduced space representation (1D vector).
        """

        # Define a wrapper function for _operg that computes the forward projection
        def operg_func(X):
            return self._operg_jit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward projection
        _, vjp_func = jax.vjp(operg_func, jnp.zeros(self.nbasis))  # Provide a zero vector matching the reduced space shape

        # Use the vjp_func to compute the reduced space projection
        X_reduced, = vjp_func(phi_2d)

        return X_reduced

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Update State
        if State is not None:
            for name in self.name_params:
                # - Height boundary conditions hbcx - #
                if name == "hbcx" : 
                    State.params['hbcx'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                            np.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                # - Height boundary conditions hbcy - #
                elif name == "hbcy" : 
                    State.params['hbcy'] = np.concatenate((np.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1),
                                                            np.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1)),axis=1)
            # State.params[self.name_mod_var] = phi
        else:
            return phi
    
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        # if adState.params[self.name_mod_var] is None:
        #     adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        # Getting the parameters 
        # if phi is not None: # If provided through phi ndarray argument 
        #     for name in self.slice_params_phys.keys():
        #         param[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])

        adparams = np.zeros((self.nphys))
        if adState is not None: # If provided through adState object argument 
            for name in self.name_params:
                if name == "hbcx" : 
                    # adparams["hbcS"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcS"])
                    # adparams["hbcN"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcN"])
                    adparams[self.slice_params_phys["hbcS"]] = adState.params[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcN"]] = adState.params[name][:,1,:,:,:].flatten()
                elif name == "hbcy" : 
                    # adparams["hbcE"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcE"])
                    # adparams["hbcW"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcW"])
                    adparams[self.slice_params_phys["hbcE"]] = adState.params[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcW"]] = adState.params[name][:,1,:,:,:].flatten()
                # else :
                #     param[name] = adState.params[name].reshape(self.shape_params_phys[name])
        # adparams = adparams.flatten()
        # adparams = adState.getparams(self.name_params,vect=True)

        adX = self._operg_reduced_jit(t, adparams)
        
        for _param in self.name_params : 
            adState.params[_param] *= 0.
        
        return adX