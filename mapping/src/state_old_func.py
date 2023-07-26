
def ini_mask(self,config):
    
    """
    NAME
        ini_mask

    DESCRIPTION
        Read mask file, interpolate it to state grid, 
        and apply to state variable
    """
    
    self.mask = {} # mask of boolean for variable and parameter
    self.mask_int = {} # mask of int (0,1,NaN) for variable integration

    # Read mask
    if config.GRID.name_init_mask is not None and os.path.exists(config.GRID.name_init_mask):
        ds = xr.open_dataset(config.GRID.name_init_mask).squeeze()
        name_lon = config.GRID.name_var_mask['lon']
        name_lat = config.GRID.name_var_mask['lat']
        name_var = config.GRID.name_var_mask['var']
    else:
        ### mask by default ###
        mask = (np.isnan(self.lon) + np.isnan(self.lat)).astype(bool)
        self.set_mask(mask,config) # setting as mask of boolean for variable and parameter
        self.set_mask_int(mask,config) # setting a mask of int (0,1,NaN) for integration
        return

    # Convert longitudes
    if np.sign(ds[name_lon].data.min())==-1 and self.lon_unit=='0_360':
        ds = ds.assign_coords({name_lon:((name_lon, ds[name_lon].data % 360))})
    elif np.sign(ds[name_lon].data.min())==1 and self.lon_unit=='-180_180':
        ds = ds.assign_coords({name_lon:((name_lon, (ds[name_lon].data + 180) % 360 - 180))})
    ds = ds.sortby(ds[name_lon])    

    dlon =  np.nanmax(self.lon[:,1:] - self.lon[:,:-1])
    dlat =  np.nanmax(self.lat[1:,:] - self.lat[:-1,:])
    dlon +=  np.nanmax(ds[name_lon].data[1:] - ds[name_lon].data[:-1])
    dlat +=  np.nanmax(ds[name_lat].data[1:] - ds[name_lat].data[:-1])

    ds = ds.sel(
        {name_lon:slice(self.lon_min-dlon,self.lon_max+dlon),
            name_lat:slice(self.lat_min-dlat,self.lat_max+dlat)})

    lon = ds[name_lon].values
    lat = ds[name_lat].values
    var = ds[name_var]
            
    if len(var.shape)==2:
        mask = var
    elif len(var.shape)==3:
        mask = var[0,:,:]

    # Interpolate to state grid
    x_source_axis = pyinterp.Axis(lon, is_circle=False)
    y_source_axis = pyinterp.Axis(lat)
    x_target = self.lon.T
    y_target = self.lat.T
    grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, mask.T)
    mask_interp = pyinterp.bivariate(grid_source,
                                    x_target.flatten(),
                                    y_target.flatten(),
                                    interpolator = config.GRID.interp_method_mask,
                                    bounds_error=False).reshape(x_target.shape).T


    # Convert to bool if float type     
    if mask_interp.dtype!=bool : 
        mask = np.empty((self.ny,self.nx),dtype='bool')
        ind_mask = (np.isnan(mask_interp)) | (mask_interp==1) | (np.abs(mask_interp)>10)
        mask[ind_mask] = True
        mask[~ind_mask] = False
    else:
        mask = mask_interp.copy()

    mask += (np.isnan(self.lon) + np.isnan(self.lat)).astype(bool)

    self.set_mask(mask,config) # setting as mask of boolean for variable and parameter  
    self.set_mask_int(mask,config) # setting a mask of int (0,1,NaN) for integration 

def set_mask(self,mask,config): 
    """
    NAME
        set_mask

    ARGUMENT 
        mask : mask to set 

    DESCRIPTION
        Sets the mask attribute of the State object. The mask is a dictionnary containing the masks of all variables and parameters.  
    """
    for varname in config.MOD.name_var:
        if varname == "SSH" : 
            self.mask[config.MOD.name_var[varname]] = mask
        elif varname == "U" :
            self.mask[config.MOD.name_var[varname]] = (mask[:,:-1]*mask[:,1:]).astype('bool') # mask for "U" grid 
        elif varname == "V" : 
            self.mask[config.MOD.name_var[varname]] = (mask[:-1,:]*mask[1:,:]).astype('bool') # mask for "V" grid 
        elif varname == "SST" : 
            self.mask[config.MOD.name_var[varname]] = mask

    # setting mask for parameters
    if config.MOD.super == "MOD_SW1L_JAX":

        for param in config.MOD.name_params:

            # equivalent height He mask #
            if param == "He" : 
                self.mask["He"] = mask 

            # x boundary conditions mask #
            if param == "hbcx" :
                shapehbcx = [len(np.asarray(config.MOD.w_waves)), # tide frequencies
                            2, # South/North
                            2, # cos/sin
                            config.MOD.Ntheta*2+1, # Angles
                            self.nx # NX
                            ]
                mask_hbcx = np.zeros((shapehbcx),dtype="bool")
                mask_hbcx[:,0,:,:,mask[0,:]==True] = True # South frontier 
                mask_hbcx[:,1,:,:,mask[-1,:]==True] = True # North frontier 
                self.mask["hbcx"] = mask_hbcx

            # y boundary conditions mask #
            if param == "hbcy" :
                shapehbcy = [len(np.asarray(config.MOD.w_waves)), # tide frequencies
                            2, # West/East
                            2, # cos/sin
                            config.MOD.Ntheta*2+1, # Angles
                            self.ny # NY
                            ]
                mask_hbcy = np.zeros((shapehbcy),dtype="bool")
                mask_hbcy[:,0,:,:,mask[:,0]==True] = True # West frontier 
                mask_hbcy[:,1,:,:,mask[:,-1]==True] = True # East frontier 
                self.mask["hbcy"] = mask_hbcy

            # internal tide generation mask #
            if param == "itg" :
                self.mask["itg"] = np.repeat(np.expand_dims(mask,axis=0),repeats=2,axis=0) 

def _detect_coast(self,mask,axis):
    """
    NAME
        _detect_coast

    ARGUMENT 
        mask : mask of continents (N,n) shaped array
        axis : either "x" or "y"

    DESCRIPTION
        Detects coast between pixels. 

    RETURNS 
        (N-1,n) or (N,n-1) array with True if it is a coast (transisition continent - ocean) False otherwise. 
    """
    if axis == "x": 
        a1 = mask[:,1:]
        a2 = mask[:,:-1]
    elif axis == "y": 
        a1 = mask[1:,:]
        a2 = mask[:-1,:]
    p1 = np.logical_and(a1,np.invert(a2))
    p2 = np.logical_and(a2,np.invert(a1))
    return np.logical_or(p1,p2)

def set_mask_int(self,mask,config): 

    """
    NAME
        set_mask_int

    ARGUMENT 
        mask : mask to set 

    DESCRIPTION
        Sets the mask_int attribute of the State object. The mask is a dictionnary containing the masks of variables, represented by an int (0,1 or Nane). 
        For "SSH" the mask is : 
            - 1 if ocean 
            - 999 if continent (NaN value)
        For "U" and "V" the mask is : 
            - 1 if ocean 
            - 999 if continent (NaN value)
            - 0 if normal to the coast    
    """

    for varname in config.MOD.name_var:

        if varname == "SSH" : 
            mask_ssh = np.ones(mask.shape,dtype='float')
            mask_ssh[mask==True]=np.nan
            self.mask_int[config.MOD.name_var[varname]] = mask_ssh

        elif varname == "U" :
            mask_u = np.ones(mask[:,1:].shape,dtype='float')
            mask_u[np.logical_and(mask[:,1:],mask[:,:-1])]=np.nan
            mask_u[self._detect_coast(mask,"x")]=0
            self.mask_int[config.MOD.name_var[varname]] = mask_u

        elif varname == "V" : 
            mask_v = np.ones(mask[1:,:].shape,dtype='float')
            mask_v[np.logical_and(mask[1:,:],mask[:-1,:])]=np.nan
            mask_v[self._detect_coast(mask,"y")]=0
            self.mask_int[config.MOD.name_var[varname]] = mask_v
            
        elif varname == "SST" : 
            mask_sst = np.ones(mask.shape,dtype='float')
            mask_sst[mask==True]=0
            self.mask_int[config.MOD.name_var[varname]] = mask_sst



def save_output(self,date,name_var=None):
    
    filename = os.path.join(self.path_save,f'{self.name_exp_save}'\
        f'_y{date.year}'\
        f'm{str(date.month).zfill(2)}'\
        f'd{str(date.day).zfill(2)}'\
        f'h{str(date.hour).zfill(2)}'\
        f'm{str(date.minute).zfill(2)}.nc')
    
    coords = {}
    coords[self.name_time] = ((self.name_time), [pd.to_datetime(date)],)

    if self.geo_grid:
            coords[self.name_lon] = ((self.name_lon,), self.lon[0,:])
            coords[self.name_lat] = ((self.name_lat,), self.lat[:,0])
            dims = (self.name_time,self.name_lat,self.name_lon)
    else:
        coords[self.name_lon] = (('y','x',), self.lon)
        coords[self.name_lat] = (('y','x',), self.lat)
        dims = ('time','y','x')

    if name_var is None:
        name_var = self.var.keys()
        
    var = {}              
    for name in name_var:

        var_to_save = +self.var[name]

        # Apply Mask
        if self.mask is not None:
            var_to_save[self.mask] = np.nan
    
        if len(var_to_save.shape)==2:
            var_to_save = var_to_save[np.newaxis,:,:]
        
        var[name] = (dims, var_to_save)
        
    ds = xr.Dataset(var, coords=coords)
    ds.to_netcdf(filename,
                    encoding={'time': {'units': 'days since 1900-01-01'}},
                    unlimited_dims={'time':True})
    
    ds.close()
    del ds
    
    return 

def save(self,filename=None):
    """
    NAME
        save

    DESCRIPTION
        Save State in a netcdf file
        Args:
            filename (str): path (dir+name) of the netcdf file.
            date (datetime): present date
            """
    
    
    # Variables
    _namey = {}
    _namex = {}
    outvars = {}
    cy,cx = 1,1
    for name,var in self.var.items():
        y1,x1 = var.shape
        if y1 not in _namey:
            _namey[y1] = 'y'+str(cy)
            cy += 1
        if x1 not in _namex:
            _namex[x1] = 'x'+str(cx)
            cx += 1
        outvars[name] = ((_namey[y1],_namex[x1],), var[:,:])
    ds = xr.Dataset(outvars)
    ds.to_netcdf(filename,group='var')
    ds.close()
    
    # Parameters
    _namey = {}
    _namex = {}
    _namez = {}
    outparams = {}
    cy,cx,cz = 1,1,1
    for name,var in self.params.items():
        if len(var.shape)==2:
            y1,x1 = var.shape
            if y1 not in _namey:
                _namey[y1] = 'y'+str(cy)
                cy += 1
            if x1 not in _namex:
                _namex[x1] = 'x'+str(cx)
                cx += 1
            outparams[name] = ((_namey[y1],_namex[x1],), var[:,:])
        else:
            z1 = var.size
            if z1 not in _namez:
                _namez[z1] = 'z'+str(cz)
                cz += 1
            outparams[name] = ((_namez[z1],), var.flatten())

    ds = xr.Dataset(outparams)
    ds.to_netcdf(filename,group='params',mode='a')
    ds.close()
    
    return

def load_output(self,date,name_var=None):
    filename = os.path.join(self.path_save,f'{self.name_exp_save}'\
        f'_y{date.year}'\
        f'm{str(date.month).zfill(2)}'\
        f'd{str(date.day).zfill(2)}'\
        f'h{str(date.hour).zfill(2)}'\
        f'm{str(date.minute).zfill(2)}.nc')
        
    ds = xr.open_dataset(filename)
    
    ds1 = ds.copy().squeeze()
    
    ds.close()
    del ds
    
    if name_var is None:
        return ds1
    
    else:
        return np.array([ds1[name].values for name in name_var])

def load(self,filename):

    with xr.open_dataset(filename,group='var') as ds:
        for name in self.var.keys():
            self.var[name] = ds[name].values
    
    with xr.open_dataset(filename,group='params') as ds:
        for name in self.params.keys():
            self.params[name] = ds[name].values
        

def random(self,ampl=1):
    other = self.copy(free=True)
    for name in self.var.keys():
        other.var[name] = ampl * np.random.random(self.var[name].shape).astype('float64')
        other.var[name][self.mask[name]] = np.nan
    for name in self.params.keys():
        other.params[name] = ampl * np.random.random(self.params[name].shape).astype('float64')
        other.params[name][self.mask[name]] = np.nan 
    return other


def copy(self, free=False):

    # Create new instance
    other = State(self.config,first=False)

    # Copy all attributes
    other.ny = self.ny
    other.nx = self.nx
    other.DX = self.DX
    other.DY = self.DY
    other.X = self.X
    other.Y = self.Y
    other.dx = self.dx
    other.dy = self.dy
    other.f = self.f
    other.mask = self.mask
    other.lon = self.lon
    other.lat = self.lat
    other.geo_grid = self.geo_grid

    # (deep)Copy model variables
    for name in self.var.keys():
        if free:
            other.var[name] = np.zeros_like(self.var[name])
        else:
            other.var[name] = deepcopy(self.var[name])
    
    # (deep)Copy model parameters
    for name in self.params.keys():
        if free:
            other.params[name] = np.zeros_like(self.params[name])
        else:
            other.params[name] = deepcopy(self.params[name])

    return other

def getvar(self,name_var=None,vect=False):
    if name_var is not None:
        if type(name_var) in (list,np.ndarray):
            var_to_return = []
            for name in name_var:
                if vect:
                    var_to_return = np.concatenate((var_to_return,self.var[name].ravel()))
                else:
                    var_to_return.append(self.var[name])
                
        else:
            var_to_return = self.var[name_var]
            if vect:
                var_to_return = var_to_return.ravel()
    else:
        var_to_return = []
        for name in self.var.keys():
            if vect:
                var_to_return = np.concatenate((var_to_return,self.var[name].ravel()))
            else:
                var_to_return.append(self.var[name])

    return deepcopy(np.asarray(var_to_return))

def getparams(self,name_params=None,vect=False):
    if name_params is not None:
        if type(name_params) in (list,np.ndarray):
            params_to_return = []
            for name in name_params:
                if vect:
                    params_to_return = np.concatenate((params_to_return,self.params[name].ravel()))
                else:
                    params_to_return.append(self.params[name])
                
        else:
            params_to_return = self.params[name_params]
            if vect:
                params_to_return = params_to_return.ravel()
    else:
        params_to_return = []
        for name in self.params:
            if vect:
                params_to_return = np.concatenate((params_to_return,self.params[name].ravel()))
            else:
                params_to_return.append(self.params[name])

    return deepcopy(np.asarray(params_to_return))

def setvar(self,var,name_var=None,add=False):

    if name_var is None:
        for i,name in enumerate(self.var):
            if add:
                self.var[name] += var[i]
            else:
                self.var[name] = deepcopy(var[i])
    else:
        if type(name_var) in (list,np.ndarray):
            for i,name in enumerate(name_var):
                if add:
                    self.var[name] += var[i]
                else:
                    self.var[name] = deepcopy(var[i])
        else:
            if add:
                self.var[name_var] += var
            else:
                self.var[name_var] = deepcopy(var)

def scalar(self,coeff,copy=False):
    if copy:
        State1 = self.copy()
        for name in self.var.keys():
            State1.var[name] *= coeff
        for name in self.params.keys():
            State1.params[name] *= coeff
        return State1
    else:
        for name in self.var.keys():
            self.var[name] *= coeff
        for name in self.params.keys():
            self.params[name] *= coeff
    
def Sum(self,State1,copy=False):
    if copy:
        State2 = self.copy()
        for name in self.var.keys():
            State2.var[name] += State1.var[name]
        for name in self.params.keys():
            State2.params[name] += State1.params[name]
        return State2
    else:
        for name in self.var.keys():
            self.var[name] += State1.var[name]
        for name in self.params.keys():
            self.params[name] += State1.params[name]
        
def plot(self,title=None,cmap='RdBu_r',ind=None,params=False):
    
    if self.flag_plot<1:
        return
    
    if ind is not None:
        indvar = ind
    else:
        if not params:
            indvar = np.arange(0,len(self.var.keys()))
        else:
            indvar = np.arange(0,len(self.params.keys()))
    nvar = len(indvar)

    fig,axs = plt.subplots(1,nvar,figsize=(nvar*7,5))
    
    if title is not None:
        fig.suptitle(title)
        
    if nvar==1:
        axs = [axs]
    
    if not params:
        for ax,name_var in zip(axs,self.var):
            ax.set_title(name_var)
            _min = np.nanmin(self.var[name_var])
            _max = np.nanmax(self.var[name_var])
            _max_abs = np.nanmax(np.absolute(self.var[name_var]))
            if np.sign(_min)!=np.sign(_max) and ((_max-np.abs(_min))<.5*_max_abs):
                im = ax.pcolormesh(self.var[name_var],cmap=cmap,\
                                shading='auto', vmin = -_max_abs, vmax = _max_abs)
            else:
                im = ax.pcolormesh(self.var[name_var], shading='auto')
            plt.colorbar(im,ax=ax)
    else:
        for ax,name_var in zip(axs,self.params):
            ax.set_title(name_var)
            if np.sign(np.nanmin(self.params[name_var]))!=np.sign(np.nanmax(self.params[name_var])):
                cmap_range = np.nanmax(np.absolute(self.params[name_var]))
                im = ax.pcolormesh(self.params[name_var],cmap=cmap,\
                                shading='auto', vmin = -cmap_range, vmax = cmap_range)
            else:
                im = ax.pcolormesh(self.params[name_var],shading='auto')
            plt.colorbar(im,ax=ax)
    
    plt.show()








