External boundary conditions
============================

MASSH allows to provide external boundary conditions to prevent any numerical instabilities on the edges of the model domain. 
For instance, these boundary conditions can come from other observational products. 

The space-time interpolation from the gird of the boundary conditions to the model grid is handled inline.  
So, the external fields can be defined on any grid (in space and time). The only requirement is that this grid includes the model grid. 

A relaxation zone induces a smooth integration of the boundary conditions into the inner region. 
The relaxation is performed with a Gaspari-Cohn function (Gaspari and Cohn 1999) taking the external product
value at the outer boundary, and the model value at the inner boundary of the relaxation zone.

Note that it is also possible to provide auxiliary data for the bathymetry, allowing to set boundary conditions for coastal areas. 

The associated functions handling boundary conditions setting are in the `mapping/src/gird.py` script

Here are the specific parameters (with their default values) for setting boundary conditions:

.. code-block:: python

    # Flag (True or False) to set or not boundary conditions
    flag_use_bc = False 

    # Path to the netcdf file(s) in which the boundary conditions fields are stored 
    # if None, 0 values are set as boundary conditions
    file_bc = None 

    # Width of the relaxation zone (in km) for which the boundary conditions are applied
    lenght_bc = 50 

    # Names of the boundary conditions variables
    name_var_bc = {'time':'','lon':'','lat':'','var':''}

    # Add mdt to boundary conditions. Useful only if boundary conditions are on sea level 
    add_mdt_bc = False 

    # use boundary conditions on coast. Useful only if a mask is provided 
    use_bc_on_coast = True 

    # Path to the netcdf file(s) in which the topography is stored
    file_depth = None 

    # Names of the topography variables
    name_var_depth = {'time':'','lon':'','lat':'','var':''} 

    # Minimal depth below which boundary conditions are used  
    bc_mindepth = None 