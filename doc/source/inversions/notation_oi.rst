Optimal Interpolation
=====================

Optimal interpolation (OI) is widely used in geosciences to interpolate sparse data onto regular grids. It is a static inversion, meaning that it does not rely on any dynamical constrain. 

Method
------
OI consists in estimating an analysed state :math:`X_a` combining the available observations :math:`Y` to approximate the real state :math:`X`:

.. math::
    X_a = BH^T(HBH^T+R)^{-1}Y 

where :math:`B` (resp. :math:`R`) is the convariance matrix of :math:`X` (resp. :math:`Y`).

:math:`B` has to be carefully chosen to represent the statistics of the field to be estimated. 
For now, :math:`B` represents the variability of an isotropic field with unique time and space scales:

.. math:: 
    B = \left( e^{-\left(\frac{t_i-t_j}{L_t}\right)^2} e^{-\left(\frac{x_i-x_j}{L_x}\right)^2} e^{-\left(\frac{y_i-y_j}{L_y}\right)^2} \right)_{(i,j)}

where :math:`(t,x,y)` are the 3D time-space coordinates, :math:`L_t` is the time scale and :math:`(L_x,L_y)` are the spatial scales.

Similarly, :math:`R` should represent the statistics of the observational and representativeness errors. For now, we only consider non-correlated constant errors.

Configuration parameters
------------------------
Here are the parameters specific to the OI technique which may be prescribed in the configuration file. If some are not prescribed, then the default values shown hereafter will be used.

.. code-block:: python

    # Name of the inversion technique
    name_analysis = 'OI'

    # Time scale (in days)
    oi_Lt = 7 

    # Longitudinal space scale (in degree)
    oi_Lx = 1 

    # Latitudinal space scale (in degree)
    oi_Ly = 1 

    # Observational error (in the same units as the observed quantity)
    oi_noise = 5e-2 