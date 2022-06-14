Diffusion model
===============

Equation solved
---------------

The diffusion model is the simplest model used in MASSH. Its unique prognostic equation applies to the SSH :math:`\eta` :

.. math::
   \partial_t \eta - K_{diff} \nabla^2 \eta = 0

where :math:`\nabla^2 = \partial^2_x + \partial^2_y` and :math:`K_{diffus}` is the diffusion coefficient, indicating the strenght of the diffusion. 

This model is linear and auto-adjoint, greatly facilating its implementation with 4Dvar data assimilation technique.
Note that setting :math:`K_{diffus}=0` gives the identity model, which can be used for static inversions.

Numerical implementation
------------------------
As the numerical implementation is straightforward, the code is located directly in the class :code:`Diffusion` in the ``mapping/src/mod.py`` directory. The discretization is done with a forward Euler scheme and a first order central spatial scheme.

