Dynamical models
================

Let's call :math:`X` the model state defined on a spatial grid (regular or not). Its temporal evolution is given by the dynamical operator :math:`\mathcal{M}`, such as:

.. math::
    \frac{\partial X}{\partial t} = \mathcal{M}(X,t)

The codes for the dynamical models are gathered in the ``mapping/src/mod.py`` script. For each model, a specific class is implemented. Each class ``M`` has the following functions:

    - ``M.step``: forward propagation in time 
    - ``M.step_tgl``: forward propagation in time of the linear tangent model
    - ``M.step_adj``: backward propagation in time of the adjoint model

For some models, the class call an external library located in ``mapping/models/``.

In the configuration file, you can select a specific model by setting the value of ``name_model``. 


Here is a detailed description of the dynamical models used in MASSH and how to parametrize them:

.. toctree::
    :maxdepth: 2

    models/notation_diff
    models/notation_qg1l
    models/notation_sw1l
    models/notation_bc


