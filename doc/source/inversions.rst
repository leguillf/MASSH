Inversion techniques 
====================

Inversion consists in estimating the gridded (model) state :math:`X` assuming a vector of observations, noted :math:`Y` defined as:

.. math::
    Y = HX+\epsilon

where :math:`H` is a linear observation operator between the reconstruction grid space and the observation space and :math:`\epsilon` is an independent observation error. 

Different inversion techniques (static and dynamic) are implemented in MASSH and the codes are gathered in the ``mapping/src/ana.py`` script.

In the configuration file, you can select a specific analysis by setting the value of ``name_analysis``. 
If ``None`` value is set, then a forward propagation of the selected model is performed.

Hereafter we provide some details on the main inversion techniques used in MASSH:

.. toctree::
    :maxdepth: 2

    inversions/notation_oi
    inversions/notation_bfn
    inversions/notation_4dvar
