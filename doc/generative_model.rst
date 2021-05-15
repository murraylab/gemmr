.. _sample_size_calculation_tutorial:

Usage of generative model
=========================

.. currentmodule: gemmr.generative_model

To use the generative model, import and instantiate a :class:`GEMMR` object

.. ipython:: python

    from gemmr.generative_model import GEMMR
    gm = GEMMR('cca', px=10, py=5, r_between=0.3)

Here, we have set the number of X and Y features to 10 and y, respectively, and the between-set correlation to 0.3.
With this, synthetic data can be generated:

.. ipython:: python

    X, Y = gm.generate_data(n=1234)

As a sanity check, we can analyze these data with CCA.

.. ipython:: python

   from gemmr.estimators import SVDCCA
   cca = SVDCCA().fit(X, Y)
   
Inspecting the estimated canonical correlation, we get

.. ipython:: python

   cca.corrs_

and we see that it is close to the assumed one in the generative model (0.3).
