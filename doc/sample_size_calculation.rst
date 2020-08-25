.. _sample_size_calculation_tutorial:

Sample size calculation
=======================

.. currentmodule: gemmr.sample_size.linear_model

We demonstrate here the options available for the sample size calculation
functions :func:`.cca_sample_size` and :func:`.pls_sample_size`.

We first import these functions

.. ipython:: python

    from gemmr import cca_sample_size, pls_sample_size

The basic use case only requires the number of features and (for PLS) the
principal component spectrum decay constants:

.. ipython:: python

    cca_sample_size(5, 10)
    pls_sample_size(5, 10, -0.5, -1.5)

Instead of the number of features, the functions also accept data matrices.
To demonstrate this, we first generate a dataset

.. ipython:: python

    from gemmr.data import generate_example_dataset
    Xcca, Ycca = generate_example_dataset('cca', px=5, py=10)
    Xpls, Ypls = generate_example_dataset('pls', px=5, py=10, ax=-.5, ay=-1.5, n=10000)

Then we can calculate sample sizes for these datasets as follows:

.. ipython:: python

    cca_sample_size(Xcca, Ycca)
    pls_sample_size(Xpls, Ypls)

Note that for PLS only the data matrices are given as arguments, not the
principal component decay constants that we needed above. That is because the
decay constants are estimated from the data matrices. Correspondingly, the
PLS sample sizes here are similar but not identical to the ones we got above
(as the estimated decay constants are only approximately equal to the true ones).
For CCA, on the other hand, we get the same sample sizes as before, as expected.

The assumed true correlations for which sample sizes are calculated can be specified as follows:

.. ipython:: python

    cca_sample_size(Xcca, Ycca, rs=(.2, .7))
    pls_sample_size(Xpls, Ypls, rs=(.4, .6))

It is also possible to specify the target power and error levels:

.. ipython:: python
    :okwarning:

    cca_sample_size(5, 10, target_power=0.8, target_error=.5)

Finally, the criterion on which the calculation is based, can be specified. By
default, the `"combined"` criterion is used, meaning that power, association
stength error, weight error, score error and loading error are considered at
the same time and the linear model predicts the maximum sample size across all
these metrics. Alternatively, the calculation can be based on each of these
metrics alone:

.. ipython:: python
    :okwarning:

    cca_sample_size(5, 10, criterion='power')
    pls_sample_size(5, 10, -0.5, -1.5, criterion='association_strength')
    cca_sample_size(5, 10, criterion='weight')
    pls_sample_size(5, 10, -0.5, -1.5, criterion='score')
    cca_sample_size(5, 10, criterion='loading')
