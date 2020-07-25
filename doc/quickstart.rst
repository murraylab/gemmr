.. currentmodule: gemmr

Quickstart
==========

.. ipython:: python
   :suppress:

   import matplotlib.pyplot as plt

I want to use CCA or PLS. How many samples are required?
---------------------------------------------------------

Simply import ``gemmr``

.. ipython:: python

    from gemmr import *

Then, for CCA, only the number of features in each of the 2 datasets needs to be specified, e.g.

.. ipython:: python
    :okwarning:

    cca_sample_size(100, 50)

The result is a dictionary with keys indicating assumed ground truth correlations and values giving the corresponding
sample size estimate. For PLS, in addition to the number of features in each dataset the powerlaw decay constant for the
within-set principal component spectrum needs to be specified:

.. ipython:: python
    :okwarning:

    pls_sample_size(100, 50, -1.5, -.5)

.. NOTE::
    Required sample sizes are calculated to obtain at least 90% power and less
    than 10% error in a number of other metrics. See the [gemmr]_ publication
    for more details.

More use cases and options of the sample size functions are discussed in
:ref:`sample_size_calculation_tutorial`.

How can I generate synthetic data for CCA or PLS?
-------------------------------------------------

The functionality is provided in module :mod:`generative_model` and requires two steps

.. ipython:: python

    from gemmr.generative_model import setup_model, generate_data


First, a model needs to be specified. The required parameters are:

* the number of features in `X` and `Y`
* the assumed true correlation between scores in `X` and `Y`
* the power-law exponents describing the within-set principal component spectra of `X` and `Y`

.. ipython:: python

    px, py = 3, 5
    r_between = 0.3
    ax, ay = -1, -.5
    Sigma = setup_model('cca', px=px, py=py, ax=ax, ay=ay, r_between=r_between)

Analogously, if a model for PLS is desired the first argument becomes ``'pls'``.

Second, given the model, data can be drawn from the normal distribution associated with the covariance matrix ``Sigma``:

.. ipython:: python

    X, Y = generate_data(Sigma, px, n=5000)
    X.shape, Y.shape

See the API reference for :func:`.generative_model.setup_model` and :func:`.generative_model.generate_data` for more details.

How do the provided CCA or PLS estimators work?
-----------------------------------------------

We assume two data arrays ``X`` and ``Y`` are given and shall be analyzed with CCA or PLS. The provided estimators
work like those in **sklearn**. For example, to perform a CCA:

.. ipython:: python

    from gemmr.estimators import SVDCCA

    cca = SVDCCA(n_components=1)
    cca.fit(X, Y)

After fitting several attributes become available. Estimated canonical correlations are stored in

.. ipython:: python

    cca.corrs_

weight (rotation) vectors in

.. ipython:: python

    cca.x_rotations_

and analogously in ``cca.y_rotations_``, and the attributes ``x_scores_`` and ``y_scores_`` provide the in-sample scores:

.. ipython:: python

    @savefig svdcca_scatter_scores.png width=4in
    plt.scatter(cca.x_scores_, cca.y_scores_, s=1)

``SVDPLS`` works analogously, but note that it finds maximal covariances instead of correlations, and correspondingly has an attribute ``covs_``.

For more information see the reference pages for :class:`.estimators.SVDCCA` and :class:`.estimators.SVDPLS`.

A sparse CCA estimator, based on the R-package *PMA*, is implemented as :class:`.estimators.SparseCCA`.


How can I investigate parameter dependencies of CCA or PLS?
-----------------------------------------------------------

This can be done with the function :func:`.sample_analysis.analyze_model_parameters`.
A basic use case is shown here:

.. ipython:: python

    from gemmr.sample_analysis import *
    results = analyze_model_parameters(
        'cca',
        pxs=(2, 5), rs=(.3, .5), n_per_ftrs=(2, 10, 30, 100),
        n_rep=10, n_perm=1,
    )
    results

The variable ``results`` contains a number of outcome metrics by default,
and further ones can be obtained through add-ons specified
as keyword-argument ``addons`` to :func:`.sample_analysis.analyze_model_parameters`.

Dependence of outcomes on, for example, sample size, can then be inspected:

.. ipython:: python

    plt.loglog(results.n_per_ftr, results.between_assocs.sel(px=5, r=0.3, Sigma_id=0).mean('rep'), label='r=0.3')
    plt.loglog(results.n_per_ftr, results.between_assocs.sel(px=5, r=0.5, Sigma_id=0).mean('rep'), label='r=0.5')

    plt.xlabel('samples per feature')
    plt.ylabel('canonical correlation')
    plt.legend()

    @savefig canonical_correlation_vs_n.png width=4in
    plt.gcf().tight_layout()

See :ref:`model_param_ana` for a more extensive example and the reference page for
:func:`.sample_analysis.analyzers.analyze_model_parameters` for more details.