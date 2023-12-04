.. _model_param_ana:

Model parameter analysis
========================

.. currentmodule:: gemmr.sample_analysis.analyzers

Parameter sweeps of multiple outcome metrics can be performed with the
function :func:`analyze_model_parameters`. As it has a large number of
parameters, we demonstrate and comment on them here.

A typical use case of :func:`analyze_model_parameters` is the following:

.. ipython:: python

    from gemmr.sample_analysis import *
    results = analyze_model_parameters(
        model='cca',
        estr='CCA',
        n_rep=10,
        n_perm=10,
        n_Sigmas=1,
        n_test=1000,
        pxs=(2,),
        rs=(0.9,),
        powerlaw_decay=('random_sum', 0, 0),
        n_per_ftrs='auto',
        addons=[
            addon.weights_true_cossim,
            addon.test_scores,
            addon.test_scores_true_pearson,
            addon.loadings_true_pearson,
            addon.remove_weights_loadings,
            addon.remove_test_scores
        ],
        mk_test_statistics=addon.mk_test_statistics_scores,
        saved_perm_features=['between_assocs'],
        postprocessors=[
            postproc.power,
            postproc.remove_between_assocs_perm
        ],
        random_state=42,
        show_progress=True
    )
    results

.. currentmodule:: gemmr.sample_analysis

Here's what all the parameters are for:

* ``model`` indicates whether synthetic data for ``'cca'`` or ``'pls'`` should be generated
* ``estr`` specifies an estimator with which the synthetic datasets are analyzed. This can be ``None`` or ``'auto'`` in which case an estimator corresponding to the model will be used. It can also be more specifically ``'CCA'``, ``'PLS'`` or ``'SparseCCA'``, or an instance of an estimator class.
* ``n_rep`` is the number of synthetic datasets drawn from each normal distribution
* ``n_perm`` is the number of times the rows of the :math:`Y` dataset are permuted. For each permutation the resulting dataset is analyzed in exactly the same way as the unpermuted dataset. As the permutations destroy associations between :math:`X` and :math:`Y`, null-distributions of quantities can be obtained in this way. Specifically, the permutations are required to calculate statistical power. We suggest to use at least 1000 permutations. Note also that the total computational cost heavily depends on ``n_perm``.
* ``n_Sigmas`` specifies how many normal distributions (more specifically: joint covariance matrices) are set up. If more than 1 is used they differ in the within-set variance spectrum (see ``axPlusay_range``) and in the direction of the true weight vectors relative to the principal component axes. For CCA, when ``axPlusay_range=(0,0)``, given the number of features and true correlation, all covariance matrices are identical, so that a small number for ``n_Sigmas`` should be sufficient to explore random fluctutations.
* ``n_test`` is the sample size used for a separate test dataset drawn from the same normal distribution as the synthetic dataset analyzed. Each draw from the normal distributions results in data for different "subjects", i.e. the rows of the generated data matrices have independent identities across repetitions. Some add-on functions that intend to compare generated data across repetitions therefore use a common.
* ``pxs`` is an iterable specifying the number of features for dataset :math:`X`. The number of features for dataset :math:`Y` is assumed to be identical by default, but see argument ``py`` of :func:`ccapwr.sample_analysis.analyzers.analyze_model_parameters`
* ``rs`` is an iterable specifying the assumed true correlations between datasets
* ``powerlaw_decay`` is a tuple specifying the minimum and maximum value for :math:`a_x+a_y`. :math:`a_x` and :math:`a_y` are, respectively, the decay constants for the powerlaws describing the within-set variance spectrum for datasets :math:`X` and :math:`Y`. Each time a covariance matrix is set up values for :math:`a_x+a_y` are drawn uniformly within this range.
* ``n_per_ftrs`` is an iterable giving the number of samples per total number of features (i.e. the number of features in :math:`X` plus the number of features in :math:`Y`) to use. It can also be set to ``'auto'`` in which case a crude experience-based heuristic is used to choose the set of numbers
* ``addons`` is a list of add-on functions that allow to run arbitrary analyses on each synthetic dataset after it has been fitted. A number of such functions is provided in module :mod:`addon`,
* ``mk_test_statistics`` is call-able object providing statistics of the test dataset that are made available to all add-on functions
* ``saved_perm_features`` allows to specify which outcomes are saved for permuted datasets. Each permuted dataset is analyzed in exactly the same way as the unpermuted dataset, but if only a subset of the outcomes are of interest, these can be specified here
* ``postprocessors`` is a list of functions that are called after the loop over all other parameters has finished. For example, statistical power can be calculated with this mechanism. A number of such functions i provided in module :mod:`postproc`.
* ``random_state`` must be set to distinct values if ``analyze_model_parameters`` is is called multiple times to explore the variability across covariance matrices
* ``show_progress`` shows progress bars for the loops over parameters if set to ``True``

