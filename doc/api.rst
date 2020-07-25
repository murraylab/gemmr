.. currentmodule:: gemmr

API reference
=============

Sample size calculation
-----------------------

.. autosummary::
   :toctree: _autosummary

   cca_sample_size
   pls_sample_size
   pearson_sample_size
   sample_size.linear_model.cca_req_corr
   sample_size.linear_model.pls_req_corr

Estimators
----------

.. autosummary::
   :toctree: _autosummary

   estimators.SVDPLS
   estimators.SVDCCA
   estimators.NIPALSPLS
   estimators.NIPALSCCA
   estimators.SparseCCA

Synthetic data generation
-------------------------

.. autosummary::
   :toctree: _autosummary

   generative_model.setup_model
   generative_model.generate_data

Analysis of CCA/PLS results
---------------------------

.. autosummary::
   :toctree: _autosummary

   sample_analysis.analyzers.analyze_dataset
   sample_analysis.analyzers.analyze_resampled
   sample_analysis.analyzers.analyze_subsampled
   sample_analysis.analyzers.analyze_model
   sample_analysis.analyzers.analyze_model_parameters

Analysis add-ons
^^^^^^^^^^^^^^^^

The functions in :mod:`sample_analysis.analyzers` only fit an estimator and return association strengths, weights and
loadings. Additional analyses can be specified in the form of add-on functions. The following functions are provided,
and arbitrary custom ones can be used as long as they have the same function signature.

.. autosummary::
   :toctree: _autosummary

   sample_analysis.addon.remove_weights_loadings
   sample_analysis.addon.remove_cv_weights
   sample_analysis.addon.weights_true_cossim
   sample_analysis.addon.scores_true_spearman
   sample_analysis.addon.loadings_true_pearson
   sample_analysis.addon.test_scores
   sample_analysis.addon.remove_test_scores
   sample_analysis.addon.assoc_test
   sample_analysis.addon.weights_pc_cossim
   sample_analysis.addon.sparseCCA_penalties
   sample_analysis.addon.cv

Some of these add-ons require some help to set them up for work:

.. autosummary::
    :toctree: _autosummary

    sample_analysis.addon.mk_scorers_for_cv
    sample_analysis.addon.mk_test_statistics_scores

Analyses, that look into relations across datasets, and therefore require outcomes of more than a given current dataset
to work,  can be specified as *postprocessors*:

.. autosummary::
    :toctree: _autosummary

    sample_analysis.postproc.power
    sample_analysis.postproc.remove_between_assocs_perm
    sample_analysis.postproc.weights_pairwise_cossim_stats
    sample_analysis.postproc.scores_pairwise_spearmansim_stats
    sample_analysis.postproc.remove_weights_loadings
    sample_analysis.postproc.remove_test_scores

Finally, there are a number of analysis building blocks that we found useful:

.. autosummary::
    :toctree: _autosummary

    sample_analysis.macros.calc_p_value
    sample_analysis.macros.analyze_subsampled_and_resampled
    sample_analysis.macros.pairwise_weight_cosine_similarity

Model selection
---------------

.. autosummary::
    :toctree: _autosummary

    model_selection.max_min_detector
    model_selection.n_components_to_explain_variance

Plotting
--------

.. autosummary::
    :toctree: _autosummary

    plot.mean_metric_curve
    plot.heatmap_n_req
    plot.polar_hist

Data
----

Preprocessing
^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    data.preprocessing.preproc_smith

Handling of included data files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    data.loaders.set_data_home
    data.load_outcomes
    data.generate_example_dataset
    data.print_ds_stats

Utility functions
-----------------

.. autosummary::
    :toctree: _autosummary

    util.rank_based_inverse_normal_trafo
    util.pc_spectrum_decay_constant
