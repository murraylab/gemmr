Private API reference
=====================

generative_model
----------------

.. currentmodule:: gemmr.generative_model

.. autosummary::
   :toctree: _autosummary

   _mk_Sigmaxy
   _find_latent_mode_vectors
   _find_latent_mode_vectors_qr
   _find_latent_mode_vectors_opti
   _find_latent_mode_vectors_pc1
   _add_lowvariance_subspace_components
   _generate_random_dominant_subspace_rotations
   _generate_dominant_subspace_rotations_from_opti
   _assemble_Sigmaxy_pls
   _assemble_Sigmaxy_cca
   _Sigmaxy_negative_min_eval
   _variance_explained_by_latent_modes
   calc_schur_complement
   _check_subspace_dimension

metric
------

.. currentmodule:: gemmr

.. autosummary::
    :toctree: _autosummary

    metrics.mk_betweenAssocRelError
    metrics.mk_betweenAssocRelError_cv
    metrics.mk_meanBetweenAssocRelError
    metrics.mk_weightError
    metrics.mk_scoreError
    metrics.mk_loadingError
    metrics.mk_crossloadingError

sample_size
-----------

.. currentmodule:: gemmr.sample_size

.. autosummary::
    :toctree: _autosummary

    interpolation.calc_n_required
    interpolation.calc_n_required_all_metrics
    interpolation.calc_max_n_required
    linear_model.do_fit_lm
    linear_model.prep_data_for_lm
    linear_model.fit_linear_model
    linear_model.get_lm_coefs
    linear_model._save_linear_model

data
----

.. currentmodule:: gemmr.data

.. autosummary::
    :toctree: _autosummary

    loaders.load_metaanalysis_outcomes
    loaders.load_metaanalysis
    loaders._fetch_synthanad
    loaders._check_data_home
    loaders._check_outcome_data_exists

util
----

.. currentmodule:: gemmr.util

.. autosummary::
    :toctree: _autosummary

    check_positive_definite
    align_weights
    nPerFtr2n
