"""Preprocessing pipeline from Smith et al. (2015)."""

import warnings

import numpy as np
from scipy.stats import zscore

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import SimpleImputer, IterativeImputer

from statsmodels.stats.correlation_tools import cov_nearest

from tqdm import trange

from ..util import rank_based_inverse_normal_trafo


__all__ = ['preproc_smith', 'preproc_minimal', 'select_sm_features',
           'prepare_confounders', 'deconfound']


_smith_feature_names = tuple(
    'PicVocab_Unadj PicVocab_AgeAdj PMAT24_A_CR DDisc_AUC_200 THC '
    'LifeSatisf_Unadj ListSort_AgeAdj ReadEng_Unadj SCPT_SPEC ReadEng_AgeAdj '
    'ListSort_Unadj DDisc_AUC_40K Avg_Weekday_Any_Tobacco_7days '
    'Num_Days_Used_Any_Tobacco_7days Total_Any_Tobacco_7days PicSeq_AgeAdj '
    'FamHist_Fath_DrgAlc PicSeq_Unadj Avg_Weekday_Cigarettes_7days '
    'Avg_Weekend_Any_Tobacco_7days Total_Cigarettes_7days Dexterity_AgeAdj '
    'Avg_Weekend_Cigarettes_7days Dexterity_Unadj Times_Used_Any_Tobacco_Today'
    ' PSQI_Score AngAggr_Unadj Taste_AgeAdj ASR_Rule_Raw Taste_Unadj '
    'ASR_Thot_Raw EVA_Denom SSAGA_TB_Still_Smoking FamHist_Fath_None '
    'ASR_Thot_Pct PercStress_Unadj ProcSpeed_AgeAdj ASR_Rule_Pct P'
    'rocSpeed_Unadj DSM_Antis_Raw ER40_CR NEOFAC_A ASR_Crit_Raw VSPLOT_TC '
    'NEOFAC_O ER40ANG VSPLOT_OFF SSAGA_Times_Used_Stimulants ASR_Soma_Pct '
    'SSAGA_Mj_Times_Used DSM_Antis_Pct CardSort_AgeAdj ASR_Extn_Raw '
    'ASR_Oth_Raw ASR_Totp_T ASR_Extn_T ASR_Totp_Raw EmotSupp_Unadj '
    'DSM_Anxi_Pct PercReject_Unadj ER40NOE DSM_Anxi_Raw ASR_TAO_Sum '
    'SSAGA_TB_Smoking_History CardSort_Unadj PosAffect_Unadj '
    'SSAGA_ChildhoodConduct Odor_AgeAdj ASR_Witd_Raw SSAGA_Alc_Hvy_Frq_Drk '
    'ASR_Soma_Raw DSM_Depr_Pct ASR_Aggr_Pct SSAGA_Alc_12_Max_Drinks '
    'DSM_Depr_Raw Mars_Final PercHostil_Unadj DSM_Somp_Pct '
    'SSAGA_Alc_Age_1st_Use ASR_Witd_Pct IWRD_TOT PainInterf_Tscore MMSE_Score '
    'SSAGA_Alc_12_Frq_Drk Odor_Unadj SSAGA_Alc_D4_Ab_Sx SSAGA_Mj_Use '
    'ASR_Aggr_Raw SSAGA_Mj_Ab_Dep DSM_Somp_Raw FearSomat_Unadj '
    'SSAGA_Alc_12_Drinks_Per_Day Mars_Log_Score SelfEff_Unadj SCPT_SEN '
    'NEOFAC_N SSAGA_Agoraphobia ASR_Intn_T AngHostil_Unadj '
    'Num_Days_Drank_7days SSAGA_Times_Used_Cocaine Loneliness_Unadj '
    'ASR_Intn_Raw SSAGA_Alc_Hvy_Drinks_Per_Day MeanPurp_Unadj DSM_Avoid_Pct '
    'NEOFAC_E Total_Beer_Wine_Cooler_7days DSM_Avoid_Raw '
    'Avg_Weekday_Wine_7days Flanker_AgeAdj ASR_Anxd_Pct '
    'Avg_Weekend_Beer_Wine_Cooler_7days SSAGA_Alc_D4_Ab_Dx Total_Drinks_7days '
    'SSAGA_Alc_Hvy_Max_Drinks FearAffect_Unadj Total_Wine_7days '
    'Avg_Weekday_Drinks_7days ER40SAD Flanker_Unadj ER40FEAR '
    'Avg_Weekday_Beer_Wine_Cooler_7days SSAGA_Times_Used_Illicits '
    'Avg_Weekend_Drinks_7days SSAGA_Alc_D4_Dp_Sx NEOFAC_C '
    'Total_Hard_Liquor_7days Correction SSAGA_Alc_Hvy_Frq_5plus '
    'DSM_Adh_Pct ASR_Attn_Pct VSPLOT_CRTE SSAGA_Depressive_Ep AngAffect_Unadj '
    'SSAGA_PanicDisorder Avg_Weekend_Hard_Liquor_7days FamHist_Moth_Dep '
    'ASR_Anxd_Raw SSAGA_Times_Used_Opiates SSAGA_Times_Used_Sedatives '
    'SSAGA_Alc_Hvy_Frq SSAGA_Alc_12_Frq_5plus Friendship_Unadj '
    'SSAGA_Depressive_Sx ASR_Attn_Raw ASR_Intr_Raw SSAGA_Alc_12_Frq '
    'FamHist_Fath_Dep InstruSupp_Unadj ASR_Intr_Pct '
    'SSAGA_Times_Used_Hallucinogens Avg_Weekend_Wine_7days FamHist_Moth_None '
    'Sadness_Unadj DSM_Hype_Raw DSM_Adh_Raw DSM_Inat_Raw'.split())


def deconfound(X, confounders, demean_confounders=False):
    """Deconfound a data matrix.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        data matrix to be confounded
    confounders : np.ndarray (n_samples, n_confounds)
        confound matrix
    demean_confounders : bool
        if ``True`` demeaned confounders are used

    Returns
    -------
    deconfounded : np.ndarray (n_samples, n_features)
        deconfounded data matrix
    """
    if confounders.shape[1] == 0:  # nothing to do
        return X
    if demean_confounders:
        confounders = confounders - confounders.mean(0, keepdims=True)
    deconfound_beta = np.linalg.pinv(confounders) @ X
    return X - confounders @ deconfound_beta


def preproc_smith(fc, sm,
                  feature_names=None, final_sm_deconfound=True,
                  confounders=tuple(),
                  hcp_confounders=False,
                  hcp_confounder_software_version=True,
                  squared_confounders=False,
                  hcp_data_dict_correct_pct_to_t=False,
                  include_N2=False,
                  confounds_impute=0):
    """Data preprocessing pipeline from Smith et al. (2015).

    Parameters
    ----------
    fc : np.ndarray, pd.DataFrame or xr.DataArray (n_samples, n_X_features)
        neuroimaging data matrix
    sm : pd.DataFrame (n_samples, n_Y_features)
        behavioral and demographic data matrix. Names of features to include,
        and confounds must be column names
    feature_names : None, slice or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default (i.e. from Smith et al. 2015, applicable to HCP data) feature
        names are used
    final_sm_deconfound : bool
        if ``True`` the subject measure data matrix will be deconfounded again
        as a very last preprocessing step, as in Smith et al. (2015). In that
        case, however, the resulting columns of Y will NOT be principal
        component scores.
    confounders : tuple of str
        column-names in ``sm`` to be used as confounders. If some are
        not found a warning is issued and the code will continue without the
        missing ones.
    hcp_confounders : bool
        if ``True`` 'Weight', 'Height', 'BPSystolic', 'BPDiastolic', 'HbA1C'
        as well as the cubic roots of 'FS_BrainSeg_Vol', 'FS_IntraCranial_Vol'
        are included as confounders
    hcp_confounder_software_version : bool
        if ``True`` and ``hcp_confounders`` is also ``True``, then the feature
        'fMRI_3T_ReconVrs' (encoded as a dummy variable) is used as confounder
    squared_confounders : bool
        if ``True`` the squares of all confounders (except software version, if
        used) are used as additional confounders
    hcp_data_dict_correct_pct_to_t : bool
        concerns only feature_names from HCP data dictionary. If ``True``
        a number of feature_names are replaced, see
        :func:`_check_feature_names`.
    include_N2 : bool
        if True, the data matrix, normalized by the absolute value of the mean
        of each feature, will be used as additional features, concatenating it
        horizontally to the z-scored data matrix
    confounds_impute : None, 0 or "mice"
        if 0, missing confound values are imputed with 0 (after an
        inverse normal transformation), if "mice"
        sklearn.impute.IterativeImputer is used, if None no imputation is
        performed

    Returns
    -------
    preprocessed_data : dict
        with items:

        * X : np.ndarray (n_samples, n_X_features)
            dataset X (principal component scores)
        * Y : np.ndarray (n_samples, n_Y_features)
            dataset Y (principal component scores)
        * X_whitened : np.ndarray (n_samples, n_X_features)
            whitened dataset X
        * Y_whitened : np.ndarray (n_samples, n_Y_features)
            whitened dataset Y
        * Y_raw : np.ndarray (n_samples, n_Y_features)
            unprocessed Y data comprising only the selected features (i.e.
            the matrix S4_raw)
        * feature_names : list
            ordered list of feature names corresponding to the columns of Y
        * X_pc_axes : np.ndarray (n_X_features, n_components)
            X principal component axes
        * confounders : np.ndarray (n_samples, n_features)
            confounder data matrix
        * X_preproc : np.ndarray (n_samples, n_X_features)
            preprocessed X data (not PC-ed)
        * Y_preproc : np.ndarray (n_samples, n_Y_features)
            preprocessed Y data (not PC-ed)

    References
    ----------
    Smith et al., A positive-negative mode of population covariation links
    brain connectivity, demographics and behavior, Nature Neuroscience
    (2015)
    """

    confounders = prepare_confounders(
        sm, confounders=confounders, hcp_confounders=hcp_confounders,
        hcp_confounder_software_version=hcp_confounder_software_version,
        squared_confounders=squared_confounders,
        impute=confounds_impute)

    uu1, uu1_white, vv1, N4 = preproc_fc(
        fc, confounders, include_N2=include_N2)

    uu2, uu2_white, S4_raw, S4_deconfounded, feature_names = preproc_sm(
        sm, confounders, final_deconfound=final_sm_deconfound,
        feature_names=feature_names,
        hcp_data_dict_correct_pct_to_t=hcp_data_dict_correct_pct_to_t)

    return dict(
        X=uu1,
        Y=uu2,
        X_whitened=uu1_white,
        Y_whitened=uu2_white,
        Y_raw=S4_raw,
        feature_names=feature_names,
        X_pc_axes=vv1,
        confounders=confounders,
        X_preproc=N4,
        Y_preproc=S4_deconfounded,
    )


def preproc_sm(sm, confounders, final_deconfound=True, feature_names=None,
               hcp_data_dict_correct_pct_to_t=True,
               nearest_psd_threshold=1e-6):
    """Preprocessing of subject measures.

    Parameters
    ----------
    confounders : np.ndarray (n_samples, n_features)
        confounder data matrix
    sm : pd.DataFrame (n_samples, n_Y_features)
        behavioral and demographic data matrix. Names of features to include,
        and confounds must be column names    final_deconfound : bool
        if ``True`` the final scores are once more deconfounded before they
        are returned
    feature_names : None, slice or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default (i.e. from Smith et al. 2015, applicable to HCP data) feature
        names are used
    hcp_data_dict_correct_pct_to_t : bool
        whether to correct HCP data dict names, see :func:`_check_features`
    nearest_psd_threshold : float
        threshold for finding an acceptable nearest positive definite matrix

    Returns
    -------
    uu2 : np.ndarray (n_samples, n_Y_features)
        processed dataset Y
    uu2_white : np.ndarray (n_samples, n_Y_features)
        whitened processed dataset Y
    S4_raw : np.ndarray (n_samples, n_Y_features)
        unprocessed Y data comprising only the selected features
    S4_deconfounded : np.ndarray (n_samples, n_Y_features)
        filtered and transformed data matrix (but still contains missing
        values)
    feature_names : list
        ordered list of feature names corresponding to the columns of Y
    """
    S4_raw, S4_deconfounded, feature_names = prepare_sm(
        sm, confounders, feature_names, hcp_data_dict_correct_pct_to_t)

    # estimate covariance-matrix, ignoring missing values
    # NOTE: This is the n_subjects x n_subjects covariance matrix across
    # features!
    # S_cov = np.nan * np.empty((S4_raw.shape[0], S4_raw.shape[0]))
    # for i in trange(len(S_cov), desc='subject', leave=False):
    #     for j in range(i + 1):
    #         mask = np.isfinite(S4_deconfounded[i]) \
    #                & np.isfinite(S4_deconfounded[j])
    #         S_cov[i, j] = S_cov[j, i] = np.cov(
    #             S4_deconfounded[i, mask], S4_deconfounded[j, mask])[0, 1]
    S_cov = np.ma.cov(np.ma.masked_invalid(S4_deconfounded)).data
    print('S_cov', S_cov.shape)
    assert np.isfinite(S_cov).all()
    S_cov_psd = cov_nearest(S_cov, threshold=nearest_psd_threshold)
    assert np.isfinite(S_cov_psd).all()
    assert np.allclose(S_cov_psd, S_cov_psd.T)
    assert np.linalg.matrix_rank(S_cov_psd) == len(S_cov_psd)
    print('smallest sval S_cov =',
          np.linalg.svd(S_cov, compute_uv=False, hermitian=True).min())
    print('smallest sval S_cov_psd =',
          np.linalg.svd(S_cov_psd, compute_uv=False, hermitian=True).min())
    print('rank S_cov =', np.linalg.matrix_rank(S_cov))
    print('rank S_cov_psd =', np.linalg.matrix_rank(S_cov_psd))

    # --- PCA ---

    dd2, uu2 = np.linalg.eigh(S_cov_psd)
    assert np.allclose((uu2**2).sum(0), 1)
    order = np.argsort(dd2)[::-1]
    dd2 = dd2[order]
    uu2 = uu2[:, order]
    assert np.all(np.diff(dd2) <= 0)

    uu2_white = uu2 / uu2.std(0)
    uu2 = uu2 * (np.sqrt(dd2) / uu2.std(0)).reshape(1, -1)

    # uu2 doesn't have mean 0, probably because of the way it's computed,
    # i.e. with cov_nearest, ...
    #assert np.allclose(uu2.mean(0), 0)
    assert np.allclose(uu2_white.var(0), 1)
    assert np.allclose(uu2.var(0), dd2)

    if final_deconfound:
        # deconfound again, just to be safe
        uu2_white = deconfound(uu2_white, confounders)
        uu2 = deconfound(uu2, confounders)

    return uu2, uu2_white, S4_raw, S4_deconfounded, feature_names


def prepare_sm(sm, confounders, feature_names,
               hcp_data_dict_correct_pct_to_t=True):
    """Filter and transform subject-measure data matrix.

    Selects features as indicated by argument ``feature_names``, applies
    rank-based inverse normal transform, and decounfounds.

    Parameters
    ----------
    sm : pd.DataFrame (n_samples, n_Y_features)
        behavioral and demographic data matrix. Names of features to include,
        and confounds must be column names    final_deconfound : bool
        if ``True`` the final scores are once more deconfounded before they
        are returned
    confounders : np.ndarray (n_samples, n_features)
        confounder data matrix
    feature_names : None, slice or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default (i.e. from Smith et al. 2015, applicable to HCP data) feature
        names are used
    hcp_data_dict_correct_pct_to_t : bool
        whether to correct HCP data dict names, see :func:`_check_features`

    Returns
    -------
    S4_raw : np.ndarray (n_samples, n_Y_features)
        unprocessed Y data comprising only the selected features
    S4_deconfounded : np.ndarray (n_samples, n_Y_features)
        filtered and transformed data matrix (but still contains missing
        values)
    feature_names : list
        ordered list of feature names corresponding to the columns of Y
    """
    S4_raw, feature_names = select_sm_features(sm, feature_names,
                                               hcp_data_dict_correct_pct_to_t)

    # gaussianise and deconfound
    S4_deconfounded = np.nan * np.empty_like(S4_raw)
    for c in range(S4_raw.shape[1]):
        data = S4_raw[:, c]
        is_finite = np.isfinite(data)
        data_gauss = rank_based_inverse_normal_trafo(data[is_finite])
        S4_deconfounded[is_finite, c] = zscore(deconfound(
            data_gauss.reshape(-1, 1),  # comprising only the "finite" subjects
            zscore(confounders[is_finite])  # use the same "finite" subjects
        )[:, 0]  # deconfound returns a matrix with 1 column
                                               )

    return S4_raw, S4_deconfounded, feature_names


def select_sm_features(sm, feature_names, hcp_data_dict_correct_pct_to_t):
    """Select features from SM matrix.

    Parameters
    ----------
    sm : pd.DataFrame (n_samples, n_Y_features)
        behavioral and demographic data matrix. Names of features to include,
        and confounds must be column names    final_deconfound : bool
        if ``True`` the final scores are once more deconfounded before they
        are returned
    feature_names : None, slice or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default (i.e. from Smith et al. 2015, applicable to HCP data) feature
        names are used
    hcp_data_dict_correct_pct_to_t : bool
        whether to correct HCP data dict names, see :func:`_check_features`

    Returns
    -------
    S4_raw : np.ndarray (n_samples, n_Y_features)
        unprocessed Y data comprising only the selected features
    feature_names : list
        ordered list of feature names corresponding to the columns of Y
    """
    feature_names = _check_features(feature_names,
                                    sm.columns.values,
                                    hcp_data_dict_correct_pct_to_t)
    total_n_features = len(feature_names)
    # keep only available features
    feature_names = [f for f in feature_names if f in sm]
    S4_raw = sm[feature_names]
    S4_raw = S4_raw.values.astype(float)
    assert S4_raw.shape[1] == len(feature_names)
    print(f'{S4_raw.shape[1]} out of {total_n_features} features found')
    if S4_raw.shape[1] < total_n_features:
        print("Missing features (after potential renaming in _check_features)"
              ":", [f for f in feature_names if f not in sm])
    return S4_raw, feature_names


def _check_features(feature_names, available_feature_names,
                    hcp_data_dict_correct_pct_to_t):
    """Make sure feature names are unique.

    Parameters
    ----------

    feature_names : None, slice or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default (i.e. from Smith et al. 2015, applicable to HCP data) feature
        names are used
    available_feature_names : iterable
        all available feature names (used to retrieve feature names in case
        ``feature_names`` is a ``slice``), ignored otherwise
    hcp_data_dict_correct_pct_to_t : bool
        if ``True`` replaces a number of feature names with "corrected" names,
        see Reference.

    Returns
    -------
    feature_names : list
        filtered and potentially corrected feature names

    References
    ----------
    https://wiki.humanconnectome.org/display/PublicData/HCP+Data+Release+Updates%3A+Known+Issues+and+Planned+fixes
        (accessed May 15, 2020)
    """
    if feature_names is None:
        feature_names = _smith_feature_names
    elif isinstance(feature_names, slice):
        feature_names = available_feature_names[feature_names]
    # else: pass  # use feature_names as is
    # unique feature_names, keeping order (dict is insertion ordered!)

    feature_names = np.array(list(dict.fromkeys(feature_names)))

    if hcp_data_dict_correct_pct_to_t:
        for old, new in [
            # ('ASR_Anxd_Pct', 'ASR_Anxd_T'),  # this is mentioned in
            # reference, but it seems the old name still exists
            ('ASR_Witd_Pct', 'ASR_Witd_T'),
            ('ASR_Soma_Pct', 'ASR_Soma_T'),
            ('ASR_Thot_Pct', 'ASR_Thot_T'),
            ('ASR_Attn_Pct', 'ASR_Attn_T'),
            ('ASR_Aggr_Pct', 'ASR_Aggr_T'),
            ('ASR_Rule_Pct', 'ASR_Rule_T'),
            ('ASR_Intr_Pct', 'ASR_Intr_T'),
            ('DSM_Depr_Pct', 'DSM_Depr_T'),
            ('DSM_Anxi_Pct', 'DSM_Anxi_T'),
            ('DSM_Somp_Pct', 'DSM_Somp_T'),
            ('DSM_Avoid_Pct', 'DSM_Avoid_T'),
            ('DSM_Adh_Pct', 'DSM_Adh_T'),
            ('DSM_Antis_Pct', 'DSM_Antis_T'),
        ]:
            feature_names = np.where(feature_names == old, new, feature_names)

    return feature_names.tolist()


def preproc_fc(fc, confounders, include_N2=True):
    """Preprocess the neuroimaging data matrix.

    Parameters
    ----------
    fc : xr.DataArray, pd.DataFrame or np.ndarray (n_samples, n_features)
        data matrix
    confounders : np.ndarray (n_samples, n_confounds)
        confound matrix
    include_N2 : bool
        if True, the data matrix, normalized by the absolute value of the mean
        of each feature, will be used as additional features, concatenating it
        horizontally to the z-scored data matrix

    Returns
    -------
    uu1 : np.ndarray (n_samples, n_features)
        preprocessed principal component scores of ``fc``
    uu1_white : np.ndarray (n_samples, n_features)
        whitened, preprocessed  principal component scores of ``fc``
    vv1 : np.ndarray (n_features, n_components)
        principal component axes
    N4 : np.ndarray (n_samples, n_features)
        preprocessed original-variable data
    """

    try:
        N = fc.values  # in case ``fc`` is pd.DataFrame or xr.DataArray
    except AttributeError:
        N = fc  # assume ``fc`` is np.ndarray

    N1 = zscore(N)
    if include_N2:
        abs_mean_N = np.abs(np.mean(N, axis=0, keepdims=True))
        N2 = zscore((N / abs_mean_N)[:, abs_mean_N[0] >= 0.1])
        N3 = np.hstack([N1, N2])
    else:
        N3 = N1
    N4 = deconfound(N3, confounders)
    N4 -= N4.mean(0, keepdims=True)
    print(N4.shape)

    # PCA
    uu1, ss1, vv1h = np.linalg.svd(N4, full_matrices=False)
    vv1 = vv1h.T

    assert np.allclose((uu1**2).sum(0), 1)
    uu1 = uu1 / uu1.std(0, keepdims=True)

    uu1_white = uu1
    uu1 = uu1 * ss1.reshape(1, -1)

    # due to noise it's possible that some low-variance PCs don't pass the
    # asserts
    n_check = int(.95*uu1.shape[1])
    assert np.allclose(uu1.mean(0)[:n_check], 0)
    assert np.allclose(uu1_white.var(0)[:n_check], 1)
    assert np.allclose(uu1.var(0)[:n_check], ss1[:n_check]**2)

    return uu1, uu1_white, vv1, N4[:, :N.shape[1]]


class PreprocWarning(Warning):
    pass


def prepare_confounders(
        sm,
        confounders=tuple(),
        hcp_confounders=False,
        hcp_confounder_software_version=True,
        squared_confounders=False,
        impute=0,
        #headmotion_features=('movement_AbsoluteRMS_mean',
        #                     'movement_RelativeRMS_mean')
):
    """Prepare the confounder matrix.

    Parameters
    ----------
    sm : pd.DataFrame (n_samples, n_features)
        behavioral data matrix
    confounders : tuple of str
        column-names in ``sm`` to be used as confounders. If some are
        not found a warning is issued and the code will continue without the
        missing ones.
    hcp_confounders : bool
        if ``True`` 'Weight', 'Height', 'BPSystolic', 'BPDiastolic', 'HbA1C'
        as well as the cubic roots of 'FS_BrainSeg_Vol', 'FS_IntraCranial_Vol'
        are included as confounders
    hcp_confounder_software_version : bool
        if ``True`` and ``hcp_confounders`` is also ``True``, then the feature
        'fMRI_3T_ReconVrs' (encoded as a dummy variable) is used as confounder
    squared_confounders : bool
        if ``True`` the squares of all confounders (except software version, if
        used) are used as additional confounders
    impute : None, 0, "mean", "median", "mice"
        if 0, missing confound values are imputed with 0 (after an
        inverse normal transformation), if "median" median value is imputed,
        if "mice" sklearn.impute.IterativeImputer is used, if None no imputation is
        performed

    Returns
    -------
    confounders : np.ndarray (n_samples, n_features)
        confounder data matrix, if impute is ``None`` it can have ``NaN``s

    Raises
    ------
    ValueError
        if confounders couldn't be found in ``sm``
    """

    _confounders = [f for f in confounders if f in sm]
    if len(_confounders) != len(confounders):
        missing_confounders = [f for f in confounders if f not in sm]
        raise ValueError('Confounders not found: '
                         '{}'.format(missing_confounders))
    confounders_matrix = sm[_confounders].values

    if hcp_confounders:
        sm_confounders = sm[['Weight', 'Height', 'BPSystolic', 'BPDiastolic',
                             'HbA1C', ]].values
        fs_confounders = sm[['FS_BrainSeg_Vol',
                             'FS_IntraCranial_Vol']].values ** (1. / 3)
        confounders_matrix = np.hstack([confounders_matrix, sm_confounders,
                                        fs_confounders])

    if squared_confounders:
        confounders_matrix = np.hstack([confounders_matrix,
                                        confounders_matrix**2])

    if hcp_confounders and hcp_confounder_software_version:
        # software reconstruction version
        reconvrs = sm['fMRI_3T_ReconVrs'].values
        used_reconvrss = np.unique(reconvrs)
        print('used fMRI 3T reconstruction software versions are:',
              used_reconvrss)
        assert set(used_reconvrss.tolist()) == {'r177', 'r177 r227', 'r227'}

        # dummy-coding: r177 -> 0, r227 -> 1, "r177 r227" -> 1
        reconvrs = np.where(reconvrs == 'r177', 0, 1).reshape(-1, 1)

        confounders_matrix = np.hstack([confounders_matrix, reconvrs])

    if confounders_matrix.shape[1] > 0:

        # inverse normal transform (this also results in mean 0)
        # confounders_matrix = \
        #     rank_based_inverse_normal_trafo(confounders_matrix)

        if impute is not None:
            # impute 0 for missing values
            print('{:.2f}% of values in confounders missing, imputing'.format(
                100 * (1 - np.isfinite(confounders_matrix).mean())))

            if impute == 0:
                # inverse normal transform (this also results in mean 0)
                confounders_matrix = \
                   rank_based_inverse_normal_trafo(confounders_matrix)
                confounders_matrix[~np.isfinite(confounders_matrix)] = 0

            elif impute in ("mean", "median"):
                confounders_matrix = SimpleImputer(strategy=impute) \
                    .fit_transform(confounders_matrix)

            elif impute == 'mice':
                confounders_matrix = IterativeImputer(random_state=0) \
                    .fit_transform(confounders_matrix)

            else:
                raise ValueError(f"Invalid impute: {impute}")

        else:
            print('{:.2f}% of values in confounders missing'.format(
                100 * (1 - np.isfinite(confounders_matrix).mean()))
            )

        # normalise
        confounders_matrix = zscore(confounders_matrix, nan_policy='omit')

    else:
        print('No confounders specified')

    return confounders_matrix


def preproc_minimal(fc, sm,
                  feature_names=None, final_sm_deconfound=True,
                  confounders=tuple(),
                  hcp_confounders=False,
                  hcp_confounder_software_version=True,
                  squared_confounders=False,
                  hcp_data_dict_correct_pct_to_t=False,
                  confounds_impute=False):
    """Minimal data preprocessing pipeline.

    Imputation can be chosen as an option for confounders. Subsequently, only
    subjects without missing values in fc, sm and confounders are used. Both
    FC and SM are processed identically: z-scored, deconfounded, z-scored
    again, PCA.

    Parameters
    ----------
    fc : np.ndarray, pd.DataFrame or xr.DataArray (n_samples, n_X_features)
        neuroimaging data matrix
    sm : pd.DataFrame (n_samples, n_Y_features)
        behavioral and demographic data matrix. Names of features to include,
        and confounds must be column names
    feature_names : None or list-like
        names of features to use, names must be columns in ``sm``. If ``None``
        default feature names are used
    confounders : tuple of str
        column-names in ``sm`` to be used as confounders. If some are
        not found a warning is issued and the code will continue without the
        missing ones.
    hcp_confounders : bool
        if ``True`` 'Weight', 'Height', 'BPSystolic', 'BPDiastolic', 'HbA1C'
        as well as the cubic roots of 'FS_BrainSeg_Vol', 'FS_IntraCranial_Vol'
        are included as confounders
    hcp_confounder_software_version : bool
        if ``True`` and ``hcp_confounders`` is also ``True``, then the feature
        'fMRI_3T_ReconVrs' (encoded as a dummy variable) is used as confounder
    squared_confounders : bool
        if ``True`` the squares of all confounders (except software version, if
        used) are used as additional confounders
    hcp_data_dict_correct_pct_to_t : bool
        concerns only feature_names from HCP data dictionary. If ``True``
        a number of feature_names are replaced, see
        :func:`_check_feature_names`.
    confounds_impute : None, 0 or "mice"
        if 0, missing confound values are imputed with 0 (after an
        inverse normal transformation), if "mice"
        sklearn.impute.IterativeImputer is used, if None no imputation is
        performed

    Returns
    -------
    preprocessed_data : dict
        with items:

        * Xpreproc : np.ndarray (n_samples, n_Xraw_features)
            The preprocessed X data, deconfounded and z-scored, but NOT PCA-ed
        * Ypreproc : np.ndarray (n_samples, n_Yraw_features)
            The preprocessed Y data, deconfounded and z-scored, but NOT PCA-ed
        * X : np.ndarray (n_samples, n_X_features)
            dataset X
        * Y : np.ndarray (n_samples, n_Y_features)
            dataset Y
        * X_whitened : np.ndarray (n_samples, n_X_features)
            whitened dataset X
        * Y_whitened : np.ndarray (n_samples, n_Y_features)
            whitened dataset Y
        * Y_raw : np.ndarray (n_samples, n_Y_features)
            unprocessed Y data comprising only the selected features (i.e.
            the matrix S4_raw)
        * feature_names : list
            ordered list of feature names corresponding to the columns of Y
        * X_pc_axes : np.ndarray (n_X_features, n_components)
            X principal component axes
        * Y_pc_axes : np.ndarray (n_Y_features, n_components)
            Y principal component axes
        * confounders : np.ndarray (n_samples, n_features)
            confounder data matrix
        * subjects_mask : np.ndarray (n_samples,) of bool
            indicates which subjects have no missing values and have thus been
            included in the outputs
    """

    confounders = prepare_confounders(
        sm, confounders=confounders, hcp_confounders=hcp_confounders,
        hcp_confounder_software_version=hcp_confounder_software_version,
        squared_confounders=squared_confounders,
        impute=confounds_impute)

    sm_raw, feature_names = \
        select_sm_features(sm, feature_names, hcp_data_dict_correct_pct_to_t)

    subjects_mask = np.isfinite(fc).all(axis=1) \
               & np.isfinite(sm_raw).all(axis=1) \
               & np.isfinite(confounders).all(axis=1)
    print(f'Using {subjects_mask.sum()} subjects which have no missing values')

    fc = fc[subjects_mask]
    sm_raw = sm_raw[subjects_mask]
    confounders = confounders[subjects_mask]

    def _minimal_preproc(X, confounders):
        X = zscore(X)

        Xd = deconfound(X, confounders)

        Xdz = zscore(Xd)  # just to be safe that X has mean 0 for SVD (PCA)

        _U, _s, _Vh = np.linalg.svd(Xdz, full_matrices=False)
        V = _Vh.T

        U_white = _U / _U.std(0, keepdims=True)
        U = U_white * _s.reshape(1, -1)

        assert np.allclose(U.mean(0), 0)
        assert np.allclose(U_white.var(0), 1)
        assert np.allclose(U.var(0), _s ** 2)

        return Xdz, U, U_white, V

    Xpreproc, UX, UX_white, VX = _minimal_preproc(fc, confounders)
    Ypreproc, UY, UY_white, VY = _minimal_preproc(sm_raw, confounders)

    return dict(
        Xpreproc=Xpreproc,
        Ypreproc=Ypreproc,
        X=UX,
        Y=UY,
        X_whitened=UX_white,
        Y_whitened=UY_white,
        Y_raw=sm_raw,
        feature_names=feature_names,
        X_pc_axes=VX,
        Y_pc_axes=VY,
        confounders=confounders,
        subjects_mask=subjects_mask
    )
