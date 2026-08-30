from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _advanced_data(*, condition: bool = False) -> pd.DataFrame:
    n_time = 8
    times = np.tile(np.arange(n_time, dtype=float), 4)
    data = pd.DataFrame(
        {
            "pupil": 3.0 + 0.02 * times + np.linspace(0.0, 0.1, len(times)),
            "time_ms": times,
            "participant_id": np.repeat(["p1", "p2"], 2 * n_time),
        }
    )
    if condition:
        data["condition"] = np.tile(
            np.repeat(["control", "treatment"], n_time),
            2,
        )
    return data


@pytest.fixture(scope="module")
def advanced_fit():
    data = _advanced_data()
    specification = p.specify_advanced_pupil_timecourse_model(
        data,
        temporal_structure="linear",
        autocorrelation="none",
        allow_high_complexity=True,
    )
    return p.fit_advanced_pupil_model_backend(
        specification,
        backend="analytic",
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=12001,
    )


def test_blink_attempt_with_no_eligible_timepoints():
    simulation = p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=1,
        n_items=None,
        sampling_frequency=10,
        time_window=(-0.2, 0.0),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=1.0,
        blink_duration=0.12,
        seed=12010,
    )
    assert not simulation.data["blink"].any()
    assert simulation.data["pupil_mm"].notna().all()


def test_paired_eye_audit_columns_with_no_complete_pairs():
    simulation = p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        sampling_frequency=10,
        time_window=(-0.2, 0.4),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=12011,
    )
    data = simulation.data.copy()
    data["left_mm"] = np.nan
    data["right_mm"] = np.nan
    contract = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit="millimetres",
        sampling_frequency=10,
        condition_col="condition",
        left_pupil_col="left_mm",
        right_pupil_col="right_mm",
        channel_audit_unit="millimetres",
        baseline_window=(-0.2, 0.0),
    )
    prepared = p.prepare_pupil_timecourse(data, contract)
    audit = p.audit_pupil_readiness(prepared)
    value = audit.summary.loc[
        audit.summary["metric"] == "left_right_pupil_disagreement",
        "value",
    ].iloc[0]
    assert np.isnan(float(value))


def test_translation_with_missingness_predictor_builds_multifamily():
    data = _advanced_data()
    data["cov"] = np.linspace(0.0, 1.0, len(data))
    missingness = p.create_pupil_missingness_spec(
        response="exclude",
        predictors=("cov",),
    )
    specification = p.specify_advanced_pupil_timecourse_model(
        data,
        temporal_structure="linear",
        autocorrelation="none",
        covariates=("cov",),
        missingness_model=missingness,
        allow_high_complexity=True,
    )
    translation = p.translate_advanced_pupil_model_to_brms(specification)
    assert isinstance(translation.family, tuple)
    assert translation.family == ("gaussian", "gaussian")


def test_arma_conditional_prediction_requires_explicit_newdata(advanced_fit):
    arma_specification = replace(
        advanced_fit.specification,
        autocorrelation=p.create_pupil_arma_spec(1, 0),
    )
    arma_fit = replace(
        advanced_fit,
        specification=arma_specification,
        translation=replace(
            advanced_fit.translation,
            specification=arma_specification,
        ),
    )
    with pytest.raises(GP3BayesError, match="explicit series-aware"):
        p.predict_advanced_pupil_trajectory(
            arma_fit,
            newdata=None,
            population_only=False,
        )


def test_residual_scale_multiplier_remaining_branch_matrix(advanced_fit):
    base_data = advanced_fit.translation.data.copy()

    time_spec = replace(
        advanced_fit.specification,
        residual_scale="time",
    )
    time_fit = replace(
        advanced_fit,
        specification=time_spec,
        translation=replace(
            advanced_fit.translation,
            specification=time_spec,
            data=base_data,
        ),
    )
    time_multiplier = p._residual_scale_multiplier(time_fit, base_data)
    assert time_multiplier.shape == (len(base_data),)
    assert np.isfinite(time_multiplier).all()

    condition_data = base_data.copy()
    condition_data["condition"] = np.tile(
        np.repeat(["control", "treatment"], 8),
        2,
    )
    condition_mapping = dict(advanced_fit.specification.mapping)
    condition_mapping["condition"] = "condition"
    condition_spec = replace(
        advanced_fit.specification,
        data=condition_data,
        mapping=condition_mapping,
        residual_scale="condition",
    )
    condition_fit = replace(
        advanced_fit,
        specification=condition_spec,
        translation=replace(
            advanced_fit.translation,
            specification=condition_spec,
            data=condition_data,
        ),
    )
    condition_multiplier = p._residual_scale_multiplier(
        condition_fit,
        condition_data,
    )
    assert condition_multiplier.shape == (len(condition_data),)
    assert np.isfinite(condition_multiplier).all()

    sparse_residuals = np.full_like(advanced_fit.residuals, np.nan, dtype=float)
    sparse_residuals[:2] = [0.1, -0.1]
    sparse_time_fit = replace(time_fit, residuals=sparse_residuals)
    sparse_multiplier = p._residual_scale_multiplier(
        sparse_time_fit,
        base_data,
    )
    assert np.allclose(sparse_multiplier, 1.0)


def test_binocular_correlation_rejects_nonfit():
    with pytest.raises(GP3BayesError, match="Expected a binocular fit"):
        p.pupil_binocular_correlation(object())  # type: ignore[arg-type]


def test_ppc_generic_plot_with_no_numeric_columns():
    import matplotlib.pyplot as plt

    empty = pd.DataFrame()
    ppc = p.PupilPPC(
        trajectory=empty,
        distribution=empty,
        features=empty,
        residuals=empty,
        residual_trajectory=empty,
        autocorrelation=empty,
        heterogeneity=empty,
        measurement_context=pd.DataFrame(
            {
                "metric": ["source"],
                "value": ["declared"],
            }
        ),
        probability=0.95,
        declared_window=None,
        unit="millimetres",
    )
    figure = p.plot_pupil_ppc(ppc, "measurement_context")
    assert figure is not None
    plt.close(figure)
