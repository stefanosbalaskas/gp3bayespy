from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _prepared(seed=4201, *, unit="millimetres"):
    sim = p.simulate_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.5),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=seed,
    )
    contract = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit=unit,
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
        baseline_window=(-0.2, 0.0),
    )
    return p.prepare_pupil_timecourse(sim.data, contract)


def test_pupil_readiness_measurement_and_prior_scale_guards():
    prepared = _prepared()
    audit = p.audit_pupil_readiness(prepared)
    for component in ("summary", "participant", "condition", "trial"):
        assert isinstance(
            p.pupil_readiness_table(audit, component),
            pd.DataFrame,
        )
    with pytest.raises(GP3BayesError):
        p.pupil_readiness_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_readiness_table(audit, "bad")  # type: ignore[arg-type]

    measurement = p.audit_pupil_measurement_context(prepared)
    assert not p.pupil_measurement_audit_table(measurement).empty
    with pytest.raises(GP3BayesError):
        p.audit_pupil_measurement_context(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_measurement_audit_table(object())  # type: ignore[arg-type]

    arbitrary = _prepared(4202, unit="arbitrary_units")
    with pytest.raises(GP3BayesError):
        p._default_prior_scales(arbitrary, None)

    center, scales = p._default_prior_scales(
        arbitrary,
        {
            "intercept": 1,
            "coefficient": 1,
            "group_sd": 1,
            "residual": 1,
            "smooth_sd": 1,
            "ar": 0.5,
        },
    )
    assert np.isfinite(center)
    assert scales["ar"] == 0.5

    with pytest.raises(GP3BayesError):
        p._default_prior_scales(
            prepared,
            {"coefficient": 0},
        )


def test_pupil_specification_guard_and_design_matrix_matrix():
    prepared = _prepared(4210)

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(object())  # type: ignore[arg-type]

    one_pid = prepared.data[
        prepared.data[".participant"] == prepared.data[".participant"].iloc[0]
    ].copy()
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(replace(prepared, data=one_pid))

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            temporal_structure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            autocorrelation="bad",  # type: ignore[arg-type]
        )

    irregular = replace(
        prepared,
        timing={
            **dict(prepared.timing),
            "cv_dt": 1.0,
            "irregularity_review_cv": 0.1,
        },
    )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            irregular,
            autocorrelation="ar1",
        )

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            covariates=("missing",),
            autocorrelation="none",
        )

    nonnumeric = prepared.data.copy()
    nonnumeric["cov"] = "x"
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            replace(prepared, data=nonnumeric),
            covariates=("cov",),
            autocorrelation="none",
        )

    data = prepared.data.copy()
    data["cov"] = np.linspace(0, 1, len(data))
    prep_cov = replace(prepared, data=data)
    smooth = p.specify_pupil_timecourse_model(
        prep_cov,
        temporal_structure="smooth",
        autocorrelation="ar1",
        participant_trajectory="factor_smooth",
        covariates=("cov",),
    )
    table = p.pupil_specification_table(smooth)
    assert "formula" in set(table["field"])

    X, names = p._design_matrix(data, smooth)
    assert X.shape[0] == len(data)
    assert "event_time_pow2" in names
    assert "cov" in names
    assert any(name.startswith("condition[") for name in names)

    linear = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        autocorrelation="none",
        condition_trajectory=False,
        item_effects=False,
    )
    X2, names2 = p._design_matrix(prepared.data, linear)
    assert "event_time_pow2" not in names2
    assert not any(name.startswith("condition[") for name in names2)

    with pytest.raises(GP3BayesError):
        p.pupil_specification_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.translate_pupil_model_to_brms(object())  # type: ignore[arg-type]

    translated = p.translate_pupil_model_to_brms(linear)
    assert translated["compile"] is False

    with pytest.raises(GP3BayesError):
        p.check_pupil_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_pupil_prior_predictive(linear, draws=0)
    with pytest.raises(GP3BayesError):
        p.check_pupil_prior_predictive(
            linear,
            probability=1.0,
        )

    prior = p.check_pupil_prior_predictive(
        linear,
        execute=True,
        draws=5,
    )
    assert prior.executed is True


def test_pupil_fit_wrappers_and_prediction_extra_guards():
    prepared = _prepared(4220)
    spec = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        autocorrelation="none",
    )

    with pytest.raises(GP3BayesError):
        p.fit_pupil_model_backend(object())  # type: ignore[arg-type]

    fit = p.fit_pupil_model(
        spec,
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=4221,
    )
    assert fit.backend == "rstan"

    cmd = p.fit_pupil_model_cmdstanr(
        spec,
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=4222,
    )
    assert cmd.backend == "cmdstanr"

    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            population_only=False,
        )

    explicit = prepared.data[[".event_time", ".condition"]].drop_duplicates().head(4)
    pred = p.predict_pupil_trajectory(
        fit,
        newdata=explicit,
        type="expected",
        ndraws=5,
    )
    assert pred.draws.shape[1] == len(explicit)

    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            newdata=pd.DataFrame({"x": [1]}),
        )
    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            ndraws=5,
            max_grid=1,
        )
