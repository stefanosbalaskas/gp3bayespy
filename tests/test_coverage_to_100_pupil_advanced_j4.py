from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _prepared(seed=4501):
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
        pupil_unit="millimetres",
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
        baseline_window=(-0.2, 0.0),
    )
    return p.prepare_pupil_timecourse(sim.data, contract)


def test_advanced_pupil_declaration_constructors_and_mapping_guards():
    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(covariate_errors={"": "se"})
    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(
            baseline_error="se1",
            covariate_errors={"baseline": "se2"},
        )

    model = p.create_pupil_measurement_model(
        response_error="response_se",
        covariate_errors={"cov": "cov_se"},
    )
    assert model.response_error == "response_se"

    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(response="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(assumptions="MNAR")
    missing = p.create_pupil_missingness_spec(
        response="model",
        predictors=("cov", "cov"),
        auxiliary_predictors=("aux",),
    )
    assert missing.predictors == ("cov",)

    for kwargs in (
        {"kernel": "bad"},
        {"basis": "bad"},
        {"scale": 1},
        {"k": 2},
        {"k": 201},
    ):
        with pytest.raises(GP3BayesError):
            p.create_pupil_gp_spec(**kwargs)
    exact = p.create_pupil_gp_spec(basis="exact")
    assert exact.k is None

    for args in ((True, 0), (4, 0), (0, 3), (0, 0)):
        with pytest.raises(GP3BayesError):
            p.create_pupil_arma_spec(*args)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(2, 1, covariance=True)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(1, 1, covariance=1)  # type: ignore[arg-type]

    assert p._ac_spec("none") is None
    assert p._ac_spec("ar1").p == 1
    assert p._ac_spec("ar2").p == 2
    assert p._ac_spec("arma11").q == 1
    with pytest.raises(GP3BayesError):
        p._ac_spec("bad")

    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution(family="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution(residual_scale="bad")  # type: ignore[arg-type]
    dist = p.specify_pupil_distribution("student", "time")
    assert p.pupil_distribution_table(dist).iloc[0]["family"] == "student"
    with pytest.raises(GP3BayesError):
        p.pupil_distribution_table(object())  # type: ignore[arg-type]

    prepared = _prepared()
    mapping = p.pupil_advanced_mapping_table(prepared)
    assert {"response", "time", "participant"}.issubset(set(mapping["role"]))
    with pytest.raises(GP3BayesError):
        p._advanced_mapping(pd.DataFrame())
    with pytest.raises(GP3BayesError):
        p._advanced_mapping(pd.DataFrame({"participant_id": ["p1"]}))
    raw = pd.DataFrame(
        {
            "pupil": [1.0, 1.1],
            "time_ms": [0.0, 1.0],
            "participant_id": ["p1", "p2"],
        }
    )
    _, raw_mapping = p._advanced_mapping(raw)
    assert raw_mapping["response"] == "pupil"


def test_advanced_pupil_specification_guard_matrix():
    prepared = _prepared(4510)

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            distribution=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            temporal_structure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            family="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            residual_scale="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            family="student",
            autocorrelation="ar1",
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            allow_high_complexity=1,  # type: ignore[arg-type]
        )

    no_condition = prepared.data.drop(
        columns=[name for name in (".condition", "condition") if name in prepared.data.columns]
    ).copy()
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            no_condition,
            residual_scale="condition",
        )

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            covariates=(".event_time",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            prepared,
            covariates=("missing",),
        )

    text = prepared.data.copy()
    text["cov"] = "x"
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            text,
            covariates=("cov",),
        )

    data = prepared.data.copy()
    data["cov"] = np.linspace(0, 1, len(data))
    data["cov_se"] = 0.1
    data["response_se"] = 0.1
    data["aux"] = np.linspace(1, 2, len(data))

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            measurement_model=object(),  # type: ignore[arg-type]
        )

    measurement = p.create_pupil_measurement_model(
        response_error="response_se",
        covariate_errors={"cov": "cov_se"},
    )
    missing_se = data.drop(columns="cov_se")
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            missing_se,
            covariates=("cov",),
            measurement_model=measurement,
        )

    bad_se = data.copy()
    bad_se["cov_se"] = 0.0
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            bad_se,
            covariates=("cov",),
            measurement_model=measurement,
        )

    miss_unknown = p.create_pupil_missingness_spec(predictors=("missing",))
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            missingness_model=miss_unknown,
        )

    miss_cov = p.create_pupil_missingness_spec(predictors=("cov",))
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            missingness_model=miss_cov,
        )

    miss_model = p.create_pupil_missingness_spec(
        response="model",
        predictors=("cov",),
        auxiliary_predictors=("aux",),
    )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("cov",),
            missingness_model=miss_model,
            autocorrelation="ar1",
        )

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("cov",),
            prior_scales={"b": 0},
        )

    spec = p.specify_advanced_pupil_timecourse_model(
        data,
        temporal_structure="gaussian_process",
        gp_spec=p.create_pupil_gp_spec(
            kernel="matern52",
            basis="approximate",
            k=8,
        ),
        residual_scale="time",
        covariates=("cov",),
        measurement_model=measurement,
        missingness_model=miss_cov,
        allow_high_complexity=True,
    )
    assert spec.complexity_audit is not None
    assert not p.pupil_advanced_specification_table(spec).empty
    with pytest.raises(GP3BayesError):
        p.pupil_advanced_specification_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.audit_pupil_computational_budget(object())  # type: ignore[arg-type]


def test_advanced_simulation_features_prior_translation_fit_and_scoring():
    for kwargs in (
        {"n_participants": 1},
        {"time_points": 7},
        {"family": "bad"},
        {"ar": (0.1, 0.2, 0.3, 0.4)},
        {"ma": (0.1, 0.2, 0.3)},
        {"ar": (0.99,)},
        {"family": "student", "student_df": 2},
    ):
        with pytest.raises(GP3BayesError):
            p.simulate_advanced_pupil_timecourse(**kwargs)

    sim = p.simulate_advanced_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        time_points=10,
        family="student",
        ar=(0.2,),
        ma=(0.1,),
        outlier_fraction=0,
        missing_fraction=0,
        seed=4520,
    )
    assert len(sim.data) == 40

    prepared = _prepared(4521)
    data = prepared.data.copy()
    data["cov"] = np.linspace(0, 1, len(data))
    spec = p.specify_advanced_pupil_timecourse_model(
        data,
        temporal_structure="gaussian_process",
        family="student",
        gp_spec=p.create_pupil_gp_spec(basis="exact"),
        covariates=("cov",),
        allow_high_complexity=True,
    )

    one = p._central_summary(np.array([1.0]), 0.9)
    assert one.iloc[0]["sd"] == 0

    no_trial_mapping = dict(spec.mapping)
    no_trial_mapping["trial"] = None
    keys = p._series_keys(data, no_trial_mapping)
    assert keys.nunique() == data[".participant"].nunique()

    assert p._advanced_reference_value(pd.Series([1.0, 2.0])) == 1.5
    assert p._advanced_reference_value(pd.Series(["a", "b"])) == "a"
    assert np.isnan(p._advanced_reference_value(pd.Series([], dtype=float)))

    X, names, meta = p._advanced_feature_matrix(data, spec)
    assert X.shape[0] == len(data)
    assert any(name.startswith("gp_basis[") for name in names)
    X2, names2, _ = p._advanced_feature_matrix(data, spec, metadata=meta)
    assert X2.shape == X.shape
    assert names2 == names

    prior = p.create_advanced_pupil_prior_specification(spec)
    assert {"nu", "sdgp", "lscale"}.issubset(set(prior.table["parameter"]))
    with pytest.raises(GP3BayesError):
        p.create_advanced_pupil_prior_specification(object())  # type: ignore[arg-type]

    translated = p.translate_advanced_pupil_model_to_brms(spec)
    assert "gp(" in translated.formula
    with pytest.raises(GP3BayesError):
        p.translate_advanced_pupil_model_to_brms(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p.check_advanced_pupil_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_advanced_pupil_prior_predictive(spec, iter=10, warmup=10)
    prior_check = p.check_advanced_pupil_prior_predictive(
        spec,
        chains=1,
        iter=60,
        warmup=10,
        seed=4522,
    )
    assert prior_check.executed

    with pytest.raises(GP3BayesError):
        p.fit_advanced_pupil_model_backend(object())  # type: ignore[arg-type]

    fit = p.fit_advanced_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=80,
        warmup=20,
        cores=1,
        seed=4523,
    )
    assert fit.fit_performed

    observed = np.array([1.0, 2.0])
    with pytest.raises(GP3BayesError):
        p.score_pupil_predictions(observed, np.ones((3, 3)))
    with pytest.raises(GP3BayesError):
        p.score_pupil_predictions(
            np.array([np.nan, np.nan]),
            np.ones((3, 2)),
        )

    score1 = p.score_pupil_predictions(
        observed,
        np.array([[1.1, 1.9]]),
    )
    assert score1.pointwise["crps"].isna().all()
    score2 = p.score_pupil_predictions(
        observed,
        np.array([[1.0, 2.0], [1.2, 1.8]]),
    )
    assert np.isfinite(
        score2.table.loc[
            score2.table["metric"] == "approx_crps",
            "value",
        ].iloc[0]
    )


def test_binocular_simulation_and_prepare_guard_matrix():
    with pytest.raises(GP3BayesError):
        p.simulate_binocular_pupil_timecourse(residual_correlation=0.99)

    sim = p.simulate_binocular_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        time_points=8,
        outlier_fraction=0,
        missing_fraction=0,
        seed=4530,
    )
    data = sim.data

    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse([])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(data.drop(columns="pupil_left"))
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(
            data,
            left_col="pupil_left",
            right_col="pupil_left",
        )

    one_condition = data.copy()
    one_condition["condition"] = "control"
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(one_condition)

    missing_pid = data.copy()
    missing_pid.loc[missing_pid.index[0], "participant_id"] = None
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(missing_pid)

    bad_time = data.copy()
    bad_time.loc[bad_time.index[0], "time_ms"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(bad_time)

    prepared = p.prepare_binocular_pupil_timecourse(data)
    assert prepared.mapping["left"] == "pupil_left"
