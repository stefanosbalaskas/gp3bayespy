from __future__ import annotations

import builtins
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


@pytest.fixture(scope="module")
def base_bundle():
    sim = p.simulate_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.5),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=11901,
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
    prepared = p.prepare_pupil_timecourse(sim.data, contract)
    specification = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        autocorrelation="none",
    )
    fit = p.fit_pupil_model(
        specification,
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=11902,
    )
    return sim.data.copy(), contract, prepared, specification, fit


def _advanced_raw(n_time: int = 8, *, condition: bool = False) -> pd.DataFrame:
    times = np.tile(np.arange(n_time, dtype=float), 4)
    participants = np.repeat(["p1", "p2"], 2 * n_time)
    data = pd.DataFrame(
        {
            "pupil": 3.0 + 0.02 * times + np.linspace(0.0, 0.1, len(times)),
            "time_ms": times,
            "participant_id": participants,
        }
    )
    if condition:
        data["condition"] = np.tile(
            np.repeat(["control", "treatment"], n_time),
            2,
        )
    return data


@pytest.fixture(scope="module")
def advanced_bundle():
    raw = _advanced_raw(8, condition=False)
    specification = p.specify_advanced_pupil_timecourse_model(
        raw,
        temporal_structure="linear",
        autocorrelation="none",
        allow_high_complexity=True,
    )
    fit = p.fit_advanced_pupil_model_backend(
        specification,
        backend="analytic",
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=11903,
    )
    return raw, specification, fit


def test_simulation_forced_blink_path():
    sim = p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        sampling_frequency=10,
        time_window=(-0.2, 0.8),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=1,
        blink_duration=0.2,
        seed=11910,
    )
    assert bool(sim.data["blink"].any())
    assert bool(sim.data.loc[sim.data["blink"], "pupil_mm"].isna().all())


def test_millisecond_condition_free_paired_audit_and_raw_readiness(base_bundle):
    data = base_bundle[0].copy()
    data["event_time_ms"] = data["event_time"] * 1000.0
    data["left_mm"] = data["pupil_mm"] - 0.01
    data["right_mm"] = data["pupil_mm"] + 0.01

    contract = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time_ms",
        pupil_unit="millimetres",
        sampling_frequency=10,
        time_unit="milliseconds",
        item_col="item_id",
        left_pupil_col="left_mm",
        right_pupil_col="right_mm",
        channel_audit_unit="millimetres",
        baseline_window=(-200.0, 0.0),
    )
    prepared = p.prepare_pupil_timecourse(data, contract)
    assert prepared.baseline_window == pytest.approx((-0.2, 0.0))
    assert ".condition" not in prepared.data
    assert {".pupil_left_audit", ".pupil_right_audit"}.issubset(prepared.data.columns)
    audit = p.audit_pupil_readiness(prepared)
    assert audit.by_condition.empty
    disagreement = audit.summary.loc[
        audit.summary["metric"] == "left_right_pupil_disagreement",
        "value",
    ].iloc[0]
    assert float(disagreement) > 0

    with pytest.raises(GP3BayesError, match="Supply `contract`"):
        p.audit_pupil_readiness(data)
    raw_audit = p.audit_pupil_readiness(data, contract)
    assert isinstance(raw_audit, p.PupilReadiness)


def test_sequence_prior_scales_path(base_bundle):
    prepared = base_bundle[2]
    center, scales = p._default_prior_scales(
        prepared,
        [1.0, 1.1, 1.2, 1.3, 1.4, 0.5],
    )
    assert np.isfinite(center)
    assert scales["intercept"] == pytest.approx(1.0)
    assert scales["ar"] == pytest.approx(0.5)


def test_condition_contrast_skips_incomplete_timepoint():
    grid = pd.DataFrame(
        {
            ".event_time": [0.0, 0.0, 1.0],
            ".condition": ["A", "B", "A"],
        }
    )
    prediction = p.PupilPrediction(
        "test",
        "pupil",
        "expected",
        np.array(
            [
                [1.0, 0.8, 1.1],
                [1.1, 0.9, 1.2],
                [0.9, 0.7, 1.0],
            ]
        ),
        grid,
        "millimetres",
        3,
        3,
        "synthetic",
    )
    out = p.pupil_condition_contrast(prediction, ("A", "B"))
    assert len(out.table) == 1
    assert out.table[".event_time"].iloc[0] == 0.0


def test_functional_parts_derivative_fast_path():
    derivative = p.PupilTrajectoryDerivative(
        pd.DataFrame({".event_time": [0.0, 1.0]}),
        np.ones((3, 2)),
        1,
        0.95,
        {"mapping": {"time": ".event_time", "condition": None}},
    )
    grid, draws, specification, order = p._functional_parts(derivative)
    assert len(grid) == 2
    assert draws.shape == (3, 2)
    assert specification["mapping"]["time"] == ".event_time"
    assert order == 1


def test_prepared_from_spec_fit_future_validation_and_fit_loglik(base_bundle):
    _, _, prepared, specification, fit = base_bundle
    assert p._prepared_from_any(specification) is prepared
    assert p._prepared_from_any(fit) is prepared

    log_lik = p._fit_log_likelihood(fit)
    assert log_lik.ndim == 2
    assert log_lik.shape[1] == len(prepared.data)

    plan = p.create_pupil_validation_plan(
        prepared,
        target="future_segment",
        future_fraction=0.25,
        seed=11920,
    )
    validation = p.validate_pupil_model(
        fit,
        plan,
        execute=True,
        ndraws=20,
    )
    assert validation.executed
    assert {"n_train", "n_test", "rmse", "mae"}.issubset(validation.table.columns)


def test_advanced_distribution_and_no_trial_budget_paths():
    raw = _advanced_raw(8)
    distribution = p.specify_pupil_distribution("gaussian", "time")
    specification = p.specify_advanced_pupil_timecourse_model(
        raw,
        distribution=distribution,
        temporal_structure="linear",
        autocorrelation="none",
        allow_high_complexity=True,
    )
    assert specification.residual_scale == "time"
    audit = p.audit_pupil_computational_budget(specification)
    assert audit.series == raw["participant_id"].nunique()


def test_advanced_high_budget_guard(monkeypatch):
    raw = _advanced_raw(8)
    high = p.PupilComplexityAudit(
        "high",
        len(raw),
        raw["time_ms"].nunique(),
        1,
        raw["participant_id"].nunique(),
        raw["participant_id"].nunique(),
        pd.DataFrame(
            {
                "check": ["synthetic"],
                "status": ["high"],
                "message": ["synthetic high-complexity branch"],
            }
        ),
    )
    monkeypatch.setattr(p, "audit_pupil_computational_budget", lambda x: high)
    with pytest.raises(GP3BayesError, match="complexity budget"):
        p.specify_advanced_pupil_timecourse_model(
            raw,
            temporal_structure="linear",
            autocorrelation="none",
            allow_high_complexity=False,
        )


def test_advanced_sensitivity_false_paths_and_fallthrough(advanced_bundle):
    _, specification, _ = advanced_bundle
    suite = p.create_pupil_advanced_sensitivity_suite(specification, include=())
    assert suite.scenarios["scenario"].tolist() == ["baseline"]

    manual = p.AdvancedPupilSensitivitySuite(
        specification,
        pd.DataFrame(
            {
                "scenario": ["other_dimension"],
                "dimension": ["other"],
                "value": ["declared"],
            }
        ),
    )
    materialized = p.materialize_pupil_advanced_sensitivity_scenario(
        manual,
        "other_dimension",
    )
    assert isinstance(materialized, p.AdvancedPupilSpecification)


def test_advanced_constant_time_prior_translation_and_tiny_fit_guards():
    constant = _advanced_raw(4)
    constant["time_ms"] = 1.0
    constant["pupil"] = 3.0
    specification = p.specify_advanced_pupil_timecourse_model(
        constant,
        temporal_structure="linear",
        autocorrelation="none",
        allow_high_complexity=True,
    )
    X, names, metadata = p._advanced_feature_matrix(constant, specification)
    assert X.shape[0] == len(constant)
    assert names == ("Intercept", "time")
    assert metadata["time_scale"] == pytest.approx(1.0)

    prior = p.create_advanced_pupil_prior_specification(specification)
    assert (prior.table["scale"] > 0).all()

    measured = _advanced_raw(8)
    measured["response_se"] = 0.1
    measurement = p.create_pupil_measurement_model(response_error="response_se")
    measured_spec = p.specify_advanced_pupil_timecourse_model(
        measured,
        temporal_structure="linear",
        residual_scale="time",
        autocorrelation="none",
        measurement_model=measurement,
        allow_high_complexity=True,
    )
    translation = p.translate_advanced_pupil_model_to_brms(measured_spec)
    assert "sigma ~ time" in translation.formula
    assert translation.formula.startswith("mi(")

    tiny = pd.DataFrame(
        {
            "pupil": [3.0, 3.1, 3.2, 3.3],
            "time_ms": [0.0, 1.0, 2.0, 3.0],
            "participant_id": ["p1", "p1", "p2", "p2"],
        }
    )
    tiny_spec = p.specify_advanced_pupil_timecourse_model(
        tiny,
        temporal_structure="linear",
        autocorrelation="none",
        allow_high_complexity=True,
    )
    with pytest.raises(GP3BayesError, match="Too few complete observations"):
        p.fit_advanced_pupil_model_backend(
            tiny_spec,
            backend="analytic",
            chains=1,
            iter=20,
            warmup=5,
            cores=1,
        )


def test_cmdstan_wrapper_and_advanced_prediction_grid_guards(advanced_bundle):
    raw, specification, fit = advanced_bundle
    cmd = p.fit_advanced_pupil_model_cmdstanr(
        specification,
        chains=1,
        iter=30,
        warmup=10,
        cores=1,
        seed=11930,
    )
    assert cmd.backend == "cmdstanr"

    with pytest.raises(GP3BayesError, match="fitted advanced"):
        p._advanced_prediction_grid(object(), None)
    with pytest.raises(GP3BayesError, match="data frame"):
        p._advanced_prediction_grid(fit, "bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="exceeds"):
        p._advanced_prediction_grid(
            fit,
            pd.DataFrame({"time_ms": [0.0, 1.0]}),
            max_grid=1,
        )

    grid = p._advanced_prediction_grid(fit, None)
    assert "participant_id" in grid.columns
    assert "condition" not in grid.columns

    with pytest.raises(GP3BayesError, match="Prediction grid exceeds"):
        p._advanced_prediction_grid(fit, None, max_grid=1)

    long_data = pd.DataFrame(
        {
            "pupil": 3.0 + 0.001 * np.arange(205),
            "time_ms": np.arange(205, dtype=float),
            "participant_id": np.where(np.arange(205) % 2, "p1", "p2"),
        }
    )
    long_translation = replace(fit.translation, data=long_data)
    long_fit = replace(fit, translation=long_translation)
    long_grid = p._advanced_prediction_grid(long_fit, None, max_grid=5000)
    assert len(long_grid) <= 200


def test_advanced_prediction_grid_reference_and_measurement_paths(advanced_bundle):
    _, _, fit = advanced_bundle
    raw = _advanced_raw(8)
    raw["cov"] = np.linspace(0.0, 1.0, len(raw))
    raw["aux"] = np.linspace(1.0, 2.0, len(raw))
    raw["response_se"] = 0.1
    missingness = p.create_pupil_missingness_spec(auxiliary_predictors=("aux",))
    measurement = p.create_pupil_measurement_model(response_error="response_se")
    specification = p.specify_advanced_pupil_timecourse_model(
        raw,
        temporal_structure="linear",
        autocorrelation="none",
        covariates=("cov", "response_se"),
        measurement_model=measurement,
        missingness_model=missingness,
        allow_high_complexity=True,
    )
    adapted = replace(
        fit,
        specification=specification,
        translation=replace(
            fit.translation,
            specification=specification,
            data=specification.data.copy(),
        ),
    )
    grid = p._advanced_prediction_grid(adapted, None)
    assert {"cov", "aux", "response_se", "participant_id"}.issubset(grid.columns)

    protected_cov = replace(specification, covariates=("time_ms",))
    protected_fit = replace(
        adapted,
        specification=protected_cov,
        translation=replace(
            adapted.translation,
            specification=protected_cov,
            data=protected_cov.data.copy(),
        ),
    )
    protected_grid = p._advanced_prediction_grid(protected_fit, None)
    assert "time_ms" in protected_grid.columns

    raw2 = raw.copy()
    raw2["cov_se"] = 0.2
    measurement2 = p.create_pupil_measurement_model(
        covariate_errors={"cov": "cov_se"},
    )
    specification2 = p.specify_advanced_pupil_timecourse_model(
        raw2,
        temporal_structure="linear",
        autocorrelation="none",
        covariates=("cov",),
        measurement_model=measurement2,
        allow_high_complexity=True,
    )
    adapted2 = replace(
        fit,
        specification=specification2,
        translation=replace(
            fit.translation,
            specification=specification2,
            data=specification2.data.copy(),
        ),
    )
    grid2 = p._advanced_prediction_grid(adapted2, None)
    assert "cov_se" in grid2.columns
    table = p.pupil_measurement_uncertainty_table(adapted2)
    assert set(table["role"]) == {"predictor"}


def test_gp_missing_basis_and_missingness_fallbacks(advanced_bundle):
    raw, specification, fit = advanced_bundle
    gp_specification = replace(
        specification,
        temporal_structure="gaussian_process",
        gp_spec=p.create_pupil_gp_spec(),
    )
    gp_fit = replace(fit, specification=gp_specification)
    with pytest.raises(GP3BayesError, match="No GP basis"):
        p.pupil_gp_hyperparameters(gp_fit)

    with pytest.raises(GP3BayesError, match="advanced specification"):
        p.audit_pupil_missingness(object())  # type: ignore[arg-type]

    miss_raw = raw.copy()
    miss_raw["aux"] = np.linspace(0.0, 1.0, len(miss_raw))
    missingness = p.create_pupil_missingness_spec(auxiliary_predictors=("aux",))
    miss_spec = p.specify_advanced_pupil_timecourse_model(
        miss_raw,
        temporal_structure="linear",
        autocorrelation="none",
        missingness_model=missingness,
        allow_high_complexity=True,
    )
    constant_time = miss_raw.copy()
    constant_time["time_ms"] = 1.0
    constant_spec = replace(miss_spec, data=constant_time)
    audit = p.audit_pupil_missingness(constant_spec)
    assert not audit.by_time.empty


def test_advanced_summary_identifiability_and_calibration_guards(advanced_bundle):
    _, specification, fit = advanced_bundle
    summary = p.summarise_pupil_posterior(fit)
    assert summary.outcome_unit == "unknown"

    ident = p.audit_advanced_pupil_identifiability(specification)
    assert "minimum_condition_rows" not in set(ident.table["check"])

    ar_spec = p.specify_advanced_pupil_timecourse_model(
        _advanced_raw(8),
        temporal_structure="linear",
        autocorrelation="ar1",
        allow_high_complexity=True,
    )
    ident_ar = p.audit_advanced_pupil_identifiability(ar_spec)
    assert "arma_series_support" in set(ident_ar.table["check"])

    with pytest.raises(GP3BayesError, match="advanced pupil fit"):
        p.audit_pupil_predictive_calibration(object(), pd.DataFrame())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="data frame"):
        p.audit_pupil_predictive_calibration(fit, [])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="contain the pupil response"):
        p.audit_pupil_predictive_calibration(
            fit,
            pd.DataFrame({"time_ms": [0.0], "participant_id": ["p1"]}),
        )


def _binocular_frame(n_time: int, *, item: bool = False, covariate: bool = False) -> pd.DataFrame:
    times = np.tile(np.arange(n_time, dtype=float), 2)
    conditions = np.repeat(["control", "treatment"], n_time)
    data = pd.DataFrame(
        {
            "pupil_left": 3.0 + 0.01 * times,
            "pupil_right": 3.02 + 0.01 * times,
            "participant_id": np.where(np.arange(len(times)) % 2, "p1", "p2"),
            "time_ms": times,
            "condition": conditions,
        }
    )
    if item:
        data["item_id"] = np.where(np.arange(len(data)) % 2, "i1", "i2")
    if covariate:
        data["cov"] = np.linspace(0.0, 1.0, len(data))
    return data


def test_binocular_optional_mapping_and_support_guards():
    optional = _binocular_frame(8, item=True)
    prepared = p.prepare_binocular_pupil_timecourse(
        optional,
        trial_col=None,
        item_col="item_id",
    )
    assert prepared.mapping["trial"] is None
    assert prepared.mapping["item"] == "item_id"

    low = p.prepare_binocular_pupil_timecourse(
        _binocular_frame(4),
        trial_col=None,
    )
    with pytest.raises(GP3BayesError, match="Too little condition-specific temporal support"):
        p.specify_binocular_pupil_model(low, temporal_structure="smooth")

    medium = p.prepare_binocular_pupil_timecourse(
        _binocular_frame(6),
        trial_col=None,
    )
    with pytest.raises(GP3BayesError, match="exceeds the governed support"):
        p.specify_binocular_pupil_model(
            medium,
            temporal_structure="smooth",
            smooth_basis_dimension=6,
        )

    large = p.prepare_binocular_pupil_timecourse(
        _binocular_frame(301),
        trial_col=None,
    )
    with pytest.raises(GP3BayesError, match="Exact GP exceeds"):
        p.specify_binocular_pupil_model(
            large,
            temporal_structure="gaussian_process",
            gp_spec=p.create_pupil_gp_spec(basis="exact"),
            allow_high_complexity=False,
        )


def test_binocular_grid_helpers_and_disabled_correlation():
    data = _binocular_frame(205, covariate=True)
    prepared = p.prepare_binocular_pupil_timecourse(
        data,
        trial_col=None,
        covariates=("cov",),
    )
    fake_fit = SimpleNamespace(
        specification=SimpleNamespace(prepared=prepared),
    )
    grid = p._binocular_grid(fake_fit, max_grid=5000)
    assert len(grid) <= 400
    assert "cov" in grid.columns
    assert "trial_id" not in grid.columns

    fake_base_fit = SimpleNamespace(
        specification=SimpleNamespace(covariates=("cov",)),
    )
    base_grid = p._binocular_to_base_grid(fake_base_fit, grid, prepared.mapping)
    assert "cov" in base_grid.columns

    specification = p.specify_binocular_pupil_model(
        p.prepare_binocular_pupil_timecourse(
            _binocular_frame(8),
            trial_col=None,
        ),
        temporal_structure="linear",
        residual_correlation=False,
    )
    binocular_fit = p.BinocularPupilFit(
        specification,
        None,  # type: ignore[arg-type]
        "analytic",
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        np.array([0.0]),
    )
    with pytest.raises(GP3BayesError, match="disabled"):
        p.pupil_binocular_correlation(binocular_fit)


def test_lfo_empty_split_guard(advanced_bundle):
    _, _, fit = advanced_bundle
    plan = p.PupilLFOPlan(
        pd.DataFrame(
            {
                "refit": [1],
                "train_through_index": [0],
                "test_from_index": [1],
                "test_through_index": [1],
            }
        ),
        0.6,
        1,
        1,
        1,
        fit,
    )
    with pytest.raises(GP3BayesError, match="empty train/test"):
        p.validate_pupil_leave_future_out(fit, plan, execute=True)


def test_plot_import_guard_gp_plot_and_generic_ppc(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("synthetic matplotlib failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(GP3BayesError, match="Matplotlib is required"):
        p._mpl_axes("synthetic")
    monkeypatch.setattr(builtins, "__import__", original_import)

    import matplotlib.pyplot as plt

    gp = p.PupilGPHyperparameters(
        pd.DataFrame(
            {
                "parameter": ["sdgp", "lscale"],
                "mean": [0.2, 1.0],
            }
        ),
        0.95,
        p.create_pupil_gp_spec(),
    )
    fig1 = p.plot_pupil_gp_hyperparameters(gp)
    plt.close(fig1)

    blank = pd.DataFrame()
    ppc = p.PupilPPC(
        trajectory=blank,
        distribution=blank,
        features=blank,
        residuals=pd.DataFrame({"residual": [0.1, -0.1, 0.05]}),
        residual_trajectory=blank,
        autocorrelation=blank,
        heterogeneity=blank,
        measurement_context=blank,
        probability=0.95,
        declared_window=None,
        unit="millimetres",
    )
    fig2 = p.plot_pupil_ppc(ppc, "residuals")
    plt.close(fig2)
