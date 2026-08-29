from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _data(n_time: int = 12) -> pd.DataFrame:
    rows = []
    for participant in ("p1", "p2"):
        for trial, condition in ((1, "A"), (2, "B")):
            for t in range(n_time):
                rows.append(
                    {
                        "participant_id": participant,
                        "trial_id": f"{participant}-t{trial}",
                        "item_id": f"i{trial}",
                        "condition": condition,
                        "event_time": float(t * 100),
                        "pupil_mm": (
                            3.0
                            + (0.15 if condition == "B" else 0.0)
                            + (0.05 if participant == "p2" else 0.0)
                            + 0.2 * np.sin(t / 2)
                        ),
                        "baseline": 3.0 + 0.02 * t,
                        "luminance": 50.0 + t,
                        "baseline_se": 0.05 + 0.001 * t,
                        "response_se": 0.03 + 0.001 * t,
                    }
                )
    return pd.DataFrame(rows)


def _spec(
    *,
    temporal: str = "smooth",
    residual: str = "constant",
    gp_spec=None,
    measurement=None,
    missingness=None,
    covariates=("baseline",),
):
    return p.specify_advanced_pupil_timecourse_model(
        _data(),
        temporal_structure=temporal,
        family="gaussian",
        residual_scale=residual,
        gp_spec=gp_spec,
        covariates=covariates,
        measurement_model=measurement,
        missingness_model=missingness,
        smooth_basis_dimension=8,
    )


def _fit(specification, seed: int = 1601):
    X, names, metadata = p._advanced_feature_matrix(
        specification.data,
        specification,
    )
    rng = np.random.default_rng(seed)
    coefficients = np.linalg.lstsq(
        X,
        specification.data[str(specification.mapping["response"])].to_numpy(float),
        rcond=None,
    )[0]
    posterior = rng.normal(
        coefficients[None, :],
        0.025,
        size=(120, len(coefficients)),
    )
    sigma = np.abs(rng.normal(0.12, 0.01, size=120))
    residuals = (
        specification.data[str(specification.mapping["response"])].to_numpy(float)
        - X @ coefficients
    )
    translation = p.AdvancedPupilTranslation(
        specification=specification,
        formula="pupil ~ time",
        family=specification.family,
        priors=pd.DataFrame(),
        data=specification.data.copy(),
    )
    return p.AdvancedPupilFit(
        specification=specification,
        translation=translation,
        backend="analytic",
        coefficients=coefficients,
        coefficient_names=names,
        covariance=np.eye(len(coefficients)) * 0.01,
        residual_scale=float(np.std(residuals)),
        posterior_coefficients=posterior,
        posterior_sigma=sigma,
        residuals=residuals,
        log_likelihood=np.zeros((120, len(specification.data))),
        design_metadata=metadata,
        sampling={"seed": seed},
    )


def test_advanced_measurement_missingness_gp_arma_distribution_constructors():
    measurement = p.create_pupil_measurement_model(
        baseline_error="baseline_se",
        response_error="response_se",
    )
    assert measurement.covariate_errors["baseline"] == "baseline_se"
    assert measurement.response_error == "response_se"

    custom = p.create_pupil_measurement_model(covariate_errors={"luminance": "baseline_se"})
    assert custom.covariate_errors["luminance"] == "baseline_se"

    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(covariate_errors={"": "x"})
    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(
            baseline_error="a",
            covariate_errors={"baseline": "b"},
        )

    missing = p.create_pupil_missingness_spec(
        response="exclude",
        predictors=("baseline", "baseline"),
        auxiliary_predictors=("luminance",),
    )
    assert missing.predictors == ("baseline",)
    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(response="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(assumptions="MNAR")

    for kernel in ("matern32", "matern52", "exp_quad"):
        gp = p.create_pupil_gp_spec(kernel=kernel, basis="approximate", k=10)
        assert gp.k == 10
    exact = p.create_pupil_gp_spec(basis="exact")
    assert exact.k is None
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(kernel="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(basis="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(scale=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(k=4)

    assert p.create_pupil_arma_spec(1, 0).p == 1
    assert p.create_pupil_arma_spec(1, 1, covariance=True).covariance
    for args in ((0, 0, False), (4, 0, False), (1, 3, False)):
        with pytest.raises(GP3BayesError):
            p.create_pupil_arma_spec(*args)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(1, 0, covariance=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(2, 0, covariance=True)

    dist = p.specify_pupil_distribution("student", "time")
    assert p.pupil_distribution_table(dist).loc[0, "family"] == "student"
    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution("bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution("gaussian", "bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_distribution_table(object())  # type: ignore[arg-type]


def test_advanced_mapping_specification_and_complexity_branches():
    data = _data()
    mapped = p.pupil_advanced_mapping_table(data)
    assert set(mapped["role"]) >= {"response", "time", "participant"}

    _, mapping = p._advanced_mapping(data)
    assert mapping["response"] == "pupil_mm"
    with pytest.raises(GP3BayesError):
        p._advanced_mapping(pd.DataFrame())
    with pytest.raises(GP3BayesError):
        p._advanced_mapping(pd.DataFrame({"pupil_mm": [3.0]}))

    assert p._ac_spec("none") is None
    assert p._ac_spec("ar1").p == 1
    assert p._ac_spec("ar2").p == 2
    assert p._ac_spec("arma11").q == 1
    declared = p.create_pupil_arma_spec(1, 0)
    assert p._ac_spec(declared) is declared
    with pytest.raises(GP3BayesError):
        p._ac_spec("bad")

    spec = _spec()
    table = p.pupil_advanced_specification_table(spec)
    assert table.loc[0, "family"] == "gaussian"
    audit = p.audit_pupil_computational_budget(spec)
    assert audit.overall_status in {"ok", "review", "high"}

    gp_spec = _spec(
        temporal="gaussian_process",
        gp_spec=p.create_pupil_gp_spec(basis="exact"),
    )
    gp_audit = p.audit_pupil_computational_budget(gp_spec)
    assert "exact_gp" in set(gp_audit.checks["check"])

    approx = _spec(
        temporal="gaussian_process",
        gp_spec=p.create_pupil_gp_spec(k=120),
    )
    approx_audit = p.audit_pupil_computational_budget(approx)
    assert "approximate_gp" in set(approx_audit.checks["check"])

    arma = p.specify_advanced_pupil_timecourse_model(
        data,
        autocorrelation="arma11",
        covariates=("baseline",),
    )
    arma_audit = p.audit_pupil_computational_budget(arma)
    assert "arma_order" in set(arma_audit.checks["check"])

    with pytest.raises(GP3BayesError):
        p.audit_pupil_computational_budget(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_advanced_specification_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, temporal_structure="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, family="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, residual_scale="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            family="student",
            autocorrelation="ar1",
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data.drop(columns="condition"),
            residual_scale="condition",
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, covariates=("missing",))
    bad_text = data.assign(text=["x"] * len(data))
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(bad_text, covariates=("text",))
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, covariates=("pupil_mm",))
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(data, prior_scales={"x": 0})


def test_advanced_sensitivity_suite_materializes_every_dimension():
    spec = _spec()
    suite = p.create_pupil_advanced_sensitivity_suite(spec)
    assert suite.scenarios.iloc[0]["scenario"] == "baseline"
    assert p.materialize_pupil_advanced_sensitivity_scenario(suite, "baseline") is spec

    dimensions = set()
    for scenario in suite.scenarios["scenario"].astype(str):
        materialized = p.materialize_pupil_advanced_sensitivity_scenario(
            suite,
            scenario,
        )
        assert isinstance(materialized, p.AdvancedPupilSpecification)
        row = suite.scenarios.loc[suite.scenarios["scenario"] == scenario].iloc[0]
        dimensions.add(str(row["dimension"]))
    assert {"family", "residual_scale", "autocorrelation", "temporal_structure"}.issubset(
        dimensions
    )

    gp_spec = _spec(
        temporal="gaussian_process",
        gp_spec=p.create_pupil_gp_spec("matern32", "approximate", 10),
    )
    gp_suite = p.create_pupil_advanced_sensitivity_suite(gp_spec)
    gp_rows = gp_suite.scenarios.loc[gp_suite.scenarios["dimension"] == "gp_kernel"]
    assert len(gp_rows) >= 1
    gp_alt = p.materialize_pupil_advanced_sensitivity_scenario(
        gp_suite,
        str(gp_rows.iloc[0]["scenario"]),
    )
    assert gp_alt.gp_spec.kernel != gp_spec.gp_spec.kernel

    with pytest.raises(GP3BayesError):
        p.create_pupil_advanced_sensitivity_suite(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_advanced_sensitivity_suite(spec, include=("bad",))
    with pytest.raises(GP3BayesError):
        p.materialize_pupil_advanced_sensitivity_scenario(object(), "baseline")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.materialize_pupil_advanced_sensitivity_scenario(suite, "missing")


def test_pupil_model_card_measurement_and_missingness_audits():
    measurement = p.create_pupil_measurement_model(
        baseline_error="baseline_se",
        response_error="response_se",
    )
    missingness = p.create_pupil_missingness_spec(
        predictors=("baseline",),
        auxiliary_predictors=("luminance",),
    )
    spec = _spec(measurement=measurement, missingness=missingness)

    card = p.pupil_model_card(spec)
    assert len(card.table) == 15
    assert p.pupil_model_card_table(card).equals(card.table)
    fit_like = SimpleNamespace(
        specification=spec,
        fit_performed=True,
        backend="analytic",
    )
    fit_card = p.pupil_model_card(fit_like)
    assert "True" in set(fit_card.table["value"].astype(str))
    with pytest.raises(GP3BayesError):
        p.pupil_model_card(object())

    uncertainty = p.pupil_measurement_uncertainty_table(measurement)
    assert set(uncertainty["role"]) == {"predictor", "response"}
    assert len(p.pupil_measurement_uncertainty_table(spec)) == 2
    assert p.pupil_measurement_uncertainty_table(None).empty

    audit = p.audit_pupil_measurement_model(spec)
    assert audit.status == "pass"

    review_data = spec.data.copy()
    review_data.loc[0, "baseline_se"] = np.nan
    review = p.audit_pupil_measurement_model(replace(spec, data=review_data))
    assert review.status == "review"

    fail_data = spec.data.copy()
    fail_data.loc[0, "baseline_se"] = 0.0
    failure = p.audit_pupil_measurement_model(replace(spec, data=fail_data))
    assert failure.status == "failure"

    with pytest.raises(GP3BayesError):
        p.audit_pupil_measurement_model(_spec())
    with pytest.raises(GP3BayesError):
        p.audit_pupil_measurement_model(object())  # type: ignore[arg-type]

    missing_data = spec.data.copy()
    missing_data.loc[missing_data.index[::7], "pupil_mm"] = np.nan
    missing_data.loc[missing_data.index[::9], "baseline"] = np.nan
    missing_spec = replace(spec, data=missing_data)
    missing_audit = p.audit_pupil_missingness(missing_spec)
    assert set(missing_audit.table["role"]) == {
        "response",
        "modelled_predictor",
        "auxiliary",
    }
    assert not missing_audit.by_time.empty
    assert p.pupil_missingness_table(missing_audit).equals(missing_audit.table)

    flat_time = missing_data.copy()
    flat_time["event_time"] = 0.0
    flat = p.audit_pupil_missingness(replace(missing_spec, data=flat_time))
    assert len(flat.by_time) == 1

    with pytest.raises(GP3BayesError):
        p.audit_pupil_missingness(_spec())
    with pytest.raises(GP3BayesError):
        p.pupil_missingness_table(object())  # type: ignore[arg-type]


def test_temporal_dependence_mapping_acf_and_tables():
    data = _data()
    audit = p.audit_pupil_temporal_dependence(data, max_lag=4)
    assert len(audit.series) == 4
    assert p.pupil_autocorrelation_table(audit, "summary").equals(audit.summary)
    assert p.pupil_autocorrelation_table(audit, "series").equals(audit.series)

    assert np.isnan(p._acf_values(np.array([1.0]), 3)).all()
    assert np.isnan(p._acf_values(np.ones(10), 3)).all()
    assert len(p._acf_values(np.arange(10.0), 3)) == 3

    spec = _spec()
    d1, m1 = p._pupil_data_mapping(spec)
    assert len(d1) == len(data)
    assert m1["response"] == "pupil_mm"

    simulation = p.AdvancedPupilSimulation(data, {"seed": 1})
    d2, m2 = p._pupil_data_mapping(simulation)
    assert len(d2) == len(data)
    assert m2["time"] == "event_time"

    with pytest.raises(GP3BayesError):
        p._pupil_data_mapping(object())
    with pytest.raises(GP3BayesError):
        p.pupil_autocorrelation_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_autocorrelation_table(audit, "bad")  # type: ignore[arg-type]


def test_advanced_fit_residual_scale_gp_diagnostics_autocorrelation_and_spectrum():
    spec = _spec()
    fit = _fit(spec)

    diag = p.diagnose_advanced_pupil_fit(fit, ess_threshold=400)
    assert (diag.metrics["status"] == "review").any()
    diag_pass = p.diagnose_advanced_pupil_fit(fit, ess_threshold=50)
    assert (diag_pass.metrics["status"] == "pass").all()
    with pytest.raises(GP3BayesError):
        p.diagnose_advanced_pupil_fit(object())  # type: ignore[arg-type]

    residual = p.estimate_pupil_residual_scale(fit, ndraws=40)
    table = p.pupil_residual_scale_table(residual)
    assert len(table) == len(residual.grid)
    assert len(residual.draws) == 40
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_residual_scale(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_residual_scale_table(object())  # type: ignore[arg-type]

    condition_time_spec = _spec(residual="condition_time")
    condition_time_fit = _fit(condition_time_spec, 1610)
    ct = p.estimate_pupil_residual_scale(condition_time_fit, ndraws=30)
    assert np.isfinite(ct.draws).all()

    all_nan_fit = replace(
        condition_time_fit,
        residuals=np.full(len(condition_time_spec.data), np.nan),
    )
    nan_scale = p.estimate_pupil_residual_scale(all_nan_fit, ndraws=10)
    assert np.isfinite(nan_scale.draws).all()

    gp_specification = _spec(
        temporal="gaussian_process",
        gp_spec=p.create_pupil_gp_spec("matern32", "approximate", 10),
    )
    gp_fit = _fit(gp_specification, 1620)
    hyper = p.pupil_gp_hyperparameters(gp_fit)
    assert set(hyper.table["parameter"]) == {"sdgp", "lscale"}
    assert p.pupil_gp_table(hyper).equals(hyper.table)
    with pytest.raises(GP3BayesError):
        p.pupil_gp_hyperparameters(fit)
    with pytest.raises(GP3BayesError):
        p.pupil_gp_table(object())  # type: ignore[arg-type]

    comparison = p.compare_pupil_autocorrelation(
        {"m1": fit, "m2": _fit(spec, 1630)},
        max_lag=3,
        ndraws=30,
    )
    assert set(comparison.table["model"]) == {"m1", "m2"}
    with pytest.raises(GP3BayesError):
        p.compare_pupil_autocorrelation(fit)
    with pytest.raises(GP3BayesError):
        p.compare_pupil_autocorrelation({"m1": fit, "bad": object()})

    spectrum = p.pupil_residual_spectrum(fit, ndraws=30)
    assert spectrum.n_series == 4
    assert len(spectrum.table) == 100
    with pytest.raises(GP3BayesError):
        p.pupil_residual_spectrum(object())  # type: ignore[arg-type]

    short_spec = p.specify_advanced_pupil_timecourse_model(
        _data(n_time=4),
        covariates=("baseline",),
    )
    short_fit = _fit(short_spec, 1640)
    with pytest.raises(GP3BayesError, match="Too few"):
        p.pupil_residual_spectrum(short_fit, ndraws=20)


def test_advanced_feature_helpers_and_reference_values():
    spec = _spec()
    X, names, meta = p._advanced_feature_matrix(spec.data, spec)
    X2, names2, meta2 = p._advanced_feature_matrix(spec.data, spec, meta)
    assert X.shape == X2.shape
    assert names == names2
    assert meta2["time_center"] == meta["time_center"]

    gp_specification = _spec(
        temporal="gaussian_process",
        gp_spec=p.create_pupil_gp_spec("matern52", "approximate", 8),
    )
    Xgp, names_gp, meta_gp = p._advanced_feature_matrix(
        gp_specification.data,
        gp_specification,
    )
    assert any(name.startswith("gp_basis[") for name in names_gp)
    Xgp2, names_gp2, _ = p._advanced_feature_matrix(
        gp_specification.data,
        gp_specification,
        meta_gp,
    )
    assert Xgp.shape == Xgp2.shape
    assert names_gp == names_gp2

    series = p._series_keys(spec.data, spec.mapping)
    assert series.nunique() == 4
    assert p._advanced_reference_value(pd.Series([1.0, 2.0, 3.0])) == 2.0
    assert p._advanced_reference_value(pd.Series(["b", "a"])) == "b"
    assert np.isnan(p._advanced_reference_value(pd.Series([], dtype=float)))
