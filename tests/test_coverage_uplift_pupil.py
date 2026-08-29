from __future__ import annotations

from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gp3bayespy import pupil as p
from gp3bayespy.exceptions import GP3BayesError


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _base_fixture():
    sim = p.simulate_pupil_timecourse(
        n_participants=4,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.6),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0.0,
        include_gaze=True,
        include_luminance=True,
        seed=211,
    )
    contract = p.create_pupil_contract(
        "pupil_mm",
        "participant_id",
        "trial_id",
        "event_time",
        "millimetres",
        10,
        item_col="item_id",
        condition_col="condition",
        validity_col="valid",
        interpolation_col="interpolated",
        blink_col="blink",
        gaze_x_col="gaze_x",
        gaze_y_col="gaze_y",
        luminance_col="luminance",
        baseline_window=(-0.2, 0.0),
        source_vendor="synthetic",
    )
    prepared = p.prepare_pupil_timecourse(
        sim.data,
        contract,
        baseline_operation="subtract",
        baseline_window=(-0.2, 0.0),
    )
    spec = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="smooth",
        smooth_basis_dimension=5,
        condition_trajectory=True,
        autocorrelation="ar1",
        covariates=(),
    )
    fit = p.fit_pupil_model_backend(
        spec, backend="analytic", chains=1, iter=140, warmup=70, cores=1, seed=212
    )
    return sim, contract, prepared, spec, fit


def _advanced_fixture():
    sim = p.simulate_advanced_pupil_timecourse(
        n_participants=4,
        trials_per_participant=3,
        time_points=10,
        missing_fraction=0.0,
        outlier_fraction=0.0,
        seed=221,
    )
    dt = np.diff(np.sort(sim.data["time_ms"].unique())).mean() / 1000
    contract = p.create_pupil_contract(
        "pupil",
        "participant_id",
        "trial_id",
        "time_ms",
        "millimetres",
        1 / dt,
        time_unit="milliseconds",
        condition_col="condition",
        luminance_col="luminance",
    )
    prepared = p.prepare_pupil_timecourse(sim.data, contract)
    spec = p.specify_advanced_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        covariates=("luminance",),
        allow_high_complexity=True,
    )
    fit = p.fit_advanced_pupil_model(spec, chains=1, iter=140, warmup=70, cores=1, seed=222)
    return sim, prepared, spec, fit


def test_base_pupil_contract_simulation_preparation_specification_fit_prediction_and_estimands():
    sim, contract, prepared, spec, fit = _base_fixture()
    assert len(sim.data) > 0
    assert p.audit_pupil_readiness(prepared).status in {"pass", "review", "failure"}
    assert not p.pupil_specification_table(spec).empty
    assert p.translate_pupil_model_to_brms(spec)["compile"] is False
    prior = p.check_pupil_prior_predictive(spec, execute=False, draws=60)
    assert prior.executed is False

    expected = p.predict_pupil_trajectory(fit, type="expected", ndraws=40)
    predictive = p.predict_pupil_trajectory(fit, type="posterior_predictive", ndraws=35)
    linear = p.predict_pupil_trajectory(fit, type="linear", ndraws=30)
    for prediction in (expected, predictive, linear):
        assert prediction.draws.shape[0] > 0

    pointwise = p.estimate_pupil_trajectory(expected, probability=0.9, interval="pointwise")
    simultaneous = p.estimate_pupil_trajectory(expected, probability=0.9, interval="simultaneous")
    assert not p.pupil_trajectory_table(pointwise).empty
    assert simultaneous.finite_grid_qualification is True

    time_values = expected.grid[".event_time"].to_numpy(float)
    window = (float(np.quantile(time_values, 0.25)), float(np.quantile(time_values, 0.75)))
    estimands = (
        p.estimate_pupil_window(expected, window=window),
        p.estimate_pupil_auc(expected, window=window),
        p.estimate_pupil_peak(expected, window=window),
        p.estimate_pupil_peak_latency(expected, window=window),
    )
    levels = tuple(str(v) for v in pd.unique(expected.grid[".condition"]))
    contrast = p.pupil_condition_contrast(expected, levels[:2], probability=0.9)
    assert not contrast.table.empty
    assert all(not x.table.empty for x in estimands)

    figures = [
        p.plot_pupil_readiness(p.audit_pupil_readiness(prepared)),
        p.plot_pupil_observed_trajectory(prepared, summary=True),
        p.plot_pupil_observed_trajectory(prepared, summary=False),
        p.plot_pupil_posterior_trajectory(pointwise),
        p.plot_pupil_estimand(contrast),
    ]
    figures.extend(p.plot_pupil_estimand(x) for x in estimands)
    assert all(fig.axes for fig in figures)


def test_base_pupil_diagnostics_ppc_validation_sensitivity_and_measurement_paths():
    sim, contract, prepared, spec, fit = _base_fixture()
    diagnostics = p.diagnose_pupil_fit(fit, ndraws=40, max_lag=4)
    acf = p.pupil_residual_acf(diagnostics)
    assert not acf.empty

    ppc = p.check_pupil_posterior_predictive(
        fit,
        ndraws=35,
        probability=0.9,
        window=(0.0, 0.4),
    )
    for component in (
        "trajectory",
        "distribution",
        "features",
        "residuals",
        "residual_trajectory",
        "autocorrelation",
        "heterogeneity",
        "measurement_context",
    ):
        assert isinstance(p.pupil_ppc_table(ppc, component), pd.DataFrame)
    plots = [
        p.plot_pupil_ppc(ppc, "trajectory"),
        p.plot_pupil_ppc(ppc, "distribution"),
        p.plot_pupil_ppc(ppc, "features"),
        p.plot_pupil_ppc(ppc, "autocorrelation"),
        p.plot_pupil_residual_acf(diagnostics),
    ]

    posterior = p.summarise_pupil_posterior(fit, probability=0.9)
    assert not posterior.table.empty

    plan_trial = p.create_pupil_validation_plan(
        prepared, target="new_trial_known_participant", K=2, seed=3
    )
    plan_participant = p.create_pupil_validation_plan(
        prepared, target="new_participant", K=2, seed=4
    )
    plan_future = p.create_pupil_validation_plan(
        prepared, target="future_segment", future_fraction=0.25, seed=5
    )
    plan_sample = p.create_pupil_validation_plan(
        prepared, target="new_sample_known_trial", K=3, seed=6
    )
    for plan in (plan_trial, plan_participant, plan_future, plan_sample):
        validation = p.validate_pupil_model(fit, plan, execute=False)
        assert validation.executed is False
        plots.append(p.plot_pupil_validation(validation))

    suite = p.create_pupil_sensitivity_suite(
        spec,
        smooth_basis_dimensions=(4, 6),
        autocorrelation=("none",),
        analysis_windows=((0.0, 0.3),),
    )
    assert p.pupil_sensitivity_table(suite).shape[0] == 4
    plots.append(p.plot_pupil_sensitivity(suite))

    measurement = p.audit_pupil_measurement_context(prepared)
    assert not p.pupil_measurement_audit_table(measurement).empty
    plots.append(p.plot_pupil_measurement_audit(measurement))
    assert all(fig.axes for fig in plots)


def test_gazepoint_schema_and_functional_dynamic_contrast_paths():
    raw = pd.DataFrame(
        {
            "TIME": [0.0, 0.1, 0.2],
            "LPD": [3.1, 3.2, 3.3],
            "LPV": [1, 1, 1],
            "LPOGX": [0.4, 0.5, 0.6],
            "LPOGY": [0.4, 0.5, 0.6],
        }
    )
    schema = p.inspect_gazepoint_pupil_schema(raw)
    assert schema.status == "single_pupil_candidate"
    assert not p.gazepoint_pupil_mapping_table(schema).empty

    _, prepared, spec, fit = _advanced_fixture()
    trajectory = p.predict_advanced_pupil_trajectory(fit, ndraws=50)
    derivative = p.estimate_pupil_trajectory_derivative(trajectory, order=1, probability=0.9)
    assert not p.pupil_trajectory_derivative_table(derivative).empty

    levels = tuple(str(v) for v in pd.unique(trajectory.grid[".condition"]))
    contrast = p.estimate_pupil_dynamic_contrast(trajectory, levels[:2], threshold=0.01)
    assert not p.pupil_dynamic_contrast_table(contrast).empty
    duration = p.estimate_pupil_threshold_duration(
        contrast, direction="absolute", threshold=0.01, probability=0.9
    )
    assert not duration.summary.empty

    figures = (
        p.plot_pupil_trajectory_derivative(derivative),
        p.plot_pupil_dynamic_contrast(contrast),
    )
    assert all(fig.axes for fig in figures)


def test_advanced_pupil_diagnostics_calibration_model_comparison_and_plot_matrix():
    sim, prepared, spec, fit = _advanced_fixture()
    prior = p.create_advanced_pupil_prior_specification(spec)
    prior_check = p.check_advanced_pupil_prior_predictive(
        spec, chains=1, iter=80, warmup=40, seed=7
    )
    assert not prior.table.empty and prior_check.executed

    trajectory = p.predict_advanced_pupil_trajectory(fit, ndraws=45)
    temporal = p.audit_pupil_temporal_dependence(prepared, max_lag=4)
    diagnostics = p.diagnose_advanced_pupil_fit(fit)
    assert diagnostics is not None
    identifiability = p.audit_advanced_pupil_identifiability(spec)
    calibration = p.audit_pupil_predictive_calibration(fit, spec.data.head(30), ndraws=25)

    alternate_spec = p.specify_advanced_pupil_timecourse_model(
        prepared,
        temporal_structure="smooth",
        smooth_basis_dimension=5,
        covariates=("luminance",),
        allow_high_complexity=True,
    )
    alternate = p.fit_advanced_pupil_model(
        alternate_spec, chains=1, iter=120, warmup=60, cores=1, seed=223
    )
    model_set = p.create_pupil_model_set({"linear": fit, "smooth": alternate})
    comparison = p.compare_pupil_models(model_set)
    weights = p.pupil_model_weights(comparison)
    assert np.isclose(weights["weight"].sum(), 1.0)

    lfo = p.create_pupil_lfo_plan(fit, initial_fraction=0.6, horizon=2, step=2, max_refits=2)
    validation = p.validate_pupil_leave_future_out(fit, lfo, execute=False)

    plots = (
        p.plot_advanced_pupil_simulation(sim, observed=True),
        p.plot_advanced_pupil_simulation(sim, observed=False),
        p.plot_advanced_pupil_trajectory(trajectory),
        p.plot_pupil_temporal_dependence(temporal),
        p.plot_pupil_identifiability_audit(identifiability),
        p.plot_pupil_predictive_calibration(calibration),
        p.plot_pupil_model_complexity(spec),
        p.plot_pupil_model_comparison(comparison),
        p.plot_pupil_lfo(lfo),
        p.plot_pupil_lfo(validation),
    )
    assert all(fig.axes for fig in plots)


def test_binocular_and_response_shape_paths_and_graphics():
    sim = p.simulate_binocular_pupil_timecourse(
        n_participants=4,
        trials_per_participant=3,
        time_points=10,
        missing_fraction=0.0,
        seed=231,
    )
    prepared = p.prepare_binocular_pupil_timecourse(sim.data)
    audit = p.audit_binocular_pupil_readiness(prepared)
    assert audit is not None
    spec = p.specify_binocular_pupil_model(
        prepared, temporal_structure="linear", smooth_basis_dimension=5
    )
    fit = p.fit_binocular_pupil_model(spec, chains=1, iter=120, warmup=60, cores=1, seed=232)
    trajectory = p.estimate_binocular_pupil_trajectory(fit, ndraws=30)
    assert not p.pupil_binocular_difference(trajectory).empty
    assert not p.pupil_binocular_correlation(fit).empty
    assert not p.pupil_binocular_agreement_table(trajectory).empty
    assert p.plot_binocular_pupil_trajectory(trajectory).axes

    response = p.simulate_pupil_response_shape(
        n_participants=4, trials_per_participant=3, time_points=12, seed=233
    )
    rspec = p.specify_pupil_response_shape_model(response.data)
    rfit = p.fit_pupil_response_shape_model(rspec, chains=1, iter=120, warmup=60, cores=1, seed=234)
    estimand = p.estimate_pupil_response_parameters(rfit)
    assert not p.pupil_response_parameter_table(estimand).empty
    assert p.plot_pupil_response_parameters(estimand).axes


def test_pupil_validation_error_and_guardrail_branches():
    sim, contract, prepared, spec, fit = _base_fixture()
    with pytest.raises(GP3BayesError, match="Unsupported pupil unit"):
        p.as_pupil_prediction_draws(
            np.ones((10, 2)),
            pd.DataFrame({".event_time": [0.0, 0.1]}),
            "bad-unit",
        )
    with pytest.raises(GP3BayesError, match="pointwise or simultaneous"):
        pred = p.predict_pupil_trajectory(fit, ndraws=20)
        p.estimate_pupil_trajectory(pred, interval="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="Unsupported pupil PPC component"):
        p.pupil_ppc_table(p.check_pupil_posterior_predictive(fit, ndraws=20), "bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="at least four"):
        tiny_prepared = replace(prepared, data=prepared.data.head(3).copy())
        p.create_pupil_validation_plan(tiny_prepared)
