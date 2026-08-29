from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _advanced_data(n_time: int = 12) -> pd.DataFrame:
    rows = []
    for participant in ("p1", "p2", "p3"):
        for trial, condition in ((1, "A"), (2, "B"), (3, "A")):
            for t in range(n_time):
                rows.append(
                    {
                        "participant_id": participant,
                        "trial_id": f"{participant}-t{trial}",
                        "item_id": f"i{trial}",
                        "condition": condition,
                        "event_time": float(t * 100),
                        "pupil_mm": (
                            3.0 + (0.15 if condition == "B" else 0.0) + 0.15 * np.sin(t / 2)
                        ),
                        "baseline": 3.0 + 0.01 * t,
                    }
                )
    return pd.DataFrame(rows)


def _advanced_fit(seed: int = 2901):
    spec = p.specify_advanced_pupil_timecourse_model(
        _advanced_data(),
        temporal_structure="linear",
        family="gaussian",
        residual_scale="constant",
        covariates=("baseline",),
    )
    fit = p.fit_advanced_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=140,
        warmup=40,
        cores=1,
        seed=seed,
    )
    return spec, fit


def test_executed_lfo_and_plot_adapter_tail():
    spec, fit = _advanced_fit()

    plan = p.create_pupil_lfo_plan(
        fit,
        initial_fraction=0.5,
        horizon=2,
        step=3,
        max_refits=2,
    )
    executed = p.validate_pupil_leave_future_out(
        fit,
        plan,
        execute=True,
        cores=1,
        seed=2910,
    )
    assert executed.executed
    assert executed.scores is not None
    assert len(executed.scores) >= 1

    cmp = p.compare_pupil_lfo({"a": executed, "b": executed})
    assert set(cmp.table["model"]) == {"a", "b"}

    with pytest.raises(GP3BayesError):
        p.validate_pupil_leave_future_out(fit, object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.compare_pupil_lfo(executed, executed)

    assert p.plot_pupil_lfo(plan).axes
    unexecuted = p.validate_pupil_leave_future_out(
        fit,
        plan,
        execute=False,
    )
    assert p.plot_pupil_lfo(unexecuted).axes
    assert p.plot_pupil_lfo(executed).axes
    assert p.plot_pupil_lfo(cmp).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_lfo(object())  # type: ignore[arg-type]

    sim = p.simulate_advanced_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        time_points=10,
        missing_fraction=0,
        outlier_fraction=0,
        seed=2920,
    )
    assert p.plot_advanced_pupil_simulation(sim, observed=True).axes
    assert p.plot_advanced_pupil_simulation(sim, observed=False).axes
    with pytest.raises(GP3BayesError):
        p.plot_advanced_pupil_simulation(object())  # type: ignore[arg-type]

    trajectory = p.predict_advanced_pupil_trajectory(
        fit,
        type="expected",
        ndraws=20,
    )
    assert p.plot_advanced_pupil_trajectory(trajectory, 0.9).axes

    residual = p.estimate_pupil_residual_scale(
        fit,
        ndraws=20,
        probability=0.9,
    )
    assert p.plot_pupil_residual_scale(residual).axes

    temporal = p.audit_pupil_temporal_dependence(fit, max_lag=3)
    assert p.plot_pupil_temporal_dependence(temporal).axes

    constant_temporal = p.PupilTemporalDependenceAudit(
        pd.DataFrame({"lag1": [0.5, 0.5, 0.5]}),
        pd.DataFrame(),
        3,
    )
    assert p.plot_pupil_temporal_dependence(constant_temporal).axes

    empty_temporal = p.PupilTemporalDependenceAudit(
        pd.DataFrame({"lag1": [np.nan, np.nan]}),
        pd.DataFrame(),
        3,
    )
    assert p.plot_pupil_temporal_dependence(empty_temporal).axes

    ac = p.PupilAutocorrelationComparison(
        pd.DataFrame(
            {
                "model": ["a", "a", "b", "b"],
                "lag": [1, 2, 1, 2],
                "median_acf": [0.2, 0.1, 0.3, 0.15],
                "median_abs_acf": [0.2, 0.1, 0.3, 0.15],
            }
        ),
        2,
    )
    assert p.plot_pupil_autocorrelation_comparison(ac, absolute=True).axes
    assert p.plot_pupil_autocorrelation_comparison(ac, absolute=False).axes

    missing = p.PupilMissingnessAudit(
        pd.DataFrame(
            {
                "variable": ["pupil", "baseline"],
                "missing_fraction": [0.1, 0.0],
            }
        ),
        pd.DataFrame({"time": [0.0], "missing_fraction": [0.1]}),
        "MAR",
    )
    assert p.plot_pupil_missingness(missing).axes

    measurement = p.PupilMeasurementAudit05(
        pd.DataFrame(
            {
                "variable": ["baseline"],
                "missing_fraction": [0.0],
            }
        ),
        "pass",
    )
    assert p.plot_pupil_measurement_uncertainty(measurement).axes

    bino = p.BinocularPupilTrajectory(
        pd.DataFrame({"time_ms": [0.0, 100.0, 200.0]}),
        np.tile([3.0, 3.1, 3.2], (20, 1)),
        np.tile([3.05, 3.15, 3.25], (20, 1)),
        0.9,
        {"time": "time_ms"},
    )
    assert p.plot_binocular_pupil_trajectory(bino).axes
    assert p.plot_binocular_pupil_trajectory(bino, 0.8).axes

    model_set = p.PupilModelSet({"a": fit, "b": fit}, "future_segment")
    model_cmp = p.PupilModelComparison(
        "loo",
        "future_segment",
        {},
        pd.DataFrame(
            {
                "model": ["a", "b"],
                "elpd": [-10.0, -11.0],
                "se": [1.0, 1.2],
                "elpd_diff": [0.0, -1.0],
            }
        ),
        model_set,
    )
    assert p.plot_pupil_model_comparison(model_cmp).axes

    params = p.PupilResponseParameters(
        pd.DataFrame(
            {
                "parameter": ["baseline", "onset"],
                "mean": [3.0, 250.0],
            }
        ),
        0.95,
    )
    assert p.plot_pupil_response_parameters(params).axes

    spectrum = p.PupilResidualSpectrum(
        pd.DataFrame(
            {
                "frequency": [0.0, 0.1, 0.2],
                "median_power": [1.0, 0.7, 0.4],
                "q25_power": [0.8, 0.5, 0.3],
                "q75_power": [1.2, 0.9, 0.5],
            }
        ),
        3,
    )
    assert p.plot_pupil_residual_spectrum(spectrum).axes

    complexity = p.PupilComplexityAudit(
        "review",
        100,
        10,
        2,
        3,
        9,
        pd.DataFrame(
            {
                "check": ["basis", "rows"],
                "status": ["pass", "review"],
            }
        ),
    )
    assert p.plot_pupil_model_complexity(complexity).axes
    assert p.plot_pupil_model_complexity(spec).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_model_complexity(object())  # type: ignore[arg-type]

    functional_grid = pd.DataFrame(
        {
            "event_time": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "condition": ["A", "A", "A", "B", "B", "B"],
        }
    )
    functional_draws = np.tile(
        [0.0, 0.1, 0.25, 0.0, 0.05, 0.1],
        (20, 1),
    )
    functional = type(
        "FunctionalPrediction",
        (),
        {
            "grid": functional_grid,
            "draws": functional_draws,
            "specification": {
                "mapping": {
                    "time": "event_time",
                    "condition": "condition",
                }
            },
        },
    )()
    derivative = p.estimate_pupil_trajectory_derivative(functional)
    dyn = p.estimate_pupil_dynamic_contrast(functional, ("A", "B"))
    assert p.plot_pupil_trajectory_derivative(derivative).axes
    assert p.plot_pupil_dynamic_contrast(dyn).axes

    ident = p.PupilIdentifiabilityAudit(
        pd.DataFrame(
            {
                "check": ["basis", "condition"],
                "status": ["pass", "review"],
            }
        ),
        "review",
        spec,
    )
    assert p.plot_pupil_identifiability_audit(ident).axes

    score = p.PupilPredictiveScore(
        pd.DataFrame({"metric": ["rmse"], "value": [0.1]}),
        pd.DataFrame(
            {
                "predicted_mean": [3.0, 3.2],
                "observed": [3.05, 3.15],
            }
        ),
        0.9,
    )
    assert p.plot_pupil_predictive_calibration(score).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_predictive_calibration(object())  # type: ignore[arg-type]

    observed = pd.DataFrame(
        {
            "event_time": [0.0, 0.1, 0.0, 0.1],
            "pupil": [3.0, 3.2, 3.1, 3.3],
            "condition": ["A", "A", "B", "B"],
        }
    )
    assert p.plot_pupil_observed_trajectory(observed, summary=True).axes
    assert p.plot_pupil_observed_trajectory(observed, summary=False).axes
    assert p.plot_pupil_observed_trajectory(
        observed.drop(columns="condition"),
        summary=True,
    ).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_observed_trajectory(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_pupil_observed_trajectory(pd.DataFrame({"x": [1]}))
