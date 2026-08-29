from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.advanced_optional_workflows as aow
import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
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
                            3.0
                            + (0.15 if condition == "B" else 0.0)
                            + (0.04 if participant == "p2" else 0.0)
                            + (0.08 if participant == "p3" else 0.0)
                            + 0.18 * np.sin(t / 2)
                        ),
                        "baseline": 3.0 + 0.01 * t,
                        "luminance": 50.0 + t,
                        "baseline_se": 0.05 + 0.001 * t,
                        "response_se": 0.03 + 0.001 * t,
                    }
                )
    return pd.DataFrame(rows)


def _advanced_spec(
    temporal: str = "smooth",
    family: str = "gaussian",
    residual: str = "constant",
    gp_spec=None,
    autocorrelation="none",
):
    return p.specify_advanced_pupil_timecourse_model(
        _advanced_data(),
        temporal_structure=temporal,
        family=family,
        residual_scale=residual,
        gp_spec=gp_spec,
        autocorrelation=autocorrelation,
        covariates=("baseline",),
        smooth_basis_dimension=8,
    )


def _advanced_fit(specification, seed: int = 2001):
    return p.fit_advanced_pupil_model_backend(
        specification,
        backend="analytic",
        chains=1,
        iter=140,
        warmup=40,
        cores=1,
        seed=seed,
    )


def _base_objects(seed: int = 2010):
    sim = p.simulate_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.8),
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
        blink_col="blink",
        gaze_x_col="gaze_x",
        gaze_y_col="gaze_y",
        luminance_col="luminance",
        baseline_window=(-0.2, 0.0),
    )
    prepared = p.prepare_pupil_timecourse(
        sim.data,
        contract,
        baseline_operation="none",
    )
    spec = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        autocorrelation="none",
    )
    fit = p.fit_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=120,
        warmup=40,
        cores=1,
        seed=seed + 1,
    )
    return sim, contract, prepared, spec, fit


def test_base_validation_targets_execute_and_sensitivity_suite():
    _, _, prepared, spec, fit = _base_objects()

    plans = {
        target: p.create_pupil_validation_plan(
            prepared,
            target=target,
            K=2,
            future_fraction=0.25,
            seed=2020,
        )
        for target in (
            "new_sample_known_trial",
            "new_trial_known_participant",
            "new_participant",
            "future_segment",
        )
    }
    assert plans["new_sample_known_trial"].strategy == "observation_kfold"
    assert plans["new_trial_known_participant"].K == 2
    assert plans["new_participant"].strategy == "grouped_participant_kfold"
    assert not plans["future_segment"].split_table.empty
    assert not any(plan.leakage_detected for plan in plans.values())

    unexecuted = p.validate_pupil_model(
        fit,
        plans["new_sample_known_trial"],
        execute=False,
    )
    assert not unexecuted.executed
    assert p.pupil_validation_table(unexecuted).loc[0, "executed"] == False  # noqa: E712

    executed = p.validate_pupil_model(
        fit,
        plans["new_sample_known_trial"],
        execute=True,
        ndraws=30,
    )
    assert executed.executed
    assert {"rmse", "mae"}.issubset(executed.table.columns)

    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(prepared, target="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(
            replace(prepared, data=prepared.data.head(3)),
        )
    with pytest.raises(GP3BayesError):
        p.validate_pupil_model(object(), plans["new_sample_known_trial"])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.validate_pupil_model(fit, object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_validation_table(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p.create_pupil_sensitivity_suite(
            spec,
            baseline_windows=((-0.2, 0.0),),
        )

    suite = p.create_pupil_sensitivity_suite(
        spec,
        baseline_windows=((-0.2, 0.0),),
        baseline_window_operation="subtract",
        baseline_operations=("none", "subtract"),
        interpolation_policy=("none", "linear"),
        blink_adjacent_margins=(0.05,),
        gaze_adjustment=("none",),
        luminance_adjustment=("none",),
        pfe_prepared={"same": prepared},
        smooth_basis_dimensions=(6,),
        autocorrelation=("none",),
        analysis_windows=((0.0, 0.5),),
    )
    assert len(suite.scenarios) >= 10
    assert p.pupil_sensitivity_table(suite).equals(suite.scenarios)

    materialized = {}
    for sid in suite.scenarios["scenario_id"]:
        materialized[sid] = p.materialize_pupil_sensitivity_scenario(
            suite,
            str(sid),
        )
        assert materialized[sid]["fit_performed"] is False
        assert materialized[sid]["pfe_correction_performed"] is False

    axes = {
        row.axis: materialized[row.scenario_id] for row in suite.scenarios.itertuples(index=False)
    }
    assert axes["smooth_basis_dimension"]["specification"].smooth_basis_dimension == 6
    assert axes["analysis_window"]["analysis_window"] == (0.0, 0.5)
    assert axes["pfe_prepared"]["prepared"] is prepared

    with pytest.raises(GP3BayesError):
        p.materialize_pupil_sensitivity_scenario(object(), "S001")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.materialize_pupil_sensitivity_scenario(suite, "missing")

    est1 = p.PupilEstimand(
        pd.DataFrame({"estimate": [0.1], "lower": [0.0], "upper": [0.2]}),
        "window_mean",
        "millimetres",
        0.95,
        (0.0, 0.5),
    )
    est2 = p.PupilEstimand(
        pd.DataFrame({"estimate": [0.2], "lower": [0.1], "upper": [0.3]}),
        "window_mean",
        "millimetres",
        0.95,
        (0.0, 0.5),
    )
    comp = p.compare_pupil_sensitivity_estimands({"a": est1, "b": est2})
    assert set(comp.table["scenario_id"]) == {"a", "b"}
    assert p.pupil_sensitivity_table(comp).equals(comp.table)

    with pytest.raises(GP3BayesError):
        p.compare_pupil_sensitivity_estimands({})
    with pytest.raises(GP3BayesError):
        p.compare_pupil_sensitivity_estimands({"bad": object()})  # type: ignore[dict-item]
    with pytest.raises(GP3BayesError):
        p.pupil_sensitivity_table(object())  # type: ignore[arg-type]

    assert p.plot_pupil_validation(executed).axes
    assert p.plot_pupil_sensitivity(suite).axes
    assert p.plot_pupil_sensitivity(comp).axes


@pytest.mark.filterwarnings(
    "ignore:covariance is not symmetric positive-semidefinite:RuntimeWarning"
)
def test_advanced_prior_translation_fit_prediction_residual_gp_and_temporal_paths():
    gp_spec = p.create_pupil_gp_spec("matern32", "approximate", 8)
    spec = _advanced_spec(
        temporal="gaussian_process",
        family="student",
        residual="condition_time",
        gp_spec=gp_spec,
    )

    prior = p.create_advanced_pupil_prior_specification(spec)
    assert {"Intercept", "b", "sigma", "nu", "sdgp", "lscale"}.issubset(
        set(prior.table["parameter"])
    )
    translation = p.translate_advanced_pupil_model_to_brms(spec)
    assert "gp(" in translation.formula
    assert ".gp3bayes_time_index" in translation.data

    prior_check = p.check_advanced_pupil_prior_predictive(
        spec,
        backend="analytic",
        chains=1,
        iter=100,
        warmup=20,
        cores=1,
        seed=2030,
    )
    assert prior_check.draws >= 50
    assert len(prior_check.table) == 4

    with pytest.raises(GP3BayesError):
        p.create_advanced_pupil_prior_specification(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.translate_advanced_pupil_model_to_brms(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_advanced_pupil_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_advanced_pupil_prior_predictive(spec, iter=100, warmup=100)

    fit = _advanced_fit(spec, 2031)
    assert fit.backend == "analytic"
    assert fit.log_likelihood.shape[1] == len(spec.data)

    expected = p.predict_advanced_pupil_trajectory(
        fit,
        type="expected",
        ndraws=25,
    )
    linear = p.predict_advanced_pupil_trajectory(
        fit,
        type="linear",
        ndraws=20,
    )
    posterior = p.predict_advanced_pupil_trajectory(
        fit,
        type="posterior_predictive",
        ndraws=20,
    )
    assert expected.draws.shape[1] == len(expected.grid)
    assert linear.draws.shape == posterior.draws.shape
    assert len(p.advanced_pupil_trajectory_table(expected)) == len(expected.grid)

    with pytest.raises(GP3BayesError):
        p.predict_advanced_pupil_trajectory(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.predict_advanced_pupil_trajectory(fit, type="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.predict_advanced_pupil_trajectory(fit, population_only=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._advanced_prediction_grid(fit, "bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._advanced_prediction_grid(fit, fit.translation.data, max_grid=2)
    with pytest.raises(GP3BayesError):
        p.advanced_pupil_trajectory_table(object())  # type: ignore[arg-type]

    residual = p.estimate_pupil_residual_scale(
        fit,
        ndraws=30,
        probability=0.9,
    )
    residual_table = p.pupil_residual_scale_table(residual)
    assert len(residual_table) == len(residual.grid)
    assert np.isfinite(residual.draws).all()
    assert np.any(np.ptp(residual.draws, axis=1) > 0)

    gp = p.pupil_gp_hyperparameters(fit, 0.9)
    assert set(p.pupil_gp_table(gp)["parameter"]) == {"sdgp", "lscale"}
    with pytest.raises(GP3BayesError):
        p.pupil_gp_hyperparameters(_advanced_fit(_advanced_spec(), 2032))
    with pytest.raises(GP3BayesError):
        p.pupil_gp_table(object())  # type: ignore[arg-type]

    temporal = p.audit_pupil_temporal_dependence(fit, max_lag=4)
    assert len(p.pupil_autocorrelation_table(temporal, "series")) >= 1
    with pytest.raises(GP3BayesError):
        p.pupil_autocorrelation_table(temporal, "bad")  # type: ignore[arg-type]

    smooth_fit = _advanced_fit(_advanced_spec("smooth"), 2033)
    comparison = p.compare_pupil_autocorrelation(
        {"gp": fit, "smooth": smooth_fit},
        max_lag=3,
        ndraws=20,
    )
    assert set(comparison.table["model"]) == {"gp", "smooth"}
    with pytest.raises(GP3BayesError):
        p.compare_pupil_autocorrelation(fit)
    with pytest.raises(GP3BayesError):
        p.compare_pupil_autocorrelation({"good": fit, "bad": object()})  # type: ignore[dict-item]

    spectrum = p.pupil_residual_spectrum(smooth_fit, ndraws=20)
    assert spectrum.n_series >= 1
    short = replace(
        smooth_fit,
        translation=replace(
            smooth_fit.translation,
            data=smooth_fit.translation.data.groupby(
                ["participant_id", "trial_id"],
                observed=True,
                sort=False,
            ).head(4),
        ),
    )
    with pytest.raises(GP3BayesError):
        p.pupil_residual_spectrum(short, ndraws=10)


def test_advanced_model_set_loo_kfold_weights_and_lfo(monkeypatch):
    fit1 = _advanced_fit(_advanced_spec("linear"), 2040)
    fit2 = _advanced_fit(_advanced_spec("smooth"), 2041)

    model_set = p.create_pupil_model_set(
        {"linear": fit1, "smooth": fit2},
        predictive_target="new_trial_known_participant",
    )
    assert len(model_set.models) == 2

    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set({"one": fit1})
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set([], predictive_target="new_trial_known_participant")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set(
            {"a": fit1, "b": fit2},
            predictive_target="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set({"a": fit1, "b": object()})

    kfold = p.compare_pupil_models(model_set, criterion="kfold", K=3, group="participant")
    assert kfold.criterion == "kfold"
    assert len(kfold.table) == 2
    with pytest.raises(GP3BayesError):
        p.compare_pupil_models(model_set, criterion="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.compare_pupil_models(model_set, criterion="kfold", K=1)

    counter = {"i": 0}

    def fake_loo(log_lik):
        counter["i"] += 1
        return SimpleNamespace(
            table=pd.DataFrame(
                {
                    "elpd_loo": [-10.0 - counter["i"]],
                    "se_elpd_loo": [1.0 + counter["i"] / 10],
                }
            )
        )

    monkeypatch.setattr(aow, "compute_psis_loo_from_log_lik", fake_loo)
    loo_cmp = p.compare_pupil_models(model_set, criterion="loo")
    assert loo_cmp.criterion == "loo"
    assert len(p.pupil_model_comparison_table(loo_cmp)) == 2

    weights = p.pupil_model_weights(loo_cmp, method="stacking")
    assert np.isclose(weights["weight"].sum(), 1.0)
    weights2 = p.pupil_model_weights(model_set, method="pseudobma")
    assert np.isclose(weights2["weight"].sum(), 1.0)

    with pytest.raises(GP3BayesError):
        p.pupil_model_comparison_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_model_weights(kfold)
    with pytest.raises(GP3BayesError):
        p.pupil_model_weights(loo_cmp, method="bad")  # type: ignore[arg-type]

    plan = p.create_pupil_lfo_plan(
        fit1,
        initial_fraction=0.5,
        horizon=2,
        step=2,
        max_refits=3,
    )
    assert 1 <= len(plan.table) <= 3
    unexecuted = p.validate_pupil_leave_future_out(
        fit1,
        plan,
        execute=False,
    )
    assert not unexecuted.executed

    with pytest.raises(GP3BayesError):
        p.create_pupil_lfo_plan(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_lfo_plan(fit1, initial_fraction=0.99, horizon=50)
    with pytest.raises(GP3BayesError):
        p.validate_pupil_leave_future_out(object(), plan)  # type: ignore[arg-type]

    scores1 = pd.DataFrame(
        {
            "refit": [1, 2],
            "test_rows": [10, 10],
            "elpd_future": [-10.0, -9.0],
            "mean_log_score": [-1.0, -0.9],
        }
    )
    scores2 = pd.DataFrame(
        {
            "refit": [1, 2],
            "test_rows": [10, 10],
            "elpd_future": [-11.0, -10.0],
            "mean_log_score": [-1.1, -1.0],
        }
    )
    v1 = p.PupilLFOValidation(
        plan.table.copy(),
        True,
        scores1,
        float(scores1["elpd_future"].sum()),
        plan.interpretation,
    )
    v2 = p.PupilLFOValidation(
        plan.table.copy(),
        True,
        scores2,
        float(scores2["elpd_future"].sum()),
        plan.interpretation,
    )
    lfo_cmp = p.compare_pupil_lfo({"linear": v1, "smooth": v2})
    assert set(lfo_cmp.table["model"]) == {"linear", "smooth"}

    with pytest.raises(GP3BayesError):
        p.compare_pupil_lfo(v1, v2)
    with pytest.raises(GP3BayesError):
        p.compare_pupil_lfo({"a": unexecuted, "b": v2})
