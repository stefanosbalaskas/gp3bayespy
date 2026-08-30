from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _advanced(seed=4701):
    sim = p.simulate_advanced_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        time_points=12,
        outlier_fraction=0,
        missing_fraction=0,
        seed=seed,
    )
    return p.specify_advanced_pupil_timecourse_model(
        sim.data,
        temporal_structure="smooth",
        family="gaussian",
        residual_scale="constant",
        autocorrelation="none",
        smooth_basis_dimension=5,
    )


def test_advanced_sensitivity_suite_materializes_every_dimension():
    spec = _advanced()
    with pytest.raises(GP3BayesError):
        p.create_pupil_advanced_sensitivity_suite(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_advanced_sensitivity_suite(spec, include=("bad",))

    suite = p.create_pupil_advanced_sensitivity_suite(spec)
    assert not suite.scenarios.empty
    assert p.materialize_pupil_advanced_sensitivity_scenario(suite, "baseline") is spec

    for scenario in suite.scenarios["scenario"].astype(str):
        materialized = p.materialize_pupil_advanced_sensitivity_scenario(suite, scenario)
        assert isinstance(materialized, p.AdvancedPupilSpecification)

    with pytest.raises(GP3BayesError):
        p.materialize_pupil_advanced_sensitivity_scenario(object(), "x")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.materialize_pupil_advanced_sensitivity_scenario(suite, "missing")

    gp_spec = p.specify_advanced_pupil_timecourse_model(
        spec.data,
        temporal_structure="gaussian_process",
        family="gaussian",
        gp_spec=p.create_pupil_gp_spec(kernel="matern32", basis="approximate", k=8),
        allow_high_complexity=True,
    )
    gp_suite = p.create_pupil_advanced_sensitivity_suite(gp_spec)
    gp_rows = gp_suite.scenarios[gp_suite.scenarios["dimension"].eq("gp_kernel")]
    assert len(gp_rows) >= 1
    for scenario in gp_rows["scenario"].astype(str):
        changed = p.materialize_pupil_advanced_sensitivity_scenario(gp_suite, scenario)
        assert changed.gp_spec is not None

    card = p.pupil_model_card(spec)
    assert not p.pupil_model_card_table(card).empty
    fit = p.fit_advanced_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=4702,
    )
    assert not p.pupil_model_card_table(fit).empty
    with pytest.raises(GP3BayesError):
        p.pupil_model_card(object())


def test_response_shape_complete_contract_and_fit():
    for kwargs in (
        {"n_participants": 1},
        {"time_points": 7},
        {"conditions": ()},
        {"amplitude": 0},
        {"rise": 0},
        {"duration": 0},
        {"decay": 0},
        {"condition_amplitude_ratio": 0},
        {"residual_sd": 0},
    ):
        with pytest.raises(GP3BayesError):
            p.simulate_pupil_response_shape(**kwargs)

    sim = p.simulate_pupil_response_shape(
        n_participants=3,
        trials_per_participant=3,
        time_points=12,
        seed=4710,
    )

    for kwargs in (
        {"family": "bad"},
        {"condition_effects": ("bad",)},
        {"participant_effects": ("bad",)},
        {"covariates": ("missing",)},
        {"prior_scales": {"x": 0}},
    ):
        with pytest.raises(GP3BayesError):
            p.specify_pupil_response_shape_model(sim.data, **kwargs)

    no_condition = sim.data.drop(columns=["condition"])
    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(no_condition)

    spec = p.specify_pupil_response_shape_model(
        sim.data,
        family="student",
        condition_effects=("amplitude", "onset"),
        participant_effects=("baseline",),
    )
    translation = p.translate_pupil_response_shape_to_brms(spec)
    assert "inv_logit" in translation.formula
    with pytest.raises(GP3BayesError):
        p.translate_pupil_response_shape_to_brms(object())  # type: ignore[arg-type]

    curve = p._shape_curve(
        np.array([-1e6, 0.0, 1e6]),
        3.0,
        0.5,
        0.0,
        0.0,
        100.0,
        0.0,
    )
    assert np.isfinite(curve).all()

    with pytest.raises(GP3BayesError):
        p.fit_pupil_response_shape_model(object())  # type: ignore[arg-type]
    fit = p.fit_pupil_response_shape_model(
        spec,
        backend="analytic",
        chains=1,
        iter=80,
        warmup=20,
        cores=1,
        seed=4711,
    )
    params = p.estimate_pupil_response_parameters(fit)
    table = p.pupil_response_parameter_table(params)
    assert set(table["parameter"]) == set(fit.parameter_draws.columns)
    assert p.plot_pupil_response_parameters(params).axes
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_response_parameters(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_response_parameter_table(object())  # type: ignore[arg-type]


def test_model_set_kfold_weights_and_lfo_nonexecuted_paths():
    spec = _advanced(4720)
    fit1 = p.fit_advanced_pupil_model_backend(
        spec, backend="analytic", chains=1, iter=70, warmup=20, cores=1, seed=4721
    )
    fit2 = p.fit_advanced_pupil_model_backend(
        spec, backend="analytic", chains=1, iter=70, warmup=20, cores=1, seed=4722
    )

    with pytest.raises(GP3BayesError):
        p._fit_log_likelihood(object())
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set([fit1, fit2])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set({"only": fit1})
    with pytest.raises(GP3BayesError):
        p.create_pupil_model_set(
            {"a": fit1, "b": fit2},
            predictive_target="bad",  # type: ignore[arg-type]
        )

    model_set = p.create_pupil_model_set({"a": fit1, "b": fit2})
    with pytest.raises(GP3BayesError):
        p.compare_pupil_models(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.compare_pupil_models(model_set, criterion="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.compare_pupil_models(model_set, criterion="kfold", K=1)

    comparison = p.compare_pupil_models(model_set, criterion="kfold", K=3)
    assert len(p.pupil_model_comparison_table(comparison)) == 2
    assert p.plot_pupil_model_comparison(comparison).axes
    with pytest.raises(GP3BayesError):
        p.pupil_model_comparison_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_model_weights(comparison)
    with pytest.raises(GP3BayesError):
        p.pupil_model_weights(model_set, method="bad")  # type: ignore[arg-type]

    plan = p.create_pupil_lfo_plan(fit1, initial_fraction=0.5, horizon=2, step=1, max_refits=3)
    assert len(plan.table) <= 3
    validation = p.validate_pupil_leave_future_out(fit1, plan, execute=False)
    assert not validation.executed
    assert p.plot_pupil_lfo(plan).axes
    assert p.plot_pupil_lfo(validation).axes

    with pytest.raises(GP3BayesError):
        p.create_pupil_lfo_plan(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_lfo_plan(fit1, initial_fraction=0.99, horizon=99)
    with pytest.raises(GP3BayesError):
        p.validate_pupil_leave_future_out(fit1, object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.compare_pupil_lfo(validation)

    scores1 = pd.DataFrame(
        {
            "refit": [1, 2],
            "test_rows": [3, 3],
            "elpd_future": [-2.0, -1.0],
            "mean_log_score": [-0.7, -0.3],
        }
    )
    scores2 = scores1.assign(
        elpd_future=[-2.5, -1.5],
        mean_log_score=[-0.8, -0.4],
    )
    v1 = p.PupilLFOValidation(plan.table, True, scores1, -3.0, "x")
    v2 = p.PupilLFOValidation(plan.table, True, scores2, -4.0, "x")
    compared = p.compare_pupil_lfo({"a": v1, "b": v2})
    assert len(compared.table) == 2
    assert p.plot_pupil_lfo(compared).axes
