from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_identifiability_and_predictive_score_branches():
    sim = p.simulate_pupil_response_shape(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        seed=1801,
    )
    data = sim.data.copy()
    data["luminance"] = np.linspace(40.0, 60.0, len(data))
    missingness = p.create_pupil_missingness_spec(
        response="model",
        predictors=("luminance",),
    )
    spec = p.specify_advanced_pupil_timecourse_model(
        data,
        family="student",
        residual_scale="condition_time",
        covariates=("luminance",),
        missingness_model=missingness,
    )
    audit = p.audit_advanced_pupil_identifiability(spec)
    assert audit.overall in {"review", "high", "pass"}
    assert p.pupil_identifiability_table(audit).equals(audit.table)

    with pytest.raises(GP3BayesError):
        p.audit_advanced_pupil_identifiability(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_identifiability_table(object())  # type: ignore[arg-type]

    observed = np.array([1.0, 2.0, np.nan, 4.0])
    draws = np.array(
        [
            [0.9, 1.9, 3.0, 4.2],
            [1.1, 2.1, 3.2, 3.8],
            [1.0, 2.0, 2.8, 4.0],
        ]
    )
    score = p.score_pupil_predictions(observed, draws, 0.8)
    assert len(score.table) == 6
    assert len(score.pointwise) == 3

    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        single = p.score_pupil_predictions(
            [1.0, 2.0],
            np.array([[1.1, 1.9]]),
        )
    assert single.pointwise["crps"].isna().all()

    with pytest.raises(GP3BayesError):
        p.score_pupil_predictions([1.0, 2.0], np.ones((2, 3)))
    with pytest.raises(GP3BayesError):
        p.score_pupil_predictions([np.nan, np.nan], np.ones((2, 2)))


def test_binocular_simulation_preparation_audit_specification_and_translation():
    sim = p.simulate_binocular_pupil_timecourse(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        missing_fraction=0.0,
        outlier_fraction=0.0,
        seed=1810,
    )
    assert {"pupil_left", "pupil_right"}.issubset(sim.data.columns)

    with pytest.raises(GP3BayesError):
        p.simulate_binocular_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            time_points=8,
            residual_correlation=0.99,
        )

    prepared = p.prepare_binocular_pupil_timecourse(sim.data)
    assert prepared.mapping["left"] == "pupil_left"
    audit = p.audit_binocular_pupil_readiness(prepared)
    assert audit.status == "pass"

    sparse = prepared.data.copy()
    sparse.loc[sparse.index[: int(len(sparse) * 0.8)], ["pupil_left", "pupil_right"]] = np.nan
    sparse_audit = p.audit_binocular_pupil_readiness(p.prepare_binocular_pupil_timecourse(sparse))
    assert sparse_audit.status in {"review", "failure"}

    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse([])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(sim.data.drop(columns="pupil_left"))
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(
            sim.data,
            left_col="pupil_left",
            right_col="pupil_left",
        )

    one_condition = sim.data.copy()
    one_condition["condition"] = "only"
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(one_condition)

    bad_participant = sim.data.copy()
    bad_participant.loc[bad_participant.index[0], "participant_id"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(bad_participant)

    bad_time = sim.data.copy()
    bad_time.loc[bad_time.index[0], "time_ms"] = np.inf
    with pytest.raises(GP3BayesError):
        p.prepare_binocular_pupil_timecourse(bad_time)

    with pytest.raises(GP3BayesError):
        p.audit_binocular_pupil_readiness(object())  # type: ignore[arg-type]

    smooth = p.specify_binocular_pupil_model(prepared)
    assert smooth.temporal_structure == "smooth"
    assert smooth.smooth_basis_dimension <= smooth.smooth_basis_dimension_requested
    linear = p.specify_binocular_pupil_model(prepared, temporal_structure="linear")
    assert linear.gp_spec is None
    gp_spec = p.specify_binocular_pupil_model(
        prepared,
        temporal_structure="gaussian_process",
        gp_spec=p.create_pupil_gp_spec("matern52", "approximate", 8),
    )
    assert gp_spec.gp_spec is not None

    translation = p.translate_binocular_pupil_model_to_brms(smooth)
    assert len(translation.formula) == 2
    assert len(translation.priors) == 4

    with pytest.raises(GP3BayesError):
        p.specify_binocular_pupil_model(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_binocular_pupil_model(prepared, temporal_structure="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_binocular_pupil_model(prepared, smooth_basis_dimension=True)
    with pytest.raises(GP3BayesError):
        p.specify_binocular_pupil_model(prepared, smooth_basis_dimension=3)
    with pytest.raises(GP3BayesError):
        p.translate_binocular_pupil_model_to_brms(object())  # type: ignore[arg-type]


def test_binocular_fit_trajectory_difference_correlation_agreement_and_grid():
    sim = p.simulate_binocular_pupil_timecourse(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        missing_fraction=0.0,
        outlier_fraction=0.0,
        seed=1820,
    )
    prepared = p.prepare_binocular_pupil_timecourse(sim.data)
    spec = p.specify_binocular_pupil_model(prepared, temporal_structure="smooth")

    fit = p.fit_binocular_pupil_model(
        spec,
        chains=1,
        iter=80,
        warmup=40,
        cores=1,
        seed=1821,
    )
    assert fit.backend == "rstan"
    assert len(fit.residual_correlation_draws) >= 50

    trajectory = p.estimate_binocular_pupil_trajectory(fit, ndraws=30)
    diff = p.pupil_binocular_difference(trajectory)
    corr = p.pupil_binocular_correlation(fit, 0.9)
    agreement = p.pupil_binocular_agreement_table(trajectory, tolerance=0.2)
    assert len(diff) == len(trajectory.grid)
    assert corr.loc[0, "parameter"] == "rescor__pupil_left__pupil_right"
    assert "probability_within_tolerance" in agreement

    newdata = trajectory.grid.head(3).copy()
    assert len(p._binocular_grid(fit, newdata)) == 3
    with pytest.raises(GP3BayesError):
        p._binocular_grid(fit, "bad")  # type: ignore[arg-type]

    no_corr_spec = replace(spec, residual_correlation=False)
    no_corr_fit = replace(fit, specification=no_corr_spec)
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_correlation(no_corr_fit)

    with pytest.raises(GP3BayesError):
        p.estimate_binocular_pupil_trajectory(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_difference(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_agreement_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_agreement_table(trajectory, tolerance=0)


def test_response_shape_simulation_spec_translation_fit_and_parameters():
    sim = p.simulate_pupil_response_shape(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        seed=1830,
    )
    assert len(sim.data) == 2 * 4 * 12
    assert "mean" in sim.truth

    for kwargs in (
        {"n_participants": 1},
        {"time_points": 7},
        {"amplitude": 0},
        {"rise": 0},
        {"duration": 0},
        {"decay": 0},
        {"condition_amplitude_ratio": 0},
        {"residual_sd": 0},
        {"conditions": ("",)},
    ):
        with pytest.raises(GP3BayesError):
            p.simulate_pupil_response_shape(**kwargs)

    spec = p.specify_pupil_response_shape_model(
        sim.data,
        family="gaussian",
        condition_effects=("amplitude", "onset"),
        participant_effects=("baseline",),
    )
    translation = p.translate_pupil_response_shape_to_brms(spec)
    assert "inv_logit" in translation.formula
    assert len(translation.priors) == 6

    curve = p._shape_curve(
        np.linspace(-1, 1, 5),
        3.0,
        0.5,
        0.0,
        0.2,
        1.0,
        0.3,
    )
    assert np.isfinite(curve).all()

    fit = p.fit_pupil_response_shape_model(
        spec,
        chains=1,
        iter=120,
        warmup=60,
        cores=1,
        seed=1831,
    )
    params = p.estimate_pupil_response_parameters(fit, 0.9)
    table = p.pupil_response_parameter_table(params)
    assert set(table["parameter"]) == {
        "baseline",
        "logAmplitude",
        "onset",
        "logRise",
        "logDuration",
        "logDecay",
    }

    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(sim.data, family="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(
            sim.data,
            condition_effects=("bad",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(
            sim.data,
            participant_effects=("bad",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(
            sim.data,
            covariates=("time_ms",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_response_shape_model(
            sim.data,
            prior_scales={"x": 0},
        )
    with pytest.raises(GP3BayesError):
        p.translate_pupil_response_shape_to_brms(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.fit_pupil_response_shape_model(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_response_parameters(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_response_parameter_table(object())  # type: ignore[arg-type]


def test_new_pupil_plot_surface_smoke():
    sim = p.simulate_pupil_response_shape(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        seed=1840,
    )
    response_spec = p.specify_pupil_response_shape_model(sim.data)
    response_fit = p.fit_pupil_response_shape_model(
        response_spec,
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=1841,
    )
    params = p.estimate_pupil_response_parameters(response_fit)

    advanced_spec = p.specify_advanced_pupil_timecourse_model(
        sim.data,
        covariates=(),
    )
    ident = p.audit_advanced_pupil_identifiability(advanced_spec)
    temporal = p.audit_pupil_temporal_dependence(sim.data, max_lag=3)

    score = p.score_pupil_predictions(
        [1.0, 2.0, 3.0],
        np.array(
            [
                [0.9, 2.1, 3.0],
                [1.1, 1.9, 3.2],
                [1.0, 2.0, 2.8],
            ]
        ),
    )

    binocular_sim = p.simulate_binocular_pupil_timecourse(
        n_participants=2,
        trials_per_participant=4,
        time_points=12,
        missing_fraction=0.0,
        outlier_fraction=0.0,
        seed=1842,
    )
    prepared = p.prepare_binocular_pupil_timecourse(binocular_sim.data)
    binocular_spec = p.specify_binocular_pupil_model(prepared)
    binocular_fit = p.fit_binocular_pupil_model(
        binocular_spec,
        chains=1,
        iter=80,
        warmup=40,
        cores=1,
        seed=1843,
    )
    trajectory = p.estimate_binocular_pupil_trajectory(binocular_fit, ndraws=20)

    figures = [
        p.plot_advanced_pupil_simulation(p.AdvancedPupilSimulation(sim.data.copy(), sim.truth)),
        p.plot_pupil_temporal_dependence(temporal),
        p.plot_binocular_pupil_trajectory(trajectory),
        p.plot_pupil_response_parameters(params),
        p.plot_pupil_model_complexity(advanced_spec),
        p.plot_pupil_identifiability_audit(ident),
        p.plot_pupil_predictive_calibration(score),
        p.plot_pupil_observed_trajectory(sim.data, summary=True),
        p.plot_pupil_observed_trajectory(sim.data, summary=False),
    ]
    assert all(fig.axes for fig in figures)

    with pytest.raises(GP3BayesError):
        p.plot_pupil_model_complexity(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_pupil_predictive_calibration(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_pupil_observed_trajectory(object())  # type: ignore[arg-type]
