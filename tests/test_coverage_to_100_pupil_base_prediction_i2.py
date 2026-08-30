from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _base(seed: int = 2801):
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
        temporal_structure="smooth",
        condition_trajectory=True,
        covariates=(".luminance",),
        smooth_basis_dimension=6,
        autocorrelation="none",
    )
    fit = p.fit_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=140,
        warmup=40,
        cores=1,
        seed=seed + 1,
    )
    return prepared, spec, fit


def _functional_prediction() -> p.PupilPrediction:
    times = np.array([0.0, 0.1, 0.2])
    grid = pd.DataFrame(
        {
            ".event_time": np.r_[times, times],
            ".condition": ["A"] * 3 + ["B"] * 3,
        }
    )
    base = np.array([3.0, 3.2, 3.4, 3.0, 3.1, 3.15])
    draws = base[None, :] + np.linspace(-0.08, 0.08, 30)[:, None]
    return p.as_pupil_prediction_draws(
        draws,
        grid,
        "millimetres",
        type="expected",
    )


def test_base_prediction_and_estimand_tail_matrix():
    prepared, spec, fit = _base()

    assert p.translate_pupil_model_to_brms(spec)["family"] == "gaussian"
    with pytest.raises(GP3BayesError):
        p.translate_pupil_model_to_brms(object())  # type: ignore[arg-type]

    prior = p.check_pupil_prior_predictive(
        spec,
        execute=True,
        backend="analytic",
        draws=55,
    )
    assert prior.executed
    with pytest.raises(GP3BayesError):
        p.check_pupil_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_pupil_prior_predictive(spec, draws=0)

    with pytest.raises(GP3BayesError):
        p.fit_pupil_model_backend(object())  # type: ignore[arg-type]

    cmd = p.fit_pupil_model_cmdstanr(
        spec,
        chains=1,
        iter=100,
        warmup=40,
        cores=1,
        seed=2805,
    )
    assert cmd.backend == "cmdstanr"

    grid = pd.DataFrame({".event_time": [0.0, 0.1, 0.2]})
    supplied = p.as_pupil_prediction_draws(
        np.ones((4, 3)),
        grid,
        "millimetres",
        type="linear",
    )
    assert supplied.type == "linear"

    with pytest.raises(GP3BayesError):
        p.as_pupil_prediction_draws(np.ones((4, 3)), grid, "bad")
    with pytest.raises(GP3BayesError):
        p.as_pupil_prediction_draws(
            np.ones((4, 3)),
            grid,
            "millimetres",
            type="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.as_pupil_prediction_draws(
            np.ones((4, 3)),
            pd.DataFrame(),
            "millimetres",
        )
    with pytest.raises(GP3BayesError):
        p.as_pupil_prediction_draws(
            np.ones((4, 2)),
            grid,
            "millimetres",
        )
    with pytest.raises(GP3BayesError):
        p.as_pupil_prediction_draws(
            np.ones((4, 3)),
            grid,
            "millimetres",
            max_cells=2,
        )

    default_grid = p._prediction_grid(fit, None, True)
    assert not default_grid.empty
    with pytest.raises(GP3BayesError):
        p._prediction_grid(fit, pd.DataFrame(), True)

    expected = p.predict_pupil_trajectory(
        fit,
        type="expected",
        ndraws=25,
    )
    posterior = p.predict_pupil_trajectory(
        fit,
        type="posterior_predictive",
        ndraws=25,
    )
    assert expected.draws.shape == posterior.draws.shape

    explicit = p.predict_pupil_trajectory(
        fit,
        newdata=default_grid,
        type="linear",
        ndraws=20,
        population_only=False,
        allow_new_levels=True,
    )
    assert explicit.n_grid == len(default_grid)

    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            population_only=False,
        )
    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            max_grid=1,
        )
    with pytest.raises(GP3BayesError):
        p.predict_pupil_trajectory(
            fit,
            type="bad",  # type: ignore[arg-type]
            ndraws=10,
        )

    simultaneous = p.estimate_pupil_trajectory(
        posterior,
        probability=0.9,
        interval="simultaneous",
    )
    assert simultaneous.finite_grid_qualification
    assert len(p.pupil_trajectory_table(simultaneous)) == len(posterior.grid)

    with pytest.raises(GP3BayesError):
        p.estimate_pupil_trajectory(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_trajectory(
            posterior,
            interval="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.pupil_trajectory_table(object())  # type: ignore[arg-type]

    pred = _functional_prediction()
    for estimator in (
        p.estimate_pupil_window,
        p.estimate_pupil_auc,
        p.estimate_pupil_peak,
        p.estimate_pupil_peak_latency,
    ):
        result = estimator(pred, (0.0, 0.2), probability=0.9)
        assert len(result.table) == 2

    contrast = p.pupil_condition_contrast(
        pred,
        ("A", "B"),
        threshold=0.01,
        probability=0.9,
    )
    assert len(contrast.table) == 3

    with pytest.raises(GP3BayesError):
        p.estimate_pupil_window(pred, (1.0, 2.0))
    with pytest.raises(AssertionError):
        p._estimand(pred, (0.0, 0.2), "bad", 0.95)
    with pytest.raises(GP3BayesError):
        p.pupil_condition_contrast(
            p.as_pupil_prediction_draws(
                np.ones((5, 3)),
                pd.DataFrame({".event_time": [0.0, 0.1, 0.2]}),
                "millimetres",
            ),
            ("A", "B"),
        )
    with pytest.raises(GP3BayesError):
        p.pupil_condition_contrast(pred, ("A",))

    mask = np.ones(len(grid), dtype=bool)
    groups = p._groups(
        p.as_pupil_prediction_draws(
            np.ones((5, 3)),
            grid,
            "millimetres",
        ),
        mask,
    )
    assert len(groups) == 1
