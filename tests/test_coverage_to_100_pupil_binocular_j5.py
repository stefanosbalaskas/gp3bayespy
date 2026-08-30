from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _prepared(seed=4801):
    sim = p.simulate_binocular_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        time_points=10,
        outlier_fraction=0,
        missing_fraction=0,
        seed=seed,
    )
    return p.prepare_binocular_pupil_timecourse(sim.data)


def test_binocular_specification_translation_and_fit_matrix():
    prepared = _prepared()

    with pytest.raises(GP3BayesError):
        p.specify_binocular_pupil_model(object())  # type: ignore[arg-type]
    for kwargs in (
        {"temporal_structure": "bad"},
        {"family": "bad"},
        {"smooth_basis_dimension": True},
        {"smooth_basis_dimension": 3},
        {"smooth_basis_dimension": 101},
    ):
        with pytest.raises(GP3BayesError):
            p.specify_binocular_pupil_model(prepared, **kwargs)

    spec = p.specify_binocular_pupil_model(
        prepared,
        temporal_structure="linear",
        family="student",
        residual_correlation=False,
        item_effects=False,
    )
    translated = p.translate_binocular_pupil_model_to_brms(spec)
    assert len(translated.formula) == 2
    with pytest.raises(GP3BayesError):
        p.translate_binocular_pupil_model_to_brms(object())  # type: ignore[arg-type]

    smooth = p.specify_binocular_pupil_model(
        prepared,
        temporal_structure="smooth",
        smooth_basis_dimension=10,
    )
    assert smooth.smooth_basis_dimension_effective >= 4

    gp_spec = p.specify_binocular_pupil_model(
        prepared,
        temporal_structure="gaussian_process",
        gp_spec=p.create_pupil_gp_spec(basis="approximate", k=8),
    )
    assert gp_spec.gp_spec is not None

    with pytest.raises(GP3BayesError):
        p.fit_binocular_pupil_model(object())  # type: ignore[arg-type]

    fit = p.fit_binocular_pupil_model(
        spec,
        backend="analytic",
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=4802,
    )
    assert fit.fit_performed

    with pytest.raises(GP3BayesError):
        p.estimate_binocular_pupil_trajectory(object())  # type: ignore[arg-type]
    trajectory = p.estimate_binocular_pupil_trajectory(fit, ndraws=5)
    assert len(trajectory.grid) == trajectory.left_draws.shape[1]
    assert p.plot_binocular_pupil_trajectory(trajectory).axes

    difference = p.pupil_binocular_difference(trajectory)
    assert "mean" in difference
    agreement = p.pupil_binocular_agreement_table(trajectory, tolerance=0.2)
    assert "probability_within_tolerance" in agreement

    with pytest.raises(GP3BayesError):
        p.pupil_binocular_difference(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_agreement_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_binocular_correlation(fit)

    grid = trajectory.grid.head(3).copy()
    assert len(p._binocular_grid(fit, grid)) == 3
    with pytest.raises(GP3BayesError):
        p._binocular_grid(fit, pd.DataFrame({"x": range(6000)}), max_grid=5000)


def test_binocular_residual_correlation_enabled():
    prepared = _prepared(4810)
    spec = p.specify_binocular_pupil_model(
        prepared,
        temporal_structure="linear",
        residual_correlation=True,
    )
    fit = p.fit_binocular_pupil_model(
        spec,
        backend="analytic",
        chains=1,
        iter=70,
        warmup=20,
        cores=1,
        seed=4811,
    )
    table = p.pupil_binocular_correlation(fit)
    assert table.iloc[0]["parameter"].startswith("rescor")
