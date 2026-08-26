from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError
from gp3bayespy.ppc import _binary_summary, _duration_summary, _predictive_status


class _Variable:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _Posterior(dict[str, _Variable]):
    pass


def _binary_fit() -> gp.BinaryFit:
    simulation = gp.simulate_hierarchical_binary_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=8201,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
    )
    prepared = gp.prepare_hierarchical_binary_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    spec = gp.specify_binary_model(prepared, baseline=0.35)
    n_participants = len(pd.unique(prepared.data["participant_id"]))
    n_items = len(pd.unique(prepared.data["item_id"]))
    draws = 60
    posterior = _Posterior(
        {
            "b_Intercept": _Variable(np.full((1, draws), -0.3)),
            "b": _Variable(np.full((1, draws, 1), 0.7)),
            "sd_participant": _Variable(np.full((1, draws), 0.25)),
            "participant_z": _Variable(
                np.linspace(-1, 1, draws * n_participants).reshape(
                    1, draws, n_participants
                )
            ),
            "sd_item": _Variable(np.full((1, draws), 0.12)),
            "item_z": _Variable(
                np.linspace(-0.5, 0.5, draws * n_items).reshape(
                    1, draws, n_items
                )
            ),
        }
    )
    return gp.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=gp.translate_binary_model_to_brms(spec),
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"chains": 1, "iter": 60, "warmup": 0},
        package_versions={},
    )


def _duration_fit() -> gp.DurationFit:
    simulation = gp.simulate_hierarchical_duration_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=8202,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    n_participants = len(pd.unique(prepared.data["participant_id"]))
    n_items = len(pd.unique(prepared.data["item_id"]))
    draws = 60
    posterior = _Posterior(
        {
            "b_Intercept": _Variable(np.full((1, draws), np.log(500.0))),
            "b": _Variable(np.full((1, draws, 1), np.log(1.1))),
            "sd_participant": _Variable(np.full((1, draws), 0.2)),
            "participant_z": _Variable(
                np.linspace(-1, 1, draws * n_participants).reshape(
                    1, draws, n_participants
                )
            ),
            "sd_item": _Variable(np.full((1, draws), 0.1)),
            "item_z": _Variable(
                np.linspace(-0.5, 0.5, draws * n_items).reshape(
                    1, draws, n_items
                )
            ),
            "sigma": _Variable(np.full((1, draws), 0.3)),
        }
    )
    return gp.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        specification=spec,
        translation=gp.translate_duration_model_to_brms(spec),
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"chains": 1, "iter": 60, "warmup": 0},
        package_versions={},
    )


def test_predictive_status_matches_frozen_conservative_boundaries():
    assert _predictive_status(0.5, 0.4, 0.6, 0.3, 0.7) == "pass"
    assert _predictive_status(0.65, 0.4, 0.6, 0.3, 0.7) == "review"
    assert _predictive_status(0.8, 0.4, 0.6, 0.3, 0.7) == "fail"
    assert _predictive_status(np.nan, 0.4, 0.6, 0.3, 0.7) == "not_applicable"


def test_binary_observed_summary_preserves_frozen_statistic_order():
    y = [0, 1, 0, 1, 1, 1, 0, 0]
    result = _binary_summary(
        y,
        condition=[-0.5, 0.5] * 4,
        participant=["p1"] * 4 + ["p2"] * 4,
        item=["i1", "i2"] * 4,
    )
    assert list(result) == [
        "overall_rate",
        "condition_low_rate",
        "condition_high_rate",
        "condition_rate_contrast",
        "participant_rate_sd",
        "item_rate_sd",
    ]
    assert result["overall_rate"] == pytest.approx(np.mean(y))


def test_duration_summary_preserves_positive_scale_features():
    result = _duration_summary(
        [100, 200, 300, 400, 500, 600, 700, 800],
        condition=[-0.5, 0.5] * 4,
        participant=["p1"] * 4 + ["p2"] * 4,
        item=["i1", "i2"] * 4,
    )
    assert set(result) == {
        "median",
        "mean",
        "q90",
        "q99",
        "coefficient_of_variation",
        "condition_median_ratio",
        "participant_log_median_sd",
        "item_log_median_sd",
        "nonfinite_fraction",
    }
    assert result["median"] > 0
    assert result["condition_median_ratio"] > 0


def test_binary_ppc_rejects_non_fit_objects():
    with pytest.raises(GP3BayesError, match="gp3bayes_fit"):
        gp.check_binary_posterior_predictive(object())


def test_duration_ppc_rejects_non_fit_objects():
    with pytest.raises(GP3BayesError, match="gp3bayes_fit"):
        gp.check_duration_posterior_predictive(object())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"draws": 49}, "draws"),
        ({"seed": -1}, "seed"),
        ({"pass_probability": 0}, "pass_probability"),
        ({"review_probability": 1}, "review_probability"),
        (
            {"pass_probability": 0.95, "review_probability": 0.80},
            "must be smaller",
        ),
    ],
)
def test_binary_ppc_validates_restricted_controls(kwargs, match):
    with pytest.raises(GP3BayesError, match=match):
        gp.check_binary_posterior_predictive(_binary_fit(), **kwargs)


def test_binary_ppc_returns_structural_parity_and_conservative_claims():
    result = gp.check_binary_posterior_predictive(_binary_fit(), draws=50, seed=17)
    assert result.family == "binary"
    assert result.draws == 50
    assert result.seed == 17
    assert result.replicated.shape == (50, 6)
    assert list(result.observed) == result.checks["statistic"].tolist()
    assert result.status in {"pass", "review", "fail"}
    assert np.isfinite(result.brier_score)
    assert result.posterior_predictive_performed is True
    assert result.adequacy_established is False


def test_binary_ppc_is_seed_reproducible():
    fit = _binary_fit()
    first = gp.check_binary_posterior_predictive(fit, draws=50, seed=19)
    second = gp.check_binary_posterior_predictive(fit, draws=50, seed=19)
    pd.testing.assert_frame_equal(first.replicated, second.replicated)
    assert first.brier_score == pytest.approx(second.brier_score)


def test_duration_ppc_returns_structural_parity_and_positive_draw_summaries():
    result = gp.check_duration_posterior_predictive(
        _duration_fit(), draws=50, seed=23
    )
    assert result.family == "duration"
    assert result.outcome_unit == "milliseconds"
    assert result.draws == 50
    assert result.replicated.shape == (50, 9)
    assert np.isfinite(result.replicated.to_numpy(dtype=float)).all()
    assert np.isfinite(result.log_scale_rmse)
    assert result.posterior_predictive_performed is True
    assert result.adequacy_established is False


def test_ppc_public_signatures_match_frozen_control_surface():
    assert list(inspect.signature(gp.check_binary_posterior_predictive).parameters) == [
        "fit",
        "draws",
        "seed",
        "pass_probability",
        "review_probability",
    ]
    assert list(inspect.signature(gp.check_duration_posterior_predictive).parameters) == [
        "fit",
        "draws",
        "seed",
        "pass_probability",
        "review_probability",
    ]


def test_ppc_public_signatures_expose_no_unrestricted_backend_escape_hatches():
    forbidden = {"kwargs", "formula", "family", "backend", "algorithm", "prior", "stanvars"}
    for function in (
        gp.check_binary_posterior_predictive,
        gp.check_duration_posterior_predictive,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)


def test_gpb_py07_historical_preclosure_ledger_floor_is_preserved():
    counts = gp.parity_counts()
    assert counts["implemented"] >= 33
    assert counts["implemented_initial"] == 1
    assert sum(counts.values()) == 458
