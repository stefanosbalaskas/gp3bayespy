from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.hierarchical_effects_advanced as h
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class Values:
    def __init__(self, values):
        self.values = np.asarray(values)


def _fit(random_slope: bool = False, family: str = "binary"):
    chains, draws, levels = 2, 12, 3
    rng = np.random.default_rng(2201)
    if random_slope:
        posterior = {
            "participant_chol_stds": Values(np.abs(rng.normal(0.4, 0.05, (chains, draws, 2)))),
            "participant_z": Values(rng.normal(0, 1, (chains, draws, levels, 2))),
            "sd_item": Values(np.abs(rng.normal(0.2, 0.03, (chains, draws)))),
            "item_z": Values(rng.normal(0, 1, (chains, draws, 2))),
        }
    else:
        posterior = {
            "sd_participant": Values(np.abs(rng.normal(0.4, 0.05, (chains, draws)))),
            "participant_z": Values(rng.normal(0, 1, (chains, draws, levels))),
            "sd_item": Values(np.abs(rng.normal(0.2, 0.03, (chains, draws)))),
            "item_z": Values(rng.normal(0, 1, (chains, draws, 2))),
        }
    if family == "duration":
        posterior["sigma"] = Values(np.abs(rng.normal(0.3, 0.02, (chains, draws))))
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "item_id": ["i1", "i2", "i1"],
            "condition": [-0.5, 0.5, -0.5],
        }
    )
    contract = SimpleNamespace(
        mappings={
            "participant": "participant_id",
            "item": "item_id",
            "condition": "condition",
        }
    )
    prepared = SimpleNamespace(data=data, contract=contract)
    specification = SimpleNamespace(prepared=prepared)
    return SimpleNamespace(
        fit_performed=True,
        family=family,
        backend_fit=SimpleNamespace(posterior=posterior),
        specification=specification,
    )


def test_group_arrays_draw_tables_rank_and_guard_paths():
    fit = _fit(False)
    arrays = h._group_arrays(fit)
    assert set(arrays) == {"participant_id", "item_id"}

    table = h.group_effect_draws_table(
        fit,
        groups=("participant_id",),
        coefficients=("Intercept",),
        ndraws=8,
        seed=2,
    )
    assert len(table) == 8 * 3

    ranks = h.group_effect_rank_probability_table(
        fit,
        "participant_id",
        ndraws=8,
        seed=2,
    )
    assert len(ranks) == 3
    assert np.all((ranks["probability_highest"] >= 0) & (ranks["probability_highest"] <= 1))

    slope_fit = _fit(True)
    slope = h.group_effect_draws_table(
        slope_fit,
        groups="participant_id",
        coefficients="condition",
        ndraws=6,
    )
    assert set(slope["coefficient"]) == {"condition"}

    with pytest.raises(GP3BayesError):
        h._integer(True, "x")
    with pytest.raises(GP3BayesError):
        h.group_effect_draws_table(fit, groups="missing")
    with pytest.raises(GP3BayesError):
        h.group_effect_draws_table(fit, coefficients="missing")
    with pytest.raises(GP3BayesError):
        h.group_effect_draws_table(fit, seed=-1)
    with pytest.raises(GP3BayesError):
        h.group_effect_draws_table(
            fit,
            groups="participant_id",
            ndraws=8,
            max_rows=2,
        )

    no_group = _fit(False)
    no_group.backend_fit.posterior = {"x": Values(np.ones((2, 10)))}
    with pytest.raises(GP3BayesError):
        h._group_arrays(no_group)


def test_variance_partition_binary_duration_and_plots():
    bfit = _fit(False, "binary")
    bpart = h.random_intercept_variance_partition(
        bfit,
        probs=(0.1, 0.5, 0.9),
    )
    btab = h.random_intercept_variance_partition_table(bpart)
    assert "logit_residual" in set(btab["component"])
    assert np.all((btab["fraction_median"] >= 0) & (btab["fraction_median"] <= 1))

    dfit = _fit(False, "duration")
    dpart = h.random_intercept_variance_partition(dfit)
    assert "lognormal_residual" in set(dpart.table["component"])

    with pytest.raises(GP3BayesError):
        h.random_intercept_variance_partition(bfit, probs=(0, 0.5, 0.9))
    with pytest.raises(GP3BayesError):
        h.random_intercept_variance_partition_table(object())  # type: ignore[arg-type]

    no_sigma = _fit(False, "duration")
    no_sigma.backend_fit.posterior.pop("sigma")
    with pytest.raises(GP3BayesError):
        h.random_intercept_variance_partition(no_sigma)

    draws = h.group_effect_draws_table(
        bfit,
        groups="participant_id",
        ndraws=10,
    )
    ranks = h.group_effect_rank_probability_table(
        bfit,
        "participant_id",
        ndraws=10,
    )
    assert h.plot_group_effect_distribution(draws, max_levels=2).axes
    assert h.plot_group_effect_rank_probability(ranks).axes
    assert h.plot_random_intercept_variance_partition(bpart).axes
    assert h.plot_random_intercept_variance_partition(bpart.table).axes

    with pytest.raises(GP3BayesError):
        h.plot_group_effect_distribution(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        h.plot_group_effect_rank_probability(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        h.plot_random_intercept_variance_partition(pd.DataFrame({"x": [1]}))
