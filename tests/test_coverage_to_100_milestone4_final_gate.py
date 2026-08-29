from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.fitting as fitting
import gp3bayespy.posterior_validation_core as pvc
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_final_gate_fitting_success_and_missing_prior_class(monkeypatch):
    monkeypatch.setattr(fitting, "_pymc_available", lambda: True)
    assert fitting._require_pymc("exercise the successful backend gate") is None

    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("x",),
        random_slope=False,
    )
    priors = gp.create_prior_specification(contract)

    with pytest.raises(GP3BayesError, match="exactly one `sigma` row"):
        fitting._prior_row(priors, "sigma")


def test_final_gate_posterior_validation_false_branches(monkeypatch):
    monkeypatch.setattr(pvc, "_validate_fit_like", lambda fit: fit)

    components = {
        "b": np.arange(84, dtype=float).reshape(7, 12),
    }
    monkeypatch.setattr(pvc, "_posterior_components", lambda fit: components)

    trace = pvc.plot_sampling_diagnostics(
        object(),
        type="trace",
        variables="b",
    )
    assert len(trace.axes) == 1
    assert trace.axes[0].get_legend() is None

    sampler = pd.DataFrame(
        {
            "Parameter": ["diverging"] * 4,
            "Chain": [1, 1, 2, 2],
            "Iteration": [1, 2, 1, 2],
            "Value": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: sampler,
    )

    divergence = pvc.plot_sampling_diagnostics(
        object(),
        type="divergence",
    )
    assert divergence.axes
