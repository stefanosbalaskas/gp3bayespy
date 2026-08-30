from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

po = importlib.import_module("gp3bayespy.posterior")
pb = importlib.import_module("gp3bayespy.prior_posterior_bridge")


def _binary_spec(seed=5301):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=3,
        trials_per_participant=5,
        n_items=2,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("trial_covariate",),
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return gp.specify_binary_model(prepared)


def test_posterior_summary_repr_array_mapping_and_selection(monkeypatch):
    table = pd.DataFrame({"variable": ["b"]})
    no_unit = po._PosteriorSummaryResult("0.1", "binary", 0.95, table, {}, outcome_unit=None)
    assert "Outcome unit:" not in repr(no_unit)
    unit = po._PosteriorSummaryResult(
        "0.1", "duration", 0.95, table, {}, outcome_unit="milliseconds"
    )
    assert "Outcome unit: milliseconds" in repr(unit)

    arr = np.array([[1.0, 2.0]])
    assert po._array_values(arr).shape == (1, 2)

    prepared_contract = SimpleNamespace(mappings={"participant": "pid"})
    fit = SimpleNamespace(
        specification=SimpleNamespace(
            contract=None,
            prepared=SimpleNamespace(contract=prepared_contract),
        )
    )
    assert po._mapping_name(fit, "participant") == "pid"

    monkeypatch.setattr(
        po,
        "_posterior_components",
        lambda x: {"b_x": np.ones((2, 3)), "sd_p": np.ones((2, 3))},
    )
    with pytest.raises(GP3BayesError):
        po._select_components(object(), variables=("",))
    with pytest.raises(GP3BayesError):
        po._select_components(object(), regex=1)  # type: ignore[arg-type]


def test_posterior_chain_table_none_stats_energy_and_zero_variance(monkeypatch):
    monkeypatch.setattr(
        po,
        "_posterior_components",
        lambda fit: {"b_x": np.ones((2, 4))},
    )

    varying = SimpleNamespace(
        backend_fit=SimpleNamespace(
            sample_stats={"energy": np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 6.0, 10.0]])}
        )
    )
    table = po._chain_table(varying, 12)
    assert table["divergences"].isna().all()
    assert table["treedepth_hits"].isna().all()
    assert np.isfinite(table["ebfmi"]).all()

    constant = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats={"energy": np.ones((2, 4))})
    )
    table2 = po._chain_table(constant, 12)
    assert table2["ebfmi"].isna().all()

    no_stats = SimpleNamespace(backend_fit=SimpleNamespace(sample_stats={}))
    table3 = po._chain_table(no_stats, 12)
    assert table3["ebfmi"].isna().all()


def test_prior_bridge_variables_inferred_non_dataframe_posterior_and_mpl_error(monkeypatch):
    spec = _binary_spec()
    fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
        specification=spec,
    )

    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *a, **k: pd.DataFrame(
            {
                "b_Intercept": np.linspace(-0.2, 0.2, 60),
                "b_trial_covariate": np.linspace(-0.1, 0.1, 60),
            }
        ),
    )
    prior = pb.simulate_declared_prior_draws(
        fit,
        variables=None,
        ndraws=50,
        seed=2,
    )
    assert set(prior.columns) == {"b_Intercept", "b_trial_covariate"}

    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *a, **k: [
            {"b_Intercept": -0.2},
            {"b_Intercept": 0.0},
            {"b_Intercept": 0.3},
        ],
    )
    bridge = pb.prior_posterior_bridge(fit, ndraws=50, seed=3)
    assert bridge.variables == ("b_Intercept",)

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("synthetic missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(GP3BayesError):
        pb._mpl()
