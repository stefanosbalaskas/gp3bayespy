from __future__ import annotations

import importlib
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")

po = importlib.import_module("gp3bayespy.posterior")
loo = importlib.import_module("gp3bayespy.loo")
ds = importlib.import_module("gp3bayespy.design_support_diagnostics")
ppc = importlib.import_module("gp3bayespy.ppc")
eg = importlib.import_module("gp3bayespy.evidence_graphics_gg")
u = importlib.import_module("gp3bayespy.unified_workflow_api")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_posterior_numeric_selection_and_array_paths(monkeypatch):
    with pytest.raises(GP3BayesError):
        po._numeric_scalar(True, "x")
    with pytest.raises(GP3BayesError):
        po._numeric_scalar(2.0, "x", upper=1)
    assert po._validate_probability(0.5, "p", open=True) == 0.5
    with pytest.raises(GP3BayesError):
        po._validate_probability(0, "p", open=True)

    monkeypatch.setattr(
        po,
        "_posterior_components",
        lambda fit: {"foo": np.ones((2, 3))},
    )
    with pytest.raises(GP3BayesError):
        po._select_components(object(), parameters_only=True)
    with pytest.raises(GP3BayesError):
        po._select_components(object(), regex=1)  # type: ignore[arg-type]

    monkeypatch.setattr(
        po,
        "_posterior_components",
        lambda fit: {"b_x": np.ones((2, 3))},
    )
    arr = po.extract_draws(object(), format="array")
    assert arr.shape == (2, 3, 1)

    spec = SimpleNamespace(
        prepared=SimpleNamespace(contract=SimpleNamespace(mappings={"participant": "pid"})),
        contract=None,
    )
    assert po._mapping_name(SimpleNamespace(specification=spec), "participant") == "pid"


def test_loo_pointwise_input_branches_and_table_helpers():
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(object())

    x1 = SimpleNamespace(pointwise=np.array([1.0, 2.0]), pareto_k=np.array([0.2, 0.8]))
    t1 = loo.loo_pointwise_table(x1)
    assert len(t1) == 2

    x2 = SimpleNamespace(
        pointwise=np.array([[1.0, 0.1], [2.0, 0.2]]),
        pareto_k=np.array([0.2, 0.3]),
    )
    assert len(loo.loo_pointwise_table(x2)) == 2

    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(SimpleNamespace(pointwise=np.ones((2, 2, 2)), pareto_k=[0.2, 0.3]))
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(SimpleNamespace(pointwise=np.array([1.0]), pareto_k=[0.2, 0.3]))

    frame = pd.DataFrame({"pareto_k": [0.2]})
    assert loo._table(frame).equals(frame)


def test_design_missingness_empty_and_threshold_branches():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x",),
    )
    with pytest.raises(GP3BayesError):
        ds.audit_missingness_structure(
            pd.DataFrame({"y": [0], "p": ["p1"], "x": [1.0]}),
            contract,
            review_fraction=0.5,
            fail_fraction=0.2,
        )

    empty = pd.DataFrame({"y": [], "p": [], "x": []})
    audit = ds.audit_missingness_structure(empty, contract)
    assert not audit.column_table.empty

    data = pd.DataFrame(
        {"y": [0, 1, 0, 1], "p": ["a", "a", "b", "b"], "x": [1.0, np.nan, 3.0, 4.0]}
    )
    review = ds.audit_missingness_structure(data, contract, review_fraction=0.2, fail_fraction=0.8)
    assert "review" in set(review.column_table["status"])


def test_ppc_summary_tail_branches():
    binary = ppc._binary_summary(
        [0, 1],
        condition=None,
        participant=["p1", "p1"],
        item=None,
    )
    assert np.isnan(binary["condition_rate_contrast"])
    assert np.isnan(binary["item_rate_sd"])

    one_condition = ppc._binary_summary(
        [0, 1],
        condition=[0, 0],
        participant=["p1", "p2"],
        item=["i1", "i1"],
    )
    assert np.isnan(one_condition["condition_rate_contrast"])

    invalid = ppc._duration_summary(
        [np.nan, -1],
        condition=None,
        participant=["p1", "p1"],
        item=None,
    )
    assert invalid["median"] == np.inf

    one = ppc._duration_summary(
        [1.0],
        condition=[0],
        participant=["p1"],
        item=["i1"],
    )
    assert np.isnan(one["condition_median_ratio"])
    assert np.isnan(one["coefficient_of_variation"])


def test_evidence_status_plot_and_unified_status_edges():
    table = pd.DataFrame({"status": ["pass", "review"], "label": ["a", "b"]})
    assert eg._status_plot(table, "x", "label").axes
    assert eg._status_plot(pd.DataFrame({"status": ["pass"]}), "x").axes

    status = u.model_workflow_status(SimpleNamespace())
    assert isinstance(status, pd.DataFrame)
    fit = SimpleNamespace(
        fit_performed=True,
        specification=SimpleNamespace(
            prepared=SimpleNamespace(data=pd.DataFrame({"x": [1]}), transformations={}),
            contract=SimpleNamespace(contract_version="0.1"),
            priors=object(),
            formula="x",
        ),
    )
    status2 = u.model_workflow_status(fit)
    assert isinstance(status2, pd.DataFrame)
