from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

ppc = importlib.import_module("gp3bayespy.ppc")
u = importlib.import_module("gp3bayespy.unified_workflow_api")
aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
hea = importlib.import_module("gp3bayespy.hierarchical_effects_advanced")
ds = importlib.import_module("gp3bayespy.design_support_diagnostics")


def test_ppc_duration_two_condition_ratio_final_branch():
    summary = ppc._duration_summary(
        [1.0, 2.0, 4.0, 8.0],
        condition=[0, 0, 1, 1],
        participant=["p1", "p1", "p2", "p2"],
        item=None,
    )
    assert np.isclose(summary["condition_median_ratio"], 4.0)


def test_unified_explicit_pupil_and_unsupported_family_guards(monkeypatch):
    monkeypatch.setattr(u, "validate_gp3bayes_object", lambda *a, **k: True)
    monkeypatch.setattr(u, "_family", lambda x: "pupil")
    with pytest.raises(GP3BayesError, match="binary and duration"):
        u.estimate_model_estimands(object())

    monkeypatch.setattr(u, "_family", lambda x: "mystery")
    with pytest.raises(GP3BayesError, match="Unsupported"):
        u.estimate_model_estimands(object())


def test_advanced_capabilities_and_hierarchical_level_resolution():
    capabilities = aow.bayesian_backend_capabilities()
    assert isinstance(capabilities, pd.DataFrame)
    assert not capabilities.empty

    data = pd.DataFrame({"pid": ["a", "b"]})
    fit = SimpleNamespace(
        specification=SimpleNamespace(
            prepared=SimpleNamespace(
                data=data,
                contract=SimpleNamespace(mappings={"participant": "pid"}),
            )
        )
    )
    assert hea._levels(fit, "participant", 2) == ["a", "b"]
    assert hea._levels(fit, "participant", 3) == ["1", "2", "3"]


def test_design_declared_column_order_and_empty_missingness_grouping():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        condition_col="c",
        predictors=("x",),
    )
    declared = ds._declared_columns(contract)
    assert declared[:3] == ("y", "p", "c")
    assert "x" in declared

    data = pd.DataFrame({"y": [], "p": [], "c": [], "x": []})
    audit = ds.audit_missingness_structure(data, contract)
    assert audit.grouping_table.empty
