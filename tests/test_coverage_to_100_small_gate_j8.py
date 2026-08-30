from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd

import gp3bayespy as gp

ds = importlib.import_module("gp3bayespy.design_support_diagnostics")
u = importlib.import_module("gp3bayespy.unified_workflow_api")


def test_missingness_no_declared_columns_available_branch():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x",),
    )
    audit = ds.audit_missingness_structure(
        pd.DataFrame({"other": [1, 2]}),
        contract,
    )
    assert set(audit.absent_columns) == {"y", "p", "x"}
    assert audit.grouping_table.empty


def test_workflow_status_source_fit_supplies_specification_branch():
    spec = SimpleNamespace(
        priors=object(),
        formula="y ~ 1",
        prepared=SimpleNamespace(
            data=pd.DataFrame({"y": [0, 1]}),
            transformations={},
        ),
        contract=SimpleNamespace(contract_version="0.1"),
    )
    wrapper = SimpleNamespace(
        fit=SimpleNamespace(
            fit_performed=True,
            specification=spec,
        )
    )
    status = u.model_workflow_status(wrapper)
    assert "specification" in set(status["stage"])
