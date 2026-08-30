from __future__ import annotations

import importlib

import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

r = importlib.import_module("gp3bayespy.readiness")


def _contract():
    return gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=(),
        random_slope=False,
    )


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


def test_readiness_repr_issue_and_no_issue_paths():
    contract = _contract()
    passing = r.ReadinessAudit(
        "0.1",
        "binary",
        "hierarchical_binary",
        True,
        "ready",
        2,
        2,
        {"pass": 1, "warn": 0, "fail": 0},
        pd.DataFrame(
            [
                {
                    "check_id": "x",
                    "category": "data",
                    "status": "pass",
                    "message": "ok",
                    "n_affected": None,
                }
            ]
        ),
        {},
        {},
        contract,
    )
    assert "Issues:" not in repr(passing)

    warning = r.ReadinessAudit(
        "0.1",
        "binary",
        "hierarchical_binary",
        True,
        "ready_with_warnings",
        2,
        2,
        {"pass": 0, "warn": 1, "fail": 0},
        pd.DataFrame(
            [
                {
                    "check_id": "w",
                    "category": "data",
                    "status": "warn",
                    "message": "review",
                    "n_affected": 1,
                }
            ]
        ),
        {},
        {},
        contract,
    )
    text = repr(warning)
    assert "Issues:" in text and "[WARN]" in text


def test_readiness_observed_levels_and_structure_missing_paths():
    duplicate = pd.DataFrame([[1, 2]], columns=["x", "x"])
    assert r._observed_levels(duplicate, "x") is None

    rows, add = _collector()
    r._audit_trial_structure(pd.DataFrame({"p": ["p1"]}), "p", "trial", add)
    assert rows == []

    r._audit_condition_structure(pd.DataFrame({"p": ["p1"]}), "p", "condition", False, add)
    assert rows == []

    rows.clear()
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("a", "b"),
        interaction=("a", "b"),
    )
    no_complete = pd.DataFrame(
        {
            "a": ["x", "y", None],
            "b": ["u", None, "v"],
        }
    )
    r._audit_interaction_structure(no_complete, contract, add)
    assert rows[-1][2] == "fail"
    assert "Fewer than two interaction combinations" in rows[-1][3]


def test_readiness_internal_status_guard_and_no_existing_columns(monkeypatch):
    contract = _contract()
    data = pd.DataFrame({"y": [0, 1], "p": ["p1", "p2"]})

    def invalid_status(data, mappings, add):
        add("synthetic", "internal", "unsupported", "bad", None)

    monkeypatch.setattr(r, "_audit_identifier_types", invalid_status)
    with pytest.raises(GP3BayesError, match="unsupported readiness-check status"):
        r.audit_model_readiness(data, contract)

    monkeypatch.undo()
    absent = r.audit_model_readiness(pd.DataFrame({"other": [1, 2]}), contract)
    row = absent.checks.loc[absent.checks["check_id"].eq("analysis_missingness")]
    assert row.iloc[0]["status"] == "fail"
    assert "No declared analysis columns" in row.iloc[0]["message"]


def test_readiness_object_identifier_scalar_matrix():
    good = pd.Series(["a", None, 1, True], dtype=object)
    assert r._supported_identifier(good)
    bad = pd.Series([{"x": 1}], dtype=object)
    assert not r._supported_identifier(bad)
