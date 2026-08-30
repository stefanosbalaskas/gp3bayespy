from __future__ import annotations

import importlib

import pandas as pd

import gp3bayespy as gp

r = importlib.import_module("gp3bayespy.readiness")


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


def test_supported_identifier_second_object_dtype_branch(monkeypatch):
    series = pd.Series([1, "a", None], dtype=object)
    original = r.ptypes.is_string_dtype
    monkeypatch.setattr(r.ptypes, "is_string_dtype", lambda dtype: False)
    try:
        assert r._supported_identifier(series)
    finally:
        monkeypatch.setattr(r.ptypes, "is_string_dtype", original)


def test_trial_structure_unsupported_identifier_return():
    rows, add = _collector()
    data = pd.DataFrame(
        {
            "p": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "trial": ["t1", "t2"],
        }
    )
    r._audit_trial_structure(data, "p", "trial", add)
    assert rows == []


def test_interaction_frame_empty_after_dropna():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("a", "b"),
        interaction=("a", "b"),
    )
    data = pd.DataFrame(
        {
            "a": ["x", "y", None, None],
            "b": [None, None, "u", "v"],
        }
    )
    rows, add = _collector()
    r._audit_interaction_structure(data, contract, add)
    assert rows[-1][2] == "fail"
    assert "No complete interaction combinations" in rows[-1][3]
