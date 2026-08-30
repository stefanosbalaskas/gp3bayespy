from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from gp3bayespy.exceptions import GP3BayesError

s = importlib.import_module("gp3bayespy.sensitivity")


class BlankPath:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ""


class FakeSBC:
    def __init__(self, ranks, simulations):
        self.ranks = ranks
        self.simulations = simulations
        self.plan = SimpleNamespace(
            specification=SimpleNamespace(family="binary"),
            backend="analytic",
            n_sims=2,
        )


def test_evidence_report_empty_path_guard(monkeypatch):
    evidence = s.ModelEvidence(
        "0.2",
        "binary",
        None,
        {},
        pd.DataFrame(),
    )
    monkeypatch.setattr(s, "Path", BlankPath)
    with pytest.raises(GP3BayesError, match="non-empty path"):
        s.create_model_evidence_report(evidence, "")


def test_random_slope_object_comparison_branch():
    table = pd.DataFrame(
        {
            "alternative": ["a"],
            "reference_median": [1.0],
            "alternative_median": [1.1],
            "alternative_lower": [0.9],
            "alternative_upper": [1.3],
        }
    )
    holder = SimpleNamespace(comparison=SimpleNamespace(table=table))
    assert s.random_slope_sensitivity_table(holder).equals(table)


def test_powerscale_raw_conversion_exception():
    with pytest.raises(GP3BayesError, match="Could not convert"):
        s.powerscale_sensitivity_table(SimpleNamespace(raw=1))


def test_sbc_result_ranks_simulations_and_overview_branches(monkeypatch):
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
    monkeypatch.setattr(aow, "SBCResult", FakeSBC)

    ranks = pd.DataFrame(
        {
            "parameter": ["b", "b"],
            "rank": [1, 2],
        }
    )
    sims = pd.DataFrame(
        {
            "parameter": ["b"],
            "rank": [1],
        }
    )
    with_ranks = FakeSBC(ranks, sims)
    assert s.sbc_stats_table(with_ranks).equals(ranks)
    overview = s.sbc_overview_table(with_ranks)
    assert overview.iloc[0]["status"] == "completed"
    assert overview.iloc[0]["diagnostics_inspected"]

    without_ranks = FakeSBC(pd.DataFrame(), sims)
    assert s.sbc_stats_table(without_ranks).equals(sims)
    overview2 = s.sbc_overview_table(without_ranks)
    assert overview2.iloc[0]["status"] == "completed"
    assert not overview2.iloc[0]["diagnostics_inspected"]
