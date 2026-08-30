from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.predictive as predictive
from gp3bayespy.exceptions import GP3BayesError

ds = importlib.import_module("gp3bayespy.design_support_diagnostics")
postfit = importlib.import_module("gp3bayespy.postfit_exploration")
ppc = importlib.import_module("gp3bayespy.ppc")


class _ModelContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _HalfStudentT:
    @staticmethod
    def dist(**kwargs):
        return 1.0


class _Math:
    @staticmethod
    def dot(a, b):
        return np.dot(a, b)


class _FakePyMC:
    HalfStudentT = _HalfStudentT
    math = _Math()

    @staticmethod
    def Model():
        return _ModelContext()

    @staticmethod
    def Normal(name, **kwargs):
        shape = kwargs.get("shape")
        if shape is None:
            return 0.0
        if isinstance(shape, tuple):
            return np.zeros(shape, dtype=float)
        return np.zeros(int(shape), dtype=float)

    @staticmethod
    def LKJCholeskyCov(name, **kwargs):
        return np.eye(2), None, None

    @staticmethod
    def Deterministic(name, value):
        return value


def _prior_row(*args, **kwargs):
    return pd.Series(
        {
            "location": 0.0,
            "scale": 1.0,
            "df": 3.0,
            "shape": 2.0,
        }
    )


def _invalid_random_slope_specification(family: str):
    outcome = "y"
    participant = "participant"
    contract = SimpleNamespace(
        family=family,
        mappings={
            "outcome": outcome,
            "participant": participant,
            "item": None,
            "condition": None,
        },
        random_slope=True,
    )
    data = pd.DataFrame(
        {
            outcome: [1, 0] if family == "binary" else [1.0, 2.0],
            participant: ["p1", "p2"],
        }
    )
    return SimpleNamespace(
        prepared=SimpleNamespace(data=data),
        contract=contract,
        priors=object(),
    )


def test_declared_columns_duplicate_predictor_and_new_interaction_paths():
    contract = SimpleNamespace(
        mappings={"outcome": "y"},
        predictors=("x", "x"),
        interaction=("z", "z"),
    )
    assert ds._declared_columns(contract) == ("y", "x", "z")


def test_ppc_duration_single_condition_skips_ratio():
    summary = ppc._duration_summary(
        [1.0, 2.0, 4.0, 8.0],
        condition=["only", "only", "only", "only"],
        participant=["p1", "p1", "p2", "p2"],
        item=None,
    )
    assert np.isnan(summary["condition_median_ratio"])


def test_binary_pymc_intercept_only_and_missing_condition_guard(monkeypatch):
    specification = _invalid_random_slope_specification("binary")
    monkeypatch.setattr(binary, "_load_pymc", lambda: _FakePyMC)
    monkeypatch.setattr(binary, "_prior_row", _prior_row)
    monkeypatch.setattr(
        binary,
        "_fixed_model_matrix",
        lambda data, contract: (
            np.ones((len(data), 1), dtype=float),
            ("(Intercept)",),
        ),
    )
    with pytest.raises(GP3BayesError, match="condition_col"):
        binary._run_binary_pymc(specification, {})


def test_duration_summary_positive_ratio_and_pymc_defensive_guard(monkeypatch):
    summary = duration._duration_summary(
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array(["p1", "p1", "p2", "p2"]),
        None,
    )
    assert summary["condition_median_ratio"] > 0

    specification = _invalid_random_slope_specification("duration")
    monkeypatch.setattr(duration, "_load_pymc", lambda: _FakePyMC)
    monkeypatch.setattr(duration, "_prior_row", _prior_row)
    monkeypatch.setattr(
        duration,
        "_fixed_model_matrix",
        lambda data, contract: (
            np.ones((len(data), 1), dtype=float),
            ("(Intercept)",),
        ),
    )
    with pytest.raises(GP3BayesError, match="condition_col"):
        duration._run_duration_pymc(specification, {})


def test_predictive_final_type_and_max_rows_guards(monkeypatch):
    with pytest.raises(GP3BayesError, match="gp3bayes prediction"):
        predictive.prediction_exceedance_probability(object(), 0.5)

    fake_prediction = SimpleNamespace(draws=np.ones((3, 2), dtype=float))
    monkeypatch.setattr(
        predictive,
        "_advanced_prediction",
        lambda x: fake_prediction,
    )
    with pytest.raises(GP3BayesError, match="greater than or equal to 2"):
        predictive.prediction_pairwise_contrasts(
            object(),
            rows=[1, 2],
            measure="difference",
            max_rows=1,
        )


def test_ess_pair_sequence_after_dead_branch_cleanup():
    value = postfit._ess_1d(np.array([0.0, 1.0, 0.0, 1.0, 0.5]))
    assert np.isfinite(value)
    assert 0 < value <= 5
