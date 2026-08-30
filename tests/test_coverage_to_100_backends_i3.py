from __future__ import annotations

import builtins
import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from gp3bayespy.exceptions import GP3BayesError

be = importlib.import_module("gp3bayespy.backends")


def _caps(package_ok: bool, runtime_ok: bool, backend: str):
    return pd.DataFrame(
        {
            "backend": [backend],
            "backend_package_available": [package_ok],
            "backend_package_version": ["1.0" if package_ok else None],
            "external_runtime_available": [runtime_ok],
            "external_runtime_version": ["2.0" if runtime_ok else None],
            "external_runtime_path": ["/tmp/runtime" if runtime_ok else None],
            "ready_for_package_interface": [package_ok and runtime_ok],
            "algorithm": ["NUTS"],
            "model_family_scope": ["test"],
            "unrestricted_modeling": [False],
        }
    )


def test_backend_capability_and_environment_branch_matrix(monkeypatch):
    monkeypatch.setattr(be, "_version", lambda package: None)
    monkeypatch.setattr(
        be,
        "_available",
        lambda package: package == "cmdstanpy",
    )

    fake = types.ModuleType("cmdstanpy")
    fake.cmdstan_path = lambda: "/tmp/cmdstan"
    fake.cmdstan_version = lambda: (2, 35, 0)
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)

    caps = be.backend_capabilities()
    row = caps.loc[caps["backend"] == "cmdstanpy"].iloc[0]
    assert bool(row["external_runtime_available"])
    assert row["external_runtime_version"] == "2.35.0"

    broken = types.ModuleType("cmdstanpy")
    broken.cmdstan_path = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    broken.cmdstan_version = lambda: (2, 35, 0)
    monkeypatch.setitem(sys.modules, "cmdstanpy", broken)
    caps = be.backend_capabilities()
    row = caps.loc[caps["backend"] == "cmdstanpy"].iloc[0]
    assert not bool(row["external_runtime_available"])

    with pytest.raises(GP3BayesError):
        be.validate_backend_environment("bad")

    monkeypatch.setattr(
        be,
        "backend_capabilities",
        lambda: _caps(False, False, "pymc"),
    )
    env = be.validate_backend_environment(
        "pymc",
        compile_test=True,
        strict=False,
    )
    assert env.status == "fail"
    assert env.checks.iloc[-1]["detail"] == "prerequisite check failed"

    with pytest.raises(GP3BayesError):
        be.validate_backend_environment(
            "pymc",
            compile_test=True,
            strict=True,
        )

    monkeypatch.setattr(
        be,
        "backend_capabilities",
        lambda: _caps(True, True, "pymc"),
    )
    original_import = builtins.__import__

    def import_ok(name, *args, **kwargs):
        if name == "pymc":
            return types.ModuleType("pymc")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_ok)
    env = be.validate_backend_environment(
        "rstan",
        compile_test=True,
    )
    assert env.backend == "pymc"
    assert env.status == "pass"

    def import_fail(name, *args, **kwargs):
        if name == "pymc":
            raise ImportError("synthetic pymc import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_fail)
    env = be.validate_backend_environment(
        "pymc",
        compile_test=True,
    )
    assert env.status == "fail"

    monkeypatch.setattr(
        be,
        "backend_capabilities",
        lambda: _caps(True, True, "cmdstanpy"),
    )
    good_cmdstan = types.ModuleType("cmdstanpy")
    good_cmdstan.cmdstan_path = lambda: "/tmp/cmdstan"
    monkeypatch.setitem(sys.modules, "cmdstanpy", good_cmdstan)
    env = be.validate_backend_environment(
        "cmdstanr",
        compile_test=True,
    )
    assert env.backend == "cmdstanpy"
    assert env.status == "pass"

    bad_cmdstan = types.ModuleType("cmdstanpy")
    bad_cmdstan.cmdstan_path = lambda: (_ for _ in ()).throw(RuntimeError("cmdstan failure"))
    monkeypatch.setitem(sys.modules, "cmdstanpy", bad_cmdstan)
    env = be.validate_backend_environment(
        "cmdstanpy",
        compile_test=True,
    )
    assert env.status == "fail"


def test_backend_draw_summary_and_schema_edge_matrix(monkeypatch):
    summary = pd.DataFrame(
        {
            "variable": ["a"],
            "mean": [1.0],
            "sd": [0.5],
        }
    )
    out = be._draw_summary(summary)
    assert np.isnan(out.loc[0, "mcse_mean"])

    postfit = importlib.import_module("gp3bayespy.postfit_exploration")
    monkeypatch.setattr(
        postfit,
        "extract_posterior_draws",
        lambda *args, **kwargs: pd.DataFrame(
            {
                ".chain": [1, 1, 2, 2],
                ".iteration": [1, 2, 1, 2],
                "a": [1.0, 2.0, 3.0, 4.0],
                "all_nan": [np.nan, np.nan, np.nan, np.nan],
                "text": ["x", "y", "z", "w"],
            }
        ),
    )
    out = be._draw_summary(object())
    assert "a" in set(out["variable"])
    assert "all_nan" not in set(out["variable"])

    monkeypatch.setattr(
        postfit,
        "extract_posterior_draws",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    with pytest.raises(GP3BayesError):
        be._draw_summary(object())

    monkeypatch.setattr(
        postfit,
        "extract_posterior_draws",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad fit")),
    )
    with pytest.raises(GP3BayesError):
        be._draw_summary(object())

    with pytest.raises(GP3BayesError):
        be.capture_gp3bayes_schema(1)
    with pytest.raises(GP3BayesError):
        be.capture_gp3bayes_schema({}, max_depth=-1)

    schema = be.capture_gp3bayes_schema(
        {"a": [1, {"b": 2}]},
        max_depth=3,
    )
    assert not schema.fields.empty

    class NoLen:
        pass

    rows = be._schema_node(NoLen(), "root", 0, 1)
    assert rows[0]["length"] == 1
