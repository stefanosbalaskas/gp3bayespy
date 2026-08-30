from __future__ import annotations

import importlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gp3bayespy.exceptions import GP3BayesError

pe = importlib.import_module("gp3bayespy.postfit_exploration")


class Dataset(dict):
    @property
    def data_vars(self):
        return self


class MatrixLike:
    columns = ["x", "y"]

    def __array__(self, dtype=None, copy=None):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        if dtype is not None:
            data = data.astype(dtype)
        return data.copy() if copy else data


def test_postfit_draw_matrix_fit_matrixlike_and_mcmc_selection(monkeypatch):
    components = {
        "a": np.arange(8.0).reshape(2, 4),
        "b": np.arange(8.0, 16.0).reshape(2, 4),
    }
    monkeypatch.setattr(pe, "_posterior_components", lambda fit: components)
    fit = SimpleNamespace(fit_performed=True)

    frame = pe._draw_matrix(fit)
    assert list(frame.columns) == ["a", "b"]
    assert list(pe._draw_matrix(MatrixLike()).columns) == ["x", "y"]

    only_a = pe.mcmc_diagnostic_table(fit, variables="a")
    assert list(only_a["variable"]) == ["a"]
    only_b = pe.mcmc_diagnostic_table(fit, regex=r"^b$")
    assert list(only_b["variable"]) == ["b"]
    with pytest.raises(GP3BayesError):
        pe.mcmc_diagnostic_table(fit, variables=("missing",))
    with pytest.raises(re.error):
        pe.mcmc_diagnostic_table(fit, regex="[")

    assert np.isfinite(pe._ess_1d(np.arange(12.0)))
    assert np.isfinite(pe._ess_1d(np.array([1.0, -1.0] * 6)))


def test_postfit_sampler_conversion_component_and_quality_except(monkeypatch):
    stats = Dataset(
        {
            "bad": object(),
            "scalar": np.array([1.0, 2.0]),
            "vector": np.arange(24.0).reshape(2, 3, 4),
        }
    )
    fit = SimpleNamespace(backend_fit=SimpleNamespace(sample_stats=stats), sampling={})
    extracted = pe.extract_sampler_diagnostics(fit)
    assert len(extracted) == 24
    assert set(extracted["Parameter"]) == {"vector"}

    no_energy = Dataset(
        {
            "diverging": np.zeros((2, 3)),
            "tree_depth": np.ones((2, 3)),
        }
    )
    table = pe.sampler_diagnostic_table(
        SimpleNamespace(
            backend_fit=SimpleNamespace(sample_stats=no_energy),
            sampling={"max_treedepth": 12},
        )
    )
    assert not table["metric"].astype(str).str.startswith("ebfmi").any()

    constant_energy = Dataset({"energy": np.ones((2, 4))})
    energy_table = pe.sampler_diagnostic_table(
        SimpleNamespace(
            backend_fit=SimpleNamespace(sample_stats=constant_energy),
            sampling={},
        )
    )
    ebfmi = energy_table[energy_table["metric"].astype(str).str.startswith("ebfmi")]
    assert ebfmi["flagged"].all()

    diagnostic = pd.DataFrame(
        {
            "variable": ["a"],
            "sd": [1.0],
            "rhat": [1.0],
            "ess_bulk": [500.0],
            "ess_tail": [500.0],
            "mcse_mean": [0.01],
        }
    )
    monkeypatch.setattr(pe, "mcmc_diagnostic_table", lambda *a, **k: diagnostic)
    monkeypatch.setattr(
        pe,
        "sampler_diagnostic_table",
        lambda *a, **k: (_ for _ in ()).throw(GP3BayesError("no sampler")),
    )
    quality = pe.summarise_mcmc_quality(SimpleNamespace(family="binary"))
    assert quality.sampler.empty
    assert quality.flagged_sampler_metrics == 0


def _group_fit():
    data = pd.DataFrame({"pid": ["p1", "p2"]})
    contract = SimpleNamespace(mappings={"participant": "pid", "item": None})
    specification = SimpleNamespace(
        prepared=SimpleNamespace(data=data),
        contract=contract,
    )
    return SimpleNamespace(specification=specification)


def test_postfit_group_effect_direct_array_flattened_canonical_and_errors(monkeypatch):
    fit = _group_fit()

    monkeypatch.setattr(
        pe,
        "_posterior_components",
        lambda x: {
            "sd_participant": np.ones((2, 3)),
            "participant_z": np.arange(12.0).reshape(2, 3, 2),
        },
    )
    table = pe.group_effect_table(fit)
    assert len(table) == 2

    monkeypatch.setattr(
        pe,
        "_posterior_components",
        lambda x: {
            "sd_participant": np.ones((2, 3)),
            "participant_z": np.ones((2, 3, 2)),
            "participant_z[1]": np.full((2, 3), 2.0),
            "participant_z[2]": np.full((2, 3), 3.0),
        },
    )
    direct = pe.group_effect_table(fit, groups="participant")
    assert len(direct) == 2

    monkeypatch.setattr(
        pe,
        "_posterior_components",
        lambda x: {
            "r_participant[p1]": np.ones((2, 3)),
            "r_participant[p2]": np.ones((2, 3)) * 2,
        },
    )
    canonical = pe.group_effect_table(fit)
    assert set(canonical["level"]) == {"p1", "p2"}

    with pytest.raises(GP3BayesError):
        pe.group_effect_table(fit, groups=("unknown",))

    monkeypatch.setattr(pe, "_posterior_components", lambda x: {"foo": np.ones((2, 3))})
    with pytest.raises(GP3BayesError):
        pe.group_effect_table(fit)


def test_postfit_loo_raw_quantity_comparison_and_weight_object_paths():
    raw = SimpleNamespace(
        estimates=pd.DataFrame(
            {"estimate": [1.0], "se": [0.1]},
            index=["elpd_loo"],
        )
    )
    table = pe.loo_summary_table(SimpleNamespace(raw=raw))
    assert table.iloc[0]["quantity"] == "elpd_loo"

    already = pd.DataFrame({"quantity": ["elpd_loo"], "estimate": [1.0], "se": [0.1]})
    assert pe.loo_summary_table(SimpleNamespace(estimates=already)).equals(already)

    comparison = pd.DataFrame(
        {
            "model": ["m1"],
            "elpd_diff": [0.0],
            "se_diff": [0.0],
        }
    )
    out = pe.model_comparison_table(SimpleNamespace(comparison=comparison))
    assert out.iloc[0]["model"] == "m1"

    weights = pe.model_weights_table(SimpleNamespace(weights={"m1": 0.6, "m2": 0.4}))
    assert np.isclose(weights["weight"].sum(), 1.0)
