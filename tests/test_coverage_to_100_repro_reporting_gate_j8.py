from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

rp = importlib.import_module("gp3bayespy.reproducibility")
rep = importlib.import_module("gp3bayespy.reporting")


class ColumnArray(np.ndarray):
    pass


def test_manifest_fit_without_spec_and_freeze_without_file():
    fit = SimpleNamespace(
        family="binary",
        specification=None,
        sampling={},
    )
    manifest = rp.create_analysis_manifest(fit=fit)
    # Without a retained specification, the manifest deliberately has no
    # specification-derived family even though the fit family was validated.
    assert manifest.family is None
    assert manifest.sampling == {}
    frozen = rp.freeze_analysis_manifest(manifest, file=None)
    assert frozen.frozen


def test_analysis_bundle_prediction_observed_none_and_no_loo(monkeypatch):
    pe = importlib.import_module("gp3bayespy.postfit_exploration")
    pr = importlib.import_module("gp3bayespy.predictive")

    token = pd.DataFrame({"x": [1.0]})
    monkeypatch.setattr(pe, "posterior_interval_table", lambda *a, **k: token)
    monkeypatch.setattr(pe, "summarise_mcmc_quality", lambda *a, **k: token)
    monkeypatch.setattr(pe, "group_effect_table", lambda *a, **k: token)
    monkeypatch.setattr(pe, "variance_component_table", lambda *a, **k: token)
    monkeypatch.setattr(pr, "audit_prediction_support", lambda *a, **k: token)
    monkeypatch.setattr(
        pr,
        "predict_model",
        lambda *a, **k: SimpleNamespace(observed=None),
    )

    fit = SimpleNamespace(
        family="binary",
        specification=SimpleNamespace(prepared=SimpleNamespace(data=pd.DataFrame({"y": [0, 1]}))),
    )
    bundle = rp.create_analysis_bundle(
        fit,
        include_loo=False,
    )
    assert "scores" not in bundle.components
    assert "calibration" not in bundle.components
    assert "coverage" not in bundle.components
    assert "loo" not in bundle.components


def test_reporting_posterior_frame_array_columns_dataframe_fallback_and_partial_threshold(
    monkeypatch,
):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]]).view(ColumnArray)
    arr.columns = ["custom_a", "custom_b"]
    frame = rep._posterior_frame(arr)
    assert frame["variable"].tolist() == ["custom_a", "custom_b"]

    pe = importlib.import_module("gp3bayespy.postfit_exploration")
    fallback = pd.DataFrame(
        {
            "variable": ["b"],
            "lower": [0.0],
            "median": [1.0],
            "upper": [2.0],
        }
    )
    monkeypatch.setattr(pe, "posterior_interval_table", lambda *a, **k: fallback)
    invalid_frame = pd.DataFrame({"variable": ["b"], "mean": [1.0]})
    assert rep._posterior_frame(invalid_frame).equals(fallback)

    partial = pd.DataFrame(
        {
            "threshold": [0.5],
            "accuracy": [0.8],
        }
    )
    fig = rep.plot_binary_threshold_metrics(partial)
    assert fig.axes
