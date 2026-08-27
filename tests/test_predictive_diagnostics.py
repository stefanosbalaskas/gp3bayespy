from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.predictive as predictive_module
from gp3bayespy.exceptions import GP3BayesError


def _support(rows: int) -> gp.PredictionSupport:
    return gp.PredictionSupport(
        table=pd.DataFrame(),
        rows=rows,
        has_extrapolation=False,
        has_novel_levels=False,
        has_missing_required=False,
    )


def _prediction(
    draws: np.ndarray,
    *,
    family: str = "binary",
    type: str = "expected",
    observed: list[float] | None = None,
) -> gp.Prediction:
    values = np.asarray(draws, dtype=float)
    probs = (0.025, 0.5, 0.975)
    q = np.quantile(values, probs, axis=0, method="linear")
    summary = pd.DataFrame(
        {
            "observation": np.arange(1, values.shape[1] + 1),
            "predicted_mean": np.mean(values, axis=0),
            "predicted_sd": np.std(values, axis=0, ddof=1),
            "lower": q[0],
            "predicted_median": q[1],
            "upper": q[2],
        }
    )
    observed_series = None if observed is None else pd.Series(observed, dtype=float)
    if observed_series is not None:
        summary["observed"] = observed_series.to_numpy()
    return gp.Prediction(
        family=family,
        type=type,  # type: ignore[arg-type]
        scale="response",
        draws=values,
        summary=summary,
        newdata=pd.DataFrame({"row": np.arange(values.shape[1])}),
        observed=observed_series,
        support=_support(values.shape[1]),
        include_group_effects=False,
        allow_new_levels=False,
        probs=probs,
        seed=1,
    )


def test_prediction_contrast_difference_keeps_r_one_based_rows():
    prediction = _prediction(np.array([[1, 2], [2, 4], [3, 6], [4, 8]]), family="duration")
    result = gp.prediction_contrast(prediction, 1, 2)
    values = prediction.draws[:, 1] - prediction.draws[:, 0]
    assert result.loc[0, "row1"] == 1
    assert result.loc[0, "row2"] == 2
    assert result.loc[0, "measure"] == "difference"
    assert result.loc[0, "mean"] == pytest.approx(float(np.mean(values)))
    assert result.loc[0, "median"] == pytest.approx(float(np.quantile(values, 0.5)))
    assert result.loc[0, "probability_gt_reference"] == pytest.approx(1.0)
    assert bool(result.loc[0, "automatic_decision"]) is False


def test_prediction_contrast_ratio_requires_positive_denominator_draws():
    prediction = _prediction(np.array([[0, 2], [1, 3], [2, 4]]), family="duration")
    with pytest.raises(GP3BayesError, match="positive denominator"):
        gp.prediction_contrast(prediction, 1, 2, measure="ratio")


def test_prediction_contrast_odds_ratio_requires_binary_expected_probability():
    duration = _prediction(np.array([[1, 2], [2, 3], [3, 4]]), family="duration")
    with pytest.raises(GP3BayesError, match="binary expected probabilities"):
        gp.prediction_contrast(duration, 1, 2, measure="odds_ratio")

    binary = _prediction(np.array([[0.2, 0.4], [0.25, 0.5], [0.3, 0.6]]))
    result = gp.prediction_contrast(binary, 1, 2, measure="odds_ratio")
    assert result.loc[0, "mean"] > 1
    assert result.loc[0, "probability_gt_reference"] == pytest.approx(1.0)


def test_prediction_contrast_validates_prediction_rows_and_measure():
    prediction = _prediction(np.array([[0.2, 0.4], [0.3, 0.5]]))
    with pytest.raises(GP3BayesError, match="identify prediction rows"):
        gp.prediction_contrast(prediction, 0, 2)
    with pytest.raises(GP3BayesError, match="measure"):
        gp.prediction_contrast(prediction, 1, 2, measure="bad")  # type: ignore[arg-type]


def test_prediction_exceedance_probability_uses_strict_direction_and_no_decision():
    prediction = _prediction(np.array([[0.2, 0.5], [0.5, 0.7], [0.8, 0.5]]))
    above = gp.prediction_exceedance_probability(prediction, 0.5, direction="above")
    below = gp.prediction_exceedance_probability(prediction, 0.5, direction="below")
    assert above["probability"].tolist() == pytest.approx([1 / 3, 1 / 3])
    assert below["probability"].tolist() == pytest.approx([1 / 3, 0.0])
    assert above["observation"].tolist() == [1, 2]
    assert not above["automatic_decision"].any()


def test_prediction_exceedance_probability_validates_threshold_and_direction():
    prediction = _prediction(np.array([[0.2, 0.4], [0.3, 0.5]]))
    with pytest.raises(GP3BayesError, match="finite number"):
        gp.prediction_exceedance_probability(prediction, float("nan"))
    with pytest.raises(GP3BayesError, match="direction"):
        gp.prediction_exceedance_probability(
            prediction, 0.5, direction="sideways"  # type: ignore[arg-type]
        )


def test_prediction_uncertainty_decomposition_is_descriptive(monkeypatch):
    expected = _prediction(np.array([[1, 2], [2, 4], [3, 6], [4, 8]]), family="duration")
    predictive = _prediction(
        np.array([[0, 1], [2, 5], [4, 7], [6, 11]]),
        family="duration",
        type="predictive",
    )
    calls: list[str] = []

    def fake_predict_model(*args, **kwargs):
        calls.append(kwargs["type"])
        return expected if kwargs["type"] == "expected" else predictive

    monkeypatch.setattr(predictive_module, "predict_model", fake_predict_model)
    result = gp.prediction_uncertainty_decomposition(
        object(), ndraws=4, seed=9  # type: ignore[arg-type]
    )
    expected_var = np.var(expected.draws, axis=0, ddof=1)
    total_var = np.var(predictive.draws, axis=0, ddof=1)
    assert calls == ["expected", "predictive"]
    assert result.table["expected_response_variance"].to_numpy() == pytest.approx(
        expected_var
    )
    assert result.table["total_predictive_variance"].to_numpy() == pytest.approx(total_var)
    assert (result.table["residual_component"] >= 0).all()
    assert result.causal_variance_decomposition is False
    assert "not a causal variance decomposition" in result.interpretation


def test_prediction_uncertainty_validates_controls_before_prediction():
    with pytest.raises(GP3BayesError, match="positive integer"):
        gp.prediction_uncertainty_decomposition(object(), ndraws=0)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="non-negative integer"):
        gp.prediction_uncertainty_decomposition(object(), seed=-1)  # type: ignore[arg-type]


def _fake_fit(family: str, data: pd.DataFrame):
    specification = SimpleNamespace(
        prepared=SimpleNamespace(data=data),
        contract=SimpleNamespace(mappings={"outcome": "y"}),
    )
    return SimpleNamespace(family=family, specification=specification)


def test_grouped_prediction_check_summarises_groups_without_exclusion(monkeypatch):
    data = pd.DataFrame({"group": ["b", "a", "b", "a"], "y": [0, 1, 1, 1]})
    fit = _fake_fit("binary", data)
    prediction = _prediction(
        np.array([[0, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
        type="predictive",
        observed=data["y"].tolist(),
    )
    monkeypatch.setattr(predictive_module, "_validate_fit", lambda value: value)
    monkeypatch.setattr(predictive_module, "predict_model", lambda *a, **k: prediction)
    result = gp.grouped_prediction_check(fit, "group", ndraws=4)
    assert result.table["group"].tolist() == ["a", "b"]
    assert result.table["n"].tolist() == [2, 2]
    assert result.table["observed"].tolist() == pytest.approx([1.0, 0.5])
    assert result.draws.shape == (4, 2)
    assert result.automatic_exclusion is False


def test_grouped_prediction_check_validates_group_column(monkeypatch):
    fit = _fake_fit("binary", pd.DataFrame({"y": [0, 1]}))
    monkeypatch.setattr(predictive_module, "_validate_fit", lambda value: value)
    with pytest.raises(GP3BayesError, match="prepared model data"):
        gp.grouped_prediction_check(fit, "missing", ndraws=2)


def test_predictive_residuals_binary_defaults_to_raw_and_supports_pearson(monkeypatch):
    data = pd.DataFrame({"y": [0.0, 1.0]})
    fit = _fake_fit("binary", data)
    prediction = _prediction(
        np.array([[0.2, 0.7], [0.4, 0.9], [0.3, 0.8]]),
        observed=[0, 1],
    )
    monkeypatch.setattr(predictive_module, "_validate_fit", lambda value: value)
    monkeypatch.setattr(predictive_module, "predict_model", lambda *a, **k: prediction)
    raw = gp.predictive_residuals(fit, ndraws=3)
    pearson = gp.predictive_residuals(fit, type="pearson", ndraws=3)
    expected = prediction.summary["predicted_mean"].to_numpy()
    assert raw["type"].unique().tolist() == ["raw"]
    assert raw["residual"].to_numpy() == pytest.approx(np.array([0.0, 1.0]) - expected)
    denominator = np.sqrt(np.maximum(expected * (1 - expected), np.finfo(float).eps))
    assert pearson["residual"].to_numpy() == pytest.approx(
        (np.array([0.0, 1.0]) - expected) / denominator
    )


def test_predictive_residuals_duration_defaults_to_log_and_supports_relative(monkeypatch):
    data = pd.DataFrame({"y": [10.0, 20.0]})
    fit = _fake_fit("duration", data)
    prediction = _prediction(
        np.array([[8, 18], [10, 20], [12, 22]]),
        family="duration",
        observed=[10, 20],
    )
    monkeypatch.setattr(predictive_module, "_validate_fit", lambda value: value)
    monkeypatch.setattr(predictive_module, "predict_model", lambda *a, **k: prediction)
    log_residual = gp.predictive_residuals(fit, ndraws=3)
    relative = gp.predictive_residuals(fit, type="relative", ndraws=3)
    expected = prediction.summary["predicted_mean"].to_numpy()
    assert log_residual["type"].unique().tolist() == ["log"]
    assert log_residual["residual"].to_numpy() == pytest.approx(
        np.log(np.array([10.0, 20.0])) - np.log(expected)
    )
    assert relative["residual"].to_numpy() == pytest.approx(
        (np.array([10.0, 20.0]) - expected) / expected
    )


def test_predictive_residuals_reject_family_incompatible_types(monkeypatch):
    binary = _fake_fit("binary", pd.DataFrame({"y": [0.0, 1.0]}))
    duration = _fake_fit("duration", pd.DataFrame({"y": [10.0, 20.0]}))
    monkeypatch.setattr(predictive_module, "_validate_fit", lambda value: value)
    with pytest.raises(GP3BayesError, match="Binary residual"):
        gp.predictive_residuals(binary, type="log", ndraws=2)
    with pytest.raises(GP3BayesError, match="Duration residual"):
        gp.predictive_residuals(duration, type="pearson", ndraws=2)


def test_gpb_py09_public_signatures_are_restricted():
    expected = {
        "prediction_contrast": ["x", "row1", "row2", "measure", "probs"],
        "prediction_exceedance_probability": ["x", "threshold", "direction"],
        "prediction_uncertainty_decomposition": [
            "fit",
            "newdata",
            "include_group_effects",
            "allow_new_levels",
            "ndraws",
            "seed",
        ],
        "grouped_prediction_check": ["fit", "group", "ndraws", "probs", "seed"],
        "predictive_residuals": ["fit", "type", "ndraws"],
    }
    forbidden = {"backend", "formula", "family", "likelihood", "prior", "kwargs"}
    for name, parameters in expected.items():
        function = getattr(gp, name)
        assert list(inspect.signature(function).parameters) == parameters
        assert not forbidden.intersection(inspect.signature(function).parameters)


def test_gpb_py09_exports_are_public_but_ledger_stays_preclosure():
    for name in (
        "prediction_contrast",
        "prediction_exceedance_probability",
        "prediction_uncertainty_decomposition",
        "grouped_prediction_check",
        "predictive_residuals",
    ):
        assert name in gp.__all__
        assert callable(getattr(gp, name))
    assert gp.parity_counts() == {
        "mapped_not_implemented": 414,
        "implemented": 43,
        "implemented_initial": 1,
    }
