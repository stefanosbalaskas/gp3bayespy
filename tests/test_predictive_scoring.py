import math

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError
from gp3bayespy.predictive import Prediction, PredictionSupport, _prediction_summary


def _prediction(
    *,
    family: str,
    type: str,
    draws: np.ndarray,
    observed: list[float] | None,
) -> Prediction:
    observed_series = None if observed is None else pd.Series(observed, dtype=float)
    support = PredictionSupport(
        table=pd.DataFrame(),
        rows=draws.shape[1],
        has_extrapolation=False,
        has_novel_levels=False,
        has_missing_required=False,
    )
    return Prediction(
        family=family,
        type=type,  # type: ignore[arg-type]
        scale="response",
        draws=np.asarray(draws, dtype=float),
        summary=_prediction_summary(
            np.asarray(draws, dtype=float),
            (0.025, 0.5, 0.975),
            observed_series,
        ),
        newdata=pd.DataFrame({"row": range(draws.shape[1])}),
        observed=observed_series,
        support=support,
        include_group_effects=False,
        allow_new_levels=False,
        probs=(0.025, 0.5, 0.975),
        seed=1,
    )


def test_binary_prediction_scores_match_perfect_reference_case():
    out = gp.binary_prediction_scores([0.05, 0.95, 0.9, 0.1], [0, 1, 1, 0])
    assert out.loc[0, "accuracy"] == pytest.approx(1.0)
    assert out.loc[0, "sensitivity"] == pytest.approx(1.0)
    assert out.loc[0, "specificity"] == pytest.approx(1.0)
    assert out.loc[0, "auc"] > 0.99
    assert bool(out.loc[0, "automatic_decision"]) is False


def test_binary_prediction_scores_match_manual_brier_and_log_loss():
    p = np.array([0.2, 0.8, 0.6, 0.3])
    y = np.array([0, 1, 0, 1])
    out = gp.binary_prediction_scores(p, y, threshold=0.6)
    assert out.loc[0, "brier"] == pytest.approx(np.mean((p - y) ** 2))
    manual_log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert out.loc[0, "log_loss"] == pytest.approx(manual_log_loss)
    assert out.loc[0, "threshold"] == pytest.approx(0.6)


def test_binary_prediction_scores_use_average_rank_auc_for_ties():
    out = gp.binary_prediction_scores([0.2, 0.5, 0.5, 0.8], [0, 0, 1, 1])
    assert out.loc[0, "auc"] == pytest.approx(0.875)


def test_binary_prediction_scores_allow_single_class_with_nan_auc():
    out = gp.binary_prediction_scores([0.1, 0.2], [0, 0])
    assert math.isnan(float(out.loc[0, "auc"]))
    assert math.isnan(float(out.loc[0, "sensitivity"]))
    assert out.loc[0, "specificity"] == pytest.approx(1.0)
    assert out.loc[0, "balanced_accuracy"] == pytest.approx(1.0)


def test_binary_prediction_scores_reject_invalid_inputs():
    with pytest.raises(GP3BayesError, match="outcomes in"):
        gp.binary_prediction_scores([0.1, 1.2], [0, 1])
    with pytest.raises(GP3BayesError, match="threshold"):
        gp.binary_prediction_scores([0.1, 0.9], [0, 1], threshold=1.1)
    with pytest.raises(GP3BayesError, match="epsilon"):
        gp.binary_prediction_scores([0.1, 0.9], [0, 1], epsilon=0.5)


def test_binary_threshold_metrics_preserve_requested_thresholds():
    thresholds = [0.25, 0.5, 0.75]
    out = gp.binary_threshold_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], thresholds)
    assert out["threshold"].tolist() == thresholds
    assert not out["automatic_decision"].any()


def test_binary_threshold_metrics_reject_out_of_range_thresholds():
    with pytest.raises(GP3BayesError, match="thresholds"):
        gp.binary_threshold_metrics([0.1, 0.9], [0, 1], [-0.1, 0.5])


def test_prediction_object_can_feed_binary_scores():
    draws = np.array([[0.1, 0.8], [0.2, 0.9], [0.3, 0.7]])
    pred = _prediction(family="binary", type="expected", draws=draws, observed=[0, 1])
    out = gp.binary_prediction_scores(pred)
    assert out.loc[0, "n"] == 2
    assert out.loc[0, "brier"] == pytest.approx(
        np.mean((draws.mean(axis=0) - np.array([0, 1])) ** 2)
    )


def test_prediction_object_without_observed_outcomes_rejects_scoring():
    draws = np.array([[0.1, 0.8], [0.2, 0.9]])
    pred = _prediction(family="binary", type="expected", draws=draws, observed=None)
    with pytest.raises(GP3BayesError, match="no observed outcome"):
        gp.binary_prediction_scores(pred)


def test_binary_calibration_table_collapses_constant_predictions_to_one_bin():
    draws = np.full((6, 4), 0.5)
    pred = _prediction(
        family="binary",
        type="expected",
        draws=draws,
        observed=[0, 1, 0, 1],
    )
    out = gp.binary_calibration_table(pred, bins=4)
    assert len(out) == 1
    assert out.loc[0, "bin"] == 1
    assert out.loc[0, "n"] == 4
    assert out.loc[0, "observed_rate"] == pytest.approx(0.5)
    assert out.loc[0, "posterior_median"] == pytest.approx(0.5)


def test_binary_calibration_table_returns_equal_frequency_style_bins():
    draws = np.array(
        [
            [0.05, 0.25, 0.65, 0.95],
            [0.10, 0.30, 0.70, 0.90],
            [0.15, 0.35, 0.75, 0.85],
        ]
    )
    pred = _prediction(
        family="binary",
        type="expected",
        draws=draws,
        observed=[0, 0, 1, 1],
    )
    out = gp.binary_calibration_table(pred, bins=2)
    assert out["n"].sum() == 4
    assert list(out.columns) == [
        "bin",
        "n",
        "mean_predicted_probability",
        "observed_rate",
        "posterior_lower",
        "posterior_median",
        "posterior_upper",
    ]


def test_binary_calibration_table_requires_binary_expected_prediction():
    draws = np.ones((3, 2))
    pred = _prediction(family="duration", type="predictive", draws=draws, observed=[1, 1])
    with pytest.raises(GP3BayesError, match="binary expected-response"):
        gp.binary_calibration_table(pred)
    with pytest.raises(GP3BayesError, match="bins"):
        good = _prediction(family="binary", type="expected", draws=draws / 2, observed=[0, 1])
        gp.binary_calibration_table(good, bins=1)


def test_duration_prediction_scores_match_manual_values():
    pred = np.array([100.0, 120.0, 140.0])
    obs = np.array([105.0, 118.0, 150.0])
    out = gp.duration_prediction_scores(pred, obs)
    error = pred - obs
    log_error = np.log(pred) - np.log(obs)
    assert out.loc[0, "mae"] == pytest.approx(np.mean(np.abs(error)))
    assert out.loc[0, "rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert out.loc[0, "log_rmse"] == pytest.approx(np.sqrt(np.mean(log_error**2)))
    assert bool(out.loc[0, "automatic_decision"]) is False


def test_duration_prediction_scores_reject_nonpositive_values():
    with pytest.raises(GP3BayesError, match="finite positive"):
        gp.duration_prediction_scores([100, 0], [90, 80])
    with pytest.raises(GP3BayesError, match="finite positive"):
        gp.duration_prediction_scores([100, 90], [90, -1])


def test_duration_quantile_calibration_matches_empirical_definition():
    draws = np.array(
        [
            [80, 160],
            [100, 180],
            [120, 200],
            [140, 220],
        ],
        dtype=float,
    )
    pred = _prediction(
        family="duration",
        type="predictive",
        draws=draws,
        observed=[110, 190],
    )
    out = gp.duration_quantile_calibration(pred, quantiles=[0.5])
    median = np.quantile(draws, 0.5, axis=0, method="linear")
    empirical = np.mean(np.array([110, 190]) <= median)
    assert out.loc[0, "empirical"] == pytest.approx(empirical)
    assert out.loc[0, "calibration_gap"] == pytest.approx(empirical - 0.5)


def test_duration_quantile_calibration_requires_predictive_duration_object():
    pred = _prediction(
        family="duration",
        type="expected",
        draws=np.ones((3, 2)),
        observed=[1, 1],
    )
    with pytest.raises(GP3BayesError, match="duration posterior predictive"):
        gp.duration_quantile_calibration(pred)
    pred2 = _prediction(
        family="duration",
        type="predictive",
        draws=np.ones((3, 2)),
        observed=[1, 1],
    )
    with pytest.raises(GP3BayesError, match="quantiles"):
        gp.duration_quantile_calibration(pred2, quantiles=[0, 0.5])


def test_duration_pit_table_matches_draw_fraction():
    draws = np.array([[80, 160], [100, 180], [120, 200], [140, 220]], dtype=float)
    pred = _prediction(
        family="duration",
        type="predictive",
        draws=draws,
        observed=[110, 190],
    )
    out = gp.duration_pit_table(pred)
    assert out["observation"].tolist() == [1, 2]
    assert out["pit"].tolist() == pytest.approx([0.5, 0.5])


def test_predictive_coverage_table_matches_interval_definition():
    draws = np.array(
        [
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
            [4, 14],
        ],
        dtype=float,
    )
    pred = _prediction(
        family="binary",
        type="predictive",
        draws=draws,
        observed=[2, 20],
    )
    out = gp.predictive_coverage_table(pred, levels=[0.8])
    lo = np.quantile(draws, 0.1, axis=0, method="linear")
    hi = np.quantile(draws, 0.9, axis=0, method="linear")
    expected = np.mean((np.array([2, 20]) >= lo) & (np.array([2, 20]) <= hi))
    assert out.loc[0, "empirical_coverage"] == pytest.approx(expected)
    assert out.loc[0, "mean_interval_width"] == pytest.approx(np.mean(hi - lo))


def test_predictive_coverage_table_requires_observed_predictive_object():
    pred = _prediction(
        family="binary",
        type="expected",
        draws=np.ones((3, 2)),
        observed=[0, 1],
    )
    with pytest.raises(GP3BayesError, match="posterior predictive object"):
        gp.predictive_coverage_table(pred)


def test_posterior_predictive_summary_table_reuses_prediction_summary_contract():
    draws = np.array([[0, 1], [1, 2], [2, 3]], dtype=float)
    pred = _prediction(
        family="binary",
        type="predictive",
        draws=draws,
        observed=[1, 2],
    )
    out = gp.posterior_predictive_summary_table(pred, probs=[0.25, 0.5, 0.75])
    assert out["predicted_mean"].tolist() == pytest.approx([1, 2])
    assert out["predicted_median"].tolist() == pytest.approx([1, 2])
    assert out["observed"].tolist() == pytest.approx([1, 2])
    assert out["lower"].to_numpy() == pytest.approx(
        np.quantile(draws, 0.25, axis=0, method="linear")
    )


def test_posterior_predictive_summary_table_rejects_expected_prediction():
    pred = _prediction(
        family="binary",
        type="expected",
        draws=np.ones((3, 2)) / 2,
        observed=[0, 1],
    )
    with pytest.raises(GP3BayesError, match="posterior predictive gp3bayes prediction"):
        gp.posterior_predictive_summary_table(pred)
