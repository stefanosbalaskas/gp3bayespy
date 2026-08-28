from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
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
    newdata: pd.DataFrame | None = None,
) -> gp.Prediction:
    values = np.asarray(draws, dtype=float)
    probs = (0.025, 0.5, 0.975)
    quantiles = np.quantile(values, probs, axis=0, method="linear")
    summary = pd.DataFrame(
        {
            "observation": np.arange(1, values.shape[1] + 1),
            "predicted_mean": np.mean(values, axis=0),
            "predicted_sd": np.std(values, axis=0, ddof=1),
            "lower": quantiles[0],
            "predicted_median": quantiles[1],
            "upper": quantiles[2],
        }
    )
    observed_series = None if observed is None else pd.Series(observed, dtype=float)
    if observed_series is not None:
        summary["observed"] = observed_series.to_numpy()
    frame = (
        pd.DataFrame({"condition": [f"c{i}" for i in range(values.shape[1])]})
        if newdata is None
        else newdata.copy()
    )
    return gp.Prediction(
        family=family,
        type=type,  # type: ignore[arg-type]
        scale="response",
        draws=values,
        summary=summary,
        newdata=frame,
        observed=observed_series,
        support=_support(values.shape[1]),
        include_group_effects=False,
        allow_new_levels=False,
        probs=probs,
        seed=1,
    )


def test_prediction_draws_long_preserves_r_observation_major_layout():
    prediction = _prediction(np.array([[1, 10], [2, 20], [3, 30]]))
    result = gp.prediction_draws_long(prediction)
    assert result["draw"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result["observation"].tolist() == [1, 1, 1, 2, 2, 2]
    assert result["value"].tolist() == [1, 2, 3, 10, 20, 30]


def test_prediction_draws_long_subsampling_is_seeded_and_renumbers_draws():
    prediction = _prediction(np.arange(30, dtype=float).reshape(10, 3))
    first = gp.prediction_draws_long(prediction, max_draws=4, seed=7)
    second = gp.prediction_draws_long(prediction, max_draws=4, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["draw"].unique().tolist()) == [1, 2, 3, 4]
    assert len(first) == 12


def test_posterior_predictive_statistic_mean_and_table_are_descriptive():
    prediction = _prediction(
        np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1]]),
        type="predictive",
        observed=[0, 1, 1],
    )
    result = gp.posterior_predictive_statistic(prediction, statistic="mean")
    table = gp.ppc_statistic_table(result)
    assert result.observed == pytest.approx(2 / 3)
    assert result.replicated.tolist() == pytest.approx([2 / 3, 1, 1 / 3, 2 / 3])
    assert result.automatic_adequacy_verdict is False
    assert bool(table.loc[0, "automatic_adequacy_verdict"]) is False
    assert table.loc[0, "two_sided_tail_probability"] == pytest.approx(
        result.two_sided_tail_probability
    )


def test_posterior_predictive_statistic_tail_rate_requires_threshold():
    prediction = _prediction(
        np.array([[1, 2], [2, 3], [3, 4]]),
        family="duration",
        type="predictive",
        observed=[2, 4],
    )
    with pytest.raises(GP3BayesError, match="threshold"):
        gp.posterior_predictive_statistic(prediction, statistic="tail_rate")
    result = gp.posterior_predictive_statistic(prediction, statistic="tail_rate", threshold=2.5)
    assert 0 <= result.two_sided_tail_probability <= 1


def test_posterior_predictive_statistic_rejects_nonpredictive_or_no_observed():
    expected = _prediction(np.array([[0.2, 0.8], [0.3, 0.7]]), observed=[0, 1])
    with pytest.raises(GP3BayesError, match="x.type"):
        gp.posterior_predictive_statistic(expected)
    predictive = _prediction(np.array([[0, 1], [1, 1]]), type="predictive", observed=None)
    with pytest.raises(GP3BayesError, match="Observed outcomes"):
        gp.posterior_predictive_statistic(predictive)


def test_binary_confusion_table_matches_frozen_four_cell_order():
    result = gp.binary_confusion_table([0.1, 0.8, 0.7, 0.2], [0, 1, 1, 0])
    assert result[["observed", "predicted"]].values.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert result["count"].tolist() == [2, 0, 0, 2]
    assert result["count"].sum() == 4


def test_binary_roc_and_precision_recall_curves_are_deterministic():
    probabilities = [0.05, 0.2, 0.8, 0.95]
    observed = [0, 0, 1, 1]
    roc = gp.binary_roc_curve(probabilities, observed)
    pr = gp.binary_precision_recall_curve(probabilities, observed)
    assert ((roc["false_positive_rate"] >= 0) & (roc["false_positive_rate"] <= 1)).all()
    assert ((roc["true_positive_rate"] >= 0) & (roc["true_positive_rate"] <= 1)).all()
    assert ((pr["recall"] >= 0) & (pr["recall"] <= 1)).all()
    assert ((pr["precision"] >= 0) & (pr["precision"] <= 1)).all()
    pd.testing.assert_frame_equal(roc, gp.binary_roc_curve(probabilities, observed))


def test_binary_curves_retain_explicit_thresholds_and_allow_infinities():
    thresholds = [0.5, float("inf"), float("-inf"), 0.5]
    roc = gp.binary_roc_curve([0.2, 0.8], [0, 1], thresholds=thresholds)
    assert set(roc["threshold"].tolist()) == {0.5, float("inf"), float("-inf")}
    with pytest.raises(GP3BayesError, match="thresholds"):
        gp.binary_roc_curve([0.2, 0.8], [0, 1], thresholds=[float("nan")])


def test_binary_calibration_error_uses_conservative_no_verdict_contract():
    result = gp.binary_calibration_error([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], bins=2)
    assert result.loc[0, "expected_calibration_error"] >= 0
    assert result.loc[0, "maximum_calibration_error"] >= 0
    assert result.loc[0, "bins_requested"] == 2
    assert bool(result.loc[0, "automatic_adequacy_verdict"]) is False


def test_binary_calibration_error_collapses_constant_predictions_to_one_bin():
    result = gp.binary_calibration_error([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1], bins=5)
    assert result.loc[0, "bins_used"] == 1
    assert result.loc[0, "expected_calibration_error"] == pytest.approx(0.0)


def test_binary_group_calibration_uses_prediction_newdata_groups():
    prediction = _prediction(
        np.array([[0.1, 0.3, 0.7, 0.9], [0.1, 0.3, 0.7, 0.9]]),
        observed=[0, 0, 1, 1],
        newdata=pd.DataFrame({"site": ["b", "a", "b", "a"]}),
    )
    result = gp.binary_group_calibration(prediction, "site")
    assert result["group"].tolist() == ["a", "b"]
    assert result["n"].tolist() == [2, 2]
    assert result["calibration_gap"].to_numpy() == pytest.approx(
        result["observed_rate"] - result["predicted_probability"]
    )


def test_duration_qq_table_matches_observed_and_predictive_quantile_shapes():
    prediction = _prediction(
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        family="duration",
        type="predictive",
        observed=[1.5, 2.5, 3.5, 4.5],
    )
    result = gp.duration_qq_table(prediction, probs=[0.25, 0.5, 0.75])
    assert result["probability"].tolist() == [0.25, 0.5, 0.75]
    assert len(result) == 3
    assert np.isfinite(result.drop(columns="probability").to_numpy()).all()


def test_duration_tail_check_is_descriptive_and_requires_positive_threshold():
    prediction = _prediction(
        np.array([[1, 2, 4], [2, 3, 5], [1, 4, 6], [2, 5, 7]]),
        family="duration",
        type="predictive",
        observed=[1, 3, 6],
    )
    result = gp.duration_tail_check(prediction, 3)
    assert 0 <= result.loc[0, "posterior_probability_rate_ge_observed"] <= 1
    assert bool(result.loc[0, "automatic_adequacy_verdict"]) is False
    with pytest.raises(GP3BayesError, match="positive duration"):
        gp.duration_tail_check(prediction, 0)


def test_group_prediction_summary_handles_one_and_multiple_group_columns():
    prediction = _prediction(
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        family="duration",
        observed=[1, 2, 3, 4],
        newdata=pd.DataFrame({"condition": ["b", "a", "b", "a"], "site": ["x", "x", "y", "y"]}),
    )
    one = gp.group_prediction_summary(prediction, "condition")
    two = gp.group_prediction_summary(prediction, ["condition", "site"])
    assert one["condition"].tolist() == ["a", "b"]
    assert one["n"].tolist() == [2, 2]
    assert len(two) == 4
    assert np.isfinite(one["observed"]).all()


def test_group_prediction_summary_allows_missing_observed_values_as_na_summary():
    prediction = _prediction(
        np.array([[1, 2], [2, 3], [3, 4]]),
        family="duration",
        observed=None,
        newdata=pd.DataFrame({"condition": ["a", "b"]}),
    )
    result = gp.group_prediction_summary(prediction, "condition")
    assert result["observed"].isna().all()


def test_prediction_pairwise_contrasts_returns_all_unique_pairs_without_decisions():
    prediction = _prediction(np.array([[1, 2, 4], [2, 4, 8], [3, 6, 12]]), family="duration")
    result = gp.prediction_pairwise_contrasts(prediction)
    assert len(result) == 3
    assert list(zip(result["row1"], result["row2"], strict=True)) == [(1, 2), (1, 3), (2, 3)]
    assert not result["automatic_decision"].any()


def test_prediction_pairwise_contrasts_honours_rows_and_maximum():
    prediction = _prediction(np.arange(20, dtype=float).reshape(5, 4) + 1, family="duration")
    result = gp.prediction_pairwise_contrasts(prediction, rows=[4, 2, 4], max_rows=2)
    assert len(result) == 1
    assert result.loc[0, "row1"] == 4
    assert result.loc[0, "row2"] == 2
    with pytest.raises(GP3BayesError, match="explicit maximum"):
        gp.prediction_pairwise_contrasts(prediction, max_rows=3)


def test_prediction_interval_width_is_upper_minus_lower():
    prediction = _prediction(np.array([[1, 3], [2, 4], [3, 5], [4, 6]]), family="duration")
    result = gp.prediction_interval_width(prediction)
    assert result["observation"].tolist() == [1, 2]
    assert result["interval_width"].to_numpy() == pytest.approx(
        result["upper"].to_numpy() - result["lower"].to_numpy()
    )


def test_prediction_ranking_never_selects_automatically():
    draws = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1]], dtype=float)
    prediction = _prediction(draws, family="duration")
    result = gp.prediction_rank_probabilities(prediction)
    assert len(result) == 3
    assert not result["automatic_selection"].any()
    assert result["probability_rank_1"].sum() == pytest.approx(1.0)


def test_prediction_ranking_direction_changes_rank_orientation():
    prediction = _prediction(np.array([[1, 3], [2, 4], [1, 5]]), family="duration")
    higher = gp.prediction_rank_probabilities(prediction, direction="higher")
    lower = gp.prediction_rank_probabilities(prediction, direction="lower")
    assert higher.loc[1, "probability_rank_1"] == pytest.approx(1.0)
    assert lower.loc[0, "probability_rank_1"] == pytest.approx(1.0)


def test_advanced_binary_diagnostics_validate_binary_probability_contract():
    with pytest.raises(GP3BayesError, match="probabilities"):
        gp.binary_confusion_table([1.2, 0.2], [1, 0])
    prediction = _prediction(np.array([[0, 1], [1, 1]]), type="predictive", observed=[0, 1])
    with pytest.raises(GP3BayesError, match="expected"):
        gp.binary_roc_curve(prediction)


def test_gpb_py10_public_signatures_expose_no_backend_escape_hatches():
    functions = [
        gp.prediction_draws_long,
        gp.posterior_predictive_statistic,
        gp.ppc_statistic_table,
        gp.binary_confusion_table,
        gp.binary_roc_curve,
        gp.binary_precision_recall_curve,
        gp.binary_calibration_error,
        gp.binary_group_calibration,
        gp.duration_qq_table,
        gp.duration_tail_check,
        gp.group_prediction_summary,
        gp.prediction_pairwise_contrasts,
        gp.prediction_interval_width,
        gp.prediction_rank_probabilities,
    ]
    forbidden = {"backend", "formula", "family", "likelihood", "prior", "kwargs"}
    for function in functions:
        assert not forbidden.intersection(inspect.signature(function).parameters)


def test_gpb_py10_does_not_promote_ledger_before_runtime_closure():
    counts = gp.parity_counts()
    assert counts["implemented"] >= 48
    assert counts["implemented_initial"] in {0, 1}
    assert sum(counts.values()) == 458
