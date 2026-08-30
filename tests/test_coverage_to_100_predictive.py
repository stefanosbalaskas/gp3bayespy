from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.predictive as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _support(rows: int = 4) -> p.PredictionSupport:
    return p.PredictionSupport(
        table=pd.DataFrame(
            {
                "variable": ["x"],
                "type": ["numeric"],
                "training_min": [0.0],
                "training_max": [1.0],
                "outside_support": [0],
                "novel_levels": [np.nan],
                "missing_values": [0],
                "detail": ["within observed range"],
            }
        ),
        rows=rows,
        has_extrapolation=False,
        has_novel_levels=False,
        has_missing_required=False,
    )


def _prediction(
    family: str = "binary",
    type: str = "expected",
    *,
    seed: int = 1301,
    rows: int = 4,
) -> p.Prediction:
    rng = np.random.default_rng(seed)
    if family == "binary":
        observed = pd.Series(([0, 1] * ((rows + 1) // 2))[:rows], dtype=float)
        base = np.linspace(0.15, 0.85, rows)
        if type == "predictive":
            probs = np.broadcast_to(base[None, :], (80, rows))
            draws = rng.binomial(1, probs).astype(float)
        elif type == "linear":
            draws = rng.normal(np.linspace(-1.0, 1.0, rows), 0.2, size=(80, rows))
        else:
            draws = np.clip(
                rng.normal(base, 0.06, size=(80, rows)),
                0.001,
                0.999,
            )
    else:
        observed = pd.Series(np.linspace(300.0, 900.0, rows), dtype=float)
        base = np.linspace(320.0, 880.0, rows)
        if type == "linear":
            draws = rng.normal(np.log(base), 0.08, size=(80, rows))
        else:
            draws = np.maximum(
                rng.normal(base, 45.0, size=(80, rows)),
                10.0,
            )

    summary = p._prediction_summary(
        draws,
        (0.025, 0.5, 0.975),
        observed,
    )
    newdata = pd.DataFrame(
        {
            "group": (["a", "a", "b", "b"] * ((rows + 3) // 4))[:rows],
            "x": np.linspace(0.0, 1.0, rows),
        }
    )
    return p.Prediction(
        family=family,
        type=type,  # type: ignore[arg-type]
        scale="response",
        draws=draws,
        summary=summary,
        newdata=newdata,
        observed=observed,
        support=_support(rows),
        include_group_effects=False,
        allow_new_levels=False,
        probs=(0.025, 0.5, 0.975),
        seed=seed,
    )


def test_scalar_sequence_and_type_helpers_cover_success_and_errors():
    assert p._flag(True, "x") is True
    with pytest.raises(GP3BayesError):
        p._flag(1, "x")

    assert p._positive_integer_or_none(None, "n") is None
    assert p._positive_integer_or_none(3.0, "n") == 3
    assert p._positive_integer(2, "n") == 2
    assert p._nonnegative_integer(0, "seed") == 0

    for value in (True, 0, -1, 1.2, np.inf, "2"):
        with pytest.raises(GP3BayesError):
            p._positive_integer_or_none(value, "n")
    with pytest.raises(GP3BayesError):
        p._positive_integer(None, "n")
    for value in (True, -1, 1.2, np.inf):
        with pytest.raises(GP3BayesError):
            p._nonnegative_integer(value, "seed")

    assert p._probabilities((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    for values in ((0.1, 0.9), (0.5, 0.4, 0.9), (-0.1, 0.5, 0.9), ("x", 0.5, 0.9)):
        with pytest.raises(GP3BayesError):
            p._probabilities(values)  # type: ignore[arg-type]

    assert p._finite_scalar(2, "x") == 2.0
    for value in (True, np.inf, "x"):
        with pytest.raises(GP3BayesError):
            p._finite_scalar(value, "x")

    assert p._probability_vector((0, 0.5, 1), "p", open_interval=False) == (0.0, 0.5, 1.0)
    assert p._probability_vector((0.1, 0.9), "p", open_interval=True) == (0.1, 0.9)
    for values, open_interval in (
        ((), False),
        ((np.nan,), False),
        ((-0.1,), False),
        ((0.0,), True),
        (("x",), False),
    ):
        with pytest.raises(GP3BayesError):
            p._probability_vector(values, "p", open_interval=open_interval)  # type: ignore[arg-type]

    assert p._advanced_probabilities((0.9, 0.1, 0.5, 0.5)) == (0.1, 0.5, 0.9)
    for values in ((), (0,), (1,), (np.nan,), ("x",)):
        with pytest.raises(GP3BayesError):
            p._advanced_probabilities(values)  # type: ignore[arg-type]

    assert p._as_values(pd.Series([1, 2])) == [1, 2]
    assert p._as_values(pd.Index(["a", "b"])) == ["a", "b"]
    assert p._as_values(np.array([1, 2])) == [1, 2]
    assert p._as_values(np.array(3)) == [3]
    assert p._as_values((1, 2)) == [1, 2]
    assert p._as_values("x") == ["x"]


def test_restore_representative_and_default_grid_value_helpers():
    categorical = pd.Series(pd.Categorical(["b", "a"], categories=["a", "b"], ordered=True))
    integer = pd.Series([1, 2], dtype="int64")
    boolean = pd.Series([True, False], dtype=bool)
    text = pd.Series(["x", "y"], dtype=object)
    numeric = pd.Series([1.0, 3.0], dtype=float)

    restored_cat = p._restore_type(["a", "b"], categorical)
    assert isinstance(restored_cat.dtype, pd.CategoricalDtype)
    assert p._restore_type([2, 1], integer).dtype == integer.dtype
    assert p._restore_type([True, False], boolean).dtype == bool
    assert p._restore_type(["x", "y"], text).dtype == object
    assert p._restore_type([2, 4], numeric).dtype == float

    assert p._representative_value(categorical, "median") == "a"
    assert p._representative_value(boolean, "median") in {True, False}
    assert p._representative_value(numeric, "median") == 2.0
    assert p._representative_value(numeric, "mean") == 2.0
    assert p._representative_value(text, "median") == "x"

    assert p._default_grid_values(categorical, "median") == ["a", "b"]
    assert set(p._default_grid_values(boolean, "median")) == {True, False}
    assert p._default_grid_values(numeric, "median") == [2.0]
    assert p._default_grid_values(text, "median") == ["x", "y"]

    with pytest.raises(GP3BayesError):
        p._representative_value(
            pd.Series(pd.Categorical([], categories=[])),
            "median",
        )
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([], dtype=bool), "median")
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([np.nan], dtype=float), "median")
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([], dtype=object), "median")


def test_prediction_support_prediction_repr_tables_and_input_normalization():
    support = _support()
    assert "Automatic rejection: FALSE" in repr(support)
    assert len(p.prediction_support_table(support)) == 1
    with pytest.raises(GP3BayesError):
        p.prediction_support_table(object())  # type: ignore[arg-type]

    binary = _prediction("binary", "expected")
    assert "<gp3bayes_prediction>" in repr(binary)
    assert len(p.prediction_table(binary)) == 4
    with pytest.raises(GP3BayesError):
        p.prediction_table(object())  # type: ignore[arg-type]

    predicted, observed, draws = p._prediction_inputs(binary)
    assert predicted.shape == observed.shape == (4,)
    assert draws is not None and draws.shape == (80, 4)

    numeric_pred, numeric_obs, no_draws = p._prediction_inputs(
        [0.2, 0.8],
        [0, 1],
    )
    assert no_draws is None
    assert np.allclose(numeric_pred, [0.2, 0.8])
    assert np.allclose(numeric_obs, [0, 1])

    missing_obs = p.Prediction(
        family=binary.family,
        type=binary.type,
        scale=binary.scale,
        draws=binary.draws,
        summary=binary.summary.drop(columns=["observed"]),
        newdata=binary.newdata,
        observed=None,
        support=binary.support,
        include_group_effects=False,
        allow_new_levels=False,
        probs=binary.probs,
        seed=1,
    )
    with pytest.raises(GP3BayesError, match="no observed"):
        p._prediction_inputs(missing_obs)
    with pytest.raises(GP3BayesError):
        p._prediction_inputs([[0.1, 0.2]], [0, 1])
    with pytest.raises(GP3BayesError):
        p._prediction_inputs([0.1, 0.2], None)
    with pytest.raises(GP3BayesError):
        p._prediction_inputs([0.1, 0.2], [0])
    with pytest.raises(GP3BayesError):
        p._prediction_inputs(["x"], [0])  # type: ignore[list-item]


def test_binary_and_duration_scores_calibration_and_coverage_matrix():
    binary = _prediction("binary", "expected")
    binary_predictive = _prediction("binary", "predictive")
    duration = _prediction("duration", "predictive")

    scores = p.binary_prediction_scores(binary)
    assert scores.loc[0, "n"] == 4
    assert 0 <= scores.loc[0, "brier"] <= 1
    thresholds = p.binary_threshold_metrics(binary, thresholds=(0.25, 0.5, 0.75))
    assert len(thresholds) == 3
    calibration = p.binary_calibration_table(binary, bins=3)
    assert not calibration.empty

    constant = _prediction("binary", "expected")
    constant_draws = np.full((80, 4), 0.5)
    constant = p.Prediction(
        family="binary",
        type="expected",
        scale="response",
        draws=constant_draws,
        summary=p._prediction_summary(
            constant_draws,
            (0.025, 0.5, 0.975),
            constant.observed,
        ),
        newdata=constant.newdata,
        observed=constant.observed,
        support=constant.support,
        include_group_effects=False,
        allow_new_levels=False,
        probs=constant.probs,
        seed=1,
    )
    assert len(p.binary_calibration_table(constant, bins=3)) == 1

    duration_scores = p.duration_prediction_scores(
        [320, 510, 690, 910],
        [300, 500, 700, 900],
    )
    assert duration_scores.loc[0, "rmse"] > 0
    assert len(p.duration_quantile_calibration(duration, quantiles=(0.25, 0.5, 0.75))) == 3
    assert len(p.duration_pit_table(duration)) == 4
    assert len(p.predictive_coverage_table(duration, levels=(0.5, 0.8))) == 2
    assert len(p.posterior_predictive_summary_table(duration)) == 4

    # Degenerate-outcome branches for sensitivity/specificity/AUC.
    only_zero = p.binary_prediction_scores([0.1, 0.2], [0, 0])
    only_one = p.binary_prediction_scores([0.8, 0.9], [1, 1])
    assert math.isnan(only_zero.loc[0, "sensitivity"])
    assert math.isnan(only_one.loc[0, "specificity"])

    for predicted, observed in (
        ([1.2], [1]),
        ([0.2], [2]),
        ([np.nan], [1]),
    ):
        with pytest.raises(GP3BayesError):
            p.binary_prediction_scores(predicted, observed)
    for threshold in (-0.1, 1.1):
        with pytest.raises(GP3BayesError):
            p.binary_prediction_scores([0.2, 0.8], [0, 1], threshold=threshold)
    for epsilon in (0, 0.5):
        with pytest.raises(GP3BayesError):
            p.binary_prediction_scores([0.2, 0.8], [0, 1], epsilon=epsilon)
    with pytest.raises(GP3BayesError):
        p.binary_calibration_table(binary_predictive)
    with pytest.raises(GP3BayesError):
        p.binary_calibration_table(binary, bins=1)
    with pytest.raises(GP3BayesError):
        p.duration_prediction_scores([0, 2], [1, 2])
    with pytest.raises(GP3BayesError):
        p.duration_quantile_calibration(binary_predictive)
    with pytest.raises(GP3BayesError):
        p.duration_pit_table(binary_predictive)
    with pytest.raises(GP3BayesError):
        p.predictive_coverage_table(binary)
    with pytest.raises(GP3BayesError):
        p.posterior_predictive_summary_table(binary)


def test_advanced_predictive_diagnostics_and_error_paths():
    binary = _prediction("binary", "expected")
    binary_pp = _prediction("binary", "predictive")
    duration = _prediction("duration", "predictive")

    assert p._advanced_prediction(binary) is binary
    assert (
        p._advanced_prediction(binary, types={"expected"}, family="binary", observed=True) is binary
    )
    with pytest.raises(GP3BayesError):
        p._advanced_prediction(object())
    with pytest.raises(GP3BayesError):
        p._advanced_prediction(binary, types={"predictive"})
    with pytest.raises(GP3BayesError):
        p._advanced_prediction(binary, family="duration")

    no_obs = p.Prediction(
        family=binary.family,
        type=binary.type,
        scale=binary.scale,
        draws=binary.draws,
        summary=binary.summary.drop(columns=["observed"]),
        newdata=binary.newdata,
        observed=None,
        support=binary.support,
        include_group_effects=False,
        allow_new_levels=False,
        probs=binary.probs,
        seed=1,
    )
    with pytest.raises(GP3BayesError):
        p._advanced_prediction(no_obs, observed=True)

    probs, outcomes = p._binary_probability_inputs(binary)
    assert probs.shape == outcomes.shape == (4,)
    probs2, outcomes2 = p._binary_probability_inputs([0.1, 0.9], [0, 1])
    assert np.allclose(probs2, [0.1, 0.9])
    assert np.allclose(outcomes2, [0, 1])
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs(binary_pp)
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs([0.1, 0.9])
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs([[0.1]], [0])
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs([1.2], [1])

    long_all = p.prediction_draws_long(binary)
    long_small = p.prediction_draws_long(binary, max_draws=10, seed=4)
    assert len(long_all) == 320
    assert len(long_small) == 40

    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert p._statistic_values(values, "mean", None, axis=None) == 2.5
    assert np.isfinite(p._statistic_values(values, "sd", None, axis=None))
    assert p._statistic_values(values, "median", None, axis=None) == 2.5
    assert p._statistic_values(values, "q90", None, axis=None) > 3
    assert p._statistic_values(values, "q95", None, axis=None) > 3
    assert p._statistic_values(values, "max", None, axis=None) == 4
    assert p._statistic_values(values, "tail_rate", 2.0, axis=None) == 0.5

    for statistic in ("mean", "sd", "median", "q90", "q95", "max"):
        result = p.posterior_predictive_statistic(duration, statistic=statistic)  # type: ignore[arg-type]
        assert result.statistic == statistic
        assert len(p.ppc_statistic_table(result)) == 1
    tail = p.posterior_predictive_statistic(
        duration,
        statistic="tail_rate",
        threshold=600,
    )
    assert tail.threshold == 600

    with pytest.raises(GP3BayesError):
        p.posterior_predictive_statistic(binary)
    with pytest.raises(GP3BayesError):
        p.posterior_predictive_statistic(duration, statistic="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.posterior_predictive_statistic(duration, statistic="tail_rate")
    with pytest.raises(GP3BayesError):
        p.ppc_statistic_table(object())  # type: ignore[arg-type]


def test_curves_calibration_group_duration_and_contrast_tables():
    binary = _prediction("binary", "expected")
    duration = _prediction("duration", "predictive")

    confusion = p.binary_confusion_table(binary, threshold=0.5)
    assert confusion["count"].sum() == 4

    auto_thresholds = p._binary_curve_thresholds(
        np.array([0.2, 0.5, 0.8]),
        None,
    )
    declared_thresholds = p._binary_curve_thresholds(
        np.array([0.2, 0.5, 0.8]),
        (0.25, 0.75),
    )
    assert np.isposinf(auto_thresholds[0])
    assert np.allclose(declared_thresholds, [0.75, 0.25])

    roc = p.binary_roc_curve(binary)
    pr = p.binary_precision_recall_curve(binary)
    assert {"false_positive_rate", "true_positive_rate"}.issubset(roc)
    assert {"recall", "precision"}.issubset(pr)

    no_positive = p.binary_roc_curve([0.1, 0.2], [0, 0])
    no_negative = p.binary_roc_curve([0.8, 0.9], [1, 1])
    assert no_positive["true_positive_rate"].isna().all()
    assert no_negative["false_positive_rate"].isna().all()

    pr_no_positive = p.binary_precision_recall_curve([0.1, 0.2], [0, 0])
    assert pr_no_positive["recall"].isna().all()

    cal_error = p.binary_calibration_error(binary, bins=3)
    constant_cal_error = p.binary_calibration_error([0.5] * 4, [0, 1, 0, 1], bins=3)
    assert cal_error.loc[0, "bins_used"] >= 1
    assert constant_cal_error.loc[0, "bins_used"] == 1

    grouped_cal = p.binary_group_calibration(binary, "group")
    assert len(grouped_cal) == 2

    qq = p.duration_qq_table(duration, probs=(0.25, 0.5, 0.75))
    tail = p.duration_tail_check(duration, threshold=600)
    assert len(qq) == 3
    assert len(tail) == 1

    grouped = p.group_prediction_summary(binary, "group")
    grouped_two = p.group_prediction_summary(binary, ("group", "x"))
    assert len(grouped) == 2
    assert len(grouped_two) == 4

    diff = p.prediction_contrast(binary, 1, 2, measure="difference")
    ratio = p.prediction_contrast(binary, 1, 2, measure="ratio")
    odds = p.prediction_contrast(binary, 1, 2, measure="odds_ratio")
    assert diff.loc[0, "measure"] == "difference"
    assert ratio.loc[0, "measure"] == "ratio"
    assert odds.loc[0, "measure"] == "odds_ratio"

    exceed_above = p.prediction_exceedance_probability(binary, 0.5, direction="above")
    exceed_below = p.prediction_exceedance_probability(binary, 0.5, direction="below")
    assert len(exceed_above) == len(exceed_below) == 4

    pairwise = p.prediction_pairwise_contrasts(
        binary,
        rows=(1, 2, 3),
        measure="difference",
        max_rows=3,
    )
    widths = p.prediction_interval_width(binary)
    ranks_high = p.prediction_rank_probabilities(binary, rows=(1, 2, 3), direction="higher")
    ranks_low = p.prediction_rank_probabilities(binary, rows=(1, 2, 3), direction="lower")
    assert len(pairwise) == 3
    assert len(widths) == 4
    assert len(ranks_high) == len(ranks_low) == 3

    assert p._prediction_rows(None, 3) == [1, 2, 3]
    assert p._prediction_rows((1, 1, 3), 3) == [1, 3]

    with pytest.raises(GP3BayesError):
        p.binary_confusion_table(binary, threshold=2)
    with pytest.raises(GP3BayesError):
        p._binary_curve_thresholds(np.array([0.2]), [[0.5]])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._binary_curve_thresholds(np.array([0.2]), [np.nan])
    with pytest.raises(GP3BayesError):
        p.binary_calibration_error(binary, bins=1)
    with pytest.raises(GP3BayesError):
        p.binary_group_calibration(binary, "missing")
    with pytest.raises(GP3BayesError):
        p.duration_tail_check(duration, threshold=0)
    with pytest.raises(GP3BayesError):
        p.group_prediction_summary(binary, ())
    with pytest.raises(GP3BayesError):
        p.prediction_contrast(object(), 1, 2)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_contrast(binary, 0, 2)
    with pytest.raises(GP3BayesError):
        p.prediction_contrast(binary, 1, 2, measure="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_contrast(duration, 1, 2, measure="odds_ratio")
    with pytest.raises(GP3BayesError):
        p.prediction_exceedance_probability(binary, 0.5, direction="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._prediction_rows((0,), 3)
    with pytest.raises(GP3BayesError):
        p.prediction_pairwise_contrasts(binary, rows=(1,), max_rows=3)
    with pytest.raises(GP3BayesError):
        p.prediction_pairwise_contrasts(binary, rows=(1, 2, 3), max_rows=2)
    with pytest.raises(GP3BayesError):
        p.prediction_pairwise_contrasts(binary, rows=(1, 2), measure="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_rank_probabilities(binary, direction="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_rank_probabilities(binary, max_rows=2)


def test_profile_surface_atlas_uncertainty_objects_tables_and_plots():
    binary = _prediction("binary", "expected")
    duration = _prediction("duration", "predictive")

    profile_table = binary.summary.copy()
    profile_table["profile_x"] = np.linspace(0.0, 1.0, 4)
    profile = p._PredictionProfile(
        variable="x",
        table=profile_table,
        prediction=binary,
    )
    gradient = p.prediction_gradient_table(profile)
    assert len(p.prediction_profile_table(profile)) == 4
    assert len(gradient) == 3

    bad_profile = p._PredictionProfile(
        variable="x",
        table=profile_table.assign(profile_x=[0, 0, 1, 2]),
        prediction=binary,
    )
    with pytest.raises(GP3BayesError):
        p.prediction_gradient_table(bad_profile)

    surface_table = binary.summary.copy()
    surface_table["surface_x"] = [0.0, 1.0, 0.0, 1.0]
    surface_table["surface_y"] = [0.0, 0.0, 1.0, 1.0]
    surface_table["interval_width"] = surface_table["upper"] - surface_table["lower"]
    surface = p._PredictionSurface(
        x="x",
        y="y",
        table=surface_table,
        prediction=binary,
    )
    assert len(p.prediction_surface_table(surface)) == 4

    contrast_table = pd.DataFrame(
        {
            "profile_x": [0.0, 0.5, 1.0],
            "contrast_level_1": ["a"] * 3,
            "contrast_level_2": ["b"] * 3,
            "measure": ["difference"] * 3,
            "contrast_mean": [0.1, 0.2, 0.3],
            "contrast_lower": [-0.1, 0.0, 0.1],
            "contrast_median": [0.1, 0.2, 0.3],
            "contrast_upper": [0.3, 0.4, 0.5],
            "probability_gt_reference": [0.7, 0.8, 0.9],
            "automatic_interaction_decision": [False] * 3,
        }
    )
    contrast = p._PredictionContrastProfile(
        variable="x",
        contrast_variable="group",
        contrast_levels=("a", "b"),
        measure="difference",
        table=contrast_table,
        draws=np.ones((80, 3)),
        prediction=binary,
    )
    assert len(p.prediction_contrast_profile_table(contrast)) == 3

    observed_stats = p._atlas_stat(duration.observed.to_numpy(float))
    draw_stats = pd.DataFrame([p._atlas_stat(row) for row in duration.draws])
    draw_stats.insert(0, "draw", np.arange(1, len(draw_stats) + 1))
    atlas = p._PredictiveDistributionAtlas(
        family="duration",
        prediction=duration,
        observed=duration.observed.to_numpy(float),
        observed_statistics=observed_stats,
        draw_statistics=draw_stats,
        include_group_effects=False,
    )
    assert len(p.predictive_distribution_atlas_table(atlas)) == 80
    assert p._atlas_get(atlas, 10, False, 1) is atlas
    envelope = p.predictive_quantile_envelope(
        atlas,
        probabilities=(0.25, 0.5, 0.75),
        probs=(0.1, 0.5, 0.9),
    )
    assert len(envelope) == 3

    score_draws = pd.DataFrame(
        {
            "draw": np.tile(np.arange(1, 11), 2),
            "metric": ["brier"] * 10 + ["log_loss"] * 10,
            "value": np.linspace(0.1, 0.3, 20),
        }
    )
    score_summary = pd.DataFrame(
        {
            "metric": ["brier", "log_loss"],
            "mean": [0.15, 0.25],
            "lower": [0.1, 0.2],
            "median": [0.15, 0.25],
            "upper": [0.2, 0.3],
        }
    )
    score_uncertainty = p._PredictionScoreUncertainty(
        family="binary",
        scope="supplied_data",
        draws=score_draws,
        summary=score_summary,
        prediction=binary,
    )
    assert len(p.prediction_score_uncertainty_table(score_uncertainty)) == 2

    cal_table = pd.DataFrame(
        {
            "bin": [1, 2],
            "n": [2, 2],
            "observed_rate": [0.0, 1.0],
            "predicted_mean": [0.2, 0.8],
            "predicted_lower": [0.1, 0.7],
            "predicted_median": [0.2, 0.8],
            "predicted_upper": [0.3, 0.9],
        }
    )
    cal_uncertainty = p._BinaryCalibrationUncertainty(
        table=cal_table,
        bins_requested=2,
        prediction=binary,
        scope="supplied_data",
    )
    assert len(p.binary_calibration_uncertainty_table(cal_uncertainty)) == 2

    figures = [
        p.plot_prediction_profile(profile),
        p.plot_prediction_gradient(profile),
        p.plot_prediction_gradient(gradient),
        p.plot_prediction_surface(surface),
        p.plot_prediction_surface_uncertainty(surface),
        p.plot_prediction_contrast_profile(contrast),
        p.plot_predictive_atlas_statistics(atlas),
        p.plot_predictive_quantile_envelope(envelope),
        p.plot_prediction_score_uncertainty(score_uncertainty),
        p.plot_binary_calibration_uncertainty(cal_uncertainty),
        p.plot_binary_calibration_uncertainty(cal_table),
    ]
    for fig in figures:
        assert fig.axes
        plt.close(fig)

    with pytest.raises(GP3BayesError):
        p.prediction_profile_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_gradient_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_surface_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_contrast_profile_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.predictive_distribution_atlas_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._atlas_get(object(), 10, False, 1)  # type: ignore[arg-type]
    for values in ((), (0,), (1,), (np.nan,), ("x",)):
        with pytest.raises(GP3BayesError):
            p.predictive_quantile_envelope(atlas, probabilities=values)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prediction_score_uncertainty_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.binary_calibration_uncertainty_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_prediction_gradient(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_predictive_atlas_statistics(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_predictive_quantile_envelope(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_prediction_score_uncertainty(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_binary_calibration_uncertainty(pd.DataFrame({"x": [1]}))


def test_remaining_prediction_plot_adapters_and_validation():
    binary = _prediction("binary", "expected")
    duration = _prediction("duration", "predictive")

    ppc = p.posterior_predictive_statistic(duration, statistic="mean")
    roc = p.binary_roc_curve(binary)
    pr = p.binary_precision_recall_curve(binary)
    group_cal = p.binary_group_calibration(binary, "group")
    qq = p.duration_qq_table(duration, probs=(0.25, 0.5, 0.75))
    tail = p.duration_tail_check(duration, 600)
    group_pred = p.group_prediction_summary(binary, "group")
    widths = p.prediction_interval_width(binary)
    ranks = p.prediction_rank_probabilities(binary, rows=(1, 2, 3))

    makers = [
        lambda: p.plot_prediction_draws(binary, observations=(1, 2), max_draws=20),
        lambda: p.plot_ppc_statistic(ppc, bins=10),
        lambda: p.plot_binary_roc(roc),
        lambda: p.plot_binary_precision_recall(pr),
        lambda: p.plot_binary_group_calibration(group_cal),
        lambda: p.plot_duration_qq(qq),
        lambda: p.plot_duration_tail(tail),
        lambda: p.plot_group_predictions(group_pred, "group"),
        lambda: p.plot_prediction_interval_width(widths),
        lambda: p.plot_prediction_rank_probabilities(ranks),
    ]
    for make in makers:
        fig = make()
        assert fig.axes
        plt.close(fig)

    with pytest.raises(GP3BayesError):
        p.plot_prediction_draws(binary, observations=("x",))  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_prediction_draws(binary, observations=(99,))
    with pytest.raises(GP3BayesError):
        p.plot_ppc_statistic(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.plot_ppc_statistic(ppc, bins=1)
    with pytest.raises(GP3BayesError):
        p.plot_binary_group_calibration(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_duration_qq(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_duration_tail(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_group_predictions(pd.DataFrame({"x": [1]}), "group")
    with pytest.raises(GP3BayesError):
        p.plot_prediction_interval_width(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_prediction_rank_probabilities(pd.DataFrame({"x": [1]}))
