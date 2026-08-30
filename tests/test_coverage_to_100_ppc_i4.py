from __future__ import annotations

import math

import numpy as np
import pandas as pd

import gp3bayespy.ppc as p


def test_ppc_duration_binary_and_status_tail_matrix():
    assert math.isnan(p._sample_sd([1.0]))

    binary = p._binary_summary(
        [0, 1, 1, 0],
        condition=None,
        participant=["p1", "p1", "p2", "p2"],
        item=None,
    )
    assert math.isnan(binary["condition_rate_contrast"])
    assert math.isnan(binary["item_rate_sd"])

    binary_two = p._binary_summary(
        [0, 1, 1, 1],
        condition=["A", "A", "B", "B"],
        participant=["p1", "p1", "p2", "p2"],
        item=["i1", "i2", "i1", "i2"],
    )
    assert np.isfinite(binary_two["condition_rate_contrast"])
    assert np.isfinite(binary_two["item_rate_sd"])

    invalid = p._duration_summary(
        [np.nan, -1.0, 0.0],
        condition=None,
        participant=["p1", "p1", "p2"],
        item=None,
    )
    assert math.isinf(invalid["median"])

    duration = p._duration_summary(
        [1.0, 2.0, 3.0, 6.0],
        condition=["A", "A", "B", "B"],
        participant=["p1", "p1", "p2", "p2"],
        item=["i1", "i2", "i1", "i2"],
    )
    assert duration["condition_median_ratio"] > 1
    assert np.isfinite(duration["item_log_median_sd"])
    assert np.isfinite(duration["coefficient_of_variation"])

    one = p._duration_summary(
        [1.0],
        condition=None,
        participant=["p1"],
        item=None,
    )
    assert math.isnan(one["coefficient_of_variation"])

    assert p._predictive_status(math.nan, 0, 1, -1, 2) == "not_applicable"
    assert p._predictive_status(0.5, 0, 1, -1, 2) == "pass"
    assert p._predictive_status(1.5, 0, 1, -1, 2) == "review"
    assert p._predictive_status(3.0, 0, 1, -1, 2) == "fail"


def test_ppc_check_table_not_applicable_and_finite_paths():
    replicated = pd.DataFrame(
        {
            "finite": np.linspace(0.0, 1.0, 100),
            "empty": [np.nan] * 100,
        }
    )
    table = p._check_table(
        {
            "finite": 0.5,
            "empty": 1.0,
            "nonfinite_observed": math.nan,
        },
        replicated.assign(nonfinite_observed=np.linspace(0.0, 1.0, 100)),
        pass_probability=0.8,
        review_probability=0.95,
    )
    statuses = dict(zip(table["statistic"], table["status"], strict=True))
    assert statuses["finite"] == "pass"
    assert statuses["empty"] == "not_applicable"
    assert statuses["nonfinite_observed"] == "not_applicable"

    replicated_rows = p._replicated_table(
        np.array([[0, 1], [1, 1]], dtype=float),
        summary_function=p._binary_summary,
        condition=["A", "B"],
        participant=["p1", "p1"],
        item=None,
    )
    assert len(replicated_rows) == 2
