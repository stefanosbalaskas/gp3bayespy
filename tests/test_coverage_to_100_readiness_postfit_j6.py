from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

r = importlib.import_module("gp3bayespy.readiness")
pe = importlib.import_module("gp3bayespy.postfit_exploration")


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


class Dataset(dict):
    @property
    def data_vars(self):
        return self


def _interaction_contract():
    return gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("a", "b"),
        interaction=("a", "b"),
        random_slope=False,
    )


def test_readiness_trial_condition_time_and_predictor_remaining_branches():
    rows, add = _collector()

    unique = pd.DataFrame({"p": ["p1", "p1"], "t": ["t1", "t2"]})
    r._audit_trial_structure(unique, "p", "t", add)
    assert rows[-1][2] == "pass"

    rows.clear()
    r._audit_trial_structure(pd.DataFrame({"p": ["p1"]}), "p", "t", add)
    assert rows == []

    rows.clear()
    invalid_pid = pd.DataFrame({"p": pd.to_datetime(["2026-01-01", "2026-01-02"]), "c": [0, 1]})
    r._audit_condition_structure(invalid_pid, "p", "c", True, add)
    assert any(x[0] == "random_slope_support" and x[2] == "fail" for x in rows)

    rows.clear()
    replicated = pd.DataFrame(
        {
            "p": ["p1"] * 4 + ["p2"] * 4,
            "c": [0, 0, 1, 1] * 2,
        }
    )
    r._audit_condition_structure(replicated, "p", "c", True, add)
    assert any(x[0] == "random_slope_support" and x[2] == "pass" for x in rows)
    assert any(x[0] == "random_slope_replication" and x[2] == "pass" for x in rows)

    rows.clear()
    r._audit_time_structure(pd.DataFrame({"p": ["p1"]}), "p", "missing", add)
    assert rows == []

    rows.clear()
    time_bad_pid = pd.DataFrame(
        {
            "p": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "time": [0.0, 1.0],
        }
    )
    r._audit_time_structure(time_bad_pid, "p", "time", add)
    assert any(x[0] == "time_within_participant" and x[2] == "fail" for x in rows)

    rows.clear()
    good_time = pd.DataFrame({"p": ["p1", "p1", "p2", "p2"], "time": [0.0, 1.0, 0.0, 1.0]})
    r._audit_time_structure(good_time, "p", "time", add)
    assert any(x[0] == "time_within_participant" and x[2] == "pass" for x in rows)

    rows.clear()
    predictors = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "blank": ["a", "", "b"],
            "cat": pd.Series(pd.Categorical(["a", "a", "b"], categories=["a", "b", "unused"])),
        }
    )
    r._audit_predictor_structure(predictors, ("num", "blank", "cat"), add)
    assert any(x[0] == "predictor_blanks" and x[2] == "fail" for x in rows)
    assert any(x[0] == "predictor_factor_levels" and x[2] == "warn" for x in rows)


def test_readiness_interaction_branch_matrix():
    contract = _interaction_contract()
    rows, add = _collector()

    r._audit_interaction_structure(pd.DataFrame({"a": [1, 2]}), contract, add)
    assert rows[-1][2] == "fail"

    rows.clear()
    unsupported = pd.DataFrame(
        {
            "a": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "b": ["x", "y"],
        }
    )
    r._audit_interaction_structure(unsupported, contract, add)
    assert rows[-1][2] == "fail"

    rows.clear()
    invariant = pd.DataFrame({"a": ["x", "x"], "b": ["u", "v"]})
    r._audit_interaction_structure(invariant, contract, add)
    assert rows[-1][2] == "fail"

    rows.clear()
    numeric = pd.DataFrame({"a": [1.0, 2.0], "b": ["u", "v"]})
    r._audit_interaction_structure(numeric, contract, add)
    assert rows[-1][2] == "pass"

    rows.clear()
    no_complete = pd.DataFrame(
        {
            "a": ["x", "y", None],
            "b": ["u", None, "v"],
        }
    )
    r._audit_interaction_structure(no_complete, contract, add)
    assert rows[-1][2] == "fail"

    rows.clear()
    weak = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    r._audit_interaction_structure(weak, contract, add)
    assert rows[-1][2] == "warn"

    rows.clear()
    replicated = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["u", "u", "v", "v"]})
    r._audit_interaction_structure(replicated, contract, add)
    assert rows[-1][2] == "pass"


def test_postfit_draw_probability_correlation_and_diagnostic_helpers():
    mapping = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    assert pe._draw_matrix(mapping).shape == (2, 2)
    assert pe._draw_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    with pytest.raises(GP3BayesError):
        pe._draw_matrix(np.ones((2, 2, 2)))
    with pytest.raises(GP3BayesError):
        pe._draw_matrix(mapping, variables=("missing",))
    with pytest.raises(GP3BayesError):
        pe._draw_matrix(mapping, regex="[")
    with pytest.raises(GP3BayesError):
        pe._draw_matrix(mapping, regex="nomatch")
    with pytest.raises(GP3BayesError):
        pe._draw_matrix({"a": [1.0, np.inf]})

    assert pe._probs((0.1, 0.9), three=False) == (0.1, 0.9)
    with pytest.raises(GP3BayesError):
        pe._probs((0.1, 0.2), three=True)
    with pytest.raises(GP3BayesError):
        pe._probs((0.9, 0.1), three=False)

    one = pe.posterior_interval_table(pd.DataFrame({"a": [1.0]}))
    assert one.iloc[0]["sd"] == 0.0
    rope = pe.posterior_probability_table(pd.DataFrame({"a": [-1.0, 0.0, 1.0]}), rope=(-0.2, 0.2))
    assert "probability_in_rope" in rope
    with pytest.raises(GP3BayesError):
        pe.posterior_probability_table(pd.DataFrame({"a": [1.0]}), rope=(1, 0))

    corr = pe.posterior_correlation_table(
        pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), method="spearman"
    )
    assert corr.iloc[0]["method"] == "spearman"
    with pytest.raises(GP3BayesError):
        pe.posterior_correlation_table(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(GP3BayesError):
        pe.posterior_correlation_table(pd.DataFrame({"a": [1, 2], "b": [2, 3]}), method="bad")

    assert np.isnan(pe._split_rhat(np.ones((1, 3))))
    assert pe._split_rhat(np.ones((2, 4))) == 1.0
    assert pe._ess_1d(np.array([1.0, 2.0, 3.0])) == 3.0
    assert pe._ess_1d(np.ones(10)) == 10.0

    with pytest.raises(GP3BayesError):
        pe.identify_mcmc_issues(pd.DataFrame(), rhat_threshold=1.0)
    with pytest.raises(GP3BayesError):
        pe.identify_mcmc_issues(object())
    with pytest.raises(GP3BayesError):
        pe.identify_mcmc_issues(pd.DataFrame({"variable": ["a"]}))

    d = pd.DataFrame(
        {
            "variable": ["a"],
            "sd": [0.0],
            "rhat": [np.nan],
            "ess_bulk": [np.nan],
            "ess_tail": [10.0],
            "mcse_mean": [0.1],
        }
    )
    issues = pe.identify_mcmc_issues(d)
    assert bool(issues.iloc[0]["flagged"])


def test_postfit_sampler_loglik_loo_and_weight_branches():
    with pytest.raises(GP3BayesError):
        pe.extract_sampler_diagnostics(SimpleNamespace(backend_fit=SimpleNamespace()))

    empty_fit = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats=Dataset({"x": np.ones(3)}))
    )
    assert pe.extract_sampler_diagnostics(empty_fit).empty
    assert pe.sampler_diagnostic_table(empty_fit).empty

    stats = Dataset(
        {
            "diverging": np.array([[0, 1, 0], [0, 0, 0]]),
            "tree_depth": np.array([[2, 12, 2], [2, 2, 12]]),
            "energy": np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]]),
        }
    )
    fit = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats=stats),
        sampling={"max_treedepth": 12},
    )
    sampler = pe.sampler_diagnostic_table(fit)
    assert "divergent_transitions" in set(sampler["metric"])
    assert any(sampler["metric"].astype(str).str.startswith("ebfmi_chain_"))

    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit, newdata=pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit)

    bad_group = Dataset()
    fit_ll = SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=bad_group))
    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit_ll)

    fit_ll.backend_fit.log_likelihood = Dataset({"y": np.ones(3)})
    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit_ll)

    fit_ll.backend_fit.log_likelihood = Dataset({"y": np.arange(12.0).reshape(2, 3, 2)})
    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit_ll, ndraws=0)
    arr = pe.extract_log_likelihood(fit_ll, ndraws=2)
    assert arr.shape == (2, 2)

    fit_ll.backend_fit.log_likelihood = Dataset({"y": np.array([[[1.0], [np.inf]]])})
    with pytest.raises(GP3BayesError):
        pe.extract_log_likelihood(fit_ll)

    with pytest.raises(GP3BayesError):
        pe.loo_diagnostic_table(object())
    loo = pe.loo_diagnostic_table({"pareto_k": [0.2, 0.6, 0.8, 1.2, np.nan]})
    assert set(loo["category"]) >= {"good", "okay", "review", "severe"}

    estimates = pd.DataFrame({"estimate": [1.0]}, index=["elpd_loo"])
    summary = pe.loo_summary_table(SimpleNamespace(estimates=estimates))
    assert "quantity" in summary
    scalar = pe.loo_summary_table(
        SimpleNamespace(elpd_loo=1.0, se_elpd_loo=0.1, p_loo=2.0, looic=3.0)
    )
    assert len(scalar) == 3
    with pytest.raises(GP3BayesError):
        pe.loo_summary_table(object())

    comp = pe.model_comparison_table(
        pd.DataFrame({"elpd_diff": [0.0], "se_diff": [0.0]}, index=["m1"])
    )
    assert comp.iloc[0]["model"] == "m1"
    with pytest.raises(GP3BayesError):
        pe.model_comparison_table(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        pe.model_comparison_table(object())

    assert len(pe.model_weights_table({"a": 0.7, "b": 0.3})) == 2
    assert len(pe.model_weights_table(pd.Series([0.4, 0.6], index=["a", "b"]))) == 2
    assert len(pe.model_weights_table([0.4, 0.6])) == 2
    with pytest.raises(GP3BayesError):
        pe.model_weights_table([])
    with pytest.raises(GP3BayesError):
        pe.model_weights_table([0.5, np.nan])
