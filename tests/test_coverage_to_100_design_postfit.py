from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.design_support_diagnostics as design
import gp3bayespy.postfit_exploration as postfit
from gp3bayespy.exceptions import GP3BayesError


def _fixture():
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=8, trials_per_participant=6, n_items=4, seed=1201
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=True,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_binary_model(prepared)
    return sim.data, contract, prepared, spec


def test_design_input_status_missingness_and_fixed_effect_branches(monkeypatch):
    data, contract, prepared, spec = _fixture()

    assert design._status(["pass", "pass"]) == "pass"
    assert design._status(["pass", "review"]) == "review"
    assert design._status(["pass", "fail"]) == "fail"

    frame, c, s = design._input(data, contract)
    assert s is None and c is contract and len(frame) == len(data)
    frame2, _, _ = design._input(prepared)
    assert len(frame2) == len(data)
    frame3, _, spec3 = design._input(spec)
    assert spec3 is spec and len(frame3) == len(data)

    with pytest.raises(GP3BayesError, match="contract"):
        design._input(data)
    with pytest.raises(GP3BayesError):
        design._input(object())

    clean = design.audit_missingness_structure(data, contract)
    assert clean.status in {"pass", "review"}

    missing = data.copy()
    missing.loc[missing.index[:4], "trial_covariate"] = np.nan
    audit = design.audit_missingness_structure(
        missing, contract, review_fraction=0.01, fail_fraction=0.9
    )
    assert audit.status in {"review", "fail"}
    assert not audit.grouping_table.empty

    absent = design.audit_missingness_structure(data.drop(columns=["trial_covariate"]), contract)
    assert absent.status == "fail"
    assert "trial_covariate" in absent.absent_columns

    with pytest.raises(GP3BayesError, match="thresholds"):
        design.audit_missingness_structure(data, contract, review_fraction=0.5, fail_fraction=0.2)

    fixed = design.audit_fixed_effect_design(spec)
    assert fixed.status in {"pass", "review", "fail"}
    assert fixed.n_rows > 0

    empty = data.copy()
    for column in ("selected", "condition", "participant_covariate", "trial_covariate"):
        empty[column] = np.nan
    fixed_empty = design.audit_fixed_effect_design(empty, contract)
    assert fixed_empty.status == "fail"
    assert fixed_empty.error

    monkeypatch.setattr(
        design,
        "_fixed_model_matrix",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("matrix failure")),
    )
    fixed_error = design.audit_fixed_effect_design(data, contract)
    assert fixed_error.status == "fail"
    assert "matrix failure" in fixed_error.error

    with pytest.raises(GP3BayesError, match="condition-number"):
        design.audit_fixed_effect_design(
            data, contract, condition_number_review=50, condition_number_fail=20
        )
    with pytest.raises(GP3BayesError, match="leverage_multiplier"):
        design.audit_fixed_effect_design(data, contract, leverage_multiplier=0.5)


def test_random_effect_support_branches_and_thresholds():
    data, contract, _, spec = _fixture()
    audit = design.audit_random_effects_support(spec)
    assert audit.status in {"pass", "review", "fail"}
    assert not audit.participant_table.empty

    sparse = data.copy()
    sparse["item_id"] = np.arange(len(sparse))
    sparse_audit = design.audit_random_effects_support(sparse, contract)
    assert sparse_audit.status in {"review", "fail", "pass"}

    no_participant = data.drop(columns=["participant_id"])
    no_part = design.audit_random_effects_support(no_participant, contract)
    assert no_part.status == "fail"
    assert no_part.error

    with pytest.raises(GP3BayesError, match="positive integers"):
        design.audit_random_effects_support(data, contract, minimum_repeated_rows=0)


def test_postfit_draw_matrix_probability_interval_correlation_error_matrix():
    draws = pd.DataFrame(
        {
            "a": np.linspace(-1, 1, 100),
            "b": np.linspace(0, 2, 100),
            "c": np.linspace(2, 3, 100),
            "text": ["x"] * 100,
        }
    )
    numeric = postfit._draw_matrix(draws)
    assert list(numeric) == ["a", "b", "c"]
    assert list(postfit._draw_matrix({"a": [1, 2], "b": [3, 4]})) == ["a", "b"]
    assert postfit._draw_matrix(np.arange(10.0)).shape == (10, 1)
    assert postfit._draw_matrix(np.arange(20.0).reshape(10, 2)).shape == (10, 2)

    assert postfit._probs((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    assert postfit._probs((0.1, 0.9), three=False) == (0.1, 0.9)

    interval = postfit.posterior_interval_table(draws, variables=("a", "b"))
    probability = postfit.posterior_probability_table(draws, regex="^[ab]$", rope=(-0.2, 0.2))
    corr = postfit.posterior_correlation_table(draws, variables=("a", "b", "c"))
    assert len(interval) == 2
    assert "probability_in_rope" in probability
    assert len(corr) == 3

    with pytest.raises(GP3BayesError, match="2-D"):
        postfit._draw_matrix(np.zeros((2, 2, 2)))
    with pytest.raises(GP3BayesError, match="Unknown posterior"):
        postfit._draw_matrix(draws, variables="missing")
    with pytest.raises(GP3BayesError, match="valid regular"):
        postfit._draw_matrix(draws, regex="[")
    with pytest.raises(GP3BayesError, match="No posterior"):
        postfit._draw_matrix(draws, regex="^missing$")
    with pytest.raises(GP3BayesError, match="finite"):
        postfit._draw_matrix(pd.DataFrame({"a": [1.0, np.nan]}))
    with pytest.raises(GP3BayesError, match="wrong length"):
        postfit._probs((0.1, 0.9))
    with pytest.raises(GP3BayesError, match="increasing"):
        postfit._probs((0.5, 0.4, 0.9))
    with pytest.raises(GP3BayesError, match="rope"):
        postfit.posterior_probability_table(draws, rope=(1, 0))
    with pytest.raises(GP3BayesError, match="At least two"):
        postfit.posterior_correlation_table(draws, variables="a")
    with pytest.raises(GP3BayesError, match="pearson"):
        postfit.posterior_correlation_table(draws, method="bad")


def test_postfit_rhat_ess_issue_and_sampler_branches(monkeypatch):
    assert np.isnan(postfit._split_rhat(np.ones((1, 3))))
    assert postfit._split_rhat(np.ones((2, 6))) == 1.0
    nonconstant = np.array([[0.0, 0.1, 0.2, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4, 0.3, 0.2]])
    assert np.isfinite(postfit._split_rhat(nonconstant))

    assert postfit._ess_1d(np.array([1.0, 2.0, 3.0])) == 3
    assert postfit._ess_1d(np.ones(20)) == 20
    assert 0 < postfit._ess_1d(np.linspace(-1, 1, 100)) <= 100

    diag = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "sd": [1.0, 0.0],
            "rhat": [1.0, np.nan],
            "ess_bulk": [500, 10],
            "ess_tail": [500, 10],
            "mcse_mean": [0.01, np.nan],
        }
    )
    issues = postfit.identify_mcmc_issues(diag)
    assert issues.loc[1, "flagged"]
    with pytest.raises(GP3BayesError, match="greater than 1"):
        postfit.identify_mcmc_issues(diag, rhat_threshold=1.0)
    with pytest.raises(GP3BayesError, match="fit or diagnostic"):
        postfit.identify_mcmc_issues(object())
    with pytest.raises(GP3BayesError, match="missing required"):
        postfit.identify_mcmc_issues(pd.DataFrame({"variable": ["a"]}))

    class Stats:
        data_vars = ("energy", "tree_depth", "diverging", "one_d", "bad")

        def __getitem__(self, key):
            if key == "bad":
                raise TypeError("skip")
            if key == "one_d":
                return np.array([1.0, 2.0])
            if key == "energy":
                return np.array([[1.0, 1.2, 1.1], [1.3, 1.1, 1.2]])
            if key == "tree_depth":
                return np.array([[8, 12, 8], [9, 10, 12]])
            return np.array([[0, 1, 0], [0, 0, 0]])

    fit = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats=Stats()),
        sampling={"max_treedepth": 12},
    )
    extracted = postfit.extract_sampler_diagnostics(fit)
    assert not extracted.empty
    summary = postfit.sampler_diagnostic_table(fit)
    assert set(summary["metric"]) >= {"divergent_transitions", "max_treedepth_hits"}

    empty_fit = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats=SimpleNamespace(data_vars=()))
    )
    assert postfit.extract_sampler_diagnostics(empty_fit).empty
    assert postfit.sampler_diagnostic_table(empty_fit).empty

    with pytest.raises(GP3BayesError, match="unavailable"):
        postfit.extract_sampler_diagnostics(SimpleNamespace(backend_fit=None))
