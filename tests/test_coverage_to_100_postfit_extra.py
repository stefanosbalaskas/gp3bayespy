from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.postfit_exploration as postfit
from gp3bayespy.exceptions import GP3BayesError


class DataVars:
    def __init__(self, mapping):
        self._mapping = mapping
        self.data_vars = tuple(mapping)

    def __getitem__(self, key):
        return self._mapping[key]


def test_sampler_diagnostics_extra_dimensions_aliases_and_ebfmi_branches():
    stats = DataVars(
        {
            "divergent__": np.array([[0, 1, 0], [0, 0, 0]], dtype=float),
            "treedepth__": np.array([[8, 12, 8], [12, 10, 9]], dtype=float),
            "energy__": np.array([[1.0, 1.2, 1.1], [2.0, 2.0, 2.0]], dtype=float),
            "vector_stat": np.arange(12.0).reshape(2, 3, 2),
            "one_d": np.array([1.0, 2.0]),
        }
    )
    fit = SimpleNamespace(
        backend_fit=SimpleNamespace(sample_stats=stats),
        sampling={"max_treedepth": 12},
    )
    extracted = postfit.extract_sampler_diagnostics(fit)
    assert set(extracted["Parameter"]) >= {
        "divergent__",
        "treedepth__",
        "energy__",
        "vector_stat",
    }
    assert len(extracted.loc[extracted["Parameter"] == "vector_stat"]) == 12

    summary = postfit.sampler_diagnostic_table(fit)
    assert summary.loc[summary["metric"] == "divergent_transitions", "value"].iloc[0] == 1
    assert summary.loc[summary["metric"] == "max_treedepth_hits", "value"].iloc[0] == 2

    ebfmi = summary.loc[summary["metric"].str.startswith("ebfmi_chain_")]
    assert len(ebfmi) == 2
    assert ebfmi["flagged"].any()

    empty = SimpleNamespace(backend_fit=SimpleNamespace(sample_stats=SimpleNamespace(data_vars=())))
    assert postfit.extract_sampler_diagnostics(empty).empty
    assert postfit.sampler_diagnostic_table(empty).empty


def test_mcmc_diagnostic_table_selection_regex_and_quality_fallback(monkeypatch):
    components = {
        "a": np.array([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]]),
        "b": np.ones((2, 4)),
        "c": np.array([[1.0, 1.1, 0.9, 1.0], [1.1, 1.0, 1.2, 0.8]]),
    }
    monkeypatch.setattr(postfit, "_posterior_components", lambda fit: components)

    all_diag = postfit.mcmc_diagnostic_table(object())
    assert set(all_diag["variable"]) == {"a", "b", "c"}

    selected = postfit.mcmc_diagnostic_table(object(), variables=("a", "c"))
    assert set(selected["variable"]) == {"a", "c"}

    regex = postfit.mcmc_diagnostic_table(object(), regex="^[ab]$")
    assert set(regex["variable"]) == {"a", "b"}

    with pytest.raises(GP3BayesError, match="Unknown posterior"):
        postfit.mcmc_diagnostic_table(object(), variables="missing")

    fit = SimpleNamespace(family="binary")
    monkeypatch.setattr(
        postfit,
        "sampler_diagnostic_table",
        lambda fit: (_ for _ in ()).throw(GP3BayesError("not stored")),
    )
    quality = postfit.summarise_mcmc_quality(
        fit,
        min_bulk_ess=1,
        min_tail_ess=1,
        max_mcse_fraction=10,
    )
    assert quality.family == "binary"
    assert quality.sampler.empty
    assert quality.to_frame().equals(quality.issues)


def test_extract_log_likelihood_success_validation_and_shapes():
    group = DataVars(
        {
            "obs": np.arange(24.0).reshape(2, 3, 4) / -10.0,
        }
    )
    fit = SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=group))

    full = postfit.extract_log_likelihood(fit)
    first_two = postfit.extract_log_likelihood(fit, ndraws=2)
    assert full.shape == (6, 4)
    assert first_two.shape == (2, 4)

    with pytest.raises(GP3BayesError, match="fitted-data"):
        postfit.extract_log_likelihood(fit, newdata=pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError, match="positive integer"):
        postfit.extract_log_likelihood(fit, ndraws=0)
    with pytest.raises(GP3BayesError, match="positive integer"):
        postfit.extract_log_likelihood(fit, ndraws=1.5)  # type: ignore[arg-type]

    missing = SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=None))
    with pytest.raises(GP3BayesError, match="not stored"):
        postfit.extract_log_likelihood(missing)

    empty_group = SimpleNamespace(data_vars=())
    with pytest.raises(GP3BayesError, match="no variables"):
        postfit.extract_log_likelihood(
            SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=empty_group))
        )

    one_d = DataVars({"obs": np.array([1.0, 2.0])})
    with pytest.raises(GP3BayesError, match="unsupported shape"):
        postfit.extract_log_likelihood(
            SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=one_d))
        )

    nonfinite = DataVars({"obs": np.array([[[1.0, np.nan]]])})
    with pytest.raises(GP3BayesError, match="not finite"):
        postfit.extract_log_likelihood(
            SimpleNamespace(backend_fit=SimpleNamespace(log_likelihood=nonfinite))
        )


def _fit_for_group_effects():
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "item_id": ["i1", "i2"],
        }
    )
    contract = SimpleNamespace(
        mappings={
            "participant": "participant_id",
            "item": "item_id",
        }
    )
    spec = SimpleNamespace(
        prepared=SimpleNamespace(data=data),
        contract=contract,
    )
    return SimpleNamespace(specification=spec)


def test_group_effect_table_latent_and_flattened_components(monkeypatch):
    fit = _fit_for_group_effects()

    latent = {
        "sd_participant": np.ones((2, 3)) * 0.5,
        "participant_z": np.zeros((2, 3, 2)),
        "participant_z[1]": np.ones((2, 3)) * 0.2,
        "participant_z[2]": np.ones((2, 3)) * -0.4,
        "sd_item": np.ones((2, 3)) * 0.25,
        "item_z": np.zeros((2, 3, 2)),
        "item_z[1]": np.ones((2, 3)) * 0.1,
        "item_z[2]": np.ones((2, 3)) * 0.3,
    }
    monkeypatch.setattr(postfit, "_posterior_components", lambda fit: latent)
    table = postfit.group_effect_table(fit)
    assert len(table) == 4
    assert set(table["group"]) == {"participant", "item"}

    only_participant = postfit.group_effect_table(fit, groups="participant", probs=(0.1, 0.9))
    assert len(only_participant) == 2

    flattened = {
        "r_participant[p1,Intercept]": np.array([[0.1, 0.2], [0.0, 0.1]]),
        "r_participant[p2,Intercept]": np.array([[-0.1, -0.2], [0.0, -0.1]]),
    }
    monkeypatch.setattr(postfit, "_posterior_components", lambda fit: flattened)
    flat_table = postfit.group_effect_table(fit, groups="participant")
    assert len(flat_table) == 2

    with pytest.raises(GP3BayesError, match="Unknown grouping"):
        postfit.group_effect_table(fit, groups="site")

    no_context = SimpleNamespace(specification=None)
    monkeypatch.setattr(postfit, "_posterior_components", lambda fit: {})
    with pytest.raises(GP3BayesError, match="retain prepared"):
        postfit.group_effect_table(no_context)

    with pytest.raises(GP3BayesError, match="No group-level"):
        postfit.group_effect_table(fit, groups="participant")


def test_variance_component_table_delegation(monkeypatch):
    expected = pd.DataFrame(
        {
            "variable": ["sigma"],
            "mean": [1.0],
            "median": [1.0],
            "sd": [0.1],
            "lower": [0.8],
            "upper": [1.2],
        }
    )
    calls = {}

    def fake(fit, variables=None, regex=None, probs=None):
        calls["regex"] = regex
        calls["probs"] = probs
        return expected

    monkeypatch.setattr(postfit, "posterior_interval_table", fake)
    result = postfit.variance_component_table(object(), probs=(0.1, 0.5, 0.9))
    assert result.equals(expected)
    assert calls["regex"] == r"^(sd_|cor_|sigma$)"
    assert calls["probs"] == (0.1, 0.5, 0.9)


def test_loo_diagnostic_summary_model_comparison_and_weights():
    diagnostic = postfit.loo_diagnostic_table({"pareto_k": [0.2, 0.6, 0.8, 1.2, np.nan]})
    assert list(diagnostic["category"][:4]) == [
        "good",
        "okay",
        "review",
        "severe",
    ]
    assert diagnostic["flagged"].iloc[-1]

    obj = SimpleNamespace(pareto_k=np.array([0.1, 0.9]))
    assert len(postfit.loo_diagnostic_table(obj)) == 2

    with pytest.raises(GP3BayesError, match="Pareto"):
        postfit.loo_diagnostic_table(object())

    estimates = pd.DataFrame(
        {
            "Estimate": [-10.0, 2.0],
            "SE": [1.0, 0.2],
        },
        index=["elpd_loo", "p_loo"],
    )
    direct = postfit.loo_summary_table(SimpleNamespace(estimates=estimates))
    assert "quantity" in direct

    raw = postfit.loo_summary_table(
        SimpleNamespace(
            estimates=None,
            raw=SimpleNamespace(estimates=estimates),
        )
    )
    assert len(raw) == 2

    scalars = postfit.loo_summary_table(
        SimpleNamespace(
            elpd_loo=-10.0,
            se_elpd_loo=1.0,
            p_loo=2.0,
            se_p_loo=0.2,
            looic=20.0,
            se_looic=2.0,
        )
    )
    assert set(scalars["quantity"]) == {"elpd_loo", "p_loo", "looic"}

    with pytest.raises(GP3BayesError, match="LOO summary"):
        postfit.loo_summary_table(object())

    comparison = pd.DataFrame(
        {
            "elpd_diff": [0.0, -2.0],
            "se_diff": [0.0, 1.1],
        },
        index=["m1", "m2"],
    )
    table = postfit.model_comparison_table(comparison)
    assert list(table["model"]) == ["m1", "m2"]
    assert not table["automatic_selection"].any()

    wrapped = postfit.model_comparison_table(
        SimpleNamespace(comparison=comparison.reset_index().rename(columns={"index": "model"}))
    )
    assert len(wrapped) == 2

    with pytest.raises(GP3BayesError, match="ELPD"):
        postfit.model_comparison_table(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        postfit.model_comparison_table(object())

    mapping_weights = postfit.model_weights_table({"m1": 0.7, "m2": 0.3})
    series_weights = postfit.model_weights_table(pd.Series([0.6, 0.4], index=["a", "b"]))
    array_weights = postfit.model_weights_table([0.8, 0.2])
    wrapped_weights = postfit.model_weights_table(SimpleNamespace(weights={"x": 1.0}))
    assert list(mapping_weights["model"]) == ["m1", "m2"]
    assert list(series_weights["model"]) == ["a", "b"]
    assert list(array_weights["model"]) == ["model_1", "model_2"]
    assert wrapped_weights.loc[0, "model"] == "x"
    assert not mapping_weights["automatic_selection"].any()

    for bad in ([], [0.5, np.nan]):
        with pytest.raises(GP3BayesError, match="finite numeric"):
            postfit.model_weights_table(bad)
