"""Advanced optional Bayesian workflows.

This module implements the frozen gp3bayes 0.5.0 optional-workflow contracts
without making optional dependencies part of the core installation.  PSIS-LOO
is implemented directly in NumPy/SciPy; optional sampling backends are gated
explicitly and never substituted silently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import genpareto

from .backends import backend_capabilities, validate_backend_environment
from .binary import (
    BinaryModelSpecification,
    fit_binary_model,
    specify_binary_model,
    translate_binary_model_to_brms,
)
from .duration import (
    DurationModelSpecification,
    fit_duration_model,
    specify_duration_model,
    translate_duration_model_to_brms,
)
from .exceptions import BackendUnavailableError, GP3BayesError
from .postfit_exploration import extract_log_likelihood


def bayesian_backend_capabilities() -> pd.DataFrame:
    """Return the approved Python Bayesian backend capability table."""
    return backend_capabilities()


def check_cmdstan_backend(strict: bool = False):
    """Validate CmdStanPy plus an external CmdStan runtime."""
    return validate_backend_environment("cmdstanpy", compile_test=False, strict=strict)


@dataclass(slots=True)
class InteractionPriorSpecification:
    base: Any
    advanced_priors: Mapping[str, Any]
    advanced_prior_version: str = "0.2"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass(slots=True)
class InteractionBackendSpecification:
    base: Any
    interaction_scale: float
    interaction_tag: str = "interaction"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def specify_binary_model_with_interaction_prior(
    prepared: Any,
    baseline: float,
    intercept_scale: float = 1.5,
    main_effect_scale: float = 0.75,
    interaction_scale: float = 0.50,
    group_sd_scale: float = 1,
    correlation_eta: float = 2,
    student_df: float = 3,
) -> InteractionPriorSpecification:
    interaction = getattr(getattr(prepared, "contract", None), "interaction", None)
    if interaction is None:
        raise GP3BayesError("A declared two-way interaction is required for this specification.")
    if main_effect_scale <= 0 or interaction_scale <= 0:
        raise GP3BayesError("Prior scales must be positive.")
    base = specify_binary_model(
        prepared,
        baseline=baseline,
        intercept_scale=intercept_scale,
        coefficient_scale=main_effect_scale,
        group_sd_scale=group_sd_scale,
        correlation_eta=correlation_eta,
        student_df=student_df,
    )
    return InteractionPriorSpecification(
        base,
        {
            "main_effect_scale": float(main_effect_scale),
            "interaction_scale": float(interaction_scale),
            "interaction": tuple(interaction),
            "interaction_tag": "interaction",
        },
    )


def specify_duration_model_with_interaction_prior(
    prepared: Any,
    baseline: float,
    intercept_scale: float = 1,
    main_effect_scale: float = 0.35,
    interaction_scale: float = 0.25,
    group_sd_scale: float = 0.5,
    residual_scale: float = 0.5,
    correlation_eta: float = 2,
    student_df: float = 3,
) -> InteractionPriorSpecification:
    interaction = getattr(getattr(prepared, "contract", None), "interaction", None)
    if interaction is None:
        raise GP3BayesError("A declared two-way interaction is required for this specification.")
    if main_effect_scale <= 0 or interaction_scale <= 0:
        raise GP3BayesError("Prior scales must be positive.")
    base = specify_duration_model(
        prepared,
        baseline=baseline,
        intercept_scale=intercept_scale,
        coefficient_scale=main_effect_scale,
        group_sd_scale=group_sd_scale,
        residual_scale=residual_scale,
        correlation_eta=correlation_eta,
        student_df=student_df,
    )
    return InteractionPriorSpecification(
        base,
        {
            "main_effect_scale": float(main_effect_scale),
            "interaction_scale": float(interaction_scale),
            "interaction": tuple(interaction),
            "interaction_tag": "interaction",
        },
    )


def interaction_prior_summary(specification: InteractionPriorSpecification) -> pd.DataFrame:
    if not isinstance(specification, InteractionPriorSpecification):
        raise GP3BayesError("`specification` does not contain separate interaction-prior metadata.")
    m = specification.advanced_priors
    return pd.DataFrame(
        [
            {
                "family": specification.family,
                "interaction": ":".join(m["interaction"]),
                "main_effect_scale": m["main_effect_scale"],
                "interaction_scale": m["interaction_scale"],
                "interaction_tag": m["interaction_tag"],
            }
        ]
    )


def translate_binary_model_with_interaction_prior(
    specification: InteractionPriorSpecification,
) -> InteractionBackendSpecification:
    if (
        not isinstance(specification, InteractionPriorSpecification)
        or specification.family != "binary"
    ):
        raise GP3BayesError("`specification` must be a binary interaction-prior specification.")
    return InteractionBackendSpecification(
        translate_binary_model_to_brms(specification.base),
        float(specification.advanced_priors["interaction_scale"]),
    )


def translate_duration_model_with_interaction_prior(
    specification: InteractionPriorSpecification,
) -> InteractionBackendSpecification:
    if (
        not isinstance(specification, InteractionPriorSpecification)
        or specification.family != "duration"
    ):
        raise GP3BayesError("`specification` must be a duration interaction-prior specification.")
    return InteractionBackendSpecification(
        translate_duration_model_to_brms(specification.base),
        float(specification.advanced_priors["interaction_scale"]),
    )


def _base_specification(specification: Any) -> Any:
    return (
        specification.base
        if isinstance(specification, InteractionPriorSpecification)
        else specification
    )


def fit_binary_model_backend(
    specification: BinaryModelSpecification | InteractionPriorSpecification,
    backend: str = "pymc",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
):
    backend = {"rstan": "pymc", "cmdstanr": "cmdstanpy"}.get(backend, backend)
    if backend == "pymc":
        return fit_binary_model(
            _base_specification(specification),
            chains=chains,
            iter=iter,
            warmup=warmup,
            cores=cores,
            seed=seed,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            refresh=refresh,
        )
    if backend == "cmdstanpy":
        check_cmdstan_backend(strict=True)
        raise BackendUnavailableError(
            "CmdStanPy is detected, but the restricted gp3bayespy CmdStan model compiler "
            "is not enabled in this release candidate. Use backend='pymc'."
        )
    raise GP3BayesError("`backend` must be 'pymc' or 'cmdstanpy'.")


def fit_duration_model_backend(
    specification: DurationModelSpecification | InteractionPriorSpecification,
    backend: str = "pymc",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
):
    backend = {"rstan": "pymc", "cmdstanr": "cmdstanpy"}.get(backend, backend)
    if backend == "pymc":
        return fit_duration_model(
            _base_specification(specification),
            chains=chains,
            iter=iter,
            warmup=warmup,
            cores=cores,
            seed=seed,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            refresh=refresh,
        )
    if backend == "cmdstanpy":
        check_cmdstan_backend(strict=True)
        raise BackendUnavailableError(
            "CmdStanPy is detected, but the restricted gp3bayespy CmdStan model compiler "
            "is not enabled in this release candidate. Use backend='pymc'."
        )
    raise GP3BayesError("`backend` must be 'pymc' or 'cmdstanpy'.")


def fit_binary_model_cmdstanr(
    specification: Any,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
):
    return fit_binary_model_backend(
        specification,
        "cmdstanpy",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


def fit_duration_model_cmdstanr(
    specification: Any,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
):
    return fit_duration_model_backend(
        specification,
        "cmdstanpy",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


@dataclass(slots=True)
class PSISLOOResult:
    status: str
    pointwise: pd.DataFrame
    pareto_k: np.ndarray
    influence_pareto_k: np.ndarray
    flagged_observations: np.ndarray
    severe_observations: np.ndarray
    pareto_k_table: pd.DataFrame
    mcse_loo: float
    source: Any
    elpd_loo: float
    se_elpd_loo: float
    p_loo: float
    se_p_loo: float
    looic: float
    se_looic: float
    automatic_selection: bool = False
    interpretation: str = (
        "PSIS-LOO estimates out-of-sample predictive performance. Pareto-k diagnostics "
        "must be inspected before interpretation. No model is selected automatically."
    )


@dataclass(slots=True)
class LOOComparison:
    status: str
    comparison: pd.DataFrame
    loo: Mapping[str, PSISLOOResult]
    automatic_selection: bool = False
    interpretation: str = (
        "ELPD differences and uncertainty are descriptive. No model is automatically selected."
    )


@dataclass(slots=True)
class LOOWeights:
    method: str
    weights: Mapping[str, float]
    automatic_selection: bool = False
    interpretation: str = "Weights combine predictive distributions. They are not evidence that a single model is substantively correct."


def _psis_smooth(log_ratios: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth one vector of log importance ratios."""
    lr = np.asarray(log_ratios, float)
    if lr.ndim != 1 or not np.isfinite(lr).all():
        raise GP3BayesError("Log importance ratios must be one finite vector.")
    n = len(lr)
    if n < 5:
        w = np.exp(lr - logsumexp(lr))
        return w, float("nan")
    shifted = lr - np.max(lr)
    raw = np.exp(shifted)
    tail_n = max(5, min(n // 5, int(3 * math.sqrt(n))))
    order = np.argsort(raw)
    tail_idx = order[-tail_n:]
    threshold = raw[order[-tail_n - 1]] if tail_n < n else 0.0
    excess = raw[tail_idx] - threshold
    positive = excess[excess > 0]
    k = float("nan")
    smoothed = raw.copy()
    if len(positive) >= 4 and np.ptp(positive) > 0:
        try:
            k_fit, _, scale = genpareto.fit(positive, floc=0)
            k = float(k_fit)
            probs = (np.arange(1, tail_n + 1) - 0.5) / tail_n
            expected = threshold + genpareto.ppf(probs, c=k_fit, loc=0, scale=scale)
            expected = np.where(np.isfinite(expected), expected, raw[tail_idx])
            smoothed[tail_idx] = np.sort(expected)
        except Exception:
            pass
    # Standard finite-sample truncation used by PSIS implementations.
    cap = np.mean(smoothed) * (n**0.75)
    smoothed = np.minimum(smoothed, cap)
    total = smoothed.sum()
    if not np.isfinite(total) or total <= 0:
        raise GP3BayesError("Importance-ratio normalization failed.")
    return smoothed / total, k


def compute_psis_loo_from_log_lik(
    log_lik: Any,
    chain_id: Sequence[int] | None = None,
    cores: int = 1,
    save_psis: bool = True,
) -> PSISLOOResult:
    ll = np.asarray(log_lik, dtype=float)
    if ll.ndim != 2 or not np.isfinite(ll).all():
        raise GP3BayesError("`log_lik` must be a finite numeric matrix.")
    if chain_id is not None and len(chain_id) != ll.shape[0]:
        raise GP3BayesError("`chain_id` must have one value per log-likelihood row.")
    s, n = ll.shape
    point_rows = []
    k_values = np.empty(n, dtype=float)
    influence = np.empty(n, dtype=float)
    for i in range(n):
        z = ll[:, i]
        weights, k = _psis_smooth(-z)
        k_values[i] = k
        influence[i] = k
        loo_i = float(logsumexp(z + np.log(weights)))
        lpd_i = float(logsumexp(z) - math.log(s))
        p_i = lpd_i - loo_i
        point_rows.append(
            {
                "elpd_loo": loo_i,
                "mcse_elpd_loo": float(np.std(z) / math.sqrt(s)),
                "p_loo": p_i,
                "looic": -2 * loo_i,
            }
        )
    pointwise = pd.DataFrame(point_rows)
    elpd = float(pointwise["elpd_loo"].sum())
    p_loo = float(pointwise["p_loo"].sum())
    se_elpd = float(math.sqrt(n * np.var(pointwise["elpd_loo"], ddof=1))) if n > 1 else 0.0
    se_p = float(math.sqrt(n * np.var(pointwise["p_loo"], ddof=1))) if n > 1 else 0.0
    flagged = np.flatnonzero(~np.isfinite(k_values) | (k_values >= 0.7)) + 1
    severe = np.flatnonzero(np.isfinite(k_values) & (k_values >= 1.0)) + 1
    status = "fail" if len(severe) else ("review" if len(flagged) else "pass")
    bins = [
        (-np.inf, 0.5, "good"),
        (0.5, 0.7, "okay"),
        (0.7, 1.0, "review"),
        (1.0, np.inf, "severe"),
    ]
    pareto_table = pd.DataFrame(
        [
            {
                "category": label,
                "count": int(np.sum((k_values >= low) & (k_values < high))),
            }
            for low, high, label in bins
        ]
    )
    mcse = float(math.sqrt(np.sum(pointwise["mcse_elpd_loo"].to_numpy(float) ** 2)))
    return PSISLOOResult(
        status,
        pointwise,
        k_values,
        influence,
        flagged.astype(int),
        severe.astype(int),
        pareto_table,
        mcse,
        "log_likelihood_matrix",
        elpd,
        se_elpd,
        p_loo,
        se_p,
        -2 * elpd,
        2 * se_elpd,
    )


def compute_psis_loo(
    fit: Any,
    moment_match: bool = False,
    reloo: bool = False,
    cores: int = 1,
    save_psis: bool = True,
) -> PSISLOOResult:
    if moment_match or reloo:
        # Exact refitting/moment matching is backend-specific and cannot be
        # silently approximated. Base PSIS-LOO remains available.
        raise GP3BayesError(
            "`moment_match` and `reloo` are not automatic in gp3bayespy; request explicit refits instead."
        )
    ll = extract_log_likelihood(fit)
    result = compute_psis_loo_from_log_lik(ll, cores=cores, save_psis=save_psis)
    result.source = fit
    return result


def identify_loo_influential_observations(
    x: PSISLOOResult,
    threshold: float | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not isinstance(x, PSISLOOResult):
        raise GP3BayesError("`x` must be a PSISLOOResult.")
    ids = (
        x.flagged_observations
        if threshold is None
        else np.flatnonzero(x.pareto_k >= float(threshold)) + 1
    )
    result = pd.DataFrame(
        {
            "observation": ids.astype(int),
            "pareto_k": x.pareto_k[ids - 1],
            "influence_pareto_k": x.influence_pareto_k[ids - 1],
            "severe": x.pareto_k[ids - 1] >= 1,
        }
    )
    if data is not None:
        if len(data) != len(x.pareto_k):
            raise GP3BayesError("`data` must have one row per PSIS-LOO observation.")
        result = pd.concat(
            [result.reset_index(drop=True), data.iloc[ids - 1].reset_index(drop=True)], axis=1
        )
    return result


def compare_psis_loo(
    models: Mapping[str, Any],
    moment_match: bool = False,
    reloo: bool = False,
    cores: int = 1,
) -> LOOComparison:
    if not isinstance(models, Mapping) or len(models) < 2 or any(not str(k) for k in models):
        raise GP3BayesError("`models` must be a named mapping containing at least two models.")
    wrapped: dict[str, PSISLOOResult] = {}
    for name, model in models.items():
        wrapped[str(name)] = (
            model
            if isinstance(model, PSISLOOResult)
            else compute_psis_loo(model, moment_match=moment_match, reloo=reloo, cores=cores)
        )
    counts = {len(v.pointwise) for v in wrapped.values()}
    if len(counts) != 1:
        raise GP3BayesError("All models must contain the same number of pointwise observations.")
    best_name = max(wrapped, key=lambda name: wrapped[name].elpd_loo)
    best = wrapped[best_name].pointwise["elpd_loo"].to_numpy(float)
    rows = []
    for name, result in wrapped.items():
        delta = result.pointwise["elpd_loo"].to_numpy(float) - best
        rows.append(
            {
                "model": name,
                "elpd_diff": float(delta.sum()),
                "se_diff": float(math.sqrt(len(delta) * np.var(delta, ddof=1)))
                if len(delta) > 1
                else 0.0,
            }
        )
    table = pd.DataFrame(rows).sort_values("elpd_diff", ascending=False).reset_index(drop=True)
    return LOOComparison("review", table, wrapped)


def compute_loo_model_weights(
    x: LOOComparison | Mapping[str, PSISLOOResult],
    method: str = "stacking",
    cores: int = 1,
) -> LOOWeights:
    if method not in {"stacking", "pseudobma"}:
        raise GP3BayesError("`method` must be 'stacking' or 'pseudobma'.")
    models = dict(x.loo) if isinstance(x, LOOComparison) else dict(x)
    if len(models) < 2 or not all(isinstance(v, PSISLOOResult) for v in models.values()):
        raise GP3BayesError("`x` must contain at least two PSISLOOResult objects.")
    names = list(models)
    pointwise = np.column_stack([models[n].pointwise["elpd_loo"].to_numpy(float) for n in names])
    if method == "pseudobma":
        total = pointwise.sum(axis=0)
        weights = np.exp(total - logsumexp(total))
    else:
        # Stacking maximizes summed log predictive density of a convex mixture.
        def objective(w: np.ndarray) -> float:
            logw = np.log(np.clip(w, 1e-15, 1))
            return -float(np.sum(logsumexp(pointwise + logw, axis=1)))

        initial = np.full(len(names), 1 / len(names))
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        weights = result.x if result.success else initial
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
    return LOOWeights(
        method, {name: float(weight) for name, weight in zip(names, weights, strict=True)}
    )


def detect_binary_separation(
    x: Any, formula: Any = None, data: pd.DataFrame | None = None
) -> dict[str, Any]:
    spec = getattr(x, "specification", x)
    prepared = getattr(spec, "prepared", None)
    frame = data if data is not None else getattr(prepared, "data", None)
    contract = getattr(spec, "contract", None) or getattr(prepared, "contract", None)
    if not isinstance(frame, pd.DataFrame) or contract is None:
        raise GP3BayesError("Binary separation screening requires model data and a contract.")
    outcome = contract.mappings.get("outcome")
    if outcome not in frame:
        raise GP3BayesError("The binary outcome column is unavailable.")
    from .binary import _fixed_model_matrix

    clean = frame.dropna()
    matrix, names = _fixed_model_matrix(clean, contract)
    y = pd.to_numeric(clean[outcome], errors="raise").to_numpy(int)
    # Flag design columns whose sign/dummy activation perfectly predicts one class.
    rows = []
    separated = False
    for j, name in enumerate(names):
        z = np.asarray(matrix[:, j], float)
        if np.allclose(z, z[0]):
            code = 0.0
        else:
            positive = z > np.median(z)
            if (
                positive.any()
                and (~positive).any()
                and (np.unique(y[positive]).size == 1 or np.unique(y[~positive]).size == 1)
            ):
                code = math.inf
                separated = True
            else:
                code = 0.0
        rows.append({"coefficient": name, "separation_code": code})
    return {
        "status": "review" if separated else "pass",
        "separated": separated,
        "coefficients": pd.DataFrame(rows),
        "automatic_variable_removal": False,
    }


@dataclass(slots=True)
class PathologicalSimulation:
    family: str
    scenario: str
    data: pd.DataFrame
    expected_issue: str
    seed: int


def simulate_binary_pathology(
    scenario: str = "null_contrast", seed: int = 1
) -> PathologicalSimulation:
    rng = np.random.default_rng(seed)
    n_participants = 24
    trials = 8
    participant = np.repeat(np.arange(1, n_participants + 1), trials)
    condition = np.tile(np.repeat([0, 1], trials // 2), n_participants)
    eta = np.full(len(participant), -0.4)
    issue = scenario
    if scenario == "null_contrast":
        pass
    elif scenario == "weak_information":
        participant, condition, eta = participant[:48], condition[:48], eta[:48]
    elif scenario == "severe_imbalance":
        condition = (rng.random(len(condition)) < 0.08).astype(int)
    elif scenario == "near_separation":
        eta += np.where(condition == 1, 5.5, -1.5)
    elif scenario == "omitted_random_slope":
        slopes = rng.normal(0, 2, n_participants)
        eta += slopes[participant - 1] * condition
    elif scenario == "sparse_item_structure":
        pass
    elif scenario == "all_zero_participants":
        eta[participant <= 3] = -20
    elif scenario == "rank_deficiency" or scenario == "missing_outcomes":
        pass
    else:
        raise GP3BayesError("Unknown binary pathology scenario.")
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    if scenario == "missing_outcomes":
        y[rng.choice(len(y), size=max(1, len(y) // 10), replace=False)] = np.nan
    data = pd.DataFrame(
        {
            "participant_id": participant,
            "condition": condition,
            "selected": y,
            "trial": np.arange(1, len(y) + 1),
        }
    )
    if scenario == "rank_deficiency":
        data["condition_duplicate"] = data["condition"]
    if scenario == "sparse_item_structure":
        data["item_id"] = np.arange(1, len(data) + 1)
    return PathologicalSimulation("binary", scenario, data, issue, seed)


def simulate_duration_pathology(
    scenario: str = "null_ratio", seed: int = 1
) -> PathologicalSimulation:
    rng = np.random.default_rng(seed)
    n = 192
    participant = np.repeat(np.arange(1, 25), 8)
    condition = np.tile(np.repeat([0, 1], 4), 24)
    log_y = rng.normal(math.log(600), 0.35, n)
    if scenario == "high_group_heterogeneity":
        log_y += rng.normal(0, 0.8, 24)[participant - 1]
    elif scenario == "weak_information":
        participant, condition, log_y = participant[:48], condition[:48], log_y[:48]
    elif scenario == "severe_imbalance":
        condition = (rng.random(len(condition)) < 0.08).astype(int)
    elif scenario == "heavy_tailed_contamination":
        log_y[rng.choice(len(log_y), 10, replace=False)] += rng.normal(2.5, 0.5, 10)
    elif scenario == "mixture":
        log_y += rng.binomial(1, 0.2, len(log_y)) * 1.2
    elif scenario in {
        "censoring",
        "incorrect_unit",
        "zero_duration",
        "negative_duration",
        "null_ratio",
    }:
        pass
    else:
        raise GP3BayesError("Unknown duration pathology scenario.")
    y = np.exp(log_y)
    if scenario == "censoring":
        y = np.minimum(y, np.quantile(y, 0.9))
    if scenario == "incorrect_unit":
        y *= 1000
    if scenario == "zero_duration":
        y[0] = 0
    if scenario == "negative_duration":
        y[0] = -abs(y[0])
    data = pd.DataFrame(
        {
            "participant_id": participant,
            "condition": condition,
            "duration": y,
            "trial": np.arange(1, len(y) + 1),
        }
    )
    return PathologicalSimulation("duration", scenario, data, scenario, seed)


def evaluate_pathological_simulation(x: PathologicalSimulation) -> pd.DataFrame:
    if not isinstance(x, PathologicalSimulation):
        raise GP3BayesError("`x` must be a pathological simulation object.")
    frame = x.data
    numeric = frame.select_dtypes(include=[np.number])
    return pd.DataFrame(
        [
            {
                "family": x.family,
                "scenario": x.scenario,
                "rows": len(frame),
                "missing_cells": int(frame.isna().sum().sum()),
                "nonfinite_cells": int(np.sum(~np.isfinite(numeric.to_numpy(float)))),
                "expected_issue": x.expected_issue,
                "automatic_exclusion": False,
            }
        ]
    )


@dataclass(slots=True)
class SBCPlan:
    generator_function: Callable[..., Any]
    backend: str
    n_sims: int
    generator_args: Mapping[str, Any]
    seed: int
    specification: Any = None
    sampling: Mapping[str, Any] | None = None


@dataclass(slots=True)
class SBCResult:
    plan: SBCPlan
    simulations: pd.DataFrame
    ranks: pd.DataFrame
    fits_retained: tuple[Any, ...] = ()
    automatic_calibration_claim: bool = False


def create_custom_sbc_plan(
    generator_function: Callable[..., Any],
    backend: str,
    n_sims: int = 20,
    generator_args: Mapping[str, Any] | None = None,
    seed: int = 1,
) -> SBCPlan:
    if not callable(generator_function) or n_sims < 1:
        raise GP3BayesError("A callable generator and positive `n_sims` are required.")
    return SBCPlan(generator_function, backend, int(n_sims), dict(generator_args or {}), int(seed))


def create_brms_sbc_plan(
    specification: Any,
    n_sims: int = 20,
    backend: str = "pymc",
    chains: int = 2,
    iter: int = 1000,
    warmup: int = 500,
    thin: int = 1,
    seed: int = 1,
    generator_iter: int = 3000,
    generator_warmup: int = 2000,
) -> SBCPlan:
    family = getattr(specification, "family", None)
    if family == "binary":
        generator = simulate_binary_pathology
        args = {"scenario": "null_contrast"}
    elif family == "duration":
        generator = simulate_duration_pathology
        args = {"scenario": "null_ratio"}
    else:
        raise GP3BayesError("SBC plans require an approved binary or duration specification.")
    plan = SBCPlan(generator, backend, int(n_sims), args, int(seed), specification)
    plan.sampling = {
        "chains": chains,
        "iter": iter,
        "warmup": warmup,
        "thin": thin,
        "generator_iter": generator_iter,
        "generator_warmup": generator_warmup,
    }
    return plan


def run_sbc_plan(
    plan: SBCPlan,
    cores_per_fit: int = 1,
    keep_fits: bool = False,
    thin_ranks: int | None = None,
    cache_mode: str = "none",
    cache_location: str | None = None,
) -> SBCResult:
    if not isinstance(plan, SBCPlan):
        raise GP3BayesError("`plan` must be an SBCPlan.")
    rng = np.random.default_rng(plan.seed)
    sim_rows = []
    rank_rows = []
    retained = []  # type: ignore[var-annotated]
    for i in range(plan.n_sims):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        generated = plan.generator_function(seed=seed, **dict(plan.generator_args))
        data = generated.data if hasattr(generated, "data") else generated
        sim_rows.append(
            {"simulation": i + 1, "seed": seed, "rows": len(data), "status": "generated"}
        )
        # Custom plans may return truth/posterior pairs directly.
        if isinstance(generated, Mapping) and "truth" in generated and "draws" in generated:
            truth = generated["truth"]
            draws = generated["draws"]
            for parameter, value in truth.items():
                z = np.asarray(draws[parameter], float).reshape(-1)
                rank_rows.append(
                    {
                        "simulation": i + 1,
                        "parameter": parameter,
                        "rank": int(np.sum(z < float(value))),
                        "draws": len(z),
                    }
                )
    return SBCResult(plan, pd.DataFrame(sim_rows), pd.DataFrame(rank_rows), tuple(retained))


def summarise_sbc_result(x: SBCResult) -> dict[str, pd.DataFrame]:
    if not isinstance(x, SBCResult):
        raise GP3BayesError("`x` must be an SBCResult.")
    if x.ranks.empty:
        overview = pd.DataFrame(
            [{"simulations": len(x.simulations), "rank_records": 0, "calibration_assessed": False}]
        )
        return {"overview": overview, "parameters": pd.DataFrame()}
    rows = []
    for parameter, frame in x.ranks.groupby("parameter", sort=False):
        u = frame["rank"].to_numpy(float) / np.maximum(frame["draws"].to_numpy(float), 1)
        rows.append(
            {
                "parameter": parameter,
                "simulations": len(frame),
                "mean_rank_fraction": float(np.mean(u)),
                "rank_fraction_sd": float(np.std(u, ddof=1)) if len(u) > 1 else 0.0,
            }
        )
    return {
        "overview": pd.DataFrame(
            [
                {
                    "simulations": len(x.simulations),
                    "rank_records": len(x.ranks),
                    "calibration_assessed": True,
                }
            ]
        ),
        "parameters": pd.DataFrame(rows),
    }


def powerscale_sequence_for_fit(
    fit: Any,
    variable: str | None = None,
    prior_selection: Any = None,
    likelihood_selection: Any = None,
    component: str = "both",
) -> pd.DataFrame:
    if component not in {"both", "prior", "likelihood"}:
        raise GP3BayesError("`component` must be 'both', 'prior', or 'likelihood'.")
    alpha = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    pieces = ["prior", "likelihood"] if component == "both" else [component]
    return pd.DataFrame(
        [
            {
                "component": part,
                "alpha": float(a),
                "variable": variable,
                "automatic_decision": False,
            }
            for part in pieces
            for a in alpha
        ]
    )


def assess_powerscaled_sensitivity(
    fit: Any,
    variable: str | None = None,
    prior_selection: Any = None,
    likelihood_selection: Any = None,
) -> pd.DataFrame:
    sequence = powerscale_sequence_for_fit(fit, variable, prior_selection, likelihood_selection)
    # A backend-independent approximation reports the declared sequence and
    # marks execution as not assessed unless a specialized powerscaling engine
    # is installed. No robustness claim is made.
    sequence["distance"] = np.nan
    sequence["status"] = "not_assessed"
    sequence["robustness_established"] = False
    return sequence


__all__ = [
    "InteractionBackendSpecification",
    "InteractionPriorSpecification",
    "LOOComparison",
    "LOOWeights",
    "PSISLOOResult",
    "PathologicalSimulation",
    "SBCPlan",
    "SBCResult",
    "assess_powerscaled_sensitivity",
    "bayesian_backend_capabilities",
    "check_cmdstan_backend",
    "compare_psis_loo",
    "compute_loo_model_weights",
    "compute_psis_loo",
    "compute_psis_loo_from_log_lik",
    "create_brms_sbc_plan",
    "create_custom_sbc_plan",
    "detect_binary_separation",
    "evaluate_pathological_simulation",
    "fit_binary_model_backend",
    "fit_binary_model_cmdstanr",
    "fit_duration_model_backend",
    "fit_duration_model_cmdstanr",
    "identify_loo_influential_observations",
    "interaction_prior_summary",
    "powerscale_sequence_for_fit",
    "run_sbc_plan",
    "simulate_binary_pathology",
    "simulate_duration_pathology",
    "specify_binary_model_with_interaction_prior",
    "specify_duration_model_with_interaction_prior",
    "summarise_sbc_result",
    "translate_binary_model_with_interaction_prior",
    "translate_duration_model_with_interaction_prior",
]
