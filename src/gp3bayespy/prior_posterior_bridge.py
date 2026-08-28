"""Declared-prior versus fitted-posterior bridges."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .exceptions import GP3BayesError
from .postfit_exploration import extract_posterior_draws
from .specification import PriorSpecification


def _mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        raise GP3BayesError(
            "Matplotlib is required for plotting; install gp3bayespy[plots]."
        ) from exc
    return plt


def _prior_object(x: Any) -> PriorSpecification:
    if isinstance(x, PriorSpecification):
        return x
    priors = getattr(getattr(x, "specification", x), "priors", None)
    if isinstance(priors, PriorSpecification):
        return priors
    raise GP3BayesError("`x` must be a fit, model specification, or PriorSpecification.")


def _class_for_variable(variable: str) -> str | None:
    if variable in {"b_Intercept", "Intercept"}:
        return "Intercept"
    if variable.startswith("b_"):
        return "b"
    if variable.startswith("sd_"):
        return "sd"
    if variable.startswith("cor_"):
        return "cor"
    if variable == "sigma":
        return "sigma"
    return None


def prior_specification_table(x: Any) -> pd.DataFrame:
    return _prior_object(x).table.copy()


def _sample_prior(row: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    distribution = str(row["distribution"])
    location = float(row["location"]) if pd.notna(row["location"]) else 0.0
    scale = float(row["scale"]) if pd.notna(row["scale"]) else 1.0
    if distribution == "normal":
        return rng.normal(location, scale, n)
    if distribution == "student_t":
        df = float(row["df"])
        z = location + scale * rng.standard_t(df, n)
        lower = float(row["lower"]) if pd.notna(row["lower"]) else -math.inf
        if np.isfinite(lower) and lower >= 0:
            z = np.abs(z)
        return z
    if distribution == "lkj":
        eta = float(row["shape"])
        return 2 * rng.beta(eta, eta, n) - 1
    raise GP3BayesError(f"Unsupported declared prior distribution {distribution!r}.")


def simulate_declared_prior_draws(
    x: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    ndraws: int = 4000,
    seed: int = 1,
) -> pd.DataFrame:
    if ndraws < 50 or seed < 0:
        raise GP3BayesError("`ndraws` must be >= 50 and `seed` non-negative.")
    priors = _prior_object(x)
    if variables is None:
        if not getattr(x, "fit_performed", False):
            raise GP3BayesError("`variables` is required when `x` is not a fitted model.")
        posterior = extract_posterior_draws(x, format="matrix")
        variables = list(getattr(posterior, "columns", []))
        if not variables and isinstance(posterior, np.ndarray):
            # Canonical extractor returns a DataFrame-like matrix in current port.
            raise GP3BayesError("Posterior variable names are required to infer declared priors.")
        variables = [v for v in variables if _class_for_variable(str(v)) is not None]
    requested = [variables] if isinstance(variables, str) else list(variables)
    if regex is not None:
        import re

        pattern = re.compile(regex)
        requested = [v for v in requested if pattern.search(v)]
    if not requested:
        raise GP3BayesError("No supported prior variables remain.")
    rng = np.random.default_rng(seed)
    table = priors.table
    out: dict[str, np.ndarray] = {}
    for variable in dict.fromkeys(requested):
        cls = _class_for_variable(variable)
        if cls is None:
            raise GP3BayesError(f"Unsupported variable: {variable}.")
        rows = table.loc[table["parameter_class"] == cls]
        if len(rows) != 1:
            raise GP3BayesError(f"No unique declared prior for class {cls!r}.")
        out[str(variable)] = _sample_prior(rows.iloc[0], ndraws, rng)
    return pd.DataFrame(out)


@dataclass(slots=True)
class PriorPosteriorBridge:
    bridge_version: str
    family: str
    variables: tuple[str, ...]
    prior_draws: pd.DataFrame
    posterior_draws: pd.DataFrame
    summary: pd.DataFrame
    distances: pd.DataFrame
    probs: tuple[float, float, float]
    seed: int
    marginal_prior_simulation: bool = True
    backend_saved_prior_draws_required: bool = False
    automatic_prior_decision: bool = False
    interpretation: str = (
        "Shift, contraction, interval overlap, and distribution distances are descriptive "
        "marginal comparisons and do not establish prior adequacy."
    )

    def to_frame(self) -> pd.DataFrame:
        return self.summary.copy()


def _summary(z: np.ndarray, probs: tuple[float, float, float]) -> dict[str, float]:
    q = np.quantile(z, probs, method="linear")
    return {
        "mean": float(np.mean(z)),
        "sd": float(np.std(z, ddof=1)),
        "lower": float(q[0]),
        "median": float(q[1]),
        "upper": float(q[2]),
    }


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return inter / union if np.isfinite(union) and union > 0 else np.nan


def prior_posterior_bridge(
    fit: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = r"^(b_|sd_|cor_|sigma$)",
    ndraws: int = 4000,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
) -> PriorPosteriorBridge:
    p = tuple(float(v) for v in probs)
    if len(p) != 3 or not p[0] < p[1] < p[2]:
        raise GP3BayesError("`probs` must contain three increasing probabilities.")
    posterior = extract_posterior_draws(fit, variables=variables, regex=regex, format="df")
    if not isinstance(posterior, pd.DataFrame):
        posterior = pd.DataFrame(posterior)
    posterior = posterior.drop(
        columns=[c for c in (".chain", ".iteration", ".draw", "chain", "draw") if c in posterior],
        errors="ignore",
    )
    supported = [c for c in posterior.columns if _class_for_variable(str(c)) is not None]
    posterior = posterior.loc[:, supported]
    if posterior.empty:
        raise GP3BayesError("No supported posterior variables remain.")
    if len(posterior) > ndraws:
        posterior = (
            posterior.sample(n=ndraws, random_state=seed + 1).sort_index().reset_index(drop=True)
        )
    prior = simulate_declared_prior_draws(fit, variables=supported, ndraws=ndraws, seed=seed)
    summary_rows = []
    distance_rows = []
    for variable in supported:
        a = prior[variable].to_numpy(float)
        b = posterior[variable].to_numpy(float)
        pa = _summary(a, p)
        pb = _summary(b, p)
        ratio = pb["sd"] / pa["sd"] if pa["sd"] > 0 else np.nan
        summary_rows.append(
            {
                "variable": variable,
                "parameter_class": _class_for_variable(variable),
                "prior_mean": pa["mean"],
                "prior_sd": pa["sd"],
                "prior_lower": pa["lower"],
                "prior_median": pa["median"],
                "prior_upper": pa["upper"],
                "posterior_mean": pb["mean"],
                "posterior_sd": pb["sd"],
                "posterior_lower": pb["lower"],
                "posterior_median": pb["median"],
                "posterior_upper": pb["upper"],
                "median_shift": pb["median"] - pa["median"],
                "standardized_location_shift": (pb["median"] - pa["median"]) / pa["sd"]
                if pa["sd"] > 0
                else np.nan,
                "sd_ratio": ratio,
                "contraction": 1 - ratio if np.isfinite(ratio) else np.nan,
                "interval_overlap_fraction": _overlap(
                    pa["lower"], pa["upper"], pb["lower"], pb["upper"]
                ),
            }
        )
        probs_w = np.linspace(0.001, 0.999, 999)
        w = float(np.mean(np.abs(np.quantile(a, probs_w) - np.quantile(b, probs_w))))
        distance_rows.append(
            {
                "variable": variable,
                "ks_distance": float(ks_2samp(a, b, method="auto").statistic),
                "quantile_wasserstein": w,
                "standardized_quantile_wasserstein": w / np.std(a, ddof=1)
                if np.std(a, ddof=1) > 0
                else np.nan,
            }
        )
    return PriorPosteriorBridge(
        "0.3",
        str(getattr(fit, "family", "unknown")),
        tuple(supported),
        prior,
        posterior,
        pd.DataFrame(summary_rows),
        pd.DataFrame(distance_rows),
        p,  # type: ignore[arg-type]
        int(seed),
    )


def prior_posterior_summary_table(x: PriorPosteriorBridge) -> pd.DataFrame:
    if not isinstance(x, PriorPosteriorBridge):
        raise GP3BayesError("`x` must be a PriorPosteriorBridge.")
    return x.summary.copy()


def prior_posterior_distance_table(x: PriorPosteriorBridge) -> pd.DataFrame:
    if not isinstance(x, PriorPosteriorBridge):
        raise GP3BayesError("`x` must be a PriorPosteriorBridge.")
    return x.distances.copy()


def prior_posterior_draws_long(
    x: PriorPosteriorBridge,
    max_draws: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    if not isinstance(x, PriorPosteriorBridge) or max_draws < 50:
        raise GP3BayesError("A PriorPosteriorBridge and `max_draws >= 50` are required.")
    pieces = []
    for offset, (label, frame) in enumerate(
        (("prior", x.prior_draws), ("posterior", x.posterior_draws))
    ):
        sample = frame
        if len(sample) > max_draws:
            sample = sample.sample(max_draws, random_state=seed + offset).sort_index()
        long = sample.reset_index(drop=True).melt(var_name="variable", value_name="value")
        long.insert(0, "draw", np.tile(np.arange(1, len(sample) + 1), sample.shape[1]))
        long.insert(2, "distribution", label)
        pieces.append(long)
    return pd.concat(pieces, ignore_index=True)


def plot_prior_posterior_density(x: PriorPosteriorBridge, max_draws: int = 1000):
    d = prior_posterior_draws_long(x, max_draws=max_draws)
    plt = _mpl()
    variables = list(dict.fromkeys(d["variable"]))
    fig, axes = plt.subplots(len(variables), 1, squeeze=False, figsize=(7, 3 * len(variables)))
    for ax, variable in zip(axes[:, 0], variables, strict=True):
        for label, frame in d[d["variable"] == variable].groupby("distribution", sort=False):
            values = frame["value"].to_numpy(float)
            ax.hist(values, bins=40, density=True, histtype="step", label=str(label))
        ax.set_title(str(variable))
        ax.legend()
    fig.tight_layout()
    return fig


def plot_prior_posterior_intervals(x: PriorPosteriorBridge):
    d = x.summary
    plt = _mpl()
    fig, ax = plt.subplots()
    y = np.arange(len(d))
    ax.hlines(y - 0.12, d["prior_lower"], d["prior_upper"])
    ax.scatter(d["prior_median"], y - 0.12, label="Prior")
    ax.hlines(y + 0.12, d["posterior_lower"], d["posterior_upper"])
    ax.scatter(d["posterior_median"], y + 0.12, label="Posterior")
    ax.set_yticks(y, d["variable"])
    ax.legend()
    ax.set_title("Declared prior vs posterior intervals")
    return fig


def plot_prior_posterior_shift(x: PriorPosteriorBridge):
    d = x.summary
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.barh(d["variable"], d["standardized_location_shift"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Standardized median shift")
    return fig


def plot_prior_posterior_contraction(x: PriorPosteriorBridge):
    d = x.summary
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.barh(d["variable"], d["contraction"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("1 - posterior SD / prior SD")
    return fig


__all__ = [
    "PriorPosteriorBridge",
    "plot_prior_posterior_contraction",
    "plot_prior_posterior_density",
    "plot_prior_posterior_intervals",
    "plot_prior_posterior_shift",
    "prior_posterior_bridge",
    "prior_posterior_distance_table",
    "prior_posterior_draws_long",
    "prior_posterior_summary_table",
    "prior_specification_table",
    "simulate_declared_prior_draws",
]
