"""Advanced hierarchical-effect diagnostics.

These helpers preserve the frozen gp3bayes 0.5.0 contracts while adapting
brms group-level arrays to the restricted PyMC fit objects used by gp3bayespy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError
from .posterior import _posterior_components, _posterior_data_vars, _validate_fit_like


def _mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        raise GP3BayesError(
            "Matplotlib is required for plotting; install gp3bayespy[plots]."
        ) from exc
    return plt


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise GP3BayesError(f"`{name}` must be one integer >= {minimum}.")
    return int(value)


def _levels(fit: Any, mapping: str, n: int) -> list[str]:
    spec = getattr(fit, "specification", None)
    prepared = getattr(spec, "prepared", None)
    data = getattr(prepared, "data", None)
    contract = getattr(prepared, "contract", None)
    mappings = getattr(contract, "mappings", {}) if contract is not None else {}
    column = mappings.get(mapping) if hasattr(mappings, "get") else None
    if isinstance(data, pd.DataFrame) and column in data:
        levels = [str(v) for v in pd.unique(data[column])]
        if len(levels) == n:
            return levels
    return [str(i) for i in range(1, n + 1)]


def _group_arrays(fit: Any) -> dict[str, tuple[np.ndarray, list[str], list[str]]]:
    fit = _validate_fit_like(fit)
    variables = _posterior_data_vars(fit)
    out: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}
    if "participant_chol_stds" in variables and "participant_z" in variables:
        stds = np.asarray(variables["participant_chol_stds"].values, float)
        z = np.asarray(variables["participant_z"].values, float)
        # z: chain, draw, level, coefficient; stds: chain, draw, coefficient
        arr = z * stds[:, :, None, :]
        levels = _levels(fit, "participant", arr.shape[2])
        condition = getattr(getattr(fit.specification, "prepared", None), "contract", None)
        mappings = getattr(condition, "mappings", {}) if condition is not None else {}
        slope = str(mappings.get("condition", "condition"))
        out[str(mappings.get("participant", "participant"))] = (
            arr,
            levels,
            ["Intercept", slope],
        )
    elif "sd_participant" in variables and "participant_z" in variables:
        sd = np.asarray(variables["sd_participant"].values, float)
        z = np.asarray(variables["participant_z"].values, float)
        arr = z * sd[:, :, None]
        if arr.ndim == 3:
            arr = arr[:, :, :, None]
        prepared = getattr(fit.specification, "prepared", None)
        mappings = getattr(getattr(prepared, "contract", None), "mappings", {})
        name = str(mappings.get("participant", "participant"))
        out[name] = (arr, _levels(fit, "participant", arr.shape[2]), ["Intercept"])
    if "sd_item" in variables and "item_z" in variables:
        sd = np.asarray(variables["sd_item"].values, float)
        z = np.asarray(variables["item_z"].values, float)
        arr = (z * sd[:, :, None])[:, :, :, None]
        prepared = getattr(fit.specification, "prepared", None)
        mappings = getattr(getattr(prepared, "contract", None), "mappings", {})
        name = str(mappings.get("item", "item"))
        out[name] = (arr, _levels(fit, "item", arr.shape[2]), ["Intercept"])
    if not out:
        raise GP3BayesError("No group-level effects were returned.")
    return out


def group_effect_draws_table(
    fit: Any,
    groups: Sequence[str] | str | None = None,
    coefficients: Sequence[str] | str | None = None,
    ndraws: int | None = None,
    seed: int = 1,
    max_rows: int = 1_000_000,
) -> pd.DataFrame:
    arrays = _group_arrays(fit)
    requested_groups = (
        None if groups is None else ([groups] if isinstance(groups, str) else list(groups))
    )
    if requested_groups is not None:
        missing = sorted(set(requested_groups) - set(arrays))
        if missing:
            raise GP3BayesError("Unknown grouping factors: " + ", ".join(missing) + ".")
    requested_coef = (
        None
        if coefficients is None
        else ([coefficients] if isinstance(coefficients, str) else list(coefficients))
    )
    max_rows = _integer(max_rows, "max_rows")
    if ndraws is not None:
        ndraws = _integer(ndraws, "ndraws")
    if seed < 0:
        raise GP3BayesError("`seed` must be one non-negative integer.")
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    total = 0
    for group, (a4, levels, coefs) in arrays.items():
        if requested_groups is not None and group not in requested_groups:
            continue
        flat = a4.reshape(-1, a4.shape[2], a4.shape[3])
        ids = np.arange(flat.shape[0])
        if ndraws is not None and len(ids) > ndraws:
            ids = np.sort(rng.choice(ids, ndraws, replace=False))
        keep = coefs if requested_coef is None else [c for c in coefs if c in requested_coef]
        for coef in keep:
            ci = coefs.index(coef)
            total += len(ids) * len(levels)
            if total > max_rows:
                raise GP3BayesError(
                    "Requested draw table exceeds `max_rows`; reduce selectors/draws or increase `max_rows` explicitly."
                )
            values = flat[ids, :, ci]
            pieces.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "level": np.repeat(levels, len(ids)),
                        "coefficient": coef,
                        "draw": np.tile(ids + 1, len(levels)),
                        "value": values.T.reshape(-1),
                    }
                )
            )
    if not pieces:
        raise GP3BayesError("No requested group-effect draws were available.")
    return pd.concat(pieces, ignore_index=True)


def group_effect_rank_probability_table(
    fit: Any,
    group: str,
    coefficient: str = "Intercept",
    ndraws: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    d = group_effect_draws_table(
        fit, groups=group, coefficients=coefficient, ndraws=ndraws, seed=seed
    )
    matrix = d.pivot(index="draw", columns="level", values="value")
    ranks = matrix.rank(axis=1, ascending=False, method="average")
    nlevels = matrix.shape[1]
    rows = []
    for level in matrix.columns:
        z = matrix[level]
        r = ranks[level]
        rows.append(
            {
                "group": group,
                "coefficient": coefficient,
                "level": level,
                "posterior_mean": float(z.mean()),
                "posterior_sd": float(z.std(ddof=1)),
                "mean_rank": float(r.mean()),
                "median_rank": float(r.median()),
                "probability_highest": float((r == 1).mean()),
                "probability_lowest": float((r == nlevels).mean()),
                "probability_positive": float((z > 0).mean()),
                "automatic_ranking_decision": False,
            }
        )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class RandomInterceptVariancePartition:
    family: str
    table: pd.DataFrame
    probs: tuple[float, float, float]
    random_slope_variance_included: bool = False
    automatic_importance_decision: bool = False
    interpretation: str = (
        "Fractions describe baseline latent variance partitioning only. "
        "They are not causal variance attributions or automatic rankings."
    )


def _summary(z: np.ndarray, probs: tuple[float, float, float]) -> dict[str, float]:
    q = np.quantile(z, probs, method="linear")
    return {
        "mean": float(np.mean(z)),
        "lower": float(q[0]),
        "median": float(q[1]),
        "upper": float(q[2]),
    }


def random_intercept_variance_partition(
    fit: Any, probs: Sequence[float] = (0.025, 0.5, 0.975)
) -> RandomInterceptVariancePartition:
    fit = _validate_fit_like(fit)
    p = tuple(float(v) for v in probs)
    if len(p) != 3 or any(v <= 0 or v >= 1 for v in p) or not p[0] < p[1] < p[2]:
        raise GP3BayesError(
            "`probs` must contain three increasing probabilities strictly inside (0, 1)."
        )
    components = _posterior_components(fit)
    sd_names = [n for n in components if n.startswith("sd_") and n.endswith("__Intercept")]
    if not sd_names:
        raise GP3BayesError("No random-intercept SD draws were found.")
    group_var = np.column_stack([components[n].reshape(-1) ** 2 for n in sd_names])
    names = [n.removeprefix("sd_").removesuffix("__Intercept") for n in sd_names]
    if fit.family == "binary":
        residual = np.full(group_var.shape[0], np.pi**2 / 3)
        residual_name = "logit_residual"
    else:
        if "sigma" not in components:
            raise GP3BayesError("Duration variance partition requires posterior `sigma`.")
        residual = components["sigma"].reshape(-1) ** 2
        residual_name = "lognormal_residual"
    total = group_var.sum(axis=1) + residual
    rows = []
    for idx, name in enumerate(names):
        v = _summary(group_var[:, idx], p)
        f = _summary(group_var[:, idx] / total, p)
        rows.append(
            {
                "component": name,
                "component_type": "random_intercept",
                **{f"variance_{k}": val for k, val in v.items()},
                **{f"fraction_{k}": val for k, val in f.items()},
            }
        )
    v = _summary(residual, p)
    f = _summary(residual / total, p)
    rows.append(
        {
            "component": residual_name,
            "component_type": "residual",
            **{f"variance_{k}": val for k, val in v.items()},
            **{f"fraction_{k}": val for k, val in f.items()},
        }
    )
    return RandomInterceptVariancePartition(fit.family, pd.DataFrame(rows), p)


def random_intercept_variance_partition_table(x: RandomInterceptVariancePartition) -> pd.DataFrame:
    if not isinstance(x, RandomInterceptVariancePartition):
        raise GP3BayesError("`x` must be a gp3bayes random-intercept variance partition.")
    return x.table.copy()


def plot_group_effect_distribution(x: pd.DataFrame, max_levels: int = 20):
    max_levels = _integer(max_levels, "max_levels")
    required = {"group", "level", "coefficient", "value"}
    if not isinstance(x, pd.DataFrame) or not required <= set(x):
        raise GP3BayesError("`x` must be a group-effect draw table.")
    plt = _mpl()
    importance = (
        x.groupby("level", observed=False)["value"]
        .apply(lambda z: np.mean(np.abs(z)))
        .sort_values(ascending=False)
    )
    d = x[x["level"].isin(importance.head(max_levels).index)]
    fig, ax = plt.subplots()
    for level, frame in d.groupby("level", observed=False):
        ax.hist(frame["value"], bins=30, density=True, histtype="step", label=str(level))
    ax.set_xlabel("Group-level deviation")
    ax.set_ylabel("Density")
    ax.set_title("Group-level posterior distributions")
    if d["level"].nunique() <= 12:
        ax.legend()
    return fig


def plot_group_effect_rank_probability(x: pd.DataFrame):
    required = {"level", "mean_rank", "probability_highest"}
    if not isinstance(x, pd.DataFrame) or not required <= set(x):
        raise GP3BayesError("`x` must be a group-effect rank-probability table.")
    plt = _mpl()
    d = x.sort_values("mean_rank", ascending=False)
    fig, ax = plt.subplots()
    ax.barh(d["level"].astype(str), d["probability_highest"])
    ax.set_xlabel("Posterior probability of largest deviation")
    ax.set_ylabel("Grouping level")
    ax.set_title("Group-level rank uncertainty")
    return fig


def plot_random_intercept_variance_partition(x: RandomInterceptVariancePartition | pd.DataFrame):
    d = x.table if isinstance(x, RandomInterceptVariancePartition) else x
    required = {"component", "fraction_median", "fraction_lower", "fraction_upper"}
    if not isinstance(d, pd.DataFrame) or not required <= set(d):
        raise GP3BayesError("`x` does not contain variance-partition summaries.")
    plt = _mpl()
    fig, ax = plt.subplots()
    y = np.arange(len(d))
    med = d["fraction_median"].to_numpy(float)
    lower = med - d["fraction_lower"].to_numpy(float)
    upper = d["fraction_upper"].to_numpy(float) - med
    ax.errorbar(med, y, xerr=np.vstack([lower, upper]), fmt="o")
    ax.set_yticks(y, d["component"].astype(str))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Latent variance fraction")
    ax.set_title("Baseline random-intercept variance partition")
    return fig


__all__ = [
    "group_effect_draws_table",
    "group_effect_rank_probability_table",
    "random_intercept_variance_partition",
    "random_intercept_variance_partition_table",
    "plot_group_effect_distribution",
    "plot_group_effect_rank_probability",
    "plot_random_intercept_variance_partition",
    "RandomInterceptVariancePartition",
]
