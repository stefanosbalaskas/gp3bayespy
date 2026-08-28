"""Pointwise and grouped PSIS-LOO influence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError


def _mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        raise GP3BayesError(
            "Matplotlib is required for plotting; install gp3bayespy[plots]."
        ) from exc
    return plt


def _table(x: Any, data: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(x, LOOInfluenceAtlas):
        return x.table.copy()
    if isinstance(x, pd.DataFrame) and "pareto_k" in x:
        return x.copy()
    return loo_pointwise_table(x, data=data)


def loo_pointwise_table(x: Any, data: pd.DataFrame | None = None) -> pd.DataFrame:
    pointwise = getattr(x, "pointwise", None)
    if pointwise is None and hasattr(x, "raw"):
        pointwise = getattr(x.raw, "pointwise", None)
    if pointwise is None:
        raise GP3BayesError("`x` must contain pointwise LOO estimates.")
    if isinstance(pointwise, pd.DataFrame):
        out = pointwise.copy().reset_index(drop=True)
    else:
        arr = np.asarray(pointwise, float)
        if arr.ndim == 1:
            out = pd.DataFrame({"elpd_loo": arr})
        elif arr.ndim == 2:
            names = ["elpd_loo", "mcse_elpd_loo", "p_loo", "looic"][: arr.shape[1]]
            out = pd.DataFrame(arr, columns=names)
        else:
            raise GP3BayesError("Pointwise LOO estimates have an unsupported shape.")
    k = getattr(x, "pareto_k", None)
    if k is None and "pareto_k" in out:
        k = out["pareto_k"].to_numpy(float)
    if k is None:
        raise GP3BayesError("Pareto-k diagnostics are unavailable.")
    k = np.asarray(k, float).reshape(-1)
    if len(k) != len(out):
        raise GP3BayesError("LOO pointwise and Pareto-k diagnostics differ in length.")
    influence = getattr(x, "influence_pareto_k", None)
    influence_arr = np.asarray(influence if influence is not None else k, float).reshape(-1)
    result = pd.concat(
        [
            pd.DataFrame({"observation": np.arange(1, len(out) + 1)}),
            out.drop(columns=["pareto_k"], errors="ignore"),
        ],
        axis=1,
    )
    result["pareto_k"] = k
    result["influence_pareto_k"] = influence_arr
    result["flagged"] = ~np.isfinite(k) | (k >= 0.7)
    result["severe"] = np.isfinite(k) & (k >= 1.0)
    if data is not None:
        if not isinstance(data, pd.DataFrame) or len(data) != len(result):
            raise GP3BayesError("`data` must have one row per LOO observation.")
        result = pd.concat([result.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
    return result


def loo_influence_summary(x: Any) -> pd.DataFrame:
    d = _table(x)
    finite = d.loc[np.isfinite(d["pareto_k"]), "pareto_k"].to_numpy(float)
    q = (
        np.quantile(finite, [0.5, 0.9, 0.95, 0.99], method="linear")
        if len(finite)
        else np.full(4, np.nan)
    )
    return pd.DataFrame(
        [
            {
                "observations": len(d),
                "finite_pareto_k": len(finite),
                "median_pareto_k": q[0],
                "p90_pareto_k": q[1],
                "p95_pareto_k": q[2],
                "p99_pareto_k": q[3],
                "flagged_k_ge_0_7": int(
                    np.sum(np.isfinite(d["pareto_k"]) & (d["pareto_k"] >= 0.7))
                ),
                "severe_k_ge_1": int(np.sum(np.isfinite(d["pareto_k"]) & (d["pareto_k"] >= 1.0))),
                "automatic_exclusion": False,
            }
        ]
    )


def loo_flagged_data(x: Any, threshold: float = 0.7) -> pd.DataFrame:
    if not np.isfinite(threshold):
        raise GP3BayesError("`threshold` must be one finite number.")
    d = _table(x)
    return d.loc[np.isfinite(d["pareto_k"]) & (d["pareto_k"] >= threshold)].copy()


def loo_group_influence_table(
    x: Any,
    group: str,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    d = _table(x, data)
    if not isinstance(group, str) or group not in d:
        raise GP3BayesError("`group` must name one pointwise-table or supplied-data column.")
    rows = []
    for value, frame in d.groupby(group, observed=False, dropna=False, sort=False):
        k = pd.to_numeric(frame["pareto_k"], errors="coerce").to_numpy(float)
        elpd = pd.to_numeric(
            frame.get("elpd_loo", pd.Series(np.nan, index=frame.index)), errors="coerce"
        ).to_numpy(float)
        rows.append(
            {
                "group": group,
                "group_value": str(value),
                "observations": len(frame),
                "mean_pareto_k": float(np.nanmean(k)) if np.isfinite(k).any() else np.nan,
                "max_pareto_k": float(np.nanmax(k)) if np.isfinite(k).any() else np.nan,
                "flagged_k_ge_0_7": int(np.sum(np.isfinite(k) & (k >= 0.7))),
                "severe_k_ge_1": int(np.sum(np.isfinite(k) & (k >= 1.0))),
                "total_elpd_loo": float(np.nansum(elpd)) if np.isfinite(elpd).any() else np.nan,
                "mean_elpd_loo": float(np.nanmean(elpd)) if np.isfinite(elpd).any() else np.nan,
                "automatic_group_exclusion": False,
            }
        )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class LOOInfluenceAtlas:
    atlas_version: str
    threshold: float
    table: pd.DataFrame
    flagged: pd.DataFrame
    summary: pd.DataFrame
    automatic_exclusion: bool = False
    interpretation: str = (
        "Pointwise predictive contributions and PSIS influence are reported together. "
        "Flagged rows request review and are not excluded automatically."
    )

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()


def create_loo_influence_atlas(
    x: Any,
    data: pd.DataFrame | None = None,
    threshold: float = 0.7,
) -> LOOInfluenceAtlas:
    table = loo_pointwise_table(x, data=data)
    return LOOInfluenceAtlas(
        "0.3",
        float(threshold),
        table,
        loo_flagged_data(table, threshold),
        loo_influence_summary(table),
    )


def loo_influence_atlas_table(x: LOOInfluenceAtlas) -> pd.DataFrame:
    if not isinstance(x, LOOInfluenceAtlas):
        raise GP3BayesError("`x` must be a gp3bayespy LOO influence atlas.")
    return x.table.copy()


def plot_loo_pointwise_elpd(x: Any):
    d = _table(x)
    if "elpd_loo" not in d:
        raise GP3BayesError("`elpd_loo` is unavailable.")
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.scatter(d["observation"], d["elpd_loo"])
    ax.set(
        xlabel="Observation",
        ylabel="Pointwise ELPD-LOO",
        title="Pointwise leave-one-out predictive contribution",
    )
    return fig


def plot_loo_pareto_vs_elpd(x: Any):
    d = _table(x)
    if "elpd_loo" not in d:
        raise GP3BayesError("`elpd_loo` is unavailable.")
    plt = _mpl()
    fig, ax = plt.subplots()
    for line in (0.5, 0.7, 1.0):
        ax.axvline(line, linestyle="--")
    ax.scatter(d["pareto_k"], d["elpd_loo"])
    ax.set(
        xlabel="Pareto k",
        ylabel="Pointwise ELPD-LOO",
        title="LOO influence and predictive contribution",
    )
    return fig


def plot_loo_influence_rank(x: Any):
    d = (
        _table(x)
        .sort_values("pareto_k", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    plt = _mpl()
    fig, ax = plt.subplots()
    for line in (0.5, 0.7, 1.0):
        ax.axhline(line, linestyle="--")
    ax.scatter(np.arange(1, len(d) + 1), d["pareto_k"])
    ax.set(xlabel="Influence rank", ylabel="Pareto k", title="Ranked PSIS-LOO influence")
    return fig


def plot_loo_group_influence(x: pd.DataFrame):
    if not isinstance(x, pd.DataFrame) or not {"group_value", "max_pareto_k"}.issubset(x):
        raise GP3BayesError("`x` must be a grouped LOO influence table.")
    d = x.sort_values("max_pareto_k")
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.barh(d["group_value"].astype(str), d["max_pareto_k"])
    for line in (0.5, 0.7, 1.0):
        ax.axvline(line, linestyle="--")
    ax.set(xlabel="Maximum Pareto k", ylabel="Group", title="Grouped PSIS-LOO influence")
    return fig


def plot_loo_group_elpd(x: pd.DataFrame):
    if not isinstance(x, pd.DataFrame) or not {"group_value", "total_elpd_loo"}.issubset(x):
        raise GP3BayesError("`x` must be a grouped LOO influence table.")
    if not np.isfinite(pd.to_numeric(x["total_elpd_loo"], errors="coerce")).any():
        raise GP3BayesError("Pointwise `elpd_loo` is unavailable.")
    d = x.sort_values("total_elpd_loo")
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.barh(d["group_value"].astype(str), d["total_elpd_loo"])
    ax.set(
        xlabel="Sum of pointwise ELPD-LOO", ylabel="Group", title="Grouped predictive contribution"
    )
    return fig


__all__ = [
    "LOOInfluenceAtlas",
    "create_loo_influence_atlas",
    "loo_flagged_data",
    "loo_group_influence_table",
    "loo_influence_atlas_table",
    "loo_influence_summary",
    "loo_pointwise_table",
    "plot_loo_group_elpd",
    "plot_loo_group_influence",
    "plot_loo_influence_rank",
    "plot_loo_pareto_vs_elpd",
    "plot_loo_pointwise_elpd",
]
