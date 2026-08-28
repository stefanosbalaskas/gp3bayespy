"""Evidence-table and publication-graphics adapters.

Matplotlib adaptation of gp3bayes 0.5.0 ``evidence-graphics-gg.R``.  These
helpers visualize existing evidence objects; they never recompute analyses or
convert descriptive statuses into automatic decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError


def _field(x: Any, *names: str) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    for name in names:
        value = getattr(x, name, None)
        if value is None and isinstance(x, dict):
            value = x.get(name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    raise GP3BayesError("The evidence object does not expose the requested table.")


def sensitivity_suite_table(x: Any) -> pd.DataFrame:
    try:
        from .sensitivity import summarise_sensitivity_suite

        return summarise_sensitivity_suite(x)
    except Exception:
        return _field(x, "component_status", "table")


def model_evidence_table(x: Any) -> pd.DataFrame:
    return _field(x, "component_table", "table")


def backend_parity_table(x: Any) -> pd.DataFrame:
    return _field(x, "table")


def backend_environment_table(x: Any) -> pd.DataFrame:
    return _field(x, "table")


def manifest_comparison_table(x: Any) -> pd.DataFrame:
    return _field(x, "table")


def schema_comparison_table(x: Any) -> pd.DataFrame:
    return _field(x, "table")


def design_support_table(x: Any) -> pd.DataFrame:
    return _field(x, "table", "checks")


def missingness_audit_table(x: Any) -> pd.DataFrame:
    return _field(x, "table", "missingness", "checks")


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GP3BayesError("Matplotlib is required for evidence graphics.") from exc
    return plt


def _status_plot(
    d: pd.DataFrame, title: str, label_col: str | None = None, status_col: str = "status"
):
    plt = _plt()
    fig, ax = plt.subplots()
    ax.set_title(title)
    if label_col and label_col in d.columns and status_col in d.columns:
        categories = list(dict.fromkeys(d[status_col].astype(str)))
        codes = {name: i for i, name in enumerate(categories)}
        ax.barh(d[label_col].astype(str), [codes[v] for v in d[status_col].astype(str)])
        ax.set_xticks(list(codes.values()), list(codes.keys()))
    elif status_col in d.columns:
        counts = d[status_col].astype(str).value_counts()
        ax.bar(counts.index, counts.values)
    else:
        numeric = d.select_dtypes(include=[np.number])
        if numeric.empty:
            ax.text(0.5, 0.5, "Evidence recorded", ha="center")
        else:
            ax.bar(range(len(numeric.columns)), numeric.mean().to_numpy(float))
            ax.set_xticks(range(len(numeric.columns)), numeric.columns, rotation=90)
    return fig


def plot_sensitivity_suite_gg(x: Any):
    return _status_plot(sensitivity_suite_table(x), "Sensitivity suite", "component")


def plot_model_evidence_gg(x: Any):
    return _status_plot(model_evidence_table(x), "Model evidence inventory", "component")


def plot_backend_parity_gg(x: Any):
    d = backend_parity_table(x)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.set_title("Backend posterior parity")
    pairs = [
        (a, b)
        for a, b in (
            ("rstan_mean", "cmdstanr_mean"),
            ("reference_mean", "alternative_mean"),
            ("left_mean", "right_mean"),
        )
        if a in d.columns and b in d.columns
    ]
    if pairs:
        a, b = pairs[0]
        ax.scatter(d[a], d[b])
        lo = min(float(d[a].min()), float(d[b].min()))
        hi = max(float(d[a].max()), float(d[b].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
    else:
        return _status_plot(
            d, "Backend posterior parity", "variable" if "variable" in d.columns else None
        )
    return fig


def plot_backend_environment_gg(x: Any):
    return _status_plot(
        backend_environment_table(x),
        "Backend environment",
        "component" if "component" in backend_environment_table(x).columns else None,
    )


def plot_manifest_comparison_gg(x: Any):
    d = manifest_comparison_table(x)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.set_title("Analysis-manifest comparison")
    ax.barh(d["component"].astype(str), d["identical"].astype(int))
    ax.set_xlim(0, 1)
    return fig


def plot_schema_comparison_gg(x: Any):
    return _status_plot(
        schema_comparison_table(x),
        "Object-schema comparison",
        "component" if "component" in schema_comparison_table(x).columns else None,
    )


def plot_design_support_gg(x: Any):
    return _status_plot(
        design_support_table(x),
        "Design-support diagnostics",
        "check" if "check" in design_support_table(x).columns else None,
    )


def plot_missingness_gg(x: Any):
    d = missingness_audit_table(x)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.set_title("Missingness audit")
    label = next((c for c in ("variable", "column", "component") if c in d.columns), None)
    value = next(
        (c for c in ("missing_fraction", "fraction", "n_missing", "missing") if c in d.columns),
        None,
    )
    if label and value:
        ax.barh(d[label].astype(str), pd.to_numeric(d[value], errors="coerce").fillna(0))
    else:
        ax.text(0.5, 0.5, "Missingness evidence recorded", ha="center")
    return fig


__all__ = [
    "backend_environment_table",
    "backend_parity_table",
    "design_support_table",
    "manifest_comparison_table",
    "missingness_audit_table",
    "model_evidence_table",
    "plot_backend_environment_gg",
    "plot_backend_parity_gg",
    "plot_design_support_gg",
    "plot_manifest_comparison_gg",
    "plot_missingness_gg",
    "plot_model_evidence_gg",
    "plot_schema_comparison_gg",
    "plot_sensitivity_suite_gg",
    "schema_comparison_table",
    "sensitivity_suite_table",
]
