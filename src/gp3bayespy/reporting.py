"""Publication graphics, model cards, registries, and diagnostic dashboards.

This module is the Matplotlib adaptation of gp3bayes 0.5.0 reporting helpers.
All graphics are descriptive.  Registries, model cards, and dashboards never
write files or launch analyses implicitly and never issue automatic adequacy,
causal-identification, exclusion, or model-selection decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise GP3BayesError("Matplotlib is required for publication graphics.") from exc
    return plt


def theme_gp3bayes(base_size: float = 11, base_family: str = "") -> dict[str, Any]:
    """Return inspectable Matplotlib theme metadata used by gp3bayespy plots."""
    if not np.isfinite(float(base_size)) or float(base_size) <= 0:
        raise GP3BayesError("`base_size` must be strictly positive.")
    return {
        "base_size": float(base_size),
        "base_family": str(base_family),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def _figure(title: str = ""):
    plt = _plt()
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    return fig, ax


def _df(x: Any, *fields: str) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    for field_name in fields:
        value = getattr(x, field_name, None)
        if value is None and isinstance(x, Mapping):
            value = x.get(field_name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    raise GP3BayesError("The supplied object does not expose the required table.")


def _status(x: Any) -> str:
    if x is None:
        return "not_available"
    value = getattr(x, "status", None)
    if value is None and isinstance(x, Mapping):
        value = x.get("status")
    return "available" if value is None else str(value)


def _posterior_frame(
    x: Any, variables: Sequence[str] | None = None, regex: str | None = None
) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        frame = x.copy()
        if "variable" in frame.columns and {"lower", "median", "upper"}.issubset(frame.columns):
            return frame
    arr = np.asarray(x) if isinstance(x, (np.ndarray, list, tuple)) else None
    if arr is not None and arr.ndim in {1, 2}:
        if arr.ndim == 1:
            arr = arr[:, None]
        names = [f"V{i + 1}" for i in range(arr.shape[1])]
        if hasattr(x, "columns"):
            names = [str(v) for v in x.columns]
        rows = []
        for j, name in enumerate(names):
            z = arr[:, j].astype(float)
            rows.append(
                {
                    "variable": name,
                    "lower": float(np.quantile(z, 0.025)),
                    "median": float(np.median(z)),
                    "upper": float(np.quantile(z, 0.975)),
                    "mean": float(np.mean(z)),
                    "sd": float(np.std(z, ddof=1)),
                }
            )
        return pd.DataFrame(rows)
    from .postfit_exploration import posterior_interval_table

    return posterior_interval_table(x, variables=variables, regex=regex)


@dataclass(slots=True)
class FigureSet:
    title: str
    figures: Mapping[str, Any]
    names: tuple[str, ...]
    automatic_writing: bool = False


@dataclass(slots=True)
class PublicationRegistry:
    registry_version: str = "0.3"
    label: str | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    entries: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["name", "type", "caption", "source"])
    )
    automatic_writing: bool = False


@dataclass(slots=True)
class RegistryValidation:
    valid: bool
    checks: pd.DataFrame
    automatic_writing: bool = False


@dataclass(slots=True)
class EvidenceInventory:
    inventory_version: str
    label: str | None
    objects: Mapping[str, Any]
    table: pd.DataFrame
    automatic_decision: bool = False


@dataclass(slots=True)
class DiagnosticDashboard:
    dashboard_version: str
    label: str | None
    objects: Mapping[str, Any]
    table: pd.DataFrame
    automatic_decision: bool = False
    interpretation: str = "The dashboard indexes supplied evidence. It does not launch expensive analyses or issue an automatic adequacy verdict."


@dataclass(slots=True)
class ModelCard:
    card_version: str
    label: str | None
    family: str
    model_family: str | None
    formula: str
    sampling_backend: str | None
    sampling: Mapping[str, Any]
    package_versions: Mapping[str, str]
    diagnosis: Any
    workflow: Any
    analysis_bundle: Any
    manifest: Any
    evidence: pd.DataFrame
    model_adequacy_certified: bool = False
    causal_identification_certified: bool = False
    automatic_model_selection: bool = False
    interpretation: str = "This model card records analysis identity and available evidence. It does not certify model adequacy, causal identification, or substantive validity."


def create_figure_set(
    figures: Mapping[str, Any],
    title: str = "gp3bayes figure set",
) -> FigureSet:
    if not isinstance(figures, Mapping) or not figures:
        raise GP3BayesError("Supply a non-empty mapping of named figures.")
    if any(not name for name in figures):
        raise GP3BayesError("Figure names must be non-empty.")
    if any(not hasattr(fig, "savefig") for fig in figures.values()):
        raise GP3BayesError("Every entry must be a Matplotlib Figure.")
    return FigureSet(str(title), dict(figures), tuple(figures))


def save_figure_set(
    x: FigureSet,
    directory: str | Path,
    width: float = 7,
    height: float = 5,
    dpi: int = 300,
    device: str = "png",
    overwrite: bool = False,
) -> pd.DataFrame:
    if not isinstance(x, FigureSet):
        raise GP3BayesError("`x` must be a gp3bayes figure set.")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    ext = str(device).lstrip(".")
    rows = []
    for name, fig in x.figures.items():
        target = path / f"{name}.{ext}"
        if target.exists() and not overwrite:
            raise GP3BayesError(f"Figure file already exists: {target}")
        fig.set_size_inches(float(width), float(height))
        fig.savefig(target, dpi=int(dpi), bbox_inches="tight")
        rows.append({"name": name, "file": str(target.resolve())})
    return pd.DataFrame(rows)


def create_publication_registry(label: str | None = None) -> PublicationRegistry:
    if label is not None and (not isinstance(label, str) or not label):
        raise GP3BayesError("`label` must be a non-empty string.")
    return PublicationRegistry(label=label)


def _registry(registry: PublicationRegistry) -> PublicationRegistry:
    if not isinstance(registry, PublicationRegistry):
        raise GP3BayesError("`registry` must be a gp3bayes publication registry.")
    return registry


def register_publication_table(
    registry: PublicationRegistry,
    name: str,
    table: pd.DataFrame,
    caption: str | None = None,
    source: str | None = None,
) -> PublicationRegistry:
    registry = _registry(registry)
    if not isinstance(name, str) or not name:
        raise GP3BayesError("`name` must be one non-empty string.")
    if not isinstance(table, pd.DataFrame):
        raise GP3BayesError("`table` must be a DataFrame.")
    if name in registry.entries["name"].tolist():
        raise GP3BayesError("Registry name already exists.")
    registry.tables[name] = table.copy()
    registry.entries = pd.concat(
        [
            registry.entries,
            pd.DataFrame([{"name": name, "type": "table", "caption": caption, "source": source}]),
        ],
        ignore_index=True,
    )
    return registry


def register_publication_figure(
    registry: PublicationRegistry,
    name: str,
    figure: Any,
    caption: str | None = None,
    source: str | None = None,
) -> PublicationRegistry:
    registry = _registry(registry)
    if not isinstance(name, str) or not name:
        raise GP3BayesError("`name` must be one non-empty string.")
    if not hasattr(figure, "savefig"):
        raise GP3BayesError("`figure` must be a Matplotlib Figure.")
    if name in registry.entries["name"].tolist():
        raise GP3BayesError("Registry name already exists.")
    registry.figures[name] = figure
    registry.entries = pd.concat(
        [
            registry.entries,
            pd.DataFrame([{"name": name, "type": "figure", "caption": caption, "source": source}]),
        ],
        ignore_index=True,
    )
    return registry


def publication_registry_table(x: PublicationRegistry) -> pd.DataFrame:
    return _registry(x).entries.copy()


def validate_publication_registry(x: PublicationRegistry) -> RegistryValidation:
    x = _registry(x)
    duplicated = x.entries["name"].duplicated().any()
    known = set(x.entries["name"])
    actual = set(x.tables) | set(x.figures)
    checks = pd.DataFrame(
        [
            {"check": "unique_names", "status": "fail" if duplicated else "pass"},
            {"check": "entry_objects_match", "status": "pass" if known == actual else "fail"},
        ]
    )
    return RegistryValidation(bool((checks["status"] == "pass").all()), checks)


def write_publication_registry(
    x: PublicationRegistry, file: str | Path, overwrite: bool = False
) -> str:
    validate = validate_publication_registry(x)
    if not validate.valid:
        raise GP3BayesError("Publication registry validation failed.")
    path = Path(file)
    if path.exists() and not overwrite:
        raise GP3BayesError("`file` already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# gp3bayes publication registry",
        "",
        f"Label: {x.label or 'not specified'}",
        "",
        "## Entries",
        "",
        x.entries.to_string(index=False),
        "",
        "Automatic writing: FALSE",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def save_publication_registry_figures(
    x: PublicationRegistry,
    directory: str | Path,
    width: float = 7,
    height: float = 5,
    dpi: int = 300,
    device: str = "png",
    overwrite: bool = False,
) -> pd.DataFrame:
    _registry(x)
    if not x.figures:
        return pd.DataFrame(columns=["name", "file"])
    return save_figure_set(
        FigureSet(x.label or "publication registry", x.figures, tuple(x.figures)),
        directory,
        width,
        height,
        dpi,
        device,
        overwrite,
    )


def create_complete_evidence_inventory(
    objects: Mapping[str, Any],
    label: str | None = None,
) -> EvidenceInventory:
    if not isinstance(objects, Mapping) or not objects:
        raise GP3BayesError("Supply a non-empty mapping of named evidence objects.")
    rows = [
        {
            "component": name,
            "class": type(value).__name__ if value is not None else "NoneType",
            "status": _status(value),
            "available": value is not None,
            "automatic_decision": False,
        }
        for name, value in objects.items()
    ]
    return EvidenceInventory("0.3", label, dict(objects), pd.DataFrame(rows))


def evidence_inventory_table(x: EvidenceInventory) -> pd.DataFrame:
    if not isinstance(x, EvidenceInventory):
        raise GP3BayesError("`x` must be a gp3bayes evidence inventory.")
    return x.table.copy()


def create_diagnostic_dashboard(
    fit: Any = None,
    analysis_bundle: Any = None,
    model_card: Any = None,
    loo: Any = None,
    prior_posterior: Any = None,
    sensitivity: Any = None,
    recovery: Any = None,
    sbc: Any = None,
    label: str | None = None,
) -> DiagnosticDashboard:
    objects = {
        "fit": fit,
        "analysis_bundle": analysis_bundle,
        "model_card": model_card,
        "loo": loo,
        "prior_posterior": prior_posterior,
        "sensitivity": sensitivity,
        "recovery": recovery,
        "sbc": sbc,
    }
    if all(v is None for v in objects.values()):
        raise GP3BayesError("Supply at least one evidence component.")
    table = pd.DataFrame(
        [
            {
                "component": name,
                "available": value is not None,
                "class": "" if value is None else type(value).__name__,
                "status": _status(value),
            }
            for name, value in objects.items()
        ]
    )
    return DiagnosticDashboard("0.3", label, objects, table)


def diagnostic_dashboard_table(x: DiagnosticDashboard) -> pd.DataFrame:
    if not isinstance(x, DiagnosticDashboard):
        raise GP3BayesError("`x` must be a gp3bayes diagnostic dashboard.")
    return x.table.copy()


def plot_diagnostic_dashboard(x: DiagnosticDashboard):
    d = diagnostic_dashboard_table(x)
    fig, ax = _figure(x.label or "gp3bayes diagnostic dashboard")
    ax.barh(d["component"], d["available"].astype(int))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["Unavailable", "Available"])
    return fig


def create_diagnostic_dashboard_figures(x: DiagnosticDashboard) -> FigureSet:
    if not isinstance(x, DiagnosticDashboard):
        raise GP3BayesError("`x` must be a gp3bayes diagnostic dashboard.")
    z = x.objects
    plots = {}
    if z.get("fit") is not None:
        plots["posterior_intervals"] = plot_posterior_intervals(z["fit"], regex=r"^(b_|sd_|sigma$)")
    if z.get("model_card") is not None:
        plots["reporting_checklist"] = plot_reporting_checklist(z["model_card"])
    if z.get("prior_posterior") is not None:
        from .prior_posterior_bridge import (
            plot_prior_posterior_contraction,
            plot_prior_posterior_shift,
        )

        plots["prior_posterior_shift"] = plot_prior_posterior_shift(z["prior_posterior"])
        plots["prior_posterior_contraction"] = plot_prior_posterior_contraction(
            z["prior_posterior"]
        )
    if z.get("loo") is not None:
        plots["loo_influence"] = plot_loo_influence(z["loo"])
    if z.get("recovery") is not None:
        from .sensitivity import plot_recovery_bias, plot_recovery_coverage

        plots["recovery_bias"] = plot_recovery_bias(z["recovery"])
        plots["recovery_coverage"] = plot_recovery_coverage(z["recovery"])
    if not plots:
        raise GP3BayesError("No dashboard component can be plotted.")
    return create_figure_set(plots, title=x.label or "gp3bayes diagnostic dashboard")


def write_diagnostic_dashboard_report(
    x: DiagnosticDashboard, file: str | Path, overwrite: bool = False
) -> str:
    if not isinstance(x, DiagnosticDashboard):
        raise GP3BayesError("`x` must be a gp3bayes diagnostic dashboard.")
    path = Path(file)
    if path.exists() and not overwrite:
        raise GP3BayesError("`file` already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# gp3bayes diagnostic dashboard",
        "",
        f"- Label: {x.label or 'not specified'}",
        f"- Available components: {int(x.table['available'].sum())}/{len(x.table)}",
        "- Automatic decision: `FALSE`",
        "",
        "## Evidence availability",
        "",
        x.table.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        x.interpretation,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def create_model_card(
    fit: Any, analysis_bundle: Any = None, manifest: Any = None, label: str | None = None
) -> ModelCard:
    family = getattr(fit, "family", None)
    if family not in {"binary", "duration"}:
        raise GP3BayesError("`fit` must be an approved gp3bayes fit.")
    from .unified_workflow_api import diagnose_model_fit, model_workflow_status

    try:
        diagnosis = diagnose_model_fit(fit)
    except Exception as exc:
        diagnosis = exc
    try:
        workflow = model_workflow_status(fit)
    except Exception as exc:
        workflow = exc  # type: ignore[assignment]
    evidence = pd.DataFrame(
        {
            "component": [
                "model_fit",
                "model_diagnosis",
                "workflow_status",
                "analysis_bundle",
                "analysis_manifest",
            ],
            "status": [
                "available",
                _status(diagnosis),
                _status(workflow),
                _status(analysis_bundle),
                _status(manifest),
            ],
        }
    )
    spec = getattr(fit, "specification", None)
    formula = str(getattr(spec, "formula_text", getattr(spec, "formula", "")))
    return ModelCard(
        "0.3",
        label,
        family,
        getattr(spec, "model_family", None),
        formula,
        getattr(fit, "sampling_backend", getattr(fit, "backend_interface", None)),
        dict(getattr(fit, "sampling", {})),
        dict(getattr(fit, "package_versions", {})),
        diagnosis,
        workflow,
        analysis_bundle,
        manifest,
        evidence,
    )


def model_card_table(x: ModelCard) -> pd.DataFrame:
    if not isinstance(x, ModelCard):
        raise GP3BayesError("`x` must be a gp3bayes model card.")
    return x.evidence.copy()


def create_reporting_checklist(x: Any) -> pd.DataFrame:
    card = x if isinstance(x, ModelCard) else create_model_card(x)
    rows = [
        ("model_family_recorded", bool(card.model_family)),
        ("formula_recorded", bool(card.formula)),
        ("sampling_backend_recorded", bool(card.sampling_backend)),
        ("sampling_settings_recorded", bool(card.sampling)),
        ("diagnostics_available", not isinstance(card.diagnosis, Exception)),
        ("workflow_status_available", not isinstance(card.workflow, Exception)),
        ("analysis_bundle_available", card.analysis_bundle is not None),
        ("analysis_manifest_available", card.manifest is not None),
        ("interpretation_boundary_recorded", bool(card.interpretation)),
    ]
    return pd.DataFrame(
        [
            {"item": name, "available": available, "automatic_requirement": False}
            for name, available in rows
        ]
    )


def plot_reporting_checklist(x: Any):
    d = x.copy() if isinstance(x, pd.DataFrame) else create_reporting_checklist(x)
    fig, ax = _figure("gp3bayes reporting evidence inventory")
    ax.barh(d["item"], d["available"].astype(int))
    ax.set_xlim(0, 1)
    return fig


def write_model_card(x: ModelCard, file: str | Path, overwrite: bool = False) -> str:
    if not isinstance(x, ModelCard):
        raise GP3BayesError("`x` must be a gp3bayes model card.")
    path = Path(file)
    if path.exists() and not overwrite:
        raise GP3BayesError("`file` already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    checklist = create_reporting_checklist(x)
    lines = [
        "# gp3bayes model card",
        "",
        f"- Label: {x.label or 'not specified'}",
        f"- Family: `{x.family}`",
        f"- Model family: `{x.model_family}`",
        f"- Formula: `{x.formula}`",
        f"- Sampling backend: `{x.sampling_backend}`",
        "",
        "## Evidence inventory",
        "",
        x.evidence.to_string(index=False),
        "",
        "## Reporting checklist",
        "",
        checklist.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        x.interpretation,
        "",
        "- Model adequacy certified automatically: `FALSE`",
        "- Causal identification certified automatically: `FALSE`",
        "- Automatic model selection: `FALSE`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


# Plotting adapters ----------------------------------------------------------
def plot_posterior_intervals(
    x: Any,
    variables: Sequence[str] | None = None,
    regex: str | None = None,
    prob: float = 0.8,
    prob_outer: float = 0.95,
):
    d = _posterior_frame(x, variables, regex)
    fig, ax = _figure("Posterior intervals")
    y = np.arange(len(d))
    med = d["median"].to_numpy(float)
    low = d["lower"].to_numpy(float)
    high = d["upper"].to_numpy(float)
    ax.errorbar(med, y, xerr=[med - low, high - med], fmt="o")
    ax.axvline(0, linestyle="--")
    ax.set_yticks(y, d["variable"].astype(str))
    return fig


def plot_posterior_areas(
    x: Any,
    variables: Sequence[str] | None = None,
    regex: str | None = None,
    prob: float = 0.5,
    prob_outer: float = 0.95,
):
    return plot_posterior_intervals(x, variables, regex, prob, prob_outer)


def plot_posterior_density(
    x: Any, variables: Sequence[str] | None = None, regex: str | None = None
):
    from .postfit_exploration import extract_posterior_draws

    draws = (
        extract_posterior_draws(x, variables=variables, regex=regex)
        if not isinstance(x, (np.ndarray, pd.DataFrame))
        else x
    )
    frame = pd.DataFrame(draws)
    fig, ax = _figure("Posterior density")
    for col in frame.columns:
        ax.hist(
            frame[col].dropna().to_numpy(float),
            bins=30,
            density=True,
            histtype="step",
            label=str(col),
        )
    ax.legend()
    return fig


def plot_posterior_pairs(
    fit: Any, variables: Sequence[str] | None = None, regex: str = "^b_", max_variables: int = 8
):
    from .postfit_exploration import extract_posterior_draws

    d = extract_posterior_draws(fit, variables=variables, regex=regex)
    d = d.iloc[:, :max_variables]
    _plt()
    axes = pd.plotting.scatter_matrix(d, diagonal="hist")
    return axes[0, 0].figure


def plot_rank_diagnostics(fit: Any, variables: Sequence[str] | None = None, regex: str = "^b_"):
    from .postfit_exploration import extract_posterior_draws

    d = extract_posterior_draws(fit, variables=variables, regex=regex)
    fig, ax = _figure("Posterior rank diagnostics")
    for col in d.columns:
        ranks = pd.Series(d[col]).rank(pct=True)
        ax.hist(ranks, bins=20, histtype="step", label=str(col))
    ax.legend()
    return fig


def plot_autocorrelation(
    fit: Any, variables: Sequence[str] | None = None, regex: str = "^b_", lags: int = 20
):
    from .postfit_exploration import extract_posterior_draws

    d = extract_posterior_draws(fit, variables=variables, regex=regex)
    fig, ax = _figure("Posterior autocorrelation")
    for col in d.columns:
        z = d[col].to_numpy(float)
        z = z - z.mean()
        denom = np.dot(z, z)
        ac = [1.0] + [
            float(np.dot(z[:-lag], z[lag:]) / denom)
            for lag in range(1, min(int(lags), len(z) - 1) + 1)
        ]
        ax.plot(range(len(ac)), ac, label=str(col))
    ax.axhline(0, linestyle="--")
    ax.legend()
    return fig


def plot_posterior_correlations(
    x: Any,
    variables: Sequence[str] | None = None,
    regex: str | None = None,
    method: str = "pearson",
):
    if isinstance(x, pd.DataFrame) and x.shape[1] > 1 and "variable1" not in x.columns:
        frame = x
    elif isinstance(x, np.ndarray):
        frame = pd.DataFrame(x)
    else:
        from .postfit_exploration import posterior_correlation_table

        table = posterior_correlation_table(x, variables=variables, regex=regex, method=method)
        names = sorted(set(table["variable1"]) | set(table["variable2"]))
        matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for r in table.itertuples():
            matrix.loc[r.variable1, r.variable2] = matrix.loc[r.variable2, r.variable1] = (
                r.correlation
            )
        frame = matrix
    corr = frame if frame.index.equals(frame.columns) else frame.corr(method=method)  # type: ignore[arg-type]
    fig, ax = _figure("Posterior correlations")
    im = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(im, ax=ax)
    return fig


def plot_mcmc_quality(x: Any):
    d = _df(x, "issues", "table", "component_table")
    status_col = "status" if "status" in d.columns else None
    fig, ax = _figure("MCMC quality")
    counts = (
        d[status_col].astype(str).value_counts() if status_col else pd.Series({"recorded": len(d)})
    )
    ax.bar(counts.index, counts.values)
    return fig


def plot_sampler_diagnostics(fit: Any):
    from .postfit_exploration import sampler_diagnostic_table

    d = sampler_diagnostic_table(fit)
    fig, ax = _figure("Sampler diagnostics")
    numeric = d.select_dtypes(include=[np.number])
    if numeric.empty:
        ax.text(0.5, 0.5, "No numeric sampler diagnostics", ha="center")
    else:
        ax.boxplot([numeric[c].dropna() for c in numeric], tick_labels=list(numeric.columns))
    return fig


def plot_estimand_intervals(
    x: Any, quantities: Sequence[str] | None = None, probs: Sequence[float] = (0.025, 0.5, 0.975)
):
    if hasattr(x, "draws") and isinstance(x.draws, Mapping):
        rows = []
        for name, z in x.draws.items():
            if quantities is not None and name not in quantities:
                continue
            arr = np.asarray(z, float)
            q = np.quantile(arr, probs)
            rows.append({"variable": name, "lower": q[0], "median": q[1], "upper": q[2]})
        return plot_posterior_intervals(pd.DataFrame(rows))
    return plot_posterior_intervals(x)


def plot_prediction_intervals(x: Any, max_rows: int = 100):
    d = _df(x, "summary").head(max_rows)
    fig, ax = _figure("Prediction intervals")
    y = np.arange(len(d))
    mean_col = "predicted_mean" if "predicted_mean" in d.columns else "mean"
    low = next((c for c in ("lower", "posterior_lower") if c in d.columns), None)
    high = next((c for c in ("upper", "posterior_upper") if c in d.columns), None)
    ax.plot(d[mean_col].to_numpy(float), y, "o")
    if low and high:
        ax.errorbar(d[mean_col], y, xerr=[d[mean_col] - d[low], d[high] - d[mean_col]], fmt="none")
    ax.set_ylabel("Observation")
    return fig


def plot_binary_calibration(x: Any, bins: int = 10):
    if not isinstance(x, pd.DataFrame):
        from .predictive import binary_calibration_table

        x = binary_calibration_table(x, bins=bins)
    d = x
    xp = next(
        c
        for c in ("mean_predicted_probability", "predicted_mean", "mean_probability")
        if c in d.columns
    )
    yp = next(c for c in ("observed_rate", "observed") if c in d.columns)
    fig, ax = _figure("Binary calibration")
    ax.plot(d[xp], d[yp], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_binary_threshold_metrics(
    x: Any,
    observed: Sequence[float] | None = None,
    thresholds: Sequence[float] = tuple(np.arange(0.1, 0.91, 0.05)),
):
    if not isinstance(x, pd.DataFrame):
        from .predictive import binary_threshold_metrics

        x = binary_threshold_metrics(x, observed=observed, thresholds=thresholds)
    d = x
    fig, ax = _figure("Binary threshold metrics")
    for c in ("accuracy", "sensitivity", "specificity", "balanced_accuracy"):
        if c in d.columns:
            ax.plot(d["threshold"], d[c], label=c)
    ax.set_ylim(0, 1)
    ax.legend()
    return fig


def plot_duration_quantile_calibration(x: Any):
    d = _df(x, "table")
    p = next(
        c for c in ("probability", "quantile_probability", "nominal_probability") if c in d.columns
    )
    e = next(
        c
        for c in ("empirical_probability", "empirical_coverage", "observed_probability")
        if c in d.columns
    )
    fig, ax = _figure("Duration quantile calibration")
    ax.plot(d[p], d[e], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    return fig


def plot_duration_pit(x: Any, bins: int = 10):
    d = _df(x, "table")
    pit = next(c for c in ("pit", "value") if c in d.columns)
    fig, ax = _figure("Duration PIT")
    ax.hist(d[pit], bins=int(bins), range=(0, 1))
    return fig


def plot_exceedance_probability(x: Any):
    d = _df(x, "table") if not isinstance(x, pd.DataFrame) else x
    fig, ax = _figure("Prediction exceedance probability")
    ax.bar(np.arange(len(d)), d["probability"])
    ax.set_ylim(0, 1)
    return fig


def plot_predictive_coverage(x: Any):
    d = _df(x, "table") if not isinstance(x, pd.DataFrame) else x
    n = next(c for c in ("nominal_coverage", "level") if c in d.columns)
    e = next(c for c in ("empirical_coverage", "coverage") if c in d.columns)
    fig, ax = _figure("Predictive coverage")
    ax.plot(d[n], d[e], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    return fig


def plot_predictive_residuals(x: Any):
    d = _df(x, "table") if not isinstance(x, pd.DataFrame) else x
    residual = "residual"
    fig, ax = _figure("Predictive residuals")
    ax.axhline(0, linestyle="--")
    ax.scatter(np.arange(len(d)), d[residual])
    return fig


def plot_prediction_support(x: Any):
    d = _df(x, "table")
    fig, ax = _figure("Prediction support audit")
    status = next((c for c in ("status", "support_status") if c in d.columns), None)
    counts = d[status].astype(str).value_counts() if status else pd.Series({"recorded": len(d)})
    ax.bar(counts.index, counts.values)
    return fig


def plot_uncertainty_decomposition(x: Any, max_rows: int = 100):
    d = _df(x, "table").head(max_rows)
    cols = [
        c for c in ("epistemic_variance", "residual_variance", "total_variance") if c in d.columns
    ]
    fig, ax = _figure("Prediction uncertainty decomposition")
    for c in cols:
        ax.plot(np.arange(len(d)), d[c], label=c)
    if cols:
        ax.legend()
        return fig


def plot_grouped_prediction_check(x: Any):
    d = _df(x, "table")
    next(c for c in ("group", "level") if c in d.columns)
    fig, ax = _figure("Grouped prediction check")
    ax.scatter(d["observed"], d["predicted_mean"])
    lo = min(float(d["observed"].min()), float(d["predicted_mean"].min()))
    hi = max(float(d["observed"].max()), float(d["predicted_mean"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    return fig


def plot_group_effects(x: Any, groups: Sequence[str] | None = None):
    d = _df(x, "table") if not isinstance(x, pd.DataFrame) else x
    if groups is not None and "group" in d.columns:
        d = d[d["group"].astype(str).isin([str(v) for v in groups])]
    label = next((c for c in ("level", "group_level", "variable") if c in d.columns), d.columns[0])
    value = next((c for c in ("median", "mean", "estimate") if c in d.columns), None)
    fig, ax = _figure("Group effects")
    if value:
        ax.barh(d[label].astype(str), d[value].astype(float))
    return fig


def plot_variance_components(x: Any):
    d = _df(x, "table") if not isinstance(x, pd.DataFrame) else x
    label = next((c for c in ("component", "variable") if c in d.columns), d.columns[0])
    value = next((c for c in ("median", "variance", "mean") if c in d.columns), None)
    fig, ax = _figure("Variance components")
    if value:
        ax.barh(d[label].astype(str), d[value].astype(float))
    return fig


def plot_loo_influence(x: Any):
    try:
        from .loo import loo_pointwise_table

        d = loo_pointwise_table(x)
    except Exception:
        d = _df(x, "table", "pointwise")
    k = next(c for c in ("pareto_k", "k") if c in d.columns)
    fig, ax = _figure("PSIS-LOO influence")
    ax.scatter(np.arange(1, len(d) + 1), d[k])
    ax.axhline(0.7, linestyle="--")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Pareto k")
    return fig


def plot_model_comparison(x: Any):
    from .postfit_exploration import model_comparison_table

    d = x if isinstance(x, pd.DataFrame) else model_comparison_table(x)
    label = next(c for c in ("model", "name") if c in d.columns)
    value = next(c for c in ("elpd_loo", "estimate", "difference") if c in d.columns)
    fig, ax = _figure("Model comparison")
    ax.barh(d[label].astype(str), d[value].astype(float))
    return fig


def plot_model_weights(x: Any):
    from .postfit_exploration import model_weights_table

    d = x if isinstance(x, pd.DataFrame) else model_weights_table(x)
    label = next(c for c in ("model", "name") if c in d.columns)
    value = next(c for c in ("weight", "stacking_weight") if c in d.columns)
    fig, ax = _figure("Model weights")
    ax.barh(d[label].astype(str), d[value].astype(float))
    ax.set_xlim(0, 1)
    return fig


__all__ = [
    name
    for name in globals()
    if name.startswith(("create_", "plot_", "register_", "save_", "validate_", "write_"))
    or name
    in {
        "DiagnosticDashboard",
        "EvidenceInventory",
        "FigureSet",
        "ModelCard",
        "PublicationRegistry",
        "RegistryValidation",
        "diagnostic_dashboard_table",
        "evidence_inventory_table",
        "model_card_table",
        "publication_registry_table",
        "theme_gp3bayes",
    }
]
