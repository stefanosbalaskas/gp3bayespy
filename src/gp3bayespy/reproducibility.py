"""Reproducible analysis manifests and post-fit bundles.

The frozen R package serializes manifests as RDS and fingerprints with MD5 over
R serialization.  gp3bayespy uses canonical Python serialization for its own
round trips while retaining the same change-detection, explicit-file, and
interpretation-boundary semantics.  Hashes are provenance fingerprints, not
cryptographic authenticity proofs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(c) for c in value.columns],
            "dtypes": [str(d) for d in value.dtypes],
            "records": [
                [_stable(v) for v in row] for row in value.itertuples(index=True, name=None)
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "name": str(value.name),
            "dtype": str(value.dtype),
            "values": [_stable(v) for v in value.tolist()],
        }
    if isinstance(value, np.ndarray):
        return [_stable(v) for v in value.tolist()]
    if is_dataclass(value):
        return _stable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list, set)):
        return [_stable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _stable({k: v for k, v in vars(value).items() if not k.startswith("backend_")})
    return repr(value)


def _hash(value: Any) -> str:
    payload = json.dumps(
        _stable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.md5(payload).hexdigest()  # nosec B324 - change detection only


def _data_fingerprint(data: pd.DataFrame | None) -> dict[str, Any]:
    if data is None:
        return {
            "available": False,
            "hash": None,
            "hash_method": "MD5-of-canonical-Python-v1",
            "row_order_sensitive": True,
            "rows": None,
            "columns": None,
        }
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a pandas DataFrame when supplied.")
    return {
        "available": True,
        "hash": _hash(data),
        "hash_method": "MD5-of-canonical-Python-v1",
        "row_order_sensitive": True,
        "rows": len(data),
        "columns": len(data.columns),
        "column_names": tuple(str(c) for c in data.columns),
    }


def _signature(value: Any) -> dict[str, Any]:
    if value is None:
        return {"available": False, "hash": None}
    return {"available": True, "hash": _hash(value), "value": _stable(value)}


def _versions() -> dict[str, str]:
    result = {}
    for package in ("gp3bayespy", "numpy", "pandas", "scipy", "pymc", "matplotlib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


@dataclass(slots=True)
class AnalysisManifest:
    manifest_version: str
    fingerprint_method: str
    fingerprint_security: str
    created_at: str
    label: str | None
    family: str | None
    model_family: str | None
    contract: Mapping[str, Any]
    specification: Mapping[str, Any]
    transformations: Mapping[str, Any]
    data: Mapping[str, Any]
    estimands: Any
    sensitivity_plan: Any
    seed: int | None
    sampling: Mapping[str, Any] | None
    package_versions: Mapping[str, str]
    python_version: str
    platform: str
    notes: tuple[str, ...]
    frozen: bool = False
    manifest_hash: str | None = None
    adequacy_established: bool = False
    causal_identification_established: bool = False
    frozen_at: str | None = None


@dataclass(slots=True)
class ManifestValidation:
    status: str
    checks: pd.DataFrame


@dataclass(slots=True)
class ManifestComparison:
    comparison_version: str
    identical: bool
    changed_components: tuple[str, ...]
    table: pd.DataFrame
    left: AnalysisManifest
    right: AnalysisManifest
    interpretation: str = "Differences are provenance differences only. No difference is automatically interpreted as inferential failure."


@dataclass(slots=True)
class CapturedComponent:
    ok: bool
    value: Any = None
    error: str | None = None


@dataclass(slots=True)
class AnalysisBundle:
    bundle_version: str
    family: str
    fit: Any
    components: Mapping[str, CapturedComponent]
    status: pd.DataFrame
    include_loo: bool
    automatic_decision: bool = False
    interpretation: str = "The bundle collects post-fit evidence for inspection and reporting. It does not automatically establish adequacy, robustness, or causal validity."


def create_analysis_manifest(
    specification: Any = None,
    fit: Any = None,
    data: pd.DataFrame | None = None,
    estimands: Sequence[str] | Mapping[str, Any] | None = (),
    sensitivity_plan: Any = None,
    seed: int | None = None,
    label: str | None = None,
    notes: Sequence[str] = (),
) -> AnalysisManifest:
    if fit is not None:
        family = getattr(fit, "family", None)
        if family not in {"binary", "duration"}:
            raise GP3BayesError("`fit` must be an approved gp3bayes fit.")
        if specification is None:
            specification = getattr(fit, "specification", None)
    if specification is not None and getattr(specification, "family", None) not in {
        "binary",
        "duration",
    }:
        raise GP3BayesError("`specification` must be an approved gp3bayes model specification.")
    prepared = getattr(specification, "prepared", None) if specification is not None else None
    contract = getattr(specification, "contract", None) if specification is not None else None
    if data is None and prepared is not None:
        data = getattr(prepared, "data", None)
    if seed is None and fit is not None:
        seed = getattr(fit, "sampling", {}).get("seed")
    if seed is not None and (isinstance(seed, bool) or int(seed) != seed or int(seed) < 0):
        raise GP3BayesError("`seed` must be NULL or one non-negative integer.")
    if label is not None and (not isinstance(label, str) or not label):
        raise GP3BayesError("`label` must be NULL or one non-empty string.")
    note_values = tuple(str(v) for v in notes)
    if any(not v for v in note_values):
        raise GP3BayesError("`notes` must contain non-empty strings.")
    transformations = getattr(prepared, "transformations", None)
    sampling = dict(getattr(fit, "sampling", {})) if fit is not None else None
    return AnalysisManifest(
        "0.2",
        "MD5-of-canonical-Python-v1",
        "change-detection only; not a cryptographic authenticity proof",
        _utc(),
        label,
        getattr(specification, "family", None),
        getattr(specification, "model_family", None),
        _signature(contract),
        _signature(specification),
        _signature(transformations),
        _data_fingerprint(data),
        tuple(estimands) if isinstance(estimands, (list, tuple, set)) else estimands,
        sensitivity_plan,
        None if seed is None else int(seed),
        sampling,
        _versions(),
        platform.python_version(),
        platform.platform(),
        note_values,
    )


def validate_analysis_manifest(
    manifest: AnalysisManifest, strict: bool = False
) -> ManifestValidation:
    rows = []
    valid_class = isinstance(manifest, AnalysisManifest)
    rows.append(
        {
            "check": "manifest_class",
            "status": "pass" if valid_class else "fail",
            "detail": type(manifest).__name__,
        }
    )
    if valid_class:
        required_ok = all(
            hasattr(manifest, k)
            for k in (
                "manifest_version",
                "family",
                "specification",
                "data",
                "estimands",
                "package_versions",
                "frozen",
                "manifest_hash",
            )
        )
        rows.append(
            {
                "check": "required_fields",
                "status": "pass" if required_ok else "fail",
                "detail": "complete" if required_ok else "missing",
            }
        )
        hash_ok = not manifest.data.get("available", False) or bool(manifest.data.get("hash"))
        rows.append(
            {
                "check": "data_fingerprint",
                "status": "pass" if hash_ok else "fail",
                "detail": str(manifest.data.get("hash")),
            }
        )
        if manifest.family is not None:
            rows.append(
                {
                    "check": "approved_family",
                    "status": "pass" if manifest.family in {"binary", "duration"} else "fail",
                    "detail": str(manifest.family),
                }
            )
    table = pd.DataFrame(rows)
    status = "fail" if (table["status"] == "fail").any() else "pass"
    if strict and status == "fail":
        raise GP3BayesError(
            "Manifest validation failed: "
            + ", ".join(table.loc[table["status"].eq("fail"), "check"])
        )
    return ManifestValidation(status, table)


def freeze_analysis_manifest(
    manifest: AnalysisManifest, file: str | Path | None = None, overwrite: bool = False
) -> AnalysisManifest:
    validate_analysis_manifest(manifest, strict=True)
    canonical = {
        k: getattr(manifest, k)
        for k in (
            "manifest_version",
            "fingerprint_method",
            "label",
            "family",
            "model_family",
            "contract",
            "specification",
            "transformations",
            "data",
            "estimands",
            "sensitivity_plan",
            "seed",
            "sampling",
            "package_versions",
            "python_version",
            "platform",
            "notes",
        )
    }
    manifest.manifest_hash = _hash(canonical)
    manifest.frozen = True
    manifest.frozen_at = _utc()
    if file is not None:
        path = Path(file)
        if path.exists() and not overwrite:
            raise GP3BayesError("`file` already exists. Set `overwrite=True` to replace it.")
        if not path.parent.exists():
            raise GP3BayesError("The parent directory for `file` does not exist.")
        path.write_bytes(pickle.dumps(manifest, protocol=5))
    return manifest


def read_analysis_manifest(file: str | Path) -> AnalysisManifest:
    path = Path(file)
    if not path.exists():
        raise GP3BayesError("Manifest file does not exist.")
    obj = pickle.loads(path.read_bytes())  # nosec B301 - explicit trusted local manifest format
    validate_analysis_manifest(obj, strict=True)
    return obj


def compare_analysis_manifests(x: AnalysisManifest, y: AnalysisManifest) -> ManifestComparison:
    validate_analysis_manifest(x, strict=True)
    validate_analysis_manifest(y, strict=True)
    components = {
        "family": (x.family, y.family),
        "contract_hash": (x.contract.get("hash"), y.contract.get("hash")),
        "specification_hash": (x.specification.get("hash"), y.specification.get("hash")),
        "transformation_hash": (x.transformations.get("hash"), y.transformations.get("hash")),
        "data_hash": (x.data.get("hash"), y.data.get("hash")),
        "estimands": (x.estimands, y.estimands),
        "sensitivity_plan": (x.sensitivity_plan, y.sensitivity_plan),
        "seed": (x.seed, y.seed),
        "sampling": (x.sampling, y.sampling),
        "package_versions": (x.package_versions, y.package_versions),
    }
    rows = []
    for name, (left, right) in components.items():
        same = _stable(left) == _stable(right)
        rows.append(
            {
                "component": name,
                "identical": same,
                "left": repr(_stable(left)),
                "right": repr(_stable(right)),
            }
        )
    table = pd.DataFrame(rows)
    changed = tuple(table.loc[~table["identical"], "component"].astype(str))
    return ManifestComparison("0.2", not changed, changed, table, x, y)


def analysis_manifest_table(manifest: AnalysisManifest) -> pd.DataFrame:
    validate_analysis_manifest(manifest, strict=True)
    return pd.DataFrame(
        [
            {
                "component": "data",
                "available": bool(manifest.data.get("available")),
                "hash": manifest.data.get("hash"),
            },
            {
                "component": "contract",
                "available": bool(manifest.contract.get("available")),
                "hash": manifest.contract.get("hash"),
            },
            {
                "component": "specification",
                "available": bool(manifest.specification.get("available")),
                "hash": manifest.specification.get("hash"),
            },
            {
                "component": "transformations",
                "available": bool(manifest.transformations.get("available")),
                "hash": manifest.transformations.get("hash"),
            },
            {"component": "manifest", "available": manifest.frozen, "hash": manifest.manifest_hash},
        ]
    )


def write_reproducibility_report(
    manifest: AnalysisManifest, file: str | Path, overwrite: bool = False
) -> str:
    validate_analysis_manifest(manifest, strict=True)
    path = Path(file)
    if path.exists() and not overwrite:
        raise GP3BayesError("`file` already exists. Set `overwrite=True` to replace it.")
    if not path.parent.exists():
        raise GP3BayesError("The report parent directory does not exist.")
    lines = [
        "# gp3bayes reproducibility report",
        "",
        f"Family: {manifest.family}",
        f"Frozen: {manifest.frozen}",
        f"Manifest hash: {manifest.manifest_hash}",
        "",
        "## Data fingerprint",
        "",
        f"- Method: {manifest.data.get('hash_method')}",
        f"- Hash: {manifest.data.get('hash')}",
        f"- Row-order sensitive: {manifest.data.get('row_order_sensitive')}",
        "",
        "## Declared estimands",
        "",
        repr(manifest.estimands),
        "",
        "## Software versions",
        "",
        *[f"- {k}: {v}" for k, v in manifest.package_versions.items()],
        "",
        "## Interpretation boundary",
        "",
        "This report records computational provenance. It does not establish model adequacy, robustness, causal identification, or substantive validity.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def _capture(function: Any, *args: Any, **kwargs: Any) -> CapturedComponent:
    try:
        return CapturedComponent(True, function(*args, **kwargs), None)
    except Exception as exc:
        return CapturedComponent(False, None, str(exc))


def create_analysis_bundle(
    fit: Any,
    newdata: pd.DataFrame | None = None,
    ndraws: int = 1000,
    include_group_effects: bool = False,
    include_loo: bool = False,
) -> AnalysisBundle:
    family = getattr(fit, "family", None)
    if family not in {"binary", "duration"}:
        raise GP3BayesError("`fit` must be an approved gp3bayes fit.")
    from .postfit_exploration import (
        group_effect_table,
        posterior_interval_table,
        summarise_mcmc_quality,
        variance_component_table,
    )
    from .predictive import (
        audit_prediction_support,
        binary_calibration_table,
        binary_prediction_scores,
        duration_prediction_scores,
        duration_quantile_calibration,
        predict_model,
        predictive_coverage_table,
    )

    training = getattr(getattr(getattr(fit, "specification", None), "prepared", None), "data", None)
    target = training if newdata is None else newdata
    components = {
        "posterior": _capture(posterior_interval_table, fit, regex=r"^(b_|sd_|cor_|sigma$)"),
        "mcmc": _capture(summarise_mcmc_quality, fit),
        "prediction_support": _capture(audit_prediction_support, fit, target),
        "expected_prediction": _capture(
            predict_model,
            fit,
            newdata=newdata,
            type="median" if family == "duration" else "expected",
            include_group_effects=include_group_effects,
            ndraws=ndraws,
        ),
        "predictive": _capture(
            predict_model,
            fit,
            newdata=newdata,
            type="predictive",
            include_group_effects=include_group_effects,
            ndraws=ndraws,
            seed=1,
        ),
        "group_effects": _capture(group_effect_table, fit),
        "variance_components": _capture(variance_component_table, fit),
    }
    expected = components["expected_prediction"]
    if expected.ok and getattr(expected.value, "observed", None) is not None:
        components["scores"] = _capture(
            binary_prediction_scores if family == "binary" else duration_prediction_scores,
            expected.value,
        )
        if family == "binary":
            components["calibration"] = _capture(binary_calibration_table, expected.value)
    predictive = components["predictive"]
    if predictive.ok and getattr(predictive.value, "observed", None) is not None:
        components["coverage"] = _capture(predictive_coverage_table, predictive.value)
        if family == "duration":
            components["quantile_calibration"] = _capture(
                duration_quantile_calibration, predictive.value
            )
    if include_loo:
        from .advanced_optional_workflows import compute_psis_loo

        components["loo"] = _capture(compute_psis_loo, fit)
    status = pd.DataFrame(
        [
            {
                "component": name,
                "available": item.ok,
                "error": "" if item.error is None else item.error,
            }
            for name, item in components.items()
        ]
    )
    return AnalysisBundle("0.3", family, fit, components, status, bool(include_loo))


def analysis_bundle_table(x: AnalysisBundle) -> pd.DataFrame:
    if not isinstance(x, AnalysisBundle):
        raise GP3BayesError("`x` must be a gp3bayes analysis bundle.")
    return x.status.copy()


def create_publication_table_set(x: AnalysisBundle) -> dict[str, pd.DataFrame]:
    if not isinstance(x, AnalysisBundle):
        raise GP3BayesError("`x` must be a gp3bayes analysis bundle.")
    from .postfit_exploration import loo_summary_table

    out = {}
    for name, item in x.components.items():
        if not item.ok:
            continue
        value = item.value
        table = None
        if isinstance(value, pd.DataFrame):
            table = value
        elif hasattr(value, "summary") and isinstance(value.summary, pd.DataFrame):
            table = value.summary
        elif hasattr(value, "issues") and isinstance(value.issues, pd.DataFrame):
            table = value.issues
        elif hasattr(value, "table") and isinstance(value.table, pd.DataFrame):
            table = value.table
        elif value.__class__.__name__ == "PSISLOOResult":
            table = loo_summary_table(value)
        if table is not None:
            out[name] = table.copy()
    return out


def create_analysis_figure_set(x: AnalysisBundle):
    if not isinstance(x, AnalysisBundle):
        raise GP3BayesError("`x` must be a gp3bayes analysis bundle.")
    from .reporting import (
        create_figure_set,
        plot_group_effects,
        plot_loo_influence,
        plot_mcmc_quality,
        plot_prediction_intervals,
        plot_prediction_support,
        plot_variance_components,
    )

    plots = {}
    if x.components["mcmc"].ok:
        plots["mcmc_quality"] = plot_mcmc_quality(x.components["mcmc"].value)
    if x.components["prediction_support"].ok:
        plots["prediction_support"] = plot_prediction_support(
            x.components["prediction_support"].value
        )
    if x.components["expected_prediction"].ok:
        plots["prediction_intervals"] = plot_prediction_intervals(
            x.components["expected_prediction"].value
        )
    if x.components["group_effects"].ok:
        plots["group_effects"] = plot_group_effects(x.components["group_effects"].value)
    if x.components["variance_components"].ok:
        plots["variance_components"] = plot_variance_components(
            x.components["variance_components"].value
        )
    if "loo" in x.components and x.components["loo"].ok:
        plots["loo_influence"] = plot_loo_influence(x.components["loo"].value)
    if not plots:
        raise GP3BayesError("No bundle components could be converted to figures.")
    return create_figure_set(**plots, title="gp3bayes analysis figures")


def write_analysis_bundle_report(x: AnalysisBundle, file: str | Path) -> str:
    if not isinstance(x, AnalysisBundle):
        raise GP3BayesError("`x` must be a gp3bayes analysis bundle.")
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = create_publication_table_set(x)
    lines = [
        "# gp3bayes post-fit analysis bundle",
        "",
        f"- Family: `{x.family}`",
        f"- Components available: {int(x.status['available'].sum())}/{len(x.status)}",
        "- Automatic adequacy/model-selection decision: `FALSE`",
        "",
        "## Component status",
        "",
        x.status.to_string(index=False),
        "",
    ]
    for name, table in tables.items():
        lines += [f"## {name.replace('_', ' ')}", "", table.to_string(index=False), ""]
    lines += ["## Interpretation boundary", "", x.interpretation, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.resolve())


__all__ = [
    "AnalysisBundle",
    "AnalysisManifest",
    "CapturedComponent",
    "ManifestComparison",
    "ManifestValidation",
    "analysis_bundle_table",
    "analysis_manifest_table",
    "compare_analysis_manifests",
    "create_analysis_bundle",
    "create_analysis_figure_set",
    "create_analysis_manifest",
    "create_publication_table_set",
    "freeze_analysis_manifest",
    "read_analysis_manifest",
    "validate_analysis_manifest",
    "write_analysis_bundle_report",
    "write_reproducibility_report",
]
