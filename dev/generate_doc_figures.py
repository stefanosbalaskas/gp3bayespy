#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gp3bayespy as gp
from gp3bayespy import evidence_graphics_gg as eg
from gp3bayespy import loo as loo_mod
from gp3bayespy import reporting as rep

DEFAULT_OUTPUT = Path("docs/assets/gallery")


class DummyLOO:
    def __init__(self) -> None:
        self.pointwise = pd.DataFrame(
            {
                "elpd_loo": [-1.1, -0.8, -1.4, -0.9, -1.7, -1.0],
                "mcse_elpd_loo": [0.08, 0.05, 0.10, 0.06, 0.12, 0.07],
            }
        )
        self.pareto_k = np.array([0.20, 0.45, 0.72, 0.35, 1.05, 0.63])
        self.influence_pareto_k = self.pareto_k.copy()


def _save(fig, root: Path, group: str, name: str, title: str, functions: str, article: str):
    directory = root / group
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.png"
    fig.set_size_inches(7.2, 4.6)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if not target.exists() or target.stat().st_size < 2000:
        raise RuntimeError(f"Figure generation failed: {target}")
    return {
        "group": group,
        "name": name,
        "title": title,
        "file": f"{group}/{name}.png",
        "functions": functions,
        "article": article,
    }


def generate_all(root: Path = DEFAULT_OUTPUT):
    root.mkdir(parents=True, exist_ok=True)
    out = []

    probability = np.array([0.03, 0.08, 0.18, 0.32, 0.68, 0.78, 0.91, 0.97])
    observed = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    out.append(
        _save(
            gp.plot_binary_roc(gp.binary_roc_curve(probability, observed)),
            root,
            "predictive",
            "binary-roc",
            "Binary ROC curve",
            "binary_roc_curve(); plot_binary_roc()",
            "advanced-predictive-diagnostics",
        )
    )
    out.append(
        _save(
            gp.plot_binary_precision_recall(
                gp.binary_precision_recall_curve(probability, observed)
            ),
            root,
            "predictive",
            "binary-precision-recall",
            "Precision-recall curve",
            "binary_precision_recall_curve(); plot_binary_precision_recall()",
            "advanced-predictive-diagnostics",
        )
    )

    predictive = [
        (
            "binary-calibration",
            rep.plot_binary_calibration(
                pd.DataFrame(
                    {
                        "mean_predicted_probability": [0.08, 0.28, 0.52, 0.73, 0.91],
                        "observed_rate": [0.06, 0.31, 0.49, 0.76, 0.89],
                    }
                )
            ),
            "Binary calibration",
            "plot_binary_calibration()",
            "advanced-predictive-diagnostics",
        ),
        (
            "binary-threshold-metrics",
            rep.plot_binary_threshold_metrics(
                pd.DataFrame(
                    {
                        "threshold": [0.2, 0.4, 0.6, 0.8],
                        "accuracy": [0.65, 0.78, 0.82, 0.74],
                        "sensitivity": [0.96, 0.88, 0.76, 0.52],
                        "specificity": [0.36, 0.68, 0.87, 0.96],
                        "balanced_accuracy": [0.66, 0.78, 0.82, 0.74],
                    }
                )
            ),
            "Threshold diagnostics",
            "plot_binary_threshold_metrics()",
            "prediction-calibration-and-scoring",
        ),
        (
            "prediction-intervals",
            rep.plot_prediction_intervals(
                pd.DataFrame(
                    {
                        "predicted_mean": [0.15, 0.35, 0.72, 0.88],
                        "lower": [0.05, 0.20, 0.58, 0.76],
                        "upper": [0.28, 0.50, 0.84, 0.96],
                    }
                )
            ),
            "Prediction intervals",
            "plot_prediction_intervals()",
            "prediction-profiles-surfaces-and-contrasts",
        ),
        (
            "predictive-coverage",
            rep.plot_predictive_coverage(
                pd.DataFrame(
                    {"nominal_coverage": [0.5, 0.8, 0.95], "empirical_coverage": [0.53, 0.79, 0.93]}
                )
            ),
            "Predictive coverage",
            "plot_predictive_coverage()",
            "predictive-distribution-and-calibration-uncertainty",
        ),
        (
            "predictive-residuals",
            rep.plot_predictive_residuals(
                pd.DataFrame({"residual": [-0.4, 0.1, 0.25, -0.08, 0.03, 0.16]})
            ),
            "Predictive residuals",
            "plot_predictive_residuals()",
            "advanced-predictive-diagnostics",
        ),
    ]
    for name, fig, title, functions, article in predictive:
        out.append(_save(fig, root, "predictive", name, title, functions, article))

    rng = np.random.default_rng(20260829)
    draws = pd.DataFrame(
        {
            "b_0": rng.normal(0.0, 0.25, 500),
            "b_condition": rng.normal(0.55, 0.20, 500),
            "sigma": rng.lognormal(-0.05, 0.12, 500),
        }
    )
    interval = pd.DataFrame(
        {
            "variable": ["b_0", "b_condition", "sigma"],
            "lower": [-0.30, 0.12, 0.74],
            "median": [0.00, 0.55, 0.96],
            "upper": [0.31, 0.94, 1.22],
            "mean": [0.01, 0.56, 0.97],
            "sd": [0.15, 0.20, 0.13],
        }
    )
    posterior = [
        (
            "posterior-intervals",
            rep.plot_posterior_intervals(interval),
            "Posterior intervals",
            "plot_posterior_intervals()",
            "posterior-exploration-and-graphics",
        ),
        (
            "posterior-areas",
            rep.plot_posterior_areas(interval),
            "Posterior areas",
            "plot_posterior_areas()",
            "posterior-exploration-and-graphics",
        ),
        (
            "posterior-density",
            rep.plot_posterior_density(draws),
            "Posterior density",
            "plot_posterior_density()",
            "posterior-exploration-and-graphics",
        ),
        (
            "posterior-correlations",
            rep.plot_posterior_correlations(draws),
            "Posterior correlations",
            "plot_posterior_correlations()",
            "posterior-exploration-and-graphics",
        ),
        (
            "mcmc-quality",
            rep.plot_mcmc_quality(pd.DataFrame({"status": ["pass", "pass", "review", "pass"]})),
            "MCMC quality inventory",
            "plot_mcmc_quality()",
            "posterior-diagnostics",
        ),
    ]
    for name, fig, title, functions, article in posterior:
        out.append(_save(fig, root, "posterior", name, title, functions, article))

    loo = DummyLOO()
    metadata = pd.DataFrame({"participant": ["p1", "p1", "p2", "p2", "p3", "p3"]})
    atlas = loo_mod.create_loo_influence_atlas(loo, data=metadata)
    grouped = loo_mod.loo_group_influence_table(atlas.table, "participant")
    loo_items = [
        (
            "pointwise-elpd",
            loo_mod.plot_loo_pointwise_elpd(atlas),
            "Pointwise ELPD-LOO",
            "plot_loo_pointwise_elpd()",
            "loo-influence-atlas",
        ),
        (
            "pareto-vs-elpd",
            loo_mod.plot_loo_pareto_vs_elpd(atlas),
            "Pareto-k versus ELPD",
            "plot_loo_pareto_vs_elpd()",
            "loo-influence-atlas",
        ),
        (
            "influence-rank",
            loo_mod.plot_loo_influence_rank(atlas),
            "Ranked PSIS-LOO influence",
            "plot_loo_influence_rank()",
            "loo-influence-atlas",
        ),
        (
            "group-influence",
            loo_mod.plot_loo_group_influence(grouped),
            "Grouped PSIS-LOO influence",
            "plot_loo_group_influence()",
            "grouped-loo-influence",
        ),
        (
            "group-elpd",
            loo_mod.plot_loo_group_elpd(grouped),
            "Grouped predictive contribution",
            "plot_loo_group_elpd()",
            "grouped-loo-influence",
        ),
    ]
    for name, fig, title, functions, article in loo_items:
        out.append(_save(fig, root, "loo", name, title, functions, article))

    status = pd.DataFrame(
        {
            "component": ["prior scale", "group deletion", "alternative estimand", "SBC"],
            "status": ["pass", "review", "pass", "pass"],
        }
    )
    sensitivity = [
        (
            "sensitivity-suite",
            eg.plot_sensitivity_suite_gg({"table": status}),
            "Sensitivity suite",
            "plot_sensitivity_suite_gg()",
            "sensitivity-atlas",
        ),
        (
            "model-evidence",
            eg.plot_model_evidence_gg({"table": status}),
            "Model evidence inventory",
            "plot_model_evidence_gg()",
            "evidence-graphics-and-governance",
        ),
        (
            "backend-parity",
            eg.plot_backend_parity_gg(
                {
                    "table": pd.DataFrame(
                        {
                            "variable": ["b_0", "b_condition", "sigma"],
                            "reference_mean": [0.01, 0.55, 0.97],
                            "alternative_mean": [0.02, 0.53, 0.99],
                        }
                    )
                }
            ),
            "Backend posterior parity",
            "plot_backend_parity_gg()",
            "backend-reliability",
        ),
        (
            "manifest-comparison",
            eg.plot_manifest_comparison_gg(
                {
                    "table": pd.DataFrame(
                        {
                            "component": ["data", "contract", "transformations", "seed"],
                            "identical": [1, 1, 1, 0],
                        }
                    )
                }
            ),
            "Analysis-manifest comparison",
            "plot_manifest_comparison_gg()",
            "reproducible-analysis-manifests",
        ),
        (
            "design-support",
            eg.plot_design_support_gg(
                {
                    "table": pd.DataFrame(
                        {
                            "check": ["fixed-effect rank", "condition number", "group repetition"],
                            "status": ["pass", "review", "pass"],
                        }
                    )
                }
            ),
            "Design-support diagnostics",
            "plot_design_support_gg()",
            "pre-fit-design-diagnostics",
        ),
        (
            "missingness",
            eg.plot_missingness_gg(
                {
                    "table": pd.DataFrame(
                        {
                            "variable": ["pupil", "gaze_x", "gaze_y", "luminance"],
                            "missing_fraction": [0.02, 0.08, 0.04, 0.01],
                        }
                    )
                }
            ),
            "Missingness audit",
            "plot_missingness_gg()",
            "measurement-error-and-missing-pupil-data",
        ),
    ]
    for name, fig, title, functions, article in sensitivity:
        out.append(_save(fig, root, "sensitivity", name, title, functions, article))

    dashboard = rep.create_diagnostic_dashboard(
        loo={"status": "review"},
        sensitivity={"status": "pass"},
        recovery={"status": "pass"},
        sbc={"status": "pass"},
        label="Evidence dashboard",
    )
    publication = [
        (
            "diagnostic-dashboard",
            rep.plot_diagnostic_dashboard(dashboard),
            "Diagnostic dashboard",
            "plot_diagnostic_dashboard()",
            "publication-registries-and-dashboards",
        ),
        (
            "reporting-checklist",
            rep.plot_reporting_checklist(
                pd.DataFrame(
                    {
                        "item": ["Contract", "Diagnostics", "Sensitivity", "Manifest", "LOO"],
                        "available": [1, 1, 1, 1, 1],
                    }
                )
            ),
            "Reporting checklist",
            "plot_reporting_checklist()",
            "model-cards-and-reporting-inventories",
        ),
        (
            "model-comparison",
            rep.plot_model_comparison(
                pd.DataFrame({"model": ["M1", "M2", "M3"], "elpd_loo": [-105.2, -101.7, -103.1]})
            ),
            "Model comparison",
            "plot_model_comparison()",
            "end-to-end-evidence-showcase",
        ),
        (
            "model-weights",
            rep.plot_model_weights(
                pd.DataFrame({"model": ["M1", "M2", "M3"], "weight": [0.18, 0.62, 0.20]})
            ),
            "Model weights",
            "plot_model_weights()",
            "end-to-end-evidence-showcase",
        ),
        (
            "uncertainty-decomposition",
            rep.plot_uncertainty_decomposition(
                pd.DataFrame(
                    {
                        "epistemic_variance": [0.02, 0.04, 0.03, 0.05],
                        "residual_variance": [0.08, 0.07, 0.09, 0.06],
                        "total_variance": [0.10, 0.11, 0.12, 0.11],
                    }
                )
            ),
            "Uncertainty decomposition",
            "plot_uncertainty_decomposition()",
            "hierarchical-effects-and-uncertainty",
        ),
        (
            "group-effects",
            rep.plot_group_effects(
                pd.DataFrame({"level": ["A", "B", "C"], "median": [-0.2, 0.1, 0.45]})
            ),
            "Group effects",
            "plot_group_effects()",
            "hierarchical-effect-atlas",
        ),
        (
            "variance-components",
            rep.plot_variance_components(
                pd.DataFrame(
                    {"component": ["participant", "item", "residual"], "variance": [0.4, 0.2, 0.8]}
                )
            ),
            "Variance components",
            "plot_variance_components()",
            "hierarchical-effect-atlas",
        ),
    ]
    for name, fig, title, functions, article in publication:
        out.append(_save(fig, root, "publication", name, title, functions, article))
    return out


def check_manifest(root: Path = DEFAULT_OUTPUT):
    path = Path("dev/doc_figures_manifest.json")
    if not path.exists():
        raise SystemExit("dev/doc_figures_manifest.json is missing.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    for item in manifest:
        image = root / item["file"]
        if not image.exists() or image.stat().st_size < 2000:
            missing.append(str(image))
    if missing:
        raise SystemExit("Missing/invalid documentation figures:\n" + "\n".join(missing))
    print(f"Documentation figure manifest PASS: {len(manifest)} figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_manifest(args.output)
        return
    manifest = generate_all(args.output)
    Path("dev/doc_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(manifest)} documentation figures.")


if __name__ == "__main__":
    main()
