from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "backend_status.py",
    "binary_workflow.py",
    "duration_workflow.py",
    "predictive_diagnostics.py",
    "pupil_workflow.py",
    "loo_model_comparison.py",
    "sensitivity_workflow.py",
    "reproducibility_workflow.py",
)


def test_all_public_examples_execute_under_coverage():
    for name in EXAMPLES:
        namespace = runpy.run_path(str(ROOT / "examples" / name), run_name=f"coverage_{name}")
        assert isinstance(namespace, dict)
