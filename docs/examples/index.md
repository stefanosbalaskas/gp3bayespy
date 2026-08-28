# Executable examples

The repository includes runnable scripts under `examples/`:

- `binary_workflow.py` — simulation, preparation, specification, prior predictive checking.
- `duration_workflow.py` — duration simulation, preparation, specification, prior predictive checking.
- `pupil_workflow.py` — advanced pupil simulation, governed analytic fitting, prediction, temporal audit.
- `loo_model_comparison.py` — direct PSIS-LOO from pointwise log-likelihood draws.
- `predictive_diagnostics.py` — ROC, precision-recall, and calibration diagnostics without a fitting backend.
- `sensitivity_workflow.py` — declarative sensitivity planning with automatic decisions disabled.
- `reproducibility_workflow.py` — deterministic analysis-manifest capture.
- `backend_status.py` — backend capability/environment inspection without compilation.

Run from a source checkout with:

```bash
PYTHONPATH=src python examples/pupil_workflow.py
```

After installation, the same scripts run with ordinary `python` because `gp3bayespy` is importable from the environment.
