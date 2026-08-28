# Executable examples

Eight small scripts exercise the main workflow families. They are intentionally compact enough to read in one sitting and concrete enough to run as smoke tests.

<div class="gp-example-grid">

<div class="gp-example-card">
  <span class="gp-card-tag">Foundation</span>
  <h3>Binary workflow</h3>
  <p>Simulation → preparation → specification → prior predictive checking.</p>
  <code>python examples/binary_workflow.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/binary_workflow.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Foundation</span>
  <h3>Duration workflow</h3>
  <p>Positive-duration simulation and lognormal workflow specification.</p>
  <code>python examples/duration_workflow.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/duration_workflow.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Diagnostics</span>
  <h3>Predictive diagnostics</h3>
  <p>Backend-free ROC, precision-recall, and calibration diagnostics.</p>
  <code>python examples/predictive_diagnostics.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/predictive_diagnostics.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Model comparison</span>
  <h3>PSIS-LOO comparison</h3>
  <p>Pointwise log-likelihood draws, Pareto-k diagnostics, and descriptive ELPD comparison.</p>
  <code>python examples/loo_model_comparison.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/loo_model_comparison.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Sensitivity</span>
  <h3>Sensitivity workflow</h3>
  <p>Declarative sensitivity planning with automatic model selection and exclusion disabled.</p>
  <code>python examples/sensitivity_workflow.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/sensitivity_workflow.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Reproducibility</span>
  <h3>Analysis manifest</h3>
  <p>Capture deterministic fingerprints for data, contract, specification, and transformations.</p>
  <code>python examples/reproducibility_workflow.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/reproducibility_workflow.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Pupillometry</span>
  <h3>Pupil workflow</h3>
  <p>Advanced simulation, governed analytic fitting, trajectory prediction, and temporal audit.</p>
  <code>python examples/pupil_workflow.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/pupil_workflow.py">View source →</a>
</div>

<div class="gp-example-card">
  <span class="gp-card-tag">Environment</span>
  <h3>Backend status</h3>
  <p>Inspect optional backend capability and environment state without compilation.</p>
  <code>python examples/backend_status.py</code>
  <a href="https://github.com/stefanosbalaskas/gp3bayespy/blob/main/examples/backend_status.py">View source →</a>
</div>

</div>

## Run from a source checkout

```bash
PYTHONPATH=src python examples/predictive_diagnostics.py
```

After installation, ordinary `python` is sufficient because `gp3bayespy` is importable from the active environment.

## Suggested order

1. `binary_workflow.py`
2. `predictive_diagnostics.py`
3. `loo_model_comparison.py`
4. `sensitivity_workflow.py`
5. `reproducibility_workflow.py`
6. `pupil_workflow.py`

For a guided narrative rather than scripts, continue to the [Article library](../articles/index.md).
