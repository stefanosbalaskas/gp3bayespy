---
hide:
  - navigation
  - toc
---

<div class="gp-hero">
  <div class="gp-hero__copy">
    <div class="gp-kicker">gp3bayes 0.5.0 → Python · released</div>
    <h1>Bayesian workflows that keep every decision visible.</h1>
    <p class="gp-lead">
      Contract-first modelling for repeated-measures and hierarchical behavioural data,
      with predictive diagnostics, sensitivity analysis, reproducibility, and dynamic pupillometry.
    </p>
    <div class="gp-actions">
      <a class="gp-btn gp-btn--primary" href="getting-started.md">Get started</a>
      <a class="gp-btn gp-btn--secondary" href="reference/index.md">Browse the API</a>
      <a class="gp-btn gp-btn--ghost" href="articles/index.md">Explore 59 articles</a>
    </div>
    <div class="gp-chip-row">
      <span>Python ≥ 3.11</span>
      <span>PyMC</span>
      <span>CmdStanPy</span>
      <span>ArviZ</span>
      <span>Matplotlib</span>
    </div>
  </div>

  <div class="gp-code-panel">
    <div class="gp-code-panel__bar">
      <span></span><span></span><span></span>
      <strong>first_model.py</strong>
    </div>
    <pre><code>from gp3bayespy import (
    create_model_contract,
    audit_model_readiness,
)

contract = create_model_contract(
    family="binary",
    outcome_col="selected",
    participant_col="participant_id",
    condition_col="condition",
)

audit = audit_model_readiness(data, contract)
print(audit.status)</code></pre>
  </div>
</div>

<div class="gp-stats">
  <div class="gp-stat"><strong>458 / 458</strong><span>frozen R exports implemented</span></div>
  <div class="gp-stat"><strong>59 / 59</strong><span>canonical articles ported</span></div>
  <div class="gp-stat"><strong>321 / 321</strong><span>release tests passing</span></div>
  <div class="gp-stat"><strong>8 / 8</strong><span>executable examples</span></div>
  <div class="gp-stat"><strong>3 OS</strong><span>Linux · macOS · Windows CI</span></div>
</div>

## Choose your path

<div class="gp-paths">
  <a class="gp-path" href="articles/binary-end-to-end.md">
    <span class="gp-path__eyebrow">Behavioural models</span>
    <h3>Binary & duration workflows</h3>
    <p>Simulate, prepare, specify, fit, diagnose, predict, recover, and report hierarchical outcomes.</p>
    <strong>Start a model →</strong>
  </a>

  <a class="gp-path" href="articles/advanced-predictive-diagnostics.md">
    <span class="gp-path__eyebrow">Predictive evidence</span>
    <h3>Diagnostics, calibration & PSIS-LOO</h3>
    <p>Inspect out-of-sample performance, calibration, uncertainty, influence, and comparison without automatic selection.</p>
    <strong>Inspect evidence →</strong>
  </a>

  <a class="gp-path" href="articles/bayesian-dynamic-pupillometry.md">
    <span class="gp-path__eyebrow">Time-course modelling</span>
    <h3>Dynamic pupillometry</h3>
    <p>Model pupil trajectories with explicit measurement context, temporal diagnostics, binocular extensions, GP models, and sensitivity workflows.</p>
    <strong>Explore pupil workflows →</strong>
  </a>
</div>

## One workflow, explicit gates

<div class="gp-flow">
  <div><strong>1</strong><span>Contract</span></div>
  <div><strong>2</strong><span>Prepare</span></div>
  <div><strong>3</strong><span>Specify</span></div>
  <div><strong>4</strong><span>Fit</span></div>
  <div><strong>5</strong><span>Diagnose</span></div>
  <div><strong>6</strong><span>Estimate</span></div>
  <div><strong>7</strong><span>Sensitivity</span></div>
</div>

`gp3bayespy` deliberately keeps these stages separate. A fitted model is not treated as proof of convergence, adequacy, robustness, causal identification, or substantive interpretation.

## Install in seconds

=== "Core"

    ```bash
    python -m pip install gp3bayespy
    ```

    NumPy, pandas, and SciPy workflows with no Bayesian backend required.

=== "Bayesian + plots"

    ```bash
    python -m pip install "gp3bayespy[bayes,plots]"
    ```

    Adds PyMC, CmdStanPy, ArviZ/xarray, and Matplotlib integrations.

=== "Everything"

    ```bash
    python -m pip install "gp3bayespy[all]"
    ```

    Reproduces the complete development, documentation, plotting, and validation environment.

[Open the installation guide →](getting-started.md#installation)

## What is inside?

<div class="grid cards" markdown>

-   **Contracts & readiness**

    Define the model family, mappings, assumptions, priors, and readiness gates before fitting.

    [Getting started →](getting-started.md)

-   **Posterior & predictive validation**

    Sampler diagnostics, posterior predictive checks, ROC/PR, calibration, scoring, uncertainty, and prediction surfaces.

    [Predictive workflows →](articles/advanced-predictive-diagnostics.md)

-   **PSIS-LOO & influence**

    Pointwise and grouped influence, Pareto-*k* diagnostics, ELPD comparison, and governed model weights.

    [LOO workflows →](articles/loo-influence-and-model-comparison.md)

-   **Sensitivity & recovery**

    Prior-scale sensitivity, power-scaling, structural alternatives, group deletion, SBC, and parameter recovery.

    [Sensitivity workflows →](articles/sensitivity-evidence-workflow.md)

-   **Dynamic pupillometry**

    Preparation, baseline/gaze/luminance sensitivity, time-course fitting, temporal validation, binocular models, GP trajectories, and robust extensions.

    [Pupillometry →](articles/bayesian-dynamic-pupillometry.md)

-   **Reproducible evidence**

    Analysis manifests, model cards, evidence inventories, publication bundles, registries, dashboards, and transformation replay.

    [Reproducibility →](articles/reproducible-analysis-manifests.md)

</div>

## See the package, not just the API

<div class="gp-showcase">
  <a href="plot-gallery.md">
    <strong>Plot gallery</strong>
    <span>Real graphics generated from gp3bayespy workflows →</span>
  </a>
  <a href="examples/index.md">
    <strong>8 executable examples</strong>
    <span>Small scripts that run end to end →</span>
  </a>
  <a href="articles/index.md">
    <strong>59 guided articles</strong>
    <span>Browse by workflow instead of filename →</span>
  </a>
  <a href="reference/index.md">
    <strong>API reference hub</strong>
    <span>458 functions organized by module →</span>
  </a>
</div>

## Release confidence

<div class="gp-release-strip">
  <div><span>Release</span><strong>0.5.0</strong></div>
  <div><span>Parity</span><strong>458 / 458</strong></div>
  <div><span>Coverage</span><strong>47.9482%</strong></div>
  <div><span>Public `**kwargs`</span><strong>0</strong></div>
  <div><span>Static gates</span><strong>Ruff + mypy PASS</strong></div>
</div>

The release is frozen against **gp3bayes 0.5.0**. The full validation record is available in [Release v0.5.0](release.md) and the machine-readable closure evidence is documented under [R → Python parity](development/parity.md).

!!! warning "Evidence is not an automatic conclusion"

    `gp3bayespy` does not automatically select a preferred model, exclude participants,
    certify adequacy, establish causality, or infer cognitive or emotional states.
    Those remain analyst decisions supported by the relevant evidence.

<div class="gp-bottom-cta">
  <div>
    <span class="gp-kicker">Ready to explore?</span>
    <h2>Start with a complete workflow.</h2>
  </div>
  <div class="gp-actions">
    <a class="gp-btn gp-btn--primary" href="getting-started.md">Getting started</a>
    <a class="gp-btn gp-btn--secondary" href="examples/index.md">Run an example</a>
  </div>
</div>
