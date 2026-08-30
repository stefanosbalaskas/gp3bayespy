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
      <a class="gp-btn gp-btn--primary" href="/gp3bayespy/getting-started/">Get started</a>
      <a class="gp-btn gp-btn--secondary" href="/gp3bayespy/reference/">Browse the API</a>
      <a class="gp-btn gp-btn--ghost" href="/gp3bayespy/articles/">Explore 59 articles</a>
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
  <div class="gp-stat"><strong>689 / 689</strong><span>current main tests passing</span></div>
  <div class="gp-stat"><strong>100.00%</strong><span>current main branch coverage</span></div>
  <div class="gp-stat"><strong>3 OS × 3 Py</strong><span>Python 3.11 · 3.12 · 3.13 CI</span></div>
</div>

## Choose your path

<div class="gp-paths">
  <a class="gp-path" href="/gp3bayespy/articles/binary-end-to-end/">
    <span class="gp-path__eyebrow">Behavioural models</span>
    <h3>Binary & duration workflows</h3>
    <p>Simulate, prepare, specify, fit, diagnose, predict, recover, and report hierarchical outcomes.</p>
    <strong>Start a model →</strong>
  </a>

  <a class="gp-path" href="/gp3bayespy/articles/advanced-predictive-diagnostics/">
    <span class="gp-path__eyebrow">Predictive evidence</span>
    <h3>Diagnostics, calibration & PSIS-LOO</h3>
    <p>Inspect out-of-sample performance, calibration, uncertainty, influence, and comparison without automatic selection.</p>
    <strong>Inspect evidence →</strong>
  </a>

  <a class="gp-path" href="/gp3bayespy/articles/bayesian-dynamic-pupillometry/">
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
  <a href="/gp3bayespy/plot-gallery/">
    <strong>Plot gallery</strong>
    <span>Real graphics generated from gp3bayespy workflows →</span>
  </a>
  <a href="/gp3bayespy/examples/">
    <strong>8 executable examples</strong>
    <span>Small scripts that run end to end →</span>
  </a>
  <a href="/gp3bayespy/articles/">
    <strong>59 guided articles</strong>
    <span>Browse by workflow instead of filename →</span>
  </a>
  <a href="/gp3bayespy/reference/">
    <strong>API reference hub</strong>
    <span>458 functions organized by module →</span>
  </a>
</div>

## Current main confidence

<div class="gp-release-strip">
  <div><span>Tests</span><strong>689 / 689</strong></div>
  <div><span>Coverage</span><strong>100.00%</strong></div>
  <div><span>Exclusions</span><strong>0</strong></div>
  <div><span>CI matrix</span><strong>3 OS × 3 Py</strong></div>
  <div><span>Static gates</span><strong>Ruff + mypy PASS</strong></div>
</div>

These are the validation metrics for the current `main` branch after the exact-coverage hardening campaign. The historical **v0.5.0** release remains immutable and retains its original release-time validation record.

### Frozen v0.5.0 release record

<div class="gp-release-strip">
  <div><span>Release</span><strong>0.5.0</strong></div>
  <div><span>Release tests</span><strong>321 / 321</strong></div>
  <div><span>Release coverage</span><strong>47.9482%</strong></div>
  <div><span>Parity</span><strong>458 / 458</strong></div>
  <div><span>Public `**kwargs`</span><strong>0</strong></div>
</div>

The release is frozen against **gp3bayes 0.5.0**. The full validation record is available in [Release v0.5.0](release.md) and the machine-readable closure evidence is documented under [R → Python parity](development/parity.md).

Archived software DOI: [**10.5281/zenodo.22150746**](https://doi.org/10.5281/zenodo.22150746). See [Citing gp3bayespy](citation.md) for citation-ready metadata.

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
    <a class="gp-btn gp-btn--primary" href="/gp3bayespy/getting-started/">Getting started</a>
    <a class="gp-btn gp-btn--secondary" href="/gp3bayespy/examples/">Run an example</a>
  </div>
</div>
