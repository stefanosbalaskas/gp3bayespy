# Getting started

`gp3bayespy` is designed around explicit analysis stages rather than a single opaque fitting call. This page gets you from installation to an inspectable model specification and shows where to go next.

## Installation

Choose the smallest environment that matches your task.

=== "Core"

    ```bash
    python -m pip install gp3bayespy
    ```

    Use this for contracts, readiness, simulation, preparation, backend-free diagnostics, reproducibility, and many analytic workflows.

=== "Bayesian + plots"

    ```bash
    python -m pip install "gp3bayespy[bayes,plots]"
    ```

    Adds optional Bayesian backends and Matplotlib integration.

=== "Complete"

    ```bash
    python -m pip install "gp3bayespy[all]"
    ```

    Matches the broad development and documentation environment.

Verify the installation:

```python
import gp3bayespy as gp

print(gp.__version__)
```

Expected release version:

```text
0.5.0
```

## The contract-first workflow

<div class="gp-flow gp-flow--compact">
  <div><strong>1</strong><span>Contract</span></div>
  <div><strong>2</strong><span>Audit</span></div>
  <div><strong>3</strong><span>Priors</span></div>
  <div><strong>4</strong><span>Specify</span></div>
  <div><strong>5</strong><span>Fit</span></div>
  <div><strong>6</strong><span>Diagnose</span></div>
  <div><strong>7</strong><span>Estimate</span></div>
</div>

### 1. Declare the model contract

```python
from gp3bayespy import create_model_contract

contract = create_model_contract(
    family="binary",
    outcome_col="selected",
    participant_col="participant_id",
    trial_col="trial_id",
    condition_col="condition",
)
```

The contract records the approved model family and neutral column mappings. It does not fit a model.

### 2. Audit readiness

```python
from gp3bayespy import audit_model_readiness

audit = audit_model_readiness(data, contract)
print(audit.status)
```

Readiness checks are evidence about whether the declared workflow can proceed. They are not substantive adequacy claims.

### 3. Declare priors

```python
from gp3bayespy import create_prior_specification

priors = create_prior_specification(
    contract,
    baseline=0.5,
)
```

### 4. Close the specification

```python
from gp3bayespy import create_model_specification

specification = create_model_specification(
    contract,
    audit,
    priors,
)

print(specification.formula_text)
```

At this point the data mappings, readiness evidence, prior declaration, and model structure are inspectable before fitting.

## Where to go next

<div class="grid cards" markdown>

-   **Binary outcomes**

    Continue through simulation, preparation, fitting, posterior diagnostics, PPC, prediction, sensitivity, and recovery.

    [Binary end-to-end →](articles/binary-end-to-end.md)

-   **Positive durations**

    Use the lognormal-duration workflow for strictly positive uncensored duration outcomes.

    [Duration end-to-end →](articles/duration-end-to-end.md)

-   **Predictive evidence**

    Work with scoring, calibration, ROC/PR, predictive uncertainty, PSIS-LOO, and model comparison.

    [Predictive diagnostics →](articles/advanced-predictive-diagnostics.md)

-   **Dynamic pupil responses**

    Start with measurement context, preparation, temporal structure, and explicit pupil estimands.

    [Dynamic pupillometry →](articles/bayesian-dynamic-pupillometry.md)

</div>

## Check optional backends

The package can inspect optional backend availability without compiling a model:

```python
import gp3bayespy as gp

print(gp.backend_capabilities())
```

For installation and portability details, see [Backend Portability and Installation](articles/backend-portability.md).

## Run the examples

The repository ships eight executable workflow scripts. Browse them on the [Executable examples](examples/index.md) page or clone the repository and run:

```bash
python examples/binary_workflow.py
python examples/predictive_diagnostics.py
python examples/pupil_workflow.py
```

## Interpretation boundaries

!!! warning

    Passing readiness checks, fitting a model, or obtaining favourable diagnostics does not automatically establish convergence, adequacy, robustness, causal identification, preferred-model status, exclusion decisions, or psychological interpretation.

See [Governance](governance.md) for the package-wide boundaries.
