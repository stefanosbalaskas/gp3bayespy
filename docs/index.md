# gp3bayespy

**Contract-first Bayesian workflows for hierarchical behavioural data in Python**

[Get started](#quick-start){ .md-button .md-button--primary }
[API reference](api.md){ .md-button }
[Browse 59 articles](articles/index.md){ .md-button }

!!! success "gp3bayespy 0.5.0 is released"

    The first public Python release is frozen against **gp3bayes 0.5.0** with **458/458 canonical exports implemented**, **59/59 articles ported**, and the full cross-platform release gate passing.

`gp3bayespy` brings the governed modelling contracts of the R package `gp3bayes` into Python. It is designed for repeated-measures and hierarchical behavioural data where the analysis needs to remain explicit about priors, model specification, diagnostics, predictive evidence, sensitivity, and interpretation.

<div class="grid cards" markdown>

- **Contract-first modelling**

  Declare outcome, grouping, condition, priors, and model family before fitting. Readiness and specification checks remain explicit.

- **Posterior & predictive validation**

  Work with sampler diagnostics, PPC, calibration, scoring, PSIS-LOO, influence diagnostics, model comparison, and uncertainty summaries.

- **Sensitivity & recovery**

  Run prior-scale sensitivity, power-scaling, group deletion, alternative estimands, SBC, and parameter-recovery workflows.

- **Dynamic pupillometry**

  Analyse baseline/gaze/luminance sensitivity, time courses, temporal dependence, binocular responses, Gaussian-process trajectories, robust models, and missing data.

- **Reproducible evidence**

  Create analysis manifests, model cards, evidence inventories, publication bundles, registries, and transformation-replay records.

- **Governed conclusions**

  The package does not automatically select a preferred model, exclude participants, certify adequacy, establish causality, or infer cognitive/emotional states.

</div>

## Install

=== "Core"

    ```bash
    python -m pip install gp3bayespy
    ```

=== "Bayesian + plots"

    ```bash
    python -m pip install "gp3bayespy[bayes,plots]"
    ```

=== "Complete environment"

    ```bash
    python -m pip install "gp3bayespy[all]"
    ```

Core workflows use NumPy, pandas, and SciPy. PyMC, CmdStanPy, ArviZ/xarray, and Matplotlib are optional extras.

## Quick start

```python
import pandas as pd
from gp3bayespy import (
    audit_model_readiness,
    create_model_contract,
    create_model_specification,
    create_prior_specification,
)

data = pd.DataFrame(
    {
        "participant_id": ["p1"] * 4 + ["p2"] * 4,
        "trial_id": [1, 2, 3, 4] * 2,
        "condition": ["control", "treatment"] * 4,
        "selected": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

contract = create_model_contract(
    family="binary",
    outcome_col="selected",
    participant_col="participant_id",
    trial_col="trial_id",
    condition_col="condition",
)

audit = audit_model_readiness(data, contract)
priors = create_prior_specification(contract, baseline=0.5)
specification = create_model_specification(contract, audit, priors)

print(specification.formula_text)
```

The workflow deliberately separates **readiness → prior declaration → specification → fitting → diagnostics → estimands → sensitivity**.

## Choose a workflow

| Goal | Documentation |
| --- | --- |
| Hierarchical binary models | [Binary end-to-end](articles/binary-end-to-end.md) |
| Positive duration models | [Duration end-to-end](articles/duration-end-to-end.md) |
| Posterior diagnostics | [Posterior diagnostics](articles/posterior-diagnostics.md) |
| Predictive diagnostics | [Advanced predictive diagnostics](articles/advanced-predictive-diagnostics.md) |
| PSIS-LOO and model comparison | [LOO influence and model comparison](articles/loo-influence-and-model-comparison.md) |
| Sensitivity analysis | [Unified sensitivity suites](articles/sensitivity-evidence-workflow.md) |
| Dynamic pupillometry | [Bayesian dynamic pupillometry](articles/bayesian-dynamic-pupillometry.md) |
| Reproducible analysis | [Analysis manifests](articles/reproducible-analysis-manifests.md) |

## Explore

<div class="grid cards" markdown>

- **[API reference](api.md)**

  Public Python API and signatures.

- **[59 articles](articles/index.md)**

  Python-facing ports of the canonical gp3bayes 0.5.0 vignettes.

- **[Executable examples](examples/index.md)**

  Eight end-to-end scripts covering the main workflow families.

- **[Plot gallery](plot-gallery.md)**

  Publication-oriented evidence graphics.

- **[Migration guide](migration.md)**

  R-to-Python guidance for gp3bayes users.

- **[Parity record](development/parity.md)**

  Frozen R reference, function ledger, and closure rules.

</div>

## Release validation

| Gate | gp3bayespy 0.5.0 |
| --- | ---: |
| Frozen exports | **458 / 458** |
| Canonical articles | **59 / 59** |
| Tests | **321 / 321** |
| Branch-aware coverage | **47.9482%** |
| Public unrestricted `**kwargs` | **0** |
| Source examples | **8 / 8** |
| Installed-wheel examples | **8 / 8** |
| Ruff / mypy | **PASS / PASS** |
| Cross-platform CI | **PASS** |
| Strict docs | **PASS** |

See the [R → Python parity record](development/parity.md) for the machine-readable closure evidence.

## Governance

!!! warning "Evidence is not an automatic conclusion"

    Creating or fitting a model does **not** automatically establish convergence, model adequacy, causal identification, robustness, exclusion decisions, preferred-model status, or psychological/cognitive/emotional interpretation. Those remain analyst decisions supported by the relevant diagnostics and evidence.

## Project links

- [GitHub repository](https://github.com/stefanosbalaskas/gp3bayespy)
- [PyPI package](https://pypi.org/project/gp3bayespy/)
- [v0.5.0 release](https://github.com/stefanosbalaskas/gp3bayespy/releases/tag/v0.5.0)
- [R reference package](https://cran.r-project.org/package=gp3bayes)
