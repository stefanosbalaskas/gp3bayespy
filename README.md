<div align="center">

# gp3bayespy

**Contract-first Bayesian workflows for hierarchical behavioural data in Python**

[![PyPI](https://img.shields.io/pypi/v/gp3bayespy.svg)](https://pypi.org/project/gp3bayespy/)
[![Python](https://img.shields.io/pypi/pyversions/gp3bayespy.svg)](https://pypi.org/project/gp3bayespy/)
[![CI](https://github.com/stefanosbalaskas/gp3bayespy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stefanosbalaskas/gp3bayespy/actions/workflows/ci.yml)
[![Docs](https://github.com/stefanosbalaskas/gp3bayespy/actions/workflows/docs.yml/badge.svg?branch=main)](https://stefanosbalaskas.github.io/gp3bayespy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![R reference](https://img.shields.io/badge/R%20reference-gp3bayes%200.5.0-276DC3.svg)](https://cran.r-project.org/package=gp3bayes)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22150746.svg)](https://doi.org/10.5281/zenodo.22150746)

[Documentation](https://stefanosbalaskas.github.io/gp3bayespy/) ·
[PyPI](https://pypi.org/project/gp3bayespy/) ·
[Examples](examples/) ·
[Articles](docs/articles/) ·
[Migration guide](docs/migration.md) ·
[Release v0.5.0](https://github.com/stefanosbalaskas/gp3bayespy/releases/tag/v0.5.0) ·
[DOI](https://doi.org/10.5281/zenodo.22150746)

</div>

`gp3bayespy` is the Python port of the R package [`gp3bayes`](https://cran.r-project.org/package=gp3bayes). It provides governed Bayesian workflows for repeated-measures and hierarchical behavioural data, with first-class support for model contracts, posterior and predictive diagnostics, sensitivity analysis, PSIS-LOO model comparison, reproducibility, and dynamic pupillometry.

Version **0.5.0** is the first public Python release frozen against **gp3bayes 0.5.0**.

## Why gp3bayespy?

| Capability | What it provides |
| --- | --- |
| **Contract-first modelling** | Explicit outcome, grouping, condition, prior, model-family, and analysis contracts before fitting. |
| **Hierarchical Bayesian workflows** | Binary and positive lognormal-duration simulation, preparation, fitting, prediction, PPC, recovery, and reporting. |
| **Posterior & predictive validation** | Sampler diagnostics, calibration, scoring, uncertainty summaries, PSIS-LOO, influence diagnostics, and model comparison. |
| **Sensitivity & robustness** | Prior-scale sensitivity, power-scaling, group deletion, alternative estimands, SBC, and recovery diagnostics. |
| **Dynamic pupillometry** | Baseline/gaze/luminance sensitivity, time-course models, temporal diagnostics, binocular models, GP trajectories, robust/distributional models, and missing-data workflows. |
| **Reproducible evidence** | Analysis manifests, model cards, publication bundles, evidence inventories, registries, dashboards, and transformation replay. |
| **Governed outputs** | No automatic model selection, participant exclusion, adequacy declaration, causal claim, or psychological-state inference. |

## Installation

### Core

```bash
python -m pip install gp3bayespy
```

### Bayesian backends and plotting

```bash
python -m pip install "gp3bayespy[bayes,plots]"
```

### Complete environment

```bash
python -m pip install "gp3bayespy[all]"
```

The core installation contains NumPy, pandas, and SciPy workflows. PyMC, CmdStanPy, ArviZ/xarray, and Matplotlib are optional and installed through the corresponding extras.

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

The package separates **data readiness**, **prior declaration**, **model specification**, **fitting**, **diagnostics**, **estimands**, and **sensitivity** rather than hiding those decisions inside a single convenience call.

## Workflow map

| Goal | Start with |
| --- | --- |
| Build a binary hierarchical workflow | [`examples/binary_workflow.py`](examples/binary_workflow.py) · [article](docs/articles/binary-end-to-end.md) |
| Build a duration workflow | [`examples/duration_workflow.py`](examples/duration_workflow.py) · [article](docs/articles/duration-end-to-end.md) |
| Inspect posterior/predictive diagnostics | [`examples/predictive_diagnostics.py`](examples/predictive_diagnostics.py) · [article](docs/articles/advanced-predictive-diagnostics.md) |
| Compare models with PSIS-LOO | [`examples/loo_model_comparison.py`](examples/loo_model_comparison.py) · [article](docs/articles/loo-influence-and-model-comparison.md) |
| Run sensitivity analyses | [`examples/sensitivity_workflow.py`](examples/sensitivity_workflow.py) · [article](docs/articles/sensitivity-evidence-workflow.md) |
| Analyse pupil time courses | [`examples/pupil_workflow.py`](examples/pupil_workflow.py) · [pupillometry articles](docs/articles/index.md) |
| Record reproducible analysis state | [`examples/reproducibility_workflow.py`](examples/reproducibility_workflow.py) · [article](docs/articles/reproducible-analysis-manifests.md) |
| Check optional backend status | [`examples/backend_status.py`](examples/backend_status.py) · [backend guide](docs/articles/backend-portability.md) |

## Documentation

The documentation site is the primary user guide:

- **[Getting started](https://stefanosbalaskas.github.io/gp3bayespy/)** — package orientation and first workflow.
- **[API reference](https://stefanosbalaskas.github.io/gp3bayespy/reference/)** — public Python API.
- **[59 articles](https://stefanosbalaskas.github.io/gp3bayespy/articles/)** — Python-facing ports of the canonical gp3bayes vignettes.
- **[Executable examples](https://stefanosbalaskas.github.io/gp3bayespy/examples/)** — eight end-to-end scripts.
- **[Plot gallery](https://stefanosbalaskas.github.io/gp3bayespy/plot-gallery/)** — publication-oriented plotting workflows.
- **[Migration guide](https://stefanosbalaskas.github.io/gp3bayespy/migration/)** — R-to-Python mapping for gp3bayes users.

## Parity and validation

The 0.5.0 release is frozen against the CRAN source archive `gp3bayes_0.5.0.tar.gz`.

| Validation item | Release state |
| --- | ---: |
| Frozen R exports | **458 / 458 implemented** |
| Canonical articles | **59 / 59 ported** |
| Public unrestricted `**kwargs` | **0** |
| Released v0.5.0 test suite | **321 / 321 passing** |
| Current `main` measured test suite | **350 / 350 passing** |
| Released v0.5.0 branch-aware coverage | **47.9482%** |
| Current `main` branch-aware coverage | **68.06%** |
| Source examples | **8 / 8** |
| Installed-wheel examples | **8 / 8** |
| Ruff / mypy | **PASS / PASS** |
| Linux / macOS / Windows CI | **PASS** |
| Strict documentation build | **PASS** |

The machine-readable parity ledger is [`dev/parity/function_map.csv`](dev/parity/function_map.csv), the article map is [`dev/parity/articles.json`](dev/parity/articles.json), and the final freeze record is [`dev/parity/final_deep_freeze_0.5.0.json`](dev/parity/final_deep_freeze_0.5.0.json).

<details>
<summary><strong>Frozen R reference details</strong></summary>

The reference archive contains 458 public exports, 230 S3 registrations, 60 R source files, 465 Rd help files, 59 vignette sources, and 54 `tests/testthat/test-*.R` files plus the package-level runner.

Reference archive SHA-256:

```text
537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a
```

</details>

## Governance

`gp3bayespy` intentionally separates statistical evidence from analyst conclusions. Creating or fitting a model does **not** automatically establish convergence, model adequacy, causal identification, robustness, a preferred model, participant/group exclusion, or psychological/cognitive/emotional interpretation.

Automatic-selection and automatic-adequacy fields remain explicit and conservative where relevant.

## Citation

If you use `gp3bayespy`, cite the archived software record for the version you used. The DOI for **gp3bayespy 0.5.0** is [`10.5281/zenodo.22150746`](https://doi.org/10.5281/zenodo.22150746). Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff). Cite the `gp3bayes` R reference where it is methodologically relevant.

## License

MIT License. See [`LICENSE`](LICENSE).
