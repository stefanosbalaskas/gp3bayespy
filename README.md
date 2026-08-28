# gp3bayespy

**gp3bayespy** is the Python port of the R package **gp3bayes**, a contract-first Bayesian workflow package for repeated-measures, hierarchical behavioural data, posterior validation, predictive diagnostics, sensitivity analysis, and dynamic pupillometry.

> **Status:** **gp3bayespy 0.5.0** is the first public Python release with frozen parity against **gp3bayes 0.5.0**: 458/458 canonical exports implemented, 59/59 canonical articles ported, eight executable workflow examples, cross-platform CI, and a committed deep-freeze validation manifest.

## Frozen reference

The port is governed by the frozen CRAN source archive `gp3bayes_0.5.0.tar.gz`:

- 458 public exports
- 230 S3 registrations
- 60 R source files
- 465 Rd help files
- 59 vignette sources
- 54 `tests/testthat/test-*.R` test files plus the package-level test runner
- SHA-256: `537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a`

`dev/parity/function_map.csv` is the machine-readable 458-function ledger. `dev/parity/articles.json` tracks the 59-article documentation map.

## Release coverage

The current candidate includes the complete public namespace across:

- model contracts, readiness, formula/prior/specification closure;
- hierarchical binary and lognormal-duration simulation, preparation, fitting, diagnostics, PPC, prediction, sensitivity, recovery, and reporting;
- posterior extraction, sampler diagnostics, hierarchical effects, prior/posterior comparison, PSIS-LOO, influence diagnostics, model comparison, predictive scoring/calibration, surfaces, uncertainty and atlases;
- simulation-based calibration, power-scaling and governed optional Bayesian workflows;
- reproducibility manifests, analysis bundles, evidence inventories, model cards, publication registries and Matplotlib graphics;
- ordinary, advanced, binocular, Gaussian-process, temporal/ARMA, robust/distributional, missing-data/measurement-error, response-shape and model-comparison pupillometry workflows;
- all 59 mapped Python-facing articles and eight executable workflow examples.

The final closure tests require all 458 exports to be root-importable, all ledger rows to be `implemented`, all 59 articles to be present, and no frozen public function to expose an unrestricted `**kwargs` catchall.

## Installation

Core numerical functionality:

```bash
python -m pip install gp3bayespy
```

Bayesian backends and plotting:

```bash
python -m pip install "gp3bayespy[bayes,plots]"
```

Everything used by the completion/release gate:

```bash
python -m pip install "gp3bayespy[all]"
```

## Minimal example

```python
import pandas as pd
from gp3bayespy import (
    audit_model_readiness,
    create_model_contract,
    create_model_specification,
    create_prior_specification,
)

data = pd.DataFrame({
    "participant_id": ["p1"] * 4 + ["p2"] * 4,
    "trial_id": [1, 2, 3, 4] * 2,
    "condition": ["control", "treatment"] * 4,
    "selected": [0, 1, 0, 1, 1, 0, 1, 0],
})

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

See `examples/`, `docs/articles/`, `docs/migration.md`, and `docs/plot-gallery.md` for end-to-end workflows.

## Governance

Creating or fitting a model does **not** automatically establish convergence, model adequacy, causal identification, robustness, exclusion decisions, preferred-model status, or psychological/cognitive/emotional interpretation. The Python port preserves these boundaries from gp3bayes and makes automatic-selection/adequacy flags explicit where relevant.

## Python integrations

Core: NumPy, pandas, SciPy. Optional: PyMC, CmdStanPy, ArviZ/xarray and Matplotlib where the gp3bayes model contract can be preserved.
