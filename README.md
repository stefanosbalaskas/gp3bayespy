# gp3bayespy

**gp3bayespy** is the Python port of the R package **gp3bayes**, a contract-first Bayesian workflow package for repeated-measures and hierarchical behavioural data.

> **Status:** pre-alpha parity port. The frozen reference is **gp3bayes 0.5.0**. The package is not yet a substitute for the complete R release.

## Frozen reference

The initial port is governed by the uploaded CRAN source archive `gp3bayes_0.5.0.tar.gz`:

- 458 public exports
- 230 S3 registrations
- 60 R source files
- 465 Rd help files
- 59 vignette sources
- 54 `tests/testthat/test-*.R` test files plus the package-level test runner
- SHA-256: `537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a`

`dev/parity/function_map.csv` is the machine-readable 458-function ledger.

## Current first tranche

The repository already contains a functional backend-independent foundation for:

- `create_model_contract()`
- `audit_model_readiness()` (initial parity implementation; edge-case warning parity remains open)
- `build_model_formula()`
- `create_prior_specification()`
- `validate_prior_specification()`
- `create_model_specification()`
- `backend_capabilities()`

No fitting backend is imported by the core package.

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
spec = create_model_specification(contract, audit, priors)
print(spec.formula_text)
```

## Governance

Creating or fitting a model does not automatically establish convergence, model adequacy, causal identification, robustness, exclusion decisions, or psychological interpretation. The Python port preserves these boundaries from gp3bayes.

## Planned Python integrations

Core: NumPy, pandas, SciPy. Optional posterior/fitting layers: PyMC, CmdStanPy, ArviZ/xarray, and later NumPyro/JAX only where the gp3bayes model contract can be preserved.
