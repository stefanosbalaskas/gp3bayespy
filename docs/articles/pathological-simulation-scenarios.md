# Pathological Simulation Scenarios

> Python-facing port of `pathological-simulation-scenarios.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The dedicated pathology generators make failure cases reproducible instead of
leaving them as prose-only limitations.

## Binary scenarios

A rank-deficient design is an explicit structural failure:

## Duration scenarios

Censoring and incorrect measurement units are semantic contract failures even
when their numeric values could otherwise pass a simple range check.

Heavy tails and mixtures are adequacy stress tests. They do not trigger an
automatic switch to another likelihood.

## Python API mapping

- `gp3bayespy.evaluate_pathological_simulation`
- `gp3bayespy.simulate_binary_pathology`
- `gp3bayespy.simulate_duration_pathology`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
