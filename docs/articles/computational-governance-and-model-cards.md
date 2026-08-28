# Computational Governance and Model Cards

> Python-facing port of `computational-governance-and-model-cards.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Complexity is audited before Stan

The complexity gate is not a statistical adequacy test. It is a reproducible guard against accidentally requesting models that combine many expensive layers or exact Gaussian processes over very large grids.

# Model card

A model card records family, temporal structure, residual scale, autocorrelation, data dimensions, measurement/missingness declarations, predictive target, complexity status, and governance text. It is designed to support methods supplements and audit trails without becoming a validity certificate.

## Python API mapping

- `gp3bayespy.audit_pupil_computational_budget`
- `gp3bayespy.create_pupil_gp_spec`
- `gp3bayespy.plot_pupil_model_complexity`
- `gp3bayespy.pupil_model_card`
- `gp3bayespy.pupil_model_card_table`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
