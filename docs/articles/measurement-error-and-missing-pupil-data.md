# Measurement Uncertainty and Missing Pupil Data

> Python-facing port of `measurement-error-and-missing-pupil-data.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Measurement uncertainty is declared, not silently corrected

The MAR label is an assumption required for this model class. Neither the audit nor a successful model fit proves that MAR holds.

# Joint brms translation

Predictor uncertainty is represented through latent `mi()` submodels. When modeled response missingness and known response uncertainty are declared together, the response uses the single `mi(sdy = ...)` mechanism so missingness and known measurement SD are represented coherently; without modeled response missingness, known response SD uses `se(..., sigma = TRUE)`. gp3bayespy (R reference 0.5) does not implement MNAR selection or pattern-mixture models.

## Python API mapping

- `gp3bayespy.audit_pupil_measurement_model`
- `gp3bayespy.audit_pupil_missingness`
- `gp3bayespy.create_pupil_measurement_model`
- `gp3bayespy.create_pupil_missingness_spec`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.plot_pupil_measurement_uncertainty`
- `gp3bayespy.plot_pupil_missingness`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`
- `gp3bayespy.translate_advanced_pupil_model_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
