# Joint Binocular Pupil Models

> Python-facing port of `binocular-pupil-models.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

No averaged pupil column is created. Left and right eyes remain separate responses.

Residual eye correlation is an association parameter. High posterior correlation does not establish that the two eyes are interchangeable or justify arbitrary eye substitution.

## Python API mapping

- `gp3bayespy.audit_binocular_pupil_readiness`
- `gp3bayespy.estimate_binocular_pupil_trajectory`
- `gp3bayespy.fit_binocular_pupil_model`
- `gp3bayespy.plot_binocular_pupil_trajectory`
- `gp3bayespy.prepare_binocular_pupil_timecourse`
- `gp3bayespy.pupil_binocular_agreement_table`
- `gp3bayespy.pupil_binocular_correlation`
- `gp3bayespy.pupil_binocular_difference`
- `gp3bayespy.simulate_binocular_pupil_timecourse`
- `gp3bayespy.specify_binocular_pupil_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
