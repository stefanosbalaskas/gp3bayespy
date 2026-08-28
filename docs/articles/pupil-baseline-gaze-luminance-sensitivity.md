# Baseline, gaze/PFE, and luminance sensitivity

> Python-facing port of `pupil-baseline-gaze-luminance-sensitivity.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Prespecify consequential analysis choices

The sensitivity layer records alternative analysis states without selecting
the scenario that produces the largest effect.

## Materialize, do not rank

Each scenario can be fitted and reduced to the same declared estimand.
`compare_pupil_sensitivity_estimands()` then places those estimands side by
side. It does not identify a winner.

PFE and luminance are handled as measurement-context variables. The 0.4
foundation can audit them and compare explicitly declared adjusted/unadjusted
specifications, but it does not invent a universal PFE correction or a
Bayesian Open-DPSM replacement.

## Python API mapping

- `gp3bayespy.compare_pupil_sensitivity_estimands`
- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.create_pupil_sensitivity_suite`
- `gp3bayespy.materialize_pupil_sensitivity_scenario`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.pupil_sensitivity_table`
- `gp3bayespy.simulate_pupil_timecourse`
- `gp3bayespy.specify_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
