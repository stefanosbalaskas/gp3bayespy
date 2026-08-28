# Experimental Interpretable Pupil Response Shape

> Python-facing port of `experimental-pupil-response-shape.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The 0.5 experimental family is intentionally singular rather than an arbitrary nonlinear-formula API. It represents a positive-amplitude asymmetric gated response using baseline, log-amplitude, onset, log-rise, log-duration, and log-decay parameters.

The parameters remain properties of this response model. gp3bayes does not relabel amplitude as attention, onset as surprise, or duration as cognitive effort.

## Python API mapping

- `gp3bayespy.estimate_pupil_response_parameters`
- `gp3bayespy.fit_pupil_response_shape_model`
- `gp3bayespy.plot_pupil_response_parameters`
- `gp3bayespy.pupil_response_parameter_table`
- `gp3bayespy.simulate_pupil_response_shape`
- `gp3bayespy.specify_pupil_response_shape_model`
- `gp3bayespy.translate_pupil_response_shape_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
