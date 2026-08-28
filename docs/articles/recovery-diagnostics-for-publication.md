# Parameter Recovery Diagnostics for Publication

> Python-facing port of `recovery-diagnostics-for-publication.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Recovery objects retain parameter summaries, repetition-level estimates, and
fit statuses. The publication layer exposes each separately.

Recovery evidence remains a validation diagnostic; no figure certifies an
implementation or model automatically.

## Python API mapping

- `gp3bayespy.plot_recovery_bias`
- `gp3bayespy.plot_recovery_coverage`
- `gp3bayespy.plot_recovery_estimates`
- `gp3bayespy.plot_recovery_fit_status`
- `gp3bayespy.plot_recovery_rmse`
- `gp3bayespy.recovery_estimate_table`
- `gp3bayespy.recovery_fit_status_table`
- `gp3bayespy.recovery_parameter_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
