# LOO Influence and Predictive Model Comparison

> Python-facing port of `loo-influence-and-model-comparison.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The LOO display layer converts existing `gp3bayes_psis_loo`,
`gp3bayes_loo_comparison`, and `gp3bayes_loo_weights` objects into explicit
tables and publication-oriented figures.

For multiple prespecified models:

ELPD differences and predictive weights are retained as descriptive predictive
evidence. The package does not promote the highest-ranked model to an
automatically preferred substantive model.

## Python API mapping

- `gp3bayespy.compare_psis_loo`
- `gp3bayespy.compute_loo_model_weights`
- `gp3bayespy.compute_psis_loo`
- `gp3bayespy.loo_diagnostic_table`
- `gp3bayespy.loo_summary_table`
- `gp3bayespy.model_comparison_table`
- `gp3bayespy.model_weights_table`
- `gp3bayespy.plot_loo_influence`
- `gp3bayespy.plot_model_comparison`
- `gp3bayespy.plot_model_weights`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
