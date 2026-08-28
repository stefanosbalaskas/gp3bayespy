# Grouped PSIS-LOO Influence

> Python-facing port of `grouped-loo-influence.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Pointwise PSIS-LOO diagnostics may be aggregated by a declared participant,
item, condition, or other observation-level grouping variable.

Aggregation supports review of concentrated predictive influence. It never
removes a group automatically.

## Python API mapping

- `gp3bayespy.loo_group_influence_table`
- `gp3bayespy.plot_loo_group_elpd`
- `gp3bayespy.plot_loo_group_influence`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/loo_model_comparison.py`.
