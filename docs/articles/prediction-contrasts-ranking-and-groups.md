# Prediction Contrasts, Rankings, and Groups

> Python-facing port of `prediction-contrasts-ranking-and-groups.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Prediction grids can be summarised by observed design variables without
expanding the approved model-family scope.

The ranking function is deliberately descriptive. A probability of rank one is
not converted into an automatic selection.

When the prediction data contain multiple rows per substantive group:

This makes aggregation explicit and reproducible rather than hiding it inside
plotting code.

## Python API mapping

- `gp3bayespy.create_prediction_grid`
- `gp3bayespy.group_prediction_summary`
- `gp3bayespy.plot_group_predictions`
- `gp3bayespy.predict_model`
- `gp3bayespy.prediction_interval_width`
- `gp3bayespy.prediction_pairwise_contrasts`
- `gp3bayespy.prediction_rank_probabilities`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
