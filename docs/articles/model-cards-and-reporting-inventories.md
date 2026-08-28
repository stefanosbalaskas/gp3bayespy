# Model Cards and Reporting Inventories

> Python-facing port of `model-cards-and-reporting-inventories.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

A model card records what was fitted and what evidence is available. It does
not turn documentation completeness into model validity.

Writing is explicit:

The card records that automatic model selection, automatic adequacy
certification, and automatic causal identification remain false.

## Python API mapping

- `gp3bayespy.create_analysis_bundle`
- `gp3bayespy.create_analysis_manifest`
- `gp3bayespy.create_model_card`
- `gp3bayespy.create_reporting_checklist`
- `gp3bayespy.model_card_table`
- `gp3bayespy.plot_reporting_checklist`
- `gp3bayespy.write_model_card`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
