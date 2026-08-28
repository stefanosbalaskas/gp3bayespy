# End-to-End Evidence and Publication Showcase

> Python-facing port of `end-to-end-evidence-showcase.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The expanded post-fit system separates fitting, diagnostics, sensitivity,
simulation validation, provenance, publication output, and interpretation.

No dashboard, registry, sensitivity result, recovery result, SBC graphic, or LOO
diagnostic automatically establishes causal validity or selects a preferred
model.

## Python API mapping

- `gp3bayespy.create_analysis_bundle`
- `gp3bayespy.create_analysis_manifest`
- `gp3bayespy.create_complete_evidence_inventory`
- `gp3bayespy.create_diagnostic_dashboard`
- `gp3bayespy.create_loo_influence_atlas`
- `gp3bayespy.create_model_card`
- `gp3bayespy.prior_posterior_bridge`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
