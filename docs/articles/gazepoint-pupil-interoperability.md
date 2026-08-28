# Gazepoint pupil-data interoperability

> Python-facing port of `gazepoint-pupil-interoperability.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Verified Open Gaze fields

The Gazepoint bridge recognizes documented Open Gaze API fields and reports a
mapping proposal. It does not select a left, right, pixel-diameter, or 3-D
pupil channel on the analyst's behalf.

The API distinguishes camera-image pupil diameter (`LPD`, `RPD`) in **pixels**
from 3-D pupil diameter (`LPUPILD`, `RPUPILD`) in **metres**. The bridge keeps
those units separate.

Because both left and right pupil channels are present, channel selection is
ambiguous and must be explicit.

## 3-D pupil diameter

A proposed schema is an interoperability audit, not evidence that a column is
appropriate for a specific scientific analysis. Export variants and
preprocessing provenance remain part of the contract.

## Python API mapping

- `gp3bayespy.gazepoint_pupil_mapping_table`
- `gp3bayespy.inspect_gazepoint_pupil_schema`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
