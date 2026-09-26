# MOVi-A direct-name / heading-up representation-noise diagnostic

This is a matched object-relative-only variant of
`kubric_movi_a_direction_object_relative_direction_representation_noise`.
It preserves the same clean controls and independent detection, centroid, and
orientation sweeps, but removes the symbol-to-entity binding step.

- **RGB overlay:** the source RGB image carries the full entity name beside
  each detected object and a red reference-heading arrow.  There are no
  symbols and no symbol legend.
- **Symbolic:** a blank symbolic layout contains full entity names at object
  locations.  The layout is rotated so the reference heading is image-up; it
  has no heading arrow, axes, symbol legend, direction text, or natural-image
  pixels.

For the orientation sweep, the RGB arrow is rotated as usual.  The symbolic
layout is instead normalized using a rotated (incorrect) heading estimate,
which rotates all displayed entities relative to image-up.  Thus the two
representations receive equivalent heading corruption without adding a
heading graphic to the symbolic form.

The same environment controls as the prior task select suites and save final
prompt images:

```bash
KUBRIC_RELATIVE_REPRESENTATION_SUITE=all \
KUBRIC_RELATIVE_REPRESENTATION_DEBUG=1 \
KUBRIC_RELATIVE_REPRESENTATION_DEBUG_DIR=/path/to/prompt_images \
python -m lmms_eval ... --tasks kubric_movi_a_direction_object_relative_direction_named_up_noise
```
