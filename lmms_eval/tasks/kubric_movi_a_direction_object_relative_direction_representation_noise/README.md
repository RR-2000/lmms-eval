# MOVi-A representation-noise diagnostic

This is a matched, object-relative-only variant of
`kubric_movi_a_direction_object_relative_direction`.  Every authored
direction/object pair is repeated with the same question, options, source
image, object symbols, and deterministic corruption seed.

There are exactly two visual representations:

- `rgb_overlay`: the original RGB image with object symbols and the reference
  heading arrow overlaid;
- `symbolic`: a blank canvas containing only object symbols and the reference
  heading arrow.  It contains no object names, direction words, title, legend,
  natural-image pixels, or coordinate labels.

The candidate-symbol legend is supplied in the *shared text prompt* for both
representations, so an object answer can be tied to a symbol without putting
semantic labels into the symbolic image.

The perturbation suites are independent one-axis sweeps (not a factorial):

| Suite | What is corrupted | Levels |
| --- | --- | --- |
| `detection` | Each non-reference object symbol is independently omitted. The RGB pixels remain visible; the symbolic node is absent. | 25%, 50%, 75% |
| `centroid` | Every non-reference 3D centroid receives deterministic isotropic Gaussian noise before projection to the displayed 2D position. | 10%, 25%, 50%, 100% of median reference-to-object distance |
| `orientation` | The supplied reference heading is rotated in the image plane. | 15, 30, 45, 90, 180 degrees |

Use one suite per run:

```bash
KUBRIC_RELATIVE_REPRESENTATION_SUITE=core python -m lmms_eval ... \
  --tasks kubric_movi_a_direction_object_relative_direction_representation_noise
KUBRIC_RELATIVE_REPRESENTATION_SUITE=detection python -m lmms_eval ...
KUBRIC_RELATIVE_REPRESENTATION_SUITE=centroid python -m lmms_eval ...
KUBRIC_RELATIVE_REPRESENTATION_SUITE=orientation python -m lmms_eval ...
```

`all` runs the clean controls plus all three sweeps.  To save exactly the
images sent to the model, set `KUBRIC_RELATIVE_REPRESENTATION_DEBUG=1` and,
optionally, `KUBRIC_RELATIVE_REPRESENTATION_DEBUG_DIR=/path/to/output`.
