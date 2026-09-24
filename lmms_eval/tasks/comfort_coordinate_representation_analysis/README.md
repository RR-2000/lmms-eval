# COMFORT coordinate-representation analysis

This task evaluates only COMFORT object-answer prompts.  Each source scene is
evaluated once per reference-relative relation with a deterministic balanced
answer-option ordering.  The task uses ground-truth object boxes and
ground-truth reference-relative axes to isolate the model-visible
representation from perception errors.

`COMFORT_REPRESENTATION_SUITE` selects conditions:

- `core` (default): `rgb_ids`, `rgb_axes_gt`, `symbolic_gt`, `hybrid_gt`.
- `jitter`: symbolic and hybrid maps with 5%, 10%, 15%, and 20% image-diagonal
  candidate-centre displacement.
- `orientation`: RGB-overlay, symbolic, and hybrid renderings with 15, 30, 45,
  90, and 180 degree reference-axis rotations.
- `all`: all of the above conditions.

An explicit comma-separated `COMFORT_REPRESENTATION_CONDITIONS` overrides the
suite.  The symbolic map uses candidate IDs and GT image-plane centres, but
never prints a direction-to-object association or highlights the queried
target.  Candidate IDs are also drawn in every RGB condition and are mapped to
the option names in the prompt, preventing visual category recognition from
confounding the representation comparison.

Useful runs:

```bash
COMFORT_REPRESENTATION_SUITE=core lmms_eval --tasks comfort_coordinate_representation_analysis ...
COMFORT_REPRESENTATION_SUITE=jitter lmms_eval --tasks comfort_coordinate_representation_analysis ...
COMFORT_REPRESENTATION_SUITE=orientation lmms_eval --tasks comfort_coordinate_representation_analysis ...
```

The submission JSON includes the condition, GT perturbation amount, prompt,
and all predictions.  It is scored on the full selected denominator.
