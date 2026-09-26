# Dir_Clr-styled RGB axis overlay versus heading-up symbolic map

This is a matched visual-style variant of
`kubric_movi_a_direction_object_relative_direction_axis_overlay_symbolic_up_noise`.
All source pairs, questions, options, symbolic images, text legends, seeds,
and perturbation schedules are unchanged.  The **only difference** is the RGB
axis renderer, matched to `Dir_Clr/pbs_run_ObjectMapInversion_BBoxDiagonal.sh`
and its `object_direction_reasoning.save_axes_overlay` implementation:

- arrows originate at the reference bounding-box centre;
- every arrow is `0.8 ×` the reference bounding-box diagonal;
- axis order is FRONT → RIGHT → BACK → LEFT;
- labels are uppercase DejaVu Sans Bold with a two-pixel black stroke;
- arrow colors are the Dir_Clr palette: front green, right orange, back red,
  and left blue;
- arrowheads are filled triangular heads and there are no label boxes.

Use the same `KUBRIC_RELATIVE_REPRESENTATION_SUITE` and
`KUBRIC_RELATIVE_REPRESENTATION_DEBUG` environment variables as the base
axis-overlay task.
