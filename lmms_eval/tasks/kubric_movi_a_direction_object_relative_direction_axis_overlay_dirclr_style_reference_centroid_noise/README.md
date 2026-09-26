# E3: reference-centroid noise on the RGB overlay

This is the reference-noise companion to
`kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_noise`.
It uses the 218 MOVi-A object-relative **object-answer** questions and creates
eight conditions: RGB overlay and heading-up symbolic map at each of
σ = 10%, 25%, 50%, and 100%. This gives 218 × 4 × 2 = 1,744 model calls.

For each question and level, one deterministic isotropic 2-D Gaussian shift
is sampled in pixel coordinates. Its per-axis standard deviation is the given
fraction of the clean projected reference-to-target distance. Exactly the same
pixel shift is applied to the Dir_Clr RGB axes' origin and to the reference
symbol in the symbolic map. RGB axis directions and lengths do not change;
candidate symbols do not move.

The submission JSON includes the sampled shift and a no-model geometric
diagnostic. `tools/summarize_kubric_reference_centroid_noise.py` computes the
same curve directly from the dataset.

For the checked-in deterministic seed schedule, the companion curve is:

| σ / reference-target distance | Trials | Wrong-box fraction | Geometric ceiling |
|---:|---:|---:|---:|
| 10% | 218 | 0.312 | 0.688 |
| 25% | 218 | 0.335 | 0.665 |
| 50% | 218 | 0.399 | 0.601 |
| 100% | 218 | 0.459 | 0.541 |

## Fig. 8 table rows

Under **Reference-centroid noise**, add these four rows:

| Perturbation | σ / reference-target distance | RGB overlay | Symbolic map |
|---|---:|---:|---:|
| Reference-centroid noise | 10% | perturbed | perturbed |
| Reference-centroid noise | 25% | perturbed | perturbed |
| Reference-centroid noise | 50% | perturbed | perturbed |
| Reference-centroid noise | 100% | perturbed | perturbed |

Suggested updated caption: “Robustness of RGB-overlay and symbolic-map
representations. RGB is perturbed under heading noise and reference-centroid
noise; it is unperturbed only by construction under candidate dropout and
candidate-centroid noise. The reference-centroid rows apply the identical
sampled shift to the RGB overlay origin and symbolic reference. The geometric
ceiling is one minus the fraction of trials whose shifted correct-axis endpoint
is nearer a wrong candidate box than the target box.”
