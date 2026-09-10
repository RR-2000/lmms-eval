# COMFORT GT_HELP component diagnostics

This task family measures whether a VLM can both generate and consume the
individual visual and textual primitives used by
`comfort_direction_object_gt_help`. Each task is independent, so a downstream
GT_HELP gain can be separated from failures in localization, object
orientation, representation production, representation reading, or axis
reasoning.

Run all tasks with:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,max_num_frames=32 \
  --tasks comfort_gt_help_components \
  --batch_size 1 --log_samples \
  --output_path outputs/comfort_gt_help_components
```

Every coordinate exposed to the model is normalized to `[0,1000]`, with
`(0,0)` at image top-left. The source is
`/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl`.

Human-facing labels are normalized during preprocessing: `horsel` and
`horser` become `horse`, `bicycle mountain` becomes `bicycle`, and `car sedan`
becomes `car`. The source JSONL is not modified. Plain-image tasks that would
be ambiguous after normalization (for example, two horses in one scene) omit
that query; tasks with an identifying box or marker can retain it.

## Task inventory

| Capability | Role | Task | Input intervention | Expected output | Primary metric |
|---|---|---|---|---|---|
| bbox | generation | `comfort_gt_component_bbox_prediction` | plain image | `[x1,y1,x2,y2]` | mean IoU |
| bbox | utilization | `comfort_gt_component_bbox_naming` | one magenta box | object name | accuracy |
| orientation label | generation | `comfort_gt_component_facing_direction` | reference box | 8-way image direction | accuracy |
| front arrow | generation | `comfort_gt_component_front_arrow` | reference box | arrow start/end JSON | direction cosine |
| front arrow | utilization | `comfort_gt_component_front_arrow_reading` | supplied front arrow | 8-way image direction | accuracy |
| left arrow | generation | `comfort_gt_component_left_arrow` | reference box | arrow start/end JSON | direction cosine |
| left arrow | utilization | `comfort_gt_component_left_arrow_reading` | supplied left arrow | 8-way image direction | accuracy |
| abstract symbols | utilization | `comfort_gt_component_symbol_to_object` | letter circles | object name | accuracy |
| abstract symbols | generation/encoding | `comfort_gt_component_object_to_symbol` | letter circles | letter | accuracy |
| long-arrow mapping | utilization | `comfort_gt_component_long_arrow_to_symbol` | GT_HELP-6-length arrows + letters | letter | accuracy |
| short-arrow mapping | utilization | `comfort_gt_component_short_arrow_to_symbol` | GT_HELP-36-length arrows + letters | letter | accuracy |
| direction vector | utilization/decoding | `comfort_gt_component_vector_to_direction` | canonical vector in text | axis direction | accuracy |
| direction vector | generation/encoding | `comfort_gt_component_direction_to_vector` | axis direction in text | `{front,up,right}` | cosine |
| projected axes | generation | `comfort_gt_component_projected_axes_prediction` | reference box | three 2D unit vectors | mean axis cosine |
| projected axes | text utilization | `comfort_gt_component_text_axes_direction` | projected front/up/right vectors in text | spatial direction | accuracy |
| projected axes | overlay utilization | `comfort_gt_component_overlay_axes_direction` | front/up/right arrows on image | spatial direction | accuracy |

## Generation/utilization controls

Seven matched comparisons make the distinction explicit:

1. bbox prediction → boxed-object naming;
2. front-arrow prediction → supplied-front-arrow reading;
3. left-arrow prediction → supplied-left-arrow reading;
4. object-to-symbol encoding → symbol-to-object decoding;
5. direction-to-vector encoding → vector-to-direction decoding;
6. projected-axis prediction → using the axes supplied numerically; and
7. projected-axis prediction → using the axes supplied as an image overlay.

All comparisons use `component_raw_accuracy` as a common binary measure. A box
is correct at IoU ≥ 0.5; an arrow, vector, or complete projected basis is
correct within 30 degrees; and a classification is an exact match. Primary
continuous metrics remain available and should be used to interpret near
misses. Generation and utilization questions use the same scene annotations,
but they are capability controls rather than chained predictions: the
utilization task always receives the ground-truth component, not output from
the generation task.

The six canonical 3D directions are encoded as:

| Word | `{front, up, right}` |
|---|---|
| front | `{1, 0, 0}` |
| back | `{-1, 0, 0}` |
| above | `{0, 1, 0}` |
| below | `{0, -1, 0}` |
| right | `{0, 0, 1}` |
| left | `{0, 0, -1}` |

## Example prompts and outputs

Values below are illustrative; actual labels and coordinates come from each
scene.

### Bounding boxes

Input image: unchanged RGB image.

```text
Locate the chair in the plain image. Return exactly one bounding box as
[x_min, y_min, x_max, y_max] in [0,1000], where (0,0) is top-left.
```

```text
[214, 330, 486, 811]
```

For bbox consumption, the image instead contains a magenta rectangle labeled
`BOX` and the prompt asks for exactly one candidate object name.

### Orientation arrows

```text
The reference chair is boxed. Predict a 2D arrow pointing toward the object's
own front. Start at the object's center. Return only
{"start": [x, y], "end": [x, y]} with coordinates in [0,1000].
```

```json
{"start": [430, 615], "end": [328, 470]}
```

The arrow direction is scored by cosine against the perspective projection of
the semantic 3D axis. The start point is scored separately against the center
of the reference bbox. The left-arrow task is identical except that it asks for
the object's own left.

The matched reading tasks instead draw the ground-truth arrow and ask for its
eight-way image direction:

```text
The reference chair is boxed. The overlaid arrow shows the object's
ground-truth front axis. In which image-plane direction does the arrow point?
Return exactly one of: right, down-right, down, down-left, left, up-left, up,
up-right.
```

```text
up-left
```

### Abstract symbols

Each target is covered by a yellow circle labeled A, B, C, or D. The assignment
is deterministically rotated between scenes so a letter cannot become a fixed
direction shortcut.

```text
Each surrounding object has an abstract letter marker. Which marker is on the
lamp? Return exactly one of: A, B, C, D.
```

```text
C
```

The long- and short-arrow tasks add the same four labeled semantic arrows used
by GT_HELP 6 and GT_HELP 36, respectively, then ask which abstract marker is in
a requested object-relative direction.

### Direction vectors and cues

```text
In the reference object's coordinates, positive axes are front, up, and right.
What direction is the vector {"front": 0, "up": -1, "right": 0}?
Return exactly one of: front, back, above, below, right, left.
```

```text
below
```

The text-axis task supplies the reference's projected front, up, and right unit
vectors numerically. The overlay-axis task supplies the same concepts as arrows
drawn at the reference. Both then ask for the direction of one named target.
Their shared generation control asks the model to produce those projected axes
from the boxed reference:

```text
Predict its screen-projected front, up, and right unit directions in image
coordinates, where x points right and y points down. Return only JSON as
{"front": [dx, dy], "up": [dx, dy], "right": [dx, dy]}.
```

```json
{"front": [-0.81, -0.59], "up": [0.02, -1.0], "right": [0.59, -0.81]}
```

Each axis receives a cosine score; the primary score averages the three. The
common raw accuracy is one only when all three axes are within 30 degrees.

## How to compare the results

1. Start with `generation_vs_utilization.png`. Compare bbox prediction IoU with bbox naming accuracy. Low IoU but high
   naming means the model can consume a supplied box but cannot produce one.
2. Compare facing-direction accuracy with front-arrow cosine. A gap separates
   categorical orientation recognition from coordinate output. Compare the
   front and left arrows to detect handedness errors.
3. Use symbol-to-object versus object-to-symbol as the two directions of symbol
   grounding. Then compare long-arrow-to-symbol against short-arrow-to-symbol;
   this is the isolated analogue of GT_HELP 6 versus 36.
4. Check vector-to-direction and direction-to-vector first. These are the
   representation floor and include up/above and down/below explicitly.
5. Compare text-axis with overlay-axis accuracy. If canonical vector tasks are
   strong but these are weak, the failure is applying axes to a scene. The gap
   between text and overlay identifies which carrier is more usable.
6. Compare these diagnostics with the original GT_HELP task only after checking
   parse success. A formatting failure should not be read as a spatial failure.

## Summarizer

```bash
python tools/summarize_comfort_gt_help_component_experiments.py \
  outputs/comfort_gt_help_components \
  --output-dir outputs/comfort_gt_help_components/analysis
```

The tool writes `summary.json`, `summary.csv`, `summary.md`,
`answer_breakdown.csv`, `generation_vs_utilization.csv`, `primary_scores.png`,
`raw_accuracy.png`, `representation_comparisons.png`, and
`generation_vs_utilization.png`. It accepts any mixture of submission JSON
files and directories and searches directories recursively. The paired table
and plot appear for every generation/utilization pair whose two submissions
are present; incomplete pairs remain visible as individual rows in the main
summary.

All tasks also expose `component_raw_accuracy`, using exact-match correctness
for classification, IoU >= 0.5 for bbox prediction, and angular error <= 30
degrees for arrow and vector prediction. This supplies one directly comparable
binary-accuracy bar for every task while retaining the continuous primary
metrics.
