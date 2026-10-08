# Appendix: Evaluation Prompts and Expected Answer Formats

This appendix records the model-visible prompt families used by the spatial
reasoning diagnostics. Text in angle brackets (for example, `<reference>`) is
replaced per example. Unless stated otherwise, an image is supplied together
with the text prompt. The displayed order of multiple-choice answers is the
order used for scoring; the expected response is the corresponding option
letter, not the answer text.

## Common paired direction/object format

Several tasks use the same four-choice wrapper with two matched formulations
of a spatial relation.

Direction-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the <reference>'s current viewpoint, where is the <target>?
Options:
A. <direction 1>
B. <direction 2>
C. <direction 3>
D. <direction 4>
```

Object-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: Which object is in the direction that the <relation> side of the <reference> points toward?
Options:
A. <object 1>
B. <object 2>
C. <object 3>
D. <object 4>
```

Expected answer format: exactly one option letter from `A`, `B`, `C`, or `D`.
The evaluator also recognizes conventional forms such as `A.`, `Answer: A`,
and `(A)`, but the requested response is the bare letter.

## `comfort_arrow_length_sweep`

This diagnostic uses the common paired direction/object format. Four labeled,
reference-relative arrows are drawn on the reference object. The experiment
varies their length over

```text
0.25, 0.45, 0.70, 0.90, 1.15, 1.50
```

times the diagonal of the reference-object bounding box. The following line is
prepended to both question formats:

```text
Four labeled reference-relative arrows are overlaid on the reference object. Their length is <scale> times the reference bbox diagonal.
```

The remainder of the prompt is:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: <authored direction or object question>
Options:
A. <option A>
B. <option B>
C. <option C>
D. <option D>
```

Expected answer format: one letter in `A`--`D`.

## `comfort_direction_object_gt_help`: conditions 0, 4, 36, and 37

All four conditions use the common paired direction/object questions and the
same `A`--`D` answer format. They differ only in the model-visible image and
the aid sentence prepended to the prompt.

### Condition 0: unassisted baseline

- Visual input: original RGB scene.
- Prompt prefix: none.

### Condition 4: reference-front arrow

- Visual input: original scene with a cyan arrow indicating the reference
  object's front direction.
- Prompt prefix:

```text
Image aid: the cyan arrow indicates the front of the reference object (<reference>).
```

### Condition 36: four short reference axes

- Visual input: four short arrows, labeled `left`, `right`, `front`, and
  `behind`, superimposed on the reference object.
- Prompt prefix:

```text
Image aid: the four labeled arrows superimposed on the reference object (<reference>) show its own left, right, front, and behind directions.
```

### Condition 37: two short reference axes

- Visual input: short `front` and `left` arrows superimposed on the reference
  object.
- Prompt prefix:

```text
Image aid: the two short labeled arrows superimposed on the reference object (<reference>) show its own front and left directions. Infer behind as opposite front and right as opposite left.
```

For every condition, the prefix is followed by:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: <authored direction or object question>
Options:
A. <option A>
B. <option B>
C. <option C>
D. <option D>
```

Expected answer format: one letter in `A`--`D`.

## `comfort_gt_help_components`

This task group contains 18 component diagnostics. Coordinates are normalized
to `[0,1000]`; image coordinates use the top-left origin, with `x` increasing
rightward and `y` increasing downward.

### Object localization and recognition

#### `comfort_gt_component_bbox_prediction`

Visual input: unmodified RGB image.

```text
Locate the <object> in the plain image. Return exactly one bounding box as [x_min, y_min, x_max, y_max] in [0,1000], where (0,0) is top-left.
```

Expected answer format:

```json
[x_min, y_min, x_max, y_max]
```

#### `comfort_gt_component_bbox_naming`

Visual input: one object enclosed by a magenta box labeled `BOX`.

```text
A magenta box labeled BOX is overlaid on one object. What object is inside it? Return exactly one name from: <candidate object names>.
```

Expected answer format: exactly one candidate object name.

### Reference-orientation estimation

#### `comfort_gt_component_facing_direction`

Visual input: reference object enclosed by a box.

```text
The reference <reference> is boxed. In which image-plane direction is its own front facing? Return exactly one of: right, down-right, down, down-left, left, up-left, up, up-right.
```

Expected answer format: one of the eight listed compass labels.

#### `comfort_gt_component_facing_direction_no_bbox`

Visual input: unmodified RGB image. Examples with a non-unique normalized
reference name are excluded.

```text
The reference object is the <reference>. In which image-plane direction is its own front facing? Return exactly one of: right, down-right, down, down-left, left, up-left, up, up-right.
```

Expected answer format: one of the eight listed compass labels.

#### `comfort_gt_component_facing_direction_flip`

Visual input: paired original and horizontally mirrored versions of the boxed
reference image. The prompt is identical for both; the horizontal component of
the gold facing direction is mirrored.

```text
The reference <reference> is boxed. In which image-plane direction is its own front facing? Return exactly one of: right, down-right, down, down-left, left, up-left, up, up-right.
```

Expected answer format: one of the eight listed compass labels.

### Axis production and reading

#### `comfort_gt_component_front_arrow`

```text
The reference <reference> is boxed. Predict a 2D arrow pointing toward the object's own front. Start at the object's center. Return only {"start": [x, y], "end": [x, y]} with coordinates in [0,1000].
```

#### `comfort_gt_component_left_arrow`

```text
The reference <reference> is boxed. Predict a 2D arrow pointing toward the object's own left. Start at the object's center. Return only {"start": [x, y], "end": [x, y]} with coordinates in [0,1000].
```

Expected answer format for both arrow-production tasks:

```json
{"start": [x, y], "end": [x, y]}
```

#### `comfort_gt_component_front_arrow_reading`

Visual input: boxed reference with its ground-truth front-axis arrow.

```text
The reference <reference> is boxed. The overlaid arrow shows the reference object's ground-truth front axis. Using the reference object's own perspective, where is the <target> relative to it? Return exactly one of: left, right, front, back.
```

#### `comfort_gt_component_left_arrow_reading`

Visual input: boxed reference with its ground-truth left-axis arrow.

```text
The reference <reference> is boxed. The overlaid arrow shows the reference object's ground-truth left axis. Using the reference object's own perspective, where is the <target> relative to it? Return exactly one of: left, right, front, back.
```

Expected answer format for both arrow-reading tasks: one of `left`, `right`,
`front`, or `back`.

### Symbol grounding and arrow-to-symbol conversion

#### `comfort_gt_component_symbol_to_object`

```text
Each surrounding object has an abstract letter marker. Which object is marked <symbol>? Return exactly one name from: <candidate object names>.
```

Expected answer format: exactly one candidate object name.

#### `comfort_gt_component_object_to_symbol`

```text
Each surrounding object has an abstract letter marker. Which marker is on the <object>? Return exactly one of: A, B, C, D.
```

Expected answer format: one letter in `A`--`D`.

#### `comfort_gt_component_long_arrow_to_symbol`

Visual input: abstract object markers and long labeled reference-relative
arrows.

```text
The reference object's labeled arrows show its own directions and the letters mark surrounding objects. Which letter marks the object <direction> of the reference? Return exactly one of: A, B, C, D.
```

#### `comfort_gt_component_short_arrow_to_symbol`

Visual input: abstract object markers and short labeled reference-relative
arrows. The textual prompt is identical to the long-arrow condition.

```text
The reference object's labeled arrows show its own directions and the letters mark surrounding objects. Which letter marks the object <direction> of the reference? Return exactly one of: A, B, C, D.
```

Expected answer format for both arrow-to-symbol tasks: one letter in `A`--`D`.

### Coordinate-vector understanding

#### `comfort_gt_component_vector_to_direction`

```text
In the reference object's coordinates, positive axes are front, up, and right. What direction is the vector {"front": <f>, "up": <u>, "right": <r>}? Return exactly one of: front, back, above, below, right, left.
```

Expected answer format: one of `front`, `back`, `above`, `below`, `right`, or
`left`.

#### `comfort_gt_component_direction_to_vector`

```text
In the reference object's coordinates, positive axes are front, up, and right. Give the unit vector for <direction>. Return only JSON as {"front": <float>, "up": <float>, "right": <float>}.
```

Expected answer format:

```text
{"front": <float>, "up": <float>, "right": <float>}
```

The values form the canonical signed unit vector for the requested direction.

#### `comfort_gt_component_projected_axes_prediction`

Visual input: reference object enclosed by a box.

```text
The reference <reference> is boxed. Predict its screen-projected front, up, and right unit directions in image coordinates, where x points right and y points down. Return only JSON as {"front": [dx, dy], "up": [dx, dy], "right": [dx, dy]}.
```

Expected answer format:

```json
{"front": [dx, dy], "up": [dx, dy], "right": [dx, dy]}
```

Each value is a two-dimensional unit direction.

### Using projected axes for spatial classification

#### `comfort_gt_component_text_axes_direction`

Visual input: boxed reference. Its projected axes are provided numerically in
the prompt.

```text
In image coordinates (x right, y down), the reference's screen-projected unit axes are front=(<fx>,<fy>), up=(<ux>,<uy>), and right=(<rx>,<ry>). Relative to the reference <reference>, where is the <target>? Return exactly one of: left, right, front, back.
```

#### `comfort_gt_component_overlay_axes_direction`

Visual input: `front`, `up`, and `right` arrows overlaid on the reference.

```text
The image overlays arrows labeled front, up, and right on the reference object. Relative to the reference <reference>, where is the <target>? Return exactly one of: left, right, front, back.
```

Expected answer format for both projected-axis classification tasks: one of
`left`, `right`, `front`, or `back`.

## `kubric_movi_a_direction_object_relative_direction_representation_noise`

Only object-relative pairs are retained. Two visual representations are used:

- `rgb_overlay`: the original RGB image with abstract object markers and a red
  reference-heading arrow;
- `symbolic`: a white canvas containing only abstract object glyphs and the red
  reference-heading arrow.

The complete condition family is the Cartesian product of the two
representations and the following perturbations:

- `clean`;
- object-detection dropout at `0.25`, `0.50`, and `0.75`;
- centroid noise at `0.10`, `0.25`, `0.50`, and `1.00` times the median
  reference-to-object distance;
- reference-orientation rotation by `15`, `30`, `45`, `90`, and `180` degrees.

The prompt text is otherwise invariant across visual/noise conditions:

```text
Answer the spatial-reasoning question using the displayed visual representation. Object-symbol legend: <symbol 1> = <object 1>; ...; R = <reference>. The red arrow is the reference object's heading. Select one answer option and respond with its letter only.
Question: <authored object-relative direction or object question>
Options:
A. <option A>
B. <option B>
C. <option C>
D. <option D>
```

Expected answer format: one letter in `A`--`D`.

## `kubric_movi_a_direction_object_relative_direction`

The task preserves the authored MOVi-A questions and options. It contains both
camera-relative and object-relative pairs.

Camera-relative direction prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the camera's perspective, where is the <target> relative to the <reference>?
Options:
A. <direction>
B. <direction>
C. <direction>
D. <direction>
```

Camera-relative object prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the camera's perspective, which object is to the <relation> of the <reference>?
Options:
A. <object>
B. <object>
C. <object>
D. <object>
```

Object-relative direction prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the <reference>'s perspective while it faces the camera, where is the <target> relative to the <reference>?
Options:
A. <direction>
B. <direction>
C. <direction>
D. <direction>
```

Object-relative object prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the <reference>'s perspective while it faces the camera, which object is to the <relation> of the <reference>?
Options:
A. <object>
B. <object>
C. <object>
D. <object>
```

Expected answer format: one letter in `A`--`D`.

## `3dsrbench_direction_object_qwen3vl_distractors`

This task retains 3DSRBench's native direction question and constructs a
matched inverse object question. Sparse object-choice sets are padded to four
choices using only visible scene objects detected offline by
`Qwen/Qwen3-VL-8B-Instruct`. A row must already contain the target and at least
one image-derived distractor; Qwen-generated objects only fill the remaining
slots. Option order is deterministically shuffled.

Direction-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: Consider the real-world 3D locations and orientations of the objects. Which side of the <reference> is facing the <target>?
Options:
A. <direction>
B. <direction>
C. <direction>
D. <direction>
```

Inverse object-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: Consider the real-world 3D locations and orientations of the objects. Which object is in the direction that the <relation> side of the <reference> points toward?
Options:
A. <object>
B. <object>
C. <object>
D. <object>
```

Expected answer format: one letter in `A`--`D`.

## `comfort_oriented_3d_direction_object`

Each annotated relation yields a matched direction-answer and object-answer
question over the same image and reference-relative relation.

Direction-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: From the <reference>'s current viewpoint, where is the <target>?
Options:
A. <direction>
B. <direction>
C. <direction>
D. <direction>
```

Object-answer prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer option and respond with its letter.
Question: Which object is in the direction that the <relation> side of the <reference> points toward?
Options:
A. <object>
B. <object>
C. <object>
D. <object>
```

The direction vocabulary is `front`, `left`, `behind`, and `right`, with the
order permuted per example. Expected answer format: one letter in `A`--`D`.

## Implementation sources

The templates above were transcribed from the task YAML files, prompt-builder
functions, and saved submission prompts in `lmms_eval/tasks/` and `outputs/`.
The principal source modules are:

- `lmms_eval/tasks/comfort_direction_object_inverse_diagnostics/utils.py`;
- `lmms_eval/tasks/comfort_direction_object_gt_help/utils.py`;
- `lmms_eval/tasks/comfort_gt_help_components/utils.py`;
- `lmms_eval/tasks/kubric_movi_a_direction_object/utils.py`;
- `lmms_eval/tasks/kubric_movi_a_direction_object_relative_direction_representation_noise/utils.py`;
- `lmms_eval/tasks/3dsrbench_custom/utils.py`;
- `lmms_eval/tasks/comfort_direction_object/utils.py`.
