# COMFORT object- and camera-basis direction prediction

This directory defines matched object-frame and camera-frame tasks over
`/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl`. Each scene contributes
four specified-target queries, one from the central reference object to each
surrounding object. Each direction, vector, or combined variant therefore has
2,000 examples.

## Tasks and output formats

### `comfort_multi_3d_object_basis_direction`

Predict only the dominant horizontal relation. The response is one lowercase
word with no JSON:

```text
left
```

Allowed answers are `left`, `right`, `front`, and `back`. The source dataset's
`behind` label is presented and evaluated as `back`.

### `comfort_multi_3d_object_basis_vector`

Predict only the target-minus-reference **unit direction vector** as valid JSON:

```json
{"front": 0.72, "up": -0.11, "right": -0.68}
```

All three keys are required. Their meanings are:

- positive `front`: in front of the reference; negative: behind/back;
- positive `up`: above the reference; negative: below;
- positive `right`: on the reference's right; negative: on its left.

The parser accepts the same object under `vector` or `relative_vector` as a
compatibility convenience, but the requested vector-only format is flat JSON.
A zero vector is rejected as a parse failure. Predictions need not already be
unit length: cosine and unit-L2 scoring normalize them first.

### `comfort_multi_3d_object_basis_combined`

Predict the categorical answer and vector together, following the relevant
Kubric viewpoint task's `answer`/`relative_vector` structure:

```json
{
  "answer": "left",
  "relative_vector": {
    "front": 0.04,
    "up": 0.12,
    "right": -0.99
  }
}
```

The group task `comfort_multi_3d_object_basis` runs these three object-frame
variants.

### Camera-frame direction, vector, and combined tasks

The camera equivalents use the same output schemas and metrics:

```text
comfort_multi_3d_camera_basis_direction
comfort_multi_3d_camera_basis_vector
comfort_multi_3d_camera_basis_combined
```

For these tasks, `right`, `up`, and `front` are camera/image-right,
camera/image-up, and the camera viewing direction. The ground-truth unit vector
is the camera-frame target position minus the camera-frame reference position.
Its dominant horizontal component supplies `left|right|front|back`.

The group `comfort_multi_3d_camera_basis` runs all three camera variants.

### Object-vs-direction answer-format tasks

Two additional tasks ask every retained spatial fact in paired forms:

```text
comfort_multi_3d_object_basis_object_direction
comfort_multi_3d_camera_basis_object_direction
```

The direction row names the target and expects one direction word:

```text
From the dog's own frame, where is the sofa relative to the dog?
```

The paired object row gives the relation and expects one candidate object name:

```text
From the dog's own frame, which surrounding object is left of the dog?
```

The object-frame task has 2,000 matched facts and 4,000 rows. Camera-frame
dominant directions are not necessarily unique: several surrounding objects
can be camera-left, for example. The camera object-answer task excludes such
ambiguous facts and contains 1,490 matched facts and 2,980 rows. The specified-
target camera direction/vector/combined tasks still retain all 2,000 examples.

The group `comfort_multi_3d_basis_object_direction` runs both paired tasks, and
`comfort_multi_3d_basis_all` runs all eight tasks in this directory.

## Ground-truth coordinate frame

Each scene supplies the reference object's 3×3 asset-orientation matrix in
camera coordinates. Its columns form an orthonormal asset-local basis, but
different assets can use different local front/right conventions. The task
therefore resolves the semantic object basis as follows:

1. Compute the controlled displacement from the scene's left object to its
   right object.
2. Select and sign the orientation-matrix column best aligned with that
   displacement as semantic `right`.
3. Compute the controlled behind-to-front displacement and select/sign the
   best remaining column as semantic `front`.
4. Set semantic `up = right × front`.
5. Project the actual 3D target-minus-reference displacement onto these axes
   and normalize it to unit length.

The four controlled placements are used only to resolve asset-axis semantics.
The returned components retain the actual target/reference center-height and
off-axis differences from the scene geometry. Dataset loading validates that
the recovered right and front signs agree with all four semantic placements.

Camera-frame variants do not use this recovered object basis for scoring. They
directly reorder the dataset camera coordinates `(x right, y front, z up)` into
the task JSON order `(front, up, right)` and normalize the displacement.

## Metrics

Direction-only metrics:

- `basis_direction_accuracy`: exact categorical accuracy;
- `basis_direction_parse_success`: a valid direction word was found.

Vector metrics:

- `basis_vector_cosine`: cosine similarity to the ground-truth unit vector;
- `basis_vector_l2_score`: `1 - unit_vector_L2 / 2`, clipped to `[0,1]`;
- `basis_vector_angle_30_accuracy`: cosine corresponds to at most 30° error;
- `basis_vector_dominant_direction_accuracy`: the largest absolute horizontal
  component yields the correct `left|right|front|back` label;
- `basis_front_sign_accuracy`, `basis_up_sign_accuracy`, and
  `basis_right_sign_accuracy`: component-sign agreement;
- `basis_full_sign_accuracy`: all three signs agree;
- `basis_vector_parse_success`: all components are finite and nonzero jointly.

Sign metrics use a `1e-4` zero dead zone so floating-point residue on an
otherwise orthogonal axis is treated as zero rather than as a meaningful sign.

The combined task reports all applicable metrics plus a mutually exclusive
outcome decomposition:

- `basis_both_correct`;
- `basis_direction_correct_vector_wrong`;
- `basis_direction_wrong_vector_correct`;
- `basis_both_wrong`;
- `basis_combined_parse_failure`.

Here “vector correct” means that the vector's dominant horizontal component
agrees with the categorical ground-truth direction. Continuous vector quality
is still reported separately by cosine, L2 score, angular accuracy, and signs.

The object-vs-direction tasks report overall, per-format, and parse accuracies,
`object_accuracy - direction_accuracy`, and these paired outcomes:

- both answers correct;
- object correct and direction wrong;
- direction correct and object wrong;
- both answers wrong.

## Running

Run the original three object-frame variants:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks comfort_multi_3d_object_basis \
  --batch_size 1 --log_samples \
  --output_path outputs/comfort_multi_3d_object_basis/evaluation
```

Run one variant by replacing the task with one of:

```text
comfort_multi_3d_object_basis_direction
comfort_multi_3d_object_basis_vector
comfort_multi_3d_object_basis_combined
```

Run the three camera-frame variants:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks comfort_multi_3d_camera_basis \
  --batch_size 1 --log_samples \
  --output_path outputs/comfort_multi_3d_camera_basis/evaluation
```

Run every direction/vector/combined and object-vs-direction task with:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks comfort_multi_3d_basis_all \
  --batch_size 1 --log_samples \
  --output_path outputs/comfort_multi_3d_basis_all/evaluation
```

Submission files are written per variant under the evaluation submission
directory and contain the parsed prediction, ground truth, per-sample metrics,
prompt, image path, and recovered object basis.

## Collecting and plotting results

Use:

```bash
python tools/plot_comfort_object_basis_results.py \
  outputs/comfort_multi_3d_object_basis/evaluation
```

The tool recursively discovers object- and camera-frame submissions and writes:

- `summary.json`, `summary.csv`, and `summary.md`;
- `metric_summary.png`;
- `primary_outcomes_100pct.png`, comparing correct/incorrect/parse-failure
  rates for each prediction format;
- `combined_outcomes_100pct.png`, showing both-correct, direction-only,
  vector-only, both-wrong, and parse-failure proportions;
- `object_direction_outcomes_100pct.png`, showing the paired object/direction
  outcome split for both frames.
- `object_vs_direction_accuracy.png`, directly comparing answer-with-object and
  answer-with-direction accuracy for each frame/model;
- `object_vs_direction_accuracy.csv` and `.md`, reporting those accuracies,
  sample counts, and the object-minus-direction accuracy gap.

Multiple files or directories may be supplied to compare models or runs. Use
`--output-dir PATH` to select the artifact directory explicitly.
