# Direction-versus-vector diagnostics

This family implements a cross-dataset diagnostic matrix that distinguishes
spatial perception, coordinate transformation, direction discretization,
numeric generation, and formatting failures. All vector schemas use `(front,
up, right)` and target-minus-reference directions. `back`, `below`, and `left`
are negative front, up, and right respectively.

The six intervention families originally designed for COMFORT now have matched
COMFORT, ScanNet, and Kubric tasks. ScanNet and Kubric additionally have angular
boundary tasks; COMFORT is omitted from that one because its controlled targets
are near-cardinal. The original dataset-specific Kubric rotation and ScanNet
depth-oracle controls remain in the group.

## Coordinate frames

- COMFORT uses the annotated reference-object orientation.
- ScanNet uses its constructed object frame: the reference object is imagined
  looking back at the camera, with world up fixed.
- Kubric uses the same anchor-looking-back-at-camera construction already used
  by its object-centric viewpoint benchmark.

Thus conditions are matched in the information they reveal and the requested
output, but not by pretending that ScanNet has semantic object yaw annotations.

## Tasks

| Family | Task suffix | Datasets | Conditions | Primary question |
|---:|---|---|---|---|
| 1 | `direction_vector_cross_output` | all three | direction only, vector only, two combined key orders | Do separate and joint outputs encode the same direction? |
| 2 | `direction_vector_output_formats` | all three | named JSON, ordered list, plain text, signs, vector prototypes | Is numeric/schema production the bottleneck? |
| 3 | `direction_vector_conversion_oracle` | all three | vector→direction, direction→canonical vector | Does the model understand the label/component mapping without vision? |
| 4 | `direction_vector_basis_oracle` | all three | nine information tiers | Which perception or transform stage repairs the answer? |
| 5 | `direction_vector_components` | all three | three signs, dominant axis, horizontal direction | Is the error axis selection or polarity? |
| 6 | `direction_vector_boundaries` | ScanNet, Kubric | angular-margin bins | Is label disagreement caused by near-boundary discretization? |
| 7 | `direction_vector_arrow_grounding` | all three | no arrow through full labeled/color axes | Can visual axes be mapped to numeric components? |
| 8 | `kubric_direction_vector_rotation` | Kubric | raster rotations 0/90/180/270° | Is object-frame output invariant to display rotation? |
| 9 | `scannet_direction_vector_depth_oracle` | ScanNet | RGB, bboxes, depths, centroids, displacement | Does real-scene failure come from localization or metric depth? |

Prefix the shared suffix with `comfort_`, `scannet_`, or `kubric_` to obtain
the concrete task name. The group contains 22 tasks. With the current datasets,
the six shared families contribute 60,000 COMFORT, 460,560 ScanNet, and 18,390
Kubric generations. Adding the two boundary tasks and dataset-specific controls
gives 634,127 generations total, so scheduler jobs should normally run tasks
separately.

## Prompt and measurement details

### 1. Cross-output consistency

The identical object pair is requested as a direction word, vector,
and combined JSON. Combined JSON is tested with direction-first and
vector-first key order. The analyzer pairs `direction_only` with `vector_only`
by `base_pair_id` and reports both-correct, direction-only, vector-only, and
both-wrong. Combined rows additionally measure whether the written direction
equals the dominant horizontal direction derived from the written vector.

### 2. Output-format controls

The target is held fixed while the output changes among:

```text
{"front":0.8,"up":0.1,"right":-0.2}
[0.8, 0.1, -0.2]                 # fixed front,up,right order
front=0.8, up=0.1, right=-0.2
{"front":1,"up":1,"right":-1}  # signs only
A                                      # one of four horizontal prototypes
```

Named JSON versus ordered list diagnoses component-order errors. Prototype
multiple choice removes numeric generation. Signs remove magnitude estimation.

### 3. Conversion oracles

`vector_to_direction` prints the gold continuous vector and asks only for its
dominant horizontal word. `direction_to_vector` prints the gold word and asks
for its canonical cardinal unit vector. These are non-visual conversion
ceilings; the latter is intentionally evaluated against the canonical vector,
not the scene's slightly non-cardinal displacement.

### 4. Basis oracle ladder

The nine tiers are RGB, reference/target boxes, a labeled front axis, all short
horizontal axes, the reference basis matrix, camera-frame displacement, basis
plus displacement, gold relative vector, and gold direction. Every tier asks
for combined direction/vector JSON. Compare per-question recovery against RGB:
boxes diagnose localization; axes diagnose orientation; basis+displacement
diagnoses transform arithmetic; gold vector diagnoses vector→word conversion;
gold direction diagnoses whether vector generation remains independently weak.
For ScanNet and Kubric, a reference-facing-camera axis can project to nearly a
point because it is depth-aligned. In that degenerate case the overlay uses a
clearly labeled canonical 2D glyph anchored on the reference box; it remains an
axis-definition oracle, not a claimed image-plane motion vector.

### 5. Component decomposition

Separate prompts request front/up/right sign, largest absolute component, or
dominant horizontal direction. `component_accuracy` is the primary metric.
This separates axis choice from positive/negative polarity and prevents one
bad numeric component from hiding otherwise correct qualitative structure.

### 6. Angular boundaries

ScanNet and Kubric object-frame combined predictions are binned by
`abs(abs(front)-abs(right))`: `0-0.1`, `0.1-0.25`, `0.25-0.5`, and `>=0.5`.
Small margins lie near the categorical left/right versus front/back boundary.
Compare full-vector cosine, horizontal cosine, and vector-derived direction
accuracy by bin.

COMFORT is intentionally excluded here because its four controlled targets are
near-cardinal: all current COMFORT examples land in the `>=0.5` bin and cannot
measure boundary sensitivity.

### 7. Arrow-to-vector grounding

Conditions are plain RGB, labeled front, labeled front+right, four labeled
short axes, and four color-coded unlabeled axes with a text legend. The prompt
always requests both outputs. Improvements isolate whether explicit axes repair
the coordinate transform; labeled versus color-only arrows test in-image text
reading versus symbolic grounding. Use matched condition deltas within each
dataset first, then compare those deltas across datasets; raw scores also absorb
differences in scene realism, object vocabulary, and geometry.

### 8. Kubric rotation equivariance

Each supported object-centric pair is rendered at four raster rotations. The
physical anchor frame and gold vector do not change; the prompt explicitly says
to undo the display rotation. This is an invariance test, not a claim that the
3D scene or camera was physically rotated. Report prediction stability and
accuracy across angles using the same `base_pair_id`.

### 9. ScanNet RGB/depth oracle

The camera-frame task progressively provides reference/target bboxes, their
camera-front depths, complete camera XYZ centroids, and exact target-minus-
reference displacement. Centroids and displacement are privileged geometry
oracles. A bbox gain indicates localization difficulty; a depth gain implicates
metric depth; failure with displacement indicates component semantics,
arithmetic, or output formatting.

## Metrics

Every submission records direction accuracy, vector-derived direction accuracy,
full 3D cosine, horizontal cosine, three component-sign accuracies, combined
internal consistency, component accuracy, and parse success. Metrics that do
not apply to a response condition remain zero in the lmms-eval aggregate; use
the condition-aware analyzer rather than interpreting a task-wide average that
mixes response formats.

## Running

Run the full 22-task matrix:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
  --tasks direction_vector_diagnostics \
  --batch_size 1 --log_samples \
  --output_path outputs/direction_vector_diagnostics
```

Each concrete task can be run independently. The combined group is large;
separate scheduler jobs are generally preferable.

Dataset-only groups are also registered as
`direction_vector_diagnostics_comfort`, `direction_vector_diagnostics_scannet`,
and `direction_vector_diagnostics_kubric`.

Analyze one model/run at a time:

```bash
python tools/analyze_direction_vector_diagnostics.py \
  --inputs outputs/direction_vector_diagnostics \
  --output-dir outputs/direction_vector_diagnostics/analysis
```

The tool writes `summary.json`, `summary.md`, per-experiment CSVs, per-direction
CSVs, conditional outcome tables, and one accuracy/cosine plot per experiment.
It namespaces outputs by dataset, writes a unified comparison CSV and
cross-dataset plots, matched transition tables for every
basis/arrow ladder and the ScanNet oracle tiers, separate direction-only versus
vector-only outcome tables per dataset, and `kubric_rotation_stability.csv`
with semantic and pairwise-vector invariance.
