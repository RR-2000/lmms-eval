# COMFORT object-versus-direction inverse diagnostics

This task family isolates why a model can correctly answer “which object is at
this relation?” while failing the inverse question “which relation contains
this object?”, or vice versa. It is separate from the numbered
`comfort_direction_object_gt_help` task and is invoked as:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=32 \
  --tasks comfort_inverse_diagnostics \
  --batch_size 1 --log_samples \
  --output_path outputs/comfort_inverse_diagnostics
```

The group contains four independently runnable tasks:

| Task | Question count | Main comparison |
|---|---:|---|
| `comfort_full_map_inversion` | 5,000 | relation→object versus object→relation maps under five cues |
| `comfort_arrow_length_sweep` | 24,000 | object versus direction accuracy at six arrow lengths |
| `comfort_map_ablation` | 32,000 | object versus direction accuracy under eight map components |
| `comfort_option_permutation` | 16,000 | stability across all four cyclic option orders |

Counts assume the current 500-scene dataset. The arrow and map experiments use
one deterministic, balanced answer permutation per scene/relation. This keeps
gold A/B/C/D positions balanced without multiplying every visual condition by
four. The permutation experiment retains all four orders explicitly.

Object strings are deliberately kept exactly as they appear in the source
annotations. In particular, `horsel` and `horser` remain distinct in this task
family: collapsing both to `horse` would make the object-keyed full map
ambiguous whenever both instances occur in one scene.

Set `COMFORT_INVERSE_DEBUG=1` to save exact model-visible images. Override the
root with `COMFORT_INVERSE_DEBUG_DIR`; images are organized by experiment and
condition.

## 1. Full-map inversion

### Purpose

The original tasks are inverse retrieval problems, not merely different output
types. This experiment asks for the complete four-edge spatial map so every
question contains the same evidence and no single queried relation or object
is privileged.

Five visual cues are crossed with two mapping formats:

| Cue | Image supplied |
|---|---|
| `none` | original RGB scene |
| `long_arrows` | four GT_HELP-6-length labeled arrows |
| `short_arrows` | four GT_HELP-36-length labeled arrows |
| `named_map` | original image plus named canonical top-down map |
| `color_map` | original image plus color-only map and a textual color→object legend |

### Relation-keyed prompt

```text
Four long labeled arrows on the reference show its left, right, front, and behind.
Use the dog’s own viewpoint and recover the complete mapping of all four
surrounding objects. Return only one JSON object with exactly these keys: left,
right, front, behind. Each value must be the name of the object at that position.
Required schema:
{"left":"<object>","right":"<object>","front":"<object>","behind":"<object>"}
```

Example answer:

```json
{"left":"sofa","right":"horser","front":"car sedan","behind":"bicycle mountain"}
```

### Object-keyed prompt

```text
Use the dog’s own viewpoint and recover the complete mapping of all four
surrounding objects. Return only one JSON object whose four keys are the
surrounding object names and whose values are exactly one of left, right,
front, or behind.
Required schema:
{"sofa":"<direction>","horser":"<direction>","car sedan":"<direction>","bicycle mountain":"<direction>"}
```

Example answer:

```json
{"sofa":"left","horser":"right","car sedan":"front","bicycle mountain":"behind"}
```

### Measurements

- `mapping_edge_accuracy`: fraction of the four key/value edges correct.
- `mapping_exact_accuracy`: all four edges correct with exactly four keys.
- `parse_success_rate`: valid JSON-object rate.

Compare the two formats within each cue. A strong relation-keyed result and weak
object-keyed result means the scene mapping is available but direction words are
harder to retrieve as values. If both improve with long arrows, the bottleneck
is spatial evidence rather than map inversion.

## 2. Arrow-length sweep

### Purpose

GT_HELP 6 and 36 differ primarily in arrow length and show a large change in
the object/direction gap. This task tests whether the model interprets arrows as
semantic axes or as visual pointers that need to approach a target.

The tested lengths are `0.25`, `0.45`, `0.70`, `0.90`, `1.15`, and `1.50` times
the reference bbox diagonal. Minimum pixel lengths scale from 10 to 48 pixels.
All other arrow properties are identical: origin, semantic direction, colors,
labels, width, and prompt.

Example direction-answer prompt:

```text
Four labeled reference-relative arrows are overlaid on the reference object.
Their length is 0.70 times the reference bbox diagonal.
Answer this spatial-reasoning question using the image. Select one answer
option and respond with its letter.
Question: From the dog’s current viewpoint, where is the sofa?
Options:
A. right
B. front
C. left
D. behind
```

The paired object-answer prompt uses the same rendered scene and length but asks
which object lies along a named direction.

### Measurements

- Overall, object, and direction accuracy.
- `object_minus_direction`.
- Object-only, direction-only, both-correct, and both-wrong paired rates.
- Parse success.
- Condition field `arrow_length_scale` for continuous length plots.

A gradual improvement suggests visibility/legibility. A sharp improvement when
tips approach target boxes suggests pointer following. Direction improving while
object accuracy remains flat suggests better axis interpretation.

## 3. Canonical-map ablation

### Purpose

GT_HELP 11 is nearly perfect for object answers yet has a severe `behind`
direction failure. This experiment removes or transforms one map component at a
time while keeping the original RGB scene alongside the diagnostic panel.

| Condition | Object names | Direction labels | Colors/targets | Heading | Rotation |
|---|---:|---:|---:|---:|---:|
| `object_names_only` | yes | no | yes | yes | 0° |
| `direction_labels_only` | text legend | yes | yes | yes | 0° |
| `object_and_direction_labels` | yes | yes | yes | yes | 0° |
| `colors_only` | text legend | no | yes | yes | 0° |
| `heading_only` | no | no | no | yes | 0° |
| `object_names_no_heading` | yes | no | yes | no | 0° |
| `rotated_labeled` | yes | yes | yes | yes | 90° |
| `rotated_unlabeled` | yes | no | yes | yes | 90° |

Example prompt:

```text
The canonical map labels objects but not direction axes.
Answer this spatial-reasoning question using the image. Select one answer option
and respond with its letter.
Question: Which object is in the direction that the behind side of the dog
points toward?
Options:
A. sofa
B. bicycle
C. horse
D. car
```

### Measurements

The metrics match the arrow sweep and are reported by map condition, answer
format, and ground-truth relation.

Key contrasts:

- Names only versus names+directions: whether explicit axis words repair inverse lookup.
- Colors only versus names only: symbol grounding versus text reading.
- Names only versus no heading: dependence on the front marker.
- Upright versus rotated: learned “top=front/bottom=behind” heuristics.
- Direction-label-only versus object-label-only: which side of the inverse map needs grounding.

## 4. Option-permutation consistency

### Purpose

Each semantic question is repeated under the four cyclic option orders already
present in `annotations.jsonl`. The image and question are unchanged. Only the
mapping from semantic answer to A/B/C/D changes.

Example repetitions:

```text
A. left   B. right  C. front  D. behind
A. behind B. left   C. right  D. front
A. front  B. behind C. left   D. right
A. right  B. front  C. behind D. left
```

The model must select different letters while preserving the same semantic
answer.

### Measurements

- Standard overall, object, and direction accuracy.
- `permutation_semantic_consistency`: all four selected option texts are identical.
- `permutation_all_correct_rate`: all four permutations select the gold semantic answer.
- Accuracy by `permutation_index` and predicted-letter distribution in the submission.

High single-run accuracy with low semantic consistency indicates option-position
sensitivity. High semantic consistency with low accuracy indicates a stable
spatial misconception rather than letter bias.

## Comparison protocol

For the arrow and map tasks, always report a condition-by-format table:

```text
condition | object accuracy | direction accuracy | object−direction
          | object only | direction only | both correct | both wrong
```

Also split every row by `left`, `right`, `front`, and `behind`. The existing
results show that aggregate object superiority is concentrated in front/behind,
so an overall mean alone is misleading.

Use paired comparisons because object and direction rows share a scene and gold
relation. For uncertainty, bootstrap by `scene_id`, not by individual question,
so repeated relations and permutations from one scene remain in the same
resample.

The most diagnostic reading order is:

1. Full-map inversion: can the model represent the complete bijection?
2. Option permutation: is the difference an option-letter artifact?
3. Arrow sweep: does usable axis extent explain the gap?
4. Map ablation: which representation component creates or removes the gap?

## Result summarizer

After evaluation, run:

```bash
python tools/analyze_comfort_inverse_diagnostics.py \
  --inputs outputs/comfort_inverse_diagnostics \
  --output-dir outputs/comfort_inverse_diagnostics/inverse_diagnostic_analysis
```

The input may be the evaluation root, a submissions directory, or individual
submission JSON files. Analyze one model/run at a time so results from different
models are not silently combined.

The tool writes `summary.json` and `summary.md`, plus condition tables in CSV.
It produces the following plots when the corresponding task is present:

- `full_map_inversion.png`: four-edge accuracy by cue and JSON orientation.
- `arrow_length_accuracy.png`: object/direction accuracy across arrow lengths.
- `map_ablation_accuracy.png`: object/direction accuracy across map components.
- `option_position_accuracy.png`: accuracy by cyclic answer position.
- `option_permutation_stability.png`: semantic consistency and all-four-correct
  rates; this plot requires complete four-permutation groups.

The arrow and map CSVs also include relation-level and paired-outcome tables,
so the aggregate gap can be traced to a specific relation and to
object-only/direction-only cases rather than inferred from two independent
means.
