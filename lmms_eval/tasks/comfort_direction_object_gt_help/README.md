# COMFORT direction/object GT-help experiments

This task is a diagnostic extension of `comfort_direction_object`. It keeps the
same examples, questions, answer options, generation settings, parser, paired
object/direction structure, and metrics. `GT_HELP` selects one controlled aid
that is prepended to the question and/or applied to the image.

The purpose of the modes is to identify which stage fails:

1. response formatting;
2. understanding the reference-perspective instruction;
3. locating and naming objects;
4. estimating the reference object's heading;
5. changing from camera coordinates to reference coordinates;
6. reading the required relation or object from that representation; and
7. matching the result to an option letter.

Scene metadata comes from
`/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl`. It supplies normalized
2D boxes, camera-frame 3D positions, and the dataset's authoritative objects at
the reference-relative left, right, front, and behind positions.

## Aid and leakage vocabulary

- **None**: does not use current-example ground truth beyond information already
  present in the normal question.
- **Localization**: supplies a box, crop, or identity but not the requested
  spatial answer.
- **Orientation**: supplies the reference heading or axes. This is privileged
  scene information, but the model must still locate the queried object and
  perform the relation lookup.
- **Geometry**: supplies the current scene in canonical reference coordinates.
  This largely removes perception and coordinate transformation.
- **Intermediate oracle**: supplies the intermediate value needed by only one
  answer format. Report the applicable object or direction metric separately.
- **Full oracle**: exposes the current answer text, answer entity, fully labeled
  relation map, or answer letter. These are ceilings and pipeline checks, not
  fair measures of independent task solving.
- **Negative control**: intentionally presents the tempting camera frame. A gain
  here can indicate camera-frame following rather than correct reasoning.

## Complete mode catalog

| Mode | Image given to model | Text added before the normal prompt | Aid/leakage | Primary diagnostic |
|---:|---|---|---|---|
| `0` | Original RGB | None | None | Baseline |
| `1` | Red box on reference | Names/explains reference box | Localization | Can it find the reference? |
| `2` | Green box on gold answer object | Names/explains answer object | Full oracle for object rows; target localization for direction rows | Ceiling and target localization |
| `3` | Labeled boxes on all objects | Explains overlay | Localization | Object detection and name grounding |
| `4` | Unlabeled cyan front arrow on reference | Says arrow is reference-front | Orientation | Does one heading cue suffice? |
| `5` | Original RGB | Gold letter and answer text | Full oracle | End-to-end compliance ceiling |
| `6` | Four direction-word-labeled arrows | Explains reference axes | Orientation | Relation lookup after axes are supplied |
| `7` | Same arrows as `6` | Axes plus reference-perspective rule | Orientation/instruction | Does explicit wording add to axes? |
| `8` | Original RGB | Reference-perspective rule only | None | Task-concept/instruction failure |
| `9` | Original RGB | Current reference answer and camera-frame contrast | Full oracle | Whether a direct frame contrast is followed |
| `10` | Four labeled arrows | Current reference answer and camera contrast | Full oracle | Answer-following ceiling with visual axes |
| `11` | RGB plus named colored-circle canonical map | Color/object legend; reference faces map-top | Geometry | Can it reason from an abstract reference map? |
| `12` | Original RGB | Exact one-letter response rule and neutral format example | None | Output formatting/parsing |
| `13` | Original RGB | Worked robot/ball example unrelated to current scene | None | Learning reference perspective from an example |
| `14` | RGB plus color-only canonical map | Color/object legend; no words inside map | Geometry | Diagram reading without rendered labels |
| `15` | RGB plus object- and direction-labeled canonical map | Explains labels | Full spatial oracle | Relation/object lookup ceiling |
| `16` | RGB plus labeled reference and camera maps | Explicitly says use reference map | Full spatial oracle | Frame-selection ceiling |
| `17` | Original RGB | States reference object's name | None | Textual reference identification control |
| `18` | Red reference box plus magnified crop panel | Explains box and crop | Localization | Small/ambiguous reference appearance |
| `19` | Numbered boxes on every object | Maps numbers to names and marks reference | Localization | Detection/name grounding without relation labels |
| `20` | Reference box plus unlabeled heading arrow | Explains arrow as heading | Orientation | Joint reference localization and heading |
| `21` | Reference box plus arrow labeled `front` | Says derive other axes from front | Orientation | Heading interpretation versus axis derivation |
| `22` | Four arrows with no rendered words | Fixed color legend in text | Orientation | Visual overlay versus in-image text reading |
| `23` | Original RGB | Canonical `(horizontal, forward)` coordinates for every object | Geometry | Symbolic relation calculation without visual geometry |
| `24` | Canonical named map only; RGB removed | Same map legend as `11` | Geometry | Whether RGB appearance distracts from the map |
| `25` | RGB plus map containing reference and answer entity only | Names yellow reference and cyan selected entity | Full oracle for object rows; entity-selection aid for direction rows | Distractor removal |
| `26` | RGB plus all-object map; answer entity cyan, distractors gray | Names highlighted answer entity | Full oracle for object rows; attention oracle for direction rows | Entity selection with clutter retained |
| `27` | RGB plus neutral-colored camera-coordinate map only | Describes camera map | Negative control | Camera-frame bias |
| `28` | RGB plus labeled reference and camera maps | Describes both, but does not command frame selection | Full spatial information | Does the model select the requested frame? |
| `29` | Same visual information as `28`/`16` | Explicitly says answer from reference map | Full spatial oracle | Effect of explicit frame-selection instruction |
| `30` | Original RGB | States which camera direction reference-front aligns with | Orientation | Text orientation cue versus visual arrow |
| `31` | Original RGB | Lists every reference axis as a normalized camera-plane vector | Orientation/transform | Relation reasoning after frame transform is supplied |
| `32` | Original RGB | Reveals gold relation on object-answer rows only | Intermediate oracle | Object retrieval once relation is known |
| `33` | Green target box on direction-answer rows only | Explains target box | Localization oracle | Direction classification once target is located |
| `34` | Original RGB | Reveals correct answer text, not its letter | Full oracle | Answer-text to option-letter matching |
| `35` | Original RGB | Reveals correct option letter only | Full oracle | Pure output compliance ceiling |

Modes `16` and `29` intentionally duplicate the explicit dual-map condition.
Mode `16` preserves the earlier experiment numbering; mode `29` completes the
new controlled `27`/`28`/`29` comparison.

## How the prompt and image are formed

`doc_to_text` first creates the unchanged base prompt:

```text
Answer this spatial-reasoning question using the image. Select one answer
option and respond with its letter.
Question: ...
Options:
A. ...
B. ...
C. ...
D. ...
```

The selected mode's text functions are joined and prepended to that prompt.
The question, options, ordering, gold letter, and parser are never rewritten.

Most visual modes retain the original RGB image. Diagram modes append square
panels to its right. Mode `24` is the exception: it replaces the RGB image with
the canonical map so that it can be compared directly with mode `11`.

Top-down map colors are fixed across examples:

- yellow: reference;
- red: reference-left object;
- blue: reference-right object;
- green: reference-front object;
- purple: reference-behind object.

Modes `25` and `26` instead use yellow for the reference, cyan for the selected
question/answer entity, and gray for distractors. That cyan selection leaks the
answer identity on object-answer rows, so those rows must be treated as oracle
conditions. Mode `27` uses yellow for the reference and the same neutral blue-gray
for every other object so its camera-frame control contains no relation-color cue.

## Recommended comparisons

Run all compared modes on exactly the same records, model checkpoint, decoding
settings, and option permutations. Prefer paired per-question changes over
comparing only aggregate accuracies.

| Comparison | Interpretation |
|---|---|
| `0` vs `12` | Gain isolated to response formatting; confirm with parse-success rate |
| `0` vs `8` vs `13` | Instruction alone versus worked-example task understanding |
| `17` vs `1` vs `18` | Naming, boxing, and magnification of the reference |
| `1` vs `3` vs `19` | Reference localization versus complete object grounding |
| `4` vs `20` | Effect of adding an explicit reference box to the same heading cue |
| `20` vs `21` | Unlabeled heading cue versus in-image `front` label |
| `21` vs `6` | One heading axis versus all four supplied axes |
| `6` vs `22` | Direction words rendered in image versus color legend in prompt |
| `6` vs `7` | Whether the perspective explanation adds value after axes are visible |
| `4` vs `30` | Visual heading cue versus textual camera-frame heading cue |
| `30` vs `31` | One heading mapping versus the complete coordinate transform |
| `11` vs `14` | Object labels inside the diagram versus prompt-only color grounding |
| `11` vs `24` | Same canonical representation with and without the RGB scene |
| `11` vs `25` | Full scene map versus removal of all distractors |
| `25` vs `26` | Distractors removed versus retained but de-emphasized |
| `0` vs `27` | Sensitivity to a tempting camera-coordinate representation |
| `28` vs `29` | Independent frame choice versus explicit reference-frame command |
| `27` vs `28` | Camera map alone versus both candidate coordinate frames |
| object metric: `0` vs `32` | Object selection after the correct relation is supplied |
| direction metric: `0` vs `33` | Direction classification after the named target is boxed |
| `34` vs `35` vs `5` | Option matching, letter compliance, and combined oracle ceiling |

Interpret a gain only at the stage modified by the comparison. In particular,
do not describe `2`, `5`, `9`, `10`, `15`, `16`, `25`, `26`, `28`, `29`,
`34`, or `35` as improved independent spatial reasoning.

For each comparison report at least:

- overall, object-answer, and direction-answer accuracy;
- overall/object/direction parse-success rate;
- object-correct/direction-wrong and direction-correct/object-wrong paired rates;
- the number of baseline wrong-to-aid correct and baseline correct-to-aid wrong
  transitions on identical `qid`s; and
- results split by whether the reference answer agrees with the camera-frame
  answer, when testing frame-selection hypotheses.

The object and direction rows are matched by `source_relation_id`. Modes `32`
and `33` are asymmetric by design: use only object accuracy for `32` and only
direction accuracy for `33`.

## Running modes

Run one mode:

```bash
GT_HELP=20 python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object_gt_help \
  --limit 10 --batch_size 1 --log_samples \
  --output_path outputs/comfort_direction_object_gt_help_20
```

Run the full catalog with separate output directories:

```bash
for gt_help in $(seq 0 35); do
  GT_HELP="${gt_help}" python -m lmms_eval \
    --model dummy --model_args response=A \
    --tasks comfort_direction_object_gt_help \
    --limit 10 --batch_size 1 --log_samples \
    --output_path "outputs/comfort_direction_object_gt_help_${gt_help}"
done
```

The experiment summarizer is
`tools/summarize_comfort_direction_object_gt_help_experiments.py`. Point it at
the common parent directory after evaluations finish.

## Saving model-visible images

Set `GT_HELP_DEBUG=1` to save the final image passed to the model. Override the
default directory with `GT_HELP_DEBUG_DIR`:

```bash
GT_HELP=24 \
GT_HELP_DEBUG=1 \
GT_HELP_DEBUG_DIR=outputs/gt_help_24_debug_images \
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object_gt_help --limit 10 --batch_size 1
```

Files are named `gt_help_<mode>_<question-id>.png`. Debug images are also
materialized from `doc_to_text`, because lmms-eval's dummy model may not resolve
visual arguments.

## Composing new conditions

`GT_HELP_PRESETS` in `utils.py` maps each mode to tuples of ordinary visual and
text functions:

```python
"20": {
    "visual": (draw_reference_bbox_and_heading_arrow,),
    "text": (describe_reference_heading_arrow,),
},
```

Several aids can be composed by placing multiple functions in the tuples. The
reusable metadata helpers are `load_scene_index`, `get_scene`,
`get_scene_objects`, `get_reference_object`, `get_answer_object`,
`get_object_at_direction`, `get_bbox_normalized`, `bbox_to_pixels`,
`get_camera_position`, and `get_camera_orientation`.
