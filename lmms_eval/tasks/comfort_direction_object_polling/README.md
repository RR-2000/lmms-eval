# COMFORT direction-to-object polling

This task evaluates a two-stage conversion for every native direction-answer
question in `COMFORT_Multi_3D`:

1. Ask the original direction question.
2. Poll the same image with object-answer questions for left, right, front,
   and back/behind.
3. Resolve the predicted direction to its corresponding poll and use that
   poll's selected object as the converted answer.

Each source direction row produces five evaluation rows. The four polls reuse
the native object options with the same controlled gold answer position as the
source direction row. The submission JSON includes raw records plus semantic
answer distributions, option-letter distributions, a direction-confusion
matrix, full-pipeline accuracy, exact-map accuracy, and collision statistics.
It also separates strict two-stage success from accidental error cancellation
and measures semantic-answer consistency across the four option permutations.

The two headline conversion metrics are `direct_answer_accuracy` for the
original direction question and `final_polled_answer_accuracy` for the final
object returned after routing through the predicted direction. The older
`direction_accuracy` and `full_pipeline_accuracy` names remain as aliases.

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object_polling --limit 20 --batch_size 1 \
  --log_samples --output_path outputs/comfort_direction_object_polling_debug
```

Use limits divisible by five so every source group remains complete.

## Target-only polling with ground-truth orientation

`comfort_direction_object_target_only_gt_polling` follows the polling and
decision procedure from `concrete_direction.target_only_inversion
--no-geometry`. It does not ask the original four-way direction question.
Instead, every source direction question produces four binary polls, one per
direction, whose choices are the named target and `None of the above`. The
target/none order is deterministically shuffled in the same way as the
standalone pipeline. The final direction is the uniquely highest target vote;
tied votes are reported as ambiguous.

Each model-visible image has the four short, labeled reference axes drawn from
COMFORT's ground-truth scene metadata using the same renderer as GT-help mode
36. No depth, target geometry, or geometry-score fusion is used.

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object_target_only_gt_polling \
  --limit 16 --batch_size 1 --log_samples \
  --output_path outputs/comfort_direction_object_target_only_gt_polling_debug
```

Use limits divisible by four so every target-only poll group remains complete.

## One-shot structured object map

`comfort_direction_object_structured_map_gt` asks for all four
direction-to-object assignments in a single JSON response. It uses the same
short ground-truth axis overlay as the target-only variant. Candidate object
names are supplied in alphabetical order so their presentation does not encode
their spatial assignment.

The required response is:

```json
{"left":"<object>","right":"<object>","front":"<object>","behind":"<object>"}
```

There is one record per scene. Metrics report per-edge object recovery, exact
four-edge map accuracy, valid-bijection rate, and JSON parse success.

```bash
python -m lmms_eval \
  --model dummy --model_args response='{}' \
  --tasks comfort_direction_object_structured_map_gt \
  --limit 4 --batch_size 1 --log_samples \
  --output_path outputs/comfort_direction_object_structured_map_gt_debug
```
