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
