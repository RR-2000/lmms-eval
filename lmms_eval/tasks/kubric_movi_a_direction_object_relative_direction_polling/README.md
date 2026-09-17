# Kubric relative-direction object polling

This task applies the five-call direction-to-object conversion evaluation to
`movi_a_relative_direction`: one authored direction question followed by the
authored object polls for left, right, front, and behind.

The source export is sparse across directions. The task keeps every map whose
available directions have one unambiguous authored target and polls only that
map's available directions. It retains 222 of 247 maps, giving 356 direction
sources and 1,084 evaluation rows. The other 25 maps contain multiple authored
targets for the same direction and are excluded rather than assigned an
arbitrary single-object ground truth.

The submission contains map coverage, answer and option-letter distributions,
direction confusion matrices, collision diagnostics, end-to-end conversion
accuracy, strict two-stage accuracy, and results split by camera-relative and
object-relative source family.

The two headline metrics are `direct_answer_accuracy` for the original
direction question and `final_polled_answer_accuracy` for the object returned
after routing through the predicted available direction.

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks kubric_movi_a_direction_object_relative_direction_polling \
  --limit 10 --batch_size 1 --log_samples \
  --output_path outputs/kubric_movi_a_relative_direction_polling_debug
```

For trustworthy grouped metrics, run the full 1,084 rows. A generic `--limit`
can split a source group because maps contain different numbers of directions.
