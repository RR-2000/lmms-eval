# COMFORT Oriented 3D direction versus object

`comfort_oriented_3d_direction_object` evaluates the paired annotations in
`/home/ramanathan/data/COMFORT_Oriented_3D/annotations.jsonl` using the same
prompts, predictions, metrics, paired summaries, and submission schema as
`comfort_direction_object`.

The dataset has 500 rendered scenes. Each scene contributes paired direction-
answer and object-answer questions for left, right, front, and behind, for
2,000 matched pairs and 4,000 examples.

Run a smoke evaluation with:

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_oriented_3d_direction_object --limit 10 --batch_size 1 \
  --log_samples --output_path outputs/comfort_oriented_3d_direction_object_debug
```

Analyze a completed submission with:

```bash
python tools/analyze_comfort_direction_object_submission.py \
  outputs/comfort_oriented_3d_direction_object_0/submissions/comfort_oriented_3d_direction_object_<model>.json
```
