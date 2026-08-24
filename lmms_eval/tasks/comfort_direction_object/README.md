# COMFORT Multi 3D direction versus object

`comfort_direction_object` evaluates the native paired annotations in
`/home/ramanathan/data/COMFORT_Multi_3D/annotations.jsonl`.

Each of the 200 rendered scenes contains one reference object and four target
objects placed at its left, right, front, and behind directions. For every
scene/relation, the dataset supplies a direction-answer prompt and an
object-answer prompt with the same image, reference, target, relation, and gold
option position. Four controlled permutations place the answer once at each of
A, B, C, and D, producing 3,200 matched pairs and 6,400 examples.

The task uses the dataset's questions and options directly. It does not add,
remove, or reshuffle options. Paired metrics group only rows sharing scene,
relation, and answer position. Submission records include the exact prompt,
options, prediction, gold answer, and resolved local image path.

Run a smoke evaluation with:

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object --limit 10 --batch_size 1 \
  --log_samples --output_path outputs/comfort_direction_object_debug
```
