# COMFORT direction versus object

`comfort_direction_object` converts COMFORT rows containing three distinct
objects (variable, reference, and addressee) plus the camera into two matched
two-option questions. The direction variant
asks for the variable object's relation to the reference object. The object
variant states that same relation and asks for the variable object.

The pair retains the source image, source qid, scene, viewpoint, objects,
relation, and frame deviation. The object choices are only the two non-anchor
objects: the target and the addressee. The direction choices are reduced to the
gold relation and one randomly selected incorrect relation. Both option sets
are shuffled with a deterministic per-sample seed. No synthetic or absent
object choices are added. Rows are dropped when the addressee has the same
relation to the anchor, because both object choices would then be correct.

The task reports overall and per-format accuracy, paired object-minus-direction
gain, the four paired outcomes, parse-success rates, and accuracy for camera,
reference, and addressee viewpoints. Submission records include the exact
model prompt, options, prediction, gold answer, and image path.

Run a smoke evaluation with:

```bash
python -m lmms_eval \
  --model dummy --model_args response=A \
  --tasks comfort_direction_object --limit 10 --batch_size 1 \
  --log_samples --output_path outputs/comfort_direction_object_debug
```
