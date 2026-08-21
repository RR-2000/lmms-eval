# GQA direction-versus-object diagnostic

`gqa_direction_object` is a paired multiple-choice diagnostic built from the
official GQA scene graphs and the same GQA images used by the existing `gqa`
task. It does not modify `lmms_eval/tasks/gqa`.

For every accepted scene-graph edge, the subject is the target object, the
relation's referenced object is the anchor, and the predicate is normalized to
one of `left`, `right`, `above`, or `below`. The builder emits two rows with the
same `source_relation_id`, image, target, anchor, relation, and stored object
candidate pool:

- direction: “Where is the target relative to the anchor?” with the fixed
  choices `left`, `right`, `above`, and `below`;
- object: “Which object is [relation] the anchor?” with four object names from
  that image.

Both prompts require only an A-D option letter. Every JSONL row stores the same
absolute local path in `image` and `image_path`; `process_docs` validates it and
`doc_to_visual` loads that file directly.

## Building the data

All source and generated data lives under `/home/ramanathan/data/GQA`. Download
and extract the [official GQA images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
there, and place the [official v1.1 scene graphs](https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip)
at `/home/ramanathan/data/GQA/sceneGraphs.zip`. Then run from the repository
root:

```bash
python tools/build_gqa_direction_object.py \
  --scene-graphs /home/ramanathan/data/GQA/sceneGraphs.zip \
  --image-dir /home/ramanathan/data/GQA/images \
  --gqa-split val \
  --output /home/ramanathan/data/GQA/gqa_direction_object.jsonl \
  --sample-output /home/ramanathan/data/GQA/sample_pairs.jsonl \
  --audit-report /home/ramanathan/data/GQA/audit_report.json
```

The task YAML reads `/home/ramanathan/data/GQA/gqa_direction_object.jsonl`.
`--seed` controls deterministic distractor selection and option shuffling. If
the input is the official zip, `--gqa-split val` selects only
`val_sceneGraphs.json`. Missing image files are rejected unless the explicit
`--allow-missing-images` debugging flag is supplied.

The accepted relation spellings are:

- left: `left`, `left of`, `to the left of`, `on the left of`;
- right: `right`, `right of`, `to the right of`, `on the right of`;
- above: `above`, `above of`, `over`, `on top of`;
- below: `below`, `below of`, `under`, `underneath`.

Rows are skipped when target or anchor names are empty/non-unique, when more
than one displayed object option satisfies the relation to the anchor, when
fewer than three unique distractor names remain, or when annotations/images
are invalid. Correctness is checked after deterministic option selection and
recorded in `diagnostic_correct_option_object_ids`,
`diagnostic_correct_object_options`, and `num_correct_object_options`. Because
names alone cannot identify duplicate instances symmetrically, the builder
skips duplicate target/anchor names rather than adding IDs to only one prompt.

## Audit and metrics

Audit an existing cache with:

```bash
python tools/build_gqa_direction_object.py \
  --audit-only /home/ramanathan/data/GQA/gqa_direction_object.jsonl \
  --audit-report /home/ramanathan/data/GQA/audit_report.json
```

The audit verifies matched metadata, exactly four choices, correct answer
indices, unique object options, and one row of each format per source relation.
It reports gold-letter, relation, and answer-format distributions. Prediction
distribution, per-format parse failures, and accuracy strata by relation,
format, gold/predicted letter, anchor, target, and option count are logged and
written into the evaluation submission JSON.

The scalar task metrics are overall/object/direction accuracy, parse-success
rates, paired object-minus-direction (also exposed as `format_switch_gain`),
and the four paired outcome rates. Paired aggregations use only complete
`source_relation_id` groups; an unmatched row never contributes.

## Validation

```bash
python -m lmms_eval --tasks list | grep gqa_direction_object
python -m lmms_eval \
  --model <MODEL_NAME> --model_args <ARGS> \
  --tasks gqa_direction_object --batch_size 1 --limit 10 \
  --log_samples --output_path outputs/gqa_direction_object_debug
```

The five matched prompt pairs in `/home/ramanathan/data/GQA/sample_pairs.jsonl`
are suitable for manual prompt inspection. Evaluation uses only the local image
paths and does not download GQA images through Hugging Face.
