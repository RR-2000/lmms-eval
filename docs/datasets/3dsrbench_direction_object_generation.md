# 3DSRBench Direction–Object Diagnostic: Generation and Provenance

This document records the implementation and the measured contents of the
`3dsrbench_direction_object` task as of 2026-09-22. Counts were recomputed by
running the current transformation over the exact source parquet identified
below; they are not inferred from evaluation outputs.

## Executive answer: which tool generated it?

No program in `tools/` generated this dataset. The dataset is constructed in
memory when LMMS-Eval loads the task:

- Task configuration: `lmms_eval/tasks/3dsrbench_custom/3dsrbench_direction_object.yaml`
- Source-data configuration: `lmms_eval/tasks/3dsrbench_custom/_default_template_yaml`
- Generator: `direction_object_process_docs()` in
  `lmms_eval/tasks/3dsrbench_custom/utils.py`
- Post-hoc result analyzer only (not a dataset generator):
  `tools/analyze_3dsrbench_direction_object_submission.py`

`tools/build_3dsr_prompt_variants_dataset.py` is unrelated. It creates
model-written natural-language prompt variants from a saved submission and is
not referenced by `3dsrbench_direction_object`.

The task and generator were introduced in Git commit
`91777fcf7d3ac255f1c8e5256a8ab3b21b4b7df7` on 2026-07-21 (commit message:
“Clean GT fix and Added Object Vs Dir Testing”). Deterministic thematic
distractor padding was added in commit
`dfd7e3b6b143a8e760e70bea722948d49107ae28` on 2026-08-21. The repository HEAD
used for this audit is `feafee20b959b404a096a3e6efc941434d1fe099`.

## Purpose and experimental unit

The task is a matched diagnostic of two ways of querying the same annotated
3D relation in a single image:

1. **Native/direction formulation:** identify which side of a subject
   (`left`, `right`, `front`, or `back`) faces a target object.
2. **Inverse/object formulation:** after supplying that side in the question,
   identify which object lies in that direction.

Every retained source relation produces exactly one native row and one inverse
row. The appropriate experimental unit for a paired comparison is therefore a
source relation (`source_qid`), not an individual prompt. Images are reused,
so neither 238 prompts nor 119 relations should be treated as independent
images.

## Source data

The task loads the `test` split from:

`/home/ramanathan/data/3DSR/dataset.parquet`

The local manifest describes this file as a 3DSRBench bounding-box export with
SAM3-derived boxes. Its image metadata identifies the image source as MS-COCO;
the selected records point to COCO `train2017` images. The diagnostic uses the
original RGB image (or a horizontal flip for `-flip` rows), not a box-overlay
image. Bounding-box coordinates are not inserted into the default diagnostic
prompt. Object names from `bbox_items` are, however, eligible for the
same-image distractor pool.

Source snapshot:

| Property | Value |
|---|---:|
| Source rows | 5,157 |
| Source columns | 30 |
| Parquet size | 1,049,380 bytes |
| Parquet modification time | 2026-06-11 15:35:37.872158790 +0800 |
| SHA-256 | `d503529754e9b24baf005db58654af70ae5cb7f35f25866836e5c19905465dca` |
| Candidate relation rows | 343 |
| Retained source relations | 119 |
| Excluded candidate rows | 224 |
| Final prompts | 238 (119 matched pairs) |
| Unique source images retained | 32 |
| Unflipped / horizontally flipped relations | 61 / 58 |

The task is evaluation-only: the YAML exposes these records as a `test` split
and defines no training or validation split.

## Exact generation procedure

The transformation is deterministic and does not call a language model.

1. Load all 5,157 parquet rows.
2. Group object terms by a stable image key, chosen in this order:
   `image_name`, `image_url`, `resolved_img_path`, `img_path`, then `qid` or
   `index`. For every row sharing an image, collect `subject`, `object1`
   (falling back to `object_1`), `object2` (falling back to `object_2`), and
   every value in `bbox_items`, preserving first-seen order and exact spelling.
3. Select rows whose normalized fields are exactly
   `qtype = "multi_object"` and
   `relation = "viewpoint towards object"`.
4. Resolve the source gold answer letter to its displayed option text and
   lowercase/whitespace-normalize it. Retain only directions in
   `{left, right, front, back}` and require nonempty `subject` and `object1`.
   All 343 candidate rows have a valid direction; the later distractor
   requirement accounts for the 224 exclusions in this snapshot.
5. Define `subject` as the oriented entity and `object1` as the target object.
   Build same-image distractors from the pooled terms, excluding the subject,
   target, and names considered the same entity. Two names are considered the
   same if, after lowercase/whitespace normalization, they are equal or either
   string is a substring of the other.
6. Require at least one distinct, image-derived distractor in addition to the
   target. Rows failing this condition are excluded. This produces 119 source
   relations from the 343 candidates (34.69% retention).
7. Create four inverse-answer options. Start with the annotated target, then
   take distinct same-image candidates in first-seen order. If fewer than four
   options are available, infer a broad theme from the target label and pad
   from a fixed thematic pool; if needed, continue through a generic pool and
   finally labels of the form `thematic scene object N`.
8. Shuffle the inverse options with a separate deterministic Python RNG seeded
   by the string `3dsrbench_direction_object_v1:<source_qid>`. This does not
   mutate global RNG state. Set the inverse gold letter to the shuffled
   position of the target.
9. Copy the source row into two records and attach audit metadata:
   `source_qid`, `source_task_family`, `diagnostic_subject`,
   `diagnostic_target_object`, `diagnostic_direction`,
   `diagnostic_sample_seed`, and
   `diagnostic_generated_object_distractors`.
10. For the native record, append `::native` to `qid` and `index`, preserve the
    source question/options/answer, set `diagnostic_variant = "native"`,
    `diagnostic_answer_format = "direction"`, and use the direction as the
    diagnostic target.
11. For the inverse record, append `::inverse`, replace the question with the
    fixed template below, replace A–D and the answer letter with the generated
    four-choice set, set `diagnostic_variant = "inverse"`,
    `diagnostic_answer_format = "object"`, and use `object1` as the target.

Inverse question template:

> Consider the real-world 3D locations and orientations of the objects. Which
> object is in the direction that the `{direction}` side of the `{subject}`
> points toward?

At visual-loading time, a local image path is preferred. If `flip` occurs in
the derived `index`, the RGB image is horizontally flipped. Otherwise, the
original RGB image is returned.

## Thematic distractor construction

Theme inference is a fixed keyword match over the target string. The themes
are checked in insertion order: vehicle, signage, person, animal, furniture,
kitchen, and street; unmatched targets use `generic`. Padding uses the
following fixed pools:

- vehicle: car, bus, van, bicycle, motorcycle, boat
- signage: traffic sign, street sign, parking sign, billboard, traffic light
- person: person, man, woman, pedestrian, child
- animal: dog, cat, horse, bird, cow, elephant
- furniture: chair, table, couch, bed, cabinet, lamp
- kitchen: refrigerator, oven, microwave, sink, bottle, cup
- street: fire hydrant, parking meter, bench, trash can, bus stop
- generic: nearby object, background object, scene landmark, other item

The current snapshot contains 357 inverse distractor positions. Of these, 146
(40.90%) come from same-image source annotations and 211 (59.10%) are thematic
padding. A total of 115/119 inverse questions (96.64%) contain at least one
thematic distractor: 4 contain zero, 19 contain one, and 96 contain two.
Thematic labels are chosen for semantic plausibility but are not verified to
be visible in the image; this must be disclosed when presenting results.

## Dataset statistics

| Statistic | Count |
|---|---:|
| Matched source relations | 119 |
| Native direction prompts | 119 |
| Inverse object prompts | 119 |
| Total prompts | 238 |
| Unique images | 32 |
| Unique subjects | 37 |
| Unique target-object strings | 45 |
| Duplicate output QIDs | 0 |

Direction distribution over the 119 matched relations:

| Direction | Count | Percent |
|---|---:|---:|
| left | 34 | 28.57% |
| right | 33 | 27.73% |
| front | 30 | 25.21% |
| back | 22 | 18.49% |

Both formulations have four answer choices in all 119 records. Gold-letter
distributions are:

| Formulation | A | B | C | D |
|---|---:|---:|---:|---:|
| Native/direction | 30 | 34 | 22 | 33 |
| Inverse/object | 24 | 35 | 30 | 30 |

## Evaluation protocol implemented by the task

The rendered prompt is:

> Answer this spatial-reasoning question using the image. Select one answer
> option and respond with its letter.  
> Question: `{question}`  
> Options:  
> A. ...  
> B. ...  
> C. ...  
> D. ...

For native rows, `original_question` is preferred; for inverse rows it is
cleared so the generated inverse question is used. Decoding is deterministic
in the task configuration (`temperature = 0`, `top_p = 1`, `do_sample = false`,
`max_new_tokens = 4096`). The evaluator extracts the first recognizable A–D
answer using a sequence of regular expressions and assigns exact-match binary
credit against the gold letter.

Reported metrics are:

- `accuracy`: mean binary accuracy over all 238 prompts.
- `direction_answer_accuracy`: mean over the 119 native direction prompts.
- `object_answer_accuracy`: mean over the 119 inverse object prompts.
- `format_switch_gain`: mean, across complete `source_qid` pairs, of
  `inverse_score - native_score`.
- `object_minus_direction`: the same paired difference grouped by answer
  format; with this construction it is numerically equivalent to
  `format_switch_gain`.

The file `tools/analyze_3dsrbench_direction_object_submission.py` consumes the
saved per-example submission JSON after evaluation. It summarizes native and
inverse accuracy and counts four paired outcomes: improved, worsened,
unchanged-correct, and unchanged-incorrect, overall and by direction. It does
not construct or modify dataset examples.

## Conference-ready methods paragraph

> We derived a matched direction–object diagnostic from the
> `multi_object/viewpoint towards object` subset of 3DSRBench. For each eligible
> relation, we retained the native four-way question asking which side of a
> subject (left, right, front, or back) faces a target object, and constructed
> an inverse four-way question that supplied the annotated side and asked for
> the corresponding target object. Object distractors were drawn first from
> other annotated object terms associated with the same COCO image; examples
> without at least one such distractor were excluded. The remaining option
> slots were padded deterministically from fixed, target-conditioned semantic
> pools, and option order was shuffled with a per-example seed. This yielded
> 119 matched relations (238 prompts) over 32 unique images: 34 left, 33 right,
> 30 front, and 22 back. Fifty-eight relations use the benchmark's
> horizontally flipped image variant. Models answered by selecting an A–D
> option under deterministic decoding. We report accuracy separately for the
> native direction and inverse object formulations and the paired per-relation
> difference (object minus direction).

## Reporting caveats and claims to avoid

- Describe this as a **derived diagnostic/task**, not a separately annotated
  independent dataset. It is generated at evaluation time from a local
  3DSRBench parquet export.
- State both 119 matched relations and 32 unique images. Avoid treating all
  prompts or relations as independent image samples.
- The two rows are semantically matched but do not differ only in output
  syntax: the question wording, supplied information, answer domain, and
  distractors change. “Direction versus object formulation” is more precise
  than “answer-format-only intervention.”
- Disclose thematic padding: 211/357 distractor slots are generated labels and
  are not visibility-verified. Calling all distractors image-grounded would be
  incorrect for the current code.
- The source manifest says its bounding boxes are SAM3-derived, but the default
  diagnostic does not display overlays or coordinates. Do not attribute gains
  to bounding-box input.
- License, annotator protocol, demographic properties, and the original
  3DSRBench collection procedure are not recoverable from this task code or the
  local export README. Those facts must be sourced from the original benchmark
  publication/repository before submission.

## Reproducibility identifiers

| Artifact | SHA-256 |
|---|---|
| `dataset.parquet` | `d503529754e9b24baf005db58654af70ae5cb7f35f25866836e5c19905465dca` |
| `utils.py` | `89ecebc75aa677d510d571943742fe492d2d6ef6fa742b4c5e1c6a8352b72979` |
| `3dsrbench_direction_object.yaml` | `2442eb45df52feeb6604056fa3818dcd05b8d08b967ed9cc22e5e6bad4937a26` |
| `_default_template_yaml` | `d88dc22f9231ceea7e7a5a179f36eef5374308b82807de0744231083cf428cc5` |
| Analyzer | `5d630426c0fea81e97027ce0d0cf63985b123aa1e6c2005370fc9a2724bc1c3f` |

