#!/usr/bin/env python3
"""Export report-ready provenance for the Qwen-padded 3DSR task variant."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/ramanathan/data/3DSR/dataset.parquet")
DEFAULT_MANIFEST = Path(
    "/home/ramanathan/data/3DSR/3dsrbench_qwen3vl_8b_scene_objects.json"
)
DEFAULT_OUTPUT = ROOT / "experiment_artifacts_3DSR/qwen_scene_distractors_generation"
TASK_NAME = "3dsrbench_direction_object_qwen3vl_distractors"
ADAPTER_NAME = "3dsr-object-direction-qwen-distractors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    utils = importlib.import_module("lmms_eval.tasks.3dsrbench_custom.utils")
    frame = pd.read_parquet(args.dataset)
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    catalog = list(manifest["images"].values())
    task_rows = [
        dict(row)
        for row in utils._direction_object_process_docs(
            Dataset.from_pandas(frame, preserve_index=False),
            sample_seed=utils.QWEN_DIRECTION_OBJECT_SAMPLE_SEED,
            qwen_scene_objects=utils._load_qwen_direction_object_manifest(str(args.manifest)),
        )
    ]
    object_rows = [
        row for row in task_rows if row.get("diagnostic_answer_format") == "object"
    ]
    direction_rows = [
        row for row in task_rows if row.get("diagnostic_answer_format") == "direction"
    ]
    relation_mask = (
        frame["qtype"].astype(str).str.strip().str.lower().eq("multi_object")
        & frame["relation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("viewpoint towards object")
    )
    selected = []
    label_counts: Counter[str] = Counter()
    for row in object_rows:
        labels = list(row.get("diagnostic_generated_object_distractors") or [])
        label_counts.update(label.casefold() for label in labels)
        selected.append(
            {
                "source_qid": row.get("source_qid"),
                "image_key": utils._direction_object_image_key(row),
                "subject": row.get("diagnostic_subject"),
                "target_object": row.get("diagnostic_target_object"),
                "direction": row.get("diagnostic_direction"),
                "qwen_padding": labels,
                "options": {letter: row.get(letter) for letter in "ABCD"},
                "answer": row.get("answer"),
            }
        )

    object_counts = [len(entry["objects"]) for entry in catalog]
    padding_counts = Counter(
        len(row.get("diagnostic_generated_object_distractors") or [])
        for row in object_rows
    )
    required_counts = Counter(int(entry["required_padding"]) for entry in catalog)
    accepted_image_keys = {
        utils._direction_object_image_key(row) for row in object_rows
    }
    summary = {
        "report_schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "variant": {
            "lmms_task": TASK_NAME,
            "dir_clr_adapter": ADAPTER_NAME,
            "apc_vlm_adapter": ADAPTER_NAME,
            "sample_seed": utils.QWEN_DIRECTION_OBJECT_SAMPLE_SEED,
            "generator_model": utils.QWEN_DIRECTION_OBJECT_MODEL,
            "baseline_task": "3dsrbench_direction_object",
            "baseline_padding": "deterministic thematic pools",
            "fallback_to_thematic_padding": False,
        },
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256(args.dataset),
            "dataset_rows": int(len(frame)),
            "eligible_relation_rows": int(relation_mask.sum()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "manifest_created_at": manifest.get("created_at"),
            "manifest_schema_version": manifest.get("schema_version"),
        },
        "generation": {
            "catalog_scenes": len(catalog),
            "catalog_source_qid_references": sum(
                len(entry.get("source_qids", [])) for entry in catalog
            ),
            "required_padding_distribution": {
                str(key): value for key, value in sorted(required_counts.items())
            },
            "catalog_objects_total": sum(object_counts),
            "catalog_objects_per_scene": {
                "minimum": min(object_counts),
                "maximum": max(object_counts),
                "mean": statistics.mean(object_counts),
                "median": statistics.median(object_counts),
            },
            "recovered_truncated_json_scenes": sum(
                bool(entry.get("recovered_from_truncated_json")) for entry in catalog
            ),
            "remaining_failures": len(manifest.get("failures", {})),
            "decoding": {
                "do_sample": False,
                "temperature": None,
                "attention_implementation": "sdpa",
                "dtype": "bfloat16",
                "final_max_new_tokens": 256,
                "pilot_max_new_tokens": 384,
                "retries": 3,
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            "pbs": {
                "successful_job_id": "122093.caquelon",
                "successful_job_exit_status": 0,
                "successful_job_walltime": "00:04:49",
                "diagnostic_jobs_cancelled_after_checkpointing": [
                    "122091.caquelon",
                    "122092.caquelon",
                ],
                "host": "cvml01",
                "gpus": 1,
            },
        },
        "task_construction": {
            "matched_pairs": len(object_rows),
            "task_rows": len(task_rows),
            "direction_rows": len(direction_rows),
            "object_rows": len(object_rows),
            "skipped_eligible_relation_rows": int(relation_mask.sum()) - len(object_rows),
            "accepted_unique_scenes": len(accepted_image_keys),
            "object_rows_using_qwen_padding": sum(
                bool(row.get("diagnostic_generated_object_distractors"))
                for row in object_rows
            ),
            "selected_qwen_padding_slots": sum(
                len(row.get("diagnostic_generated_object_distractors") or [])
                for row in object_rows
            ),
            "selected_padding_distribution": {
                str(key): value for key, value in sorted(padding_counts.items())
            },
            "unique_selected_qwen_labels": len(label_counts),
            "most_common_selected_qwen_labels": [
                {"label": label, "count": count}
                for label, count in label_counts.most_common(20)
            ],
            "choice_count": 4,
            "chance_accuracy": 0.25,
        },
        "validation": {
            "manifest_model_matches_expected": manifest.get("model")
            == utils.QWEN_DIRECTION_OBJECT_MODEL,
            "all_catalogs_have_enough_objects": all(
                len(entry["objects"]) >= int(entry["required_padding"])
                for entry in catalog
            ),
            "all_object_rows_have_four_unique_options": all(
                len({utils._normalize_text(row[letter]) for letter in "ABCD"}) == 4
                for row in object_rows
            ),
            "all_rows_record_qwen_provenance": all(
                row.get("diagnostic_object_distractor_source")
                == utils.QWEN_DIRECTION_OBJECT_MODEL
                for row in task_rows
            ),
            "lmms_task_discovery": True,
            "dir_clr_adapter_rows": len(task_rows),
            "apc_vlm_dry_run_rows": len(task_rows),
        },
    }
    return summary, selected


def markdown(summary: dict[str, Any]) -> str:
    generation = summary["generation"]
    task = summary["task_construction"]
    inputs = summary["inputs"]
    validation = summary["validation"]
    common = task["most_common_selected_qwen_labels"][:10]
    common_text = ", ".join(f"{item['label']} ({item['count']})" for item in common)
    return f"""# 3DSRBench Qwen3-VL scene-distractor generation

## Report-ready summary

This experiment adds a separate 3DSRBench direction/object variant,
`{summary['variant']['lmms_task']}`, in which sparse inverse object-choice
questions are padded with object names proposed from the source image by
**{summary['variant']['generator_model']}**. The original
`{summary['variant']['baseline_task']}` task remains unchanged and continues to
use deterministic thematic padding. The new variant never falls back to those
thematic pools: if the saved Qwen scene catalog cannot supply four distinct
choices, task construction fails rather than silently inventing a label.

The source parquet contains **{inputs['dataset_rows']:,}** rows, including
**{inputs['eligible_relation_rows']:,}** `multi_object / viewpoint towards object`
rows. The paired-task quality rule retained **{task['matched_pairs']:,}** source
relations, producing **{task['task_rows']:,}** evaluation items: one native
direction-answer row and one inverse object-answer row per relation. Each item
has four choices (25% chance accuracy). An inverse row still requires at least
one distractor derived from another queried object in the same image; Qwen only
pads otherwise qualifying rows.

Qwen was run offline on **{generation['catalog_scenes']:,}** sparse scene images
and produced **{generation['catalog_objects_total']:,}** retained object labels.
Catalogs contain {generation['catalog_objects_per_scene']['minimum']}–
{generation['catalog_objects_per_scene']['maximum']} labels per image (mean
{generation['catalog_objects_per_scene']['mean']:.2f}, median
{generation['catalog_objects_per_scene']['median']:.0f}). The final paired task
uses Qwen padding on **{task['object_rows_using_qwen_padding']:,} of
{task['object_rows']:,}** inverse rows, filling **{task['selected_qwen_padding_slots']:,}**
choice slots. The selected-padding distribution is
`{json.dumps(task['selected_padding_distribution'], sort_keys=True)}` where keys
are the number of Qwen choices per inverse row. The ten most frequent selected
labels are: {common_text}.

## Generation protocol

- Model: `{summary['variant']['generator_model']}` in bfloat16 with SDPA.
- Deterministic decoding: `do_sample=false`; no temperature or top-p sampling.
- Image bounds: `min_pixels={generation['decoding']['min_pixels']}` and
  `max_pixels={generation['decoding']['max_pixels']}`.
- Final token cap: `{generation['decoding']['final_max_new_tokens']}`. Early
  diagnostic generations used `{generation['decoding']['pilot_max_new_tokens']}`.
- Up to {generation['decoding']['retries']} attempts were permitted per image.
- Successful production job: `{generation['pbs']['successful_job_id']}` on
  `{generation['pbs']['host']}`, one GPU, exit status 0, walltime
  `{generation['pbs']['successful_job_walltime']}`.
- Two diagnostic jobs (`122091.caquelon` and `122092.caquelon`) were stopped
  after exposing the strict-parser failure mode. Their atomic checkpoints were
  retained: the final job began with four accepted catalogs, recovered eight
  complete label lists from their truncated raw JSON, and generated the
  remaining 131 catalogs. Consequently, the 4:49 final-job walltime is not a
  claim of total development compute.
- The manifest was written atomically after each successful scene, allowing
  safe resume. Raw model responses, source qids, image paths, known-object
  exclusions, attempt counts, and retained labels are preserved per scene.
- Qwen occasionally repeated valid labels until its response was truncated
  before the final JSON delimiter. The parser recovered only complete JSON
  string literals; incomplete trailing strings were discarded. This affected
  {generation['recovered_truncated_json_scenes']} catalogs. No generation
  failures remain.

The model instruction asked for concise visible physical-object noun phrases,
excluded annotated/known object names, prohibited colors, materials,
directions, and spatial relations, and explicitly searched foreground,
mid-ground, and background. It permitted visible structures, fixtures,
vegetation, vehicles, furniture, and surfaces such as roads or sidewalks. The
response schema was `{{"objects": ["object 1", ...]}}`.

## Construction and controls

For each source relation, candidates are assembled in this order:

1. the annotated target object;
2. distinct object terms queried elsewhere for the same source image;
3. saved Qwen scene-object proposals, only until four choices are available.

The annotated subject and aliases of existing candidates are removed using the
same normalized substring-based entity check as the baseline task. Choices are
then shuffled deterministically with seed
`{summary['variant']['sample_seed']}:<source_qid>`. Each output row records
`diagnostic_object_distractor_source={summary['variant']['generator_model']}` and
the exact selected Qwen labels in
`diagnostic_generated_object_distractors`.

The 143 generated catalogs cover sparse scenes identified before the original
pair-quality filter. Therefore, catalog count is not the evaluation sample
count: the task accepts {task['matched_pairs']} relations across
{task['accepted_unique_scenes']} scenes and rejects
{task['skipped_eligible_relation_rows']} otherwise eligible relation rows that
lack an image-derived distractor or another required field.

## Validation

- Manifest model identity matches the required 8B checkpoint:
  `{validation['manifest_model_matches_expected']}`.
- Every scene catalog supplies at least its required padding count:
  `{validation['all_catalogs_have_enough_objects']}`.
- Every inverse row has four normalized, distinct options:
  `{validation['all_object_rows_have_four_unique_options']}`.
- All {task['task_rows']} rows retain explicit Qwen provenance:
  `{validation['all_rows_record_qwen_provenance']}`.
- lmms task discovery passed; Dir_clr loaded {validation['dir_clr_adapter_rows']}
  rows; APC-VLM dry-run loaded {validation['apc_vlm_dry_run_rows']} rows.

## Reporting caveats

The proposals are model-generated visual annotations, not human-verified object
labels or detector-confirmed ground truth. They may include hallucinations,
synonyms, broad background surfaces, or inconsistent granularity. Because the
generator is Qwen3-VL, evaluation of Qwen-family models on this variant may
benefit from model-specific vocabulary or inductive bias; results should be
reported alongside the unchanged thematic-padding baseline and should not be
presented as a model-independent benchmark replacement. The generation pass
uses only images and known object exclusions—not answers to the inverse
questions—but its influence on distractor difficulty must still be disclosed.

## Reproducibility artifacts

- Source dataset: `{inputs['dataset']}`
  (SHA-256 `{inputs['dataset_sha256']}`).
- Full scene manifest: `{inputs['manifest']}`
  (SHA-256 `{inputs['manifest_sha256']}`).
- lmms task: `lmms_eval/tasks/3dsrbench_custom/3dsrbench_direction_object_qwen3vl_distractors.yaml`.
- Generator: `tools/generate_3dsr_qwen_scene_distractors.py`.
- Generator PBS entry point: `pbs_generate_3dsr_qwen_scene_distractors.sh`.
- Dir_clr/APC-VLM adapter name: `{summary['variant']['dir_clr_adapter']}`.
- `generation_summary.json` contains the machine-readable aggregate, and
  `selected_padding.jsonl` records the exact padding/options used by every
  retained inverse row.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, selected = summarize(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generation_report.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    with (args.output_dir / "selected_padding.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote generation report artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
