#!/usr/bin/env python3
"""Report per-source-family metrics from a Kubric direction/object submission.

The extended task produces four records for each source question:
native, inverse, direction_natural_language, and object_direction_exhaustive.
This tool reads the submission records written by the task's submission metric;
it does not rerun model inference. Answer-vs-ground-truth plots reconstruct
candidate directions from the source parquet's object and camera positions.

Example:
    python tools/analyze_kubric_direction_object_submission.py \
        outputs/kubric_movi_a_direction_object_extended_0/submissions/\
kubric_movi_a_direction_object_extended_qwen3_vl_experiments.json \
        --plot outputs/kubric_movi_a_direction_object_extended_0/kubric_paired_outcomes.png

By default, reports, the paired-outcomes plot, two answer-vs-ground-truth
plots, and their JSON counts are written to the run directory containing the
submission folder. Use ``--output-dir`` to choose another destination.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

SOURCE_FAMILIES = (
    "object_centric_relative_position",
    "object_centric_relative_position_multi",
)
VARIANTS = (
    "native",
    "inverse",
    "direction_natural_language",
    "object_direction_exhaustive",
)
RELATIONS = ("left", "right", "front", "behind")
OUTCOME_KEYS = ("improved", "worsened", "unchanged_correct", "unchanged_incorrect")
DATASET_CANDIDATES = (
    Path("/home/ramanathan/data/movi_a_3dsr/movi_a_validation.parquet"),
    Path("/home/ramanathan/data/movi_a_3dsr_better_sample/" "movi_a_validation.parquet"),
)
BASE_VARIANTS = frozenset({"native", "inverse"})


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    # The current submission writer emits a bare list. Accept a few common
    # wrappers as well, so the tool remains useful with copied submissions.
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("results", payload.get("submissions"))
    else:
        records = None

    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Expected a JSON list of submission records")
    return records


def _score(row: dict[str, Any]) -> float:
    value = row.get("score")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {row.get('qid', '<unknown>')} has no numeric score") from exc


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    return {
        "count": len(scores),
        "correct": sum(score == 1.0 for score in scores),
        "accuracy": sum(scores) / len(scores) if scores else None,
    }


def _paired_outcomes(sources: dict[str, dict[str, dict[str, Any]]], left: str, right: str) -> dict[str, int]:
    """Count correctness transitions when swapping ``right`` for ``left``."""
    outcomes = {
        "improved": 0,
        "worsened": 0,
        "unchanged_correct": 0,
        "unchanged_incorrect": 0,
    }
    for variants in sources.values():
        if left not in variants or right not in variants:
            continue
        left_correct = _score(variants[left]) == 1.0
        right_correct = _score(variants[right]) == 1.0
        if left_correct and not right_correct:
            outcomes["improved"] += 1
        elif right_correct and not left_correct:
            outcomes["worsened"] += 1
        elif left_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    outcomes["paired_count"] = sum(outcomes.values())
    return outcomes


def _format_paired_outcomes(sources: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, int]:
    """Compare the base object-answer and direction-answer prompts by format.

    ``native`` and ``inverse`` swap answer formats between the single- and
    multi-object source families, so choosing rows by ``answer_format`` is
    less error-prone than assuming one fixed variant represents each format.
    """
    outcomes = {key: 0 for key in OUTCOME_KEYS}
    for rows in sources.values():
        direction = next((row for row in rows.values() if row.get("answer_format") == "direction"), None)
        object_answer = next((row for row in rows.values() if row.get("answer_format") == "object"), None)
        if direction is None or object_answer is None:
            continue
        direction_correct = _score(direction) == 1.0
        object_correct = _score(object_answer) == 1.0
        if object_correct and not direction_correct:
            outcomes["improved"] += 1
        elif direction_correct and not object_correct:
            outcomes["worsened"] += 1
        elif object_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    outcomes["paired_count"] = sum(outcomes.values())
    return outcomes


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    unknown_families: set[str] = set()
    unknown_variants: set[str] = set()

    for row in records:
        family = str(row.get("source_task_family", ""))
        variant = str(row.get("variant", ""))
        if family not in SOURCE_FAMILIES:
            unknown_families.add(family)
        if variant not in VARIANTS:
            unknown_variants.add(variant)
        grouped[(family, variant)].append(row)
        # Include family in the key so copied/generated datasets with reused
        # source ids cannot accidentally be paired across task families.
        source_key = (family, str(row.get("source_qid", "")))
        by_source[source_key][variant] = row

    by_family = {family: {variant: _summary(grouped[(family, variant)]) for variant in VARIANTS} for family in SOURCE_FAMILIES}
    by_variant = {
        variant: _summary(grouped_for_variant) for variant in VARIANTS for grouped_for_variant in [[row for row in records if row.get("variant") == variant]]
    }

    paired = {}
    paired_outcomes = {}
    for family in SOURCE_FAMILIES:
        family_sources = {source: variants for source, variants in by_source.items() if source[0] == family}
        paired[family] = {}
        paired_outcomes[family] = {}
        # The simple direction baseline is native for the single-object
        # source family, but inverse for the multi-object family: the latter's
        # original answer is an object, so its inverse is the direction row.
        direction_baseline = "native" if family == "object_centric_relative_position" else "inverse"
        for left, right in (
            ("inverse", "native"),
            ("direction_natural_language", direction_baseline),
            ("object_direction_exhaustive", direction_baseline),
        ):
            differences = [_score(variants[left]) - _score(variants[right]) for variants in family_sources.values() if left in variants and right in variants]
            paired[family][f"{left}_minus_{right}"] = {
                "paired_count": len(differences),
                "mean_gain": sum(differences) / len(differences) if differences else None,
            }
            paired_outcomes[family][f"{left}_vs_{right}"] = _paired_outcomes(family_sources, left, right)

    by_relation: dict[str, dict[tuple[str, str], dict[str, dict[str, Any]]]] = defaultdict(dict)
    for source, variants in by_source.items():
        relation = next((str(row.get("relation", "")) for row in variants.values()), "")
        if relation:
            by_relation[relation][source] = variants
    format_paired_outcomes = {
        "overall": _format_paired_outcomes(by_source),
        **{relation: _format_paired_outcomes(by_relation[relation]) for relation in RELATIONS if relation in by_relation},
    }

    output: dict[str, Any] = {
        "records": len(records),
        "by_source_task_family": by_family,
        "by_variant": by_variant,
        "paired_gains": paired,
        "paired_outcomes": paired_outcomes,
        "format_paired_outcomes_by_relation": format_paired_outcomes,
    }
    if unknown_families:
        output["unknown_source_task_families"] = sorted(unknown_families)
    if unknown_variants:
        output["unknown_variants"] = sorted(unknown_variants)
    return output


def _print_report(report: dict[str, Any]) -> None:
    print(f"Records: {report['records']}")
    print("\nAccuracy by base question family and row variant")
    print(f"{'source_task_family':48} {'variant':32} {'correct/total':>13} {'accuracy':>10}")
    print("-" * 108)
    for family in SOURCE_FAMILIES:
        for variant in VARIANTS:
            row = report["by_source_task_family"][family][variant]
            accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
            print(f"{family:48} {variant:32} {row['correct']:>6}/{row['count']:<6} {accuracy:>10}")

    print("\nOverall by variant")
    for variant in VARIANTS:
        row = report["by_variant"][variant]
        accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        print(f"  {variant:32} {row['correct']}/{row['count']} ({accuracy})")

    print("\nPaired mean gains")
    for family, gains in report["paired_gains"].items():
        print(f"  {family}")
        for name, value in gains.items():
            gain = "N/A" if value["mean_gain"] is None else f"{value['mean_gain']:+.4f}"
            print(f"    {name}: {gain} ({value['paired_count']} pairs)")

    print("\nPaired correctness transitions (left variant versus right variant)")
    for family, comparisons in report["paired_outcomes"].items():
        print(f"  {family}")
        for name, counts in comparisons.items():
            print(
                f"    {name}: improved={counts['improved']}, worsened={counts['worsened']}, "
                f"unchanged-correct={counts['unchanged_correct']}, "
                f"unchanged-incorrect={counts['unchanged_incorrect']} "
                f"({counts['paired_count']} pairs)"
            )


def _format_report(report: dict[str, Any]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _print_report(report)
    return buffer.getvalue()

    print("\nDirection-to-object paired outcomes by relation")
    for relation, counts in report["format_paired_outcomes_by_relation"].items():
        print(
            f"  {relation}: improved={counts['improved']}, worsened={counts['worsened']}, "
            f"unchanged-correct={counts['unchanged_correct']}, "
            f"unchanged-incorrect={counts['unchanged_incorrect']} "
            f"({counts['paired_count']} pairs)"
        )


def _save_paired_outcomes_plot(report: dict[str, Any], path: Path) -> None:
    """Save a COMFORT-style stacked plot for object versus direction answers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it to use --plot.") from exc

    paired_by_relation = report["format_paired_outcomes_by_relation"]
    labels = ["overall", *(relation for relation in RELATIONS if relation in paired_by_relation)]
    labels = [label for label in labels if paired_by_relation[label]["paired_count"]]

    if not labels:
        raise ValueError("No complete direction/object pairs were found for plotting.")

    colors = {
        "improved": "#2a9d8f",
        "worsened": "#e76f51",
        "unchanged_correct": "#457b9d",
        "unchanged_incorrect": "#9aa0a6",
    }
    display = {
        "improved": "Improved (direction wrong → object correct)",
        "worsened": "Worsened (direction correct → object wrong)",
        "unchanged_correct": "Unchanged: correct",
        "unchanged_incorrect": "Unchanged: incorrect",
    }
    figure, axis = plt.subplots(figsize=(11, 6))
    bottom = [0] * len(labels)
    for key in OUTCOME_KEYS:
        values = [paired_by_relation[label][key] for label in labels]
        axis.bar(labels, values, bottom=bottom, label=display[key], color=colors[key])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Number of matched source relations")
    axis.set_title("Kubric MOVi-A direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _dominant_anchor_direction(
    anchor_position: list[float],
    object_position: list[float],
    camera_position: list[float],
) -> str | None:
    """Classify an object in the anchor-centred, camera-facing horizontal frame."""
    front_x = float(camera_position[0]) - float(anchor_position[0])
    front_y = float(camera_position[1]) - float(anchor_position[1])
    norm = math.hypot(front_x, front_y)
    if norm <= 1e-8:
        return None
    front_x, front_y = front_x / norm, front_y / norm
    # This matches the dataset convention: an anchor facing the camera has its
    # semantic right on the camera/image-left side.
    right_x, right_y = -front_y, front_x
    delta_x = float(object_position[0]) - float(anchor_position[0])
    delta_y = float(object_position[1]) - float(anchor_position[1])
    front_component = delta_x * front_x + delta_y * front_y
    right_component = delta_x * right_x + delta_y * right_y
    if abs(right_component) >= abs(front_component):
        return "right" if right_component < 0 else "left"
    return "front" if front_component >= 0 else "behind"


def _load_object_directions(dataset_path: Path) -> dict[str, dict[str, str]]:
    """Map source qids and visible object names to their semantic directions."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Answer-vs-GT plots require pyarrow") from exc

    table = pq.read_table(
        dataset_path,
        columns=["qid", "task_family", "task_metadata", "visible_objects"],
    )
    lookup: dict[str, dict[str, str]] = {}
    for doc in table.to_pylist():
        if doc.get("task_family") not in SOURCE_FAMILIES:
            continue
        metadata = doc.get("task_metadata") or {}
        anchor_name = str(metadata.get("anchor_object") or "")
        camera_position = metadata.get("camera_position")
        objects = {str(obj.get("name")): obj for obj in (doc.get("visible_objects") or []) if obj.get("name") and obj.get("position_3d")}
        anchor = objects.get(anchor_name)
        if anchor is None or not camera_position:
            continue
        directions = {anchor_name: "reference object"}
        for name, obj in objects.items():
            if name == anchor_name:
                continue
            direction = _dominant_anchor_direction(anchor["position_3d"], obj["position_3d"], camera_position)
            if direction is not None:
                directions[name] = direction
        lookup[str(doc.get("qid"))] = directions
    return lookup


def _resolve_dataset_path(records: list[dict[str, Any]], requested: Path | None) -> Path:
    """Select the parquet containing the most submission source qids."""
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Answer-vs-GT plots require pyarrow") from exc

    source_qids = {str(row.get("source_qid") or "") for row in records}
    scored = []
    for candidate in DATASET_CANDIDATES:
        if not candidate.is_file():
            continue
        dataset_qids = set(pq.read_table(candidate, columns=["qid"])["qid"].to_pylist())
        scored.append((len(source_qids & dataset_qids), candidate))
    if not scored or max(score for score, _ in scored) == 0:
        raise ValueError("Could not match submission source qids to a known MOVi-A parquet; " "provide the correct file with --dataset")
    return max(scored, key=lambda item: item[0])[1]


def _prompt_options(row: dict[str, Any]) -> dict[str, str]:
    """Recover lettered choices retained in the saved question prompt."""
    return {
        match.group(1).upper(): match.group(2).strip()
        for match in re.finditer(
            r"^\s*([A-P])\.\s*(.*?)\s*$",
            str(row.get("question_prompt") or ""),
            flags=re.MULTILINE,
        )
    }


def _selected_option(row: dict[str, Any]) -> str | None:
    parsed = str(row.get("parsed_prediction") or "").strip().upper()
    match = re.fullmatch(r"([A-P])", parsed)
    if match is None:
        match = re.match(r"\s*([A-P])(?:[.)]|\s)", str(row.get("prediction") or "").upper())
    if match is None:
        return None
    return _prompt_options(row).get(match.group(1))


def _answer_gt_counts(
    records: list[dict[str, Any]],
    object_directions: dict[str, dict[str, str]],
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], dict[str, int]]:
    """Count selected directions for the matched base direction/object rows."""
    direction_counts = {relation: Counter() for relation in RELATIONS}
    object_counts = {relation: Counter() for relation in RELATIONS}
    skipped = Counter()
    for row in records:
        if row.get("variant") not in BASE_VARIANTS:
            continue
        relation = str(row.get("relation") or "").lower()
        if relation not in RELATIONS:
            skipped["invalid_gold_relation"] += 1
            continue
        answer_format = row.get("answer_format")
        selected = _selected_option(row)
        if selected is None:
            if answer_format == "direction":
                direction_counts[relation]["parse failure"] += 1
            elif answer_format == "object":
                object_counts[relation]["parse failure"] += 1
            continue
        if answer_format == "direction":
            selected_direction = selected.strip().lower()
            direction_counts[relation][selected_direction if selected_direction in RELATIONS else "unmapped answer"] += 1
        elif answer_format == "object":
            source_lookup = object_directions.get(str(row.get("source_qid")))
            if source_lookup is None:
                skipped["source_missing_from_dataset"] += 1
                object_counts[relation]["unmapped object"] += 1
            else:
                object_counts[relation][source_lookup.get(selected, "unmapped object")] += 1
    return direction_counts, object_counts, dict(skipped)


def _save_answer_gt_plot(
    counts: dict[str, Counter[str]],
    title: str,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib") from exc

    categories = [
        *RELATIONS,
        "reference object",
        "unmapped object",
        "unmapped answer",
        "parse failure",
    ]
    colors = {
        "left": "#4C78A8",
        "right": "#F58518",
        "front": "#54A24B",
        "behind": "#E45756",
        "reference object": "#B279A2",
        "unmapped object": "#BAB0AC",
        "unmapped answer": "#9D755D",
        "parse failure": "#5F5F5F",
    }
    figure, axis = plt.subplots(figsize=(9, 6))
    bottom = [0] * len(RELATIONS)
    for category in categories:
        values = [counts[relation][category] for relation in RELATIONS]
        if not any(values):
            continue
        axis.bar(
            RELATIONS,
            values,
            bottom=bottom,
            label=category,
            color=colors[category],
        )
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_xlabel("Ground-truth direction")
    axis.set_ylabel("Number of predictions")
    axis.set_title(title)
    axis.legend(
        title="Model-selected direction",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _serialise_counts(counts: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {relation: dict(counts[relation]) for relation in RELATIONS}


def _save_answer_gt_artifacts(
    records: list[dict[str, Any]],
    dataset_path: Path,
    output_dir: Path,
) -> list[Path]:
    direction_counts, object_counts, skipped = _answer_gt_counts(records, _load_object_directions(dataset_path))
    direction_path = output_dir / "direction_gt_vs_selected_direction.png"
    object_path = output_dir / "object_gt_vs_selected_object_direction.png"
    counts_path = output_dir / "answer_gt_direction_counts.json"
    _save_answer_gt_plot(
        direction_counts,
        "Direction tasks: ground truth vs model-selected direction",
        direction_path,
    )
    _save_answer_gt_plot(
        object_counts,
        "Object tasks: ground truth vs direction of selected object",
        object_path,
    )
    counts_path.write_text(
        json.dumps(
            {
                "direction_tasks": _serialise_counts(direction_counts),
                "object_tasks": _serialise_counts(object_counts),
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [direction_path, object_path, counts_path]


def _default_output_dir(submission: Path) -> Path:
    return submission.parent.parent if submission.parent.name == "submissions" else submission.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the JSON submission list")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Artifact directory; defaults to the run directory containing submissions/",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="PATH",
        help="Override the paired-outcomes plot path (kept for compatibility)",
    )
    parser.add_argument(
        "--answer-gt-dir",
        type=Path,
        metavar="DIR",
        help="Override the directory for direction/object answer-vs-GT artifacts",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help=("Source MOVi-A parquet used for object directions; defaults to " "automatic selection by source-qid coverage"),
    )
    args = parser.parse_args()

    records = _load_records(args.submission)
    report = analyze(records)
    dataset_path = _resolve_dataset_path(records, args.dataset)
    output_dir = (args.output_dir or _default_output_dir(args.submission)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kubric_direction_object_analysis.json"
    text_path = output_dir / "kubric_direction_object_analysis.txt"
    markdown_path = output_dir / "summary.md"
    paired_plot_path = (args.plot or output_dir / "kubric_paired_outcomes.png").resolve()
    answer_gt_dir = (args.answer_gt_dir or output_dir).resolve()

    text_report = _format_report(report)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(text_report, encoding="utf-8")
    markdown_path.write_text(
        "# Kubric direction/object analysis\n\n"
        f"Submission: `{args.submission.resolve()}`\n\n"
        f"Object-direction source: `{dataset_path}`\n\n"
        "```text\n" + text_report.rstrip() + "\n```\n",
        encoding="utf-8",
    )
    _save_paired_outcomes_plot(report, paired_plot_path)
    answer_gt_paths = _save_answer_gt_artifacts(
        records,
        dataset_path=dataset_path,
        output_dir=answer_gt_dir,
    )

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(text_report, end="")
    for path in [json_path, text_path, markdown_path, paired_plot_path, *answer_gt_paths]:
        print(f"Saved: {path}", file=sys.stderr)
    print(f"Object directions loaded from: {dataset_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
