#!/usr/bin/env python3
"""Summarize matched ScanNet answer-with-object versus answer-with-direction tasks.

Accepts either submission files or directories.  Both camera-frame and
object-facing-camera submissions can be analyzed together.  The output
contains machine-readable tables, a Markdown report, paired-case rows, and
plots for accuracy, paired outcomes, and answer-versus-ground-truth direction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_INPUT = Path("/home/ramanathan/VLM/lmms-eval/outputs/scannet_basis_object_direction_8")
DEFAULT_MANIFEST = Path("/mnt/rdata4_3/Spatial_Benchmarks/scannetv2/scannet_camera_basis_manifest.jsonl")
OUTPUT_DIR_NAME = "scannet_object_direction_analysis"
DIRECTIONS = ("left", "right", "front", "back")
FORMATS = ("object", "direction")
FRAME_ORDER = ("camera", "object_facing_camera")
FRAME_LABELS = {
    "camera": "Camera frame",
    "object_facing_camera": "Object facing camera",
}
OUTCOMES = ("both_correct", "object_only", "direction_only", "both_wrong", "parse_failure")
OUTCOME_LABELS = {
    "both_correct": "Both correct",
    "object_only": "Object only correct",
    "direction_only": "Direction only correct",
    "both_wrong": "Both wrong",
    "parse_failure": "Either parse failed",
}
OUTCOME_COLORS = {
    "both_correct": "#2A9D8F",
    "object_only": "#E9C46A",
    "direction_only": "#F4A261",
    "both_wrong": "#E76F51",
    "parse_failure": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="Submission JSON files or directories searched recursively.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="ScanNet frame manifest used to map selected objects back to directions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <first input>/scannet_object_direction_analysis.",
    )
    return parser.parse_args()


def discover(inputs: Iterable[Path]) -> list[Path]:
    paths = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob("scannet_*_basis_object_direction_*.json"))
        else:
            raise FileNotFoundError(path)
    unique = sorted({path for path in paths if path.is_file()})
    if not unique:
        raise FileNotFoundError("No ScanNet object-direction submission JSON files found")
    return unique


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows, metadata = payload.get("records"), payload
    elif isinstance(payload, list):
        rows, metadata = payload, {}
    else:
        rows, metadata = None, {}
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a non-empty record list in {path}")
    if any(row.get("prediction_format") != "object_direction" for row in rows):
        raise ValueError(f"Not an object-direction submission: {path}")
    return metadata, rows


def model_name(metadata: dict[str, Any], path: Path) -> str:
    task = str(metadata.get("task", ""))
    marker = task + "_"
    return path.stem[len(marker):] if task and path.stem.startswith(marker) else path.stem


def frame_name(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("coordinate_frame")) for row in rows}
    if metadata.get("coordinate_frame"):
        values.add(str(metadata["coordinate_frame"]))
    values &= set(FRAME_ORDER)
    if len(values) != 1:
        raise ValueError(f"Expected one coordinate frame, found {sorted(values)}")
    return next(iter(values))


def score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("answer_accuracy", 0.0))
    except (TypeError, ValueError):
        return 0.0


def parsed(row: dict[str, Any]) -> bool:
    try:
        return bool(float(row.get("parse_success", 0.0)))
    except (TypeError, ValueError):
        return False


def group_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, dict[str, Any]]], list[str]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    issues = []
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        answer_format = str(row.get("answer_format", ""))
        if not pair_id or answer_format not in FORMATS:
            issues.append(str(row.get("qid", "missing qid")) + ": missing pair id/format")
            continue
        if answer_format in grouped[pair_id]:
            issues.append(str(row.get("qid", pair_id)) + ": duplicate format")
        grouped[pair_id][answer_format] = row
    complete = []
    for pair_id, pair in grouped.items():
        if set(pair) == set(FORMATS):
            complete.append(pair)
        else:
            issues.append(pair_id + ": incomplete pair")
    return complete, issues


def pair_outcome(pair: dict[str, dict[str, Any]]) -> str:
    if not all(parsed(pair[answer_format]) for answer_format in FORMATS):
        return "parse_failure"
    object_correct = score(pair["object"]) == 1.0
    direction_correct = score(pair["direction"]) == 1.0
    if object_correct and direction_correct:
        return "both_correct"
    if object_correct:
        return "object_only"
    if direction_correct:
        return "direction_only"
    return "both_wrong"


def mean(rows: list[dict[str, Any]], function) -> float:
    return sum(function(row) for row in rows) / len(rows) if rows else 0.0


def reference_key(row: dict[str, Any]) -> tuple[str, str, str]:
    # pair_id begins scene::frame::reference-id::target-id.
    parts = str(row.get("pair_id", "")).split("::")
    return tuple(parts[:3]) if len(parts) >= 3 else (str(row.get("scene_id")), "", str(row.get("reference_object")))


def object_direction_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        lookup[reference_key(row)][str(row.get("target_object"))] = str(row.get("gt_direction"))
    return lookup


def _direction_from_positions(reference: list[float], target: list[float], frame: str) -> str:
    dx = float(target[0]) - float(reference[0])
    dy = float(target[1]) - float(reference[1])
    if frame == "camera":
        right, front = dx, dy
    else:
        norm = (float(reference[0]) ** 2 + float(reference[1]) ** 2) ** 0.5
        if norm <= 1e-8:
            raise ValueError("reference lies at camera origin")
        front_x, front_y = -float(reference[0]) / norm, -float(reference[1]) / norm
        right_x, right_y = front_y, -front_x
        front = dx * front_x + dy * front_y
        right = dx * right_x + dy * right_y
    if abs(right) >= abs(front):
        return "right" if right > 0.0 else "left"
    return "front" if front > 0.0 else "back"


def manifest_direction_lookup(
    manifest: Path,
    experiments: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Recover directions for every candidate, including downsampled targets."""
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    wanted: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for experiment in experiments:
        frame = experiment["summary"]["frame"]
        for row in experiment["rows"]:
            parts = str(row.get("pair_id", "")).split("::")
            if len(parts) < 4 or not parts[1].startswith("frame"):
                continue
            wanted[(parts[0], int(parts[1][5:]))].add((parts[2], frame))

    lookup: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            view = json.loads(line)
            view_key = (str(view.get("scene_id")), int(view.get("frame_id", -1)))
            references = wanted.get(view_key)
            if not references:
                continue
            objects = {str(obj.get("object_id")): obj for obj in (view.get("objects") or [])}
            for reference_id, frame in references:
                reference = objects.get(reference_id)
                if reference is None:
                    continue
                reference_position = reference.get("position_camera_xyz_m")
                if not isinstance(reference_position, list) or len(reference_position) != 3:
                    continue
                key = (view_key[0], f"frame{view_key[1]:06d}", reference_id, frame)
                for target_id, target in objects.items():
                    if target_id == reference_id:
                        continue
                    position = target.get("position_camera_xyz_m")
                    label = str(target.get("label", ""))
                    if not label or not isinstance(position, list) or len(position) != 3:
                        continue
                    lookup[key][label] = _direction_from_positions(reference_position, position, frame)
    return lookup


def summarize_submission(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = frame_name(metadata, rows)
    model = model_name(metadata, path)
    pairs, issues = group_pairs(rows)
    format_rows = {answer_format: [row for row in rows if row.get("answer_format") == answer_format] for answer_format in FORMATS}
    counts = Counter(pair_outcome(pair) for pair in pairs)
    summary = {
        "frame": frame,
        "frame_label": FRAME_LABELS[frame],
        "model": model,
        "submission": str(path),
        "records": len(rows),
        "pairs": len(pairs),
        "scenes": len({str(row.get("scene_id")) for row in rows}),
        "object_accuracy": mean(format_rows["object"], score),
        "direction_accuracy": mean(format_rows["direction"], score),
        "object_parse_success": mean(format_rows["object"], lambda row: float(parsed(row))),
        "direction_parse_success": mean(format_rows["direction"], lambda row: float(parsed(row))),
        "issues": len(issues),
    }
    summary["object_minus_direction"] = summary["object_accuracy"] - summary["direction_accuracy"]
    for outcome in OUTCOMES:
        summary[outcome] = counts[outcome] / len(pairs) if pairs else 0.0
    return {"summary": summary, "rows": rows, "pairs": pairs, "issues": issues}


def build_pair_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for experiment in experiments:
        summary = experiment["summary"]
        for pair in experiment["pairs"]:
            direction, obj = pair["direction"], pair["object"]
            result.append({
                "frame": summary["frame"],
                "model": summary["model"],
                "pair_id": direction.get("pair_id"),
                "scene_id": direction.get("scene_id"),
                "reference_object": direction.get("reference_object"),
                "target_object": direction.get("target_object"),
                "gt_direction": direction.get("gt_direction"),
                "direction_prediction": direction.get("parsed_answer"),
                "object_prediction": obj.get("parsed_answer"),
                "direction_correct": score(direction),
                "object_correct": score(obj),
                "direction_parse_success": float(parsed(direction)),
                "object_parse_success": float(parsed(obj)),
                "outcome": pair_outcome(pair),
                "img_path": direction.get("img_path"),
            })
    return result


def build_direction_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for experiment in experiments:
        summary = experiment["summary"]
        for direction in DIRECTIONS:
            for answer_format in FORMATS:
                rows = [row for row in experiment["rows"] if row.get("answer_format") == answer_format and row.get("gt_direction") == direction]
                result.append({
                    "frame": summary["frame"],
                    "model": summary["model"],
                    "gt_direction": direction,
                    "answer_format": answer_format,
                    "count": len(rows),
                    "accuracy": mean(rows, score),
                    "parse_success": mean(rows, lambda row: float(parsed(row))),
                })
    return result


def build_confusion_rows(
    experiments: list[dict[str, Any]],
    manifest_lookup: dict[tuple[str, str, str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    result = []
    unmapped = Counter()
    for experiment in experiments:
        summary = experiment["summary"]
        retained_lookup = object_direction_lookup(experiment["rows"])
        for row in experiment["rows"]:
            gold = str(row.get("gt_direction"))
            answer_format = str(row.get("answer_format"))
            predicted_answer = row.get("parsed_answer")
            if not parsed(row) or predicted_answer is None:
                selected_direction = "parse_failure"
            elif answer_format == "direction":
                selected_direction = str(predicted_answer)
            else:
                ref_key = reference_key(row)
                full_key = (*ref_key, summary["frame"])
                selected_direction = manifest_lookup.get(full_key, {}).get(str(predicted_answer))
                if selected_direction is None:
                    selected_direction = retained_lookup.get(ref_key, {}).get(str(predicted_answer), "unmapped")
                if selected_direction == "unmapped":
                    unmapped[summary["frame"]] += 1
            result.append({
                "frame": summary["frame"],
                "model": summary["model"],
                "answer_format": answer_format,
                "gt_direction": gold,
                "selected_direction": selected_direction,
            })
    counts = Counter((row["frame"], row["model"], row["answer_format"], row["gt_direction"], row["selected_direction"]) for row in result)
    collapsed = [
        {"frame": key[0], "model": key[1], "answer_format": key[2], "gt_direction": key[3], "selected_direction": key[4], "count": count}
        for key, count in sorted(counts.items())
    ]
    return collapsed, dict(unmapped)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(summaries: list[dict[str, Any]], path: Path) -> None:
    x = list(range(len(summaries)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(8, 3.0 * len(summaries)), 5.8))
    objects = ax.bar([value - width / 2 for value in x], [row["object_accuracy"] for row in summaries], width, label="Answer with object", color="#457B9D")
    directions = ax.bar([value + width / 2 for value in x], [row["direction_accuracy"] for row in summaries], width, label="Answer with direction", color="#E9C46A")
    for bars in (objects, directions):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{row['frame_label']}\n{row['model']}" for row in summaries])
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("ScanNet object-answer versus direction-answer accuracy")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(summaries: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4.0, 1.0 + 0.8 * len(summaries))))
    for y, row in enumerate(summaries):
        left = 0.0
        for outcome in OUTCOMES:
            width = row[outcome]
            ax.barh(y, width, left=left, color=OUTCOME_COLORS[outcome], edgecolor="white", height=0.62)
            if width >= 0.05:
                ax.text(left + width / 2, y, f"{width:.1%}", ha="center", va="center", fontsize=9)
            left += width
    ax.set_yticks(range(len(summaries)), [f"{row['frame_label']} · {row['model']} (n={row['pairs']})" for row in summaries])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of matched pairs")
    ax.set_title("ScanNet paired object/direction outcomes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[key]) for key in OUTCOMES]
    ax.legend(handles, [OUTCOME_LABELS[key] for key in OUTCOMES], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(confusions: list[dict[str, Any]], summaries: list[dict[str, Any]], path: Path) -> None:
    extras = tuple(
        value for value in ("unmapped", "parse_failure")
        if any(row["selected_direction"] == value for row in confusions)
    )
    columns = (*DIRECTIONS, *extras)
    fig, axes = plt.subplots(
        len(summaries),
        len(FORMATS),
        figsize=(14, max(5.0, 4.4 * len(summaries))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, summary in enumerate(summaries):
        for column_index, answer_format in enumerate(FORMATS):
            ax = axes[row_index, column_index]
            matrix = []
            for gold in DIRECTIONS:
                counts = {column: 0 for column in columns}
                for row in confusions:
                    if row["frame"] == summary["frame"] and row["model"] == summary["model"] and row["answer_format"] == answer_format and row["gt_direction"] == gold:
                        if row["selected_direction"] in counts:
                            counts[row["selected_direction"]] += int(row["count"])
                total = sum(counts.values())
                matrix.append([counts[column] / total if total else 0.0 for column in columns])
            image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(columns)), columns)
            ax.set_yticks(range(len(DIRECTIONS)), DIRECTIONS)
            ax.set_xlabel("Direction implied by answer")
            ax.set_ylabel("Ground-truth direction")
            ax.set_title(f"{summary['frame_label']} · answer with {answer_format}")
            for y, values in enumerate(matrix):
                for x, value in enumerate(values):
                    if value >= 0.01:
                        ax.text(x, y, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Row-normalized share", shrink=0.82, pad=0.02)
    fig.suptitle("ScanNet ground truth versus direction implied by answer")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, summaries: list[dict[str, Any]], direction_rows: list[dict[str, Any]], unmapped: dict[str, int]) -> None:
    lines = [
        "# ScanNet object-answer versus direction-answer analysis",
        "",
        "The two answer formats are evaluated on matched scene relations with four object candidates and four balanced direction labels.",
        "",
        "## Main results",
        "",
        "| Frame | Model | Pairs | Object accuracy | Direction accuracy | Object−direction | Object parsed | Direction parsed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['frame_label']} | {row['model']} | {row['pairs']} | {row['object_accuracy']:.1%} | "
            f"{row['direction_accuracy']:.1%} | {row['object_minus_direction']:+.1%} | "
            f"{row['object_parse_success']:.1%} | {row['direction_parse_success']:.1%} |"
        )
    lines.extend((
        "",
        "## Paired cases",
        "",
        "| Frame | Both correct | Object only | Direction only | Both wrong | Parse failure |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in summaries:
        lines.append(f"| {row['frame_label']} | {row['both_correct']:.1%} | {row['object_only']:.1%} | {row['direction_only']:.1%} | {row['both_wrong']:.1%} | {row['parse_failure']:.1%} |")
    lines.extend((
        "",
        "## Accuracy by ground-truth direction",
        "",
        "| Frame | Direction | Object answer | Direction answer |",
        "|---|---|---:|---:|",
    ))
    keyed = {(row["frame"], row["gt_direction"], row["answer_format"]): row for row in direction_rows}
    for summary in summaries:
        for direction in DIRECTIONS:
            obj = keyed[(summary["frame"], direction, "object")]
            direct = keyed[(summary["frame"], direction, "direction")]
            lines.append(f"| {summary['frame_label']} | {direction} | {obj['accuracy']:.1%} | {direct['accuracy']:.1%} |")
    lines.extend((
        "",
        "## Artifacts",
        "",
        "- `paired_cases.csv` contains every matched pair and its outcome, making object-only and direction-only cases directly filterable.",
        "- `direction_breakdown.csv` contains accuracy and parse success by frame, relation, and answer format.",
        "- `answer_direction_confusion.csv` maps direction answers directly and object answers through the target-direction lookup available in retained records.",
        (
            f"- Unmapped selected-object directions by frame: `{json.dumps(unmapped, sort_keys=True)}`. "
            "These are shown explicitly rather than guessed."
            if unmapped
            else "- Every selected object was mapped to its geometric direction using the source ScanNet manifest."
        ),
        "- `accuracy.png`, `paired_outcomes_100pct.png`, and `answer_vs_gt_direction.png` visualize the main comparisons.",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output(inputs: list[Path], requested: Path | None) -> Path:
    if requested:
        output = requested.expanduser().resolve()
    else:
        first = inputs[0].expanduser().resolve()
        output = (first if first.is_dir() else first.parent) / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    args = parse_args()
    experiments = []
    for path in discover(args.inputs):
        metadata, rows = load(path)
        experiments.append(summarize_submission(path, metadata, rows))
    experiments.sort(key=lambda item: (FRAME_ORDER.index(item["summary"]["frame"]), item["summary"]["model"]))
    summaries = [item["summary"] for item in experiments]
    pair_rows = build_pair_rows(experiments)
    direction_rows = build_direction_rows(experiments)
    manifest_lookup = manifest_direction_lookup(args.manifest, experiments)
    confusion_rows, unmapped = build_confusion_rows(experiments, manifest_lookup)
    output = resolve_output(args.inputs, args.output_dir)

    serializable_summaries = [{key: value for key, value in row.items() if key != "issues"} for row in summaries]
    (output / "summary.json").write_text(json.dumps({"experiments": serializable_summaries, "unmapped_object_predictions": unmapped}, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "summary.csv", serializable_summaries)
    write_csv(output / "paired_cases.csv", pair_rows)
    write_csv(output / "direction_breakdown.csv", direction_rows)
    write_csv(output / "answer_direction_confusion.csv", confusion_rows)
    write_markdown(output / "summary.md", summaries, direction_rows, unmapped)
    plot_accuracy(summaries, output / "accuracy.png")
    plot_outcomes(summaries, output / "paired_outcomes_100pct.png")
    plot_confusions(confusion_rows, summaries, output / "answer_vs_gt_direction.png")

    print(f"Analyzed {len(experiments)} submissions and {sum(row['pairs'] for row in summaries)} matched pairs.")
    print(output / "summary.md")
    for row in summaries:
        print(f"{row['frame_label']}: object={row['object_accuracy']:.1%}, direction={row['direction_accuracy']:.1%}, gap={row['object_minus_direction']:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
