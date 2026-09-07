#!/usr/bin/env python3
"""Summarize the four COMFORT object/direction inverse diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_INPUT = Path("outputs/comfort_inverse_diagnostics")
OUTPUT_DIR_NAME = "inverse_diagnostic_analysis"
PREFIXES = (
    "comfort_full_map_inversion_",
    "comfort_arrow_length_sweep_",
    "comfort_map_ablation_",
    "comfort_option_permutation_",
)
FORMAT_COLORS = {"object": "#457B9D", "direction": "#E9C46A"}
MAPPING_COLORS = {"relation_to_object": "#457B9D", "object_to_relation": "#E76F51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[DEFAULT_INPUT], help="Submission files or directories searched recursively.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def discover(inputs: Iterable[Path]) -> list[Path]:
    files = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*.json") if candidate.name.startswith(PREFIXES))
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No COMFORT inverse-diagnostic submissions found")
    return files


def load(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No records in {path}")
    rows = [row for row in rows if isinstance(row, dict)]
    experiment = str(payload.get("experiment", "")) if isinstance(payload, dict) else ""
    if not experiment:
        values = {str(row.get("diagnostic_experiment")) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Cannot infer experiment for {path}")
        experiment = next(iter(values))
    return experiment, rows


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows) if rows else 0.0


def grouped_means(rows: list[dict[str, Any]], keys: tuple[str, ...], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output = []
    for values, group in sorted(grouped.items()):
        output.append({**dict(zip(keys, values)), "count": len(group), **{field: mean(group, field) for field in fields}})
    return output


def paired_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[str(row.get("source_relation_id"))][str(row.get("answer_format"))] = row
    grouped = defaultdict(list)
    for pair in pairs.values():
        if not {"object", "direction"} <= set(pair):
            continue
        key = (str(pair["object"].get("experiment_condition")), str(pair["object"].get("relation")))
        grouped[key].append(pair)
    output = []
    for (condition, relation), group in sorted(grouped.items()):
        counts = Counter()
        for pair in group:
            obj = bool(pair["object"].get("score"))
            direction = bool(pair["direction"].get("score"))
            counts["both_correct" if obj and direction else "object_only" if obj else "direction_only" if direction else "both_wrong"] += 1
        total = len(group)
        output.append(
            {
                "experiment_condition": condition,
                "relation": relation,
                "paired_count": total,
                **{name: counts[name] / total for name in ("both_correct", "object_only", "direction_only", "both_wrong")},
            }
        )
    return output


def permutation_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("permutation_group_id"))].append(row)
    records = []
    for group_id, group in grouped.items():
        if len(group) != 4:
            continue
        records.append(
            {
                "permutation_group_id": group_id,
                "answer_format": group[0].get("answer_format"),
                "relation": group[0].get("relation"),
                "semantic_consistency": float(
                    all(float(row.get("parse_success", 0.0)) == 1.0 for row in group)
                    and len({str(row.get("selected_answer")) for row in group}) == 1
                ),
                "all_correct": float(all(float(row.get("score", 0.0)) == 1.0 for row in group)),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grouped_bar(
    rows: list[dict[str, Any]],
    category: str,
    series: str,
    value: str,
    colors: dict[str, str],
    title: str,
    path: Path,
) -> None:
    if not rows:
        return
    categories = list(dict.fromkeys(row[category] for row in rows))
    series_names = list(dict.fromkeys(row[series] for row in rows))
    x = list(range(len(categories)))
    width = 0.8 / max(1, len(series_names))
    fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(categories)), 5.8))
    for index, name in enumerate(series_names):
        values = []
        for item in categories:
            match = next((row for row in rows if row[category] == item and row[series] == name), None)
            values.append(float(match[value]) if match else 0.0)
        offset = (index - (len(series_names) - 1) / 2) * width
        bars = ax.bar([position + offset for position in x], values, width, label=name.replace("_", " "), color=colors.get(name))
        for bar, score in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, min(1.02, score + 0.015), f"{score:.1%}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, [item.replace("_", " ") for item in categories], rotation=24, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel(value.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_arrow(rows: list[dict[str, Any]], output: Path) -> None:
    overall = grouped_means(rows, ("experiment_condition", "answer_format"), ("score",))
    lengths = [name for name in ("0.25", "0.45", "0.70", "0.90", "1.15", "1.50") if any(row["experiment_condition"] == name for row in overall)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for answer_format in ("object", "direction"):
        points = [
            (float(length), row["score"])
            for length in lengths
            for row in overall
            if row["experiment_condition"] == length and row["answer_format"] == answer_format
        ]
        if points:
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.2,
                label=answer_format.title(),
                color=FORMAT_COLORS[answer_format],
            )
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Arrow length / reference bbox diagonal")
    ax.set_ylabel("Accuracy")
    ax.set_title("Arrow-Length Sweep: Object vs Direction Answers")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "arrow_length_accuracy.png", dpi=200)
    plt.close(fig)


def plot_permutation(rows: list[dict[str, Any]], groups: list[dict[str, Any]], output: Path) -> None:
    position = grouped_means(rows, ("permutation_index", "answer_format"), ("score",))
    _grouped_bar(position, "permutation_index", "answer_format", "score", FORMAT_COLORS, "Accuracy by Gold Option Position", output / "option_position_accuracy.png")
    if not groups:
        return
    summary = grouped_means(groups, ("answer_format",), ("semantic_consistency", "all_correct"))
    reshaped = []
    for row in summary:
        for metric in ("semantic_consistency", "all_correct"):
            reshaped.append({"answer_format": row["answer_format"], "metric": metric, "value": row[metric]})
    _grouped_bar(reshaped, "metric", "answer_format", "value", FORMAT_COLORS, "Option-Permutation Stability", output / "option_permutation_stability.png")


def write_markdown(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# COMFORT inverse diagnostic results", ""]
    for experiment, rows in tables.items():
        lines.extend([f"## {experiment.replace('_', ' ').title()}", ""])
        if not rows:
            lines.extend(["No records.", ""])
            continue
        columns = list(rows[0])
        lines.append("| " + " | ".join(column.replace("_", " ").title() for column in columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            values = [f"{row[column]:.3f}" if isinstance(row[column], float) else str(row[column]) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    submissions = {}
    for path in discover(args.inputs):
        experiment, rows = load(path)
        if experiment in submissions:
            raise ValueError(f"More than one submission found for {experiment}; pass one model/run at a time")
        submissions[experiment] = rows
    first = args.inputs[0].expanduser().resolve()
    parent = first if first.is_dir() else first.parent
    output = args.output_dir.expanduser().resolve() if args.output_dir else parent / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)

    tables = {}
    full = submissions.get("full_map_inversion", [])
    if full:
        tables["full_map_inversion"] = grouped_means(full, ("experiment_condition", "mapping_format"), ("mapping_edge_accuracy", "mapping_exact_accuracy", "parse_success"))
        _grouped_bar(tables["full_map_inversion"], "experiment_condition", "mapping_format", "mapping_edge_accuracy", MAPPING_COLORS, "Full-Map Inversion Edge Accuracy", output / "full_map_inversion.png")
    for experiment in ("arrow_length_sweep", "map_ablation"):
        rows = submissions.get(experiment, [])
        if not rows:
            continue
        tables[experiment] = grouped_means(rows, ("experiment_condition", "answer_format"), ("score", "parse_success"))
        tables[f"{experiment}_by_relation"] = grouped_means(rows, ("experiment_condition", "answer_format", "relation"), ("score",))
        tables[f"{experiment}_paired"] = paired_outcomes(rows)
        if experiment == "arrow_length_sweep":
            plot_arrow(rows, output)
        else:
            _grouped_bar(tables[experiment], "experiment_condition", "answer_format", "score", FORMAT_COLORS, "Canonical-Map Ablation Accuracy", output / "map_ablation_accuracy.png")
    permutation = submissions.get("option_permutation", [])
    if permutation:
        groups = permutation_groups(permutation)
        tables["option_permutation"] = grouped_means(permutation, ("answer_format", "permutation_index"), ("score", "parse_success"))
        tables["option_permutation_stability"] = grouped_means(groups, ("answer_format", "relation"), ("semantic_consistency", "all_correct"))
        plot_permutation(permutation, groups, output)

    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    write_markdown(output / "summary.md", tables)
    print(f"Analyzed {len(submissions)} experiments; wrote outputs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
