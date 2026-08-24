#!/usr/bin/env python3
"""Analyze a Kubric MOVi-E direction/object submission and save its reports.

By default this reads the Qwen3-VL MOVi-E submission and writes the following
beside the run's ``submissions`` directory:

* ``movi_e_direction_object_analysis.json``
* ``movi_e_direction_object_analysis.txt``
* ``paired_outcomes.png``

The input is the wrapped submission JSON written by
``kubric_movi_e_direction_object``. Bare record lists are also accepted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUBMISSION = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/"
    "kubric_movi_e_direction_object_0/submissions/"
    "kubric_movi_e_direction_object_qwen3_vl_experiments.json"
)
ANSWER_FORMATS = ("direction", "object")
RELATIONS = ("left", "right", "front", "behind")
OUTCOME_KEYS = (
    "improved",
    "worsened",
    "unchanged_correct",
    "unchanged_incorrect",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get(
            "records", payload.get("results", payload.get("submissions"))
        )
    else:
        records = None
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError("Expected a submission record list or a wrapper with records")
    return records


def _score(row: dict[str, Any]) -> float:
    value = row.get("score")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record {row.get('qid', '<unknown>')} has no numeric score"
        ) from exc


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    parse_successes = [bool(row.get("parse_success")) for row in rows]
    return {
        "count": len(rows),
        "correct": sum(score == 1.0 for score in scores),
        "accuracy": sum(scores) / len(scores) if scores else None,
        "parse_success_count": sum(parse_successes),
        "parse_success_rate": (
            sum(parse_successes) / len(parse_successes) if parse_successes else None
        ),
    }


def _group_sources(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    sources: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_rows = []
    for row in rows:
        source_id = str(row.get("source_relation_id") or row.get("source_qid") or "")
        answer_format = str(row.get("answer_format") or row.get("variant") or "")
        if not source_id or answer_format not in ANSWER_FORMATS:
            continue
        if answer_format in sources[source_id]:
            duplicate_rows.append(str(row.get("qid", source_id)))
        sources[source_id][answer_format] = row
    return sources, duplicate_rows


def _complete_pairs(
    sources: dict[str, dict[str, dict[str, Any]]], relation: str | None = None
) -> list[dict[str, dict[str, Any]]]:
    pairs = []
    for pair in sources.values():
        if set(ANSWER_FORMATS) > set(pair):
            continue
        if relation is not None and str(pair["direction"].get("relation")) != relation:
            continue
        pairs.append(pair)
    return pairs


def _paired_report(pairs: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    outcomes = {key: 0 for key in OUTCOME_KEYS}
    differences = []
    for pair in pairs:
        object_score = _score(pair["object"])
        direction_score = _score(pair["direction"])
        differences.append(object_score - direction_score)
        object_correct = object_score == 1.0
        direction_correct = direction_score == 1.0
        if object_correct and not direction_correct:
            outcomes["improved"] += 1
        elif direction_correct and not object_correct:
            outcomes["worsened"] += 1
        elif object_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    return {
        "paired_count": len(pairs),
        "object_minus_direction": (
            sum(differences) / len(differences) if differences else None
        ),
        **outcomes,
    }


def _categorical_counts(
    rows: list[dict[str, Any]], field: str, missing: str = "parse_failure"
) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get(field) or missing) for row in rows).items())
    )


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources, duplicates = _group_sources(records)
    complete = _complete_pairs(sources)
    incomplete = {
        source_id: sorted(set(ANSWER_FORMATS) - set(pair))
        for source_id, pair in sources.items()
        if set(ANSWER_FORMATS) > set(pair)
    }

    by_answer_format = {
        answer_format: _summary(
            [row for row in records if row.get("answer_format") == answer_format]
        )
        for answer_format in ANSWER_FORMATS
    }
    by_relation = {}
    for relation in RELATIONS:
        relation_rows = [row for row in records if row.get("relation") == relation]
        by_relation[relation] = {
            "overall": _summary(relation_rows),
            **{
                answer_format: _summary(
                    [
                        row
                        for row in relation_rows
                        if row.get("answer_format") == answer_format
                    ]
                )
                for answer_format in ANSWER_FORMATS
            },
            "paired": _paired_report(_complete_pairs(sources, relation)),
        }

    by_sequence = {}
    for sequence_name in sorted(
        {str(row.get("sequence_name", "<missing>")) for row in records}
    ):
        sequence_rows = [
            row
            for row in records
            if str(row.get("sequence_name", "<missing>")) == sequence_name
        ]
        by_sequence[sequence_name] = {
            "overall": _summary(sequence_rows),
            **{
                answer_format: _summary(
                    [
                        row
                        for row in sequence_rows
                        if row.get("answer_format") == answer_format
                    ]
                )
                for answer_format in ANSWER_FORMATS
            },
        }

    report: dict[str, Any] = {
        "records": len(records),
        "source_groups": len(sources),
        "complete_pairs": len(complete),
        "overall": _summary(records),
        "by_answer_format": by_answer_format,
        "by_relation": by_relation,
        "paired": _paired_report(complete),
        "gold_option_distribution": _categorical_counts(
            records, "gold_option_letter", "missing"
        ),
        "predicted_option_distribution": _categorical_counts(
            records, "predicted_option_letter"
        ),
        "by_sequence": by_sequence,
    }
    if incomplete:
        report["incomplete_sources"] = incomplete
    if duplicates:
        report["duplicate_rows"] = duplicates
    return report


def _format_summary(label: str, summary: dict[str, Any]) -> str:
    accuracy = "N/A" if summary["accuracy"] is None else f"{summary['accuracy']:.4f}"
    parse_rate = (
        "N/A"
        if summary["parse_success_rate"] is None
        else f"{summary['parse_success_rate']:.4f}"
    )
    return (
        f"{label:16} {summary['correct']:>5}/{summary['count']:<5} "
        f"accuracy={accuracy:>7} parse_success={parse_rate:>7}"
    )


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Records: {report['records']}",
        f"Source groups: {report['source_groups']}",
        f"Complete pairs: {report['complete_pairs']}",
        "",
        "Accuracy by answer format",
        _format_summary("overall", report["overall"]),
    ]
    for answer_format in ANSWER_FORMATS:
        lines.append(
            _format_summary(
                answer_format, report["by_answer_format"][answer_format]
            )
        )

    paired = report["paired"]
    gain = (
        "N/A"
        if paired["object_minus_direction"] is None
        else f"{paired['object_minus_direction']:+.4f}"
    )
    lines.extend(
        [
            "",
            f"Paired object-minus-direction gain: {gain} ({paired['paired_count']} pairs)",
            (
                "Paired outcomes: "
                f"improved={paired['improved']}, worsened={paired['worsened']}, "
                f"unchanged-correct={paired['unchanged_correct']}, "
                f"unchanged-incorrect={paired['unchanged_incorrect']}"
            ),
            "",
            "By relation",
        ]
    )
    for relation in RELATIONS:
        relation_report = report["by_relation"][relation]
        relation_gain = relation_report["paired"]["object_minus_direction"]
        relation_gain_text = "N/A" if relation_gain is None else f"{relation_gain:+.4f}"
        lines.append(
            f"  {relation:8} direction={relation_report['direction']['accuracy']:.4f} "
            f"object={relation_report['object']['accuracy']:.4f} "
            f"gain={relation_gain_text} pairs={relation_report['paired']['paired_count']}"
        )
    lines.extend(
        [
            "",
            f"Gold option distribution: {report['gold_option_distribution']}",
            f"Predicted option distribution: {report['predicted_option_distribution']}",
        ]
    )
    if report.get("incomplete_sources"):
        lines.append(f"Incomplete sources: {len(report['incomplete_sources'])}")
    if report.get("duplicate_rows"):
        lines.append(f"Duplicate rows: {len(report['duplicate_rows'])}")
    return "\n".join(lines) + "\n"


def _save_paired_outcomes_plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib") from exc

    labels = ["overall", *RELATIONS]
    paired_reports = [
        report["paired"],
        *(report["by_relation"][relation]["paired"] for relation in RELATIONS),
    ]
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
        values = [paired[key] for paired in paired_reports]
        axis.bar(labels, values, bottom=bottom, label=display[key], color=colors[key])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Number of matched source relations")
    axis.set_title("Kubric MOVi-E direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _default_output_dir(submission: Path) -> Path:
    return submission.parent.parent if submission.parent.name == "submissions" else submission.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        default=DEFAULT_SUBMISSION,
        help=f"Submission JSON (default: {DEFAULT_SUBMISSION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the run directory containing submissions/",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="Print the JSON report instead of the text summary",
    )
    args = parser.parse_args()

    report = analyze(_load_records(args.submission))
    output_dir = args.output_dir or _default_output_dir(args.submission)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "movi_e_direction_object_analysis.json"
    text_path = output_dir / "movi_e_direction_object_analysis.txt"
    plot_path = output_dir / "paired_outcomes.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_report = format_report(report)
    text_path.write_text(text_report, encoding="utf-8")
    _save_paired_outcomes_plot(report, plot_path)

    print(json.dumps(report, indent=2) if args.print_json else text_report, end="")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved text report: {text_path}")
    print(f"Saved paired-outcomes plot: {plot_path}")


if __name__ == "__main__":
    main()
