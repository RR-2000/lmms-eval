#!/usr/bin/env python3
"""Analyze a COMFORT Multi-3D or Oriented-3D direction/object submission.

The submission contains paired ``direction`` and ``object`` rows for each
source relation.  This script writes a JSON report, a human-readable summary,
and a stacked paired-outcomes chart beside the run's ``submissions`` directory.

Bare record lists and the wrapped JSON emitted by the COMFORT task are both
accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kubric_movi_e_direction_object_submission import (
    OUTCOME_KEYS,
    RELATIONS,
    _default_output_dir,
    _load_records,
    analyze,
    format_report,
)

DEFAULT_SUBMISSION = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/"
    "comfort_oriented_3d_direction_object_0/submissions/"
    "comfort_oriented_3d_direction_object_qwen3_vl_experiments.json"
)

ORIENTED_TASK = "comfort_oriented_3d_direction_object"
MULTI_TASK = "comfort_direction_object"


def _save_paired_outcomes_plot(
    report: dict, path: Path, dataset_name: str = "COMFORT_Multi_3D"
) -> None:
    """Save the same matched-pair outcome plot used for MOVi-E analyses."""
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
    axis.set_title(f"{dataset_name} direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _submission_identity(submission: Path, records: list[dict]) -> tuple[str, str]:
    """Infer dataset/task identity from wrapper metadata, records, or filename."""
    with submission.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    wrapper = payload if isinstance(payload, dict) else {}
    first = records[0] if records else {}
    task_name = str(wrapper.get("task") or first.get("task") or "")
    dataset_name = str(wrapper.get("dataset") or first.get("dataset") or "")

    oriented_hint = "oriented" in str(submission).lower()
    if task_name not in {MULTI_TASK, ORIENTED_TASK}:
        task_name = ORIENTED_TASK if oriented_hint else MULTI_TASK
    if not dataset_name:
        dataset_name = (
            "COMFORT_Oriented_3D" if task_name == ORIENTED_TASK else "COMFORT_Multi_3D"
        )
    return dataset_name, task_name


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

    records = _load_records(args.submission)
    dataset_name, task_name = _submission_identity(args.submission, records)
    report = analyze(records)
    report = {"dataset": dataset_name, "task": task_name, **report}
    output_dir = args.output_dir or _default_output_dir(args.submission)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_stem = f"{task_name}_analysis"
    json_path = output_dir / f"{analysis_stem}.json"
    text_path = output_dir / f"{analysis_stem}.txt"
    markdown_path = output_dir / "summary.md"
    plot_name = (
        "comfort_oriented_3d_paired_outcomes.png"
        if task_name == ORIENTED_TASK
        else "comfort_paired_outcomes.png"
    )
    plot_path = output_dir / plot_name
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_report = format_report(report)
    text_path.write_text(text_report, encoding="utf-8")
    markdown_path.write_text(
        f"# {dataset_name} direction/object analysis\n\n"
        f"Submission: `{args.submission.resolve()}`\n\n"
        "```text\n" + text_report.rstrip() + "\n```\n",
        encoding="utf-8",
    )
    _save_paired_outcomes_plot(report, plot_path, dataset_name)

    print(json.dumps(report, indent=2) if args.print_json else text_report, end="")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved text report: {text_path}")
    print(f"Saved Markdown summary: {markdown_path}")
    print(f"Saved paired-outcomes plot: {plot_path}")


if __name__ == "__main__":
    main()
