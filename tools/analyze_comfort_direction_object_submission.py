#!/usr/bin/env python3
"""Analyze a COMFORT direction/object submission and save its reports.

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
    "comfort_direction_object_0/submissions/"
    "comfort_direction_object_qwen3_vl_experiments.json"
)


def _save_paired_outcomes_plot(report: dict, path: Path) -> None:
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
    axis.set_title("COMFORT direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    json_path = output_dir / "comfort_direction_object_analysis.json"
    text_path = output_dir / "comfort_direction_object_analysis.txt"
    plot_path = output_dir / "comfort_paired_outcomes.png"
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
