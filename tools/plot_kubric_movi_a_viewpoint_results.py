#!/usr/bin/env python3
"""Plot Kubric MOVi-A viewpoint metrics and generate simple inferences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_viewpoint_pred/"
    "Qwen__Qwen3-VL-4B-Instruct/20260708_112529_results.json"
)

FAMILIES = [
    "camera_relative_position",
    "camera_distance",
    "height_relative_3d",
    "object_centric_relative_position",
    "object_centric_relative_position_multi",
    "object_centric_direction_binary",
    "object_centric_camera_pose",
]

CASE_METRICS = [
    "answer_and_direction_correct",
    "answer_correct_direction_wrong",
    "answer_wrong_direction_correct",
    "answer_and_direction_wrong",
]

CASE_LABELS = {
    "answer_and_direction_correct": "Answer+Direction Correct",
    "answer_correct_direction_wrong": "Answer Correct, Direction Wrong",
    "answer_wrong_direction_correct": "Answer Wrong, Direction Correct",
    "answer_and_direction_wrong": "Answer+Direction Wrong",
}

CASE_COLORS = {
    "answer_and_direction_correct": "#2A9D8F",
    "answer_correct_direction_wrong": "#E9C46A",
    "answer_wrong_direction_correct": "#F4A261",
    "answer_and_direction_wrong": "#E76F51",
}

FAMILY_DISPLAY = {
    "camera_relative_position": "Camera Relative",
    "camera_distance": "Camera Distance",
    "height_relative_3d": "Height 3D",
    "object_centric_relative_position": "Object-Centric",
    "object_centric_relative_position_multi": "Object-Centric Multi",
    "object_centric_direction_binary": "Obj-Centric Binary",
    "object_centric_camera_pose": "Camera Pose",
}

OBJECT_CENTRIC_SINGLE_METRICS = [
    "object_centric_relative_position_camera_direction_cosine",
    "object_centric_relative_position_right_sign_accuracy",
    "object_centric_relative_position_front_sign_accuracy",
    "object_centric_relative_position_full_sign_accuracy",
    "object_centric_relative_position_camera_front_sign_accuracy",
    "object_centric_relative_position_camera_vector_nonzero",
]

OBJECT_CENTRIC_MULTI_METRICS = [
    "object_centric_relative_position_multi_camera_direction_cosine",
    "object_centric_relative_position_multi_right_sign_accuracy",
    "object_centric_relative_position_multi_front_sign_accuracy",
    "object_centric_relative_position_multi_full_sign_accuracy",
    "object_centric_relative_position_multi_camera_front_sign_accuracy",
    "object_centric_relative_position_multi_camera_vector_nonzero",
    "object_centric_relative_position_multi_candidate_aware_direction_accuracy",
    "object_centric_relative_position_multi_direction_given_predicted_object_accuracy",
    "object_centric_relative_position_multi_ranking_accuracy_on_relation_axis",
    "object_centric_relative_position_multi_ranking_score_on_relation_axis",
    "object_centric_relative_position_multi_predicted_target_in_candidate_set",
]

OBJECT_CENTRIC_BINARY_METRICS = [
    "object_centric_direction_binary_camera_direction_cosine",
    "object_centric_direction_binary_camera_distance_score",
    "object_centric_direction_binary_camera_right_sign_accuracy",
    "object_centric_direction_binary_camera_up_sign_accuracy",
    "object_centric_direction_binary_camera_front_sign_accuracy",
    "object_centric_direction_binary_camera_full_sign_accuracy",
    "object_centric_direction_binary_right_sign_accuracy",
    "object_centric_direction_binary_front_sign_accuracy",
    "object_centric_direction_binary_full_sign_accuracy",
    "object_centric_direction_binary_camera_vector_nonzero",
]

OBJECT_CENTRIC_CAMERA_POSE_METRICS = [
    "object_centric_camera_pose_camera_direction_cosine",
    "object_centric_camera_pose_camera_distance_score",
    "object_centric_camera_pose_camera_right_sign_accuracy",
    "object_centric_camera_pose_camera_up_sign_accuracy",
    "object_centric_camera_pose_camera_front_sign_accuracy",
    "object_centric_camera_pose_camera_full_sign_accuracy",
]

GEOMETRY_METRICS = [
    "viewpoint_reference_to_camera_cosine",
    "viewpoint_reference_to_camera_distance_score",
    "viewpoint_vector_cosine",
    "viewpoint_scale_score",
    "object_centric_relative_position_camera_direction_cosine",
    "object_centric_relative_position_camera_distance_score",
    "object_centric_relative_position_multi_camera_direction_cosine",
    "object_centric_relative_position_multi_camera_distance_score",
    "object_centric_direction_binary_camera_direction_cosine",
    "object_centric_direction_binary_camera_distance_score",
    "object_centric_camera_pose_camera_direction_cosine",
    "object_centric_camera_pose_camera_distance_score",
]

METRIC_LABELS = {
    "object_centric_relative_position_camera_direction_cosine": "Camera Vector Cosine",
    "object_centric_relative_position_right_sign_accuracy": "Right Sign",
    "object_centric_relative_position_front_sign_accuracy": "Front Sign",
    "object_centric_relative_position_full_sign_accuracy": "Right+Front Both",
    "object_centric_relative_position_camera_front_sign_accuracy": "Camera Front Sign",
    "object_centric_relative_position_camera_vector_nonzero": "Camera Vector Nonzero",
    "object_centric_relative_position_multi_camera_direction_cosine": "Camera Vector Cosine",
    "object_centric_relative_position_multi_right_sign_accuracy": "Right Sign",
    "object_centric_relative_position_multi_front_sign_accuracy": "Front Sign",
    "object_centric_relative_position_multi_full_sign_accuracy": "Right+Front Both",
    "object_centric_relative_position_multi_camera_front_sign_accuracy": "Camera Front Sign",
    "object_centric_relative_position_multi_camera_vector_nonzero": "Camera Vector Nonzero",
    "object_centric_relative_position_multi_candidate_aware_direction_accuracy": "Chosen Candidate On Correct Side",
    "object_centric_relative_position_multi_direction_given_predicted_object_accuracy": "Vector Matches Chosen Candidate",
    "object_centric_relative_position_multi_ranking_accuracy_on_relation_axis": "Top Candidate Chosen",
    "object_centric_relative_position_multi_ranking_score_on_relation_axis": "Candidate Rank Score",
    "object_centric_relative_position_multi_predicted_target_in_candidate_set": "Predicted Target In Candidates",
    "object_centric_direction_binary_camera_direction_cosine": "Camera Vector Cosine",
    "object_centric_direction_binary_camera_distance_score": "Camera Distance Score",
    "object_centric_direction_binary_camera_right_sign_accuracy": "Camera Right Sign",
    "object_centric_direction_binary_camera_up_sign_accuracy": "Camera Up Sign",
    "object_centric_direction_binary_camera_front_sign_accuracy": "Camera Front Sign",
    "object_centric_direction_binary_camera_full_sign_accuracy": "Camera Full Sign",
    "object_centric_direction_binary_right_sign_accuracy": "Target Right Sign",
    "object_centric_direction_binary_front_sign_accuracy": "Target Front Sign",
    "object_centric_direction_binary_full_sign_accuracy": "Target Right+Front",
    "object_centric_direction_binary_camera_vector_nonzero": "Camera Vector Nonzero",
    "object_centric_camera_pose_camera_direction_cosine": "Camera Vector Cosine",
    "object_centric_camera_pose_camera_distance_score": "Camera Distance Score",
    "object_centric_camera_pose_camera_right_sign_accuracy": "Camera Right Sign",
    "object_centric_camera_pose_camera_up_sign_accuracy": "Camera Up Sign",
    "object_centric_camera_pose_camera_front_sign_accuracy": "Camera Front Sign",
    "object_centric_camera_pose_camera_full_sign_accuracy": "Camera Full Sign",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Kubric MOVi-A viewpoint metrics and infer failure modes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to an LMMS result JSON.",
    )
    parser.add_argument(
        "--task",
        default="kubric_movi_a_viewpoint",
        help="Task key inside the LMMS result JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and reports. Defaults next to the input JSON.",
    )
    return parser.parse_args()


def load_task_results(path: Path, task_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    results = payload.get("results", {}).get(task_name)
    if not isinstance(results, dict):
        raise KeyError(f"Task '{task_name}' not found in {path}")
    return results


def metric_value(results: dict[str, Any], metric_name: str) -> float | None:
    value = results.get(f"{metric_name},none")
    return float(value) if isinstance(value, (int, float)) else None


def family_metric_value(results: dict[str, Any], family: str, suffix: str) -> float | None:
    return metric_value(results, f"{family}_viewpoint_{suffix}")


def overall_metrics(results: dict[str, Any]) -> dict[str, float]:
    keys = [
        "viewpoint_answer_accuracy",
        "viewpoint_axis_sign_accuracy",
        "viewpoint_answer_and_direction_correct",
        "viewpoint_answer_correct_direction_wrong",
        "viewpoint_answer_wrong_direction_correct",
        "viewpoint_answer_and_direction_wrong",
        "viewpoint_reference_to_camera_cosine",
        "viewpoint_reference_to_camera_distance_score",
        "viewpoint_vector_cosine",
        "viewpoint_scale_score",
        "viewpoint_combined_score",
    ]
    return {key: metric_value(results, key) or 0.0 for key in keys}


def family_case_metrics(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    family_cases: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        family_cases[family] = {
            case: family_metric_value(results, family, case) or 0.0 for case in CASE_METRICS
        }
    return family_cases


def named_metrics(results: dict[str, Any], metric_names: list[str]) -> dict[str, float]:
    return {name: metric_value(results, name) or 0.0 for name in metric_names}


def ensure_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_overall_scores(metrics: dict[str, float], output_dir: Path) -> Path:
    labels = [
        "Answer",
        "Direction Sign",
        "Both Correct",
        "Answer Only",
        "Direction Only",
        "Both Wrong",
        "Ref->Cam Cos",
        "Ref->Cam Dist",
        "Vector Cos",
        "Scale",
        "Combined",
    ]
    values = [
        metrics["viewpoint_answer_accuracy"],
        metrics["viewpoint_axis_sign_accuracy"],
        metrics["viewpoint_answer_and_direction_correct"],
        metrics["viewpoint_answer_correct_direction_wrong"],
        metrics["viewpoint_answer_wrong_direction_correct"],
        metrics["viewpoint_answer_and_direction_wrong"],
        metrics["viewpoint_reference_to_camera_cosine"],
        metrics["viewpoint_reference_to_camera_distance_score"],
        metrics["viewpoint_vector_cosine"],
        metrics["viewpoint_scale_score"],
        metrics["viewpoint_combined_score"],
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color="#457B9D")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Kubric MOVi-A Viewpoint Overall Metrics")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "overall_metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_geometry_metrics(geometry_metrics: dict[str, float], output_dir: Path) -> Path:
    """Plot the continuous geometry metrics added to the result summary."""
    labels = [
        "Ref→Cam\nCosine",
        "Ref→Cam\nDistance",
        "Viewpoint\nVector Cosine",
        "Viewpoint\nScale",
        "Obj Single\nCam Cosine",
        "Obj Single\nCam Distance",
        "Obj Multi\nCam Cosine",
        "Obj Multi\nCam Distance",
        "Binary\nCam Cosine",
        "Binary\nCam Distance",
        "Camera Pose\nCam Cosine",
        "Camera Pose\nCam Distance",
    ]
    values = [geometry_metrics[name] for name in GEOMETRY_METRICS]
    colors = [
        "#457B9D",
        "#457B9D",
        "#2A9D8F",
        "#E9C46A",
        "#3A86FF",
        "#3A86FF",
        "#8D99AE",
        "#8D99AE",
        "#F4A261",
        "#F4A261",
        "#1D3557",
        "#1D3557",
    ]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Kubric MOVi-A Viewpoint Geometry Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()

    output_path = output_dir / "geometry_metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_overall_case_split(metrics: dict[str, float], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3))
    left = 0.0
    for case in CASE_METRICS:
        value = metrics[f"viewpoint_{case}"]
        ax.barh(
            ["Overall"],
            [value],
            left=left,
            color=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )
        if value > 0.04:
            ax.text(left + value / 2.0, 0, f"{value:.3f}", ha="center", va="center", fontsize=9)
        left += value

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of samples")
    ax.set_title("Overall Answer/Direction Outcome Split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    fig.tight_layout()

    output_path = output_dir / "overall_case_split.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_family_case_splits(family_cases: dict[str, dict[str, float]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    families = list(family_cases.keys())
    left = [0.0] * len(families)
    for case in CASE_METRICS:
        values = [family_cases[family][case] for family in families]
        ax.barh(
            families,
            values,
            left=left,
            color=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )
        left = [current + value for current, value in zip(left, values)]

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of samples")
    ax.set_title("Per-Family Answer/Direction Outcome Split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()

    output_path = output_dir / "family_case_splits.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overall_case_pie(metrics: dict[str, float], output_dir: Path) -> Path:
    values = [metrics[f"viewpoint_{case}"] for case in CASE_METRICS]
    labels = [CASE_LABELS[case] for case in CASE_METRICS]
    colors = [CASE_COLORS[case] for case in CASE_METRICS]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=110,
        counterclock=False,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")
    ax.set_title("Overall Outcome Split", pad=14)
    fig.tight_layout()

    output_path = output_dir / "overall_case_pie.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_family_case_pies(family_cases: dict[str, dict[str, float]], output_dir: Path) -> Path:
    num_families = len(FAMILIES)
    cols = 3
    rows = (num_families + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = axes.flatten()
    colors = [CASE_COLORS[case] for case in CASE_METRICS]

    for ax, family in zip(axes, FAMILIES):
        values = [family_cases[family][case] for case in CASE_METRICS]
        labels = [CASE_LABELS[case] for case in CASE_METRICS]
        ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 7 else "",
            startangle=110,
            counterclock=False,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            textprops={"fontsize": 8},
        )
        ax.set_title(FAMILY_DISPLAY[family], fontsize=11)

    for ax in axes[len(FAMILIES) :]:
        ax.axis("off")

    fig.suptitle("Per-Family Outcome Pie Charts", fontsize=14, y=0.98)
    fig.tight_layout()

    output_path = output_dir / "family_case_pies.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_family_correctness(family_cases: dict[str, dict[str, float]], output_dir: Path) -> Path:
    families = list(family_cases.keys())
    both_correct = [family_cases[family]["answer_and_direction_correct"] for family in families]
    both_wrong = [family_cases[family]["answer_and_direction_wrong"] for family in families]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(families))
    width = 0.38
    ax.bar([i - width / 2 for i in x], both_correct, width=width, color="#2A9D8F", label="Both Correct")
    ax.bar([i + width / 2 for i in x], both_wrong, width=width, color="#E76F51", label="Both Wrong")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Best vs Worst Joint Outcome by Task Family")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "family_best_worst.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_family_case_grouped_bars(family_cases: dict[str, dict[str, float]], output_dir: Path) -> Path:
    families = list(family_cases.keys())
    x = range(len(families))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, case in enumerate(CASE_METRICS):
        values = [family_cases[family][case] for family in families]
        positions = [i + (idx - 1.5) * width for i in x]
        ax.bar(
            positions,
            values,
            width=width,
            color=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([FAMILY_DISPLAY[family] for family in families], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Per-Family Outcome Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_path = output_dir / "family_case_grouped_bars.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_family_summary_bars(metrics: dict[str, float], family_cases: dict[str, dict[str, float]], output_dir: Path) -> Path:
    families = list(family_cases.keys())
    answer_rates = [
        family_cases[family]["answer_and_direction_correct"] + family_cases[family]["answer_correct_direction_wrong"]
        for family in families
    ]
    direction_rates = [
        family_cases[family]["answer_and_direction_correct"] + family_cases[family]["answer_wrong_direction_correct"]
        for family in families
    ]
    both_correct = [family_cases[family]["answer_and_direction_correct"] for family in families]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(families))
    width = 0.25
    ax.bar([i - width for i in x], answer_rates, width=width, color="#457B9D", label="Answer Accuracy")
    ax.bar([i for i in x], direction_rates, width=width, color="#8D99AE", label="Direction Accuracy")
    ax.bar([i + width for i in x], both_correct, width=width, color="#2A9D8F", label="Both Correct")
    ax.set_xticks(list(x))
    ax.set_xticklabels([FAMILY_DISPLAY[family] for family in families], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Family-Level Accuracy Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "family_accuracy_summary.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_single_metrics(single_metrics: dict[str, float], output_dir: Path) -> Path:
    metric_names = OBJECT_CENTRIC_SINGLE_METRICS
    labels = [METRIC_LABELS[name] for name in metric_names]
    values = [single_metrics[name] for name in metric_names]
    colors = ["#3A86FF", "#2A9D8F", "#E9C46A", "#1D3557", "#F4A261", "#8D99AE"]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object-Centric Single-Target Diagnostics")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "object_centric_single_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_multi_metrics(multi_metrics: dict[str, float], output_dir: Path) -> Path:
    metric_names = OBJECT_CENTRIC_MULTI_METRICS
    labels = [METRIC_LABELS[name] for name in metric_names]
    values = [multi_metrics[name] for name in metric_names]
    colors = [
        "#3A86FF",
        "#2A9D8F",
        "#E9C46A",
        "#1D3557",
        "#F4A261",
        "#8D99AE",
        "#577590",
        "#43AA8B",
        "#F8961E",
        "#277DA1",
        "#90BE6D",
    ]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object-Centric Multi-Target Diagnostics")
    ax.tick_params(axis="x", rotation=33)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "object_centric_multi_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_sign_comparison(
    single_metrics: dict[str, float],
    multi_metrics: dict[str, float],
    output_dir: Path,
) -> Path:
    labels = ["Right Sign", "Front Sign", "Right+Front Both"]
    single_values = [
        single_metrics["object_centric_relative_position_right_sign_accuracy"],
        single_metrics["object_centric_relative_position_front_sign_accuracy"],
        single_metrics["object_centric_relative_position_full_sign_accuracy"],
    ]
    multi_values = [
        multi_metrics["object_centric_relative_position_multi_right_sign_accuracy"],
        multi_metrics["object_centric_relative_position_multi_front_sign_accuracy"],
        multi_metrics["object_centric_relative_position_multi_full_sign_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], single_values, width=width, color="#2A9D8F", label="Single Target")
    ax.bar([i + width / 2 for i in x], multi_values, width=width, color="#F4A261", label="Multi Target")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object-Centric Sign Accuracy Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "object_centric_sign_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_camera_comparison(
    single_metrics: dict[str, float],
    multi_metrics: dict[str, float],
    output_dir: Path,
) -> Path:
    labels = ["Camera Cosine", "Camera Front Sign", "Camera Vector Nonzero"]
    single_values = [
        single_metrics["object_centric_relative_position_camera_direction_cosine"],
        single_metrics["object_centric_relative_position_camera_front_sign_accuracy"],
        single_metrics["object_centric_relative_position_camera_vector_nonzero"],
    ]
    multi_values = [
        multi_metrics["object_centric_relative_position_multi_camera_direction_cosine"],
        multi_metrics["object_centric_relative_position_multi_camera_front_sign_accuracy"],
        multi_metrics["object_centric_relative_position_multi_camera_vector_nonzero"],
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], single_values, width=width, color="#457B9D", label="Single Target")
    ax.bar([i + width / 2 for i in x], multi_values, width=width, color="#8D99AE", label="Multi Target")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object-Centric Camera-Vector Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "object_centric_camera_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_multi_selection_metrics(multi_metrics: dict[str, float], output_dir: Path) -> Path:
    metric_names = [
        "object_centric_relative_position_multi_candidate_aware_direction_accuracy",
        "object_centric_relative_position_multi_direction_given_predicted_object_accuracy",
        "object_centric_relative_position_multi_ranking_accuracy_on_relation_axis",
        "object_centric_relative_position_multi_ranking_score_on_relation_axis",
        "object_centric_relative_position_multi_predicted_target_in_candidate_set",
    ]
    labels = [METRIC_LABELS[name] for name in metric_names]
    values = [multi_metrics[name] for name in metric_names]
    colors = ["#577590", "#43AA8B", "#F8961E", "#277DA1", "#90BE6D"]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object-Centric Multi-Target Selection Diagnostics")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "object_centric_multi_selection_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_centric_multi_selection_pie(multi_metrics: dict[str, float], output_dir: Path) -> Path:
    labels = [
        "Top Candidate Chosen",
        "In Candidate Set But Not Top",
        "Outside Candidate Set / Missing",
    ]
    top = multi_metrics["object_centric_relative_position_multi_ranking_accuracy_on_relation_axis"]
    in_set = multi_metrics["object_centric_relative_position_multi_predicted_target_in_candidate_set"]
    outside = max(0.0, 1.0 - in_set)
    mid = max(0.0, in_set - top)
    values = [top, mid, outside]
    colors = ["#2A9D8F", "#E9C46A", "#E76F51"]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    _, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=105,
        counterclock=False,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")
    ax.set_title("Multi-Target Candidate Selection Outcome", pad=14)
    fig.tight_layout()

    output_path = output_dir / "object_centric_multi_selection_pie.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_block(metric_values: dict[str, float], title: str, output_path: Path) -> Path:
    labels = [METRIC_LABELS.get(name, name) for name in metric_values]
    values = [metric_values[name] for name in metric_values]
    colors = [
        "#3A86FF",
        "#2A9D8F",
        "#E9C46A",
        "#1D3557",
        "#F4A261",
        "#8D99AE",
        "#577590",
        "#43AA8B",
        "#F8961E",
        "#277DA1",
    ][: len(labels)]

    fig, ax = plt.subplots(figsize=(max(10.5, len(labels) * 1.1), 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def infer_metrics(metrics: dict[str, float], family_cases: dict[str, dict[str, float]]) -> list[str]:
    notes: list[str] = []

    answer = metrics["viewpoint_answer_accuracy"]
    direction = metrics["viewpoint_axis_sign_accuracy"]
    answer_only = metrics["viewpoint_answer_correct_direction_wrong"]
    direction_only = metrics["viewpoint_answer_wrong_direction_correct"]
    both_correct = metrics["viewpoint_answer_and_direction_correct"]
    both_wrong = metrics["viewpoint_answer_and_direction_wrong"]
    vector_cos = metrics["viewpoint_vector_cosine"]
    scale = metrics["viewpoint_scale_score"]
    ref_cam = metrics["viewpoint_reference_to_camera_cosine"]

    if answer - direction > 0.15:
        notes.append(
            f"Answer accuracy ({answer:.3f}) is much higher than direction-sign accuracy ({direction:.3f}), suggesting the model often selects the right label without recovering the underlying spatial direction."
        )

    if answer_only > direction_only + 0.10:
        notes.append(
            f"`Answer correct, direction wrong` ({answer_only:.3f}) is much larger than `answer wrong, direction correct` ({direction_only:.3f}), which points to answer priors or shortcut object selection being stronger than geometric consistency."
        )

    if both_wrong > 0.25:
        notes.append(
            f"`Both wrong` remains substantial at {both_wrong:.3f}, so a large slice of the benchmark is still failing at both the discrete and geometric levels."
        )

    if vector_cos < 0.10:
        notes.append(
            f"Vector cosine is very low ({vector_cos:.3f}), indicating that the continuous direction vectors are close to random or are not being followed by the model even when some discrete answers are correct."
        )

    if 0.35 <= scale <= 0.65:
        notes.append(
            f"Scale score is mid-range ({scale:.3f}), which suggests the model captures some apparent-size information but not reliably enough to support precise geometric prediction."
        )

    if ref_cam <= 0.05:
        notes.append(
            f"Reference-to-camera cosine is effectively zero ({ref_cam:.3f}); in practice this usually means the model is not emitting a meaningful `camera_vector` field yet."
        )

    object_single = family_cases["object_centric_relative_position"]
    object_multi = family_cases["object_centric_relative_position_multi"]
    camera_rel = family_cases["camera_relative_position"]
    camera_dist = family_cases["camera_distance"]
    height = family_cases["height_relative_3d"]

    if object_single["answer_and_direction_wrong"] > 0.50:
        notes.append(
            f"Single-target object-centric reasoning is the clearest bottleneck: `both wrong` is {object_single['answer_and_direction_wrong']:.3f}, which suggests difficulty transforming into the anchor-centered, camera-facing frame."
        )

    if object_multi["answer_wrong_direction_correct"] > 0.25:
        notes.append(
            f"Multi-target object-centric reasoning shows a different error mode: `answer wrong, direction correct` is {object_multi['answer_wrong_direction_correct']:.3f}, suggesting the model often gets the side/front-behind geometry roughly right but fails candidate selection."
        )

    if camera_rel["answer_correct_direction_wrong"] > 0.45:
        notes.append(
            f"Camera-relative position has a strong `answer correct, direction wrong` pattern ({camera_rel['answer_correct_direction_wrong']:.3f}); object identification appears easier than producing a faithful signed vector."
        )

    if camera_dist["answer_and_direction_correct"] >= max(
        camera_rel["answer_and_direction_correct"],
        height["answer_and_direction_correct"],
        object_single["answer_and_direction_correct"],
        object_multi["answer_and_direction_correct"],
    ):
        notes.append(
            f"Camera-distance is the strongest family on joint correctness ({camera_dist['answer_and_direction_correct']:.3f}), implying depth-order judgments are currently easier for the model than anchor-centric viewpoint reasoning."
        )

    if height["answer_correct_direction_wrong"] > height["answer_and_direction_correct"]:
        notes.append(
            f"Height-relative 3D questions still show more `answer correct, direction wrong` ({height['answer_correct_direction_wrong']:.3f}) than full success ({height['answer_and_direction_correct']:.3f}), so the model often knows which object to pick without producing a consistent signed `up` value."
        )

    return notes


def write_inference_report(
    metrics: dict[str, float],
    family_cases: dict[str, dict[str, float]],
    object_centric_single_metrics: dict[str, float],
    object_centric_multi_metrics: dict[str, float],
    object_centric_binary_metrics: dict[str, float],
    object_centric_camera_pose_metrics: dict[str, float],
    output_dir: Path,
    input_path: Path,
) -> tuple[Path, Path]:
    inferences = infer_metrics(metrics, family_cases)

    txt_lines = [
        f"Input: {input_path}",
        "",
        "Overall metrics:",
    ]
    for key, value in metrics.items():
        txt_lines.append(f"- {key}: {value:.6f}")

    txt_lines.append("")
    txt_lines.append("Per-family case splits:")
    for family, case_values in family_cases.items():
        txt_lines.append(f"- {family}:")
        for case in CASE_METRICS:
            txt_lines.append(f"  - {case}: {case_values[case]:.6f}")

    txt_lines.append("")
    txt_lines.append("Object-centric single-target diagnostics:")
    for key, value in object_centric_single_metrics.items():
        txt_lines.append(f"- {key}: {value:.6f}")

    txt_lines.append("")
    txt_lines.append("Object-centric multi-target diagnostics:")
    for key, value in object_centric_multi_metrics.items():
        txt_lines.append(f"- {key}: {value:.6f}")

    txt_lines.append("")
    txt_lines.append("Object-centric binary diagnostics:")
    for key, value in object_centric_binary_metrics.items():
        txt_lines.append(f"- {key}: {value:.6f}")

    txt_lines.append("")
    txt_lines.append("Object-centric camera-pose diagnostics:")
    for key, value in object_centric_camera_pose_metrics.items():
        txt_lines.append(f"- {key}: {value:.6f}")

    txt_lines.append("")
    txt_lines.append("Inferences:")
    for note in inferences:
        txt_lines.append(f"- {note}")

    txt_path = output_dir / "inference_report.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n")

    md_lines = [
        "# Kubric MOVi-A Viewpoint Result Analysis",
        "",
        f"Input: `{input_path}`",
        "",
        "## Overall Metrics",
    ]
    for key, value in metrics.items():
        md_lines.append(f"- `{key}`: `{value:.6f}`")

    md_lines.append("")
    md_lines.append("## Per-Family Case Splits")
    for family, case_values in family_cases.items():
        md_lines.append(f"### `{family}`")
        for case in CASE_METRICS:
            md_lines.append(f"- `{case}`: `{case_values[case]:.6f}`")
        md_lines.append("")

    md_lines.append("## Object-Centric Single-Target Diagnostics")
    for key, value in object_centric_single_metrics.items():
        md_lines.append(f"- `{key}`: `{value:.6f}`")

    md_lines.append("")
    md_lines.append("## Object-Centric Multi-Target Diagnostics")
    for key, value in object_centric_multi_metrics.items():
        md_lines.append(f"- `{key}`: `{value:.6f}`")

    md_lines.append("")
    md_lines.append("## Object-Centric Binary Diagnostics")
    for key, value in object_centric_binary_metrics.items():
        md_lines.append(f"- `{key}`: `{value:.6f}`")

    md_lines.append("")
    md_lines.append("## Object-Centric Camera-Pose Diagnostics")
    for key, value in object_centric_camera_pose_metrics.items():
        md_lines.append(f"- `{key}`: `{value:.6f}`")

    md_lines.append("")
    md_lines.append("## Inferences")
    for note in inferences:
        md_lines.append(f"- {note}")

    md_path = output_dir / "inference_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    return txt_path, md_path


def main() -> int:
    args = parse_args()
    results = load_task_results(args.input, args.task)
    metrics = overall_metrics(results)
    geometry_metrics = named_metrics(results, GEOMETRY_METRICS)
    family_cases = family_case_metrics(results)
    object_centric_single_metrics = named_metrics(results, OBJECT_CENTRIC_SINGLE_METRICS)
    object_centric_multi_metrics = named_metrics(results, OBJECT_CENTRIC_MULTI_METRICS)
    object_centric_binary_metrics = named_metrics(results, OBJECT_CENTRIC_BINARY_METRICS)
    object_centric_camera_pose_metrics = named_metrics(results, OBJECT_CENTRIC_CAMERA_POSE_METRICS)
    output_dir = ensure_output_dir(args.input, args.output_dir)

    outputs = [
        plot_overall_scores(metrics, output_dir),
        plot_geometry_metrics(geometry_metrics, output_dir),
        plot_overall_case_split(metrics, output_dir),
        plot_overall_case_pie(metrics, output_dir),
        plot_family_case_splits(family_cases, output_dir),
        plot_family_case_pies(family_cases, output_dir),
        plot_family_case_grouped_bars(family_cases, output_dir),
        plot_family_summary_bars(metrics, family_cases, output_dir),
        plot_family_correctness(family_cases, output_dir),
        plot_object_centric_single_metrics(object_centric_single_metrics, output_dir),
        plot_object_centric_multi_metrics(object_centric_multi_metrics, output_dir),
        plot_object_centric_sign_comparison(object_centric_single_metrics, object_centric_multi_metrics, output_dir),
        plot_object_centric_camera_comparison(object_centric_single_metrics, object_centric_multi_metrics, output_dir),
        plot_object_centric_multi_selection_metrics(object_centric_multi_metrics, output_dir),
        plot_object_centric_multi_selection_pie(object_centric_multi_metrics, output_dir),
        plot_metric_block(
            object_centric_binary_metrics,
            "Object-Centric Binary Diagnostics",
            output_dir / "object_centric_binary_metrics.png",
        ),
        plot_metric_block(
            object_centric_camera_pose_metrics,
            "Object-Centric Camera-Pose Diagnostics",
            output_dir / "object_centric_camera_pose_metrics.png",
        ),
    ]
    report_txt, report_md = write_inference_report(
        metrics,
        family_cases,
        object_centric_single_metrics,
        object_centric_multi_metrics,
        object_centric_binary_metrics,
        object_centric_camera_pose_metrics,
        output_dir,
        args.input,
    )
    outputs.extend([report_txt, report_md])

    print("Generated analysis artifacts:")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
