#!/usr/bin/env python3
"""Build a prompt/answer catalog for the spatial experiments in this repository.

The examples are read from real lmms-eval submission records.  This keeps the
catalog synchronized with the prompts that were actually evaluated instead of
maintaining a second set of hand-written, potentially stale examples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO = Path("/home/ramanathan/VLM/lmms-eval")
DEFAULT_OUTPUT = DEFAULT_REPO / "outputs/EXPERIMENTS_README.md"

GT_HELP = {
    0: ("Baseline", "The original image and question, with no extra ground-truth aid."),
    1: ("Reference bbox", "Boxes the reference object to isolate reference localization."),
    2: ("Answer bbox", "Boxes the correct surrounding object to test whether target localization is the bottleneck."),
    3: ("All-object bboxes", "Boxes every relevant object so localization is supplied but the relation must still be inferred."),
    4: ("Reference-front arrow", "Adds one arrow showing the reference object's annotated front orientation."),
    5: ("Plain-text ground truth", "Supplies privileged scene information in text, testing whether the model can use an explicit textual oracle."),
    6: ("Long four-axis arrows", "Draws long left, right, front, and behind arrows on the reference object."),
    7: ("Long arrows plus perspective explanation", "Adds the four long arrows and explicitly explains object-relative perspective in the prompt."),
    8: ("Perspective explanation only", "Explains object-relative perspective without changing the image."),
    9: ("Reference-versus-camera example", "Gives a worked contrast between reference-frame and camera-frame answers."),
    10: ("Worked contrast plus arrows", "Combines the reference-versus-camera example with four image arrows."),
    11: ("Named top-down map", "Adds an abstract top-down map whose colored targets are labeled with object names."),
    12: ("Response-format scaffold", "Adds an explicit answer-format scaffold while leaving spatial evidence unchanged."),
    13: ("Neutral worked example", "Adds a worked example intended to teach the operation without revealing this sample's answer."),
    14: ("Color-only top-down map", "Uses abstract colored circles and a text legend instead of directly writing object names on the map."),
    36: ("Short four-axis arrows", "Ablates arrow length by shortening all four semantic arrows from GT_HELP 6."),
    37: ("Short front/left arrows", "Keeps only short front and left arrows, testing sparse-axis scaffolding."),
}

COMPONENTS = {
    "bbox_prediction": "Produce an object's normalized xyxy box from the plain image; scored primarily by IoU.",
    "bbox_naming": "Name the object inside a supplied magenta box; tests whether a VLM can consume a bbox overlay.",
    "facing_direction": "Classify the reference object's visible front into an eight-way image direction.",
    "front_arrow": "Predict a two-point image arrow for the reference object's front; scored by direction cosine and start-point error.",
    "front_arrow_reading": "Read a supplied ground-truth front arrow as an eight-way image direction; tests utilization separately from generation.",
    "left_arrow": "Predict a two-point image arrow for the reference object's left; exposes handedness errors separately from front estimation.",
    "left_arrow_reading": "Read a supplied ground-truth left arrow as an eight-way image direction; tests utilization separately from generation.",
    "symbol_to_object": "Map an abstract letter marker in the image to its underlying object name.",
    "object_to_symbol": "Perform the inverse grounding operation: map an object name to its abstract marker.",
    "long_arrow_to_symbol": "Use long semantic arrows to select the abstract symbol at a requested object-relative direction.",
    "short_arrow_to_symbol": "Repeat arrow-to-symbol grounding with the shorter GT_HELP-36 arrow geometry.",
    "vector_to_direction": "Convert a canonical {front, up, right} vector into a direction word without visual perception.",
    "direction_to_vector": "Convert a direction word into its canonical {front, up, right} vector.",
    "projected_axes_prediction": "Generate the reference object's projected front/up/right 2D basis for comparison with text and overlay utilization.",
    "text_axes_direction": "Use numerically supplied projected reference axes to answer a scene relation.",
    "overlay_axes_direction": "Use front/up/right image overlays to answer the same kind of scene relation.",
}

INVERSE = {
    "full_map_inversion": "Recover all four spatial edges at once, comparing relation-keyed and object-keyed JSON maps under multiple visual cues.",
    "arrow_length_sweep": "Vary arrow length while holding arrow semantics fixed to distinguish axis interpretation from visual pointer-following.",
    "map_ablation": "Remove or transform top-down-map components to locate which labels, colors, heading, and rotation make the inverse lookup usable.",
    "option_permutation": "Repeat semantic questions under cyclic option orders to measure letter bias and semantic consistency.",
    "binary_axis": "Restrict each question to left/right or front/behind, separating axis choice from polarity choice.",
    "oracle_ladder": "Reveal successively later pipeline stages to locate localization, orientation, mapping, retrieval, and output bottlenecks.",
}

DIRECTION_VECTOR = {
    "cross_output_consistency": "Ask for direction-only, vector-only, and combined outputs for matched pairs; tests whether representations agree.",
    "output_format_control": "Hold the target fixed while changing named JSON, list, text, signs, and prototype response formats.",
    "conversion_oracle": "Remove vision and directly test vector-to-word and word-to-canonical-vector conversion.",
    "basis_oracle_ladder": "Supply progressively stronger boxes, axes, basis, displacement, gold-vector, and gold-direction oracles.",
    "component_decomposition": "Request individual signs, dominant axis, or horizontal direction to separate axis and polarity failures.",
    "angular_boundary": "Bin examples by horizontal decision margin to test whether errors concentrate near direction boundaries.",
    "arrow_vector_grounding": "Progress from RGB to labeled or color-coded axes and measure whether overlays repair object-frame transforms.",
    "rotation_equivariance": "Rotate the displayed Kubric raster while holding the physical object frame fixed; measures output invariance.",
    "depth_oracle": "Progressively reveal ScanNet boxes, depths, centroids, and displacement to isolate real-scene geometry failures.",
}

GOLD_FIELDS = (
    "gt", "gt_mapping", "gold_answer", "gold_target", "gold_option",
    "gold_option_letter", "target", "gt_direction", "gt_vector", "gt_bbox",
    "gt_arrow_start", "gt_arrow_end", "gt_arrow_direction", "relation",
)
PREDICTION_FIELDS = (
    "prediction", "prediction_text", "parsed_prediction", "parsed_answer",
    "predicted_bbox", "predicted_direction", "predicted_vector",
    "predicted_option_letter", "selected_answer", "selected_direction",
)
SCORE_FIELDS = (
    "accuracy", "raw_accuracy", "answer_accuracy", "format_accuracy",
    "direction_accuracy", "vector_direction_accuracy", "component_accuracy",
    "mapping_edge_accuracy", "mapping_exact_accuracy", "bbox_iou", "iou",
    "arrow_cosine", "vector_cosine", "horizontal_cosine", "parse_success",
    "direction_parse_success", "vector_parse_success", "combined_parse_success",
)


@dataclass(frozen=True)
class Submission:
    family: str
    title: str
    description: str
    path: Path


@dataclass(frozen=True)
class ResultDashboard:
    title: str
    summary: str
    images: tuple[str, ...]
    interpretation: str
    main_result: str


RESULT_DASHBOARDS = (
    ResultDashboard(
        "COMFORT GT_HELP — Qwen3-VL-4B",
        "comforter_comfort_direction_object_gt_help_all_variants_debug/summary/summary.md",
        (
            "comforter_comfort_direction_object_gt_help_all_variants_debug/summary/comfort_direction_object_gt_help_experiment_summary_accuracy.png",
            "comforter_comfort_direction_object_gt_help_all_variants_debug/summary/comfort_direction_object_gt_help_experiment_summary_paired_outcomes_100pct.png",
            "comforter_comfort_direction_object_gt_help_all_variants_debug/summary/comfort_direction_object_gt_help_experiment_summary_transitions_vs_0.png",
        ),
        "Compare each help tier with mode 0, then inspect whether gains come from object answers, direction answers, or paired recovery.",
        "Baseline accuracy is 14.3%, with object answers outperforming direction answers by 16.8 points. Long four-axis arrows raise accuracy to 82.5%, while worked contrast plus arrows reaches 99.5%.",
    ),
    ResultDashboard(
        "COMFORT GT_HELP — Qwen3-VL-8B",
        "comforter_comfort_direction_object_gt_help_all_variants_debug_8/summary/summary.md",
        (
            "comforter_comfort_direction_object_gt_help_all_variants_debug_8/summary/comfort_direction_object_gt_help_experiment_summary_accuracy.png",
            "comforter_comfort_direction_object_gt_help_all_variants_debug_8/summary/comfort_direction_object_gt_help_experiment_summary_paired_outcomes_100pct.png",
            "comforter_comfort_direction_object_gt_help_all_variants_debug_8/summary/comfort_direction_object_gt_help_experiment_summary_transitions_vs_0.png",
        ),
        "Use the 4B/8B comparison to test whether a scaffold effect is stable with scale rather than relying on a single checkpoint.",
        "Baseline accuracy is 16.1%, with a 21.9-point object-answer advantage. Long four-axis arrows reach 91.2%, and worked contrast plus arrows reaches 99.4%, reproducing the scaffold effect at 8B scale.",
    ),
    ResultDashboard(
        "COMFORT component diagnostics — Qwen3-VL-4B",
        "comfort_gt_help_components_4/analysis/summary.md",
        (
            "comfort_gt_help_components_4/analysis/raw_accuracy.png",
            "comfort_gt_help_components_4/analysis/primary_scores.png",
            "comfort_gt_help_components_4/analysis/representation_comparisons.png",
            "comfort_gt_help_components_4/analysis/generation_vs_utilization.png",
        ),
        "These results determine whether failures originate in bbox use, orientation, symbol grounding, or vector semantics.",
        "BBox naming is 97.6% and symbol mapping is roughly 78–83%, but facing-direction accuracy is 2.6% and the front-arrow cosine is negative. Long arrows outperform short arrows (83.2% versus 61.0%).",
    ),
    ResultDashboard(
        "COMFORT component diagnostics — Qwen3-VL-8B",
        "comfort_gt_help_components_8/analysis/summary.md",
        (
            "comfort_gt_help_components_8/analysis/raw_accuracy.png",
            "comfort_gt_help_components_8/analysis/primary_scores.png",
            "comfort_gt_help_components_8/analysis/representation_comparisons.png",
            "comfort_gt_help_components_8/analysis/generation_vs_utilization.png",
        ),
        "Compare against the 4B panel by capability, not only overall score, to identify scaling-sensitive primitives.",
        "BBox naming reaches 98.1% and vector-to-direction conversion reaches 100%, while facing-direction accuracy is 0% and both front/left arrow cosines are strongly negative. Long arrows again outperform short arrows (77.5% versus 51.6%).",
    ),
    ResultDashboard(
        "COMFORT inverse diagnostics",
        "comfort_inverse_diagnostics_8_analysis/summary.md",
        (
            "comfort_inverse_diagnostics_8_analysis/full_map_inversion.png",
            "comfort_inverse_diagnostics_8_analysis/arrow_length_accuracy.png",
            "comfort_inverse_diagnostics_8_analysis/map_ablation_accuracy.png",
            "comfort_inverse_diagnostics_8_analysis/option_position_accuracy.png",
            "comfort_inverse_diagnostics_8_analysis/option_permutation_stability.png",
        ),
        "The most diagnostic contrasts are object versus direction within a condition, and the point at which increasing arrow length closes that gap.",
        "The object advantage falls from 39.9 points at 0.25× arrow length to −1.3 points at 1.15×. A fully labeled map makes both answer formats 100%, showing that the asymmetry is strongly scaffold-dependent.",
    ),
    ResultDashboard(
        "Cross-dataset direction/vector diagnostics",
        "direction_vector_diagnostics_analysis_8/summary.md",
        (
            "direction_vector_diagnostics_analysis_8/cross_dataset_cross_output_consistency.png",
            "direction_vector_diagnostics_analysis_8/cross_dataset_output_format_control.png",
            "direction_vector_diagnostics_analysis_8/cross_dataset_conversion_oracle.png",
            "direction_vector_diagnostics_analysis_8/cross_dataset_basis_oracle_ladder.png",
            "direction_vector_diagnostics_analysis_8/cross_dataset_component_decomposition.png",
            "direction_vector_diagnostics_analysis_8/cross_dataset_arrow_vector_grounding.png",
            "direction_vector_diagnostics_analysis_8/kubric_kubric_rotation_equivariance.png",
            "direction_vector_diagnostics_analysis_8/kubric_angular_boundary.png",
        ),
        "Read categorical direction, vector-derived direction, cosine, and parse success together; a vector gap alone does not distinguish geometry from serialization.",
        "On matched cross-output trials, direction-only accuracy exceeds vector-derived direction on COMFORT (7.5% versus 1.5%) and Kubric (16.5% versus 3.6%). Prototype choices and JSON key order materially change performance, implicating the output interface as well as geometry.",
    ),
    ResultDashboard(
        "COMFORT object/camera basis",
        "comfort_multi_3d_basis_combined_analysis/summary.md",
        (
            "comfort_multi_3d_basis_combined_analysis/metric_summary.png",
            "comfort_multi_3d_basis_combined_analysis/primary_outcomes_100pct.png",
            "comfort_multi_3d_basis_combined_analysis/combined_outcomes_100pct.png",
            "comfort_multi_3d_basis_combined_analysis/object_vs_direction_accuracy.png",
            "comfort_multi_3d_basis_combined_analysis/object_direction_outcomes_100pct.png",
        ),
        "The camera/object frame contrast is essential: the observed object-answer advantage is not frame invariant.",
        "The paired camera-frame task favors direction answers by 14.8 points, whereas the object-frame task favors object answers by 16.7 points. Camera-frame direction accuracy is 61.4%, compared with 42.4% from vector-derived directions.",
    ),
    ResultDashboard(
        "ScanNet object/camera basis",
        "scannet_basis_all_8/scannet_basis_analysis/summary.md",
        (
            "scannet_basis_all_8/scannet_basis_analysis/raw_accuracy.png",
            "scannet_basis_all_8/scannet_basis_analysis/metric_summary.png",
            "scannet_basis_all_8/scannet_basis_analysis/camera_vs_object_frame_accuracy.png",
            "scannet_basis_all_8/scannet_basis_analysis/primary_outcomes_100pct.png",
            "scannet_basis_all_8/scannet_basis_analysis/combined_outcomes_100pct.png",
            "scannet_basis_all_8/scannet_basis_analysis/object_vs_direction_accuracy.png",
            "scannet_basis_all_8/scannet_basis_analysis/object_direction_outcomes_100pct.png",
        ),
        "Use the real-scene camera/object frame contrast to test whether the representation effects found on COMFORT transfer to ScanNet.",
        "Discrete direction exceeds vector-derived direction in both the camera frame (50.8% versus 23.2%) and object-facing-camera frame (14.5% versus 10.1%). Paired answers are nearly even in the camera frame (41.8% object versus 43.4% direction), but object answers lead by 10.5 points in the object-facing-camera frame.",
    ),
    ResultDashboard(
        "COMFORT paired object/direction baseline",
        "comfort_direction_object_0/analysis/summary.md",
        (
            "comfort_direction_object_0/analysis/comfort_paired_outcomes.png",
            "comfort_direction_object_0/analysis/direction_distributions/direction_gt_vs_predicted_direction.png",
            "comfort_direction_object_0/analysis/direction_distributions/object_gt_vs_predicted_object_direction.png",
        ),
        "Use paired transitions and per-relation distributions to distinguish inverse-query asymmetry from a direction-label prior.",
        "Across 8,000 matched pairs, overall accuracy is 14.3%: object answers reach 22.8% versus 5.9% for direction answers, a 16.9-point gap.",
    ),
    ResultDashboard(
        "Kubric paired object/direction",
        "kubric_movi_a_direction_object_clean_better_sample_0/analysis/summary.md",
        (
            "kubric_movi_a_direction_object_clean_better_sample_0/analysis/kubric_paired_outcomes.png",
            "kubric_movi_a_direction_object_clean_better_sample_0/analysis/direction_gt_vs_selected_direction.png",
            "kubric_movi_a_direction_object_clean_better_sample_0/analysis/object_gt_vs_selected_object_direction.png",
        ),
        "Interpret native/inverse rows by their logged answer format because the native format differs between single- and multi-target source families.",
        "Object answers outperform direction answers in both Kubric families: 30.0% versus 5.0% for single-target examples and 44.6% versus 13.7% for multi-target examples.",
    ),
    ResultDashboard(
        "COMFORT bbox localization",
        "comfort_bbox/evaluation/submissions/bbox_visualizations/summary.md",
        ("comfort_bbox/evaluation/submissions/bbox_visualizations/scene_000000.png",),
        "The table reports localization quality; the representative overlay shows prediction and ground truth in the same scene.",
        "Across 2,500 objects in 500 scenes, 96.4% of predictions parse successfully and mean IoU is 55.7%.",
    ),
    ResultDashboard(
        "COMFORT reference/addressee viewpoint",
        "comfort_reference_orientation_viewpoint_0/analysis/summary.md",
        (
            "comfort_reference_orientation_viewpoint_0/analysis/metric_summary.png",
            "comfort_reference_orientation_viewpoint_0/analysis/outcome_split.png",
            "comfort_reference_orientation_viewpoint_0/analysis/question_type_accuracy.png",
        ),
        "Compare answer correctness with relation-axis correctness to measure answer-with-wrong-geometry cases.",
        "Answer accuracy is 11.3%, relation-axis accuracy is 3.6%, and mean vector cosine is approximately zero. Answer-correct/direction-wrong cases account for 8.9%, while 87.5% are wrong on both.",
    ),
    ResultDashboard(
        "COMFORT original viewpoint",
        "comfort_viewpoint_0/analysis/summary.md",
        (
            "comfort_viewpoint_0/analysis/metric_summary.png",
            "comfort_viewpoint_0/analysis/outcome_split.png",
            "comfort_viewpoint_0/analysis/question_type_accuracy.png",
        ),
        "This panel separates camera, reference, and addressee question types and should be compared with the filtered reference-orientation task.",
        "Overall answer accuracy is 27.2% and relation-axis accuracy is 28.3%. Camera-viewpoint questions reach 55.8%, far above reference-oriented (8.0%) and addressee-oriented (10.8%) questions.",
    ),
    ResultDashboard(
        "Kubric object-centric viewpoint",
        "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/summary.md",
        (
            "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/overall_metrics.png",
            "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/overall_case_split.png",
            "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/family_accuracy_summary.png",
            "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/geometry_metrics.png",
            "kubric_movi_a_object_centric_looking_back_better_sample_0/analysis/object_centric_camera_comparison.png",
        ),
        "Use the family split before the overall number because task families have different answer spaces and geometric requirements.",
        "Overall answer accuracy is 22.0%, axis-sign accuracy is 12.8%, and mean vector cosine is −0.257. Both answer and direction are wrong on 69.7% of examples.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a README with explanations and real prompt/answer examples."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-examples-per-submission", type=int, default=32,
        help="Safety cap after splitting a submission by experimental condition.",
    )
    parser.add_argument("--max-text", type=int, default=6000)
    return parser.parse_args()


def load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        records = payload.get("records", payload.get("samples", []))
        if isinstance(records, list):
            return payload, [row for row in records if isinstance(row, dict)]
    raise ValueError("submission is not a record list or a mapping with records/samples")


def clean_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_qwen3_vl_experiments", "_openai", "_experiments"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def first_match(path: Path, patterns: Iterable[str]) -> str | None:
    text = clean_stem(path)
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def describe_basis(stem: str) -> str:
    frame = "camera frame" if "camera_basis" in stem else "reference-object frame"
    if "object_direction" in stem:
        output = "paired answer-with-object and answer-with-direction questions"
    elif stem.endswith("_direction"):
        output = "one dominant horizontal direction word"
    elif stem.endswith("_vector"):
        output = "a continuous {front, up, right} vector"
    else:
        output = "both a direction word and continuous vector"
    return f"Predicts {output} in the {frame}."


def discover(repo: Path) -> tuple[list[Submission], list[str]]:
    outputs = repo / "outputs"
    submissions: list[Submission] = []
    missing: list[str] = []

    def add(path: Path, family: str, title: str, description: str) -> None:
        if path.is_file():
            submissions.append(Submission(family, title, description, path))
        else:
            missing.append(f"{family}: `{path.relative_to(repo)}`")

    # Prefer the 8B GT_HELP run as the canonical prompt source; the experiment
    # design is identical in the older un-suffixed run.
    gt_root = outputs / "comforter_comfort_direction_object_gt_help_all_variants_debug_8"
    for mode, (name, description) in GT_HELP.items():
        matches = sorted((gt_root / f"gt_help_{mode}" / "evaluation/submissions").glob("*.json"))
        if matches:
            add(matches[0], "COMFORT numbered GT_HELP", f"GT_HELP {mode}: {name}", description)
        else:
            missing.append(f"COMFORT numbered GT_HELP: mode {mode}")

    component_root = outputs / "comfort_gt_help_components_8/submissions"
    for key, description in COMPONENTS.items():
        matches = sorted(component_root.glob(f"comfort_gt_component_{key}_*.json"))
        if matches:
            add(matches[0], "COMFORT GT_HELP components", humanize(key), description)
        else:
            missing.append(f"COMFORT GT_HELP components: {key}")

    for key, description in INVERSE.items():
        matches = sorted((outputs / f"comfort_{key}_8/submissions").glob("*.json"))
        if matches:
            add(matches[0], "COMFORT inverse diagnostics", humanize(key), description)
        else:
            missing.append(f"COMFORT inverse diagnostics: {key}")

    for dataset in ("comfort", "scannet", "kubric"):
        root = outputs / f"direction_vector_diagnostics_{dataset}_8/submissions"
        paths = sorted(root.glob("*.json"))
        if not paths:
            missing.append(f"Direction/vector diagnostics: no {dataset} submissions")
            continue
        for path in paths:
            key = first_match(path, DIRECTION_VECTOR)
            description = DIRECTION_VECTOR.get(key or "", "A direction/vector diagnostic condition.")
            title = f"{dataset.title()}: {humanize(key or clean_stem(path))}"
            add(path, "Cross-dataset direction/vector diagnostics", title, description)

    basis_roots = (
        outputs / "comfort_multi_3d_object_basis_0/submissions",
        outputs / "comfort_multi_3d_camera_basis_0/submissions",
        outputs / "comfort_multi_3d_basis_object_direction_0/submissions",
        outputs / "scannet_basis_all_8/submissions",
        outputs / "scannet_basis_object_direction_8/submissions",
    )
    for root in basis_roots:
        for path in sorted(root.glob("*.json")):
            stem = clean_stem(path)
            family = "ScanNet basis tasks" if stem.startswith("scannet") else "COMFORT basis tasks"
            add(path, family, humanize(stem), describe_basis(stem))
    if not any(item.family == "ScanNet basis tasks" for item in submissions):
        missing.append("ScanNet basis tasks: no submissions")

    singleton_specs = (
        (
            outputs / "comfort_direction_object_0/submissions/comfort_direction_object_qwen3_vl_experiments.json",
            "Paired object-versus-direction tasks", "COMFORT paired object/direction",
            "Asks inverse multiple-choice questions on the same scene: name the object at a direction, or name the direction of an object.",
        ),
        (
            outputs / "kubric_movi_a_direction_object_clean_better_sample_0/submissions/kubric_movi_a_direction_object_qwen3_vl_experiments.json",
            "Paired object-versus-direction tasks", "Kubric paired object/direction",
            "Tests the same answer-with-object versus answer-with-direction distinction on clean object-centric Kubric samples.",
        ),
        (
            outputs / "comfort_bbox/evaluation/submissions/comfort_multi_3d_bbox_prediction_qwen3_vl_experiments.json",
            "Localization", "COMFORT bbox prediction",
            "Asks for one normalized xyxy bounding box at a time and evaluates IoU against the scene annotation.",
        ),
        (
            outputs / "comfort_reference_orientation_viewpoint_0/submissions/comfort_viewpoint_qwen3_vl_experiments.json",
            "Viewpoint tasks", "COMFORT reference/addressee orientation viewpoint",
            "Keeps examples whose relations use reference-object or addressee ground-truth orientation and evaluates answer plus inferred direction.",
        ),
        (
            outputs / "comfort_viewpoint_0/submissions/comfort_viewpoint_qwen3_vl_experiments.json",
            "Viewpoint tasks", "COMFORT original viewpoint",
            "Measures spatial answers and direction inference under the original COMFORT viewpoint filtering.",
        ),
        (
            outputs / "kubric_movi_a_object_centric_looking_back_better_sample_0/submissions/kubric_movi_a_viewpoint_qwen3_vl_experiments.json",
            "Viewpoint tasks", "Kubric object-centric looking-back viewpoint",
            "Uses an anchor-centered frame whose forward axis looks back toward the camera, testing object-centric coordinate transformation.",
        ),
    )
    for spec in singleton_specs:
        add(*spec)
    return submissions, missing


def scalar_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (str, int, float, bool)) and value is not None:
            values.append(str(value))
    return sorted(set(values))


def condition_dimensions(rows: list[dict[str, Any]]) -> list[str]:
    dimensions: list[str] = []
    for field in ("experiment_condition", "answer_format", "mapping_format", "task_family"):
        values = scalar_values(rows, field)
        if 1 < len(values) <= 16:
            dimensions.append(field)
    # `variant` usually duplicates answer_format. Use it only when no more
    # informative condition dimension exists.
    if not dimensions:
        values = scalar_values(rows, "variant")
        if 1 < len(values) <= 16:
            dimensions.append("variant")
    # Avoid accidental combinatorial explosion from redundant dimensions.
    while dimensions:
        keys = {tuple(str(row.get(field, "")) for field in dimensions) for row in rows}
        if len(keys) <= 32:
            break
        dimensions.pop()
    return dimensions


def representative_rows(rows: list[dict[str, Any]], cap: int) -> list[tuple[str, dict[str, Any], int]]:
    dimensions = condition_dimensions(rows)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "unspecified")) for field in dimensions)
        grouped.setdefault(key, []).append(row)
    results = []
    for key, group in sorted(grouped.items()):
        # Prefer a parsed record, while preserving the first evaluated sample
        # when no generic parse flag is available.
        sample = next((row for row in group if row.get("parse_success") is True), group[0])
        label = ", ".join(f"{humanize(field)} = {value}" for field, value in zip(dimensions, key))
        results.append((label or "Representative sample", sample, len(group)))
    return results[:cap]


def picked(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: row[field] for field in fields if field in row and row[field] is not None}


def clip(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    text = text.replace("```", "` ` `")
    if len(text) > limit:
        text = text[: limit - 32].rstrip() + "\n... [truncated by catalog tool]"
    return text


def fenced(lines: list[str], label: str, value: Any, limit: int, language: str = "text") -> None:
    lines.extend((f"**{label}**", "", f"```{language}", clip(value, limit), "```", ""))


def task_name(metadata: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> str:
    for source in (metadata, rows[0] if rows else {}):
        if source.get("task"):
            return str(source["task"])
        if source.get("task_name"):
            return str(source["task_name"])
    return clean_stem(path)


def _markdown_path(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output.parent)).as_posix()


def _demote_markdown_headings(text: str) -> str:
    """Keep an embedded summary inside its dashboard subsection."""
    rendered = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            level = min(6, len(match.group(1)) + 3)
            rendered.append(f"{'#' * level} {match.group(2)}")
        else:
            rendered.append(line)
    return "\n".join(rendered)


def append_result_dashboards(lines: list[str], repo: Path, output: Path) -> tuple[int, int]:
    outputs_root = repo / "outputs"
    available = []
    for dashboard in RESULT_DASHBOARDS:
        summary = outputs_root / dashboard.summary
        if not summary.is_file():
            continue
        images = [outputs_root / item for item in dashboard.images]
        available.append((dashboard, summary, [path for path in images if path.is_file()]))

    lines.extend((
        "## Current results dashboard",
        "",
        "The tables below are copied from the current generated `summary.md` files. The images are linked directly to the corresponding analysis artifacts, so rerunning the analysis and this catalog updates both the numeric and visual evidence. Expand a section to inspect its complete current tables.",
        "",
        "| Results | Summary | Inline plots |",
        "|---|---|---:|",
    ))
    for dashboard, summary, images in available:
        summary_link = _markdown_path(summary, output)
        lines.append(f"| {dashboard.title} | [open summary]({summary_link}) | {len(images)} |")
    lines.append("")

    image_count = 0
    for dashboard, summary, images in available:
        summary_link = _markdown_path(summary, output)
        lines.extend((
            f"### {dashboard.title}",
            "",
            dashboard.interpretation,
            "",
            f"**Current main result:** {dashboard.main_result}",
            "",
            f"Source tables: [{summary.name}]({summary_link})",
            "",
        ))
        for image_path in images:
            image_link = _markdown_path(image_path, output)
            label = humanize(image_path.stem)
            alt = f"{dashboard.title}: {label}".replace('"', "&quot;")
            lines.extend((f'<img src="{image_link}" alt="{alt}" width="720" loading="lazy">', ""))
            image_count += 1
        lines.extend((
            "<details>",
            "<summary>Show complete current tables and textual summary</summary>",
            "",
            _demote_markdown_headings(summary.read_text(encoding="utf-8").strip()),
            "",
            "</details>",
            "",
        ))
    return len(available), image_count


def write_readme(
    repo: Path,
    output: Path,
    submissions: list[Submission],
    missing: list[str],
    max_examples: int,
    max_text: int,
) -> tuple[int, int, list[str], int, int]:
    loaded: list[tuple[Submission, dict[str, Any], list[dict[str, Any]]]] = []
    errors: list[str] = []
    for submission in submissions:
        try:
            metadata, rows = load_records(submission.path)
            if not rows:
                raise ValueError("no records")
            loaded.append((submission, metadata, rows))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"`{submission.path.relative_to(repo)}`: {error}")

    family_counts = Counter(item.family for item, _, _ in loaded)
    lines = [
        "# Spatial experiment catalog",
        "",
        f"Generated on {datetime.now().astimezone().isoformat(timespec='seconds')} from the submission files under `{repo / 'outputs'}`.",
        "",
        "This document explains every experiment family developed in this workflow and shows prompts, expected answers, and model responses copied from the actual logged submissions. Examples are grouped by condition or answer format when a submission contains multiple interventions. They are evidence from one evaluated sample, not hand-written idealizations and not aggregate results.",
        "",
        "When the same task design was evaluated with both 4B and 8B checkpoints, the catalog uses the 8B submission as the canonical prompt source. Model-size replicas are separate result runs, not separate experiment definitions; their aggregate summaries remain in their respective output folders.",
        "",
        "## How to read and compare the experiments",
        "",
        "Use matched questions and condition deltas wherever possible. First check parse success, then the task's semantic accuracy or continuous score. Object-answer versus direction-answer gaps diagnose inverse retrieval; direction-word versus vector gaps diagnose representation/conversion; oracle ladders identify the earliest supplied stage that repairs performance. Cross-dataset raw scores also contain dataset difficulty, so COMFORT, ScanNet, and Kubric are best compared by intervention gains before absolute accuracy.",
        "",
        "The expected-answer blocks retain all useful gold fields present in the record. For multiple choice, `gold_option_letter` is the required emitted answer while `gold_answer`, `gold_target`, or `gold_option` identifies its semantics. The model-response block shows both raw and parsed fields when the logger provides them.",
        "",
        "Regenerate this file with:",
        "",
        "```bash",
        "python tools/build_conversation_experiments_readme.py",
        "```",
        "",
    ]
    dashboard_count, dashboard_images = append_result_dashboards(lines, repo, output)
    lines.extend([
        "## Experiment and example coverage",
        "",
        "| Family | Loaded submissions |",
        "|---|---:|",
    ])
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(("", f"Total: **{len(loaded)} submissions** and **{sum(len(rows) for _, _, rows in loaded):,} records**.", ""))

    if missing or errors:
        lines.extend((
            "### Results not represented by a real example",
            "",
            "These experiment definitions are retained in the task suite, but no readable submission was available when this README was generated:",
            "",
        ))
        lines.extend(f"- {item}" for item in sorted(set(missing + errors)))
        lines.append("")

    current_family = None
    example_count = 0
    for submission, metadata, rows in loaded:
        if submission.family != current_family:
            current_family = submission.family
            lines.extend((f"## {current_family}", ""))
            if current_family == "COMFORT numbered GT_HELP":
                lines.extend(("The numbered tiers vary the amount and carrier of privileged help while preserving the paired object-answer and direction-answer evaluation. Compare every tier against mode 0, and compare modes that differ in one feature (6 vs 36 for length; 36 vs 37 for axis count; 11 vs 14 for named vs symbolic maps).", ""))
            elif current_family == "COMFORT GT_HELP components":
                lines.extend(("These tasks validate that the VLM can consume or produce each primitive used by the composite GT_HELP overlays. A composite-task failure should not be attributed to spatial reasoning until its relevant primitive succeeds here.", ""))
            elif current_family == "COMFORT inverse diagnostics":
                lines.extend(("These experiments target cases where object answers and direction answers diverge. Pair records by their logged identifiers and compare the two answer formats inside each intervention condition.", ""))
            elif current_family == "Cross-dataset direction/vector diagnostics":
                lines.extend(("The same conceptual probes are run across COMFORT, ScanNet, and Kubric. Their `{front, up, right}` frames are dataset-specific, so use matched within-dataset gains before comparing absolute levels.", ""))

        rel = submission.path.relative_to(repo)
        task = task_name(metadata, rows, submission.path)
        conditions = representative_rows(rows, max_examples)
        lines.extend((
            f"### {submission.title}",
            "",
            submission.description,
            "",
            f"Task: `{task}`  ",
            f"Submission: [{rel.name}](../{rel.as_posix()})  ",
            f"Records: **{len(rows):,}**; example groups shown: **{len(conditions)}**.",
            "",
        ))
        for index, (condition, row, group_size) in enumerate(conditions, 1):
            label = condition if len(conditions) > 1 else "Representative logged example"
            lines.extend((f"#### Example {index}: {label}", "", f"This group contains {group_size:,} records.", ""))
            prompt = row.get("question_prompt", row.get("prompt", row.get("question", "[prompt field not logged]")))
            fenced(lines, "Prompt sent to the model", prompt, max_text)
            gold = picked(row, GOLD_FIELDS)
            if not gold:
                gold = {"expected": "[no recognized gold field in submission record]"}
            fenced(lines, "Expected answer / ground truth", gold, max_text, "json")
            prediction = picked(row, PREDICTION_FIELDS)
            if not prediction:
                prediction = {"response": "[no recognized prediction field in submission record]"}
            fenced(lines, "Model response", prediction, max_text, "json")
            scores = picked(row, SCORE_FIELDS)
            if scores:
                fenced(lines, "Recorded evaluation fields", scores, max_text, "json")
            example_count += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(loaded), example_count, errors, dashboard_count, dashboard_images


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    submissions, missing = discover(repo)
    loaded, examples, errors, dashboards, dashboard_images = write_readme(
        repo, output, submissions, missing,
        max(1, args.max_examples_per_submission), max(500, args.max_text),
    )
    print(f"Wrote {output}")
    print(f"Loaded {loaded} submissions and included {examples} real prompt/answer examples.")
    print(f"Embedded {dashboards} result dashboards with {dashboard_images} plots.")
    if missing:
        print(f"Documented {len(missing)} missing experiment outputs.")
    if errors:
        print(f"Skipped {len(errors)} unreadable submissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
