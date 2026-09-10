"""Camera-basis direction and vector tasks over prepared ScanNet frames."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks.comfort_multi_3d_object_basis import utils as basis
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


AXES = basis.AXES
EPSILON = basis.EPSILON
TASK_BY_FORMAT = {
    "direction": "scannet_camera_basis_direction",
    "vector": "scannet_camera_basis_vector",
    "combined": "scannet_camera_basis_combined",
}
OBJECT_TASK_BY_FORMAT = {
    "direction": "scannet_object_basis_direction",
    "vector": "scannet_object_basis_vector",
    "combined": "scannet_object_basis_combined",
}
PAIR_TASK_BY_FRAME = {
    "camera": "scannet_camera_basis_object_direction",
    "object_facing_camera": "scannet_object_basis_object_direction",
}


def _unit_camera_vector(reference: dict, target: dict) -> dict[str, float]:
    """Return target-minus-reference in ScanNet's (+X right,+Y front,+Z up) camera frame."""
    try:
        reference_position = [float(value) for value in reference["position_camera_xyz_m"]]
        target_position = [float(value) for value in target["position_camera_xyz_m"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Object lacks a valid camera-frame 3D centre") from error
    if len(reference_position) != 3 or len(target_position) != 3:
        raise ValueError("Camera-frame positions must be XYZ triples")
    right, front, up = basis._unit(basis._subtract(target_position, reference_position))
    return {"front": front, "up": up, "right": right}


def _object_facing_camera_vector(reference: dict, target: dict) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Express target-minus-anchor in an anchor frame looking back at camera.

    ScanNet does not annotate an object's semantic yaw. The valid object-centric
    frame available for every visible instance is therefore constructed from
    the anchor-to-camera direction and world up, exactly as the associated
    prompt states.
    """
    reference_position = [float(value) for value in reference["position_camera_xyz_m"]]
    target_position = [float(value) for value in target["position_camera_xyz_m"]]
    if len(reference_position) != 3 or len(target_position) != 3:
        raise ValueError("Camera-frame positions must be XYZ triples")
    # Camera is (0,0,0), so this is the horizontal anchor-to-camera vector.
    front = basis._unit([-reference_position[0], -reference_position[1], 0.0])
    up = [0.0, 0.0, 1.0]
    right = basis._unit(basis._cross(front, up))
    displacement = basis._unit(basis._subtract(target_position, reference_position))
    vector = {"front": basis._dot(displacement, front), "up": basis._dot(displacement, up), "right": basis._dot(displacement, right)}
    return vector, {"front": front, "up": up, "right": right}


def _process_docs(dataset: Dataset, prediction_format: str, coordinate_frame: str = "camera") -> Dataset:
    task_mapping = TASK_BY_FORMAT if coordinate_frame == "camera" else OBJECT_TASK_BY_FORMAT
    if prediction_format not in task_mapping:
        raise ValueError(f"Unknown ScanNet prediction format {prediction_format!r}")
    records, skipped = [], Counter()
    for source in dataset:
        view = dict(source)
        scene_id = str(view.get("scene_id", "")).strip()
        frame_id = view.get("frame_id")
        image_path = Path(str(view.get("image_path", "")))
        objects = view.get("objects") or []
        if not scene_id or frame_id is None or not image_path.is_file():
            skipped["invalid_view"] += 1
            continue
        if not isinstance(objects, list) or len(objects) < 2:
            skipped["fewer_than_two_objects"] += 1
            continue
        for reference in objects:
            for target in objects:
                if reference is target or reference.get("object_id") == target.get("object_id"):
                    continue
                try:
                    if coordinate_frame == "camera":
                        vector = _unit_camera_vector(reference, target)
                        evaluation_basis = {"front": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0], "right": [1.0, 0.0, 0.0]}
                    else:
                        vector, evaluation_basis = _object_facing_camera_vector(reference, target)
                    direction = basis.direction_from_vector(vector)
                except (TypeError, ValueError):
                    skipped["invalid_geometry"] += 1
                    continue
                if direction is None:
                    skipped["vertical_only_relation"] += 1
                    continue
                # The manifest retains ``display_name`` (e.g. ``chair #17``)
                # for traceability, but the strict builder guarantees labels
                # are unique within a retained frame. Questions should thus
                # use natural object names rather than exposing instance IDs.
                reference_name = str(reference.get("label", "")).strip()
                target_name = str(target.get("label", "")).strip()
                if not reference_name or not target_name:
                    skipped["missing_object_name"] += 1
                    continue
                pair_id = f"{scene_id}::frame{int(frame_id):06d}::{reference['object_id']}::{target['object_id']}"
                records.append({
                    "qid": f"{pair_id}::{coordinate_frame}::{prediction_format}",
                    "index": f"{pair_id}::{coordinate_frame}::{prediction_format}",
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "coordinate_frame": coordinate_frame,
                    "prediction_format": prediction_format,
                    "reference_object": reference_name,
                    "reference_object_id": str(reference["object_id"]),
                    "target_object": target_name,
                    "target_object_id": str(target["object_id"]),
                    "gt_direction": direction,
                    "gt_vector": vector,
                    "evaluation_basis_camera_frame": evaluation_basis,
                    "question": (f"From the camera's frame of reference, what is the 3D direction from the {reference_name} to the {target_name}?" if coordinate_frame == "camera" else f"From the {reference_name}'s frame while it looks back at the camera, what is the 3D direction from the {reference_name} to the {target_name}?"),
                    "img_path": str(image_path),
                    "image_path": str(image_path),
                })
    eval_logger.info(
        "ScanNet camera-basis {} task loaded {} pairs from {} views; skipped={}",
        prediction_format, len(records), len({(row['scene_id'], row['frame_id']) for row in records}), dict(skipped),
    )
    return Dataset.from_list(records)


def process_direction_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "direction")


def process_vector_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "vector")


def process_combined_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "combined")


def process_object_direction_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "direction", "object_facing_camera")


def process_object_vector_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "vector", "object_facing_camera")


def process_object_combined_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "combined", "object_facing_camera")


def _process_object_direction_docs(dataset: Dataset, coordinate_frame: str) -> Dataset:
    """Create balanced, matched answer-with-object/direction rows.

    Only five-object views are retained, giving four candidate objects for
    every reference and therefore the same four-way choice cardinality as the
    direction vocabulary. A fact is retained only when its direction occurs
    once among those four candidates, so the inverse object question has one
    valid answer. Finally, relations are downsampled deterministically to make
    left/right/front/back exactly balanced.
    """
    if coordinate_frame not in PAIR_TASK_BY_FRAME:
        raise ValueError(f"Unknown ScanNet paired coordinate frame {coordinate_frame!r}")
    base_docs = list(_process_docs(dataset, "direction", coordinate_frame))
    by_reference = defaultdict(list)
    for doc in base_docs:
        key = (str(doc["scene_id"]), int(doc["frame_id"]), str(doc["reference_object_id"]))
        by_reference[key].append(doc)

    eligible = []
    skipped = Counter()
    for reference_docs in by_reference.values():
        if len(reference_docs) != 4:
            skipped["not_four_candidates"] += len(reference_docs)
            continue
        direction_counts = Counter(str(doc["gt_direction"]) for doc in reference_docs)
        for doc in reference_docs:
            if direction_counts[str(doc["gt_direction"])] != 1:
                skipped["ambiguous_inverse_relation"] += 1
                continue
            eligible.append(doc)

    by_direction = defaultdict(list)
    for doc in eligible:
        by_direction[str(doc["gt_direction"])].append(doc)
    if set(by_direction) != set(basis.DIRECTIONS):
        raise ValueError(f"ScanNet paired rows do not cover all directions: {sorted(by_direction)}")
    per_direction = min(len(by_direction[direction]) for direction in basis.DIRECTIONS)
    balanced = []
    for direction in basis.DIRECTIONS:
        ordered = sorted(by_direction[direction], key=lambda row: str(row["pair_id"]))
        balanced.extend(ordered[:per_direction])
        skipped["relation_balance_downsample"] += len(ordered) - per_direction
    balanced.sort(key=lambda row: str(row["pair_id"]))

    records = []
    for source in balanced:
        reference_key = (str(source["scene_id"]), int(source["frame_id"]), str(source["reference_object_id"]))
        reference_docs = by_reference[reference_key]
        candidates = [str(doc["target_object"]) for doc in reference_docs]
        pair_id = f"{source['pair_id']}::{coordinate_frame}::object_direction"
        if coordinate_frame == "camera":
            perspective = "the camera's frame of reference"
        else:
            perspective = f"the {source['reference_object']}-centred frame while it looks back at the camera"
        shared = {
            **source,
            "prediction_format": "object_direction",
            "pair_id": pair_id,
            "candidate_objects": candidates,
        }
        records.extend(
            (
                {
                    **shared,
                    "qid": f"{pair_id}::direction",
                    "index": f"{pair_id}::direction",
                    "answer_format": "direction",
                    "gold_answer": source["gt_direction"],
                    "question": (
                        f"From {perspective}, where is the {source['target_object']} "
                        f"relative to the {source['reference_object']}?"
                    ),
                },
                {
                    **shared,
                    "qid": f"{pair_id}::object",
                    "index": f"{pair_id}::object",
                    "answer_format": "object",
                    "gold_answer": source["target_object"],
                    "question": (
                        f"From {perspective}, which candidate object is "
                        f"{source['gt_direction']} of the {source['reference_object']}?"
                    ),
                },
            )
        )
    eval_logger.info(
        "ScanNet {} object/direction task loaded {} balanced matched pairs "
        "({} rows, {} per relation); skipped={}",
        coordinate_frame,
        len(records) // 2,
        len(records),
        per_direction,
        dict(skipped),
    )
    return Dataset.from_list(records)


def process_camera_basis_object_direction_docs(dataset: Dataset) -> Dataset:
    return _process_object_direction_docs(dataset, "camera")


def process_object_basis_object_direction_docs(dataset: Dataset) -> Dataset:
    return _process_object_direction_docs(dataset, "object_facing_camera")


# Parsing, visual loading, targets, scoring, and scalar aggregations are shared
# with COMFORT because the output schema is deliberately identical.
doc_to_visual = basis.doc_to_visual
doc_to_target = basis.doc_to_target
aggregate_basis_direction_accuracy = basis.aggregate_basis_direction_accuracy
aggregate_basis_direction_parse_success = basis.aggregate_basis_direction_parse_success
aggregate_basis_vector_cosine = basis.aggregate_basis_vector_cosine
aggregate_basis_vector_l2_score = basis.aggregate_basis_vector_l2_score
aggregate_basis_vector_angle_30_accuracy = basis.aggregate_basis_vector_angle_30_accuracy
aggregate_basis_vector_dominant_direction_accuracy = basis.aggregate_basis_vector_dominant_direction_accuracy
aggregate_basis_front_sign_accuracy = basis.aggregate_basis_front_sign_accuracy
aggregate_basis_up_sign_accuracy = basis.aggregate_basis_up_sign_accuracy
aggregate_basis_right_sign_accuracy = basis.aggregate_basis_right_sign_accuracy
aggregate_basis_full_sign_accuracy = basis.aggregate_basis_full_sign_accuracy
aggregate_basis_vector_parse_success = basis.aggregate_basis_vector_parse_success
aggregate_basis_combined_parse_success = basis.aggregate_basis_combined_parse_success
aggregate_basis_combined_parse_failure = basis.aggregate_basis_combined_parse_failure
aggregate_basis_both_correct = basis.aggregate_basis_both_correct
aggregate_basis_direction_correct_vector_wrong = basis.aggregate_basis_direction_correct_vector_wrong
aggregate_basis_direction_wrong_vector_correct = basis.aggregate_basis_direction_wrong_vector_correct
aggregate_basis_both_wrong = basis.aggregate_basis_both_wrong
process_object_direction_results = basis.process_object_direction_results
aggregate_basis_format_accuracy = basis.aggregate_basis_format_accuracy
aggregate_basis_object_answer_accuracy = basis.aggregate_basis_object_answer_accuracy
aggregate_basis_direction_answer_accuracy = basis.aggregate_basis_direction_answer_accuracy
aggregate_basis_object_minus_direction = basis.aggregate_basis_object_minus_direction
aggregate_basis_format_switch_gain = basis.aggregate_basis_format_switch_gain
aggregate_basis_format_parse_success = basis.aggregate_basis_format_parse_success
aggregate_basis_object_parse_success = basis.aggregate_basis_object_parse_success
aggregate_basis_pair_direction_parse_success = basis.aggregate_basis_pair_direction_parse_success
aggregate_basis_object_correct_direction_wrong = basis.aggregate_basis_object_correct_direction_wrong
aggregate_basis_direction_correct_object_wrong = basis.aggregate_basis_direction_correct_object_wrong
aggregate_basis_pair_both_correct = basis.aggregate_basis_pair_both_correct
aggregate_basis_pair_both_wrong = basis.aggregate_basis_pair_both_wrong


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    entry = basis._score_entry(doc, prediction)
    metric_names = (
        "basis_direction_accuracy", "basis_direction_parse_success", "basis_vector_cosine", "basis_vector_l2_score",
        "basis_vector_angle_30_accuracy", "basis_vector_dominant_direction_accuracy", "basis_front_sign_accuracy",
        "basis_up_sign_accuracy", "basis_right_sign_accuracy", "basis_full_sign_accuracy", "basis_vector_parse_success",
        "basis_combined_parse_success", "basis_combined_parse_failure", "basis_both_correct",
        "basis_direction_correct_vector_wrong", "basis_direction_wrong_vector_correct", "basis_both_wrong",
    )
    output = {name: dict(entry) for name in metric_names}
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "target": doc_to_target(doc), "img_path": doc.get("img_path")}
    return output


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if doc.get("coordinate_frame") == "camera" and doc.get("prediction_format") != "object_direction":
        return basis.doc_to_text(doc, lmms_eval_specific_kwargs)
    kwargs = lmms_eval_specific_kwargs or {}
    prediction_format = doc["prediction_format"]
    if prediction_format == "object_direction":
        if doc.get("coordinate_frame") == "camera":
            frame_text = (
                "Use the camera frame: right is camera/image-right and front follows "
                "the camera viewing direction. Do not use an object's semantic facing direction."
            )
        else:
            frame_text = (
                f"Use the {doc['reference_object']}-centred anchor frame: imagine the reference "
                "looking back at the camera; front points toward the camera and its own right "
                "appears camera/image-left."
            )
        if doc["answer_format"] == "direction":
            output_rule = "Return exactly one lowercase direction word: left, right, front, or back."
        else:
            candidates = ", ".join(str(item) for item in doc["candidate_objects"])
            output_rule = f"Return exactly one object name from this four-object candidate list: {candidates}."
        return f"{kwargs.get('pre_prompt', '')}{doc['question']}\n{frame_text}\n{output_rule}{kwargs.get('post_prompt', '')}"
    frame_text = (
        f"Use the {doc['reference_object']}-centred anchor frame, not camera/image axes: the reference is looking back at the camera, so positive front points from it toward the camera, positive up is world up, and positive right is its own right (camera/image left)."
    )
    if prediction_format == "direction":
        output_rule = "Return exactly one lowercase direction word: left, right, front, or back. Use the dominant horizontal component and ignore up/down for this word."
    elif prediction_format == "vector":
        output_rule = 'Return only valid JSON with the target-minus-reference unit direction: {"front": <float>, "up": <float>, "right": <float>}. All three components must be present.'
    else:
        output_rule = 'Return only valid JSON containing both predictions: {"answer": "<left|right|front|back>", "relative_vector": {"front": <float>, "up": <float>, "right": <float>}}. The relative_vector is the target-minus-reference unit direction.'
    return f"{kwargs.get('pre_prompt', '')}{doc['question']}\n{frame_text}\n{output_rule}{kwargs.get('post_prompt', '')}"


def aggregate_results_for_submission(results, args):
    prediction_format = str(results[0].get("prediction_format", "unknown")) if results else "unknown"
    task_mapping = TASK_BY_FORMAT if (not results or results[0].get("coordinate_frame") == "camera") else OBJECT_TASK_BY_FORMAT
    task = task_mapping.get(prediction_format, "scannet_camera_basis")
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{task}_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "ScanNet v2 prepared camera frames", "task": task,
            "prediction_format": prediction_format, "coordinate_frame": ("camera (+X right, +Y front, +Z up)" if not results or results[0].get("coordinate_frame") == "camera" else "anchor facing camera (+Y front to camera, +Z world up, +X anchor right)"),
            "vector_definition": "unit target-minus-reference direction", "num_records": len(results), "records": results,
        }, handle, indent=2)
    eval_logger.info("ScanNet camera-basis {} records saved to {}.", prediction_format, path)


def aggregate_object_direction_submission(results, args):
    coordinate_frame = str(results[0].get("coordinate_frame", "camera")) if results else "camera"
    task = PAIR_TASK_BY_FRAME.get(coordinate_frame, "scannet_basis_object_direction")
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{task}_{model}.json", args)
    report = {
        "dataset": "ScanNet v2 prepared camera frames",
        "task": task,
        "coordinate_frame": coordinate_frame,
        "prediction_format": "object_direction",
        "candidate_count": 4,
        "relation_balanced": True,
        "num_records": len(results),
        "num_matched_pairs": len(basis._matched_format_pairs(results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("ScanNet {} object/direction records saved to {}.", coordinate_frame, path)
