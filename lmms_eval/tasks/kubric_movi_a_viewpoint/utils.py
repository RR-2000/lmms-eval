"""
Canonical viewpoint-estimation task for Kubric MOVi-A.

This task asks a model to produce a structured relative-position estimate that can
be compared directly against dataset ground truth in a consistent coordinate
standard across image-plane, camera-distance, and anchor-centric reasoning tasks.
"""

import json
import math
import re
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a.utils import (
    DIFFICULTIES,
    TASK_FAMILIES,
    _get_image_path,
    _get_object_record,
    _get_queried_object_names,
    _ground_truth_answer_text,
    doc_to_visual,
)
from lmms_eval.utils import sanitize_model_name

VIEWPOINT_TASK_FAMILIES = TASK_FAMILIES + [
    "object_centric_direction_binary",
    "object_centric_camera_pose",
]


AXIS_ALIASES = {
    "right": ("right", "x", "dx", "image_right", "anchor_right"),
    "up": ("up", "y", "dy", "image_up", "world_up", "anchor_up"),
    "front": ("front", "z", "dz", "depth", "camera_depth", "anchor_front"),
}

RELATION_TO_AXIS = {
    "left": "right",
    "right": "right",
    "above": "up",
    "below": "up",
    "higher": "up",
    "lower": "up",
    "closer": "front",
    "farther": "front",
    "front": "front",
    "behind": "front",
}

POSITIVE_RELATIONS = {"right", "above", "higher", "closer", "front"}
NEGATIVE_RELATIONS = {"left", "below", "lower", "farther", "behind"}


def _normalize_name(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _extract_json_blob(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _get_first_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _parse_relative_vector(payload: dict) -> dict[str, float]:
    vector = {"right": 0.0, "up": 0.0, "front": 0.0}
    if not isinstance(payload, dict):
        return vector

    raw_vector = payload.get("relative_vector", payload)
    if isinstance(raw_vector, list) and len(raw_vector) >= 3:
        return {
            "right": float(raw_vector[0]),
            "up": float(raw_vector[1]),
            "front": float(raw_vector[2]),
        }

    if not isinstance(raw_vector, dict):
        raw_vector = {}

    for axis_name, aliases in AXIS_ALIASES.items():
        for alias in aliases:
            value = _get_first_float(raw_vector.get(alias))
            if value is None:
                value = _get_first_float(payload.get(alias))
            if value is not None:
                vector[axis_name] = value
                break
    return vector


def _parse_named_vector(payload: dict, field_name: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {"right": 0.0, "up": 0.0, "front": 0.0}

    raw_vector = payload.get(field_name)
    if raw_vector is None:
        return {"right": 0.0, "up": 0.0, "front": 0.0}
    return _parse_relative_vector({field_name: raw_vector})


def _parse_prediction(pred_text: str) -> dict:
    payload = _extract_json_blob(pred_text) or {}
    answer = payload.get("answer")
    target_object = payload.get("target_object", payload.get("target", answer))
    scale_ratio = _get_first_float(payload.get("scale_ratio"))
    camera_distance = _get_first_float(payload.get("camera_distance"))

    if answer is None and isinstance(target_object, str):
        answer = target_object

    return {
        "raw_payload": payload,
        "answer": str(answer).strip() if answer is not None else "",
        "target_object": str(target_object).strip() if target_object is not None else "",
        "relative_vector": _parse_relative_vector(payload),
        "camera_vector": _parse_named_vector(payload, "camera_vector"),
        "camera_distance": camera_distance,
        "scale_ratio": scale_ratio,
    }


def _bbox_area(obj: dict) -> Optional[float]:
    bbox = obj.get("bbox_2d_norm")
    if bbox and len(bbox) == 4:
        ymin, xmin, ymax, xmax = bbox
        return max(0.0, float(ymax) - float(ymin)) * max(0.0, float(xmax) - float(xmin))

    bbox = obj.get("bbox_2d_xyxy_pixels")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    return None


def _scale_ratio(reference_obj: Optional[dict], target_obj: Optional[dict]) -> Optional[float]:
    if not reference_obj or not target_obj:
        return None
    reference_area = _bbox_area(reference_obj)
    target_area = _bbox_area(target_obj)
    if not reference_area or not target_area or reference_area <= 0.0 or target_area <= 0.0:
        return None
    return math.sqrt(target_area / reference_area)


def _camera_distance(obj: dict, camera_position: list[float]) -> Optional[float]:
    position = obj.get("position_3d")
    if not position or len(position) != 3 or not camera_position or len(camera_position) != 3:
        return None
    return math.sqrt(sum((float(position[i]) - float(camera_position[i])) ** 2 for i in range(3)))


def _camera_world_vector_from_anchor(doc, anchor_name: str) -> tuple[Optional[dict[str, float]], Optional[float]]:
    anchor_obj = _get_object_record(doc, anchor_name)
    metadata = doc.get("task_metadata", {}) or {}
    camera_position = (
        metadata.get("camera_position")
        or doc.get("camera_position")
        or (doc.get("camera_metadata", {}) or {}).get("position")
    )
    if not anchor_obj or not camera_position or len(camera_position) != 3:
        return None, None

    anchor_position = anchor_obj.get("position_3d")
    if not anchor_position or len(anchor_position) != 3:
        return None, None

    delta_x = float(camera_position[0]) - float(anchor_position[0])
    delta_y = float(camera_position[1]) - float(anchor_position[1])
    delta_z = float(camera_position[2]) - float(anchor_position[2])
    distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    if distance <= 1e-8:
        return None, None

    # World-aligned camera frame:
    # right = +X, up = +Z, front = +Y
    vector = {
        "right": delta_x,
        "up": delta_z,
        "front": delta_y,
    }
    return vector, distance


def _vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(len(a))]


def _vector_dot(a: list[float], b: list[float]) -> float:
    return float(sum(float(a[i]) * float(b[i]) for i in range(len(a))))


def _vector_cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    ]


def _vector_norm(a: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def _vector_normalize(a: list[float]) -> Optional[list[float]]:
    norm = _vector_norm(a)
    if norm <= 1e-8:
        return None
    return [float(x) / norm for x in a]


def _safe_unit(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(float(vector[axis]) ** 2 for axis in ("right", "up", "front")))
    if norm <= 1e-8:
        return {"right": 0.0, "up": 0.0, "front": 0.0}
    return {axis: float(vector[axis]) / norm for axis in ("right", "up", "front")}


def _cosine_similarity(pred_vector: dict[str, float], gt_vector: dict[str, float]) -> float:
    pred_unit = _safe_unit(pred_vector)
    gt_unit = _safe_unit(gt_vector)
    pred_norm = math.sqrt(sum(pred_unit[axis] ** 2 for axis in pred_unit))
    gt_norm = math.sqrt(sum(gt_unit[axis] ** 2 for axis in gt_unit))
    if pred_norm <= 1e-8 or gt_norm <= 1e-8:
        return 0.0
    return sum(pred_unit[axis] * gt_unit[axis] for axis in pred_unit)


def _sign(value: float) -> int:
    if value > 1e-8:
        return 1
    if value < -1e-8:
        return -1
    return 0


def _gt_pair(doc) -> tuple[Optional[dict], Optional[dict], list[str]]:
    names = _get_queried_object_names(doc)
    if len(names) < 2:
        return None, None, names
    return _get_object_record(doc, names[0]), _get_object_record(doc, names[1]), names


def _compute_anchor_local_vector(doc, anchor_name: str, target_name: str) -> Optional[dict[str, float]]:
    anchor_obj = _get_object_record(doc, anchor_name)
    target_obj = _get_object_record(doc, target_name)
    metadata = doc.get("task_metadata", {}) or {}
    camera_position = metadata.get("camera_position")

    if not anchor_obj or not target_obj or not camera_position:
        return None
    anchor_position = anchor_obj.get("position_3d")
    target_position = target_obj.get("position_3d")
    if not anchor_position or not target_position or len(anchor_position) != 3 or len(target_position) != 3 or len(camera_position) != 3:
        return None

    relative = _vector_sub(target_position, anchor_position)
    world_up = [0.0, 0.0, 1.0]
    forward_world = _vector_sub(camera_position, anchor_position)
    forward_horizontal = [forward_world[0], forward_world[1], 0.0]
    forward = _vector_normalize(forward_horizontal)
    if forward is None:
        return None
    # cross(forward, up) matches the natural-language right sign convention
    right = _vector_normalize(_vector_cross(forward, world_up))
    if right is None:
        return None

    return {
        "right": _vector_dot(relative, right),
        "up": relative[2],
        "front": _vector_dot(relative, forward),
    }


def _relation_axis_sign(relation: str, axis_value: float) -> int:
    axis_sign = _sign(axis_value)
    if relation in POSITIVE_RELATIONS:
        return 1 if axis_sign > 0 else -1 if axis_sign < 0 else 0
    if relation in NEGATIVE_RELATIONS:
        return -1 if axis_sign > 0 else 1 if axis_sign < 0 else 0
    return 0


def _multi_object_candidate_axis_values(doc) -> Optional[dict[str, float]]:
    if doc.get("task_family") != "object_centric_relative_position_multi":
        return None
    metadata = doc.get("task_metadata", {}) or {}
    anchor_name = metadata.get("anchor_object", "")
    relation = str(
        metadata.get("queried_relation", metadata.get("relation", ""))
    ).strip().lower()
    axis = RELATION_TO_AXIS.get(relation)
    candidates = metadata.get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
    if not anchor_name or not axis or not candidates:
        return None

    axis_values: dict[str, float] = {}
    for candidate in candidates:
        local_vector = _compute_anchor_local_vector(doc, anchor_name, candidate)
        if not local_vector:
            continue
        axis_values[_normalize_name(candidate)] = float(local_vector[axis])
    return axis_values if axis_values else None


def _multi_object_candidate_aware_scores(doc, prediction: dict) -> dict[str, float]:
    if doc.get("task_family") != "object_centric_relative_position_multi":
        return {
            "candidate_aware_direction_accuracy": 0.0,
            "direction_given_predicted_object_accuracy": 0.0,
            "ranking_accuracy_on_relation_axis": 0.0,
            "ranking_score_on_relation_axis": 0.0,
            "predicted_target_in_candidate_set": 0.0,
            "available": 0.0,
        }

    metadata = doc.get("task_metadata", {}) or {}
    relation = str(metadata.get("relation", "")).strip().lower()
    axis = RELATION_TO_AXIS.get(relation)
    pred_target = _normalize_name(prediction.get("target_object", "") or prediction.get("answer", ""))
    if not axis or not pred_target:
        return {
            "candidate_aware_direction_accuracy": 0.0,
            "direction_given_predicted_object_accuracy": 0.0,
            "ranking_accuracy_on_relation_axis": 0.0,
            "ranking_score_on_relation_axis": 0.0,
            "predicted_target_in_candidate_set": 0.0,
            "available": 1.0,
        }

    axis_values = _multi_object_candidate_axis_values(doc)
    if not axis_values or pred_target not in axis_values:
        return {
            "candidate_aware_direction_accuracy": 0.0,
            "direction_given_predicted_object_accuracy": 0.0,
            "ranking_accuracy_on_relation_axis": 0.0,
            "ranking_score_on_relation_axis": 0.0,
            "predicted_target_in_candidate_set": 0.0,
            "available": 1.0,
        }

    predicted_axis_value = float(axis_values[pred_target])
    predicted_axis_sign = _sign(predicted_axis_value)
    desired_sign = 1 if relation in POSITIVE_RELATIONS else -1 if relation in NEGATIVE_RELATIONS else 0
    candidate_aware_direction_accuracy = 1.0 if predicted_axis_sign == desired_sign and predicted_axis_sign != 0 else 0.0

    pred_vector_axis_sign = _sign(float(prediction["relative_vector"].get(axis, 0.0)))
    direction_given_predicted_object_accuracy = 1.0 if pred_vector_axis_sign == predicted_axis_sign and predicted_axis_sign != 0 else 0.0

    ordered_targets = sorted(
        axis_values.items(),
        key=lambda item: item[1],
        reverse=relation in POSITIVE_RELATIONS,
    )
    if relation in POSITIVE_RELATIONS:
        best_target = max(axis_values.items(), key=lambda item: item[1])[0]
    else:
        best_target = min(axis_values.items(), key=lambda item: item[1])[0]
    ranking_accuracy_on_relation_axis = 1.0 if pred_target == best_target else 0.0
    ordered_names = [name for name, _ in ordered_targets]
    rank_index = ordered_names.index(pred_target)
    if len(ordered_names) == 1:
        ranking_score_on_relation_axis = 1.0
    else:
        ranking_score_on_relation_axis = 1.0 - (rank_index / float(len(ordered_names) - 1))

    return {
        "candidate_aware_direction_accuracy": candidate_aware_direction_accuracy,
        "direction_given_predicted_object_accuracy": direction_given_predicted_object_accuracy,
        "ranking_accuracy_on_relation_axis": ranking_accuracy_on_relation_axis,
        "ranking_score_on_relation_axis": ranking_score_on_relation_axis,
        "predicted_target_in_candidate_set": 1.0,
        "available": 1.0,
    }


def _object_centric_sign_scores(prediction: dict, gt_spec: dict) -> dict[str, float]:
    task_family = gt_spec.get("task_family", "")
    if task_family not in {
        "object_centric_relative_position",
        "object_centric_relative_position_multi",
        "object_centric_direction_binary",
    }:
        return {
            "right_sign_accuracy": 0.0,
            "front_sign_accuracy": 0.0,
            "full_sign_accuracy": 0.0,
            "camera_front_sign_accuracy": 0.0,
            "camera_vector_nonzero": 0.0,
            "available": 0.0,
        }

    gt_right = float(gt_spec["vector"].get("right", 0.0))
    gt_front = float(gt_spec["vector"].get("front", 0.0))
    pred_right = float(prediction["relative_vector"].get("right", 0.0))
    pred_front = float(prediction["relative_vector"].get("front", 0.0))

    gt_right_sign = _sign(gt_right)
    gt_front_sign = _sign(gt_front)
    pred_right_sign = _sign(pred_right)
    pred_front_sign = _sign(pred_front)

    right_sign_accuracy = 1.0 if gt_right_sign != 0 and pred_right_sign == gt_right_sign else 0.0
    front_sign_accuracy = 1.0 if gt_front_sign != 0 and pred_front_sign == gt_front_sign else 0.0
    full_sign_accuracy = 1.0 if right_sign_accuracy == 1.0 and front_sign_accuracy == 1.0 else 0.0

    pred_camera_vector = prediction.get("camera_vector", {}) or {}
    camera_front_sign_accuracy = 1.0 if _sign(float(pred_camera_vector.get("front", 0.0))) > 0 else 0.0
    camera_vector_nonzero = 1.0 if any(abs(float(pred_camera_vector.get(axis, 0.0))) > 1e-8 for axis in ("right", "up", "front")) else 0.0

    return {
        "right_sign_accuracy": right_sign_accuracy,
        "front_sign_accuracy": front_sign_accuracy,
        "full_sign_accuracy": full_sign_accuracy,
        "camera_front_sign_accuracy": camera_front_sign_accuracy,
        "camera_vector_nonzero": camera_vector_nonzero,
        "available": 1.0,
    }


def _get_gt_spec(doc) -> dict:
    task_family = doc.get("task_family", "unknown")
    metadata = doc.get("task_metadata", {}) or {}
    relation = str(metadata.get("relation", "")).strip().lower()
    gt_answer = _ground_truth_answer_text(doc)

    spec = {
        "task_family": task_family,
        "relation": relation,
        "answer": gt_answer,
        "reference_object": "",
        "target_object": "",
        "vector": {"right": 0.0, "up": 0.0, "front": 0.0},
        "camera_vector": None,
        "camera_distance": None,
        "scale_ratio": None,
    }

    if task_family in {"camera_relative_position", "height_relative_3d", "camera_distance"}:
        reference_obj, target_obj, names = _gt_pair(doc)
        if len(names) >= 2:
            spec["reference_object"] = names[0]
            spec["target_object"] = names[1]
        spec["scale_ratio"] = _scale_ratio(reference_obj, target_obj)

        if task_family == "camera_relative_position":
            delta = metadata.get("center_delta")
            if isinstance(delta, list) and len(delta) == 2:
                spec["vector"] = {"right": float(delta[0]), "up": -float(delta[1]), "front": 0.0}
            elif reference_obj and target_obj:
                a = reference_obj.get("image_position_2d", reference_obj.get("bbox_center_norm"))
                b = target_obj.get("image_position_2d", target_obj.get("bbox_center_norm"))
                if a and b and len(a) == 2 and len(b) == 2:
                    spec["vector"] = {"right": float(b[0]) - float(a[0]), "up": float(a[1]) - float(b[1]), "front": 0.0}

        elif task_family == "height_relative_3d":
            if "object_a_z" in metadata and "object_b_z" in metadata:
                dz = float(metadata["object_b_z"]) - float(metadata["object_a_z"])
            elif reference_obj and target_obj:
                dz = float(target_obj["position_3d"][2]) - float(reference_obj["position_3d"][2])
            else:
                dz = 0.0
            spec["vector"] = {"right": 0.0, "up": dz, "front": 0.0}

        elif task_family == "camera_distance":
            distance_a = metadata.get("object_a_camera_distance")
            distance_b = metadata.get("object_b_camera_distance")
            if distance_a is not None and distance_b is not None:
                delta = float(distance_a) - float(distance_b)
            else:
                camera_position = metadata.get("camera_position")
                if reference_obj and target_obj and camera_position:
                    dist_a = _camera_distance(reference_obj, camera_position)
                    dist_b = _camera_distance(target_obj, camera_position)
                    delta = float(dist_a or 0.0) - float(dist_b or 0.0)
                else:
                    delta = 0.0
            spec["vector"] = {"right": 0.0, "up": 0.0, "front": delta}

    elif task_family == "object_centric_relative_position":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("target_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(_get_object_record(doc, anchor_name), _get_object_record(doc, target_name))
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        rel = metadata.get("relative_local_coordinates", {}) or {}
        spec["vector"] = {
            "right": -float(rel.get("right", 0.0)),
            "up": float(rel.get("up", 0.0)),
            "front": float(rel.get("front", 0.0)),
        }

    elif task_family == "object_centric_relative_position_multi":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("correct_object", gt_answer)
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(_get_object_record(doc, anchor_name), _get_object_record(doc, target_name))
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        rel = metadata.get("correct_relative_local_coordinates", {}) or {}
        spec["vector"] = {
            "right": -float(rel.get("right", 0.0)),
            "up": float(rel.get("up", 0.0)),
            "front": float(rel.get("front", 0.0)),
        }

    elif task_family == "object_centric_direction_binary":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("target_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(
            _get_object_record(doc, anchor_name),
            _get_object_record(doc, target_name),
        )
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        rel = metadata.get("relative_local_coordinates", {}) or {}
        spec["vector"] = {
            "right": -float(rel.get("right", 0.0)),
            "up": float(rel.get("up", 0.0)),
            "front": float(rel.get("front", 0.0)),
        }

    elif task_family == "object_centric_camera_pose":
        anchor_name = metadata.get("anchor_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = ""
        camera_vector = metadata.get("camera_vector_world_aligned")
        if isinstance(camera_vector, dict):
            spec["camera_vector"] = {
                "right": float(camera_vector.get("right", 0.0)),
                "up": float(camera_vector.get("up", 0.0)),
                "front": float(camera_vector.get("front", 0.0)),
            }
        else:
            spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        if spec["camera_vector"] is not None:
            spec["vector"] = dict(spec["camera_vector"])
        if spec["camera_distance"] is None and metadata.get("camera_distance") is not None:
            spec["camera_distance"] = float(metadata["camera_distance"])

    return spec


def _build_task_instructions(doc) -> str:
    gt_spec = _get_gt_spec(doc)
    task_family = gt_spec["task_family"]
    relation = gt_spec["relation"]
    answer = gt_spec["answer"]
    reference_object = gt_spec["reference_object"]
    target_object = gt_spec["target_object"]

    lines = [
        "Return only valid JSON.",
        "Use this canonical sign standard:",
        "- `right` > 0 means the target object is to the right of the reference object.",
        "- `up` > 0 means the target object is higher than the reference object.",
        "- `front` > 0 means the target object is more in front; for camera-distance questions, use `front` > 0 when the target object is closer to the camera.",
        "- `scale_ratio` means apparent target size divided by apparent reference size.",
    ]

    if task_family == "camera_relative_position":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":0.0},"scale_ratio":<float>}',
            ]
        )
    elif task_family == "height_relative_3d":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"'
                + target_object
                + '","relative_vector":{"right":0.0,"up":<float>,"front":0.0},"scale_ratio":<float>}',
            ]
        )
    elif task_family == "camera_distance":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"'
                + target_object
                + '","relative_vector":{"right":0.0,"up":0.0,"front":<float>},"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_relative_position":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Interpret the frame as standing at the anchor object and facing the camera.",
                "Also provide `camera_vector`, the true direction from the reference anchor object to the camera in a world-aligned frame.",
                "- For `camera_vector`, use `right` > 0 for larger world X, `up` > 0 for larger world Z, and `front` > 0 for larger world Y.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the discrete relation word: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<left|right|front|behind>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_relative_position_multi":
        candidates = doc.get("task_metadata", {}).get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
        candidate_text = ", ".join(str(item) for item in candidates)
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Candidate target objects: {candidate_text}",
                "Interpret the frame as standing at the anchor object and facing the camera.",
                "Also provide `camera_vector`, the true direction from the reference anchor object to the camera in a world-aligned frame.",
                "- For `camera_vector`, use `right` > 0 for larger world X, `up` > 0 for larger world Z, and `front` > 0 for larger world Y.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the chosen target object name: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"<object name>","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_direction_binary":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                f"Queried relation to verify: {relation}",
                "Interpret the frame as standing at the anchor object and facing the camera.",
                "Also provide `camera_vector`, the true direction from the reference anchor object to the camera in a world-aligned frame.",
                "- For `camera_vector`, use `right` > 0 for larger world X, `up` > 0 for larger world Z, and `front` > 0 for larger world Y.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with `yes` or `no`: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<yes|no>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_camera_pose":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                "Estimate the camera position relative to the anchor object using a world-aligned frame.",
                "- For `camera_vector`, use `right` > 0 for larger world X, `up` > 0 for larger world Z, and `front` > 0 for larger world Y.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the dominant horizontal relation word: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<left|right|front|behind>","camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>}',
            ]
        )

    return "\n".join(lines)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = str(doc.get("question", "")).strip()
    instructions = _build_task_instructions(doc)
    return f"Question: {question}\n{instructions}\n"


def doc_to_target(doc):
    return json.dumps(
        {
            "answer": _ground_truth_answer_text(doc),
            "target_object": _get_gt_spec(doc)["target_object"],
            "relative_vector": _get_gt_spec(doc)["vector"],
            "camera_vector": _get_gt_spec(doc)["camera_vector"],
            "camera_distance": _get_gt_spec(doc)["camera_distance"],
            "scale_ratio": _get_gt_spec(doc)["scale_ratio"],
        },
        ensure_ascii=True,
    )


def _answer_matches(prediction: dict, gt_spec: dict) -> float:
    gt_answer = _normalize_name(gt_spec["answer"])
    pred_answer = _normalize_name(prediction.get("answer"))
    pred_target = _normalize_name(prediction.get("target_object"))
    if pred_answer == gt_answer or pred_target == gt_answer:
        return 1.0
    if gt_spec["task_family"] == "object_centric_relative_position_multi" and pred_target == _normalize_name(gt_spec["target_object"]):
        return 1.0
    return 0.0


def _axis_sign_accuracy(prediction: dict, gt_spec: dict) -> float:
    relation = gt_spec["relation"]
    axis = RELATION_TO_AXIS.get(relation)
    if not axis:
        return 0.0

    gt_vector = gt_spec["vector"]
    pred_vector = prediction["relative_vector"]
    if gt_spec.get("task_family") == "object_centric_camera_pose":
        gt_vector = gt_spec.get("camera_vector") or gt_vector
        pred_vector = prediction.get("camera_vector") or pred_vector

    gt_value = float(gt_vector.get(axis, 0.0))
    pred_value = float(pred_vector.get(axis, 0.0))
    gt_sign = _sign(gt_value)
    pred_sign = _sign(pred_value)

    if relation in POSITIVE_RELATIONS:
        return 1.0 if pred_sign == gt_sign and gt_sign != 0 else 0.0
    if relation in NEGATIVE_RELATIONS:
        return 1.0 if pred_sign == gt_sign and gt_sign != 0 else 0.0
    return 0.0


def _scale_score(prediction: dict, gt_spec: dict) -> float:
    gt_scale = gt_spec.get("scale_ratio")
    pred_scale = prediction.get("scale_ratio")
    if gt_scale is None or pred_scale is None or gt_scale <= 0.0 or pred_scale <= 0.0:
        return 0.0
    return 1.0 / (1.0 + abs(math.log(pred_scale / gt_scale)))


def _camera_direction_cosine(prediction: dict, gt_spec: dict) -> tuple[float, float]:
    gt_camera_vector = gt_spec.get("camera_vector")
    if not gt_camera_vector:
        return 0.0, 0.0
    return _cosine_similarity(prediction["camera_vector"], gt_camera_vector), 1.0


def _camera_distance_score(prediction: dict, gt_spec: dict) -> tuple[float, float]:
    gt_camera_distance = gt_spec.get("camera_distance")
    pred_camera_distance = prediction.get("camera_distance")
    if gt_camera_distance is None:
        return 0.0, 0.0
    if pred_camera_distance is None or gt_camera_distance <= 0.0 or pred_camera_distance <= 0.0:
        return 0.0, 1.0
    return 1.0 / (1.0 + abs(math.log(pred_camera_distance / gt_camera_distance))), 1.0


def _camera_vector_sign_scores(prediction: dict, gt_spec: dict) -> dict[str, float]:
    gt_camera_vector = gt_spec.get("camera_vector")
    if not gt_camera_vector:
        return {
            "right_sign_accuracy": 0.0,
            "up_sign_accuracy": 0.0,
            "front_sign_accuracy": 0.0,
            "full_sign_accuracy": 0.0,
            "available": 0.0,
        }

    pred_camera_vector = prediction.get("camera_vector", {}) or {}
    per_axis = {}
    for axis in ("right", "up", "front"):
        gt_sign = _sign(float(gt_camera_vector.get(axis, 0.0)))
        pred_sign = _sign(float(pred_camera_vector.get(axis, 0.0)))
        per_axis[axis] = 1.0 if gt_sign != 0 and pred_sign == gt_sign else 0.0

    full_sign_accuracy = 1.0 if all(per_axis[axis] == 1.0 for axis in ("right", "up", "front")) else 0.0
    return {
        "right_sign_accuracy": per_axis["right"],
        "up_sign_accuracy": per_axis["up"],
        "front_sign_accuracy": per_axis["front"],
        "full_sign_accuracy": full_sign_accuracy,
        "available": 1.0,
    }


def _score_prediction(doc, pred_text: str) -> dict:
    gt_spec = _get_gt_spec(doc)
    prediction = _parse_prediction(pred_text)
    answer_accuracy = _answer_matches(prediction, gt_spec)
    axis_sign_accuracy = _axis_sign_accuracy(prediction, gt_spec)
    answer_and_direction_correct = 1.0 if answer_accuracy == 1.0 and axis_sign_accuracy == 1.0 else 0.0
    answer_correct_direction_wrong = 1.0 if answer_accuracy == 1.0 and axis_sign_accuracy == 0.0 else 0.0
    answer_wrong_direction_correct = 1.0 if answer_accuracy == 0.0 and axis_sign_accuracy == 1.0 else 0.0
    answer_and_direction_wrong = 1.0 if answer_accuracy == 0.0 and axis_sign_accuracy == 0.0 else 0.0
    camera_direction_cosine, camera_direction_available = _camera_direction_cosine(prediction, gt_spec)
    camera_distance_score, camera_distance_available = _camera_distance_score(prediction, gt_spec)
    camera_sign_scores = _camera_vector_sign_scores(prediction, gt_spec)
    multi_object_scores = _multi_object_candidate_aware_scores(doc, prediction)
    object_centric_scores = _object_centric_sign_scores(prediction, gt_spec)
    score_vector_pred = prediction["relative_vector"]
    score_vector_gt = gt_spec["vector"]
    if gt_spec.get("task_family") == "object_centric_camera_pose":
        score_vector_pred = prediction.get("camera_vector") or score_vector_pred
        score_vector_gt = gt_spec.get("camera_vector") or score_vector_gt
    vector_cosine = _cosine_similarity(score_vector_pred, score_vector_gt)
    scale_score = _scale_score(prediction, gt_spec)
    auxiliary_score = scale_score
    if gt_spec.get("scale_ratio") is None and camera_distance_available > 0.0:
        auxiliary_score = camera_distance_score
    combined_score = (answer_accuracy + axis_sign_accuracy + ((vector_cosine + 1.0) / 2.0) + auxiliary_score) / 4.0

    return {
        "gt_spec": gt_spec,
        "prediction": prediction,
        "answer_accuracy": answer_accuracy,
        "axis_sign_accuracy": axis_sign_accuracy,
        "answer_and_direction_correct": answer_and_direction_correct,
        "answer_correct_direction_wrong": answer_correct_direction_wrong,
        "answer_wrong_direction_correct": answer_wrong_direction_correct,
        "answer_and_direction_wrong": answer_and_direction_wrong,
        "camera_direction_cosine": camera_direction_cosine,
        "camera_direction_available": camera_direction_available,
        "camera_distance_score": camera_distance_score,
        "camera_distance_available": camera_distance_available,
        "camera_right_sign_accuracy": camera_sign_scores["right_sign_accuracy"],
        "camera_up_sign_accuracy": camera_sign_scores["up_sign_accuracy"],
        "camera_front_sign_accuracy": camera_sign_scores["front_sign_accuracy"],
        "camera_full_sign_accuracy": camera_sign_scores["full_sign_accuracy"],
        "camera_sign_metric_available": camera_sign_scores["available"],
        "object_centric_right_sign_accuracy": object_centric_scores["right_sign_accuracy"],
        "object_centric_front_sign_accuracy": object_centric_scores["front_sign_accuracy"],
        "object_centric_full_sign_accuracy": object_centric_scores["full_sign_accuracy"],
        "object_centric_camera_front_sign_accuracy": object_centric_scores["camera_front_sign_accuracy"],
        "object_centric_camera_vector_nonzero": object_centric_scores["camera_vector_nonzero"],
        "object_centric_metric_available": object_centric_scores["available"],
        "candidate_aware_direction_accuracy": multi_object_scores["candidate_aware_direction_accuracy"],
        "direction_given_predicted_object_accuracy": multi_object_scores["direction_given_predicted_object_accuracy"],
        "ranking_accuracy_on_relation_axis": multi_object_scores["ranking_accuracy_on_relation_axis"],
        "ranking_score_on_relation_axis": multi_object_scores["ranking_score_on_relation_axis"],
        "predicted_target_in_candidate_set": multi_object_scores["predicted_target_in_candidate_set"],
        "multi_object_metric_available": multi_object_scores["available"],
        "vector_cosine": vector_cosine,
        "scale_score": scale_score,
        "combined_score": combined_score,
    }


def process_results(doc, results):
    pred = results[0].strip() if results else ""
    scores = _score_prediction(doc, pred)
    task_family = doc.get("task_family", "unknown")
    difficulty = doc.get("difficulty", "unknown")
    index = doc.get("index", doc.get("qid"))
    qid = doc.get("qid", index)

    base_entry = {
        "index": index,
        "qid": qid,
        "task_family": task_family,
        "difficulty": difficulty,
        "answer_accuracy": scores["answer_accuracy"],
        "axis_sign_accuracy": scores["axis_sign_accuracy"],
        "answer_and_direction_correct": scores["answer_and_direction_correct"],
        "answer_correct_direction_wrong": scores["answer_correct_direction_wrong"],
        "answer_wrong_direction_correct": scores["answer_wrong_direction_correct"],
        "answer_and_direction_wrong": scores["answer_and_direction_wrong"],
        "camera_direction_cosine": scores["camera_direction_cosine"],
        "camera_direction_available": scores["camera_direction_available"],
        "camera_distance_score": scores["camera_distance_score"],
        "camera_distance_available": scores["camera_distance_available"],
        "camera_right_sign_accuracy": scores["camera_right_sign_accuracy"],
        "camera_up_sign_accuracy": scores["camera_up_sign_accuracy"],
        "camera_front_sign_accuracy": scores["camera_front_sign_accuracy"],
        "camera_full_sign_accuracy": scores["camera_full_sign_accuracy"],
        "camera_sign_metric_available": scores["camera_sign_metric_available"],
        "object_centric_right_sign_accuracy": scores["object_centric_right_sign_accuracy"],
        "object_centric_front_sign_accuracy": scores["object_centric_front_sign_accuracy"],
        "object_centric_full_sign_accuracy": scores["object_centric_full_sign_accuracy"],
        "object_centric_camera_front_sign_accuracy": scores["object_centric_camera_front_sign_accuracy"],
        "object_centric_camera_vector_nonzero": scores["object_centric_camera_vector_nonzero"],
        "object_centric_metric_available": scores["object_centric_metric_available"],
        "candidate_aware_direction_accuracy": scores["candidate_aware_direction_accuracy"],
        "direction_given_predicted_object_accuracy": scores["direction_given_predicted_object_accuracy"],
        "ranking_accuracy_on_relation_axis": scores["ranking_accuracy_on_relation_axis"],
        "ranking_score_on_relation_axis": scores["ranking_score_on_relation_axis"],
        "predicted_target_in_candidate_set": scores["predicted_target_in_candidate_set"],
        "multi_object_metric_available": scores["multi_object_metric_available"],
        "vector_cosine": scores["vector_cosine"],
        "scale_score": scores["scale_score"],
        "combined_score": scores["combined_score"],
    }

    result_dict = {
        "viewpoint_answer_accuracy": {**base_entry, "score": scores["answer_accuracy"]},
        "viewpoint_axis_sign_accuracy": {**base_entry, "score": scores["axis_sign_accuracy"]},
        "viewpoint_answer_and_direction_correct": {**base_entry, "score": scores["answer_and_direction_correct"]},
        "viewpoint_answer_correct_direction_wrong": {**base_entry, "score": scores["answer_correct_direction_wrong"]},
        "viewpoint_answer_wrong_direction_correct": {**base_entry, "score": scores["answer_wrong_direction_correct"]},
        "viewpoint_answer_and_direction_wrong": {**base_entry, "score": scores["answer_and_direction_wrong"]},
        "viewpoint_reference_to_camera_cosine": {**base_entry, "score": scores["camera_direction_cosine"]},
        "viewpoint_reference_to_camera_distance_score": {**base_entry, "score": scores["camera_distance_score"]},
        "object_centric_relative_position_camera_direction_cosine": {**base_entry, "score": scores["camera_direction_cosine"]},
        "object_centric_relative_position_multi_camera_direction_cosine": {**base_entry, "score": scores["camera_direction_cosine"]},
        "object_centric_relative_position_camera_distance_score": {**base_entry, "score": scores["camera_distance_score"]},
        "object_centric_relative_position_multi_camera_distance_score": {**base_entry, "score": scores["camera_distance_score"]},
        "object_centric_relative_position_camera_right_sign_accuracy": {
            **base_entry,
            "score": scores["camera_right_sign_accuracy"],
        },
        "object_centric_relative_position_camera_up_sign_accuracy": {
            **base_entry,
            "score": scores["camera_up_sign_accuracy"],
        },
        "object_centric_relative_position_right_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_right_sign_accuracy"],
        },
        "object_centric_relative_position_front_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_front_sign_accuracy"],
        },
        "object_centric_relative_position_full_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_full_sign_accuracy"],
        },
        "object_centric_relative_position_camera_front_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_camera_front_sign_accuracy"],
        },
        "object_centric_relative_position_camera_full_sign_accuracy": {
            **base_entry,
            "score": scores["camera_full_sign_accuracy"],
        },
        "object_centric_relative_position_camera_vector_nonzero": {
            **base_entry,
            "score": scores["object_centric_camera_vector_nonzero"],
        },
        "object_centric_relative_position_multi_camera_right_sign_accuracy": {
            **base_entry,
            "score": scores["camera_right_sign_accuracy"],
        },
        "object_centric_relative_position_multi_camera_up_sign_accuracy": {
            **base_entry,
            "score": scores["camera_up_sign_accuracy"],
        },
        "object_centric_relative_position_multi_right_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_right_sign_accuracy"],
        },
        "object_centric_relative_position_multi_front_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_front_sign_accuracy"],
        },
        "object_centric_relative_position_multi_full_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_full_sign_accuracy"],
        },
        "object_centric_relative_position_multi_camera_front_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_camera_front_sign_accuracy"],
        },
        "object_centric_relative_position_multi_camera_full_sign_accuracy": {
            **base_entry,
            "score": scores["camera_full_sign_accuracy"],
        },
        "object_centric_relative_position_multi_camera_vector_nonzero": {
            **base_entry,
            "score": scores["object_centric_camera_vector_nonzero"],
        },
        "object_centric_relative_position_multi_candidate_aware_direction_accuracy": {
            **base_entry,
            "score": scores["candidate_aware_direction_accuracy"],
        },
        "object_centric_relative_position_multi_direction_given_predicted_object_accuracy": {
            **base_entry,
            "score": scores["direction_given_predicted_object_accuracy"],
        },
        "object_centric_relative_position_multi_ranking_accuracy_on_relation_axis": {
            **base_entry,
            "score": scores["ranking_accuracy_on_relation_axis"],
        },
        "object_centric_relative_position_multi_ranking_score_on_relation_axis": {
            **base_entry,
            "score": scores["ranking_score_on_relation_axis"],
        },
        "object_centric_relative_position_multi_predicted_target_in_candidate_set": {
            **base_entry,
            "score": scores["predicted_target_in_candidate_set"],
        },
        "object_centric_direction_binary_camera_direction_cosine": {
            **base_entry,
            "score": scores["camera_direction_cosine"],
        },
        "object_centric_direction_binary_camera_distance_score": {
            **base_entry,
            "score": scores["camera_distance_score"],
        },
        "object_centric_direction_binary_camera_right_sign_accuracy": {
            **base_entry,
            "score": scores["camera_right_sign_accuracy"],
        },
        "object_centric_direction_binary_camera_up_sign_accuracy": {
            **base_entry,
            "score": scores["camera_up_sign_accuracy"],
        },
        "object_centric_direction_binary_camera_front_sign_accuracy": {
            **base_entry,
            "score": scores["camera_front_sign_accuracy"],
        },
        "object_centric_direction_binary_camera_full_sign_accuracy": {
            **base_entry,
            "score": scores["camera_full_sign_accuracy"],
        },
        "object_centric_direction_binary_right_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_right_sign_accuracy"],
        },
        "object_centric_direction_binary_front_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_front_sign_accuracy"],
        },
        "object_centric_direction_binary_full_sign_accuracy": {
            **base_entry,
            "score": scores["object_centric_full_sign_accuracy"],
        },
        "object_centric_direction_binary_camera_vector_nonzero": {
            **base_entry,
            "score": scores["object_centric_camera_vector_nonzero"],
        },
        "object_centric_camera_pose_camera_direction_cosine": {
            **base_entry,
            "score": scores["camera_direction_cosine"],
        },
        "object_centric_camera_pose_camera_distance_score": {
            **base_entry,
            "score": scores["camera_distance_score"],
        },
        "object_centric_camera_pose_camera_right_sign_accuracy": {
            **base_entry,
            "score": scores["camera_right_sign_accuracy"],
        },
        "object_centric_camera_pose_camera_up_sign_accuracy": {
            **base_entry,
            "score": scores["camera_up_sign_accuracy"],
        },
        "object_centric_camera_pose_camera_front_sign_accuracy": {
            **base_entry,
            "score": scores["camera_front_sign_accuracy"],
        },
        "object_centric_camera_pose_camera_full_sign_accuracy": {
            **base_entry,
            "score": scores["camera_full_sign_accuracy"],
        },
        "viewpoint_vector_cosine": {**base_entry, "score": scores["vector_cosine"]},
        "viewpoint_scale_score": {**base_entry, "score": scores["scale_score"]},
        "viewpoint_combined_score": {**base_entry, "score": scores["combined_score"]},
        "submission": {
            "index": index,
            "qid": qid,
            "task_family": task_family,
            "difficulty": difficulty,
            "question_prompt": doc_to_text(doc),
            "img_path": _get_image_path(doc),
            "gt": scores["gt_spec"],
            "prediction_text": pred,
            "parsed_prediction": scores["prediction"],
            "answer_accuracy": scores["answer_accuracy"],
            "axis_sign_accuracy": scores["axis_sign_accuracy"],
            "answer_and_direction_correct": scores["answer_and_direction_correct"],
            "answer_correct_direction_wrong": scores["answer_correct_direction_wrong"],
            "answer_wrong_direction_correct": scores["answer_wrong_direction_correct"],
            "answer_and_direction_wrong": scores["answer_and_direction_wrong"],
            "camera_direction_cosine": scores["camera_direction_cosine"],
            "camera_direction_available": scores["camera_direction_available"],
            "gt_camera_vector": scores["gt_spec"]["camera_vector"],
            "gt_camera_distance": scores["gt_spec"]["camera_distance"],
            "pred_camera_vector": scores["prediction"]["camera_vector"],
            "pred_camera_distance": scores["prediction"]["camera_distance"],
            "camera_distance_score": scores["camera_distance_score"],
            "camera_distance_available": scores["camera_distance_available"],
            "camera_right_sign_accuracy": scores["camera_right_sign_accuracy"],
            "camera_up_sign_accuracy": scores["camera_up_sign_accuracy"],
            "camera_front_sign_accuracy": scores["camera_front_sign_accuracy"],
            "camera_full_sign_accuracy": scores["camera_full_sign_accuracy"],
            "camera_sign_metric_available": scores["camera_sign_metric_available"],
            "object_centric_right_sign_accuracy": scores["object_centric_right_sign_accuracy"],
            "object_centric_front_sign_accuracy": scores["object_centric_front_sign_accuracy"],
            "object_centric_full_sign_accuracy": scores["object_centric_full_sign_accuracy"],
            "object_centric_camera_front_sign_accuracy": scores["object_centric_camera_front_sign_accuracy"],
            "object_centric_camera_vector_nonzero": scores["object_centric_camera_vector_nonzero"],
            "object_centric_metric_available": scores["object_centric_metric_available"],
            "candidate_aware_direction_accuracy": scores["candidate_aware_direction_accuracy"],
            "direction_given_predicted_object_accuracy": scores["direction_given_predicted_object_accuracy"],
            "ranking_accuracy_on_relation_axis": scores["ranking_accuracy_on_relation_axis"],
            "ranking_score_on_relation_axis": scores["ranking_score_on_relation_axis"],
            "predicted_target_in_candidate_set": scores["predicted_target_in_candidate_set"],
            "multi_object_metric_available": scores["multi_object_metric_available"],
            "vector_cosine": scores["vector_cosine"],
            "scale_score": scores["scale_score"],
            "combined_score": scores["combined_score"],
        },
    }

    for family in VIEWPOINT_TASK_FAMILIES:
        result_dict[f"{family}_viewpoint_combined_score"] = {**base_entry, "score": scores["combined_score"]}
        result_dict[f"{family}_viewpoint_answer_and_direction_correct"] = {
            **base_entry,
            "score": scores["answer_and_direction_correct"],
        }
        result_dict[f"{family}_viewpoint_answer_correct_direction_wrong"] = {
            **base_entry,
            "score": scores["answer_correct_direction_wrong"],
        }
        result_dict[f"{family}_viewpoint_answer_wrong_direction_correct"] = {
            **base_entry,
            "score": scores["answer_wrong_direction_correct"],
        }
        result_dict[f"{family}_viewpoint_answer_and_direction_wrong"] = {
            **base_entry,
            "score": scores["answer_and_direction_wrong"],
        }
    for difficulty_name in DIFFICULTIES:
        result_dict[f"{difficulty_name}_viewpoint_combined_score"] = {**base_entry, "score": scores["combined_score"]}
    return result_dict


def _aggregate_metric(results, metric_name: str, task_family: Optional[str] = None, difficulty: Optional[str] = None) -> float:
    filtered = [
        float(r.get(metric_name, 0.0))
        for r in results
        if (task_family is None or r.get("task_family") == task_family)
        and (difficulty is None or r.get("difficulty") == difficulty)
    ]
    return sum(filtered) / len(filtered) if filtered else 0.0


def aggregate_viewpoint_answer_accuracy(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_axis_sign_accuracy(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_vector_cosine(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_scale_score(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "score")


def aggregate_viewpoint_reference_to_camera_cosine(results):
    available = [r for r in results if float(r.get("camera_direction_available", 0.0)) > 0.0]
    if not available:
        return 0.0
    return sum(float(r.get("camera_direction_cosine", 0.0)) for r in available) / len(available)


def aggregate_viewpoint_reference_to_camera_distance_score(results):
    available = [r for r in results if float(r.get("camera_distance_available", 0.0)) > 0.0]
    if not available:
        return 0.0
    return sum(float(r.get("camera_distance_score", 0.0)) for r in available) / len(available)


def aggregate_camera_relative_position_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="camera_relative_position")


def aggregate_camera_relative_position_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="camera_relative_position")


def aggregate_camera_relative_position_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="camera_relative_position")


def aggregate_camera_relative_position_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="camera_relative_position")


def aggregate_camera_distance_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="camera_distance")


def aggregate_camera_distance_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="camera_distance")


def aggregate_camera_distance_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="camera_distance")


def aggregate_camera_distance_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="camera_distance")


def aggregate_height_relative_3d_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="height_relative_3d")


def aggregate_height_relative_3d_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="height_relative_3d")


def aggregate_height_relative_3d_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="height_relative_3d")


def aggregate_height_relative_3d_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="height_relative_3d")


def aggregate_object_centric_relative_position_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_camera_direction_cosine(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_relative_position"
        and float(r.get("camera_direction_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_direction_cosine", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_relative_position_camera_distance_score(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_relative_position"
        and float(r.get("camera_distance_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_distance_score", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_relative_position_camera_right_sign_accuracy(results):
    return _aggregate_metric(results, "camera_right_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_camera_up_sign_accuracy(results):
    return _aggregate_metric(results, "camera_up_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_camera_full_sign_accuracy(results):
    return _aggregate_metric(results, "camera_full_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_right_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_right_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_front_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_front_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_full_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_full_sign_accuracy", task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_camera_front_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "object_centric_camera_front_sign_accuracy",
        task_family="object_centric_relative_position",
    )


def aggregate_object_centric_relative_position_camera_vector_nonzero(results):
    return _aggregate_metric(
        results,
        "object_centric_camera_vector_nonzero",
        task_family="object_centric_relative_position",
    )


def aggregate_object_centric_relative_position_multi_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="object_centric_relative_position_multi")


def aggregate_object_centric_relative_position_multi_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="object_centric_relative_position_multi")


def aggregate_object_centric_relative_position_multi_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="object_centric_relative_position_multi")


def aggregate_object_centric_relative_position_multi_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="object_centric_relative_position_multi")


def aggregate_object_centric_relative_position_multi_camera_direction_cosine(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_relative_position_multi"
        and float(r.get("camera_direction_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_direction_cosine", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_relative_position_multi_camera_distance_score(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_relative_position_multi"
        and float(r.get("camera_distance_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_distance_score", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_relative_position_multi_camera_right_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "camera_right_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_camera_up_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "camera_up_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_camera_full_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "camera_full_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_right_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "object_centric_right_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_front_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "object_centric_front_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_full_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "object_centric_full_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_camera_front_sign_accuracy(results):
    return _aggregate_metric(
        results,
        "object_centric_camera_front_sign_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_camera_vector_nonzero(results):
    return _aggregate_metric(
        results,
        "object_centric_camera_vector_nonzero",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_candidate_aware_direction_accuracy(results):
    return _aggregate_metric(
        results,
        "candidate_aware_direction_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_direction_given_predicted_object_accuracy(results):
    return _aggregate_metric(
        results,
        "direction_given_predicted_object_accuracy",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_ranking_accuracy_on_relation_axis(results):
    return _aggregate_metric(
        results,
        "ranking_accuracy_on_relation_axis",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_ranking_score_on_relation_axis(results):
    return _aggregate_metric(
        results,
        "ranking_score_on_relation_axis",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_relative_position_multi_predicted_target_in_candidate_set(results):
    return _aggregate_metric(
        results,
        "predicted_target_in_candidate_set",
        task_family="object_centric_relative_position_multi",
    )


def aggregate_object_centric_direction_binary_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_camera_direction_cosine(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_direction_binary"
        and float(r.get("camera_direction_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_direction_cosine", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_direction_binary_camera_distance_score(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_direction_binary"
        and float(r.get("camera_distance_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_distance_score", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_direction_binary_camera_right_sign_accuracy(results):
    return _aggregate_metric(results, "camera_right_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_camera_up_sign_accuracy(results):
    return _aggregate_metric(results, "camera_up_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_camera_front_sign_accuracy(results):
    return _aggregate_metric(results, "camera_front_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_camera_full_sign_accuracy(results):
    return _aggregate_metric(results, "camera_full_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_right_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_right_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_front_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_front_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_full_sign_accuracy(results):
    return _aggregate_metric(results, "object_centric_full_sign_accuracy", task_family="object_centric_direction_binary")


def aggregate_object_centric_direction_binary_camera_vector_nonzero(results):
    return _aggregate_metric(results, "object_centric_camera_vector_nonzero", task_family="object_centric_direction_binary")


def aggregate_object_centric_camera_pose_viewpoint_answer_and_direction_correct(results):
    return _aggregate_metric(results, "answer_and_direction_correct", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_viewpoint_answer_correct_direction_wrong(results):
    return _aggregate_metric(results, "answer_correct_direction_wrong", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_viewpoint_answer_wrong_direction_correct(results):
    return _aggregate_metric(results, "answer_wrong_direction_correct", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_viewpoint_answer_and_direction_wrong(results):
    return _aggregate_metric(results, "answer_and_direction_wrong", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_camera_direction_cosine(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_camera_pose"
        and float(r.get("camera_direction_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_direction_cosine", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_camera_pose_camera_distance_score(results):
    filtered = [
        r for r in results
        if r.get("task_family") == "object_centric_camera_pose"
        and float(r.get("camera_distance_available", 0.0)) > 0.0
    ]
    if not filtered:
        return 0.0
    return sum(float(r.get("camera_distance_score", 0.0)) for r in filtered) / len(filtered)


def aggregate_object_centric_camera_pose_camera_right_sign_accuracy(results):
    return _aggregate_metric(results, "camera_right_sign_accuracy", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_camera_up_sign_accuracy(results):
    return _aggregate_metric(results, "camera_up_sign_accuracy", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_camera_front_sign_accuracy(results):
    return _aggregate_metric(results, "camera_front_sign_accuracy", task_family="object_centric_camera_pose")


def aggregate_object_centric_camera_pose_camera_full_sign_accuracy(results):
    return _aggregate_metric(results, "camera_full_sign_accuracy", task_family="object_centric_camera_pose")


def aggregate_viewpoint_combined_score(results):
    overall = _aggregate_metric(results, "score")
    eval_logger.info(f"Kubric MOVi-A viewpoint combined: {overall * 100:.2f}%")
    for family in VIEWPOINT_TASK_FAMILIES:
        family_score = _aggregate_metric(results, "combined_score", task_family=family)
        eval_logger.info(f"Kubric MOVi-A viewpoint {family}: {family_score * 100:.2f}%")
    for difficulty in DIFFICULTIES:
        difficulty_score = _aggregate_metric(results, "combined_score", difficulty=difficulty)
        eval_logger.info(f"Kubric MOVi-A viewpoint {difficulty}: {difficulty_score * 100:.2f}%")
    return overall


def _get_submission_model_tag(args) -> str:
    model_name = getattr(args, "model", "") or ""
    if model_name:
        return sanitize_model_name(model_name)

    model_args = getattr(args, "model_args", "") or ""
    for key in ("pretrained", "model_name", "model_path"):
        match = re.search(rf"(?:^|,){key}=([^,]+)", model_args)
        if match:
            return sanitize_model_name(match.group(1).strip())

    return "unknown_model"


def aggregate_results_for_submission(results, args):
    model_tag = _get_submission_model_tag(args)
    path = generate_submission_file(f"kubric_movi_a_viewpoint_{model_tag}.json", args)
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    eval_logger.info(f"Viewpoint records saved to {path}.")
