"""
Canonical viewpoint-estimation task for Kubric MOVi-A.

This task asks a model to produce a structured relative-position estimate that can
be compared directly against dataset ground truth in a consistent coordinate
standard across image-plane, camera-distance, and anchor-centric reasoning tasks.
"""

import json
import math
import os
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


def _format_signed_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "null"
    return f"{float(value):+.{digits}f}"


def _format_vector_text(vector: Optional[dict[str, float]]) -> str:
    if not vector:
        return '{"right": +0.0000, "up": +0.0000, "front": +0.0000}'
    return (
        '{'
        f'"right": {_format_signed_float(vector.get("right", 0.0))}, '
        f'"up": {_format_signed_float(vector.get("up", 0.0))}, '
        f'"front": {_format_signed_float(vector.get("front", 0.0))}'
        '}'
    )


def _exclude_gold_answer_from_hint() -> bool:
    return os.getenv("LMMS_EVAL_VIEWPOINT_HINT_EXCLUDE_GOLD_ANSWER", "0") == "1"


def _relation_polarity_text(relation: str) -> str:
    axis = RELATION_TO_AXIS.get(relation)
    if not axis:
        return ""
    if relation in POSITIVE_RELATIONS:
        return f"For relation `{relation}`, the decisive `{axis}` component should be positive."
    if relation in NEGATIVE_RELATIONS:
        return f"For relation `{relation}`, the decisive `{axis}` component should be negative."
    return ""


def _build_consistency_hint(doc, gt_spec: dict) -> str:
    relation = gt_spec.get("relation", "")
    task_family = gt_spec.get("task_family", "")
    answer = gt_spec.get("answer", "")
    axis = RELATION_TO_AXIS.get(relation)
    lines = []

    if answer and not _exclude_gold_answer_from_hint():
        lines.append(f"The gold answer is `{answer}`.")

    polarity_hint = _relation_polarity_text(relation)
    if polarity_hint:
        lines.append(polarity_hint)

    if task_family == "object_centric_relative_position_multi":
        candidate_axis_values = _multi_object_candidate_axis_values(doc) or {}
        if candidate_axis_values and axis:
            ranked = sorted(
                candidate_axis_values.items(),
                key=lambda item: item[1],
                reverse=relation in POSITIVE_RELATIONS,
            )
            ranked_text = ", ".join(
                f"{name}({_format_signed_float(value)})" for name, value in ranked
            )
            lines.append(
                f"Candidate ranking on `{axis}` from best to worst for `{relation}` is: {ranked_text}."
            )
            target_name = _normalize_name(gt_spec.get("target_object", ""))
            if target_name in candidate_axis_values:
                lines.append(
                    f"The chosen target `{gt_spec.get('target_object', '')}` has `{axis}` = "
                    f"{_format_signed_float(candidate_axis_values[target_name])}."
                )

    elif task_family == "object_centric_direction_binary":
        target_name = gt_spec.get("target_object", "")
        if axis and target_name:
            lines.append(
                f"To keep answer and direction consistent, the `{axis}` sign of `{target_name}` relative to "
                f"`{gt_spec.get('reference_object', '')}` must agree with the yes/no label."
            )

    elif task_family in {
        "camera_relative_position",
        "height_relative_3d",
        "camera_distance",
        "object_centric_relative_position",
        "object_centric_camera_pose",
    }:
        vector = gt_spec.get("camera_vector") if task_family == "object_centric_camera_pose" else gt_spec.get("vector")
        if axis and vector:
            lines.append(
                f"The decisive `{axis}` component is {_format_signed_float(vector.get(axis, 0.0))}."
            )

    return " ".join(lines)


def _build_viewpoint_GT_help(doc, help_mode: str) -> str:
    assert help_mode in {"0", "1", "2", "3", "4", "5", "6", "7", "8"}, "Invalid GT help format option"
    if help_mode == "0":
        return ""

    gt_spec = _get_gt_spec(doc)
    relation = gt_spec.get("relation", "")
    axis = RELATION_TO_AXIS.get(relation)
    answer = gt_spec.get("answer", "")
    target_object = gt_spec.get("target_object", "")
    reference_object = gt_spec.get("reference_object", "")
    vector = gt_spec.get("vector")
    camera_vector = gt_spec.get("camera_vector")
    camera_distance = gt_spec.get("camera_distance")
    scale_ratio = gt_spec.get("scale_ratio")
    exclude_gold_answer = _exclude_gold_answer_from_hint()

    if help_mode == "1":
        return "" if exclude_gold_answer else f"The gold answer is `{answer}`."

    if help_mode == "2":
        return _build_consistency_hint(doc, gt_spec)

    if help_mode == "3":
        parts = []
        if answer and not exclude_gold_answer:
            parts.append(f"The gold answer is `{answer}`.")
        if axis:
            parts.append(
                f"Use the `{axis}` component to resolve relation `{relation}`; its gold value is "
                f"{_format_signed_float((camera_vector if gt_spec.get('task_family') == 'object_centric_camera_pose' else vector).get(axis, 0.0))}."
            )
        return " ".join(parts)

    if help_mode == "4":
        answer_text = "" if exclude_gold_answer else f"answer=`{answer}`, "
        return (
            f"Gold structured target: {answer_text}target_object=`{target_object}`, "
            f"relative_vector={_format_vector_text(vector)}."
        )

    if help_mode == "5":
        answer_text = "" if exclude_gold_answer else f"answer=`{answer}`, "
        lines = [
            f"Gold structured target: {answer_text}target_object=`{target_object}`, "
            f"reference_object=`{reference_object}`.",
            f"relative_vector={_format_vector_text(vector)}.",
        ]
        if camera_vector is not None:
            lines.append(f"camera_vector={_format_vector_text(camera_vector)}.")
        if camera_distance is not None:
            lines.append(f"camera_distance={float(camera_distance):.4f}.")
        if scale_ratio is not None:
            lines.append(f"scale_ratio={float(scale_ratio):.4f}.")
        return " ".join(lines)

    if help_mode == "6":
        lines = [_build_consistency_hint(doc, gt_spec)]
        if camera_vector is not None:
            lines.append(f"Reference-to-camera vector is { _format_vector_text(camera_vector) }.")
        if camera_distance is not None:
            lines.append(f"Reference-to-camera distance is {float(camera_distance):.4f}.")
        return " ".join(line for line in lines if line)

    if help_mode == "7":
        if gt_spec.get("task_family") != "object_centric_relative_position_multi":
            return ""
        relation_axis_values = _multi_object_candidate_axis_values(doc) or {}
        if not relation_axis_values or not axis:
            return ""
        ranked = sorted(
            relation_axis_values.items(),
            key=lambda item: item[1],
            reverse=relation in POSITIVE_RELATIONS,
        )
        return (
            f"Candidate analysis for relation `{relation}` on axis `{axis}`: "
            + ", ".join(f"{name}={_format_signed_float(value)}" for name, value in ranked)
            + "."
        )

    if help_mode == "8":
        lines = [f"relation=`{relation}`."]
        if not exclude_gold_answer:
            lines.insert(0, f"Gold answer=`{answer}`.")
        if axis:
            lines.append(f"decisive_axis=`{axis}`.")
        lines.append(f"relative_vector={_format_vector_text(vector)}.")
        if camera_vector is not None:
            lines.append(f"camera_vector={_format_vector_text(camera_vector)}.")
        if camera_distance is not None:
            lines.append(f"camera_distance={float(camera_distance):.4f}.")
        if scale_ratio is not None:
            lines.append(f"scale_ratio={float(scale_ratio):.4f}.")
        return " ".join(lines)

    return ""


def _parse_relative_vector(payload: dict) -> dict[str, float]:
    vector = {"right": 0.0, "up": 0.0, "front": 0.0}
    if payload is None:
        return vector

    raw_vector = payload
    if isinstance(payload, dict):
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
            if value is None and isinstance(payload, dict):
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
    return _parse_relative_vector(raw_vector)


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


def _get_frame_index(doc) -> Optional[int]:
    metadata = doc.get("task_metadata", {}) or {}
    for key in ("frame_index", "frame_idx", "frame_number"):
        value = metadata.get(key, doc.get(key))
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    image_path = _get_image_path(doc)
    match = re.search(r"frame_(\d+)", image_path)
    if match:
        return int(match.group(1))
    return None


def _select_frame_value(value, frame_index: Optional[int]):
    if not isinstance(value, list) or not value:
        return value
    if all(not isinstance(item, list) for item in value):
        return value
    if frame_index is not None and 0 <= frame_index < len(value):
        return value[frame_index]
    return value[0]


def _get_camera_position(doc) -> Optional[list[float]]:
    metadata = doc.get("task_metadata", {}) or {}
    camera_metadata = doc.get("camera_metadata", {}) or {}
    camera_doc = doc.get("camera", {}) or {}
    frame_index = _get_frame_index(doc)

    candidates = [
        metadata.get("camera_position"),
        doc.get("camera_position"),
        camera_metadata.get("position"),
        camera_doc.get("position"),
        camera_doc.get("positions"),
    ]
    for candidate in candidates:
        selected = _select_frame_value(candidate, frame_index)
        if isinstance(selected, list) and len(selected) == 3:
            return [float(selected[i]) for i in range(3)]
    return None


def _get_camera_quaternion(doc) -> Optional[list[float]]:
    metadata = doc.get("task_metadata", {}) or {}
    camera_metadata = doc.get("camera_metadata", {}) or {}
    camera_doc = doc.get("camera", {}) or {}
    frame_index = _get_frame_index(doc)

    candidates = [
        metadata.get("camera_quaternion"),
        metadata.get("camera_rotation"),
        doc.get("camera_quaternion"),
        doc.get("camera_rotation"),
        camera_metadata.get("quaternion"),
        camera_metadata.get("rotation"),
        camera_doc.get("quaternion"),
        camera_doc.get("quaternions"),
        camera_doc.get("rotation"),
    ]
    for candidate in candidates:
        selected = _select_frame_value(candidate, frame_index)
        if isinstance(selected, list) and len(selected) == 4:
            return [float(selected[i]) for i in range(4)]
    return None


def _quaternion_to_camera_basis(quaternion: list[float]) -> Optional[dict[str, list[float]]]:
    if not quaternion or len(quaternion) != 4:
        return None

    w, x, y, z = (float(quaternion[i]) for i in range(4))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]

    right = [rotation[row][0] for row in range(3)]
    up = [rotation[row][1] for row in range(3)]
    front = [-rotation[row][2] for row in range(3)]
    return {"right": right, "up": up, "front": front}


def _camera_frame_vector_from_world(doc, world_vector: list[float]) -> dict[str, float]:
    basis = _quaternion_to_camera_basis(_get_camera_quaternion(doc) or [])
    if not basis:
        return {
            "right": float(world_vector[0]),
            "up": float(world_vector[2]),
            "front": float(world_vector[1]),
        }

    return {
        "right": _vector_dot(world_vector, basis["right"]),
        "up": _vector_dot(world_vector, basis["up"]),
        "front": _vector_dot(world_vector, basis["front"]),
    }


def _camera_frame_position(doc, position: list[float]) -> Optional[dict[str, float]]:
    camera_position = _get_camera_position(doc)
    if not position or len(position) != 3 or not camera_position or len(camera_position) != 3:
        return None
    translated = _vector_sub(position, camera_position)
    return _camera_frame_vector_from_world(doc, translated)


def _camera_distance(obj: dict, camera_position: list[float]) -> Optional[float]:
    position = obj.get("position_3d")
    if not position or len(position) != 3 or not camera_position or len(camera_position) != 3:
        return None
    return math.sqrt(sum((float(position[i]) - float(camera_position[i])) ** 2 for i in range(3)))


def _camera_world_vector_from_anchor(doc, anchor_name: str) -> tuple[Optional[dict[str, float]], Optional[float]]:
    anchor_obj = _get_object_record(doc, anchor_name)
    if not anchor_obj:
        return None, None

    anchor_position = anchor_obj.get("position_3d")
    anchor_in_camera_frame = _camera_frame_position(doc, anchor_position) if anchor_position else None
    if not anchor_in_camera_frame:
        return None, None

    vector = {
        "right": -float(anchor_in_camera_frame["right"]),
        "up": -float(anchor_in_camera_frame["up"]),
        "front": -float(anchor_in_camera_frame["front"]),
    }
    distance = math.sqrt(sum(float(vector[axis]) ** 2 for axis in ("right", "up", "front")))
    if distance <= 1e-8:
        return None, None

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
    if not anchor_obj or not target_obj:
        return None
    anchor_position = anchor_obj.get("position_3d")
    target_position = target_obj.get("position_3d")
    if not anchor_position or not target_position or len(anchor_position) != 3 or len(target_position) != 3:
        return None

    relative = _vector_sub(target_position, anchor_position)
    return _camera_frame_vector_from_world(doc, relative)


def _pair_answer_from_relation(relation: str, reference_name: str, target_name: str, axis_value: float, fallback: str) -> str:
    axis_sign = _sign(axis_value)
    if axis_sign == 0:
        return fallback
    if relation in POSITIVE_RELATIONS:
        return target_name if axis_sign > 0 else reference_name
    if relation in NEGATIVE_RELATIONS:
        return target_name if axis_sign < 0 else reference_name
    return fallback


def _binary_answer_from_relation(relation: str, axis_value: float, fallback: str) -> str:
    axis_sign = _sign(axis_value)
    if axis_sign == 0:
        return fallback
    if relation in POSITIVE_RELATIONS:
        return "yes" if axis_sign > 0 else "no"
    if relation in NEGATIVE_RELATIONS:
        return "yes" if axis_sign < 0 else "no"
    return fallback


def _dominant_horizontal_relation(vector: dict[str, float], fallback: str) -> str:
    right = float(vector.get("right", 0.0))
    front = float(vector.get("front", 0.0))
    if abs(right) >= abs(front) and abs(right) > 1e-8:
        return "right" if right > 0 else "left"
    if abs(front) > 1e-8:
        return "front" if front > 0 else "behind"
    return fallback


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
        reference_name = spec["reference_object"]
        target_name = spec["target_object"]
        reference_position = reference_obj.get("position_3d") if reference_obj else None
        target_position = target_obj.get("position_3d") if target_obj else None

        if task_family == "camera_relative_position":
            if reference_position and target_position:
                spec["vector"] = _camera_frame_vector_from_world(
                    doc,
                    _vector_sub(target_position, reference_position),
                )
            axis = RELATION_TO_AXIS.get(relation)
            if axis:
                spec["answer"] = _pair_answer_from_relation(
                    relation,
                    reference_name,
                    target_name,
                    float(spec["vector"].get(axis, 0.0)),
                    gt_answer,
                )

        elif task_family == "height_relative_3d":
            if reference_position and target_position:
                spec["vector"] = _camera_frame_vector_from_world(
                    doc,
                    _vector_sub(target_position, reference_position),
                )
            axis = RELATION_TO_AXIS.get(relation)
            if axis:
                spec["answer"] = _pair_answer_from_relation(
                    relation,
                    reference_name,
                    target_name,
                    float(spec["vector"].get(axis, 0.0)),
                    gt_answer,
                )

        elif task_family == "camera_distance":
            reference_camera_position = _camera_frame_position(doc, reference_position) if reference_position else None
            target_camera_position = _camera_frame_position(doc, target_position) if target_position else None
            if reference_camera_position and target_camera_position:
                depth_delta = float(reference_camera_position["front"]) - float(target_camera_position["front"])
                spec["vector"] = {"right": 0.0, "up": 0.0, "front": depth_delta}
            else:
                camera_position = _get_camera_position(doc) or metadata.get("camera_position")
                if reference_obj and target_obj and camera_position:
                    dist_a = _camera_distance(reference_obj, camera_position)
                    dist_b = _camera_distance(target_obj, camera_position)
                    spec["vector"] = {"right": 0.0, "up": 0.0, "front": float(dist_a or 0.0) - float(dist_b or 0.0)}
            spec["answer"] = _pair_answer_from_relation(
                relation,
                reference_name,
                target_name,
                float(spec["vector"].get("front", 0.0)),
                gt_answer,
            )

    elif task_family == "object_centric_relative_position":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("target_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(_get_object_record(doc, anchor_name), _get_object_record(doc, target_name))
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        local_vector = _compute_anchor_local_vector(doc, anchor_name, target_name)
        if local_vector:
            spec["vector"] = local_vector
            spec["answer"] = _dominant_horizontal_relation(local_vector, gt_answer)
            spec["relation"] = spec["answer"]

    elif task_family == "object_centric_relative_position_multi":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("correct_object", gt_answer)
        spec["reference_object"] = anchor_name
        axis_values = _multi_object_candidate_axis_values(doc) or {}
        candidates = metadata.get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
        axis = RELATION_TO_AXIS.get(relation)
        normalized_candidates = { _normalize_name(name): name for name in candidates }
        if axis_values and axis:
            if relation in POSITIVE_RELATIONS:
                chosen_target = max(axis_values.items(), key=lambda item: item[1])[0]
            else:
                chosen_target = min(axis_values.items(), key=lambda item: item[1])[0]
            target_name = normalized_candidates.get(chosen_target, target_name)
        spec["target_object"] = target_name
        spec["answer"] = target_name
        spec["scale_ratio"] = _scale_ratio(_get_object_record(doc, anchor_name), _get_object_record(doc, target_name))
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        local_vector = _compute_anchor_local_vector(doc, anchor_name, target_name)
        if local_vector:
            spec["vector"] = local_vector

    elif task_family == "object_centric_direction_binary":
        relation = str(
            metadata.get("queried_relation", metadata.get("relation", ""))
        ).strip().lower()
        spec["relation"] = relation
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("target_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(
            _get_object_record(doc, anchor_name),
            _get_object_record(doc, target_name),
        )
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        local_vector = _compute_anchor_local_vector(doc, anchor_name, target_name)
        if local_vector:
            spec["vector"] = local_vector
        axis = RELATION_TO_AXIS.get(relation)
        if axis:
            spec["answer"] = _binary_answer_from_relation(
                relation,
                float(spec["vector"].get(axis, 0.0)),
                gt_answer,
            )

    elif task_family == "object_centric_camera_pose":
        anchor_name = metadata.get("anchor_object", "")
        spec["reference_object"] = anchor_name
        spec["target_object"] = ""
        spec["camera_vector"], spec["camera_distance"] = _camera_world_vector_from_anchor(doc, anchor_name)
        if spec["camera_vector"] is not None:
            spec["vector"] = dict(spec["camera_vector"])
            spec["answer"] = _dominant_horizontal_relation(spec["camera_vector"], gt_answer)
            spec["relation"] = spec["answer"]
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
        "Use this canonical camera-frame convention for every 3D coordinate you return:",
        "- First translate coordinates so the camera center is at the origin.",
        "- Then rotate into the camera frame using the camera rotation.",
        "- In this camera frame, `right` is +X, `front` is +Y, and `up` is +Z.",
        "- `right` > 0 means the target is to the camera-right of the reference.",
        "- `up` > 0 means the target is above the reference in the camera frame.",
        "- `front` > 0 means the target is farther forward along the viewing direction; for camera-distance questions, use `front` > 0 when the target object is closer to the camera.",
        "- `scale_ratio` means apparent target size divided by apparent reference size.",
    ]

    if task_family == "camera_relative_position":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `<Answer>` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"scale_ratio":<float>}',
            ]
        )
    elif task_family == "height_relative_3d":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `<Answer>` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"scale_ratio":<float>}',
            ]
        )
    elif task_family == "camera_distance":
        lines.extend(
            [
                f"Reference object: {reference_object}",
                f"Target object: {target_object}",
                f"Question relation to resolve: {relation}",
                f"Answer with which object satisfies the question: `<Answer>` is the gold label format.",
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
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "Also provide `camera_vector`, the direction from the anchor object to the camera after the same camera-origin, camera-rotation transform.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the discrete relation word: `<Answer>` is the gold label format.",
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
                "Express each candidate target relative to the anchor in the camera frame described above.",
                "Also provide `camera_vector`, the direction from the anchor object to the camera after the same camera-origin, camera-rotation transform.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the chosen target object name: `<Answer>` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"<object name>","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_direction_binary":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                f"Queried relation to verify: {relation}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "Also provide `camera_vector`, the direction from the anchor object to the camera after the same camera-origin, camera-rotation transform.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with `yes` or `no`: `<Answer>` is the gold label format.",
                'JSON schema: {"answer":"<yes|no>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_camera_pose":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                "Estimate the camera position relative to the anchor object in the canonical camera frame described above.",
                "Because the camera is the origin, `camera_vector` should point from the anchor object back to the camera in that same frame.",
                "Also provide `camera_distance`, the Euclidean distance from the anchor object center to the camera.",
                f"Answer with the dominant horizontal relation word: `<Answer>` is the gold label format.",
                'JSON schema: {"answer":"<left|right|front|behind>","camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>}',
            ]
        )

    return "\n".join(lines)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = str(doc.get("question", "")).strip()
    instructions = _build_task_instructions(doc)
    gt_help = ""
    help_mode = os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0")
    if help_mode in {"1", "2", "3", "4", "5", "6", "7", "8"}:
        gt_help = _build_viewpoint_GT_help(doc, help_mode)

    prompt = f"Question: {question}\n{instructions}\n"
    if gt_help:
        prompt += f"{gt_help}\n"
    return prompt


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
            "Help_Prompt": _build_viewpoint_GT_help(doc, os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0")),
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
