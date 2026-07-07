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


def _parse_prediction(pred_text: str) -> dict:
    payload = _extract_json_blob(pred_text) or {}
    answer = payload.get("answer")
    target_object = payload.get("target_object", payload.get("target", answer))
    scale_ratio = _get_first_float(payload.get("scale_ratio"))

    if answer is None and isinstance(target_object, str):
        answer = target_object

    return {
        "raw_payload": payload,
        "answer": str(answer).strip() if answer is not None else "",
        "target_object": str(target_object).strip() if target_object is not None else "",
        "relative_vector": _parse_relative_vector(payload),
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
        rel = metadata.get("relative_local_coordinates", {}) or {}
        spec["vector"] = {
            "right": float(rel.get("right", 0.0)),
            "up": float(rel.get("up", 0.0)),
            "front": float(rel.get("front", 0.0)),
        }

    elif task_family == "object_centric_relative_position_multi":
        anchor_name = metadata.get("anchor_object", "")
        target_name = metadata.get("correct_object", gt_answer)
        spec["reference_object"] = anchor_name
        spec["target_object"] = target_name
        spec["scale_ratio"] = _scale_ratio(_get_object_record(doc, anchor_name), _get_object_record(doc, target_name))
        rel = metadata.get("correct_relative_local_coordinates", {}) or {}
        spec["vector"] = {
            "right": -float(rel.get("right", 0.0)),
            "up": float(rel.get("up", 0.0)),
            "front": float(rel.get("front", 0.0)),
        }

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
                f"Answer with the discrete relation word: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<left|right|front|behind>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"scale_ratio":<float>}',
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
                f"Answer with the chosen target object name: `{answer}` is the gold label format.",
                'JSON schema: {"answer":"<object name>","target_object":"<object name>","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"scale_ratio":<float>}',
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

    gt_value = float(gt_spec["vector"].get(axis, 0.0))
    pred_value = float(prediction["relative_vector"].get(axis, 0.0))
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


def _score_prediction(doc, pred_text: str) -> dict:
    gt_spec = _get_gt_spec(doc)
    prediction = _parse_prediction(pred_text)
    answer_accuracy = _answer_matches(prediction, gt_spec)
    axis_sign_accuracy = _axis_sign_accuracy(prediction, gt_spec)
    answer_and_direction_correct = 1.0 if answer_accuracy == 1.0 and axis_sign_accuracy == 1.0 else 0.0
    answer_correct_direction_wrong = 1.0 if answer_accuracy == 1.0 and axis_sign_accuracy == 0.0 else 0.0
    answer_wrong_direction_correct = 1.0 if answer_accuracy == 0.0 and axis_sign_accuracy == 1.0 else 0.0
    answer_and_direction_wrong = 1.0 if answer_accuracy == 0.0 and axis_sign_accuracy == 0.0 else 0.0
    vector_cosine = _cosine_similarity(prediction["relative_vector"], gt_spec["vector"])
    scale_score = _scale_score(prediction, gt_spec)
    combined_score = (answer_accuracy + axis_sign_accuracy + ((vector_cosine + 1.0) / 2.0) + scale_score) / 4.0

    return {
        "gt_spec": gt_spec,
        "prediction": prediction,
        "answer_accuracy": answer_accuracy,
        "axis_sign_accuracy": axis_sign_accuracy,
        "answer_and_direction_correct": answer_and_direction_correct,
        "answer_correct_direction_wrong": answer_correct_direction_wrong,
        "answer_wrong_direction_correct": answer_wrong_direction_correct,
        "answer_and_direction_wrong": answer_and_direction_wrong,
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
            "vector_cosine": scores["vector_cosine"],
            "scale_score": scores["scale_score"],
            "combined_score": scores["combined_score"],
        },
    }

    for family in TASK_FAMILIES:
        result_dict[f"{family}_viewpoint_combined_score"] = {**base_entry, "score": scores["combined_score"]}
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


def aggregate_viewpoint_combined_score(results):
    overall = _aggregate_metric(results, "score")
    eval_logger.info(f"Kubric MOVi-A viewpoint combined: {overall * 100:.2f}%")
    for family in TASK_FAMILIES:
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
