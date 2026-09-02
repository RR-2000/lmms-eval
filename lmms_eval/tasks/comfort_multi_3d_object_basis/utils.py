"""COMFORT object-centric direction, vector, and combined prediction tasks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Multi_3D")
AXES = ("front", "up", "right")
DIRECTIONS = ("left", "right", "front", "back")
SOURCE_TO_ANSWER = {
    "left": "left",
    "right": "right",
    "front": "front",
    "behind": "back",
    "back": "back",
}
TASK_BY_FRAME_FORMAT = {
    ("object", "direction"): "comfort_multi_3d_object_basis_direction",
    ("object", "vector"): "comfort_multi_3d_object_basis_vector",
    ("object", "combined"): "comfort_multi_3d_object_basis_combined",
    ("camera", "direction"): "comfort_multi_3d_camera_basis_direction",
    ("camera", "vector"): "comfort_multi_3d_camera_basis_vector",
    ("camera", "combined"): "comfort_multi_3d_camera_basis_combined",
}
PAIR_TASK_BY_FRAME = {
    "object": "comfort_multi_3d_object_basis_object_direction",
    "camera": "comfort_multi_3d_camera_basis_object_direction",
}
EPSILON = 1e-8


def _image_path(scene: dict) -> Path:
    value = Path(str(scene.get("image", "")))
    return value if value.is_absolute() else DATA_ROOT / value


def _position(obj: dict) -> list[float]:
    value = (obj.get("camera_frame") or {}).get("position")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Object {obj.get('object_id')!r} has no 3D camera-frame position")
    return [float(component) for component in value]


def _orientation_columns(obj: dict) -> list[list[float]]:
    matrix = (obj.get("camera_frame") or {}).get("orientation_matrix")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3 or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in matrix):
        raise ValueError(f"Object {obj.get('object_id')!r} has no 3x3 orientation matrix")
    rows = [[float(value) for value in row] for row in matrix]
    return [[rows[row][column] for row in range(3)] for column in range(3)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: list[float]) -> list[float]:
    norm = _norm(vector)
    if norm <= EPSILON:
        raise ValueError("Cannot normalize a zero vector")
    return [value / norm for value in vector]


def _scale(vector: list[float], amount: float) -> list[float]:
    return [amount * value for value in vector]


def _scene_objects(scene: dict) -> tuple[dict, dict[str, dict]]:
    objects = scene.get("objects") or []
    if not isinstance(objects, list):
        raise ValueError("Scene objects must be a list")
    reference = next(
        (obj for obj in objects if isinstance(obj, dict) and (obj.get("role") == "reference" or obj.get("object_id") == "reference")),
        None,
    )
    by_direction = {str(obj.get("reference_direction")): obj for obj in objects if isinstance(obj, dict) and obj.get("reference_direction")}
    if reference is None or not {"left", "right", "front", "behind"} <= set(by_direction):
        raise ValueError("Scene lacks the reference or one of its four surrounding objects")
    return reference, by_direction


def object_basis(scene: dict) -> dict[str, list[float]]:
    """Recover semantic object axes in camera coordinates.

    Orientation-matrix columns are the asset-local orthonormal axes. Different
    assets need not use the same local front/right convention, so the controlled
    surrounding placements identify the two horizontal columns and their signs.
    The right-handed up axis is then right cross front.
    """
    reference, targets = _scene_objects(scene)
    columns = [_unit(column) for column in _orientation_columns(reference)]
    raw_right = _unit(_subtract(_position(targets["right"]), _position(targets["left"])))
    raw_front = _unit(_subtract(_position(targets["front"]), _position(targets["behind"])))

    right_index = max(range(3), key=lambda index: abs(_dot(columns[index], raw_right)))
    remaining = [index for index in range(3) if index != right_index]
    front_index = max(remaining, key=lambda index: abs(_dot(columns[index], raw_front)))

    right = columns[right_index]
    if _dot(right, raw_right) < 0.0:
        right = _scale(right, -1.0)
    front = columns[front_index]
    if _dot(front, raw_front) < 0.0:
        front = _scale(front, -1.0)
    up = _unit(_cross(right, front))

    basis = {"front": front, "up": up, "right": right}
    for axis in AXES:
        if not math.isclose(_norm(basis[axis]), 1.0, abs_tol=1e-5):
            raise ValueError(f"Recovered {axis} axis is not unit length")
    if any(abs(_dot(basis[a], basis[b])) > 1e-4 for a, b in (("front", "up"), ("front", "right"), ("up", "right"))):
        raise ValueError("Recovered object basis is not orthogonal")
    return basis


def object_relative_unit_vector(reference: dict, target: dict, basis: dict[str, list[float]]) -> dict[str, float]:
    displacement = _subtract(_position(target), _position(reference))
    displacement = _unit(displacement)
    vector = {axis: _dot(displacement, basis[axis]) for axis in AXES}
    # Remove floating-point noise without erasing genuine small height offsets.
    return {axis: 0.0 if abs(value) < 1e-10 else float(value) for axis, value in vector.items()}


def camera_relative_unit_vector(reference: dict, target: dict) -> dict[str, float]:
    """Express target-minus-reference in the dataset camera frame."""
    right, front, up = _unit(_subtract(_position(target), _position(reference)))
    return {"front": front, "up": up, "right": right}


def _validate_semantic_basis(reference: dict, targets: dict[str, dict], basis: dict[str, list[float]]) -> None:
    vectors = {direction: object_relative_unit_vector(reference, target, basis) for direction, target in targets.items()}
    checks = {
        "left": vectors["left"]["right"] < 0.0,
        "right": vectors["right"]["right"] > 0.0,
        "front": vectors["front"]["front"] > 0.0,
        "behind": vectors["behind"]["front"] < 0.0,
    }
    if not all(checks.values()):
        raise ValueError(f"Recovered basis disagrees with semantic placements: {checks}")


def _question(reference_label: str, target_label: str, coordinate_frame: str) -> str:
    perspective = f"the {reference_label}'s own frame of reference" if coordinate_frame == "object" else "the camera's frame of reference"
    return f"From {perspective}, what is the 3D direction from the " f"{reference_label} to the {target_label}?"


def _process_docs(dataset: Dataset, prediction_format: str, coordinate_frame: str = "object") -> Dataset:
    if (coordinate_frame, prediction_format) not in TASK_BY_FRAME_FORMAT:
        raise ValueError(f"Unknown frame/format combination: {coordinate_frame}/{prediction_format}")
    records = []
    skipped = Counter()
    for source in dataset:
        scene = dict(source)
        scene_id = str(scene.get("scene_id", "")).strip()
        image_path = _image_path(scene)
        if not scene_id:
            skipped["missing_scene_id"] += 1
            continue
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue
        try:
            reference, targets = _scene_objects(scene)
            basis = object_basis(scene)
            _validate_semantic_basis(reference, targets, basis)
        except (TypeError, ValueError) as error:
            skipped[f"invalid_geometry:{type(error).__name__}"] += 1
            eval_logger.warning("Skipping COMFORT basis scene {}: {}", scene_id, error)
            continue

        reference_label = str(reference.get("label", "")).strip()
        if not reference_label:
            skipped["missing_reference_label"] += 1
            continue
        for source_direction in ("left", "right", "front", "behind"):
            target = targets[source_direction]
            target_label = str(target.get("label", "")).strip()
            if not target_label:
                skipped["missing_target_label"] += 1
                continue
            if coordinate_frame == "object":
                vector = object_relative_unit_vector(reference, target, basis)
                answer = SOURCE_TO_ANSWER[source_direction]
                evaluation_basis = basis
            else:
                vector = camera_relative_unit_vector(reference, target)
                answer = direction_from_vector(vector)
                evaluation_basis = {
                    "front": [0.0, 1.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "right": [1.0, 0.0, 0.0],
                }
            if answer is None:
                skipped["degenerate_horizontal_direction"] += 1
                continue
            qid = f"{scene_id}::{target.get('object_id')}::{coordinate_frame}::" f"{prediction_format}"
            records.append(
                {
                    "qid": qid,
                    "index": qid,
                    "pair_id": f"{scene_id}::{target.get('object_id')}",
                    "scene_id": scene_id,
                    "coordinate_frame": coordinate_frame,
                    "prediction_format": prediction_format,
                    "reference_object": reference_label,
                    "reference_object_id": str(reference.get("object_id", "reference")),
                    "target_object": target_label,
                    "target_object_id": str(target.get("object_id", "")),
                    "source_direction": source_direction,
                    "gt_direction": answer,
                    "gt_vector": vector,
                    "evaluation_basis_camera_frame": evaluation_basis,
                    "object_basis_camera_frame": basis,
                    "question": _question(reference_label, target_label, coordinate_frame),
                    "img_path": str(image_path),
                    "image_path": str(image_path),
                }
            )

    eval_logger.info(
        "COMFORT {}-frame {} task loaded {} object pairs from {} scenes; skipped={}.",
        coordinate_frame,
        prediction_format,
        len(records),
        len({record["scene_id"] for record in records}),
        dict(skipped),
    )
    return Dataset.from_list(records)


def process_direction_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "direction")


def process_vector_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "vector")


def process_combined_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "combined")


def process_camera_direction_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "direction", "camera")


def process_camera_vector_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "vector", "camera")


def process_camera_combined_docs(dataset: Dataset) -> Dataset:
    return _process_docs(dataset, "combined", "camera")


def _process_object_direction_docs(dataset: Dataset, coordinate_frame: str) -> Dataset:
    """Create matched direction-answer and object-answer rows.

    Camera-frame object questions retain only relations with exactly one target
    in that scene. Specified-target direction/vector tasks do not need this
    filter and continue to use every target.
    """
    base_docs = list(_process_docs(dataset, "direction", coordinate_frame))
    by_scene = defaultdict(list)
    for doc in base_docs:
        by_scene[str(doc["scene_id"])].append(doc)

    records = []
    ambiguous_pairs = 0
    for scene_docs in by_scene.values():
        direction_counts = Counter(str(doc["gt_direction"]) for doc in scene_docs)
        candidates = [str(doc["target_object"]) for doc in scene_docs]
        for source in scene_docs:
            if coordinate_frame == "camera" and direction_counts[str(source["gt_direction"])] != 1:
                ambiguous_pairs += 1
                continue
            pair_id = f"{source['pair_id']}::{coordinate_frame}::object_direction"
            perspective = f"the {source['reference_object']}'s own frame of reference" if coordinate_frame == "object" else "the camera's frame of reference"
            shared = {
                **source,
                "prediction_format": "object_direction",
                "pair_id": pair_id,
                "candidate_objects": candidates,
            }
            records.extend(
                [
                    {
                        **shared,
                        "qid": f"{pair_id}::direction",
                        "index": f"{pair_id}::direction",
                        "answer_format": "direction",
                        "gold_answer": source["gt_direction"],
                        "question": (f"From {perspective}, where is the {source['target_object']} " f"relative to the {source['reference_object']}?"),
                    },
                    {
                        **shared,
                        "qid": f"{pair_id}::object",
                        "index": f"{pair_id}::object",
                        "answer_format": "object",
                        "gold_answer": source["target_object"],
                        "question": (f"From {perspective}, which surrounding object is " f"{source['gt_direction']} of the {source['reference_object']}?"),
                    },
                ]
            )

    eval_logger.info(
        "COMFORT {}-frame object/direction task loaded {} matched pairs ({} rows); " "excluded {} ambiguous camera pairs.",
        coordinate_frame,
        len(records) // 2,
        len(records),
        ambiguous_pairs,
    )
    return Dataset.from_list(records)


def process_object_basis_object_direction_docs(dataset: Dataset) -> Dataset:
    return _process_object_direction_docs(dataset, "object")


def process_camera_basis_object_direction_docs(dataset: Dataset) -> Dataset:
    return _process_object_direction_docs(dataset, "camera")


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path") or doc.get("image_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT image not found for {doc.get('qid')}: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def _frame_instructions(doc: dict) -> str:
    if doc.get("coordinate_frame") == "camera":
        return (
            f"The central reference object is the {doc['reference_object']}; the target is "
            f"the {doc['target_object']}. Use the camera's orthonormal frame: positive "
            "front follows the camera viewing direction, positive right is camera/image "
            "right, and positive up is camera/image up. Negative front means back toward "
            "the camera, negative right means camera-left, and negative up means below."
        )
    return (
        f"The central reference object is the {doc['reference_object']}; the target is "
        f"the {doc['target_object']}. Use the reference object's own orthonormal frame: "
        "positive front is where it faces, positive right is its own right, and positive "
        "up is its own up. Negative front means back, negative right means left, and "
        "negative up means below."
    )


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    prediction_format = doc["prediction_format"]
    if prediction_format == "object_direction":
        frame_text = "Use the central reference object's own front/right axes, not the camera axes." if doc.get("coordinate_frame") == "object" else "Use camera/image right and the camera viewing direction, not the central object's axes."
        if doc["answer_format"] == "direction":
            output_rule = "Return exactly one lowercase direction word: left, right, front, or back."
        else:
            candidates = ", ".join(str(item) for item in doc["candidate_objects"])
            output_rule = f"Return exactly one object name from this candidate list: {candidates}."
        return f"{kwargs.get('pre_prompt', '')}{doc['question']}\n" f"{frame_text}\n{output_rule}{kwargs.get('post_prompt', '')}"
    if prediction_format == "direction":
        output_rule = "Return exactly one lowercase direction word: left, right, front, or back. " "Use the dominant horizontal component and ignore up/down for this word."
    elif prediction_format == "vector":
        output_rule = "Return only valid JSON with the target-minus-reference unit direction: " '{"front": <float>, "up": <float>, "right": <float>}. ' "All three components must be present."
    else:
        output_rule = (
            "Return only valid JSON containing both predictions: "
            '{"answer": "<left|right|front|back>", "relative_vector": '
            '{"front": <float>, "up": <float>, "right": <float>}}. '
            "The relative_vector is the target-minus-reference unit direction."
        )
    return f"{kwargs.get('pre_prompt', '')}{doc['question']}\n" f"{_frame_instructions(doc)}\n{output_rule}" f"{kwargs.get('post_prompt', '')}"


def doc_to_target(doc):
    if doc["prediction_format"] == "object_direction":
        return str(doc["gold_answer"])
    vector = {axis: round(float(doc["gt_vector"][axis]), 6) for axis in AXES}
    if doc["prediction_format"] == "direction":
        return str(doc["gt_direction"])
    if doc["prediction_format"] == "vector":
        return json.dumps(vector)
    return json.dumps({"answer": doc["gt_direction"], "relative_vector": vector})


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates = [str(text).strip()]
    match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_direction(text: str) -> Optional[str]:
    payload = _extract_json(text)
    candidates = []
    if payload is not None:
        candidates.extend((payload.get("answer"), payload.get("direction")))
    candidates.append(text)
    aliases = {"behind": "back", "backward": "back", "forwards": "front", "forward": "front"}
    for candidate in candidates:
        if candidate is None:
            continue
        match = re.search(
            r"(?<![A-Za-z])(left|right|front|back|behind|backward|forward|forwards)(?![A-Za-z])",
            str(candidate),
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).lower()
            return aliases.get(value, value)
    return None


def _normalize_answer(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_object_answer(text: str, candidates: list[str]) -> Optional[str]:
    """Resolve one candidate name from a plain-text or JSON answer."""
    if not text:
        return None
    payload = _extract_json(text)
    values = []
    if payload is not None:
        values.extend(
            (
                payload.get("answer"),
                payload.get("object"),
                payload.get("target_object"),
            )
        )
    values.append(text)
    normalized_candidates = {_normalize_answer(candidate): str(candidate) for candidate in candidates}
    for value in values:
        normalized = _normalize_answer(value)
        if normalized in normalized_candidates:
            return normalized_candidates[normalized]
        matches = [original for candidate, original in normalized_candidates.items() if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", normalized)]
        if len(matches) == 1:
            return matches[0]
    return None


def _finite_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_vector(text: str) -> Optional[dict[str, float]]:
    payload = _extract_json(text)
    if payload is None:
        return None
    raw = payload.get("relative_vector", payload.get("vector", payload))
    if not isinstance(raw, dict):
        return None
    vector = {axis: _finite_float(raw.get(axis)) for axis in AXES}
    if any(vector[axis] is None for axis in AXES):
        return None
    if math.sqrt(sum(float(vector[axis]) ** 2 for axis in AXES)) <= EPSILON:
        return None
    return {axis: float(vector[axis]) for axis in AXES}


def _unit_dict(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(float(vector[axis]) ** 2 for axis in AXES))
    if norm <= EPSILON:
        return {axis: 0.0 for axis in AXES}
    return {axis: float(vector[axis]) / norm for axis in AXES}


def _cosine(prediction: dict[str, float], target: dict[str, float]) -> float:
    pred_unit = _unit_dict(prediction)
    target_unit = _unit_dict(target)
    return sum(pred_unit[axis] * target_unit[axis] for axis in AXES)


def _l2_score(prediction: dict[str, float], target: dict[str, float]) -> float:
    pred_unit = _unit_dict(prediction)
    target_unit = _unit_dict(target)
    distance = math.sqrt(sum((pred_unit[axis] - target_unit[axis]) ** 2 for axis in AXES))
    return max(0.0, 1.0 - distance / 2.0)


def direction_from_vector(vector: dict[str, float]) -> Optional[str]:
    right = float(vector.get("right", 0.0))
    front = float(vector.get("front", 0.0))
    if abs(right) <= EPSILON and abs(front) <= EPSILON:
        return None
    if abs(right) >= abs(front):
        return "right" if right > 0.0 else "left"
    return "front" if front > 0.0 else "back"


def _sign(value: float, tolerance: float = 1e-4) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _score_entry(doc: dict, prediction: str) -> dict:
    prediction_format = str(doc["prediction_format"])
    parsed_direction = parse_direction(prediction) if prediction_format in {"direction", "combined"} else None
    parsed_vector = parse_vector(prediction) if prediction_format in {"vector", "combined"} else None
    gt_direction = str(doc["gt_direction"])
    gt_vector = {axis: float(doc["gt_vector"][axis]) for axis in AXES}
    direction_parse_success = parsed_direction is not None
    vector_parse_success = parsed_vector is not None
    direction_accuracy = float(parsed_direction == gt_direction)
    vector_cosine = _cosine(parsed_vector, gt_vector) if parsed_vector is not None else 0.0
    vector_l2_score = _l2_score(parsed_vector, gt_vector) if parsed_vector is not None else 0.0
    vector_angle_30_accuracy = float(vector_parse_success and vector_cosine >= math.cos(math.radians(30.0)))
    predicted_vector_direction = direction_from_vector(parsed_vector or {})
    vector_dominant_direction_accuracy = float(predicted_vector_direction == gt_direction)
    sign_scores = {}
    for axis in AXES:
        sign_scores[axis] = float(parsed_vector is not None and _sign(float(parsed_vector[axis])) == _sign(round(float(gt_vector[axis]), 6)))
    full_sign_accuracy = float(vector_parse_success and all(sign_scores[axis] == 1.0 for axis in AXES))
    combined_parse_success = float(direction_parse_success and vector_parse_success)
    valid_combined = prediction_format == "combined" and combined_parse_success == 1.0
    both_correct = float(valid_combined and direction_accuracy == 1.0 and vector_dominant_direction_accuracy == 1.0)
    direction_correct_vector_wrong = float(valid_combined and direction_accuracy == 1.0 and vector_dominant_direction_accuracy == 0.0)
    direction_wrong_vector_correct = float(valid_combined and direction_accuracy == 0.0 and vector_dominant_direction_accuracy == 1.0)
    both_wrong = float(valid_combined and direction_accuracy == 0.0 and vector_dominant_direction_accuracy == 0.0)
    combined_parse_failure = float(prediction_format == "combined" and not valid_combined)
    return {
        "qid": doc.get("qid"),
        "pair_id": doc.get("pair_id"),
        "scene_id": doc.get("scene_id"),
        "coordinate_frame": doc.get("coordinate_frame", "object"),
        "prediction_format": prediction_format,
        "reference_object": doc.get("reference_object"),
        "target_object": doc.get("target_object"),
        "source_direction": doc.get("source_direction"),
        "gt_direction": gt_direction,
        "gt_vector": gt_vector,
        "prediction": prediction,
        "parsed_direction": parsed_direction,
        "parsed_vector": parsed_vector,
        "predicted_vector_direction": predicted_vector_direction,
        "direction_parse_success": float(direction_parse_success),
        "vector_parse_success": float(vector_parse_success),
        "combined_parse_success": combined_parse_success,
        "combined_parse_failure": combined_parse_failure,
        "direction_accuracy": direction_accuracy,
        "vector_cosine": vector_cosine,
        "vector_l2_score": vector_l2_score,
        "vector_angle_30_accuracy": vector_angle_30_accuracy,
        "vector_dominant_direction_accuracy": vector_dominant_direction_accuracy,
        "front_sign_accuracy": sign_scores["front"],
        "up_sign_accuracy": sign_scores["up"],
        "right_sign_accuracy": sign_scores["right"],
        "full_sign_accuracy": full_sign_accuracy,
        "both_correct": both_correct,
        "direction_correct_vector_wrong": direction_correct_vector_wrong,
        "direction_wrong_vector_correct": direction_wrong_vector_correct,
        "both_wrong": both_wrong,
    }


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    entry = _score_entry(doc, prediction)
    metric_names = (
        "basis_direction_accuracy",
        "basis_direction_parse_success",
        "basis_vector_cosine",
        "basis_vector_l2_score",
        "basis_vector_angle_30_accuracy",
        "basis_vector_dominant_direction_accuracy",
        "basis_front_sign_accuracy",
        "basis_up_sign_accuracy",
        "basis_right_sign_accuracy",
        "basis_full_sign_accuracy",
        "basis_vector_parse_success",
        "basis_combined_parse_success",
        "basis_combined_parse_failure",
        "basis_both_correct",
        "basis_direction_correct_vector_wrong",
        "basis_direction_wrong_vector_correct",
        "basis_both_wrong",
    )
    output = {name: dict(entry) for name in metric_names}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "target": doc_to_target(doc),
        "img_path": doc.get("img_path"),
        "evaluation_basis_camera_frame": doc.get("evaluation_basis_camera_frame"),
        "object_basis_camera_frame": doc.get("object_basis_camera_frame"),
    }
    return output


def process_object_direction_results(doc, results):
    prediction = results[0].strip() if results else ""
    answer_format = str(doc["answer_format"])
    if answer_format == "direction":
        parsed = parse_direction(prediction)
    else:
        parsed = parse_object_answer(prediction, list(doc["candidate_objects"]))
    gold = str(doc["gold_answer"])
    entry = {
        "qid": doc.get("qid"),
        "pair_id": doc.get("pair_id"),
        "scene_id": doc.get("scene_id"),
        "coordinate_frame": doc.get("coordinate_frame"),
        "prediction_format": "object_direction",
        "answer_format": answer_format,
        "reference_object": doc.get("reference_object"),
        "target_object": doc.get("target_object"),
        "gt_direction": doc.get("gt_direction"),
        "gold_answer": gold,
        "prediction": prediction,
        "parsed_answer": parsed,
        "parse_success": float(parsed is not None),
        "answer_accuracy": float(_normalize_answer(parsed) == _normalize_answer(gold)),
    }
    metric_names = (
        "basis_format_accuracy",
        "basis_object_answer_accuracy",
        "basis_direction_answer_accuracy",
        "basis_object_minus_direction",
        "basis_format_switch_gain",
        "basis_format_parse_success",
        "basis_object_parse_success",
        "basis_direction_parse_success",
        "basis_object_correct_direction_wrong",
        "basis_direction_correct_object_wrong",
        "basis_pair_both_correct",
        "basis_pair_both_wrong",
    )
    output = {name: dict(entry) for name in metric_names}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "target": doc_to_target(doc),
        "candidate_objects": list(doc["candidate_objects"]),
        "img_path": doc.get("img_path"),
    }
    return output


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _aggregate(results, field: str) -> float:
    return _mean(float(row.get(field, 0.0)) for row in results)


def aggregate_basis_direction_accuracy(results):
    score = _aggregate(results, "direction_accuracy")
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row.get("gt_direction"))].append(float(row.get("direction_accuracy", 0.0)))
    eval_logger.info(
        "COMFORT basis direction accuracy {:.4f}; by direction={}",
        score,
        {key: _mean(values) for key, values in sorted(grouped.items())},
    )
    return score


def aggregate_basis_direction_parse_success(results):
    return _aggregate(results, "direction_parse_success")


def aggregate_basis_vector_cosine(results):
    return _aggregate(results, "vector_cosine")


def aggregate_basis_vector_l2_score(results):
    return _aggregate(results, "vector_l2_score")


def aggregate_basis_vector_angle_30_accuracy(results):
    return _aggregate(results, "vector_angle_30_accuracy")


def aggregate_basis_vector_dominant_direction_accuracy(results):
    return _aggregate(results, "vector_dominant_direction_accuracy")


def aggregate_basis_front_sign_accuracy(results):
    return _aggregate(results, "front_sign_accuracy")


def aggregate_basis_up_sign_accuracy(results):
    return _aggregate(results, "up_sign_accuracy")


def aggregate_basis_right_sign_accuracy(results):
    return _aggregate(results, "right_sign_accuracy")


def aggregate_basis_full_sign_accuracy(results):
    return _aggregate(results, "full_sign_accuracy")


def aggregate_basis_vector_parse_success(results):
    return _aggregate(results, "vector_parse_success")


def aggregate_basis_combined_parse_success(results):
    return _aggregate(results, "combined_parse_success")


def aggregate_basis_combined_parse_failure(results):
    return _aggregate(results, "combined_parse_failure")


def aggregate_basis_both_correct(results):
    return _aggregate(results, "both_correct")


def aggregate_basis_direction_correct_vector_wrong(results):
    return _aggregate(results, "direction_correct_vector_wrong")


def aggregate_basis_direction_wrong_vector_correct(results):
    return _aggregate(results, "direction_wrong_vector_correct")


def aggregate_basis_both_wrong(results):
    return _aggregate(results, "both_wrong")


def aggregate_basis_format_accuracy(results):
    return _aggregate(results, "answer_accuracy")


def aggregate_basis_object_answer_accuracy(results):
    return _mean(float(row["answer_accuracy"]) for row in results if row.get("answer_format") == "object")


def aggregate_basis_direction_answer_accuracy(results):
    return _mean(float(row["answer_accuracy"]) for row in results if row.get("answer_format") == "direction")


def _matched_format_pairs(results) -> list[dict[str, dict]]:
    grouped = defaultdict(dict)
    for row in results:
        answer_format = str(row.get("answer_format", ""))
        if answer_format in {"object", "direction"}:
            grouped[str(row.get("pair_id"))][answer_format] = row
    return [pair for pair in grouped.values() if {"object", "direction"} <= set(pair)]


def aggregate_basis_object_minus_direction(results):
    return _mean(float(pair["object"]["answer_accuracy"]) - float(pair["direction"]["answer_accuracy"]) for pair in _matched_format_pairs(results))


def aggregate_basis_format_switch_gain(results):
    return aggregate_basis_object_minus_direction(results)


def aggregate_basis_format_parse_success(results):
    return _aggregate(results, "parse_success")


def aggregate_basis_object_parse_success(results):
    return _mean(float(row["parse_success"]) for row in results if row.get("answer_format") == "object")


def aggregate_basis_pair_direction_parse_success(results):
    return _mean(float(row["parse_success"]) for row in results if row.get("answer_format") == "direction")


def _aggregate_pair_outcome(results, object_score: float, direction_score: float):
    return _mean(float(float(pair["object"]["answer_accuracy"]) == object_score and float(pair["direction"]["answer_accuracy"]) == direction_score) for pair in _matched_format_pairs(results))


def aggregate_basis_object_correct_direction_wrong(results):
    return _aggregate_pair_outcome(results, 1.0, 0.0)


def aggregate_basis_direction_correct_object_wrong(results):
    return _aggregate_pair_outcome(results, 0.0, 1.0)


def aggregate_basis_pair_both_correct(results):
    return _aggregate_pair_outcome(results, 1.0, 1.0)


def aggregate_basis_pair_both_wrong(results):
    return _aggregate_pair_outcome(results, 0.0, 0.0)


def aggregate_object_direction_submission(results, args):
    coordinate_frame = str(results[0].get("coordinate_frame", "object")) if results else "object"
    task = PAIR_TASK_BY_FRAME.get(coordinate_frame, "comfort_multi_3d_basis_object_direction")
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{task}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": task,
        "coordinate_frame": coordinate_frame,
        "prediction_format": "object_direction",
        "num_records": len(results),
        "num_matched_pairs": len(_matched_format_pairs(results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info(
        "COMFORT {}-frame object/direction records saved to {}.",
        coordinate_frame,
        path,
    )


def aggregate_results_for_submission(results, args):
    prediction_format = str(results[0].get("prediction_format", "unknown")) if results else "unknown"
    coordinate_frame = str(results[0].get("coordinate_frame", "object")) if results else "object"
    task = TASK_BY_FRAME_FORMAT.get((coordinate_frame, prediction_format), "comfort_multi_3d_basis")
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{task}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": task,
        "prediction_format": prediction_format,
        "coordinate_frame": coordinate_frame,
        "coordinate_frame_definition": ("central reference object's own front/up/right basis" if coordinate_frame == "object" else "camera front/up/right basis"),
        "vector_definition": "unit target-minus-reference direction with keys front, up, right",
        "num_records": len(results),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info(
        "COMFORT {}-frame {} records saved to {}.",
        coordinate_frame,
        prediction_format,
        path,
    )
