"""Structured viewpoint evaluation for the generated COMFORT dataset."""

import json
import math
import re
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


AXES = ("right", "up", "front")
RELATION_AXIS = {"left": "right", "right": "right", "front": "front", "behind": "front"}
RELATION_SIGN = {"left": -1, "right": 1, "front": 1, "behind": -1}


def process_reference_and_addressee_orientation_docs(dataset):
    """Retain directions expressed in the reference or addressee GT frame.

    For these source records, ``vector_between_objects`` is already the
    variable-minus-reference displacement rotated using the current observer
    asset's ground-truth pose (rather than a camera-facing convention).  The
    matching scene ``config.json`` contains the original object positions and
    rotations used to render the record.
    """
    observer_docs = dataset.filter(
        lambda doc: doc.get("viewpoint") in {"reference", "addressee"}
    )
    return observer_docs.map(
        lambda doc: {
            **doc,
            "vector_frame": "observer_ground_truth_orientation",
        }
    )


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path") or doc.get("image") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT image not found for {doc.get('qid')}: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def _vector(doc) -> dict[str, float]:
    vector = doc.get("vector_between_objects") or {}
    return {axis: float(vector.get(axis, 0.0)) for axis in AXES}


def _coordinate_text(doc) -> str:
    if doc.get("vector_frame") == "observer_ground_truth_orientation":
        reference = doc.get("reference_object") or {}
        observer = reference if doc.get("viewpoint") == "reference" else (doc.get("addressee_object") or {})
        return (
            f"The coordinate-system origin is the {reference.get('name', 'reference object')} center. "
            f"Its axes are the {observer.get('name', 'observer')}'s ground-truth right, front-facing, and up directions."
        )
    system = doc.get("coordinate_system") or {}
    Origin = system.get("origin")
    right = system.get("right")
    front = system.get("front")
    up = system.get("up")
    if Origin and right and front and up:
        return (
            f"The coordinate system has origin at {Origin}, "
            f"right vector {right}, front vector {front}, and up vector {up}."
        )
    else:
        return "The vector axes are right, up, and front."


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    observer = (doc.get("reference_object") or {}) if doc.get("viewpoint") == "reference" else (doc.get("addressee_object") or {})
    vector_instruction = (
        "Return a `between_objects` vector, it must be variable-object position minus reference-object position "
        f"expressed in {observer.get('name', 'the observer')}'s ground-truth orientation frame. "
        "Use the observer's own right, front (facing), and up axes; do not orient it toward the camera."
        if doc.get("vector_frame") == "observer_ground_truth_orientation"
        else "Return a `between_objects` vector, it must be variable-object position minus reference-object position in the camera frame."
    )
    return "\n".join(
        [
            str(doc["question"]),
            _coordinate_text(doc),
            "Return only valid JSON.",
            vector_instruction,
            "The `between_objects` field is mandatory and is heavily evaluated. Never omit it and never use [0, 0, 0] unless the two objects are exactly coincident.",
            'JSON schema: {"answer":"<left|right|front|behind>","between_objects":{"right":<float>,"front":<float>,"up":<float>}}',
        ]
    )


def doc_to_target(doc):
    gold_answer = str(doc["answer"]).lower()
    return json.dumps(
        {
            "answer": gold_answer,
            "between_objects": _vector(doc),
            "relation_axis": doc.get("target_relation_axis", RELATION_AXIS[gold_answer]),
            "relation_axis_sign": doc.get("target_relation_axis_sign", RELATION_SIGN[gold_answer]),
        },
        ensure_ascii=True,
    )


def _payload(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _prediction(text: str) -> tuple[str, dict[str, float], dict]:
    payload = _payload(text)
    raw_vector = payload.get("between_objects", payload.get("relative_vector", {}))
    raw_vector = raw_vector if isinstance(raw_vector, dict) else {}
    vector = {axis: _number(raw_vector.get(axis, raw_vector.get({"right": "x", "up": "y", "front": "z"}[axis]))) for axis in AXES}
    return str(payload.get("answer", "")).strip().lower(), vector, payload


def _sign(value: float) -> int:
    return 1 if value > 1e-8 else -1 if value < -1e-8 else 0


def _cosine(first: dict[str, float], second: dict[str, float]) -> float:
    first_norm = math.sqrt(sum(first[axis] ** 2 for axis in AXES))
    second_norm = math.sqrt(sum(second[axis] ** 2 for axis in AXES))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return 0.0
    return sum(first[axis] * second[axis] for axis in AXES) / (first_norm * second_norm)


def process_results(doc, results):
    raw = results[0].strip() if results else ""
    answer, predicted_vector, payload = _prediction(raw)
    gold_answer = str(doc["answer"]).strip().lower()
    gold_vector = _vector(doc)
    answer_accuracy = float(answer == gold_answer)
    relation_axis = doc.get("target_relation_axis", RELATION_AXIS[gold_answer])
    relation_sign = float(doc.get("target_relation_axis_sign", RELATION_SIGN[gold_answer]))
    relation_axis_accuracy = float(
        _sign(predicted_vector[relation_axis]) == relation_sign
    )
    vector_cosine = _cosine(predicted_vector, gold_vector)
    entry = {
        "qid": doc.get("qid"), "scene": doc.get("scene"), "variation": doc.get("variation"),
        "viewpoint": doc.get("viewpoint"), "frame_deviation_degrees": doc.get("frame_deviation_degrees"),
        "answer_accuracy": answer_accuracy, "relation_axis_accuracy": relation_axis_accuracy,
        "vector_cosine": vector_cosine,
        "answer_and_direction_correct": float(answer_accuracy and relation_axis_accuracy),
        "answer_correct_direction_wrong": float(answer_accuracy and not relation_axis_accuracy),
        "answer_wrong_direction_correct": float(not answer_accuracy and relation_axis_accuracy),
        "answer_and_direction_wrong": float(not answer_accuracy and not relation_axis_accuracy),
    }
    result = {
        "comfort_answer_accuracy": entry, "comfort_relation_axis_accuracy": entry,
        "comfort_vector_cosine": entry, "comfort_answer_and_direction_correct": entry,
        "comfort_answer_correct_direction_wrong": entry, "comfort_answer_wrong_direction_correct": entry,
        "comfort_answer_and_direction_wrong": entry, "comfort_camera_answer_accuracy": entry,
        "comfort_reference_answer_accuracy": entry, "comfort_addressee_answer_accuracy": entry,
        "submission": {**entry, "question_prompt": doc_to_text(doc), "img_path": doc.get("img_path"),
                       "prediction": raw, "parsed_prediction": payload, "gold_answer": gold_answer,
                       "gold_between_objects": gold_vector},
    }
    return result


def _mean(results, key, viewpoint: Optional[str] = None):
    values = [float(row[key]) for row in results if viewpoint is None or row.get("viewpoint") == viewpoint]
    return sum(values) / len(values) if values else 0.0


def aggregate_answer_accuracy(results): return _mean(results, "answer_accuracy")
def aggregate_relation_axis_accuracy(results): return _mean(results, "relation_axis_accuracy")
def aggregate_vector_cosine(results): return _mean(results, "vector_cosine")
def aggregate_answer_and_direction_correct(results): return _mean(results, "answer_and_direction_correct")
def aggregate_answer_correct_direction_wrong(results): return _mean(results, "answer_correct_direction_wrong")
def aggregate_answer_wrong_direction_correct(results): return _mean(results, "answer_wrong_direction_correct")
def aggregate_answer_and_direction_wrong(results): return _mean(results, "answer_and_direction_wrong")
def aggregate_camera_answer_accuracy(results): return _mean(results, "answer_accuracy", "camera")
def aggregate_reference_answer_accuracy(results): return _mean(results, "answer_accuracy", "reference")
def aggregate_addressee_answer_accuracy(results): return _mean(results, "answer_accuracy", "addressee")


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"comfort_viewpoint_{model}.json", args)
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    eval_logger.info("COMFORT viewpoint records saved to {}.", path)
