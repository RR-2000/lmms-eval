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
    system = doc.get("coordinate_system") or {}
    right = system.get("right_axis_world_xy")
    front = system.get("front_axis_world_xy")
    if right is None or front is None:
        return "The vector axes are right, up, and front."
    return (
        "Use the supplied viewpoint frame: "
        f"right axis in world XY={right}, front axis in world XY={front}, and up=world +Z."
    )


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return "\n".join(
        [
            str(doc["question"]),
            _coordinate_text(doc),
            "Return only valid JSON. The vector should not be a 0 vector.",
            "`between_objects` must be variable-object position minus reference-object position in this viewpoint frame.",
            'JSON schema: {"answer":"<left|right|front|behind>","between_objects":{"right":<float>,"up":<float>,"front":<float>}}',
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
