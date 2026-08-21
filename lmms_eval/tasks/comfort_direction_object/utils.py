"""Matched direction-answer versus object-answer diagnostic for COMFORT."""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


OPTION_LETTERS = tuple("AB")
DIRECTIONS = ("left", "right", "front", "behind")
ANSWER_FORMATS = ("object", "direction")
SAMPLE_SEED = "comfort_direction_object_v1"


def _name(doc: dict, field: str) -> str:
    value = doc.get(field) or {}
    return str(value.get("name", "")).strip() if isinstance(value, dict) else ""


def _normalize(value: str) -> str:
    return " ".join(str(value).lower().split())


def _relation_phrase(relation: str) -> str:
    return {
        "left": "to the left of",
        "right": "to the right of",
        "front": "in front of",
        "behind": "behind",
    }[relation]


def _object_question(doc: dict, relation: str, reference: str) -> str:
    viewpoint = str(doc.get("viewpoint", "camera"))
    if viewpoint == "camera":
        lead = f"From the camera's perspective, which object is {_relation_phrase(relation)} the {reference}?"
        convention = (
            "Use the camera-view convention: left/right mean image-left/image-right; "
            "front means farther from the camera along its viewing direction; behind "
            "means closer to the camera."
        )
    elif viewpoint == "reference":
        lead = f"From the perspective of the {reference}, which object is {_relation_phrase(relation)} the {reference}?"
        convention = (
            "Use the reference-view convention: right/left are the reference's own "
            "right/left, front is the direction the reference faces, and behind is "
            "the opposite direction."
        )
    elif viewpoint == "addressee":
        addressee = _name(doc, "addressee_object") or "addressee"
        lead = f"From the perspective of the {addressee}, which object is {_relation_phrase(relation)} the {reference}?"
        convention = (
            "Use the addressee-view convention: right/left are the addressee's own "
            "right/left, front is the direction the addressee faces, and behind is "
            "the opposite direction."
        )
    else:
        raise ValueError(f"Unknown COMFORT viewpoint: {viewpoint}")
    return f"{lead} {convention}"


def _unalign_position(value) -> Optional[tuple[float, float, float]]:
    """Undo make_COMFORT.py's [-world_y, world_x, world_z] transform."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return float(value[1]), -float(value[0]), float(value[2])


def _unit(vector: tuple[float, float]) -> Optional[tuple[float, float]]:
    length = math.hypot(*vector)
    if length <= 1e-8:
        return None
    return vector[0] / length, vector[1] / length


def _viewpoint_axes(doc: dict):
    """Reconstruct the planar axes used by the COMFORT dataset builder."""
    viewpoint = str(doc.get("viewpoint", ""))
    scene = str(doc.get("scene", ""))
    if viewpoint == "camera":
        camera = _unalign_position(doc.get("camera_position_world"))
        reference = _unalign_position((doc.get("reference_object") or {}).get("position_world"))
        if camera is None or reference is None:
            return None
        front = _unit((reference[0] - camera[0], reference[1] - camera[1]))
        return ((front[1], -front[0]), front) if front else None
    if scene == "comfort_car_ref_facing_left":
        return ((-1.0, 0.0), (0.0, -1.0)) if viewpoint == "reference" else ((1.0, 0.0), (0.0, 1.0))
    if scene == "comfort_car_ref_facing_right" and viewpoint in {"reference", "addressee"}:
        return (1.0, 0.0), (0.0, 1.0)
    return None


def _addressee_relation(doc: dict) -> Optional[str]:
    reference = _unalign_position((doc.get("reference_object") or {}).get("position_world"))
    addressee = _unalign_position((doc.get("addressee_object") or {}).get("position_world"))
    axes = _viewpoint_axes(doc)
    if reference is None or addressee is None or axes is None:
        return None
    delta = (addressee[0] - reference[0], addressee[1] - reference[1])
    right_axis, front_axis = axes
    right = delta[0] * right_axis[0] + delta[1] * right_axis[1]
    front = delta[0] * front_axis[0] + delta[1] * front_axis[1]
    if abs(right) >= abs(front):
        return "right" if right >= 0.0 else "left"
    return "front" if front >= 0.0 else "behind"


def _set_options(record: dict, values: list[str], answer: str) -> None:
    if len(values) != 2 or len({_normalize(value) for value in values}) != 2:
        raise ValueError(f"Expected two unique options, got {values}")
    record["options"] = values
    record["num_options"] = len(values)
    record["gold_option_letter"] = OPTION_LETTERS[values.index(answer)]
    record["answer"] = answer
    record["answer_idx"] = values.index(answer)


def process_docs(dataset: Dataset) -> Dataset:
    """Turn every valid COMFORT row into a matched direction/object pair."""
    docs = list(dataset)
    records = []
    skipped = Counter()

    for doc in docs:
        source_qid = str(doc.get("qid", doc.get("index", ""))).strip()
        target = _name(doc, "variable_object")
        reference = _name(doc, "reference_object")
        addressee = _name(doc, "addressee_object")
        relation = str(doc.get("target_relation", doc.get("answer", ""))).strip().lower()
        image_path = str(doc.get("img_path", doc.get("image", ""))).strip()
        if not source_qid or not target or not reference:
            skipped["missing_pair_metadata"] += 1
            continue
        if not addressee:
            skipped["not_three_objects"] += 1
            continue
        if len({_normalize(target), _normalize(reference), _normalize(addressee)}) != 3:
            skipped["not_three_distinct_objects"] += 1
            continue
        if relation not in DIRECTIONS:
            skipped["unsupported_relation"] += 1
            continue
        if not image_path:
            skipped["missing_image_path"] += 1
            continue
        if str(doc.get("viewpoint", "camera")) not in {"camera"}:
            skipped["unsupported_viewpoint"] += 1
            continue

        competing_relation = _addressee_relation(doc)
        if competing_relation is None:
            skipped["missing_addressee_geometry"] += 1
            continue
        if competing_relation == relation:
            skipped["ambiguous_relation"] += 1
            continue

        # Only show the two non-anchor objects. Keep the direction task equally
        # sized by retaining the gold direction and one randomly selected
        # incorrect direction. Seed each choice and shuffle for reproducibility.
        object_options = [target, addressee]
        object_rng = random.Random(f"{SAMPLE_SEED}:{source_qid}:object")
        object_rng.shuffle(object_options)
        direction_rng = random.Random(f"{SAMPLE_SEED}:{source_qid}:direction")
        distractor = direction_rng.choice([candidate for candidate in DIRECTIONS if candidate != relation])
        direction_options = [relation, distractor]
        direction_rng.shuffle(direction_options)
        common = dict(doc)
        common.update(
            {
                "source_qid": source_qid,
                "source_relation_id": source_qid,
                "source_task_family": doc.get("task_family"),
                "task_family": "comfort_direction_object",
                "diagnostic_anchor": reference,
                "diagnostic_target_object": target,
                "diagnostic_relation": relation,
                "diagnostic_competing_object": addressee,
                "diagnostic_competing_object_relation": competing_relation,
                "diagnostic_sample_seed": SAMPLE_SEED,
                "img_path": image_path,
                "image": image_path,
            }
        )

        direction = dict(common)
        direction.update(
            {
                "qid": f"{source_qid}::direction",
                "index": f"{source_qid}::direction",
                "diagnostic_variant": "direction",
                "diagnostic_answer_format": "direction",
                "diagnostic_question": str(doc.get("question", "")),
            }
        )
        _set_options(direction, direction_options, relation)
        records.append(direction)

        object_record = dict(common)
        object_record.update(
            {
                "qid": f"{source_qid}::object",
                "index": f"{source_qid}::object",
                "diagnostic_variant": "object",
                "diagnostic_answer_format": "object",
                "diagnostic_question": _object_question(doc, relation, reference),
            }
        )
        _set_options(object_record, object_options, target)
        records.append(object_record)

    eval_logger.info(
        "COMFORT direction/object created {} matched pairs ({} examples; seed={}); skipped={}.",
        len(records) // 2,
        len(records),
        SAMPLE_SEED,
        dict(skipped),
    )
    return Dataset.from_list(records)


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path") or doc.get("image") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT image not found for {doc.get('qid')}: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    options = "\n".join(
        f"{letter}. {value}"
        for letter, value in zip(OPTION_LETTERS, doc["options"])
    )
    return (
        f"{kwargs.get('pre_prompt', '')}Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with its letter.\n"
        f"Question: {doc['diagnostic_question']}\nOptions:\n{options}"
        f"{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return str(doc["gold_option_letter"])


def extract_option_letter(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in (
        r"^\s*([A-B])(?:[.\s):]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-B])\b",
        r"\(([A-B])\)",
    ):
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _entry(doc, prediction: str, parsed: Optional[str]) -> dict:
    gold = str(doc["gold_option_letter"]).upper()
    return {
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_relation_id": doc.get("source_relation_id"),
        "scene": doc.get("scene"),
        "variation": doc.get("variation"),
        "viewpoint": doc.get("viewpoint"),
        "frame_deviation_degrees": doc.get("frame_deviation_degrees"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "anchor": doc.get("diagnostic_anchor"),
        "target": doc.get("diagnostic_target_object"),
        "relation": doc.get("diagnostic_relation"),
        "num_options": int(doc.get("num_options", 0)),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
        "prediction": prediction,
    }


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = extract_option_letter(prediction)
    entry = _entry(doc, prediction, parsed)
    metrics = (
        "accuracy",
        "object_answer_accuracy",
        "direction_answer_accuracy",
        "object_minus_direction",
        "format_switch_gain",
        "parse_success_rate",
        "object_parse_success_rate",
        "direction_parse_success_rate",
        "object_correct_direction_wrong",
        "direction_correct_object_wrong",
        "both_correct",
        "both_wrong",
        "camera_accuracy",
        "reference_accuracy",
        "addressee_accuracy",
    )
    result = {metric: dict(entry) for metric in metrics}
    result["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "img_path": doc.get("img_path"),
        "image_path": doc.get("img_path"),
        "options": list(doc["options"]),
        "gold_answer": doc.get("answer"),
    }
    return result


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    _log_report(results)
    return _mean(row["score"] for row in results)


def aggregate_object_answer_accuracy(results):
    return _mean(row["score"] for row in results if row["answer_format"] == "object")


def aggregate_direction_answer_accuracy(results):
    return _mean(row["score"] for row in results if row["answer_format"] == "direction")


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_object_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results if row["answer_format"] == "object")


def aggregate_direction_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results if row["answer_format"] == "direction")


def _matched_pairs(results):
    grouped = defaultdict(dict)
    for row in results:
        if row.get("source_relation_id") and row.get("answer_format") in ANSWER_FORMATS:
            grouped[row["source_relation_id"]][row["answer_format"]] = row
    return [pair for pair in grouped.values() if set(ANSWER_FORMATS) <= set(pair)]


def aggregate_object_minus_direction(results):
    return _mean(pair["object"]["score"] - pair["direction"]["score"] for pair in _matched_pairs(results))


def aggregate_format_switch_gain(results):
    return aggregate_object_minus_direction(results)


def _paired_rate(results, object_score: float, direction_score: float):
    return _mean(
        float(pair["object"]["score"] == object_score and pair["direction"]["score"] == direction_score)
        for pair in _matched_pairs(results)
    )


def aggregate_object_correct_direction_wrong(results):
    return _paired_rate(results, 1.0, 0.0)


def aggregate_direction_correct_object_wrong(results):
    return _paired_rate(results, 0.0, 1.0)


def aggregate_both_correct(results):
    return _paired_rate(results, 1.0, 1.0)


def aggregate_both_wrong(results):
    return _paired_rate(results, 0.0, 0.0)


def _viewpoint_accuracy(results, viewpoint: str):
    return _mean(row["score"] for row in results if row.get("viewpoint") == viewpoint)


def aggregate_camera_accuracy(results):
    return _viewpoint_accuracy(results, "camera")


def aggregate_reference_accuracy(results):
    return _viewpoint_accuracy(results, "reference")


def aggregate_addressee_accuracy(results):
    return _viewpoint_accuracy(results, "addressee")


def _stratify(results, field):
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row.get(field, "<missing>"))].append(row["score"])
    return {
        key: {"accuracy": _mean(scores), "count": len(scores)}
        for key, scores in sorted(grouped.items())
    }


def _log_report(results):
    eval_logger.info(
        "COMFORT direction/object matched pairs: {} across {} source rows.",
        len(_matched_pairs(results)),
        len({row.get("source_relation_id") for row in results}),
    )
    for field in ("relation", "answer_format", "viewpoint", "scene", "gold_option_letter"):
        eval_logger.info("COMFORT direction/object by {}: {}", field, _stratify(results, field))
    eval_logger.info(
        "COMFORT direction/object prediction distribution: {}",
        dict(sorted(Counter(row.get("predicted_option_letter") or "parse_failure" for row in results).items())),
    )


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"comfort_direction_object_{model}.json", args)
    report = {
        "num_records": len(results),
        "num_matched_pairs": len(_matched_pairs(results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT direction/object records saved to {}.", path)
