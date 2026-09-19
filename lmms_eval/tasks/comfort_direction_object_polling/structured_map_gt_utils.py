"""One-shot structured direction-to-object maps with ground-truth axes."""

import json
from collections import Counter

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.tasks.comfort_direction_object_gt_help import utils as gt_aids
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "comfort_direction_object_structured_map_gt"


def _normalize(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def process_docs(dataset: Dataset) -> Dataset:
    """Create one complete direction-to-object mapping request per scene."""
    by_scene = {}
    for row in base.process_docs(dataset):
        by_scene.setdefault(str(row["scene_id"]), row)

    records = []
    skipped = Counter()
    for scene_id, source in sorted(by_scene.items()):
        relation_to_object = {direction: str(gt_aids.get_object_at_direction(source, direction).get("label", "")).strip() for direction in base.DIRECTIONS}
        candidates = sorted(relation_to_object.values(), key=_normalize)
        if any(not value for value in candidates):
            skipped["empty_object_name"] += 1
            continue
        if len(set(map(_normalize, candidates))) != len(base.DIRECTIONS):
            skipped["duplicate_object_name"] += 1
            continue
        qid = f"{scene_id}::structured_direction_object_map"
        record = dict(source)
        record.update(
            {
                "qid": qid,
                "index": qid,
                "source_qid": qid,
                "source_relation_id": qid,
                "task_family": TASK_NAME,
                "diagnostic_variant": "structured_direction_object_map",
                "diagnostic_answer_format": "structured_object_map",
                "relation_to_object": relation_to_object,
                "candidate_objects": candidates,
                "gt_mapping_json": json.dumps(relation_to_object, sort_keys=True),
            }
        )
        records.append(record)

    eval_logger.info(
        "COMFORT one-shot structured-map task loaded {} scenes; skipped={}.",
        len(records),
        dict(skipped),
    )
    return Dataset.from_list(records)


def doc_to_visual(doc):
    """Use the short four-axis overlay from GT-help mode 36."""
    image = base.doc_to_visual(doc)[0]
    return [gt_aids.draw_short_reference_direction_arrows(doc, image)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    reference = str(doc["diagnostic_anchor"])
    candidates = ", ".join(str(value) for value in doc["candidate_objects"])
    schema = '{"left":"<object>","right":"<object>","front":"<object>","behind":"<object>"}'
    return (
        f"{kwargs.get('pre_prompt', '')}"
        "Answer this spatial-reasoning question using the image and the four "
        "labeled arrows drawn on the reference object. "
        f"Using the {reference}'s own viewpoint, assign each candidate object "
        "to its direction around the reference.\n"
        f"Candidate objects (alphabetical order): {candidates}.\n"
        "Return only one JSON object with exactly the keys left, right, front, "
        "and behind. Use every candidate object exactly once.\n"
        f"Required schema: {schema}"
        f"{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return str(doc["gt_mapping_json"])


def _extract_json(text: str):
    if not text:
        return None
    raw = str(text).strip()
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.insert(0, raw[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize_mapping(mapping) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    output = {}
    for key, value in mapping.items():
        normalized_key = _normalize(key)
        if normalized_key == "back":
            normalized_key = "behind"
        output[normalized_key] = _normalize(value)
    return output


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = _extract_json(prediction)
    gold_raw = json.loads(doc["gt_mapping_json"])
    gold = _normalize_mapping(gold_raw)
    predicted = _normalize_mapping(parsed)
    edge_scores = {direction: float(predicted.get(direction) == gold[direction]) for direction in base.DIRECTIONS}
    candidate_set = {_normalize(value) for value in doc["candidate_objects"]}
    predicted_values = [predicted.get(direction) for direction in base.DIRECTIONS]
    bijective = float(parsed is not None and set(predicted) == set(base.DIRECTIONS) and len(set(predicted_values)) == len(base.DIRECTIONS) and set(predicted_values) == candidate_set)
    exact = float(parsed is not None and predicted == gold)
    entry = {
        "qid": doc["qid"],
        "scene_id": doc["scene_id"],
        "anchor": doc["diagnostic_anchor"],
        "candidate_objects": list(doc["candidate_objects"]),
        "gt_mapping": gold_raw,
        "parsed_mapping": parsed,
        "normalized_mapping": predicted,
        "edge_scores": edge_scores,
        "mapping_edge_accuracy": sum(edge_scores.values()) / len(base.DIRECTIONS),
        "mapping_exact_accuracy": exact,
        "bijective_map_rate": bijective,
        "parse_success": float(parsed is not None),
        "prediction": prediction,
    }
    metrics = (
        "mapping_edge_accuracy",
        "object_recovery_accuracy",
        "mapping_exact_accuracy",
        "bijective_map_rate",
        "parse_success_rate",
    )
    output = {name: dict(entry) for name in metrics}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "img_path": doc.get("img_path"),
        "image_path": doc.get("image_path"),
        "orientation_source": "COMFORT ground-truth scene metadata",
    }
    return output


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_mapping_edge_accuracy(results):
    return _mean(float(row["mapping_edge_accuracy"]) for row in results)


aggregate_object_recovery_accuracy = aggregate_mapping_edge_accuracy


def aggregate_mapping_exact_accuracy(results):
    return _mean(float(row["mapping_exact_accuracy"]) for row in results)


def aggregate_bijective_map_rate(results):
    return _mean(float(row["bijective_map_rate"]) for row in results)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": TASK_NAME,
        "num_records": len(results),
        "orientation_source": "COMFORT ground-truth scene metadata",
        "analysis": {
            "mapping_edge_accuracy": aggregate_mapping_edge_accuracy(results),
            "mapping_exact_accuracy": aggregate_mapping_exact_accuracy(results),
            "bijective_map_rate": aggregate_bijective_map_rate(results),
            "parse_success_rate": aggregate_parse_success_rate(results),
        },
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT structured-map records saved to {}.", path)
