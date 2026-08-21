"""Runtime helpers and metrics for the GQA direction/object diagnostic."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import datasets
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


OPTION_LETTERS = tuple("ABCD")
ANSWER_FORMATS = ("object", "direction")
def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Validate the explicit local image paths emitted by the builder."""
    missing = sorted(
        str(doc.get("image_path", ""))
        for doc in dataset
        if not Path(str(doc.get("image_path", ""))).is_file()
    )
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Could not resolve {len(missing)} local GQA diagnostic image(s): {preview}"
        )
    return dataset


def doc_to_visual(doc):
    with Image.open(doc["image_path"]) as image:
        return [image.convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    options = doc["options"]
    if doc["diagnostic_answer_format"] == "object":
        relation_phrase = {
            "left": "to the left of",
            "right": "to the right of",
            "above": "above",
            "below": "below",
        }[doc["diagnostic_relation"]]
        question = (
            f"Look at the image. Which object is {relation_phrase} the "
            f"{doc['diagnostic_anchor']}? Answer with the option letter only."
        )
    else:
        question = (
            f"Look at the image. Where is the {doc['diagnostic_target_object']} "
            f"relative to the {doc['diagnostic_anchor']}? "
            "Answer with the option letter only."
        )
    option_lines = "\n".join(
        f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options)
    )
    return f"{pre_prompt}{question}\n{option_lines}{post_prompt}"


def doc_to_target(doc):
    return str(doc["gold_option_letter"])


def extract_option_letter(text: str) -> Optional[str]:
    """Parse an A-D response without treating arbitrary words as answers."""
    if not text:
        return None
    patterns = (
        r"^\s*([A-D])(?:[.\s):]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
        r"\(([A-D])\)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _metric_entry(doc, prediction, parsed):
    gold = str(doc["gold_option_letter"]).upper()
    return {
        "source_relation_id": doc["source_relation_id"],
        "source_qid": doc.get("source_qid", doc["source_relation_id"]),
        "image_id": str(doc["image_id"]),
        "variant": doc["diagnostic_variant"],
        "answer_format": doc["diagnostic_answer_format"],
        "anchor": doc["diagnostic_anchor"],
        "target": doc["diagnostic_target_object"],
        "relation": doc["diagnostic_relation"],
        "num_options": int(doc["num_options"]),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
        "prediction": prediction,
    }


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = extract_option_letter(prediction)
    entry = _metric_entry(doc, prediction, parsed)
    metric_names = (
        "accuracy",
        "object_answer_accuracy",
        "direction_answer_accuracy",
        "format_switch_gain",
        "object_minus_direction",
        "parse_success_rate",
        "object_parse_success_rate",
        "direction_parse_success_rate",
        "object_correct_direction_wrong",
        "direction_correct_object_wrong",
        "both_correct",
        "both_wrong",
        "submission",
    )
    result = {metric: dict(entry) for metric in metric_names}
    result["submission"].update(
        {
            "question_prompt": doc_to_text(doc),
            "img_path": doc["image_path"],
            "image_path": doc["image_path"],
            "options": list(doc["options"]),
            "gold_answer": doc["answer"],
        }
    )
    return result


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    _log_stratified_report(results)
    return _mean(item["score"] for item in results)


def aggregate_object_answer_accuracy(results):
    return _mean(
        item["score"] for item in results if item["answer_format"] == "object"
    )


def aggregate_direction_answer_accuracy(results):
    return _mean(
        item["score"] for item in results if item["answer_format"] == "direction"
    )


def aggregate_parse_success_rate(results):
    return _mean(float(item["parse_success"]) for item in results)


def aggregate_object_parse_success_rate(results):
    return _mean(
        float(item["parse_success"])
        for item in results
        if item["answer_format"] == "object"
    )


def aggregate_direction_parse_success_rate(results):
    return _mean(
        float(item["parse_success"])
        for item in results
        if item["answer_format"] == "direction"
    )


def _matched_pairs(results):
    grouped = defaultdict(dict)
    for item in results:
        source_id = item.get("source_relation_id")
        answer_format = item.get("answer_format")
        if source_id and answer_format in ANSWER_FORMATS:
            grouped[source_id][answer_format] = item
    return [
        pair for pair in grouped.values() if set(ANSWER_FORMATS) <= set(pair)
    ]


def aggregate_object_minus_direction(results):
    return _mean(
        pair["object"]["score"] - pair["direction"]["score"]
        for pair in _matched_pairs(results)
    )


def aggregate_format_switch_gain(results):
    """Alias for paired object-answer minus direction-answer accuracy."""
    return aggregate_object_minus_direction(results)


def _paired_outcome_rate(results, object_score, direction_score):
    return _mean(
        float(
            pair["object"]["score"] == object_score
            and pair["direction"]["score"] == direction_score
        )
        for pair in _matched_pairs(results)
    )


def aggregate_object_correct_direction_wrong(results):
    return _paired_outcome_rate(results, 1.0, 0.0)


def aggregate_direction_correct_object_wrong(results):
    return _paired_outcome_rate(results, 0.0, 1.0)


def aggregate_both_correct(results):
    return _paired_outcome_rate(results, 1.0, 1.0)


def aggregate_both_wrong(results):
    return _paired_outcome_rate(results, 0.0, 0.0)


def _stratify(results, field):
    grouped = defaultdict(list)
    for item in results:
        grouped[str(item.get(field, "<missing>"))].append(item["score"])
    return {
        key: {"accuracy": _mean(scores), "count": len(scores)}
        for key, scores in sorted(grouped.items())
    }


def _log_stratified_report(results):
    eval_logger.info(
        "GQA direction/object matched pairs: {} of {} source relation groups.",
        len(_matched_pairs(results)),
        len({item.get("source_relation_id") for item in results}),
    )
    for field in (
        "relation",
        "answer_format",
        "gold_option_letter",
        "predicted_option_letter",
        "anchor",
        "target",
        "num_options",
    ):
        eval_logger.info(
            "GQA direction/object by {}: {}", field, _stratify(results, field)
        )

    prediction_distribution = Counter(
        item.get("predicted_option_letter") or "parse_failure" for item in results
    )
    eval_logger.info(
        "GQA direction/object prediction distribution: {}",
        dict(sorted(prediction_distribution.items())),
    )


def _model_tag(args):
    return sanitize_model_name(getattr(args, "model", "") or "unknown_model")


def aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"gqa_direction_object_{_model_tag(args)}.json", args
    )
    report = {
        "num_records": len(results),
        "num_matched_pairs": len(_matched_pairs(results)),
        "prediction_distribution": dict(
            Counter(item.get("predicted_option_letter") or "parse_failure" for item in results)
        ),
        "stratified_accuracy": {
            field: _stratify(results, field)
            for field in (
                "relation",
                "answer_format",
                "gold_option_letter",
                "predicted_option_letter",
                "anchor",
                "target",
                "num_options",
            )
        },
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("GQA direction/object records and strata saved to {}.", path)
