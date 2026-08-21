"""Evaluation helpers for the matched MOVi-E direction/object diagnostic."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


OPTION_LETTERS = tuple("ABCD")
ANSWER_FORMATS = ("object", "direction")


def _image_path(doc: dict) -> str:
    return str(doc.get("image_path") or doc.get("img_path") or doc.get("image") or "")


def process_docs(dataset: Dataset) -> Dataset:
    """Validate the pre-built matched pairs and their local image paths."""
    missing = []
    malformed = []
    for doc in dataset:
        if doc.get("diagnostic_answer_format") not in ANSWER_FORMATS:
            malformed.append(str(doc.get("qid", "unknown")))
        if not Path(_image_path(doc)).is_file():
            missing.append(_image_path(doc))
    if missing or malformed:
        problems = []
        if missing:
            problems.append(f"{len(missing)} image(s) missing (e.g. {missing[0]!r})")
        if malformed:
            problems.append(f"{len(malformed)} malformed row(s) (e.g. {malformed[0]!r})")
        raise ValueError("Invalid MOVi-E direction/object dataset: " + "; ".join(problems))
    return dataset


def doc_to_visual(doc):
    with Image.open(_image_path(doc)) as image:
        return [image.convert("RGB")]


def _options(doc: dict) -> list[str]:
    values = doc.get("options")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    return [str(doc[letter]) for letter in OPTION_LETTERS if doc.get(letter) is not None]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    option_lines = "\n".join(
        f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, _options(doc))
    )
    return f"{pre_prompt}{doc['question']}\n{option_lines}{post_prompt}"


def doc_to_target(doc):
    return str(doc.get("gold_option_letter", doc.get("answer", ""))).strip().upper()


def _extract_option_letter(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in (
        r"^\s*([A-D])(?:[.\s):]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
        r"\(([A-D])\)",
    ):
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _entry(doc, prediction: str, parsed: Optional[str]) -> dict:
    gold = doc_to_target(doc)
    return {
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_relation_id": doc.get("source_relation_id", doc.get("source_qid")),
        "sequence_name": doc.get("sequence_name"),
        "frame_index": doc.get("frame_index"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "relation": doc.get("diagnostic_relation"),
        "anchor": doc.get("diagnostic_anchor"),
        "target": doc.get("diagnostic_target_object"),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
    }


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = _extract_option_letter(prediction)
    entry = _entry(doc, prediction, parsed)
    metrics = (
        "accuracy", "object_answer_accuracy", "direction_answer_accuracy",
        "format_switch_gain", "object_minus_direction", "parse_success_rate",
        "object_parse_success_rate", "direction_parse_success_rate", "submission",
    )
    output = {metric: dict(entry) for metric in metrics}
    output["submission"].update({
        "question_prompt": doc_to_text(doc), "image_path": _image_path(doc),
        "options": _options(doc), "prediction": prediction,
    })
    return output


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    eval_logger.info("MOVi-E direction/object: {} complete matched pairs out of {} source groups.", len(_matched_pairs(results)), len({r.get("source_relation_id") for r in results}))
    return _mean(item["score"] for item in results)


def aggregate_object_answer_accuracy(results):
    return _mean(item["score"] for item in results if item["answer_format"] == "object")


def aggregate_direction_answer_accuracy(results):
    return _mean(item["score"] for item in results if item["answer_format"] == "direction")


def aggregate_parse_success_rate(results):
    return _mean(float(item["parse_success"]) for item in results)


def aggregate_object_parse_success_rate(results):
    return _mean(float(item["parse_success"]) for item in results if item["answer_format"] == "object")


def aggregate_direction_parse_success_rate(results):
    return _mean(float(item["parse_success"]) for item in results if item["answer_format"] == "direction")


def _matched_pairs(results):
    grouped = defaultdict(dict)
    for item in results:
        source_id, answer_format = item.get("source_relation_id"), item.get("answer_format")
        if source_id and answer_format in ANSWER_FORMATS:
            grouped[source_id][answer_format] = item
    return [pair for pair in grouped.values() if set(ANSWER_FORMATS) <= set(pair)]


def aggregate_object_minus_direction(results):
    return _mean(pair["object"]["score"] - pair["direction"]["score"] for pair in _matched_pairs(results))


def aggregate_format_switch_gain(results):
    """Matched object-answer minus direction-answer accuracy."""
    return aggregate_object_minus_direction(results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"kubric_movi_e_direction_object_{model}.json", args)
    report = {
        "num_records": len(results),
        "num_matched_pairs": len(_matched_pairs(results)),
        "prediction_distribution": dict(Counter(item.get("predicted_option_letter") or "parse_failure" for item in results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("MOVi-E direction/object records saved to {}.", path)
