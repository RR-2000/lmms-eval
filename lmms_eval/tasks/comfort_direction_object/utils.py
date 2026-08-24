"""Direction-versus-object evaluation for COMFORT_Multi_3D."""

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


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Multi_3D")
OPTION_LETTERS = tuple("ABCD")
DIRECTIONS = ("left", "right", "front", "behind")
ANSWER_FORMATS = ("object", "direction")


def _relation(doc: dict) -> str:
    """Return the native relation, retaining `behind` rather than `back`."""
    value = doc.get("answer") if doc.get("task") == "direction" else doc.get("direction")
    return str(value or "").strip().lower()


def _target(doc: dict) -> str:
    value = doc.get("target_object") if doc.get("task") == "direction" else doc.get("answer")
    return str(value or "").strip()


def _image_path(doc: dict) -> str:
    value = Path(str(doc.get("image", "")))
    return str(value if value.is_absolute() else DATA_ROOT / value)


def _source_relation_id(doc: dict) -> str:
    """Pair direction/object rows with the same controlled gold position."""
    return f"{doc['scene_id']}::{_relation(doc)}::{int(doc['answer_index'])}"


def _group_pairs(rows):
    grouped = defaultdict(dict)
    for row in rows:
        source_id = row.get("source_relation_id")
        answer_format = row.get("diagnostic_answer_format", row.get("answer_format"))
        if source_id and answer_format in ANSWER_FORMATS:
            grouped[source_id][answer_format] = row
    return grouped


def process_docs(dataset: Dataset) -> Dataset:
    """Validate and normalize the native COMFORT_Multi_3D annotations."""
    records = []
    skipped = Counter()
    for source in dataset:
        doc = dict(source)
        answer_format = str(doc.get("task", "")).strip().lower()
        relation = _relation(doc)
        target = _target(doc)
        anchor = str(doc.get("reference_object", "")).strip()
        options = [str(option).strip() for option in (doc.get("options") or [])]
        try:
            answer_index = int(doc.get("answer_index"))
        except (TypeError, ValueError):
            skipped["invalid_answer_index"] += 1
            continue

        if answer_format not in ANSWER_FORMATS:
            skipped["unsupported_answer_format"] += 1
            continue
        if relation not in DIRECTIONS or not target or not anchor:
            skipped["invalid_pair_metadata"] += 1
            continue
        if len(options) != 4 or len(set(options)) != 4:
            skipped["invalid_options"] += 1
            continue
        if answer_index not in range(4) or options[answer_index] != str(doc.get("answer", "")).strip():
            skipped["answer_option_mismatch"] += 1
            continue

        image_path = _image_path(doc)
        if not Path(image_path).is_file():
            skipped["missing_image"] += 1
            continue

        source_id = _source_relation_id(doc)
        doc.update(
            {
                "qid": str(doc.get("id")),
                "index": str(doc.get("id")),
                "source_qid": source_id,
                "source_relation_id": source_id,
                "source_task_family": "COMFORT_Multi_3D",
                "task_family": "comfort_direction_object",
                "diagnostic_variant": answer_format,
                "diagnostic_answer_format": answer_format,
                "diagnostic_anchor": anchor,
                "diagnostic_target_object": target,
                "diagnostic_relation": relation,
                "diagnostic_question": str(doc.get("question", "")).strip(),
                "options": options,
                "num_options": 4,
                "gold_option_letter": OPTION_LETTERS[answer_index],
                "answer_idx": answer_index,
                "img_path": image_path,
                "image_path": image_path,
                "viewpoint": "reference_object",
            }
        )
        records.append(doc)

    incomplete_ids = {
        source_id
        for source_id, pair in _group_pairs(records).items()
        if set(pair) != set(ANSWER_FORMATS)
    }
    if incomplete_ids:
        skipped["incomplete_pair"] += sum(
            record["source_relation_id"] in incomplete_ids for record in records
        )
        records = [
            record
            for record in records
            if record["source_relation_id"] not in incomplete_ids
        ]

    eval_logger.info(
        "COMFORT_Multi_3D direction/object loaded {} matched pairs ({} examples); skipped={}.",
        len(records) // 2,
        len(records),
        dict(skipped),
    )
    return Dataset.from_list(records)


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path") or doc.get("image_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT_Multi_3D image not found for {doc.get('qid')}: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    option_lines = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(OPTION_LETTERS, doc["options"])
    )
    return (
        f"{kwargs.get('pre_prompt', '')}Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with its letter.\n"
        f"Question: {doc['diagnostic_question']}\nOptions:\n{option_lines}"
        f"{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return str(doc["gold_option_letter"])


def extract_option_letter(text: str) -> Optional[str]:
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
    gold = str(doc["gold_option_letter"]).upper()
    return {
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_relation_id": doc.get("source_relation_id"),
        "scene_id": doc.get("scene_id"),
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
    metric_names = (
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
    )
    result = {metric: dict(entry) for metric in metric_names}
    result["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "img_path": doc.get("img_path"),
        "image_path": doc.get("image_path"),
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
    return _mean(
        float(row["parse_success"])
        for row in results
        if row["answer_format"] == "object"
    )


def aggregate_direction_parse_success_rate(results):
    return _mean(
        float(row["parse_success"])
        for row in results
        if row["answer_format"] == "direction"
    )


def _matched_pairs(results):
    return [
        pair
        for pair in _group_pairs(results).values()
        if set(ANSWER_FORMATS) <= set(pair)
    ]


def aggregate_object_minus_direction(results):
    return _mean(
        pair["object"]["score"] - pair["direction"]["score"]
        for pair in _matched_pairs(results)
    )


def aggregate_format_switch_gain(results):
    return aggregate_object_minus_direction(results)


def _paired_rate(results, object_score: float, direction_score: float):
    return _mean(
        float(
            pair["object"]["score"] == object_score
            and pair["direction"]["score"] == direction_score
        )
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
        "COMFORT_Multi_3D matched pairs: {} across {} source groups.",
        len(_matched_pairs(results)),
        len({row.get("source_relation_id") for row in results}),
    )
    for field in ("relation", "answer_format", "scene_id", "gold_option_letter"):
        eval_logger.info("COMFORT_Multi_3D by {}: {}", field, _stratify(results, field))
    eval_logger.info(
        "COMFORT_Multi_3D prediction distribution: {}",
        dict(
            sorted(
                Counter(
                    row.get("predicted_option_letter") or "parse_failure"
                    for row in results
                ).items()
            )
        ),
    )


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"comfort_direction_object_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "num_records": len(results),
        "num_matched_pairs": len(_matched_pairs(results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT_Multi_3D direction/object records saved to {}.", path)
