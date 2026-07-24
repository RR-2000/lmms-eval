"""Matched direction-versus-object diagnostic for the Kubric MOVi-A export.

This task uses only the two object-centric source families.  Each source item
creates a matched native/inverse pair with the same image, anchor, target, and
relation:

* single-object relative-position questions keep their native direction answer
  and receive an object-selection variant with multiple object choices;
* multi-object relative-position questions keep their native object-selection
  answer and receive a direction-answer variant for their selected object.

The paired metrics therefore measure the effect of prompt answer format rather
than scene content.
"""

import json
import random
import re
from collections import defaultdict
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a.utils import _get_image_path, _get_options, doc_to_visual
from lmms_eval.utils import sanitize_model_name


SOURCE_FAMILIES = {
    "object_centric_relative_position",
    "object_centric_relative_position_multi",
}
DIRECTIONS = ("left", "right", "front", "behind")
SAMPLE_SEED = "kubric_movi_a_direction_object_v1"


def _answer_text(doc: dict) -> str:
    answer = str(doc.get("answer", "")).strip()
    options = _get_options(doc)
    return str(options.get(answer, answer)).strip()


def _relation(doc: dict) -> Optional[str]:
    relation = str((doc.get("task_metadata") or {}).get("relation", "")).strip().lower()
    return relation if relation in DIRECTIONS else None


def _relation_phrase(relation: str) -> str:
    return {
        "left": "to the left of",
        "right": "to the right of",
        "front": "in front of",
        "behind": "behind",
    }[relation]


def _multiple_object_options(doc: dict, anchor: str, target: str, source_qid: str) -> list[str]:
    """Select up to four visible object candidates, always including target and anchor."""
    visible_names = [str(obj.get("name", "")) for obj in doc.get("visible_objects", []) if obj.get("name")]
    candidates = [target]
    if anchor != target:
        candidates.append(anchor)
    candidates.extend(name for name in visible_names if name not in candidates)
    candidates = candidates[:4]
    random.Random(source_qid).shuffle(candidates)
    return candidates


def _set_options(doc: dict, values: list[str], answer_value: str) -> None:
    for index, letter in enumerate("ABCD"):
        doc[letter] = values[index] if index < len(values) else None
    doc["answer"] = "ABCD"[values.index(answer_value)]


def process_docs(dataset: Dataset) -> Dataset:
    """Sample both source families equally, then create native/inverse variants."""
    records = []
    skipped = 0
    candidates = {family: [] for family in SOURCE_FAMILIES}
    for doc in dataset:
        family = doc.get("task_family")
        if family in candidates:
            candidates[family].append(doc)

    sampled_count = min(len(items) for items in candidates.values()) if candidates else 0
    sampler = random.Random(SAMPLE_SEED)
    sampled_docs = []
    for family in sorted(candidates):
        family_docs = candidates[family]
        sampled_docs.extend(sampler.sample(family_docs, sampled_count))

    eval_logger.info(
        "Kubric direction/object task sampled %d source items per family "
        "(%d source items total; %d native/inverse examples total; seed=%s).",
        sampled_count,
        sampled_count * len(candidates),
        sampled_count * len(candidates) * 2,
        SAMPLE_SEED,
    )

    for doc in sampled_docs:
        family = doc.get("task_family")
        relation = _relation(doc)
        metadata = doc.get("task_metadata") or {}
        anchor = str(metadata.get("anchor_object", "")).strip()
        if doc["task_family"] == "object_centric_relative_position":
            target = str(metadata.get("target_object", "")).strip()
        else:
            target = str(metadata.get("correct_object", _answer_text(doc))).strip()
        if not relation or not anchor or not target:
            skipped += 1
            continue

        source_qid = str(doc.get("qid", doc.get("index", "unknown")))
        common = dict(doc)
        common.update(
            {
                "source_qid": source_qid,
                "source_task_family": doc["task_family"],
                "diagnostic_relation": relation,
                "diagnostic_anchor": anchor,
                "diagnostic_target_object": target,
                "diagnostic_sampled_source_count_per_family": sampled_count,
                "diagnostic_sample_seed": SAMPLE_SEED,
            }
        )

        native = dict(common)
        native.update(
            {
                "qid": f"{source_qid}::native",
                "index": f"{source_qid}::native",
                "diagnostic_variant": "native",
                "diagnostic_answer_format": "direction" if doc["task_family"] == "object_centric_relative_position" else "object",
                "diagnostic_target": relation if doc["task_family"] == "object_centric_relative_position" else target,
            }
        )
        records.append(native)

        inverse = dict(common)
        inverse.update(
            {
                "qid": f"{source_qid}::inverse",
                "index": f"{source_qid}::inverse",
                "diagnostic_variant": "inverse",
            }
        )
        if doc["task_family"] == "object_centric_relative_position":
            options = _multiple_object_options(doc, anchor, target, source_qid)
            inverse["question"] = (
                f"Imagine standing at the {anchor} and facing the camera. Which object is "
                f"{_relation_phrase(relation)} the {anchor}?"
            )
            _set_options(inverse, options, target)
            inverse["diagnostic_answer_format"] = "object"
            inverse["diagnostic_target"] = target
        else:
            inverse["question"] = (
                f"Imagine standing at the {anchor} and facing the camera. "
                f"Where is the {target} relative to the {anchor}?"
            )
            _set_options(inverse, list(DIRECTIONS), relation)
            inverse["diagnostic_answer_format"] = "direction"
            inverse["diagnostic_target"] = relation
        records.append(inverse)

    if skipped:
        eval_logger.warning(f"Skipped {skipped} Kubric rows without a valid matched transformation.")
    return Dataset.from_list(records)


EXTENDED_DIRECTION_VARIANT = "direction_natural_language"
EXTENDED_EXHAUSTIVE_VARIANT = "object_direction_exhaustive"
EXTENDED_OPTION_LABELS = tuple("ABCDEFGHIJKLMNOP")


def _set_extended_options(doc: dict, values: list[str], answer_value: str) -> None:
    """Store an arbitrary number of labelled options for the extended task."""
    doc["extended_options"] = values
    doc["extended_option_labels"] = list(EXTENDED_OPTION_LABELS[: len(values)])
    doc["extended_answer"] = EXTENDED_OPTION_LABELS[values.index(answer_value)]


def _direction_document_options(doc: dict) -> list[str]:
    return [
        str(value).strip()
        for value in _get_options(doc).values()
        if value is not None
    ]


def _build_extended_variants(records: list[dict]) -> list[dict]:
    """Add natural-language and exhaustive options to native/inverse pairs."""
    by_source = defaultdict(list)
    for doc in records:
        by_source[doc.get("source_qid")].append(doc)

    extended = []
    for source_qid, pair in by_source.items():
        direction_doc = next(
            (doc for doc in pair if doc.get("diagnostic_answer_format") == "direction"),
            None,
        )
        object_doc = next(
            (doc for doc in pair if doc.get("diagnostic_answer_format") == "object"),
            None,
        )
        if not direction_doc or not object_doc:
            continue

        relation = direction_doc["diagnostic_relation"]
        anchor = direction_doc["diagnostic_anchor"]
        target = direction_doc["diagnostic_target_object"]
        direction_options = [
            f"{target} {_relation_phrase(option)} {anchor}"
            for option in DIRECTIONS
        ]

        natural = dict(direction_doc)
        natural.update(
            {
                "qid": f"{source_qid}::direction_natural_language",
                "index": f"{source_qid}::direction_natural_language",
                "question": (
                    f"Which statement correctly describes where the {target} is "
                    f"relative to the {anchor}?"
                ),
                "diagnostic_variant": EXTENDED_DIRECTION_VARIANT,
                "diagnostic_answer_format": "direction",
                "diagnostic_target": direction_options[DIRECTIONS.index(relation)],
            }
        )
        _set_extended_options(natural, direction_options, direction_options[DIRECTIONS.index(relation)])
        extended.append(natural)

        object_options = _direction_document_options(object_doc)
        if target not in object_options:
            object_options.insert(0, target)
        object_options = object_options[:4]
        exhaustive_options = [
            f"{obj} {_relation_phrase(option)} {anchor}"
            for obj in object_options
            for option in DIRECTIONS
        ]
        correct_exhaustive_option = f"{target} {_relation_phrase(relation)} {anchor}"
        exhaustive = dict(object_doc)
        exhaustive.update(
            {
                "qid": f"{source_qid}::object_direction_exhaustive",
                "index": f"{source_qid}::object_direction_exhaustive",
                "question": (
                    f"Which statement correctly describes the position of an object "
                    f"relative to the {anchor}?"
                ),
                "diagnostic_variant": EXTENDED_EXHAUSTIVE_VARIANT,
                "diagnostic_answer_format": "object_direction",
                "diagnostic_target": correct_exhaustive_option,
            }
        )
        _set_extended_options(exhaustive, exhaustive_options, correct_exhaustive_option)
        extended.append(exhaustive)

    return extended


def extended_process_docs(dataset: Dataset) -> Dataset:
    """Keep native/inverse rows and add both extended option formats."""
    base_records = list(process_docs(dataset))
    return Dataset.from_list(base_records + _build_extended_variants(base_records))


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    options = _get_options(doc)
    prompt = (
        "Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with its letter.\n"
        f"Question: {doc['question']}\nOptions:\n"
    )
    for letter, value in options.items():
        prompt += f"{letter}. {value}\n"
    return prompt


def extended_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Render the standard rows and the extended natural-language options."""
    if not doc.get("extended_options"):
        return doc_to_text(doc, lmms_eval_specific_kwargs)

    prompt = (
        "Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with its option letter.\n"
        f"Question: {doc['question']}\nOptions:\n"
    )
    for label, option in zip(doc["extended_option_labels"], doc["extended_options"]):
        prompt += f"{label}. {option}\n"
    return prompt


def extended_doc_to_target(doc):
    return str(doc.get("extended_answer", doc.get("answer", ""))).strip()


def doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def direct_answer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Prompt the matched diagnostic without letter-labelled answer choices."""
    del lmms_eval_specific_kwargs
    options = [str(value).strip() for value in _get_options(doc).values() if value is not None]
    prompt = (
        "Answer this spatial-reasoning question using the image. "
        "Respond with the exact object or direction text, not an option letter.\n"
        f"Question: {doc['question']}\nPossible answers: {'; '.join(options)}\n"
    )
    return prompt


def direct_answer_doc_to_target(doc):
    return str(doc.get("diagnostic_target", "")).strip()


def _extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in (r"^\s*([A-D])(?:[.\s)]|$)", r"\b(?:answer|option)\s*(?:is|:)\s*([A-D])\b", r"\(([A-D])\)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _extract_extended_answer(text: str, labels: list[str]) -> Optional[str]:
    if not text:
        return None
    label_pattern = "|".join(re.escape(label) for label in labels)
    for pattern in (
        rf"^\s*({label_pattern})(?:[.\s)]|$)",
        rf"\b(?:answer|option)\s*(?:is|:)\s*({label_pattern})\b",
        rf"\(({label_pattern})\)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _normalize_direct_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _extract_direct_answer(text: str, options: list[str]) -> Optional[str]:
    """Match a direct object/direction response to one of the displayed values."""
    normalized = _normalize_direct_answer(text).strip(".?!:;\"'")
    if not normalized:
        return None
    normalized_options = {_normalize_direct_answer(option): option for option in options}
    if normalized in normalized_options:
        return normalized_options[normalized]

    # Permit short natural responses such as "The answer is left" while never
    # accepting a letter as a surrogate answer.
    for option_norm, option in sorted(normalized_options.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(option_norm)}(?!\w)", normalized):
            return option
    return None


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = _extract_answer(prediction)
    score = float(parsed == str(doc.get("answer", "")).strip())
    entry = {
        "index": doc.get("index"),
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_task_family": doc.get("source_task_family"),
        "difficulty": doc.get("difficulty", "unknown"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "relation": doc.get("diagnostic_relation"),
        "score": score,
    }
    result = {
        "accuracy": entry,
        "object_answer_accuracy": entry,
        "direction_answer_accuracy": entry,
        "format_switch_gain": entry,
        "object_minus_direction": entry,
        "direction_natural_language_gain_from_simple": entry,
        "object_direction_exhaustive_gain_from_simple": entry,
        "submission": {
            **entry,
            "question_prompt": doc_to_text(doc),
            "img_path": _get_image_path(doc),
            "prediction": prediction,
            "parsed_prediction": parsed,
            "gold_option": doc.get("answer"),
            "gold_target": doc.get("diagnostic_target"),
        },
    }
    for source_family in SOURCE_FAMILIES:
        result[f"{source_family}_format_switch_gain"] = entry
    return result


def extended_process_results(doc, results):
    """Score standard rows plus the two extended option-format variants."""
    if not doc.get("extended_options"):
        return process_results(doc, results)

    prediction = results[0].strip() if results else ""
    labels = doc["extended_option_labels"]
    parsed = _extract_extended_answer(prediction, labels)
    score = float(parsed == str(doc.get("extended_answer", "")).strip().upper())
    entry = {
        "index": doc.get("index"),
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_task_family": doc.get("source_task_family"),
        "difficulty": doc.get("difficulty", "unknown"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "relation": doc.get("diagnostic_relation"),
        "score": score,
    }
    submission = {
        **entry,
        "question_prompt": extended_doc_to_text(doc),
        "img_path": _get_image_path(doc),
        "prediction": prediction,
        "parsed_prediction": parsed,
        "gold_option": doc.get("extended_answer"),
        "gold_target": doc.get("diagnostic_target"),
        "options": doc.get("extended_options"),
    }
    result = {
        "accuracy": entry,
        "object_answer_accuracy": entry,
        "direction_answer_accuracy": entry,
        "direction_natural_language_gain_from_simple": entry,
        "object_direction_exhaustive_gain_from_simple": entry,
        "direction_natural_language_accuracy": entry,
        "object_direction_exhaustive_accuracy": entry,
        "submission": submission,
    }
    for source_family in SOURCE_FAMILIES:
        result[f"{source_family}_format_switch_gain"] = entry
    return result


def direct_answer_process_results(doc, results):
    prediction = results[0].strip() if results else ""
    options = [str(value).strip() for value in _get_options(doc).values() if value is not None]
    parsed = _extract_direct_answer(prediction, options)
    gold = direct_answer_doc_to_target(doc)
    score = float(_normalize_direct_answer(parsed or "") == _normalize_direct_answer(gold))
    entry = {
        "index": doc.get("index"),
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "source_task_family": doc.get("source_task_family"),
        "difficulty": doc.get("difficulty", "unknown"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "relation": doc.get("diagnostic_relation"),
        "score": score,
    }
    result = {
        "accuracy": entry,
        "object_answer_accuracy": entry,
        "direction_answer_accuracy": entry,
        "format_switch_gain": entry,
        "object_minus_direction": entry,
        "submission": {
            **entry,
            "question_prompt": direct_answer_doc_to_text(doc),
            "img_path": _get_image_path(doc),
            "prediction": prediction,
            "parsed_prediction": parsed,
            "gold_target": gold,
        },
    }
    for source_family in SOURCE_FAMILIES:
        result[f"{source_family}_format_switch_gain"] = entry
    return result


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    return _mean([result["score"] for result in results])


def aggregate_object_answer_accuracy(results):
    return _mean([result["score"] for result in results if result.get("answer_format") == "object"])


def aggregate_direction_answer_accuracy(results):
    return _mean([result["score"] for result in results if result.get("answer_format") == "direction"])


def aggregate_direction_natural_language_accuracy(results):
    return _mean(
        [
            result["score"]
            for result in results
            if result.get("variant") == EXTENDED_DIRECTION_VARIANT
        ]
    )


def aggregate_object_direction_exhaustive_accuracy(results):
    return _mean(
        [
            result["score"]
            for result in results
            if result.get("variant") == EXTENDED_EXHAUSTIVE_VARIANT
        ]
    )


def _aggregate_gain_from_simple_direction(results, variant):
    paired = defaultdict(dict)
    for result in results:
        pair = paired[result.get("source_qid")]
        result_variant = result.get("variant")
        if result_variant == variant:
            pair["target"] = result["score"]
        elif result_variant in {"native", "inverse"} and result.get("answer_format") == "direction":
            pair["simple"] = result["score"]

    gains = [
        pair["target"] - pair["simple"]
        for pair in paired.values()
        if "target" in pair and "simple" in pair
    ]
    return _mean(gains)


def aggregate_direction_natural_language_gain_from_simple(results):
    return _aggregate_gain_from_simple_direction(results, EXTENDED_DIRECTION_VARIANT)


def aggregate_object_direction_exhaustive_gain_from_simple(results):
    return _aggregate_gain_from_simple_direction(results, EXTENDED_EXHAUSTIVE_VARIANT)


def _paired_differences(results, source_family=None, comparison="switch"):
    paired = defaultdict(dict)
    for result in results:
        if source_family is None or result.get("source_task_family") == source_family:
            paired[result.get("source_qid")][result.get("variant")] = result["score"]
    if comparison == "switch":
        differences = [pair["inverse"] - pair["native"] for pair in paired.values() if {"native", "inverse"} <= pair.keys()]
    else:
        by_format = defaultdict(dict)
        for result in results:
            if source_family is None or result.get("source_task_family") == source_family:
                by_format[result.get("source_qid")][result.get("answer_format")] = result["score"]
        differences = [pair["object"] - pair["direction"] for pair in by_format.values() if {"object", "direction"} <= pair.keys()]
    return _mean(differences)


def aggregate_format_switch_gain(results):
    """Paired inverse-minus-native accuracy; positive means the changed prompt helped."""
    return _paired_differences(results)


def aggregate_object_minus_direction(results):
    """Paired object-answer minus direction-answer accuracy."""
    return _paired_differences(results, comparison="format")


def aggregate_object_centric_relative_position_format_switch_gain(results):
    return _paired_differences(results, "object_centric_relative_position")


def aggregate_object_centric_relative_position_multi_format_switch_gain(results):
    return _paired_differences(results, "object_centric_relative_position_multi")


def _submission_model_tag(args) -> str:
    model_name = getattr(args, "model", "") or ""
    if model_name:
        return sanitize_model_name(model_name)
    return "unknown_model"


def aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"kubric_movi_a_direction_object_{_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"Kubric direction/object records saved to {path}.")


def direct_answer_aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"kubric_movi_a_direction_object_direct_answer_{_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"Kubric direct-answer direction/object records saved to {path}.")


def extended_aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"kubric_movi_a_direction_object_extended_{_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"Kubric extended direction/object records saved to {path}.")
