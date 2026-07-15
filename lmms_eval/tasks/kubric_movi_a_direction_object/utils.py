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


def doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def _extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in (r"^\s*([A-D])(?:[.\s)]|$)", r"\b(?:answer|option)\s*(?:is|:)\s*([A-D])\b", r"\(([A-D])\)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
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


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    return _mean([result["score"] for result in results])


def aggregate_object_answer_accuracy(results):
    return _mean([result["score"] for result in results if result.get("answer_format") == "object"])


def aggregate_direction_answer_accuracy(results):
    return _mean([result["score"] for result in results if result.get("answer_format") == "direction"])


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
