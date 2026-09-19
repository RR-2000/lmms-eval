"""Ground-truth-axis target-only polling for COMFORT direction questions.

This task mirrors ``concrete_direction.target_only_inversion --no-geometry``:
each source direction question becomes four binary target/none polls, and the
direction with the strongest target vote is returned.  The only visual aid is
the short four-axis overlay derived from COMFORT's ground-truth scene metadata.
"""

import json
import random
import re
from collections import Counter, defaultdict

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.tasks.comfort_direction_object_gt_help import utils as gt_aids
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "comfort_direction_object_target_only_gt_polling"
NONE_OPTION = "None of the above"
MINIMUM_DIRECTION_MARGIN = 0.10


def process_docs(dataset: Dataset) -> Dataset:
    """Expand each native direction row into four target-vs-none polls."""
    direction_rows = [row for row in base.process_docs(dataset) if row["diagnostic_answer_format"] == "direction"]
    records = []
    for direction_row in direction_rows:
        source_id = str(direction_row["qid"])
        target = str(direction_row["diagnostic_target_object"])
        gold_direction = str(direction_row["diagnostic_relation"])
        for poll_direction in base.DIRECTIONS:
            target_letter = random.Random(f"{source_id}::{poll_direction}").choice(("A", "B"))
            none_letter = "B" if target_letter == "A" else "A"
            options_by_letter = {
                target_letter: target,
                none_letter: NONE_OPTION,
            }
            gold_letter = target_letter if poll_direction == gold_direction else none_letter
            qid = f"{source_id}::target_poll_{poll_direction}"
            poll = dict(direction_row)
            poll.update(
                {
                    "qid": qid,
                    "index": qid,
                    "task_family": TASK_NAME,
                    "conversion_source_qid": source_id,
                    "source_qid": source_id,
                    "source_direction_question": direction_row["diagnostic_question"],
                    "source_gold_direction": gold_direction,
                    "source_target_object": target,
                    "diagnostic_variant": f"target_poll_{poll_direction}",
                    "diagnostic_answer_format": "binary_object",
                    "poll_direction": poll_direction,
                    "target_option_letter": target_letter,
                    "none_option_letter": none_letter,
                    "options": [options_by_letter[letter] for letter in "AB"],
                    "num_options": 2,
                    "answer": options_by_letter[gold_letter],
                    "answer_idx": "AB".index(gold_letter),
                    "gold_option_letter": gold_letter,
                }
            )
            records.append(poll)

    eval_logger.info(
        "COMFORT target-only GT polling loaded {} source questions and {} binary polls.",
        len(direction_rows),
        len(records),
    )
    return Dataset.from_list(records)


def doc_to_visual(doc):
    """Render the short GT axes that replace the learned orientation overlay."""
    image = base.doc_to_visual(doc)[0]
    return [gt_aids.draw_short_reference_direction_arrows(doc, image)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Use the exact binary prompt structure from target-only inversion."""
    kwargs = lmms_eval_specific_kwargs or {}
    direction = str(doc["poll_direction"])
    side = "back" if direction == "behind" else direction
    reference = str(doc["diagnostic_anchor"])
    question = f"Which object is where the {side} side of the {reference} is facing?"
    option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip("AB", doc["options"]))
    return (
        f"{kwargs.get('pre_prompt', '')}"
        "Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with only its letter.\n"
        f"Question: {question}\nOptions:\n{option_lines}"
        f"{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return str(doc["gold_option_letter"])


def parse_binary_vote(raw: str, target_letter: str):
    """Match the target-only runner's first-standalone-A/B vote parser."""
    letters = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", str(raw).upper())
    answer = letters[0] if letters else None
    if answer is None:
        return 0.0, None
    return (1.0 if answer == target_letter else -1.0), answer


def _selected_answer(doc, letter):
    return doc["options"]["AB".index(letter)] if letter in {"A", "B"} else None


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    evidence, parsed = parse_binary_vote(prediction, str(doc["target_option_letter"]))
    gold = str(doc["gold_option_letter"])
    entry = {
        "qid": doc.get("qid"),
        "source_qid": doc.get("conversion_source_qid"),
        "scene_id": doc.get("scene_id"),
        "variant": doc.get("diagnostic_variant"),
        "poll_direction": doc.get("poll_direction"),
        "anchor": doc.get("diagnostic_anchor"),
        "source_gold_direction": doc.get("source_gold_direction"),
        "source_target_object": doc.get("source_target_object"),
        "source_direction_question": doc.get("source_direction_question"),
        "target_option_letter": doc.get("target_option_letter"),
        "none_option_letter": doc.get("none_option_letter"),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "selected_answer": _selected_answer(doc, parsed),
        "target_evidence": evidence,
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
        "prediction": prediction,
    }
    metric_names = (
        "accuracy",
        "target_poll_accuracy",
        "target_positive_accuracy",
        "target_negative_accuracy",
        "final_direction_accuracy",
        "direction_selection_coverage",
        "ambiguous_direction_rate",
        "exact_poll_set_accuracy",
        "parse_success_rate",
    )
    output = {name: dict(entry) for name in metric_names}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "options": list(doc["options"]),
        "img_path": doc.get("img_path"),
        "image_path": doc.get("image_path"),
        "orientation_source": "COMFORT ground-truth scene metadata",
        "geometry_used": False,
    }
    return output


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _groups(results):
    grouped = defaultdict(dict)
    for row in results:
        grouped[row.get("source_qid")][row.get("poll_direction")] = row
    return [group for group in grouped.values() if set(group) >= set(base.DIRECTIONS)]


def _select_direction(group):
    """Apply the no-geometry score selection, including stable tie handling."""
    ranked = sorted(
        enumerate(base.DIRECTIONS),
        key=lambda item: -float(group[item[1]]["target_evidence"]),
    )
    best_direction = ranked[0][1]
    best_score = float(group[best_direction]["target_evidence"])
    second_score = float(group[ranked[1][1]]["target_evidence"])
    margin = best_score - second_score
    return (
        best_direction if margin >= MINIMUM_DIRECTION_MARGIN else None,
        margin,
    )


def aggregate_accuracy(results):
    return _mean(row["score"] for row in results)


aggregate_target_poll_accuracy = aggregate_accuracy


def aggregate_target_positive_accuracy(results):
    return _mean(row["score"] for row in results if row["poll_direction"] == row["source_gold_direction"])


def aggregate_target_negative_accuracy(results):
    return _mean(row["score"] for row in results if row["poll_direction"] != row["source_gold_direction"])


def aggregate_final_direction_accuracy(results):
    return _mean(float(_select_direction(group)[0] == next(iter(group.values()))["source_gold_direction"]) for group in _groups(results))


def aggregate_direction_selection_coverage(results):
    return _mean(float(_select_direction(group)[0] is not None) for group in _groups(results))


def aggregate_ambiguous_direction_rate(results):
    return _mean(float(_select_direction(group)[0] is None) for group in _groups(results))


def aggregate_exact_poll_set_accuracy(results):
    return _mean(float(all(group[direction]["score"] == 1.0 for direction in base.DIRECTIONS)) for group in _groups(results))


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def _analysis(results):
    groups = _groups(results)
    confusion = {direction: Counter() for direction in base.DIRECTIONS}
    evidence_patterns = Counter()
    margins = []
    for group in groups:
        gold = next(iter(group.values()))["source_gold_direction"]
        predicted, margin = _select_direction(group)
        confusion[gold][predicted or "ambiguous"] += 1
        evidence_patterns[tuple(group[direction]["target_evidence"] for direction in base.DIRECTIONS)] += 1
        margins.append(margin)
    return {
        "direction_order": list(base.DIRECTIONS),
        "minimum_direction_margin": MINIMUM_DIRECTION_MARGIN,
        "geometry_used": False,
        "orientation_source": "COMFORT ground-truth scene metadata",
        "gold_to_predicted_direction": {direction: dict(sorted(counts.items())) for direction, counts in confusion.items()},
        "target_evidence_patterns": {",".join(str(value) for value in pattern): count for pattern, count in sorted(evidence_patterns.items())},
        "mean_direction_margin": _mean(margins),
        "target_poll_accuracy": aggregate_target_poll_accuracy(results),
        "target_positive_accuracy": aggregate_target_positive_accuracy(results),
        "target_negative_accuracy": aggregate_target_negative_accuracy(results),
        "final_direction_accuracy": aggregate_final_direction_accuracy(results),
        "direction_selection_coverage": aggregate_direction_selection_coverage(results),
        "ambiguous_direction_rate": aggregate_ambiguous_direction_rate(results),
        "exact_poll_set_accuracy": aggregate_exact_poll_set_accuracy(results),
    }


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": TASK_NAME,
        "pipeline": "target_only_inversion_no_geometry",
        "num_records": len(results),
        "num_source_questions": len(_groups(results)),
        "analysis": _analysis(results),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT target-only GT polling records saved to {}.", path)
