"""Validate direction-to-object conversion by polling all four directions."""

import json
from collections import Counter, defaultdict

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "comfort_direction_object_polling"
BASELINE_VARIANT = "direction"
POLL_VARIANTS = tuple(f"poll_{direction}" for direction in base.DIRECTIONS)
ALL_VARIANTS = (BASELINE_VARIANT, *POLL_VARIANTS)


def process_docs(dataset: Dataset) -> Dataset:
    """Expand every direction row into a baseline plus four object polls."""
    normalized = list(base.process_docs(dataset))
    direction_rows = [
        row for row in normalized if row["diagnostic_answer_format"] == "direction"
    ]
    object_rows = {
        (row["scene_id"], row["diagnostic_relation"], row["answer_idx"]): row
        for row in normalized
        if row["diagnostic_answer_format"] == "object"
    }

    records = []
    skipped = Counter()
    for direction_row in direction_rows:
        source_id = direction_row["qid"]
        common = {
            "conversion_source_qid": source_id,
            "source_qid": source_id,
            "source_relation_id": source_id,
            "source_direction_question": direction_row["diagnostic_question"],
            "source_gold_direction": direction_row["diagnostic_relation"],
            "source_target_object": direction_row["diagnostic_target_object"],
            "source_answer_index": direction_row["answer_idx"],
            "task_family": TASK_NAME,
        }

        baseline = dict(direction_row)
        baseline.update(common)
        baseline.update(
            {
                "diagnostic_variant": BASELINE_VARIANT,
                "poll_direction": None,
            }
        )

        polls = []
        complete = True
        for poll_direction in base.DIRECTIONS:
            key = (
                direction_row["scene_id"],
                poll_direction,
                direction_row["answer_idx"],
            )
            native_object = object_rows.get(key)
            if native_object is None:
                skipped[f"missing_{poll_direction}_poll"] += 1
                complete = False
                break
            poll = dict(native_object)
            poll.update(common)
            poll.update(
                {
                    "qid": f"{source_id}::poll_{poll_direction}",
                    "index": f"{source_id}::poll_{poll_direction}",
                    "diagnostic_variant": f"poll_{poll_direction}",
                    "diagnostic_answer_format": "object",
                    "diagnostic_question": (
                        "Which object is where the "
                        f"{'back' if poll_direction == 'behind' else poll_direction} "
                        f"side of the {direction_row['diagnostic_anchor']} is facing?"
                    ),
                    "poll_direction": poll_direction,
                }
            )
            polls.append(poll)
        if complete:
            records.extend([baseline, *polls])

    eval_logger.info(
        "COMFORT direction-to-object polling loaded {} source questions and {} polls; skipped={}.",
        len(records) // 5,
        (len(records) // 5) * 4,
        dict(skipped),
    )
    return Dataset.from_list(records)


doc_to_visual = base.doc_to_visual
doc_to_text = base.doc_to_text
doc_to_target = base.doc_to_target


def _selected_answer(doc: dict, letter):
    if letter not in base.OPTION_LETTERS:
        return None
    return doc["options"][base.OPTION_LETTERS.index(letter)]


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = base.extract_option_letter(prediction)
    gold = str(doc["gold_option_letter"]).upper()
    selected_answer = _selected_answer(doc, parsed)
    entry = {
        "qid": doc.get("qid"),
        "source_qid": doc.get("conversion_source_qid"),
        "scene_id": doc.get("scene_id"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "poll_direction": doc.get("poll_direction"),
        "anchor": doc.get("diagnostic_anchor"),
        "source_gold_direction": doc.get("source_gold_direction"),
        "source_target_object": doc.get("source_target_object"),
        "source_answer_index": doc.get("source_answer_index"),
        "source_direction_question": doc.get("source_direction_question"),
        "gold_answer": doc.get("answer"),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "selected_answer": selected_answer,
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
        "prediction": prediction,
    }
    metric_names = (
        "accuracy",
        "direct_answer_accuracy",
        "final_polled_answer_accuracy",
        "direction_accuracy",
        "poll_accuracy",
        "left_poll_accuracy",
        "right_poll_accuracy",
        "front_poll_accuracy",
        "behind_poll_accuracy",
        "full_pipeline_accuracy",
        "strict_two_stage_accuracy",
        "chosen_poll_accuracy",
        "oracle_direction_poll_accuracy",
        "pipeline_accuracy_given_direction_correct",
        "pipeline_accuracy_given_direction_wrong",
        "exact_poll_map_accuracy",
        "collision_free_poll_map_rate",
        "mean_unique_poll_answer_ratio",
        "parse_success_rate",
    )
    result = {name: dict(entry) for name in metric_names}
    result["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "options": list(doc["options"]),
        "img_path": doc.get("img_path"),
        "image_path": doc.get("image_path"),
    }
    return result


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _groups(results):
    grouped = defaultdict(dict)
    for row in results:
        grouped[row.get("source_qid")][row.get("variant")] = row
    return [group for group in grouped.values() if set(group) >= set(ALL_VARIANTS)]


def aggregate_accuracy(results):
    return _mean(row["score"] for row in results)


def aggregate_direction_accuracy(results):
    return _mean(
        row["score"] for row in results if row["variant"] == BASELINE_VARIANT
    )


def aggregate_direct_answer_accuracy(results):
    return aggregate_direction_accuracy(results)


def aggregate_poll_accuracy(results):
    return _mean(row["score"] for row in results if row["variant"] in POLL_VARIANTS)


def _aggregate_named_poll(results, direction):
    return _mean(
        row["score"] for row in results if row["variant"] == f"poll_{direction}"
    )


def aggregate_left_poll_accuracy(results):
    return _aggregate_named_poll(results, "left")


def aggregate_right_poll_accuracy(results):
    return _aggregate_named_poll(results, "right")


def aggregate_front_poll_accuracy(results):
    return _aggregate_named_poll(results, "front")


def aggregate_behind_poll_accuracy(results):
    return _aggregate_named_poll(results, "behind")


def _pipeline_correct(group):
    direction = group[BASELINE_VARIANT].get("selected_answer")
    poll = group.get(f"poll_{direction}")
    return bool(
        poll
        and poll.get("selected_answer")
        == group[BASELINE_VARIANT].get("source_target_object")
    )


def aggregate_full_pipeline_accuracy(results):
    return _mean(float(_pipeline_correct(group)) for group in _groups(results))


def aggregate_final_polled_answer_accuracy(results):
    return aggregate_full_pipeline_accuracy(results)


def _chosen_poll(group):
    direction = group[BASELINE_VARIANT].get("selected_answer")
    return group.get(f"poll_{direction}")


def aggregate_strict_two_stage_accuracy(results):
    return _mean(
        float(
            group[BASELINE_VARIANT]["score"] == 1.0
            and _chosen_poll(group) is not None
            and _chosen_poll(group)["score"] == 1.0
        )
        for group in _groups(results)
    )


def aggregate_chosen_poll_accuracy(results):
    return _mean(
        float(poll["score"] == 1.0)
        for group in _groups(results)
        if (poll := _chosen_poll(group)) is not None
    )


def aggregate_oracle_direction_poll_accuracy(results):
    return _mean(
        group[f"poll_{group[BASELINE_VARIANT]['source_gold_direction']}"]["score"]
        for group in _groups(results)
    )


def aggregate_pipeline_accuracy_given_direction_correct(results):
    groups = [
        group for group in _groups(results) if group[BASELINE_VARIANT]["score"] == 1.0
    ]
    return _mean(float(_pipeline_correct(group)) for group in groups)


def aggregate_pipeline_accuracy_given_direction_wrong(results):
    groups = [
        group for group in _groups(results) if group[BASELINE_VARIANT]["score"] == 0.0
    ]
    return _mean(float(_pipeline_correct(group)) for group in groups)


def aggregate_exact_poll_map_accuracy(results):
    return _mean(
        float(all(group[variant]["score"] == 1.0 for variant in POLL_VARIANTS))
        for group in _groups(results)
    )


def _unique_poll_answers(group):
    return len(
        {
            group[variant].get("selected_answer")
            for variant in POLL_VARIANTS
            if group[variant].get("selected_answer") is not None
        }
    )


def aggregate_collision_free_poll_map_rate(results):
    return _mean(
        float(_unique_poll_answers(group) == len(POLL_VARIANTS))
        for group in _groups(results)
    )


def aggregate_mean_unique_poll_answer_ratio(results):
    return _mean(
        _unique_poll_answers(group) / len(POLL_VARIANTS) for group in _groups(results)
    )


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def _analysis(results):
    selected_by_variant = {}
    letters_by_variant = {}
    for variant in ALL_VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        selected_by_variant[variant] = dict(
            sorted(Counter(row.get("selected_answer") or "parse_failure" for row in rows).items())
        )
        letters_by_variant[variant] = dict(
            sorted(Counter(row.get("predicted_option_letter") or "parse_failure" for row in rows).items())
        )

    direction_confusion = {
        direction: Counter() for direction in base.DIRECTIONS
    }
    baseline_confusion = {direction: Counter() for direction in base.DIRECTIONS}
    for group in _groups(results):
        baseline = group[BASELINE_VARIANT]
        baseline_confusion[baseline["source_gold_direction"]][
            baseline.get("selected_answer") or "parse_failure"
        ] += 1
        object_to_direction = {
            group[f"poll_{direction}"]["gold_answer"]: direction
            for direction in base.DIRECTIONS
        }
        for direction in base.DIRECTIONS:
            selected = group[f"poll_{direction}"].get("selected_answer")
            selected_direction = object_to_direction.get(selected, "parse_failure_or_other")
            direction_confusion[direction][selected_direction] += 1

    permutation_answers = defaultdict(list)
    for row in results:
        permutation_answers[
            (row["scene_id"], row["source_gold_direction"], row["variant"])
        ].append(row.get("selected_answer"))
    permutation_consistency = {}
    for variant in ALL_VARIANTS:
        groups = [
            answers
            for (_, _, grouped_variant), answers in permutation_answers.items()
            if grouped_variant == variant and len(answers) == len(base.OPTION_LETTERS)
        ]
        permutation_consistency[variant] = {
            "groups": len(groups),
            "consistent_groups": sum(
                len(set(answers)) == 1 and answers[0] is not None for answers in groups
            ),
            "rate": _mean(
                float(len(set(answers)) == 1 and answers[0] is not None)
                for answers in groups
            ),
        }

    return {
        "selected_answer_distribution_by_variant": selected_by_variant,
        "predicted_letter_distribution_by_variant": letters_by_variant,
        "poll_direction_to_selected_object_direction": {
            direction: dict(sorted(counts.items()))
            for direction, counts in direction_confusion.items()
        },
        "gold_direction_to_predicted_direction": {
            direction: dict(sorted(counts.items()))
            for direction, counts in baseline_confusion.items()
        },
        "semantic_answer_permutation_consistency": permutation_consistency,
        "direct_answer_accuracy": aggregate_direct_answer_accuracy(results),
        "final_polled_answer_accuracy": aggregate_final_polled_answer_accuracy(results),
        "full_pipeline_accuracy": aggregate_full_pipeline_accuracy(results),
        "strict_two_stage_accuracy": aggregate_strict_two_stage_accuracy(results),
        "chosen_poll_accuracy": aggregate_chosen_poll_accuracy(results),
        "oracle_direction_poll_accuracy": aggregate_oracle_direction_poll_accuracy(results),
        "pipeline_accuracy_given_direction_correct": (
            aggregate_pipeline_accuracy_given_direction_correct(results)
        ),
        "pipeline_accuracy_given_direction_wrong": (
            aggregate_pipeline_accuracy_given_direction_wrong(results)
        ),
        "exact_poll_map_accuracy": aggregate_exact_poll_map_accuracy(results),
        "collision_free_poll_map_rate": aggregate_collision_free_poll_map_rate(results),
        "mean_unique_poll_answer_ratio": aggregate_mean_unique_poll_answer_ratio(results),
    }


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": TASK_NAME,
        "num_records": len(results),
        "num_source_questions": len(_groups(results)),
        "analysis": _analysis(results),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT direction-to-object polling records saved to {}.", path)
