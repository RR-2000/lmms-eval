"""Validate direction-to-object conversion on available Kubric directions."""

import json
from collections import Counter, defaultdict

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object_polling import utils as polling
from lmms_eval.tasks.kubric_movi_a.utils import _get_image_path, _get_options
from lmms_eval.tasks.kubric_movi_a_direction_object import utils as base
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "kubric_movi_a_direction_object_relative_direction_polling"
BASELINE_VARIANT = polling.BASELINE_VARIANT
POLL_VARIANTS = polling.POLL_VARIANTS


def _map_key(doc):
    return (
        _get_image_path(doc),
        str(doc.get("coordinate_frame", "")),
        str(doc.get("reference_object", "")),
        str(doc.get("source_task_family", "")),
    )


def process_docs(dataset: Dataset) -> Dataset:
    """Poll every uniquely authored direction available for each map."""
    normalized = list(base.relative_direction_process_docs(dataset))
    pairs = defaultdict(dict)
    for row in normalized:
        pairs[row["source_qid"]][row["diagnostic_answer_format"]] = row

    maps = defaultdict(lambda: defaultdict(list))
    for pair in pairs.values():
        direction = pair.get("direction")
        object_answer = pair.get("object")
        if direction is None or object_answer is None:
            continue
        maps[_map_key(direction)][direction["diagnostic_relation"]].append(pair)

    usable_maps = {
        key: {direction: rows[0] for direction, rows in by_direction.items()}
        for key, by_direction in maps.items()
        if all(len(rows) == 1 for rows in by_direction.values())
    }

    records = []
    for map_index, (key, by_direction) in enumerate(sorted(usable_maps.items())):
        image_path, coordinate_frame, anchor, family = key
        map_id = f"map_{map_index:04d}"
        available_directions = [
            direction for direction in base.DIRECTIONS if direction in by_direction
        ]
        for source_direction in available_directions:
            direction_row = by_direction[source_direction]["direction"]
            source_id = direction_row["source_qid"]
            common = {
                "conversion_source_qid": source_id,
                "source_qid": source_id,
                "source_direction_question": direction_row["question"],
                "source_gold_direction": source_direction,
                "source_target_object": direction_row["diagnostic_target_object"],
                "conversion_map_id": map_id,
                "conversion_coordinate_frame": coordinate_frame,
                "conversion_source_family": family,
                "conversion_total_maps": len(maps),
                "conversion_usable_maps": len(usable_maps),
                "available_poll_directions": available_directions,
                "conversion_image_path": image_path,
                "conversion_anchor": anchor,
            }

            baseline = dict(direction_row)
            baseline.update(common)
            baseline.update(
                {
                    "diagnostic_variant": BASELINE_VARIANT,
                    "poll_direction": None,
                }
            )
            rows = [baseline]
            for poll_direction in available_directions:
                native_object = by_direction[poll_direction]["object"]
                poll = dict(native_object)
                poll.update(common)
                poll.update(
                    {
                        "qid": f"{source_id}::poll_{poll_direction}",
                        "index": f"{source_id}::poll_{poll_direction}",
                        "diagnostic_variant": f"poll_{poll_direction}",
                        "diagnostic_answer_format": "object",
                        "poll_direction": poll_direction,
                    }
                )
                rows.append(poll)
            records.extend(rows)

    eval_logger.info(
        "Kubric relative-direction polling retained {} of {} unambiguous maps, producing {} source questions and {} total rows.",
        len(usable_maps),
        len(maps),
        sum(len(by_direction) for by_direction in usable_maps.values()),
        len(records),
    )
    return Dataset.from_list(records)


doc_to_visual = base.doc_to_visual
doc_to_text = base.doc_to_text
doc_to_target = base.doc_to_target


def _selected_answer(doc, letter):
    return _get_options(doc).get(letter) if letter else None


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = base._extract_answer(prediction)
    gold = str(doc.get("answer", "")).strip()
    entry = {
        "qid": doc.get("qid"),
        "source_qid": doc.get("conversion_source_qid"),
        "scene_id": doc.get("conversion_map_id"),
        "map_id": doc.get("conversion_map_id"),
        "sequence_name": doc.get("sequence_name"),
        "frame_index": doc.get("frame_index"),
        "source_task_family": doc.get("conversion_source_family"),
        "coordinate_frame": doc.get("conversion_coordinate_frame"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "poll_direction": doc.get("poll_direction"),
        "anchor": doc.get("conversion_anchor"),
        "source_gold_direction": doc.get("source_gold_direction"),
        "source_target_object": doc.get("source_target_object"),
        "source_direction_question": doc.get("source_direction_question"),
        "gold_answer": doc.get("diagnostic_target"),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "selected_answer": _selected_answer(doc, parsed),
        "parse_success": parsed is not None,
        "score": float(parsed == gold),
        "prediction": prediction,
        "total_maps": doc.get("conversion_total_maps"),
        "usable_maps": doc.get("conversion_usable_maps"),
        "available_poll_directions": doc.get("available_poll_directions"),
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
        "options": list(_get_options(doc).values()),
        "img_path": _get_image_path(doc),
    }
    return result


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _groups(results):
    grouped = defaultdict(dict)
    for row in results:
        grouped[row.get("source_qid")][row.get("variant")] = row
    complete = []
    for group in grouped.values():
        baseline = group.get(BASELINE_VARIANT)
        if baseline is None:
            continue
        required = {
            f"poll_{direction}"
            for direction in baseline.get("available_poll_directions", [])
        }
        if required <= set(group):
            complete.append(group)
    return complete


def _available_directions(group):
    return list(group[BASELINE_VARIANT].get("available_poll_directions", []))


def aggregate_accuracy(results):
    return _mean(row["score"] for row in results)


def aggregate_direct_answer_accuracy(results):
    return _mean(row["score"] for row in results if row["variant"] == BASELINE_VARIANT)


aggregate_direction_accuracy = aggregate_direct_answer_accuracy


def aggregate_poll_accuracy(results):
    return _mean(row["score"] for row in results if str(row["variant"]).startswith("poll_"))


def _aggregate_named_poll(results, direction):
    return _mean(row["score"] for row in results if row["variant"] == f"poll_{direction}")


def aggregate_left_poll_accuracy(results):
    return _aggregate_named_poll(results, "left")


def aggregate_right_poll_accuracy(results):
    return _aggregate_named_poll(results, "right")


def aggregate_front_poll_accuracy(results):
    return _aggregate_named_poll(results, "front")


def aggregate_behind_poll_accuracy(results):
    return _aggregate_named_poll(results, "behind")


def _chosen_poll(group):
    direction = group[BASELINE_VARIANT].get("selected_answer")
    return group.get(f"poll_{direction}")


def _pipeline_correct(group):
    poll = _chosen_poll(group)
    return bool(
        poll
        and poll.get("selected_answer")
        == group[BASELINE_VARIANT].get("source_target_object")
    )


def aggregate_final_polled_answer_accuracy(results):
    return _mean(float(_pipeline_correct(group)) for group in _groups(results))


aggregate_full_pipeline_accuracy = aggregate_final_polled_answer_accuracy


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
    groups = [group for group in _groups(results) if group[BASELINE_VARIANT]["score"] == 1.0]
    return _mean(float(_pipeline_correct(group)) for group in groups)


def aggregate_pipeline_accuracy_given_direction_wrong(results):
    groups = [group for group in _groups(results) if group[BASELINE_VARIANT]["score"] == 0.0]
    return _mean(float(_pipeline_correct(group)) for group in groups)


def aggregate_exact_poll_map_accuracy(results):
    return _mean(
        float(all(group[f"poll_{direction}"]["score"] == 1.0 for direction in _available_directions(group)))
        for group in _groups(results)
    )


def _unique_poll_answers(group):
    return len(
        {
            group[f"poll_{direction}"].get("selected_answer")
            for direction in _available_directions(group)
            if group[f"poll_{direction}"].get("selected_answer") is not None
        }
    )


def aggregate_collision_free_poll_map_rate(results):
    return _mean(
        float(_unique_poll_answers(group) == len(_available_directions(group)))
        for group in _groups(results)
    )


def aggregate_mean_unique_poll_answer_ratio(results):
    return _mean(
        _unique_poll_answers(group) / len(_available_directions(group))
        for group in _groups(results)
    )


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def _analysis(results):
    variants = (BASELINE_VARIANT, *POLL_VARIANTS)
    selected_by_variant = {}
    letters_by_variant = {}
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        selected_by_variant[variant] = dict(
            sorted(Counter(row.get("selected_answer") or "parse_failure" for row in rows).items())
        )
        letters_by_variant[variant] = dict(
            sorted(Counter(row.get("predicted_option_letter") or "parse_failure" for row in rows).items())
        )

    baseline_confusion = {direction: Counter() for direction in base.DIRECTIONS}
    poll_confusion = {direction: Counter() for direction in base.DIRECTIONS}
    for group in _groups(results):
        baseline = group[BASELINE_VARIANT]
        baseline_confusion[baseline["source_gold_direction"]][
            baseline.get("selected_answer") or "parse_failure"
        ] += 1
        object_to_direction = {
            group[f"poll_{direction}"]["gold_answer"]: direction
            for direction in _available_directions(group)
        }
        for direction in _available_directions(group):
            selected = group[f"poll_{direction}"].get("selected_answer")
            selected_direction = object_to_direction.get(selected, "other_or_parse_failure")
            poll_confusion[direction][selected_direction] += 1

    analysis = {
        "selected_answer_distribution_by_variant": selected_by_variant,
        "predicted_letter_distribution_by_variant": letters_by_variant,
        "gold_direction_to_predicted_direction": {
            direction: dict(sorted(counts.items()))
            for direction, counts in baseline_confusion.items()
        },
        "poll_direction_to_selected_object_direction": {
            direction: dict(sorted(counts.items()))
            for direction, counts in poll_confusion.items()
        },
        "direct_answer_accuracy": aggregate_direct_answer_accuracy(results),
        "final_polled_answer_accuracy": aggregate_final_polled_answer_accuracy(results),
        "poll_accuracy": aggregate_poll_accuracy(results),
        "strict_two_stage_accuracy": aggregate_strict_two_stage_accuracy(results),
        "chosen_poll_accuracy": aggregate_chosen_poll_accuracy(results),
        "oracle_direction_poll_accuracy": aggregate_oracle_direction_poll_accuracy(results),
        "pipeline_accuracy_given_direction_correct": aggregate_pipeline_accuracy_given_direction_correct(results),
        "pipeline_accuracy_given_direction_wrong": aggregate_pipeline_accuracy_given_direction_wrong(results),
        "exact_poll_map_accuracy": aggregate_exact_poll_map_accuracy(results),
        "collision_free_poll_map_rate": aggregate_collision_free_poll_map_rate(results),
        "mean_unique_poll_answer_ratio": aggregate_mean_unique_poll_answer_ratio(results),
    }
    by_family = {}
    for family in sorted({row.get("source_task_family") for row in results}):
        family_rows = [row for row in results if row.get("source_task_family") == family]
        by_family[family] = {
            "source_questions": len(_groups(family_rows)),
            "direct_answer_accuracy": aggregate_direct_answer_accuracy(family_rows),
            "poll_accuracy": aggregate_poll_accuracy(family_rows),
            "final_polled_answer_accuracy": aggregate_final_polled_answer_accuracy(family_rows),
            "strict_two_stage_accuracy": aggregate_strict_two_stage_accuracy(family_rows),
            "exact_poll_map_accuracy": aggregate_exact_poll_map_accuracy(family_rows),
        }
    analysis["by_source_task_family"] = by_family
    return analysis


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    first = results[0] if results else {}
    report = {
        "dataset": "movi_a_relative_direction",
        "task": TASK_NAME,
        "map_coverage": {
            "usable_maps": first.get("usable_maps", 0),
            "total_maps": first.get("total_maps", 0),
            "coverage": (
                first.get("usable_maps", 0) / first.get("total_maps", 1)
                if first.get("total_maps", 0)
                else 0.0
            ),
        },
        "num_records": len(results),
        "num_source_questions": len(_groups(results)),
        "analysis": _analysis(results),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("Kubric relative-direction polling records saved to {}.", path)
