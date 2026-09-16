"""COMFORT_Oriented_3D variant of the direction-versus-object task."""

from pathlib import Path

from datasets import Dataset

from lmms_eval.tasks.comfort_direction_object import utils as base


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Oriented_3D")
DATASET_NAME = "COMFORT_Oriented_3D"
TASK_NAME = "comfort_oriented_3d_direction_object"


def process_docs(dataset: Dataset) -> Dataset:
    return base.process_docs_for_dataset(
        dataset,
        data_root=DATA_ROOT,
        dataset_name=DATASET_NAME,
        task_name=TASK_NAME,
        pair_by_answer_index=False,
    )


doc_to_visual = base.doc_to_visual
doc_to_text = base.doc_to_text
doc_to_target = base.doc_to_target
process_results = base.process_results
aggregate_accuracy = base.aggregate_accuracy
aggregate_object_answer_accuracy = base.aggregate_object_answer_accuracy
aggregate_direction_answer_accuracy = base.aggregate_direction_answer_accuracy
aggregate_object_minus_direction = base.aggregate_object_minus_direction
aggregate_format_switch_gain = base.aggregate_format_switch_gain
aggregate_parse_success_rate = base.aggregate_parse_success_rate
aggregate_object_parse_success_rate = base.aggregate_object_parse_success_rate
aggregate_direction_parse_success_rate = base.aggregate_direction_parse_success_rate
aggregate_object_correct_direction_wrong = base.aggregate_object_correct_direction_wrong
aggregate_direction_correct_object_wrong = base.aggregate_direction_correct_object_wrong
aggregate_both_correct = base.aggregate_both_correct
aggregate_both_wrong = base.aggregate_both_wrong


def aggregate_results_for_submission(results, args):
    return base.aggregate_results_for_submission_for_task(
        results,
        args,
        dataset_name=DATASET_NAME,
        task_name=TASK_NAME,
    )
