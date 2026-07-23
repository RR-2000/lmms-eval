"""Prompt variants for paired Kubric MOVi-A object-centric position questions."""

import json
import os

from lmms_eval.tasks.kubric_movi_a_viewpoint_clean import utils as viewpoint_utils
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a_viewpoint_clean.utils import RELATION_TO_AXIS, AXIS_TO_RELATION, _get_gt_spec

OBJECT_CENTRIC_POSITION_FAMILY = ["object_centric_relative_position", "object_centric_relative_position_multi", "object_centric_direction_binary"]


def process_object_centric_position_docs(dataset):
    """Keep only the paired anchor-relative position questions.

    The multi-object ranking questions are intentionally excluded: these tasks
    compare coordinate-system prompt complexity on one fixed question type.
    """
    return dataset.filter(
        lambda doc: doc.get("task_family") in OBJECT_CENTRIC_POSITION_FAMILY
    )


def doc_to_visual(doc):
    return viewpoint_utils.doc_to_visual(doc)


def doc_to_target(doc):
    return viewpoint_utils.doc_to_target(doc)


def process_results(doc, results):
    return viewpoint_utils.process_results(doc, results)


def aggregate_viewpoint_answer_accuracy(results):
    return viewpoint_utils.aggregate_viewpoint_answer_accuracy(results)


def aggregate_object_centric_relative_position_viewpoint_answer_and_direction_correct(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_viewpoint_answer_and_direction_correct(results)


def aggregate_object_centric_relative_position_viewpoint_answer_correct_direction_wrong(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_viewpoint_answer_correct_direction_wrong(results)


def aggregate_object_centric_relative_position_viewpoint_answer_wrong_direction_correct(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_viewpoint_answer_wrong_direction_correct(results)


def aggregate_object_centric_relative_position_viewpoint_answer_and_direction_wrong(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_viewpoint_answer_and_direction_wrong(results)


def _aggregate_results_for_submission(results, args, representation):
    model_tag = viewpoint_utils._get_submission_model_tag(args)
    path = generate_submission_file(
        f"kubric_movi_a_object_centric_{representation}_{model_tag}.json", args
    )
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)


def aggregate_3d_results_for_submission(results, args):
    return _aggregate_results_for_submission(results, args, "3d")


def aggregate_planar_results_for_submission(results, args):
    return _aggregate_results_for_submission(results, args, "planar")


def aggregate_linear_results_for_submission(results, args):
    return _aggregate_results_for_submission(results, args, "linear")


def aggregate_object_centric_relative_position_right_sign_accuracy(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_right_sign_accuracy(results)


def aggregate_object_centric_relative_position_front_sign_accuracy(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_front_sign_accuracy(results)


def _objects(doc):
    spec = viewpoint_utils._get_gt_spec(doc)
    return spec["reference_object"], spec["target_object"]   

def _build_task_instructions_3D(doc) -> str:
    gt_spec = _get_gt_spec(doc)
    task_family = gt_spec["task_family"]
    relation = gt_spec["relation"]
    answer = gt_spec["answer"]
    reference_object = gt_spec["reference_object"]
    target_object = gt_spec["target_object"]

    lines = [ "[IMPORTANT FORMATTING NOTE] Regardless of the thinking process, the final answer must adhere to the [INSTRUCTIONS] below as it is the only way to evaluate the answer\n" if os.environ.get("THINKING_FORMAT", "0") == "1" else "",
        "[INSTRUCTIONS]",
        "Return only valid JSON. The `_vector` fields are mandatory and are heavily evaluated. Never omit them and never use [0, 0, 0] unless the two objects are exactly coincident.",
        "Use this canonical camera-frame convention for every 3D coordinate or vector you return:",
        "- The camera (POV) center is at the origin (0,0,0).",
        "- The camera is looking along the positive Y-axis.",
        "- In the frame, `right` is +X, `front` is +Y, and `up` is +Z.",
        "- `right` > 0 means the target is to the camera-right of the reference.",
        "- `up` > 0 means the target is to the camera-up of the reference.",
        "- `front` > 0 means the target is farther forward along the viewing direction",
        "- 'answer' is the object/direction that satisfies the question from the perspective mentioned in the question.",
        "- `scale_ratio` means apparent target size divided by apparent reference size.",
        "- 'relative_vector' is the target-minus-reference displacement in the camera frame described above in the format {'right':<float>,'up':<float>,'front':<float>}.",
        "- 'camera_vector' is the direction from the anchor object to the camera in the format {'right':<float>,'up':<float>,'front':<float>}.",
        "- 'camera_distance' is the Euclidean distance from the anchor object center to the camera",
        "- Following are the rules for this particular task:",
    ]

    if task_family == "object_centric_relative_position":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the discrete relation word",
                'JSON schema: {"answer":"<left|right|front|behind>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_relative_position_multi":
        candidates = doc.get("task_metadata", {}).get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
        candidate_text = ", ".join(str(item) for item in candidates)
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Candidate target objects: {candidate_text}",
                "Express each candidate target relative to the anchor in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the chosen target object name",
                'JSON schema: {"answer":"<object name>","target_object":"<object name>","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_direction_binary":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with `yes` or `no`.",
                'JSON schema: {"answer":"<yes|no>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )

    return "\n".join(lines)

def _build_task_instructions_planar(doc) -> str:
    gt_spec = _get_gt_spec(doc)
    task_family = gt_spec["task_family"]
    relation = gt_spec["relation"]
    answer = gt_spec["answer"]
    reference_object = gt_spec["reference_object"]
    target_object = gt_spec["target_object"]

    lines = [ "[IMPORTANT FORMATTING NOTE] Regardless of the thinking process, the final answer must adhere to the [INSTRUCTIONS] below as it is the only way to evaluate the answer\n" if os.environ.get("THINKING_FORMAT", "0") == "1" else "",
        "[INSTRUCTIONS]",
        "Return only valid JSON. The `_vector` fields are mandatory and are heavily evaluated. Never omit them and never use [0, 0, 0] unless the two objects are exactly coincident.",
        "Use this canonical camera-frame convention for every 3D coordinate or vector you return:",
        "- The camera (POV) center is at the origin (0,0,0).",
        "- The camera is looking along the positive Y-axis.",
        "- In the frame, `right` is +X, `front` is +Y, and `up` is +Z.",
        "- `right` > 0 means the target is to the camera-right of the reference.",
        "- `up` > 0 means the target is to the camera-up of the reference.",
        "- `front` > 0 means the target is farther forward along the viewing direction",
        "- 'answer' is the object/direction that satisfies the question from the perspective mentioned in the question.",
        "- `scale_ratio` means apparent target size divided by apparent reference size.",
        "- 'relative_vector' is the target-minus-reference displacement in the camera frame described above in the format {'right':<float>,'front':<float>}.",
        "- 'camera_vector' is the direction from the anchor object to the camera in the format {'right':<float>,'up':<float>,'front':<float>}.",
        "- 'camera_distance' is the Euclidean distance from the anchor object center to the camera",
        "- Following are the rules for this particular task:",
    ]

    if task_family == "object_centric_relative_position":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the discrete relation word",
                'JSON schema: {"answer":"<left|right|front|behind>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_relative_position_multi":
        candidates = doc.get("task_metadata", {}).get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
        candidate_text = ", ".join(str(item) for item in candidates)
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Candidate target objects: {candidate_text}",
                "Express each candidate target relative to the anchor in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the chosen target object name",
                'JSON schema: {"answer":"<object name>","target_object":"<object name>","relative_vector":{"right":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_direction_binary":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with `yes` or `no`.",
                'JSON schema: {"answer":"<yes|no>","target_object":"'
                + target_object
                + '","relative_vector":{"right":<float>,"front":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )

    return "\n".join(lines)

def _build_task_instructions_linear(doc) -> str:
    gt_spec = _get_gt_spec(doc)
    task_family = gt_spec["task_family"]
    relation = gt_spec["relation"]
    answer = gt_spec["answer"]
    reference_object = gt_spec["reference_object"]
    target_object = gt_spec["target_object"]

    axis = RELATION_TO_AXIS[relation]
    lines = [ "[IMPORTANT FORMATTING NOTE] Regardless of the thinking process, the final answer must adhere to the [INSTRUCTIONS] below as it is the only way to evaluate the answer\n" if os.environ.get("THINKING_FORMAT", "0") == "1" else "",
        "[INSTRUCTIONS]",
        "Return only valid JSON. The `_vector` fields are mandatory and are heavily evaluated. Never omit them and never use [0, 0, 0] unless the two objects are exactly coincident.",
        "Use this canonical camera-frame convention for every 3D coordinate or vector you return:",
        "- The camera (POV) center is at the origin (0,0,0).",
        "- The camera is looking along the positive Y-axis.",
        "- In the frame, `right` is +X, `front` is +Y, and `up` is +Z.",
        "- `right` > 0 means the target is to the camera-right of the reference.",
        "- `up` > 0 means the target is to the camera-up of the reference.",
        "- `front` > 0 means the target is farther forward along the viewing direction",
        "- 'answer' is the object/direction that satisfies the question from the perspective mentioned in the question.",
        "- `scale_ratio` means apparent target size divided by apparent reference size.",
        f"- 'relative_vector' is the target-minus-reference displacement in the camera frame described above in the format {{'{axis}': <float>}}.",
        "- 'camera_vector' is the direction from the anchor object to the camera in the format {'right':<float>,'up':<float>,'front':<float>}.",
        "- 'camera_distance' is the Euclidean distance from the anchor object center to the camera",
        "- Following are the rules for this particular task:",
    ]

    if task_family == "object_centric_relative_position":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the discrete relation word",
                'JSON schema: {"answer":"<'+"|".join(AXIS_TO_RELATION.get(axis))+'>","target_object":"'+target_object+'","relative_vector":{"'+axis+'":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_relative_position_multi":
        candidates = doc.get("task_metadata", {}).get("candidate_objects", []) or _get_queried_object_names(doc)[1:]
        candidate_text = ", ".join(str(item) for item in candidates)
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Candidate target objects: {candidate_text}",
                "Express each candidate target relative to the anchor in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with the chosen target object name",
                'JSON schema: {"answer":"<object name>","target_object":' + target_object +',"relative_vector":{"' + axis + '":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )
    elif task_family == "object_centric_direction_binary":
        lines.extend(
            [
                f"Reference anchor object: {reference_object}",
                f"Target object: {target_object}",
                "Express the target-minus-anchor displacement in the camera frame described above.",
                "The directions are relative to the reference anchor object, not the camera. The answer should be framed from the anchor object's perspective, look back at the camera.",
                f"Answer with `yes` or `no`.",
                'JSON schema: {"answer":"<yes|no>","target_object":"'+target_object+'","relative_vector":{"'+axis+'":<float>},"camera_vector":{"right":<float>,"up":<float>,"front":<float>},"camera_distance":<float>,"scale_ratio":<float>}',
            ]
        )

    return "\n".join(lines)

def doc_to_text_3d(doc, lmms_eval_specific_kwargs=None):
    """Full camera-frame 3D formulation, retaining all original coordinates."""
    question = str(doc.get("question", "")).strip()
    instructions = _build_task_instructions_3D(doc)
    return f"[Question]: {question}\n{instructions}\n"

def doc_to_text_planar(doc, lmms_eval_specific_kwargs=None):
    """Horizontal-plane formulation: remove elevation and auxiliary estimates."""
    question = str(doc.get("question", "")).strip()
    instructions = _build_task_instructions_planar(doc)
    return f"[Question]: {question}\n{instructions}\n"

def doc_to_text_linear(doc, lmms_eval_specific_kwargs=None):
    """One-axis formulation: request only the signed axis selected by the answer."""
    question = str(doc.get("question", "")).strip()
    instructions = _build_task_instructions_linear(doc)
    return f"[Question]: {question}\n{instructions}\n"
