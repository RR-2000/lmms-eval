"""Prompt variants for paired Kubric MOVi-A object-centric position questions."""

from lmms_eval.tasks.kubric_movi_a_viewpoint_clean import utils as viewpoint_utils


OBJECT_CENTRIC_POSITION_FAMILY = "object_centric_relative_position"


def process_object_centric_position_docs(dataset):
    """Keep only the paired anchor-relative position questions.

    The multi-object ranking questions are intentionally excluded: these tasks
    compare coordinate-system prompt complexity on one fixed question type.
    """
    return dataset.filter(
        lambda doc: doc.get("task_family") == OBJECT_CENTRIC_POSITION_FAMILY
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


def aggregate_object_centric_relative_position_right_sign_accuracy(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_right_sign_accuracy(results)


def aggregate_object_centric_relative_position_front_sign_accuracy(results):
    return viewpoint_utils.aggregate_object_centric_relative_position_front_sign_accuracy(results)


def _objects(doc):
    spec = viewpoint_utils._get_gt_spec(doc)
    return spec["reference_object"], spec["target_object"]


def doc_to_text_3d(doc, lmms_eval_specific_kwargs=None):
    """Full camera-frame 3D formulation, retaining all original coordinates."""
    question = str(doc.get("question", "")).strip()
    instructions = viewpoint_utils._build_task_instructions(doc)
    return f"Question: {question}\n{instructions}\n"


def doc_to_text_planar(doc, lmms_eval_specific_kwargs=None):
    """Horizontal-plane formulation: remove elevation and auxiliary estimates."""
    anchor, target = _objects(doc)
    question = str(doc.get("question", "")).strip()
    return "\n".join(
        [
            f"Question: {question}",
            "Use only the horizontal plane. Ignore height, distance, and object size.",
            f"Stand at {anchor} and face the camera. Decide where {target} is relative to {anchor}.",
            "On this plane, right/left is one axis and front/behind is the other axis.",
            "Return only valid JSON. Include a signed horizontal target-minus-anchor vector; omit `up`.",
            "JSON schema: {\"answer\":\"<left|right|front|behind>\",\"relative_vector\":{\"right\":<float>,\"front\":<float>}}.",
        ]
    )


def doc_to_text_linear(doc, lmms_eval_specific_kwargs=None):
    """One-axis formulation: request only the signed axis selected by the answer."""
    anchor, target = _objects(doc)
    return "\n".join(
        [
            f"Imagine you are standing at {anchor} and facing the camera.",
            f"Is {target} to your left, right, in front of you, or behind you?",
            "Return only valid JSON. Also include one signed target-minus-anchor vector component: use `right` for a left/right answer, or `front` for a front/behind answer.",
            "JSON schema: {\"answer\":\"<left|right|front|behind>\",\"relative_vector\":{\"<right|front>\":<float>}}.",
        ]
    )
