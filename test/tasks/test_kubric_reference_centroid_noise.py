import numpy as np
from datasets import Dataset

from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_reference_centroid_noise import (
    utils,
)


def _dataset():
    objects = []
    specifications = (
        ("reference", [20, 40, 40, 60], [0.3, 0.5], [0.0, 0.0, 0.0]),
        ("target", [70, 40, 90, 60], [0.8, 0.5], [1.0, 0.0, 0.0]),
        ("wrong one", [45, 10, 55, 20], [0.5, 0.15], [0.0, 1.0, 0.0]),
        ("wrong two", [45, 80, 55, 90], [0.5, 0.85], [0.0, -1.0, 0.0]),
        ("wrong three", [5, 5, 15, 15], [0.1, 0.1], [-1.0, 1.0, 0.0]),
    )
    for index, (name, box, image_position, position) in enumerate(specifications):
        objects.append({
            "object_idx": index,
            "name": name,
            "bbox_2d_xyxy_pixels": box,
            "image_position_2d": image_position,
            "position_3d": position,
        })
    common = {
        "pair_id": "pair-1",
        "task_family": "object_relative_direction",
        "reference_object": "reference",
        "target_object": "target",
        "relation": "right",
        "visible_objects": objects,
        "candidate_objects": ["target", "wrong one", "wrong two", "wrong three"],
        "task_metadata": {"relative_coordinates": {"right": 1.0, "front": 0.0}},
        "A": "target",
        "B": "wrong one",
        "C": "wrong two",
        "D": "wrong three",
        "answer": "A",
    }
    return Dataset.from_list([
        {**common, "qid": "pair-1::direction", "index": "pair-1::direction", "answer_format": "direction", "question": "Direction?", "A": "right", "B": "left", "C": "front", "D": "behind"},
        {**common, "qid": "pair-1::object", "index": "pair-1::object", "answer_format": "object", "question": "Which object?"},
    ])


def test_e3_has_four_levels_two_views_and_only_object_answers():
    docs = utils.process_docs(_dataset())
    assert len(docs) == 8
    assert set(docs["diagnostic_answer_format"]) == {"object"}
    assert set(docs["representation"]) == set(utils.REPRESENTATIONS)
    assert set(docs["noise_level"]) == set(utils.LEVELS)


def test_shift_is_shared_between_views_and_scaled_across_levels():
    docs = [dict(doc) for doc in utils.process_docs(_dataset())]
    for level in utils.LEVELS:
        shifts = [doc["reference_centroid_shift_pixels"] for doc in docs if doc["noise_level"] == level]
        assert len(shifts) == 2
        assert np.allclose(shifts[0], shifts[1])
    base = utils._reference_shift_pixels(docs[0], utils.LEVELS[0]) / utils.LEVELS[0]
    for level in utils.LEVELS[1:]:
        assert np.allclose(utils._reference_shift_pixels(docs[0], level) / level, base)


def test_geometric_curve_counts_each_question_once_not_once_per_view():
    docs = utils.process_docs(_dataset())
    curve = utils.geometric_curve(docs)
    assert [row["num_trials"] for row in curve] == [1, 1, 1, 1]
    assert all(0.0 <= row["wrong_box_fraction"] <= 1.0 for row in curve)
