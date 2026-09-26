import importlib
import importlib.util
from pathlib import Path

import pytest


utils = importlib.import_module("lmms_eval.tasks.3dsrbench_custom.utils")


def _row(qid, target):
    return {
        "qid": qid,
        "index": qid,
        "image_name": "scene-1",
        "qtype": "multi_object",
        "relation": "viewpoint towards object",
        "subject": "car",
        "object1": target,
        "original_question": f"Which side of the car points toward the {target}?",
        "A": "left",
        "B": "right",
        "C": "front",
        "D": "back",
        "answer": "A",
    }


def test_qwen_padding_uses_manifest_objects_without_thematic_fallback():
    result = utils._direction_object_process_docs(
        [_row("q1", "traffic sign"), _row("q2", "bench")],
        sample_seed=utils.QWEN_DIRECTION_OBJECT_SAMPLE_SEED,
        qwen_scene_objects={"scene-1": ["tree", "bicycle", "lamp post"]},
    )
    rows = [dict(row) for row in result]
    assert len(rows) == 4
    inverse = next(row for row in rows if row["qid"] == "q1::inverse")
    assert set(inverse[letter] for letter in "ABCD") == {
        "traffic sign", "bench", "tree", "bicycle"
    }
    assert inverse["diagnostic_generated_object_distractors"] == ["tree", "bicycle"]
    assert inverse["diagnostic_object_distractor_source"] == utils.QWEN_DIRECTION_OBJECT_MODEL


def test_qwen_padding_rejects_an_incomplete_scene_catalog():
    with pytest.raises(ValueError, match="did not provide enough"):
        utils._direction_object_process_docs(
            [_row("q1", "traffic sign"), _row("q2", "bench")],
            sample_seed=utils.QWEN_DIRECTION_OBJECT_SAMPLE_SEED,
            qwen_scene_objects={"scene-1": ["tree"]},
        )


def test_generator_recovers_complete_labels_from_truncated_qwen_json():
    path = Path(__file__).parents[2] / "tools/generate_3dsr_qwen_scene_distractors.py"
    spec = importlib.util.spec_from_file_location("qwen_distractor_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.extract_objects('{"objects": ["tree", "lamp post", "partial') == [
        "tree", "lamp post"
    ]
