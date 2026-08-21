from datasets import Dataset

from lmms_eval.tasks.comfort_direction_object import utils


def _doc(qid, target="red Sphere", reference="blue Sphere", viewpoint="camera"):
    return {
        "qid": qid,
        "index": qid,
        "img_path": "/tmp/example.png",
        "image": "/tmp/example.png",
        "question": f"Where is the {target} relative to the {reference}?",
        "answer": "left",
        "target_relation": "left",
        "task_family": "comfort_viewpoint_spatial_reasoning",
        "scene": "comfort_car_ref_facing_right",
        "variation": "test",
        "viewpoint": viewpoint,
        "frame_deviation_degrees": 0.0,
        "variable_object": {"name": target, "position_world": [0.0, -1.0, 0.0]},
        "reference_object": {"name": reference, "position_world": [0.0, 0.0, 0.0]},
        "addressee_object": {"name": "Woman", "position_world": [0.0, 1.0, 0.0]},
        "camera_position_world": [10.0, 0.0, 1.0],
    }


def _records():
    source = Dataset.from_list(
        [
            _doc("one"),
            _doc("two", "Basketball", "Chair"),
            _doc("three", "yellow Sphere", "green Sphere"),
        ]
    )
    return list(utils.process_docs(source))


def test_process_docs_emits_deterministic_matched_pairs():
    first = _records()
    second = _records()
    assert first == second
    assert len(first) == 6
    for source_qid in {row["source_qid"] for row in first}:
        pair = [row for row in first if row["source_qid"] == source_qid]
        assert {row["diagnostic_answer_format"] for row in pair} == {"direction", "object"}
        assert all(len(row["options"]) == 2 for row in pair)
        assert all(len(set(row["options"])) == 2 for row in pair)
        assert len({row["img_path"] for row in pair}) == 1
        direction_row = next(row for row in pair if row["diagnostic_answer_format"] == "direction")
        assert direction_row["diagnostic_relation"] in direction_row["options"]
        assert len(set(direction_row["options"]) - {direction_row["diagnostic_relation"]}) == 1
        object_row = next(row for row in pair if row["diagnostic_answer_format"] == "object")
        assert set(object_row["options"]) == {object_row["diagnostic_target_object"], "Woman"}
        assert object_row["diagnostic_anchor"] not in object_row["options"]


def test_prompt_parser_submission_and_paired_metric():
    rows = _records()
    object_row = next(row for row in rows if row["source_qid"] == "one" and row["diagnostic_answer_format"] == "object")
    direction_row = next(row for row in rows if row["source_qid"] == "one" and row["diagnostic_answer_format"] == "direction")
    assert "which object is to the left of the blue Sphere?" in utils.doc_to_text(object_row)
    assert utils.extract_option_letter("Option B") == "B"
    assert utils.extract_option_letter("Option C") is None
    assert utils.extract_option_letter("left") is None

    object_result = utils.process_results(object_row, [object_row["gold_option_letter"]])
    wrong = next(letter for letter in utils.OPTION_LETTERS if letter != direction_row["gold_option_letter"])
    direction_result = utils.process_results(direction_row, [wrong])
    paired = [object_result["object_minus_direction"], direction_result["object_minus_direction"]]
    assert utils.aggregate_object_minus_direction(paired) == 1.0
    assert object_result["submission"]["question_prompt"] == utils.doc_to_text(object_row)
    assert object_result["submission"]["img_path"] == object_row["img_path"]
