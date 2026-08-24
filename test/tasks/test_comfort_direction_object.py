from datasets import Dataset

from lmms_eval.tasks.comfort_direction_object import utils


def _source_rows():
    common = {
        "scene_id": "scene_000000",
        "image": "scenes/scene_000000.png",
        "reference_object": "bench",
        "answer_index": 1,
    }
    return [
        {
            **common,
            "id": "scene_000000_direction_left_1",
            "task": "direction",
            "question": "From the bench's current viewpoint, where is the laptop?",
            "options": ["right", "left", "front", "behind"],
            "answer": "left",
            "answer_3dsrbench": "left",
            "target_object": "laptop",
            "direction": None,
            "direction_3dsrbench": None,
        },
        {
            **common,
            "id": "scene_000000_object_left_1",
            "task": "object",
            "question": "Which object is in the direction that the left side of the bench points toward?",
            "options": ["duck", "laptop", "bicycle mountain", "sophia"],
            "answer": "laptop",
            "answer_3dsrbench": None,
            "direction": "left",
            "direction_3dsrbench": "left",
            "target_object": None,
        },
    ]


def _records():
    return list(utils.process_docs(Dataset.from_list(_source_rows())))


def test_process_docs_uses_native_matched_pair():
    first = _records()
    second = _records()
    assert first == second
    assert len(first) == 2
    assert {row["diagnostic_answer_format"] for row in first} == {"direction", "object"}
    assert len({row["source_relation_id"] for row in first}) == 1
    assert all(row["gold_option_letter"] == "B" for row in first)
    assert all(len(row["options"]) == 4 for row in first)
    assert all(row["img_path"].endswith("scenes/scene_000000.png") for row in first)
    direction = next(row for row in first if row["diagnostic_answer_format"] == "direction")
    object_row = next(row for row in first if row["diagnostic_answer_format"] == "object")
    assert direction["diagnostic_target_object"] == object_row["diagnostic_target_object"] == "laptop"
    assert direction["diagnostic_relation"] == object_row["diagnostic_relation"] == "left"


def test_prompt_parser_submission_and_paired_metric():
    rows = _records()
    object_row = next(row for row in rows if row["diagnostic_answer_format"] == "object")
    direction_row = next(row for row in rows if row["diagnostic_answer_format"] == "direction")
    prompt = utils.doc_to_text(object_row)
    assert object_row["question"] in prompt
    assert "A. duck" in prompt and "D. sophia" in prompt
    assert utils.extract_option_letter("Option C") == "C"
    assert utils.extract_option_letter("laptop") is None

    object_result = utils.process_results(object_row, [object_row["gold_option_letter"]])
    direction_result = utils.process_results(direction_row, ["A"])
    paired = [
        object_result["object_minus_direction"],
        direction_result["object_minus_direction"],
    ]
    assert utils.aggregate_object_minus_direction(paired) == 1.0
    assert object_result["submission"]["question_prompt"] == prompt
    assert object_result["submission"]["img_path"] == object_row["img_path"]
