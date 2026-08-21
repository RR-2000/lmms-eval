from tools.build_gqa_direction_object import audit_records, build_records

from lmms_eval.tasks.gqa_direction_object import utils


def _scene():
    return {
        "objects": {
            "1": {
                "name": "ball",
                "relations": [{"name": "to the left of", "object": "2"}],
            },
            "2": {"name": "chair", "relations": []},
            "3": {"name": "table", "relations": []},
            "4": {"name": "lamp", "relations": []},
            "5": {"name": "window", "relations": []},
        }
    }


def test_builder_emits_deterministic_matched_pair():
    first, report = build_records({"image-1": _scene()}, seed="test")
    second, _ = build_records({"image-1": _scene()}, seed="test")

    assert first == second
    assert report["valid"]
    assert report["num_matched_pairs"] == 1
    assert {row["diagnostic_answer_format"] for row in first} == {
        "direction",
        "object",
    }
    assert all(len(row["options"]) == 4 for row in first)
    assert first[0]["source_relation_id"] == first[1]["source_relation_id"]
    assert first[0]["image"] == first[0]["image_path"]
    assert first[0]["image_path"].endswith("image-1.jpg")
    assert len(set(first[1]["options"])) == 4


def test_audit_rejects_pair_metadata_mismatch():
    records, _ = build_records({"image-1": _scene()}, seed="test")
    records[1]["diagnostic_anchor"] = "different"
    report = audit_records(records)
    assert not report["valid"]
    assert any("pair metadata differs" in error for error in report["errors"])


def test_builder_drops_multiple_correct_object_options():
    scene = _scene()
    scene["objects"]["3"]["relations"] = [
        {"name": "left of", "object": "2"}
    ]
    records, report = build_records({"image-1": scene}, seed="test")
    assert records == []
    assert report["skipped_counts"]["ambiguous_relation"] == 2


def test_prompt_and_strict_letter_parser():
    records, _ = build_records({"image-1": _scene()}, seed="test")
    direction = next(
        row for row in records if row["diagnostic_answer_format"] == "direction"
    )
    prompt = utils.doc_to_text(direction)
    assert "Where is the ball relative to the chair?" in prompt
    assert "A. left" in prompt
    assert utils.extract_option_letter("Option C") == "C"
    assert utils.extract_option_letter("A table") == "A"
    assert utils.extract_option_letter("chair") is None


def test_metrics_use_only_complete_pairs():
    records, _ = build_records({"image-1": _scene()}, seed="test")
    object_row = next(
        row for row in records if row["diagnostic_answer_format"] == "object"
    )
    direction_row = next(
        row for row in records if row["diagnostic_answer_format"] == "direction"
    )
    object_result = utils.process_results(
        object_row, [object_row["gold_option_letter"]]
    )["object_minus_direction"]
    wrong_letter = next(
        letter
        for letter in utils.OPTION_LETTERS
        if letter != direction_row["gold_option_letter"]
    )
    direction_result = utils.process_results(direction_row, [wrong_letter])[
        "object_minus_direction"
    ]
    unmatched = dict(object_result, source_relation_id="unmatched", score=0.0)

    assert utils.aggregate_object_minus_direction(
        [object_result, direction_result, unmatched]
    ) == 1.0
    assert utils.aggregate_object_correct_direction_wrong(
        [object_result, direction_result, unmatched]
    ) == 1.0


def test_parse_failure_scores_zero_and_is_reported():
    records, _ = build_records({"image-1": _scene()}, seed="test")
    processed = utils.process_results(records[0], ["I cannot tell"])
    result = processed["accuracy"]
    assert result["score"] == 0.0
    assert result["parse_success"] is False
    assert result["predicted_option_letter"] is None
    submission = processed["submission"]
    assert submission["question_prompt"] == utils.doc_to_text(records[0])
    assert submission["img_path"] == records[0]["image_path"]
    assert submission["image_path"] == records[0]["image_path"]
    assert submission["options"] == records[0]["options"]
