import json

from datasets import Dataset
from PIL import Image

from lmms_eval.tasks.kubric_movi_a_motion import utils


def _fixture(tmp_path):
    sequence_dir = tmp_path / "fixture"
    frames_dir = sequence_dir / "frames"
    masks_dir = sequence_dir / "segmentations"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    paths = []
    objects = []
    for frame in range(3):
        path = frames_dir / f"frame_{frame:05d}.png"
        Image.new("RGB", (64, 64), (20 + frame * 20, 30, 40)).save(path)
        mask = Image.new("L", (64, 64), 0)
        for x in range(8 + frame, 20 + frame):
            for y in range(10, 22):
                mask.putpixel((x, y), 1)
        mask.save(masks_dir / f"segmentations_{frame:05d}.png")
        paths.append(str(path))
    for object_id in range(5):
        rows = []
        for frame in range(3):
            position = [float(frame + object_id), float(object_id), float(2 - frame)]
            rows.append(
                {
                    "frame_index": frame,
                    "timestamp_seconds": frame / 12,
                    "position_world": position,
                    "position_camera": [position[0], position[2], -position[1]],
                    "velocity_world": [1.0 + object_id, 0.0, -1.0],
                    "speed": ((1.0 + object_id) ** 2 + 1.0) ** 0.5,
                    "acceleration_world": [0.0, 0.0, -9.8],
                    "acceleration_magnitude": 9.8,
                    "distance_to_camera": 10.0 + frame,
                    "visibility_pixels": 144,
                    "image_position_normalized": [0.2 + frame * 0.02, 0.25],
                    "bbox_2d_normalized": [0.15, 0.12 + frame * 0.01, 0.35, 0.32 + frame * 0.01],
                }
            )
        objects.append({"object_id": object_id, "description": f"object {object_id}", "frame_quantities": rows})
    sequence_gt = {
        "objects": objects,
        "camera": {"field_of_view": 0.857556, "quaternions": [[1, 0, 0, 0]] * 3},
        "collisions": {"episodes": [{"instances": [0, 65535], "frame_start": 1, "frame_end": 1, "max_force": 2.0}]},
    }
    gt = {
        "metric": "displacement",
        "camera_coordinate_delta": [2.0, -2.0, 0.0],
        "distance_to_camera_delta": 2.0,
        "displacement": 2.828427,
    }
    doc = {
        "qid": "fixture_q",
        "sequence_id": "fixture",
        "subtask": "displacement",
        "question": "What is the displacement of object 0?",
        "frame_start": 0,
        "frame_end": 2,
        "fps": 12.0,
        "target_object_id": 0,
        "target_description": "object 0",
        "A": "1 scene units",
        "B": "2 scene units",
        "C": "2.828 scene units",
        "D": "4 scene units",
        "E": "5 scene units",
        "answer": "C",
        "answer_detail": "2.828427",
        "answer_value": 2.828427,
        "answer_unit": "scene units",
        "ground_truth_json": json.dumps(gt),
        "frames_json": json.dumps(paths),
        "_sequence_gt": sequence_gt,
    }
    return doc


def test_every_scaffold_renders_text_and_visuals(tmp_path):
    source = _fixture(tmp_path)
    for condition in utils.SCAFFOLDS:
        doc = {**source, "scaffold_condition": condition, "scaffold_suite": "test"}
        prompt = utils.doc_to_text(doc)
        visuals = utils.doc_to_visual(doc)
        assert "Question:" in prompt
        assert "Options:" in prompt
        if condition == "state_only_no_visual":
            assert visuals == []
        else:
            assert all(isinstance(image, Image.Image) for image in visuals)


def test_oracle_ladder_is_paired_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVI_MOTION_SAMPLES_PER_SUBTASK", "1")
    source = _fixture(tmp_path)
    processed = list(utils.process_oracle_ladder_docs(Dataset.from_list([source])))
    assert len(processed) == len(utils.SUITES["oracle_ladder"])
    assert {row["source_qid"] for row in processed} == {"fixture_q"}
    assert {row["scaffold_condition"] for row in processed} == set(utils.SUITES["oracle_ladder"])


def test_results_report_paired_gain_and_numeric_error(tmp_path):
    source = _fixture(tmp_path)
    baseline = {**source, "source_qid": "fixture_q", "qid": "fixture_q::baseline", "scaffold_condition": "baseline", "scaffold_suite": "test"}
    assisted = {**baseline, "qid": "fixture_q::derived", "scaffold_condition": "derived_quantity_oracle"}
    wrong = utils.process_results(baseline, ["A"])["accuracy"]
    correct = utils.process_results(assisted, ["C"])["accuracy"]
    assert wrong["score"] == 0
    assert correct["score"] == 1
    assert correct["numeric_option_error"] < 0.001
    assert utils.aggregate_paired_gain([wrong, correct]) == 1.0


def test_detailed_trace_saves_prompt_visuals_and_score(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("MOVI_MOTION_TRACE_DIR", str(trace_dir))
    source = _fixture(tmp_path)
    doc = {
        **source,
        "source_qid": source["qid"],
        "qid": f"{source['qid']}::bbox_track",
        "scaffold_condition": "bbox_track",
        "scaffold_suite": "perception",
    }

    prompt = utils.doc_to_text(doc)
    visuals = utils.doc_to_visual(doc)
    result = utils.process_results(doc, ["C"])["accuracy"]

    records = list((trace_dir / "records").glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text())
    assert payload["task_prompt"] == prompt
    assert payload["visual_input"]["count"] == len(visuals)
    assert payload["visual_input"]["contact_sheet"].endswith(".jpg")
    assert payload["scored_output"]["correct"] is True
    assert payload["source_document"]["question"] == source["question"]
    for visual in payload["visual_input"]["ordered_media"]:
        assert (trace_dir / visual["path"]).is_file()
    assert (trace_dir / payload["visual_input"]["contact_sheet"]).is_file()
    assert result["score"] == 1


def test_comparison_scaffolds_treat_all_candidates_uniformly(tmp_path):
    doc = _fixture(tmp_path)
    candidate_values = [
        {"object_id": index, "description": f"object {index}", "value": float(index + 1)}
        for index in range(5)
    ]
    doc.update(
        {
            "subtask": "fastest_object",
            "question": "Which object is fastest?",
            "ground_truth_json": json.dumps({"metric": "instantaneous_speed", "candidate_values": candidate_values}),
            "target_object_id": 4,
            "target_description": "object 4",
        }
    )
    raw = utils.aid_raw_state(doc)
    trace = utils.aid_2d_motion(doc)
    assert all(f"object {index}" in raw for index in range(5))
    assert all(f"object {index}" in trace for index in range(5))
    visuals = utils.visual_bbox_track(doc, [Image.open(path).convert("RGB") for path in json.loads(doc["frames_json"])])
    assert len(visuals) == 3


def test_derived_frame_library_rebuilds_target_and_preserves_original_indices(tmp_path):
    source = _fixture(tmp_path)
    derived = utils._derive_frame_limited_doc(source, 2)

    assert json.loads(derived["selected_frame_indices_json"]) == [0, 2]
    assert len(json.loads(derived["frames_json"])) == 2
    assert derived["derived_from_frame_count"] == 3
    assert derived["derived_frame_count"] == 2
    assert derived["answer"] in utils.OPTION_LETTERS
    assert abs(json.loads(derived["ground_truth_json"])["value"] - 2.8284271247461903) < 1e-9

    doc = {**derived, "scaffold_condition": "frame_labels", "scaffold_suite": "temporal"}
    prompt = utils.doc_to_text(doc)
    visuals = utils.doc_to_visual(doc)
    assert "Original frame IDs=[0, 2]" in prompt
    assert len(visuals) == 2


def test_derived_frame_library_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVI_MOTION_SAMPLES_PER_SUBTASK", "1")
    monkeypatch.setenv("MOVI_MOTION_DERIVED_NUM_FRAMES", "2")
    processed = list(utils.process_temporal_docs(Dataset.from_list([_fixture(tmp_path)])))
    assert len(processed) == len(utils.SUITES["temporal"])
    assert all(json.loads(row["selected_frame_indices_json"]) == [0, 2] for row in processed)


def test_numeric_scale_guide_is_shared_by_baseline_and_all_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVI_MOTION_NUMERIC_SCALE_GUIDE", "1")
    source = _fixture(tmp_path)
    for condition in utils.SCAFFOLDS:
        doc = {**source, "scaffold_condition": condition, "scaffold_suite": "test"}
        prompt = utils.doc_to_text(doc)
        visuals = utils.doc_to_visual(doc)
        assert "Metric scale guide:" in prompt
        assert visuals
        if condition == "state_only_no_visual":
            assert visuals[0].getpixel((8, 8)) != (20, 30, 40)


def test_scale_guide_does_not_change_categorical_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVI_MOTION_NUMERIC_SCALE_GUIDE", "1")
    source = _fixture(tmp_path)
    source.update({"subtask": "dominant_motion_direction", "answer_value": None})
    doc = {**source, "scaffold_condition": "baseline", "scaffold_suite": "test"}
    prompt = utils.doc_to_text(doc)
    visuals = utils.doc_to_visual(doc)
    assert "Metric scale guide:" not in prompt
    assert visuals[0].getpixel((5, 58)) == (20, 30, 40)
