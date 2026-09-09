"""Controlled experiments for COMFORT inverse spatial queries.

The family contains full-map inversion, an arrow-length sweep, canonical-map
ablations, and option-permutation consistency.  It deliberately reuses the
same geometry and drawing primitives as the numbered GT_HELP task.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.tasks.comfort_direction_object_gt_help import utils as aids
from lmms_eval.utils import sanitize_model_name


DIRECTIONS = base.DIRECTIONS
ANSWER_FORMATS = base.ANSWER_FORMATS
OPTION_LETTERS = base.OPTION_LETTERS
ARROW_LENGTHS = (
    ("0.25", 0.25, 10.0),
    ("0.45", 0.45, 18.0),
    ("0.70", 0.70, 24.0),
    ("0.90", 0.90, 30.0),
    ("1.15", 1.15, 38.0),
    ("1.50", 1.50, 48.0),
)
FULL_MAP_CUES = ("none", "long_arrows", "short_arrows", "named_map", "color_map")
MAP_ABLATIONS = (
    "object_names_only",
    "direction_labels_only",
    "object_and_direction_labels",
    "colors_only",
    "heading_only",
    "object_names_no_heading",
    "rotated_labeled",
    "rotated_unlabeled",
)
BINARY_AXES = {
    "left_right": ("left", "right"),
    "front_behind": ("front", "behind"),
}
ORACLE_LADDER = (
    "baseline",
    "reference_localized",
    "heading_given",
    "axes_given",
    "intermediate_oracle",
    "spatial_map_oracle",
    "answer_text_oracle",
    "answer_letter_oracle",
)
DEBUG_DEFAULT_DIR = Path("outputs/comfort_inverse_diagnostics_debug")


def _balanced_permutation_index(doc: dict) -> int:
    scene_text = str(doc.get("scene_id", "0"))
    digits = "".join(character for character in scene_text if character.isdigit())
    scene_index = int(digits or 0)
    return (scene_index + DIRECTIONS.index(str(doc["diagnostic_relation"]))) % 4


def _normalized_docs(dataset: Dataset, *, all_permutations: bool) -> list[dict]:
    rows = []
    for source in base.process_docs(dataset):
        doc = dict(source)
        if all_permutations or int(doc["answer_idx"]) == _balanced_permutation_index(doc):
            rows.append(doc)
    return rows


def _conditioned_doc(doc: dict, experiment: str, condition: str) -> dict:
    output = dict(doc)
    original_source_id = str(doc["source_relation_id"])
    pair_stem = "::".join(original_source_id.split("::")[:2])
    permutation_index = int(doc["answer_idx"])
    output.update(
        {
            "diagnostic_experiment": experiment,
            "experiment_condition": condition,
            "original_source_relation_id": original_source_id,
            "source_relation_id": f"{pair_stem}::{experiment}::{condition}::{permutation_index}",
            "source_qid": f"{pair_stem}::{experiment}::{condition}::{permutation_index}",
            "qid": f"{doc['qid']}::{experiment}::{condition}",
            "index": f"{doc['qid']}::{experiment}::{condition}",
            "permutation_index": permutation_index,
        }
    )
    return output


def process_arrow_length_sweep_docs(dataset: Dataset) -> Dataset:
    records = []
    for doc in _normalized_docs(dataset, all_permutations=False):
        for name, scale, minimum in ARROW_LENGTHS:
            record = _conditioned_doc(doc, "arrow_length_sweep", name)
            record.update({"arrow_length_scale": scale, "arrow_minimum_pixels": minimum})
            records.append(record)
    return _finish(records, "arrow_length_sweep")


def process_map_ablation_docs(dataset: Dataset) -> Dataset:
    records = [
        _conditioned_doc(doc, "map_ablation", condition)
        for doc in _normalized_docs(dataset, all_permutations=False)
        for condition in MAP_ABLATIONS
    ]
    return _finish(records, "map_ablation")


def process_option_permutation_docs(dataset: Dataset) -> Dataset:
    records = []
    for doc in _normalized_docs(dataset, all_permutations=True):
        record = _conditioned_doc(doc, "option_permutation", "plain_image")
        record["permutation_group_id"] = (
            f"{doc['scene_id']}::{doc['diagnostic_relation']}::"
            f"{doc['diagnostic_answer_format']}"
        )
        records.append(record)
    return _finish(records, "option_permutation")


def _binary_option_order(doc: dict, choices: tuple[str, str]) -> list[str]:
    scene_text = str(doc.get("scene_id", "0"))
    digits = "".join(character for character in scene_text if character.isdigit())
    scene_index = int(digits or 0)
    relation_index = DIRECTIONS.index(str(doc["diagnostic_relation"]))
    return list(choices if (scene_index + relation_index) % 2 == 0 else choices[::-1])


def process_binary_axis_docs(dataset: Dataset) -> Dataset:
    records = []
    for doc in _normalized_docs(dataset, all_permutations=False):
        relation = str(doc["diagnostic_relation"])
        condition = next(name for name, directions in BINARY_AXES.items() if relation in directions)
        directions = BINARY_AXES[condition]
        if doc["diagnostic_answer_format"] == "direction":
            semantic_choices = directions
            gold_answer = relation
        else:
            semantic_choices = tuple(
                str(aids.get_object_at_direction(doc, direction).get("label", "")).strip()
                for direction in directions
            )
            gold_answer = str(aids.get_object_at_direction(doc, relation).get("label", "")).strip()
        options = _binary_option_order(doc, semantic_choices)
        gold_index = options.index(gold_answer)
        record = _conditioned_doc(doc, "binary_axis", condition)
        record.update(
            {
                "options": options,
                "answer_idx": gold_index,
                "gold_option_letter": OPTION_LETTERS[gold_index],
                "binary_axis": condition,
            }
        )
        records.append(record)
    return _finish(records, "binary_axis")


def process_oracle_ladder_docs(dataset: Dataset) -> Dataset:
    records = [
        _conditioned_doc(doc, "oracle_ladder", condition)
        for doc in _normalized_docs(dataset, all_permutations=False)
        for condition in ORACLE_LADDER
    ]
    return _finish(records, "oracle_ladder")


def process_full_map_inversion_docs(dataset: Dataset) -> Dataset:
    # One normalized annotation row supplies the scene lookup and image path;
    # each full-map question covers all four relations at once.
    by_scene = {}
    for doc in _normalized_docs(dataset, all_permutations=False):
        by_scene.setdefault(str(doc["scene_id"]), doc)
    records = []
    for scene_id, source in sorted(by_scene.items()):
        relation_map = {
            direction: str(aids.get_object_at_direction(source, direction).get("label", "")).strip()
            for direction in DIRECTIONS
        }
        object_map = {object_name: direction for direction, object_name in relation_map.items()}
        if len(object_map) != 4:
            eval_logger.warning("Skipping full-map scene {} with duplicate object labels", scene_id)
            continue
        for cue in FULL_MAP_CUES:
            for mapping_format, gold_map in (
                ("relation_to_object", relation_map),
                ("object_to_relation", object_map),
            ):
                qid = f"{scene_id}::full_map_inversion::{cue}::{mapping_format}"
                records.append(
                    {
                        **dict(source),
                        "qid": qid,
                        "index": qid,
                        "source_qid": qid,
                        "source_relation_id": qid,
                        "diagnostic_experiment": "full_map_inversion",
                        "experiment_condition": cue,
                        "mapping_format": mapping_format,
                        "gt_mapping_json": json.dumps(gold_map, sort_keys=True),
                        "relation_to_object": relation_map,
                    }
                )
    return _finish(records, "full_map_inversion")


def _finish(records: list[dict], experiment: str) -> Dataset:
    eval_logger.info(
        "COMFORT inverse diagnostic {} loaded {} records from {} scenes; conditions={}",
        experiment,
        len(records),
        len({row["scene_id"] for row in records}),
        dict(Counter(row["experiment_condition"] for row in records)),
    )
    return Dataset.from_list(records)


def _map_positions(size: int, rotation_quarters: int) -> tuple[tuple[float, float], dict[str, tuple[float, float]], float]:
    center = (size / 2, size / 2 + size * 0.025)
    distance = size * 0.29
    base_offsets = {
        "left": (-distance, 0.0),
        "right": (distance, 0.0),
        "front": (0.0, -distance),
        "behind": (0.0, distance),
    }
    positions = {}
    angle = rotation_quarters * math.pi / 2.0
    for direction, (x, y) in base_offsets.items():
        rotated = (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        positions[direction] = (center[0] + rotated[0], center[1] + rotated[1])
    return center, positions, distance


def _custom_map_panel(
    doc: dict,
    size: int,
    *,
    object_labels: bool,
    direction_labels: bool,
    heading: bool,
    target_circles: bool = True,
    rotation_quarters: int = 0,
) -> Image.Image:
    panel = Image.new("RGB", (size, size), aids.TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font = aids._map_font(max(15, round(size / 30)))
    label_font = aids._map_font(max(12, round(size / 42)))
    axis_font = aids._map_font(max(13, round(size / 36)))
    aids._centered_map_text(draw, (size / 2, 12), "Reference-centered diagnostic map", title_font, (20, 20, 20), size)
    center, positions, distance = _map_positions(size, rotation_quarters)
    radius = max(16.0, size * 0.045)
    aids._draw_map_axes(draw, center, distance)
    reference = aids.get_reference_object(doc)
    aids._draw_map_circle(
        draw,
        center,
        radius,
        aids.TOP_DOWN_COLORS["reference"][0],
        str(reference.get("label", "")) if object_labels else None,
        label_font,
        size,
    )
    if target_circles:
        for direction in DIRECTIONS:
            obj = aids.get_object_at_direction(doc, direction)
            aids._draw_map_circle(
                draw,
                positions[direction],
                radius,
                aids.TOP_DOWN_COLORS[direction][0],
                str(obj.get("label", "")) if object_labels else None,
                label_font,
                size,
            )
    if heading:
        heading_angle = -math.pi / 2.0 + rotation_quarters * math.pi / 2.0
        tip = (center[0] + radius * 0.9 * math.cos(heading_angle), center[1] + radius * 0.9 * math.sin(heading_angle))
        perpendicular = (-math.sin(heading_angle), math.cos(heading_angle))
        base_center = (center[0] + radius * 0.25 * math.cos(heading_angle), center[1] + radius * 0.25 * math.sin(heading_angle))
        draw.polygon(
            [
                tip,
                (base_center[0] + radius * 0.28 * perpendicular[0], base_center[1] + radius * 0.28 * perpendicular[1]),
                (base_center[0] - radius * 0.28 * perpendicular[0], base_center[1] - radius * 0.28 * perpendicular[1]),
            ],
            fill=(20, 20, 20),
        )
    if direction_labels:
        for direction, position in positions.items():
            dx, dy = position[0] - center[0], position[1] - center[1]
            norm = math.hypot(dx, dy)
            text_position = (center[0] + dx / norm * size * 0.43, center[1] + dy / norm * size * 0.43)
            aids._centered_map_text(draw, text_position, direction, axis_font, aids.TOP_DOWN_COLORS[direction][0], size)
    return panel


def _map_ablation_image(doc: dict, image: Image.Image) -> Image.Image:
    condition = str(doc["experiment_condition"])
    settings = {
        "object_names_only": dict(object_labels=True, direction_labels=False, heading=True),
        "direction_labels_only": dict(object_labels=False, direction_labels=True, heading=True),
        "object_and_direction_labels": dict(object_labels=True, direction_labels=True, heading=True),
        "colors_only": dict(object_labels=False, direction_labels=False, heading=True),
        "heading_only": dict(object_labels=False, direction_labels=False, heading=True, target_circles=False),
        "object_names_no_heading": dict(object_labels=True, direction_labels=False, heading=False),
        "rotated_labeled": dict(object_labels=True, direction_labels=True, heading=True, rotation_quarters=1),
        "rotated_unlabeled": dict(object_labels=True, direction_labels=False, heading=True, rotation_quarters=1),
    }[condition]
    panel = _custom_map_panel(doc, image.height, **settings)
    return aids._append_map_panels(image, [panel])


def _cue_image(doc: dict, image: Image.Image, cue: str) -> Image.Image:
    if cue == "none":
        return image
    if cue == "long_arrows":
        return aids.draw_reference_direction_arrows(doc, image)
    if cue == "short_arrows":
        return aids.draw_short_reference_direction_arrows(doc, image)
    if cue == "named_map":
        return aids.draw_reference_top_down_map(doc, image)
    if cue == "color_map":
        return aids.draw_unlabeled_reference_top_down_map(doc, image)
    raise ValueError(f"Unknown cue {cue!r}")


ORACLE_VISUAL_AIDS = {
    "baseline": (),
    "reference_localized": (aids.draw_reference_bbox,),
    "heading_given": (aids.draw_reference_bbox_and_labeled_front_arrow,),
    "axes_given": (aids.draw_short_reference_direction_arrows,),
    "intermediate_oracle": (aids.draw_target_bbox_for_direction_questions,),
    "spatial_map_oracle": (aids.draw_labeled_reference_top_down_map,),
    "answer_text_oracle": (),
    "answer_letter_oracle": (),
}


ORACLE_TEXT_AIDS = {
    "baseline": (),
    "reference_localized": (aids.describe_reference_bbox,),
    "heading_given": (aids.describe_labeled_reference_front_arrow,),
    "axes_given": (aids.describe_reference_direction_arrows,),
    "intermediate_oracle": (
        aids.reveal_relation_for_object_questions,
        aids.reveal_target_for_direction_questions,
    ),
    "spatial_map_oracle": (aids.labeled_top_down_mapping,),
    "answer_text_oracle": (aids.ground_truth_free_text,),
    "answer_letter_oracle": (aids.ground_truth_letter_only,),
}


def _oracle_image(doc: dict, image: Image.Image) -> Image.Image:
    for visual_aid in ORACLE_VISUAL_AIDS[str(doc["experiment_condition"])]:
        image = visual_aid(doc, image)
    return image


def _oracle_text(doc: dict) -> str:
    lines = [
        text
        for text_aid in ORACLE_TEXT_AIDS[str(doc["experiment_condition"])]
        if (text := text_aid(doc))
    ]
    return " ".join(lines)


def _debug_enabled() -> bool:
    return str(os.getenv("COMFORT_INVERSE_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _save_debug(doc: dict, image: Image.Image) -> None:
    if not _debug_enabled():
        return
    root = Path(os.getenv("COMFORT_INVERSE_DEBUG_DIR", str(DEBUG_DEFAULT_DIR))).expanduser()
    output_dir = root / str(doc["diagnostic_experiment"]) / str(doc["experiment_condition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(doc["qid"]))
    image.save(output_dir / f"{safe_id}.png", format="PNG")


def doc_to_visual(doc):
    image = base.doc_to_visual(doc)[0]
    experiment = str(doc["diagnostic_experiment"])
    if experiment == "arrow_length_sweep":
        image = aids._draw_reference_direction_arrows_with_scale(
            doc,
            image,
            length_scale=float(doc["arrow_length_scale"]),
            minimum_length=float(doc["arrow_minimum_pixels"]),
        )
    elif experiment == "map_ablation":
        image = _map_ablation_image(doc, image)
    elif experiment == "full_map_inversion":
        image = _cue_image(doc, image, str(doc["experiment_condition"]))
    elif experiment == "oracle_ladder":
        image = _oracle_image(doc, image)
    elif experiment not in {"option_permutation", "binary_axis"}:
        raise ValueError(f"Unknown experiment {experiment!r}")
    _save_debug(doc, image)
    return [image]


CUE_TEXT = {
    "none": "No additional spatial aid is provided.",
    "long_arrows": "Four long labeled arrows on the reference show its left, right, front, and behind.",
    "short_arrows": "Four short labeled arrows on the reference show its left, right, front, and behind.",
}


MAP_TEXT = {
    "object_names_only": "The canonical map labels objects but not direction axes.",
    "direction_labels_only": "The canonical map labels direction axes but not objects.",
    "object_and_direction_labels": "The canonical map labels both objects and direction axes.",
    "colors_only": "The canonical map has colored object circles and a heading marker but no text labels.",
    "heading_only": "The canonical map shows only the reference and its heading marker.",
    "object_names_no_heading": "The canonical map labels objects but removes the reference heading marker.",
    "rotated_labeled": "The canonical reference map is rotated 90 degrees and retains object and semantic direction labels.",
    "rotated_unlabeled": "The canonical reference map is rotated 90 degrees, retains object names and heading, but omits direction labels.",
}


def _color_object_mapping(doc: dict) -> str:
    records = []
    for direction in DIRECTIONS:
        color = aids.TOP_DOWN_COLORS[direction][1]
        label = aids.get_object_at_direction(doc, direction).get("label", "")
        records.append(f"{color}={label}")
    return "Color-to-object legend: " + "; ".join(records) + "."


def _cue_text(doc: dict) -> str:
    cue = str(doc["experiment_condition"])
    if cue == "named_map":
        return "The added top-down panel names the objects in a reference-centered canonical map."
    if cue == "color_map":
        return (
            "The added canonical map uses colored circles and a heading marker without direction labels. "
            + _color_object_mapping(doc)
        )
    return CUE_TEXT[cue]


def _map_text(doc: dict) -> str:
    condition = str(doc["experiment_condition"])
    text = MAP_TEXT[condition]
    if condition in {"direction_labels_only", "colors_only"}:
        text += " " + _color_object_mapping(doc)
    return text


def _mc_prompt(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    options = "\n".join(f"{letter}. {answer}" for letter, answer in zip(OPTION_LETTERS, doc["options"]))
    experiment = str(doc["diagnostic_experiment"])
    if experiment == "arrow_length_sweep":
        aid_text = (
            "Four labeled reference-relative arrows are overlaid on the reference object. "
            f"Their length is {doc['arrow_length_scale']} times the reference bbox diagonal."
        )
    elif experiment == "map_ablation":
        aid_text = _map_text(doc)
    elif experiment == "binary_axis":
        axis = str(doc["experiment_condition"]).replace("_", "/")
        aid_text = (
            f"Binary-axis diagnostic: this question is restricted to the reference object's {axis} axis. "
            "Choose between the two supplied alternatives only."
        )
    elif experiment == "oracle_ladder":
        aid_text = _oracle_text(doc)
        if not aid_text:
            aid_text = "No oracle aid is supplied in this baseline condition."
    else:
        aid_text = "No spatial aid is added; this condition tests sensitivity to answer-option ordering."
    return (
        f"{kwargs.get('pre_prompt', '')}{aid_text}\n"
        "Answer this spatial-reasoning question using the image. Select one answer option "
        "and respond with its letter.\n"
        f"Question: {doc['diagnostic_question']}\nOptions:\n{options}"
        f"{kwargs.get('post_prompt', '')}"
    )


def _full_map_prompt(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    reference = str(doc["diagnostic_anchor"])
    cue_text = _cue_text(doc)
    if doc["mapping_format"] == "relation_to_object":
        instruction = (
            "Return only one JSON object with exactly these keys: left, right, front, behind. "
            "Each value must be the name of the object at that position."
        )
        schema = '{"left":"<object>","right":"<object>","front":"<object>","behind":"<object>"}'
    else:
        objects = list(doc["relation_to_object"].values())
        instruction = (
            "Return only one JSON object whose four keys are the surrounding object names and whose "
            "values are exactly one of left, right, front, or behind."
        )
        schema = "{" + ",".join(f'"{name}":"<direction>"' for name in objects) + "}"
    return (
        f"{kwargs.get('pre_prompt', '')}{cue_text}\n"
        f"Use the {reference}'s own viewpoint and recover the complete mapping of all four surrounding objects. "
        f"{instruction}\nRequired schema: {schema}{kwargs.get('post_prompt', '')}"
    )


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if doc["diagnostic_experiment"] == "full_map_inversion":
        return _full_map_prompt(doc, lmms_eval_specific_kwargs)
    return _mc_prompt(doc, lmms_eval_specific_kwargs)


def doc_to_target(doc):
    if doc["diagnostic_experiment"] == "full_map_inversion":
        return str(doc["gt_mapping_json"])
    return str(doc["gold_option_letter"])


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates = [str(text).strip()]
    start, end = str(text).find("{"), str(text).rfind("}")
    if 0 <= start < end:
        candidates.insert(0, str(text)[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def process_full_map_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = _extract_json(prediction)
    raw_gold = json.loads(doc["gt_mapping_json"])
    gold = {_normalize(key): _normalize(value) for key, value in raw_gold.items()}
    normalized = {_normalize(key): _normalize(value) for key, value in parsed.items()} if parsed is not None else {}
    key_accuracy = sum(float(normalized.get(key) == value) for key, value in gold.items()) / 4.0
    exact = float(parsed is not None and len(normalized) == 4 and normalized == gold)
    entry = {
        "qid": doc["qid"],
        "scene_id": doc["scene_id"],
        "diagnostic_experiment": doc["diagnostic_experiment"],
        "experiment_condition": doc["experiment_condition"],
        "mapping_format": doc["mapping_format"],
        "gt_mapping": raw_gold,
        "parsed_mapping": parsed,
        "prediction": prediction,
        "parse_success": float(parsed is not None),
        "mapping_edge_accuracy": key_accuracy,
        "mapping_exact_accuracy": exact,
    }
    output = {name: dict(entry) for name in ("mapping_edge_accuracy", "mapping_exact_accuracy", "parse_success_rate")}
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "img_path": doc["img_path"]}
    return output


def process_mc_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = base.extract_option_letter(prediction)
    gold = str(doc["gold_option_letter"])
    valid_letters = OPTION_LETTERS[: len(doc["options"])]
    valid_selection = parsed in valid_letters
    selected_answer = doc["options"][valid_letters.index(parsed)] if valid_selection else None
    entry = {
        "qid": doc["qid"],
        "source_qid": doc["source_qid"],
        "source_relation_id": doc["source_relation_id"],
        "original_source_relation_id": doc.get("original_source_relation_id"),
        "permutation_group_id": doc.get("permutation_group_id"),
        "scene_id": doc["scene_id"],
        "diagnostic_experiment": doc["diagnostic_experiment"],
        "experiment_condition": doc["experiment_condition"],
        "answer_format": doc["diagnostic_answer_format"],
        "relation": doc["diagnostic_relation"],
        "anchor": doc["diagnostic_anchor"],
        "target": doc["diagnostic_target_object"],
        "permutation_index": int(doc["permutation_index"]),
        "options": list(doc["options"]),
        "gold_option_letter": gold,
        "predicted_option_letter": parsed,
        "selected_answer": selected_answer,
        "parse_success": float(valid_selection),
        "score": float(parsed == gold),
        "prediction": prediction,
    }
    if "arrow_length_scale" in doc:
        entry["arrow_length_scale"] = float(doc["arrow_length_scale"])
    metrics = (
        "accuracy",
        "object_answer_accuracy",
        "direction_answer_accuracy",
        "object_minus_direction",
        "object_correct_direction_wrong",
        "direction_correct_object_wrong",
        "both_correct",
        "both_wrong",
        "parse_success_rate",
    )
    output = {name: dict(entry) for name in metrics}
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "img_path": doc["img_path"]}
    return output


def process_permutation_results(doc, results):
    output = process_mc_results(doc, results)
    entry = dict(output["submission"])
    output["permutation_semantic_consistency"] = entry
    output["permutation_all_correct_rate"] = entry
    return output


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    _log_conditions(results)
    return _mean(float(row["score"]) for row in results)


def aggregate_object_answer_accuracy(results):
    return _mean(float(row["score"]) for row in results if row["answer_format"] == "object")


def aggregate_direction_answer_accuracy(results):
    return _mean(float(row["score"]) for row in results if row["answer_format"] == "direction")


def _matched_pairs(results):
    grouped = defaultdict(dict)
    for row in results:
        grouped[str(row["source_relation_id"])][str(row["answer_format"])] = row
    return [pair for pair in grouped.values() if set(ANSWER_FORMATS) <= set(pair)]


def aggregate_object_minus_direction(results):
    return _mean(pair["object"]["score"] - pair["direction"]["score"] for pair in _matched_pairs(results))


def _paired_rate(results, object_score: float, direction_score: float) -> float:
    return _mean(float(pair["object"]["score"] == object_score and pair["direction"]["score"] == direction_score) for pair in _matched_pairs(results))


def aggregate_object_correct_direction_wrong(results):
    return _paired_rate(results, 1.0, 0.0)


def aggregate_direction_correct_object_wrong(results):
    return _paired_rate(results, 0.0, 1.0)


def aggregate_both_correct(results):
    return _paired_rate(results, 1.0, 1.0)


def aggregate_both_wrong(results):
    return _paired_rate(results, 0.0, 0.0)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_mapping_edge_accuracy(results):
    _log_conditions(results)
    return _mean(float(row["mapping_edge_accuracy"]) for row in results)


def aggregate_mapping_exact_accuracy(results):
    return _mean(float(row["mapping_exact_accuracy"]) for row in results)


def _permutation_groups(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row["permutation_group_id"])].append(row)
    return [rows for rows in grouped.values() if len(rows) == 4]


def aggregate_permutation_semantic_consistency(results):
    return _mean(
        float(
            all(float(row.get("parse_success", 0.0)) == 1.0 for row in rows)
            and len({_normalize(row.get("selected_answer")) for row in rows}) == 1
        )
        for rows in _permutation_groups(results)
    )


def aggregate_permutation_all_correct_rate(results):
    return _mean(float(all(float(row["score"]) == 1.0 for row in rows)) for rows in _permutation_groups(results))


def _log_conditions(results) -> None:
    grouped = defaultdict(list)
    for row in results:
        grouped[(str(row.get("experiment_condition")), str(row.get("answer_format", row.get("mapping_format"))))].append(row)
    report = {}
    for key, rows in sorted(grouped.items()):
        score_field = "score" if "score" in rows[0] else "mapping_edge_accuracy"
        report["/".join(key)] = {"count": len(rows), "score": _mean(float(row[score_field]) for row in rows)}
    eval_logger.info("COMFORT inverse diagnostic by condition/format: {}", report)


def aggregate_results_for_submission(results, args):
    experiment = str(results[0].get("diagnostic_experiment", "inverse_diagnostic")) if results else "inverse_diagnostic"
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"comfort_{experiment}_{model}.json", args)
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": f"comfort_{experiment}",
        "experiment": experiment,
        "num_records": len(results),
        "conditions": sorted({str(row.get("experiment_condition")) for row in results}),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT inverse diagnostic records saved to {}", path)
