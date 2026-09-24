"""Matched GT RGB-overlay and symbolic-map representation diagnostics.

The task deliberately holds boxes, reference orientation, candidate IDs,
questions, and answer choices fixed.  Conditions differ only in whether Qwen
receives original RGB, an RGB coordinate overlay, a symbolic image-plane map,
or both.  Jitter corrupts map candidate locations only; orientation corruption
rotates the supplied axes only.  Thus neither condition exposes the gold
direction-to-object binding as text.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.tasks.comfort_direction_object_gt_help import utils as aids
from lmms_eval.utils import sanitize_model_name


CORE = ("rgb_ids", "rgb_axes_gt", "symbolic_gt", "hybrid_gt")
JITTER_LEVELS = (0.05, 0.10, 0.15, 0.20)
ORIENTATION_DEGREES = (15, 30, 45, 90, 180)
JITTER = tuple(
    f"{representation}_jitter_{level:0.2f}".replace(".", "p")
    for representation in ("symbolic", "hybrid")
    for level in JITTER_LEVELS
)
ORIENTATION = tuple(
    f"{representation}_orientation_{degrees}"
    for representation in ("rgb_axes", "symbolic", "hybrid")
    for degrees in ORIENTATION_DEGREES
)
ALL_CONDITIONS = CORE + JITTER + ORIENTATION
DEBUG_DEFAULT_DIR = Path("outputs/comfort_coordinate_representation_debug")


def _balanced_option_index(doc: dict) -> int:
    scene_digits = "".join(character for character in str(doc["scene_id"]) if character.isdigit())
    return (int(scene_digits or 0) + base.DIRECTIONS.index(str(doc["diagnostic_relation"]))) % 4


def _selected_conditions() -> tuple[str, ...]:
    explicit = str(os.getenv("COMFORT_REPRESENTATION_CONDITIONS", "")).strip()
    if explicit:
        selected = tuple(item.strip() for item in explicit.split(",") if item.strip())
    else:
        suite = str(os.getenv("COMFORT_REPRESENTATION_SUITE", "core")).strip().lower()
        suites = {"core": CORE, "jitter": JITTER, "orientation": ORIENTATION, "all": ALL_CONDITIONS}
        if suite not in suites:
            raise ValueError(f"Unknown COMFORT_REPRESENTATION_SUITE={suite!r}; choose {sorted(suites)}")
        selected = suites[suite]
    unknown = sorted(set(selected) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown representation conditions: {unknown}")
    return selected


def _condition_doc(doc: dict, condition: str) -> dict:
    item = dict(doc)
    qid = f"{doc['qid']}::coordinate_representation::{condition}"
    item.update(
        {
            "qid": qid,
            "index": qid,
            "source_qid": str(doc["qid"]),
            "source_relation_id": str(doc["source_relation_id"]),
            "diagnostic_experiment": "coordinate_representation",
            "experiment_condition": condition,
        }
    )
    return item


def process_docs(dataset: Dataset) -> Dataset:
    """Use one balanced ordering for every object-answer relation and condition."""
    source = base.process_docs(dataset)
    rows = []
    conditions = _selected_conditions()
    for doc in source:
        if doc["diagnostic_answer_format"] != "object":
            continue
        if int(doc["answer_idx"]) != _balanced_option_index(doc):
            continue
        rows.extend(_condition_doc(doc, condition) for condition in conditions)
    eval_logger.info(
        "COMFORT coordinate-representation task loaded {} object prompts from {} scenes; conditions={}",
        len(rows), len({row["scene_id"] for row in rows}), Counter(row["experiment_condition"] for row in rows),
    )
    return Dataset.from_list(rows)


def _candidate_objects(doc: dict) -> list[dict]:
    """Reference first, then the four non-reference candidates in semantic order."""
    return [aids.get_reference_object(doc), *(aids.get_object_at_direction(doc, direction) for direction in base.DIRECTIONS)]


def _candidate_id(index: int) -> str:
    return "R" if index == 0 else chr(ord("A") + index - 1)


def _id_legend(doc: dict) -> str:
    entries = []
    for index, obj in enumerate(_candidate_objects(doc)):
        entries.append(f"{_candidate_id(index)}={obj.get('label', '')}")
    return "; ".join(entries)


def _draw_ids(doc: dict, image: Image.Image) -> Image.Image:
    output = image.copy()
    for index, obj in enumerate(_candidate_objects(doc)):
        color = aids.REFERENCE_COLOR if index == 0 else aids.ALL_OBJECT_COLORS[(index - 1) % len(aids.ALL_OBJECT_COLORS)]
        aids._draw_box(output, obj, color, _candidate_id(index))
    return output


def _rotated(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    x, y = vector
    return (x * math.cos(radians) - y * math.sin(radians), x * math.sin(radians) + y * math.cos(radians))


def _draw_axes(doc: dict, image: Image.Image, rotation_degrees: float = 0.0) -> Image.Image:
    output = image.copy()
    reference = aids.get_reference_object(doc)
    left, top, right, bottom = aids.bbox_to_pixels(reference, output)
    start = ((left + right) / 2.0, (top + bottom) / 2.0)
    length = max(30.0, 0.90 * math.hypot(right - left, bottom - top))
    draw = ImageDraw.Draw(output)
    width = aids._line_width(output) + 1
    for direction in base.DIRECTIONS:
        aids._draw_direction_arrow(
            draw, start, _rotated(aids._direction_screen_direction(doc, direction), rotation_degrees),
            length, aids.DIRECTION_ARROW_COLORS[direction], direction, width, output,
        )
    return output


def _jitter_offset(scene_id: str, candidate_index: int, amount: float, size: int) -> tuple[float, float]:
    """Deterministic, zero-mean position perturbation measured in image diagonal."""
    digits = "".join(character for character in scene_id if character.isdigit())
    phase = (int(digits or 0) * 37 + candidate_index * 97) % 360
    magnitude = amount * math.sqrt(2.0) * size
    radians = math.radians(phase)
    return (magnitude * math.cos(radians), magnitude * math.sin(radians))


def _parse_condition(condition: str) -> tuple[str, float, float]:
    """Return representation, candidate jitter fraction, and axis rotation."""
    if condition in {"rgb_ids", "rgb_axes_gt", "symbolic_gt", "hybrid_gt"}:
        return condition.removesuffix("_gt"), 0.0, 0.0
    representation, kind, value = condition.rsplit("_", 2)
    if kind == "jitter":
        return representation, float(value.replace("p", ".")), 0.0
    if kind == "orientation":
        return representation, 0.0, float(value)
    raise ValueError(f"Cannot parse representation condition {condition!r}")


def _symbolic_panel(doc: dict, size: int, *, jitter: float, rotation_degrees: float) -> Image.Image:
    """Image-plane symbolic rendering; labels identify candidates but not relations."""
    panel = Image.new("RGB", (size, size), aids.TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font = aids._map_font(max(15, round(size / 30)))
    label_font = aids._map_font(max(13, round(size / 34)))
    aids._centered_map_text(draw, (size / 2, 10), "Symbolic image-plane layout", title_font, (20, 20, 20), size)
    reference = aids.get_reference_object(doc)
    ref_left, ref_top, ref_right, ref_bottom = aids.bbox_to_pixels(reference, Image.new("RGB", (size, size)))
    ref_center = ((ref_left + ref_right) / 2.0, (ref_top + ref_bottom) / 2.0)
    # Draw the reference and candidates at their GT image-plane centres.
    radius = max(11.0, size * 0.028)
    for index, obj in enumerate(_candidate_objects(doc)):
        left, top, right, bottom = aids.bbox_to_pixels(obj, panel)
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        if index:
            dx, dy = _jitter_offset(str(doc["scene_id"]), index, jitter, size)
            center = (max(radius, min(size - radius, center[0] + dx)), max(radius + 28, min(size - radius, center[1] + dy)))
        color = aids.REFERENCE_COLOR if index == 0 else aids.ALL_OBJECT_COLORS[(index - 1) % len(aids.ALL_OBJECT_COLORS)]
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=color, outline=(20, 20, 20), width=2)
        aids._centered_map_text(draw, (center[0], center[1] - label_font.size / 2), _candidate_id(index), label_font, (0, 0, 0), size)
    length = max(30.0, 0.90 * math.hypot(ref_right - ref_left, ref_bottom - ref_top))
    width = max(2, aids._line_width(panel))
    for direction in base.DIRECTIONS:
        aids._draw_direction_arrow(
            draw, ref_center, _rotated(aids._direction_screen_direction(doc, direction), rotation_degrees),
            length, aids.DIRECTION_ARROW_COLORS[direction], direction, width, panel,
        )
    return panel


def _render(doc: dict) -> Image.Image:
    image = base.doc_to_visual(doc)[0]
    representation, jitter, rotation = _parse_condition(str(doc["experiment_condition"]))
    if representation == "rgb_ids":
        return _draw_ids(doc, image)
    if representation == "rgb_axes":
        return _draw_axes(doc, _draw_ids(doc, image), rotation)
    panel = _symbolic_panel(doc, image.height, jitter=jitter, rotation_degrees=rotation)
    if representation == "symbolic":
        return panel
    if representation == "hybrid":
        return aids._append_map_panels(_draw_axes(doc, _draw_ids(doc, image), rotation), [panel])
    raise ValueError(f"Unknown representation {representation!r}")


def _debug_enabled() -> bool:
    return str(os.getenv("COMFORT_REPRESENTATION_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}


def doc_to_visual(doc):
    image = _render(doc)
    if _debug_enabled():
        root = Path(os.getenv("COMFORT_REPRESENTATION_DEBUG_DIR", str(DEBUG_DEFAULT_DIR)))
        directory = root / str(doc["experiment_condition"])
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(doc["qid"]))
        image.save(directory / f"{safe}.png", format="PNG")
    return [image]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    options = "\n".join(f"{letter}. {option}" for letter, option in zip(base.OPTION_LETTERS, doc["options"]))
    representation, _, _ = _parse_condition(str(doc["experiment_condition"]))
    representation_instruction = (
        "Use the displayed reference-relative coordinate axes to locate the requested candidate. "
        if representation != "rgb_ids"
        else "Use the original image to answer the reference-relative spatial question. "
    )
    return (
        f"{kwargs.get('pre_prompt', '')}Candidate-ID legend: {_id_legend(doc)}. "
        f"{representation_instruction}"
        "The symbolic layout, when present, is an image-plane abstraction; its letters identify candidates but do not encode their direction labels. "
        "Select one answer option and respond with its letter only.\n"
        f"Question: {doc['diagnostic_question']}\nOptions:\n{options}{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return str(doc["gold_option_letter"])


def _entry(doc: dict, results) -> dict:
    prediction = results[0].strip() if results else ""
    parsed = base.extract_option_letter(prediction)
    representation, jitter, rotation = _parse_condition(str(doc["experiment_condition"]))
    return {
        "qid": doc["qid"], "source_qid": doc["source_qid"], "scene_id": doc["scene_id"],
        "task": "comfort_coordinate_representation_analysis", "experiment_condition": doc["experiment_condition"],
        "representation": representation, "candidate_jitter_fraction": jitter, "axis_rotation_degrees": rotation,
        "answer_format": "object", "relation": doc["diagnostic_relation"], "anchor": doc["diagnostic_anchor"],
        "target": doc["diagnostic_target_object"], "options": list(doc["options"]),
        "gold_option_letter": doc["gold_option_letter"], "predicted_option_letter": parsed,
        "parse_success": float(parsed in base.OPTION_LETTERS), "score": float(parsed == doc["gold_option_letter"]),
        "prediction": prediction,
    }


def process_results(doc, results):
    entry = _entry(doc, results)
    output = {metric: dict(entry) for metric in ("accuracy", "parse_success_rate")}
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "img_path": doc["img_path"]}
    return output


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _condition_summary(results) -> dict:
    grouped = defaultdict(list)
    for row in results:
        grouped[row["experiment_condition"]].append(row)
    return {
        condition: {"count": len(rows), "accuracy": _mean(float(row["score"]) for row in rows),
                    "parse_success": _mean(float(row["parse_success"]) for row in rows)}
        for condition, rows in sorted(grouped.items())
    }


def aggregate_accuracy(results):
    summary = _condition_summary(results)
    eval_logger.info("COMFORT coordinate-representation results by condition: {}", summary)
    return _mean(float(row["score"]) for row in results)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"comfort_coordinate_representation_analysis_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "COMFORT_Multi_3D", "task": "comfort_coordinate_representation_analysis",
            "num_records": len(results), "condition_summary": _condition_summary(results), "records": results,
        }, handle, indent=2)
    eval_logger.info("COMFORT coordinate-representation records saved to {}", path)
