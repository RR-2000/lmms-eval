"""A rendering-only Dir_Clr style variant of the RGB-axis diagnostic."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a import utils as kubric
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_axis_overlay_symbolic_up_noise import utils as base_task
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_representation_noise import utils as noise
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_noise"
DEBUG_DEFAULT_DIR = Path("outputs/kubric_axis_overlay_dirclr_style_noise_debug")
AXIS_COLORS = {
    "front": (45, 220, 90),
    "right": (255, 170, 35),
    "back": (235, 70, 70),
    "left": (60, 165, 255),
}


def process_docs(dataset: Dataset) -> Dataset:
    """Preserve every non-rendering aspect of the base diagnostic."""
    rows = []
    for source in base_task.process_docs(dataset):
        doc = dict(source)
        qid = f"{doc['source_qid']}::{doc['diagnostic_answer_format']}::axis_overlay_dirclr_style::{doc['experiment_condition']}"
        doc.update({"qid": qid, "index": qid, "diagnostic_experiment": "axis_overlay_dirclr_style_vs_symbolic_heading_up_noise"})
        rows.append(doc)
    return Dataset.from_list(rows)


def _font(image: Image.Image) -> ImageFont.ImageFont:
    """Exact font policy from Dir_Clr object_direction_reasoning._font."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, round(min(image.size) / 70)))
    except OSError:
        return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: tuple[int, int, int],
    label: str,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    """Exact line, triangle-head, and stroked-label treatment from Dir_Clr."""
    end = start + direction * length
    draw.line((*start, *end), fill=color, width=width)
    perpendicular = np.asarray([-direction[1], direction[0]])
    head = max(7.0, 2.4 * width)
    draw.polygon([
        tuple(end),
        tuple(end - head * direction + 0.45 * head * perpendicular),
        tuple(end - head * direction - 0.45 * head * perpendicular),
    ], fill=color)
    draw.text(tuple(end + 3 * direction), label.upper(), fill=color, font=font, stroke_width=2, stroke_fill="black")


def _reference_box(doc: dict, image: Image.Image) -> tuple[float, float, float, float]:
    reference = noise._reference_name(doc)
    obj = next((item for item in noise._objects(doc) if str(item["name"]) == reference), None)
    if obj:
        bbox = obj.get("bbox_2d_xyxy_pixels")
        if isinstance(bbox, list) and len(bbox) == 4:
            return tuple(float(value) for value in bbox)
    # Defensive fallback only; normal MOVi-A rows have reference pixel boxes.
    position = noise._display_nodes(doc, "clean", 0.0).get(reference, np.asarray([0.5, 0.5]))
    x, y = float(position[0]) * image.width, float(position[1]) * image.height
    return (x - 10.0, y - 10.0, x + 10.0, y + 10.0)


def _dirclr_rgb_axes(doc: dict, perturbation: str, level: float) -> Image.Image:
    image = kubric._load_image(doc.get("image")) or kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    output = image.convert("RGB").copy()
    x1, y1, x2, y2 = _reference_box(doc, output)
    start = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    # The BBoxDiagonal PBS entry point sets this value to 0.8.
    length = 0.8 * math.hypot(x2 - x1, y2 - y1)
    front = noise._heading_screen_vector(doc)
    if perturbation == "orientation":
        front = noise._rotate(front, level)
    directions = {
        "front": front,
        "right": noise._rotate(front, 90),
        "back": -front,
        "left": noise._rotate(front, -90),
    }
    draw = ImageDraw.Draw(output)
    width = max(2, round(min(output.size) / 220))
    font = _font(output)
    for label in ("front", "right", "back", "left"):
        _draw_arrow(draw, start, directions[label], length, AXIS_COLORS[label], label, width, font)
    return output


def _render(doc: dict) -> Image.Image:
    representation, perturbation, level = noise._condition_parts(str(doc["experiment_condition"]))
    if representation == "rgb_overlay":
        return _dirclr_rgb_axes(doc, perturbation, level)
    # Deliberately unchanged from the base experiment.
    return base_task._symbolic_map(doc, perturbation, level)


def _debug_enabled() -> bool:
    return str(os.getenv("KUBRIC_RELATIVE_REPRESENTATION_DEBUG", "0")).lower() in {"1", "true", "yes", "on"}


def doc_to_visual(doc):
    image = _render(doc)
    if _debug_enabled():
        root = Path(os.getenv("KUBRIC_RELATIVE_REPRESENTATION_DEBUG_DIR", str(DEBUG_DEFAULT_DIR)))
        directory = root / str(doc["experiment_condition"])
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(doc["qid"]))
        image.save(directory / f"{safe}.png", format="PNG")
    return [image]


# The text, targets, scoring, and metric definitions are intentionally shared.
doc_to_text = base_task.doc_to_text
doc_to_target = base_task.doc_to_target
process_results = base_task.process_results
aggregate_accuracy = base_task.aggregate_accuracy
aggregate_parse_success_rate = base_task.aggregate_parse_success_rate


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "MOVi-A relative direction", "task": TASK_NAME, "num_records": len(results),
            "condition_summary": base_task._condition_summary(results), "records": results,
        }, handle, indent=2)
    eval_logger.info("MOVi-A Dir_Clr-style axis-overlay records saved to {}", path)
