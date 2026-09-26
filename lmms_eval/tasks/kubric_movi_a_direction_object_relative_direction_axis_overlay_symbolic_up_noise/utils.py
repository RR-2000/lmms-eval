"""Axis-only RGB overlays and symbol-only heading-up maps for MOVi-A."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a import utils as kubric
from lmms_eval.tasks.kubric_movi_a_direction_object import utils as base
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_representation_noise import utils as noise
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "kubric_movi_a_direction_object_relative_direction_axis_overlay_symbolic_up_noise"
DEBUG_DEFAULT_DIR = Path("outputs/kubric_axis_overlay_symbolic_up_noise_debug")
AXIS_COLORS = {"front": (220, 35, 35), "right": (35, 100, 220), "back": (240, 145, 25), "left": (40, 150, 75)}


def process_docs(dataset: Dataset) -> Dataset:
    """Reuse the exact source pairs, conditions, and deterministic corruptions."""
    rows = []
    for source in noise.process_docs(dataset):
        doc = dict(source)
        qid = f"{doc['source_qid']}::{doc['diagnostic_answer_format']}::axis_overlay_symbolic_up::{doc['experiment_condition']}"
        doc.update({"qid": qid, "index": qid, "diagnostic_experiment": "axis_overlay_vs_symbolic_heading_up_noise"})
        rows.append(doc)
    return Dataset.from_list(rows)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], vector: np.ndarray, length: float, color: tuple[int, int, int], label: str, canvas: Image.Image) -> None:
    end = (start[0] + float(vector[0]) * length, start[1] + float(vector[1]) * length)
    width = max(2, round(min(canvas.size) / 110))
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(float(vector[1]), float(vector[0]))
    head = max(8.0, length * 0.18)
    for offset in (2.55, -2.55):
        point = (end[0] + head * math.cos(angle + offset), end[1] + head * math.sin(angle + offset))
        draw.line((end, point), fill=color, width=width)
    font = noise._font(max(12, round(min(canvas.size) / 22)))
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    x = min(max(2, end[0] - text_width / 2), canvas.width - text_width - 2)
    y = min(max(2, end[1] - text_height / 2), canvas.height - text_height - 2)
    draw.rectangle((x - 2, y - 1, x + text_width + 2, y + text_height + 1), fill=(255, 255, 255), outline=color, width=1)
    draw.text((x, y - text_box[1]), label, fill=color, font=font)


def _rgb_axis_overlay(doc: dict, perturbation: str, level: float) -> Image.Image:
    image = kubric._load_image(doc.get("image")) or kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    canvas = image.copy()
    reference = noise._reference_name(doc)
    nodes = noise._display_nodes(doc, "clean", 0.0)
    if reference not in nodes:
        return canvas
    front = noise._heading_screen_vector(doc)
    if perturbation == "orientation":
        front = noise._rotate(front, level)
    vectors = {
        "front": front,
        "right": noise._rotate(front, 90),
        "back": -front,
        "left": noise._rotate(front, -90),
    }
    center = (float(nodes[reference][0]) * canvas.width, float(nodes[reference][1]) * canvas.height)
    length = max(26.0, min(canvas.size) * 0.18)
    draw = ImageDraw.Draw(canvas)
    for label in ("front", "right", "back", "left"):
        _draw_arrow(draw, center, vectors[label], length, AXIS_COLORS[label], label, canvas)
    return canvas


def _heading_up_nodes(doc: dict, perturbation: str, level: float) -> dict[str, np.ndarray]:
    nodes = noise._display_nodes(doc, perturbation, level)
    reference = noise._reference_name(doc)
    ref = nodes.get(reference)
    if ref is None:
        return nodes
    front = noise._heading_screen_vector(doc)
    if perturbation == "orientation":
        front = noise._rotate(front, level)
    angle = math.atan2(float(front[1]), float(front[0]))
    rotation = math.degrees(-math.pi / 2.0 - angle)
    rotated = {name: noise._rotate(point - ref, rotation) for name, point in nodes.items()}
    # Refit the rotated layout into the symbolic canvas so alignment never
    # crops symbols at an image edge.
    extent = max((float(np.max(np.abs(vector))) for vector in rotated.values()), default=1.0)
    scale = 0.40 / max(extent, 1e-6)
    return {name: np.asarray([0.5, 0.5]) + vector * scale for name, vector in rotated.items()}


def _symbolic_map(doc: dict, perturbation: str, level: float) -> Image.Image:
    image = kubric._load_image(doc.get("image")) or kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    canvas = Image.new("RGB", image.size, (255, 255, 255))
    nodes = _heading_up_nodes(doc, perturbation, level)
    kept = noise._kept_names(doc, perturbation, level)
    # Reuse the prior symbol-only drawing; it contains circles and letters,
    # never names, arrows, direction text, or a legend inside the image.
    noise._draw_nodes(canvas, doc, nodes, kept, transparent=False)
    return canvas


def _render(doc: dict) -> Image.Image:
    representation, perturbation, level = noise._condition_parts(str(doc["experiment_condition"]))
    if representation == "rgb_overlay":
        return _rgb_axis_overlay(doc, perturbation, level)
    return _symbolic_map(doc, perturbation, level)


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


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    representation, _, _ = noise._condition_parts(str(doc["experiment_condition"]))
    if representation == "rgb_overlay":
        instruction = "The four labelled arrows define the reference object's front, right, back, and left directions. "
    else:
        instruction = (
            f"Object-symbol legend: {noise._symbol_legend(doc)}. "
            "The symbolic layout is reference-aligned: the reference object's front is image-up. "
        )
    options = kubric._get_options(doc)
    return (
        "Answer the spatial-reasoning question using the displayed visual representation. " + instruction
        + "Select one answer option and respond with its letter only.\n"
        + f"Question: {doc['question']}\nOptions:\n"
        + "".join(f"{letter}. {value}\n" for letter, value in options.items())
    )


def doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def _entry(doc: dict, results) -> dict:
    prediction = results[0].strip() if results else ""
    parsed = base._extract_answer(prediction)
    return {
        "qid": doc["qid"], "source_qid": doc["source_qid"], "pair_id": doc["source_qid"],
        "source_task_family": doc["source_task_family"], "answer_format": doc["diagnostic_answer_format"],
        "relation": doc["diagnostic_relation"], "reference_object": noise._reference_name(doc),
        "target_object": doc["diagnostic_target_object"], "experiment_condition": doc["experiment_condition"],
        "representation": doc["representation"], "noise_family": doc["noise_family"], "noise_level": doc["noise_level"],
        "prediction": prediction, "parsed_prediction": parsed, "gold_option": doc_to_target(doc),
        "gold_target": doc["diagnostic_target"], "parse_success": float(parsed in {"A", "B", "C", "D"}),
        "score": float(parsed == doc_to_target(doc)),
    }


def process_results(doc, results):
    entry = _entry(doc, results)
    output = {metric: dict(entry) for metric in ("accuracy", "parse_success_rate")}
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "img_path": kubric._get_image_path(doc)}
    return output


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _condition_summary(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment_condition"]].append(row)
    return {
        condition: {
            "count": len(items), "accuracy": _mean(float(item["score"]) for item in items),
            "parse_success": _mean(float(item["parse_success"]) for item in items),
            "direction_accuracy": _mean(float(item["score"]) for item in items if item["answer_format"] == "direction"),
            "object_accuracy": _mean(float(item["score"]) for item in items if item["answer_format"] == "object"),
        }
        for condition, items in sorted(grouped.items())
    }


def aggregate_accuracy(results):
    summary = _condition_summary(results)
    eval_logger.info("MOVi-A axis-overlay / heading-up symbolic results by condition: {}", summary)
    return _mean(float(row["score"]) for row in results)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "MOVi-A relative direction", "task": TASK_NAME, "num_records": len(results),
            "condition_summary": _condition_summary(results), "records": results,
        }, handle, indent=2)
    eval_logger.info("MOVi-A axis-overlay / heading-up symbolic records saved to {}", path)
