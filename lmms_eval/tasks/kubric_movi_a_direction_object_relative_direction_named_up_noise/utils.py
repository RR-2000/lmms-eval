"""Direct entity-name representations for the MOVi-A noise diagnostic."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a import utils as kubric
from lmms_eval.tasks.kubric_movi_a_direction_object import utils as base
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_representation_noise import utils as noise
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "kubric_movi_a_direction_object_relative_direction_named_up_noise"
DEBUG_DEFAULT_DIR = Path("outputs/kubric_relative_named_up_noise_debug")


def process_docs(dataset):
    """Reuse exactly the matched pairs, conditions, and corruption seeds."""
    records = []
    for source in noise.process_docs(dataset):
        doc = dict(source)
        qid = f"{doc['source_qid']}::{doc['diagnostic_answer_format']}::named_up_noise::{doc['experiment_condition']}"
        doc.update({
            "qid": qid,
            "index": qid,
            "diagnostic_experiment": "direct_names_vs_heading_up_symbolic_noise",
        })
        records.append(doc)
    return _dataset(records)


def _dataset(records):
    # Keep this local rather than importing a private Dataset helper.
    from datasets import Dataset
    return Dataset.from_list(records)


def _shown_names(doc: dict) -> set[str]:
    """Show only task-relevant candidates, not every scene distractor."""
    names = {_name for _name in (_reference_name(doc), str(doc.get("diagnostic_target_object", ""))) if _name}
    names.update(str(name) for name in doc.get("candidate_objects", []) if name)
    return names


def _reference_name(doc: dict) -> str:
    return noise._reference_name(doc)


def _overlap_area(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))


def _draw_name(
    draw: ImageDraw.ImageDraw,
    name: str,
    center: tuple[float, float],
    canvas: Image.Image,
    *,
    rgb: bool,
    occupied: list[tuple[float, float, float, float]],
) -> None:
    font = noise._font(max(11, round(min(canvas.size) / 24)))
    box = draw.textbbox((0, 0), name, font=font)
    text_width, text_height = box[2] - box[0], box[3] - box[1]
    distance = max(10, round(min(canvas.size) * 0.045))
    # Long labels need room to fan out in crowded scenes.  Leader lines retain
    # the object association when a nearer slot is occupied.
    candidates = []
    for radius in (distance, max(70, distance * 3), max(145, distance * 6), max(220, distance * 9)):
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            candidates.append((
                center[0] + radius * math.cos(radians) - text_width / 2,
                center[1] + radius * math.sin(radians) - text_height / 2,
            ))
    boxes = []
    for proposed_x, proposed_y in candidates:
        x = min(max(2, proposed_x), canvas.width - text_width - 2)
        y = min(max(2, proposed_y), canvas.height - text_height - 2)
        boxes.append((x - 2, y - 1, x + text_width + 2, y + text_height + 1))
    rectangle = min(boxes, key=lambda box: sum(_overlap_area(box, prior) for prior in occupied))
    x, y = rectangle[0] + 2, rectangle[1] + 1
    # Opaque backing makes names legible on RGB; on symbolic it simply anchors
    # names without introducing symbols, directions, or a legend.
    fill = (255, 255, 255) if rgb else (245, 245, 245)
    draw.line((center, (x, y + text_height / 2)), fill=(20, 20, 20), width=1)
    draw.rectangle(rectangle, fill=fill, outline=(20, 20, 20), width=1)
    draw.text((x, y - box[1]), name, fill=(15, 15, 15), font=font)
    occupied.append(rectangle)


def _heading_up_nodes(doc: dict, perturbation: str, level: float) -> dict[str, np.ndarray]:
    """Rotate and fit the symbolic layout so its estimated heading is up."""
    nodes = noise._display_nodes(doc, perturbation, level)
    reference = noise._reference_name(doc)
    ref = nodes.get(reference)
    if ref is None:
        return nodes
    heading = noise._heading_screen_vector(doc)
    if perturbation == "orientation":
        heading = noise._rotate(heading, level)
    angle = math.atan2(float(heading[1]), float(heading[0]))
    # Rotate the node vectors until the (possibly corrupted) heading points up.
    delta_degrees = math.degrees(-math.pi / 2.0 - angle)
    rotated = {name: noise._rotate(point - ref, delta_degrees) for name, point in nodes.items()}
    extent = max((float(np.max(np.abs(vector))) for vector in rotated.values()), default=1.0)
    scale = 0.40 / max(extent, 1e-6)
    return {name: np.asarray([0.5, 0.5]) + vector * scale for name, vector in rotated.items()}


def _render(doc: dict) -> Image.Image:
    representation, perturbation, level = noise._condition_parts(str(doc["experiment_condition"]))
    image = kubric._load_image(doc.get("image")) or kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    rgb = representation == "rgb_overlay"
    # Entity names are long (for example, "large green metal cylinder").
    # Render at 512 px so they remain legible to the VLM rather than becoming
    # an accidental OCR-noise perturbation.
    side = max(512, image.width, image.height)
    size = (side, side)
    canvas = image.resize(size, Image.Resampling.LANCZOS) if rgb else Image.new("RGB", size, (255, 255, 255))
    nodes = noise._display_nodes(doc, perturbation, level) if rgb else _heading_up_nodes(doc, perturbation, level)
    kept = noise._kept_names(doc, perturbation, level)
    draw = ImageDraw.Draw(canvas)
    occupied = []
    shown = _shown_names(doc)
    for name, point in nodes.items():
        if name in kept and name in shown:
            _draw_name(
                draw, name, (float(point[0]) * canvas.width, float(point[1]) * canvas.height), canvas,
                rgb=rgb, occupied=occupied,
            )
    if rgb:
        reference = _reference_name(doc)
        if reference in nodes:
            heading = noise._heading_screen_vector(doc)
            if perturbation == "orientation":
                heading = noise._rotate(heading, level)
            center = (float(nodes[reference][0]) * canvas.width, float(nodes[reference][1]) * canvas.height)
            noise._draw_arrow(draw, center, heading, max(22.0, min(canvas.size) * 0.14), max(2, round(min(canvas.size) / 110)))
    return canvas


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
    representation_instruction = (
        "Each visible entity is labelled directly by name; the red arrow shows the reference object's heading. "
        if representation == "rgb_overlay"
        else "Each visible entity is labelled directly by name. The layout is normalized so the reference object's heading is image-up. "
    )
    options = kubric._get_options(doc)
    return (
        "Answer the spatial-reasoning question using the displayed visual representation. "
        + representation_instruction
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
    eval_logger.info("MOVi-A direct-name / heading-up results by condition: {}", summary)
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
    eval_logger.info("MOVi-A direct-name / heading-up records saved to {}", path)
