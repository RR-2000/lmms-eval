"""E3 reference-centroid noise for matched Dir_Clr RGB and symbolic views."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a_direction_object import utils as relative
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_noise import utils as dirclr
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_axis_overlay_symbolic_up_noise import utils as symbolic
from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_representation_noise import utils as noise
from lmms_eval.utils import sanitize_model_name


TASK_NAME = "kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_reference_centroid_noise"
DEBUG_DEFAULT_DIR = Path("outputs/kubric_axis_overlay_dirclr_reference_centroid_noise_debug")
LEVELS = (0.10, 0.25, 0.50, 1.00)
REPRESENTATIONS = ("rgb_overlay", "symbolic")


def _condition(representation: str, level: float) -> str:
    return f"{representation}_reference_centroid_{level:0.2f}".replace(".", "p")


def _condition_parts(condition: str) -> tuple[str, float]:
    for representation in REPRESENTATIONS:
        prefix = f"{representation}_reference_centroid_"
        if condition.startswith(prefix):
            return representation, float(condition.removeprefix(prefix).replace("p", "."))
    raise ValueError(f"Unknown E3 condition {condition!r}")


def _object(doc: dict, name: str) -> dict | None:
    return next((item for item in noise._objects(doc) if str(item.get("name")) == name), None)


def _box(doc: dict, name: str) -> tuple[float, float, float, float] | None:
    item = _object(doc, name)
    value = item.get("bbox_2d_xyxy_pixels") if item else None
    if not (isinstance(value, list) and len(value) == 4):
        return None
    box = tuple(float(coordinate) for coordinate in value)
    return box if all(np.isfinite(box)) else None


def _center(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def _reference_target_distance(doc: dict) -> float:
    reference_box = _box(doc, noise._reference_name(doc))
    target_box = _box(doc, str(doc["diagnostic_target_object"]))
    if reference_box is None or target_box is None:
        return 0.0
    return float(np.linalg.norm(_center(target_box) - _center(reference_box)))


def _reference_shift_pixels(doc: dict, level: float) -> np.ndarray:
    """Return the shared RGB/symbolic N(0, (level * ref-target distance)^2 I) draw."""
    sigma = level * _reference_target_distance(doc)
    # Reuse the same standard-normal variate across levels, so each question's
    # sweep is a scaled trajectory rather than four unrelated perturbations.
    seed = noise._stable_seed(doc.get("source_qid"), "reference_centroid")
    return np.random.default_rng(seed).normal(0.0, sigma, size=2)


def process_docs(dataset: Dataset) -> Dataset:
    """Build 218 object-answer questions × four levels × two representations."""
    rows = []
    normalized = relative.relative_direction_process_docs(dataset)
    sources = [
        dict(doc) for doc in normalized
        if doc.get("source_task_family") == "object_relative_direction"
        and doc.get("diagnostic_answer_format") == "object"
    ]
    for source in sources:
        for level in LEVELS:
            shift = _reference_shift_pixels(source, level)
            for representation in REPRESENTATIONS:
                condition = _condition(representation, level)
                doc = dict(source)
                qid = f"{source['source_qid']}::object::dirclr_e3::{condition}"
                doc.update({
                    "qid": qid,
                    "index": qid,
                    "experiment_condition": condition,
                    "representation": representation,
                    "noise_family": "reference_centroid",
                    "noise_level": level,
                    "reference_centroid_shift_pixels": shift.tolist(),
                    "reference_target_distance_pixels": _reference_target_distance(source),
                    "diagnostic_experiment": "dirclr_e3_reference_centroid_noise",
                })
                rows.append(doc)
    eval_logger.info(
        "MOVi-A E3 loaded {} records ({} object-answer questions × {} levels × {} representations).",
        len(rows), len(sources), len(LEVELS), len(REPRESENTATIONS),
    )
    return Dataset.from_list(rows)


def _rgb_overlay(doc: dict, level: float) -> Image.Image:
    # Render the original clean axes, changing only their common origin.
    image = dirclr.kubric._load_image(doc.get("image")) or dirclr.kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    output = image.convert("RGB").copy()
    x1, y1, x2, y2 = dirclr._reference_box(doc, output)
    start = _center((x1, y1, x2, y2)) + _reference_shift_pixels(doc, level)
    length = 0.8 * float(np.hypot(x2 - x1, y2 - y1))
    front = noise._heading_screen_vector(doc)
    directions = {
        "front": front,
        "right": noise._rotate(front, 90),
        "back": -front,
        "left": noise._rotate(front, -90),
    }
    draw = dirclr.ImageDraw.Draw(output)
    width = max(2, round(min(output.size) / 220))
    font = dirclr._font(output)
    for label in ("front", "right", "back", "left"):
        dirclr._draw_arrow(draw, start, directions[label], length, dirclr.AXIS_COLORS[label], label, width, font)
    return output


def _symbolic_nodes(doc: dict, level: float, image: Image.Image) -> dict[str, np.ndarray]:
    nodes = symbolic._heading_up_nodes(doc, "clean", 0.0)
    reference = noise._reference_name(doc)
    if reference in nodes:
        delta = _reference_shift_pixels(doc, level)
        nodes[reference] = nodes[reference] + np.asarray([delta[0] / image.width, delta[1] / image.height])
    return nodes


def _symbolic_map(doc: dict, level: float) -> Image.Image:
    image = dirclr.kubric._load_image(doc.get("image")) or dirclr.kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    canvas = Image.new("RGB", image.size, (255, 255, 255))
    nodes = _symbolic_nodes(doc, level, canvas)
    noise._draw_nodes(canvas, doc, nodes, set(nodes), transparent=False)
    return canvas


def _render(doc: dict) -> Image.Image:
    representation, level = _condition_parts(str(doc["experiment_condition"]))
    return _rgb_overlay(doc, level) if representation == "rgb_overlay" else _symbolic_map(doc, level)


def _point_box_distance(point: np.ndarray, box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return float(np.hypot(dx, dy))


def geometric_trial(doc: dict, level: float | None = None) -> dict:
    """Score whether the shifted correct-axis endpoint favors a wrong candidate box."""
    if level is None:
        _, level = _condition_parts(str(doc["experiment_condition"]))
    reference_box = _box(doc, noise._reference_name(doc))
    target = str(doc["diagnostic_target_object"])
    target_box = _box(doc, target)
    candidates = [str(name) for name in doc.get("candidate_objects", [])]
    wrong_boxes = [(name, _box(doc, name)) for name in candidates if name != target]
    wrong_boxes = [(name, box) for name, box in wrong_boxes if box is not None]
    if reference_box is None or target_box is None or not wrong_boxes:
        return {"valid": False, "wrong_box_nearer": None}

    start = _center(reference_box) + _reference_shift_pixels(doc, level)
    length = 0.8 * float(np.hypot(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1]))
    relation = str(doc["diagnostic_relation"]).lower()
    front = noise._heading_screen_vector(doc)
    vectors = {"front": front, "right": noise._rotate(front, 90), "behind": -front, "left": noise._rotate(front, -90)}
    endpoint = start + vectors[relation] * length
    target_distance = _point_box_distance(endpoint, target_box)
    wrong_name, wrong_distance = min(
        ((name, _point_box_distance(endpoint, box)) for name, box in wrong_boxes),
        key=lambda item: item[1],
    )
    confused = wrong_distance < target_distance
    return {
        "valid": True,
        "endpoint_pixels": endpoint.tolist(),
        "target_box_distance_pixels": target_distance,
        "nearest_wrong_object": wrong_name,
        "nearest_wrong_box_distance_pixels": wrong_distance,
        "wrong_box_nearer": bool(confused),
    }


def geometric_curve(docs) -> list[dict]:
    """Return the no-VLM wrong-box fraction and its complementary ceiling."""
    unique = {}
    for doc in docs:
        source_qid = str(doc["source_qid"])
        if "geometric_trial" in doc and "noise_level" in doc:
            unique[(source_qid, float(doc["noise_level"]))] = doc["geometric_trial"]
        else:
            for level in LEVELS:
                unique[(source_qid, level)] = geometric_trial(doc, level)
    curve = []
    for level in LEVELS:
        trials = [trial for (qid, trial_level), trial in unique.items() if trial_level == level and trial["valid"]]
        wrong_fraction = sum(bool(trial["wrong_box_nearer"]) for trial in trials) / len(trials) if trials else 0.0
        curve.append({
            "sigma_fraction": level,
            "num_trials": len(trials),
            "wrong_box_fraction": wrong_fraction,
            "geometric_ceiling": 1.0 - wrong_fraction,
        })
    return curve


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


doc_to_target = symbolic.doc_to_target


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    representation, _ = _condition_parts(str(doc["experiment_condition"]))
    if representation == "rgb_overlay":
        instruction = "The four labelled arrows define the reference object's front, right, back, and left directions. "
    else:
        instruction = (
            f"Object-symbol legend: {noise._symbol_legend(doc)}. "
            "The symbolic layout is reference-aligned: the reference object's front is image-up. "
        )
    options = dirclr.kubric._get_options(doc)
    return (
        "Answer the spatial-reasoning question using the displayed visual representation. "
        + instruction
        + "Select one answer option and respond with its letter only.\n"
        + f"Question: {doc['question']}\nOptions:\n"
        + "".join(f"{letter}. {value}\n" for letter, value in options.items())
    )


def _entry(doc: dict, results) -> dict:
    entry = symbolic._entry(doc, results)
    return {
        **entry,
        "reference_centroid_shift_pixels": list(doc["reference_centroid_shift_pixels"]),
        "reference_target_distance_pixels": float(doc["reference_target_distance_pixels"]),
        "geometric_trial": geometric_trial(doc),
    }


def process_results(doc, results):
    entry = _entry(doc, results)
    output = {metric: dict(entry) for metric in ("accuracy", "parse_success_rate")}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "img_path": dirclr.kubric._get_image_path(doc),
    }
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
            "count": len(items),
            "accuracy": _mean(float(item["score"]) for item in items),
            "parse_success": _mean(float(item["parse_success"]) for item in items),
            "wrong_box_fraction": _mean(
                float(item["geometric_trial"]["wrong_box_nearer"])
                for item in items if item["geometric_trial"]["valid"]
            ),
        }
        for condition, items in sorted(grouped.items())
    }


def aggregate_accuracy(results):
    eval_logger.info("MOVi-A E3 results by condition: {}", _condition_summary(results))
    return _mean(float(row["score"]) for row in results)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{TASK_NAME}_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "MOVi-A relative direction",
            "task": TASK_NAME,
            "num_records": len(results),
            "condition_summary": _condition_summary(results),
            "geometric_curve": geometric_curve(results),
            "records": results,
        }, handle, indent=2)
    eval_logger.info("MOVi-A E3 reference-centroid records saved to {}", path)
