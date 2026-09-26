"""RGB-overlay versus symbol-only robustness diagnostics for MOVi-A.

The symbolic rendering is intentionally austere: object glyphs and one
reference-heading arrow only.  Names are supplied as a textual legend shared
with the RGB condition, never rendered into the symbolic image.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a import utils as kubric
from lmms_eval.tasks.kubric_movi_a_direction_object import utils as base
from lmms_eval.utils import sanitize_model_name


REPRESENTATIONS = ("rgb_overlay", "symbolic")
DETECTION_LEVELS = (0.25, 0.50, 0.75)
CENTROID_LEVELS = (0.10, 0.25, 0.50, 1.00)
ORIENTATION_LEVELS = (15, 30, 45, 90, 180)
CORE = tuple(f"{representation}_clean" for representation in REPRESENTATIONS)
DETECTION = tuple(
    f"{representation}_detection_{level:0.2f}".replace(".", "p")
    for representation in REPRESENTATIONS for level in DETECTION_LEVELS
)
CENTROID = tuple(
    f"{representation}_centroid_{level:0.2f}".replace(".", "p")
    for representation in REPRESENTATIONS for level in CENTROID_LEVELS
)
ORIENTATION = tuple(
    f"{representation}_orientation_{degrees}"
    for representation in REPRESENTATIONS for degrees in ORIENTATION_LEVELS
)
ALL_CONDITIONS = CORE + DETECTION + CENTROID + ORIENTATION
DEBUG_DEFAULT_DIR = Path("outputs/kubric_relative_representation_noise_debug")


def _selected_conditions() -> tuple[str, ...]:
    explicit = str(os.getenv("KUBRIC_RELATIVE_REPRESENTATION_CONDITIONS", "")).strip()
    if explicit:
        selected = tuple(item.strip() for item in explicit.split(",") if item.strip())
    else:
        suite = str(os.getenv("KUBRIC_RELATIVE_REPRESENTATION_SUITE", "core")).strip().lower()
        suites = {
            "core": CORE,
            "detection": DETECTION,
            "centroid": CENTROID,
            "orientation": ORIENTATION,
            "all": ALL_CONDITIONS,
        }
        if suite not in suites:
            raise ValueError(
                "Unknown KUBRIC_RELATIVE_REPRESENTATION_SUITE="
                f"{suite!r}; choose {sorted(suites)}"
            )
        selected = suites[suite]
    unknown = sorted(set(selected) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown representation conditions: {unknown}")
    return selected


def _parse_condition(condition: str) -> tuple[str, str, float]:
    representation, kind, value = condition.rsplit("_", 2)
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation in {condition!r}")
    if kind == "clean" and value == "clean":  # defensive; never produced
        return representation, "clean", 0.0
    # clean has only two components, so handle it before the generic parser.
    raise ValueError(f"Cannot parse representation condition {condition!r}")


def _condition_parts(condition: str) -> tuple[str, str, float]:
    if condition.endswith("_clean"):
        return condition.removesuffix("_clean"), "clean", 0.0
    representation, kind, value = condition.rsplit("_", 2)
    if kind in {"detection", "centroid"}:
        return representation, kind, float(value.replace("p", "."))
    if kind == "orientation":
        return representation, kind, float(value)
    raise ValueError(f"Cannot parse representation condition {condition!r}")


def _stable_seed(*parts: object) -> int:
    raw = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _glyph(index: int, is_reference: bool) -> str:
    if is_reference:
        return "R"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[index % len(alphabet)]


def _objects(doc: dict) -> list[dict]:
    objects = [dict(obj) for obj in doc.get("visible_objects", []) if obj.get("name")]
    return sorted(objects, key=lambda obj: (int(obj.get("object_idx", 10**6)), str(obj["name"])))


def _reference_name(doc: dict) -> str:
    return str(doc.get("diagnostic_anchor", doc.get("reference_object", ""))).strip()


def _symbol_map(doc: dict) -> dict[str, str]:
    reference = _reference_name(doc)
    symbols = {}
    next_index = 0
    for obj in _objects(doc):
        name = str(obj["name"])
        if name == reference:
            symbols[name] = "R"
        else:
            symbols[name] = _glyph(next_index, False)
            next_index += 1
    return symbols


def _symbol_legend(doc: dict) -> str:
    symbols = _symbol_map(doc)
    return "; ".join(f"{symbol} = {name}" for name, symbol in symbols.items())


def _positions_and_projector(doc: dict) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Return 3D positions plus an affine 3D-to-image projector fitted per scene."""
    positions: dict[str, np.ndarray] = {}
    source, destination = [], []
    for obj in _objects(doc):
        position = obj.get("position_3d")
        image_position = obj.get("image_position_2d")
        if not (isinstance(position, list) and len(position) == 3):
            continue
        point = np.asarray(position, dtype=float)
        positions[str(obj["name"])] = point
        if isinstance(image_position, list) and len(image_position) == 2:
            source.append([*point, 1.0])
            destination.append([float(image_position[0]), float(image_position[1])])
    if len(source) < 4:
        return positions, None
    # Coefficients map homogeneous 3D coordinates to normalized image x/y.
    coefficient, *_ = np.linalg.lstsq(np.asarray(source), np.asarray(destination), rcond=None)
    return positions, coefficient


def _fallback_screen_position(obj: dict) -> np.ndarray:
    value = obj.get("image_position_2d", obj.get("bbox_center_norm", [0.5, 0.5]))
    if isinstance(value, list) and len(value) == 2:
        return np.asarray(value, dtype=float)
    return np.asarray([0.5, 0.5], dtype=float)


def _project(point: np.ndarray, coefficient: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if coefficient is None:
        return fallback
    output = np.asarray([*point, 1.0]) @ coefficient
    if not np.all(np.isfinite(output)):
        return fallback
    return np.clip(output, 0.03, 0.97)


def _centroid_scale(positions: dict[str, np.ndarray], reference: str) -> float:
    ref = positions.get(reference)
    if ref is None:
        return 1.0
    distances = [float(np.linalg.norm(point - ref)) for name, point in positions.items() if name != reference]
    return float(np.median(distances)) if distances else 1.0


def _display_nodes(doc: dict, perturbation: str, level: float) -> dict[str, np.ndarray]:
    reference = _reference_name(doc)
    positions, projector = _positions_and_projector(doc)
    scale = _centroid_scale(positions, reference)
    seed_root = _stable_seed(doc.get("source_qid"), "centroid")
    nodes = {}
    for obj in _objects(doc):
        name = str(obj["name"])
        point = positions.get(name)
        fallback = _fallback_screen_position(obj)
        if point is not None and perturbation == "centroid" and name != reference:
            rng = np.random.default_rng(_stable_seed(seed_root, name))
            point = point + rng.normal(0.0, level * scale, size=3)
        nodes[name] = _project(point, projector, fallback) if point is not None else fallback
    return nodes


def _kept_names(doc: dict, perturbation: str, level: float) -> set[str]:
    reference = _reference_name(doc)
    names = {str(obj["name"]) for obj in _objects(doc)}
    if perturbation != "detection" or not level:
        return names
    kept = {reference}
    for name in sorted(names - {reference}):
        rng = np.random.default_rng(_stable_seed(doc.get("source_qid"), "detection", name))
        if float(rng.random()) >= level:
            kept.add(name)
    return kept


def _heading_screen_vector(doc: dict) -> np.ndarray:
    """Infer an image-plane reference-forward vector from exported local geometry.

    MOVi-A stores local relative coordinates but not a pose quaternion.  With
    its upright objects, a world XY target vector and its local (right, front)
    vector determine yaw.  The affine scene projector converts that yaw to an
    image-plane heading.
    """
    reference = _reference_name(doc)
    target = str(doc.get("diagnostic_target_object", doc.get("target_object", "")))
    positions, projector = _positions_and_projector(doc)
    relative = (doc.get("task_metadata") or {}).get("relative_coordinates") or {}
    ref, tgt = positions.get(reference), positions.get(target)
    if ref is None or tgt is None or projector is None:
        return np.asarray([0.0, -1.0])
    world = tgt - ref
    right, front = float(relative.get("right", 0.0)), float(relative.get("front", 0.0))
    if np.hypot(world[0], world[1]) < 1e-6 or np.hypot(right, front) < 1e-6:
        return np.asarray([0.0, -1.0])
    world_angle = math.atan2(world[1], world[0])
    local_angle = math.atan2(front, right)
    forward_world_angle = world_angle + math.pi / 2.0 - local_angle
    forward_world = np.asarray([math.cos(forward_world_angle), math.sin(forward_world_angle), 0.0])
    # The first three rows are the linear portion of the affine projector.
    vector = forward_world @ projector[:3]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.asarray([0.0, -1.0])


def _rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    matrix = np.asarray(((math.cos(radians), -math.sin(radians)), (math.sin(radians), math.cos(radians))))
    return vector @ matrix.T


def _draw_arrow(draw: ImageDraw.ImageDraw, center: tuple[float, float], vector: np.ndarray, length: float, width: int) -> None:
    end = (center[0] + float(vector[0]) * length, center[1] + float(vector[1]) * length)
    draw.line((center, end), fill=(220, 30, 30), width=width)
    angle = math.atan2(float(vector[1]), float(vector[0]))
    head = max(8.0, length * 0.24)
    for offset in (2.55, -2.55):
        point = (end[0] + head * math.cos(angle + offset), end[1] + head * math.sin(angle + offset))
        draw.line((end, point), fill=(220, 30, 30), width=width)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_nodes(canvas: Image.Image, doc: dict, nodes: dict[str, np.ndarray], kept: set[str], *, transparent: bool) -> None:
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    reference = _reference_name(doc)
    symbols = _symbol_map(doc)
    radius = max(11, round(min(width, height) * 0.035))
    font = _font(max(12, round(radius * 1.2)))
    for name, position in nodes.items():
        if name not in kept:
            continue
        x, y = float(position[0]) * width, float(position[1]) * height
        fill = (30, 30, 30) if name == reference else (245, 245, 245)
        text_fill = (255, 255, 255) if name == reference else (20, 20, 20)
        outline = (255, 255, 255) if transparent else (20, 20, 20)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        glyph = symbols[name]
        box = draw.textbbox((0, 0), glyph, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]), glyph, fill=text_fill, font=font)


def _render(doc: dict) -> Image.Image:
    representation, perturbation, level = _condition_parts(str(doc["experiment_condition"]))
    image = kubric._load_image(doc.get("image")) or kubric._load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image for {doc.get('qid', 'unknown')}")
    if representation == "rgb_overlay":
        canvas = image.copy()
        transparent = True
    else:
        # No title, legend, names, direction words, axes, or natural-image data.
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        transparent = False
    nodes = _display_nodes(doc, perturbation, level)
    kept = _kept_names(doc, perturbation, level)
    _draw_nodes(canvas, doc, nodes, kept, transparent=transparent)
    reference = _reference_name(doc)
    if reference in nodes:
        vector = _heading_screen_vector(doc)
        if perturbation == "orientation":
            vector = _rotate(vector, level)
        center = (float(nodes[reference][0]) * canvas.width, float(nodes[reference][1]) * canvas.height)
        _draw_arrow(ImageDraw.Draw(canvas), center, vector, max(22.0, min(canvas.size) * 0.14), max(2, round(min(canvas.size) / 110)))
    return canvas


def _debug_enabled() -> bool:
    return str(os.getenv("KUBRIC_RELATIVE_REPRESENTATION_DEBUG", "0")).lower() in {"1", "true", "yes", "on"}


def process_docs(dataset: Dataset) -> Dataset:
    """Create matched visual variants for authored object-relative pairs only."""
    normalized = base.relative_direction_process_docs(dataset)
    conditions = _selected_conditions()
    rows = []
    for source in normalized:
        doc = dict(source)
        if doc.get("source_task_family") != "object_relative_direction":
            continue
        if not _objects(doc) or _reference_name(doc) not in _symbol_map(doc):
            continue
        for condition in conditions:
            item = dict(doc)
            qid = f"{doc['qid']}::representation_noise::{condition}"
            representation, noise_family, noise_level = _condition_parts(condition)
            item.update({
                "qid": qid,
                "index": qid,
                "experiment_condition": condition,
                "representation": representation,
                "noise_family": noise_family,
                "noise_level": noise_level,
                "diagnostic_experiment": "rgb_overlay_vs_symbolic_noise",
            })
            rows.append(item)
    eval_logger.info(
        "MOVi-A representation-noise task loaded {} records ({} source pairs); conditions={}",
        len(rows), len({row["source_qid"] for row in rows}), Counter(row["experiment_condition"] for row in rows),
    )
    return Dataset.from_list(rows)


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
    options = kubric._get_options(doc)
    return (
        "Answer the spatial-reasoning question using the displayed visual representation. "
        f"Object-symbol legend: {_symbol_legend(doc)}. "
        "The red arrow is the reference object's heading. Select one answer option and respond with its letter only.\n"
        f"Question: {doc['question']}\nOptions:\n"
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
        "relation": doc["diagnostic_relation"], "reference_object": _reference_name(doc),
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
            "count": len(items),
            "accuracy": _mean(float(item["score"]) for item in items),
            "parse_success": _mean(float(item["parse_success"]) for item in items),
            "direction_accuracy": _mean(float(item["score"]) for item in items if item["answer_format"] == "direction"),
            "object_accuracy": _mean(float(item["score"]) for item in items if item["answer_format"] == "object"),
        }
        for condition, items in sorted(grouped.items())
    }


def aggregate_accuracy(results):
    summary = _condition_summary(results)
    eval_logger.info("MOVi-A representation-noise results by condition: {}", summary)
    return _mean(float(row["score"]) for row in results)


def aggregate_parse_success_rate(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(
        f"kubric_movi_a_direction_object_relative_direction_representation_noise_{model}.json", args
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "MOVi-A relative direction", "task": "kubric_movi_a_direction_object_relative_direction_representation_noise",
            "num_records": len(results), "condition_summary": _condition_summary(results), "records": results,
        }, handle, indent=2)
    eval_logger.info("MOVi-A representation-noise records saved to {}", path)
