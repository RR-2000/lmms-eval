"""Composable diagnostics for the visual primitives used by COMFORT GT_HELP.

The tasks deliberately isolate localization, orientation, abstract-symbol
grounding, and direction-vector reasoning.  All image coordinates exposed to a
model use the same [0, 1000] frame used by the COMFORT localization task.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_multi_3d_bbox_prediction.utils import compute_iou, parse_bbox
from lmms_eval.tasks.comfort_multi_3d_object_basis.utils import object_basis, parse_vector
from lmms_eval.utils import sanitize_model_name


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Multi_3D")
COORDINATE_MAX = 1000.0
AXES = ("front", "up", "right")
HORIZONTAL_DIRECTIONS = ("left", "right", "front", "back")
SOURCE_DIRECTIONS = ("left", "right", "front", "behind")
SOURCE_TO_ANSWER = {"left": "left", "right": "right", "front": "front", "behind": "back"}
COMPASS_DIRECTIONS = (
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
    "up",
    "up-right",
)
SYMBOLS = ("A", "B", "C", "D")
SYMBOL_FILL = (255, 223, 65)
SYMBOL_OUTLINE = (20, 20, 20)
BOX_COLOR = (238, 50, 230)
ARROW_COLORS = {
    "front": (0, 235, 255),
    "back": (255, 70, 70),
    "left": (90, 245, 90),
    "right": (255, 180, 30),
    "up": (190, 110, 255),
}
EPSILON = 1e-8
LABEL_ALIASES = {
    "horsel": "horse",
    "horser": "horse",
    "bicycle mountain": "bicycle",
    "car sedan": "car",
}

# These labels are copied into every submission row so analysis can compare
# producing a scaffold with consuming the same scaffold.  "generation" means
# emitting the representation; "utilization" means receiving it and decoding
# or applying it to another question.
COMPONENT_ROLES = {
    "comfort_gt_component_bbox_prediction": ("bbox", "generation"),
    "comfort_gt_component_bbox_naming": ("bbox", "utilization"),
    "comfort_gt_component_facing_direction": ("orientation_label", "generation"),
    "comfort_gt_component_front_arrow": ("front_arrow", "generation"),
    "comfort_gt_component_front_arrow_reading": ("front_arrow", "utilization"),
    "comfort_gt_component_left_arrow": ("left_arrow", "generation"),
    "comfort_gt_component_left_arrow_reading": ("left_arrow", "utilization"),
    "comfort_gt_component_symbol_to_object": ("abstract_symbol", "utilization"),
    "comfort_gt_component_object_to_symbol": ("abstract_symbol", "generation"),
    "comfort_gt_component_long_arrow_to_symbol": ("long_direction_arrows", "utilization"),
    "comfort_gt_component_short_arrow_to_symbol": ("short_direction_arrows", "utilization"),
    "comfort_gt_component_vector_to_direction": ("direction_vector", "utilization"),
    "comfort_gt_component_direction_to_vector": ("direction_vector", "generation"),
    "comfort_gt_component_projected_axes_prediction": ("projected_axes", "generation"),
    "comfort_gt_component_text_axes_direction": ("projected_axes_text", "utilization"),
    "comfort_gt_component_overlay_axes_direction": ("projected_axes_overlay", "utilization"),
}


def _display_label(value) -> str:
    """Return the human-facing class name used in prompts and evaluation."""
    label = " ".join(str(value or "").strip().lower().split())
    return LABEL_ALIASES.get(label, label)


def _image_path(scene: dict) -> Path:
    value = Path(str(scene.get("image", "")))
    return value if value.is_absolute() else DATA_ROOT / value


def _bbox_1000(obj: dict) -> list[float]:
    bbox = obj.get("bbox_2d_normalized_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("missing normalized xyxy bbox")
    values = [float(value) * COORDINATE_MAX for value in bbox]
    if not all(0.0 <= value <= COORDINATE_MAX for value in values):
        raise ValueError("bbox lies outside the normalized image")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox is empty")
    return values


def _position(obj: dict) -> list[float]:
    value = (obj.get("camera_frame") or {}).get("position")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("missing camera-frame position")
    return [float(component) for component in value]


def _scene_objects(scene: dict) -> tuple[dict, dict[str, dict], list[dict]]:
    objects = [obj for obj in (scene.get("objects") or []) if isinstance(obj, dict)]
    reference = next((obj for obj in objects if obj.get("role") == "reference"), None)
    targets = {
        str(obj.get("reference_direction")): obj
        for obj in objects
        if obj.get("role") == "target" and obj.get("reference_direction")
    }
    if reference is None or not set(SOURCE_DIRECTIONS) <= set(targets):
        raise ValueError("scene does not contain a reference and four directional targets")
    return reference, targets, objects


def _unit_xy(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(*vector)
    if norm <= EPSILON:
        raise ValueError("zero screen direction")
    return vector[0] / norm, vector[1] / norm


def _project_camera_vector(reference: dict, vector: list[float]) -> tuple[float, float]:
    """Perspective-project a camera-frame tangent at the reference into x/y."""
    x, y, z = _position(reference)
    vx, vy, vz = [float(value) for value in vector]
    if abs(y) <= EPSILON:
        raise ValueError("reference lies on the camera plane")
    du = (vx * y - x * vy) / (y * y)
    dv = -(vz * y - z * vy) / (y * y)
    if math.hypot(du, dv) <= EPSILON:
        return (0.0, -1.0 if vy >= 0.0 else 1.0)
    return _unit_xy((du, dv))


def _screen_direction(reference: dict, target: dict) -> tuple[float, float]:
    """Perspective-project a reference-to-target tangent into image x/y."""
    target_position = _position(target)
    reference_position = _position(reference)
    displacement = [target_position[index] - reference_position[index] for index in range(3)]
    return _project_camera_vector(reference, displacement)


def _direction_vectors(reference: dict, targets: dict[str, dict]) -> dict[str, list[float]]:
    return {
        SOURCE_TO_ANSWER[source]: list(_screen_direction(reference, targets[source]))
        for source in SOURCE_DIRECTIONS
    }


def _compass_label(vector: list[float]) -> str:
    # Image y increases downward; sector zero points image-right.
    angle = math.atan2(float(vector[1]), float(vector[0]))
    index = int(round(angle / (math.pi / 4.0))) % 8
    return COMPASS_DIRECTIONS[index]


def _symbol_map(scene_id: str) -> dict[str, str]:
    offset = int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:8], 16) % 4
    symbols = SYMBOLS[offset:] + SYMBOLS[:offset]
    return {source: symbols[index] for index, source in enumerate(SOURCE_DIRECTIONS)}


def _render_objects(objects: list[dict]) -> list[dict]:
    return [
        {
            "object_id": str(obj.get("object_id", "")),
            "label": _display_label(obj.get("label")),
            "role": str(obj.get("role", "")),
            "reference_direction": obj.get("reference_direction"),
            "bbox": _bbox_1000(obj),
        }
        for obj in objects
    ]


def _base_scenes(dataset: Dataset):
    skipped = Counter()
    for source in dataset:
        scene = dict(source)
        scene_id = str(scene.get("scene_id", "")).strip()
        image_path = _image_path(scene)
        try:
            reference, targets, objects = _scene_objects(scene)
            basis = object_basis(scene)
            rendered = _render_objects(objects)
            screen = _direction_vectors(reference, targets)
            basis_screen = {axis: list(_project_camera_vector(reference, basis[axis])) for axis in AXES}
            basis_screen["left"] = [-value for value in basis_screen["right"]]
        except (TypeError, ValueError) as error:
            skipped[type(error).__name__] += 1
            continue
        if not scene_id or not image_path.is_file():
            skipped["missing_scene_or_image"] += 1
            continue
        labels = [_display_label(obj.get("label")) for obj in objects]
        if not all(labels):
            skipped["missing_label"] += 1
            continue
        yield {
            "scene": scene,
            "scene_id": scene_id,
            "img_path": str(image_path),
            "reference": reference,
            "targets": targets,
            "objects": objects,
            "render_objects": rendered,
            "basis": basis,
            "screen_directions": screen,
            "basis_screen_directions": basis_screen,
            "reference_label": _display_label(reference.get("label")),
            "candidate_objects": list(dict.fromkeys(labels)),
            "label_counts": dict(Counter(labels)),
            "symbol_map": _symbol_map(scene_id),
        }
    if skipped:
        eval_logger.warning("COMFORT component preprocessing skipped rows: {}", dict(skipped))


def _record(base: dict, task: str, qid_suffix: str, **values) -> dict:
    component_name, component_role = COMPONENT_ROLES.get(task, ("unclassified", "unclassified"))
    return {
        "qid": f"{base['scene_id']}::{qid_suffix}",
        "index": f"{base['scene_id']}::{qid_suffix}",
        "diagnostic_task": task,
        "component_name": component_name,
        "component_role": component_role,
        "scene_id": base["scene_id"],
        "img_path": base["img_path"],
        "reference_object": base["reference_label"],
        "candidate_objects": base["candidate_objects"],
        "label_counts": base["label_counts"],
        "render_objects": base["render_objects"],
        "screen_directions": base["screen_directions"],
        "basis_screen_directions": base["basis_screen_directions"],
        "symbol_map": base["symbol_map"],
        **values,
    }


def _finish(records: list[dict], name: str) -> Dataset:
    eval_logger.info(
        "COMFORT component {} loaded {} rows from {} scenes.",
        name,
        len(records),
        len({row["scene_id"] for row in records}),
    )
    return Dataset.from_list(records)


def process_bbox_prediction_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_bbox_prediction"
    records = []
    for base in _base_scenes(dataset):
        for obj in base["objects"]:
            object_id = str(obj.get("object_id", ""))
            label = _display_label(obj.get("label"))
            # A plain-image name cannot identify one of two instances that now
            # share a normalized class label (for example HorseL and HorseR).
            if base["label_counts"][label] > 1:
                continue
            records.append(_record(
                base, task, f"bbox::{object_id}", visual_mode="plain",
                object_id=object_id, object_label=label, object_role=obj.get("role"),
                gt_bbox=_bbox_1000(obj),
                prompt=(f"Locate the {label} in the plain image. Return exactly one bounding box "
                        "as [x_min, y_min, x_max, y_max] in [0,1000], where (0,0) is top-left."),
            ))
    return _finish(records, task)


def process_bbox_naming_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_bbox_naming"
    records = []
    for base in _base_scenes(dataset):
        for obj in base["objects"]:
            object_id = str(obj.get("object_id", ""))
            label = _display_label(obj.get("label"))
            records.append(_record(
                base, task, f"bbox_name::{object_id}", visual_mode="boxed_object",
                overlay_object_id=object_id, gold_answer=label,
                answer_choices=base["candidate_objects"],
                prompt=("A magenta box labeled BOX is overlaid on one object. What object is inside it? "
                        f"Return exactly one name from: {', '.join(base['candidate_objects'])}."),
            ))
    return _finish(records, task)


def process_facing_direction_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_facing_direction"
    records = []
    for base in _base_scenes(dataset):
        gold = _compass_label(base["basis_screen_directions"]["front"])
        records.append(_record(
            base, task, "facing", visual_mode="boxed_reference", gold_answer=gold,
            answer_choices=list(COMPASS_DIRECTIONS),
            prompt=(f"The reference {base['reference_label']} is boxed. In which image-plane direction "
                    "is its own front facing? Return exactly one of: " + ", ".join(COMPASS_DIRECTIONS) + "."),
        ))
    return _finish(records, task)


def _process_arrow_docs(dataset: Dataset, direction: str) -> Dataset:
    task = f"comfort_gt_component_{direction}_arrow"
    records = []
    for base in _base_scenes(dataset):
        reference_render = next(obj for obj in base["render_objects"] if obj["role"] == "reference")
        x1, y1, x2, y2 = reference_render["bbox"]
        start = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        unit = base["basis_screen_directions"][direction]
        raw_end = [start[0] + 180.0 * unit[0], start[1] + 180.0 * unit[1]]
        end = [max(0.0, min(1000.0, value)) for value in raw_end]
        scored_unit = list(_unit_xy((end[0] - start[0], end[1] - start[1])))
        records.append(_record(
            base, task, f"{direction}_arrow", visual_mode="boxed_reference",
            arrow_axis=direction, gt_arrow_start=start, gt_arrow_end=end,
            gt_arrow_direction=scored_unit,
            prompt=(f"The reference {base['reference_label']} is boxed. Predict a 2D arrow pointing "
                    f"toward the object's own {direction}. Start at the object's center. Return only "
                    '{"start": [x, y], "end": [x, y]} with coordinates in [0,1000].'),
        ))
    return _finish(records, task)


def process_front_arrow_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_docs(dataset, "front")


def process_left_arrow_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_docs(dataset, "left")


def _process_arrow_reading_docs(dataset: Dataset, direction: str) -> Dataset:
    """Supply a gold overlay and ask the model to read its screen direction."""
    task = f"comfort_gt_component_{direction}_arrow_reading"
    records = []
    for base in _base_scenes(dataset):
        gold = _compass_label(base["basis_screen_directions"][direction])
        records.append(_record(
            base,
            task,
            f"{direction}_arrow_reading",
            visual_mode="single_axis_arrow",
            overlay_axis=direction,
            gold_answer=gold,
            answer_choices=list(COMPASS_DIRECTIONS),
            prompt=(
                f"The reference {base['reference_label']} is boxed. The overlaid arrow shows the "
                f"object's ground-truth {direction} axis. In which image-plane direction does the "
                "arrow point? Return exactly one of: " + ", ".join(COMPASS_DIRECTIONS) + "."
            ),
        ))
    return _finish(records, task)


def process_front_arrow_reading_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_reading_docs(dataset, "front")


def process_left_arrow_reading_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_reading_docs(dataset, "left")


def process_symbol_to_object_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_symbol_to_object"
    records = []
    for base in _base_scenes(dataset):
        for source in SOURCE_DIRECTIONS:
            target = base["targets"][source]
            symbol = base["symbol_map"][source]
            records.append(_record(
                base, task, f"symbol_to_object::{source}", visual_mode="symbols",
                query_symbol=symbol, gold_answer=_display_label(target.get("label")),
                answer_choices=base["candidate_objects"],
                prompt=(f"Each surrounding object has an abstract letter marker. Which object is marked {symbol}? "
                        f"Return exactly one name from: {', '.join(base['candidate_objects'])}."),
            ))
    return _finish(records, task)


def process_object_to_symbol_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_object_to_symbol"
    records = []
    for base in _base_scenes(dataset):
        for source in SOURCE_DIRECTIONS:
            target = base["targets"][source]
            symbol = base["symbol_map"][source]
            target_label = _display_label(target.get("label"))
            if base["label_counts"][target_label] > 1:
                continue
            records.append(_record(
                base, task, f"object_to_symbol::{source}", visual_mode="symbols",
                query_object=target_label, gold_answer=symbol,
                answer_choices=list(SYMBOLS),
                prompt=(f"Each surrounding object has an abstract letter marker. Which marker is on the "
                        f"{target_label}? Return exactly one of: {', '.join(SYMBOLS)}."),
            ))
    return _finish(records, task)


def _process_arrow_to_symbol_docs(dataset: Dataset, short: bool) -> Dataset:
    length_name = "short" if short else "long"
    task = f"comfort_gt_component_{length_name}_arrow_to_symbol"
    records = []
    for base in _base_scenes(dataset):
        for source in SOURCE_DIRECTIONS:
            answer_direction = SOURCE_TO_ANSWER[source]
            records.append(_record(
                base, task, f"{length_name}_arrow_symbol::{source}",
                visual_mode=f"symbols_{length_name}_arrows", gold_answer=base["symbol_map"][source],
                answer_choices=list(SYMBOLS), query_direction=answer_direction,
                prompt=(f"The reference object's labeled arrows show its own directions and the letters mark "
                        f"surrounding objects. Which letter marks the object {answer_direction} of the reference? "
                        f"Return exactly one of: {', '.join(SYMBOLS)}."),
            ))
    return _finish(records, task)


def process_long_arrow_to_symbol_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_to_symbol_docs(dataset, False)


def process_short_arrow_to_symbol_docs(dataset: Dataset) -> Dataset:
    return _process_arrow_to_symbol_docs(dataset, True)


CANONICAL_VECTORS = {
    "front": {"front": 1.0, "up": 0.0, "right": 0.0},
    "back": {"front": -1.0, "up": 0.0, "right": 0.0},
    "above": {"front": 0.0, "up": 1.0, "right": 0.0},
    "below": {"front": 0.0, "up": -1.0, "right": 0.0},
    "right": {"front": 0.0, "up": 0.0, "right": 1.0},
    "left": {"front": 0.0, "up": 0.0, "right": -1.0},
}


def process_vector_to_direction_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_vector_to_direction"
    records = []
    for base in _base_scenes(dataset):
        for answer, vector in CANONICAL_VECTORS.items():
            records.append(_record(
                base, task, f"vector_to_direction::{answer}", visual_mode="plain",
                gt_direction=answer, gold_answer=answer, answer_choices=list(CANONICAL_VECTORS),
                input_vector=vector,
                prompt=("In the reference object's coordinates, positive axes are front, up, and right. "
                        f"What direction is the vector {json.dumps(vector)}? Return exactly one of: "
                        + ", ".join(CANONICAL_VECTORS) + "."),
            ))
    return _finish(records, task)


def process_direction_to_vector_docs(dataset: Dataset) -> Dataset:
    task = "comfort_gt_component_direction_to_vector"
    records = []
    for base in _base_scenes(dataset):
        for direction, vector in CANONICAL_VECTORS.items():
            records.append(_record(
                base, task, f"direction_to_vector::{direction}", visual_mode="plain",
                query_direction=direction, gt_vector=vector,
                prompt=("In the reference object's coordinates, positive axes are front, up, and right. "
                        f"Give the unit vector for {direction}. Return only JSON as "
                        '{"front": <float>, "up": <float>, "right": <float>}.'),
            ))
    return _finish(records, task)


def process_projected_axes_prediction_docs(dataset: Dataset) -> Dataset:
    """Generate the three 2D axes later consumed by text/overlay tasks."""
    task = "comfort_gt_component_projected_axes_prediction"
    records = []
    for base in _base_scenes(dataset):
        target = {
            axis: [float(value) for value in base["basis_screen_directions"][axis]]
            for axis in AXES
        }
        records.append(_record(
            base,
            task,
            "projected_axes_prediction",
            visual_mode="boxed_reference",
            gt_basis_screen=target,
            prompt=(
                f"The reference {base['reference_label']} is boxed. Predict its screen-projected "
                "front, up, and right unit directions in image coordinates, where x points right "
                "and y points down. Return only JSON as "
                '{"front": [dx, dy], "up": [dx, dy], "right": [dx, dy]}.'
            ),
        ))
    return _finish(records, task)


def _process_axes_direction_docs(dataset: Dataset, overlay: bool) -> Dataset:
    cue = "overlay" if overlay else "text"
    task = f"comfort_gt_component_{cue}_axes_direction"
    records = []
    for base in _base_scenes(dataset):
        front = base["basis_screen_directions"]["front"]
        up = base["basis_screen_directions"]["up"]
        right = base["basis_screen_directions"]["right"]
        for source in SOURCE_DIRECTIONS:
            target = base["targets"][source]
            target_label = _display_label(target.get("label"))
            if base["label_counts"][target_label] > 1:
                continue
            if overlay:
                cue_text = "The image overlays arrows labeled front, up, and right on the reference object."
                visual_mode = "axes_overlay"
            else:
                cue_text = (
                    "In image coordinates (x right, y down), the reference's screen-projected unit axes are "
                    f"front=({front[0]:.3f},{front[1]:.3f}), up=({up[0]:.3f},{up[1]:.3f}), "
                    f"and right=({right[0]:.3f},{right[1]:.3f})."
                )
                visual_mode = "boxed_reference"
            records.append(_record(
                base, task, f"{cue}_axes::{source}", visual_mode=visual_mode,
                query_object=target_label, gold_answer=SOURCE_TO_ANSWER[source],
                answer_choices=list(HORIZONTAL_DIRECTIONS),
                prompt=(f"{cue_text} Relative to the reference {base['reference_label']}, where is the "
                        f"{target_label}? Return exactly one of: {', '.join(HORIZONTAL_DIRECTIONS)}."),
            ))
    return _finish(records, task)


def process_text_axes_direction_docs(dataset: Dataset) -> Dataset:
    return _process_axes_direction_docs(dataset, False)


def process_overlay_axes_direction_docs(dataset: Dataset) -> Dataset:
    return _process_axes_direction_docs(dataset, True)


def _bbox_pixels(bbox: list[float], image: Image.Image) -> tuple[int, int, int, int]:
    return tuple(round(value / COORDINATE_MAX * size) for value, size in zip(bbox, (image.width, image.height, image.width, image.height)))


def _object_by_id(doc: dict, object_id: str) -> dict:
    return next(obj for obj in doc["render_objects"] if str(obj["object_id"]) == str(object_id))


def _reference_render(doc: dict) -> dict:
    return next(obj for obj in doc["render_objects"] if obj["role"] == "reference")


def _draw_box(image: Image.Image, obj: dict, label: str = "BOX") -> None:
    draw = ImageDraw.Draw(image)
    box = _bbox_pixels(obj["bbox"], image)
    width = max(3, round(min(image.size) / 180))
    draw.rectangle(box, outline=BOX_COLOR, width=width)
    draw.rectangle((box[0], max(0, box[1] - 16), box[0] + 30, box[1]), fill=BOX_COLOR)
    draw.text((box[0] + 2, max(0, box[1] - 15)), label, fill=(0, 0, 0))


def _draw_arrow(image: Image.Image, start: tuple[float, float], direction: list[float], length: float, label: str) -> None:
    draw = ImageDraw.Draw(image)
    dx, dy = _unit_xy((float(direction[0]), float(direction[1])))
    end = (
        max(5.0, min(image.width - 6.0, start[0] + dx * length)),
        max(5.0, min(image.height - 6.0, start[1] + dy * length)),
    )
    color = ARROW_COLORS[label]
    width = max(3, round(min(image.size) / 180))
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(10.0, width * 3.0)
    draw.polygon([
        end,
        (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6)),
        (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6)),
    ], fill=color)
    draw.text((round(end[0] + 3 * dx), round(end[1] + 3 * dy)), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def _draw_symbols(image: Image.Image, doc: dict) -> None:
    draw = ImageDraw.Draw(image)
    for source in SOURCE_DIRECTIONS:
        obj = next(item for item in doc["render_objects"] if item.get("reference_direction") == source)
        x1, y1, x2, y2 = _bbox_pixels(obj["bbox"], image)
        x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        radius = max(12, round(min(image.size) / 42))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=SYMBOL_FILL, outline=SYMBOL_OUTLINE, width=3)
        symbol = doc["symbol_map"][source]
        text_box = draw.textbbox((0, 0), symbol)
        draw.text((x - (text_box[2] - text_box[0]) / 2, y - (text_box[3] - text_box[1]) / 2), symbol, fill=(0, 0, 0), stroke_width=1)


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT component image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    mode = doc.get("visual_mode", "plain")
    if mode == "plain":
        return [image]
    if mode == "boxed_object":
        _draw_box(image, _object_by_id(doc, doc["overlay_object_id"]))
    elif mode == "boxed_reference":
        _draw_box(image, _reference_render(doc), "REF")
    elif mode.startswith("symbols"):
        _draw_symbols(image, doc)
        if mode.endswith("_arrows"):
            reference = _reference_render(doc)
            x1, y1, x2, y2 = _bbox_pixels(reference["bbox"], image)
            start = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            scale = 0.45 if "short" in mode else 1.15
            minimum = 18.0 if "short" in mode else 38.0
            length = max(minimum, scale * math.hypot(x2 - x1, y2 - y1))
            for direction in HORIZONTAL_DIRECTIONS:
                _draw_arrow(image, start, doc["screen_directions"][direction], length, direction)
    elif mode == "axes_overlay":
        reference = _reference_render(doc)
        _draw_box(image, reference, "REF")
        x1, y1, x2, y2 = _bbox_pixels(reference["bbox"], image)
        start = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        length = max(30.0, 0.8 * math.hypot(x2 - x1, y2 - y1))
        for direction in ("front", "up", "right"):
            _draw_arrow(image, start, doc["basis_screen_directions"][direction], length, direction)
    elif mode == "single_axis_arrow":
        reference = _reference_render(doc)
        _draw_box(image, reference, "REF")
        x1, y1, x2, y2 = _bbox_pixels(reference["bbox"], image)
        start = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        length = max(30.0, 0.8 * math.hypot(x2 - x1, y2 - y1))
        axis = str(doc["overlay_axis"])
        _draw_arrow(image, start, doc["basis_screen_directions"][axis], length, axis)
    else:
        raise ValueError(f"Unknown visual mode {mode!r}")
    return [image]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc['prompt']}{kwargs.get('post_prompt', '')}"


def doc_to_target(doc):
    if "gt_bbox" in doc:
        return json.dumps([round(float(value), 3) for value in doc["gt_bbox"]])
    if "gt_arrow_start" in doc:
        return json.dumps({"start": doc["gt_arrow_start"], "end": doc["gt_arrow_end"]})
    if "gt_vector" in doc:
        return json.dumps(doc["gt_vector"])
    if "gt_basis_screen" in doc:
        return json.dumps(doc["gt_basis_screen"])
    return str(doc.get("gold_answer", ""))


def _normal(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_choice(text: str, choices: list[str]) -> Optional[str]:
    normalized = {_normal(choice): str(choice) for choice in choices}
    answer = _normal(text)
    if answer in normalized:
        return normalized[answer]
    matches = [(key, original) for key, original in normalized.items() if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", answer)]
    if not matches:
        return None
    longest = max(len(key) for key, _ in matches)
    best = [original for key, original in matches if len(key) == longest]
    return best[0] if len(best) == 1 else None


def _extract_json(text: str):
    candidates = [str(text).strip()]
    match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            pass
    return None


def _point(value) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(v) and 0.0 <= v <= COORDINATE_MAX for v in point) else None


def parse_arrow(text: str) -> Optional[dict[str, list[float]]]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return None
    start, end = _point(payload.get("start")), _point(payload.get("end"))
    if start is None or end is None or math.dist(start, end) <= EPSILON:
        return None
    return {"start": start, "end": end}


def parse_projected_axes(text: str) -> Optional[dict[str, list[float]]]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return None
    parsed = {}
    for axis in AXES:
        value = payload.get(axis)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            vector = [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(component) for component in vector) or math.hypot(*vector) <= EPSILON:
            return None
        parsed[axis] = vector
    return parsed


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def process_bbox_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = parse_bbox(prediction)
    iou = compute_iou(doc["gt_bbox"], parsed) if parsed is not None else 0.0
    raw_accuracy = float(iou >= 0.5)
    entry = _common_entry(doc, prediction, parsed_answer=parsed, parse_success=float(parsed is not None),
                          bbox_iou=iou, bbox_acc_0_5=raw_accuracy, raw_accuracy=raw_accuracy)
    return _outputs(entry, ("component_bbox_iou", "component_bbox_acc_0_5", "component_raw_accuracy", "component_parse_success"), doc)


def process_classification_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = _parse_choice(prediction, list(doc["answer_choices"]))
    accuracy = float(_normal(parsed) == _normal(doc["gold_answer"]))
    entry = _common_entry(doc, prediction, parsed_answer=parsed, parse_success=float(parsed is not None),
                          accuracy=accuracy, raw_accuracy=accuracy)
    return _outputs(entry, ("component_accuracy", "component_raw_accuracy", "component_parse_success"), doc)


def process_arrow_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = parse_arrow(prediction)
    cosine = 0.0
    start_score = 0.0
    if parsed is not None:
        vector = [parsed["end"][0] - parsed["start"][0], parsed["end"][1] - parsed["start"][1]]
        unit = _unit_xy((vector[0], vector[1]))
        cosine = unit[0] * doc["gt_arrow_direction"][0] + unit[1] * doc["gt_arrow_direction"][1]
        start_score = max(0.0, 1.0 - math.dist(parsed["start"], doc["gt_arrow_start"]) / math.hypot(1000.0, 1000.0))
    angle_accuracy = float(parsed is not None and cosine >= math.cos(math.radians(30.0)))
    entry = _common_entry(
        doc, prediction, parsed_answer=parsed, parse_success=float(parsed is not None),
        arrow_cosine=cosine, arrow_angle_30_accuracy=angle_accuracy,
        arrow_start_score=start_score, raw_accuracy=angle_accuracy,
    )
    return _outputs(entry, ("component_arrow_cosine", "component_arrow_angle_30_accuracy", "component_arrow_start_score", "component_raw_accuracy", "component_parse_success"), doc)


def _vector_cosine(predicted: dict, target: dict) -> float:
    pred_norm = math.sqrt(sum(float(predicted[key]) ** 2 for key in AXES))
    target_norm = math.sqrt(sum(float(target[key]) ** 2 for key in AXES))
    return sum(float(predicted[key]) * float(target[key]) for key in AXES) / (pred_norm * target_norm)


def process_vector_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = parse_vector(prediction)
    cosine = _vector_cosine(parsed, doc["gt_vector"]) if parsed is not None else 0.0
    signs = float(parsed is not None and all((float(parsed[key]) > 1e-4) - (float(parsed[key]) < -1e-4) == (float(doc["gt_vector"][key]) > 1e-4) - (float(doc["gt_vector"][key]) < -1e-4) for key in AXES))
    angle_accuracy = float(parsed is not None and cosine >= math.cos(math.radians(30.0)))
    entry = _common_entry(
        doc, prediction, parsed_answer=parsed, parse_success=float(parsed is not None),
        vector_cosine=cosine, vector_angle_30_accuracy=angle_accuracy,
        vector_full_sign_accuracy=signs, raw_accuracy=angle_accuracy,
    )
    return _outputs(entry, ("component_vector_cosine", "component_vector_angle_30_accuracy", "component_vector_full_sign_accuracy", "component_raw_accuracy", "component_parse_success"), doc)


def process_projected_axes_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = parse_projected_axes(prediction)
    cosines = {axis: 0.0 for axis in AXES}
    if parsed is not None:
        for axis in AXES:
            predicted = _unit_xy(tuple(parsed[axis]))
            target = _unit_xy(tuple(doc["gt_basis_screen"][axis]))
            cosines[axis] = predicted[0] * target[0] + predicted[1] * target[1]
    mean_cosine = _mean(cosines.values())
    all_accurate = float(parsed is not None and all(value >= math.cos(math.radians(30.0)) for value in cosines.values()))
    entry = _common_entry(
        doc,
        prediction,
        parsed_answer=parsed,
        parse_success=float(parsed is not None),
        basis_mean_cosine=mean_cosine,
        basis_all_angle_30_accuracy=all_accurate,
        front_axis_cosine=cosines["front"],
        up_axis_cosine=cosines["up"],
        right_axis_cosine=cosines["right"],
        raw_accuracy=all_accurate,
    )
    return _outputs(
        entry,
        (
            "component_basis_mean_cosine",
            "component_basis_all_angle_30_accuracy",
            "component_raw_accuracy",
            "component_parse_success",
        ),
        doc,
    )


def _common_entry(doc: dict, prediction: str, **scores) -> dict:
    keys = ("qid", "scene_id", "diagnostic_task", "component_name", "component_role", "reference_object", "object_id", "object_label", "query_object", "query_symbol", "query_direction", "gold_answer", "gt_bbox", "gt_arrow_start", "gt_arrow_end", "gt_arrow_direction", "gt_vector", "gt_basis_screen", "input_vector", "visual_mode", "overlay_axis")
    return {**{key: doc.get(key) for key in keys if key in doc}, "prediction": prediction, **scores}


def _outputs(entry: dict, metrics: tuple[str, ...], doc: dict) -> dict:
    output = {metric: dict(entry) for metric in metrics}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "target": doc_to_target(doc),
        "img_path": doc.get("img_path"),
    }
    return output


def _aggregate(results, field: str) -> float:
    return _mean(float(row.get(field, 0.0)) for row in results)


def aggregate_accuracy(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row.get("gold_answer"))].append(float(row.get("accuracy", 0.0)))
    score = _aggregate(results, "accuracy")
    eval_logger.info("COMFORT component accuracy {:.4f}; by answer={}", score, {key: _mean(value) for key, value in sorted(grouped.items())})
    return score


def aggregate_parse_success(results):
    return _aggregate(results, "parse_success")


def aggregate_raw_accuracy(results):
    """Comparable binary accuracy shared by every component task."""
    return _aggregate(results, "raw_accuracy")


def aggregate_bbox_iou(results):
    return _aggregate(results, "bbox_iou")


def aggregate_bbox_acc_0_5(results):
    return _aggregate(results, "bbox_acc_0_5")


def aggregate_arrow_cosine(results):
    return _aggregate(results, "arrow_cosine")


def aggregate_arrow_angle_30_accuracy(results):
    return _aggregate(results, "arrow_angle_30_accuracy")


def aggregate_arrow_start_score(results):
    return _aggregate(results, "arrow_start_score")


def aggregate_vector_cosine(results):
    return _aggregate(results, "vector_cosine")


def aggregate_vector_angle_30_accuracy(results):
    return _aggregate(results, "vector_angle_30_accuracy")


def aggregate_vector_full_sign_accuracy(results):
    return _aggregate(results, "vector_full_sign_accuracy")


def aggregate_basis_mean_cosine(results):
    return _aggregate(results, "basis_mean_cosine")


def aggregate_basis_all_angle_30_accuracy(results):
    return _aggregate(results, "basis_all_angle_30_accuracy")


def aggregate_results_for_submission(results, args):
    task = str(results[0].get("diagnostic_task", "comfort_gt_component")) if results else "comfort_gt_component"
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"{task}_{model}.json", args)
    numeric_fields = sorted({key for row in results for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": task,
        "num_records": len(results),
        "metrics": {field: _aggregate(results, field) for field in numeric_fields},
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT component submission saved to {}", path)
