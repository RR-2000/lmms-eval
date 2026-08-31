"""Composable visual and textual GT aids for COMFORT_Multi_3D.

Set ``GT_HELP`` to 0..35.  The preset registry near the bottom is deliberately
made from ordinary functions: experiments can combine aids by putting several
draw/describe functions in one preset without changing the task entry points.
"""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont

from lmms_eval.tasks.comfort_direction_object import utils as base
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Multi_3D")
SCENES_PATH = DATA_ROOT / "scenes.jsonl"
VALID_GT_HELP = frozenset(str(value) for value in range(36))
DEBUG_SAVE_DEFAULT_DIR = Path("outputs/comfort_direction_object_gt_help_debug_images")

REFERENCE_COLOR = (255, 64, 64)
ANSWER_COLOR = (48, 220, 96)
FRONT_ARROW_COLOR = (0, 220, 255)
DIRECTION_ARROW_COLORS = {
    "left": (255, 170, 32),
    "right": (64, 160, 255),
    "front": (48, 220, 96),
    "behind": (210, 80, 255),
}
TOP_DOWN_COLORS = {
    "reference": ((245, 200, 40), "yellow"),
    "left": ((225, 70, 70), "red"),
    "right": ((65, 125, 220), "blue"),
    "front": ((65, 180, 95), "green"),
    "behind": ((165, 85, 205), "purple"),
}
TOP_DOWN_BACKGROUND = (242, 245, 248)
ALL_OBJECT_COLORS = (
    (255, 64, 64),
    (255, 170, 32),
    (64, 160, 255),
    (64, 220, 120),
    (210, 80, 255),
)

# Keep the original task's validation, targets, parsing, and scalar metrics.
process_docs = base.process_docs
doc_to_target = base.doc_to_target
extract_option_letter = base.extract_option_letter
aggregate_accuracy = base.aggregate_accuracy
aggregate_object_answer_accuracy = base.aggregate_object_answer_accuracy
aggregate_direction_answer_accuracy = base.aggregate_direction_answer_accuracy
aggregate_object_minus_direction = base.aggregate_object_minus_direction
aggregate_format_switch_gain = base.aggregate_format_switch_gain
aggregate_parse_success_rate = base.aggregate_parse_success_rate
aggregate_object_parse_success_rate = base.aggregate_object_parse_success_rate
aggregate_direction_parse_success_rate = base.aggregate_direction_parse_success_rate
aggregate_object_correct_direction_wrong = base.aggregate_object_correct_direction_wrong
aggregate_direction_correct_object_wrong = base.aggregate_direction_correct_object_wrong
aggregate_both_correct = base.aggregate_both_correct
aggregate_both_wrong = base.aggregate_both_wrong


@lru_cache(maxsize=1)
def load_scene_index(path: Path = SCENES_PATH) -> dict[str, dict]:
    """Read the per-scene bounding boxes, 3D positions, and orientations."""
    scenes = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            scene = json.loads(line)
            scene_id = str(scene.get("scene_id", ""))
            if not scene_id:
                raise ValueError(f"Missing scene_id in {path}:{line_number}")
            scenes[scene_id] = scene
    return scenes


def get_scene(doc: dict) -> dict:
    """Resolve an annotation row to its record in ``scenes.jsonl``."""
    scene_id = str(doc.get("scene_id", ""))
    try:
        return load_scene_index()[scene_id]
    except KeyError as error:
        raise KeyError(f"No COMFORT scene metadata for {scene_id!r}") from error


def get_scene_objects(doc: dict) -> list[dict]:
    return list(get_scene(doc).get("objects") or [])


def get_reference_object(doc: dict) -> dict:
    for obj in get_scene_objects(doc):
        if obj.get("role") == "reference" or obj.get("object_id") == "reference":
            return obj
    raise KeyError(f"Scene {doc.get('scene_id')!r} has no reference object")


def get_answer_object(doc: dict) -> dict:
    """Return the physical target object that instantiates the gold relation."""
    return get_object_at_direction(doc, base._relation(doc))


def get_object_at_direction(doc: dict, direction: str) -> dict:
    """Return the target object placed at a reference-relative direction."""
    for obj in get_scene_objects(doc):
        if obj.get("reference_direction") == direction:
            return obj
    raise KeyError(
        f"Scene {doc.get('scene_id')!r} has no target at direction {direction!r}"
    )


def get_bbox_normalized(obj: dict) -> tuple[float, float, float, float]:
    bbox = obj.get("bbox_2d_normalized_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Object {obj.get('object_id')!r} has no normalized xyxy box")
    return tuple(float(value) for value in bbox)


def get_camera_position(obj: dict) -> tuple[float, float, float]:
    """Return ``(right, forward, up)`` camera-frame position in scene units."""
    position = (obj.get("camera_frame") or {}).get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"Object {obj.get('object_id')!r} has no 3D camera position")
    return tuple(float(value) for value in position)


def get_camera_orientation(obj: dict) -> tuple[tuple[float, float, float], ...]:
    """Return the 3x3 matrix mapping asset-root axes into camera axes."""
    matrix = (obj.get("camera_frame") or {}).get("orientation_matrix")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise ValueError(f"Object {obj.get('object_id')!r} has no 3x3 orientation")
    if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in matrix):
        raise ValueError(f"Object {obj.get('object_id')!r} has an invalid orientation")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def bbox_to_pixels(obj: dict, image: Image.Image) -> tuple[int, int, int, int]:
    """Convert normalized top-left-origin xyxy coordinates to clamped pixels."""
    xmin, ymin, xmax, ymax = get_bbox_normalized(obj)
    width, height = image.size
    return (
        max(0, min(width - 1, round(xmin * width))),
        max(0, min(height - 1, round(ymin * height))),
        max(0, min(width - 1, round(xmax * width))),
        max(0, min(height - 1, round(ymax * height))),
    )


def _line_width(image: Image.Image) -> int:
    return max(3, round(min(image.size) / 170))


def _draw_box(
    image: Image.Image,
    obj: dict,
    color: tuple[int, int, int],
    label: Optional[str] = None,
) -> None:
    draw = ImageDraw.Draw(image)
    bbox = bbox_to_pixels(obj, image)
    draw.rectangle(bbox, outline=color, width=_line_width(image))
    if not label:
        return
    left, top, _, _ = bbox
    text_bbox = draw.textbbox((left, top), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_top = max(0, top - text_height - 5)
    draw.rectangle(
        (left, label_top, min(image.width - 1, left + text_width + 6), top),
        fill=color,
    )
    draw.text((left + 3, label_top + 1), label, fill=(0, 0, 0))


def draw_reference_bbox(doc: dict, image: Image.Image) -> Image.Image:
    """Draw a red box around only the reference object."""
    output = image.copy()
    obj = get_reference_object(doc)
    _draw_box(output, obj, REFERENCE_COLOR, f"reference: {obj.get('label', '')}")
    return output


def draw_answer_bbox(doc: dict, image: Image.Image) -> Image.Image:
    """Draw a green box around the target object used by the gold answer."""
    output = image.copy()
    obj = get_answer_object(doc)
    _draw_box(output, obj, ANSWER_COLOR, f"answer object: {obj.get('label', '')}")
    return output


def draw_all_object_bboxes(doc: dict, image: Image.Image) -> Image.Image:
    """Draw and name every object, without printing its direction relation."""
    output = image.copy()
    for index, obj in enumerate(get_scene_objects(doc)):
        role = "reference" if obj.get("role") == "reference" else "object"
        label = f"{role}: {obj.get('label', '')}"
        _draw_box(output, obj, ALL_OBJECT_COLORS[index % len(ALL_OBJECT_COLORS)], label)
    return output


def draw_numbered_object_bboxes(doc: dict, image: Image.Image) -> Image.Image:
    """Draw numbered boxes around every object without direction annotations."""
    output = image.copy()
    for index, obj in enumerate(get_scene_objects(doc), start=1):
        _draw_box(
            output,
            obj,
            ALL_OBJECT_COLORS[(index - 1) % len(ALL_OBJECT_COLORS)],
            f"#{index}",
        )
    return output


def draw_target_bbox_for_direction_questions(
    doc: dict, image: Image.Image
) -> Image.Image:
    """Localize the named target only on direction-answer rows."""
    if doc.get("diagnostic_answer_format") != "direction":
        return image
    output = image.copy()
    target = get_answer_object(doc)
    _draw_box(output, target, ANSWER_COLOR, f"question target: {target.get('label', '')}")
    return output


def _semantic_direction_camera_vector(
    doc: dict, direction: str
) -> tuple[float, float, float]:
    """Return a reference-to-direction-target vector in camera coordinates.

    The generated scene explicitly places a target at each semantic direction.
    Its 3D displacement is authoritative and avoids asset-local base-rotation
    ambiguities while remaining independent of rendered 2D box geometry.
    """
    reference_position = get_camera_position(get_reference_object(doc))
    direction_position = get_camera_position(get_object_at_direction(doc, direction))
    return tuple(
        direction_position[index] - reference_position[index] for index in range(3)
    )


def _direction_screen_direction(doc: dict, direction: str) -> tuple[float, float]:
    """Perspective-project a 3D direction and return its normalized 2D tangent."""
    x, y, z = get_camera_position(get_reference_object(doc))
    vx, vy, vz = _semantic_direction_camera_vector(doc, direction)
    if abs(y) < 1e-8:
        raise ValueError("Cannot project a reference object on the camera plane")
    # u=x/y and v=-z/y in a right/forward/up pinhole camera.  The focal
    # length cancels because the resulting screen tangent is normalized.
    du = (vx * y - x * vy) / (y * y)
    dv = -(vz * y - z * vy) / (y * y)
    norm = math.hypot(du, dv)
    if norm < 1e-8:
        # Front can point almost exactly into/out of the camera.  Keep the aid
        # visible and deterministic rather than drawing a zero-length arrow.
        return (0.0, -1.0 if vy >= 0 else 1.0)
    return (du / norm, dv / norm)


def _front_screen_direction(doc: dict) -> tuple[float, float]:
    """Backward-compatible helper for the original front-arrow preset."""
    return _direction_screen_direction(doc, "front")


def draw_reference_front_arrow(doc: dict, image: Image.Image) -> Image.Image:
    """Draw a cyan arrow from the reference centre toward its projected front."""
    output = image.copy()
    reference = get_reference_object(doc)
    left, top, right, bottom = bbox_to_pixels(reference, output)
    start = ((left + right) / 2.0, (top + bottom) / 2.0)
    dx, dy = _front_screen_direction(doc)
    length = max(28.0, 0.85 * math.hypot(right - left, bottom - top))
    end = (
        max(4.0, min(output.width - 5.0, start[0] + dx * length)),
        max(4.0, min(output.height - 5.0, start[1] + dy * length)),
    )
    draw = ImageDraw.Draw(output)
    width = _line_width(output) + 1
    draw.line((start, end), fill=FRONT_ARROW_COLOR, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(10.0, width * 3.0)
    points = [
        end,
        (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6)),
        (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6)),
    ]
    draw.polygon(points, fill=FRONT_ARROW_COLOR)
    return output


def _draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    screen_direction: tuple[float, float],
    length: float,
    color: tuple[int, int, int],
    label: str,
    width: int,
    image: Image.Image,
) -> None:
    dx, dy = screen_direction
    end = (
        max(5.0, min(image.width - 6.0, start[0] + dx * length)),
        max(5.0, min(image.height - 6.0, start[1] + dy * length)),
    )
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(10.0, width * 3.0)
    draw.polygon(
        [
            end,
            (
                end[0] - head * math.cos(angle - math.pi / 6),
                end[1] - head * math.sin(angle - math.pi / 6),
            ),
            (
                end[0] - head * math.cos(angle + math.pi / 6),
                end[1] - head * math.sin(angle + math.pi / 6),
            ),
        ],
        fill=color,
    )
    if not label:
        return
    text_box = draw.textbbox((0, 0), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_x = max(0, min(image.width - text_width - 6, round(end[0] + dx * 4)))
    label_y = max(0, min(image.height - text_height - 4, round(end[1] + dy * 4)))
    draw.rectangle(
        (label_x, label_y, label_x + text_width + 6, label_y + text_height + 4),
        fill=color,
    )
    draw.text((label_x + 3, label_y + 2), label, fill=(0, 0, 0))


def draw_reference_direction_arrows(doc: dict, image: Image.Image) -> Image.Image:
    """Superimpose the reference object's four labeled semantic directions."""
    output = image.copy()
    reference = get_reference_object(doc)
    left, top, right, bottom = bbox_to_pixels(reference, output)
    start = ((left + right) / 2.0, (top + bottom) / 2.0)
    length = max(38.0, 1.15 * math.hypot(right - left, bottom - top))
    width = _line_width(output) + 1
    draw = ImageDraw.Draw(output)
    radius = max(4, width + 1)
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=max(1, width // 2),
    )
    for direction in base.DIRECTIONS:
        _draw_direction_arrow(
            draw,
            start,
            _direction_screen_direction(doc, direction),
            length,
            DIRECTION_ARROW_COLORS[direction],
            direction,
            width,
            output,
        )
    return output


def draw_reference_bbox_and_heading_arrow(doc: dict, image: Image.Image) -> Image.Image:
    """Box the reference and add an arrow with no text rendered in the image."""
    return draw_reference_front_arrow(doc, draw_reference_bbox(doc, image))


def draw_reference_bbox_and_labeled_front_arrow(
    doc: dict, image: Image.Image
) -> Image.Image:
    """Box the reference and label its single projected heading arrow as front."""
    output = draw_reference_bbox(doc, image)
    reference = get_reference_object(doc)
    left, top, right, bottom = bbox_to_pixels(reference, output)
    start = ((left + right) / 2.0, (top + bottom) / 2.0)
    _draw_direction_arrow(
        ImageDraw.Draw(output),
        start,
        _front_screen_direction(doc),
        max(28.0, 0.85 * math.hypot(right - left, bottom - top)),
        FRONT_ARROW_COLOR,
        "front",
        _line_width(output) + 1,
        output,
    )
    return output


def draw_reference_symbolic_direction_arrows(
    doc: dict, image: Image.Image
) -> Image.Image:
    """Draw four color-coded arrows without direction words inside the image."""
    output = image.copy()
    reference = get_reference_object(doc)
    left, top, right, bottom = bbox_to_pixels(reference, output)
    start = ((left + right) / 2.0, (top + bottom) / 2.0)
    length = max(38.0, 1.15 * math.hypot(right - left, bottom - top))
    width = _line_width(output) + 1
    draw = ImageDraw.Draw(output)
    radius = max(4, width + 1)
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=max(1, width // 2),
    )
    for direction in base.DIRECTIONS:
        _draw_direction_arrow(
            draw,
            start,
            _direction_screen_direction(doc, direction),
            length,
            DIRECTION_ARROW_COLORS[direction],
            "",
            width,
            output,
        )
    return output


def _map_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _centered_map_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    panel_width: int,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    x = max(4, min(panel_width - width - 4, round(xy[0] - width / 2)))
    draw.text((x, round(xy[1])), text, font=font, fill=fill)


def _map_object_records(doc: dict) -> list[tuple[str, dict]]:
    return [
        ("reference", get_reference_object(doc)),
        *((direction, get_object_at_direction(doc, direction)) for direction in base.DIRECTIONS),
    ]


def _draw_map_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    label: Optional[str],
    font: ImageFont.ImageFont,
    panel_width: int,
) -> None:
    x, y = center
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=(25, 25, 25),
        width=max(2, round(radius / 8)),
    )
    if label:
        _centered_map_text(
            draw,
            (x, y + radius + 5),
            label,
            font,
            (20, 20, 20),
            panel_width,
        )


def _draw_map_axes(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    distance: float,
    color: tuple[int, int, int] = (155, 160, 168),
) -> None:
    x, y = center
    draw.line((x - distance, y, x + distance, y), fill=color, width=2)
    draw.line((x, y - distance, x, y + distance), fill=color, width=2)


def _reference_map_panel(
    doc: dict,
    size: int,
    *,
    object_labels: bool,
    direction_labels: bool,
    title: str = "Reference-centered top-down map",
) -> Image.Image:
    panel = Image.new("RGB", (size, size), TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font = _map_font(max(15, round(size / 30)))
    label_font = _map_font(max(12, round(size / 42)))
    axis_font = _map_font(max(13, round(size / 36)))
    _centered_map_text(draw, (size / 2, 12), title, title_font, (20, 20, 20), size)
    center = (size / 2, size / 2 + size * 0.025)
    distance = size * 0.29
    radius = max(16.0, size * 0.045)
    positions = {
        "reference": center,
        "left": (center[0] - distance, center[1]),
        "right": (center[0] + distance, center[1]),
        "front": (center[0], center[1] - distance),
        "behind": (center[0], center[1] + distance),
    }
    _draw_map_axes(draw, center, distance)
    for direction, obj in _map_object_records(doc):
        color = TOP_DOWN_COLORS[direction][0]
        label = str(obj.get("label", "")) if object_labels else None
        _draw_map_circle(draw, positions[direction], radius, color, label, label_font, size)
    # The small black heading marker makes the reference's canonical facing
    # direction visible even when direction words are intentionally omitted.
    draw.polygon(
        [
            (center[0], center[1] - radius * 0.9),
            (center[0] - radius * 0.28, center[1] - radius * 0.25),
            (center[0] + radius * 0.28, center[1] - radius * 0.25),
        ],
        fill=(20, 20, 20),
    )
    if direction_labels:
        axis_positions = {
            "left": (size * 0.07, center[1] - axis_font.size),
            "right": (size * 0.93, center[1] - axis_font.size),
            "front": (center[0], size * 0.10),
            "behind": (center[0], size * 0.94),
        }
        for direction, position in axis_positions.items():
            _centered_map_text(
                draw,
                position,
                direction,
                axis_font,
                TOP_DOWN_COLORS[direction][0],
                size,
            )
    return panel


def _camera_map_panel(
    doc: dict, size: int, *, semantic_colors: bool = True
) -> Image.Image:
    """Render actual object positions in the camera right/forward plane."""
    panel = Image.new("RGB", (size, size), TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font = _map_font(max(15, round(size / 30)))
    label_font = _map_font(max(12, round(size / 42)))
    axis_font = _map_font(max(13, round(size / 36)))
    _centered_map_text(draw, (size / 2, 12), "Camera-frame top-down map", title_font, (20, 20, 20), size)
    center = (size / 2, size / 2 + size * 0.025)
    extent = size * 0.32
    _draw_map_axes(draw, center, extent)
    reference = get_reference_object(doc)
    reference_position = get_camera_position(reference)
    records = _map_object_records(doc)
    deltas = {}
    maximum = 1e-8
    for direction, obj in records:
        position = get_camera_position(obj)
        delta = (
            position[0] - reference_position[0],
            position[1] - reference_position[1],
        )
        deltas[direction] = delta
        maximum = max(maximum, abs(delta[0]), abs(delta[1]))
    scale = extent / maximum
    radius = max(16.0, size * 0.045)
    for direction, obj in records:
        delta_x, delta_forward = deltas[direction]
        position = (
            center[0] + delta_x * scale,
            center[1] - delta_forward * scale,
        )
        _draw_map_circle(
            draw,
            position,
            radius,
            TOP_DOWN_COLORS[direction][0]
            if semantic_colors
            else (245, 200, 40)
            if direction == "reference"
            else (115, 145, 180),
            str(obj.get("label", "")),
            label_font,
            size,
        )
    for text, xy in (
        ("camera left", (22, center[1] - axis_font.size / 2)),
        ("camera right", (size - 58, center[1] - axis_font.size / 2)),
        ("camera front", (center[0], center[1] - extent - axis_font.size - 8)),
        ("camera behind", (center[0], center[1] + extent + 8)),
    ):
        _centered_map_text(draw, xy, text, axis_font, (75, 80, 88), size)
    return panel


def _query_map_panel(doc: dict, size: int, *, include_distractors: bool) -> Image.Image:
    """Render a canonical map with the current question entities highlighted.

    The cyan object is an oracle for object-answer questions because that entity
    is what the question asks the model to find.  It is non-oracle entity
    selection for direction-answer questions, where both objects are named.
    """
    panel = Image.new("RGB", (size, size), TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font = _map_font(max(15, round(size / 30)))
    label_font = _map_font(max(12, round(size / 42)))
    _centered_map_text(
        draw,
        (size / 2, 12),
        "Question-entity reference map",
        title_font,
        (20, 20, 20),
        size,
    )
    center = (size / 2, size / 2 + size * 0.025)
    distance = size * 0.29
    radius = max(16.0, size * 0.045)
    positions = {
        "reference": center,
        "left": (center[0] - distance, center[1]),
        "right": (center[0] + distance, center[1]),
        "front": (center[0], center[1] - distance),
        "behind": (center[0], center[1] + distance),
    }
    _draw_map_axes(draw, center, distance)
    answer_object = get_answer_object(doc)
    for direction, obj in _map_object_records(doc):
        selected = direction == "reference" or obj is answer_object
        if not include_distractors and not selected:
            continue
        color = (
            (245, 200, 40)
            if direction == "reference"
            else (35, 205, 220)
            if selected
            else (190, 194, 200)
        )
        _draw_map_circle(
            draw,
            positions[direction],
            radius,
            color,
            str(obj.get("label", "")),
            label_font,
            size,
        )
    draw.polygon(
        [
            (center[0], center[1] - radius * 0.9),
            (center[0] - radius * 0.28, center[1] - radius * 0.25),
            (center[0] + radius * 0.28, center[1] - radius * 0.25),
        ],
        fill=(20, 20, 20),
    )
    return panel


def _append_map_panels(image: Image.Image, panels: Iterable[Image.Image]) -> Image.Image:
    panels = list(panels)
    output = Image.new(
        "RGB",
        (image.width + sum(panel.width for panel in panels), max([image.height, *(panel.height for panel in panels)])),
        (255, 255, 255),
    )
    output.paste(image.convert("RGB"), (0, 0))
    x = image.width
    for panel in panels:
        output.paste(panel, (x, 0))
        x += panel.width
    return output


def draw_reference_bbox_and_crop(doc: dict, image: Image.Image) -> Image.Image:
    """Box the reference and append a magnified crop of that object."""
    reference = get_reference_object(doc)
    boxed = draw_reference_bbox(doc, image)
    left, top, right, bottom = bbox_to_pixels(reference, image)
    padding = max(6, round(0.2 * max(right - left, bottom - top)))
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding + 1),
        min(image.height, bottom + padding + 1),
    )
    crop = image.convert("RGB").crop(crop_box)
    panel = Image.new("RGB", (image.height, image.height), TOP_DOWN_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    font = _map_font(max(15, round(image.height / 30)))
    _centered_map_text(
        draw,
        (panel.width / 2, 12),
        f"Reference crop: {reference.get('label', '')}",
        font,
        (20, 20, 20),
        panel.width,
    )
    available = max(1, image.height - max(48, round(image.height * 0.12)))
    scale = min((panel.width - 24) / crop.width, (available - 12) / crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel.paste(
        resized,
        ((panel.width - resized.width) // 2, image.height - resized.height - 12),
    )
    return _append_map_panels(boxed, [panel])


def draw_reference_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append a reference-canonical map with object names on colored circles."""
    size = image.height
    return _append_map_panels(
        image,
        [_reference_map_panel(doc, size, object_labels=True, direction_labels=False)],
    )


def draw_reference_top_down_map_only(doc: dict, image: Image.Image) -> Image.Image:
    """Replace the RGB scene with the same canonical map used by mode 11."""
    return _reference_map_panel(
        doc,
        image.height,
        object_labels=True,
        direction_labels=False,
    )


def draw_query_pair_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append a map containing only the reference and current answer entity."""
    return _append_map_panels(
        image,
        [_query_map_panel(doc, image.height, include_distractors=False)],
    )


def draw_highlighted_query_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append an all-object map with the current answer entity highlighted."""
    return _append_map_panels(
        image,
        [_query_map_panel(doc, image.height, include_distractors=True)],
    )


def draw_camera_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append only the camera-coordinate map as a frame-bias control."""
    return _append_map_panels(
        image,
        [_camera_map_panel(doc, image.height, semantic_colors=False)],
    )


def draw_unlabeled_reference_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append color-only reference-canonical geometry with no direction words."""
    size = image.height
    return _append_map_panels(
        image,
        [_reference_map_panel(doc, size, object_labels=False, direction_labels=False)],
    )


def draw_labeled_reference_top_down_map(doc: dict, image: Image.Image) -> Image.Image:
    """Append a fully object- and direction-labeled reference map."""
    size = image.height
    return _append_map_panels(
        image,
        [_reference_map_panel(doc, size, object_labels=True, direction_labels=True)],
    )


def draw_reference_and_camera_top_down_maps(doc: dict, image: Image.Image) -> Image.Image:
    """Append matched reference-frame and camera-frame top-down maps."""
    size = image.height
    return _append_map_panels(
        image,
        [
            _reference_map_panel(
                doc,
                size,
                object_labels=True,
                direction_labels=True,
                title="Reference-frame top-down map",
            ),
            _camera_map_panel(doc, size),
        ],
    )


def describe_reference_bbox(doc: dict) -> str:
    obj = get_reference_object(doc)
    return f"Image aid: the red box marks the reference object ({obj.get('label')})."


def describe_answer_bbox(doc: dict) -> str:
    obj = get_answer_object(doc)
    return f"Image aid: the green box marks the object involved in the correct answer ({obj.get('label')})."


def describe_all_object_bboxes(doc: dict) -> str:
    return "Image aid: colored, labeled boxes mark every object; the reference box is labeled 'reference'."


def response_format_scaffold(doc: dict) -> str:
    return (
        "Response-format rule: return exactly one option letter, A, B, C, or D. "
        "For example, if the second option is correct, respond with B."
    )


def neutral_reference_perspective_example(doc: dict) -> str:
    return (
        "Worked example unrelated to the current image: suppose a robot is the "
        "reference object and faces away from the camera. A ball placed on the "
        "robot's own left may appear on image-right. A direction question asking "
        "where the ball is should still be answered 'left'. An object question "
        "asking what is to the robot's left should be answered 'ball'. First solve "
        "from the reference object's axes, then return the letter of the matching option."
    )


def identify_reference_object(doc: dict) -> str:
    obj = get_reference_object(doc)
    return f"Reference-identity help: the reference object is the {obj.get('label')}."


def describe_reference_crop(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        f"Image aid: the red box and magnified inset identify the reference object "
        f"({obj.get('label')})."
    )


def numbered_object_mapping(doc: dict) -> str:
    mappings = []
    for index, obj in enumerate(get_scene_objects(doc), start=1):
        role = " (reference)" if obj.get("role") == "reference" else ""
        mappings.append(f"#{index}={obj.get('label')}{role}")
    return "Numbered object boxes: " + "; ".join(mappings) + "."


def describe_reference_front_arrow(doc: dict) -> str:
    obj = get_reference_object(doc)
    return f"Image aid: the cyan arrow indicates the front of the reference object ({obj.get('label')})."


def describe_reference_heading_arrow(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        f"Image aid: the red box marks the reference object ({obj.get('label')}); "
        "the cyan arrow without an image label shows its heading (its own front)."
    )


def describe_labeled_reference_front_arrow(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        f"Image aid: the red box marks the reference object ({obj.get('label')}); "
        "its single cyan arrow is labeled 'front'. Derive the other three axes from it."
    )


def describe_symbolic_direction_arrows(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        f"Image aid: four arrows originate at the reference object ({obj.get('label')}). "
        "The fixed color legend is orange=left, blue=right, green=front, and "
        "purple=behind; direction words are not rendered inside the image."
    )


def describe_reference_direction_arrows(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        "Image aid: the four labeled arrows superimposed on the reference object "
        f"({obj.get('label')}) show its own left, right, front, and behind directions."
    )


def explain_reference_perspective(doc: dict) -> str:
    obj = get_reference_object(doc)
    return (
        f"Reference-perspective rule: mentally stand at the {obj.get('label')} and face "
        "the same way it faces. Interpret left, right, front, and behind using that "
        "object's axes, which rotate with the object. Do not use image-left/image-right "
        "or the camera's facing/depth axes."
    )


def _camera_direction_for_object(doc: dict, obj: dict) -> str:
    """Classify reference-to-object displacement in the camera right/forward plane."""
    reference_position = get_camera_position(get_reference_object(doc))
    object_position = get_camera_position(obj)
    right_delta = object_position[0] - reference_position[0]
    forward_delta = object_position[1] - reference_position[1]
    if abs(right_delta) >= abs(forward_delta):
        return "right" if right_delta >= 0 else "left"
    return "front" if forward_delta >= 0 else "behind"


def reference_vs_camera_example(doc: dict) -> str:
    """Give the current question's expected answer and its camera-frame contrast."""
    relation = base._relation(doc)
    if doc.get("diagnostic_answer_format") == "direction":
        target = get_answer_object(doc)
        camera_answer = _camera_direction_for_object(doc, target)
        contrast = (
            f"Use {relation}, the reference-frame answer, not {camera_answer}, the "
            "camera-frame description."
            if camera_answer != relation
            else f"Both frames happen to give {relation} here, but derive it from the reference frame."
        )
        return (
            f"Frame comparison for this question: the {target.get('label')} is "
            f"{relation} from the reference object's perspective, so the expected "
            f"answer is {relation}. From the camera perspective the same object is "
            f"{camera_answer}. {contrast}"
        )

    expected_object = get_answer_object(doc)
    camera_direction = _camera_direction_for_object(doc, expected_object)
    contrast = (
        f"Thus camera-{relation} would be the wrong rule for finding it."
        if camera_direction != relation
        else "The two direction labels happen to agree here, but the reference-frame rule still governs."
    )
    return (
        f"Frame comparison for this question: at reference-relative {relation}, the "
        f"expected answer is {expected_object.get('label')}. From the camera perspective, "
        f"that same object lies {camera_direction}. {contrast} Use "
        f"{expected_object.get('label')}, the reference-relative answer."
    )


def top_down_color_mapping(doc: dict) -> str:
    mappings = []
    for direction, obj in _map_object_records(doc):
        color_name = TOP_DOWN_COLORS[direction][1]
        role = " (reference)" if direction == "reference" else ""
        mappings.append(f"{color_name}={obj.get('label')}{role}")
    return (
        "Top-down map aid: the reference object is centered and its facing direction "
        "is toward the top of the map. Color-to-object mapping: "
        + "; ".join(mappings)
        + ". Use the map's reference-centered geometry when answering the question."
    )


def unlabeled_top_down_color_mapping(doc: dict) -> str:
    mappings = [
        f"{TOP_DOWN_COLORS[direction][1]}={obj.get('label')}"
        for direction, obj in _map_object_records(doc)
    ]
    return (
        "Color-only top-down aid: the yellow circle at the center is the reference "
        "object, and its black heading marker points toward the top of the map. Circle "
        "colors map to objects as follows: "
        + "; ".join(mappings)
        + ". The map itself intentionally contains no object or direction words."
    )


def labeled_top_down_mapping(doc: dict) -> str:
    return (
        top_down_color_mapping(doc)
        + " The map also labels the reference-relative left, right, front, and behind axes explicitly."
    )


def dual_top_down_mapping(doc: dict) -> str:
    return (
        top_down_color_mapping(doc)
        + " The first panel canonicalizes the reference object's own axes; the second "
        "panel shows the same colored objects in camera right/forward coordinates. "
        "Answer from the first, reference-frame panel, not from the camera-frame panel."
    )


def canonical_numeric_layout(doc: dict) -> str:
    """Expose canonical coordinates without directly printing object relations."""
    coordinates = {
        "reference": (0, 0),
        "left": (-1, 0),
        "right": (1, 0),
        "front": (0, 1),
        "behind": (0, -1),
    }
    records = []
    for direction, obj in _map_object_records(doc):
        x, y = coordinates[direction]
        records.append(f"{obj.get('label')}=({x},{y})")
    return (
        "Reference-relative numeric layout: coordinates are (horizontal, forward), "
        "with positive horizontal toward the reference object's right and positive "
        "forward toward its front. "
        + "; ".join(records)
        + ". Infer the requested relation or object from these coordinates."
    )


def describe_map_only(doc: dict) -> str:
    return (
        top_down_color_mapping(doc)
        + " This canonical map replaces the original RGB scene, removing appearance and background cues."
    )


def describe_query_pair_map(doc: dict) -> str:
    reference = get_reference_object(doc)
    target = get_answer_object(doc)
    return (
        "Question-entity map: yellow marks the centered reference object "
        f"({reference.get('label')}) and cyan marks {target.get('label')}; all other "
        "objects are removed. The reference heading marker points toward the top."
    )


def describe_highlighted_query_map(doc: dict) -> str:
    target = get_answer_object(doc)
    return (
        "Attention-oracle map: yellow marks the reference, cyan highlights the "
        f"question's answer entity ({target.get('label')}), and gray circles are "
        "distractors. The reference heading marker points toward the top."
    )


def describe_camera_map_control(doc: dict) -> str:
    return (
        "Camera-frame control: the added panel places the named objects using camera "
        "left/right/front/behind coordinates, centered on the reference object."
    )


def describe_dual_maps_without_selection(doc: dict) -> str:
    return (
        "Two coordinate views are provided: the first panel is centered on the "
        "reference object's own axes and the second uses the camera's axes. Both "
        "panels contain the same color-to-object correspondences: "
        + "; ".join(
            f"{TOP_DOWN_COLORS[direction][1]}={obj.get('label')}"
            for direction, obj in _map_object_records(doc)
        )
        + "."
    )


def describe_reference_heading_in_camera_frame(doc: dict) -> str:
    reference = get_reference_object(doc)
    camera_direction = _camera_direction_for_object(
        doc, get_object_at_direction(doc, "front")
    )
    return (
        f"Orientation help: the {reference.get('label')}'s own front points mostly "
        f"toward camera-{camera_direction}. Use that heading to derive its other axes."
    )


def describe_reference_to_camera_axis_mapping(doc: dict) -> str:
    mappings = []
    for direction in base.DIRECTIONS:
        camera_right, camera_forward, _ = _semantic_direction_camera_vector(
            doc, direction
        )
        norm = math.hypot(camera_right, camera_forward)
        if norm < 1e-8:
            raise ValueError(f"Degenerate camera-plane direction for {direction!r}")
        mappings.append(
            f"reference-{direction}=({camera_right / norm:+.2f} camera-right, "
            f"{camera_forward / norm:+.2f} camera-forward)"
        )
    return (
        "Reference-to-camera axis mapping using normalized camera-plane vectors: "
        + "; ".join(mappings)
        + "."
    )


def reveal_relation_for_object_questions(doc: dict) -> str:
    if doc.get("diagnostic_answer_format") != "object":
        return ""
    return (
        "Relation-oracle help for this object-answer row: the required "
        f"reference-relative relation is {base._relation(doc)}."
    )


def reveal_target_for_direction_questions(doc: dict) -> str:
    if doc.get("diagnostic_answer_format") != "direction":
        return ""
    return (
        "Target-localization oracle for this direction-answer row: the green box "
        f"marks the named question target ({get_answer_object(doc).get('label')}). "
        "Classify its position relative to the reference object."
    )


def ground_truth_free_text(doc: dict) -> str:
    index = int(doc["answer_idx"])
    return (
        "Answer-text oracle: the correct answer text is "
        f"'{doc['options'][index]}'. Match that text to the options and return its letter."
    )


def ground_truth_letter_only(doc: dict) -> str:
    return f"Answer-letter oracle: respond with {doc['gold_option_letter']}."


def ground_truth_plain_text(doc: dict) -> str:
    """Reveal both the gold option letter and its literal option text."""
    index = int(doc["answer_idx"])
    return f"Ground-truth help: the correct answer is {doc['gold_option_letter']}. {doc['options'][index]}."


VisualAid = Callable[[dict, Image.Image], Image.Image]
TextAid = Callable[[dict], str]


def compose_visual_aids(
    doc: dict, image: Image.Image, aids: Iterable[VisualAid]
) -> Image.Image:
    output = image
    for aid in aids:
        output = aid(doc, output)
    return output


def compose_text_aids(doc: dict, aids: Iterable[TextAid]) -> list[str]:
    return [text for aid in aids if (text := aid(doc))]


# Edit these tuples to compose arbitrary experiments.  For example, mode 4 can
# combine a box and arrow by using ``(draw_reference_bbox,
# draw_reference_front_arrow)`` and their two matching text functions.
GT_HELP_PRESETS: dict[str, dict[str, tuple]] = {
    "0": {"visual": (), "text": ()},
    "1": {"visual": (draw_reference_bbox,), "text": (describe_reference_bbox,)},
    "2": {"visual": (draw_answer_bbox,), "text": (describe_answer_bbox,)},
    "3": {"visual": (draw_all_object_bboxes,), "text": (describe_all_object_bboxes,)},
    "4": {"visual": (draw_reference_front_arrow,), "text": (describe_reference_front_arrow,)},
    "5": {"visual": (), "text": (ground_truth_plain_text,)},
    "6": {
        "visual": (draw_reference_direction_arrows,),
        "text": (describe_reference_direction_arrows,),
    },
    "7": {
        "visual": (draw_reference_direction_arrows,),
        "text": (describe_reference_direction_arrows, explain_reference_perspective),
    },
    "8": {"visual": (), "text": (explain_reference_perspective,)},
    "9": {"visual": (), "text": (reference_vs_camera_example,)},
    "10": {
        "visual": (draw_reference_direction_arrows,),
        "text": (describe_reference_direction_arrows, reference_vs_camera_example),
    },
    "11": {
        "visual": (draw_reference_top_down_map,),
        "text": (top_down_color_mapping,),
    },
    "12": {"visual": (), "text": (response_format_scaffold,)},
    "13": {"visual": (), "text": (neutral_reference_perspective_example,)},
    "14": {
        "visual": (draw_unlabeled_reference_top_down_map,),
        "text": (unlabeled_top_down_color_mapping,),
    },
    "15": {
        "visual": (draw_labeled_reference_top_down_map,),
        "text": (labeled_top_down_mapping,),
    },
    "16": {
        "visual": (draw_reference_and_camera_top_down_maps,),
        "text": (dual_top_down_mapping,),
    },
    "17": {"visual": (), "text": (identify_reference_object,)},
    "18": {
        "visual": (draw_reference_bbox_and_crop,),
        "text": (describe_reference_crop,),
    },
    "19": {
        "visual": (draw_numbered_object_bboxes,),
        "text": (numbered_object_mapping,),
    },
    "20": {
        "visual": (draw_reference_bbox_and_heading_arrow,),
        "text": (describe_reference_heading_arrow,),
    },
    "21": {
        "visual": (draw_reference_bbox_and_labeled_front_arrow,),
        "text": (describe_labeled_reference_front_arrow,),
    },
    "22": {
        "visual": (draw_reference_symbolic_direction_arrows,),
        "text": (describe_symbolic_direction_arrows,),
    },
    "23": {"visual": (), "text": (canonical_numeric_layout,)},
    "24": {
        "visual": (draw_reference_top_down_map_only,),
        "text": (describe_map_only,),
    },
    "25": {
        "visual": (draw_query_pair_top_down_map,),
        "text": (describe_query_pair_map,),
    },
    "26": {
        "visual": (draw_highlighted_query_top_down_map,),
        "text": (describe_highlighted_query_map,),
    },
    "27": {
        "visual": (draw_camera_top_down_map,),
        "text": (describe_camera_map_control,),
    },
    "28": {
        "visual": (draw_reference_and_camera_top_down_maps,),
        "text": (describe_dual_maps_without_selection,),
    },
    "29": {
        "visual": (draw_reference_and_camera_top_down_maps,),
        "text": (dual_top_down_mapping,),
    },
    "30": {
        "visual": (),
        "text": (describe_reference_heading_in_camera_frame,),
    },
    "31": {
        "visual": (),
        "text": (describe_reference_to_camera_axis_mapping,),
    },
    "32": {
        "visual": (),
        "text": (reveal_relation_for_object_questions,),
    },
    "33": {
        "visual": (draw_target_bbox_for_direction_questions,),
        "text": (reveal_target_for_direction_questions,),
    },
    "34": {"visual": (), "text": (ground_truth_free_text,)},
    "35": {"visual": (), "text": (ground_truth_letter_only,)},
}


def get_gt_help_mode() -> str:
    mode = str(os.getenv("GT_HELP", "0")).strip()
    if mode not in VALID_GT_HELP:
        raise ValueError(f"GT_HELP must be one of {sorted(VALID_GT_HELP)}, got {mode!r}")
    return mode


def _debug_save_enabled() -> bool:
    value = str(os.getenv("GT_HELP_DEBUG", "0")).strip().lower()
    if value in {"0", "false", "no", "off", ""}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "GT_HELP_DEBUG must be a boolean value such as 0/1 or false/true, "
        f"got {value!r}"
    )


def save_debug_image(doc: dict, image: Image.Image, mode: str) -> Optional[Path]:
    """Save the final model-visible image when ``GT_HELP_DEBUG`` is enabled."""
    if not _debug_save_enabled():
        return None

    output_dir = Path(
        os.getenv("GT_HELP_DEBUG_DIR", str(DEBUG_SAVE_DEFAULT_DIR))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_id = str(doc.get("qid") or doc.get("id") or doc.get("index") or "sample")
    safe_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in raw_id
    ).strip("_") or "sample"
    output_path = output_dir / f"gt_help_{mode}_{safe_id}.png"
    image.save(output_path, format="PNG")
    eval_logger.info("Saved GT-help debug image to {}.", output_path)
    return output_path


def _render_final_image(doc: dict, mode: Optional[str] = None) -> Image.Image:
    """Build the exact final image shared by debugging and model inference."""
    mode = mode or get_gt_help_mode()
    image = base.doc_to_visual(doc)[0]
    return compose_visual_aids(doc, image, GT_HELP_PRESETS[mode]["visual"])


def doc_to_visual(doc):
    mode = get_gt_help_mode()
    output = _render_final_image(doc, mode)
    save_debug_image(doc, output, mode)
    return [output]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    mode = get_gt_help_mode()
    # The lmms-eval dummy model never resolves visual request arguments. Save
    # from the always-evaluated prompt path too, using the same renderer as
    # doc_to_visual, so dry-run jobs still produce inspectable images.
    if _debug_save_enabled():
        save_debug_image(doc, _render_final_image(doc, mode), mode)
    prompt = base.doc_to_text(doc, lmms_eval_specific_kwargs)
    aid_lines = compose_text_aids(doc, GT_HELP_PRESETS[mode]["text"])
    if not aid_lines:
        return prompt
    return f"{' '.join(aid_lines)}\n{prompt}"


def process_results(doc, results):
    output = base.process_results(doc, results)
    mode = get_gt_help_mode()
    aid_lines = compose_text_aids(doc, GT_HELP_PRESETS[mode]["text"])
    output["submission"].update(
        {
            "gt_help_mode": mode,
            "gt_help_text": aid_lines,
            "question_prompt": doc_to_text(doc),
        }
    )
    return output


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(
        f"comfort_direction_object_gt_help_{get_gt_help_mode()}_{model}.json", args
    )
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": "comfort_direction_object_gt_help",
        "gt_help_mode": get_gt_help_mode(),
        "num_records": len(results),
        "num_matched_pairs": len(base._matched_pairs(results)),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT GT-help direction/object records saved to {}.", path)
