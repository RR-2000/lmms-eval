"""COMFORT_Multi_3D single-object bbox prediction and IoU evaluation."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


DATA_ROOT = Path("/home/ramanathan/data/COMFORT_Multi_3D")
COORDINATE_MAX = 1000.0
IOU_THRESHOLDS = (0.25, 0.5, 0.75)
DEBUG_DEFAULT_DIR = Path("outputs/comfort_multi_3d_bbox_prediction_debug")
GT_COLOR = (40, 220, 80)
PRED_COLOR = (255, 60, 60)


def _image_path(scene: dict) -> Path:
    path = Path(str(scene.get("image", "")))
    return path if path.is_absolute() else DATA_ROOT / path


def _bbox_1000(obj: dict) -> list[float]:
    bbox = obj.get("bbox_2d_normalized_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Object {obj.get('object_id')!r} has no normalized xyxy bbox")
    values = [float(value) for value in bbox]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Normalized bbox is outside [0,1]: {values}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid normalized xyxy bbox: {values}")
    return [value * COORDINATE_MAX for value in values]


def _question_for_object(obj: dict) -> str:
    label = str(obj.get("label", "")).strip()
    if obj.get("role") == "reference":
        return f"Locate the reference object, the {label}, in the image."
    return f"Locate the {label} in the image."


def process_docs(dataset: Dataset) -> Dataset:
    """Expand every scene into one independent query for each visible object."""
    records = []
    skipped = Counter()
    for source in dataset:
        scene = dict(source)
        scene_id = str(scene.get("scene_id", "")).strip()
        image_path = _image_path(scene)
        objects = scene.get("objects") or []
        if not scene_id:
            skipped["missing_scene_id"] += 1
            continue
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue
        if not isinstance(objects, list) or not objects:
            skipped["missing_objects"] += 1
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                skipped["invalid_object"] += 1
                continue
            object_id = str(obj.get("object_id", "")).strip()
            label = str(obj.get("label", "")).strip()
            role = str(obj.get("role", "")).strip()
            if not object_id or not label or role not in {"reference", "target"}:
                skipped["invalid_object_metadata"] += 1
                continue
            try:
                gt_bbox = _bbox_1000(obj)
            except (TypeError, ValueError):
                skipped["invalid_bbox"] += 1
                continue
            qid = f"{scene_id}::{object_id}"
            records.append(
                {
                    "qid": qid,
                    "index": qid,
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "object_label": label,
                    "object_role": role,
                    "reference_direction": obj.get("reference_direction"),
                    "question": _question_for_object(obj),
                    "gt_bbox_1000_xyxy": gt_bbox,
                    "bbox_2d_normalized_xyxy": [value / COORDINATE_MAX for value in gt_bbox],
                    "img_path": str(image_path),
                    "image_path": str(image_path),
                }
            )

    eval_logger.info(
        "COMFORT bbox prediction loaded {} object queries from {} scenes; skipped={}.",
        len(records),
        len({record["scene_id"] for record in records}),
        dict(skipped),
    )
    return Dataset.from_list(records)


def doc_to_visual(doc):
    path = Path(str(doc.get("img_path") or doc.get("image_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"COMFORT bbox image not found for {doc.get('qid')}: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return (
        f"{kwargs.get('pre_prompt', '')}{doc['question']} "
        "Return exactly one bounding box as [x_min, y_min, x_max, y_max]. "
        "Use coordinates normalized to the range [0, 1000], with (0, 0) at "
        "the image's top-left and (1000, 1000) at its bottom-right."
        f"{kwargs.get('post_prompt', '')}"
    )


def doc_to_target(doc):
    return json.dumps([round(value, 3) for value in doc["gt_bbox_1000_xyxy"]])


_BBOX_PATTERN = re.compile(
    r"[\[\(]?\s*(-?\d+(?:\.\d+)?)\s*[,\s]+"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]+"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]+"
    r"(-?\d+(?:\.\d+)?)\s*[\]\)]?"
)


def parse_bbox(text: str) -> Optional[list[float]]:
    """Parse the first valid, ordered xyxy box in the required [0,1000] frame."""
    if not text:
        return None
    for match in _BBOX_PATTERN.finditer(str(text)):
        values = [float(match.group(index)) for index in range(1, 5)]
        x1, y1, x2, y2 = values
        if (
            all(0.0 <= value <= COORDINATE_MAX for value in values)
            and x2 > x1
            and y2 > y1
        ):
            return values
    return None


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def _debug_enabled() -> bool:
    value = str(os.getenv("COMFORT_BBOX_DEBUG", "0")).strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "COMFORT_BBOX_DEBUG must be a boolean such as 0/1 or false/true, "
        f"got {value!r}"
    )


def _bbox_to_pixels(box: list[float], image: Image.Image) -> tuple[int, int, int, int]:
    return (
        round(box[0] / COORDINATE_MAX * image.width),
        round(box[1] / COORDINATE_MAX * image.height),
        round(box[2] / COORDINATE_MAX * image.width),
        round(box[3] / COORDINATE_MAX * image.height),
    )


def _draw_labeled_box(
    image: Image.Image,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    pixels = _bbox_to_pixels(box, image)
    width = max(3, round(min(image.size) / 170))
    draw.rectangle(pixels, outline=color, width=width)
    x, y = pixels[:2]
    text_box = draw.textbbox((x, y), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_y = max(0, y - text_height - 5)
    draw.rectangle((x, label_y, x + text_width + 6, y), fill=color)
    draw.text((x + 3, label_y + 1), label, fill=(0, 0, 0))


def save_prediction_overlay(
    doc: dict,
    predicted_bbox: Optional[list[float]],
    iou: float,
) -> Optional[Path]:
    """Save GT (green) and prediction (red) on the source image in debug mode."""
    if not _debug_enabled():
        return None
    image = doc_to_visual(doc)[0]
    gt_bbox = [float(value) for value in doc["gt_bbox_1000_xyxy"]]
    _draw_labeled_box(image, gt_bbox, GT_COLOR, "GT")
    if predicted_bbox is not None:
        _draw_labeled_box(image, predicted_bbox, PRED_COLOR, f"prediction IoU={iou:.3f}")
    else:
        draw = ImageDraw.Draw(image)
        message = "prediction: parse failure"
        text_box = draw.textbbox((5, 5), message)
        draw.rectangle((3, 3, text_box[2] + 8, text_box[3] + 8), fill=PRED_COLOR)
        draw.text((6, 6), message, fill=(0, 0, 0))

    output_dir = Path(
        os.getenv("COMFORT_BBOX_DEBUG_DIR", str(DEBUG_DEFAULT_DIR))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_qid = re.sub(r"[^A-Za-z0-9_-]+", "_", str(doc.get("qid", "sample"))).strip("_")
    output_path = output_dir / f"{safe_qid or 'sample'}.png"
    image.save(output_path, format="PNG")
    return output_path


def _metric_entry(doc: dict, prediction: str, parsed: Optional[list[float]]) -> dict:
    gt_bbox = [float(value) for value in doc["gt_bbox_1000_xyxy"]]
    iou = compute_iou(gt_bbox, parsed) if parsed is not None else 0.0
    return {
        "qid": doc.get("qid"),
        "scene_id": doc.get("scene_id"),
        "object_id": doc.get("object_id"),
        "object_label": doc.get("object_label"),
        "object_role": doc.get("object_role"),
        "reference_direction": doc.get("reference_direction"),
        "gt_bbox": gt_bbox,
        "predicted_bbox": parsed,
        "prediction": prediction,
        "parse_success": parsed is not None,
        "iou": iou,
        "bbox_acc_0_25": float(iou >= 0.25),
        "bbox_acc_0_5": float(iou >= 0.5),
        "bbox_acc_0_75": float(iou >= 0.75),
    }


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = parse_bbox(prediction)
    entry = _metric_entry(doc, prediction, parsed)
    overlay_path = save_prediction_overlay(doc, parsed, entry["iou"])
    metric_names = (
        "bbox_iou",
        "bbox_acc_0_25",
        "bbox_acc_0_5",
        "bbox_acc_0_75",
        "bbox_parse_success",
        "reference_bbox_iou",
        "target_bbox_iou",
    )
    output = {metric: dict(entry) for metric in metric_names}
    output["submission"] = {
        **entry,
        "question_prompt": doc_to_text(doc),
        "img_path": doc.get("img_path"),
        "debug_overlay_path": str(overlay_path) if overlay_path else None,
    }
    return output


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_bbox_iou(results):
    values = [float(row["iou"]) for row in results]
    eval_logger.info("COMFORT bbox mean IoU: {:.4f} across {} objects.", _mean(values), len(values))
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row.get("object_label", "<missing>"))].append(float(row["iou"]))
    eval_logger.info(
        "COMFORT bbox IoU by object label: {}",
        {label: {"mean_iou": _mean(scores), "count": len(scores)} for label, scores in sorted(grouped.items())},
    )
    return _mean(values)


def aggregate_bbox_acc_0_25(results):
    return _mean(float(row["bbox_acc_0_25"]) for row in results)


def aggregate_bbox_acc_0_5(results):
    return _mean(float(row["bbox_acc_0_5"]) for row in results)


def aggregate_bbox_acc_0_75(results):
    return _mean(float(row["bbox_acc_0_75"]) for row in results)


def aggregate_bbox_parse_success(results):
    return _mean(float(row["parse_success"]) for row in results)


def aggregate_reference_bbox_iou(results):
    return _mean(float(row["iou"]) for row in results if row.get("object_role") == "reference")


def aggregate_target_bbox_iou(results):
    return _mean(float(row["iou"]) for row in results if row.get("object_role") == "target")


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(
        f"comfort_multi_3d_bbox_prediction_{model}.json", args
    )
    report = {
        "dataset": "COMFORT_Multi_3D",
        "task": "comfort_multi_3d_bbox_prediction",
        "coordinate_format": "normalized [x_min, y_min, x_max, y_max] in [0,1000]",
        "num_records": len(results),
        "mean_iou": _mean(float(row["iou"]) for row in results),
        "parse_success_rate": _mean(float(row["parse_success"]) for row in results),
        "records": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    eval_logger.info("COMFORT bbox prediction records saved to {}.", path)
