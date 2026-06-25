"""
Utilities for evaluating local Kubric MOVi-A 3D spatial reasoning exports in LMMS-eval.
"""

import io
import json
import os
import re
import string
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name


TASK_FAMILIES = [
    "camera_relative_position",
    "camera_distance",
    "height_relative_3d",
    "object_centric_relative_position",
    "object_centric_relative_position_multi",
]

DIFFICULTIES = ["easy", "hard", "very_hard"]

COLOR_LABELS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]

# Map Kubric MOVi-A color labels to distinct LMMS-eval colors for bounding box overlays.
COLOR_MAP = {
    "blue": "black",
    "brown": "darkblue",
    "cyan": "orange",
    "gray": "pink",
    "green": "magenta",
    "purple": "yellow",
    "red": "cyan",
    "yellow": "green",
}

KUBRIC_BBOX_IOU_THRESHOLDS = {
    "bbox_iou_acc_0_1": 0.1,
    "bbox_iou_acc_0_25": 0.25,
    "bbox_iou_acc_0_5": 0.5,
    "bbox_iou_acc_0_75": 0.75,
}

KUBRIC_BBOX_PIXEL_THRESHOLDS = {
    "bbox_center_acc_px10": 10.0,
    "bbox_center_acc_px25": 25.0,
    "bbox_center_acc_px50": 50.0,
    "bbox_center_acc_px100": 100.0,
}


def _get_overlay_color_name(color_name: str) -> str:
    return COLOR_MAP.get(color_name, color_name)


def _normalize_coord_1_to_1000(value: float) -> int:
    value = max(0.0, min(1.0, float(value)))
    return max(1, min(1000, int(round(value * 999)) + 1))


def _get_bbox_xyxy_1000(obj: dict) -> Optional[list[int]]:
    bbox_norm = obj.get("bbox_2d_norm")
    if bbox_norm and len(bbox_norm) == 4:
        ymin, xmin, ymax, xmax = bbox_norm
        return [
            _normalize_coord_1_to_1000(xmin),
            _normalize_coord_1_to_1000(ymin),
            _normalize_coord_1_to_1000(xmax),
            _normalize_coord_1_to_1000(ymax),
        ]

    bbox_pixels = obj.get("bbox_2d_xyxy_pixels")
    if bbox_pixels and len(bbox_pixels) == 4:
        return [int(v) for v in bbox_pixels]
    return None


def _get_bbox_center_1000(obj: dict) -> Optional[list[int]]:
    bbox_norm = obj.get("bbox_2d_norm")
    if bbox_norm and len(bbox_norm) == 4:
        ymin, xmin, ymax, xmax = bbox_norm
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        return [
            _normalize_coord_1_to_1000(center_x),
            _normalize_coord_1_to_1000(center_y),
        ]

    bbox_pixels = obj.get("bbox_2d_xyxy_pixels")
    if bbox_pixels and len(bbox_pixels) == 4:
        center_x = (bbox_pixels[0] + bbox_pixels[2]) // 2
        center_y = (bbox_pixels[1] + bbox_pixels[3]) // 2
        return [int(center_x), int(center_y)]
    return None


def _load_image(value):
    if value is None:
        return None
    if hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") and os.path.isfile(value["path"]):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and os.path.isfile(value):
        return Image.open(value).convert("RGB")
    return None


def _get_image_path(doc) -> str:
    for key in ("img_path", "image"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and value.get("path"):
            return value["path"]
    return ""


def _get_visible_objects(doc) -> list[dict]:
    objects = doc.get("visible_objects", [])
    return objects if isinstance(objects, list) else []


def _get_object_map(doc) -> dict[str, dict]:
    return {
        obj.get("name"): obj
        for obj in _get_visible_objects(doc)
        if isinstance(obj, dict) and obj.get("name")
    }


def _get_queried_object_names(doc) -> list[str]:
    items = doc.get("bbox_items", [])
    if isinstance(items, list) and items:
        return [item for item in items if item]
    object_map = _get_object_map(doc)
    return list(object_map.keys())


def _get_object_record(doc, object_name: str) -> Optional[dict]:
    return _get_object_map(doc).get(object_name)


def _get_bbox_prompt_object_names(doc) -> list[str]:
    items = doc.get("bbox_items", [])
    if isinstance(items, list) and items:
        return [item for item in items if item]

    answer_text = _ground_truth_answer_text(doc)
    if answer_text and _get_object_record(doc, answer_text):
        return [answer_text]
    if len(_get_visible_objects(doc)) == 1:
        only_object = _get_visible_objects(doc)[0]
        if isinstance(only_object, dict) and only_object.get("name"):
            return [only_object["name"]]
    return []


def _get_options(doc) -> dict[str, str]:
    options = {}
    for cand in string.ascii_uppercase[:4]:
        if cand in doc and doc[cand] is not None:
            value = doc[cand]
            if isinstance(value, str) and value.lower() == "nan":
                continue
            if pd.isna(value):
                continue
            options[cand] = value
    return options


def _ground_truth_answer_text(doc) -> str:
    answer = str(doc.get("answer", "")).strip()
    options = _get_options(doc)
    if answer in options:
        return str(options[answer]).strip()
    return answer


def _format_bbox_context(doc) -> str:
    if os.getenv("LMMS_EVAL_INCLUDE_LOCATION_TEXT", "0") != "1":
        return ""

    prompt_lines = []
    object_names = _get_queried_object_names(doc)
    if object_names:
        prompt_lines.append(f"The image contains the following queried objects: {', '.join(object_names)}.")
        prompt_lines.append("Bounding boxes are given as normalized 1-1000 integer coordinates [x_min, y_min, x_max, y_max].")

    frame_entries = []
    for object_name in object_names:
        obj = _get_object_record(doc, object_name)
        if not obj:
            continue
        bbox = _get_bbox_xyxy_1000(obj)
        if not bbox:
            continue
        frame_entries.append(f"{object_name}[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

    if frame_entries:
        prompt_lines.append("Frame 0: " + ", ".join(frame_entries))

    return "\n".join(prompt_lines) + ("\n" if prompt_lines else "")


def _format_bbox_dimensions_prompt(doc) -> str:
    object_names = _get_queried_object_names(doc)
    if not object_names:
        return ""

    prompt_lines = ["The queried objects have the following bounding box(es):"]
    found_any = False
    for object_name in object_names:
        obj = _get_object_record(doc, object_name)
        if not obj:
            continue
        bbox = _get_bbox_xyxy_1000(obj)
        if not bbox:
            continue
        prompt_lines.append(f"{object_name}: 1[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        found_any = True
    return "\n".join(prompt_lines) if found_any else ""


def _format_centroids_prompt(doc) -> str:
    object_names = _get_queried_object_names(doc)
    if not object_names:
        return ""

    prompt_lines = ["The queried objects have the following bounding box center point(s):"]
    found_any = False
    for object_name in object_names:
        obj = _get_object_record(doc, object_name)
        if not obj:
            continue
        center = _get_bbox_center_1000(obj)
        if not center:
            continue
        prompt_lines.append(f"{object_name}: 1[{center[0]}, {center[1]}]")
        found_any = True
    return "\n".join(prompt_lines) if found_any else ""


def _format_bbox_colors_prompt(doc) -> str:
    object_names = _get_queried_object_names(doc)
    if not object_names:
        return ""

    prompt_lines = ["The queried objects have bounding boxes with the following colors:"]
    found_any = False
    for object_name in object_names:
        obj = _get_object_record(doc, object_name)
        if not obj:
            continue
        color_name = obj.get("color_name")
        if not color_name:
            continue
        prompt_lines.append(f"{object_name}: {_get_overlay_color_name(color_name)}")
        found_any = True
    return "\n".join(prompt_lines) if found_any else ""


def _build_bbox_prompt(doc, lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer the above in the bbox format: [x_min, y_min, x_max, y_max]. Each in the range [0, 1000].\n",
    )

    prompt = pre_prompt
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"

    question = doc.get("question", "").strip()
    if question:
        prompt += f"Question: {question}\n"

    object_names = _get_bbox_prompt_object_names(doc)
    if object_names:
        prompt += "Look at the image and find the bbox(es) of:\n"
        for idx, object_name in enumerate(object_names, start=1):
            prompt += f"{idx}. {object_name}\n"
        prompt += "\n"

    prompt += post_prompt
    return prompt


def _format_gt_bbox_answers(doc) -> dict[str, list[int]]:
    formatted_answers = {}
    for object_name in _get_bbox_prompt_object_names(doc):
        obj = _get_object_record(doc, object_name)
        bbox = _get_bbox_xyxy_1000(obj) if obj else None
        if bbox:
            formatted_answers[object_name] = bbox
    return formatted_answers


def _coerce_bbox_to_1000(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    values = [float(v) for v in bbox]
    if max(values) <= 1.0 and min(values) >= 0.0:
        return [v * 1000.0 for v in values]
    if max(values) <= 999.0 and min(values) >= 0.0:
        return [max(0.0, min(1000.0, v)) for v in values]
    return [max(0.0, min(1000.0, v)) for v in values]


def _sanitize_bbox_xyxy(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = _coerce_bbox_to_1000(bbox)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [left, top, right, bottom]


def _extract_bbox_sequences(text: str) -> list[list[float]]:
    if not text:
        return []

    pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    matches = re.findall(pattern, text)
    return [[float(match[i]) for i in range(4)] for match in matches]


def _parse_predicted_bboxes(text: str, object_names: list[str]) -> dict[str, list[float]]:
    predictions = {}
    if not text or not object_names:
        return predictions

    named_patterns = [
        r"(?im)^\s*(?:\d+\.\s*)?(?P<name>[^:\n]+?)\s*:\s*\[(?P<bbox>[^\]]+)\]",
        r"(?im)^\s*(?P<name>[^:\n]+?)\s*-\s*\[(?P<bbox>[^\]]+)\]",
    ]
    object_lookup = {name.lower(): name for name in object_names}

    for pattern in named_patterns:
        for match in re.finditer(pattern, text):
            name_key = match.group("name").strip().lower()
            if name_key not in object_lookup:
                continue
            bbox_values = _extract_bbox_sequences(f"[{match.group('bbox')}]")
            if bbox_values:
                predictions[object_lookup[name_key]] = _sanitize_bbox_xyxy(bbox_values[0])

    remaining_names = [name for name in object_names if name not in predictions]
    if remaining_names and not predictions:
        raw_boxes = _extract_bbox_sequences(text)
        for object_name, bbox in zip(remaining_names, raw_boxes):
            predictions.setdefault(object_name, _sanitize_bbox_xyxy(bbox))

    return predictions


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _compute_center_distance(box1: list[float], box2: list[float]) -> float:
    center_1_x = (box1[0] + box1[2]) / 2.0
    center_1_y = (box1[1] + box1[3]) / 2.0
    center_2_x = (box2[0] + box2[2]) / 2.0
    center_2_y = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(center_1_x - center_2_x, center_1_y - center_2_y))


def _score_bbox_prediction(doc, pred_text: str) -> dict:
    gt_answers = _format_gt_bbox_answers(doc)
    object_names = list(gt_answers.keys())
    pred_answers = _parse_predicted_bboxes(pred_text, object_names)

    per_object = []
    for object_name in object_names:
        gt_bbox = [float(v) for v in gt_answers[object_name]]
        pred_bbox = pred_answers.get(object_name, [0.0, 0.0, 0.0, 0.0])
        iou = _compute_iou(gt_bbox, pred_bbox)
        center_distance = _compute_center_distance(gt_bbox, pred_bbox)
        per_object.append(
            {
                "object_name": object_name,
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_bbox,
                "iou": iou,
                "center_distance": center_distance,
            }
        )

    if not per_object:
        return {
            "mean_iou": 0.0,
            "mean_center_distance": 0.0,
            "per_object": [],
            **{metric_name: 0.0 for metric_name in KUBRIC_BBOX_IOU_THRESHOLDS},
            **{metric_name: 0.0 for metric_name in KUBRIC_BBOX_PIXEL_THRESHOLDS},
        }

    mean_iou = sum(item["iou"] for item in per_object) / len(per_object)
    mean_center_distance = sum(item["center_distance"] for item in per_object) / len(per_object)

    scores = {
        "mean_iou": mean_iou,
        "mean_center_distance": mean_center_distance,
        "per_object": per_object,
    }
    for metric_name, threshold in KUBRIC_BBOX_IOU_THRESHOLDS.items():
        scores[metric_name] = sum(item["iou"] >= threshold for item in per_object) / len(per_object)
    for metric_name, threshold in KUBRIC_BBOX_PIXEL_THRESHOLDS.items():
        scores[metric_name] = sum(item["center_distance"] <= threshold for item in per_object) / len(per_object)
    return scores


def _build_text_only_position_hint(doc, gt_answer: str) -> str:
    task_family = doc.get("task_family")
    metadata = doc.get("task_metadata", {}) or {}
    relation = metadata.get("relation")
    queried = _get_queried_object_names(doc)

    if task_family == "camera_relative_position" and len(queried) >= 2:
        obj_a, obj_b = queried[:2]
        if relation == "left":
            return f"The correct answer is the object that is more to the left. Between these two, that is {gt_answer}."
        if relation == "right":
            return f"The correct answer is the object that is more to the right. Between these two, that is {gt_answer}."
        if relation == "above":
            return f"The correct answer is the object that is higher in the image. Between these two, that is {gt_answer}."
        if relation == "below":
            return f"The correct answer is the object that is lower in the image. Between these two, that is {gt_answer}."

    if task_family == "height_relative_3d" and len(queried) >= 2:
        target = relation or ("higher" if "higher" in doc.get("question", "") else "lower")
        return f"Considering the real-world 3D locations, the {gt_answer} is {target}."

    if task_family == "camera_distance" and len(queried) >= 2:
        target = relation or ("closer" if "closer" in doc.get("question", "") else "farther")
        return f"Relative to the camera, the {gt_answer} is {target}."

    if task_family == "object_centric_relative_position":
        anchor = metadata.get("anchor_object")
        target_object = metadata.get("target_object")
        if anchor and target_object and relation:
            return f"From the {anchor}'s camera-facing viewpoint, the {target_object} is to the {relation} of the {anchor}."

    if task_family == "object_centric_relative_position_multi":
        anchor = metadata.get("anchor_object")
        if anchor and relation:
            phrase = {
                "left": "to the left of",
                "right": "to the right of",
                "front": "in front of",
                "behind": "behind",
            }.get(relation, relation)
            return f"From the {anchor}'s camera-facing viewpoint, the correct answer is the object farthest {phrase} the {anchor}: {gt_answer}."

    return f"The correct answer is: {gt_answer}."


def _build_GT_help(doc, answer_format: str) -> str:
    assert answer_format in {"0", "1", "2", "3", "4", "5", "6", "7", "8"}, "Invalid GT help format option"

    gt_answer = _ground_truth_answer_text(doc)
    gt_object = _get_object_record(doc, gt_answer)

    if not gt_answer:
        return ""
    if answer_format == "1":
        return f"The correct answer is: {gt_answer}."
    if answer_format == "2":
        if not gt_object:
            return ""
        bbox = _get_bbox_xyxy_1000(gt_object)
        if not bbox:
            return ""
        return f"The correct answer is the object with bounding box(es): 1[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
    if answer_format == "3":
        if not gt_object or not gt_object.get("color_name"):
            return ""
        overlay_color = _get_overlay_color_name(gt_object["color_name"])
        return f"The correct answer is the object with bounding box in the color: {overlay_color}."
    if answer_format == "4":
        return _build_text_only_position_hint(doc, gt_answer)
    if answer_format == "5":
        return _format_bbox_dimensions_prompt(doc)
    if answer_format == "6":
        return _format_bbox_colors_prompt(doc)
    if answer_format == "7":
        if not gt_object:
            return ""
        center = _get_bbox_center_1000(gt_object)
        if not center:
            return ""
        return f"The correct answer is the object(s) at: 1[{center[0]}, {center[1]}]"
    if answer_format == "8":
        return _format_centroids_prompt(doc)
    return ""


def _draw_bbox_overlays(doc, img: Image.Image) -> Image.Image:
    help_mode = os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0")
    if help_mode not in {"2", "3", "5", "6"}:
        return img

    draw = ImageDraw.Draw(img)
    object_names = []
    if help_mode in {"2", "3"}:
        gt_answer = _ground_truth_answer_text(doc)
        if gt_answer:
            object_names = [gt_answer]
    elif help_mode in {"5", "6"}:
        object_names = _get_queried_object_names(doc)

    for object_name in object_names:
        obj = _get_object_record(doc, object_name)
        if not obj:
            continue
        bbox = obj.get("bbox_2d_xyxy_pixels")
        if not bbox or len(bbox) != 4:
            continue
        color = _get_overlay_color_name(obj.get("color_name", "red"))
        draw.rectangle(bbox, outline=color, width=3)

    return img


def doc_to_visual(doc):
    image = _load_image(doc.get("image"))
    if image is None:
        image = _load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image found for sample {doc.get('index', doc.get('qid', 'unknown'))}")
    if os.getenv("LMMS_MASK_IMAGE", "0") == "1":
        image = Image.fromarray(np.array(image) * 0)
    return [_draw_bbox_overlays(doc, image)]


def bbox_doc_to_visual(doc):
    image = _load_image(doc.get("image"))
    if image is None:
        image = _load_image(doc.get("img_path"))
    if image is None:
        raise FileNotFoundError(f"No usable image found for sample {doc.get('index', doc.get('qid', 'unknown'))}")
    return [image]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Please select the correct answer directly from the options above and include the option letter (A, B, C, or D).\n",
    )

    question = doc["question"]
    options = _get_options(doc)
    bbox_context = _format_bbox_context(doc)
    if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in {"1", "2", "3", "4", "5", "6", "7", "8"}:
        gt_help = _build_GT_help(doc, os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0"))
        if gt_help:
            post_prompt = gt_help + "\n" + post_prompt

    prompt = ""
    if pre_prompt:
        prompt += pre_prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
    if bbox_context:
        prompt += bbox_context
    prompt += f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}. {value}\n"
    if post_prompt:
        prompt += post_prompt
    return prompt


def bbox_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _build_bbox_prompt(doc, lmms_eval_specific_kwargs)


def doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None

    text = text.strip()
    patterns = [
        r"^([A-D])[\.\s\)]",
        r"^([A-D])$",
        r"[Aa]nswer[:\s]+([A-D])",
        r"[Tt]he answer is[:\s]+([A-D])",
        r"[Mm]y answer is[:\s]+([A-D])",
        r"\(([A-D])\)",
        r"([A-D])\.",
        r"\b([A-D])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def process_results(doc, results):
    pred = results[0].strip()
    pred_answer = extract_answer(pred)
    gt_answer = str(doc.get("answer", "")).strip()
    score = 1.0 if pred_answer == gt_answer else 0.0

    task_family = doc.get("task_family", "unknown")
    difficulty = doc.get("difficulty", "unknown")
    index = doc.get("index", doc.get("qid"))
    qid = doc.get("qid", index)

    base_entry = {
        "index": index,
        "qid": qid,
        "score": score,
        "task_family": task_family,
        "difficulty": difficulty,
    }

    result_dict = {
        "vanilla_accuracy": base_entry,
        "submission": {
            "index": index,
            "qid": qid,
            "task_family": task_family,
            "difficulty": difficulty,
            "question_prompt": doc_to_text(doc),
            "img_path": _get_image_path(doc),
            "prediction": pred,
            "parsed_prediction": pred_answer,
            "gt_answer": gt_answer,
            "gt_answer_text": _ground_truth_answer_text(doc),
            "Help_Prompt": _build_GT_help(doc, os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0")),
            "score": score,
        },
    }

    for family in TASK_FAMILIES:
        result_dict[f"{family}_accuracy"] = base_entry
    for difficulty_name in DIFFICULTIES:
        result_dict[f"{difficulty_name}_accuracy"] = base_entry
    for family in TASK_FAMILIES:
        for difficulty_name in DIFFICULTIES:
            result_dict[f"{family}_{difficulty_name}_accuracy"] = base_entry

    return result_dict


def bbox_process_results(doc, results):
    pred = results[0].strip() if results else ""
    scores = _score_bbox_prediction(doc, pred)
    index = doc.get("index", doc.get("qid"))
    qid = doc.get("qid", index)

    base_entry = {
        "index": index,
        "qid": qid,
        "score": scores["mean_iou"],
        "mean_iou": scores["mean_iou"],
        "mean_center_distance": scores["mean_center_distance"],
        "per_object": scores["per_object"],
    }
    for metric_name in KUBRIC_BBOX_IOU_THRESHOLDS:
        base_entry[metric_name] = scores[metric_name]
    for metric_name in KUBRIC_BBOX_PIXEL_THRESHOLDS:
        base_entry[metric_name] = scores[metric_name]

    result_dict = {
        "submission": {
            "index": index,
            "qid": qid,
            "question_prompt": _build_bbox_prompt(doc),
            "img_path": _get_image_path(doc),
            "gt_answers": _format_gt_bbox_answers(doc),
            "parsed_prediction": {
                item["object_name"]: item["pred_bbox"] for item in scores["per_object"]
            },
            "model_answer": pred,
            "per_object": scores["per_object"],
            "mean_iou": scores["mean_iou"],
            "mean_center_distance": scores["mean_center_distance"],
        },
        "bbox_mean_iou": base_entry,
    }

    for metric_name in KUBRIC_BBOX_IOU_THRESHOLDS:
        result_dict[metric_name] = base_entry
    for metric_name in KUBRIC_BBOX_PIXEL_THRESHOLDS:
        result_dict[metric_name] = base_entry

    return result_dict


def _aggregate_accuracy(results, task_family=None, difficulty=None):
    filtered = [
        r["score"]
        for r in results
        if (task_family is None or r.get("task_family") == task_family)
        and (difficulty is None or r.get("difficulty") == difficulty)
    ]
    return sum(filtered) / len(filtered) if filtered else 0.0


def aggregate_vanilla_accuracy(results):
    if not results:
        return 0.0

    overall = _aggregate_accuracy(results)
    eval_logger.info(f"Kubric MOVi-A overall: {overall * 100:.2f}%")
    for family in TASK_FAMILIES:
        family_acc = _aggregate_accuracy(results, task_family=family)
        eval_logger.info(f"Kubric MOVi-A {family}: {family_acc * 100:.2f}%")
    for difficulty in DIFFICULTIES:
        diff_acc = _aggregate_accuracy(results, difficulty=difficulty)
        eval_logger.info(f"Kubric MOVi-A {difficulty}: {diff_acc * 100:.2f}%")

    return overall


def _get_submission_model_tag(args) -> str:
    model_name = getattr(args, "model", "") or ""
    if model_name:
        return sanitize_model_name(model_name)

    model_args = getattr(args, "model_args", "") or ""
    for key in ("pretrained", "model_name", "model_path"):
        match = re.search(rf"(?:^|,){key}=([^,]+)", model_args)
        if match:
            return sanitize_model_name(match.group(1).strip())

    return "unknown_model"


def aggregate_results_for_submission(results, args):
    model_tag = _get_submission_model_tag(args)
    path = generate_submission_file(f"kubric_movi_a_combined_records_{model_tag}.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    eval_logger.info(f"Combined records saved to {path}.")


def bbox_aggregate_results_for_submission(results, args):
    model_tag = _get_submission_model_tag(args)
    path = generate_submission_file(f"kubric_movi_a_bbox_predictions_{model_tag}.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    eval_logger.info(f"BBox predictions saved to {path}.")


def _aggregate_bbox_metric(results, metric_name: str) -> float:
    if not results:
        return 0.0
    values = [float(r.get(metric_name, 0.0)) for r in results]
    return sum(values) / len(values) if values else 0.0


def aggregate_bbox_mean_iou(results):
    return _aggregate_bbox_metric(results, "mean_iou")


def aggregate_bbox_iou_acc_0_1(results):
    return _aggregate_bbox_metric(results, "bbox_iou_acc_0_1")


def aggregate_bbox_iou_acc_0_25(results):
    return _aggregate_bbox_metric(results, "bbox_iou_acc_0_25")


def aggregate_bbox_iou_acc_0_5(results):
    return _aggregate_bbox_metric(results, "bbox_iou_acc_0_5")


def aggregate_bbox_iou_acc_0_75(results):
    return _aggregate_bbox_metric(results, "bbox_iou_acc_0_75")


def aggregate_bbox_center_acc_px10(results):
    return _aggregate_bbox_metric(results, "bbox_center_acc_px10")


def aggregate_bbox_center_acc_px25(results):
    return _aggregate_bbox_metric(results, "bbox_center_acc_px25")


def aggregate_bbox_center_acc_px50(results):
    return _aggregate_bbox_metric(results, "bbox_center_acc_px50")


def aggregate_bbox_center_acc_px100(results):
    return _aggregate_bbox_metric(results, "bbox_center_acc_px100")


def aggregate_camera_relative_position_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_relative_position")


def aggregate_camera_distance_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_distance")


def aggregate_height_relative_3d_accuracy(results):
    return _aggregate_accuracy(results, task_family="height_relative_3d")


def aggregate_object_centric_relative_position_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position")


def aggregate_object_centric_relative_position_multi_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position_multi")


def aggregate_easy_accuracy(results):
    return _aggregate_accuracy(results, difficulty="easy")


def aggregate_hard_accuracy(results):
    return _aggregate_accuracy(results, difficulty="hard")


def aggregate_very_hard_accuracy(results):
    return _aggregate_accuracy(results, difficulty="very_hard")


def aggregate_camera_relative_position_easy_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_relative_position", difficulty="easy")


def aggregate_camera_relative_position_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_relative_position", difficulty="hard")


def aggregate_camera_relative_position_very_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_relative_position", difficulty="very_hard")


def aggregate_camera_distance_easy_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_distance", difficulty="easy")


def aggregate_camera_distance_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_distance", difficulty="hard")


def aggregate_camera_distance_very_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="camera_distance", difficulty="very_hard")


def aggregate_height_relative_3d_easy_accuracy(results):
    return _aggregate_accuracy(results, task_family="height_relative_3d", difficulty="easy")


def aggregate_height_relative_3d_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="height_relative_3d", difficulty="hard")


def aggregate_height_relative_3d_very_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="height_relative_3d", difficulty="very_hard")


def aggregate_object_centric_relative_position_easy_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position", difficulty="easy")


def aggregate_object_centric_relative_position_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position", difficulty="hard")


def aggregate_object_centric_relative_position_very_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position", difficulty="very_hard")


def aggregate_object_centric_relative_position_multi_easy_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position_multi", difficulty="easy")


def aggregate_object_centric_relative_position_multi_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position_multi", difficulty="hard")


def aggregate_object_centric_relative_position_multi_very_hard_accuracy(results):
    return _aggregate_accuracy(results, task_family="object_centric_relative_position_multi", difficulty="very_hard")
