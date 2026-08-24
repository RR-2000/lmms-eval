"""
3DSRBench utilities adapted for local parquet exports with optional bbox prompt text.
"""

import json
import os
import random
import re
import string
from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.utils import sanitize_model_name

# Category mapping from detailed categories to main categories
CATEGORY_MAPPING = {
    "height_higher": "height",
    "location_above": "location",
    "location_closer_to_camera": "location",
    "location_next_to": "location",
    "orientation_in_front_of": "orientation",
    "orientation_on_the_left": "orientation",
    "orientation_viewpoint": "orientation",
    "multi_object_closer_to": "multi_object",
    "multi_object_facing": "multi_object",
    "multi_object_viewpoint_towards_object": "multi_object",
    "multi_object_parallel": "multi_object",
    "multi_object_same_direction": "multi_object",
}

# Semantic mapping for objects in the 3DSRBench dataset

map_subject = "large red cube"
map_object1 = "small blue sphere"
map_object2 = "medium green cylinder"

MAIN_CATEGORIES = ["height", "location", "orientation", "multi_object"]

DEFAULT_COLOR_ORDER = [
    "green",
    "red",
    "blue",
    "yellow",
    "purple",
    "black",
    "cyan",
    "magenta",
    "orange",
    "lime",
    "pink",
    "teal",
    "navy",
    "maroon",
]
IMG_BASE="/home/ramanathan/data/3dsrbench_data/images/coco_images"

# Direction-versus-object diagnostic -------------------------------------------------
#
# 3DSRBench's ``multi_object_viewpoint_towards_object`` items ask which side of a
# subject points towards a target object.  The diagnostic below keeps that exact
# relation and makes a paired object-answer version: given the relevant side of
# the subject, select the object it points towards.  Distractors are other
# objects queried for the same underlying COCO image, rather than invented
# names, so both variants remain grounded in the image.
DIRECTION_OBJECT_QTYPE = "multi_object"
DIRECTION_OBJECT_RELATION = "viewpoint towards object"
DIRECTION_OBJECT_DIRECTIONS = {"left", "right", "front", "back"}
DIRECTION_OBJECT_SAMPLE_SEED = "3dsrbench_direction_object_v1"

# Used only when a source image does not provide enough distinct queried
# objects for the inverse object-choice prompt.  The pools make the synthetic
# distractors semantically consistent with the target rather than padding with
# arbitrary labels.  They deliberately contain broad COCO-style object names,
# so this remains deterministic and requires no external model at evaluation
# time.
DIRECTION_OBJECT_THEMATIC_POOLS = {
    "vehicle": ["car", "bus", "van", "bicycle", "motorcycle", "boat"],
    "signage": ["traffic sign", "street sign", "parking sign", "billboard", "traffic light"],
    "person": ["person", "man", "woman", "pedestrian", "child"],
    "animal": ["dog", "cat", "horse", "bird", "cow", "elephant"],
    "furniture": ["chair", "table", "couch", "bed", "cabinet", "lamp"],
    "kitchen": ["refrigerator", "oven", "microwave", "sink", "bottle", "cup"],
    "street": ["fire hydrant", "parking meter", "bench", "trash can", "bus stop"],
    "generic": ["nearby object", "background object", "scene landmark", "other item"],
}

def _parse_json_maybe(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_text(text) -> str:
    return " ".join(str(text).strip().lower().split())


def _get_valid_boxes(boxes) -> list[list[float]]:
    if not isinstance(boxes, list):
        return []
    return [box for box in boxes if isinstance(box, list) and len(box) == 4]


def _get_all_items_bbox_entries(doc) -> list[tuple[str, list[list[float]], str]]:
    bbox_data = _parse_json_maybe(doc.get("bboxes_by_item", {}))
    if not isinstance(bbox_data, dict) or not bbox_data:
        return []

    color_data = _parse_json_maybe(doc.get("bbox_colors_by_item", {}))
    entries = []
    for item_name, frame_map in bbox_data.items():
        frame_map = _parse_json_maybe(frame_map)
        if not isinstance(frame_map, dict):
            return []
        boxes = _get_valid_boxes(frame_map.get("0", []))
        # if not boxes:
        if len(boxes) != 0:
            return []

        color_name = DEFAULT_COLOR_ORDER[len(entries) % len(DEFAULT_COLOR_ORDER)]
        if isinstance(color_data, dict):
            item_color = _parse_json_maybe(color_data.get(item_name, {}))
            if isinstance(item_color, dict) and item_color.get("name"):
                color_name = item_color["name"]

        entries.append((item_name, boxes, color_name))

    return entries


def _load_image_from_doc(doc, use_overlay: bool = False):
    if "image" in doc and hasattr(doc["image"], "convert"):
        return doc["image"].convert("RGB")

    path_candidates = []
    if use_overlay:
        path_candidates.append(doc.get("overlay_png_path"))
    path_candidates.extend(
        [
            doc.get("resolved_img_path"),
            doc.get("img_path"),
            doc.get("image"),
            doc.get("image_url"),
        ]
    )
    # if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["2", "3"]:
    #     for path_idx in range(len(path_candidates)):
    #         path_candidates[path_idx] = path_candidates[path_idx].replace("3DSR_fixed", "3DSR")
    for image_path in path_candidates:
        if not image_path:
            continue
        if os.path.isfile(image_path):
            drawn = False
            img = Image.open(image_path).convert("RGB")
            gt_answer = None
            gt_bbox = []
            gt_bbox_color = "red"
            if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["2", "3"]:
                # Draw GT bounding boxes on the image for help
                gt_answer = doc.get(doc.get("answer", ""))
                gt_bbox = _parse_json_maybe(doc.get("bboxes_by_item", "")).get(gt_answer, {}).get("0", [])
                gt_bbox_color = _parse_json_maybe(doc.get("bbox_colors_by_item", "")).get(gt_answer, {}).get("name", "red")
                if len(gt_bbox) == 1:
                    draw = ImageDraw.Draw(img)
                    width, height = img.size
                    # print(f'DEBUG: GT {doc.get("index")} bboxes {doc.get("qtype")}_{doc.get("relation")} for answer: "{gt_answer}": {gt_bbox}')
                    for box in gt_bbox:
                        if len(box) != 4:
                            continue
                        x, y, w, h = box
                        draw.rectangle([x*width, y*height, (x + w)*width, (y + h)*height], outline=gt_bbox_color, width=3)
                        drawn = True
            if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["5", "6"]:
                bbox_entries = _get_all_items_bbox_entries(doc)
                if bbox_entries:
                    draw = ImageDraw.Draw(img)
                    width, height = img.size
                    for _, boxes, box_color in bbox_entries:
                        for box in boxes:
                            x, y, w, h = box
                            draw.rectangle([x*width, y*height, (x + w)*width, (y + h)*height], outline=box_color, width=3)
                            drawn = True
            # Save image for debugging
            
            if "flip" in str(doc.get("index")):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            
            if os.getenv("LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR", ""):
                save_path = os.path.join(os.getenv("LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR"), "TMP_IMGS", f"{doc.get('qid', doc.get('index', 'unknown'))}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)
                if drawn:
                    print(f"Debug: Drew bbox overlays on image for doc_id={doc.get('index')} at {save_path}")
            
            if os.getenv("LMMS_MASK_IMAGE", "0") == "1":
                img = Image.fromarray(np.array(img) * 0)
            return img
        
        # if image_path is a URL, try to download it
        if image_path.startswith("http://") or image_path.startswith("https://"):
            img_path = os.path.join(IMG_BASE, image_path.split("/")[-2], image_path.split("/")[-1])
            img = Image.open(img_path).convert("RGB")

            if "flip" in str(doc.get("index")):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if os.getenv("LMMS_MASK_IMAGE", "0") == "1":
                img = Image.fromarray(np.array(img) * 0)
            return img
                
    raise FileNotFoundError("No image path found in document.")


def _get_relation_category(doc) -> str:
    if doc.get("category"):
        return doc["category"]

    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", "")).replace(" ", "_")
    category = f"{qtype}_{relation}".strip("_")
    if category in CATEGORY_MAPPING:
        return category
    return "unknown"


def _build_fallback_question(doc) -> str:
    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", ""))
    subject = doc.get("subject")
    object1 = doc.get("object1", doc.get("object_1"))
    object2 = doc.get("object2", doc.get("object_2"))

    templates = {
        ("height", "higher"): f"Consider the real-world 3D location of the objects. Which object is higher in the image: {subject} or {object1}?",
        ("location", "above"): f"Consider the real-world 3D location of the objects. Is {object1} above {subject}?",
        ("location", "closer to camera"): f"Consider the real-world 3D location of the objects. Which object is closer to the camera: {subject} or {object1}?",
        ("location", "next to"): f"Consider the real-world 3D location of the objects. Is {subject} next to {object1}?",
        ("orientation", "in front of"): f"From {object1}'s perspective, is {subject} in front of {object1}?",
        ("orientation", "on the left"): f"From {object1}'s perspective, is {subject} on the left of {object1}?",
        ("orientation", "viewpoint"): f"From the camera's viewpoint, which direction of {subject} is facing the camera?",
        ("multi_object", "closer to"): f"Which object is {subject} closer to: {object1} or {object2}?",
        ("multi_object", "facing"): f"Which object is {subject} facing: {object1} or {object2}?",
        ("multi_object", "parallel"): f"Is {subject} parallel to {object1}?",
        ("multi_object", "same direction"): f"Are {subject} and {object1} facing the same direction?",
        ("multi_object", "viewpoint towards object"): f"Relative to its current viewpoint, which direction should {object1} turn to face {subject}?",
    }
    return templates.get((qtype, relation), f"What is the correct answer about {subject}?")


def _build_text_only_position_hint(doc, gt) -> str:
    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", ""))
    subject = doc.get("subject")
    object1 = doc.get("object1", doc.get("object_1"))
    object2 = doc.get("object2", doc.get("object_2"))

    hints = {
        ("height", "higher"): f"The {gt} is {'higher' if 'higher' in str(doc.get('original_question', '')) else 'lower'}.",
        # f"Determine the relative vertical positions of {subject} and {object1}, then answer which one is higher in the image.",
        ("location", "above"): f"The {subject} is directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}." if gt == "yes" else f"The {subject} is not directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}.",
        # f"Determine whether {subject} is vertically positioned above {object1} in the image.",
        ("location", "closer to camera"): f"The {gt} is closer to the camera." if "closer to the camera" in str(doc.get('original_question', '')) else f"The {gt} is farther from the camera.",
        # f"Determine the relative depth of {subject} and {object1}, then answer which object is closer to the camera.",
        ("location", "next to"): f"The {subject} is far from {object1}." if gt == 'far away from each other' else f"The {subject} is close to {object1}.",
        # f"Determine whether {subject} is positioned adjacent to {object1} in the image.",
        ("orientation", "in front of"): f"The {subject} is in front of {object1} from {object1}'s perspective." if gt == 'in front of' else f"The {subject} is behind {object1} from {object1}'s perspective.",
        # f"Determine the relative front-back positions of {subject} and {object1}, then answer whether {subject} is in front of {object1}.",
        ("orientation", "on the left"): f"The {subject} is on the left of {object1} from {object1}'s perspective." if gt == 'on the left' else f"The {subject} is on the right of {object1} from {object1}'s perspective.",
        # f"Determine the relative left-right positions of {subject} and {object1}, then answer whether {subject} is on the left of {object1}.",
        ("orientation", "viewpoint"): f"The {subject}'s {gt} side is facing the camera.",
        # f"Determine the viewing direction of {subject} relative to the camera viewpoint, then answer which direction it is facing.",
        ("multi_object", "closer to"): f"The {gt} is closer to {subject}.",
        # f"Determine the relative distances from {subject} to {object1} and {object2}, then answer which object {subject} is closer to.",
        ("multi_object", "facing"): f"The {subject} is facing {gt}.",
        # f"Determine whether {subject} is oriented toward {object1} or toward {object2}, then answer which object it is facing.",
        ("multi_object", "parallel"): f"The {subject} and {object1} are parallel." if gt == 'parallel' else f"The {subject} and {object1} are perpendicular.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are parallel.",
        ("multi_object", "same direction"): f"The {subject} and {object1} are facing the same direction." if gt == 'same or similar directions' else f"The {subject} and {object1} are facing different directions.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are facing the same direction.",
        ("multi_object", "viewpoint towards object"): f"The {subject}'s {gt} side is facing the {object1}",
        # f"Determine the relative position of {object1} from {subject}'s current viewpoint, then answer which direction {subject} should turn to face {object1}.",
    }
    return hints.get((qtype, relation), f"Determine the relevant relative position or orientation involving {subject} before answering.")

def _format_semantic_mapping(doc, gt):
    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", ""))
    subject = doc.get("subject")
    object1 = doc.get("object1", doc.get("object_1"))
    object2 = doc.get("object2", doc.get("object_2"))
    map_gt = map_subject if gt == subject else (map_object1 if gt == object1 else (map_object2 if gt == object2 else gt))
    if gt in [subject, object1, object2]:
        gt = map_gt  # If gt is one of the objects, then map it to the corresponding semantic representation

    mapping_prompt = f"Map the objects in the question to the following semantic representations:\n"
    if subject:
        mapping_prompt += f"  {subject} -> {map_subject}\n"
    if object1:
        mapping_prompt += f"  {object1} -> {map_object1}\n"
    if object2:
        mapping_prompt += f"  {object2} -> {map_object2}\n"

    subject = map_subject if subject else None
    object1 = map_object1 if object1 else None
    object2 = map_object2 if object2 else None
    

    hints = {
        ("height", "higher"): f"The {gt} is {'higher' if 'higher' in str(doc.get('original_question', '')) else 'lower'}.",
        # f"Determine the relative vertical positions of {subject} and {object1}, then answer which one is higher in the image.",
        ("location", "above"): f"The {subject} is directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}." if gt == "yes" else f"The {subject} is not directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}.",
        # f"Determine whether {subject} is vertically positioned above {object1} in the image.",
        ("location", "closer to camera"): f"The {gt} is closer to the camera." if "closer to the camera" in str(doc.get('original_question', '')) else f"The {gt} is farther from the camera.",
        # f"Determine the relative depth of {subject} and {object1}, then answer which object is closer to the camera.",
        ("location", "next to"): f"The {subject} is far from {object1}." if gt == 'far away from each other' else f"The {subject} is close to {object1}.",
        # f"Determine whether {subject} is positioned adjacent to {object1} in the image.",
        ("orientation", "in front of"): f"The {subject} is in front of {object1} from {object1}'s perspective." if gt == 'in front of' else f"The {subject} is behind {object1} from {object1}'s perspective.",
        # f"Determine the relative front-back positions of {subject} and {object1}, then answer whether {subject} is in front of {object1}.",
        ("orientation", "on the left"): f"The {subject} is on the left of {object1} from {object1}'s perspective." if gt == 'on the left' else f"The {subject} is on the right of {object1} from {object1}'s perspective.",
        # f"Determine the relative left-right positions of {subject} and {object1}, then answer whether {subject} is on the left of {object1}.",
        ("orientation", "viewpoint"): f"The {subject}'s {gt} side is facing the camera.",
        # f"Determine the viewing direction of {subject} relative to the camera viewpoint, then answer which direction it is facing.",
        ("multi_object", "closer to"): f"The {gt} is closer to {subject}.",
        # f"Determine the relative distances from {subject} to {object1} and {object2}, then answer which object {subject} is closer to.",
        ("multi_object", "facing"): f"The {subject} is facing {gt}.",
        # f"Determine whether {subject} is oriented toward {object1} or toward {object2}, then answer which object it is facing.",
        ("multi_object", "parallel"): f"The {subject} and {object1} are parallel." if gt == 'parallel' else f"The {subject} and {object1} are perpendicular.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are parallel.",
        ("multi_object", "same direction"): f"The {subject} and {object1} are facing the same direction." if gt == 'same or similar directions' else f"The {subject} and {object1} are facing different directions.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are facing the same direction.",
        ("multi_object", "viewpoint towards object"): f"The {subject}'s {gt} side is facing the {object1}",
        # f"Determine the relative position of {object1} from {subject}'s current viewpoint, then answer which direction {subject} should turn to face {object1}.",
    }
    return mapping_prompt + hints.get((qtype, relation), f"Determine the relevant relative position or orientation involving {subject} before answering.")


def _format_semantic_override(doc, gt):
    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", ""))
    subject = doc.get("subject")
    object1 = doc.get("object1", doc.get("object_1"))
    object2 = doc.get("object2", doc.get("object_2"))
    map_gt = map_subject if gt == subject else (map_object1 if gt == object1 else (map_object2 if gt == object2 else gt))
    if gt in [subject, object1, object2]:
        gt = map_gt  # If gt is one of the objects, then map it to the corresponding semantic representation

    # mapping_prompt = f"Map the objects in the question to the following semantic representations:\n"
    # if subject:
    #     mapping_prompt += f"  {subject} -> {map_subject}\n"
    # if object1:
    #     mapping_prompt += f"  {object1} -> {map_object1}\n"
    # if object2:
    #     mapping_prompt += f"  {object2} -> {map_object2}\n"

    subject = map_subject if subject else None
    object1 = map_object1 if object1 else None
    object2 = map_object2 if object2 else None
    

    hints = {
        ("height", "higher"): f"The {gt} is {'higher' if 'higher' in str(doc.get('original_question', '')) else 'lower'}.",
        # f"Determine the relative vertical positions of {subject} and {object1}, then answer which one is higher in the image.",
        ("location", "above"): f"The {subject} is directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}." if gt == "yes" else f"The {subject} is not directly {'above' if 'above' in str(doc.get('original_question', '')) else 'underneath'} {object1}.",
        # f"Determine whether {subject} is vertically positioned above {object1} in the image.",
        ("location", "closer to camera"): f"The {gt} is closer to the camera." if "closer to the camera" in str(doc.get('original_question', '')) else f"The {gt} is farther from the camera.",
        # f"Determine the relative depth of {subject} and {object1}, then answer which object is closer to the camera.",
        ("location", "next to"): f"The {subject} is far from {object1}." if gt == 'far away from each other' else f"The {subject} is close to {object1}.",
        # f"Determine whether {subject} is positioned adjacent to {object1} in the image.",
        ("orientation", "in front of"): f"The {subject} is in front of {object1} from {object1}'s perspective." if gt == 'in front of' else f"The {subject} is behind {object1} from {object1}'s perspective.",
        # f"Determine the relative front-back positions of {subject} and {object1}, then answer whether {subject} is in front of {object1}.",
        ("orientation", "on the left"): f"The {subject} is on the left of {object1} from {object1}'s perspective." if gt == 'on the left' else f"The {subject} is on the right of {object1} from {object1}'s perspective.",
        # f"Determine the relative left-right positions of {subject} and {object1}, then answer whether {subject} is on the left of {object1}.",
        ("orientation", "viewpoint"): f"The {subject}'s {gt} side is facing the camera.",
        # f"Determine the viewing direction of {subject} relative to the camera viewpoint, then answer which direction it is facing.",
        ("multi_object", "closer to"): f"The {gt} is closer to {subject}.",
        # f"Determine the relative distances from {subject} to {object1} and {object2}, then answer which object {subject} is closer to.",
        ("multi_object", "facing"): f"The {subject} is facing {gt}.",
        # f"Determine whether {subject} is oriented toward {object1} or toward {object2}, then answer which object it is facing.",
        ("multi_object", "parallel"): f"The {subject} and {object1} are parallel." if gt == 'parallel' else f"The {subject} and {object1} are perpendicular.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are parallel.",
        ("multi_object", "same direction"): f"The {subject} and {object1} are facing the same direction." if gt == 'same or similar directions' else f"The {subject} and {object1} are facing different directions.",
        # f"Determine the relative orientations of {subject} and {object1}, then answer whether they are facing the same direction.",
        ("multi_object", "viewpoint towards object"): f"The {subject}'s {gt} side is facing the {object1}",
        # f"Determine the relative position of {object1} from {subject}'s current viewpoint, then answer which direction {subject} should turn to face {object1}.",
    }
    return hints.get((qtype, relation), f"Determine the relevant relative position or orientation involving {subject} before answering.")

def _build_options(doc) -> dict[str, str]:
    options = {}
    for cand in string.ascii_uppercase[:4]:
        if cand in doc and doc[cand] is not None:
            val = doc[cand]
            if isinstance(val, str) and val.lower() == "nan":
                continue
            if pd.isna(val):
                continue
            options[cand] = val
    if options:
        return options

    qtype = _normalize_text(doc.get("qtype", ""))
    relation = _normalize_text(doc.get("relation", ""))
    subject = doc.get("subject")
    object1 = doc.get("object1", doc.get("object_1"))
    object2 = doc.get("object2", doc.get("object_2"))

    derived_options = []
    if (qtype, relation) in {
        ("height", "higher"),
        ("location", "closer to camera"),
    }:
        derived_options = [subject, object1]
    elif (qtype, relation) in {
        ("location", "above"),
        ("location", "next to"),
        ("orientation", "in front of"),
        ("orientation", "on the left"),
        ("multi_object", "parallel"),
        ("multi_object", "same direction"),
    }:
        derived_options = ["yes", "no"]
    elif (qtype, relation) in {
        ("orientation", "viewpoint"),
        ("multi_object", "viewpoint towards object"),
    }:
        derived_options = ["left", "right", "front", "back"]
    elif (qtype, relation) in {
        ("multi_object", "closer to"),
        ("multi_object", "facing"),
    }:
        derived_options = [object1, object2]

    return {
        string.ascii_uppercase[idx]: option
        for idx, option in enumerate(derived_options)
        if option is not None
    }


def _ground_truth_answer_letter(doc) -> Optional[str]:
    answer = str(doc.get("answer", "")).strip()
    if answer in string.ascii_uppercase[:4]:
        return answer

    answer_norm = _normalize_text(answer)
    for letter, option in _build_options(doc).items():
        if answer_norm == _normalize_text(option):
            return letter
    return None


def _format_bbox_context(doc) -> str:
    if os.getenv("LMMS_EVAL_INCLUDE_LOCATION_TEXT", "0") != "1":
        return ""

    bbox_data = _parse_json_maybe(doc.get("bboxes_by_item", {}))
    if not isinstance(bbox_data, dict) or not bbox_data:
        return ""

    object_names = list(bbox_data.keys())
    prompt_lines = [
        f"The image contains the following queried objects: {', '.join(object_names)}.",
        "Bounding boxes are given in normalized coordinates as [x_min, y_min, x_max, y_max].",
    ]

    frame_entries = []
    for obj_name, frame_map in bbox_data.items():
        frame_map = _parse_json_maybe(frame_map)
        if not isinstance(frame_map, dict):
            continue
        boxes = frame_map.get("0", [])
        for box_idx, box in enumerate(boxes, start=1):
            if len(box) != 4:
                continue
            x, y, w, h = box
            frame_entries.append(
                f"{obj_name}_{box_idx}[{x:.2f}, {y:.2f}, {x + w:.2f}, {y + h:.2f}]"
            )
    if frame_entries:
        prompt_lines.append("Frame 0: " + ", ".join(frame_entries))

    return "\n".join(prompt_lines) + "\n"

def _get_color_idx(box_text, answer):
    option_splits = box_text.split('\"0\"')
    # print(option_splits)
    for idx, split in enumerate(option_splits):
        if answer in split:
            return idx
    return None


def _format_bbox_dimensions_prompt(doc) -> str:
    bbox_entries = _get_all_items_bbox_entries(doc)
    if not bbox_entries:
        return ""

    # img = _load_image_from_doc(doc, use_overlay=False)
    image_width, image_height = 1000, 1000 #img.size

    prompt_lines = ["The queried objects have the following bounding box(es):"]
    for item_name, boxes, _ in bbox_entries:
        formatted_boxes = []
        for box_idx, (x, y, w, h) in enumerate(boxes, start=1):
            if "flip" in str(doc.get("index")):
                x = 1 - (x + w)
                w = 1 - x - w
            x1, y1, x2, y2 = int(x * image_width), int(y * image_height), int((x + w) * image_width), int((y + h) * image_height)
            formatted_boxes.append(f"{box_idx}[{x1}, {y1}, {x2}, {y2}]")
        prompt_lines.append(f"{item_name}: {', '.join(formatted_boxes)}")
    return "\n".join(prompt_lines)

def _format_centroids_prompt(doc) -> str:
    bbox_entries = _get_all_items_bbox_entries(doc)
    if not bbox_entries:
        return ""

    # img = _load_image_from_doc(doc, use_overlay=False)
    image_width, image_height = 1000, 1000 #img.size

    prompt_lines = ["The queried objects have the following bounding box(es):"]
    for item_name, boxes, _ in bbox_entries:
        formatted_boxes = []
        for box_idx, (x, y, w, h) in enumerate(boxes, start=1):
            x1, y1, x2, y2 = int(x * image_width), int(y * image_height), int((x + w) * image_width), int((y + h) * image_height)
            formatted_boxes.append(f"{box_idx}[{(x1+x2)//2}, {(y1+y2)//2}]")
        prompt_lines.append(f"{item_name}: {', '.join(formatted_boxes)}")
    return "\n".join(prompt_lines)


def _build_bbox_prompt(doc, lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer the above in the bbox format: [x_min, y_min, x_max, y_max]. Each in the range [0, 1000].\n",
    )
    items = doc.get("bbox_items", [])

    prompt = pre_prompt
    question = doc.get("question") or doc.get("original_question") or _build_fallback_question(doc)
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"
    prompt += f"Question: {question}\n"

    if items:
        prompt += "Look at the image and find the bbox(es) of:\n"
        for idx, item in enumerate(items, start=1):
            prompt += f"{idx}. {item}\n"
        prompt += "\n"

    prompt += post_prompt
    return prompt


def _format_gt_bbox_answers(doc) -> dict[str, list[list[int]]]:
    bbox_data = _parse_json_maybe(doc.get("bboxes_by_item", {}))
    if not isinstance(bbox_data, dict):
        return {}

    item_names = doc.get("bbox_items") or list(bbox_data.keys())
    image_width, image_height = 1000, 1000
    is_flip = "flip" in str(doc.get("index", doc.get("qid", "")))

    formatted_answers = {}
    for item_name in item_names:
        frame_map = _parse_json_maybe(bbox_data.get(item_name, {}))
        if not isinstance(frame_map, dict):
            formatted_answers[item_name] = []
            continue

        boxes = _get_valid_boxes(frame_map.get("0", []))
        formatted_boxes = []
        for x, y, w, h in boxes:
            if is_flip:
                x = 1 - (x + w)
            x1 = int(x * image_width)
            y1 = int(y * image_height)
            x2 = int((x + w) * image_width)
            y2 = int((y + h) * image_height)
            formatted_boxes.append([x1, y1, x2, y2])

        formatted_answers[item_name] = formatted_boxes

    return formatted_answers


def _format_bbox_colors_prompt(doc) -> str:
    bbox_entries = _get_all_items_bbox_entries(doc)
    if not bbox_entries:
        return ""

    prompt_lines = ["The queried objects have bounding boxes with the following colors:"]
    for item_name, _, color_name in bbox_entries:
        prompt_lines.append(f"{item_name}: {color_name}")
    return "\n".join(prompt_lines)


def _build_GT_help(doc, options, answer_format) -> str:

    assert answer_format in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, "Invalid GT help format option"

    gt_answer = doc.get("answer", "").strip()
    
    # Letter to term
    if gt_answer in ['A', 'B', 'C', 'D']:
        gt_answer = doc.get(gt_answer)
        
    if gt_answer == "":
        return ""
    if answer_format == "1":
        return f"The correct answer is: {gt_answer}."
    
    elif answer_format == "2":
        if gt_answer not in _parse_json_maybe(doc.get("bboxes_by_item", "")):
            return ""
        gt_bboxes = _parse_json_maybe(doc.get("bboxes_by_item", ""))[gt_answer]["0"]
        
        return_prompt = ""
        if len(gt_bboxes) != 1:
            return ""
        for box_idx, box in enumerate(gt_bboxes, start=1):
            if len(box) != 4:
                continue
            # img = _load_image_from_doc(doc, use_overlay=False)
            image_width, image_height = 1000, 1000#img.size
            x, y, w, h = box
            if "flip" in str(doc.get("index")):
                x = 1 - (x + w)
                w = 1 - x - w
            
            x1, y1, x2, y2 = int(x * image_width), int(y * image_height), int((x + w) * image_width), int((y + h) * image_height)
            return_prompt += f"{box_idx}[{x1}, {y1}, {x2}, {y2}] \n"
        return "The correct answer is the object with bounding box(es): " + return_prompt if return_prompt != "" else ""
    
    elif answer_format == "3":
        if gt_answer not in _parse_json_maybe(doc.get("bboxes_by_item", "")):
            return ""
        gt_bbox = _parse_json_maybe(doc.get("bboxes_by_item", ""))[gt_answer]["0"]
        # print(_parse_json_maybe(doc.get("bboxes_by_item", "")))
        # print(gt_answer)
        # print(gt_bbox)
        # print()

        # if not gt_bbox:
        if len(gt_bbox) != 1:
            return ""
        
        # items = [
        #     doc.get("subject"),
        #     doc.get("object_1", doc.get("object1")),
        #     doc.get("object_2", doc.get("object2")),
        # ]
        # gt_idx = -1
        # for item in items:
        #     if item and item in gt_answer:
        #         gt_idx = items.index(item)
        #         break
        
        # if gt_idx == -1:
        #     return ""
        
        # color = DEFAULT_COLOR_ORDER[gt_idx % len(DEFAULT_COLOR_ORDER)]
        color = gt_bbox = _parse_json_maybe(doc.get("bbox_colors_by_item", ""))[gt_answer]["name"]
        
        return f"The correct answer is the object with bounding box in the color: {color}."
    elif answer_format == "4":
        return _build_text_only_position_hint(doc, gt_answer)
    elif answer_format == "5":
        # Only prints bbox dimensions if all items have bboxes, to avoid giving partial hints
        return _format_bbox_dimensions_prompt(doc)
    elif answer_format == "6":
        # Only prints bboxes colors if all items have bboxes, to avoid giving partial hints
        return _format_bbox_colors_prompt(doc)
    elif answer_format == "7":
        # Provides the center point of the GT bbox as a hint
        if gt_answer not in _parse_json_maybe(doc.get("bboxes_by_item", "")):
            return ""
        gt_bboxes = _parse_json_maybe(doc.get("bboxes_by_item", ""))[gt_answer]["0"]
        
        return_prompt = ""
        for box_idx, box in enumerate(gt_bboxes, start=1):
            if len(box) != 4:
                continue
            # img = _load_image_from_doc(doc, use_overlay=False)
            image_width, image_height = 1000, 1000#img.size
            x, y, w, h = box
            x1, y1, x2, y2 = int(x * image_width), int(y * image_height), int((x + w) * image_width), int((y + h) * image_height)
            return_prompt += f"{box_idx}[{ (x1 + x2) // 2 }, { (y1 + y2) // 2 }] \n"
        return "The correct answer is the object(s) at: " + return_prompt if return_prompt != "" else ""
    elif answer_format == "8":
        # Provides centrois for all items
        return _format_centroids_prompt(doc)
    elif answer_format == "9":
        # Makes the semantic mapping
        return _format_semantic_mapping(doc, gt_answer)

    elif answer_format == "10":
        # Makes the semantic override
        return _format_semantic_override(doc, gt_answer)

    return ""

def doc_to_visual(doc):
    return [_load_image_from_doc(doc, use_overlay=False)]


def bbox_doc_to_visual(doc):
    if "image" in doc and hasattr(doc["image"], "convert"):
        return [doc["image"].convert("RGB")]

    path_candidates = []
    path_candidates.extend(
        [
            doc.get("resolved_img_path"),
            doc.get("img_path"),
            doc.get("image"),
        ]
    )

    for image_path in path_candidates:
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            if "flip" in str(doc.get("index")):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return [img]

    return [_load_image_from_doc(doc, use_overlay=False)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Please select the correct answer from the options above and include the Option letter (A, B, C, D). \n",
    )

    bbox_context = _format_bbox_context(doc)
    question = doc.get("original_question") or _build_fallback_question(doc)
    options = _build_options(doc)
    if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        gt_help = _build_GT_help(doc, options, os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0"))
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
        for key, item in options.items():
            prompt += f"{key}. {item}\n"
    if post_prompt:
        prompt += post_prompt

    return prompt

def variants_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Please select the correct answer from the options above and include the Option letter (A, B, C, D). \n",
    )

    gt_help = doc.get("help", "")
    question = doc.get("question", "")
    options = doc.get("options", "")

    assert question, "Question is missing in the variants document."
    assert gt_help, "Help text is missing in the variants document."
    assert options, "Options are missing in the variants document."
    
    prompt = ""
    if pre_prompt:
        prompt += pre_prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
    prompt += f"Question: {question}\n"
    if options:
        prompt += "Options:\n" + options + "\n"
    if gt_help:
        prompt += f"{gt_help}\n"
    if post_prompt:
        prompt += post_prompt

    return prompt


def bbox_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _build_bbox_prompt(doc, lmms_eval_specific_kwargs)


def doc_to_target(doc):
    return _ground_truth_answer_letter(doc) or doc.get("answer")


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


def get_qid_key(qid: str, mode: str = "base") -> str:
    if mode == "base":
        return qid[:8]
    elif mode == "flip":
        return qid[:8]
    elif mode == "circular":
        if "-flip" in qid:
            return qid[:13]
        else:
            return qid[:8]
    else:
        return qid


def get_main_category(category: str) -> str:
    return CATEGORY_MAPPING.get(category, "other")


def process_results(doc, results):
    pred = results[0].strip()
    pred_answer = extract_answer(pred)
    gt_answer = _ground_truth_answer_letter(doc) or str(doc.get("answer", "")).strip() or doc.get("parent_gt_answer", "")
    gt_help_format = os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0")
    gt_help_text = doc.get("help", "")
    if not gt_help_text and gt_help_format in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        gt_help_text = _build_GT_help(doc, _build_options(doc), gt_help_format)

    score = 1.0 if pred_answer == gt_answer else 0.0

    category = _get_relation_category(doc)
    main_category = get_main_category(category)

    index = doc.get("index", doc.get("qid"))
    qid = doc.get("qid", index)

    base_entry = {
        "index": index,
        "qid": qid,
        "score": score,
        "category": category,
        "main_category": main_category,
    }

    question = ""
    try:
        question = variants_doc_to_text(doc)
    except Exception as e:
        question = doc_to_text(doc)

    submission_entry = {
        "index": index,
        "qid": qid,
        "question_prompt": question,
        "image_url": doc.get("image_url", doc.get("image")),
        "gt_answer": gt_answer,
        "gt_help_text": gt_help_text,
        "model_answer": pred,
        "pred_answer": pred_answer,
        "score": score,
        "category": category,
        "main_category": main_category,
    }

    result_dict = {
        "submission": submission_entry,
        "vanilla_accuracy": base_entry,
        "flip_accuracy": base_entry,
        "circular_accuracy": base_entry,
        "flip_circular_accuracy": base_entry,
        "height_accuracy": base_entry,
        "location_accuracy": base_entry,
        "orientation_accuracy": base_entry,
        "multi_object_accuracy": base_entry,
    }

    for cat_name in CATEGORY_MAPPING:
        result_dict[f"{cat_name}_accuracy"] = base_entry

    return result_dict


def aggregate_results_for_submission(results, args):
    model_tag = _get_submission_model_tag(args)
    path = generate_submission_file(f"3dsrbench_predictions_{model_tag}.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    eval_logger.info(f"Results saved to {path}.")


def bbox_process_results(doc, results):
    pred = results[0].strip() if results else ""
    index = doc.get("index", doc.get("qid"))
    qid = doc.get("qid", index)
    return {
        "submission": {
            "index": index,
            "qid": qid,
            "question_prompt": _build_bbox_prompt(doc),
            "image_url": doc.get("image_url", doc.get("image")),
            "gt_answers": _format_gt_bbox_answers(doc),
            "model_answer": pred,
        }
    }


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


def _direction_object_image_key(doc: dict) -> str:
    """Return a stable key for collecting candidate objects from one image."""
    return str(
        doc.get("image_name")
        or doc.get("image_url")
        or doc.get("resolved_img_path")
        or doc.get("img_path")
        or doc.get("qid", doc.get("index", ""))
    )


def _direction_object_terms(doc: dict) -> list[str]:
    """Extract object names available in a source row without duplicates."""
    values = [
        doc.get("subject"),
        doc.get("object1", doc.get("object_1")),
        doc.get("object2", doc.get("object_2")),
    ]
    bbox_items = _parse_json_maybe(doc.get("bbox_items", []))
    if isinstance(bbox_items, list):
        values.extend(bbox_items)

    terms = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in terms:
            terms.append(value)
    return terms


def _direction_object_same_entity(left: str, right: str) -> bool:
    """Treat abbreviated references (for example, ``truck``) as the subject."""
    left = _normalize_text(left)
    right = _normalize_text(right)
    return bool(left and right and (left == right or left in right or right in left))


def _direction_object_theme(*terms: str) -> str:
    """Infer a broad visual theme for deterministic fallback distractors."""
    text = _normalize_text(" ".join(str(term or "") for term in terms))
    themes = {
        "vehicle": ("car", "truck", "bus", "van", "bicycle", "bike", "motorcycle", "boat", "train", "airplane"),
        "signage": ("sign", "logo", "plaque", "board", "letter"),
        "person": ("person", "man", "woman", "girl", "boy", "baby", "skier", "policeman", "pedestrian"),
        "animal": ("dog", "cat", "horse", "cow", "elephant", "giraffe", "bird", "bear", "animal"),
        "furniture": ("chair", "table", "couch", "bed", "cabinet", "lamp", "mirror", "stairs", "fireplace"),
        "kitchen": ("fridge", "refrigerator", "oven", "microwave", "faucet", "sink", "bottle", "cup", "stove"),
        "street": ("hydrant", "parking", "bench", "trash", "street", "sidewalk", "bus stop"),
    }
    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            return theme
    return "generic"


def _direction_object_four_options(
    target: str, subject: str, image_candidates: list[str]
) -> tuple[list[str], list[str]]:
    """Return target plus three distinct, in-theme distractors.

    Image-grounded object names are preferred.  If fewer than four are
    available, a deterministic thematic pool supplies the remaining labels.
    The generated labels are recorded separately for auditability.
    """
    candidates = []
    generated = []

    def add(value: str, is_generated: bool = False) -> None:
        value = str(value or "").strip()
        if not value or _direction_object_same_entity(value, subject):
            return
        if any(_direction_object_same_entity(value, candidate) for candidate in candidates):
            return
        candidates.append(value)
        if is_generated:
            generated.append(value)

    # Keep the annotated target even when its wording overlaps the subject
    # (for example, ``bus`` and ``bus stop``).
    candidates.append(str(target).strip())
    for candidate in image_candidates:
        add(candidate)
        if len(candidates) == 4:
            return candidates, generated

    # The target's class is the strongest indicator of an appropriate
    # distractor theme; the subject may belong to a different class (for
    # example, a truck pointing at a stop sign).
    theme = _direction_object_theme(target)
    for candidate in DIRECTION_OBJECT_THEMATIC_POOLS[theme] + DIRECTION_OBJECT_THEMATIC_POOLS["generic"]:
        add(candidate, is_generated=True)
        if len(candidates) == 4:
            return candidates, generated

    # The fixed pools above provide more than enough distinct candidates for
    # normal 3DSR rows; retain a safe final fallback for unusual free-form text.
    suffix = 1
    while len(candidates) < 4:
        add(f"thematic scene object {suffix}", is_generated=True)
        suffix += 1
    return candidates, generated


def _direction_object_set_options(doc: dict, values: list[str], answer_value: str) -> None:
    for index, letter in enumerate("ABCD"):
        doc[letter] = values[index] if index < len(values) else None
    doc["answer"] = "ABCD"[values.index(answer_value)]


def direction_object_process_docs(dataset):
    """Create matched direction-answer and object-answer 3DSRBench examples.

    Only ``multi_object_viewpoint_towards_object`` is used because it specifies
    both the direction (its native answer) and the target object.  Each source
    row yields a native direction item plus an inverse object item.
    """
    image_terms = defaultdict(list)
    docs = list(dataset)
    for doc in docs:
        bucket = image_terms[_direction_object_image_key(doc)]
        for term in _direction_object_terms(doc):
            if term not in bucket:
                bucket.append(term)

    records = []
    skipped = 0
    for doc in docs:
        if (
            _normalize_text(doc.get("qtype", "")) != DIRECTION_OBJECT_QTYPE
            or _normalize_text(doc.get("relation", "")) != DIRECTION_OBJECT_RELATION
        ):
            continue

        direction = _build_options(doc).get(_ground_truth_answer_letter(doc) or "", "")
        direction = _normalize_text(direction)
        subject = str(doc.get("subject", "")).strip()
        target = str(doc.get("object1", doc.get("object_1", ""))).strip()
        if direction not in DIRECTION_OBJECT_DIRECTIONS or not subject or not target:
            skipped += 1
            continue

        source_qid = str(doc.get("qid", doc.get("index", "unknown")))
        image_candidates = [
            term
            for term in image_terms[_direction_object_image_key(doc)]
            if not _direction_object_same_entity(term, subject)
            and not _direction_object_same_entity(term, target)
        ]
        # Keep the original data-quality requirement: a paired object question
        # must have at least one image-derived distractor in addition to its
        # target.  The thematic generator only pads qualifying examples from
        # two-to-four choices; it never turns a single-object row into an
        # object-selection question.
        image_derived_options = [target]
        for candidate in image_candidates:
            if not any(
                _direction_object_same_entity(candidate, existing)
                for existing in image_derived_options
            ):
                image_derived_options.append(candidate)
        if len(image_derived_options) < 2:
            skipped += 1
            continue
        candidates, generated_distractors = _direction_object_four_options(
            target, subject, image_candidates
        )
        random.Random(f"{DIRECTION_OBJECT_SAMPLE_SEED}:{source_qid}").shuffle(candidates)

        common = dict(doc)
        common.update(
            {
                "source_qid": source_qid,
                "source_task_family": "multi_object_viewpoint_towards_object",
                "diagnostic_subject": subject,
                "diagnostic_target_object": target,
                "diagnostic_direction": direction,
                "diagnostic_sample_seed": DIRECTION_OBJECT_SAMPLE_SEED,
                "diagnostic_generated_object_distractors": generated_distractors,
            }
        )

        native = dict(common)
        native.update(
            {
                "qid": f"{source_qid}::native",
                "index": f"{source_qid}::native",
                "diagnostic_variant": "native",
                "diagnostic_answer_format": "direction",
                "diagnostic_target": direction,
            }
        )
        records.append(native)

        inverse = dict(common)
        inverse.update(
            {
                "qid": f"{source_qid}::inverse",
                "index": f"{source_qid}::inverse",
                "question": (
                    "Consider the real-world 3D locations and orientations of the objects. "
                    f"Which object is in the direction that the {direction} side of the "
                    f"{subject} points toward?"
                ),
                "original_question": "",
                "diagnostic_variant": "inverse",
                "diagnostic_answer_format": "object",
                "diagnostic_target": target,
            }
        )
        _direction_object_set_options(inverse, candidates, target)
        records.append(inverse)

    if skipped:
        eval_logger.warning("Skipped %d 3DSR direction/object rows without a valid paired transformation.", skipped)
    eval_logger.info(
        "3DSR direction/object task created %d matched examples from %d source rows (seed=%s).",
        len(records),
        len(records) // 2,
        DIRECTION_OBJECT_SAMPLE_SEED,
    )
    try:
        from datasets import Dataset

        return Dataset.from_list(records)
    except ImportError:
        # Kept for lightweight unit tests outside the lmms-eval environment.
        return records


def direction_object_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    options = _build_options(doc)
    prompt = (
        "Answer this spatial-reasoning question using the image. "
        "Select one answer option and respond with its letter.\n"
        f"Question: {doc.get('original_question') or doc.get('question') or _build_fallback_question(doc)}\nOptions:\n"
    )
    for letter, value in options.items():
        prompt += f"{letter}. {value}\n"
    return prompt


def direction_object_doc_to_target(doc):
    return _ground_truth_answer_letter(doc) or str(doc.get("answer", "")).strip()


def direction_object_direct_answer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Render the diagnostic with unlabelled values and a direct-text answer."""
    del lmms_eval_specific_kwargs
    options = [str(value).strip() for value in _build_options(doc).values() if value is not None]
    return (
        "Answer this spatial-reasoning question using the image. "
        "Respond with the exact object or direction text, not an option letter.\n"
        f"Question: {doc.get('original_question') or doc.get('question') or _build_fallback_question(doc)}\n"
        f"Possible answers: {'; '.join(options)}\n"
    )


def direction_object_direct_answer_doc_to_target(doc):
    return str(doc.get("diagnostic_target", "")).strip()


def _direction_object_extract_direct_answer(text: str, options: list[str]) -> Optional[str]:
    """Extract one displayed object/direction value, never a choice letter."""
    normalized = _normalize_text(text).strip(".?!:;\"'")
    if not normalized:
        return None
    normalized_options = {_normalize_text(option): option for option in options}
    if normalized in normalized_options:
        return normalized_options[normalized]
    for option_norm, option in sorted(normalized_options.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(option_norm)}(?!\w)", normalized):
            return option
    return None


def direction_object_process_results(doc, results):
    prediction = results[0].strip() if results else ""
    parsed = extract_answer(prediction)
    gold = direction_object_doc_to_target(doc)
    score = float(parsed == gold)
    entry = {
        "index": doc.get("index"),
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "direction": doc.get("diagnostic_direction"),
        "score": score,
    }
    return {
        "accuracy": entry,
        "object_answer_accuracy": entry,
        "direction_answer_accuracy": entry,
        "format_switch_gain": entry,
        "object_minus_direction": entry,
        "submission": {
            **entry,
            "question_prompt": direction_object_doc_to_text(doc),
            "image_url": doc.get("image_url", doc.get("image")),
            "prediction": prediction,
            "parsed_prediction": parsed,
            "gold_option": gold,
            "gold_target": doc.get("diagnostic_target"),
        },
    }


def direction_object_direct_answer_process_results(doc, results):
    prediction = results[0].strip() if results else ""
    options = [str(value).strip() for value in _build_options(doc).values() if value is not None]
    parsed = _direction_object_extract_direct_answer(prediction, options)
    gold = direction_object_direct_answer_doc_to_target(doc)
    score = float(_normalize_text(parsed or "") == _normalize_text(gold))
    entry = {
        "index": doc.get("index"),
        "qid": doc.get("qid"),
        "source_qid": doc.get("source_qid"),
        "variant": doc.get("diagnostic_variant"),
        "answer_format": doc.get("diagnostic_answer_format"),
        "direction": doc.get("diagnostic_direction"),
        "score": score,
    }
    return {
        "accuracy": entry,
        "object_answer_accuracy": entry,
        "direction_answer_accuracy": entry,
        "format_switch_gain": entry,
        "object_minus_direction": entry,
        "submission": {
            **entry,
            "question_prompt": direction_object_direct_answer_doc_to_text(doc),
            "image_url": doc.get("image_url", doc.get("image")),
            "prediction": prediction,
            "parsed_prediction": parsed,
            "gold_target": gold,
        },
    }


def _direction_object_mean(results, answer_format=None):
    if answer_format is not None:
        results = [result for result in results if result.get("answer_format") == answer_format]
    return sum(result["score"] for result in results) / len(results) if results else 0.0


def direction_object_aggregate_accuracy(results):
    return _direction_object_mean(results)


def direction_object_aggregate_object_answer_accuracy(results):
    return _direction_object_mean(results, "object")


def direction_object_aggregate_direction_answer_accuracy(results):
    return _direction_object_mean(results, "direction")


def direction_object_aggregate_format_switch_gain(results):
    paired = defaultdict(dict)
    for result in results:
        paired[result.get("source_qid")][result.get("variant")] = result["score"]
    differences = [
        pair["inverse"] - pair["native"]
        for pair in paired.values()
        if {"native", "inverse"} <= pair.keys()
    ]
    return sum(differences) / len(differences) if differences else 0.0


def direction_object_aggregate_object_minus_direction(results):
    paired = defaultdict(dict)
    for result in results:
        paired[result.get("source_qid")][result.get("answer_format")] = result["score"]
    differences = [
        pair["object"] - pair["direction"]
        for pair in paired.values()
        if {"object", "direction"} <= pair.keys()
    ]
    return sum(differences) / len(differences) if differences else 0.0


def direction_object_aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"3dsrbench_direction_object_{_get_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"3DSR direction/object records saved to {path}.")


def direction_object_direct_answer_aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"3dsrbench_direction_object_direct_answer_{_get_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"3DSR direct-answer direction/object records saved to {path}.")


# Multi-family direction-versus-object diagnostic -----------------------------
#
# This expands the original viewpoint-only diagnostic to the requested source
# families.  Every pair is built with the source row's number of native choices
# (two or four), and is discarded if that many distinct image-grounded object
# labels are not available for an object-answer inverse.
MULTI_FAMILY_DIRECTION_OBJECT_CATEGORIES = {
    "orientation_in_front_of",
    "orientation_on_the_left",
    "orientation_viewpoint",
    "multi_object_viewpoint_towards_object",
    "multi_object_facing",
    "multi_object_closer_to",
}
MULTI_FAMILY_DIRECTION_OBJECT_SEED = "3dsrbench_direction_object_multifamily_v1"


def _multifamily_object_choices(
    target: str, subject: str, image_terms: list[str], count: int, source_qid: str
) -> list[str] | None:
    """Return exactly ``count`` distinct, image-grounded choices including target."""
    choices = [target]
    for term in image_terms:
        if _direction_object_same_entity(term, subject):
            continue
        if any(_direction_object_same_entity(term, choice) for choice in choices):
            continue
        choices.append(term)
        if len(choices) == count:
            break
    if len(choices) != count:
        return None
    random.Random(f"{MULTI_FAMILY_DIRECTION_OBJECT_SEED}:{source_qid}").shuffle(choices)
    return choices


def _multifamily_pair_spec(doc: dict, category: str) -> dict[str, object] | None:
    """Return the matched native/inverse prompt specification for one source row."""
    subject = str(doc.get("subject", "")).strip()
    object1 = str(doc.get("object1", doc.get("object_1", ""))).strip()
    object2 = str(doc.get("object2", doc.get("object_2", ""))).strip()
    source_options = _build_options(doc)
    gold_letter = _ground_truth_answer_letter(doc)
    gold_source = _normalize_text(source_options.get(gold_letter, ""))
    if not subject or not object1:
        return None

    if category == "orientation_in_front_of":
        choices = ["in front of", "behind"]
        gold = "in front of" if gold_source in {"yes", "in front of"} else "behind"
        question = f"From {object1}'s perspective, where is {subject}?"
        return {"native_options": choices, "native_answer": gold, "native_target": gold,
                "native_question": question, "native_format": "direction",
                "inverse_options": None, "inverse_answer": object1, "inverse_target": object1}
    if category == "orientation_on_the_left":
        choices = ["on the left", "on the right"]
        gold = "on the left" if gold_source in {"yes", "on the left"} else "on the right"
        question = f"From {object1}'s perspective, where is {subject}?"
        return {"native_options": choices, "native_answer": gold, "native_target": gold,
                "native_question": question, "native_format": "direction",
                "inverse_options": None, "inverse_answer": object1, "inverse_target": object1}
    if category in {"orientation_viewpoint", "multi_object_viewpoint_towards_object"}:
        choices = ["left", "right", "front", "back"]
        gold = gold_source
        if gold not in choices:
            return None
        if category == "orientation_viewpoint":
            question = f"From the camera's viewpoint, which direction of {subject} faces the camera?"
            target = object1
        else:
            question = f"Relative to its current viewpoint, which direction should {object1} turn to face {subject}?"
            target = object1
        return {"native_options": choices, "native_answer": gold, "native_target": gold,
                "native_question": question, "native_format": "direction",
                "inverse_options": None, "inverse_answer": target, "inverse_target": target}
    if category == "multi_object_facing":
        native_choices = [str(value).strip() for value in source_options.values() if str(value).strip()]
        if len(native_choices) != 2 or gold_source not in {_normalize_text(value) for value in native_choices}:
            return None
        target = next(value for value in native_choices if _normalize_text(value) == gold_source)
        return {"native_options": native_choices, "native_answer": target, "native_target": target,
                "native_question": str(doc.get("original_question") or doc.get("question") or _build_fallback_question(doc)),
                "native_format": "object", "inverse_options": ["toward", "away from"],
                "inverse_answer": "toward", "inverse_target": "toward",
                "inverse_question": f"Is {subject} facing toward or away from {target}?"}
    if category == "multi_object_closer_to":
        native_choices = [str(value).strip() for value in source_options.values() if str(value).strip()]
        if len(native_choices) != 2 or gold_source not in {_normalize_text(value) for value in native_choices}:
            return None
        target = next(value for value in native_choices if _normalize_text(value) == gold_source)
        return {"native_options": native_choices, "native_answer": target, "native_target": target,
                "native_question": str(doc.get("original_question") or doc.get("question") or _build_fallback_question(doc)),
                "native_format": "object", "inverse_options": ["closer to", "farther from"],
                "inverse_answer": "closer to", "inverse_target": "closer to",
                "inverse_question": f"Is {subject} closer to or farther from {target}?"}
    return None


def multifamily_direction_object_process_docs(dataset):
    """Create matched pairs from orientation and requested multi-object rows."""
    docs = list(dataset)
    image_terms = defaultdict(list)
    for doc in docs:
        bucket = image_terms[_direction_object_image_key(doc)]
        for term in _direction_object_terms(doc):
            if term not in bucket:
                bucket.append(term)

    records = []
    skipped = 0
    for doc in docs:
        category = _get_relation_category(doc)
        if category not in MULTI_FAMILY_DIRECTION_OBJECT_CATEGORIES:
            continue
        transformed = _multifamily_pair_spec(doc, category)
        if transformed is None:
            skipped += 1
            continue
        native_choices = list(transformed["native_options"])
        native_answer = str(transformed["native_answer"])
        native_target = str(transformed["native_target"])
        native_question = str(transformed["native_question"])
        native_format = str(transformed["native_format"])
        inverse_choices = transformed["inverse_options"]
        inverse_answer = str(transformed["inverse_answer"])
        inverse_target = str(transformed["inverse_target"])
        target = inverse_target if native_format == "direction" else native_target
        subject = str(doc.get("subject", "")).strip()
        source_qid = str(doc.get("qid", doc.get("index", "unknown")))
        if native_format == "direction":
            inverse_choices = _multifamily_object_choices(
                target, subject, image_terms[_direction_object_image_key(doc)], len(native_choices), source_qid
            )
            if inverse_choices is None:
                skipped += 1
                continue
        if not isinstance(inverse_choices, list) or len(inverse_choices) != len(native_choices):
            skipped += 1
            continue

        common = dict(doc)
        common.update({
            "source_qid": source_qid,
            "source_task_family": category,
            "diagnostic_subject": subject,
            "diagnostic_target_object": target,
            "diagnostic_direction": native_answer if native_format == "direction" else inverse_answer,
            "diagnostic_num_options": len(native_choices),
            "diagnostic_sample_seed": MULTI_FAMILY_DIRECTION_OBJECT_SEED,
        })
        native = dict(common)
        native.update({
            "qid": f"{source_qid}::native", "index": f"{source_qid}::native",
            "question": native_question, "original_question": "",
            "diagnostic_variant": "native", "diagnostic_answer_format": native_format,
            "diagnostic_target": native_target,
        })
        _direction_object_set_options(native, native_choices, native_answer)
        inverse = dict(common)
        inverse.update({
            "qid": f"{source_qid}::inverse", "index": f"{source_qid}::inverse",
            "question": (
                str(transformed.get("inverse_question")) if native_format == "object" else
                "Consider the real-world 3D locations and orientations of the objects. "
                f"Which object is related to {subject} as {native_answer}?"
            ),
            "original_question": "", "diagnostic_variant": "inverse",
            "diagnostic_answer_format": "direction" if native_format == "object" else "object",
            "diagnostic_target": inverse_target,
        })
        _direction_object_set_options(inverse, inverse_choices, inverse_answer)
        records.extend((native, inverse))

    eval_logger.info(
        "3DSR multi-family direction/object task created %d pairs; skipped %d source rows.",
        len(records) // 2, skipped,
    )
    try:
        from datasets import Dataset
        return Dataset.from_list(records)
    except ImportError:
        return records


def multifamily_direction_object_aggregate_results_for_submission(results, args):
    path = generate_submission_file(
        f"3dsrbench_direction_object_multifamily_{_get_submission_model_tag(args)}.json", args
    )
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    eval_logger.info(f"3DSR multi-family direction/object records saved to {path}.")


def bbox_aggregate_results_for_submission(results, args):
    model_tag = _get_submission_model_tag(args)
    path = generate_submission_file(f"3dsrbench_bbox_predictions_{model_tag}.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    eval_logger.info(f"Results saved to {path}.")


def aggregate_vanilla_accuracy(results):
    if not results:
        return 0.0
    total_score = sum(r["score"] for r in results)
    return total_score / len(results)


def aggregate_flip_accuracy(results):
    if not results:
        return 0.0

    qid_groups = {}
    for r in results:
        qid = r["qid"]
        key = qid.replace("-flip", "")

        if key not in qid_groups:
            qid_groups[key] = []
        qid_groups[key].append(r["score"])

    correct = 0
    total = 0
    for scores in qid_groups.values():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_circular_accuracy(results):
    if not results:
        return 0.0

    qid_groups = {}
    for r in results:
        qid = r["qid"]
        key = get_qid_key(qid, mode="circular")

        if key not in qid_groups:
            qid_groups[key] = []
        qid_groups[key].append(r["score"])

    correct = 0
    total = 0
    for scores in qid_groups.values():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_flip_circular_accuracy(results):
    if not results:
        return 0.0

    qid_groups = {}
    for r in results:
        qid = r["qid"]
        base_id = qid[:8]

        if base_id not in qid_groups:
            qid_groups[base_id] = []
        qid_groups[base_id].append(r["score"])

    correct = 0
    total = 0
    for scores in qid_groups.values():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_height_accuracy(results):
    if not results:
        return 0.0
    scores = [r["score"] for r in results if r.get("main_category") == "height"]
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_location_accuracy(results):
    if not results:
        return 0.0
    scores = [r["score"] for r in results if r.get("main_category") == "location"]
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_orientation_accuracy(results):
    if not results:
        return 0.0
    scores = [r["score"] for r in results if r.get("main_category") == "orientation"]
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_multi_object_accuracy(results):
    if not results:
        return 0.0
    scores = [r["score"] for r in results if r.get("main_category") == "multi_object"]
    return sum(scores) / len(scores) if scores else 0.0


def _aggregate_detailed_category(results, target_category: str) -> float:
    scores = [r["score"] for r in results if r.get("category") == target_category]
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_height_higher_accuracy(results):
    return _aggregate_detailed_category(results, "height_higher")


def aggregate_location_above_accuracy(results):
    return _aggregate_detailed_category(results, "location_above")


def aggregate_location_closer_to_camera_accuracy(results):
    return _aggregate_detailed_category(results, "location_closer_to_camera")


def aggregate_location_next_to_accuracy(results):
    return _aggregate_detailed_category(results, "location_next_to")


def aggregate_orientation_in_front_of_accuracy(results):
    return _aggregate_detailed_category(results, "orientation_in_front_of")


def aggregate_orientation_on_the_left_accuracy(results):
    return _aggregate_detailed_category(results, "orientation_on_the_left")


def aggregate_orientation_viewpoint_accuracy(results):
    return _aggregate_detailed_category(results, "orientation_viewpoint")


def aggregate_multi_object_closer_to_accuracy(results):
    return _aggregate_detailed_category(results, "multi_object_closer_to")


def aggregate_multi_object_facing_accuracy(results):
    return _aggregate_detailed_category(results, "multi_object_facing")


def aggregate_multi_object_parallel_accuracy(results):
    return _aggregate_detailed_category(results, "multi_object_parallel")


def aggregate_multi_object_same_direction_accuracy(results):
    return _aggregate_detailed_category(results, "multi_object_same_direction")


def aggregate_multi_object_viewpoint_towards_object_accuracy(results):
    return _aggregate_detailed_category(results, "multi_object_viewpoint_towards_object")


def aggregate_category_accuracy(results):
    if not results:
        return 0.0

    for main_cat in MAIN_CATEGORIES:
        scores = [r["score"] for r in results if r.get("main_category") == main_cat]
        if scores:
            acc = sum(scores) / len(scores)
            eval_logger.info(f"3DSRBench {main_cat}: {acc * 100:.2f}%")

    all_scores = [r["score"] for r in results]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0
