"""
3DSRBench utilities adapted for local parquet exports with optional bbox prompt text.
"""

import json
import os
import re
import string
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

    assert answer_format in {"0", "1", "2", "3", "4", "5", "6", "7", "8"}, "Invalid GT help format option"

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
    if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["1", "2", "3", "4", "5", "6", "7", "8"]:
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
    if not gt_help_text and gt_help_format in {"1", "2", "3", "4", "5", "6", "7", "8"}:
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
