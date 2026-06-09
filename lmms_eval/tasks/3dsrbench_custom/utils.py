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
from loguru import logger as eval_logger

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
        ]
    )
    for image_path in path_candidates:
        if image_path:
            # debug save IMG
            img = Image.open(image_path).convert("RGB")
            if os.getenv("LMMS_MASK_IMAGE", "0") == "1":
                img = Image.fromarray(np.array(img) * 0)
            # dir = "./test/3dsr"
            # os.makedirs(dir, exists_ok=true)
            # img.save(f'{dir}/{image_path.split("/")[-1]}')
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
        ("height", "higher"): f"Which object is higher in the image: {subject} or {object1}?",
        ("location", "above"): f"Is {subject} above {object1}?",
        ("location", "closer to camera"): f"Which object is closer to the camera: {subject} or {object1}?",
        ("location", "next to"): f"Is {subject} next to {object1}?",
        ("orientation", "in front of"): f"Is {subject} in front of {object1}?",
        ("orientation", "on the left"): f"Is {subject} on the left of {object1}?",
        ("orientation", "viewpoint"): f"From the camera's viewpoint, which direction is {subject} facing?",
        ("multi_object", "closer to"): f"Which object is {subject} closer to: {object1} or {object2}?",
        ("multi_object", "facing"): f"Which object is {subject} facing: {object1} or {object2}?",
        ("multi_object", "parallel"): f"Is {subject} parallel to {object1}?",
        ("multi_object", "same direction"): f"Are {subject} and {object1} facing the same direction?",
        ("multi_object", "viewpoint towards object"): f"Relative to its current viewpoint, which direction should {subject} turn to face {object1}?",
    }
    return templates.get((qtype, relation), f"What is the correct answer about {subject}?")


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

def _build_GT_help(doc, options, answer_format) -> str:

    assert answer_format in {"0", "1", "2", "3"}, "Invalid GT help format option"

    gt_answer = doc.get("answer", "").strip()
    if gt_answer == "":
        return ""
    if answer_format == "1":
        return f"The correct answer is: {gt_answer}."
    
    elif answer_format == "2":
        if len(options) == 4 or 'yes' in options.values():
            return ""
        gt_bboxes = _parse_json_maybe(doc.get("bboxes_by_item", ""))[gt_answer]["0"]
        
        return_prompt = "The correct answer is the object with bounding box(es):"
        for box_idx, box in enumerate(gt_bboxes, start=1):
            if len(box) != 4:
                continue
            x, y, w, h = box
            return_prompt += f"{box_idx}[{x:.2f}, {y:.2f}, {x + w:.2f}, {y + h:.2f}] \n"
        return return_prompt
    
    elif answer_format == "3":
        if len(options) == 4 or 'yes' in options.values():
            return ""
        gt_bbox = _parse_json_maybe(doc.get("bboxes_by_item", ""))[gt_answer]["0"]
        # print(_parse_json_maybe(doc.get("bboxes_by_item", "")))
        # print(gt_answer)
        # print(gt_bbox)
        # print()

        if not gt_bbox:
            return ""
        
        color_idx = _get_color_idx(doc.get("bboxes_by_item", ""), gt_answer)
        if color_idx is None:
            return ""
        color = DEFAULT_COLOR_ORDER[color_idx % len(DEFAULT_COLOR_ORDER)]
        
        return f"The correct answer is the object with bounding box in the color: {color}."
    
    
    return ""

def doc_to_visual(doc):
    return [_load_image_from_doc(doc, use_overlay=False)]


def bbox_doc_to_visual(doc):
    return [_load_image_from_doc(doc, use_overlay=True)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Please select the correct answer from the options above. \n",
    )

    bbox_context = _format_bbox_context(doc)
    question = doc.get("question") or _build_fallback_question(doc)
    options = _build_options(doc)
    if os.getenv("LMMS_EVAL_INCLUDE_GT_HELP_TEXT", "0") in ["1", "2", "3"]:
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


def bbox_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    bbox_kwargs = dict(lmms_eval_specific_kwargs)
    bbox_kwargs.setdefault(
        "post_prompt",
        "Some queried objects are highlighted with bounding boxes. Please select the correct answer from the options above. \n",
    )
    return doc_to_text(doc, bbox_kwargs)


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
    gt_answer = _ground_truth_answer_letter(doc) or str(doc.get("answer", "")).strip()

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

    result_dict = {
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
