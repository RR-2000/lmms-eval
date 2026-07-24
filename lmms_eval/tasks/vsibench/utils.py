import os
import re
import json
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

DIR_TO_VECTOR = {
    "A. front-left": [1, 1],
    "B. front-right": [1, -1],
    "C. back-left": [-1, 1],
    "D. back-right": [-1, -1]
}

DIRECTION_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
}

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
overwrite_cache_dir = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["dataset_path"]


def vsibench_doc_to_visual(doc):
    if os.path.exists(overwrite_cache_dir):
        cache_dir = overwrite_cache_dir
    else:
        cache_dir = os.path.join(base_cache_dir, cache_name)
    
    doc["dataset"] = doc["dataset"].replace("_frames_24", "_frames_6")
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    if "bboxes" in doc['dataset']:
        video_path_alt = doc["dataset"].replace("bboxes", "bboxes_dense") + "/" + doc["scene_name"] + ".mp4"
        if os.path.exists(os.path.join(cache_dir, video_path_alt)):
            video_path = video_path_alt
        
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists('_'.join(video_path[:-4].split('_')[:-1])+'.mp4'):
        video_path = '_'.join(video_path[:-4].split('_')[:-1]) + '.mp4'
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def path_to_resolution(video_path):
    # without loading the video, check the metadata to get the resolution

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    try:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        video.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Unable to read video resolution from metadata: {video_path}")

    return width, height

def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None, include_pred_vec=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    LOC_TEXT = os.getenv("LMMS_EVAL_INCLUDE_LOCATION_TEXT", "0") == "1"
    
    if LOC_TEXT and ("object_rel" in doc["question_type"] or doc["question_type"] in ["obj_appearance_order", "object_counting", "object_size_estimation"]):
        vid_path = vsibench_doc_to_visual(doc)[0]
        width, height = 1,1 # path_to_resolution(vid_path)
        vid_path = vid_path.replace(".mp4", "/")
        objects = sorted(os.listdir(vid_path))
        obj_names = [obj.replace('.json', '').replace('bboxes_', '') for obj in objects]
        max_frames = 32
        frame_interval = 1
        fps = 12
        obj_bboxes = []
        pre_prompt += (
            f" The video contains the following objects: {', '.join(obj_names)}. "
            "Each object may have bounding box annotations for each frame, given in the "
            "format (coordinates are in fractions of dimensions): "
            "time:[<object_name>[x_min, y_min, x_max, y_max], "
            "<object_name>[x_min, y_min, x_max, y_max]].\n"
        )
        for obj in objects:
            json_file = os.path.join(vid_path, obj)
            if os.path.isfile(json_file) and obj.endswith('.json'):
                with open(json_file, 'r') as f:
                    data = yaml.safe_load(f)
                    num_frames = len(data)
                    if num_frames < max_frames:
                        raise ValueError(f"Number of frames for object {obj} is less than max_frames ({num_frames} < {max_frames}).")
                    frame_interval = max(1, num_frames // max_frames)
                    sampled_frames = [data[str(i)] for i in range(0, num_frames, frame_interval)][:max_frames]
                    obj_bboxes.append(sampled_frames)
                    if len(obj_bboxes) >= 1:
                        assert len(obj_bboxes[-1]) == len(obj_bboxes[0]), f"Number of sampled frames for object {obj} is different from the first object ({len(obj_bboxes[-1])} != {len(obj_bboxes[0])})."
        
        for frame_idx in range(len(obj_bboxes[0])):
            time = frame_interval * frame_idx / fps
            frame_entries = []
            for obj_idx, obj in enumerate(obj_names):
                bbox = list(obj_bboxes[obj_idx][frame_idx])

                for bbox_idx, bbox_i in enumerate(bbox):
                    frame_entries.append(
                        f"{obj}_{bbox_idx+1}[{bbox_i[0]*width:.2f}, {bbox_i[1]*height:.2f}, {(bbox_i[0]+bbox_i[2])*width:.2f}, {(bbox_i[1] + bbox_i[3])*height:.2f}]"
                    )
            if len(frame_entries) > 0:
                pre_prompt += f" At time {time:.2f}s:[{', '.join(frame_entries)}]\n"
    # print(f"pre_prompt: {pre_prompt}")
    post_question = ""
    if include_pred_vec and doc["question_type"] in DIRECTION_QUESTION_TYPES:
        answer = doc.get("ground_truth", None)
        gt_dir = doc.get("options", ["A. front-left", "B. front-right", "C. back-left", "D. back-right"])[ord(answer) - ord("A")] if answer is not None else None
        gt_vec = DIR_TO_VECTOR.get(gt_dir, None) if gt_dir is not None else None
        if gt_vec is not None:
            post_question += f"The right is +X and forward is +Y with the origin at the reference object and facing in the direction mentioned.\n"

    # exit()
    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, post_question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def vsibench_vector_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Prompt the direction variants for both the option letter and its vector."""
    return vsibench_doc_to_text(doc, lmms_eval_specific_kwargs, include_pred_vec=True)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


INVERSE_DIRECTION_TASK = "object_rel_direction_hard_inverse"
_HARD_DIRECTION_RE = re.compile(
    r"standing by the (.+?) and facing the (.+?), is the (.+?) to my",
    re.IGNORECASE,
)
_DIRECTION_LABEL_RE = re.compile(r"^[A-D][.) :]\s*", re.IGNORECASE)
_VSI_JSONL_PATH = "/home/ramanathan/data/VSI-Bench_new/test.jsonl"


@lru_cache(maxsize=1)
def _load_vsi_scene_object_pools():
    """Collect object names mentioned by non-hard-direction rows per scene."""
    jsonl_path = os.getenv("VSI_BENCH_JSONL", _VSI_JSONL_PATH)
    pools = {}
    with open(jsonl_path, encoding="utf-8") as data_file:
        rows = [json.loads(line) for line in data_file if line.strip()]

    for row in rows:
        if row.get("question_type") == "object_rel_direction_hard":
            continue
        question = str(row.get("question", ""))
        objects = []

        # Object-counting, size, and distance questions.
        objects.extend(re.findall(r"How many (.+?)\(s\)", question, re.IGNORECASE))
        objects.extend(
            re.findall(
                r"longest dimension \(length, width, or height\) of the (.+?), measured",
                question,
                re.IGNORECASE,
            )
        )
        objects.extend(
            value
            for match in re.findall(r"between the (.+?) and the (.+?) \(", question, re.IGNORECASE)
            for value in match
        )

        # Multiple-choice questions often provide the scene objects directly.
        for match in re.findall(r"\(([^()]*)\)", question):
            if "," in match:
                objects.extend(match.split(","))
        appearance_match = re.search(
            r"appearance order of the following categories in the video:\s*(.+?)(?:\n|$)",
            question,
            re.IGNORECASE,
        )
        if appearance_match:
            objects.extend(appearance_match.group(1).split(","))

        scene_pool = pools.setdefault(row.get("scene_name"), set())
        scene_pool.update(
            value.strip().lower()
            for value in objects
            if value.strip() and not value.strip().isdigit()
        )
    return pools


def _hard_direction_parts(doc):
    match = _HARD_DIRECTION_RE.search(str(doc.get("question", "")))
    if not match:
        raise ValueError(f"Could not parse hard-direction question: {doc.get('question')!r}")
    return tuple(part.strip() for part in match.groups())


def _direction_text(doc):
    answer = str(doc.get("ground_truth", "")).strip().upper()
    options = doc.get("options") or []
    index = ord(answer) - ord("A")
    if not 0 <= index < len(options):
        raise ValueError(f"Invalid hard-direction answer/options for row {doc.get('id')}")
    return _DIRECTION_LABEL_RE.sub("", str(options[index])).strip().lower()


def _inverse_object_options(doc, target, anchor, facing):
    pool = set(_load_vsi_scene_object_pools().get(doc.get("scene_name"), set()))
    target = target.lower()
    pool.add(target)
    # The answer is always present; distractors are sampled deterministically
    # from objects mentioned by other task families in this same scene.
    distractors = sorted(pool - {target, anchor.lower(), facing.lower()})
    seed_text = f"{doc.get('scene_name', '')}:{doc.get('id', '')}"
    seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_text)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    if len(distractors) > 3:
        distractors = list(rng.choice(distractors, size=3, replace=False))
    options = [target] + distractors
    rng.shuffle(options)
    return [str(option) for option in options]


def vsibench_inverse_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Create paired original and object-answer inverse hard-direction rows."""
    records = []
    for source_doc in dataset:
        if source_doc.get("question_type") != "object_rel_direction_hard":
            continue
        anchor, facing, target = _hard_direction_parts(source_doc)
        direction = _direction_text(source_doc)
        options = _inverse_object_options(source_doc, target, anchor, facing)
        inverse_answer = chr(ord("A") + options.index(target.lower()))
        pair_id = str(source_doc.get("id", source_doc.get("question_id")))

        direct = dict(source_doc)
        direct.update(
            {
                "question_type": INVERSE_DIRECTION_TASK,
                "pair_id": pair_id,
                "variant": "direct",
                "original_question": source_doc["question"],
                "inverse_direction": direction,
                "direct_ground_truth": source_doc["ground_truth"],
                "inverse_ground_truth": inverse_answer,
            }
        )
        records.append(direct)

        inverse = dict(source_doc)
        inverse.update(
            {
                "question_type": INVERSE_DIRECTION_TASK,
                "question": (
                    f"If I am standing by the {anchor} and facing the {facing}, "
                    f"what is to my {direction}?"
                ),
                "options": [f"{letter}. {value}" for letter, value in zip("ABCD", options)],
                "ground_truth": inverse_answer,
                "pair_id": pair_id,
                "variant": "inverse",
                "original_question": source_doc["question"],
                "inverse_direction": direction,
                "direct_ground_truth": source_doc["ground_truth"],
                "inverse_ground_truth": inverse_answer,
            }
        )
        records.append(inverse)

    eval_logger.info("VSiBench inverse hard-direction task created %d paired rows", len(records) // 2)
    return datasets.Dataset.from_list(records)


def vsibench_inverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    options = "\n".join(str(option) for option in doc.get("options", []))
    return (
        "Answer the spatial reasoning question using the video. "
        "Respond with the letter of the correct option.\n"
        f"Question: {doc['question']}\nOptions:\n{options}\n"
    )


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def _extract_answer_letter(prediction):
    prediction = str(prediction).strip()
    labeled = re.search(r"(?:answer|option)\s*(?:is|:|=)?\s*([A-D])\b", prediction, re.I)
    if labeled:
        return labeled.group(1).upper()
    leading = re.match(r"\s*([A-D])(?:\s*[.) :]\s*|$)", prediction, re.I)
    return leading.group(1).upper() if leading else None


def _extract_direction_vector(prediction):
    number = r"[-+]?\d+(?:\.\d+)?"
    match = re.search(
        rf"(?:dir(?:ection)?\s*)?vector\s*[:=]?\s*[\"'`]?\s*"
        rf"\[\s*({number})\s*,\s*({number})\s*\]",
        str(prediction),
        re.I,
    )
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


def _direction_option_vector(option):
    text = re.sub(r"^\s*[A-D]\s*[.):]\s*", "", str(option), flags=re.I)
    text = text.lower().replace("_", "-").replace(" ", "-")
    for direction, vector in DIR_TO_VECTOR.items():
        direction_text = direction.split(". ", 1)[-1].replace(" ", "-")
        if direction_text == text or direction_text in text:
            return tuple(float(value) for value in vector)
    return None


def _direction_ground_truth_vector(doc):
    answer = str(doc.get("ground_truth", "")).strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "Z":
        return None
    options = doc.get("options") or [
        "A. front-left", "B. front-right", "C. back-left", "D. back-right"
    ]
    index = ord(answer) - ord("A")
    return _direction_option_vector(options[index]) if index < len(options) else None


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def vsibench_process_results(doc, results):
    doc["prediction"] = results[0]
    print(results[0])
    
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc["prediction"])), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {
        "vsibench_overall": doc,
        "submission": _submission_record(doc),
        "obj_appearance_order_accuracy": doc,
        "object_abs_distance_mra": doc,
        "object_counting_mra": doc,
        "object_rel_distance_accuracy": doc,
        "object_size_estimation_mra": doc,
        "room_size_estimation_mra": doc,
        "route_planning_accuracy": doc,
        "object_rel_direction_accuracy": doc,
    }


def _submission_record(doc):
    """Return the serializable fields needed to reproduce a VSiBench result."""
    fields = (
        "id",
        "question_id",
        "dataset",
        "scene_name",
        "question_type",
        "question",
        "options",
        "ground_truth",
        "prediction",
    )
    return {field: doc[field] for field in fields if field in doc}


def vsibench_aggregate_submission(results, args):
    """Save predictions for the current VSiBench task as a JSON submission."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = generate_submission_file(f"vsibench_submission_{timestamp}.json", args)
    with open(path, "w", encoding="utf-8") as submission_file:
        json.dump(results, submission_file, ensure_ascii=False, indent=2, default=str)
    eval_logger.info(f"VSiBench submission saved to {path}")
    return path


def vsibench_inverse_process_results(doc, results):
    """Score either member of a direct/inverse hard-direction pair."""
    prediction = str(results[0]).strip() if results else ""
    parsed = _extract_answer_letter(prediction)
    score = float(parsed == str(doc.get("ground_truth", "")).strip().upper())
    entry = {
        "id": doc.get("id"),
        "pair_id": doc.get("pair_id"),
        "variant": doc.get("variant"),
        "score": score,
    }
    submission = {
        **entry,
        "scene_name": doc.get("scene_name"),
        "original_question": doc.get("original_question"),
        "question": doc.get("question"),
        "options": doc.get("options"),
        "prediction": prediction,
        "parsed_prediction": parsed,
        "ground_truth": doc.get("ground_truth"),
        "inverse_direction": doc.get("inverse_direction"),
    }
    return {
        "submission": submission,
        "direct_accuracy": entry,
        "inverse_accuracy": entry,
        "difference": entry,
    }


def _aggregate_inverse_accuracy(results, variant):
    scores = [row["score"] for row in results if row.get("variant") == variant]
    return round(float(np.mean(scores)), 6) if scores else 0.0


def vsibench_aggregate_direct_accuracy(results):
    return _aggregate_inverse_accuracy(results, "direct")


def vsibench_aggregate_inverse_accuracy(results):
    return _aggregate_inverse_accuracy(results, "inverse")


def vsibench_aggregate_inverse_difference(results):
    paired = {}
    for row in results:
        pair = paired.setdefault(row.get("pair_id"), {})
        pair[row.get("variant")] = row["score"]
    differences = [
        pair["inverse"] - pair["direct"]
        for pair in paired.values()
        if "direct" in pair and "inverse" in pair
    ]
    return round(float(np.mean(differences)), 6) if differences else 0.0


def vsibench_aggregate_inverse_submission(results, args):
    """Save both direct and inverse predictions in one submission JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = generate_submission_file(
        f"vsibench_object_rel_direction_hard_inverse_{timestamp}.json", args
    )
    with open(path, "w", encoding="utf-8") as submission_file:
        json.dump(results, submission_file, ensure_ascii=False, indent=2, default=str)
    eval_logger.info(f"VSiBench inverse hard-direction submission saved to {path}")
    return path


VECTOR_METRIC_NAMES = (
    "object_rel_direction_answer_accuracy",
    "object_rel_direction_vector_accuracy",
    "object_rel_direction_both_correct",
    "object_rel_direction_answer_correct_vector_wrong",
    "object_rel_direction_answer_wrong_vector_correct",
    "object_rel_direction_both_wrong",
)


def vsibench_process_direction_vector_results(doc, results):
    """Score the answer letter and direction vector independently and jointly."""
    prediction = results[0]
    answer_correct = _extract_answer_letter(prediction) == str(doc.get("ground_truth", "")).strip().upper()
    predicted_vector = _extract_direction_vector(prediction)
    target_vector = _direction_ground_truth_vector(doc)
    vector_correct = (
        predicted_vector is not None
        and target_vector is not None
        and np.allclose(predicted_vector, target_vector)
    )

    outcomes = {
        "object_rel_direction_answer_accuracy": float(answer_correct),
        "object_rel_direction_vector_accuracy": float(vector_correct),
        "object_rel_direction_both_correct": float(answer_correct and vector_correct),
        "object_rel_direction_answer_correct_vector_wrong": float(answer_correct and not vector_correct),
        "object_rel_direction_answer_wrong_vector_correct": float(not answer_correct and vector_correct),
        "object_rel_direction_both_wrong": float(not answer_correct and not vector_correct),
    }
    doc["prediction"] = prediction
    doc.update(outcomes)
    return {
        **{metric: doc for metric in VECTOR_METRIC_NAMES},
        "submission": _submission_record(doc),
    }


def _aggregate_direction_vector_metric(results, metric):
    if not results:
        return 0.0
    return round(float(np.mean([row[metric] for row in results])), 6)


def vsibench_aggregate_direction_answer_accuracy(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_answer_accuracy")


def vsibench_aggregate_direction_vector_accuracy(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_vector_accuracy")


def vsibench_aggregate_direction_both_correct(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_both_correct")


def vsibench_aggregate_direction_answer_correct_vector_wrong(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_answer_correct_vector_wrong")


def vsibench_aggregate_direction_answer_wrong_vector_correct(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_answer_wrong_vector_correct")


def vsibench_aggregate_direction_both_wrong(results):
    return _aggregate_direction_vector_metric(results, "object_rel_direction_both_wrong")


def _compute_all_subscores(results) -> dict:
    """Compute all sub-category scores from raw results. Shared logic for all aggregation functions."""
    df = pd.DataFrame(results)
    output = {}

    for question_type, question_type_indexes in df.groupby("question_type").groups.items():
        per_question_type = df.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    direction_keys = [
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    ]
    direction_scores = [output.pop(key) for key in direction_keys if key in output]
    output["object_rel_direction_accuracy"] = (
        sum(direction_scores) / len(direction_scores) if direction_scores else 0.0
    )

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    return output


def vsibench_aggregate_overall(results):
    output = _compute_all_subscores(results)
    eval_logger.info(f"Evaluation results: {output}")
    return round(output["overall"], 6)


def vsibench_aggregate_obj_appearance_order_accuracy(results):
    return round(_compute_all_subscores(results).get("obj_appearance_order_accuracy", 0.0), 6)


def vsibench_aggregate_object_abs_distance_mra(results):
    return round(_compute_all_subscores(results).get("object_abs_distance_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_object_counting_mra(results):
    return round(_compute_all_subscores(results).get("object_counting_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_object_rel_distance_accuracy(results):
    return round(_compute_all_subscores(results).get("object_rel_distance_accuracy", 0.0), 6)


def vsibench_aggregate_object_size_estimation_mra(results):
    return round(_compute_all_subscores(results).get("object_size_estimation_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_room_size_estimation_mra(results):
    return round(_compute_all_subscores(results).get("room_size_estimation_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_route_planning_accuracy(results):
    return round(_compute_all_subscores(results).get("route_planning_accuracy", 0.0), 6)


def vsibench_aggregate_object_rel_direction_accuracy(results):
    return round(_compute_all_subscores(results).get("object_rel_direction_accuracy", 0.0), 6)


# Keep backward compatibility
def vsibench_aggregate_results(results):
    return vsibench_aggregate_overall(results)
