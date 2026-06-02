import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger

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

def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    LOC_TEXT = os.getenv("LMMS_EVAL_INCLUDE_LOCATION_TEXT", "0") == "1"
    
    if LOC_TEXT and ("object_rel" in doc["question_type"] or doc["question_type"] in ["obj_appearance_order", "object_counting", "object_size_estimation"]):
        vid_path = vsibench_doc_to_visual(doc)[0]
        width, height = path_to_resolution(vid_path)
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
            "format: "
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
                        f"{obj}_{bbox_idx+1}[{bbox_i[0]*width:.2f}, {bbox_i[1]*height:.2f}, {bbox_i[2]*width:.2f}, {bbox_i[3]*height:.2f}]"
                    )
            if len(frame_entries) > 0:
                pre_prompt += f" At time {time:.2f}s:[{', '.join(frame_entries)}]\n"
    print(f"pre_prompt: {pre_prompt}")
    # exit()
    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


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
    # print(results[0])
    # exit()
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
        "obj_appearance_order_accuracy": doc,
        "object_abs_distance_mra": doc,
        "object_counting_mra": doc,
        "object_rel_distance_accuracy": doc,
        "object_size_estimation_mra": doc,
        "room_size_estimation_mra": doc,
        "route_planning_accuracy": doc,
        "object_rel_direction_accuracy": doc,
    }


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
