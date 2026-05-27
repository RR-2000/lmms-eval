import os
from pathlib import Path

import datasets
import pandas as pd
import yaml
from loguru import logger as eval_logger

TASK_NAMES = [
    "3D Video Grounding",
    "Dimensional Measurement",
    "Displacement & Path Length",
    "Ego-Centric Orientation",
    "Pose Estimation",
    "Spatial Relation",
    "Speed & Acceleration",
    "Trajectory Description",
]


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace(" ", "_").replace("&", "and")


TASK_KEY_MAP = {name: _normalize_task_name(name) for name in TASK_NAMES}
MCA_QUESTION_TYPES = sorted(TASK_KEY_MAP.values())
NA_QUESTION_TYPES = []

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "stibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def stibench_doc_to_visual(doc):
    cache_dir = Path(base_cache_dir) / cache_name
    video_name = doc["Video"]
    source = doc.get("Source", "")

    candidate_paths = [
        cache_dir / video_name,
        cache_dir / source / video_name,
        cache_dir / "videos" / video_name,
        cache_dir / "video" / video_name,
        cache_dir / source / "videos" / video_name,
        cache_dir / source / "video" / video_name,
    ]

    for path in candidate_paths:
        if path.exists():
            return [str(path)]

    tried = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"video path not found. tried: {tried}")


def stibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    question = doc["Question"].strip()
    prompt = (doc.get("Prompt") or "").strip()
    question_type = (doc.get("Task") or "").strip().lower()

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."

    if question_type in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."

    ts,te = doc.get("time_start"), doc.get("time_end")
    if ts is not None and te is not None:
        pre_prompt += f" The question is about the video segment from {ts} seconds to {te} seconds."
    options = []
    candidates = doc.get("Candidates") or {}
    if isinstance(candidates, dict):
        for key in sorted(candidates.keys()):
            options.append(f"{key}. {candidates[key]}")
    if options:
        options_block = "Options:\n" + "\n".join(options)
    else:
        options_block = ""

    parts = [pre_prompt]
    if prompt:
        parts.append(prompt)
    parts.append(question)
    if options_block:
        parts.append(options_block)
    parts.append(post_prompt)
    return "\n".join(parts)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred: str) -> str:
    pred = (pred or "").strip()
    for char in pred:
        if char.isalpha():
            return char.upper()
    return pred


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
}


def stibench_process_results(doc, results):
    doc["prediction"] = results[0]
    task_name = (doc.get("Task") or "").strip()
    task_key = _normalize_task_name(task_name) if task_name else ""
    doc["question_type"] = task_key
    ground_truth = (doc.get("Answer") or "").strip()

    if task_key in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), ground_truth)
    elif task_key in NA_QUESTION_TYPES:
        for key in METRICS_FOR_NA.keys():
            doc[key] = WORST_CASE_FOR_METRICS.get(key, 0.0)
    else:
        raise ValueError(f"Unknown task type: {task_name}")

    output = {"stibench_overall": doc}
    for task_key in MCA_QUESTION_TYPES:
        output[f"task_{task_key}_accuracy"] = doc
    return output


def _compute_all_subscores(results) -> dict:
    """Compute per-task accuracy scores from raw results."""
    df = pd.DataFrame(results)
    output = {}

    if "Task" not in df.columns:
        output["overall"] = df.get("accuracy", pd.Series([0.0])).mean()
        return output

    for task_name, task_indexes in df.groupby("Task").groups.items():
        per_task = df.iloc[task_indexes]
        task_key = _normalize_task_name(str(task_name))
        output[f"task_{task_key}_accuracy"] = per_task["accuracy"].mean()

    output["overall"] = sum(output.values()) / len(output) if output else 0.0
    return output


def stibench_aggregate_overall(results):
    output = _compute_all_subscores(results)
    eval_logger.info(f"Evaluation results: {output}")
    return round(output["overall"], 6)


def _aggregate_task(results, task_key: str) -> float:
    output = _compute_all_subscores(results)
    return round(output.get(f"task_{task_key}_accuracy", 0.0), 6)


def stibench_aggregate_3d_video_grounding_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["3D Video Grounding"])


def stibench_aggregate_dimensional_measurement_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Dimensional Measurement"])


def stibench_aggregate_displacement_and_path_length_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Displacement & Path Length"])


def stibench_aggregate_ego_centric_orientation_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Ego-Centric Orientation"])


def stibench_aggregate_pose_estimation_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Pose Estimation"])


def stibench_aggregate_spatial_relation_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Spatial Relation"])


def stibench_aggregate_speed_and_acceleration_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Speed & Acceleration"])


def stibench_aggregate_trajectory_description_accuracy(results):
    return _aggregate_task(results, TASK_KEY_MAP["Trajectory Description"])


# # Backward compatibility for existing configs
# vsibench_doc_to_visual = stibench_doc_to_visual
# vsibench_doc_to_text = stibench_doc_to_text
# vsibench_process_results = stibench_process_results
# vsibench_aggregate_overall = stibench_aggregate_overall
# vsibench_aggregate_results = stibench_aggregate_overall
