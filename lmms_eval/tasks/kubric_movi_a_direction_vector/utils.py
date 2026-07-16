"""Paired camera/object-direction diagnostic for Kubric MOVi-A.

Only object-centric relative-position items are retained: they provide an
anchor object, one or more visible targets, and a text answer that can be
matched across direction-answer and object-answer prompt formats. Every
vector is expressed in the camera-aligned, camera-centered viewpoint frame;
the discrete text answer separately follows the source task's reference-object
position and orientation. Every matched text prompt is evaluated twice, once
asking for camera-to-object vectors and once for object-to-camera vectors.
"""

import json
import random
import re
from typing import Optional

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.kubric_movi_a.utils import _get_image_path, _get_object_record, _get_options, doc_to_visual
from lmms_eval.tasks.kubric_movi_a_viewpoint.utils import _camera_frame_position, _camera_frame_vector_from_world, _cosine_similarity, _sign, _vector_sub
from lmms_eval.utils import sanitize_model_name


SOURCE_FAMILIES = {"object_centric_relative_position", "object_centric_relative_position_multi"}
DIRECTIONS = ("left", "right", "front", "behind")
VECTOR_VARIANTS = ("camera_to_object", "object_to_camera")
SAMPLE_SEED = "kubric_movi_a_direction_vector_v1"
AXES = ("right", "up", "front")


def _answer_text(doc: dict) -> str:
    answer = str(doc.get("answer", "")).strip()
    return str(_get_options(doc).get(answer, answer)).strip()


def _relation(doc: dict) -> Optional[str]:
    relation = str((doc.get("task_metadata") or {}).get("relation", "")).strip().lower()
    return relation if relation in DIRECTIONS else None


def _relation_phrase(relation: str) -> str:
    return {"left": "to the left of", "right": "to the right of", "front": "in front of", "behind": "behind"}[relation]


def _set_options(doc: dict, values: list[str], answer_value: str) -> None:
    for index, letter in enumerate("ABCD"):
        doc[letter] = values[index] if index < len(values) else None
    doc["answer"] = "ABCD"[values.index(answer_value)]


def _multiple_object_options(doc: dict, anchor: str, target: str, source_qid: str) -> list[str]:
    names = [str(obj.get("name")) for obj in doc.get("visible_objects", []) if obj.get("name")]
    candidates = [target] + ([anchor] if anchor != target else [])
    candidates.extend(name for name in names if name not in candidates)
    candidates = candidates[:4]
    random.Random(source_qid).shuffle(candidates)
    return candidates


def process_docs(dataset: Dataset) -> Dataset:
    """Balance the two source families and make text-format × vector-direction pairs."""
    by_family = {family: [] for family in SOURCE_FAMILIES}
    for doc in dataset:
        if doc.get("task_family") in by_family:
            by_family[doc["task_family"]].append(doc)
    count = min((len(rows) for rows in by_family.values()), default=0)
    sampler = random.Random(SAMPLE_SEED)
    sources = [doc for family in sorted(by_family) for doc in sampler.sample(by_family[family], count)]
    records, skipped = [], 0

    for doc in sources:
        metadata = doc.get("task_metadata") or {}
        relation = _relation(doc)
        anchor = str(metadata.get("anchor_object", "")).strip()
        target = str(metadata.get("target_object") if doc["task_family"] == "object_centric_relative_position" else metadata.get("correct_object", _answer_text(doc))).strip()
        if not (relation and anchor and target and _get_object_record(doc, anchor) and _get_object_record(doc, target)):
            skipped += 1
            continue
        source_qid = str(doc.get("qid", doc.get("index", "unknown")))
        common = dict(doc)
        common.update({"source_qid": source_qid, "source_task_family": doc["task_family"], "diagnostic_anchor": anchor, "diagnostic_target": target, "diagnostic_relation": relation})
        text_variants = []
        native = dict(common)
        native.update({"diagnostic_text_variant": "native", "diagnostic_answer_format": "direction" if doc["task_family"] == "object_centric_relative_position" else "object", "diagnostic_text_target": relation if doc["task_family"] == "object_centric_relative_position" else target})
        text_variants.append(native)
        inverse = dict(common)
        inverse.update({"diagnostic_text_variant": "inverse"})
        if doc["task_family"] == "object_centric_relative_position":
            _set_options(inverse, _multiple_object_options(doc, anchor, target, source_qid), target)
            inverse["question"] = f"Imagine standing at the {anchor} and facing the camera. Which object is {_relation_phrase(relation)} the {anchor}?"
            inverse.update({"diagnostic_answer_format": "object", "diagnostic_text_target": target})
        else:
            _set_options(inverse, list(DIRECTIONS), relation)
            inverse["question"] = f"Imagine standing at the {anchor} and facing the camera. Where is the {target} relative to the {anchor}?"
            inverse.update({"diagnostic_answer_format": "direction", "diagnostic_text_target": relation})
        text_variants.append(inverse)
        for text_doc in text_variants:
            for vector_variant in VECTOR_VARIANTS:
                variant = dict(text_doc)
                variant.update({"qid": f"{source_qid}::{text_doc['diagnostic_text_variant']}::{vector_variant}", "index": f"{source_qid}::{text_doc['diagnostic_text_variant']}::{vector_variant}", "diagnostic_vector_variant": vector_variant})
                records.append(variant)
    eval_logger.info("Kubric direction/vector task sampled %d items per family (%d records).", count, len(records))
    if skipped:
        eval_logger.warning("Skipped %d source items without usable 3D object records.", skipped)
    return Dataset.from_list(records)


def _gt_vectors(doc: dict) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    names = [doc["diagnostic_anchor"], doc["diagnostic_target"]]
    if doc.get("source_task_family") == "object_centric_relative_position_multi":
        names.extend(str(name) for name in (doc.get("task_metadata") or {}).get("candidate_objects", []) if name)
    vectors = {}
    for name in dict.fromkeys(names):
        obj = _get_object_record(doc, name)
        position = obj.get("position_3d") if obj else None
        camera_to_object = _camera_frame_position(doc, position) if position else None
        if camera_to_object:
            vectors[name] = {"camera_to_object": camera_to_object, "object_to_camera": {axis: -value for axis, value in camera_to_object.items()}}
    anchor = _get_object_record(doc, doc["diagnostic_anchor"])
    target = _get_object_record(doc, doc["diagnostic_target"])
    between = {axis: 0.0 for axis in AXES}
    if anchor and target and anchor.get("position_3d") and target.get("position_3d"):
        # This remains a camera-frame vector even though the text label below
        # is evaluated with the source task's anchor-relative convention.
        between = _camera_frame_vector_from_world(doc, _vector_sub(target["position_3d"], anchor["position_3d"]))
    return vectors, between


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_vector(value) -> dict[str, float]:
    value = value if isinstance(value, dict) else {}
    return {axis: _number(value.get(axis, value.get({"right": "x", "up": "y", "front": "z"}[axis], 0.0))) for axis in AXES}


def _parse_prediction(text: str) -> dict:
    payload = _extract_json(text)
    raw_objects = payload.get("object_vectors", {})
    objects = {}
    if isinstance(raw_objects, dict):
        for name, vector in raw_objects.items():
            objects[str(name)] = _parse_vector(vector)
    return {"payload": payload, "answer": str(payload.get("answer", "")).strip(), "target_object": str(payload.get("target_object", "")).strip(), "object_vectors": objects, "between_objects": _parse_vector(payload.get("between_objects"))}


def _axis_alignment(pred: dict[str, float], gold: dict[str, float]) -> float:
    informative = [axis for axis in AXES if _sign(gold[axis]) != 0]
    return sum(_sign(pred[axis]) == _sign(gold[axis]) for axis in informative) / len(informative) if informative else 0.0


def _matches_text(prediction: dict, doc: dict) -> float:
    expected = str(doc["diagnostic_text_target"]).strip().lower()
    values = [prediction["answer"].strip().lower(), prediction["target_object"].strip().lower()]
    return float(expected in values)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    requested = doc["diagnostic_vector_variant"]
    direction = "from the camera to each object" if requested == "camera_to_object" else "from each object to the camera"
    names = list(_gt_vectors(doc)[0])
    lines = [
        f"Question: {doc['question']}",
        "Return only valid JSON.",
        "All vectors use the canonical viewpoint frame: camera center at the origin; right=+X, front=+Y, up=+Z.",
        f"For every listed object, estimate a direction vector {direction} in that frame.",
        f"Listed objects: {', '.join(names)}.",
        "Give `between_objects` as target-minus-anchor, also in that camera frame (right/up/front).",
        "Keep vector coordinates separate from the text answer: the text answer must use the direction word or object name implied by the question's reference object and orientation, never an option letter.",
        'JSON schema: {"answer":"<text answer>","target_object":"<selected object when applicable>","object_vectors":{"<object name>":{"right":<float>,"up":<float>,"front":<float>}},"between_objects":{"right":<float>,"up":<float>,"front":<float>}}',
    ]
    return "\n".join(lines)


def doc_to_target(doc):
    vectors, between = _gt_vectors(doc)
    requested = doc["diagnostic_vector_variant"]
    return json.dumps({"answer": doc["diagnostic_text_target"], "target_object": doc["diagnostic_target"], "object_vectors": {name: values[requested] for name, values in vectors.items()}, "between_objects": between})


def process_results(doc, results):
    raw = results[0].strip() if results else ""
    prediction = _parse_prediction(raw)
    gold_vectors, gold_between = _gt_vectors(doc)
    requested = doc["diagnostic_vector_variant"]
    cosine_scores, axis_scores = [], []
    for name, vectors in gold_vectors.items():
        if name in prediction["object_vectors"]:
            predicted = prediction["object_vectors"][name]
            cosine_scores.append(_cosine_similarity(predicted, vectors[requested]))
            axis_scores.append(_axis_alignment(predicted, vectors[requested]))
        else:
            cosine_scores.append(0.0)
            axis_scores.append(0.0)
    vector_cosine = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
    vector_axis = sum(axis_scores) / len(axis_scores) if axis_scores else 0.0
    between_cosine = _cosine_similarity(prediction["between_objects"], gold_between)
    between_axis = _axis_alignment(prediction["between_objects"], gold_between)
    text_correct = _matches_text(prediction, doc)
    directions_correct = float(vector_axis == 1.0 and between_axis == 1.0)
    entry = {"qid": doc.get("qid"), "source_qid": doc.get("source_qid"), "task_family": doc.get("source_task_family"), "text_variant": doc.get("diagnostic_text_variant"), "vector_variant": requested, "text_answer_accuracy": text_correct, "vector_cosine": vector_cosine, "vector_axis_alignment": vector_axis, "between_objects_cosine": between_cosine, "between_objects_axis_alignment": between_axis, "directions_correct": directions_correct}
    entry.update({"answer_and_direction_correct": float(text_correct and directions_correct), "answer_correct_direction_wrong": float(text_correct and not directions_correct), "answer_wrong_direction_correct": float(not text_correct and directions_correct), "answer_and_direction_wrong": float(not text_correct and not directions_correct)})
    return {"text_answer_accuracy": entry, "camera_to_object_cosine": entry, "object_to_camera_cosine": entry, "camera_to_object_axis_alignment": entry, "object_to_camera_axis_alignment": entry, "between_objects_cosine": entry, "between_objects_axis_alignment": entry, "answer_and_direction_correct": entry, "answer_correct_direction_wrong": entry, "answer_wrong_direction_correct": entry, "answer_and_direction_wrong": entry, "submission": {**entry, "question_prompt": doc_to_text(doc), "prediction": raw, "parsed_prediction": prediction, "gold_vectors": gold_vectors, "gold_between_objects": gold_between, "img_path": _get_image_path(doc)}}


def _mean(results, key, variant=None):
    selected = [float(row[key]) for row in results if variant is None or row.get("vector_variant") == variant]
    return sum(selected) / len(selected) if selected else 0.0


def aggregate_text_answer_accuracy(results): return _mean(results, "text_answer_accuracy")
def aggregate_camera_to_object_cosine(results): return _mean(results, "vector_cosine", "camera_to_object")
def aggregate_object_to_camera_cosine(results): return _mean(results, "vector_cosine", "object_to_camera")
def aggregate_camera_to_object_axis_alignment(results): return _mean(results, "vector_axis_alignment", "camera_to_object")
def aggregate_object_to_camera_axis_alignment(results): return _mean(results, "vector_axis_alignment", "object_to_camera")
def aggregate_between_objects_cosine(results): return _mean(results, "between_objects_cosine")
def aggregate_between_objects_axis_alignment(results): return _mean(results, "between_objects_axis_alignment")
def aggregate_answer_and_direction_correct(results): return _mean(results, "answer_and_direction_correct")
def aggregate_answer_correct_direction_wrong(results): return _mean(results, "answer_correct_direction_wrong")
def aggregate_answer_wrong_direction_correct(results): return _mean(results, "answer_wrong_direction_correct")
def aggregate_answer_and_direction_wrong(results): return _mean(results, "answer_and_direction_wrong")


def aggregate_results_for_submission(results, args):
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    path = generate_submission_file(f"kubric_movi_a_direction_vector_{model}.json", args)
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    eval_logger.info("Kubric direction/vector records saved to %s.", path)
