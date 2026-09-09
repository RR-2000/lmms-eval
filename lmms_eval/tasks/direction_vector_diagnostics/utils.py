"""Matched direction/vector diagnostics over COMFORT, ScanNet, and Kubric."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.comfort_direction_object_gt_help import utils as comfort_aids
from lmms_eval.tasks.comfort_multi_3d_object_basis import utils as basis
from lmms_eval.tasks.kubric_movi_a_viewpoint_clean import utils as kubric
from lmms_eval.tasks.scannet_camera_basis import utils as scannet
from lmms_eval.utils import sanitize_model_name

AXES = basis.AXES
SHARED_EXPERIMENTS = {
    "cross_output_consistency": ("direction_only", "vector_only", "combined_direction_first", "combined_vector_first"),
    "output_format_control": ("named_json", "ordered_list", "plain_text", "sign_json", "prototype_mc"),
    "conversion_oracle": ("vector_to_direction", "direction_to_vector"),
    "basis_oracle_ladder": (
        "rgb",
        "bboxes",
        "front_axis",
        "full_axes",
        "basis_text",
        "displacement_text",
        "basis_and_displacement",
        "gold_vector",
        "gold_direction",
    ),
    "component_decomposition": ("front_sign", "up_sign", "right_sign", "dominant_axis", "horizontal_direction"),
    "arrow_vector_grounding": ("plain", "front", "front_right", "full_labeled", "full_color"),
}
KUBRIC_ROTATIONS = (0, 90, 180, 270)
SCANNET_ORACLES = ("rgb", "bboxes", "depth_text", "centroids_text", "displacement_text")


def _comfort_base(dataset) -> list[dict]:
    scenes = {str(row["scene_id"]): dict(row) for row in dataset}
    rows = []
    for doc in basis._process_docs(dataset, "combined", "object"):
        scene = scenes[str(doc["scene_id"])]
        merged = {**scene, **dict(doc), "base_pair_id": doc["pair_id"], "dataset_name": "comfort"}
        objects = scene["objects"]
        reference = next(obj for obj in objects if obj.get("role") == "reference")
        target = next(obj for obj in objects if str(obj.get("object_id")) == str(doc["target_object_id"]))
        reference_position = [float(x) for x in reference["camera_frame"]["position"]]
        target_position = [float(x) for x in target["camera_frame"]["position"]]
        merged.update(
            {
                "reference_bbox": reference["bbox_2d_normalized_xyxy"],
                "target_bbox": target["bbox_2d_normalized_xyxy"],
                "reference_position": reference_position,
                "target_position": target_position,
                "camera_displacement": [target_position[i] - reference_position[i] for i in range(3)],
                "basis_camera": doc["object_basis_camera_frame"],
            }
        )
        rows.append(merged)
    return rows


def _scannet_base(dataset) -> list[dict]:
    views = {(str(row["scene_id"]), int(row["frame_id"])): dict(row) for row in dataset}
    rows = []
    for source in scannet._process_docs(dataset, "combined", "object_facing_camera"):
        doc = dict(source)
        view = views[(str(doc["scene_id"]), int(doc["frame_id"]))]
        objects = {str(obj["object_id"]): obj for obj in view["objects"]}
        reference = objects[str(doc["reference_object_id"])]
        target = objects[str(doc["target_object_id"])]
        rp = [float(x) for x in reference["position_camera_xyz_m"]]
        tp = [float(x) for x in target["position_camera_xyz_m"]]
        doc.update(
            {
                "base_pair_id": doc["pair_id"],
                "dataset_name": "scannet",
                "reference_bbox": reference["bbox_2d_normalized_xyxy"],
                "target_bbox": target["bbox_2d_normalized_xyxy"],
                "reference_position": rp,
                "target_position": tp,
                "camera_displacement": [tp[i] - rp[i] for i in range(3)],
                "basis_camera": doc["evaluation_basis_camera_frame"],
            }
        )
        rows.append(doc)
    return rows


def _kubric_bbox(obj):
    box = obj.get("bbox_2d_norm") if obj else None
    if box and len(box) == 4:
        ymin, xmin, ymax, xmax = map(float, box)
        return [xmin, ymin, xmax, ymax]
    return None


def _kubric_base(dataset) -> list[dict]:
    rows, skipped = [], Counter()
    for raw in kubric.process_object_centric_looking_back_docs(dataset):
        if raw.get("task_family") != "object_centric_relative_position":
            continue
        doc = dict(raw)
        try:
            spec = kubric._get_gt_spec(doc)
            reference = kubric._get_object_record(doc, spec["reference_object"])
            target = kubric._get_object_record(doc, spec["target_object"])
            vector = basis._unit([float(spec["vector"][axis]) for axis in AXES])
            vector = dict(zip(AXES, vector))
            direction = basis.direction_from_vector(vector)
            if not reference or not target or direction is None:
                raise ValueError
            image_path = str(kubric._get_image_path(doc))
            reference_world = [float(x) for x in reference["position_3d"]]
            target_world = [float(x) for x in target["position_3d"]]
            displacement_world = [target_world[i] - reference_world[i] for i in range(3)]
            displacement_camera_dict = kubric._camera_frame_vector_from_world(doc, displacement_world)
            displacement_camera = [
                displacement_camera_dict["right"],
                displacement_camera_dict["front"],
                displacement_camera_dict["up"],
            ]
            reference_camera_dict = kubric._camera_frame_position(doc, reference_world)
            target_camera_dict = kubric._camera_frame_position(doc, target_world)
            reference_camera = [reference_camera_dict["right"], reference_camera_dict["front"], reference_camera_dict["up"]]
            target_camera = [target_camera_dict["right"], target_camera_dict["front"], target_camera_dict["up"]]
            camera = kubric._get_camera_position(doc)
            front_world = basis._unit([camera[i] - reference_world[i] for i in range(3)])
            front_world[2] = 0.0
            front_world = basis._unit(front_world)
            right_world = basis._unit(basis._cross(front_world, [0.0, 0.0, 1.0]))

            def in_camera(world_axis):
                value = kubric._camera_frame_vector_from_world(doc, world_axis)
                return [value["right"], value["front"], value["up"]]

            base_id = str(doc.get("qid", doc.get("index")))
            doc.update(
                {
                    "base_pair_id": base_id,
                    "dataset_name": "kubric",
                    "prediction_format": "combined",
                    "gt_direction": direction,
                    "gt_vector": vector,
                    "reference_object": spec["reference_object"],
                    "target_object": spec["target_object"],
                    "reference_bbox": _kubric_bbox(reference),
                    "target_bbox": _kubric_bbox(target),
                    "reference_position": reference_camera,
                    "target_position": target_camera,
                    "camera_displacement": displacement_camera,
                    "basis_camera": {
                        "front": in_camera(front_world),
                        "up": in_camera([0.0, 0.0, 1.0]),
                        "right": in_camera(right_world),
                    },
                    "img_path": image_path,
                    "image_path": image_path,
                    "question": f"From the {spec['reference_object']}'s frame while it looks back at the camera, what is the 3D direction from the {spec['reference_object']} to the {spec['target_object']}?",
                }
            )
            rows.append(doc)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            skipped["invalid_geometry"] += 1
    eval_logger.info("Kubric shared diagnostic base loaded {} rows; skipped={}", len(rows), dict(skipped))
    return rows


def _dataset_base(dataset, dataset_name: str) -> list[dict]:
    return {"comfort": _comfort_base, "scannet": _scannet_base, "kubric": _kubric_base}[dataset_name](dataset)


def _expand_shared(dataset, dataset_name: str, experiment: str) -> Dataset:
    records = []
    for source in _dataset_base(dataset, dataset_name):
        for condition in SHARED_EXPERIMENTS[experiment]:
            doc = dict(source)
            doc.update(
                {
                    "diagnostic_experiment": experiment,
                    "experiment_condition": condition,
                    "qid": f"{source['base_pair_id']}::{experiment}::{condition}",
                    "index": f"{source['base_pair_id']}::{experiment}::{condition}",
                }
            )
            if experiment == "conversion_oracle" and condition == "direction_to_vector":
                canonical = {
                    "front": {"front": 1.0, "up": 0.0, "right": 0.0},
                    "back": {"front": -1.0, "up": 0.0, "right": 0.0},
                    "right": {"front": 0.0, "up": 0.0, "right": 1.0},
                    "left": {"front": 0.0, "up": 0.0, "right": -1.0},
                }
                doc["perceptual_gt_vector"] = doc["gt_vector"]
                doc["gt_vector"] = canonical[doc["gt_direction"]]
            records.append(doc)
    eval_logger.info("Direction/vector diagnostic {} loaded {} {} rows", experiment, len(records), dataset_name)
    return Dataset.from_list(records)


def process_cross_output_docs(dataset):
    return _expand_shared(dataset, "comfort", "cross_output_consistency")


def process_output_format_docs(dataset):
    return _expand_shared(dataset, "comfort", "output_format_control")


def process_conversion_oracle_docs(dataset):
    return _expand_shared(dataset, "comfort", "conversion_oracle")


def process_basis_oracle_docs(dataset):
    return _expand_shared(dataset, "comfort", "basis_oracle_ladder")


def process_component_docs(dataset):
    return _expand_shared(dataset, "comfort", "component_decomposition")


def process_arrow_grounding_docs(dataset):
    return _expand_shared(dataset, "comfort", "arrow_vector_grounding")


def process_scannet_cross_output_docs(dataset):
    return _expand_shared(dataset, "scannet", "cross_output_consistency")


def process_scannet_output_format_docs(dataset):
    return _expand_shared(dataset, "scannet", "output_format_control")


def process_scannet_conversion_oracle_docs(dataset):
    return _expand_shared(dataset, "scannet", "conversion_oracle")


def process_scannet_basis_oracle_docs(dataset):
    return _expand_shared(dataset, "scannet", "basis_oracle_ladder")


def process_scannet_component_docs(dataset):
    return _expand_shared(dataset, "scannet", "component_decomposition")


def process_scannet_arrow_grounding_docs(dataset):
    return _expand_shared(dataset, "scannet", "arrow_vector_grounding")


def process_kubric_cross_output_docs(dataset):
    return _expand_shared(dataset, "kubric", "cross_output_consistency")


def process_kubric_output_format_docs(dataset):
    return _expand_shared(dataset, "kubric", "output_format_control")


def process_kubric_conversion_oracle_docs(dataset):
    return _expand_shared(dataset, "kubric", "conversion_oracle")


def process_kubric_basis_oracle_docs(dataset):
    return _expand_shared(dataset, "kubric", "basis_oracle_ladder")


def process_kubric_component_docs(dataset):
    return _expand_shared(dataset, "kubric", "component_decomposition")


def process_kubric_arrow_grounding_docs(dataset):
    return _expand_shared(dataset, "kubric", "arrow_vector_grounding")


def process_kubric_rotation_docs(dataset) -> Dataset:
    records = []
    for source in _kubric_base(dataset):
        base_id = source["base_pair_id"]
        for rotation in KUBRIC_ROTATIONS:
            records.append(
                {
                    **source,
                    "qid": f"{base_id}::raster_rotation::{rotation}",
                    "index": f"{base_id}::raster_rotation::{rotation}",
                    "diagnostic_experiment": "kubric_rotation_equivariance",
                    "experiment_condition": f"rotate_{rotation}",
                    "raster_rotation_degrees": rotation,
                }
            )
    eval_logger.info("Kubric rotation diagnostic loaded {} rows", len(records))
    return Dataset.from_list(records)


def process_scannet_depth_docs(dataset) -> Dataset:
    views = {(str(row["scene_id"]), int(row["frame_id"])): dict(row) for row in dataset}
    records = []
    for source in scannet._process_docs(dataset, "combined", "camera"):
        view = views[(str(source["scene_id"]), int(source["frame_id"]))]
        objects = {str(obj["object_id"]): obj for obj in view["objects"]}
        reference, target = objects[source["reference_object_id"]], objects[source["target_object_id"]]
        for condition in SCANNET_ORACLES:
            pair_id = source["pair_id"]
            records.append(
                {
                    **dict(source),
                    "qid": f"{pair_id}::scannet_depth::{condition}",
                    "index": f"{pair_id}::scannet_depth::{condition}",
                    "base_pair_id": pair_id,
                    "dataset_name": "scannet",
                    "diagnostic_experiment": "scannet_rgb_depth_oracle",
                    "experiment_condition": condition,
                    "reference_bbox": reference["bbox_2d_normalized_xyxy"],
                    "target_bbox": target["bbox_2d_normalized_xyxy"],
                    "reference_position": reference["position_camera_xyz_m"],
                    "target_position": target["position_camera_xyz_m"],
                }
            )
    eval_logger.info("ScanNet RGB/depth diagnostic loaded {} rows", len(records))
    return Dataset.from_list(records)


def _process_boundary_docs(dataset, dataset_name: str) -> Dataset:
    records = []
    for source in _dataset_base(dataset, dataset_name):
        doc = dict(source)
        front, right = abs(float(doc["gt_vector"]["front"])), abs(float(doc["gt_vector"]["right"]))
        margin = abs(front - right)
        pair_id = doc["base_pair_id"]
        doc.update(
            {
                "qid": f"{pair_id}::angular_boundary",
                "index": f"{pair_id}::angular_boundary",
                "base_pair_id": pair_id,
                "diagnostic_experiment": "angular_boundary",
                "experiment_condition": "matched_combined",
                "angular_margin": margin,
                "angular_margin_bin": "0-0.1" if margin < 0.1 else "0.1-0.25" if margin < 0.25 else "0.25-0.5" if margin < 0.5 else ">=0.5",
            }
        )
        records.append(doc)
    eval_logger.info("{} angular-boundary diagnostic loaded {} rows", dataset_name, len(records))
    return Dataset.from_list(records)


def process_scannet_boundary_docs(dataset) -> Dataset:
    return _process_boundary_docs(dataset, "scannet")


def process_kubric_boundary_docs(dataset) -> Dataset:
    return _process_boundary_docs(dataset, "kubric")


def _bbox(image, box, color, label):
    if not box:
        return
    x1, y1, x2, y2 = box
    coords = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 180))
    draw.rectangle(coords, outline=color, width=width)
    draw.text((coords[0] + 2, max(0, coords[1] - 13)), label, fill=color)


def _project_axis(reference_position, axis):
    """Approximate a camera-XYZ 3D tangent as an image-plane direction."""
    x, y, z = map(float, reference_position)
    vx, vy, vz = map(float, axis)
    if abs(y) < basis.EPSILON:
        return None
    dx = (vx * y - x * vy) / (y * y)
    dy = -(vz * y - z * vy) / (y * y)
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-5 else None


def _arrow(draw, start, direction, length, color, label=None):
    dx, dy = direction
    end = (start[0] + dx * length, start[1] + dy * length)
    width = max(3, round(length / 18))
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(dy, dx)
    head = max(8, length * 0.22)
    points = [
        end,
        (end[0] - head * math.cos(angle - 0.5), end[1] - head * math.sin(angle - 0.5)),
        (end[0] - head * math.cos(angle + 0.5), end[1] - head * math.sin(angle + 0.5)),
    ]
    draw.polygon(points, fill=color)
    if label:
        draw.text((end[0] + 3, end[1] + 3), label, fill=color, stroke_width=1, stroke_fill="white")


def _draw_generic_axes(doc, image, directions, colored_only=False):
    """Draw supplied reference-frame axes, projecting them when non-degenerate."""
    image = image.copy()
    box = doc.get("reference_bbox")
    _bbox(image, box, "red", "reference")
    if box:
        center = ((box[0] + box[2]) * image.width / 2, (box[1] + box[3]) * image.height / 2)
        box_size = max((box[2] - box[0]) * image.width, (box[3] - box[1]) * image.height)
    else:
        center = (image.width / 2, image.height / 2)
        box_size = min(image.size) / 4
    length = max(18, min(min(image.size) * 0.18, box_size * 0.75))
    basis_camera = doc.get("basis_camera") or {}
    colors = {"front": "lime", "back": "magenta", "right": "deepskyblue", "left": "orange"}
    opposites = {"back": "front", "left": "right"}
    fallback = {"front": (0.0, -1.0), "back": (0.0, 1.0), "right": (-1.0, 0.0), "left": (1.0, 0.0)}
    draw = ImageDraw.Draw(image)
    for name in directions:
        axis_name = opposites.get(name, name)
        direction = _project_axis(doc.get("reference_position", [0, 1, 0]), basis_camera.get(axis_name, [0, 0, 0]))
        if direction is not None and name in opposites:
            direction = (-direction[0], -direction[1])
        direction = direction or fallback[name]
        _arrow(draw, center, direction, length, colors[name], None if colored_only else name)
    return image


def _draw_aid_axes(doc, image, directions, colored_only=False):
    if doc.get("dataset_name") == "comfort":
        if tuple(directions) == ("front",):
            return comfort_aids.draw_reference_bbox_and_labeled_front_arrow(doc, image)
        if tuple(directions) == ("front", "right"):
            return comfort_aids._draw_reference_direction_arrows_with_scale(doc, image, length_scale=0.45, minimum_length=18, directions=directions)
        if colored_only:
            return comfort_aids.draw_reference_symbolic_direction_arrows(doc, image)
        return comfort_aids.draw_short_reference_direction_arrows(doc, image)
    return _draw_generic_axes(doc, image, directions, colored_only)


def doc_to_visual(doc):
    if doc["diagnostic_experiment"] == "conversion_oracle":
        return []
    with Image.open(str(doc["img_path"])) as handle:
        image = handle.convert("RGB")
    experiment, condition = doc["diagnostic_experiment"], doc["experiment_condition"]
    if experiment == "kubric_rotation_equivariance":
        return [image.rotate(-int(doc["raster_rotation_degrees"]), expand=True)]
    if experiment == "scannet_rgb_depth_oracle" and condition == "bboxes":
        image = image.copy()
        _bbox(image, doc["reference_bbox"], "red", "reference")
        _bbox(image, doc["target_bbox"], "lime", "target")
    if experiment == "basis_oracle_ladder":
        if condition == "bboxes":
            image = image.copy()
            _bbox(image, doc.get("reference_bbox"), "red", "reference")
            _bbox(image, doc.get("target_bbox"), "lime", "target")
        elif condition == "front_axis":
            image = _draw_aid_axes(doc, image, ("front",))
        elif condition == "full_axes":
            image = _draw_aid_axes(doc, image, ("front", "back", "right", "left"))
    if experiment == "arrow_vector_grounding":
        if condition == "front":
            image = _draw_aid_axes(doc, image, ("front",))
        elif condition == "front_right":
            image = _draw_aid_axes(doc, image, ("front", "right"))
        elif condition == "full_labeled":
            image = _draw_aid_axes(doc, image, ("front", "back", "right", "left"))
        elif condition == "full_color":
            image = _draw_aid_axes(doc, image, ("front", "back", "right", "left"), colored_only=True)
    return [image]


def _vector_json(vector):
    return json.dumps({axis: round(float(vector[axis]), 6) for axis in AXES})


def _common(doc):
    return (
        f"Reference object: {doc.get('reference_object')}. Target object: {doc.get('target_object')}. "
        "Use the requested frame. Components are front, up, right; negative values mean back, below, left. "
    )


def _combined_rule(vector_first=False):
    schema = (
        '{"relative_vector":{"front":<float>,"up":<float>,"right":<float>},"answer":"<left|right|front|back>"}'
        if vector_first
        else '{"answer":"<left|right|front|back>","relative_vector":{"front":<float>,"up":<float>,"right":<float>}}'
    )
    return "Return only valid JSON: " + schema


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    exp, cond = doc["diagnostic_experiment"], doc["experiment_condition"]
    text = _common(doc) + str(doc.get("question", "")) + "\n"
    if exp == "cross_output_consistency":
        text += (
            "Return exactly one direction word."
            if cond == "direction_only"
            else (
                'Return only JSON: {"front":<float>,"up":<float>,"right":<float>}' if cond == "vector_only" else _combined_rule(cond == "combined_vector_first")
            )
        )
    elif exp == "output_format_control":
        rules = {
            "named_json": 'Return only {"front":<float>,"up":<float>,"right":<float>}.',
            "ordered_list": "Return only [front, up, right].",
            "plain_text": "Return only: front=<float>, up=<float>, right=<float>.",
            "sign_json": 'Return only component signs using -1, 0, or 1: {"front":<sign>,"up":<sign>,"right":<sign>}.',
            "prototype_mc": "Choose the closest dominant horizontal prototype: A=[1,0,0] front; B=[-1,0,0] back; C=[0,0,1] right; D=[0,0,-1] left. Return only A, B, C, or D.",
        }
        text += rules[cond]
    elif exp == "conversion_oracle":
        text = (
            f"Given the gold reference-relative vector {_vector_json(doc['gt_vector'])}, return only its dominant horizontal direction: left, right, front, or back."
            if cond == "vector_to_direction"
            else f"Given the gold direction '{doc['gt_direction']}', return only a canonical unit JSON vector using front, up, right components."
        )
    elif exp == "basis_oracle_ladder":
        basis_text = json.dumps(doc.get("basis_camera"), separators=(",", ":"))
        displacement = [round(float(value), 6) for value in doc["camera_displacement"]]
        aids = {
            "rgb": "RGB only.",
            "bboxes": "Red is reference; green is target.",
            "front_axis": "The labeled arrow is reference-front.",
            "full_axes": "The overlay gives all horizontal reference axes.",
            "basis_text": f"Gold reference basis in camera XYZ: {basis_text}.",
            "displacement_text": f"Gold target-minus-reference camera XYZ displacement: {displacement}.",
            "basis_and_displacement": f"Gold basis: {basis_text}. Gold camera XYZ displacement: {displacement}.",
            "gold_vector": f"Gold relative vector: {_vector_json(doc['gt_vector'])}.",
            "gold_direction": f"Gold direction: {doc['gt_direction']}.",
        }
        text = aids[cond] + "\n" + text + _combined_rule()
    elif exp == "component_decomposition":
        rules = {
            "front_sign": "Return only -1, 0, or 1 for the front component sign.",
            "up_sign": "Return only -1, 0, or 1 for the up component sign.",
            "right_sign": "Return only -1, 0, or 1 for the right component sign.",
            "dominant_axis": "Return only the largest-absolute-magnitude axis: front, up, or right.",
            "horizontal_direction": "Return only left, right, front, or back using the dominant horizontal component.",
        }
        text += rules[cond]
    elif exp == "arrow_vector_grounding":
        text = (
            {
                "plain": "No overlay.",
                "front": "Front is labeled.",
                "front_right": "Front and right are labeled; infer opposites.",
                "full_labeled": "All horizontal axes are labeled.",
                "full_color": "Colored arrows encode axes: green=front, magenta=back, blue=right, orange=left.",
            }[cond]
            + "\n"
            + text
            + _combined_rule()
        )
    elif exp == "scannet_rgb_depth_oracle":
        rp, tp = doc["reference_position"], doc["target_position"]
        extra = {
            "rgb": "RGB only.",
            "bboxes": "Red box is reference and green box is target.",
            "depth_text": f"Gold camera-front depths: reference={rp[1]:.4f}m, target={tp[1]:.4f}m.",
            "centroids_text": f"Gold camera XYZ centroids: reference={rp}, target={tp}.",
            "displacement_text": f"Gold target-minus-reference camera XYZ displacement={[tp[i]-rp[i] for i in range(3)]}.",
        }[cond]
        text = extra + "\n" + text + _combined_rule()
    elif exp == "angular_boundary":
        text += _combined_rule()
    else:
        rotation = doc["raster_rotation_degrees"]
        text = (
            f"The raster has been rotated clockwise {rotation} degrees. This does not rotate the physical reference-object frame; mentally undo the raster rotation.\n"
            + text
            + _combined_rule()
        )
    return pre + text + post


def doc_to_target(doc):
    exp, cond = doc["diagnostic_experiment"], doc["experiment_condition"]
    vector = {axis: round(float(doc["gt_vector"][axis]), 6) for axis in AXES}
    if exp == "cross_output_consistency":
        if cond == "direction_only":
            return doc["gt_direction"]
        if cond == "vector_only":
            return json.dumps(vector)
    if exp == "output_format_control":
        if cond == "ordered_list":
            return json.dumps([vector[axis] for axis in AXES])
        if cond == "plain_text":
            return ", ".join(f"{axis}={vector[axis]}" for axis in AXES)
        if cond == "sign_json":
            return json.dumps({axis: basis._sign(vector[axis]) for axis in AXES})
        if cond == "prototype_mc":
            return {"front": "A", "back": "B", "right": "C", "left": "D"}[doc["gt_direction"]]
        return json.dumps(vector)
    if exp == "conversion_oracle":
        return doc["gt_direction"] if cond == "vector_to_direction" else json.dumps(vector)
    if exp == "component_decomposition":
        if cond.endswith("_sign"):
            return str(basis._sign(vector[cond[:-5]]))
        if cond == "dominant_axis":
            return max(AXES, key=lambda axis: abs(vector[axis]))
        return doc["gt_direction"]
    return json.dumps({"answer": doc["gt_direction"], "relative_vector": vector})


def _parse_special(doc, prediction):
    exp, cond = doc["diagnostic_experiment"], doc["experiment_condition"]
    if exp == "output_format_control":
        if cond == "ordered_list":
            try:
                values = json.loads(prediction)
                return None, dict(zip(AXES, map(float, values))) if isinstance(values, list) and len(values) == 3 else None, None
            except (ValueError, TypeError, json.JSONDecodeError):
                return None, None, None
        if cond == "plain_text":
            number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            found = {axis: re.search(rf"{axis}\s*=\s*{number}", prediction, re.I) for axis in AXES}
            return None, ({axis: float(found[axis].group(1)) for axis in AXES} if all(found.values()) else None), None
        if cond == "prototype_mc":
            match = re.search(r"\b([ABCD])\b", prediction.upper())
            letter = match.group(1) if match else None
            vectors = {
                "A": {"front": 1, "up": 0, "right": 0},
                "B": {"front": -1, "up": 0, "right": 0},
                "C": {"front": 0, "up": 0, "right": 1},
                "D": {"front": 0, "up": 0, "right": -1},
            }
            return None, vectors.get(letter), None
    if exp == "component_decomposition":
        if cond.endswith("_sign"):
            match = re.search(r"(?<!\d)(-1|0|1)(?!\d)", prediction)
            expected = basis._sign(float(doc["gt_vector"][cond[:-5]]))
            return None, None, (float(int(match.group(1)) == expected) if match else None)
        if cond == "dominant_axis":
            pred = next((axis for axis in AXES if re.search(rf"\b{axis}\b", prediction.lower())), None)
            gold = max(AXES, key=lambda axis: abs(float(doc["gt_vector"][axis])))
            return None, None, (float(pred == gold) if pred else None)
        direction = basis.parse_direction(prediction)
        return direction, None, (float(direction == doc["gt_direction"]) if direction else None)
    return basis.parse_direction(prediction), basis.parse_vector(prediction), None


def process_results(doc, results):
    prediction = results[0].strip() if results else ""
    exp, cond = doc["diagnostic_experiment"], doc["experiment_condition"]
    mode = "combined"
    if exp == "cross_output_consistency":
        mode = "direction" if cond == "direction_only" else "vector" if cond == "vector_only" else "combined"
    elif exp == "conversion_oracle":
        mode = "direction" if cond == "vector_to_direction" else "vector"
    elif exp in {"output_format_control", "component_decomposition"}:
        mode = "special"
    parsed_direction, parsed_vector, component_accuracy = _parse_special(doc, prediction)
    if mode != "special":
        parsed_direction = basis.parse_direction(prediction) if mode in {"direction", "combined"} else None
        parsed_vector = basis.parse_vector(prediction) if mode in {"vector", "combined"} else None
    gt = {axis: float(doc["gt_vector"][axis]) for axis in AXES}
    vector_direction = basis.direction_from_vector(parsed_vector or {})
    vector_cosine = basis._cosine(parsed_vector, gt) if parsed_vector else 0.0
    hp = {"front": float((parsed_vector or {}).get("front", 0)), "up": 0.0, "right": float((parsed_vector or {}).get("right", 0))}
    hg = {"front": gt["front"], "up": 0.0, "right": gt["right"]}
    horizontal_cosine = basis._cosine(hp, hg) if parsed_vector and (abs(hp["front"]) + abs(hp["right"])) > 0 else 0.0
    sign_scores = {axis: float(parsed_vector is not None and basis._sign(parsed_vector[axis]) == basis._sign(gt[axis])) for axis in AXES}
    direction_accuracy = float(parsed_direction == doc["gt_direction"])
    internal = float(parsed_direction is not None and vector_direction is not None and parsed_direction == vector_direction)
    if mode == "direction":
        parse_success = float(parsed_direction is not None)
    elif mode == "vector":
        parse_success = float(parsed_vector is not None)
    elif mode == "combined":
        parse_success = float(parsed_direction is not None and parsed_vector is not None)
    elif exp == "component_decomposition":
        parse_success = float(component_accuracy is not None)
    else:
        parse_success = float(parsed_vector is not None)
    entry = {
        "qid": doc["qid"],
        "base_pair_id": doc["base_pair_id"],
        "scene_id": doc.get("scene_id"),
        "dataset_name": doc["dataset_name"],
        "diagnostic_experiment": exp,
        "experiment_condition": cond,
        "response_mode": mode,
        "gt_direction": doc["gt_direction"],
        "gt_vector": gt,
        "prediction": prediction,
        "parsed_direction": parsed_direction,
        "parsed_vector": parsed_vector,
        "predicted_vector_direction": vector_direction,
        "direction_accuracy": direction_accuracy,
        "vector_direction_accuracy": float(vector_direction == doc["gt_direction"]),
        "vector_cosine": vector_cosine,
        "horizontal_cosine": horizontal_cosine,
        "front_sign_accuracy": sign_scores["front"],
        "up_sign_accuracy": sign_scores["up"],
        "right_sign_accuracy": sign_scores["right"],
        "internal_consistency": internal,
        "component_accuracy": float(component_accuracy or 0.0),
        "parse_success": parse_success,
        "angular_margin_bin": doc.get("angular_margin_bin"),
    }
    output = {
        name: dict(entry)
        for name in (
            "diagnostic_direction_accuracy",
            "diagnostic_vector_direction_accuracy",
            "diagnostic_vector_cosine",
            "diagnostic_horizontal_cosine",
            "diagnostic_front_sign_accuracy",
            "diagnostic_up_sign_accuracy",
            "diagnostic_right_sign_accuracy",
            "diagnostic_internal_consistency",
            "diagnostic_component_accuracy",
            "diagnostic_parse_success",
        )
    }
    output["submission"] = {**entry, "question_prompt": doc_to_text(doc), "img_path": doc.get("img_path")}
    return output


def _mean(results, field):
    return sum(float(row.get(field, 0)) for row in results) / len(results) if results else 0.0


def aggregate_direction_accuracy(r):
    return _mean(r, "direction_accuracy")


def aggregate_vector_direction_accuracy(r):
    return _mean(r, "vector_direction_accuracy")


def aggregate_vector_cosine(r):
    return _mean(r, "vector_cosine")


def aggregate_horizontal_cosine(r):
    return _mean(r, "horizontal_cosine")


def aggregate_front_sign_accuracy(r):
    return _mean(r, "front_sign_accuracy")


def aggregate_up_sign_accuracy(r):
    return _mean(r, "up_sign_accuracy")


def aggregate_right_sign_accuracy(r):
    return _mean(r, "right_sign_accuracy")


def aggregate_internal_consistency(r):
    return _mean(r, "internal_consistency")


def aggregate_component_accuracy(r):
    return _mean(r, "component_accuracy")


def aggregate_parse_success(r):
    return _mean(r, "parse_success")


def aggregate_results_for_submission(results, args):
    experiment = str(results[0]["diagnostic_experiment"]) if results else "direction_vector_diagnostic"
    dataset_name = str(results[0].get("dataset_name", "unknown")) if results else "unknown"
    model = sanitize_model_name(getattr(args, "model", "") or "unknown_model")
    file_experiment = experiment if experiment.startswith(f"{dataset_name}_") else f"{dataset_name}_{experiment}"
    path = generate_submission_file(f"{file_experiment}_{model}.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"dataset_name": dataset_name, "experiment": experiment, "num_records": len(results), "records": results}, handle, indent=2)
    eval_logger.info("Direction/vector diagnostic saved to {}", path)
