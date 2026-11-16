"""Composable diagnostics for the exhaustive MOVi-A motion dataset.

The task reads only ``qa.parquet`` through lmms-eval. Rich assistance is loaded
lazily from ``ground_truth/<sequence_id>.json`` so task development and task
registration do not depend on the dataset-generation job being complete.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import threading
from typing import Callable, Iterable, Optional

from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image, ImageDraw


DATA_HOME = Path(os.getenv("RAMANATHAN_DATA_ROOT", "/home/ramanathan/data"))
DATA_ROOT = Path(os.getenv("MOVI_A_MOTION_ROOT", str(DATA_HOME / "movi_a_motion")))
FRAME_ROOT = Path(os.getenv("MOVI_A_TRAIN_EXPORT_ROOT", str(DATA_HOME / "movi_a_train_export")))
OPTION_LETTERS = "ABCDE"
DEFAULT_SAMPLES_PER_SUBTASK = 100
COLORS = ((255, 65, 65), (50, 190, 255), (70, 220, 110), (240, 170, 40), (190, 80, 240))
NUMERIC_ANSWER_METRICS = {
    "displacement",
    "path_length",
    "average_speed",
    "instantaneous_speed",
    "instantaneous_acceleration",
}
_TRACE_LOCK = threading.Lock()


def _trace_root() -> Optional[Path]:
    value = os.getenv("MOVI_MOTION_TRACE_DIR", "").strip()
    return Path(value) if value else None


def _numeric_scale_guide_enabled() -> bool:
    return os.getenv("MOVI_MOTION_NUMERIC_SCALE_GUIDE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _trace_record_path(doc: dict) -> Optional[Path]:
    root = _trace_root()
    if root is None:
        return None
    suite = str(doc.get("scaffold_suite", "unknown"))
    qid = str(doc.get("qid", doc.get("index", "unknown")))
    digest = hashlib.sha1(f"{suite}::{qid}".encode()).hexdigest()
    condition = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc.get("scaffold_condition", "unknown")))
    return root / "records" / f"{digest}_{condition}.json"


def _trace_update(doc: dict, **fields) -> None:
    """Atomically merge task-side trace fields when trace mode is enabled."""

    path = _trace_record_path(doc)
    if path is None:
        return
    with _TRACE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload.update(fields)
        payload.setdefault("trace_version", 1)
        payload.setdefault("qid", str(doc.get("qid", "")))
        payload.setdefault("source_qid", str(doc.get("source_qid", doc.get("qid", ""))))
        payload.setdefault("sequence_id", str(doc.get("sequence_id", "")))
        payload.setdefault("subtask", str(doc.get("subtask", "")))
        payload.setdefault("scaffold_suite", str(doc.get("scaffold_suite", "")))
        payload.setdefault("scaffold_condition", str(doc.get("scaffold_condition", "baseline")))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, path)


def _trace_write_bytes(path: Path, content: bytes) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    try:
        os.replace(temporary, path)
    except FileNotFoundError:
        # Another worker may have installed the same content-addressed file.
        pass


def _trace_visuals(doc: dict, images: list[Image.Image], source_paths: list[str], source_indices: list[int]) -> None:
    root = _trace_root()
    if root is None:
        return
    visual_entries = []
    visual_hashes = []
    original_frame_indices = frame_indices(doc, len(source_paths))
    for output_index, image in enumerate(images):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("media") / digest[:2] / f"{digest}.png"
        _trace_write_bytes(root / relative, content)
        source_position = source_indices[output_index] if output_index < len(source_indices) else None
        source_frame_index = original_frame_indices[source_position] if source_position is not None else None
        visual_entries.append(
            {
                "order": output_index,
                "sha256": digest,
                "path": str(relative),
                "width": image.width,
                "height": image.height,
                "source_frame_index": source_frame_index,
                "source_path": source_paths[source_position] if source_position is not None else None,
                "is_generated_panel": source_position is None,
            }
        )
        visual_hashes.append(digest)

    preview_path = None
    if images:
        sequence_digest = hashlib.sha256("\n".join(visual_hashes).encode()).hexdigest()
        relative_preview = Path("previews") / f"{sequence_digest}.jpg"
        absolute_preview = root / relative_preview
        if not absolute_preview.is_file():
            columns = min(6, len(images))
            rows = math.ceil(len(images) / columns)
            tile_width, tile_height, label_height = 180, 180, 20
            sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
            draw = ImageDraw.Draw(sheet)
            for index, image in enumerate(images):
                thumbnail = image.copy()
                thumbnail.thumbnail((tile_width, tile_height))
                x = (index % columns) * tile_width + (tile_width - thumbnail.width) // 2
                y = (index // columns) * (tile_height + label_height) + (tile_height - thumbnail.height) // 2
                sheet.paste(thumbnail, (x, y))
                draw.text(((index % columns) * tile_width + 4, (index // columns) * (tile_height + label_height) + tile_height + 2), f"input {index}", fill="black")
            buffer = io.BytesIO()
            sheet.save(buffer, format="JPEG", quality=88)
            _trace_write_bytes(absolute_preview, buffer.getvalue())
        preview_path = str(relative_preview)

    condition = str(doc.get("scaffold_condition", "baseline"))
    spec = SCAFFOLDS[condition]
    visual_aids = [getattr(aid, "__name__", str(aid)) for aid in spec.visual_aids]
    if (
        _has_numeric_answer(doc)
        and _numeric_scale_guide_enabled()
        and str(doc.get("scaffold_suite", "")) != "visual_scale"
        and condition not in {"corner_metric_scale_guide", "object_anchored_metric_scale_guide"}
    ):
        visual_aids.append("visual_metric_scale_guide")
    _trace_update(
        doc,
        visual_input={
            "count": len(images),
            "ordered_media": visual_entries,
            "contact_sheet": preview_path,
            "frame_policy": spec.frame_policy,
            "visual_aids": visual_aids,
            "resolved_source_frame_paths": source_paths,
        },
    )


class ScaffoldSpec:
    """Task-local scaffold configuration.

    Keep this as a regular class rather than a dataclass: lmms-eval loads task
    utility modules directly from their paths without first registering them in
    ``sys.modules``, which Python 3.10 dataclasses require when resolving the
    postponed annotations used by this module.
    """

    __slots__ = ("text_aids", "visual_aids", "frame_policy")

    def __init__(
        self,
        text_aids: tuple[Callable[[dict], str], ...] = (),
        visual_aids: tuple[Callable[[dict, list[Image.Image]], list[Image.Image]], ...] = (),
        frame_policy: Optional[str] = None,
    ):
        self.text_aids = text_aids
        self.visual_aids = visual_aids
        self.frame_policy = frame_policy


def _strict_json(value, default):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def ground_truth(doc: dict) -> dict:
    return _strict_json(doc.get("ground_truth_json"), {})


def frame_paths(doc: dict) -> list[str]:
    paths = _strict_json(doc.get("frames_json"), [])
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{doc.get('qid')}: frames_json must contain at least one path")
    sequence_id = str(doc.get("sequence_id", ""))
    resolved = []
    for value in paths:
        embedded = Path(str(value))
        rebased = FRAME_ROOT / sequence_id / "frames" / embedded.name
        # Prefer the configured moved-data root when present. Retain fixture
        # and portable-dataset paths when no corresponding local frame exists.
        resolved.append(str(rebased if rebased.is_file() else embedded))
    return resolved


def frame_indices(doc: dict, count: Optional[int] = None) -> list[int]:
    """Return the original MOVi frame index represented by each visual."""

    values = _strict_json(doc.get("selected_frame_indices_json"), [])
    if isinstance(values, list) and values and (count is None or len(values) == count):
        return [int(value) for value in values]
    size = len(frame_paths(doc)) if count is None else count
    return list(range(size))


@lru_cache(maxsize=32)
def load_sequence_gt(sequence_id: str) -> dict:
    path = DATA_ROOT / "ground_truth" / f"{sequence_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"MOVi-A motion GT sidecar is missing: {path}. Set MOVI_A_MOTION_ROOT "
            "to the generated dataset root or finish the generator first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def sequence_gt(doc: dict) -> dict:
    fixture = doc.get("_sequence_gt")
    return fixture if isinstance(fixture, dict) else load_sequence_gt(str(doc["sequence_id"]))


def _objects(doc: dict) -> list[dict]:
    return list(sequence_gt(doc).get("objects") or [])


def _object(doc: dict, object_id: Optional[int] = None) -> dict:
    wanted = int(doc["target_object_id"] if object_id is None else object_id)
    for item in _objects(doc):
        if int(item["object_id"]) == wanted:
            return item
    raise KeyError(f"{doc.get('qid')}: object {wanted} is absent from its GT sidecar")


def _frames(doc: dict, object_id: Optional[int] = None) -> list[dict]:
    return list(_object(doc, object_id).get("frame_quantities") or [])


def _relevant_frames(doc: dict, object_id: Optional[int] = None) -> list[dict]:
    start, end = int(doc["frame_start"]), int(doc["frame_end"])
    selected = _strict_json(doc.get("selected_frame_indices_json"), [])
    selected_set = {int(value) for value in selected} if isinstance(selected, list) and selected else None
    return [
        row for row in _frames(doc, object_id)
        if start <= int(row["frame_index"]) <= end
        and (selected_set is None or int(row["frame_index"]) in selected_set)
    ]


def _candidate_ids(doc: dict) -> list[int]:
    values = ground_truth(doc).get("candidate_values") or []
    ids = [int(item["object_id"]) for item in values if "object_id" in item]
    return ids or [int(doc["target_object_id"])]


def _stable_rng(doc: dict, salt: str) -> random.Random:
    seed = hashlib.sha1(f"{doc.get('qid')}::{salt}".encode()).hexdigest()
    return random.Random(seed)


def _fmt_vector(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _precision(doc: dict) -> int:
    condition = str(doc.get("scaffold_condition", ""))
    return {"precision_2dp": 2, "precision_4dp": 4, "precision_full": 8}.get(condition, 4)


def _metric(doc: dict) -> str:
    return str(doc.get("subtask", ground_truth(doc).get("metric", "")))


def _is_comparison(doc: dict) -> bool:
    return _metric(doc) in {"fastest_object", "greatest_path_length_object"}


def _has_numeric_answer(doc: dict) -> bool:
    """True only when the displayed answer choices are numerical quantities."""

    return _metric(doc) in NUMERIC_ANSWER_METRICS and doc.get("answer_value") is not None


def _select_frame_indices(doc: dict, limit: int) -> list[int]:
    """Select a deterministic, label-aware set spanning the original clip."""

    total = len(frame_paths(doc))
    if limit <= 0 or limit >= total:
        return list(range(total))
    gt = ground_truth(doc)
    required = {
        max(0, min(total - 1, int(doc.get("frame_start", 0)))),
        max(0, min(total - 1, int(doc.get("frame_end", total - 1)))),
    }
    if "query_frame" in gt:
        query = max(0, min(total - 1, int(gt["query_frame"])))
        required.update({query, max(0, query - 1), min(total - 1, query + 1)})
    # Preserve the full temporal extent whenever capacity permits.
    required.update({0, total - 1})
    selected = set(sorted(required)[:limit])
    while len(selected) < limit:
        remaining = [index for index in range(total) if index not in selected]
        # Farthest-point sampling spreads the remaining evidence over time.
        chosen = max(remaining, key=lambda index: (min(abs(index - prior) for prior in selected), -index))
        selected.add(chosen)
    return sorted(selected)


def _numeric_options(value: float, unit: str, rng: random.Random) -> tuple[dict[str, str], str]:
    scale = max(abs(value), 0.1)
    candidates = [value, max(0.0, value - 0.50 * scale), max(0.0, value - 0.25 * scale), value + 0.25 * scale, value + 0.50 * scale]
    unique = []
    for candidate in candidates:
        if all(abs(candidate - prior) > max(1e-6, scale * 0.02) for prior in unique):
            unique.append(candidate)
    offset = 0.75
    while len(unique) < 5:
        unique.append(value + offset * scale)
        offset += 0.25
    correct = unique[0]
    rng.shuffle(unique)
    answer_index = min(range(5), key=lambda index: abs(unique[index] - correct))
    return (
        {letter: f"{unique[index]:.3f} {unit}".strip() for index, letter in enumerate(OPTION_LETTERS)},
        OPTION_LETTERS[answer_index],
    )


def _derived_interval(doc: dict, object_id: int, selected: set[int]) -> dict:
    start, end = int(doc["frame_start"]), int(doc["frame_end"])
    rows = [
        row for row in _frames(doc, object_id)
        if start <= int(row["frame_index"]) <= end and int(row["frame_index"]) in selected
    ]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows or int(rows[0]["frame_index"]) != start or int(rows[-1]["frame_index"]) != end:
        raise ValueError(f"{doc.get('qid')}: derived frames do not preserve interval endpoints {start}-{end}")
    first, last = rows[0], rows[-1]
    segments = [
        math.sqrt(sum((float(right["position_world"][axis]) - float(left["position_world"][axis])) ** 2 for axis in range(3)))
        for left, right in zip(rows, rows[1:])
    ]
    displacement_vector = [float(last["position_world"][axis]) - float(first["position_world"][axis]) for axis in range(3)]
    displacement = math.sqrt(sum(value * value for value in displacement_vector))
    path_length = sum(segments)
    duration = float(last["timestamp_seconds"]) - float(first["timestamp_seconds"])
    camera_delta = [float(last["position_camera"][axis]) - float(first["position_camera"][axis]) for axis in range(3)]
    distance_delta = float(last["distance_to_camera"]) - float(first["distance_to_camera"])
    return {
        "frame_start": start,
        "frame_end": end,
        "sampled_frame_indices": [int(row["frame_index"]) for row in rows],
        "time_start_seconds": float(first["timestamp_seconds"]),
        "time_end_seconds": float(last["timestamp_seconds"]),
        "duration_seconds": duration,
        "start_position_world": list(map(float, first["position_world"])),
        "end_position_world": list(map(float, last["position_world"])),
        "displacement_vector_world": displacement_vector,
        "displacement": displacement,
        "path_length": path_length,
        "straightness": displacement / path_length if path_length > 1e-12 else 0.0,
        "average_speed_from_path": path_length / duration if duration > 0 else 0.0,
        "mean_annotated_speed": sum(float(row["speed"]) for row in rows) / len(rows),
        "start_velocity_world": list(map(float, first["velocity_world"])),
        "end_velocity_world": list(map(float, last["velocity_world"])),
        "velocity_change_world": [float(last["velocity_world"][axis]) - float(first["velocity_world"][axis]) for axis in range(3)],
        "camera_coordinate_delta": camera_delta,
        "start_distance_to_camera": float(first["distance_to_camera"]),
        "end_distance_to_camera": float(last["distance_to_camera"]),
        "distance_to_camera_delta": distance_delta,
        "visible_fraction": sum(int(row.get("visibility_pixels", 0)) >= 200 for row in rows) / len(rows),
    }


def _derived_motion_direction(interval: dict) -> tuple[str, float]:
    dx, dy, _ = interval["camera_coordinate_delta"]
    dz = float(interval["distance_to_camera_delta"])
    scores = {"left": dx, "right": -dx, "up": dy, "down": -dy, "away from the camera": dz, "toward the camera": -dz}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[0][0], float(ranked[0][1] - ranked[1][1])


def _derive_frame_limited_doc(source: dict, limit: int) -> dict:
    """Rebuild one QA record so its target exactly matches its visible frames."""

    doc = dict(source)
    original_paths = frame_paths(doc)
    indices = _select_frame_indices(doc, limit)
    selected = set(indices)
    doc["frames_json"] = json.dumps([original_paths[index] for index in indices], separators=(",", ":"))
    doc["selected_frame_indices_json"] = json.dumps(indices, separators=(",", ":"))
    doc["derived_frame_count"] = len(indices)
    doc["derived_from_frame_count"] = len(original_paths)
    metric = _metric(doc)
    old_gt = ground_truth(doc)
    unit = str(doc.get("answer_unit", old_gt.get("unit", "")))

    if metric in {"fastest_object", "greatest_path_length_object"}:
        candidates = []
        for item in old_gt.get("candidate_values") or []:
            object_id = int(item["object_id"])
            if metric == "fastest_object":
                query = int(doc["frame_start"])
                row = next(row for row in _frames(doc, object_id) if int(row["frame_index"]) == query)
                value = float(row["speed"])
            else:
                value = float(_derived_interval(doc, object_id, selected)["path_length"])
            candidates.append({**item, "value": value})
        winner = max(candidates, key=lambda item: float(item["value"]))
        options = {letter: str(doc[letter]) for letter in OPTION_LETTERS}
        answer = next(letter for letter, text in options.items() if text == str(winner["description"]))
        doc.update({
            "answer": answer,
            "answer_detail": str(winner["description"]),
            "answer_value": float(winner["value"]),
            "target_object_id": int(winner["object_id"]),
            "target_description": str(winner["description"]),
            "ground_truth_json": json.dumps({
                "metric": "instantaneous_speed" if metric == "fastest_object" else "path_length",
                "unit": unit,
                "winner_object_id": int(winner["object_id"]),
                "winner_description": str(winner["description"]),
                "winner_value": float(winner["value"]),
                "candidate_values": candidates,
                "sampled_frame_indices": indices,
            }, separators=(",", ":")),
        })
        return doc

    interval = _derived_interval(doc, int(doc["target_object_id"]), selected)
    if metric == "displacement":
        value = float(interval["displacement"])
    elif metric == "path_length":
        value = float(interval["path_length"])
    elif metric == "average_speed":
        value = float(interval["average_speed_from_path"])
    elif metric in {"instantaneous_speed", "instantaneous_acceleration"}:
        query = int(old_gt.get("query_frame", (int(doc["frame_start"]) + int(doc["frame_end"])) // 2))
        row = next(row for row in _frames(doc, int(doc["target_object_id"])) if int(row["frame_index"]) == query)
        value = float(row["speed"] if metric == "instantaneous_speed" else row["acceleration_magnitude"])
        interval["query_frame"] = query
    elif metric == "dominant_motion_direction":
        direction, margin = _derived_motion_direction(interval)
        answer = next(letter for letter in OPTION_LETTERS if str(doc[letter]) == direction)
        interval.update({"direction": direction, "direction_margin": margin})
        doc.update({
            "answer": answer,
            "answer_detail": direction,
            "answer_value": None,
            "ground_truth_json": json.dumps(interval, separators=(",", ":")),
        })
        return doc
    else:
        raise ValueError(f"Unsupported derived motion metric: {metric}")

    options, answer = _numeric_options(value, unit, _stable_rng(doc, f"derived_{limit}_options"))
    interval.update({"metric": metric, "value": value, "unit": unit})
    doc.update(options)
    doc.update({
        "answer": answer,
        "answer_detail": f"{value:.9g}",
        "answer_value": value,
        "ground_truth_json": json.dumps(interval, separators=(",", ":")),
    })
    return doc


def _formula(doc: dict) -> str:
    metric = _metric(doc)
    formulas = {
        "displacement": "displacement = sqrt((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2)",
        "path_length": "path length = sum_i ||position[i+1] - position[i]||_2",
        "average_speed": "average speed = path length / elapsed time",
        "instantaneous_speed": "speed = sqrt(vx^2 + vy^2 + vz^2)",
        "instantaneous_acceleration": "a[t] = (v[t+1]-v[t-1])/(2/fps); magnitude = ||a[t]||_2",
        "dominant_motion_direction": "choose the largest signed camera-frame change among left/right/up/down/toward/away",
        "fastest_object": "compute ||velocity||_2 for every candidate and choose the maximum",
        "greatest_path_length_object": "sum consecutive 3D segment lengths for every candidate and choose the maximum",
    }
    return formulas.get(metric, "Use the supplied physical state to compute the requested quantity.")


def aid_formula(doc: dict) -> str:
    return "Computation scaffold: " + _formula(doc)


def _state_for_object(doc: dict, object_id: int, precision: int) -> str:
    rows = _relevant_frames(doc, object_id)
    metric = _metric(doc)
    description = _object(doc, object_id).get("description", f"object {object_id}")
    if not rows:
        return f"{description}: no relevant state rows"
    if metric in {"instantaneous_speed", "fastest_object"}:
        query = int(ground_truth(doc).get("query_frame", doc.get("frame_start", 0) + 1))
        row = min(rows, key=lambda item: abs(int(item["frame_index"]) - query))
        return f"{description}, frame {row['frame_index']}: velocity={_fmt_vector(row['velocity_world'], precision)}"
    if metric == "instantaneous_acceleration":
        return "; ".join(
            f"{description}, frame {row['frame_index']}: velocity={_fmt_vector(row['velocity_world'], precision)}"
            for row in rows
        )
    if metric in {"path_length", "average_speed", "greatest_path_length_object"}:
        return (
            f"{description} positions: "
            + "; ".join(f"f{row['frame_index']}={_fmt_vector(row['position_world'], precision)}" for row in rows)
        )
    first, last = rows[0], rows[-1]
    coordinate_key = "position_camera" if metric == "dominant_motion_direction" else "position_world"
    return (
        f"{description}: frame {first['frame_index']} {coordinate_key}={_fmt_vector(first[coordinate_key], precision)}; "
        f"frame {last['frame_index']} {coordinate_key}={_fmt_vector(last[coordinate_key], precision)}"
    )


def aid_raw_state(doc: dict) -> str:
    precision = _precision(doc)
    ids = _candidate_ids(doc) if _metric(doc) in {"fastest_object", "greatest_path_length_object"} else [int(doc["target_object_id"])]
    lines = [_state_for_object(doc, object_id, precision) for object_id in ids]
    return "Authoritative GT physical-state scaffold (scene units):\n" + "\n".join(lines)


def _representation_state(doc: dict, precision: int = 6) -> dict[str, list[dict]]:
    """Return only the GT quantities needed to solve this document.

    The complete, full-precision state remains in the per-sequence GT sidecar.
    JSON and table scaffolds share this payload so they differ only in their
    representation, without injecting irrelevant fields or excessive tokens.
    """

    metric = _metric(doc)
    ids = _candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]

    def vector(values) -> list[float]:
        return [round(float(value), precision) for value in values]

    payload = {}
    for object_id in ids:
        rows = _relevant_frames(doc, object_id)
        if not rows:
            payload[str(object_id)] = []
            continue

        if metric in {"instantaneous_speed", "fastest_object"}:
            query = int(ground_truth(doc).get("query_frame", doc.get("frame_start", 0) + 1))
            selected = [min(rows, key=lambda row: abs(int(row["frame_index"]) - query))]
            payload[str(object_id)] = [
                {"frame": int(row["frame_index"]), "velocity_world": vector(row["velocity_world"])}
                for row in selected
            ]
        elif metric == "instantaneous_acceleration":
            payload[str(object_id)] = [
                {
                    "frame": int(row["frame_index"]),
                    "time": round(float(row["timestamp_seconds"]), 9),
                    "velocity_world": vector(row["velocity_world"]),
                }
                for row in rows
            ]
        elif metric in {"path_length", "average_speed", "greatest_path_length_object"}:
            payload[str(object_id)] = [
                {
                    "frame": int(row["frame_index"]),
                    **({"time": round(float(row["timestamp_seconds"]), 9)} if metric == "average_speed" else {}),
                    "position_world": vector(row["position_world"]),
                }
                for row in rows
            ]
        elif metric == "dominant_motion_direction":
            payload[str(object_id)] = [
                {"frame": int(row["frame_index"]), "position_camera": vector(row["position_camera"])}
                for row in (rows[0], rows[-1])
            ]
        else:
            payload[str(object_id)] = [
                {"frame": int(row["frame_index"]), "position_world": vector(row["position_world"])}
                for row in (rows[0], rows[-1])
            ]
    return payload


def aid_state_json(doc: dict) -> str:
    return "Authoritative task-relevant GT JSON (scene units):\n" + json.dumps(
        _representation_state(doc), separators=(",", ":"), sort_keys=True
    )


def aid_state_table(doc: dict) -> str:
    payload = _representation_state(doc)
    fields = []
    for records in payload.values():
        for record in records:
            for field in record:
                if field != "frame" and field not in fields:
                    fields.append(field)
    lines = ["object | frame" + "".join(f" | {field}" for field in fields)]
    for object_id, records in payload.items():
        for record in records:
            values = [object_id, str(record["frame"])]
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, list):
                    value = "[" + ",".join(f"{float(item):g}" for item in value) + "]"
                values.append(str(value))
            lines.append(" | ".join(values))
    return "Authoritative task-relevant GT table (scene units):\n" + "\n".join(lines)


def aid_2d_motion(doc: dict) -> str:
    traces = []
    for object_id in (_candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]):
        values = "; ".join(
            f"f{row['frame_index']}={_fmt_vector(row['image_position_normalized'], 3)}"
            for row in _relevant_frames(doc, object_id)
        )
        traces.append(f"object {object_id}: {values}")
    return "GT image-space traces: " + " | ".join(traces)


def aid_visibility(doc: dict) -> str:
    records = []
    for object_id in (_candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]):
        records.append(
            f"object {object_id}: "
            + ", ".join(f"f{r['frame_index']}={r['visibility_pixels']}" for r in _relevant_frames(doc, object_id))
        )
    return "GT visibility pixels: " + " | ".join(records)


def aid_fps(doc: dict) -> str:
    fps = float(doc.get("fps", 12.0))
    start, end = int(doc["frame_start"]), int(doc["frame_end"])
    return f"Timing scaffold: FPS={fps:g}, one-frame dt={1/fps:.6f}s, query duration={(end-start)/fps:.6f}s."


def aid_collision_timeline(doc: dict) -> str:
    start, end = int(doc["frame_start"]), int(doc["frame_end"])
    episodes = [
        item for item in (sequence_gt(doc).get("collisions") or {}).get("episodes", [])
        if int(item["frame_end"]) >= start and int(item["frame_start"]) <= end
    ]
    if not episodes:
        return "GT collision timeline: no consolidated collision episode overlaps the query interval."
    return "GT collision timeline: " + "; ".join(
        f"objects={item['instances']} frames={item['frame_start']}-{item['frame_end']} max_force={item['max_force']:.4f}"
        for item in episodes
    )


def aid_world_positions(doc: dict) -> str:
    return "World-coordinate scaffold. " + aid_raw_state(doc)


def aid_camera_positions(doc: dict) -> str:
    records = []
    for object_id in (_candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]):
        records.append(
            f"object {object_id}: "
            + "; ".join(f"f{r['frame_index']}={_fmt_vector(r['position_camera'])}" for r in _relevant_frames(doc, object_id))
        )
    return "GT camera-frame positions (+X left, +Y up): " + " | ".join(records)


def aid_camera_basis(doc: dict) -> str:
    camera = sequence_gt(doc).get("camera") or {}
    start = int(doc["frame_start"])
    quaternion = (camera.get("quaternions") or [[1, 0, 0, 0]])[start]
    return (
        "Camera-coordinate convention: +X is image-left, +Y is up; increasing camera distance means away. "
        f"Camera quaternion (wxyz) at frame {start}={_fmt_vector(quaternion, 6)}."
    )


def aid_distance_trace(doc: dict) -> str:
    records = []
    for object_id in (_candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]):
        records.append(
            f"object {object_id}: "
            + ", ".join(f"f{r['frame_index']}={r['distance_to_camera']:.4f}" for r in _relevant_frames(doc, object_id))
        )
    return "GT distance-to-camera traces: " + " | ".join(records)


def aid_signed_axis_deltas(doc: dict) -> str:
    gt = ground_truth(doc)
    return (
        "GT signed motion components: camera_delta="
        f"{_fmt_vector(gt.get('camera_coordinate_delta', [0, 0, 0]))}, "
        f"distance_to_camera_delta={float(gt.get('distance_to_camera_delta', 0)):.4f}."
    )


def aid_direction_definition(doc: dict) -> str:
    return (
        "Direction rule: positive camera-X=left, negative camera-X=right, positive camera-Y=up, "
        "negative camera-Y=down, decreasing camera distance=toward, increasing distance=away."
    )


def _positions(doc: dict) -> list[list[float]]:
    return [list(map(float, row["position_world"])) for row in _relevant_frames(doc)]


def aid_component_differences(doc: dict) -> str:
    positions = _positions(doc)
    delta = [positions[-1][i] - positions[0][i] for i in range(3)] if positions else [0, 0, 0]
    return "Intermediate GT endpoint difference (dx,dy,dz)=" + _fmt_vector(delta, 6)


def aid_squared_components(doc: dict) -> str:
    positions = _positions(doc)
    delta = [positions[-1][i] - positions[0][i] for i in range(3)] if positions else [0, 0, 0]
    return "Intermediate squared components=" + _fmt_vector([value * value for value in delta], 6)


def _segment_vectors(doc: dict, object_id: Optional[int] = None) -> list[list[float]]:
    positions = [list(map(float, row["position_world"])) for row in _relevant_frames(doc, object_id)]
    return [[b[i] - a[i] for i in range(3)] for a, b in zip(positions, positions[1:])]


def aid_segment_vectors(doc: dict) -> str:
    ids = _candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]
    return "GT consecutive displacement vectors: " + " | ".join(
        f"object {object_id}: " + "; ".join(_fmt_vector(v, 4) for v in _segment_vectors(doc, object_id))
        for object_id in ids
    )


def aid_segment_lengths(doc: dict) -> str:
    ids = _candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]
    records = []
    for object_id in ids:
        lengths = [math.sqrt(sum(value * value for value in vector)) for vector in _segment_vectors(doc, object_id)]
        records.append(f"object {object_id}: " + _fmt_vector(lengths, 6))
    return "GT consecutive segment lengths: " + " | ".join(records)


def aid_velocity_difference(doc: dict) -> str:
    rows = _relevant_frames(doc)
    if len(rows) < 2:
        return "Velocity-difference scaffold unavailable for fewer than two frames."
    first, last = rows[0]["velocity_world"], rows[-1]["velocity_world"]
    delta = [float(last[i]) - float(first[i]) for i in range(3)]
    dt = float(rows[-1]["timestamp_seconds"]) - float(rows[0]["timestamp_seconds"])
    return f"GT velocity difference={_fmt_vector(delta, 6)}, dt={dt:.6f}s."


def aid_norm_ready(doc: dict) -> str:
    metric = _metric(doc)
    if metric == "fastest_object":
        return "Final GT velocity vectors before taking magnitudes: " + " | ".join(
            f"object {object_id}: {_fmt_vector(_relevant_frames(doc, object_id)[0]['velocity_world'], 8)}"
            for object_id in _candidate_ids(doc)
        )
    if metric == "instantaneous_speed":
        vector = _relevant_frames(doc)[len(_relevant_frames(doc)) // 2]["velocity_world"]
    elif metric == "instantaneous_acceleration":
        vector = _relevant_frames(doc)[len(_relevant_frames(doc)) // 2]["acceleration_world"]
    else:
        positions = _positions(doc)
        vector = [positions[-1][i] - positions[0][i] for i in range(3)] if positions else [0, 0, 0]
    return "Final GT vector before taking its Euclidean magnitude=" + _fmt_vector(vector, 8)


def aid_derived(doc: dict) -> str:
    gt = ground_truth(doc)
    if _metric(doc) in {"fastest_object", "greatest_path_length_object"}:
        values = gt.get("candidate_values") or []
        return "Authoritative derived candidate quantities: " + "; ".join(
            f"{item['description']}={float(item['value']):.9g} {gt.get('unit', doc.get('answer_unit', ''))}" for item in values
        )
    if _metric(doc) == "dominant_motion_direction":
        return f"Authoritative derived direction: {gt.get('direction', doc.get('answer_detail'))}."
    return f"Authoritative derived {_metric(doc)}={float(doc['answer_value']):.12g} {doc.get('answer_unit', '')}."


def aid_rounded_derived(doc: dict) -> str:
    if doc.get("answer_value") is None:
        return aid_derived(doc)
    return f"Rounded authoritative value={float(doc['answer_value']):.3f} {doc.get('answer_unit', '')}."


def aid_answer_text(doc: dict) -> str:
    return f"Answer-text oracle: the correct answer text is '{doc[str(doc['answer'])]}'. Match it to an option letter."


def aid_answer_letter(doc: dict) -> str:
    return f"Answer-letter oracle: respond with {doc['answer']}."


def aid_worked_example(doc: dict) -> str:
    metric = _metric(doc)
    examples = {
        "displacement": "Unrelated example: from [0,0,0] to [3,4,0], displacement=sqrt(9+16)=5.",
        "path_length": "Unrelated example: segment lengths 1, 2, and 3 give path length 6, regardless of endpoint displacement.",
        "average_speed": "Unrelated example: path 6 over 2 seconds gives average speed 3.",
        "instantaneous_speed": "Unrelated example: velocity [3,4,0] has speed 5.",
        "instantaneous_acceleration": "Unrelated example: velocity change [2,0,0] over 0.5s gives acceleration [4,0,0].",
    }
    return examples.get(metric, "Unrelated example: compare all candidate values and select the largest when asked for a maximum.")


def aid_distractor_state(doc: dict) -> str:
    target = int(doc["target_object_id"])
    others = [item for item in _objects(doc) if int(item["object_id"]) != target]
    if not others:
        return "Distractor-state control: no non-target object is available."
    distractor = others[0]
    return "Explicitly irrelevant distractor state; do not use it. " + _state_for_object(doc, int(distractor["object_id"]), 4)


def aid_mismatched_state(doc: dict) -> str:
    target = int(doc["target_object_id"])
    others = [item for item in _objects(doc) if int(item["object_id"]) != target]
    if not others:
        return "Mismatched-state control unavailable."
    wrong = others[0]
    return (
        "INTENTIONALLY MISMATCHED CONTROL: the following belongs to a different object and must not be used "
        "as target evidence. " + _state_for_object(doc, int(wrong["object_id"]), 4)
    )


def aid_noisy_state(doc: dict) -> str:
    rng = _stable_rng(doc, "coordinate_noise_low")
    lines = []
    for object_id in (_candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]):
        values = []
        for row in _relevant_frames(doc, object_id):
            noisy = [float(value) + rng.gauss(0, 0.02) for value in row["position_world"]]
            values.append(f"f{row['frame_index']}={_fmt_vector(noisy, 4)}")
        lines.append(f"object {object_id}: " + "; ".join(values))
    return "Low-noise estimated positions (sigma=0.02 scene units): " + "; ".join(lines)


def _bbox_pixels(row: dict, image: Image.Image) -> Optional[tuple[int, int, int, int]]:
    bbox = row.get("bbox_2d_normalized")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    ymin, xmin, ymax, xmax = map(float, bbox)
    return (round(xmin * image.width), round(ymin * image.height), round(xmax * image.width), round(ymax * image.height))


def visual_frame_labels(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    fps = float(doc.get("fps", 12.0))
    output = []
    for index, image in zip(frame_indices(doc, len(images)), images):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        label = f"frame {index}  t={index/fps:.3f}s"
        draw.rectangle((0, 0, min(canvas.width, 150), 18), fill=(0, 0, 0))
        draw.text((3, 3), label, fill=(255, 255, 255))
        output.append(canvas)
    return output


def visual_endpoint_highlight(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    start, end = int(doc["frame_start"]), int(doc["frame_end"])
    output = []
    for index, image in zip(frame_indices(doc, len(images)), images):
        canvas = image.copy()
        if index in {start, end}:
            draw = ImageDraw.Draw(canvas)
            color = (30, 220, 80) if index == start else (255, 70, 70)
            draw.rectangle((1, 1, canvas.width - 2, canvas.height - 2), outline=color, width=5)
            draw.text((5, 22), "START" if index == start else "END", fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        output.append(canvas)
    return output


def _track_ids(doc: dict, uniform_candidates: bool = False) -> list[int]:
    return _candidate_ids(doc) if uniform_candidates else [int(doc["target_object_id"])]


def _visual_boxes(doc: dict, images: list[Image.Image], uniform_candidates: bool) -> list[Image.Image]:
    ids = _track_ids(doc, uniform_candidates)
    by_id = {object_id: {int(row["frame_index"]): row for row in _frames(doc, object_id)} for object_id in ids}
    output = []
    for frame_index, image in zip(frame_indices(doc, len(images)), images):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for color_index, object_id in enumerate(ids):
            row = by_id[object_id].get(frame_index)
            box = _bbox_pixels(row, canvas) if row else None
            if box:
                color = COLORS[color_index % len(COLORS)]
                draw.rectangle(box, outline=color, width=3)
                draw.text((box[0] + 2, max(1, box[1] - 12)), f"obj {object_id}", fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
        output.append(canvas)
    return output


def visual_bbox_track(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    return _visual_boxes(doc, images, _is_comparison(doc))


def visual_candidate_tracks(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    return _visual_boxes(doc, images, True)


def visual_segmentation_track(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    ids = _track_ids(doc, _metric(doc) in {"fastest_object", "greatest_path_length_object"})
    paths = frame_paths(doc)
    original_indices = frame_indices(doc, len(images))
    output = []
    for path_position, (frame_index, image) in enumerate(zip(original_indices, images)):
        canvas = image.convert("RGBA")
        sequence_dir = Path(paths[path_position]).parent.parent
        mask_path = sequence_dir / "segmentations" / f"segmentations_{frame_index:05d}.png"
        if mask_path.is_file():
            mask = Image.open(mask_path).convert("L")
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            for color_index, object_id in enumerate(ids):
                binary = mask.point(lambda value, sid=object_id + 1: 100 if value == sid else 0)
                color = COLORS[color_index % len(COLORS)] + (0,)
                layer = Image.new("RGBA", canvas.size, color)
                layer.putalpha(binary)
                overlay = Image.alpha_composite(overlay, layer)
            canvas = Image.alpha_composite(canvas, overlay)
        output.append(canvas.convert("RGB"))
    return output


def visual_center_track(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    ids = _candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]
    rows_by_id = {
        object_id: {int(row["frame_index"]): row for row in _frames(doc, object_id)}
        for object_id in ids
    }
    points_by_id: dict[int, list[tuple[int, int]]] = {object_id: [] for object_id in ids}
    output = []
    for frame_index, image in zip(frame_indices(doc, len(images)), images):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for color_index, object_id in enumerate(ids):
            row = rows_by_id[object_id].get(frame_index)
            if row:
                x, y = map(float, row["image_position_normalized"])
                points_by_id[object_id].append((round(x * canvas.width), round(y * canvas.height)))
            points = points_by_id[object_id]
            color = COLORS[color_index % len(COLORS)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        output.append(canvas)
    return output


def _scale_bar_geometry(doc: dict, frame_index: int, image_width: int) -> tuple[float, int]:
    """Return scene units and pixels for a camera-plane ruler at target depth."""

    row = next(
        (item for item in _frames(doc) if int(item["frame_index"]) == int(frame_index)),
        None,
    )
    if row is None:
        raise ValueError(f"{doc.get('qid')}: target has no GT row for frame {frame_index}")
    camera = sequence_gt(doc).get("camera") or {}
    field_of_view = float(camera.get("field_of_view", 0.0))
    if field_of_view <= 0.0:
        focal_length = float(camera.get("focal_length", 0.0))
        sensor_width = float(camera.get("sensor_width", 0.0))
        if focal_length <= 0.0 or sensor_width <= 0.0:
            raise ValueError(f"{doc.get('qid')}: camera metadata cannot define a metric scale guide")
        field_of_view = 2.0 * math.atan(sensor_width / (2.0 * focal_length))
    position_camera = row.get("position_camera") or []
    depth = abs(float(position_camera[2])) if len(position_camera) >= 3 else 0.0
    if depth <= 1e-6:
        depth = float(row.get("distance_to_camera", 0.0))
    if depth <= 1e-6:
        raise ValueError(f"{doc.get('qid')}: target depth is unavailable for frame {frame_index}")

    focal_pixels = (float(image_width) / 2.0) / math.tan(field_of_view / 2.0)
    pixels_per_unit = focal_pixels / depth
    scene_units = 1.0
    # Keep the ruler legible and inside the lower-left legend while retaining
    # an exact power-of-two relationship to one scene unit.
    while pixels_per_unit * scene_units > image_width * 0.42:
        scene_units /= 2.0
    while pixels_per_unit * scene_units < max(24.0, image_width * 0.12):
        scene_units *= 2.0
    return scene_units, max(1, round(pixels_per_unit * scene_units))


def _draw_metric_scale_guide(doc: dict, image: Image.Image, frame_index: int) -> Image.Image:
    """Overlay a perspective-correct metric ruler at the target's depth."""

    units, bar_pixels = _scale_bar_geometry(doc, frame_index, image.width)
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin = max(5, round(image.width * 0.02))
    box_height = 43
    box_width = min(image.width - 2 * margin, max(158, bar_pixels + 24))
    top = image.height - margin - box_height
    draw.rounded_rectangle(
        (margin, top, margin + box_width, image.height - margin),
        radius=4,
        fill=(0, 0, 0, 190),
        outline=(255, 255, 255, 225),
        width=1,
    )
    draw.text((margin + 6, top + 3), "SCALE AT TARGET DEPTH", fill=(255, 255, 255, 255))
    x0, y = margin + 8, top + 26
    x1 = min(x0 + bar_pixels, margin + box_width - 8)
    draw.line((x0, y, x1, y), fill=(255, 230, 60, 255), width=3)
    for x in (x0, round((x0 + x1) / 2), x1):
        draw.line((x, y - 5, x, y + 5), fill=(255, 230, 60, 255), width=2)
    unit_label = f"{units:g} scene unit" + ("s" if units != 1.0 else "")
    label_x = min(x1 + 5, margin + box_width - 88)
    draw.text((label_x, y - 7), unit_label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def visual_metric_scale_guide(
    doc: dict,
    images: list[Image.Image],
    source_frame_indices: Optional[list[int]] = None,
) -> list[Image.Image]:
    """Add the shared metric guide, or a scale-only panel for no-scene input."""

    indices = source_frame_indices if source_frame_indices is not None else frame_indices(doc, len(images))
    if images:
        return [
            _draw_metric_scale_guide(doc, image, indices[position])
            if position < len(indices)
            else image
            for position, image in enumerate(images)
        ]

    # Preserve the state-only ablation's lack of scene pixels while ensuring
    # every numerical condition still receives the requested plotted guide.
    paths = frame_paths(doc)
    source_indices = frame_indices(doc, len(paths))
    reference_position = len(source_indices) // 2
    reference_frame = source_indices[reference_position]
    with Image.open(paths[reference_position]) as reference:
        width, height = reference.size
    panel = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "SCALE-ONLY INPUT", fill=(0, 0, 0))
    draw.text((8, 24), "Scene pixels intentionally hidden", fill=(0, 0, 0))
    return [_draw_metric_scale_guide(doc, panel, reference_frame)]


def visual_corner_metric_scale_guide(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    """Condition-local version of the legacy lower-left metric ruler."""

    return [
        _draw_metric_scale_guide(doc, image, frame_index)
        for frame_index, image in zip(frame_indices(doc, len(images)), images)
    ]


def _rectangle_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _anchored_scale_placement(
    target_box: tuple[int, int, int, int],
    obstacle_boxes: list[tuple[int, int, int, int]],
    bar_pixels: int,
    image: Image.Image,
) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    """Choose a low-occlusion ruler parallel to the target's major image side."""

    left, top, right, bottom = target_box
    width, height = image.size
    gap, edge = max(5, round(width * 0.02)), 4
    horizontal = (right - left) >= (bottom - top)
    candidates = []
    if horizontal:
        x0 = max(edge, min((left + right - bar_pixels) // 2, width - edge - bar_pixels))
        for y in (bottom + gap, top - gap):
            start, end = (x0, y), (x0 + bar_pixels, y)
            footprint = (x0 - 5, y - 18, x0 + max(bar_pixels, 94) + 5, y + 8)
            candidates.append(("horizontal", start, end, footprint))
    else:
        y0 = max(edge, min((top + bottom - bar_pixels) // 2, height - edge - bar_pixels))
        for x in (right + gap, left - gap):
            start, end = (x, y0), (x, y0 + bar_pixels)
            footprint = (x - 8, y0 - 5, x + 104, y0 + bar_pixels + 5)
            candidates.append(("vertical", start, end, footprint))

    def score(candidate) -> tuple[int, int]:
        _, _, _, footprint = candidate
        outside = (
            max(0, -footprint[0]) + max(0, -footprint[1])
            + max(0, footprint[2] - width) + max(0, footprint[3] - height)
        )
        overlap = sum(_rectangle_overlap(footprint, box) for box in obstacle_boxes)
        return outside * 10000 + overlap, abs(footprint[0] - left) + abs(footprint[1] - top)

    return min(candidates, key=score)


def _draw_object_anchored_metric_scale(doc: dict, image: Image.Image, frame_index: int) -> tuple[Image.Image, dict]:
    """Draw a ruler beside the target and explicitly connect it to that target."""

    rows = [row for item in _objects(doc) for row in item.get("frame_quantities", []) if int(row["frame_index"]) == int(frame_index)]
    target_id = int(doc["target_object_id"])
    target_row = next((row for row in rows if int(row["object_id"]) == target_id), None)
    target_box = _bbox_pixels(target_row, image) if target_row else None
    if target_box is None:
        # Preserve a usable visual if a target box is unavailable.
        return _draw_metric_scale_guide(doc, image, frame_index), {
            "frame_index": frame_index,
            "fallback": "corner_metric_scale_guide",
        }

    obstacle_boxes = [box for row in rows if int(row["object_id"]) != target_id if (box := _bbox_pixels(row, image))]
    # Reserve the cyan TARGET tag as well as other scene objects when choosing
    # between the two outside placements.
    obstacle_boxes.append((target_box[0], max(0, target_box[1] - 13), target_box[0] + 45, target_box[1]))
    units, bar_pixels = _scale_bar_geometry(doc, frame_index, image.width)
    orientation, start, end, footprint = _anchored_scale_placement(
        target_box, obstacle_boxes, bar_pixels, image,
    )
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cyan, yellow = (30, 235, 255, 255), (255, 230, 60, 255)
    target_center = ((target_box[0] + target_box[2]) // 2, (target_box[1] + target_box[3]) // 2)
    bar_center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)

    draw.rectangle(target_box, outline=cyan, width=2)
    draw.text(
        (target_box[0] + 2, max(1, target_box[1] - 12)), "TARGET",
        fill=cyan, stroke_width=2, stroke_fill=(0, 0, 0, 230),
    )
    draw.line((target_center, bar_center), fill=(30, 235, 255, 190), width=1)
    draw.line((start, end), fill=(0, 0, 0, 230), width=7)
    draw.line((start, end), fill=yellow, width=3)
    if orientation == "horizontal":
        for x in (start[0], round((start[0] + end[0]) / 2), end[0]):
            draw.line((x, start[1] - 5, x, start[1] + 5), fill=(0, 0, 0, 230), width=5)
            draw.line((x, start[1] - 5, x, start[1] + 5), fill=yellow, width=2)
        label_position = (start[0], max(1, start[1] - 17))
    else:
        for y in (start[1], round((start[1] + end[1]) / 2), end[1]):
            draw.line((start[0] - 5, y, start[0] + 5, y), fill=(0, 0, 0, 230), width=5)
            draw.line((start[0] - 5, y, start[0] + 5, y), fill=yellow, width=2)
        label_position = (min(image.width - 90, start[0] + 7), max(1, start[1]))

    label = f"{units:g} scene unit" + ("s" if units != 1.0 else "")
    text_box = draw.textbbox(label_position, label, stroke_width=1)
    draw.rounded_rectangle(
        (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
        radius=3, fill=(0, 0, 0, 195),
    )
    draw.text(label_position, label, fill=(255, 255, 255, 255))
    geometry = {
        "frame_index": int(frame_index),
        "target_object_id": target_id,
        "target_bbox_pixels": list(target_box),
        "orientation": orientation,
        "scene_units": units,
        "bar_pixels": bar_pixels,
        "bar_start_pixels": list(start),
        "bar_end_pixels": list(end),
        "placement_footprint_pixels": list(footprint),
        "reference_depth": "target_center_camera_depth",
    }
    return Image.alpha_composite(canvas, overlay).convert("RGB"), geometry


def visual_object_anchored_metric_scale_guide(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    """Add a target-associated, perspective-correct ruler to each source frame."""

    outputs, geometry = [], []
    for frame_index, image in zip(frame_indices(doc, len(images)), images):
        rendered, record = _draw_object_anchored_metric_scale(doc, image, frame_index)
        outputs.append(rendered)
        geometry.append(record)
    _trace_update(
        doc,
        object_anchored_scale_guide={
            "description": "Camera-plane ruler at target-center depth, aligned to the target's major projected side.",
            "frames": geometry,
        },
    )
    return outputs


def visual_trajectory_plot(doc: dict, images: list[Image.Image]) -> list[Image.Image]:
    size = 320
    panel = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), "GT top-down world trajectory (X/Y)", fill=(0, 0, 0))
    ids = _candidate_ids(doc) if _is_comparison(doc) else [int(doc["target_object_id"])]
    rows_by_id = {object_id: _relevant_frames(doc, object_id) for object_id in ids}
    all_rows = [row for rows in rows_by_id.values() for row in rows]
    if all_rows:
        xs = [float(row["position_world"][0]) for row in all_rows]
        ys = [float(row["position_world"][1]) for row in all_rows]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        span_x, span_y = max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6)
        for color_index, object_id in enumerate(ids):
            rows = rows_by_id[object_id]
            color = COLORS[color_index % len(COLORS)]
            points = [
                (30 + round((float(row["position_world"][0]) - min_x) / span_x * 260),
                 290 - round((float(row["position_world"][1]) - min_y) / span_y * 240))
                for row in rows
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            for point in points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
            if points:
                draw.text((points[-1][0] + 5, points[-1][1]), f"obj {object_id}", fill=color)
    return [*images, panel]


SCAFFOLDS: dict[str, ScaffoldSpec] = {
    "baseline": ScaffoldSpec(),
    "corner_metric_scale_guide": ScaffoldSpec(visual_aids=(visual_corner_metric_scale_guide,)),
    "object_anchored_metric_scale_guide": ScaffoldSpec(visual_aids=(visual_object_anchored_metric_scale_guide,)),
    "temporal_oracle": ScaffoldSpec(visual_aids=(visual_frame_labels, visual_endpoint_highlight), frame_policy="interval"),
    "tracking_oracle": ScaffoldSpec(visual_aids=(visual_frame_labels, visual_bbox_track), frame_policy="interval"),
    "2d_motion_oracle": ScaffoldSpec(text_aids=(aid_2d_motion,), visual_aids=(visual_center_track,), frame_policy="interval"),
    "physical_state_oracle": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "state_formula_oracle": ScaffoldSpec(text_aids=(aid_raw_state, aid_formula)),
    "derived_quantity_oracle": ScaffoldSpec(text_aids=(aid_derived,)),
    "answer_text_oracle": ScaffoldSpec(text_aids=(aid_answer_text,)),
    "answer_letter_oracle": ScaffoldSpec(text_aids=(aid_answer_letter,)),
    "bbox_track": ScaffoldSpec(visual_aids=(visual_bbox_track,)),
    "segmentation_track": ScaffoldSpec(visual_aids=(visual_segmentation_track,)),
    "visibility_timeline": ScaffoldSpec(text_aids=(aid_visibility,)),
    "center_track": ScaffoldSpec(text_aids=(aid_2d_motion,), visual_aids=(visual_center_track,)),
    "uniform_candidate_tracks": ScaffoldSpec(visual_aids=(visual_candidate_tracks,)),
    "frame_labels": ScaffoldSpec(visual_aids=(visual_frame_labels,)),
    "endpoint_highlight": ScaffoldSpec(visual_aids=(visual_endpoint_highlight,)),
    "query_triplet": ScaffoldSpec(visual_aids=(visual_frame_labels,), frame_policy="triplet"),
    "interval_crop": ScaffoldSpec(frame_policy="interval"),
    "collision_timeline": ScaffoldSpec(text_aids=(aid_collision_timeline,)),
    "fps_and_dt": ScaffoldSpec(text_aids=(aid_fps,)),
    "world_position_table": ScaffoldSpec(text_aids=(aid_world_positions,)),
    "camera_position_table": ScaffoldSpec(text_aids=(aid_camera_positions,)),
    "camera_basis": ScaffoldSpec(text_aids=(aid_camera_basis,)),
    "distance_to_camera_trace": ScaffoldSpec(text_aids=(aid_distance_trace,)),
    "top_down_trajectory": ScaffoldSpec(visual_aids=(visual_trajectory_plot,)),
    "signed_axis_deltas": ScaffoldSpec(text_aids=(aid_signed_axis_deltas,)),
    "direction_definition": ScaffoldSpec(text_aids=(aid_direction_definition,)),
    "formula_only": ScaffoldSpec(text_aids=(aid_formula,)),
    "raw_inputs_only": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "inputs_plus_formula": ScaffoldSpec(text_aids=(aid_raw_state, aid_formula)),
    "component_differences": ScaffoldSpec(text_aids=(aid_component_differences,)),
    "squared_components": ScaffoldSpec(text_aids=(aid_squared_components,)),
    "segment_vectors": ScaffoldSpec(text_aids=(aid_segment_vectors,)),
    "segment_lengths": ScaffoldSpec(text_aids=(aid_segment_lengths,)),
    "velocity_difference": ScaffoldSpec(text_aids=(aid_velocity_difference,)),
    "norm_ready_vector": ScaffoldSpec(text_aids=(aid_norm_ready,)),
    "unrounded_derived_value": ScaffoldSpec(text_aids=(aid_derived,)),
    "rounded_derived_value": ScaffoldSpec(text_aids=(aid_rounded_derived,)),
    "worked_unrelated_example": ScaffoldSpec(text_aids=(aid_worked_example,)),
    "state_only_no_visual": ScaffoldSpec(text_aids=(aid_raw_state,), frame_policy="none"),
    "full_trace_no_tracking_overlay": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "compact_table": ScaffoldSpec(text_aids=(aid_state_table,), frame_policy="interval"),
    "json": ScaffoldSpec(text_aids=(aid_state_json,), frame_policy="interval"),
    "trajectory_plot": ScaffoldSpec(visual_aids=(visual_trajectory_plot,)),
    "precision_2dp": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "precision_4dp": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "precision_full": ScaffoldSpec(text_aids=(aid_raw_state,)),
    "coordinate_noise_low": ScaffoldSpec(text_aids=(aid_noisy_state,)),
    "reversed_frames": ScaffoldSpec(frame_policy="reverse"),
    "distractor_state": ScaffoldSpec(text_aids=(aid_distractor_state,)),
    "mismatched_object_state": ScaffoldSpec(text_aids=(aid_mismatched_state,)),
}


SUITES = {
    "baseline": ("baseline",),
    "visual_scale": ("baseline", "corner_metric_scale_guide", "object_anchored_metric_scale_guide"),
    "oracle_ladder": (
        "baseline", "temporal_oracle", "tracking_oracle", "2d_motion_oracle",
        "physical_state_oracle", "state_formula_oracle", "derived_quantity_oracle",
        "answer_text_oracle", "answer_letter_oracle",
    ),
    "perception": ("baseline", "bbox_track", "segmentation_track", "visibility_timeline", "center_track", "uniform_candidate_tracks"),
    "temporal": ("baseline", "frame_labels", "endpoint_highlight", "query_triplet", "interval_crop", "collision_timeline", "fps_and_dt"),
    "coordinate": ("baseline", "world_position_table", "camera_position_table", "camera_basis", "distance_to_camera_trace", "top_down_trajectory", "signed_axis_deltas", "direction_definition"),
    "calculation": (
        "baseline", "formula_only", "raw_inputs_only", "inputs_plus_formula", "component_differences",
        "squared_components", "segment_vectors", "segment_lengths", "velocity_difference",
        "norm_ready_vector", "unrounded_derived_value", "rounded_derived_value", "worked_unrelated_example",
        "state_only_no_visual", "full_trace_no_tracking_overlay",
    ),
    "representation": ("baseline", "compact_table", "json", "trajectory_plot"),
    "precision": ("baseline", "precision_2dp", "precision_4dp", "precision_full"),
    "robustness": ("baseline", "coordinate_noise_low", "reversed_frames", "distractor_state", "mismatched_object_state"),
}


CONDITION_SUBTASKS = {
    "corner_metric_scale_guide": NUMERIC_ANSWER_METRICS,
    "object_anchored_metric_scale_guide": NUMERIC_ANSWER_METRICS,
    "uniform_candidate_tracks": {"fastest_object", "greatest_path_length_object"},
    "signed_axis_deltas": {"dominant_motion_direction"},
    "direction_definition": {"dominant_motion_direction"},
    "component_differences": {"displacement"},
    "squared_components": {"displacement"},
    "segment_vectors": {"path_length", "average_speed", "greatest_path_length_object"},
    "segment_lengths": {"path_length", "average_speed", "greatest_path_length_object"},
    "velocity_difference": {"instantaneous_acceleration"},
    "norm_ready_vector": {"displacement", "instantaneous_speed", "instantaneous_acceleration", "fastest_object"},
}


def _condition_applies(doc: dict, condition: str) -> bool:
    allowed = CONDITION_SUBTASKS.get(condition)
    return allowed is None or _metric(doc) in allowed


def _sample_docs(dataset: Dataset) -> list[dict]:
    limit = int(os.getenv("MOVI_MOTION_SAMPLES_PER_SUBTASK", str(DEFAULT_SAMPLES_PER_SUBTASK)))
    derived_frames = int(os.getenv("MOVI_MOTION_DERIVED_NUM_FRAMES", "0"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for doc in dataset:
        grouped[str(doc.get("subtask", "unknown"))].append(dict(doc))
    selected = []
    for subtask in sorted(grouped):
        ranked = sorted(
            grouped[subtask],
            key=lambda doc: hashlib.sha1(str(doc.get("qid", "")).encode()).hexdigest(),
        )
        chosen = ranked[:limit] if limit >= 0 else ranked
        selected.extend(
            _derive_frame_limited_doc(doc, derived_frames) if derived_frames > 0 else doc
            for doc in chosen
        )
    return selected


def _process_suite(dataset: Dataset, suite: str) -> Dataset:
    conditions = SUITES[suite]
    records = []
    for source in _sample_docs(dataset):
        if suite == "visual_scale" and not _has_numeric_answer(source):
            continue
        source_qid = str(source["qid"])
        for condition in conditions:
            if not _condition_applies(source, condition):
                continue
            record = dict(source)
            record.update(
                {
                    "source_qid": source_qid,
                    "qid": f"{source_qid}::{condition}",
                    "index": f"{source_qid}::{condition}",
                    "scaffold_suite": suite,
                    "scaffold_condition": condition,
                }
            )
            records.append(record)
    eval_logger.info(
        "MOVi-A motion suite={} source_questions={} expanded_records={} conditions={}",
        suite, len({row["source_qid"] for row in records}), len(records), conditions,
    )
    return Dataset.from_list(records)


def process_baseline_docs(dataset): return _process_suite(dataset, "baseline")
def process_visual_scale_docs(dataset): return _process_suite(dataset, "visual_scale")
def process_oracle_ladder_docs(dataset): return _process_suite(dataset, "oracle_ladder")
def process_perception_docs(dataset): return _process_suite(dataset, "perception")
def process_temporal_docs(dataset): return _process_suite(dataset, "temporal")
def process_coordinate_docs(dataset): return _process_suite(dataset, "coordinate")
def process_calculation_docs(dataset): return _process_suite(dataset, "calculation")
def process_representation_docs(dataset): return _process_suite(dataset, "representation")
def process_precision_docs(dataset): return _process_suite(dataset, "precision")
def process_robustness_docs(dataset): return _process_suite(dataset, "robustness")


def _policy_indices(doc: dict, total: int, policy: Optional[str]) -> list[int]:
    original_indices = frame_indices(doc, total)
    if policy == "none":
        return []
    if policy == "interval":
        start, end = int(doc["frame_start"]), int(doc["frame_end"])
        return [position for position, frame_index in enumerate(original_indices) if start <= frame_index <= end]
    if policy == "triplet":
        query = int(ground_truth(doc).get("query_frame", (int(doc["frame_start"]) + int(doc["frame_end"])) // 2))
        return [position for position, frame_index in enumerate(original_indices) if query - 1 <= frame_index <= query + 1]
    positions = list(range(total))
    return list(reversed(positions)) if policy == "reverse" else positions


def doc_to_visual(doc):
    paths = frame_paths(doc)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{doc.get('qid')}: missing {len(missing)} frame(s), first={missing[0]}")
    # Visual aids use original frame indices. Apply them before temporal slicing.
    images = [Image.open(path).convert("RGB") for path in paths]
    spec = SCAFFOLDS[str(doc.get("scaffold_condition", "baseline"))]
    for aid in spec.visual_aids:
        images = aid(doc, images)
    # A plot aid adds a panel after the source frames; preserve that panel.
    source_count = len(paths)
    source_images, extra_images = images[:source_count], images[source_count:]
    indices = _policy_indices(doc, source_count, spec.frame_policy)
    outputs = [source_images[index] for index in indices] + extra_images
    condition_has_scale = str(doc.get("scaffold_condition", "baseline")) in {
        "corner_metric_scale_guide", "object_anchored_metric_scale_guide",
    }
    global_scale_active = (
        _has_numeric_answer(doc)
        and _numeric_scale_guide_enabled()
        and str(doc.get("scaffold_suite", "")) != "visual_scale"
    )
    if global_scale_active and not condition_has_scale:
        original_frame_indices = frame_indices(doc, source_count)
        source_outputs = visual_metric_scale_guide(
            doc,
            outputs[:len(indices)],
            [original_frame_indices[index] for index in indices],
        )
        if not indices:
            source_outputs = visual_metric_scale_guide(doc, [], [])
        outputs = source_outputs + outputs[len(indices):]
    _trace_visuals(doc, outputs, paths, indices)
    return outputs


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    condition = str(doc.get("scaffold_condition", "baseline"))
    spec = SCAFFOLDS[condition]
    aids = [text for function in spec.text_aids if (text := function(doc))]
    options = "\n".join(f"{letter}. {doc[letter]}" for letter in OPTION_LETTERS)
    parts = [str(kwargs.get("pre_prompt", "")).strip()]
    selected = _strict_json(doc.get("selected_frame_indices_json"), [])
    if isinstance(selected, list) and selected:
        fps = float(doc.get("fps", 12.0))
        timestamps = [round(int(index) / fps, 6) for index in selected]
        parts.append(
            "Derived-frame input: the images retain their original clip frame IDs. "
            f"Original frame IDs={list(map(int, selected))}; timestamps_seconds={timestamps}. "
            "Path-based targets are calculated over the displayed frames within the requested interval."
        )
    scale_condition = condition if condition in {
        "corner_metric_scale_guide", "object_anchored_metric_scale_guide",
    } else None
    global_scale_active = (
        _has_numeric_answer(doc)
        and _numeric_scale_guide_enabled()
        and str(doc.get("scaffold_suite", "")) != "visual_scale"
    )
    scale_guide_active = _has_numeric_answer(doc) and (global_scale_active or scale_condition is not None)
    if scale_guide_active:
        if scale_condition == "object_anchored_metric_scale_guide":
            scale_note = (
                "Object-anchored metric scale guide: the cyan TARGET box identifies the object whose center depth "
                "defines the perspective-correct ruler. The connected yellow ruler is labelled in scene units, "
                "lies parallel to the target's longer projected side, and represents a camera-plane segment at "
                "that depth. Use each frame's own ruler because its pixel length can change with target depth."
            )
        else:
            scale_note = (
                "Metric scale guide: each displayed scene frame includes a perspective-correct ruler labelled in "
                "scene units for a camera-plane segment at the target object's depth. Its on-screen length can vary "
                "across frames as the target depth changes."
            )
        if condition == "state_only_no_visual":
            scale_note += " The SCALE-ONLY INPUT panel contains no scene pixels."
        parts.append(scale_note)
    if aids:
        parts.append("\n".join(aids))
    parts.extend(
        [
            f"Question: {doc['question']}",
            "Options:\n" + options,
            str(kwargs.get("post_prompt", "Answer with only the option letter.")).strip(),
        ]
    )
    prompt = "\n\n".join(part for part in parts if part)
    _trace_update(
        doc,
        task_prompt=prompt,
        prompt_components={
            "pre_prompt": str(kwargs.get("pre_prompt", "")).strip(),
            "text_aids": aids,
            "question": str(doc.get("question", "")),
            "options": {letter: str(doc.get(letter, "")) for letter in OPTION_LETTERS},
            "post_prompt": str(kwargs.get("post_prompt", "Answer with only the option letter.")).strip(),
        },
        scaffold={
            "frame_policy": spec.frame_policy,
            "text_aids": [getattr(aid, "__name__", str(aid)) for aid in spec.text_aids],
            "visual_aids": [getattr(aid, "__name__", str(aid)) for aid in spec.visual_aids]
            + (["visual_metric_scale_guide"] if scale_guide_active and scale_condition is None else []),
        },
        source_document=dict(doc),
        target={
            "answer_letter": str(doc.get("answer", "")),
            "answer_text": str(doc.get(str(doc.get("answer", "")), "")),
            "answer_value": doc.get("answer_value"),
            "answer_unit": doc.get("answer_unit"),
        },
    )
    return prompt


def doc_to_target(doc):
    return str(doc["answer"])


def _extract_letter(text: str) -> Optional[str]:
    text = str(text or "")
    for pattern in (r"^\s*([A-E])(?:[\s.)]|$)", r"\b(?:answer|option)\s*(?:is|:)?\s*([A-E])\b", r"\(([A-E])\)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def process_results(doc, results):
    prediction = str(results[0] if results else "")
    parsed = _extract_letter(prediction)
    gold = str(doc["answer"]).upper()
    score = float(parsed == gold)
    numeric_error = None
    if parsed in OPTION_LETTERS and doc.get("answer_value") is not None:
        selected = str(doc.get(parsed, ""))
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", selected)
        if match:
            numeric_error = abs(float(match.group()) - float(doc["answer_value"]))
    entry = {
        "qid": doc.get("qid"), "source_qid": doc.get("source_qid"),
        "sequence_id": doc.get("sequence_id"), "subtask": doc.get("subtask"),
        "suite": doc.get("scaffold_suite"), "condition": doc.get("scaffold_condition"),
        "score": score, "parse_success": float(parsed is not None),
        "numeric_option_error": numeric_error, "prediction": prediction,
        "parsed_prediction": parsed, "gold": gold,
    }
    _trace_update(
        doc,
        scored_output={
            "model_response": prediction,
            "parsed_prediction": parsed,
            "gold": gold,
            "correct": bool(score),
            "parse_success": parsed is not None,
            "numeric_option_error": numeric_error,
        },
    )
    return {
        "accuracy": entry,
        "baseline_accuracy": entry,
        "assisted_accuracy": entry,
        "paired_scaffold_gain": entry,
        "parse_success": entry,
        "numeric_option_error": entry,
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_accuracy(results):
    for (condition, subtask), items in sorted(_group(results, "condition", "subtask").items()):
        eval_logger.info("MOVi motion condition={} subtask={} accuracy={:.4f} n={}", condition, subtask, _mean([x["score"] for x in items]), len(items))
    return _mean([item["score"] for item in results])


def _group(results, *keys):
    grouped = defaultdict(list)
    for item in results:
        grouped[tuple(item.get(key) for key in keys)].append(item)
    return grouped


def aggregate_baseline_accuracy(results):
    values = [item["score"] for item in results if item.get("condition") == "baseline"]
    return _mean(values)


def aggregate_assisted_accuracy(results):
    values = [item["score"] for item in results if item.get("condition") != "baseline"]
    return _mean(values)


def aggregate_paired_gain(results):
    by_source = defaultdict(dict)
    for item in results:
        by_source[item["source_qid"]][item["condition"]] = item["score"]
    gains = []
    rescue = Counter()
    first_rescue = Counter()
    regressions = Counter()
    for conditions in by_source.values():
        if "baseline" not in conditions:
            continue
        for condition, score in conditions.items():
            if condition != "baseline":
                gains.append(score - conditions["baseline"])
                if conditions["baseline"] == 0 and score == 1:
                    rescue[condition] += 1
                if conditions["baseline"] == 1 and score == 0:
                    regressions[condition] += 1
        if conditions["baseline"] == 0:
            for condition in SUITES["oracle_ladder"][1:]:
                if conditions.get(condition) == 1:
                    first_rescue[condition] += 1
                    break
    eval_logger.info("MOVi motion baseline-error rescues by condition: {}", dict(rescue))
    if first_rescue:
        eval_logger.info("MOVi motion first oracle-ladder rescue stage: {}", dict(first_rescue))
    if regressions:
        eval_logger.info("MOVi motion baseline-correct regressions by condition: {}", dict(regressions))
    return _mean(gains)


def aggregate_parse_success(results):
    return _mean([item["parse_success"] for item in results])


def aggregate_numeric_error(results):
    values = [item["numeric_option_error"] for item in results if item.get("numeric_option_error") is not None]
    return _mean(values)
