#!/usr/bin/env python3
"""Build the ScanNet camera-basis evaluation manifest from prepared scenes.

The input layout is produced by ``Spatial_Benchmarks/scannet_download.sh``.
Each emitted row is one RGB view with 4--5 visible, uniquely identifiable
instances;
the task expands it into ordered reference/target spatial-relation queries.
Run this again after adding prepared ScanNet scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/mnt/rdata4_3/Spatial_Benchmarks/scannetv2")
# These classes are scene structure rather than individually identifiable
# objects for a reference/target relation. Keep the list intentionally limited
# to structural architecture; large furniture remains a valid object.
STRUCTURAL_LABELS = {"floor", "wall", "ceiling", "door", "window", "windowsill"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames-per-scene", type=int, default=8)
    parser.add_argument("--max-objects-per-frame", type=int, default=5)
    parser.add_argument("--min-objects-per-frame", type=int, default=4)
    parser.add_argument("--min-visible-pixels", type=int, default=800)
    return parser.parse_args()


def _selected_evenly(rows: list[dict], maximum: int) -> list[dict]:
    if len(rows) <= maximum:
        return rows
    if maximum == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (maximum - 1))] for index in range(maximum)]


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label(instance: dict, source: dict | None) -> str:
    return str(instance.get("label", source.get("label") if source else "unknown")).strip().lower()


def _view_objects(frame: dict, by_object_id: dict[int, dict], camera_from_first: np.ndarray, width: int, height: int, minimum_pixels: int, maximum_objects: int) -> list[dict]:
    # Count every annotated non-structural visible instance before applying the
    # size threshold. If two chairs are visible, neither is identifiable from a
    # label-only prompt even when one happens to be too small for evaluation.
    visible_label_counts: dict[str, int] = {}
    for instance in frame.get("objects", []):
        source = by_object_id.get(int(instance["object_id"]))
        label = _label(instance, source)
        if source is not None and label not in STRUCTURAL_LABELS:
            visible_label_counts[label] = visible_label_counts.get(label, 0) + 1

    visible = []
    for instance in frame.get("objects", []):
        object_id = int(instance["object_id"])
        source = by_object_id.get(object_id)
        label = _label(instance, source)
        if (source is None or label in STRUCTURAL_LABELS or visible_label_counts.get(label) != 1
                or int(instance.get("visible_pixel_count", 0)) < minimum_pixels):
            continue
        point = np.asarray([*source["bbox3d_center_xyz_m"], 1.0], dtype=np.float64)
        camera_point = camera_from_first @ point
        # The instance mask can contain a very small visible fragment while its
        # 3D centre is behind the camera. Such centres are unsuitable for a
        # camera-frame direction target.
        if not np.isfinite(camera_point).all() or camera_point[1] <= 1e-4:
            continue
        xmin, ymin, xmax, ymax = (int(value) for value in instance["bbox2d_xyxy_px"])
        visible.append({
            "object_id": object_id,
            "label": label,
            "display_name": f"{label} #{object_id}",
            "position_camera_xyz_m": [round(float(value), 6) for value in camera_point[:3]],
            "bbox_2d_normalized_xyxy": [
                round(xmin / width, 6), round(ymin / height, 6),
                round((xmax + 1) / width, 6), round((ymax + 1) / height, 6),
            ],
            "visible_pixel_count": int(instance["visible_pixel_count"]),
        })
    visible.sort(key=lambda item: (-item["visible_pixel_count"], item["object_id"]))
    return visible[:maximum_objects]


def main() -> None:
    args = parse_args()
    if args.frames_per_scene < 1 or args.min_visible_pixels < 1 or args.min_objects_per_frame < 4:
        raise ValueError("frames-per-scene/min-visible-pixels must be positive and min-objects-per-frame must be at least 4")
    if args.max_objects_per_frame < args.min_objects_per_frame:
        raise ValueError("max-objects-per-frame must be at least min-objects-per-frame")
    root = args.root.resolve()
    output = (args.output or root / "scannet_camera_basis_manifest.jsonl").resolve()
    records, summary = [], {}
    for scene_dir in sorted(path for path in root.glob("scene*_??") if (path / "SUCCESS").is_file()):
        metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
        objects = json.loads((scene_dir / "objects.json").read_text(encoding="utf-8"))["objects"]
        by_object_id = {int(obj["object_id"]): obj for obj in objects}
        frames = _load_rows(scene_dir / "frames.jsonl")
        with np.load(scene_dir / "camera.npz") as camera:
            camera_from_first = np.asarray(camera["camera_from_first"], dtype=np.float64)
        candidates = []
        for frame in frames:
            frame_id = int(frame["frame_id"])
            if not frame.get("pose_valid") or frame_id >= len(camera_from_first):
                continue
            view_objects = _view_objects(
                frame, by_object_id, camera_from_first[frame_id], int(metadata["image_width"]),
                int(metadata["image_height"]), args.min_visible_pixels, args.max_objects_per_frame,
            )
            image_path = scene_dir / str(frame["rgb_file"])
            # ``SUCCESS`` means the preparation script has already validated
            # RGB/frame count agreement, so avoid an expensive network-filesystem
            # stat for every video frame here. Task loading still verifies each
            # selected image before evaluation.
            if len(view_objects) >= args.min_objects_per_frame:
                candidates.append({
                    "scene_id": str(metadata["scene_id"]), "frame_id": frame_id,
                    "image_path": str(image_path), "objects": view_objects,
                })
        selected = _selected_evenly(candidates, args.frames_per_scene)
        records.extend(selected)
        summary[scene_dir.name] = {"eligible_views": len(candidates), "selected_views": len(selected)}
    if not records:
        raise RuntimeError(f"No usable views found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(output), "views": len(records), "ordered_pairs": sum(len(row["objects"]) * (len(row["objects"]) - 1) for row in records), "scenes": len(summary), "per_scene": summary}, indent=2))


if __name__ == "__main__":
    main()
