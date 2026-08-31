#!/usr/bin/env python3
"""Plot all COMFORT_Multi_3D GT and predicted bboxes, grouped by image."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SUBMISSION = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/comfort_bbox/evaluation/"
    "submissions/comfort_multi_3d_bbox_prediction_qwen3_vl_experiments.json"
)
COORDINATE_MAX = 1000.0
GT_COLOR = (35, 220, 80)
PRED_COLOR = (255, 65, 65)
PANEL_BACKGROUND = (245, 245, 245)
TEXT_COLOR = (20, 20, 20)
DIRECTION_ORDER = {None: 0, "left": 1, "right": 2, "front": 3, "behind": 4}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Expected a record list or a wrapped submission with 'records'")
    return records


def group_by_scene(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        scene_id = str(row.get("scene_id") or "").strip()
        if not scene_id:
            raise ValueError(f"Record {row.get('qid', '<unknown>')} has no scene_id")
        grouped[scene_id].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                0 if row.get("object_role") == "reference" else 1,
                DIRECTION_ORDER.get(row.get("reference_direction"), 99),
                str(row.get("object_id", "")),
            )
        )
    return dict(sorted(grouped.items()))


def _valid_bbox(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (
        all(0.0 <= item <= COORDINATE_MAX for item in box)
        and x2 > x1
        and y2 > y1
    ):
        return None
    return box


def _to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box[0] / COORDINATE_MAX * width),
        round(box[1] / COORDINATE_MAX * height),
        round(box[2] / COORDINATE_MAX * width),
        round(box[3] / COORDINATE_MAX * height),
    )


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_tag(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    canvas_size: tuple[int, int],
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = max(0, min(canvas_size[0] - text_width - 6, position[0]))
    y = max(0, min(canvas_size[1] - text_height - 4, position[1] - text_height - 4))
    draw.rectangle((x, y, x + text_width + 6, y + text_height + 4), fill=color)
    draw.text((x + 3, y + 2), text, fill=(0, 0, 0), font=font)


def _format_bbox(box: Optional[list[float]]) -> str:
    if box is None:
        return "parse failure"
    return "[" + ", ".join(f"{value:.0f}" for value in box) + "]"


def render_scene(
    scene_id: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    image_paths = {str(row.get("img_path") or "") for row in rows}
    if len(image_paths) != 1:
        raise ValueError(f"Scene {scene_id} has inconsistent image paths: {image_paths}")
    image_path = Path(next(iter(image_paths)))
    if not image_path.is_file():
        raise FileNotFoundError(f"Scene image not found: {image_path}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_width, image_height = image.size
    scale = max(1.0, min(image.size) / 512.0)
    font = _font(max(12, round(13 * scale)))
    title_font = _font(max(15, round(17 * scale)))
    line_height = max(18, round(21 * scale))
    panel_width = max(410, round(image_width * 0.82))
    panel_height = max(image_height, 94 + len(rows) * line_height * 3)
    canvas = Image.new("RGB", (image_width + panel_width, panel_height), PANEL_BACKGROUND)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(min(image.size) / 150))

    parsed_count = 0
    ious = []
    object_summaries = []
    for number, row in enumerate(rows, start=1):
        gt_bbox = _valid_bbox(row.get("gt_bbox"))
        if gt_bbox is None:
            raise ValueError(f"Record {row.get('qid')} has an invalid GT bbox")
        predicted_bbox = _valid_bbox(row.get("predicted_bbox"))
        gt_pixels = _to_pixels(gt_bbox, image_width, image_height)
        draw.rectangle(gt_pixels, outline=GT_COLOR, width=line_width)
        _draw_tag(draw, gt_pixels[:2], f"G{number}", GT_COLOR, font, image.size)

        if predicted_bbox is not None:
            parsed_count += 1
            pred_pixels = _to_pixels(predicted_bbox, image_width, image_height)
            draw.rectangle(pred_pixels, outline=PRED_COLOR, width=line_width)
            _draw_tag(draw, pred_pixels[:2], f"P{number}", PRED_COLOR, font, image.size)
        try:
            iou = float(row.get("iou", 0.0))
        except (TypeError, ValueError):
            iou = 0.0
        ious.append(iou)
        object_summaries.append(
            {
                "number": number,
                "qid": row.get("qid"),
                "object_id": row.get("object_id"),
                "object_label": row.get("object_label"),
                "object_role": row.get("object_role"),
                "reference_direction": row.get("reference_direction"),
                "iou": iou,
                "gt_bbox": gt_bbox,
                "predicted_bbox": predicted_bbox,
            }
        )

    panel_x = image_width + 18
    draw.text((panel_x, 14), scene_id, fill=TEXT_COLOR, font=title_font)
    draw.text((panel_x, 43), "GT boxes/tags", fill=GT_COLOR, font=font)
    draw.text((panel_x + 115, 43), "Prediction boxes/tags", fill=PRED_COLOR, font=font)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    draw.text(
        (panel_x, 66),
        f"Scene mean IoU: {mean_iou:.3f} | parsed: {parsed_count}/{len(rows)}",
        fill=TEXT_COLOR,
        font=font,
    )

    y = 94
    for item in object_summaries:
        direction = item["reference_direction"]
        descriptor = str(item["object_role"])
        if direction:
            descriptor += f", {direction}"
        draw.text(
            (panel_x, y),
            f"{item['number']}. {item['object_label']} ({descriptor})  IoU={item['iou']:.3f}",
            fill=TEXT_COLOR,
            font=font,
        )
        draw.text(
            (panel_x + 14, y + line_height),
            f"GT   {_format_bbox(item['gt_bbox'])}",
            fill=GT_COLOR,
            font=font,
        )
        draw.text(
            (panel_x + 14, y + 2 * line_height),
            f"Pred {_format_bbox(item['predicted_bbox'])}",
            fill=PRED_COLOR,
            font=font,
        )
        y += 3 * line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return {
        "scene_id": scene_id,
        "source_image": str(image_path),
        "output_image": str(output_path),
        "object_count": len(rows),
        "parsed_count": parsed_count,
        "parse_failure_count": len(rows) - parsed_count,
        "mean_iou": mean_iou,
        "objects": object_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        default=DEFAULT_SUBMISSION,
        help=f"Wrapped bbox submission JSON (default: {DEFAULT_SUBMISSION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination; defaults beside the submission as bbox_visualizations/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Render only the first N scenes for a quick check",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not args.submission.is_file():
        parser.error(f"Submission not found: {args.submission}")

    try:
        grouped = group_by_scene(load_records(args.submission))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    scene_items = list(grouped.items())
    if args.limit is not None:
        scene_items = scene_items[: args.limit]
    output_dir = (
        args.output_dir
        or args.submission.parent / "bbox_visualizations"
    ).resolve()

    scene_reports = []
    for index, (scene_id, rows) in enumerate(scene_items, start=1):
        report = render_scene(scene_id, rows, output_dir / f"{scene_id}.png")
        scene_reports.append(report)
        if index % 50 == 0 or index == len(scene_items):
            print(f"Rendered {index}/{len(scene_items)} scenes")

    total_objects = sum(report["object_count"] for report in scene_reports)
    total_parsed = sum(report["parsed_count"] for report in scene_reports)
    summary = {
        "submission": str(args.submission.resolve()),
        "output_dir": str(output_dir),
        "scene_count": len(scene_reports),
        "object_count": total_objects,
        "parsed_count": total_parsed,
        "parse_failure_count": total_objects - total_parsed,
        "mean_iou": (
            sum(
                item["iou"]
                for report in scene_reports
                for item in report["objects"]
            )
            / total_objects
            if total_objects
            else None
        ),
        "scenes": scene_reports,
    }
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(scene_reports)} scene visualizations to {output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
