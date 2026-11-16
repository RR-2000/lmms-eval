#!/usr/bin/env python3
"""Render examples of an object-anchored metric ruler for MOVi-A motion tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/ramanathan/VLM/lmms-eval")
DATASET = Path("/home/ramanathan/data/movi_a_motion/qa.parquet")
OUTPUT = ROOT / "outputs/kubric_movi_a_motion_12f_full_scale_guide/analysis/examples/object_anchored_scale"
NUMERIC = {
    "displacement",
    "path_length",
    "average_speed",
    "instantaneous_speed",
    "instantaneous_acceleration",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def bbox_pixels(row: dict, image: Image.Image) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = map(float, row["bbox_2d_normalized"])
    return (
        round(xmin * image.width),
        round(ymin * image.height),
        round(xmax * image.width),
        round(ymax * image.height),
    )


def choose_anchor(
    bbox: tuple[int, int, int, int], bar_pixels: int, width: int, height: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Place a bar outside the bbox, parallel to its longer projected side."""

    left, top, right, bottom = bbox
    gap, pad = 8, 7
    horizontal = (right - left) >= (bottom - top)
    if horizontal:
        x0 = max(pad, min((left + right - bar_pixels) // 2, width - pad - bar_pixels))
        if bottom + gap + 18 < height:
            y = bottom + gap
        else:
            y = max(18, top - gap)
        return "horizontal", (x0, y), (x0 + bar_pixels, y)

    y0 = max(pad, min((top + bottom - bar_pixels) // 2, height - pad - bar_pixels))
    if right + gap + 18 < width:
        x = right + gap
    else:
        x = max(18, left - gap)
    return "vertical", (x, y0), (x, y0 + bar_pixels)


def draw_object_anchored_scale(utils, doc: dict, image: Image.Image, frame_index: int) -> Image.Image:
    row = next(item for item in utils._frames(doc) if int(item["frame_index"]) == frame_index)
    bbox = bbox_pixels(row, image)
    units, bar_pixels = utils._scale_bar_geometry(doc, frame_index, image.width)
    orientation, start, end = choose_anchor(bbox, bar_pixels, image.width, image.height)

    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cyan = (30, 235, 255, 255)
    yellow = (255, 226, 45, 255)
    label_font = font(11, bold=True)
    small_font = font(9)

    # The target box and leader line make the ruler's reference depth explicit.
    draw.rectangle(bbox, outline=cyan, width=2)
    draw.text((bbox[0] + 2, max(1, bbox[1] - 13)), "TARGET", fill=cyan, font=small_font,
              stroke_width=2, stroke_fill=(0, 0, 0, 230))
    object_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    bar_center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    draw.line((object_center, bar_center), fill=(30, 235, 255, 190), width=1)

    draw.line((start, end), fill=(0, 0, 0, 230), width=7)
    draw.line((start, end), fill=yellow, width=3)
    if orientation == "horizontal":
        ticks = (start[0], (start[0] + end[0]) // 2, end[0])
        for x in ticks:
            draw.line((x, start[1] - 5, x, start[1] + 5), fill=(0, 0, 0, 230), width=5)
            draw.line((x, start[1] - 5, x, start[1] + 5), fill=yellow, width=2)
        label_xy = (start[0], max(1, start[1] - 17))
    else:
        ticks = (start[1], (start[1] + end[1]) // 2, end[1])
        for y in ticks:
            draw.line((start[0] - 5, y, start[0] + 5, y), fill=(0, 0, 0, 230), width=5)
            draw.line((start[0] - 5, y, start[0] + 5, y), fill=yellow, width=2)
        label_xy = (min(image.width - 95, start[0] + 7), max(1, start[1]))

    unit_label = f"{units:g} scene unit" + ("s" if units != 1 else "")
    text_box = draw.textbbox(label_xy, unit_label, font=label_font, stroke_width=1)
    draw.rounded_rectangle(
        (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
        radius=3,
        fill=(0, 0, 0, 190),
    )
    draw.text(label_xy, unit_label, fill=(255, 255, 255, 255), font=label_font)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main() -> None:
    os.environ["MOVI_MOTION_DERIVED_NUM_FRAMES"] = "12"
    os.environ["MOVI_MOTION_NUMERIC_SCALE_GUIDE"] = "0"
    from lmms_eval.tasks.kubric_movi_a_motion import utils

    dataset = Dataset.from_parquet(str(DATASET))
    selected: list[dict] = []
    seen_targets: set[str] = set()
    for item in dataset:
        doc = dict(item)
        if str(doc["subtask"]) not in NUMERIC:
            continue
        shape = str(doc.get("target_description", "object")).split()[-1].lower()
        target_key = str(doc.get("target_description", "object")).lower()
        if target_key in seen_targets:
            continue
        derived = utils._derive_frame_limited_doc(doc, 12)
        indices = utils.frame_indices(derived)
        rows = {int(row["frame_index"]): row for row in utils._frames(derived)}
        viable = [
            index for index in indices
            if index in rows and rows[index].get("bbox_2d_normalized") and int(rows[index].get("visibility_pixels", 0)) >= 200
        ]
        if not viable:
            continue
        # Prefer the frame with the largest visible target footprint.
        frame_index = max(
            viable,
            key=lambda idx: (
                float(rows[idx]["bbox_2d_normalized"][2]) - float(rows[idx]["bbox_2d_normalized"][0])
            ) * (
                float(rows[idx]["bbox_2d_normalized"][3]) - float(rows[idx]["bbox_2d_normalized"][1])
            ),
        )
        selected.append({"doc": derived, "frame_index": frame_index, "shape": shape})
        seen_targets.add(target_key)
        if len(selected) == 12:
            break

    if len(selected) < 4:
        raise RuntimeError(f"Only found {len(selected)} viable distinct targets")

    # Guarantee geometric diversity before filling the remaining rows with
    # additional appearances.
    diverse: list[dict] = []
    used_shapes: set[str] = set()
    for example in selected:
        if example["shape"] not in used_shapes:
            diverse.append(example)
            used_shapes.add(example["shape"])
    for example in selected:
        if example not in diverse:
            diverse.append(example)
        if len(diverse) == 6:
            break
    selected = diverse[:6]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tile, caption_h = 256, 48
    sheet = Image.new("RGB", (tile * 3, (tile + caption_h) * len(selected)), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    heading_font, caption_font = font(13, True), font(10)
    records = []

    for row_index, example in enumerate(selected):
        doc, frame_index = example["doc"], example["frame_index"]
        path_by_index = dict(zip(utils.frame_indices(doc), utils.frame_paths(doc)))
        with Image.open(path_by_index[frame_index]) as opened:
            raw = opened.convert("RGB")
        corner = utils._draw_metric_scale_guide(doc, raw, frame_index)
        anchored = draw_object_anchored_scale(utils, doc, raw, frame_index)
        y = row_index * (tile + caption_h)
        for column, image in enumerate((raw, corner, anchored)):
            sheet.paste(image, (column * tile, y))
        titles = ("Raw frame", "Current corner guide", "Proposed target-anchored guide")
        for column, title in enumerate(titles):
            sheet_draw.text((column * tile + 5, y + tile + 3), title, fill="black", font=heading_font)
        detail = f"{doc['target_description']} · frame {frame_index} · {doc['subtask']}"
        sheet_draw.text((5, y + tile + 23), detail, fill=(60, 70, 85), font=caption_font)

        image_path = OUTPUT / f"example_{row_index + 1}_{example['shape']}.png"
        anchored.save(image_path)
        units, pixels = utils._scale_bar_geometry(doc, frame_index, raw.width)
        records.append({
            "qid": doc["qid"],
            "sequence_id": doc["sequence_id"],
            "target": doc["target_description"],
            "subtask": doc["subtask"],
            "frame_index": frame_index,
            "ruler_scene_units": units,
            "ruler_pixels": pixels,
            "image": image_path.name,
        })

    sheet_path = OUTPUT / "comparison_sheet.png"
    sheet.save(sheet_path)
    (OUTPUT / "examples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Object-anchored scale-guide prototype\n\n"
        "Each row compares the raw source frame, the current corner ruler, and a proposed ruler "
        "placed outside the target's projected bounding box and parallel to its longer projected side. "
        "The cyan box and leader identify the object whose center depth defines the scale.\n\n"
        "![Comparison](comparison_sheet.png)\n",
        encoding="utf-8",
    )
    print(f"Wrote {sheet_path}")
    print(f"Wrote {len(records)} individual examples")


if __name__ == "__main__":
    main()
