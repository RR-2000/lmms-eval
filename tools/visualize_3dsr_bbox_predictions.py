#!/usr/bin/env python3
"""Visualize 3DSR bbox predictions by drawing model outputs on source images."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw, ImageFont

DEFAULT_INPUT = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_4B_bbox_pred/submissions/3dsrbench_bbox_predictions.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "visualized_bboxes"
DEFAULT_IMAGE_CACHE_DIR = DEFAULT_OUTPUT_DIR / "_image_cache"
DEFAULT_DATASET_JSONL = Path("/home/ramanathan/data/3DSR/dataset.jsonl")

BOX_COLOR_CYCLE = [
    "#E63946",
    "#1D3557",
    "#2A9D8F",
    "#F4A261",
    "#6A4C93",
    "#E76F51",
    "#457B9D",
    "#8D99AE",
]


@dataclass
class ParsedBox:
    bbox: tuple[int, int, int, int]
    label: str


@dataclass
class IoUMatch:
    label: str
    pred_bbox: tuple[int, int, int, int]
    gt_bbox: tuple[int, int, int, int]
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a 3DSR prediction JSON file, parse bbox predictions from "
            "model_answer, download each source image, and save annotated images."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the prediction JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotated images will be written.",
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=DEFAULT_IMAGE_CACHE_DIR,
        help="Directory to cache downloaded source images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this zero-based record index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotated output images.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force redownloading source images even if cached.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for image downloads.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=DEFAULT_DATASET_JSONL,
        help="Dataset JSONL used to load GT bboxes for IoU calculation.",
    )
    parser.add_argument(
        "--skip-iou",
        action="store_true",
        help="Skip loading dataset JSONL and computing IoU metrics.",
    )
    parser.add_argument(
        "--iou-report",
        type=Path,
        default=None,
        help="Optional path to write an IoU JSON report.",
    )
    return parser.parse_args()


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def normalize_bbox(values: list[Any]) -> tuple[int, int, int, int] | None:
    if len(values) != 4:
        return None

    try:
        coords = [int(round(float(value))) for value in values]
    except (TypeError, ValueError):
        return None

    coords = [max(0, min(1000, coord)) for coord in coords]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_named_boxes_from_python_object(obj: Any) -> list[ParsedBox]:
    parsed: list[ParsedBox] = []

    if isinstance(obj, dict):
        bbox_payload = obj.get("bbox_2d") or obj.get("bbox") or obj.get("box")
        label = obj.get("label") or obj.get("name") or obj.get("object") or "bbox_1"
        if isinstance(bbox_payload, (list, tuple)):
            bbox = normalize_bbox(list(bbox_payload))
            if bbox is not None:
                parsed.append(ParsedBox(bbox=bbox, label=str(label)))
        return parsed

    if isinstance(obj, list):
        if all(not isinstance(item, (dict, list, tuple)) for item in obj):
            bbox = normalize_bbox(list(obj))
            if bbox is not None:
                parsed.append(ParsedBox(bbox=bbox, label="bbox_1"))
            return parsed

        for item in obj:
            parsed.extend(extract_named_boxes_from_python_object(item))

    return parsed


def deduplicate_boxes(boxes: list[ParsedBox]) -> list[ParsedBox]:
    seen: set[tuple[tuple[int, int, int, int], str]] = set()
    deduped: list[ParsedBox] = []
    for box in boxes:
        key = (box.bbox, box.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def parse_model_answer(answer: str) -> list[ParsedBox]:
    cleaned = strip_code_fences(answer)

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed_obj = parser(cleaned)
        except Exception:
            continue
        parsed_boxes = deduplicate_boxes(extract_named_boxes_from_python_object(parsed_obj))
        if parsed_boxes:
            return parsed_boxes

    dict_pattern = re.compile(
        r'"bbox_2d"\s*:\s*\[([^\]]+)\].*?"label"\s*:\s*"([^"]+)"',
        re.DOTALL,
    )
    dict_boxes: list[ParsedBox] = []
    for coords_text, label in dict_pattern.findall(cleaned):
        values = [part.strip() for part in coords_text.split(",")]
        bbox = normalize_bbox(values)
        if bbox is not None:
            dict_boxes.append(ParsedBox(bbox=bbox, label=label))
    if dict_boxes:
        return deduplicate_boxes(dict_boxes)

    bracket_boxes: list[ParsedBox] = []
    for index, match in enumerate(re.findall(r"\[([^\[\]]+)\]", cleaned), start=1):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", match)
        if len(numbers) == 4:
            bbox = normalize_bbox(numbers)
            if bbox is not None:
                bracket_boxes.append(ParsedBox(bbox=bbox, label=f"bbox_{index}"))
    if bracket_boxes:
        return deduplicate_boxes(bracket_boxes)

    flat_numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    sequential_boxes: list[ParsedBox] = []
    for index in range(0, len(flat_numbers) // 4):
        chunk = flat_numbers[index * 4 : (index + 1) * 4]
        bbox = normalize_bbox(chunk)
        if bbox is not None:
            sequential_boxes.append(ParsedBox(bbox=bbox, label=f"bbox_{index + 1}"))
    return deduplicate_boxes(sequential_boxes)


def group_boxes_by_label(boxes: list[ParsedBox]) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for box in boxes:
        grouped.setdefault(box.label, []).append(box.bbox)
    return grouped


def normalized_center_box_to_bbox_1000(values: list[Any]) -> tuple[int, int, int, int] | None:
    if len(values) != 4:
        return None

    try:
        center_x, center_y, width, height = [float(value) for value in values]
    except (TypeError, ValueError):
        return None

    if any(not math.isfinite(value) for value in (center_x, center_y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    x1 = (center_x - width / 2) * 1000
    y1 = (center_y - height / 2) * 1000
    x2 = (center_x + width / 2) * 1000
    y2 = (center_y + height / 2) * 1000
    return normalize_bbox([x1, y1, x2, y2])


def maybe_flip_bbox_horizontally(
    bbox: tuple[int, int, int, int], should_flip: bool
) -> tuple[int, int, int, int]:
    if not should_flip:
        return bbox
    x1, y1, x2, y2 = bbox
    return (1000 - x2, y1, 1000 - x1, y2)


def load_dataset_records(dataset_jsonl: Path) -> dict[str, dict[str, Any]]:
    records_by_index: dict[str, dict[str, Any]] = {}
    with dataset_jsonl.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse dataset JSONL {dataset_jsonl} at line {line_number}: {exc}"
                ) from exc

            record_index = record.get("index")
            if isinstance(record_index, str) and record_index:
                records_by_index[record_index] = record
    return records_by_index


def extract_gt_boxes_by_label(record: dict[str, Any], flip_image: bool) -> dict[str, list[tuple[int, int, int, int]]]:
    raw_bboxes = record.get("bboxes_by_item")
    if not isinstance(raw_bboxes, dict):
        return {}

    extracted: dict[str, list[tuple[int, int, int, int]]] = {}
    for label, frame_map in raw_bboxes.items():
        if not isinstance(label, str) or not isinstance(frame_map, dict):
            continue

        label_boxes: list[tuple[int, int, int, int]] = []
        for frame_boxes in frame_map.values():
            if not isinstance(frame_boxes, list):
                continue
            for normalized_box in frame_boxes:
                if not isinstance(normalized_box, list):
                    continue
                bbox = normalized_center_box_to_bbox_1000(normalized_box)
                if bbox is None:
                    continue
                label_boxes.append(maybe_flip_bbox_horizontally(bbox, flip_image))

        if label_boxes:
            extracted[label] = label_boxes

    return extracted


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # print(box_a)
    # print(box_b)
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area_1 = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area_1 = area_a + area_b - intersection_area_1
    

    # Also check for flipped box from pred, take the max of the 2 IoUs
    ax1, ay1, ax2, ay2 = 1000 -ax1, ay1, 1000 - ax2, ay2
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area_2 = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area_2 = area_a + area_b - intersection_area_2
    if union_area_1 <= 0 and union_area_2 <= 0:
        return 0.0

    if intersection_area_1 > intersection_area_2:
        return intersection_area_1 / union_area_1
    else:
        return intersection_area_2 / union_area_2

    return 0.0

def compute_iou_matches(
    predicted_boxes: list[ParsedBox],
    gt_boxes_by_label: dict[str, list[tuple[int, int, int, int]]],
) -> list[IoUMatch]:
    matches: list[IoUMatch] = []
    predicted_by_label = group_boxes_by_label(predicted_boxes)

    for label in sorted(set(predicted_by_label) & set(gt_boxes_by_label)):
        gt_candidates = list(gt_boxes_by_label[label])
        if not gt_candidates:
            continue

        for pred_bbox in predicted_by_label[label]:
            best_index = -1
            best_iou = -1.0
            for candidate_index, gt_bbox in enumerate(gt_candidates):
                current_iou = bbox_iou(pred_bbox, gt_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = candidate_index

            if best_index >= 0:
                best_gt_bbox = gt_candidates.pop(best_index)
                matches.append(
                    IoUMatch(
                        label=label,
                        pred_bbox=pred_bbox,
                        gt_bbox=best_gt_bbox,
                        iou=best_iou,
                    )
                )

    return matches


def scale_bbox_to_pixels(
    bbox_1000: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_1000
    return (
        round(x1 / 1000 * image_width),
        round(y1 / 1000 * image_height),
        round(x2 / 1000 * image_width),
        round(y2 / 1000 * image_height),
    )


def image_name_from_url(image_url: str) -> str:
    path = urlparse(image_url).path
    name = Path(path).name
    return name or "image.jpg"


def sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def download_image(
    image_url: str,
    cache_dir: Path,
    redownload: bool,
    timeout: float,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_name = image_name_from_url(image_url)
    cached_path = cache_dir / sanitize_for_filename(image_name)
    if cached_path.exists() and not redownload:
        return cached_path

    request = Request(
        image_url,
        headers={
            "User-Agent": "lmms-eval-3dsr-bbox-visualizer/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        cached_path.write_bytes(response.read())
    return cached_path


def get_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for drawing bbox visualizations. Install it with `pip install pillow`."
        ) from exc

    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(image: "Image.Image", boxes: list[ParsedBox], flip_image: bool) -> "Image.Image":
    try:
        from PIL import ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for drawing bbox visualizations. Install it with `pip install pillow`."
        ) from exc

    annotated = image.convert("RGB").copy()
    if flip_image:
        annotated = annotated.transpose(Image.FLIP_LEFT_RIGHT)
    draw = ImageDraw.Draw(annotated)
    font = get_font()

    for index, parsed_box in enumerate(boxes):
        color = BOX_COLOR_CYCLE[index % len(BOX_COLOR_CYCLE)]
        x1, y1, x2, y2 = scale_bbox_to_pixels(parsed_box.bbox, annotated.width, annotated.height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

        label = f"{parsed_box.label} [{parsed_box.bbox[0]}, {parsed_box.bbox[1]}, {parsed_box.bbox[2]}, {parsed_box.bbox[3]}]"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left
        label_height = bottom - top

        text_x = x1
        text_y = y1 - label_height - 6
        if text_y < 0:
            text_y = min(annotated.height - label_height - 2, y1 + 4)
        if text_x + label_width + 6 > annotated.width:
            text_x = max(0, annotated.width - label_width - 6)

        draw.rectangle(
            (text_x, text_y, text_x + label_width + 6, text_y + label_height + 4),
            fill=color,
        )
        draw.text((text_x + 3, text_y + 2), label, fill="white", font=font)

    return annotated


def load_image(path: Path) -> "Image.Image":
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for loading images. Install it with `pip install pillow`."
        ) from exc

    with path.open("rb") as handle:
        return Image.open(BytesIO(handle.read())).convert("RGB")


def build_output_path(output_dir: Path, record_index: str, image_url: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = image_name_from_url(image_url)
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix or ".jpg"
    filename = f"{sanitize_for_filename(record_index)}_{sanitize_for_filename(stem)}{suffix}"
    return output_dir / filename


def iter_records(records: list[dict[str, Any]], start: int, limit: int | None) -> list[dict[str, Any]]:
    sliced = records[start:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    image_cache_dir = args.image_cache_dir.resolve()
    dataset_jsonl = args.dataset_jsonl.resolve()
    iou_report_path = args.iou_report.resolve() if args.iou_report is not None else None

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    with input_path.open() as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        print(f"Expected a JSON list in {input_path}", file=sys.stderr)
        return 1

    dataset_records_by_index: dict[str, dict[str, Any]] = {}
    if not args.skip_iou:
        if not dataset_jsonl.is_file():
            print(f"Dataset JSONL not found: {dataset_jsonl}", file=sys.stderr)
            return 1
        dataset_records_by_index = load_dataset_records(dataset_jsonl)

    total = 0
    saved = 0
    skipped = 0
    failed = 0
    iou_record_count = 0
    iou_match_count = 0
    iou_sum = 0.0
    records_missing_dataset = 0
    records_with_no_common_boxes = 0
    iou_report_records: list[dict[str, Any]] = []

    for record in iter_records(records, args.start, args.limit):
        total += 1
        record_index = str(record.get("index", f"record_{args.start + total - 1}"))
        image_url = record.get("image_url")
        model_answer = record.get("model_answer", "")
        flip_image = "flip" in str(record.get("qid", ""))

        if not isinstance(image_url, str) or not image_url:
            print(f"[skip] {record_index}: missing image_url", file=sys.stderr)
            skipped += 1
            continue

        boxes = parse_model_answer(str(model_answer))
        if not boxes:
            print(f"[skip] {record_index}: could not parse any bbox from model_answer", file=sys.stderr)
            skipped += 1
            continue

        output_path = build_output_path(output_dir, record_index, image_url)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {record_index}: output exists at {output_path}", file=sys.stderr)
            skipped += 1
            continue

        try:
            cached_image = download_image(
                image_url=image_url,
                cache_dir=image_cache_dir,
                redownload=args.redownload,
                timeout=args.timeout,
            )
            image = load_image(cached_image)
            annotated = draw_boxes(image, boxes, flip_image)
            annotated.save(output_path)
            saved += 1
            print(f"[saved] {record_index}: {output_path}")
        except Exception as exc:
            failed += 1
            print(f"[error] {record_index}: {exc}", file=sys.stderr)
            continue

        if args.skip_iou:
            continue

        dataset_record = dataset_records_by_index.get(record_index)
        if dataset_record is None:
            records_missing_dataset += 1
            print(f"[iou-skip] {record_index}: dataset record not found", file=sys.stderr)
            continue

        gt_boxes_by_label = extract_gt_boxes_by_label(dataset_record, flip_image=flip_image)
        matches = compute_iou_matches(boxes, gt_boxes_by_label)
        if not matches:
            records_with_no_common_boxes += 1
            print(f"[iou-skip] {record_index}: no common GT/pred bbox labels", file=sys.stderr)
            continue

        iou_record_count += 1
        record_mean_iou = sum(match.iou for match in matches) / len(matches)
        iou_match_count += len(matches)
        iou_sum += sum(match.iou for match in matches)
        print(
            f"[iou] {record_index}: matched={len(matches)} mean_iou={record_mean_iou:.4f}",
            file=sys.stderr,
        )
        iou_report_records.append(
            {
                "index": record_index,
                "qid": record.get("qid"),
                "matched_labels": [match.label for match in matches],
                "mean_iou": record_mean_iou,
                "matches": [
                    {
                        "label": match.label,
                        "pred_bbox": list(match.pred_bbox),
                        "gt_bbox": list(match.gt_bbox),
                        "iou": match.iou,
                    }
                    for match in matches
                ],
            }
        )

    print(
        f"Processed={total} Saved={saved} Skipped={skipped} Failed={failed}",
        file=sys.stderr,
    )
    if not args.skip_iou:
        overall_mean_iou = iou_sum / iou_match_count if iou_match_count else 0.0
        print(
            "IoU "
            f"RecordsWithMatches={iou_record_count} "
            f"MatchedBoxes={iou_match_count} "
            f"MeanIoU={overall_mean_iou:.4f} "
            f"MissingDataset={records_missing_dataset} "
            f"NoCommonBoxes={records_with_no_common_boxes}",
            file=sys.stderr,
        )
        if iou_report_path is not None:
            iou_report_path.parent.mkdir(parents=True, exist_ok=True)
            iou_report_path.write_text(
                json.dumps(
                    {
                        "input": str(input_path),
                        "dataset_jsonl": str(dataset_jsonl),
                        "processed_records": total,
                        "saved_records": saved,
                        "records_with_iou_matches": iou_record_count,
                        "matched_boxes": iou_match_count,
                        "mean_iou": overall_mean_iou,
                        "records_missing_dataset": records_missing_dataset,
                        "records_with_no_common_boxes": records_with_no_common_boxes,
                        "records": iou_report_records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"[iou-report] wrote {iou_report_path}", file=sys.stderr)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
