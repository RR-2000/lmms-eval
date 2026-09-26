#!/usr/bin/env python3
"""Generate visible-object padding for the 3DSR direction/object task.

Qwen3-VL-8B-Instruct is run once for each relevant sparse scene. The resulting
JSON is an auditable, resumable offline artifact; evaluation never invokes the
generator model.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATASET = Path("/home/ramanathan/data/3DSR/dataset.parquet")
DEFAULT_OUTPUT = Path(
    "/home/ramanathan/data/3DSR/3dsrbench_qwen3vl_8b_scene_objects.json"
)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def same_entity(left: str, right: str) -> bool:
    left, right = normalize(left), normalize(right)
    return bool(left and right and (left == right or left in right or right in left))


def image_key(row: dict[str, Any]) -> str:
    return str(
        row.get("image_name")
        or row.get("image_url")
        or row.get("resolved_img_path")
        or row.get("img_path")
        or row.get("qid", row.get("index", ""))
    )


def row_terms(row: dict[str, Any]) -> list[str]:
    values = [row.get("subject"), row.get("object1"), row.get("object_1")]
    bbox_items = row.get("bbox_items")
    if isinstance(bbox_items, str) and bbox_items.strip().startswith("["):
        try:
            bbox_items = json.loads(bbox_items)
        except json.JSONDecodeError:
            bbox_items = []
    if isinstance(bbox_items, list):
        values.extend(bbox_items)
    result = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def resolve_image(row: dict[str, Any]) -> Path:
    for field in ("resolved_img_path", "img_path", "image"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend(
                [
                    Path("/home/ramanathan/data/3dsrbench_data") / path,
                    Path("/home/ramanathan/data/3dsrbench_data/images/coco_images")
                    / Path(*path.parts[-2:]),
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image for {image_key(row)!r}")


def sparse_scenes(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    terms_by_image: dict[str, list[str]] = defaultdict(list)
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = image_key(row)
        rows_by_image[key].append(row)
        for term in row_terms(row):
            if term not in terms_by_image[key]:
                terms_by_image[key].append(term)

    result = {}
    for row in records:
        if normalize(row.get("qtype")) != "multi_object" or normalize(
            row.get("relation")
        ) != "viewpoint towards object":
            continue
        key = image_key(row)
        subject = str(row.get("subject") or "").strip()
        target = str(row.get("object1") or row.get("object_1") or "").strip()
        options = [target]
        for term in terms_by_image[key]:
            if same_entity(term, subject) or any(same_entity(term, item) for item in options):
                continue
            options.append(term)
        if len(options) < 4:
            scene = result.setdefault(
                key,
                {
                    "image_path": str(resolve_image(row)),
                    "known_objects": terms_by_image[key],
                    "required_padding": 0,
                    "source_qids": [],
                },
            )
            scene["required_padding"] = max(scene["required_padding"], 4 - len(options))
            scene["source_qids"].append(str(row.get("qid", row.get("index", ""))))
    return result


def extract_objects(text: str) -> list[str]:
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for start, end in (("{", "}"), ("[", "]")):
        left, right = text.find(start), text.rfind(end)
        if left >= 0 and right > left:
            candidates.append(text[left : right + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        values = payload.get("objects") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            continue
        result = []
        for value in values:
            value = " ".join(str(value or "").strip().split())
            if value and value.casefold() not in {item.casefold() for item in result}:
                result.append(value)
        if result:
            return result
    # Qwen sometimes emits a valid list prefix, then repeats objects until the
    # token limit cuts off the final JSON delimiters. Recover only complete JSON
    # string literals from that list prefix; an incomplete trailing string is
    # deliberately ignored.
    prefix = re.search(r'"objects"\s*:\s*\[', text)
    if prefix:
        result = []
        for match in re.finditer(r'"(?:\\.|[^"\\])*"', text[prefix.end() :]):
            value = " ".join(str(json.loads(match.group(0))).strip().split())
            if value and value.casefold() not in {item.casefold() for item in result}:
                result.append(value)
        if result:
            return result
    raise ValueError("Model response did not contain a JSON object list")


class QwenSceneObjectGenerator:
    def __init__(self, args: argparse.Namespace):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        kwargs: dict[str, Any] = {
            "dtype": "bfloat16",
            "device_map": args.device_map,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.process_vision_info = process_vision_info
        self.torch = torch
        self.args = args

    def generate(
        self,
        image_path: str,
        known_objects: list[str],
        required_padding: int,
        attempt: int,
    ) -> tuple[list[str], str]:
        prompt = (
            "Identify distinct physical objects that are clearly visible in this image. "
            "Return concise everyday noun phrases, not colors, materials, directions, or spatial relations. "
            "Include small objects, background structures, fixtures, vegetation, vehicles, furniture, "
            "and surfaces such as a road or sidewalk when visible. "
            "Do not repeat or rename any known object listed below. Return JSON only as "
            '{"objects": ["object 1", "object 2", ...]}. '
            f"You MUST find at least {required_padding} objects outside the known list; list exactly 12 "
            "when possible and never repeat an object. "
            "Inspect the foreground, middle ground, and background before answering.\n"
            f"Known objects to exclude: {json.dumps(known_objects, ensure_ascii=False)}\n"
            f"Retry number: {attempt}. On retries, look for additional background and small objects."
        )
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = self.process_vision_info([messages])
        inputs = self.processor(
            text=[rendered], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        device = "cuda" if self.args.device_map == "auto" else self.args.device_map
        inputs = inputs.to(device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output)]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            return extract_objects(raw), raw
        except ValueError as exc:
            raise ValueError(f"{exc}; raw response={raw!r}") from exc


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def recover_truncated_failures(payload: dict[str, Any]) -> int:
    recovered = 0
    for key, failure in list(payload.get("failures", {}).items()):
        error = str(failure.get("error", ""))
        marker = "raw response="
        if marker not in error:
            continue
        try:
            raw = ast.literal_eval(error.split(marker, 1)[1])
            objects = extract_objects(raw)
        except (SyntaxError, ValueError):
            continue
        known = failure.get("known_objects", [])
        usable = [
            value for value in objects
            if not any(same_entity(value, item) for item in known)
        ]
        if len(usable) < int(failure.get("required_padding", 0)):
            continue
        payload["images"][key] = {
            **{field: value for field, value in failure.items() if field not in {"error", "attempts"}},
            "objects": usable,
            "raw_response": raw,
            "attempts": failure.get("attempts", 1),
            "recovered_from_truncated_json": True,
        }
        del payload["failures"][key]
        recovered += 1
    return recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Report sparse scenes without loading Qwen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = pd.read_parquet(args.dataset).to_dict(orient="records")
    scenes = sparse_scenes(records)
    print(f"Found {len(scenes)} sparse scenes among {len(records)} rows.", flush=True)
    if args.inspect:
        print(json.dumps(scenes, indent=2, ensure_ascii=False))
        return

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "images": {},
        "failures": {},
    }
    if args.output.is_file() and not args.overwrite:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("model") != args.model:
            raise ValueError("Existing manifest model differs; pass --overwrite to replace it")
        recovered = recover_truncated_failures(payload)
        if recovered:
            write_manifest(args.output, payload)
            print(f"Recovered {recovered} truncated JSON responses from the checkpoint.", flush=True)

    generator = QwenSceneObjectGenerator(args)
    pending = [(key, scene) for key, scene in scenes.items() if key not in payload["images"]]
    if args.limit is not None:
        pending = pending[: args.limit]
    failures = []
    for ordinal, (key, scene) in enumerate(pending, start=1):
        error = None
        for attempt in range(1, args.retries + 1):
            try:
                objects, raw = generator.generate(
                    scene["image_path"],
                    scene["known_objects"],
                    scene["required_padding"],
                    attempt,
                )
                usable = [
                    value for value in objects
                    if not any(same_entity(value, known) for known in scene["known_objects"])
                ]
                if len(usable) < scene["required_padding"]:
                    raise ValueError(
                        f"needed {scene['required_padding']} novel objects, got {usable!r}; "
                        f"raw response={raw!r}"
                    )
                payload["images"][key] = {
                    **scene, "objects": usable, "raw_response": raw, "attempts": attempt,
                }
                payload.setdefault("failures", {}).pop(key, None)
                write_manifest(args.output, payload)
                print(f"[{ordinal}/{len(pending)}] {key}: {usable}", flush=True)
                error = None
                break
            except Exception as exc:  # preserve per-scene progress on model/parse failures
                error = exc
                print(f"[{ordinal}/{len(pending)}] {key} attempt {attempt}: {exc}", flush=True)
        if error is not None:
            failures.append((key, str(error)))
            payload.setdefault("failures", {})[key] = {
                **scene, "error": str(error), "attempts": args.retries,
            }
            write_manifest(args.output, payload)
    if failures:
        raise RuntimeError(f"Failed to generate {len(failures)} scenes: {failures}")
    print(f"Wrote {len(payload['images'])} scene catalogs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
