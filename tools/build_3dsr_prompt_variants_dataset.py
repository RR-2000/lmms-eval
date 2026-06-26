#!/usr/bin/env python3
"""
Build a derived 3DSR dataset from a saved submission JSON.

This version pulls the base question from the original 3DSR dataset JSONL and
uses a locally loaded Qwen-VL model through Hugging Face Transformers to
generate natural-language variants of the question and GT help text.


python tools/build_3dsr_prompt_variants_dataset.py \
  outputs/3dsrbench_4B_GT_4_Blank/submissions/3dsrbench_predictions_qwen3_vl_experiments.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --limit-per-category 10 \
  --jsonl-only

"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import time
from typing import Any
from pathlib import Path

DEFAULT_SOURCE_JSONL = Path("/home/ramanathan/data/3DSR/dataset.jsonl")
OPTION_LINE_RE = re.compile(r"^([A-D])\.\s*(.*)$")


def optional_import(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name), True
    except (ImportError, AttributeError):
        return None, False


process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", type=Path, help="Path to the saved submission JSON file.")
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=DEFAULT_SOURCE_JSONL,
        help=f"Original 3DSR dataset JSONL. Default: {DEFAULT_SOURCE_JSONL}",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix without extension. Defaults to '<input>_variants'.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=10,
        help="Number of NL variants to create per source record. Default: 10.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of source records to process.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Optional limit on the number of parent records to process for each category.",
    )
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Write only JSONL. Useful when parquet dependencies are unavailable.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for loading the Qwen-VL model. Default: auto.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen-VL Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum image pixels for AutoProcessor. Default matches repo Qwen config.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1605632,
        help="Maximum image pixels for AutoProcessor. Default matches repo Qwen config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for rewriting. Default: 0.8.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum output tokens per rewrite call. Default: 1200.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for generation when temperature > 0. Default: 0.95.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for each rewrite generation/parse attempt. Default: 3.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional sleep between requests in seconds. Default: 0.",
    )
    return parser.parse_args()


def _resolve_model_class(pretrained: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration, "torch_dtype"

    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration, "dtype"


class QwenVLRewriter:
    def __init__(
        self,
        *,
        model_name: str,
        device_map: str,
        attn_implementation: str | None,
        min_pixels: int,
        max_pixels: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        if not _has_qwen_vl:
            raise ImportError("qwen_vl_utils is required. Install it with `pip install qwen-vl-utils`.")

        model_cls, dtype_key = _resolve_model_class(model_name)
        model_kwargs: dict[str, Any] = {
            dtype_key: "bfloat16",
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = model_cls.from_pretrained(model_name, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.device_map = device_map
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._torch = torch

    def _move_inputs(self, inputs):
        if self.device_map == "auto":
            return inputs.to("cuda" if self._torch.cuda.is_available() else "cpu")
        if isinstance(self.device_map, str) and self.device_map.startswith("cuda"):
            return inputs.to(self.device_map)
        return inputs

    def generate_variants(
        self,
        *,
        image: Image.Image | None,
        original_question: str,
        options_map: dict[str, str],
        gt_answer: str,
        gt_help_text: str,
        variants: int,
    ) -> list[dict[str, str]]:
        messages = build_rewrite_messages(
            image=image,
            original_question=original_question,
            options_map=options_map,
            gt_answer=gt_answer,
            gt_help_text=gt_help_text,
            variants=variants,
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = extract_json_object(output_text)
        variant_items = parsed.get("variants")
        if not isinstance(variant_items, list):
            raise ValueError("Model JSON response did not contain a 'variants' list.")

        cleaned: list[dict[str, str]] = []
        for item in variant_items[:variants]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            help_text = str(item.get("help", "")).strip()
            if question and help_text:
                cleaned.append({"question": question, "help": help_text})
        if len(cleaned) != variants:
            raise ValueError(f"Expected {variants} usable variants, got {len(cleaned)}.")
        return cleaned


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return data


def load_image_from_record(record: dict[str, Any]):
    from PIL import Image

    path_candidates = [
        record.get("resolved_img_path"),
        record.get("img_path"),
        record.get("image"),
    ]
    for image_path in path_candidates:
        if isinstance(image_path, str) and image_path and Path(image_path).is_file():
            return Image.open(image_path).convert("RGB")
    return None


def limit_records_per_category(records: list[dict[str, Any]], per_category_limit: int) -> list[dict[str, Any]]:
    if per_category_limit <= 0:
        raise ValueError("--limit-per-category must be a positive integer")

    category_counts: dict[str, int] = {}
    filtered_records: list[dict[str, Any]] = []
    for record in records:
        category = str(record.get("category", "unknown"))
        current_count = category_counts.get(category, 0)
        if current_count >= per_category_limit:
            continue
        filtered_records.append(record)
        category_counts[category] = current_count + 1

    return filtered_records


def load_source_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_qid: dict[str, dict[str, Any]] = {}
    by_index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_number}: {exc}") from exc
            qid = str(record.get("qid", ""))
            index = str(record.get("index", ""))
            if qid:
                by_qid[qid] = record
            if index:
                by_index[index] = record
    return by_qid, by_index


def resolve_source_record(
    saved_record: dict[str, Any],
    source_by_qid: dict[str, dict[str, Any]],
    source_by_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    qid = str(saved_record.get("qid", ""))
    index = str(saved_record.get("index", ""))
    source = source_by_qid.get(qid) or source_by_index.get(index)
    if source is None:
        raise KeyError(f"Could not find source dataset row for qid={qid!r}, index={index!r}")
    return source


def get_source_original_question(source_record: dict[str, Any]) -> str:
    original_question = str(source_record.get("original_question", "") or "").strip()
    if original_question:
        return original_question
    return str(source_record.get("question", "") or "").strip()


def parse_prompt_parts(question_prompt: str, gt_help_text: str) -> dict[str, Any]:
    prompt = question_prompt.replace("\r\n", "\n").strip("\n")
    question_text = prompt
    options_lines: list[str] = []
    post_prompt = ""

    if prompt.startswith("Question: "):
        prompt = prompt[len("Question: ") :]

    if "\nOptions:\n" in prompt:
        _, remainder = prompt.split("\nOptions:\n", 1)
    else:
        remainder = ""

    remainder_lines = remainder.splitlines()
    first_non_option_idx = len(remainder_lines)
    for idx, line in enumerate(remainder_lines):
        if OPTION_LINE_RE.match(line.strip()):
            options_lines.append(line.strip())
        else:
            first_non_option_idx = idx
            break

    trailing_text = "\n".join(remainder_lines[first_non_option_idx:]).strip("\n")
    help_text = (gt_help_text or "").strip()

    if help_text:
        if trailing_text.startswith(help_text):
            post_prompt = trailing_text[len(help_text) :].lstrip("\n")
        else:
            help_idx = trailing_text.find(help_text)
            if help_idx >= 0:
                post_prompt = (trailing_text[:help_idx] + trailing_text[help_idx + len(help_text) :]).strip("\n")
            else:
                post_prompt = trailing_text
    else:
        post_prompt = trailing_text

    options_map: dict[str, str] = {}
    for line in options_lines:
        match = OPTION_LINE_RE.match(line)
        if match:
            options_map[match.group(1)] = match.group(2)

    return {
        "question": question_text,
        "options": "\n".join(options_lines),
        "options_map": options_map,
        "help": help_text,
        "post_prompt": post_prompt,
    }


def build_question_prompt(question: str, options_lines: str, help_text: str, post_prompt: str) -> str:
    parts = [f"Question: {question}"]
    if options_lines:
        parts.append("Options:")
        parts.append(options_lines)
    if help_text:
        parts.append(help_text)
    if post_prompt:
        parts.append(post_prompt)
    return "\n".join(parts).rstrip() + "\n"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Model response did not contain a valid JSON object.")


def build_rewrite_messages(
    *,
    image,
    original_question: str,
    options_map: dict[str, str],
    gt_answer: str,
    gt_help_text: str,
    variants: int,
) -> list[dict[str, Any]]:
    option_lines = [f"{label}. {text}" for label, text in options_map.items()]
    correct_option_text = options_map.get(gt_answer, "")

    user_text = (
        "Create natural-language rewrites for a multiple-choice spatial reasoning question.\n"
        f"Generate exactly {variants} variants.\n"
        "Requirements:\n"
        "- Preserve the original meaning exactly.\n"
        "- Keep the correct answer unchanged.\n"
        "- Do not introduce new objects, relations, or ambiguity.\n"
        "- Rewrite only in natural language.\n"
        "- The help text must remain a semantically equivalent hint.\n"
        "- Keep each help text concise, ideally one sentence.\n"
        "- Do not mention the answer letter in the rewritten question or help unless the original help already implies it naturally.\n"
        "- Return JSON only with the shape {\"variants\": [{\"question\": \"...\", \"help\": \"...\"}, ...]}.\n\n"
        f"Original question: {original_question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n"
        f"Correct answer letter: {gt_answer}\n"
        f"Correct answer text: {correct_option_text}\n"
        f"Original help text: {gt_help_text}\n"
    )

    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_text})

    return [
        {
            "role": "system",
            "content": (
                "You are a careful data augmentation assistant for visual question answering datasets. "
                "You preserve semantics exactly and follow JSON output instructions strictly."
            ),
        },
        {"role": "user", "content": content},
    ]


def generate_variants_with_model(
    *,
    rewriter: QwenVLRewriter,
    image,
    original_question: str,
    options_map: dict[str, str],
    gt_answer: str,
    gt_help_text: str,
    variants: int,
    retries: int,
) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return rewriter.generate_variants(
                image=image,
                original_question=original_question,
                options_map=options_map,
                gt_answer=gt_answer,
                gt_help_text=gt_help_text,
                variants=variants,
            )
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(5.0 * attempt, 15.0))

    raise RuntimeError(f"Failed to generate variants after {retries} attempts: {last_error}") from last_error


def build_variant_record(
    *,
    saved_record: dict[str, Any],
    source_record: dict[str, Any],
    variant_idx: int,
    variant_payload: dict[str, str],
) -> dict[str, Any]:
    parent_index = str(saved_record.get("index", ""))
    parent_qid = str(saved_record.get("qid", parent_index))
    gt_help_text = str(saved_record.get("gt_help_text", "") or "")
    parsed_prompt = parse_prompt_parts(str(saved_record.get("question_prompt", "")), gt_help_text)

    variant_suffix = f"v{variant_idx:02d}"
    variant_index = f"{parent_index}__{variant_suffix}"
    variant_qid = f"{parent_qid}__{variant_suffix}"

    options_map = parsed_prompt["options_map"]
    question_variant = variant_payload["question"]
    help_variant = variant_payload["help"]
    question_prompt = build_question_prompt(
        question=question_variant,
        options_lines=parsed_prompt["options"],
        help_text=help_variant,
        post_prompt=parsed_prompt["post_prompt"],
    )

    variant_record = {
        "parent_index": parent_index,
        "parent_qid": parent_qid,
        "variant_id": variant_idx,
        "index": variant_index,
        "qid": variant_qid,
        "question_prompt": question_prompt,
        "question": question_variant,
        "original_question": source_record.get("original_question", source_record.get("question")),
        "source_question": source_record.get("question"),
        "options": parsed_prompt["options"],
        "option_A": options_map.get("A"),
        "option_B": options_map.get("B"),
        "option_C": options_map.get("C"),
        "option_D": options_map.get("D"),
        "help": help_variant,
        "post_prompt": parsed_prompt["post_prompt"],
        "image_url": saved_record.get("image_url", source_record.get("image_url")),
        "gt_answer": saved_record.get("gt_answer", source_record.get("answer")),
        "gt_help_text": help_variant,
        "original_gt_help_text": gt_help_text,
        "category": saved_record.get("category", source_record.get("category")),
        "main_category": saved_record.get("main_category"),
    }

    # Carry through every field from the parent saved record so each variant row
    # remains fully traceable back to its source submission payload.
    for key, value in saved_record.items():
        variant_record[f"parent_{key}"] = value

    return variant_record


def build_dataset(
    *,
    saved_records: list[dict[str, Any]],
    source_by_qid: dict[str, dict[str, Any]],
    source_by_index: dict[str, dict[str, Any]],
    rewriter: QwenVLRewriter,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.variants <= 0:
        raise ValueError("--variants must be a positive integer")

    output_records: list[dict[str, Any]] = []
    for record_idx, saved_record in enumerate(saved_records, start=1):
        source_record = resolve_source_record(saved_record, source_by_qid, source_by_index)
        image = load_image_from_record(source_record)
        parsed_prompt = parse_prompt_parts(
            str(saved_record.get("question_prompt", "")),
            str(saved_record.get("gt_help_text", "") or ""),
        )
        
        variants = generate_variants_with_model(
            rewriter=rewriter,
            image=image,
            original_question=get_source_original_question(source_record),
            options_map=parsed_prompt["options_map"],
            gt_answer=str(saved_record.get("gt_answer", source_record.get("answer", ""))),
            gt_help_text=str(saved_record.get("gt_help_text", "")),
            variants=args.variants,
            retries=args.retries,
        )

        for variant_idx, variant_payload in enumerate(variants):
            output_records.append(
                build_variant_record(
                    saved_record=saved_record,
                    source_record=source_record,
                    variant_idx=variant_idx,
                    variant_payload=variant_payload,
                )
            )

        print(
            f"Processed {record_idx}/{len(saved_records)} parent records "
            f"({saved_record.get('qid', saved_record.get('index', 'unknown'))})"
        )
        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

    return output_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet export requires pandas with a parquet backend (for example pyarrow). "
            "The JSONL output was still written successfully."
        ) from exc

    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> None:
    args = parse_args()

    input_path = args.input_json.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    source_jsonl = args.source_jsonl.resolve()
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"Source JSONL not found: {source_jsonl}")

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = input_path.with_suffix("")
        output_prefix = output_prefix.with_name(output_prefix.name + "_variants")
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    saved_records = load_records(input_path)
    if args.limit_per_category is not None:
        saved_records = limit_records_per_category(saved_records, args.limit_per_category)
    if args.limit is not None:
        saved_records = saved_records[: args.limit]

    source_by_qid, source_by_index = load_source_records(source_jsonl)
    rewriter = QwenVLRewriter(
        model_name=args.model,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    variant_records = build_dataset(
        saved_records=saved_records,
        source_by_qid=source_by_qid,
        source_by_index=source_by_index,
        rewriter=rewriter,
        args=args,
    )

    jsonl_path = output_prefix.with_suffix(".jsonl")
    parquet_path = output_prefix.with_suffix(".parquet")

    write_jsonl(jsonl_path, variant_records)
    print(f"Loaded {len(saved_records)} source records from {input_path}")
    print(f"Wrote {len(variant_records)} variant records to {jsonl_path}")

    if args.jsonl_only:
        return

    write_parquet(parquet_path, variant_records)
    print(f"Wrote {len(variant_records)} variant records to {parquet_path}")


if __name__ == "__main__":
    main()
