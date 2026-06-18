#!/usr/bin/env python3
"""Build summary JSON files from 3DSR correct/incorrect artifact folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path(
    "/home/ramanathan/VLM/lmms-eval/experiment_artifacts_3dsr_split/qwen3_4B_GT_1/by_task_type"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a 3DSR experiment artifact directory and write summary JSON "
            "files containing correct and incorrect samples keyed by doc_id."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Experiment directory to scan. This can be a parent folder like "
            "by_task_type or a single task directory containing correct/incorrect "
            "folders."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path for a single task directory. If --root points to a "
            "parent folder of task directories, this option is ignored and each "
            "task gets its own summary JSON in the root folder."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def iter_labelled_json_files(root: Path, label: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.parent.name == label
    )


def iter_unlabelled_json_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*.json")
        if path.is_file() and path.name != "doc_id_summary.json"
    )


def has_task_content(root: Path) -> bool:
    return (
        any((root / label).is_dir() for label in ("correct", "incorrect"))
        or bool(iter_unlabelled_json_files(root))
    )


def find_task_dirs(root: Path) -> list[Path]:
    if has_task_content(root):
        return [root]

    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and has_task_content(path)
    )


def extract_record(data: dict[str, object], json_path: Path) -> tuple[str, dict[str, object]] | None:
    doc_id = data.get("doc_id")
    if doc_id is None:
        print(f"Skipping {json_path}: missing doc_id")
        return None

    record = {
        "answer": data.get("answer"),
        "ground_truth": data.get("ground_truth"),
        "prompt_rendered": data.get("prompt_rendered"),
    }
    return str(doc_id), record


def extract_answer(answer: object) -> str | None:
    
    if not isinstance(answer, str):
        return None

    if ":" in str(answer):
        return answer.rsplit(":", 1)[-1].strip().rsplit(".", 1)[-1].strip()
    return answer.rsplit(".", 1)[-1].strip()


def normalized_ground_truth(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip()


def is_correct_sample(data: dict[str, object]) -> bool | None:
    answer = extract_answer(data.get("answer").strip('.'))
    # if data.get("doc_id") == 0:
    #     print(f"Debug: doc_id=0, raw answer={data.get('answer')}, extracted answer={answer}")
    ground_truth = normalized_ground_truth(data.get("ground_truth"))
    if answer is None or ground_truth is None:
        return None
    if ground_truth.lower().strip('.') in answer.lower() or ground_truth[0] == data.get("answer")[0]:
        return True
    
    # print(f"Debug: doc_id={data.get('doc_id')}, answer='{answer}', raw_answer='{data.get('answer')}', ground_truth='{ground_truth}' - Marking as incorrect: {ground_truth[0]} != {data.get('answer')[0]}")
        
    return False


def load_records(json_paths: list[Path]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"Skipping unreadable JSON {json_path}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"Skipping {json_path}: expected a JSON object")
            continue

        extracted = extract_record(data, json_path)
        if extracted is None:
            continue

        doc_id, record = extracted
        if doc_id in records:
            print(f"Overwriting duplicate doc_id {doc_id} from {json_path}")
        records[doc_id] = record

    return records


def build_summary(task_dir: Path, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return False

    correct_json_paths = iter_labelled_json_files(task_dir, "correct")
    incorrect_json_paths = iter_labelled_json_files(task_dir, "incorrect")

    if correct_json_paths or incorrect_json_paths:
        correct_records = load_records(correct_json_paths)
        incorrect_records = load_records(incorrect_json_paths)
    else:
        correct_records: dict[str, dict[str, object]] = {}
        incorrect_records: dict[str, dict[str, object]] = {}

        for json_path in iter_unlabelled_json_files(task_dir):
            try:
                data = json.loads(json_path.read_text())
            except Exception as exc:
                print(f"Skipping unreadable JSON {json_path}: {exc}")
                continue

            if not isinstance(data, dict):
                print(f"Skipping {json_path}: expected a JSON object")
                continue

            extracted = extract_record(data, json_path)
            if extracted is None:
                continue

            correctness = is_correct_sample(data)
            if correctness is None:
                print(f"Skipping {json_path}: missing string answer or ground_truth")
                continue

            doc_id, record = extracted
            target_records = correct_records if correctness else incorrect_records
            if doc_id in target_records:
                print(f"Overwriting duplicate doc_id {doc_id} from {json_path}")
            target_records[doc_id] = record

    summary = {
        "correct": correct_records,
        "incorrect": incorrect_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote summary: {output_path}")
    print(f"Correct samples: {len(correct_records)}")
    print(f"Incorrect samples: {len(incorrect_records)}")
    return True


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Root directory not found: {root}")
        return 1

    task_dirs = find_task_dirs(root)
    if not task_dirs:
        print(f"No task directories with correct/incorrect folders found under: {root}")
        return 1

    wrote_count = 0

    for task_dir in task_dirs:
        if len(task_dirs) == 1 and args.output is not None:
            output_path = args.output.resolve()
        elif len(task_dirs) == 1:
            output_path = task_dir / "doc_id_summary.json"
        else:
            output_path = root / f"{task_dir.name}_doc_id_summary.json"

        if build_summary(task_dir, output_path, args.overwrite):
            wrote_count += 1

    print(f"Task directories processed: {len(task_dirs)}")
    print(f"Summary files written: {wrote_count}")
    return 0 if wrote_count == len(task_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
