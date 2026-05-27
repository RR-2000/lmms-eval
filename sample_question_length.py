#!/usr/bin/env python3
"""Histogram question lengths and sample representative artifacts per bin.

Scans a root folder for JSON result files (recursively), computes question length
(word count), builds 10 equal-width bins, and copies 10 representative samples
per bin into a sibling output folder.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path("/home/ramanathan/VLM/lmms-eval/experiment_artifacts_MMSI/qwen3_vl_experiments")
OUT_DIR = ROOT.parent / "sampled_by_question_length"
NUM_BINS = 10
SAMPLES_PER_BIN = 10
QUESTION_KEY = "question_text"


@dataclass(frozen=True)
class Item:
    json_path: Path
    length: int


def _iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def _question_length_words(text: str) -> int:
    return len((text or "").split())


def _load_items(root: Path) -> List[Item]:
    items: List[Item] = []
    for json_path in _iter_json_files(root):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        q = data.get(QUESTION_KEY)
        if not isinstance(q, str) or not q.strip():
            continue
        length = _question_length_words(q)
        items.append(Item(json_path=json_path, length=length))
    return items


def _bin_edges(lengths: List[int], num_bins: int) -> List[Tuple[int, int]]:
    lo = min(lengths)
    hi = max(lengths)
    if lo == hi:
        return [(lo, hi)] * num_bins

    width = int(math.ceil((hi - lo + 1) / float(num_bins)))
    edges = []
    start = lo
    for _ in range(num_bins):
        end = start + width - 1
        edges.append((start, end))
        start = end + 1
    # Ensure last bin captures max
    edges[-1] = (edges[-1][0], max(edges[-1][1], hi))
    return edges


def _assign_bin(value: int, edges: List[Tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(edges):
        if lo <= value <= hi:
            return i
    return max(0, len(edges) - 1)


def _representative_samples(items: List[Item], k: int) -> List[Item]:
    if not items:
        return []
    if len(items) <= k:
        return items
    items_sorted = sorted(items, key=lambda it: (it.length, str(it.json_path)))
    step = (len(items_sorted) - 1) / float(k - 1)
    picks = []
    for i in range(k):
        idx = int(round(i * step))
        picks.append(items_sorted[idx])
    return picks


def _copy_artifacts(json_path: Path, root: Path, out_root: Path) -> None:
    # Copy all files sharing the same base name (json + png/mp4/npz/etc.)
    base = json_path.with_suffix("")
    rel_dir = json_path.parent.relative_to(root)
    dest_dir = out_root / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    for path in json_path.parent.glob(base.name + ".*"):
        if path.is_file():
            dest_path = dest_dir / path.name
            shutil.copy2(path, dest_path)

    for path in json_path.parent.glob(base.name + "*.png"):
        if path.is_file():
            dest_path = dest_dir / path.name
            shutil.copy2(path, dest_path)


def main() -> int:
    items = _load_items(ROOT)
    if not items:
        print(f"No JSON files with '{QUESTION_KEY}' found under {ROOT}")
        return 1

    lengths = [it.length for it in items]
    edges = _bin_edges(lengths, NUM_BINS)

    # Bucket items
    buckets: List[List[Item]] = [[] for _ in range(NUM_BINS)]
    for it in items:
        bi = _assign_bin(it.length, edges)
        buckets[bi].append(it)

    # Print histogram
    print("Histogram (question word count):")
    for i, (lo, hi) in enumerate(edges):
        print(f"  bin {i:02d} [{lo}, {hi}]: {len(buckets[i])}")

    # Copy samples
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        lo, hi = edges[i]
        bin_dir = OUT_DIR / f"bin_{i:02d}_{lo}-{hi}"
        samples = _representative_samples(bucket, SAMPLES_PER_BIN)
        for it in samples:
            _copy_artifacts(it.json_path, ROOT, bin_dir)

    print(f"Copied samples to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
