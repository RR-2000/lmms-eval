#!/usr/bin/env python3
"""Build and audit the GQA scene-graph direction/object diagnostic.

GQA scene graphs map image IDs to objects.  Each object's ``relations`` list
describes that subject object relative to another object ID.  This script turns
every unambiguous supported relation into a matched direction/object pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


DIRECTIONS = ("left", "right", "above", "below")
OPTION_LETTERS = tuple("ABCD")
RELATION_ALIASES = {
    "left": "left",
    "left of": "left",
    "to the left of": "left",
    "on the left of": "left",
    "right": "right",
    "right of": "right",
    "to the right of": "right",
    "on the right of": "right",
    "above": "above",
    "above of": "above",
    "over": "above",
    "on top of": "above",
    "below": "below",
    "below of": "below",
    "under": "below",
    "underneath": "below",
}
SKIP_REASONS = (
    "missing_scene_graph",
    "unsupported_relation",
    "duplicate_object_names",
    "fewer_than_4_candidates",
    "ambiguous_relation",
    "invalid_image",
    "malformed_annotation",
)


def normalize_relation(value: object) -> str | None:
    normalized = " ".join(str(value or "").strip().lower().split())
    return RELATION_ALIASES.get(normalized)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_scene_graphs(paths: Iterable[Path], gqa_split: str | None = None) -> dict:
    merged = {}
    for path in paths:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith("scenegraphs.json")
                ]
                split_members = [
                    name
                    for name in members
                    if Path(name).name.lower().startswith(f"{gqa_split}_")
                ] if gqa_split else []
                if split_members:
                    members = split_members
                if not members:
                    raise ValueError(f"No *sceneGraphs.json member found in {path}")
                for member in members:
                    with archive.open(member) as handle:
                        payload = json.load(handle)
                    if not isinstance(payload, dict):
                        raise ValueError(f"Expected a JSON object in {path}:{member}")
                    merged.update(payload)
        else:
            merged.update(_load_json(path))
    return merged


def _read_id_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
            if isinstance(value, dict):
                value = value.keys()
            return {str(item) for item in value}
        return {line.strip() for line in handle if line.strip()}


def _stable_rng(source_relation_id: str, seed: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{source_relation_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _clean_name(obj: Mapping) -> str:
    return " ".join(str(obj.get("name", "")).strip().split())


def _objects_for_scene(scene: Mapping) -> dict[str, Mapping]:
    objects = scene.get("objects")
    if not isinstance(objects, Mapping):
        return {}
    return {
        str(object_id): obj
        for object_id, obj in objects.items()
        if isinstance(obj, Mapping)
    }


def _relation_index(objects: Mapping[str, Mapping], stats: Counter):
    """Return valid (subject, anchor, relation, ordinal) rows and ambiguity map."""
    rows = []
    targets_by_anchor_relation = defaultdict(set)
    for target_id, target_obj in objects.items():
        relations = target_obj.get("relations", [])
        if not isinstance(relations, list):
            stats["malformed_annotation"] += 1
            continue
        for ordinal, relation_obj in enumerate(relations):
            if not isinstance(relation_obj, Mapping):
                stats["malformed_annotation"] += 1
                continue
            relation = normalize_relation(relation_obj.get("name"))
            if relation is None:
                stats["unsupported_relation"] += 1
                continue
            anchor_id = str(relation_obj.get("object", ""))
            if not anchor_id or anchor_id not in objects or anchor_id == target_id:
                stats["malformed_annotation"] += 1
                continue
            rows.append((target_id, anchor_id, relation, ordinal))
            targets_by_anchor_relation[(anchor_id, relation)].add(target_id)
    return rows, targets_by_anchor_relation


def build_records(
    scene_graphs: Mapping[str, Mapping],
    *,
    gqa_split: str = "val",
    seed: str = "gqa_direction_object_v1",
    limit_pairs: int | None = None,
    requested_image_ids: set[str] | None = None,
    valid_image_ids: set[str] | None = None,
    image_dir: Path | None = None,
    require_images: bool = False,
) -> tuple[list[dict], dict]:
    stats = Counter({reason: 0 for reason in SKIP_REASONS})
    records = []

    if requested_image_ids is not None:
        stats["missing_scene_graph"] += len(
            requested_image_ids - {str(key) for key in scene_graphs}
        )
        image_ids = sorted(requested_image_ids & {str(key) for key in scene_graphs})
    else:
        image_ids = sorted(str(key) for key in scene_graphs)

    stop = False
    for image_id in image_ids:
        if valid_image_ids is not None and image_id not in valid_image_ids:
            stats["invalid_image"] += 1
            continue
        image_path = (image_dir / f"{image_id}.jpg") if image_dir else Path(f"{image_id}.jpg")
        scene = scene_graphs.get(image_id)
        if not isinstance(scene, Mapping):
            stats["malformed_annotation"] += 1
            continue
        objects = _objects_for_scene(scene)
        if not objects:
            stats["malformed_annotation"] += 1
            continue

        names = {object_id: _clean_name(obj) for object_id, obj in objects.items()}
        name_counts = Counter(name.casefold() for name in names.values() if name)
        relation_rows, targets_by_anchor_relation = _relation_index(objects, stats)

        for target_id, anchor_id, relation, ordinal in relation_rows:
            target = names[target_id]
            anchor = names[anchor_id]
            if not target or not anchor:
                stats["malformed_annotation"] += 1
                continue
            # A bare name cannot identify an instance if the same name occurs
            # more than once in the image.  Skip rather than silently attaching
            # object IDs to only one answer format.
            if name_counts[target.casefold()] != 1 or name_counts[anchor.casefold()] != 1:
                stats["duplicate_object_names"] += 1
                continue
            distractor_names = []
            seen = {target.casefold(), anchor.casefold()}
            for object_id in sorted(objects):
                name = names[object_id]
                key = name.casefold()
                if not name or key in seen or name_counts[key] != 1:
                    continue
                seen.add(key)
                distractor_names.append(name)
            if len(distractor_names) < 3:
                stats["fewer_than_4_candidates"] += 1
                continue
            source_relation_id = f"{image_id}:{target_id}:{ordinal}:{anchor_id}"
            rng = _stable_rng(source_relation_id, seed)
            distractors = rng.sample(distractor_names, 3)
            object_options = [target, *distractors]
            rng.shuffle(object_options)
            name_to_object_id = {
                name.casefold(): object_id
                for object_id, name in names.items()
                if name and name_counts[name.casefold()] == 1
            }
            relation_target_ids = targets_by_anchor_relation[(anchor_id, relation)]
            correct_option_ids = [
                name_to_object_id[option.casefold()]
                for option in object_options
                if name_to_object_id[option.casefold()] in relation_target_ids
            ]
            if len(correct_option_ids) != 1:
                stats["ambiguous_relation"] += 1
                continue
            if require_images and not image_path.is_file():
                stats["invalid_image"] += 1
                continue
            object_answer_idx = object_options.index(target)
            candidate_pool = list(object_options)

            resolved_image_path = str(image_path.resolve())
            common = {
                "image": resolved_image_path,
                "image_path": resolved_image_path,
                "image_id": image_id,
                "gqa_split": gqa_split,
                "source_qid": source_relation_id,
                "source_relation_id": source_relation_id,
                "diagnostic_anchor": anchor,
                "diagnostic_anchor_object_id": anchor_id,
                "diagnostic_target_object": target,
                "diagnostic_target_object_id": target_id,
                "diagnostic_relation": relation,
                "diagnostic_relation_target_object_ids": sorted(
                    relation_target_ids
                ),
                "diagnostic_correct_option_object_ids": correct_option_ids,
                "diagnostic_correct_object_options": [target],
                "num_correct_object_options": len(correct_option_ids),
                "candidate_pool": candidate_pool,
                "task_family": "gqa_direction_object",
                "relation_source": "scene_graph",
                "num_options": 4,
            }

            direction_answer_idx = DIRECTIONS.index(relation)
            records.append(
                {
                    **common,
                    "id": f"{source_relation_id}::direction",
                    "diagnostic_variant": "direction",
                    "diagnostic_answer_format": "direction",
                    "options": list(DIRECTIONS),
                    "answer": relation,
                    "answer_idx": direction_answer_idx,
                    "gold_option_letter": OPTION_LETTERS[direction_answer_idx],
                }
            )
            records.append(
                {
                    **common,
                    "id": f"{source_relation_id}::object",
                    "diagnostic_variant": "object",
                    "diagnostic_answer_format": "object",
                    "options": object_options,
                    "answer": target,
                    "answer_idx": object_answer_idx,
                    "gold_option_letter": OPTION_LETTERS[object_answer_idx],
                }
            )
            stats["accepted_pairs"] += 1
            if limit_pairs is not None and stats["accepted_pairs"] >= limit_pairs:
                stop = True
                break
        if stop:
            break

    report = audit_records(records)
    report["skipped_counts"] = {reason: stats[reason] for reason in SKIP_REASONS}
    report["accepted_pairs"] = stats["accepted_pairs"]
    report["seed"] = seed
    report["gqa_split"] = gqa_split
    return records, report


def audit_records(records: Iterable[Mapping], *, check_image_paths: bool = False) -> dict:
    records = list(records)
    errors = []
    by_source = defaultdict(list)
    for row_number, record in enumerate(records, 1):
        source_id = record.get("source_relation_id")
        by_source[source_id].append(record)
        if not record.get("image_path"):
            errors.append(f"row {row_number}: missing image_path")
        elif check_image_paths and not Path(str(record["image_path"])).is_file():
            errors.append(f"row {row_number}: image_path does not exist")
        options = record.get("options", [])
        if len(options) != 4 or record.get("num_options") != 4:
            errors.append(f"row {row_number}: expected exactly four options")
        answer_idx = record.get("answer_idx")
        if not isinstance(answer_idx, int) or not 0 <= answer_idx < len(options):
            errors.append(f"row {row_number}: invalid answer_idx")
        elif options[answer_idx] != record.get("answer"):
            errors.append(f"row {row_number}: answer does not match options[answer_idx]")
        if record.get("diagnostic_answer_format") == "object":
            folded = [str(option).casefold() for option in options]
            if len(folded) != len(set(folded)):
                errors.append(f"row {row_number}: duplicate object option names")
            correct_option_ids = record.get(
                "diagnostic_correct_option_object_ids", []
            )
            if (
                correct_option_ids != [record.get("diagnostic_target_object_id")]
                or record.get("diagnostic_correct_object_options")
                != [record.get("answer")]
                or record.get("num_correct_object_options") != 1
            ):
                errors.append(
                    f"row {row_number}: multiple or invalid correct object options"
                )

    matched_pairs = 0
    for source_id, pair in by_source.items():
        formats = {record.get("diagnostic_answer_format"): record for record in pair}
        if set(formats) != {"direction", "object"} or len(pair) != 2:
            errors.append(f"source {source_id}: expected one direction and one object row")
            continue
        shared_fields = (
            "image",
            "image_path",
            "image_id",
            "diagnostic_anchor",
            "diagnostic_target_object",
            "diagnostic_relation",
            "candidate_pool",
        )
        if any(formats["direction"].get(field) != formats["object"].get(field) for field in shared_fields):
            errors.append(f"source {source_id}: pair metadata differs")
            continue
        matched_pairs += 1

    gold_letters = Counter(str(record.get("gold_option_letter")) for record in records)
    relations = Counter(str(record.get("diagnostic_relation")) for record in records)
    formats = Counter(str(record.get("diagnostic_answer_format")) for record in records)
    return {
        "valid": not errors,
        "errors": errors,
        "num_records": len(records),
        "num_source_relation_groups": len(by_source),
        "num_matched_pairs": matched_pairs,
        "gold_option_letter_distribution": dict(sorted(gold_letters.items())),
        "gold_relation_distribution": dict(sorted(relations.items())),
        "answer_format_distribution": dict(sorted(formats.items())),
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-graphs",
        type=Path,
        action="append",
        help="GQA sceneGraphs JSON or zip; repeat for multiple files",
    )
    parser.add_argument("--output", type=Path, help="Output diagnostic JSONL")
    parser.add_argument("--audit-only", type=Path, help="Audit an existing JSONL")
    parser.add_argument("--audit-report", type=Path, help="Write the audit JSON")
    parser.add_argument("--sample-output", type=Path, help="Write the first five pairs")
    parser.add_argument("--gqa-split", default="val")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/ramanathan/data/GQA/images"),
        help="Directory containing GQA images named {image_id}.jpg",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Generate paths without requiring each image file to exist",
    )
    parser.add_argument("--seed", default="gqa_direction_object_v1")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--image-ids", type=Path, help="Optional requested image IDs")
    parser.add_argument("--valid-image-ids", type=Path, help="Optional known-valid image IDs")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        records = read_jsonl(args.audit_only)
        report = audit_records(
            records, check_image_paths=not args.allow_missing_images
        )
    else:
        if not args.scene_graphs or not args.output:
            raise SystemExit("build mode requires --scene-graphs and --output")
        scene_graphs = load_scene_graphs(args.scene_graphs, args.gqa_split)
        records, report = build_records(
            scene_graphs,
            gqa_split=args.gqa_split,
            seed=args.seed,
            limit_pairs=args.limit_pairs,
            requested_image_ids=_read_id_file(args.image_ids),
            valid_image_ids=_read_id_file(args.valid_image_ids),
            image_dir=args.image_dir,
            require_images=not args.allow_missing_images,
        )
        write_jsonl(args.output, records)
        if args.sample_output:
            write_jsonl(args.sample_output, records[:10])

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
