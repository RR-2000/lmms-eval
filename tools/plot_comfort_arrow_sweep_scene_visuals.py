#!/usr/bin/env python3
"""Render one COMFORT scene at every arrow-length and label-position step.

This script calls the task's own ``doc_to_visual`` function so the resulting
panels are identical to the visual inputs used during evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from datasets import Dataset

from lmms_eval.tasks.comfort_direction_object_inverse_diagnostics import utils


def load_scene_rows(annotation_path: Path, scene_id: str) -> Dataset:
    rows = []
    with annotation_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("scene_id")) == scene_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"Scene {scene_id!r} was not found in {annotation_path}")
    # Direction and object rows have a few task-specific fields.  Fill their
    # union explicitly because Dataset.from_list otherwise infers its schema
    # from the first row and can discard fields that appear only later.
    all_keys = set().union(*(row.keys() for row in rows))
    normalized_rows = [{key: row.get(key) for key in all_keys} for row in rows]
    return Dataset.from_list(normalized_rows)


def select_condition_docs(
    dataset: Dataset,
    process_docs: Callable[[Dataset], Dataset],
    *,
    scene_id: str,
    relation: str,
    answer_format: str,
) -> list[dict]:
    docs = [
        dict(doc)
        for doc in process_docs(dataset)
        if str(doc["scene_id"]) == scene_id
        and str(doc["diagnostic_relation"]) == relation
        and str(doc["diagnostic_answer_format"]) == answer_format
    ]
    if len(docs) != 6:
        raise ValueError(
            f"Expected six conditions for {scene_id}/{relation}/{answer_format}; "
            f"found {len(docs)}"
        )
    return docs


def plot_contact_sheet(
    docs: list[dict],
    *,
    panel_titles: list[str],
    figure_title: str,
    output_stem: Path,
) -> None:
    if len(docs) != len(panel_titles):
        raise ValueError("Every rendered document must have one panel title")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    reference = str(docs[0]["diagnostic_anchor"])
    scene_id = str(docs[0]["scene_id"])
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.25))
    for ax, doc, title in zip(axes.flat, docs, panel_titles):
        visual = utils.doc_to_visual(doc)[0]
        ax.imshow(visual)
        ax.set_title(title, fontweight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#C7C7C7")
            spine.set_linewidth(0.8)

    fig.suptitle(
        f"{figure_title}\n{scene_id} · reference object: {reference}",
        x=0.02,
        y=0.985,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.88, wspace=0.04, hspace=0.14)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_visuals(
    docs: list[dict],
    *,
    output_dir: Path,
    filename_for_doc: Callable[[dict], str],
) -> None:
    """Save each evaluation condition as its own unmodified PNG visual."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        visual = utils.doc_to_visual(doc)[0]
        visual.save(output_dir / filename_for_doc(doc), format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene_000000")
    parser.add_argument("--relation", choices=utils.DIRECTIONS, default="left")
    parser.add_argument("--answer-format", choices=utils.ANSWER_FORMATS, default="direction")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/home/ramanathan/data/COMFORT_Multi_3D/annotations.jsonl"),
    )
    parser.add_argument(
        "--label-output-dir",
        type=Path,
        default=Path("outputs/comfort_arrow_label_endpoint_8/analysis"),
    )
    parser.add_argument(
        "--length-output-dir",
        type=Path,
        default=Path("outputs/comfort_arrow_length_sweep_8/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dataset = load_scene_rows(args.annotations, args.scene_id)

    length_docs = select_condition_docs(
        scene_dataset,
        utils.process_arrow_length_sweep_docs,
        scene_id=args.scene_id,
        relation=args.relation,
        answer_format=args.answer_format,
    )
    plot_contact_sheet(
        length_docs,
        panel_titles=[
            f"Length = {float(doc['arrow_length_scale']):g}× bbox diagonal"
            for doc in length_docs
        ],
        figure_title="COMFORT arrow-length sweep visuals",
        output_stem=args.length_output_dir
        / f"arrow_length_sweep_{args.scene_id}_visuals",
    )
    save_individual_visuals(
        length_docs,
        output_dir=args.length_output_dir / f"{args.scene_id}_visuals",
        filename_for_doc=lambda doc: (
            f"arrow_length_{float(doc['arrow_length_scale']):.2f}.png"
        ),
    )

    label_docs = select_condition_docs(
        scene_dataset,
        utils.process_arrow_label_endpoint_docs,
        scene_id=args.scene_id,
        relation=args.relation,
        answer_format=args.answer_format,
    )
    label_titles = []
    for doc in label_docs:
        position = float(doc["arrow_label_position"])
        suffix = " (origin)" if position == 0.0 else " (arrowhead)" if position == 1.0 else ""
        label_titles.append(f"Label position = {position:.1f}{suffix}")
    plot_contact_sheet(
        label_docs,
        panel_titles=label_titles,
        figure_title="COMFORT arrow-label position sweep visuals",
        output_stem=args.label_output_dir
        / f"arrow_label_position_{args.scene_id}_visuals",
    )
    save_individual_visuals(
        label_docs,
        output_dir=args.label_output_dir / f"{args.scene_id}_visuals",
        filename_for_doc=lambda doc: (
            f"label_position_{float(doc['arrow_label_position']):.1f}.png"
        ),
    )


if __name__ == "__main__":
    main()
