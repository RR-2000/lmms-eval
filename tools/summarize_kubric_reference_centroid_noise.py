#!/usr/bin/env python3
"""Compute the E3 reference-centroid geometric ceiling without a VLM."""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset

from lmms_eval.tasks.kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_reference_centroid_noise import utils


DEFAULT_DATASET = "/home/ramanathan/data/movi_a_relative_direction/movi_a_validation.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files={"test": args.dataset})["test"]
    docs = utils.process_docs(dataset)
    curve = utils.geometric_curve(docs)
    if args.format == "json":
        print(json.dumps(curve, indent=2))
        return
    print("| σ / reference-target distance | Trials | Wrong-box fraction | Geometric ceiling |")
    print("|---:|---:|---:|---:|")
    for row in curve:
        print(
            f"| {row['sigma_fraction']:.0%} | {row['num_trials']} | "
            f"{row['wrong_box_fraction']:.3f} | {row['geometric_ceiling']:.3f} |"
        )


if __name__ == "__main__":
    main()
