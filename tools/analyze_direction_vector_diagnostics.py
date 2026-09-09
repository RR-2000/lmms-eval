#!/usr/bin/env python3
"""Analyze the cross-dataset direction/vector diagnostic submissions."""

from __future__ import annotations

import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

EXPERIMENT_PREFIXES = (
    "cross_output_consistency_",
    "output_format_control_",
    "conversion_oracle_",
    "basis_oracle_ladder_",
    "component_decomposition_",
    "angular_boundary_",
    "arrow_vector_grounding_",
    "kubric_rotation_equivariance_",
    "scannet_rgb_depth_oracle_",
)
PREFIXES = EXPERIMENT_PREFIXES + tuple(f"{dataset}_{prefix}" for dataset in ("comfort", "scannet", "kubric") for prefix in EXPERIMENT_PREFIXES)
FIELDS = (
    "direction_accuracy",
    "vector_direction_accuracy",
    "vector_cosine",
    "horizontal_cosine",
    "front_sign_accuracy",
    "up_sign_accuracy",
    "right_sign_accuracy",
    "internal_consistency",
    "component_accuracy",
    "parse_success",
)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def discover(inputs):
    found = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(p for p in path.rglob("*.json") if p.name.startswith(PREFIXES))
        else:
            raise FileNotFoundError(path)
    if not found:
        raise FileNotFoundError("No direction/vector diagnostic submissions found")
    return sorted(set(found))


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No records: {path}")
    experiment = str(payload.get("experiment") or rows[0].get("diagnostic_experiment"))
    dataset_name = str(payload.get("dataset_name") or rows[0].get("dataset_name") or "unknown")
    return dataset_name, experiment, rows


def mean(rows, field):
    return sum(float(r.get(field, 0)) for r in rows) / len(rows) if rows else 0.0


def summarize(rows, keys=("experiment_condition",)):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return [{**dict(zip(keys, key)), "count": len(group), **{f: mean(group, f) for f in FIELDS}} for key, group in sorted(groups.items())]


def outcomes(rows):
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("experiment_condition"))].append(row)
    for condition, group in sorted(groups.items()):
        applicable = [
            r
            for r in group
            if r.get("response_mode") in {"combined", "special"} and (r.get("parsed_direction") is not None or r.get("parsed_vector") is not None)
        ]
        counts = Counter()
        for row in applicable:
            d = bool(row.get("direction_accuracy"))
            v = bool(row.get("vector_direction_accuracy"))
            counts["both_correct" if d and v else "direction_only" if d else "vector_only" if v else "both_wrong"] += 1
        if applicable:
            result.append(
                {
                    "experiment_condition": condition,
                    "count": len(applicable),
                    **{k: counts[k] / len(applicable) for k in ("both_correct", "direction_only", "vector_only", "both_wrong")},
                }
            )
    return result


def cross_head(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[str(row.get("base_pair_id"))][str(row.get("experiment_condition"))] = row
    counts = Counter()
    total = 0
    for group in groups.values():
        if not {"direction_only", "vector_only"} <= set(group):
            continue
        d = bool(group["direction_only"].get("direction_accuracy"))
        v = bool(group["vector_only"].get("vector_direction_accuracy"))
        total += 1
        counts["both_correct" if d and v else "direction_only" if d else "vector_only" if v else "both_wrong"] += 1
    return [{"count": total, **{key: counts[key] / total for key in ("both_correct", "direction_only", "vector_only", "both_wrong")}}] if total else []


def transitions(rows, baseline):
    groups = defaultdict(dict)
    for row in rows:
        groups[str(row.get("base_pair_id"))][str(row.get("experiment_condition"))] = row
    conditions = sorted({str(r.get("experiment_condition")) for r in rows} - {baseline})
    output = []
    for condition in conditions:
        pairs = [(g[baseline], g[condition]) for g in groups.values() if {baseline, condition} <= set(g)]
        if not pairs:
            continue
        for metric in ("direction_accuracy", "vector_direction_accuracy"):
            counts = Counter()
            for before, after in pairs:
                a, b = bool(before.get(metric)), bool(after.get(metric))
                counts["improved" if not a and b else "worsened" if a and not b else "unchanged_correct" if a else "unchanged_wrong"] += 1
            n = len(pairs)
            output.append(
                {
                    "experiment_condition": condition,
                    "metric": metric,
                    "count": n,
                    "baseline": mean([p[0] for p in pairs], metric),
                    "condition": mean([p[1] for p in pairs], metric),
                    **{k: counts[k] / n for k in ("improved", "worsened", "unchanged_correct", "unchanged_wrong")},
                }
            )
    return output


def _cos(a, b):
    axes = ("front", "up", "right")
    na = math.sqrt(sum(float(a[x]) ** 2 for x in axes))
    nb = math.sqrt(sum(float(b[x]) ** 2 for x in axes))
    return sum(float(a[x]) * float(b[x]) for x in axes) / (na * nb) if na and nb else 0.0


def rotation_stability(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("base_pair_id"))].append(row)
    output = []
    for pair_id, group in groups.items():
        if len(group) != 4:
            continue
        directions = [r.get("parsed_direction") for r in group]
        vectors = [r.get("parsed_vector") for r in group]
        valid = [v for v in vectors if isinstance(v, dict)]
        pairwise = [_cos(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
        output.append(
            {
                "base_pair_id": pair_id,
                "direction_semantic_consistency": float(None not in directions and len(set(directions)) == 1),
                "all_directions_correct": float(all(r.get("direction_accuracy") for r in group)),
                "all_vector_directions_correct": float(all(r.get("vector_direction_accuracy") for r in group)),
                "mean_pairwise_vector_cosine": sum(pairwise) / len(pairwise) if pairwise else 0.0,
                "all_vectors_parsed": float(len(valid) == 4),
            }
        )
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(summary, title, path):
    if not summary:
        return
    labels = [r["experiment_condition"].replace("_", "\n") for r in summary]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(labels)), 6))
    series = (
        ("direction_accuracy", "Direction", "#457B9D"),
        ("vector_direction_accuracy", "Vector→direction", "#E9C46A"),
        ("horizontal_cosine", "Horizontal cosine", "#2A9D8F"),
        ("component_accuracy", "Component", "#9B5DE5"),
    )
    for i, (field, label, color) in enumerate(series):
        ax.bar([j + (i - 1) * width for j in x], [r[field] for r in summary], width, label=label, color=color)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset(rows, output_dir):
    by_experiment = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(row)
    for experiment, group in by_experiment.items():
        datasets = sorted({row["dataset_name"] for row in group})
        if len(datasets) < 2:
            continue
        conditions = list(dict.fromkeys(row["condition"] for row in group))
        lookup = {(row["dataset_name"], row["condition"]): row for row in group}
        fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.35 * len(conditions)), 9), sharex=True)
        for ax, metric, title in zip(axes, ("direction_accuracy", "vector_direction_accuracy"), ("Direction answer", "Vector-derived direction")):
            for dataset_name in datasets:
                values = [lookup.get((dataset_name, condition), {}).get(metric, math.nan) for condition in conditions]
                ax.plot(range(len(conditions)), values, marker="o", linewidth=2, label=dataset_name)
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Accuracy")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend(frameon=False, ncol=len(datasets))
        axes[-1].set_xticks(range(len(conditions)), [condition.replace("_", "\n") for condition in conditions])
        fig.suptitle(experiment.replace("_", " ").title() + ": Cross-Dataset Comparison")
        fig.tight_layout()
        fig.savefig(output_dir / f"cross_dataset_{experiment}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def markdown(tables):
    lines = ["# Direction/vector diagnostic results", ""]
    for name, rows in tables.items():
        lines += [f"## {name.replace('_',' ').title()}", ""]
        if not rows:
            lines += ["No complete/applicable records.", ""]
            continue
        cols = list(rows[0])
        lines += ["| " + " | ".join(c.replace("_", " ") for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        lines += ["| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for r in rows]
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    a = args()
    output = a.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for path in discover(a.inputs):
        dataset_name, experiment, rows = load(path)
        name = f"{dataset_name}_{experiment}"
        if name in experiments:
            raise ValueError(f"Duplicate dataset/experiment {name}; analyze one model/run at a time")
        experiments[name] = (experiment, rows)
    tables = {}
    cross_dataset = []
    for name, (experiment, rows) in experiments.items():
        dataset_name = str(rows[0].get("dataset_name", "unknown"))
        keys = ("angular_margin_bin",) if experiment == "angular_boundary" else ("experiment_condition",)
        table = summarize(rows, keys)
        cross_dataset.extend(
            {
                "dataset_name": dataset_name,
                "experiment": experiment,
                "condition": row.get("experiment_condition", row.get("angular_margin_bin", "")),
                "count": row["count"],
                **{field: row[field] for field in FIELDS},
            }
            for row in table
        )
        tables[name] = table
        write_csv(output / f"{name}.csv", table)
        plot(
            table if keys == ("experiment_condition",) else [{**r, "experiment_condition": r["angular_margin_bin"]} for r in table],
            name.replace("_", " ").title(),
            output / f"{name}.png",
        )
        detail = summarize(rows, ("experiment_condition", "gt_direction"))
        tables[f"{name}_by_direction"] = detail
        write_csv(output / f"{name}_by_direction.csv", detail)
        case = outcomes(rows)
        tables[f"{name}_outcomes"] = case
        write_csv(output / f"{name}_outcomes.csv", case)
        if experiment == "cross_output_consistency":
            tables[f"{name}_paired_outcomes"] = cross_head(rows)
            write_csv(output / f"{name}_paired_outcomes.csv", tables[f"{name}_paired_outcomes"])
        baseline = {"basis_oracle_ladder": "rgb", "arrow_vector_grounding": "plain", "scannet_rgb_depth_oracle": "rgb"}.get(name)
        baseline = baseline or {"basis_oracle_ladder": "rgb", "arrow_vector_grounding": "plain", "scannet_rgb_depth_oracle": "rgb"}.get(experiment)
        if baseline:
            change = transitions(rows, baseline)
            tables[f"{name}_transitions"] = change
            write_csv(output / f"{name}_transitions.csv", change)
        if experiment == "kubric_rotation_equivariance":
            stable = rotation_stability(rows)
            tables["kubric_rotation_stability"] = stable
            write_csv(output / "kubric_rotation_stability.csv", stable)
    tables["cross_dataset_comparison"] = sorted(
        cross_dataset,
        key=lambda row: (row["experiment"], row["dataset_name"], row["condition"]),
    )
    write_csv(output / "cross_dataset_comparison.csv", tables["cross_dataset_comparison"])
    plot_cross_dataset(tables["cross_dataset_comparison"], output)
    (output / "summary.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown(tables), encoding="utf-8")
    print(f"Analyzed {len(experiments)} experiments into {output}")


if __name__ == "__main__":
    raise SystemExit(main())
