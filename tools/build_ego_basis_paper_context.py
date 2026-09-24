#!/usr/bin/env python3
"""Build a paper-writing context document from the final ego-basis runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
REPORT = OUTPUTS / "ego_basis_analysis_for_paper.md"

FAMILIES = {
    "3DSRBench": "final_3dsrbench_direction_object",
    "COMFORT-Oriented-3D": "final_comfort_oriented_3d_direction_object",
    "Kubric MOVi-A": "final_kubric_movi_a_direction_object_relative_direction",
    "ScanNet v2": "final_scannet_basis_object_direction",
}

MODEL_IDS = {
    "cambrians": "nyu-visionx/Cambrian-S-7B",
    "internvideo3": "yanziang/InternVideo3-8B-Instruct",
    "internvl3_5": "OpenGVLab/InternVL3_5-8B",
    "llava_onevision2": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
    "longva": "lmms-lab/LongVA-7B",
    "openai": "gpt-5.6-luna",
    "sat": "array/Qwen2.5-VL-SAT",
    "spatial_mllm": "Diankun/Spatial-MLLM-v1.1-Instruct-820K",
    "spatialladder": "hongxingli/SpatialLadder-3B",
    "vst": "rayruiyang/VST-7B-RL",
}

MODEL_LABELS = {
    "cambrians": "Cambrian-S-7B",
    "internvideo3": "InternVideo3-8B",
    "internvl3_5": "InternVL3.5-8B",
    "llava_onevision2": "LLaVA-OneVision-2-8B",
    "longva": "LongVA-7B",
    "openai": "GPT-5.6 Luna",
    "sat": "Qwen2.5-VL-SAT",
    "spatial_mllm": "Spatial-MLLM-v1.1",
    "spatialladder": "SpatialLadder-3B",
    "vst": "VST-7B-RL",
}

FRAME_LABELS = {
    "overall": "Overall",
    "camera_relative_direction": "Camera-relative",
    "object_relative_direction": "Object-relative",
    "camera": "Camera frame",
    "object_facing_camera": "Object-facing-camera frame",
}

RELATION_ORDER = ("left", "right", "front", "behind")


@dataclass
class Cell:
    benchmark: str
    frame: str
    model: str
    pairs: int
    direction_correct: int
    object_correct: int
    direction_parse: int
    object_parse: int
    improved: int
    worsened: int
    both_correct: int
    both_wrong: int
    relation_rows: dict[str, list[tuple[str, float, bool]]]

    @property
    def direction_accuracy(self) -> float:
        return self.direction_correct / self.pairs

    @property
    def object_accuracy(self) -> float:
        return self.object_correct / self.pairs

    @property
    def delta(self) -> float:
        return self.object_accuracy - self.direction_accuracy

    @property
    def direction_parse_rate(self) -> float:
        return self.direction_parse / self.pairs

    @property
    def object_parse_rate(self) -> float:
        return self.object_parse / self.pairs

    @property
    def valid_for_macro(self) -> bool:
        return min(self.direction_parse_rate, self.object_parse_rate) >= 0.80


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"No records list in {path}")
    return rows


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score", row.get("answer_accuracy", 0.0)) or 0.0)


def _parsed(row: dict[str, Any]) -> bool:
    if "parse_success" in row:
        return bool(float(row["parse_success"] or 0.0))
    return row.get("parsed_prediction") not in (None, "")


def _answer_format(row: dict[str, Any]) -> str:
    value = str(row.get("answer_format", ""))
    if value in {"direction", "object"}:
        return value
    return {"native": "direction", "inverse": "object"}.get(str(row.get("variant")), value)


def _relation(row: dict[str, Any]) -> str:
    value = str(row.get("relation", row.get("direction", row.get("gt_direction", "")))).lower()
    return "behind" if value == "back" else value


def _pair_id(row: dict[str, Any]) -> str:
    return str(row.get("source_qid", row.get("pair_id", "")))


def _make_cell(benchmark: str, frame: str, model: str, rows: Iterable[dict[str, Any]]) -> Cell:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        fmt = _answer_format(row)
        if fmt in {"direction", "object"}:
            pairs[_pair_id(row)][fmt] = row

    complete = [formats for formats in pairs.values() if {"direction", "object"} <= formats.keys()]
    improved = worsened = both_correct = both_wrong = 0
    relation_rows: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)
    for formats in complete:
        direction = formats["direction"]
        obj = formats["object"]
        d_ok = _score(direction) == 1.0
        o_ok = _score(obj) == 1.0
        if o_ok and not d_ok:
            improved += 1
        elif d_ok and not o_ok:
            worsened += 1
        elif d_ok:
            both_correct += 1
        else:
            both_wrong += 1
        relation = _relation(direction) or _relation(obj)
        relation_rows[relation].extend(
            [("direction", _score(direction), _parsed(direction)), ("object", _score(obj), _parsed(obj))]
        )

    return Cell(
        benchmark=benchmark,
        frame=frame,
        model=model,
        pairs=len(complete),
        direction_correct=sum(_score(x["direction"]) == 1.0 for x in complete),
        object_correct=sum(_score(x["object"]) == 1.0 for x in complete),
        direction_parse=sum(_parsed(x["direction"]) for x in complete),
        object_parse=sum(_parsed(x["object"]) for x in complete),
        improved=improved,
        worsened=worsened,
        both_correct=both_correct,
        both_wrong=both_wrong,
        relation_rows=dict(relation_rows),
    )


def collect() -> list[Cell]:
    cells: list[Cell] = []
    for benchmark, dirname in FAMILIES.items():
        family = OUTPUTS / dirname
        for model_dir in sorted(path for path in family.iterdir() if path.is_dir()):
            model = model_dir.name
            submissions = sorted((model_dir / "submissions").glob("*.json"))
            if benchmark == "ScanNet v2":
                for submission in submissions:
                    rows = _load_rows(submission)
                    frame = str(rows[0].get("coordinate_frame"))
                    cells.append(_make_cell(benchmark, frame, model, rows))
                continue

            if len(submissions) != 1:
                raise ValueError(f"Expected one submission under {model_dir}, found {len(submissions)}")
            rows = _load_rows(submissions[0])
            if benchmark == "Kubric MOVi-A":
                for frame in ("camera_relative_direction", "object_relative_direction"):
                    subset = [row for row in rows if row.get("source_task_family") == frame]
                    cells.append(_make_cell(benchmark, frame, model, subset))
            else:
                cells.append(_make_cell(benchmark, "overall", model, rows))
    return cells


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def signed_pp(value: float) -> str:
    return f"{100 * value:+.1f}"


def mcnemar_exact(improved: int, worsened: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = improved + worsened
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(improved, worsened) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _cell_table(cells: list[Cell]) -> list[str]:
    rows: list[list[str]] = []
    for cell in sorted(cells, key=lambda x: MODEL_LABELS[x.model].lower()):
        parse = f"{pct(cell.direction_parse_rate)}/{pct(cell.object_parse_rate)}"
        if not cell.valid_for_macro:
            parse += " ⚠"
        rows.append(
            [
                MODEL_LABELS[cell.model],
                str(cell.pairs),
                pct(cell.direction_accuracy),
                pct(cell.object_accuracy),
                signed_pp(cell.delta),
                str(cell.improved),
                str(cell.worsened),
                str(cell.both_correct),
                str(cell.both_wrong),
                f"{mcnemar_exact(cell.improved, cell.worsened):.3g}",
                parse,
            ]
        )
    return markdown_table(
        [
            "Model",
            "Pairs",
            "Dir. acc. %",
            "Obj. acc. %",
            "Obj−dir pp",
            "Obj only",
            "Dir only",
            "Both ✓",
            "Both ✗",
            "McNemar p",
            "Parse D/O %",
        ],
        rows,
    )


def _macro_rows(cells: list[Cell]) -> list[list[str]]:
    rows: list[list[str]] = []
    keys = sorted({(cell.benchmark, cell.frame) for cell in cells})
    for benchmark, frame in keys:
        selected = [c for c in cells if (c.benchmark, c.frame) == (benchmark, frame) and c.valid_for_macro]
        wins = sum(c.delta > 0 for c in selected)
        ties = sum(c.delta == 0 for c in selected)
        losses = sum(c.delta < 0 for c in selected)
        rows.append(
            [
                benchmark,
                FRAME_LABELS[frame],
                str(len(selected)),
                pct(sum(c.direction_accuracy for c in selected) / len(selected)),
                pct(sum(c.object_accuracy for c in selected) / len(selected)),
                signed_pp(sum(c.delta for c in selected) / len(selected)),
                f"{wins}/{ties}/{losses}",
                f"{sum(c.improved for c in selected)}/{sum(c.worsened for c in selected)}",
            ]
        )
    return rows


def _relation_table(cells: list[Cell]) -> list[list[str]]:
    rows: list[list[str]] = []
    keys = sorted({(cell.benchmark, cell.frame) for cell in cells})
    for benchmark, frame in keys:
        selected = [c for c in cells if (c.benchmark, c.frame) == (benchmark, frame) and c.valid_for_macro]
        for relation in RELATION_ORDER:
            d_scores: list[float] = []
            o_scores: list[float] = []
            for cell in selected:
                by_format: dict[str, list[float]] = defaultdict(list)
                for fmt, score, _ in cell.relation_rows.get(relation, []):
                    by_format[fmt].append(score)
                if by_format["direction"]:
                    d_scores.append(sum(by_format["direction"]) / len(by_format["direction"]))
                    o_scores.append(sum(by_format["object"]) / len(by_format["object"]))
            if d_scores:
                d_mean = sum(d_scores) / len(d_scores)
                o_mean = sum(o_scores) / len(o_scores)
                rows.append(
                    [benchmark, FRAME_LABELS[frame], relation, pct(d_mean), pct(o_mean), signed_pp(o_mean - d_mean)]
                )
    return rows


def build(cells: list[Cell]) -> str:
    valid = [cell for cell in cells if cell.valid_for_macro]
    invalid = [cell for cell in cells if not cell.valid_for_macro]
    representative_pairs = {
        (cell.benchmark, cell.frame): cell.pairs
        for cell in cells
    }
    pairs_per_model = sum(representative_pairs.values())
    total_pairs = sum(cell.pairs for cell in cells)
    best = max(valid, key=lambda cell: cell.delta)
    worst = min(valid, key=lambda cell: cell.delta)
    wins = sum(cell.delta > 0 for cell in valid)
    ties = sum(cell.delta == 0 for cell in valid)
    losses = sum(cell.delta < 0 for cell in valid)

    lines = [
        "# Ego-basis direction–object analysis: paper-writing context",
        "",
        f"Generated from the completed `outputs/final_*` runs on {date.today().isoformat()}.",
        "This is an evidence packet for drafting a conference-paper experiment section; it is not itself polished paper prose.",
        "All reported accuracies are recomputed from the saved per-example submissions.",
        "",
        "## One-paragraph experiment description",
        "",
        "We test whether multimodal models' spatial judgments depend on the required answer representation. "
        "For each underlying spatial relation, we construct a matched pair: a **direction-answer** item asks for "
        "one of four labels (left, right, front, behind/back), while an **object-answer** item asks the model to "
        "select which of four objects occupies a specified direction. The image and underlying relation are held "
        "fixed within a pair. This paired design isolates answer-format sensitivity more directly than comparing "
        "unmatched question sets. We evaluate camera-relative, object-relative, and object-facing-camera coordinate "
        "frames across 3DSRBench, COMFORT-Oriented-3D, Kubric MOVi-A, and ScanNet v2. Decoding is deterministic "
        "where specified by the task configuration; all tasks use four choices, so chance accuracy is 25%.",
        "",
        "## Experimental inventory",
        "",
    ]
    inventory = [
        ["3DSRBench", "Object/viewpoint relation", "119", "238", "Direction (`native`) vs object (`inverse`)", "4"],
        ["COMFORT-Oriented-3D", "Anchor object's current viewpoint", "2,000", "4,000", "Direction vs object", "4"],
        ["Kubric MOVi-A", "Camera-relative", "219", "438", "Direction (`native`) vs object (`inverse`)", "4"],
        ["Kubric MOVi-A", "Object-relative", "218", "436", "Direction (`native`) vs object (`inverse`)", "4"],
        ["ScanNet v2", "Camera frame", "1,988", "3,976", "Direction vs object", "4"],
        ["ScanNet v2", "Object facing camera", "1,740", "3,480", "Direction vs object", "4"],
    ]
    lines += markdown_table(["Benchmark", "Coordinate frame", "Matched pairs/model", "Items/model", "Formats", "Choices"], inventory)
    lines += [
        "",
        f"The full suite contains {pairs_per_model:,} unique matched relations per model "
        f"({2 * pairs_per_model:,} question instances), or {total_pairs:,} matched relations and "
        f"{2 * total_pairs:,} instances across the ten evaluated model entries. COMFORT contains "
        "500 rendered scenes × four relations = 2,000 pairs. The ScanNet splits each cover 102 scenes.",
        "",
        "### Pair construction and coordinate bases",
        "",
        "- **3DSRBench:** only `multi_object_viewpoint_towards_object` source rows are used because each identifies "
        "both the side of a subject and the object toward which that side points. The native member predicts the "
        "direction; the inverse member receives the direction and selects the target object. An inverse item must "
        "have at least one distractor derived from another queried object in the same image. If fewer than four "
        "choices remain, deterministic thematic COCO-style distractors pad the set (seed "
        "`3dsrbench_direction_object_v1`). This construction detail should be disclosed because generated "
        "distractors may differ in difficulty from image-derived distractors.",
        "- **COMFORT-Oriented-3D:** the authored annotation file provides matched direction- and object-answer "
        "questions for the left, right, front, and behind sides of an oriented anchor in each rendered scene. The "
        "questions use the anchor object's current viewpoint.",
        "- **Kubric MOVi-A:** the relative-direction export is already paired. Each `pair_id` has one authored "
        "four-choice direction row and one authored four-choice object row. Evaluation preserves the questions, "
        "options, image, reference/target objects, and relation. The two source families are camera-relative and "
        "object-relative direction.",
        "- **ScanNet camera basis:** visible instance 3D box centres are transformed into each RGB frame's camera "
        "coordinates, with `+X` image-right, `+Y` camera-forward, and `+Z` image-up. Ground truth is the normalized "
        "target-minus-reference vector; its dominant absolute horizontal component yields left, right, front, or back.",
        "- **ScanNet object-facing-camera basis:** ScanNet has no dependable semantic yaw for reconstructed objects. "
        "The evaluation therefore uses an explicit constructed anchor frame rather than asserting a semantic object "
        "front. The reference centre is the origin, horizontal anchor-to-camera is `+front`, world up is `+up`, and "
        "`right = front × up`; consequently, anchor-right appears camera/image-left. The same displacement is "
        "projected into this basis.",
        "- **ScanNet sampling controls:** frames retain four to five visible, non-structural instances with labels "
        "unique in that frame; duplicated labels and structural classes are removed, and frames are sampled evenly "
        "per scene. The paired task further keeps only five-object views (four candidates per reference), removes "
        "relations with multiple candidate objects, and deterministically balances the four directions. The current "
        "manifest has 980 views before paired-task filtering.",
        "",
        "### Evaluated models",
        "",
    ]
    lines += markdown_table(
        ["Report label", "Output directory key", "Recorded model identifier"],
        [[MODEL_LABELS[key], f"`{key}`", f"`{MODEL_IDS[key]}`"] for key in sorted(MODEL_IDS, key=lambda k: MODEL_LABELS[k].lower())],
    )
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **Direction accuracy**: exact multiple-choice correctness on the direction-answer member of each pair.",
        "- **Object accuracy**: exact multiple-choice correctness on the object-answer member of each pair.",
        "- **Object − direction (percentage points)**: paired mean score difference. Positive values favor object answers.",
        "- **Object only / improved**: direction member wrong and object member correct.",
        "- **Direction only / worsened**: direction member correct and object member wrong.",
        "- **Both correct / both wrong**: matched-pair agreement states.",
        "- **McNemar p**: two-sided exact test over discordant pairs (object-only vs direction-only). These p-values "
        "are uncorrected and should not be treated as the paper's final multiple-comparison analysis.",
        "- **Parse D/O**: percentage of direction/object responses successfully parsed. Unparseable responses score zero.",
        "",
        "For cross-model macro summaries only, a model/frame cell is included when both format-specific parse rates "
        "are at least 80%. This removes clearly broken runs rather than silently treating interface failure as spatial "
        "reasoning failure. Every run, including excluded runs, remains visible in the detailed tables.",
        "",
        "## Cross-model summary",
        "",
    ]
    lines += markdown_table(
        ["Benchmark", "Frame", "Valid models", "Mean dir. %", "Mean obj. %", "Mean Δ pp", "Obj win/tie/loss", "Pooled obj-only/dir-only"],
        _macro_rows(cells),
    )
    lines += [
        "",
        f"Across the {len(valid)} parse-valid model × benchmark/frame cells, object answers are better in "
        f"{wins}, tied in {ties}, and worse in {losses}. The largest positive difference is "
        f"{best.benchmark}/{FRAME_LABELS[best.frame]} for {MODEL_LABELS[best.model]} "
        f"({signed_pp(best.delta)} pp); the largest negative difference is "
        f"{worst.benchmark}/{FRAME_LABELS[worst.frame]} for {MODEL_LABELS[worst.model]} "
        f"({signed_pp(worst.delta)} pp). This heterogeneity argues against describing object answers as uniformly "
        "easier or harder; the effect depends on dataset, coordinate frame, model, and relation.",
        "",
        "## Detailed model results",
        "",
        "All accuracy and parse columns are percentages; Δ is in percentage points. `⚠` marks cells excluded "
        "from cross-model means by the 80% parse rule.",
    ]
    for benchmark in FAMILIES:
        lines += ["", f"### {benchmark}", ""]
        frames = sorted({c.frame for c in cells if c.benchmark == benchmark}, key=lambda f: list(FRAME_LABELS).index(f))
        for frame in frames:
            selected = [c for c in cells if c.benchmark == benchmark and c.frame == frame]
            if len(frames) > 1:
                lines += [f"#### {FRAME_LABELS[frame]}", ""]
            lines += _cell_table(selected)
            lines.append("")

    lines += [
        "## Relation-wise macro accuracy",
        "",
        "These are macro-averages over parse-valid models: each model contributes equally, then examples are "
        "averaged within relation. ScanNet's label `back` is normalized to `behind` here. This table is useful for "
        "finding label-specific biases, but the per-model confusion plots should be consulted before making a claim.",
        "",
    ]
    lines += markdown_table(
        ["Benchmark", "Frame", "Ground-truth relation", "Mean dir. %", "Mean obj. %", "Δ pp"],
        _relation_table(cells),
    )

    invalid_text = ", ".join(
        f"{MODEL_LABELS[c.model]} on {c.benchmark}/{FRAME_LABELS[c.frame]} "
        f"(D={pct(c.direction_parse_rate)}%, O={pct(c.object_parse_rate)}%)"
        for c in invalid
    )
    lines += [
        "",
        "## Data-quality and interpretation notes",
        "",
        f"- Cells excluded from macro summaries by the parse rule: {invalid_text}.",
        "- LongVA's near-zero values in these saved runs primarily reflect output-format/parsing failure and must not "
        "be presented as clean evidence of zero spatial competence. Re-run with a compatible prompt/parser before a "
        "camera-ready comparison.",
        "- COMFORT's Qwen2.5-VL-SAT run has a lower overall parse rate than most other non-LongVA runs; report "
        "parse-aware results or re-run it before drawing a close comparison.",
        "- Some ScanNet object-answer parse rates are materially below the corresponding direction rates. The raw "
        "accuracy therefore mixes reasoning and instruction-following. Include parse success beside accuracy and, "
        "ideally, add accuracy conditional on successful parsing.",
        "- Kubric's authored object-answer candidate set does not always provide a direction label for every distractor. "
        "This does **not** affect gold accuracy or paired transitions, but it leaves some predicted-object-to-direction "
        "entries unmapped in the diagnostic confusion plots.",
        "- `front`/`behind` semantics differ by coordinate frame. Camera-relative uses the camera/image basis; "
        "object-relative and object-facing-camera conditions use the designated object's orientation. Do not pool these "
        "frames without naming the transformation being tested.",
        "- The paired comparison isolates answer format only to the extent that the paired prompts have equivalent "
        "language and candidate difficulty. Object candidates may carry different perceptual or lexical difficulty than "
        "direction labels, so describe this as answer-representation sensitivity rather than a pure latent-geometry probe.",
        "- Per-model McNemar tests are paired and appropriate for binary outcomes, but the many model/frame/relation "
        "comparisons require multiplicity correction (e.g. Holm) for confirmatory claims.",
        "",
        "## Recommended analyses before the final paper",
        "",
        "1. Report 95% paired bootstrap confidence intervals for object−direction accuracy, resampling source pairs "
        "(and scenes when scene IDs induce clustering).",
        "2. Apply Holm correction to the exact McNemar tests, or fit a mixed-effects logistic model with answer format, "
        "coordinate frame, relation, and model as effects and source pair/scene as random intercepts.",
        "3. Report three quantities together: raw accuracy, parse success, and accuracy conditional on successful parsing.",
        "4. Quantify inverse consistency: after mapping an object prediction back to a direction, measure whether it agrees "
        "with the direct direction prediction, separately from whether either is correct.",
        "5. Analyze direction confusion matrices for left↔right and front↔behind swaps. Compare camera and object bases to "
        "test whether errors follow a systematic basis inversion rather than random guessing.",
        "6. Stratify by scene, object class, target/reference visibility, distance, occlusion, and distractor similarity where "
        "metadata permit. This can separate coordinate-transform errors from object-recognition failures.",
        "7. Add wording/order controls: permute option order, use synonymous direction phrasing, swap target/reference order, "
        "and repeat with direct free-form direction generation. This tests whether the effect is tied to labels or geometry.",
        "8. Re-run failed/low-parse model interfaces and retain a frozen parser version. Treat parser changes as part of the "
        "evaluation protocol, not post-hoc data cleaning.",
        "",
        "## Suggested experiment-section outline",
        "",
        "1. **Question.** Does changing only the answer representation expose spatial knowledge that is hidden by "
        "direction-label generation, and does this depend on the egocentric basis?",
        "2. **Paired construction.** Explain the two formats, four-way candidate sets, shared source relation, and coordinate frames.",
        "3. **Datasets and models.** Use the inventory and model-identifier tables above; state deterministic decoding and chance level.",
        "4. **Metrics.** Lead with paired object−direction difference and transition counts, with accuracy and parse rate as context.",
        "5. **Results.** Present one main table of model accuracies/deltas, then a frame × relation figure using the existing plots.",
        "6. **Interpretation.** Emphasize heterogeneous format effects and basis-dependent errors; avoid claiming that object "
        "answers recover latent direction knowledge without the inverse-consistency analysis.",
        "7. **Limitations.** Discuss parse failures, candidate-set asymmetry, dataset size (especially 119 3DSRBench pairs), "
        "and multiple comparisons.",
        "",
        "## Provenance and reusable artifacts",
        "",
        "- Task definitions: `lmms_eval/tasks/3dsrbench_custom/3dsrbench_direction_object.yaml`, "
        "`lmms_eval/tasks/comfort_oriented_3d_direction_object/comfort_oriented_3d_direction_object.yaml`, "
        "`lmms_eval/tasks/kubric_movi_a_direction_object/kubric_movi_a_direction_object_relative_direction.yaml`, "
        "and `lmms_eval/tasks/scannet_camera_basis/scannet_basis_object_direction.yaml`.",
        "- Raw per-example evidence: `outputs/final_*/<model>/submissions/*.json`.",
        "- COMFORT confusion data/figures: "
        "`outputs/final_comfort_oriented_3d_direction_object/<model>/submissions/*_direction_plots/`.",
        "- Kubric summaries and figures: `outputs/final_kubric_movi_a_direction_object_relative_direction/<model>/analysis/`.",
        "- ScanNet summaries, CSVs, and figures: `outputs/final_scannet_basis_object_direction/<model>/analysis/`.",
        "- 3DSRBench paired-outcome figures: `outputs/final_3dsrbench_direction_object/<model>/*paired_outcomes.png`.",
        "- Analysis tools: `tools/analyze_comfort_direction_object_submission.py`, "
        "`tools/analyze_kubric_direction_object_submission.py`, "
        "`tools/analyze_scannet_basis_object_direction_submission.py`, and "
        "`tools/analyze_3dsrbench_direction_object_submission.py`.",
        "- This document can be regenerated with `python tools/build_ego_basis_paper_context.py`.",
        "",
        "## Paper-writing guardrails",
        "",
        "Numbers in this file are descriptive results from the saved runs. A drafting model should not invent missing "
        "confidence intervals, significance corrections, training details, hardware details, or causal explanations. "
        "Any claim of significance should be regenerated from the raw submissions using a predeclared correction. "
        "Use 'associated with' or 'shows a difference' for observational comparisons, and distinguish answer-format "
        "sensitivity from coordinate-frame transformation ability.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cells = collect()
    REPORT.write_text(build(cells), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
