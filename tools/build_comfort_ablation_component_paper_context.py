#!/usr/bin/env python3
"""Build a paper-writing context file for the COMFORT ablation studies."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
REPORT = OUT / "comfort_ablation_component_analysis_for_paper.md"

INVERSE = OUT / "comfort_inverse_diagnostics_8_analysis" / "summary.json"
COMPONENT = OUT / "comfort_gt_help_components_8" / "comfort_gt_component_analysis" / "summary.json"
POLL = OUT / "comfort_direction_object_polling_8" / "submissions" / "comfort_direction_object_polling_qwen3_vl_experiments.json"
TARGET = OUT / "comfort_direction_object_target_only_gt_polling_8" / "submissions" / "comfort_direction_object_target_only_gt_polling_qwen3_vl_experiments.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}"


def pp(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):+.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def by_fields(rows: list[dict[str, Any]], *fields: str) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(row[field]) for field in fields): row for row in rows}


def paired_condition_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_condition"])].append(row)
    output = []
    for condition, values in sorted(grouped.items(), key=lambda item: float(item[0]) if item[0].replace(".", "", 1).isdigit() else item[0]):
        output.append(
            [
                condition,
                str(sum(int(v["paired_count"]) for v in values)),
                pct(sum(float(v["both_correct"]) for v in values) / len(values)),
                pct(sum(float(v["object_only"]) for v in values) / len(values)),
                pct(sum(float(v["direction_only"]) for v in values) / len(values)),
                pct(sum(float(v["both_wrong"]) for v in values) / len(values)),
            ]
        )
    return output


def relation_matrix(rows: list[dict[str, Any]]) -> list[list[str]]:
    indexed = by_fields(rows, "experiment_condition", "answer_format", "relation")
    conditions = sorted(
        {str(row["experiment_condition"]) for row in rows},
        key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value,
    )
    output = []
    for condition in conditions:
        row = [condition]
        for relation in ("left", "right", "front", "behind"):
            d = indexed[(condition, "direction", relation)]["score"]
            o = indexed[(condition, "object", relation)]["score"]
            row.append(f"{pct(d, 1)}/{pct(o, 1)}")
        output.append(row)
    return output


def normalized_confusion(confusion: dict[str, dict[str, int]], columns: list[str]) -> list[list[str]]:
    output = []
    for gold in ("left", "right", "front", "behind"):
        counts = confusion.get(gold, {})
        total = sum(counts.values())
        output.append([gold, *(pct(counts.get(col, 0) / total, 1) if total else "N/A" for col in columns)])
    return output


def build() -> str:
    inv = load(INVERSE)
    components = load(COMPONENT)
    poll = load(POLL)
    target = load(TARGET)
    pa = poll["analysis"]
    ta = target["analysis"]
    poll_accuracy_by_variant = {
        variant: sum(float(row["score"]) for row in poll["records"] if row["variant"] == variant)
        / sum(1 for row in poll["records"] if row["variant"] == variant)
        for variant in ("poll_left", "poll_right", "poll_front", "poll_behind")
    }
    poll_parse_success = sum(float(row["parse_success"]) for row in poll["records"]) / len(poll["records"])

    arrow = by_fields(inv["arrow_length_sweep"], "experiment_condition", "answer_format")
    map_rows = by_fields(inv["map_ablation"], "experiment_condition", "answer_format")

    lines = [
        "# COMFORT ablation and component analysis: paper-writing context",
        "",
        f"Generated on {date.today().isoformat()} from the saved COMFORT diagnostic outputs. This file is an "
        "evidence packet for a conference-paper experiment/ablation section, not polished manuscript prose.",
        "",
        "## Scope and evaluated system",
        "",
        "All results in this packet are for **Qwen/Qwen3-VL-8B-Instruct** through the "
        "`qwen3_vl_experiments` adapter. The studies use the 500-scene COMFORT Multi-3D data and deterministic "
        "decoding. Counts differ because some component tasks omit label-normalized ambiguous scenes, while the "
        "polling studies expand every source question into several prompts.",
        "",
        "The four required result roots are all represented:",
        "",
        "- `outputs/comfort_inverse_diagnostics_8_analysis` — arrow-length, map-component, full-map inversion, and option-order ablations.",
        "- `outputs/comfort_gt_help_components_8` — generation-versus-utilization controls for localization, orientation, symbols, vectors, and projected axes.",
        "- `outputs/comfort_direction_object_polling_8` — direct direction answer plus four object polls per source question.",
        "- `outputs/comfort_direction_object_target_only_gt_polling_8` — four target-versus-none polls with ground-truth short-axis overlays.",
        "",
        "## Executive findings",
        "",
        "1. **Arrow length is a dominant intervention.** Direction accuracy rises from 22.10% at 0.25× the "
        "reference-box diagonal to 91.05% at 1.15× and remains there at 1.50×. Object accuracy starts much higher "
        "(62.00%) and saturates near 90%, so the +39.90 pp object advantage at 0.25× closes and slightly reverses "
        "at long lengths.",
        "2. **Explicitly labeling both objects and directions solves the canonical-map task (100% for both formats),** "
        "whereas removing either side of the mapping causes large, relation-specific failures. The cue—not just the "
        "requested output type—determines which inverse lookup direction is easier.",
        "3. **The model can consume localization and canonical vector semantics but struggles with orientation.** It "
        "names a boxed object at 98.12% and decodes text vectors at 100%, yet predicts front/left arrows in nearly "
        "the opposite direction (mean cosine −0.806/−0.912) and cannot reliably apply supplied projected axes.",
        "4. **Long arrows are much more usable than short arrows for symbol lookup:** 77.50% versus 51.60% (+25.90 pp).",
        "5. **Naive direction-to-object polling does not validate a clean conversion.** Direct direction accuracy is "
        "5.15%; nominal final-object accuracy is 50.24%, but strict success of both stages is only 1.65%, and the "
        "poll-map is never exactly correct. The high final score is dominated by accidental error cancellation.",
        "6. **Target-only polling collapses by over-accepting the target.** Positive-poll accuracy is 99.70%, but "
        "negative-poll accuracy is 2.12%; 99.83% of source questions tie as ambiguous and final direction accuracy "
        "is 0.05%.",
        "",
        "## 1. Inverse diagnostic ablations",
        "",
        "These experiments diagnose why relation→object and object→relation retrieval can behave differently. "
        "The four-choice tasks have a 25% chance level. Parse success is 100% throughout the summarized ablations.",
        "",
        "### 1.1 Full-map inversion",
        "",
        "Each of 500 scenes requires a complete four-edge JSON map. `relation_to_object` uses direction keys and "
        "object values; `object_to_relation` uses object keys and direction values. Edge accuracy scores individual "
        "assignments, while exact accuracy requires all four assignments.",
        "",
    ]
    full_rows = []
    for row in inv["full_map_inversion"]:
        full_rows.append([
            str(row["experiment_condition"]), str(row["mapping_format"]), str(row["count"]),
            pct(row["mapping_edge_accuracy"]), pct(row["mapping_exact_accuracy"]), pct(row["parse_success"]),
        ])
    lines += table(["Cue", "Map format", "N", "Edge acc. %", "Exact acc. %", "Parse %"], full_rows)
    lines += [
        "",
        "The inversion effect changes sign with representation. Long arrows favor object→relation by +46.30 pp "
        "edge accuracy (81.55% vs 35.25%), whereas the named canonical map favors relation→object by +19.95 pp "
        "(90.35% vs 70.40%). Long arrows produce 69.00% exact object-keyed maps but only 0.40% exact "
        "relation-keyed maps. Thus the map is not simply present or absent: its accessibility depends on how the "
        "visual cue aligns with key-versus-value retrieval.",
        "",
        "### 1.2 Arrow-length sweep",
        "",
        "Arrow length is measured as a multiple of the reference bounding-box diagonal; all other rendering and "
        "prompt properties are held fixed. Each cell contains 2,000 questions (500 per relation).",
        "",
    ]
    arrow_rows = []
    for length in ("0.25", "0.45", "0.70", "0.90", "1.15", "1.50"):
        d = arrow[(length, "direction")]
        o = arrow[(length, "object")]
        arrow_rows.append([length, str(d["count"]), pct(d["score"]), pct(o["score"]), pp(o["score"] - d["score"])])
    lines += table(["Arrow scale", "N/format", "Direction %", "Object %", "Object−direction pp"], arrow_rows)
    lines += ["", "Relation cells below are `direction/object` accuracy percentages.", ""]
    lines += table(["Scale", "Left D/O %", "Right D/O %", "Front D/O %", "Behind D/O %"], relation_matrix(inv["arrow_length_sweep_by_relation"]))
    lines += ["", "Paired outcomes are percentages of matched direction/object pairs.", ""]
    lines += table(["Scale", "Pairs", "Both correct %", "Object only %", "Direction only %", "Both wrong %"], paired_condition_rows(inv["arrow_length_sweep_paired"]))
    lines += [
        "",
        "The shortest arrows create especially asymmetric front performance: at 0.25×, direction-front accuracy is "
        "0.40% while object-front accuracy is 89.40%. At 1.15×, all relation-specific direction accuracies are at "
        "least 88.4%. This is consistent with a visibility/pointer-proximity bottleneck, although arrow length also "
        "changes overlap and salience and should not be called a pure semantic-axis manipulation.",
        "",
        "### 1.3 Canonical-map component ablation",
        "",
    ]
    map_out = []
    for condition in sorted({key[0] for key in map_rows}):
        d, o = map_rows[(condition, "direction")], map_rows[(condition, "object")]
        map_out.append([condition, str(d["count"]), pct(d["score"]), pct(o["score"]), pp(o["score"] - d["score"])])
    lines += table(["Condition", "N/format", "Direction %", "Object %", "Object−direction pp"], map_out)
    lines += ["", "Relation cells below are `direction/object` accuracy percentages.", ""]
    lines += table(["Condition", "Left D/O %", "Right D/O %", "Front D/O %", "Behind D/O %"], relation_matrix(inv["map_ablation_by_relation"]))
    lines += ["", "Paired outcomes by condition:", ""]
    lines += table(["Condition", "Pairs", "Both correct %", "Object only %", "Direction only %", "Both wrong %"], paired_condition_rows(inv["map_ablation_paired"]))
    lines += [
        "",
        "Key controlled contrasts: adding direction labels to object names raises direction accuracy from 62.60% to "
        "100% and object accuracy from 76.35% to 100%. Removing the heading marker from object names lowers direction "
        "accuracy by 6.60 pp and object accuracy by 2.05 pp. Rotating a fully labeled map preserves strong overall "
        "performance but exposes a severe object-answer right-relation failure (10.4%), showing that aggregate scores "
        "can hide a label/rotation-specific mapping error. The rotated-unlabeled map is near floor.",
        "",
        "### 1.4 Option-permutation consistency",
        "",
        "Each semantic question is repeated under four cyclic A/B/C/D orderings. Semantic consistency asks whether "
        "all four predictions resolve to the same answer text; all-correct requires that text to be the gold answer.",
        "",
    ]
    perm_rows = [[str(r["answer_format"]), str(r["permutation_index"]), str(r["count"]), pct(r["score"]), pct(r["parse_success"])] for r in inv["option_permutation"]]
    lines += table(["Format", "Permutation", "N", "Accuracy %", "Parse %"], perm_rows)
    lines += [""]
    stab_rows = [[str(r["answer_format"]), str(r["relation"]), str(r["count"]), pct(r["semantic_consistency"]), pct(r["all_correct"])] for r in inv["option_permutation_stability"]]
    lines += table(["Format", "Relation", "Groups", "Semantic consistency %", "All four correct %"], stab_rows)
    lines += [
        "",
        "Direction answers are often semantically stable but stably wrong: consistency ranges from 77.2% to 99.0%, "
        "while all-four-correct ranges from 0.2% to 2.6%. Object answers are less stable (42.0–65.4%) but sometimes "
        "correct across all orders for behind (41.4%) and front (25.8%). This distinguishes a stable spatial "
        "misconception from simple option-position bias.",
        "",
        "**Coverage note:** the inverse-diagnostics design also defines binary-axis and oracle-ladder experiments, "
        "but neither is present in `comfort_inverse_diagnostics_8_analysis/summary.json`, and no corresponding "
        "`*_8` raw output folder was found. Do not imply that those two planned ablations were run.",
        "",
        "## 2. GT_HELP component diagnostics",
        "",
        "Generation tasks ask the model to produce a representation; utilization tasks provide the ground-truth "
        "representation and ask the model to use it. Utilization therefore measures an upper-bound component control, "
        "not a chained pipeline using the model's own generated intermediate output. `Raw accuracy` makes tasks "
        "binary: IoU ≥ 0.5, angular error ≤30°, or exact classification. Primary continuous scores retain IoU/cosine.",
        "",
    ]
    comp_rows = []
    for r in components["experiments"]:
        comp_rows.append([
            str(r["capability"]), str(r["label"]), str(r["component_role"]), str(r["num_records"]),
            str(r["primary_field"]), pct(r["primary_score"]), pct(r["raw_accuracy"]), pct(r["parse_success"]),
        ])
    lines += table(["Capability", "Diagnostic", "Role", "N", "Primary metric", "Primary %", "Raw acc. %", "Parse %"], comp_rows)
    lines += [
        "",
        "Negative arrow/axis cosine is meaningful: it indicates an average direction more than 90° from ground "
        "truth, not a negative accuracy. Front/left arrow starts are nevertheless localized correctly (start scores "
        "98.09% and 98.21%), so the dominant error is arrow direction rather than finding the box center.",
        "",
        "### Generation versus utilization",
        "",
    ]
    pairs = [
        ("BBox", "BBox prediction", "Name boxed object"),
        ("Front arrow", "Predict front arrow", "Use supplied front arrow"),
        ("Left arrow", "Predict left arrow", "Use supplied left arrow"),
        ("Abstract symbols", "Object to symbol", "Symbol to object"),
        ("Direction vector", "Direction to vector", "Vector to direction"),
        ("Projected axes → text", "Predict projected axes", "Text axes applied"),
        ("Projected axes → overlay", "Predict projected axes", "Overlay axes applied"),
    ]
    by_label = {str(r["label"]): r for r in components["experiments"]}
    pair_rows = []
    for label, generation, utilization in pairs:
        g, u = by_label[generation], by_label[utilization]
        pair_rows.append([label, pct(g["raw_accuracy"]), pct(u["raw_accuracy"]), pp(u["raw_accuracy"] - g["raw_accuracy"])])
    lines += table(["Component", "Generation raw %", "Utilization raw %", "Use−generation pp"], pair_rows)
    lines += [
        "",
        "Important class-level checks from `answer_breakdown`: the long-arrow symbol task is much weaker for B "
        "(40.8%) than A/C/D (92.4/91.4/85.4%), and object→symbol has the same B weakness (45.53%). The 8-way "
        "facing-direction diagnostic contains only `down-left` as the gold label in these 500 records, so its 0% "
        "result is a single-class failure, not balanced 8-way accuracy. This dataset property must be disclosed.",
        "",
        "## 3. Four-way direction-to-object polling",
        "",
        f"There are {poll['num_source_questions']:,} source direction questions and {poll['num_records']:,} records: "
        "one direct direction question plus four object polls (left/right/front/behind) for each source. The direct "
        "prediction selects which poll to route through; the selected object from that poll is the nominal converted answer.",
        "",
    ]
    polling_metrics = [
        ("Direct direction accuracy", pa["direct_answer_accuracy"], "Original four-way direction question"),
        ("Mean poll accuracy", 0.259, "Accuracy across all four object polls"),
        ("Left poll accuracy", poll_accuracy_by_variant["poll_left"], "Object selected for requested left side"),
        ("Right poll accuracy", poll_accuracy_by_variant["poll_right"], "Object selected for requested right side"),
        ("Front poll accuracy", poll_accuracy_by_variant["poll_front"], "Object selected for requested front side"),
        ("Behind poll accuracy", poll_accuracy_by_variant["poll_behind"], "Object selected for requested behind side"),
        ("Final polled answer / full pipeline", pa["final_polled_answer_accuracy"], "Routed poll returns the source target, even if both stages are individually wrong"),
        ("Strict two-stage accuracy", pa["strict_two_stage_accuracy"], "Direct direction and chosen poll are both correct"),
        ("Chosen-poll local accuracy", pa["chosen_poll_accuracy"], "Chosen poll is correct for its own requested direction"),
        ("Oracle-direction poll accuracy", pa["oracle_direction_poll_accuracy"], "Poll indexed by the gold direction"),
        ("Pipeline given direct correct", pa["pipeline_accuracy_given_direction_correct"], "Final source target conditional on correct direction"),
        ("Pipeline given direct wrong", pa["pipeline_accuracy_given_direction_wrong"], "Final source target conditional on wrong direction"),
        ("Exact four-poll map", pa["exact_poll_map_accuracy"], "All four polls locally correct"),
        ("Collision-free poll map", pa["collision_free_poll_map_rate"], "Four polls choose four unique objects"),
        ("Mean unique-answer ratio", pa["mean_unique_poll_answer_ratio"], "Unique selected objects divided by four"),
        ("Parse success", poll_parse_success, "Parse rate over direct and poll records"),
    ]
    lines += table(["Metric", "Result %", "Meaning"], [[name, pct(value), meaning] for name, value, meaning in polling_metrics])
    lines += [
        "",
        "The counterintuitive conditional result—51.23% final accuracy when the direct direction is wrong versus "
        "32.04% when it is correct—confirms that `final_polled_answer_accuracy` rewards error cancellation. It must "
        "not be presented as evidence that the two-stage direction estimate is correct. `strict_two_stage_accuracy` "
        "is the appropriate end-to-end correctness measure for validating the proposed conversion.",
        "",
        "### Direct direction confusion (% within gold row)",
        "",
    ]
    lines += table(["Gold", "Pred left", "Pred right", "Pred front", "Pred behind"], normalized_confusion(pa["gold_direction_to_predicted_direction"], ["left", "right", "front", "behind"]))
    lines += [
        "",
        "The direct predictor overwhelmingly maps gold left to right (97.1%), gold right to left (89.2%), gold "
        "front to left (93.2%), and gold behind to right (91.4%). This is structured basis/polarity confusion, not "
        "uniform four-way guessing.",
        "",
        "### Poll semantic consistency across four option permutations",
        "",
    ]
    consistency = pa["semantic_answer_permutation_consistency"]
    lines += table(
        ["Variant", "Groups", "Consistent groups", "Rate %"],
        [[variant, str(v["groups"]), str(v["consistent_groups"]), pct(v["rate"])] for variant, v in consistency.items()],
    )
    lines += [
        "",
        "Consistency falls from 85.2% for the direct direction response to 33.6–67.4% for object polls. The exact "
        "poll map is 0%, only 6.8% of maps are collision-free, and maps contain 2.675 unique objects on average "
        "(66.88% of four), so treating the four polls as a bijective spatial map is unsupported.",
        "",
        "## 4. Target-only ground-truth-axis polling",
        "",
        f"This study contains {target['num_source_questions']:,} source questions and {target['num_records']:,} binary "
        "polls. Each model-visible image contains short labeled axes rendered from ground-truth COMFORT orientation. "
        "For every direction, the model chooses the named target or `None of the above`; no geometry fusion is used. "
        "A direction is returned only when its target-evidence score uniquely exceeds the runner-up by at least 0.1.",
        "",
    ]
    target_metrics = [
        ("All binary polls", ta["target_poll_accuracy"]),
        ("Positive poll (gold direction)", ta["target_positive_accuracy"]),
        ("Negative polls (three non-gold directions)", ta["target_negative_accuracy"]),
        ("Final direction accuracy", ta["final_direction_accuracy"]),
        ("Direction-selection coverage", ta["direction_selection_coverage"]),
        ("Ambiguous direction rate", ta["ambiguous_direction_rate"]),
        ("Exact four-poll set", ta["exact_poll_set_accuracy"]),
        ("Parse success", sum(float(r["parse_success"]) for r in target["records"]) / len(target["records"])),
    ]
    lines += table(["Metric", "Result %"], [[name, pct(value)] for name, value in target_metrics])
    lines += [
        "",
        "The dominant evidence pattern is target-positive for all four directions: 6,531/8,000 source questions "
        "(81.64%). The system therefore has excellent sensitivity to the target in the one positive poll but almost "
        "no specificity on negative polls. This produces 7,986/8,000 ambiguous groups and only four correct final "
        "directions. The mean best-versus-second evidence margin is 0.003125, far below the 0.10 selection threshold. "
        "Ground-truth orientation overlays alone do not repair the decision rule when the binary visual "
        "question is answered affirmatively for every direction.",
        "",
        "### Final direction outcomes (% within gold row)",
        "",
    ]
    lines += table(["Gold", "Pred left", "Pred right", "Pred front", "Pred behind", "Ambiguous"], normalized_confusion(ta["gold_to_predicted_direction"], ["left", "right", "front", "behind", "ambiguous"]))
    lines += [
        "",
        "## 5. Cross-experiment interpretation",
        "",
        "A coherent picture is that Qwen3-VL-8B has strong object recognition/localization consumption and can "
        "decode explicit symbolic vector semantics, but it does not robustly infer or apply an object's projected "
        "egocentric axes. Long, target-reaching arrows and fully labeled maps can bypass this bottleneck. Short axes "
        "are insufficient on their own: the component test drops from 77.5% with long arrows to 51.6% with short "
        "arrows, and target-only binary polling with short ground-truth axes becomes almost universally affirmative.",
        "",
        "The data do **not** support the stronger claim that object-answer polling generally converts an incorrect "
        "direction representation into a correct spatial map. Full-map, permutation, and polling diagnostics show "
        "that inversion behavior is cue- and retrieval-direction-dependent, and the nominal pipeline metric can be "
        "inflated by correlated wrong answers.",
        "",
        "## 6. Recommended paper analyses and controls",
        "",
        "1. Use source-question or scene-clustered bootstrap confidence intervals; prompt-expanded rows are not independent.",
        "2. For paired object/direction conditions, report object-only, direction-only, both-correct, both-wrong, and an exact McNemar test with multiplicity correction.",
        "3. Treat arrow length as a continuous intervention and fit a monotone/saturating response curve; report the length where the object–direction gap crosses zero.",
        "4. Separate sensitivity and specificity in target-only polling. Overall binary accuracy obscures the 99.70%/2.12% positive/negative split.",
        "5. Report strict two-stage accuracy as the conversion headline. Keep final-target accuracy only as an explicitly labeled error-cancellation diagnostic.",
        "6. Rebalance or regenerate the 8-way facing-direction component task; the saved run contains only one gold class.",
        "7. Run the currently missing binary-axis and oracle-ladder ablations before citing conclusions about axis selection versus polarity or the earliest recoverable pipeline stage.",
        "8. Test negative prompts and calibrated/log-probability target evidence so target-only polling can reject the named target in nonmatching directions.",
        "9. Repeat key ablations across models. All results here concern one model, so they establish a detailed case study rather than model-family generality.",
        "",
        "## 7. Provenance",
        "",
        "- Inverse summary and plots: `outputs/comfort_inverse_diagnostics_8_analysis/`.",
        "- Inverse raw submissions: `outputs/comfort_full_map_inversion_8/`, `outputs/comfort_arrow_length_sweep_8/`, `outputs/comfort_map_ablation_8/`, and `outputs/comfort_option_permutation_8/`.",
        "- Component results: `outputs/comfort_gt_help_components_8/comfort_gt_component_analysis/` and `outputs/comfort_gt_help_components_8/submissions/`.",
        "- Four-way polling: `outputs/comfort_direction_object_polling_8/`.",
        "- Target-only polling: `outputs/comfort_direction_object_target_only_gt_polling_8/`.",
        "- Task documentation: `lmms_eval/tasks/comfort_direction_object_inverse_diagnostics/README.md`, `lmms_eval/tasks/comfort_gt_help_components/README.md`, and `lmms_eval/tasks/comfort_direction_object_polling/README.md`.",
        "- Relevant implementations: `lmms_eval/tasks/comfort_direction_object_polling/utils.py` and `target_only_gt_utils.py`.",
        "- Regenerate this packet with `python tools/build_comfort_ablation_component_paper_context.py`.",
        "",
        "## Paper-writing guardrails",
        "",
        "Do not invent confidence intervals, significance claims, unrun ablations, or cross-model generality. Do not "
        "compare raw two-choice and four-choice accuracy without chance adjustment. Negative cosine is not negative "
        "accuracy. Do not call 50.24% final-polled accuracy clean conversion success; the strict two-stage result is "
        "1.65%. Distinguish ground-truth-component utilization from a pipeline consuming model-generated components.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    for required in (INVERSE, COMPONENT, POLL, TARGET):
        if not required.is_file():
            raise FileNotFoundError(required)
    REPORT.write_text(build(), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
