#!/usr/bin/env bash
# Run every analysis tool created or used for the COMFORT, ScanNet, and Kubric
# experiments developed in this conversation. Missing experiment outputs are
# reported and skipped, so jobs that are still running do not block the rest.

set -uo pipefail

REPO_ROOT="/home/ramanathan/VLM/lmms-eval"
OUTPUTS="$REPO_ROOT/outputs"
CONDA_INIT="/apps/miniconda3/etc/profile.d/conda.sh"

cd "$REPO_ROOT" || exit 1
if [[ -f "$CONDA_INIT" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_INIT"
  conda activate lmms
fi

failures=0
skips=0

run_analysis() {
  local label="$1"
  shift
  echo
  echo "===== $label ====="
  if "$@"; then
    echo "[OK] $label"
  else
    echo "[FAILED] $label" >&2
    failures=$((failures + 1))
  fi
}

skip_analysis() {
  echo
  echo "[SKIPPED] $1"
  skips=$((skips + 1))
}

latest_result() {
  local directory="$1"
  find "$directory" -type f -name '*_results.json' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

# 1. COMFORT GT_HELP modes 0--14, 36, and 37.
for root in \
  "$OUTPUTS/comforter_comfort_direction_object_gt_help_all_variants_debug" \
  "$OUTPUTS/comforter_comfort_direction_object_gt_help_all_variants_debug_8"; do
  if [[ -d "$root" ]]; then
    run_analysis "COMFORT GT_HELP: $(basename "$root")" \
      python tools/summarize_comfort_direction_object_gt_help_experiments.py \
      "$root" --modes 0-14,36-37 --output-dir "$root/summary"
  else
    skip_analysis "COMFORT GT_HELP output is missing: $root"
  fi
done

# 2. COMFORT component diagnostics for every available model size.
for root in "$OUTPUTS/comfort_gt_help_components_4" "$OUTPUTS/comfort_gt_help_components_8"; do
  if [[ -d "$root/submissions" ]]; then
    run_analysis "COMFORT GT_HELP components: $(basename "$root")" \
      python tools/summarize_comfort_gt_help_component_experiments.py \
      "$root" --output-dir "$root/analysis"
  else
    skip_analysis "COMFORT component output is missing: $root"
  fi
done

# 3. COMFORT inverse diagnostics. This automatically picks up binary-axis and
# oracle-ladder results when their output directories are created.
inverse_inputs=()
for root in \
  "$OUTPUTS/comfort_full_map_inversion_8" \
  "$OUTPUTS/comfort_arrow_length_sweep_8" \
  "$OUTPUTS/comfort_map_ablation_8" \
  "$OUTPUTS/comfort_option_permutation_8" \
  "$OUTPUTS/comfort_binary_axis_8" \
  "$OUTPUTS/comfort_oracle_ladder_8"; do
  [[ -d "$root/submissions" ]] && inverse_inputs+=("$root")
done
if ((${#inverse_inputs[@]})); then
  run_analysis "COMFORT inverse diagnostics" \
    python tools/analyze_comfort_inverse_diagnostics.py \
    --inputs "${inverse_inputs[@]}" \
    --output-dir "$OUTPUTS/comfort_inverse_diagnostics_8_analysis"
else
  skip_analysis "No COMFORT inverse-diagnostic submissions were found"
fi

# 4. Matched direction/vector diagnostics. ScanNet is automatically included
# after its currently running evaluation writes submissions.
direction_vector_inputs=()
for root in \
  "$OUTPUTS/direction_vector_diagnostics_comfort_8" \
  "$OUTPUTS/direction_vector_diagnostics_kubric_8" \
  "$OUTPUTS/direction_vector_diagnostics_scannet_8"; do
  [[ -d "$root/submissions" ]] && direction_vector_inputs+=("$root")
done
if ((${#direction_vector_inputs[@]})); then
  run_analysis "Cross-dataset direction/vector diagnostics" \
    python tools/analyze_direction_vector_diagnostics.py \
    --inputs "${direction_vector_inputs[@]}" \
    --output-dir "$OUTPUTS/direction_vector_diagnostics_analysis_8"
else
  skip_analysis "No direction/vector diagnostic submissions were found"
fi

# 5. COMFORT object-frame, camera-frame, and object-vs-direction basis tasks.
comfort_basis_inputs=()
for root in \
  "$OUTPUTS/comfort_multi_3d_object_basis_0" \
  "$OUTPUTS/comfort_multi_3d_camera_basis_0" \
  "$OUTPUTS/comfort_multi_3d_basis_object_direction_0"; do
  [[ -d "$root/submissions" ]] && comfort_basis_inputs+=("$root")
done
if ((${#comfort_basis_inputs[@]})); then
  run_analysis "COMFORT object/camera basis tasks" \
    python tools/plot_comfort_object_basis_results.py \
    --inputs "${comfort_basis_inputs[@]}" \
    --output-dir "$OUTPUTS/comfort_multi_3d_basis_combined_analysis"
else
  skip_analysis "No COMFORT basis submissions were found"
fi

# 6. ScanNet camera/object basis and paired object-vs-direction tasks.
scannet_basis_inputs=()
for root in \
  "$OUTPUTS/scannet_basis_all_8" \
  "$OUTPUTS/scannet_basis_object_direction_8"; do
  [[ -d "$root/submissions" ]] && scannet_basis_inputs+=("$root")
done
if ((${#scannet_basis_inputs[@]})); then
  run_analysis "ScanNet basis tasks" \
    python tools/plot_scannet_basis_results.py \
    --inputs "${scannet_basis_inputs[@]}" \
    --output-dir "$OUTPUTS/scannet_basis_all_8/scannet_basis_analysis"
else
  skip_analysis "ScanNet basis output is missing"
fi

scannet_object_direction_root="$OUTPUTS/scannet_basis_object_direction_8"
if [[ -d "$scannet_object_direction_root/submissions" ]]; then
  run_analysis "ScanNet object-vs-direction detailed analysis" \
    python tools/analyze_scannet_object_direction_submission.py \
    "$scannet_object_direction_root" \
    --output-dir "$scannet_object_direction_root/scannet_object_direction_analysis"
else
  skip_analysis "ScanNet object/direction submissions are missing"
fi

# 7. COMFORT paired direction/object results and answer-vs-GT distributions.
comfort_direction_submission="$OUTPUTS/comfort_direction_object_0/submissions/comfort_direction_object_qwen3_vl_experiments.json"
if [[ -f "$comfort_direction_submission" ]]; then
  run_analysis "COMFORT direction-vs-object outcomes" \
    python tools/analyze_comfort_direction_object_submission.py \
    "$comfort_direction_submission" \
    --output-dir "$OUTPUTS/comfort_direction_object_0/analysis"
  run_analysis "COMFORT answer-vs-GT distributions" \
    python tools/plot_comfort_direction_object_distributions.py \
    "$comfort_direction_submission" \
    --output-dir "$OUTPUTS/comfort_direction_object_0/analysis/direction_distributions"
else
  skip_analysis "COMFORT direction/object submission is missing"
fi

# 8. Kubric paired direction/object experiment.
kubric_direction_submission="$OUTPUTS/kubric_movi_a_direction_object_clean_better_sample_0/submissions/kubric_movi_a_direction_object_qwen3_vl_experiments.json"
if [[ -f "$kubric_direction_submission" ]]; then
  run_analysis "Kubric direction-vs-object outcomes" \
    python tools/analyze_kubric_direction_object_submission.py \
    "$kubric_direction_submission" \
    --output-dir "$OUTPUTS/kubric_movi_a_direction_object_clean_better_sample_0/analysis"
else
  skip_analysis "Kubric direction/object submission is missing"
fi

# 9. COMFORT bbox prediction visualization and IoU summary.
comfort_bbox_submission="$OUTPUTS/comfort_bbox/evaluation/submissions/comfort_multi_3d_bbox_prediction_qwen3_vl_experiments.json"
if [[ -f "$comfort_bbox_submission" ]]; then
  run_analysis "COMFORT bbox predictions" \
    python tools/visualize_comfort_multi_3d_bbox_predictions.py \
    "$comfort_bbox_submission" \
    --output-dir "$OUTPUTS/comfort_bbox/evaluation/submissions/bbox_visualizations"
else
  skip_analysis "COMFORT bbox submission is missing"
fi

# 10. COMFORT reference-orientation viewpoint task.
comfort_reference_root="$OUTPUTS/comfort_reference_orientation_viewpoint_0"
comfort_reference_result="$(latest_result "$comfort_reference_root")"
comfort_reference_submission="$comfort_reference_root/submissions/comfort_viewpoint_qwen3_vl_experiments.json"
if [[ -n "$comfort_reference_result" && -f "$comfort_reference_submission" ]]; then
  run_analysis "COMFORT reference-orientation viewpoint" \
    python tools/plot_comfort_viewpoint_results.py \
    --input "$comfort_reference_result" \
    --samples "$comfort_reference_submission" \
    --task comfort_reference_orientation_viewpoint \
    --output-dir "$comfort_reference_root/analysis"
else
  skip_analysis "COMFORT reference-orientation viewpoint output is incomplete"
fi

# 11. Original COMFORT viewpoint task.
comfort_viewpoint_root="$OUTPUTS/comfort_viewpoint_0"
comfort_viewpoint_result="$(latest_result "$comfort_viewpoint_root")"
comfort_viewpoint_submission="$comfort_viewpoint_root/submissions/comfort_viewpoint_qwen3_vl_experiments.json"
if [[ -n "$comfort_viewpoint_result" && -f "$comfort_viewpoint_submission" ]]; then
  run_analysis "COMFORT viewpoint" \
    python tools/plot_comfort_viewpoint_results.py \
    --input "$comfort_viewpoint_result" \
    --samples "$comfort_viewpoint_submission" \
    --task comfort_viewpoint \
    --output-dir "$comfort_viewpoint_root/analysis"
else
  skip_analysis "COMFORT viewpoint output is incomplete"
fi

# 12. Kubric object-centric looking-back viewpoint task.
kubric_viewpoint_root="$OUTPUTS/kubric_movi_a_object_centric_looking_back_better_sample_0"
kubric_viewpoint_result="$(latest_result "$kubric_viewpoint_root")"
kubric_viewpoint_submission="$kubric_viewpoint_root/submissions/kubric_movi_a_viewpoint_qwen3_vl_experiments.json"
if [[ -n "$kubric_viewpoint_result" && -f "$kubric_viewpoint_submission" ]]; then
  run_analysis "Kubric object-centric viewpoint" \
    python tools/plot_kubric_movi_a_viewpoint_results.py \
    --input "$kubric_viewpoint_result" \
    --submission-input "$kubric_viewpoint_submission" \
    --task kubric_movi_a_object_centric_looking_back_better_sample \
    --output-dir "$kubric_viewpoint_root/analysis"
else
  skip_analysis "Kubric object-centric viewpoint output is incomplete"
fi

# 13. Refresh the prompt/answer catalog from the actual submission records.
run_analysis "Experiment prompt/answer catalog" \
  python tools/build_conversation_experiments_readme.py \
  --repo-root "$REPO_ROOT" \
  --output "$OUTPUTS/EXPERIMENTS_README.md"

echo
echo "===== Analysis run complete ====="
echo "Failures: $failures"
echo "Skipped:  $skips"
if ((failures)); then
  exit 1
fi
