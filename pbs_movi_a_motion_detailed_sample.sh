#!/bin/bash
#PBS -N lmms-movi-motion-12f-detailed-160
#PBS -J 0-3
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:host=cvml06

set -euo pipefail

source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
conda activate /home/ramanathan/.conda/envs/lmms

# Select 20 deterministic source questions from each of the eight motion
# subtasks and derive a label-aware 12-frame version of every question before
# expanding it across the complete diagnostic task library. Numeric targets,
# comparison winners, options, prompts, overlays, and scaffold GT are rebuilt
# from the displayed frames. This produces 160 source questions and exactly
# 8,180 scaffolded model calls across the four parallel array elements.
export MOVI_MOTION_SAMPLES_PER_SUBTASK=20
export MOVI_MOTION_DERIVED_NUM_FRAMES=12

# Task trace mode saves, for every expanded condition:
#   - source document, scaffold metadata, exact task and rendered chat prompts
#   - ordered transformed visual inputs as content-addressed PNGs
#   - a contact-sheet preview, model output, parsed answer, and score
# Standard lmms-eval JSONL sample logging is also enabled below.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Normal array submissions use PBS_ARRAY_INDEX.  MOVI_PART_INDEX permits one
# failed part to be resubmitted as an ordinary (non-array) PBS job.
PART_INDEX="${MOVI_PART_INDEX:-${PBS_ARRAY_INDEX:-}}"
if [[ -z "${PART_INDEX}" ]]; then
  echo "No part selected: use the PBS array directive or set MOVI_PART_INDEX=0..3" >&2
  exit 2
fi

case "${PART_INDEX}" in
  0)
    TASKS="kubric_movi_a_motion_calculation_diagnostics,kubric_movi_a_motion_representation_diagnostics"
    PART_NAME="calculation_representation"
    ;;
  1)
    TASKS="kubric_movi_a_motion_oracle_ladder,kubric_movi_a_motion_precision_diagnostics"
    PART_NAME="oracle_precision"
    ;;
  2)
    TASKS="kubric_movi_a_motion_temporal_diagnostics,kubric_movi_a_motion_perception_diagnostics"
    PART_NAME="temporal_perception"
    ;;
  3)
    TASKS="kubric_movi_a_motion_coordinate_diagnostics,kubric_movi_a_motion_robustness_controls"
    PART_NAME="coordinate_robustness"
    ;;
  *)
    echo "Unexpected part index: ${PART_INDEX} (expected 0, 1, 2, or 3)" >&2
    exit 2
    ;;
esac

RUN_ROOT="/home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_motion_12f_detailed_160"
OUTPUT_DIR="${RUN_ROOT}/part_${PART_INDEX}_${PART_NAME}"
export MOVI_MOTION_TRACE_DIR="${OUTPUT_DIR}/trace_bundle"

mkdir -p "${OUTPUT_DIR}"

echo "Motion diagnostic part index: ${PART_INDEX}"
echo "Tasks: ${TASKS}"
echo "Source questions: 160 total (20 per subtask; 158 distinct sequences in the current dataset)"
echo "Derived visual input: 12 label-aware frames per source question"
echo "Expanded scaffold conditions: 8,180"
echo "Results and JSONL samples: ${OUTPUT_DIR}"
echo "Full prompt/visual traces: ${MOVI_MOTION_TRACE_DIR}"
nvidia-smi

python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=12,pretrained="Qwen/Qwen3-VL-8B-Instruct" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit -1 \
  --log_samples \
  --log_samples_suffix detailed_traces \
  --output_path "${OUTPUT_DIR}"
