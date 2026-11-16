#!/bin/bash
#PBS -N lmms-movi-motion-12f-full-scale
#PBS -J 0-3
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:host=cvml06

set -euo pipefail

source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
conda activate /home/ramanathan/.conda/envs/lmms

# Run 400 deterministic questions from each of the eight subtasks. Rebuild
# every target and option from 12 label-aware frames, then put a
# perspective-correct scene-unit ruler on every numerical baseline and
# assistance condition. The dedicated visual-scale suite additionally keeps
# a true unassisted baseline and compares corner and object-anchored rulers.
export MOVI_MOTION_SAMPLES_PER_SUBTASK=400
export MOVI_MOTION_DERIVED_NUM_FRAMES=12
export MOVI_MOTION_NUMERIC_SCALE_GUIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    TASKS="kubric_movi_a_motion_coordinate_diagnostics,kubric_movi_a_motion_robustness_controls,kubric_movi_a_motion_visual_scale_assistance"
    PART_NAME="coordinate_robustness_visual_scale"
    ;;
  *)
    echo "Unexpected part index: ${PART_INDEX} (expected 0, 1, 2, or 3)" >&2
    exit 2
    ;;
esac

RUN_ROOT="/home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_motion_12f_full_bbox_scale_guide_400"
OUTPUT_DIR="${RUN_ROOT}/part_${PART_INDEX}_${PART_NAME}"
export MOVI_MOTION_TRACE_DIR="${OUTPUT_DIR}/trace_bundle"
mkdir -p "${OUTPUT_DIR}" "${MOVI_MOTION_TRACE_DIR}"

echo "Motion diagnostic part index: ${PART_INDEX}"
echo "Tasks: ${TASKS}"
echo "Source questions: 3,200 total (400 deterministic questions per subtask)"
echo "Derived visual input: 12 label-aware frames per source question"
echo "Shared aid: perspective-correct scene-unit scale on every numerical condition"
echo "Visual-scale comparison: unassisted vs corner ruler vs object-anchored ruler"
echo "Full library size: 169,600 model calls across all four array elements"
echo "Results and JSONL samples: ${OUTPUT_DIR}"
echo "Full prompt, transformed-image, model-I/O, and scoring traces: ${MOVI_MOTION_TRACE_DIR}"
nvidia-smi

python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=12,pretrained="Qwen/Qwen3-VL-8B-Instruct" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit -1 \
  --log_samples \
  --log_samples_suffix scale_guide \
  --output_path "${OUTPUT_DIR}"
