#!/bin/bash
#PBS -N lmms-kubric_movi_a_motion_diagnostics
#PBS -J 0-3
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:host=cvml07

set -euo pipefail

# Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate /home/ramanathan/.conda/envs/lmms
# conda activate lmms

# export MOVI_MOTION_SAMPLES_PER_SUBTASK=5

# Balance the 40,900 expanded requests across four array jobs:
#   0: 11,700 requests  (calculation 8,500 + representation 3,200)
#   1: 10,400 requests  (oracle ladder 7,200 + precision 3,200)
#   2:  9,800 requests  (temporal 5,600 + perception 4,200)
#   3:  9,000 requests  (coordinate 5,000 + robustness 4,000)
case "${PBS_ARRAY_INDEX}" in
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
    echo "Unexpected PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_motion_diagnostics_full_8/part_${PBS_ARRAY_INDEX}_${PART_NAME}"

echo "PBS array index: ${PBS_ARRAY_INDEX}"
echo "Tasks: ${TASKS}"
echo "Output: ${OUTPUT_DIR}"

python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-8B-Instruct" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit -1 \
  --output_path "${OUTPUT_DIR}"
