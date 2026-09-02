#!/bin/bash
#PBS -N comfort-direction-object-gt-debug
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:host=cvml10

set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

TASK_NAME="comfort_direction_object_gt_help"
OUTPUT_ROOT="/home/ramanathan/VLM/lmms-eval/outputs/comforter_${TASK_NAME}_all_variants_debug"

for gt_help in 37; do
  variant_dir="${OUTPUT_ROOT}/gt_help_${gt_help}"
  echo "Running ${TASK_NAME} with GT_HELP=${gt_help}"

  GT_HELP_DEBUG=1 \
  GT_HELP_DEBUG_DIR="${variant_dir}/images" \
  GT_HELP="${gt_help}" \
  python -m lmms_eval \
    --model qwen3_vl_experiments \
    --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
    --limit -1 \
    --batch_size 1 \
    --log_samples \
    --tasks "${TASK_NAME}" \
    --output_path "${variant_dir}/evaluation"
done

echo "Finished all GT_HELP variants. Results are under ${OUTPUT_ROOT}"
