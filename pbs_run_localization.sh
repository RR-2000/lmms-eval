#!/bin/bash
#PBS -N comfort-bbox
#PBS -l select=1:ncpus=8:ngpus=1:mem=10gb:host=cvml06

set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

# COMFORT_BBOX_DEBUG=1 \
# COMFORT_BBOX_DEBUG_DIR=outputs/comfort_bbox_debug/overlays \
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks comfort_multi_3d_bbox_prediction \
  --limit -1 \
  --batch_size 1 \
  --log_samples \
  --output_path outputs/comfort_bbox/evaluation