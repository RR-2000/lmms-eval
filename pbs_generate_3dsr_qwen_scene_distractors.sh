#!/bin/bash
#PBS -N 3DSR-QwenDistractors
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb:host=cvml01
#PBS -l walltime=12:00:00

set -eo pipefail
source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
conda activate /home/ramanathan/.conda/envs/APC
cd /home/ramanathan/VLM/lmms-eval
set -u

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/ramanathan/.conda/envs/APC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u tools/generate_3dsr_qwen_scene_distractors.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda:0 \
  --max-new-tokens 256 \
  --output /home/ramanathan/data/3DSR/3dsrbench_qwen3vl_8b_scene_objects.json
