#!/bin/bash
#PBS -N lmms-vsi
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:host=cvml06

# Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval

conda activate lmms

python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-2B-Instruct,max_num_frames=64 \
  --tasks vsibench \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/vsibench