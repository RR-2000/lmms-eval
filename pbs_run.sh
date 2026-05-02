#!/bin/bash
#PBS -N lmms-vsi
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:host=cvml01

# Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval

conda activate lmms

# LMMS_EVAL_EXPERIMENTS_SAVE_MP4=1
# LMMS_EVAL_EXPERIMENTS_SAVE_NPZ=1
LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args pretrained=Qwen/Qwen3-VL-2B-Instruct,max_num_frames=16 \
  --tasks vsibench_debiased \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/vsibench 


# How to enable

# Set LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1
# Optional output dir: LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts/qwen3_vl_experiments
# Or per-request via gen kwargs: save_attention=true and save_attention_dir=...
# What gets written (per sample)

# .../<tag>.json (prompt + answer + gen kwargs + basic shapes)
# .../<tag>.npz (arrays: attn_map_thw and video_uint8_thwc when available)
# .../<tag>_attn.mp4 (attention heatmap clip)
# .../<tag>.mp4 (input video clip, if video frames are available)