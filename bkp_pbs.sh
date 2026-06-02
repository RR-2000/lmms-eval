#!/bin/bash
#PBS -N lmms-mmsi_corr
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=cvml03

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

# LMMS_EVAL_EXPERIMENTS_SAVE_MP4=1
# LMMS_EVAL_EXPERIMENTS_SAVE_NPZ=1
MMSI_BENCH_DRAW_CORR=1 LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_MMSI_CORR/qwen3 LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=16,pretrained="Qwen/Qwen3-VL-2B-Instruct" \
  --tasks mmsi_bench \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/mmsi_corr

# vsibench_debiased, 3dsrbench, cv_bench_2d, mmsi_bench, stibench, vsibench_object_appearance_order, 
# vsibench_baseline_bbox_object_size_estimation
# vsibench_bbox_object_counting
# How to enable

# Set LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1
# Optional output dir: LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts/qwen3_vl_experiments
# Or per-request via gen kwargs: save_attention=true and save_attention_dir=...
# What gets written (per sample)

# .../<tag>.json (prompt + answer + gen kwargs + basic shapes)
# .../<tag>.npz (arrays: attn_map_thw and video_uint8_thwc when available)
# .../<tag>_attn.mp4 (attention heatmap clip)
# .../<tag>.mp4 (input video clip, if video frames are available)