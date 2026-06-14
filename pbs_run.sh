#!/bin/bash
#PBS -N lmms-3DSR_GT_0_Blank
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=cvml03

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

# LMMS_EVAL_EXPERIMENTS_SAVE_MP4=1
# LMMS_EVAL_EXPERIMENTS_SAVE_NPZ=1
# MMSI_BENCH_DRAW_CORR=1 LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_MMSI_CORR/qwen3 LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 
# LMMS_EVAL_INCLUDE_LOCATION_TEXT=1 \
# python -m lmms_eval \
#   --model qwen3_vl_experiments \
#   --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-2B-Instruct" \
#   --tasks vsibench_bbox_object_counting \
#   --batch_size 1 \
#   --limit -1 \
#   --output_path /home/ramanathan/VLM/lmms-eval/outputs/VSI_text_bbox_object_counting

# vsibench_debiased, 3dsrbench, cv_bench_2d, mmsi_bench, stibench, vsibench_object_appearance_order, 
# vsibench_baseline_bbox_object_size_estimation
# vsibench_bbox_object_counting
# 3dsrbench_parquet
# embspatial

LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_3dsr_split/qwen3_4B_GT_0_Blank LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
LMMS_EVAL_INCLUDE_LOCATION_TEXT=0 \
LMMS_MASK_IMAGE=1 \
LMMS_EVAL_INCLUDE_GT_HELP_TEXT=0 \
  python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks 3dsrbench_parquet \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_4B_GT_0_Blank


# Set LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1
# Optional output dir: LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts/qwen3_vl_experiments
# Or per-request via gen kwargs: save_attention=true and save_attention_dir=...
# What gets written (per sample)

# .../<tag>.json (prompt + answer + gen kwargs + basic shapes)
# .../<tag>.npz (arrays: attn_map_thw and video_uint8_thwc when available)
# .../<tag>_attn.mp4 (attention heatmap clip)
# .../<tag>.mp4 (input video clip, if video frames are available)



# LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_embspatial_vanilla/qwen3 LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
# EMBSPATIAL_BBOX_TEXT=0 \
# EMBSPATIAL_BBOX_OVERLAY=0 \
#   python -m lmms_eval \
#   --model qwen3_vl_experiments \
#   --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-2B-Instruct" \
#   --tasks embspatial \
#   --batch_size 1 \
#   --limit -1 \
#   --output_path /home/ramanathan/VLM/lmms-eval/outputs/embspatial