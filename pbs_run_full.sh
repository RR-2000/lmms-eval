#!/bin/bash
#PBS -N lmms-Synth_Q3
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=cvml03

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi

# vsibench_debiased, 3dsrbench, cv_bench_2d, mmsi_bench, stibench, vsibench_object_appearance_order, 
# vsibench_baseline_bbox_object_size_estimation
# vsibench_bbox_object_counting
# 3dsrbench_parquet
# embspatial

# qwen3_vl_experiments, Qwen/Qwen3-VL-4B-Instruct
# qwen2_5_vl, rayruiyang/VST-7B-RL
# internvl3_5, OpenGVLab/InternVL3_5-4B
# transformers=5.5.4, transformers<5

model=qwen3_vl_experiments
model_weights=Qwen/Qwen3-VL-4B-Instruct
task=kubric_movi_a #3dsrbench_parquet, kubric_movi_a
conda activate lmms

LMMS_MASK_IMAGE=1 \
LMMS_EVAL_INCLUDE_GT_HELP_TEXT="0" \
  python -m lmms_eval \
  --model $model \
  --model_args max_num_frames=32,pretrained="$model_weights" \
  --tasks $task \
  --batch_size 1 \
  --limit -1 \
  --output_path "/home/ramanathan/VLM/lmms-eval/outputs/${task}_GT_0_Blank"

LMMS_MASK_IMAGE=1 \
LMMS_EVAL_INCLUDE_GT_HELP_TEXT="1" \
  python -m lmms_eval \
  --model $model \
  --model_args max_num_frames=32,pretrained="$model_weights" \
  --tasks $task \
  --batch_size 1 \
  --limit -1 \
  --output_path "/home/ramanathan/VLM/lmms-eval/outputs/${task}_GT_1_Blank"

# LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_3dsr_split/qwen3_4B_GT_0_Blank LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
# LMMS_EVAL_INCLUDE_LOCATION_TEXT=0 \
for gt_help_text in $(seq 2 8); do
  LMMS_MASK_IMAGE=0 \
  LMMS_EVAL_INCLUDE_GT_HELP_TEXT="${gt_help_text}" \
    python -m lmms_eval \
    --model $model \
    --model_args max_num_frames=32,pretrained="$model_weights" \
    --tasks $task \
    --batch_size 1 \
    --limit -1 \
    --output_path "/home/ramanathan/VLM/lmms-eval/outputs/${task}_GT_${gt_help_text}"

  if [ $gt_help_text -eq 0 ] || [ $gt_help_text -eq 1 ] || [ $gt_help_text -eq 4 ]; then
    # Also run with masked images for GT help text = 0
    LMMS_MASK_IMAGE=1 \
    LMMS_EVAL_INCLUDE_GT_HELP_TEXT="${gt_help_text}" \
      python -m lmms_eval \
      --model $model \
      --model_args max_num_frames=32,pretrained="$model_weights" \
      --tasks $task \
      --batch_size 1 \
      --limit -1 \
      --output_path "/home/ramanathan/VLM/lmms-eval/outputs/${task}_GT_${gt_help_text}_Blank"
  fi
done