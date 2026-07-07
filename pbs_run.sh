#!/bin/bash
#PBS -N lmms-3D_Grounding
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=cvml03

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

# vsibench_debiased, 3dsrbench, cv_bench_2d, mmsi_bench, stibench, vsibench_object_appearance_order, 
# vsibench_baseline_bbox_object_size_estimation
# vsibench_bbox_object_counting
# 3dsrbench_parquet, 3dsrbench_variant, kubric_movi_a, kubric_movi_a_bbox_pred, kubric_movi_a_viewpoint
# embspatial

# qwen3_vl_experiments, Qwen/Qwen3-VL-4B-Instruct
# qwen2_5_vl, rayruiyang/VST-7B-RL
# internvl3_5, OpenGVLab/InternVL3_5-4B
# transformers=5.5.4, transformers<5

# LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_3dsr_split/qwen3_4B_GT_0_Blank LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
# LMMS_EVAL_INCLUDE_LOCATION_TEXT=0 \
LMMS_MASK_IMAGE=0 \
LMMS_EVAL_INCLUDE_GT_HELP_TEXT=0 \
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks kubric_movi_a_viewpoint \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_viewpoint_pred

# python tools/build_3dsr_prompt_variants_dataset.py --input_json /home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_4B_GT_4_Blank/submissions/3dsrbench_predictions_qwen3_vl_experiments.json --source-jsonl /home/ramanathan/data/3DSR/dataset.jsonl