#!/bin/bash
#PBS -N lmms-kubric_movi_a_direction_object_clean_better_sample
#PBS -l select=1:ncpus=4:ngpus=1:mem=20gb:host=cvml06

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

# vsibench_debiased, 3dsrbench, cv_bench_2d, mmsi_bench, stibench, vsibench_object_appearance_order, 
# vsibench_baseline_bbox_object_size_estimation, vsibench_object_rel_direction_vector_hard
# vsibench_bbox_object_counting, vsibench_baseline_bbox_object_rel_direction_hard, vsibench_object_rel_direction_hard_inverse
# 3dsrbench_parquet, 3dsrbench_variant, 3dsrbench_direction_object, 3dsrbench_direction_object_direct_answer
# kubric_movi_a, kubric_movi_a_bbox_pred, comfort_viewpoint, comfort_direction_object
# kubric_movi_a_viewpoint, kubric_movi_a_direction_object, kubric_movi_a_direction_vector,
# kubric_movi_a_viewpoint_clean, kubric_movi_a_viewpoint_clean_better_sample, kubric_movi_a_direction_object_direct_answer, 
# kubric_movi_a_direction_object_extended
# kubric_movi_a_object_centric_3d, kubric_movi_a_object_centric_planar, kubric_movi_a_object_centric_linear
# kubric_movi_e_direction_object
# embspatial
# gqa_direction_object
# 3dsrbench_direction_object_multifamily, 3dsrbench_generated_direction_object
# kubric_movi_a_object_centric_looking_back_better_sample
# comfort_reference_orientation_viewpoint
# kubric_movi_a_direction_object_clean_better_sample

# qwen3_vl_experiments, Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-4B-Thinking
# qwen2_5_vl, rayruiyang/VST-7B-RL
# internvl3_5, OpenGVLab/InternVL3_5-4B
# transformers=5.5.4, transformers<5

# LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_3dsr_split/qwen3_4B_GT_0_Blank LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
# LMMS_EVAL_INCLUDE_LOCATION_TEXT=0 \
# LMMS_MASK_IMAGE=0 \
# LMMS_EVAL_INCLUDE_GT_HELP_TEXT=6 \
# LMMS_EVAL_VIEWPOINT_HINT_EXCLUDE_GOLD_ANSWER=1 \
# LMMS_EVAL_DIRECTION_VECTOR_BALANCE_FAMILIES=1 \
# THINKING_FORMAT=1 \
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks kubric_movi_a_direction_object_clean_better_sample \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_direction_object_clean_better_sample_0
  # --output_path /home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_direction_object_direct_answer_0
  # --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_viewpoint_pred_No_GT_6
  # --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_obj_vs_dir_0

# python tools/build_3dsr_prompt_variants_dataset.py --input_json /home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_4B_GT_4_Blank/submissions/3dsrbench_predictions_qwen3_vl_experiments.json --source-jsonl /home/ramanathan/data/3DSR/dataset.jsonl