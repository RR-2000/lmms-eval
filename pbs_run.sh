#!/bin/bash
#PBS -N lmms-Synth_BBox
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
# 3dsrbench_parquet
# embspatial

# qwen3_vl_experiments, Qwen/Qwen3-VL-4B-Instruct
# qwen2_5_vl, rayruiyang/VST-7B-RL
# internvl3_5, OpenGVLab/InternVL3_5-4B
# transformers=5.5.4, transformers<5

# LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR=./experiment_artifacts_3dsr_split/qwen3_4B_GT_0_Blank LMMS_EVAL_EXPERIMENTS_SAVE_ATTN=1 \
# LMMS_EVAL_INCLUDE_LOCATION_TEXT=0 \
# LMMS_MASK_IMAGE=1 \
# LMMS_EVAL_INCLUDE_GT_HELP_TEXT=4 \
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks kubric_movi_a_bbox_pred \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_bbox_pred

# python tools/build_3dsr_prompt_variants_dataset.py  \
#   outputs/3dsrbench_4B_GT_4_Blank/submissions/3dsrbench_predictions_qwen3_vl_experiments.json  \
#   --model Qwen/Qwen3-VL-8B-Instruct \

# export HF_HOME="~/.cache/huggingface"
# # Start a Cosmos3 Reasoner server separately before running this eval.
# # Example:
# #   vllm serve nvidia/Cosmos3-Nano \
# #     --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
# #     --async-scheduling \
# #     --allowed-local-media-path / \
# #     --media-io-kwargs '{"video": {"num_frames": -1}}' \
# #     --port 8000

# MODEL='nvidia/Cosmos3-Nano'
# BASE_URL='http://127.0.0.1:8000/v1'
# TASKS='3dsrbench_parquet'
# MAX_NUM_FRAMES=32
# BATCH_SIZE=8
# MAX_NEW_TOKENS=1024

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model cosmos3 \
#     --model_args model_version=$MODEL,base_url=$BASE_URL,api_key=EMPTY,max_frames_num=$MAX_NUM_FRAMES,video_fps=4,num_concurrent=8,enable_thinking=false \
#     --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
#     --tasks $TASKS \
#     --log_samples \
#     --output_path ./logs/ \
#     --batch_size $BATCH_SIZE
