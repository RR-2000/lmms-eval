#!/bin/bash
#PBS -N lmms-cross-internvl3_5
#PBS -l select=1:ncpus=8:ngpus=1:mem=14gb:host=cvml01

# Activate the Conda environment
source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi

# 3dsrbench_direction_object
# scannet_basis_object_direction
# comfort_oriented_3d_direction_object
# kubric_movi_a_direction_object_relative_direction

# qwen3_vl_experiments, Qwen/Qwen3-VL-8B-Instruct
# llava_onevision2, lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
# cambrians, nyu-visionx/Cambrian-S-7B
# internvl3_5, OpenGVLab/InternVL3_5-8B
# spatial_mllm, Diankun/Spatial-MLLM-v1.1-Instruct-820K
# spatialladder, hongxingli/SpatialLadder-3B
# vst, rayruiyang/VST-7B-RL
# longva, lmms-lab/LongVA-7B
# sat, array/Qwen2.5-VL-SAT
# openai, gpt-5.6-luna
# internvideo3, yanziang/InternVideo3-8B-Instruct
# transformers=5.5.4, transformers<5.0

model=internvl3_5
model_weights=OpenGVLab/InternVL3_5-8B
model_type=gpt-5.6-luna
tasks=(comfort_gt_help_components comfort_full_map_inversion comfort_arrow_length_sweep comfort_map_ablation comfort_option_permutation)
conda activate /home/ramanathan/.conda/envs/lmms2
# ,attn_implementation=sdpa,model=$model_type,
# --gen_kwargs max_new_tokens=4096
for task in "${tasks[@]}"; do
  python -m lmms_eval \
    --model $model \
    --model_args pretrained="$model_weights", \
    --tasks $task \
    --batch_size 1 \
    --limit -1 \
    --gen_kwargs max_new_tokens=4096 \
    --output_path /home/ramanathan/VLM/lmms-eval/outputs/diagnosis_final_$task/$model
done
