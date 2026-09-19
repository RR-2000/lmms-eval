#!/bin/bash
#PBS -N lmms-cambrians
#PBS -l select=1:ncpus=6:ngpus=1:mem=32gb:host=cvml01

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
# transformers=5.5.4, transformers<5

model=cambrians
model_weights=nyu-visionx/Cambrian-S-7B
tasks=(3dsrbench_direction_object scannet_basis_object_direction comfort_oriented_3d_direction_object kubric_movi_a_direction_object_relative_direction)
conda activate /home/ramanathan/.conda/envs/lmms3

for task in $tasks; do
  python -m lmms_eval \
    --model $model \
    --model_args pretrained="$model_weights" \
    --tasks $task \
    --batch_size 1 \
    --limit -1 \
    --output_path /home/ramanathan/VLM/lmms-eval/outputs/final_$task/$model
done