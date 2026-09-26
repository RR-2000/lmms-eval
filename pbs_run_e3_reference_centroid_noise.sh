#!/bin/bash
#PBS -N lmms-kubric-e3-reference-centroid-noise
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:host=cvml12

source /home/ramanathan/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval || exit 1
nvidia-smi
conda activate /home/ramanathan/.conda/envs/lmms

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KUBRIC_RELATIVE_REPRESENTATION_DEBUG=1 \
KUBRIC_RELATIVE_REPRESENTATION_DEBUG_DIR=/home/ramanathan/VLM/lmms-eval/outputs/kubric_e3_reference_centroid_noise_8/prompt_images \
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-8B-Instruct" \
  --tasks kubric_movi_a_direction_object_relative_direction_axis_overlay_dirclr_style_reference_centroid_noise \
  --batch_size 1 \
  --limit -1 \
  --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_e3_reference_centroid_noise_8
