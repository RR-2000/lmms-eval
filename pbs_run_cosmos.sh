#!/bin/bash
#PBS -N lmms-Cosmos_Test
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=cvml03

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
cd /home/ramanathan/VLM/lmms-eval
nvidia-smi
conda activate lmms

if [ -n "${CONDA_PREFIX:-}" ]; then
  TORCH_LIB_DIR="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib"
  NVIDIA_SITE_PACKAGES_DIR="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
  EXTRA_LD_PATHS="${CONDA_PREFIX}/lib"
  if [ -d "$TORCH_LIB_DIR" ]; then
    EXTRA_LD_PATHS="${EXTRA_LD_PATHS}:${TORCH_LIB_DIR}"
  fi
  if [ -d "$NVIDIA_SITE_PACKAGES_DIR" ]; then
    while IFS= read -r libdir; do
      EXTRA_LD_PATHS="${EXTRA_LD_PATHS}:${libdir}"
    done < <(find "$NVIDIA_SITE_PACKAGES_DIR" -maxdepth 3 -type d \( -name lib -o -name lib64 \) | sort)
  fi
  export LD_LIBRARY_PATH="${EXTRA_LD_PATHS}:${LD_LIBRARY_PATH}"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "${CUDA_VISIBLE_DEVICES}" == GPU-* ]]; then
  UUID_TO_INDEX="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)"
  if [ -n "$UUID_TO_INDEX" ]; then
    NORMALIZED_CUDA_VISIBLE_DEVICES="$(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" UUID_TO_INDEX="$UUID_TO_INDEX" python - <<'PY'
import os

visible = [item.strip() for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
uuid_to_index = {}
for line in os.environ["UUID_TO_INDEX"].splitlines():
    parts = [part.strip() for part in line.split(",", 1)]
    if len(parts) == 2:
        uuid_to_index[parts[1]] = parts[0]

mapped = [uuid_to_index[item] for item in visible if item in uuid_to_index]
print(",".join(mapped))
PY
)"
    if [ -n "$NORMALIZED_CUDA_VISIBLE_DEVICES" ]; then
      export CUDA_VISIBLE_DEVICES="$NORMALIZED_CUDA_VISIBLE_DEVICES"
    fi
  fi
fi

check_vllm_cuda_compatibility() {
  python - <<'PY'
import os
import re
import subprocess
import sys

def parse_major_minor(version: str):
    match = re.search(r"(\d+)\.(\d+)", version or "")
    return (int(match.group(1)), int(match.group(2))) if match else None

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    driver_version = out[0].strip() if out else ""
except Exception:
    driver_version = ""

try:
    import importlib.metadata as md
    runtime_version = None
    for candidate in ("nvidia-cuda-runtime", "nvidia_cuda_runtime", "nvidia-cuda-runtime-cu12"):
        try:
            version = md.version(candidate)
        except Exception:
            continue
        if runtime_version is None:
            runtime_version = version
        if parse_major_minor(version) and parse_major_minor(version)[0] >= 13:
            runtime_version = version
            break
except Exception:
    runtime_version = None

driver = parse_major_minor(driver_version)
runtime = parse_major_minor(runtime_version) if runtime_version else None

# vLLM 0.23.0 in this env is linked against the CUDA 13 runtime package.
# NVIDIA driver 565 exposes CUDA 12.7 support, which is not enough for CUDA 13.
if runtime and runtime[0] >= 13:
    if not driver or driver[0] < 13:
        print(
            "Incompatible CUDA stack for local vLLM startup: "
            f"installed runtime package is {runtime_version}, but node driver is {driver_version or 'unknown'} "
            "(this node advertises CUDA 12.x support, not CUDA 13). "
            "Use a CUDA-12-compatible vLLM environment/wheel, upgrade the node driver, "
            "or connect lmms-eval to an external Cosmos3 server instead.",
            file=sys.stderr,
        )
        sys.exit(1)
PY
}

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
# python -m lmms_eval \
#   --model qwen3_vl_experiments \
#   --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
#   --tasks kubric_movi_a_bbox_pred \
#   --batch_size 1 \
#   --limit -1 \
#   --output_path /home/ramanathan/VLM/lmms-eval/outputs/kubric_movi_a_bbox_pred

# python tools/build_3dsr_prompt_variants_dataset.py  \
#   outputs/3dsrbench_4B_GT_4_Blank/submissions/3dsrbench_predictions_qwen3_vl_experiments.json  \
#   --model Qwen/Qwen3-VL-8B-Instruct \

# export HF_HOME="~/.cache/huggingface"
# # Start a Cosmos3 Reasoner server separately before running this eval.
# # Example:
# #   vllm serve nvidia/Cosmos3-Nano \
# #     --hf-overrides '{"architectures": ["Cosmos3ForConditionalGeneration"]}' \
# #     --async-scheduling \
# #     --allowed-local-media-path / \
# #     --media-io-kwargs '{"video": {"num_frames": -1}}' \
# #     --port 8000

MODEL='nvidia/Cosmos3-Nano'
BASE_URL='http://127.0.0.1:8000/v1'
TASKS='3dsrbench_parquet'
MAX_NUM_FRAMES=32
BATCH_SIZE=8
MAX_NEW_TOKENS=1024
START_COSMOS3_SERVER="${START_COSMOS3_SERVER:-1}"
SERVER_HOST='127.0.0.1'
SERVER_PORT='8000'
COSMOS3_HF_OVERRIDES='{"architectures": ["Cosmos3ForConditionalGeneration"]}'
SERVER_STARTUP_TIMEOUT_SEC="${SERVER_STARTUP_TIMEOUT_SEC:-1800}"

wait_for_openai_server() {
  local url="$1"
  local server_pid="$2"
  local log_path="$3"
  local startup_timeout_sec="$4"
  python - "$url" "$server_pid" "$log_path" "$startup_timeout_sec" <<'PY'
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

url = sys.argv[1].rstrip("/") + "/models"
server_pid = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 0
log_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
startup_timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 1800
deadline = time.time() + startup_timeout_sec
last_error = None


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False

while time.time() < deadline:
    if not process_alive(server_pid):
        log_tail = ""
        if log_path and log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            log_tail = "\n".join(lines[-20:])
        print("Cosmos3 server process exited before becoming ready.", file=sys.stderr)
        if log_tail:
            print(log_tail, file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status < 400:
                print(f"Server ready at {url}")
                sys.exit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)

print(
    f"Timed out waiting for server at {url} after {startup_timeout_sec}s. "
    f"Last error: {last_error}",
    file=sys.stderr,
)
sys.exit(1)
PY
}

if [ "$START_COSMOS3_SERVER" = "1" ]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "Cosmos3 server startup requested, but 'vllm' is not installed in the active environment." >&2
    echo "Install vllm in the 'lmms' conda env or rerun with START_COSMOS3_SERVER=0 and point BASE_URL at an existing server." >&2
    exit 1
  fi
  check_vllm_cuda_compatibility || exit 1
  echo "Starting Cosmos3 vLLM server on ${SERVER_HOST}:${SERVER_PORT}"
  vllm serve "$MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --hf-overrides "$COSMOS3_HF_OVERRIDES" \
    --async-scheduling \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    > cosmos3_server.log 2>&1 &
  SERVER_PID=$!
  trap 'if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
  wait_for_openai_server "$BASE_URL" "$SERVER_PID" "cosmos3_server.log" "$SERVER_STARTUP_TIMEOUT_SEC" || exit 1
else
  echo "Using external Cosmos3 server at ${BASE_URL}"
fi

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model cosmos3 \
    --model_args model_version=$MODEL,base_url=$BASE_URL,api_key=EMPTY,max_frames_num=$MAX_NUM_FRAMES,video_fps=4,num_concurrent=8,enable_thinking=false \
    --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
    --tasks $TASKS \
    --log_samples \
    --output_path ./logs/ \
    --batch_size $BATCH_SIZE \
    --limit 50
