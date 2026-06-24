export HF_HOME="~/.cache/huggingface"
# Start a Cosmos3 Reasoner server first, for example with vLLM:
#   vllm serve nvidia/Cosmos3-Nano \
#     --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
#     --async-scheduling \
#     --allowed-local-media-path / \
#     --media-io-kwargs '{"video": {"num_frames": -1}}' \
#     --port 8000

MODEL='nvidia/Cosmos3-Nano'
BASE_URL='http://127.0.0.1:8000/v1'
TASKS='3dsrbench_parquet'
MAX_NUM_FRAMES=32
BATCH_SIZE=8
MAX_NEW_TOKENS=1024

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model cosmos3 \
    --model_args model_version=$MODEL,base_url=$BASE_URL,api_key=EMPTY,max_frames_num=$MAX_NUM_FRAMES,video_fps=4,num_concurrent=8,enable_thinking=false \
    --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
    --tasks $TASKS \
    --log_samples \
    --output_path ./logs/ \
    --batch_size $BATCH_SIZE
