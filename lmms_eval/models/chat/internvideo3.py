"""InternVideo3 chat-template backend.

This adapter follows the checkpoint's trust-remote-code quickstart: multimodal
messages are passed directly to ``processor.apply_chat_template`` with
``tokenize=True`` so the bundled InternVideo3 processor owns image/video
loading, temporal sampling, timestamp insertion, and visual token expansion.
"""

from __future__ import annotations

import copy
import time
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils.versions import require_version

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@register_model("internvideo3")
class InternVideo3(lmms):
    """Hugging Face trust-remote-code wrapper for InternVideo3-8B-Instruct."""

    is_simple = False

    def __init__(
        self,
        pretrained: str = "yanziang/InternVideo3-8B-Instruct",
        device: str = "cuda",
        device_map: Optional[str] = None,
        batch_size: Union[int, str] = 1,
        use_cache: bool = True,
        attn_implementation: str = "sdpa",
        torch_dtype: str = "bfloat16",
        fps: Optional[float] = 4.0,
        max_num_frames: int = 256,
        min_pixels: int = 256 * 32 * 32,
        max_pixels: int = 256 * 4 * 32 * 32,
        max_new_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        require_version(
            "transformers>=4.57.3,<5.0.0",
            "InternVideo3 requires transformers>=4.57.3,<5.0.0 because its "
            "remote model code is not compatible with Transformers 5.",
        )
        if kwargs:
            raise TypeError(f"Unexpected model_args for internvideo3: {sorted(kwargs)}")

        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("InternVideo3 currently supports batch_size=1 only")
        if torch_dtype not in _DTYPES:
            raise ValueError(f"torch_dtype must be one of {sorted(_DTYPES)}, got {torch_dtype!r}")
        if max_num_frames < 1:
            raise ValueError("max_num_frames must be positive")
        if min_pixels < 1 or max_pixels < min_pixels:
            raise ValueError("pixel bounds must satisfy 0 < min_pixels <= max_pixels")

        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self._device = torch.device(
            f"cuda:{accelerator.local_process_index}"
            if accelerator.num_processes > 1
            else device
        )
        # One model replica per process by default. Explicit values such as
        # ``auto`` remain available for single-process tensor parallelism.
        self.device_map = device_map or str(self._device)

        load_kwargs = {
            "dtype": _DTYPES[torch_dtype],
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        eval_logger.info("[internvideo3] Loading {}", pretrained)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            **load_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config

        # The official evaluation commands combine an FPS target with a hard
        # frame cap. The processor accepts FPS per request; its max_frames
        # attribute provides the independent cap.
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None:
            video_processor.max_frames = int(max_num_frames)

        self.pretrained = pretrained
        self.use_cache = bool(use_cache)
        self.fps = float(fps) if fps is not None else None
        self.max_num_frames = int(max_num_frames)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = (
            self._resolve_system_prompt(system_prompt.replace("\\n", "\n"))
            if system_prompt
            else None
        )

        self._input_device = self._resolve_input_device()

    def _resolve_input_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return self._device

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def _prepare_messages(self, raw_messages: list) -> list:
        raw_messages = copy.deepcopy(raw_messages)
        if self.system_prompt:
            raw_messages = self._apply_system_prompt(
                raw_messages,
                self.system_prompt,
            )
        messages = ChatMessages(**{"messages": raw_messages})
        video_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        image_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        return messages.to_hf_messages(
            video_kwargs=video_kwargs,
            image_kwargs=image_kwargs,
        )

    def _generation_kwargs(self, raw_kwargs: dict) -> dict:
        user_kwargs = dict(raw_kwargs or {})
        user_kwargs.pop("until", None)
        temperature = float(user_kwargs.pop("temperature", 0.0) or 0.0)
        do_sample = bool(user_kwargs.pop("do_sample", temperature > 0))

        generation_kwargs = {
            "max_new_tokens": int(
                user_kwargs.pop("max_new_tokens", self.max_new_tokens)
            ),
            "num_beams": int(user_kwargs.pop("num_beams", 1)),
            "do_sample": do_sample,
            "use_cache": self.use_cache,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            top_p = user_kwargs.pop("top_p", None)
            if top_p is not None:
                generation_kwargs["top_p"] = float(top_p)
        else:
            user_kwargs.pop("top_p", None)

        # Forward other standard Transformers generation controls supplied by
        # a task while dropping lmms-only multimodal configuration.
        user_kwargs.pop("image_aspect_ratio", None)
        generation_kwargs.update(user_kwargs)
        return generation_kwargs

    def generate_until(
        self,
        requests: List[Instance],
    ) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        total_elapsed_time = 0.0
        total_tokens = 0
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for request in requests:
            _, doc_to_messages, raw_gen_kwargs, doc_id, task, split = request.args
            raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
            messages = self._prepare_messages(raw_messages)
            has_video = any(
                content.get("type") == "video"
                for message in messages
                for content in message.get("content", [])
            )

            # This is the checkpoint's canonical path. In particular, the
            # remote processor loads videos and injects per-frame timestamps.
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if has_video and self.fps is not None:
                # InternVideo3's reference code forwards FPS both in the media
                # item and at template-processing time.
                template_kwargs["fps"] = self.fps
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
            inputs = {
                key: value.to(self._input_device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in inputs.items()
            }
            generation_kwargs = self._generation_kwargs(raw_gen_kwargs)

            started_at = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            elapsed = time.time() - started_at

            input_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_length:]
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            output_tokens = int(generated_ids.shape[-1])
            results.append(
                GenerationResult(
                    text=answer,
                    token_counts=TokenCounts(output_tokens=output_tokens),
                )
            )
            self.cache_hook.add_partial(
                "generate_until",
                (messages, raw_gen_kwargs),
                answer,
            )
            total_elapsed_time += elapsed
            total_tokens += output_tokens
            pbar.update(1)

        pbar.close()
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=(
                total_tokens / total_elapsed_time
                if total_elapsed_time > 0
                else 0.0
            ),
            additional_metrics={"rank": self.rank},
        )
        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "InternVideo3 multi-round generation is not implemented"
        )

    def loglikelihood(
        self,
        requests: List[Instance],
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("InternVideo3 loglikelihood is not implemented")
