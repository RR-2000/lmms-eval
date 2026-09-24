"""LMMS adapter for hongxingli/SpatialLadder-3B."""

from __future__ import annotations

from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple._spatial_video import (
    SpatialVideoAdapter,
    resolve_device_map,
)

Qwen2_5_VLForConditionalGeneration, _has_qwen_model = optional_import("transformers", "Qwen2_5_VLForConditionalGeneration")
AutoModelForImageTextToText, _has_auto_image_text = optional_import("transformers", "AutoModelForImageTextToText")
process_vision_info, _has_qwen_vl_utils = optional_import("qwen_vl_utils", "process_vision_info")


@register_model("spatialladder")
class SpatialLadder(SpatialVideoAdapter):
    model_label = "SpatialLadder"

    def __init__(
        self,
        pretrained: str = "hongxingli/SpatialLadder-3B",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: Union[int, str] = 1,
        max_frames_num: Optional[int] = None,
        reverse_frames: bool = False,
        flip_frames: bool = False,
        random_clip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        if not _has_qwen_vl_utils:
            raise ImportError("SpatialLadder requires qwen-vl-utils. Install it with `pip install qwen-vl-utils`.")

        self.path = pretrained
        model_kwargs = {
            "device_map": resolve_device_map(device_map),
            "trust_remote_code": True,
        }
        if _has_qwen_model:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                **model_kwargs,
            )
        elif _has_auto_image_text:
            self._model = AutoModelForImageTextToText.from_pretrained(self.path, dtype="auto", **model_kwargs)
        else:
            raise ImportError("SpatialLadder requires a transformers version providing Qwen2_5_VLForConditionalGeneration or AutoModelForImageTextToText.")

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.path, use_fast=True, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self._config = self._model.config
        if getattr(self._model.generation_config, "eos_token_id", None) is None:
            self._model.generation_config.eos_token_id = self._tokenizer.eos_token_id
        if getattr(self._model.generation_config, "pad_token_id", None) is None:
            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        self._configure_adapter(
            device=device,
            batch_size=batch_size,
            modality=modality,
            max_frames_num=max_frames_num,
            reverse_frames=reverse_frames,
            flip_frames=flip_frames,
            random_clip=random_clip,
        )
        eval_logger.info(f"Loaded {self.model_label} from {self.path}")

    def _process_vision(self, messages):
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        return image_inputs, video_inputs, video_kwargs

    def _generate_extra_kwargs(self):
        return {"use_cache": True}
