"""LMMS adapter for Diankun/Spatial-MLLM checkpoints."""

from __future__ import annotations

import importlib
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLProcessor

from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple._spatial_video import (
    SpatialVideoAdapter,
    add_external_repo_to_path,
    resolve_device_map,
)

process_vision_info, _has_qwen_vl_utils = optional_import("qwen_vl_utils", "process_vision_info")


def _apply_qwen2_5_vl_layout_compatibility(model):
    """Expose pre-4.57 Qwen attributes expected by Spatial-MLLM.

    Spatial-MLLM was written against transformers 4.51, where the text token
    embeddings, vision tower, and RoPE helper were exposed at different levels
    of ``Qwen2_5_VLForConditionalGeneration``. Transformers 4.57 moved them
    under its multimodal base model. Use non-registering aliases so state-dict
    traversal and module placement are not changed.
    """

    base_model = model.model
    if not hasattr(base_model, "embed_tokens"):
        get_input_embeddings = getattr(base_model, "get_input_embeddings", None)
        if get_input_embeddings is None:
            raise AttributeError("The loaded Qwen2.5-VL model does not expose input embeddings.")
        object.__setattr__(base_model, "embed_tokens", get_input_embeddings())

    if not hasattr(model, "visual"):
        visual = getattr(base_model, "visual", None)
        if visual is None:
            raise AttributeError("The loaded Qwen2.5-VL model does not expose a vision tower.")
        object.__setattr__(model, "visual", visual)

    if not hasattr(model, "get_rope_index"):
        get_rope_index = getattr(base_model, "get_rope_index", None)
        if get_rope_index is None:
            raise AttributeError("The loaded Qwen2.5-VL model does not expose get_rope_index().")
        object.__setattr__(model, "get_rope_index", get_rope_index)

    return model


def _prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs, temporal_patch_size=2):
    video_tchw = []
    for video_input in video_inputs or []:
        if isinstance(video_input, torch.Tensor):
            tensor = video_input.float() / 255.0
        elif isinstance(video_input, list) and all(isinstance(image, Image.Image) for image in video_input):
            tensor = torch.stack([torch.as_tensor(np.asarray(image)).permute(2, 0, 1) for image in video_input]).float() / 255.0
        else:
            raise ValueError("Unsupported video input format for Spatial-MLLM.")
        video_tchw.append(tensor)

    image_tchw = []
    for image_input in image_inputs or []:
        if not isinstance(image_input, Image.Image):
            raise ValueError("Unsupported image input format for Spatial-MLLM.")
        # Qwen pads a still image to `temporal_patch_size` identical frames
        # before patchification. Mirror that padding for VGGT so its temporal
        # embeddings align with image_grid_thw and Qwen's image tokens.
        image_tensor = torch.as_tensor(np.asarray(image_input)).permute(2, 0, 1).float() / 255.0
        image_tchw.append(image_tensor.unsqueeze(0).repeat(int(temporal_patch_size), 1, 1, 1))

    batch.update(
        {
            "video_tchw": video_tchw or None,
            "image_tchw": image_tchw or None,
        }
    )
    return batch


@register_model("spatial_mllm")
class SpatialMLLM(SpatialVideoAdapter):
    model_label = "Spatial-MLLM"

    def __init__(
        self,
        pretrained: str = "Diankun/Spatial-MLLM-v1.1-Instruct-135K",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: Union[int, str] = 1,
        attn_implementation: Optional[str] = None,
        max_frames_num: Optional[int] = None,
        reverse_frames: bool = False,
        flip_frames: bool = False,
        random_clip: bool = False,
        spatial_mllm_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        if not _has_qwen_vl_utils:
            raise ImportError("Spatial-MLLM requires qwen-vl-utils. Install it with `pip install qwen-vl-utils`.")

        add_external_repo_to_path("Spatial-MLLM", "SPATIAL_MLLM_ROOT", spatial_mllm_root)
        try:
            model_module = importlib.import_module("src.qwenvl.model.spatial_mllm")
            config_class = model_module.SpatialMLLMConfig
            model_class = model_module.SpatialMLLMForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise ImportError("Spatial-MLLM requires its source repository. Set SPATIAL_MLLM_ROOT or pass spatial_mllm_root=/path/to/Spatial-MLLM.") from exc

        self.path = pretrained
        config = config_class.from_pretrained(self.path)
        load_kwargs = {
            "config": config,
            "torch_dtype": "auto",
            "device_map": resolve_device_map(device_map),
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self._model = model_class.from_pretrained(self.path, **load_kwargs).eval()
        _apply_qwen2_5_vl_layout_compatibility(self._model)
        self._processor = Qwen2_5_VLProcessor.from_pretrained(self.path, use_fast=False)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, padding_side="left")
        self._config = config
        self._configure_adapter(
            device=device,
            batch_size=batch_size,
            modality=modality,
            max_frames_num=max_frames_num,
            reverse_frames=reverse_frames,
            flip_frames=flip_frames,
            random_clip=random_clip,
        )

    def _process_vision(self, messages):
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs, {}

    def _prepare_inputs(self, inputs, video_inputs, image_inputs):
        temporal_patch_size = self._config.vision_config.temporal_patch_size
        inputs = _prepare_spatial_mllm_inputs(
            inputs,
            video_inputs,
            image_inputs,
            temporal_patch_size=temporal_patch_size,
        )
        inputs.pop("mm_token_type_ids", None)
        model_device = next(self._model.parameters()).device
        for key in ("image_tchw", "video_tchw"):
            if inputs.get(key) is not None:
                inputs[key] = [tensor.to(model_device) for tensor in inputs[key]]
        return inputs
