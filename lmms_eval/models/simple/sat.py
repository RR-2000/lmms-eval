"""LMMS adapter for the SAT Qwen2.5-VL spatial reasoning model."""

from __future__ import annotations

from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple import _spatial_video as spatial_video

Qwen2_5_VLForConditionalGeneration, _has_qwen_model = optional_import("transformers", "Qwen2_5_VLForConditionalGeneration")
AutoModelForImageTextToText, _has_auto_image_text = optional_import("transformers", "AutoModelForImageTextToText")
process_vision_info, _has_qwen_vl_utils = optional_import("qwen_vl_utils", "process_vision_info")
Qwen2_5_VLProcessor, _has_qwen_processor = optional_import("transformers", "Qwen2_5_VLProcessor")
Qwen2VLImageProcessor, _has_qwen_image_processor = optional_import("transformers", "Qwen2VLImageProcessor")
Qwen2VLVideoProcessor, _has_qwen_video_processor = optional_import("transformers", "Qwen2VLVideoProcessor")


def _load_sat_processor(pretrained: str, min_pixels: int, max_pixels: int):
    """Load SAT's processor, repairing its invalid image-processor metadata.

    The checkpoint currently declares ``Qwen2_5_VLImageProcessor``, but the
    actual Transformers class used by Qwen2.5-VL is
    ``Qwen2VLImageProcessor``.  Prefer the normal auto path so this workaround
    naturally becomes dormant if the upstream checkpoint is corrected.
    """

    processor_kwargs = {
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "trust_remote_code": True,
        "use_fast": False,
    }
    try:
        processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        return processor, processor.tokenizer
    except ValueError as exc:
        if "Unrecognized image processor" not in str(exc):
            raise

    if not (_has_qwen_processor and _has_qwen_image_processor and _has_qwen_video_processor):
        raise ImportError("SAT's processor metadata requires Qwen2.5-VL processor support from a newer transformers version.")

    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        pretrained,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    video_processor = Qwen2VLVideoProcessor.from_pretrained(
        pretrained,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
    return processor, tokenizer


@register_model("sat")
class SAT(spatial_video.SpatialVideoAdapter):
    """Run the merged ``array/Qwen2.5-VL-SAT`` checkpoint in LMMS-Eval."""

    model_label = "SAT"

    def __init__(
        self,
        pretrained: str = "array/Qwen2.5-VL-SAT",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: Union[int, str] = 1,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_frames_num: Optional[int] = 32,
        reverse_frames: bool = False,
        flip_frames: bool = False,
        random_clip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        if not _has_qwen_vl_utils:
            raise ImportError("SAT requires qwen-vl-utils. Install it with `pip install qwen-vl-utils[decord]==0.0.8`.")

        valid_attention = {None, "eager", "sdpa", "flash_attention_2"}
        if attn_implementation not in valid_attention:
            raise ValueError(f"attn_implementation must be one of {valid_attention}, got {attn_implementation!r}")

        self.path = pretrained
        model_kwargs = {
            "device_map": spatial_video.resolve_device_map(device_map),
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        if _has_qwen_model:
            model_class = Qwen2_5_VLForConditionalGeneration
        elif _has_auto_image_text:
            model_class = AutoModelForImageTextToText
        else:
            raise ImportError("SAT requires a transformers version with Qwen2_5_VLForConditionalGeneration or AutoModelForImageTextToText.")

        self._model = model_class.from_pretrained(self.path, **model_kwargs).eval()
        self._processor, self._tokenizer = _load_sat_processor(self.path, min_pixels, max_pixels)
        self._config = self._model.config
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
