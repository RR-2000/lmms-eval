"""LMMS adapter for rayruiyang/VST-7B-RL."""

from __future__ import annotations

import importlib
from typing import Optional, Union

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple._spatial_video import (
    SpatialVideoAdapter,
    add_external_repo_to_path,
    resolve_device_map,
)


@register_model("vst")
class VST(SpatialVideoAdapter):
    model_label = "VST"

    def __init__(
        self,
        pretrained: str = "rayruiyang/VST-7B-RL",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: Union[int, str] = 1,
        max_frames_num: Optional[int] = None,
        reverse_frames: bool = False,
        flip_frames: bool = False,
        random_clip: bool = False,
        vst_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        add_external_repo_to_path("VST", "VST_ROOT", vst_root)
        try:
            vision_module = importlib.import_module("vst.utils.vision_process")
            self._process_vision_info = vision_module.process_vision_info
        except (ImportError, AttributeError) as exc:
            raise ImportError("VST requires its source repository. Set VST_ROOT or pass vst_root=/path/to/VST.") from exc

        self.path = pretrained
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.path,
            torch_dtype="auto",
            device_map=resolve_device_map(device_map),
            trust_remote_code=True,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
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

    def _process_vision(self, messages):
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(messages, return_video_kwargs=True)
        return image_inputs, video_inputs, video_kwargs
