"""Shared video inference support for spatial-reasoning model adapters."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def resolve_device_map(device_map: Any) -> Any:
    if isinstance(device_map, str) and device_map in {"cuda", "cuda:0"}:
        return "auto"
    if isinstance(device_map, str) and device_map.startswith("cuda:"):
        return {"": device_map}
    return device_map


def add_external_repo_to_path(
    repo_name: str,
    env_var: str,
    explicit_root: str | None = None,
) -> Path:
    """Resolve a companion model repository and prepend it to ``sys.path``.

    The fallback for ``thinking-in-space`` preserves the layout used by the
    source integration while allowing callers to override it with a model arg
    or environment variable.
    """

    lmms_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(explicit_root).expanduser() if explicit_root else None,
        Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else None,
        lmms_root / repo_name,
        lmms_root.parent / "thinking-in-space" / repo_name,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            resolved = candidate.resolve()
            path = str(resolved)
            if path not in sys.path:
                sys.path.insert(0, path)
            return resolved

    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise ImportError(f"Could not find the {repo_name} repository. Set {env_var} or pass " f"the corresponding repository-root model argument. Checked: {checked}")


def sample_video_frames(
    video_path: str,
    num_frames: int | None = None,
    reverse_frames: bool = False,
    flip_frames: bool = False,
    random_clip: bool = False,
) -> list[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    frame_count = min(32, total_frames) if num_frames is None or num_frames <= 0 else min(num_frames, total_frames)
    if random_clip and total_frames > frame_count:
        start = np.random.randint(0, total_frames - frame_count + 1)
        indices = np.arange(start, start + frame_count, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
    if reverse_frames:
        indices = indices[::-1]

    frames = []
    for index in indices:
        frame = Image.fromarray(vr[int(index)].asnumpy()).convert("RGB")
        if flip_frames:
            frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        frames.append(frame)
    return frames


class SpatialVideoAdapter(lmms):
    """Common single-video generation loop used by the three integrations."""

    model_label = "spatial video model"

    def _configure_adapter(
        self,
        *,
        device: str,
        batch_size: int | str,
        modality: str,
        max_frames_num: int | None,
        reverse_frames: bool,
        flip_frames: bool,
        random_clip: bool,
    ) -> None:
        parsed_batch_size = int(batch_size)
        if parsed_batch_size != 1:
            raise ValueError(f"Batch size must be 1 for {self.model_label}, got {parsed_batch_size}.")
        if modality not in {"image", "video"}:
            raise ValueError(f"Unsupported modality for {self.model_label}: {modality}")

        self.batch_size_per_gpu = parsed_batch_size
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._rank = 0
        self._world_size = 1
        self.modality = modality
        self.max_frames_num = int(max_frames_num) if max_frames_num is not None else None
        self.reverse_frames = to_bool(reverse_frames)
        self.flip_frames = to_bool(flip_frames)
        self.random_clip = to_bool(random_clip)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def _process_vision(self, messages):
        raise NotImplementedError

    def _prepare_inputs(self, inputs, video_inputs, image_inputs):
        return inputs

    def _generate_extra_kwargs(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _flatten(items):
        return [item for group in items for item in group]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        pbar = tqdm(total=len(requests), disable=self.rank != 0, desc="Model Responding")

        for request in requests:
            contexts, request_gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            gen_kwargs = dict(request_gen_kwargs)
            visuals = self._flatten([doc_to_visual(self.task_dict[task][split][doc_id])])

            if not visuals:
                raise ValueError(f"{self.model_label} requires at least one visual input.")

            if self.modality == "image":
                visual_content = [{"type": "image", "image": visual} for visual in visuals]
            else:
                if len(visuals) != 1:
                    raise ValueError(f"{self.model_label} supports exactly one video per request, got {len(visuals)}.")
                video_path = visuals[0]
                if self.reverse_frames or self.flip_frames or self.random_clip:
                    video_value = sample_video_frames(
                        video_path,
                        num_frames=self.max_frames_num,
                        reverse_frames=self.reverse_frames,
                        flip_frames=self.flip_frames,
                        random_clip=self.random_clip,
                    )
                else:
                    video_value = str(video_path)

                video_content = {"type": "video", "video": video_value}
                if self.max_frames_num:
                    video_content["nframes"] = self.max_frames_num
                visual_content = [video_content]

            messages = [
                {
                    "role": "user",
                    "content": visual_content + [{"type": "text", "text": str(contexts)}],
                }
            ]

            gen_kwargs.pop("until", None)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", 64))
            do_sample = to_bool(gen_kwargs.pop("do_sample", False))
            temperature = float(gen_kwargs.pop("temperature", 0.0))

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = self._process_vision(messages)
            processor_kwargs = {
                "text": [text],
                "images": image_inputs or None,
                "videos": video_inputs or None,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_kwargs:
                video_kwargs = dict(video_kwargs)
                if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
                    video_kwargs["fps"] = video_kwargs["fps"][0]
                processor_kwargs.update(video_kwargs)

            inputs = self._processor(**processor_kwargs)
            inputs = self._prepare_inputs(inputs, video_inputs, image_inputs)
            model_device = next(self._model.parameters()).device
            inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}

            generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, **self._generate_extra_kwargs()}
            if do_sample:
                generation_kwargs["temperature"] = temperature
            # Preserve task-supplied generation controls supported by HF generate.
            generation_kwargs.update(gen_kwargs)
            generated_ids = self._model.generate(**inputs, **generation_kwargs)
            generated_ids = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], generated_ids)]
            output_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            responses.append(output_text)
            self.cache_hook.add_partial("generate_until", (contexts, request_gen_kwargs), output_text)
            pbar.update(1)

        pbar.close()
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(f"Loglikelihood is not implemented for {self.model_label}.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError(f"Multi-round generation is not implemented for {self.model_label}.")
