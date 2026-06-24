from typing import Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleSimple


@register_model("cosmos3")
class Cosmos3(OpenAICompatibleSimple):
    """
    Cosmos 3 Reasoner adapter backed by an OpenAI-compatible server.

    This targets the supported Cosmos deployment surfaces (vLLM / NIM) instead of
    trying to load the Cosmos checkpoint directly inside lmms-eval.
    """

    DEFAULT_REASONING_PROMPT = (
        "Answer the question using the following format:\n\n"
        "<think>\n"
        "Your reasoning.\n"
        "</think>\n\n"
        "Write your final answer immediately after the </think> tag."
    )

    def __init__(
        self,
        model_version: str = "nvidia/Cosmos3-Nano",
        model: Optional[str] = None,
        base_url: Optional[str] = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = "EMPTY",
        max_frames_num: int = 32,
        video_fps: Optional[float] = 4.0,
        batch_size: int = 8,
        num_concurrent: int = 8,
        enable_thinking: bool = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        if model is not None:
            model_version = model

        super().__init__(
            model_version=model_version,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_frames_num=max_frames_num,
            video_fps=video_fps,
            batch_size=batch_size,
            num_concurrent=num_concurrent,
            **kwargs,
        )

        self.enable_thinking = bool(enable_thinking)
        self.reasoning_prompt = (
            reasoning_prompt.replace("\\n", "\n")
            if reasoning_prompt
            else (self.DEFAULT_REASONING_PROMPT if self.enable_thinking else None)
        )
