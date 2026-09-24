import types
import unittest

import torch

from lmms_eval import models
from lmms_eval.api.instance import GenerationResult
from lmms_eval.models.chat.internvideo3 import InternVideo3


class _FakeInputs(dict):
    pass


class _FakeProcessor:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return _FakeInputs(input_ids=torch.tensor([[10, 11]]))

    def batch_decode(self, token_ids, **kwargs):
        return ["B"]


class _FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 12]])


class TestInternVideo3(unittest.TestCase):
    def test_registered_as_chat_model(self):
        resolved = models.MODEL_REGISTRY_V2.resolve("internvideo3")

        self.assertEqual(resolved.model_type, "chat")
        self.assertIs(
            models.MODEL_REGISTRY_V2.get_model_class("internvideo3"),
            InternVideo3,
        )

    def _model(self):
        model = InternVideo3.__new__(InternVideo3)
        model.processor = _FakeProcessor()
        model._model = _FakeModel()
        model._input_device = torch.device("cpu")
        model._rank = 0
        model._world_size = 1
        model.batch_size_per_gpu = 1
        model.use_cache = True
        model.fps = 4.0
        model.max_num_frames = 256
        model.min_pixels = 262144
        model.max_pixels = 1048576
        model.max_new_tokens = 1024
        model.system_prompt = None
        model.task_dict = {"demo": {"test": [{"id": 0}]}}
        model.cache_hook = types.SimpleNamespace(
            add_partial=lambda *args, **kwargs: None
        )
        return model

    def test_generate_uses_remote_processor_chat_template(self):
        model = self._model()
        request = types.SimpleNamespace(
            args=(
                "unused context",
                lambda doc: [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "url": "demo.mp4"},
                            {"type": "text", "text": "Choose an option."},
                        ],
                    }
                ],
                {"max_new_tokens": 16, "temperature": 0.0},
                0,
                "demo",
                "test",
            )
        )

        results = model.generate_until([request])

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], GenerationResult)
        self.assertEqual(results[0].text, "B")
        self.assertEqual(results[0].token_counts.output_tokens, 1)

        messages, template_kwargs = model.processor.template_calls[0]
        video = messages[0]["content"][0]
        self.assertEqual(video["video"], "demo.mp4")
        self.assertEqual(video["fps"], 4.0)
        self.assertEqual(video["min_pixels"], 262144)
        self.assertEqual(video["max_pixels"], 1048576)
        self.assertTrue(template_kwargs["tokenize"])
        self.assertTrue(template_kwargs["add_generation_prompt"])
        self.assertTrue(template_kwargs["return_dict"])
        self.assertEqual(template_kwargs["fps"], 4.0)
        self.assertEqual(model.model.generate_kwargs["max_new_tokens"], 16)
        self.assertFalse(model.model.generate_kwargs["do_sample"])

    def test_generation_kwargs_do_not_mutate_task_config(self):
        model = self._model()
        raw = {
            "until": ["stop"],
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 12,
        }

        parsed = model._generation_kwargs(raw)

        self.assertEqual(raw["until"], ["stop"])
        self.assertTrue(parsed["do_sample"])
        self.assertEqual(parsed["temperature"], 0.7)
        self.assertEqual(parsed["top_p"], 0.8)
        self.assertNotIn("until", parsed)


if __name__ == "__main__":
    unittest.main()
