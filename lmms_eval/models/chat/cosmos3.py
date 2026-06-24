from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChat
from lmms_eval.models.simple.cosmos3 import Cosmos3 as Cosmos3Simple


@register_model("cosmos3_chat")
class Cosmos3(OpenAICompatibleChat, Cosmos3Simple):
    """
    Cosmos 3 chat adapter backed by an OpenAI-compatible server.
    """

    def __init__(self, **kwargs):
        Cosmos3Simple.__init__(self, **kwargs)
