from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl_experiments import Qwen3_VL_Experiments as Qwen3_VL_Chat
from lmms_eval.models.simple.qwen3_5 import Qwen3_5 as Qwen3_5_Simple


@register_model("qwen3_5_experiments_chat")
class Qwen3_5_Experiments(Qwen3_VL_Chat, Qwen3_5_Simple):
    """
    Qwen3.5 chat model -- uses Qwen3.5 defaults (via Qwen3_5_Simple)
    and chat generate_until (via Qwen3_VL_Chat).
    """

    pass
