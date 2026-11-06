import torch 
import math 
from transformers import PreTrainedModel

def load_llama(model_id, seqlen=2048, device=None) -> PreTrainedModel:
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = False                     
    ctx = getattr(config, "max_position_embeddings", None)
    if ctx and seqlen > ctx:
        config.rope_scaling = {"type": "linear", "factor": float(math.ceil(seqlen / ctx))}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation="eager",
    )
    
    for mod in model.modules():
        mod._non_persistent_buffers_set.clear()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()

def load_smolvla(path: str, device: str = None):
    """Load SmolVLA in BF16"""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(path)
    return policy.to(device=device, dtype=torch.bfloat16).eval()