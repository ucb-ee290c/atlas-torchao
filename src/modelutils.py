import torch 
import math 
from transformers import PreTrainedModel

def load_llama(model_id, seqlen=2048, device="cuda") -> PreTrainedModel:
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
    return model.to(device).eval()

def load_smolvla(path: str, device: str = "cuda"):
    """Load SmolVLA in BF16"""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(path)
    return policy.to(device=device, dtype=torch.bfloat16).eval()