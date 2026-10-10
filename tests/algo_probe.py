"""Run tiny models through every ladder mode and save outputs and state (for tests/test_algo_base.py).

    python tests/algo_probe.py SRC_DIR OUT.pt [ALGO_SPEC]

SRC_DIR is put first on sys.path: the current src/, or an export of the l40s-run-2.1 tree.
Without ALGO_SPEC the profile is applied with quant_modes.quantize (the pre-algo entry point);
with it, through algos.quantize (the plumbing the eval CLIs use).
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
spec = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, src)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import quant_modes as qm  # noqa: E402

torch.set_num_threads(4)
FAST = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
MODES = [m for m in qm.LADDER if m != "bf16"] + ["collab_v0_hw"]


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      attn_implementation="eager", use_cache=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


def conv_net():
    torch.manual_seed(0)
    return ConvNet().to(torch.bfloat16).eval()


def apply(model, mode, skip):
    if spec is None:
        return qm.quantize(model, mode, skip=skip, options=dict(FAST))
    import algos

    return algos.quantize(model, mode, skip=skip, options=dict(FAST), algo=algos.make(spec))


def state(model):
    d = {f"sd.{k}": v for k, v in model.state_dict().items() if torch.is_tensor(v)}
    d.update({f"buf.{k}": v for k, v in model.named_buffers()})
    return {k: v.detach().clone() for k, v in d.items()}


ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
img = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
res = {}
for name, build, inp, skip in (("llama", tiny_llama, ids, ["lm_head"]), ("conv", conv_net, img, [])):
    for mode in ["bf16"] + MODES:
        model = build()
        rep = apply(model, mode, skip) if mode != "bf16" else {}
        with torch.no_grad():
            y = model(inp)
        y = getattr(y, "logits", y)
        res[f"{name}/{mode}"] = {"out": y.detach().clone(), "state": state(model),
                                 "ladder": json.dumps(rep.get("ladder"), sort_keys=True, default=str),
                                 "algo": json.dumps(rep.get("algo"), sort_keys=True, default=str)}
torch.save(res, out)
print("saved", len(res), "runs to", out)
