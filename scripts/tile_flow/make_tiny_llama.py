"""2-layer random Llama (hidden 64, 4 heads, 2 kv heads) with spread row scales, plus a tokenizer, for the smoke test.
usage: make_tiny_llama.py OUT_DIR TOKENIZER_DIR     (TOKENIZER_DIR: a model dir holding the tokenizer files to copy)"""
import shutil
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

out, src = sys.argv[1], sys.argv[2].rstrip("/") + "/"
cfg = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=False)
torch.manual_seed(0)
m = LlamaForCausalLM(cfg).to(torch.bfloat16)
# spread the row scales (as tests/test_algo_tile.py::_spread_rows) so 32-channel groups get different exponents
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and n != "lm_head":
            mod.weight.mul_((2.0 ** torch.randint(-3, 4, (mod.out_features, 1), generator=g).float()).to(mod.weight.dtype))
m.save_pretrained(out)
for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(src + f, out)
print("ok")
