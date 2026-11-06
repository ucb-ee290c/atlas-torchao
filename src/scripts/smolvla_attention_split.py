"""Assign SmolVLA's attention q/k/p/v entries to a tower from the order of attention calls.

All of SmolVLA's text and expert attention runs through ONE function (SmolVLMWithExpertModel.
eager_attention_forward), which ref_attention tags ".L<i>" by call order. The layer code
(lerobot smolvlm_with_expert.py, forward) makes the calls in a fixed pattern, with
`every = self_attn_every_n_layers` (2 for smolvla_base) and the VLM prefix and action suffix both present:

    layer l with l % every == 0  -> 1 call: self-attention over VLM + expert tokens together   -> "joint_attn"
    other layers                 -> 2 calls: the VLM's own self-attention, then the expert's
                                    cross-attention over the VLM's keys / values              -> "language", "expert"

So 16 layers give 8 + 8 + 8 = 24 calls. Cross-check: the VLM call of the last layer feeds nothing the action
loss reads, and it is exactly the entry with a zero score. If the call count does not fit the pattern the
entries are left as they are and a warning is printed.

    python src/scripts/smolvla_attention_split.py src/out/fisher/smolvla/smolvla_n32.json   # retag a report in place
"""
import json
import re
import sys
from pathlib import Path

_CALL = re.compile(r"vlm_with_expert\.L(\d+)\.[qkpv]$")


def call_kinds(n_calls, every=2):
    """['joint', 'vlm', 'expert', ...] for each call index, or None if n_calls does not fit the pattern."""
    kinds = []
    layer = 0
    while len(kinds) < n_calls:
        kinds += ["joint"] if layer % every == 0 else ["vlm", "expert"]
        layer += 1
    return kinds if len(kinds) == n_calls else None


def split_joint_attention(layers, every=2):
    """Retag the shared-function attention entries in place: tower language / expert / joint_attn and
    attn_site vlm / expert / joint. Returns {site: n_entries}, or None when nothing could be assigned."""
    hits = {n: int(m.group(1)) for n, r in layers.items()
            if r.get("kind") == "attn" and (m := _CALL.search(n))}
    if not hits:
        return None
    kinds = call_kinds(max(hits.values()) + 1, every)
    if kinds is None:
        print(f"WARNING: {max(hits.values()) + 1} attention calls do not fit the SmolVLA pattern "
              f"(every={every}); leaving the towers unchanged")
        return None
    tower = {"vlm": "language", "expert": "expert", "joint": "joint_attn"}
    count = {}
    for n, i in hits.items():
        layers[n]["tower"], layers[n]["attn_site"] = tower[kinds[i]], kinds[i]
        count[kinds[i]] = count.get(kinds[i], 0) + 1
    return count


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = json.loads(Path(path).read_text())
        c = split_joint_attention(d["layers"])
        if c:
            d["meta"]["attention_split"] = "joint / vlm / expert by call order (src/scripts/smolvla_attention_split.py)"
            Path(path).write_text(json.dumps(d, indent=1))
        print(path, "->", c)
