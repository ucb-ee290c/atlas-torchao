# TinyLlama prefill and decode FX graphs

## TL;DR

`src/fx_export.py` has two new `--boundary` options for causal LMs:

- `causal_lm_prefill` takes the mask and position ids as inputs and returns the K/V cache.
- `causal_lm_decode` takes one token plus a K/V cache.

Both were captured for the full 22-layer TinyLlama (`atlas_v1_awq_tile_static`, 64 tokens), and both pass the exporter's verification.

- **Prefill** is bit-identical to the old graph.
- **Decode** is not bit-identical to a full 65-token forward: the max logit difference is about 1.0 and the top token is the same. The cause hasn't been confirmed.

This was done for Nicolas Reed's request (Merlin PR #63) for a decode graph, and for mask and positions as inputs.

---

## 1. Why

The original graph, `--boundary causal_lm`, takes only `input_ids`. That leads to two problems for the chip compiler:

1. **It traces Hugging Face's mask code.** Hugging Face builds the position ids and causal mask inside the model, so those operations end up in the graph. That includes `arange`, `diff`, `cumsum`, `le`, `index` and the packed-sequence check. None of them depend on the input words.
2. **It has no decode graph.** Generating text token by token needs a graph that takes one token plus the saved attention state (the KV cache). Nicolas was building that graph himself (`examples/atlas/whole-model/unchanged_fx/cache_source.py` in Merlin PR #63).

## 2. The three bundle types

Each bundle is its own folder with its own `model.pt2`, fixtures, verification and manifest. Shapes are fixed at export time, so each phase and length is a separate bundle.

| `--boundary` | Inputs | Outputs | When it runs |
|---|---|---|---|
| `causal_lm` (default, unchanged) | `input_ids [1,S]` | logits `[1,S,V]` | n/a |
| `causal_lm_prefill` | `input_ids [1,S]` int64, `position_ids [1,S]` int64, `attention_mask [1,1,S,S]` bf16 | logits `[1,S,V]`, then K0, V0, …, K21, V21, each `[1,Hkv,S,D]` bf16 | once, on the prompt |
| `causal_lm_decode` | `input_ids [1,1]`, `position_ids [1,1]` (= S), `attention_mask [1,1,1,S+1]`, then K0, V0, …, K21, V21 `[1,Hkv,S,D]` | logits `[1,1,V]`, then K/V of length S+1 | once per new token |

The bundle conventions:

- **S** is `--seqlen`. For decode it is the length of the past cache.
- **The mask** is additive bf16: 0 where attention is allowed, and the bf16 minimum where it is masked.
- **The cache tensors** are stored after RoPE and before GQA expansion. For TinyLlama, Hkv = 4 and D = 64.
- **The order** of inputs and outputs matches Nicolas's `TensorCacheLogits` / `TensorPrefillLogits`.

### How they are used together (host loop)

```
prompt (64 tokens) ─▶ prefill ─▶ logits + K/V (len 64)
                                     │
host picks token 65 ─────────────────┤
                                     ▼
token 65, pos 64, mask [1,1,1,65] + K/V ─▶ decode ─▶ logits + K/V (len 65) ─▶ host picks next ...
```

The host builds the positions and the mask, picks each token and passes the K/V along. The decode bundle above works only for a past length of 64. A later step, with a past of 65 or more, needs its own decode bundle.

## 3. Code changes (`src/fx_export.py`)

All additions are next to `CausalLogits`. The existing boundaries are unchanged.

| Added | What it does |
|---|---|
| `phase_positions_and_mask(input_ids, past_length)` | Builds position ids `past … past+Q-1` and the additive bf16 causal mask on the host, outside the graph. |
| `_AppendTensorCache` | Stands in for Hugging Face's cache object. Each layer's `update()` stores (or appends to) its K/V as plain tensors. `tensors()` returns them in layer order, and raises an error if a layer skipped or repeated its update. |
| `CausalLogitsPhase(model, past_len)` | The new boundary module. Instead of calling `model(input_ids)`, it runs the pieces directly: `embed_tokens`, `rotary_emb(position_ids)`, each decoder layer with the given mask and cache, then `norm` and `lm_head`. This bypasses Hugging Face's top-level forward, which is where the mask code lives. Use `past_len=0` for prefill and `past_len=S` for decode. |
| `causal_phase_cases(...)` | Builds the two fixture cases. A decode case gets its K/V from a real eager prefill of S random tokens rather than from made-up values. |
| CLI | `--boundary` accepts `causal_lm_prefill` and `causal_lm_decode`. The export-kind check now accepts any `causal_lm*` boundary. `main()` picks the wrapper and cases. |

What it's based on: the interface and cache logic follow Nicolas's `cache_source.py` (Merlin PR #63). They were rewritten here, and his eager admission checks (`validate_phase` and the rest) were not ported.

## 4. Commands

### Environment

```bash
source /home/eecs/agarcia05/miniconda3/etc/profile.d/conda.sh
conda activate atlas-torchao
cd /rscratch/agarcia/atlas-torchao-tile-aware

export PYTHONPATH=src
export MODEL=/rscratch/agarcia/TinyLlama-1.1B-Chat-v1.0
export OUT=/scratch/agarcia05/exports/full/atlas_v1_awq_tile_static
export P=$OUT/fx_prefill
export D=$OUT/fx_decode
# optional: export CUDA_VISIBLE_DEVICES=<free gpu>
```

The tested environment was torch 2.11.0+cu130 and transformers 5.22.1. Each `--out-dir` must not already exist. Commit `src/fx_export.py` first, otherwise the manifest records `dirty`.

### Step 1: export (once per model and recipe, about 25 min)

```bash
python -u src/eval.py --model $MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static \
    --device cuda --export-dir $OUT/export --out $OUT/export_cli.json
```

### Step 2: FX bundles (each reuses `$OUT/export`)

```bash
# baseline (tokens in, logits out)
python src/fx_export.py --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx \
    --device cuda --seqlen 64 --native-fp8

# prefill: mask and positions as inputs, K/V out
python src/fx_export.py --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx_prefill \
    --device cuda --seqlen 64 --native-fp8 --boundary causal_lm_prefill

# prefill at 512 tokens (not run yet)
python src/fx_export.py --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx_prefill512 \
    --device cuda --seqlen 512 --native-fp8 --boundary causal_lm_prefill

# decode: one token after a 64-token cache
python src/fx_export.py --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx_decode \
    --device cuda --seqlen 64 --native-fp8 --boundary causal_lm_decode
```

### Step 3: checks

```bash
# built-in verification passed?
jq -c '{status,cases,error}' $P/verification.json $D/verification.json
jq -c '{status,all_outputs_byte_identical}' $P/fresh_process_verification.json $D/fresh_process_verification.json

# interface (expect prefill 3 inputs / 45 outputs, decode 47 inputs / 45 outputs)
grep -c USER_INPUT  $P/graph_signature.txt $D/graph_signature.txt
grep -c USER_OUTPUT $P/graph_signature.txt $D/graph_signature.txt

# HF mask / position ops gone (expect 0 for prefill, 10 for baseline)
grep -cE 'aten\.(arange|diff|cumsum|le\.Tensor|__and__)' $OUT/fx/graph.txt $P/graph.txt

# CPU replay of the saved graph (rewrites fresh_process_verification.json)
python src/fx_export.py --verify-dir $P --device cpu
python src/fx_export.py --verify-dir $D --device cpu

# formal checklist (written for single-input graphs; may not accept these yet, untested)
python scripts/tile_flow/verify_bundle.py $OUT/export $P $OUT/verify_prefill.json
```

### Step 4: compare against the original model

`check_phases.py` (currently a session scratch script, not in the repo) runs three checks on random prompts:

1. Prefill logits compared with `CausalLogits` on the same 64 tokens.
2. Decode-after-prefill logits compared with the last position of a 65-token `CausalLogits` forward.
3. The K/V returned by decode compared with the K/V from a 65-token prefill.

```bash
python <path>/check_phases.py $OUT/export $MODEL
```

## 5. Results (full TinyLlama, `atlas_v1_awq_tile_static`, seqlen 64, cuda)

| Bundle | Inputs | Outputs | Graph nodes | HF mask ops | `model.pt2` | Verification |
|---|---|---|---|---|---|---|
| `fx` (baseline) | 1 | 1 | 3,217 | 10 | 1.24 GB | passed |
| `fx_prefill` | 3 | 45 | 3,225 | 0 | 1.24 GB | passed |
| `fx_decode` | 47 (from layer count) | 45 | 3,247 | (not counted) | 1.24 GB | passed |

All three have 154 `atlas.qlinear`, 22 `atlas.qk_matmul` and 22 `atlas.pv_matmul` nodes. Their fresh-process reload is byte-identical to eager.

Comparison against the original model (`check_phases.py`):

| Check | 2-layer toy model (3 seeds) | Full model (2 seeds) |
|---|---|---|
| 1. prefill vs `CausalLogits` | exact (max diff 0.0) | exact (max diff 0.0) |
| 2. decode vs 65-token forward, last position | max diff 0.016 – 0.08, same argmax | max diff 0.75 – 0.98, same argmax |
| 3. decode K/V vs 65-token prefill K/V | not equal | not equal |

## 6. Known issues and open questions

- **The decode mismatch is not explained.** The leading hypothesis is untested. The recipe gives attention q/k/v a dynamic per-(batch, head) scale, so a 64-token prefill and a 65-token forward quantize K/V differently, and the difference grows with depth. If this is right, it is a property of the recipe and not a bug. The next test is to compare K/V layer by layer, and to check whether adding a 65th token changes the logits at positions 0–63.
- **The graph is not smaller.** Removing the mask block is offset by the K/V outputs, so prefill has 8 more nodes than the baseline. The mask code is a fixed number of operations at any length, so 512 tokens will not change this. The benefit is the absence of `diff`, `cumsum` and `index` ops, and of an on-chip 512×512 mask build. It is not a smaller op count.
- **Nicolas's intake tool rejects these bundles as they are.** His `extract_capsules.py` requires bundles with more than one input to have `provenance.boundary == "causal_lm_tensor_bf16_cache"`, the fields `phase`, `past_length`, `total_length` and `raw_capture`, and two sidecars (`--cache-abi`, `--cache-capture`). The exporter writes none of these yet. Whether it should is a question for him.
- **One decode bundle per cache length.** No decode bundle has been made for a past other than 64.
- **Not run:** prefill at 512, `verify_bundle.py` on the new bundles, a comparison against Nicolas's own `TensorCacheLogits`, and the unit tests (`tests/test_fx_*.py`).
- **Provenance:** the bundles above were captured from a dirty tree (`M src/fx_export.py`) on commit `c093b07`.
