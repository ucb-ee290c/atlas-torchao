# Multi-Task DiT and Pi0: what a complete, faithful FX handoff needs (scoping, 2026-11-06)

Scope: read-only on `src/` and `scripts/` (HEAD `45edd84`; `src/fx_export.py` and `src/ref_attention.py` are unmodified in the worktree, other
files under `src/` have uncommitted edits by other sessions, so cite `quant_export.py` by function, not line). Every number below is from a
command run in this session or a `file:line` read; anything else is labelled INFERENCE or NOT RUN. Scratch scripts live in the session
scratchpad (`.../scratchpad/{nr_repro*.py,stack_test.py,siglip_width.py,export_probe*.py,strict_probe.py}`); the equivalence test is reproduced
in section 5. CPU rule followed: one job at a time, `taskset -a -c 0-15 nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.

## 0. Answer in one table

| Question | Finding | Handoff-blocking |
|---|---|---|
| What boundary do the compilers want? | No written pi0 / DiT boundary spec exists. Precedent: their op inventory is the **whole inference, loop unrolled, non-strict** (SmolVLA 10 denoise steps, TinyLlama prefill + decode steps); their device route is a **hand-written host graph that owns loop, cache and stop conditions**; our own SmolVLA FX handoff was a **single flow-loss forward with 1 camera, no loop, no KV cache**. Section 1. | decision, default below |
| DiT SDPA sites | Doable by an instance-`forward` replacement on `RoPEAttention` (about 35 lines in `ref_attention.py`) plus a re-export to calibrate 6 new sites. No mask, not causal, RoPE is plain BF16 elementwise before SDPA, head_dim 64, T = 24. Section 3. | yes |
| Pi0 strict export | `NativeRule` is real and fixable in 4 places in `fx_export.py`, but it is the 5th of 6 blockers found, and the next one (`while time >= -dt/2`, `pi0_pytorch.py:407`) is in openpi. openpi pins torch 2.7.1. The compilers' own export is non-strict. Section 4.1. | no (nice to have) |
| Loop: 10 unrolled calls or one step graph x10? | Unrolled (a module called N times) is refused today (measured, found 2 / found 3). A single step graph or a per-phase graph is **also** refused today, because the verifier assumes every quantized module is called (three separate checks, measured). Recommended: per-phase graphs + a loop manifest, which needs one `capture_fx` change that also provides the multi-call counting; unrolled Pi0 graph second; DiT's 100 steps must not be unrolled. Section 4.2. | per-phase split (with the `fx_export.py` change) and manifest: yes; unrolled: no |
| Action-expert attention sites | Already covered: the `sample_actions` graph has 27 SigLIP + 18 PaliGemma + 18 expert attention sites, one qk/pv pair each (measured from `graph_inspection.json`). Section 4.3. | closed |
| Camera stacking | Measured bitwise equal to 3 separate SigLIP calls on a tiny Pi0 (quantized and BF16) and on a real-width SigLIP; a negative control (per-tensor dynamic scales) differs, so the test can fail. Section 5. | no (equivalence evidence done on tiny; real-weight run NOT RUN) |

## 1. What the compilers team consumes (evidence)

* Op inventory (atlas-kernels `coverage/export_ops.py:1-6`): `torch.export.export(strict=False)` over one complete inference with fake
  tensors. TinyLlama = 128-token prefill plus greedy decode steps (`--decode-count`, default 2, `export_ops.py:411`, `Generate.forward` at
  `:280-300` loops the decode steps in Python, DynamicCache); SmolVLA = `policy.model.sample_actions(...)` with all `config.num_steps`
  (10) denoise steps (`:307-341`), vision tower called **per image**, not stacked (`:336`). Phases are recovered by counting module calls
  (`phases()`, `:344-360`: one denoise step per `action_in_proj` call). `method_reason` (`:437`): "strict TorchDynamo export was not needed".
  Sizes: 4,540 ATen records TinyLlama, 18,923 SmolVLA (`atlas-mlir/docs/bringing-up-a-model.md:85`).
* Device route (`bringing-up-a-model.md:261-271`, "New-model adapter contract"): "Host graph and semantic boundaries: Implement nondevice work, parameter
  ownership, dispatch order, cache lifetime, interlaunch staging ... stop conditions". It is "hand-written". The ledger carries
  "inference/decode/denoise step" per call. For pi0 the doc says only "check vision/text/action joins and denoising control independently"
  (`:279`) and that pi0 "require[s] implementation and qualification" (`:261`).
* FX to MLIR (`/scratch2/lorenhung/atlas-fp8/plans/step3-mlir-plan.md:169`, risk 10): "`fx_export` captures fixed `seqlen`, no KV cache; the
  saved intake refuses symbolic programs; Atlas needs static 32x32-tiled shapes". S6 (`:187`) lists "SmolVLA, pi0, DiT, ResNet50" after S3 with no boundary stated.
* Our SmolVLA handoff, for comparison (`src/fx_export.py:1100-1116`): `SmolVLAFlowLoss`, `n_images=1`, "no sampling loop, no KV cache and no random
  draw; the full action sampler (10 denoising steps with a KV cache) is a different boundary".
* Kernel constraints that bear on the models (atlas-kernels `kernels/attention/kernel.yaml:40-57`): head_dim in {64, 128, 256}; the 128/256
  variants take `mask = none | causal`. DiT head_dim is 64. Pi0 Gemma head_dim is 256 (`openpi/src/openpi/models/gemma.py:77-95`), but SigLIP
  head_dim is 72 (q_proj `[1152,1152]` in `buffer_inventory.json`; 16 heads assumed, the so400m config) and Pi0's mask is the block mask of
  `make_att_2d_masks` (`pi0_pytorch.py:52,157-160`). Whether the kernels can express either was not checked (compiler-side, not ours).

Default adopted here, since nobody has specified otherwise: **per-phase graphs, host-owned loop** (it matches the device route and the SmolVLA
precedent), plus the unrolled whole-inference graph as an inventory artifact once the verifier allows it. Evidence that would flip this: the
compilers asking for one graph per model, or `model2MLIR` needing the loop inside the graph.

## 2. State of the two graphs (measured)

| | DiT bundle | Pi0 bundle |
|---|---|---|
| boundary | `observation_encoder.encode` + 1 `noise_predictor` call (`policy_fx_capture.py:28-39`) | `sample_actions(num_steps=1)`, explicit noise, 3 cameras stacked (`pi0_fx.py`) |
| torch / strict | 2.11.0, strict | 2.7.1, non-strict |
| graph nodes (native / decode) | 3,250 / 43,340 (`verification.json`) | 8,149 / 95,785 |
| attention sites in graph | 24 (12 CLIP vision + 12 CLIP text), 6 `aten.scaled_dot_product_attention` nodes in the DiT blocks stay BF16 (`grep -c` on `graph.txt`: 6) | 63 = 27 vision_tower + 18 language_model + 18 gemma_expert, 1 qk/pv pair each |
| true loop | 100 DDPM steps, `num_inference_steps=None` falls back to `num_train_timesteps=100` (`modeling_multi_task_dit.py` `DiffusionObjective.__init__`, `conditional_sample`) | 10 Euler steps (`sample_actions(num_steps=10)` default, `pi0_pytorch.py:377`) |
| what repeats per step | `noise_predictor` only; the encoder (CLIP towers, 145 of the 179 Linear sites; 34 are in `noise_predictor`, counted from `export/policy.json`) runs once per chunk | expert pass only; SigLIP + prefix pass run once |

Consequence: both bundles are a prefix/encoder **plus one step**. As op inventory they are complete except the DiT SDPA; as a loop body they
are wrong, because a host calling them N times would recompute the encoder/prefix N times. The manifests do not say how many steps the real
model takes (only `docs/algos/howto-model-to-fx-graph.md:223` does).

DiT per-step host work that is outside any graph: `DDPMScheduler.step` draws `variance_noise = randn_tensor(...)` when `t > 0`
(`diffusers/schedulers/scheduling_ddpm.py:543-545`), and the config has `clip_sample=True, clip_sample_range=1.0, prediction_type=epsilon,
beta_schedule=squaredcos_cap_v2` (HF `config.json`). An unrolled DiT graph would need 99 explicit noise inputs and the scheduler constants.

## 3. (a) DiT `RoPEAttention`: policy sites

What the real module does (`lerobot/policies/multi_task_dit/modeling_multi_task_dit.py:451-493`): fused `qkv_proj`, reshape to `[B,H,T,D]`, RoPE
(`RotaryPositionalEmbedding.forward`, `:411-448`: `q*cos + rotate_half(q)*sin`, BF16 elementwise, cos/sin buffers `[1,1,T,D]`), then
`F.scaled_dot_product_attention(q, k, v, dropout_p=0 in eval)` with **no `attn_mask`, not causal, default scale `head_dim**-0.5`**. Config: hidden 512,
8 heads (head_dim 64), 6 layers, horizon 24 so T = Sq = Skv = 24, `use_positional_encoding=False`. Nothing for the atlas ops to handle beyond a plain
unmasked softmax; RoPE stays an ordinary BF16 node before `quantize_act` (Q and K are quantized after rotation, as `mtdit_inout/attention.py:42-47`
states: "what a matmul unit would be fed").

Why it is skipped today: `patch_attention` (`ref_attention.py:209-246`) discovers HF modules by `config._attn_implementation` and `q_proj/k_proj/v_proj`
(`:235`), plus SmolVLA/Pi0 by class name. `RoPEAttention` has neither (fused `qkv_proj`, no config), so it is "left unchanged" (docstring `:218`).

How the existing hooks work (so the patch can reuse them):
* transformers modules: `register_hf_attention` registers `ATTN_KEY` in `AttentionInterface` (`:176`) and `_hf_attention` (`:154`) reads
  `module._atlas_ref_attn`; the config is flipped to `ATTN_KEY` (`:244-246`).
* SmolVLA: instance replacement `mod.eager_attention_forward = types.MethodType(...)` (`:227`).
* Pi0: module-level swap of `modeling_gemma.eager_attention_forward`, `modeling_siglip.eager_attention_forward` and **global `F.scaled_dot_product_attention`**,
  gated by forward pre/post hooks and a `threading.local` (`:258`, `_install_pi0_routes` `:314-325`). The reference DiT port does the same for DiT SDPA (`multitask-dit-quant/mtdit_inout/attention.py:172-225`, `DiTSdpaPatch`).
* Site discovery downstream is by `_atlas_ref_attn` on any module (`quant_export.attention_sites`, `fx_export.native_attention` at `:340-360`), so a new site needs no other registry.

Recommended patch (instance replacement, not the global SDPA swap): in `patch_attention`, a third loop for `type(mod).__name__ == "RoPEAttention"` that sets
`mod._atlas_ref_attn = {"fm_qk","fm_av","variant":"vision", **_algo_cfg(name)}` and `mod.forward = types.MethodType(_dit_rope_attention, mod)`, where
`_dit_rope_attention` is the 12-line original forward with the SDPA call replaced by `ref_attention(q, k, v, self.scale, None, ...)`, output
`.transpose(1, 2).reshape(B, T, hidden)`. Reasons: no hooks, no `threading.local`, no global monkeypatch (all three are dynamo hazards and
cross-talk risks with the CLIP sites), and the module scope in the FX graph stays the module path.
* Variant: `vision` (scores to BF16, softmax in float32, P to BF16; `ref_attention.py:103-107`). That is what `mtdit_inout` uses for the DiT SDPA
  and both CLIP towers (`attention.py:15-38`: scores `.to(bf16)`, `softmax(scores.float())`). `pi0` would run the unmasked softmax in BF16; `text` keeps float32 scores.
* Effort: about 35 lines in `src/ref_attention.py`; tests about 60 lines (patched vs original SDPA within BF16 tolerance on a tiny `RoPEAttention`, site discovered, native graph has
  the extra qk/pv pairs); re-export (385 s reported for the DiT export) and re-capture (266 s reported). Roughly 0.5 to 1 day including tests. No `fx_export.py` change expected
  (matching is by tag, `_inspect_native_attention`, `:526`), but not tested.
* Risks, all from reading: (1) `native_attention` is all-or-nothing (`fx_export.py:340-360`): once the 6 modules carry `_atlas_ref_attn`, an export whose
  policy lacks their sites makes `native_attention` return `None` and **all 24 CLIP sites silently fall back to the decode path**, so the bundle must be re-exported, not patched, and the
  verifier should assert 30 qk/pv pairs; (2) the 6 `p` operands need new static exponents from calibration; (3) T = 24 is not a multiple of 32, so every
  DiT attention matrix is a partial tile (24 x 64 Q/K/V, 24 x 24 P); `tile_native.attention_site_status` checks exponent constancy only, not shape alignment, so the compiler
  side has to confirm tail-tile handling; (4) `lerobot` modules may differ in other versions (read from the installed lerobot in `atlas-lerobot`, torch 2.11 / transformers 5.5.4).
* Existing divergence worth a decision (not a bug claim): all 24 CLIP sites in the DiT policy are `variant: "text"` (`policy.json`, counted: 24 of 24), because `patch_attention`
  picks `vision` only when the class name contains "vision" (`:237`, class is `CLIPAttention`). `mtdit_inout` routes both CLIP towers through its Idefics3/vision routine. The two differ in score precision
  before the mask and softmax. Which one the DiT bundle should follow was not decided here.

## 4. (b) Pi0

### 4.1 Strict export under torch 2.7.1

`NativeRule` (`fx_export.py:186-196`) is a `tuple` subclass with a custom `__new__(cls, mode, reduce, rule, static_exp, rows=32)` and a `.rows` attribute.
Dynamo in torch 2.7.1 cannot unpack, index or slice it; 2.11.0 can. Measured on a 10-line repro in both envs (`nr_repro2.py`):

| variant | torch 2.7.1 | torch 2.11.0 |
|---|---|---|
| `NativeRule` + `a, b, c, d = rule` (current `_qact:259`) | FAIL, "Failed to unpack object for UNPACK_SEQUENCE" | OK |
| `NativeRule` + `rule[0]` / `rule[:4]` | FAIL, "Unsupported method call" | OK |
| 5-field `NamedTuple` + attribute access | OK | OK |
| frozen dataclass + attribute access | OK | OK |
| plain 4-tuple | OK | OK |

Do **not** switch `NativeRule` to a NamedTuple: its docstring promises "a 4-tuple, so it unpacks and compares as before", and `(*rule, storage)` at
`_check_act` (`:469`) and the indexing in `ActStats.record` (`:221`, `rule[1]`) would break (unpacking 5 values into 4 names raises `ValueError` in plain Python).

Lowest-blast-radius fix: convert to plain values at closure-build time and keep `NativeRule` for verification code, which runs outside dynamo.
Passing the `NativeRule` object through as an opaque argument is fine as long as it is not unpacked (measured OK on 2.7.1 and 2.11.0, `nr_repro3.py`).
* `_qact` (`:255-268`): new signature `_qact(x, qargs, storage, tag=None, label=None, rows=32, rule=None)`; unpack `qargs` (plain `tuple(nrule)`), use the int `rows`, forward `rule` only to `_ACT_RECORDER.record` (eager pass).
* `_native_sites.linear_forward` (`:397`) and `conv_forward` (`:403`): compute `qargs, rows = tuple(nrule), nrule.rows` once when the closure is built (default arguments).
* `_native_sites.cast` (`:419-421`): look up from a prebuilt `{site: {operand: (tuple(rule), rule.rows, rule)}}` instead of `sites[site][operand]`; `REFERENCE_OPERAND` (`:363`) likewise.
About 15 changed lines, plus one test that runs `capture_fx(strict=True)` on the tiny Llama under both torch versions.

The brief's diagnosis (strict fails on `NativeRule`) is not the first strict failure with the committed boundary: `pi0_fx.py`'s `SimpleNamespace` fails first, so the brief likely describes an earlier boundary version (the DiT copy of `policy_fx_capture.py` still carries a stale `Pi0FlowLoss` docstring). The `NativeRule` fix is necessary but not sufficient. First failures on a tiny random-weight Pi0 (`pi0_tiny.build`, `algo granularity:w=channel,a=token,rule=nc`, `sample_actions(num_steps=1)`, torch 2.7.1, `torch.export.export(strict=True)`), in the order met after removing each:
1. Boundary `Pi0OneStepSampler.forward` builds a `types.SimpleNamespace`: "Dynamo does not know how to trace the function `<class 'types.SimpleNamespace'>`". Driver-only; a plain class works.
2. Boundary assigns `model.embed_prefix = ...` inside `forward`: "Mutating module attribute embed_prefix during export". Driver-only; install it once before export.
3. openpi `preprocess_observation_pytorch` calls `set(image_keys).issubset(...)` (`preprocessing_pytorch.py:31`): "Dynamo does not know how to trace method `issubset` of class `set`". openpi is pinned, so the driver must replace `_preprocess_observation` (identity for 224x224 eval inputs).
4. The decode-path baseline export (which `capture_fx` runs first, `:1018`) raised a `KeyError` inside the traced region. **Not isolated**; it may come from my `a=token` scaffolding rather than the real static recipe. The native path never reached it.
5. `NativeRule` unpack (`fx_export.py:259`), the issue in the brief.
6. With `NativeRule` neutralised (5-field NamedTuple + attribute `_qact` patched in), "Data-dependent branching" at `while time >= -dt / 2` (`pi0_pytorch.py:407`): `time` is a tensor created inside `sample_actions`. Non-strict export passes because fake-tensor constants fold it. A strict driver must re-implement the loop with host-computed float32 times (the same `time += dt` float32 recurrence, run outside the graph) and call `model.denoise_step` directly.
Not reached after 6: whether `cfg["calls"]` mutation, hooks plus `threading.local` and the global SDPA swap trace (in isolation the last two and a global-dict mutation each export strict on 2.7.1, `tl_repro.py`).

Newer torch for atlas-pi0: openpi pins `torch==2.7.1` and `transformers==4.53.2` (`openpi/pyproject.toml:29,38`; `uv.lock:3165`), and its PyTorch path
copies `transformers_replace/*` over `transformers` (`README.md:207`, `src/pi0_model.py:14`). `atlas-pi0` has torch 2.7.1+cu126 / transformers 4.53.2 / Python 3.11; the only newer torch on disk is
`atlas-lerobot` (2.11.0+cpu), which has transformers 5.5.4 and cannot host openpi's replaced files. An upgrade was not tried.

Assessment: strict Pi0 under 2.7.1 means the 4-site `fx_export.py` fix, a driver that drops three constructs of the current boundary, a re-implemented sampler loop, and an unresolved baseline
`KeyError`, to gain: no 420 float-weight meta placeholders in the signature (`verify_bundle.py:260` check fails for non-strict) and a dynamo-checked graph. The compilers' own exports are non-strict, so this is **nice to have**.
Apply the `fx_export.py` fix anyway (cheap, makes the code torch-version-robust).

### 4.2 The loop: unrolled calls, one step graph, or phase graphs

Measured with `capture_fx(native_fp8="uint8", strict=False)` on the tiny quantized Pi0 (`export_probe.py`):

| boundary | result |
|---|---|
| 3 cameras stacked, `num_steps=1` | passes, 1,075 native nodes |
| 3 cameras stacked, `num_steps=2` | refused: `...gemma_expert.model.layers.0.self_attn.q_proj: expected exactly one atlas.qlinear.default consuming the code and exponent buffers directly, found 2` |
| 3 cameras unstacked (3 SigLIP calls), `num_steps=1` | refused: `...vision_tower...patch_embedding: expected exactly one atlas.qconv2d.default ..., found 3` |

The check is `fx_export.py:495` (`len(mine) != 1 or set(code.users) != set(mine) ...`), with the Conv2d twin at `:521` (`len(row["im2col_nodes"]) != 1`). Camera stacking and loop
unrolling are the same limitation. A multi-call verifier would: take the expected call count per site from the eager replay (the `called` set at `:992-1008` becomes a `Counter`),
require `len(mine) == calls`, check each qlinear is fed by its own `quantize_act`, and record per-call node names in `graph_inspection.json`. About 40 to 60 `src` lines plus 60 test lines; other places that assume one call
(`inspect_graph`, `verify_saved`, `scripts/tile_flow/verify_bundle.py` count checks) were not tested. It also needs the driver to python-unroll the loop with host times (same item as strict blocker 6) or rely on non-strict fake-tensor folding, which already worked for `num_steps=1`.

Options:
* **Unrolled 10 steps** (the compilers' inventory precedent): faithful to the whole inference; needs the multi-call verifier; the expert part repeats 10x in one graph. For DiT, 100 unrolled steps with 99 noise inputs and host scheduler arithmetic is not sensible.
* **One step graph called N times**: correct for the compiler's host-owned loop only if the encoder/prefix is not inside it (section 2). Therefore split.
* **Phase graphs (recommended default)**: Pi0 `prefix` (3 cameras, SigLIP, PaliGemma prefix; outputs the 18 per-layer K and V tensors, plus prefix pad mask) and `denoise_step` (state, `x_t`, time, the 36 cache tensors, prefix pad mask in; `v_t` out; Euler update stays on the host or is a tiny extra node). DiT `encode` (observation to conditioning vector) and `denoise_step` (noisy actions, timestep, conditioning vec to epsilon). Each quantized module is called at most once per graph, but **the existing verifier does not accept a graph that leaves quantized modules uncalled** (measured, table below), so this is not driver-only for Pi0. The KV cache crosses the boundary as explicit tensors (consistent with "no KV cache objects" in the step-3 plan, risk 10), and the loop manifest (below) carries the control.

Measured on the tiny quantized Pi0 with the phase boundaries written as `prefix` (stacked `embed_prefix` + `paligemma_with_expert.forward(inputs_embeds=[prefix, None], use_cache=True)`, returns the pad mask and the per-layer K/V) and `denoise_step` (state, pad mask, `x_t`, time, K/V tensors rebuilt into a `DynamicCache`), `capture_fx(native_fp8="uint8", strict=False)` unchanged (`phase_probe.py`):

| check that refused (first one hit, in order) | where | prefix graph | step graph |
|---|---|---|---|
| 1. unused quantized Linear/Conv2d whose code buffers are still lifted as graph inputs: "never called by the eager replay but present in the graph" | `inspect_graph`, `fx_export.py:777-779` | refused (expert q_proj) | refused (patch_embedding) |
| 2. attention paths come from every module with `_atlas_ref_attn`, not from called ones: "attention site ... has no atlas.qk_matmul in the graph" | `_inspect_native_attention`, per-path loop (`native_attention` `:340-360` builds `paths`) | refused after check 1 was relaxed in a scratch copy | same |
| 3. "saved code/exponent inventory differs": the reloaded program lists lifted buffers of uncalled modules, the saved inventory excludes them | `verify_saved` (`:879`) vs the `_without(..., unused)` inventory | refused after checks 1 and 2 were relaxed | same |

Checks 1 and 2 were relaxed in a scratch copy of `fx_export.py` (not in the repo) to find check 3; nothing past check 3 was tested. The consistent fix is in `capture_fx`: derive a `calls: Counter` from the eager replay (the `called` set at `:992-1008`), treat 0-call modules and attention sites as absent from this graph (also in the policy-vs-graph comparisons and `verify_saved`), and expect exactly `calls[path]` nodes for the others; the same counter is the multi-call verifier of plan item 5. Estimated 30 to 60 `src` lines plus 60 test lines. A driver-only alternative for DiT (untested): a boundary that holds only `model.noise_predictor` (or only `model.observation_encoder`) under an empty `model` module, so the other half is not in the module tree. For Pi0 the phases share `paligemma_with_expert`, so that needs real surgery.

Facts the denoise graph must preserve (read, not run): with `use_cache=False` openpi's replaced Gemma concatenates the cached K/V without mutating the cache (`transformers_replace/models/gemma/modeling_gemma.py:304-310`),
so every step sees the same prefix K/V; the expert qk site therefore quantizes `K = cat(prefix K, suffix K)` with one per-head dynamic exponent over the full length every step, and that
exponent can change between steps when the suffix K amax exceeds the prefix's (INFERENCE from the code; a 1-step graph cannot show this).
Pi0 step times come from a float32 `time += dt` recurrence (`pi0_pytorch.py:402-418`), so the manifest should list the exact float32 values.

Loop manifest fields (add to `fx_manifest.json` provenance and `bundle_manifest.json`; driver-only): model steps (Pi0 10, `dt=-0.1`, times list; DiT 100, DDPM `timesteps`, `clip_sample`, `prediction_type`, `variance noise draws`), which graph repeats, which inputs change per call, KV tensor shapes, camera count and order, whether a camera mask is false.

### 4.3 Action-expert attention sites

Joint prefix+suffix attention in the flow-loss forward (`gemma_pytorch.py:196-204`) calls `modeling_gemma.eager_attention_forward` with the **PaliGemma** layer's `self_attn`
(`self.paligemma.language_model.layers[layer_idx].self_attn`, `:197,201`) and the expert's q/k/v/o projections feed it, so the 18 expert `self_attn` sites never see a matmul there; that is why the flow-loss capture was refused.
The `sample_actions` path with KV cache covers them: the prefix pass runs `paligemma.language_model.forward` (PaliGemma sites) and `denoise_step` runs `gemma_expert.model.forward` (expert sites, `gemma_pytorch.py:100-120`).
Measured from `graph_inspection.json` `native_fp8_attention_modules` of the Pi0 bundle: 63 modules = 27 `vision_tower` + 18 `language_model` + 18 `gemma_expert`, `pairs == 1` each, and `unused_quantized_modules` is empty.
Side effect to keep in mind: the PaliGemma sites in `sample_actions` see prefix-only attention while in the training/joint path they would see prefix+suffix, so the static `p` exponent calibrated through `sample_actions` is the right one for this boundary and not for a flow-loss capture.

## 5. (c) Camera stacking: equivalence tests

What must hold for `torch.cat(cameras, 0)` -> one SigLIP call -> `chunk(3)` to equal 3 calls:
1. **Policy precondition (static, no model run).** For every `vision_tower` site: the Linear activation rule is batch-independent, i.e. `mode: static`, or dynamic `reduce` in {`token`}, or dynamic `block` with tokens per image a multiple of `rows` (256 % 32 = 0); the attention q/k/v rule is per-(batch, head) and `p` is static. A dynamic `tensor` rule couples images and breaks equivalence. Pi0 bundle, from `fx/policy.json`: all 420 Linear sites static; 63 attention sites with q/k/v dynamic per-(b,h) (source `reference`) and `p` static. The check below, run on the real policy (plain Python, no torch), passes for all 189 quantized `vision_tower` sites (162 Linear + 27 attention).
2. **Eager equivalence.** Original `embed_prefix` vs `_embed_prefix_one_tower_call`: embeddings bitwise, pad masks and att masks equal; then `sample_actions` at 1 and 3 steps bitwise.
3. **Negative control** on the same code with a per-tensor dynamic Linear rule: must differ.
4. **Graph link.** Eager-stacked equals the graph is already `parity.json` (byte-identical outputs on 2 cases); with 2 this gives eager-unstacked equals graph. NOT RUN on real weights: the same check on the real checkpoint (one eager forward x 2 cases, unstacked `embed_prefix`, compare against `fx/fixtures.safetensors` outputs).

Results (tiny random-weight Pi0 from `tests/pi0_tiny.py`, atlas-pi0 env, three different non-zero images with scales 1, 0.25, 0.03, seeds 0 and 1; `stack_test.py`):

| stage | `embed_prefix` embs (50,176 elems) | pad / att masks | `sample_actions` 1 step | `sample_actions` 3 steps |
|---|---|---|---|---|
| BF16 unquantized | bitwise | equal | bitwise (128 elems) | bitwise |
| quantized `granularity:w=channel,a=token,rule=nc` | bitwise | equal | bitwise | bitwise |
| negative control `granularity:w=tensor,a=tensor,rule=ref` | differs: max abs 5.9e-3 / 6.2e-3 (abs max 0.54 / 0.63), 45,206 / 26,555 elems differ | equal | differs: max abs 2.9e-2 / 4.9e-2, 128 of 128 elems | differs: max abs 3.1e-2 / 3.3e-2, 128 of 128 |

Width check (`siglip_width.py`): random-weight SigLIP at the real width (hidden 1152, intermediate 4304, 16 heads, head_dim 72, 2 layers, BF16, sdpa), batch-3 call vs three batch-1 calls:
bitwise equal (0 of 884,736 elements differ). Limits: tiny model plus a 2-layer real-width tower, CPU only; a batched BF16 GEMM may pick a different accumulation order on another backend or with the full 27 layers, so the real-weight run in item 4 is the evidence that counts for the shipped bundle. Per-element ulp is not reported (near-zero elements make it meaningless).

Exact test (pytest form; requires `OPENPI_DIR` and the atlas-pi0 env, skip otherwise, as `tests/test_pi0.py` does):

```python
# tests/test_pi0_camera_stack.py (proposed; not added to the repo)
import pytest, torch, algos, pi0_tiny, pi0_fx

pytestmark = pytest.mark.skipif(pi0_tiny.available() is not None, reason="needs openpi + transformers_replace")
FAST = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}

def _model(spec):
    m, cfg = pi0_tiny.build("bfloat16", action_horizon=4, seed=0)
    if spec:
        algos.quantize(m, "ref_conv", options=dict(FAST), algo=algos.make(spec))
    return m, cfg

def _obs(cfg, seed):                       # three DIFFERENT images with unequal range, all masks true
    o = pi0_tiny.observation(cfg, seed=seed)
    g = torch.Generator().manual_seed(seed + 50)
    for k, s in zip(o.images, (1.0, 0.25, 0.03)):
        o.images[k] = (torch.rand(1, 3, 224, 224, generator=g) * 2 - 1) * s
        o.image_masks[k] = torch.ones(1, dtype=torch.bool)
    return o

def _embed(m, o, stacked):
    imgs, msk = list(o.images.values()), list(o.image_masks.values())
    f = (lambda *a: pi0_fx._embed_prefix_one_tower_call(m, *a)) if stacked else (lambda *a: type(m).embed_prefix(m, *a))
    with torch.no_grad():
        return f(imgs, msk, o.tokenized_prompt, o.tokenized_prompt_mask)

def _run(m, o, nz, steps, stacked):
    orig = m.embed_prefix
    if stacked:
        m.embed_prefix = lambda *a: pi0_fx._embed_prefix_one_tower_call(m, *a)
    try:
        with torch.no_grad():
            return m.sample_actions("cpu", o, noise=nz, num_steps=steps)
    finally:
        m.embed_prefix = orig

@pytest.mark.parametrize("spec", [None, "granularity:w=channel,a=token,rule=nc"])
@pytest.mark.parametrize("seed", [0, 1])
def test_stacked_equals_unstacked(spec, seed):
    m, cfg = _model(spec); o = _obs(cfg, seed); nz = pi0_tiny.noise(cfg, seed)
    a, b = _embed(m, o, False), _embed(m, o, True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    for steps in (1, 3):
        assert torch.equal(_run(m, o, nz, steps, False), _run(m, o, nz, steps, True))

def test_negative_control_per_tensor_rule_is_not_batch_independent():
    m, cfg = _model("granularity:w=tensor,a=tensor,rule=ref"); o = _obs(cfg, 0)
    assert not torch.equal(_embed(m, o, False)[0], _embed(m, o, True)[0])      # the test must be able to fail

def test_policy_vision_sites_are_batch_independent(policy_json):                  # item 1; needs a fixture returning json.load(open(<bundle>/fx/policy.json))
    for name, s in policy_json["sites"].items():
        if "vision_tower" not in name or s["status"] != "quantized":
            continue
        if s["kind"] == "attention":
            r = s["operands"]
            assert r["p"]["mode"] == "static"
            assert all(r[o]["mode"] == "dynamic" and r[o]["source"] == "reference" for o in "qkv")   # per (batch, head)
        else:
            a = s["activation"]
            assert a["mode"] == "static" or a.get("reduce") == "token" or (a.get("reduce") == "block" and 256 % a.get("rows", 32) == 0)
```

## 6. (d) Prioritised plan

Effort figures are estimates from the line counts above, not measurements. B = blocks a faithful compiler handoff; N = nice to have.

| # | Item | Where | Effort | Label |
|---|---|---|---|---|
| 1 | DiT `RoPEAttention` policy sites (section 3): instance-forward replacement, variant `vision`; re-export 385 s, re-capture 266 s, verifier asserts 30 qk/pv pairs | `src/ref_attention.py` (+35), `tests/` (+60) | 0.5-1 day | **B** (otherwise 6 of 30 DiT attention blocks are BF16 SDPA, and the bundle reads as fully quantized) |
| 2 | Loop manifest in `fx_manifest.json` / `bundle_manifest.json` (section 4.2 fields) and a one-paragraph boundary statement in the how-to | drivers, docs | 0.25 day | **B** (the manifests today do not say the graph is one step of 10 / 100) |
| 3 | Per-phase graphs: DiT `encode` + `denoise_step`; Pi0 `prefix` (KV tensors out) + `denoise_step` (KV tensors in); needs the `capture_fx` uncalled-module handling (section 4.2, three checks); re-capture both (Pi0 non-strict took 1,549 s for the current one-step graph) | drivers (+120), `src/fx_export.py` (+30-60, shared with item 5), `tests/` (+60) | 1.5-2 days | **B** for a loop-faithful handoff; the current graphs stay valid as per-op inventory |
| 4 | Equivalence evidence for stacking: add the section 5 tests; run the real-weight eager-unstacked vs fixtures check (item 4 there) | `tests/` (+60); one 2-case eager run of the real Pi0 | 0.25 day + run time | N, but needed to keep stacking in item 3 |
| 5 | Multi-call verifier (`fx_export.py:495,521`, `called` counter at `:992-1008`, graph_inspection per-call rows, `verify_bundle.py` counts) | `src/fx_export.py` (+40-60), `tests/` (+60), `scripts/tile_flow/` | 0.5-1 day | N as a feature, **but** the `calls` Counter is the same change as item 3, so do them together; unlocks 6 and removes the stacking deviation; becomes B if the compilers ask for the whole inference in one graph |
| 6 | Unrolled `sample_actions(num_steps=10)` Pi0 graph (non-strict, python-unrolled host times); not for DiT | driver | 0.5 day + capture time (not measured; 10x the expert part) | N (matches the compilers' inventory precedent) |
| 7 | `NativeRule` dynamo fix (section 4.1, 4 sites) + test under both torch versions | `src/fx_export.py` (+15), `tests/` | 0.25 day | N (robustness) |
| 8 | Strict Pi0 under 2.7.1: item 7 + driver rewrite (no `SimpleNamespace`, no in-forward mutation, `_preprocess_observation` bypass, host-time loop) + isolate the decode-baseline `KeyError` | drivers, 1 `src` item | 1+ day, outcome uncertain | N |
| 9 | Generalise `verify_bundle.py` (patch-embedding conv declared BF16/float32 by policy, parameters with no users in non-strict graphs, per-call counts, DiT SDPA row disappears after item 1) | `scripts/tile_flow/verify_bundle.py` | 0.25 day | N |
| 10 | Decide the CLIP variant for the DiT bundle (`text` today, `vision` in `mtdit_inout`) | `ref_attention.py:237` or docs | 0.25 day, needs a decision | N (fidelity question) |

Order: 1 and 3+5 both touch `src` (different files: `ref_attention.py` vs `fx_export.py`), 2 is docs/drivers; do 1, 2 and 3+5 in parallel, then 4, then 6; 7 to 10 as capacity allows. Items 1 to 3 together are about 3 days including re-captures.

## 7. Not verified here

* No real-weight Pi0 or DiT forward was run; all experiments used the tiny random-weight Pi0, a 2-layer real-width SigLIP, or 10-line repros.
* The three `capture_fx` refusals for phase graphs were measured on the tiny model with scratch relaxations; whether more checks follow check 3, and whether DiT strict (torch 2.11) prunes unused buffers differently, was not tested.
* The decode-baseline strict `KeyError` (section 4.1, blocker 4) was not isolated, and blockers after the `while` loop were not reached.
* Compiler-side acceptance (head_dim 72 SigLIP, Pi0 block masks, 24-row tiles, 10x repeated weights) is unchecked; only the kernel manifest values above were read.
* The effort numbers, the unrolled-graph size and capture time, and the claim that the per-head K exponent can vary across denoise steps are estimates or inferences.
