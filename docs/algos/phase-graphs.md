# Per-phase FX graphs and the loop manifest (Multi-Task DiT, Pi0)

Branch `tile-phase-graphs` (worktree `/scratch2/lorenhung/atlas-fp8/atlas-torchao-phases`, from `tile-aware` `4ddedee`). Plan: `docs/algos/dit-pi0-full-graph-plan.md`
items 2, 3 and 5. Every number below is from a command run in this session; what was not run is listed at the end.

## Decisions taken (nobody specified them; the plan's default)

* **Per-phase graphs, host-owned loop, plus a loop manifest.** No unrolled whole-inference graph (plan item 6 is not done).
* The Euler / DDPM update is **on the host** (the step graph returns the noise prediction or `v_t`), as are the noise draws, scheduler arithmetic, clipping and the
  action-chunk slice. The manifest lists them.
* The existing exports and bundles under `exports-tile/` were only read. The new bundles live in sibling directories
  `.../multitask-dit/atlas_v1_awq_tile_static_attn_phases/{encode,step}/` and `.../pi0/gptq_panel_static_noawq_phases/`. `loop_manifest.json` sits next to each
  new bundle's `fx/` (a copy in `encode/` and `step/`, same sha256 as the master in the phases directory) and is referenced from that bundle's
  `fx/fx_manifest.json` (`provenance.loop_manifest`, with sha256) and `bundle_manifest.json` (`loop_manifest`, with `matches_fx_manifest`). The existing bundles'
  manifests are untouched.
* The capture cases of the phase bundles are the existing single-step bundle's fixture inputs (read only), so the phases can be compared with that bundle.

## 1. Loop manifest (`scripts/tile_flow/make_loop_manifest.py`, tests `tests/test_loop_manifest.py`)

`make_loop_manifest.py dit|pi0 --single-step-bundle DIR --out PATH` (DiT in the atlas-lerobot env; Pi0 in the atlas-pi0 env with `--checkpoint-dir` and `--openpi-dir`).
Nothing is typed in; each value is read from the file or object named in `sources`.

| Field | DiT | Pi0 |
|---|---|---|
| step count | 100: `config.json` has `num_inference_steps: null`, so `DiffusionObjective.__init__` uses `num_train_timesteps` = 100 (`modeling_multi_task_dit.py:662-666`); the script builds the policy's own `DiffusionObjective` from the checkpoint config and reads `.num_inference_steps` | 10: `sample_actions(num_steps=10)` default (`openpi/.../pi0_pytorch.py:377`, read with `ast`), the reference shim passes the same (`src/pi0_model.py:204`, `pi0_fx.py:97`); the script refuses if they disagree |
| schedule | `scheduler.set_timesteps(100)`: timesteps 99 ... 0 (diffusers `DDPMScheduler`, `leading` spacing); per step `alpha_prod_t`, `alpha_prod_t_prev`, `sqrt_beta_prod_t`, `sqrt_alpha_prod_t`, `pred_original_sample_coeff`, `current_sample_coeff`, `variance_std`, whether noise is drawn (99 of 100: every `t > 0`); config `clip_sample` true, range 1.0, `prediction_type` epsilon, `beta_schedule` squaredcos_cap_v2, `variance_type` fixed_small | `dt` = float32(-1/10) = -0.10000000149011612; float32 `times` list from the same recurrence as `sample_actions` (`time >= -dt/2`, `time += dt`): 1.0, 0.8999999761581421, ..., 0.09999992698431015 (10 values) |
| graphs and calls | `encode` x1, `step` x100 per action chunk | `prefix` x1, `step` x10 |
| per-call inputs | step: `noisy_actions` (previous scheduler output) and `timestep` change every call; `conditioning_vec` is constant over the loop | step: `x_t` and `time` change; `state`, `prefix_pad_masks` and the 36 KV tensors are constant (the expert never writes the cache) |
| shapes | `conditioning_vec` [1, 4160] bf16 = (2 cameras x 768 CLIP + 32 state + 512 text) x 2 obs steps; state [1,2,32], images [1,2,2,3,84,84], tokens [1,77], noisy actions [1,24,7] | KV 36 tensors of [1, 1, 868, 256] bf16 = 18 layers x (k, v); 868 = 3 x 256 SigLIP tokens + 100 prompt tokens; state [1,32], `x_t` [1,10,32] f32, `time` [1] f32 |
| host operations | observation queues (2 frames), tokenization, normalization, initial noise draw, timestep tensor, scheduler arithmetic with clip, 99 variance-noise draws, slice `[1:17]` of horizon 24, unnormalization | image resize and preprocessing, tokenization and state normalization (reference shim), noise draw `torch.normal` f32 [1,10,32], time recurrence, Euler update, output postprocessing |

Tests (`tests/test_loop_manifest.py`): the DDPM table reproduces `DDPMScheduler.step` for every timestep for 100 and 10 steps (a table with one coefficient scaled by 1.01 fails the same check);
the float32 Euler recurrence yields exactly `num_steps` times for every step count from 1 to 40 and for 50 and 100; the step-count reader parses a signature and fails when the default is missing; end-to-end runs on the real DiT and Pi0
bundles. Result in the atlas-lerobot env: 49 passed, 1 skipped (the Pi0 end-to-end needs openpi's gemma config, importable only in the atlas-pi0 env). In the atlas-pi0 env, `-k "pi0_manifest or euler"`: 44 passed.

## 2. `capture_fx` call counting (`src/fx_export.py`, tests `tests/test_fx_calls.py`)

The eager replay now derives a per-module call count (forward hooks on the quantized Linear / Conv2d, one `Counter` per capture case; the cases must agree, otherwise the
capture refuses with "called a different number of times in different cases"). Attention sites are counted by tag instead, from the `site:q|k|p|v` casts of
`_observe_acts` (a module called as `module.forward()` fires no hook); the four operands of a site must be cast equally often.

| calls | Linear / Conv2d | attention site |
|---|---|---|
| 0 | absent from the graph (`unused_quantized_modules`); a non-strict export's lifted but unconsumed buffers are accepted only for declared-unused modules and only when no node reads them; the inventory excludes them | pruned from the plan before export, listed in `graph_inspection.json` `unused_attention_modules`, not reported as emulated; none of its tagged `quantize_act` nodes may be in the graph |
| 1 | exactly the old checks (one `qlinear`/`qconv2d` fed by its own `quantize_act`, one `im2col`) | one Q.K / P.V pair as before |
| N | N `qlinear` (or `qconv2d`) nodes on the one code/exponent buffer pair, each with its own `quantize_act` (`row["qlinear_nodes"]`, `row["quantize_act_nodes"]`, `row["calls"]`); N `im2col` nodes; no node outside the set | N pairs (`pairs == calls`) |

`graph_inspection.json` gains `native_fp8_linear_calls` / `native_fp8_conv2d_calls` (nodes; `..._sites` still counts sites), `unused_attention_modules`, per-row `calls`;
`verify_saved` compares node counts with `..._calls` (falling back to `..._sites` for bundles saved before). `inspect_graph` takes `calls`, `attn_calls`, `unused_attention`
keyword arguments; without them the old strict single-call behaviour is unchanged.

Tests: `tests/test_fx_native.py` and `tests/test_fx_export.py` (unchanged): 67 passed. `tests/test_fx_calls.py`: 26 passed (Linear called 0, 1, 2 and 3 times, decode and native, strict and non-strict;
non-strict lifted-unused buffers stay unconsumed; Conv2d called 1 and 2 times; RoPEAttention sites called 0, 1 and 2 times in both export modes; negatives: call count differing between cases, forged call counts, a called module declared
unused, two calls sharing one `quantize_act`, forged attention counts and an absent site whose nodes are present, unequal operand counts, `im2col` count not matching the calls).

## 3. DiT phase bundles

`encode` = `observation_encoder.encode` (inputs state, images, input_ids, attention_mask, output `conditioning_vec`); `step` = one `noise_predictor(noisy_actions, timestep, conditioning_vec=cond)` call.
Drivers: `scripts/tile_flow/policy_fx_capture_phases.py` (copy in the phases directory), strict export under torch 2.11 as the single-step bundle, native FP8 uint8. Capture: encode about 4 minutes, step about 1 minute (wall clock, host load average about 38).

| | encode | step | existing single-step bundle (plan, `verification.json`) |
|---|---|---|---|
| quantized Linear sites in graph (of 179) | 145 | 34 | 179 |
| attention sites in graph (of 30) | 24 (CLIP) | 6 (DiT blocks) | 30 |
| declared unused (Linear / attention) | 34 / 6 | 145 / 24 | |
| `atlas.qlinear` / `qk_matmul` / `pv_matmul` / `quantize_act` | 145 / 24 / 24 / 241 | 34 / 6 / 6 / 58 | |
| native graph nodes | 2,586 | 858 | 3,442 |
| `verify_bundle.py` | 25 of 25, `ok: true` | 25 of 25, `ok: true` | |
| capture status, same-process and fresh-process replay | passed, passed | passed, passed | |
| `still_emulated.emulated_modules` | empty | empty | |

Both bundles have `bundle_manifest.json` written by `make_bundle_manifest.py` (`loop_manifest.matches_fx_manifest: true`).

`verify_bundle.py` changes (still 25 checks): a buffer counts as held by the graph only when a node reads its placeholder (a non-strict export lifts unused ones); attention sites the graph does not hold are derived from the graph
(no `quantize_act` tagged `<site>:...`) and must equal the declared `unused_attention_modules`; the `qlinear` node count per site is read from the graph and must equal the declared `calls`.
`tests/test_verify_bundle_phases.py` runs the real phase bundles and three tampered copies (a present site declared absent, an absent site not declared, a forged call count); each must fail the matching check.

### Equivalence (`scripts/tile_flow/compose_phases.py`, output `compose_phases.json` next to the bundles)

Loaded with `torch.export.load` only. Composition `step(encode(obs), noisy_actions, t)` against the existing single-step bundle's `model.pt2` run in the same process on the same inputs: **byte-identical on both cases, max abs
difference 0.0**. `encode`'s output is byte-identical to the encode bundle's own fixture, and its shape equals the manifest's `conditioning_vec` shape [1, 4160].

Host loop of the manifest (100 `step` calls, 1 `encode` call, DDPM arithmetic from the `per_step` table, noise from the global RNG in diffusers' draw order, seed 0) against `DiffusionObjective.conditional_sample` of the eagerly rebuilt
quantized policy with the same seed: **byte-identical, max abs difference 0.0** (shape [1, 24, 7], bf16).

**A discrepancy that is not caused by the phase split:** against the *recorded fixture* of the existing single-step bundle, case 0 differs (121 of 168 elements, max abs 0.0234 in bf16, fixture abs max 3.45) and case 1 is identical. The existing bundle's
own `model.pt2` run now on its own recorded case-0 inputs gives the same 121 differing elements, and so does the eager original boundary, so the bundle does not reproduce its own recorded case 0 in this environment; repeated runs are deterministic and OMP/MKL thread counts 1, 2, 4, 8, 16 give the same result.
Forcing `ATEN_CPU_CAPABILITY=avx2` or `ONEDNN_MAX_CPU_ISA=AVX2` changes both cases (138 / 140 elements differ), so the replay depends on the CPU kernel dispatch of BF16 contractions. The `src` files the replay imports are identical between the capture commit `6b1b358` and
`4ddedee` (diff of `src` touches only `algos/gptq_awq.py`, `fx_export.py`, `pack_plan.py`, `quant_export.py`). The root cause of the case-0 difference was not found. The in-process comparison above is therefore the equivalence evidence; do not use the recorded case-0 fixture as a bit-exact reference across machines or kernel dispatch settings.

## How a compiler owner consumes it

1. Read `loop_manifest.json` first: `denoising_steps`, `graphs.<name>.calls_per_action_chunk`, `graphs.<name>.inputs/outputs` (shapes, dtypes), `loop` (pseudo-code), `host_operations`, and `scheduler.per_step` (DiT) or `euler.times` (Pi0).
2. Each phase directory is a stand-alone FX bundle (`fx/model.pt2`, `graph.txt`, `graph_inspection.json`, `policy.json`, `fixtures.safetensors`); the verifier is `scripts/tile_flow/verify_bundle.py EXPORT FX OUT`.
3. Weights are shared: the `encode` and `step` bundles hold disjoint quantized sites (`unused_quantized_modules` lists the other phase's), so the union of the two inventories is the model. Pi0 prefix and step likewise share nothing but the K/V tensors that cross the boundary.
4. The host owns: noise draws, scheduler / Euler arithmetic, the loop, cache lifetime (Pi0 K/V are produced once and read 10 times, never written by the step).
5. A site that a graph calls N times appears once in `weight_site_bindings` with `calls: N` and `qlinear_nodes` / `quantize_act_nodes`; the weights are one buffer pair.

## What is left / not verified

* No unrolled whole-inference graph (plan item 6), no strict Pi0 export (items 7, 8), no hardware or RTL check; all numerics are the framework reference emulation.
* The DiT recorded-fixture discrepancy above is open.
* The scheduler table is float32 values written as JSON floats; a consumer that recomputes the coefficients in another precision will not be byte-identical to diffusers.
