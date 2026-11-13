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

## 4. Pi0 phase bundles

`prefix` = `_preprocess_observation` + openpi's `embed_prefix` (3 SigLIP calls, cameras NOT stacked: with the call counting each SigLIP module has 3 nodes, which removes the one-tower deviation of the single-step bundle)
+ the PaliGemma prefix pass with `use_cache=True`; outputs `prefix_pad_masks` [1, 868] and 36 K/V tensors of [1, 1, 868, 256]. `step` = `denoise_step` with the 36 tensors rebuilt into a `DynamicCache` (inputs state, `prefix_pad_masks`, `x_t`, `time`, K/V; output `v_t` [1, 10, 32]).
Driver `scripts/tile_flow/pi0_phase_capture.py` (copy in the phases directory), non-strict export under torch 2.7.1 (atlas-pi0 env), native FP8 uint8, the same `aten.conv2d.padding` driver workaround as the single-step bundle.
Capture wall clock (host load average about 40): prefix 40 minutes (16:05 to 16:45), step 8.5 minutes.

| | prefix | step | existing single-step bundle |
|---|---|---|---|
| quantized Linear sites in graph (of 420) | 289 | 131 | 420 |
| of which called 3 times | 163 (27 SigLIP layers x 6 Linear + the projector) | 0 | 0 (cameras stacked) |
| attention sites in graph (of 63) | 45 (27 SigLIP, 3 pairs each; 18 PaliGemma, 1 pair) | 18 (action expert) | 63 |
| declared unused (Linear / attention) | 131 / 18 | 289 / 45 | |
| `atlas.qlinear` / `qk_matmul` / `pv_matmul` / `quantize_act` | 615 / 99 / 99 / 1,011 | 131 / 18 / 18 / 203 | |
| native graph nodes | 9,662 | 3,985 | 8,149 |
| `verify_bundle.py` | 25 of 25, `ok: true` | 25 of 25, `ok: true` | |
| capture status, same-process and fresh-process replay | passed, passed | passed, passed | |
| `still_emulated.emulated_modules` | empty | empty | |

(615 = 289 + 2 x 163; 1,011 = 615 + 4 x 99; both from the same graph counts.) Both bundles have `bundle_manifest.json` with `loop_manifest.matches_fx_manifest: true`; the loop manifest records `siglip_calls_per_prefix_call: 3`.

### Pi0 equivalence (`compose_phases.py pi0`, `compose_phases.json`)

One step with the existing bundle's own fixture inputs: `prefix`, then `step` at time 1.0, then `x = noise + (-1.0) * v` (the single-step bundle is `sample_actions(num_steps=1)`), against the existing single-step `model.pt2` (cameras stacked, one SigLIP call) run in the same process:
**byte-identical on both cases, max abs difference 0.0**, 36 K/V tensors with the manifest's shape. This is also real-weight evidence that stacking the cameras equals three SigLIP calls for this bundle (the plan's section 5 item 4 had only tiny-model evidence).
Against the *recorded fixtures* of the existing bundle both cases differ (max abs 0.036 and 0.045); the existing single-step graph itself differs from its own recorded fixture by the same amount, so as for DiT this is a property of the existing bundle in this environment, not of the phase split.
The 10-step host loop against eager `sample_actions(num_steps=10)` (`compose_phases.py pi0 ... --eager-loop`) first did not finish (killed at the deadline); it was completed afterwards, see "Pi0 10-step loop check" below. The DiT host loop (100 steps) was run, see section 3.

## How a compiler owner consumes it

1. Read `loop_manifest.json` first: `denoising_steps`, `graphs.<name>.calls_per_action_chunk`, `graphs.<name>.inputs/outputs` (shapes, dtypes), `loop` (pseudo-code), `host_operations`, and `scheduler.per_step` (DiT) or `euler.times` (Pi0).
2. Each phase directory is a stand-alone FX bundle (`fx/model.pt2`, `graph.txt`, `graph_inspection.json`, `policy.json`, `fixtures.safetensors`); the verifier is `scripts/tile_flow/verify_bundle.py EXPORT FX OUT`.
3. Weights are shared: the `encode` and `step` bundles hold disjoint quantized sites (`unused_quantized_modules` lists the other phase's), so the union of the two inventories is the model. Pi0 prefix and step likewise share nothing but the K/V tensors that cross the boundary.
4. The host owns: noise draws, scheduler / Euler arithmetic, the loop, cache lifetime (Pi0 K/V are produced once and read 10 times, never written by the step).
5. A site that a graph calls N times appears once in `weight_site_bindings` with `calls: N` and `qlinear_nodes` / `quantize_act_nodes`; the weights are one buffer pair.

## What is left / not verified

* Pi0: the 10-step host loop was checked against eager on 2 recorded cases only (below), on this host with OMP/MKL threads 8. Size: the Pi0 `model.pt2` files are 4.3 GB (prefix) and 4.1 GB (step), as large as the single-step bundle, because a non-strict export keeps the code / exponent buffers of the modules a phase never calls as lifted graph inputs that no node reads (listed in `graph_inspection.json` `unused_quantized_modules[*].buffers_lifted_unused`); a consumer must ignore them, and the DiT strict bundles do not have them (155 MB step, 184 MB encode). Pruning them from the saved program is not done.
* No unrolled whole-inference graph (plan item 6), no strict Pi0 export (items 7, 8), no hardware or RTL check; all numerics are the framework reference emulation.
* The recorded-fixture discrepancy (DiT and Pi0) is open; see "Fixture reproducibility finding" below for what was and was not ruled out.
* The scheduler table is float32 values written as JSON floats; a consumer that recomputes the coefficients in another precision will not be byte-identical to diffusers.

## Pi0 10-step loop check

Run (atlas-pi0 env, torch 2.7.1, host `bwrcix-1`, `run_eager_loop.sh` in the phases directory; load average 26 at start, 45 at the end):

```
taskset -a -c 0-15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /scratch2/lorenhung/envs/atlas-pi0/bin/python scripts/tile_flow/compose_phases.py pi0 \
  <exports-tile>/pi0/gptq_panel_static_noawq_phases <exports-tile>/pi0/gptq_panel_static_noawq <phases>/compose_phases_eager.json \
  --eager-loop <exports-tile>/pi0/gptq_panel_static_noawq/export --openpi-dir /scratch2/lorenhung/atlas-fp8/openpi \
  --checkpoint-dir /scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris --cache-dir <phases>/eager_loop_cache
```

Wall clock 18:28:41 to 18:38:25 (9.7 minutes: model load, the one-step checks, then per case an eager pass and a composed pass). **Nothing was reduced**: both cases, full
10 steps for both the eager and the composed loop, one run each. (The earlier 20-minute kill was not a property of the loop: each 10-step pass took 63 to 65 seconds here.)

| case | composed loop: 1 `prefix` + 10 `step` calls, host Euler loop of `loop_manifest.json` | eager `sample_actions(num_steps=10)`, same noise (`ins[9]`) | max abs difference | differing elements | byte-identical |
|---|---|---|---|---|---|
| 0 | [1, 10, 32] float32 | same | 0.0 | 0 of 320 | yes |
| 1 | [1, 10, 32] float32 | same | 0.0 | 0 of 320 | yes |

`ok: true`. Because the composed loop equals eager there was no mismatch to localise, so the step-1 comparison code (added to `compose_phases.py`: eager `num_steps=1` against the composed one-step output, run only on a mismatch) did not execute and is untested.
The one-step checks in the same run again gave composition equals the existing single-step graph (max abs 0.0, both cases). Caveats: equality is shown in one process at one thread count (8); only the two recorded fixture inputs
(one noise draw each) were used, not a sample of observations; the eager reference is the quantized-emulation model, so this is a consistency check of the phase split and host loop, not an accuracy check.
`compose_phases.json` in the phases directory now holds this run (with a `run` block); the previous one-step-only file is kept as `compose_phases_onestep_omp4.json`. Script changes: `--eager-cases`, `--cache-dir`,
partial JSON written after each case, `differing_elements` and timings per case.

## Fixture reproducibility finding

Hypothesis tested: the recorded `fixtures.safetensors` came from a different host, CPU or torch build. Recorded provenance (`fx/fx_manifest.json`, `bundle_manifest.json`) against this session:

| | recorded | now |
|---|---|---|
| Pi0 single-step bundle `gptq_panel_static_noawq` | host `bwrcix-1.EECS.Berkeley.EDU`, torch `2.7.1+cu126`, python 3.11.17, torchao 0.12.0, commit `45edd84` (tile-aware, dirty: true), capture source `79628ee`; `fixtures.safetensors` mtime 2026-11-06 13:14:31 | `bwrcix-1.EECS.Berkeley.EDU`, torch `2.7.1+cu126` (atlas-pi0 env) |
| DiT single-step bundle `atlas_v1_awq_tile_static_attn` | host `bwrcix-1.EECS.Berkeley.EDU`, torch `2.11.0+cpu`, python 3.12.15, torchao 0.18.0+cpu, commit `6b1b358` (tile-dit-attn, dirty: false); fixtures mtime 2026-11-06 13:04:34 | `bwrcix-1.EECS.Berkeley.EDU`, torch `2.11.0+cpu` (atlas-lerobot env) |
| CPU | not recorded (no `created_utc`, no CPU model, no thread count or ISA setting in any manifest; the mtimes above are the only timestamps) | Intel Xeon Platinum 8380 (Ice Lake), AVX-512 incl. VNNI, no `avx512_bf16`, no AMX |

Result: host name and torch build **match** for both bundles, so the "different host / different torch build" part of the hypothesis is not supported by the provenance. It cannot be fully excluded: the CPU model and the process
environment of the recording run were not recorded, and a hostname does not prove the same CPU. The other hosts in `~/.ssh/config` (`workhorse`, `merlin-a/b/c`, `merlin-compute-1/2`, `merlin-w1`, `tvm-chunk-f2`) are AWS instances reached via SSM; with `BatchMode` `workhorse` timed out
at banner exchange and the others returned `TargetNotConnected` (stopped). No bundle was produced there (paths are `/scratch2` on this host), so nothing was checked on another machine; the two `atlas-hw-*` boxes were not touched.

One further observation made while doing this (not the asked hypothesis, a side finding): **the Pi0 single-step graph's output depends on the intra-op thread count.** `scripts/tile_flow/pi0_thread_repro.py N OUTDIR` (load `model.pt2`,
run both recorded cases, `torch.set_num_threads(N)`, same process otherwise; results in `thread_repro.jsonl`), run with `taskset -c 0-15` and OMP/MKL = N:

| threads | case 0 max abs vs fixture | case 1 max abs vs fixture | differing elements vs fixture |
|---|---|---|---|
| 1 | 0.03618 | 0.04545 | 320 of 320 (both) |
| 4 | 0.03642 | 0.04545 | 320 of 320 (both) |
| 8 | 0.03815 | 0.05568 | 320 of 320 (both) |

Between thread counts (same graph, same inputs): case 0 differs in 320 of 320 elements for 1 vs 4 (max 0.0443), 1 vs 8 (0.0401), 4 vs 8 (0.0416); case 1: 1 vs 4 identical, 1 vs 8 and 4 vs 8 differ in 320 elements (max 0.0474).
So a change of thread count alone moves the output by about the same size as the fixture mismatch (0.036 to 0.056), and none of the three tested counts reproduces the fixture. (The DiT section above reports thread counts 1 to 16 giving the same DiT result; that was not re-run.)
What this supports: a fixture recorded at a different thread count, or under different CPU load, would not reproduce bit-exactly. What it does not show: the thread count (or anything else) of the recording run, nor which setting would reproduce the fixture. The recorded run's
thread count could be probed by trying more counts (2, 3, 5 to 16, 32+), which was not done. For DiT the cause is still not found. Practical consequence stays as stated in section 3: use same-process comparisons, and fix the thread count when comparing across runs.
