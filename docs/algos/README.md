# Quantization algorithms (`--algo`)

`--algo NAME[:k=v,...]` selects an FP8 quantization algorithm on `src/eval.py`, `src/eval_policy.py`
and `src/eval_pi0.py`. It is orthogonal to the ladder modes (`--modes`): every mode except `bf16`
runs with the algorithm's hooks, and each result entry records it under `algo` next to the
unchanged `ladder` record.

`--algo none` (the default) creates no algorithm object: the quantized modules take exactly the
code paths of `l40s-run-2.1`. `tests/test_algo_base.py` checks this bit for bit on every ladder
step against an export of that tag.

Several algorithms can be stacked with `+`, e.g.
`--algo smoothquant:alpha=0.5+gptq_awq:method=gptq+granularity:w=channel,a=token,rule=nc`. The
order, the conflict rule and the record are in [stacking.md](stacking.md) (NEW).

Named recipes expand to documented stacks: `--algo atlas_v1`, `atlas_v1_awq` and `atlas_v1_static`
([atlas_v1.md](atlas_v1.md), NEW).

## Hook API (`src/algos/base.py`)

Subclass `algos.base.Algo`, override only the hooks you need, and set `ALGO = YourClass` in the
module. Every default is a no-op.

Model-level hooks run in `algos.quantize` for every mode except `bf16`, in this order:

| Hook | | Purpose |
|---|---|---|
| `bf16_layers(model, mode) -> [fqn]` | (e) | Layers kept in BF16. They are passed to the profile as extra `skip` entries and listed in `algo.bf16_opt_out`. The chip coverage report counts them as host work on their own `<op>@bf16_opt_out` rows. They are never silent: the MXU has no BF16-operand mode. |
| `calibrate(model, batches, run) -> dict` | (f) | Runs on the unquantized model. `run(model, batch)` is one forward. Called only if `needs_calibration` is true. Whatever it returns is stored in `algo.calibration.result`. |
| `prepare(model, mode) -> dict` | (d, folded) | In-place transforms folded into the weights or norms, such as smoothing or rotations. Runs before any layer is quantized. |

Module-level hooks run only in the reference-numerics modules: `RefQuantLinear`,
`RefQuantConv2d` and the reference attention routine, so the steps `ref_linear` through
`chip_rope_vpu`. `collab_v0`, `collab_v0_hw` and `torchao` get the model-level hooks only, and the
record says so in `module_hooks_applied: false`.

| Hook | | Purpose |
|---|---|---|
| `transform_weight(name, w, kind) -> w` | (a) | Applied right before the FP8 cast. `w` is 2-D `[out, in]`; a Conv2d weight is flattened. |
| `weight_scale(name, w, kind) -> Scale or None` | (b) | `None` keeps the reference per-tensor rule. |
| `act_scale(name, x, kind, dims) -> Scale or None` | (c) | `None` keeps the dynamic per-tensor rule. Attention operands use `dims=(-2, -1)`, which is per (batch, head). |
| `online_transform(name, kind) -> callable or None` | (d, online) | `x -> x'` applied at run time before the activation cast. Its weight-side counterpart goes in `transform_weight`. |

`kind` is one of `linear`, `conv2d` or `attention.{q,k,p,v}`. For attention, `name` is the site.

`Scale(exp, granularity, block=0, dynamic=True)` gives an int32 power-of-two exponent: the real
value is `e4m3(code) * 2^exp`, and the codes come from the vpack encoder (`ref_numerics.e4m3_vpack`).

- **per_tensor**: `exp` is `[]`.
- **per_channel**: `exp` is `[out, 1]`. The output rescale uses `[out]`.
- **per_token**: `exp` is `[..., T, 1]`.
- **per_block**: set `block`, and give `exp` broadcastable against `[..., n_blocks, block]`.
- **static**: `dynamic=False`. Under `chip`, a static cast is counted as vpack only, with no amax.

Per-block scales work only under framework accumulation (`ref_linear` .. `ref_conv`). Under an MXU
accumulation model (`reference_hw`, `chip`, `chip_rope_vpu`) they raise `NotImplementedError`,
because per-block partial sums need a hardware ruling and are not defined here.

The granularity actually used is counted per layer in the result:
`algo.granularity.{weight, activation, online_transforms}`.

Helpers:

- `algos.base.observe_inputs(model, batches, run, select, reduce)` reduces each Linear/Conv2d input
  over the calibration set; the default reduction is the BF16 amax.
- `cast_with` and `dequant` are the explicit-scale cast and dequantize.

## Calibration rules

Calibration data must never overlap the evaluation data (`src/algos/calib.py`):

| Model | Calibration source (`--calib-source`) | Default `--calib-n` |
|---|---|---|
| TinyLlama | WikiText-2 **train** split, random windows of `--seqlen` (`wikitext2_train`) | 16 |
| SmolVLA, DiT | Frames from episodes outside eval_policy's held-out test episodes: same shuffle, same `--seed` (`non_eval_episodes`) | 32 |
| Pi0 | `auto`: with `--obs droid`, DROID frames from the episodes after the evaluated ones in the same seeded order (overlap is checked and refused). With `--obs dummy`, dummy observations under `--calib-seed`, which must differ from `--seed`. Otherwise model-level `mxu_dummy`. | 8 |

`--calib-seed` (default 1234) picks the sample. The record
`algo.calibration = {source, n_samples, seed, ...}` lists the exact windows, frames or episodes
used. Calibration data is loaded once per run and reused for every mode.

## Naming and layout for a track

- Branch `algo/<track>`, created from `algo/base`.
- Code in `src/algos/<track>.py`, plus one line in `algos.REGISTRY`: `"<track>": "algos.<track>"`.
- Tests in `tests/test_algo_<track>.py`. Use `tests/test_algo_example_identity.py` as the template,
  and keep `tests/test_algo_base.py` passing.
- Docs in `docs/algos/<track>.md`. Cite a source for every numerics choice (paper with section,
  reference file:line, or RTL), and mark anything without one as NEW, with a reason.
- Parameters go in the spec, e.g. `--algo smoothquant:alpha=0.5`. Values parse as int, float,
  true/false or string.

## Comparison rules

- Use the same windows and frames as l40s-run-2: the same protocol, `--seqlen`,
  `--max-windows` / `--nsamples`, `--seed`, `--n-frames` and `--n-obs`. Every mode of one run
  sees identical inputs.
- Report each `(algo, ladder step)` against the best existing profile at the **same ladder step**,
  not only against `--algo none` at that step.
- Label CPU screening results as screening (small samples), never as final.
- A change that needs hardware the RTL does not have (for example per-block partial sums, or a
  BF16-operand MXU) is reported as such. It is not folded silently into a ladder step.
