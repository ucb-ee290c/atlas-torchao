# gptq_awq `dead_columns=zero|rtn`: do not zero input channels that saw only zeros in calibration (2026-11-06, CPU)

Fixes the cause diagnosed in `siglip-fc2-flush.md` (commit a6cd07e): `gptq()` in `src/algos/gptq_awq.py` zeroed every weight column whose calibration input is exactly 0
(`dead = diag(H) == 0`). Default stays as it was; the fix is opt-in. Every number below comes from a command run in this worktree
(`tests/test_gptq_dead_columns.py`, `scripts/tile_diag/gptq_dead_columns_validate.py`, raw results `/scratch2/lorenhung/atlas-fp8/tmp_fc2/gptq_dead_columns_l23.json`).

## Option

`gptq_awq:...,dead_columns=zero|rtn` (parsed like `producer_aware`; `parse()` turns the value into a string, anything else raises `ValueError`; works inside a stack).

| value | behaviour |
|---|---|
| `zero` (default) | the original code path: `W[:, dead] = 0`. Bit-identical to the parent commit (test against a frozen copy of the old routine, 4 cases: actorder x block). |
| `rtn` | dead columns keep their weight and are cast with `qdq(., s)`: plain round-to-nearest on the same pinned exponents (`s` is the per-tensor or the stacked chooser's panel `nc` exponent, computed from the weight before GPTQ, so the option cannot move it). |

Semantics of `rtn`:
- Error feedback: dead columns neither give nor receive feedback. In GPTQ's own algebra this is already true (a dead channel has `H[dead, :] = 0`, so after `H[dead, dead] = 1` the matrix is block diagonal and `Hinv` has no cross terms),
  and the loop enforces it exactly: a dead column's error is not propagated, and the dead columns are overwritten at the end with `qdq(W_original[:, dead], s)`. Consequence: live columns are bit-identical to `zero`.
  This is the choice consistent with GPTQ (the update `W[:, j:] -= err * Hinv[i, j:]` is zero for them), not a new approximation.
- The dampening uses the same `mean(diag(H))` as `zero` (dead diagonals count as 1 in both), so live columns do not move.
- Grid: unchanged. Exponents are panel-constant (checked with `tile_native.check_constant_groups`), codes carry no 0x7F / 0xFF.
- Record: `rec["dead_columns"] = "rtn"` only when on (like `producer_aware`); the `params` dict of the record shows what the spec string contained. A default-spec record has no new key.
- `gptq()` also puts `info["dead"]` (count of dead columns) in its `info` dict (always, harmless).
- The GPTQ cache key includes the option.

Code: `src/algos/gptq_awq.py` (`gptq()` signature and body, `GptqAwq.__init__`, `transform_weight`, `record`).

## Tests (`tests/test_gptq_dead_columns.py`, 13 tests, synthetic layer 64 x 96 with 72 of 96 input channels dead, panel `nc` exponents)

1. Default bit-identical to a frozen copy of the original routine (actorder False/True x block 16/128), also through `transform_weight` on the algo; record has no `dead_columns`.
2. `rtn`: dead columns equal `qdq(W, s)` exactly (actorder False/True, and scalar exponent); live columns equal the default's bit for bit and differ from plain RTN (GPTQ still acts on them);
   weight error on the layer 0.0300 vs plain RTN 0.0269 vs default 0.8616 (seed 3); Gaussian-probe output error under 1.5x RTN's; calibration-set error on the live columns not above RTN's; with no dead column `rtn` equals the default.
3. Exponent grid: `s0`, `weight_scale` exponent equal the panel `nc` of the input weight and are equal between `zero` and `rtn`; constant per 32 rows; no reserved codes; codes decode back to the output weight; dead columns carry real codes only in `rtn`.
4. Option parsing: default, `rtn`, `zero`, inside a stack, three invalid spellings raise `ValueError`, invalid value to `gptq()` raises.

Existing tests (CPU, `PYTHONPATH=src`, one pytest process over the six files): 115 passed, 0 failed
(`test_algo_tile`, `test_awq_producer_aware`, `test_algo_gptq_awq`, `test_algo_hw_calib`, `test_tile_flow_defects`, `test_gptq_dead_columns`; 46 + 10 + 18 + 4 + 24 + 13 in that order).
The full suite was not run.

## Real tensor: Pi0 SigLIP layer 23 `mlp.fc2` ([1152 out, 4304 in], stock BF16 weight)

Setup: stock weight, panel `nc` exponents (32 rows, range -11 .. -10, constant per 32), GPTQ `block=128 damp=0.01 actorder=true` (the export's settings), `H = X^T X` of the captured fc2 inputs
(`tmp_fc2/fc2_in.pt`, DROID episode 252 frame 30, 512 tokens = 2 images x 256), float32. Gaussian probe: 1024 x 4304 standard normal rows, seed 0. All errors are relative Frobenius norms.
Real-input output error is `||X (Wq - W)^T|| / ||X W^T||`.

Reproducing the export's zero columns: not possible from one frame. The export (8 calibration frames) has 4177 zero columns (127 alive channels); this frame alone gives 4229 dead channels in `H` (75 alive), as `siglip-fc2-flush.md` says.
The table therefore uses this frame's `H`, not the export's `H`.

A. Calibration = the whole captured frame (4229 dead columns). Real error is in-sample.

| variant | zero columns | weight err | real-input out err (calib frame) | Gaussian out err |
|---|---|---|---|---|
| RTN, same exponents | 0 | 0.0266 | 0.0254 | 0.0266 |
| GPTQ `zero` | 4229 | 0.9894 | 0.0187 | 0.9891 |
| GPTQ `rtn` | 0 | 0.0411 | 0.0187 | 0.0407 |

B. Calibration = image 0 only (4232 dead columns), image 1 held out (real tokens that excite channels image 0 never did).

| variant | zero columns | weight err | out err image 0 (calib) | out err image 1 (held out) | Gaussian out err |
|---|---|---|---|---|---|
| RTN, same exponents | 0 | 0.0266 | 0.0253 | 0.0255 | 0.0266 |
| GPTQ `zero` | 4232 | 0.9905 | 0.0164 | 0.1156 | 0.9902 |
| GPTQ `rtn` | 0 | 0.0477 | 0.0164 | 0.0226 | 0.0483 |

In both runs: live columns of `rtn` are bit-identical to those of `zero`; dead columns of `rtn` equal plain RTN; every entry on the pinned grid; 0 reserved codes; `damp` stayed 0.01.
Runtime 3 to 6 s per GPTQ run (CPU, shared host).

Reading: the calibration-frame error is identical for `zero` and `rtn` (dead channels carry no input there), so the option costs nothing on calibration data; on the held-out image `zero` is 4.5x worse than RTN's real error (0.1156 vs 0.0255)
and `rtn` is better than RTN (0.0226). The weight error of `rtn` (0.041 to 0.048) is above RTN's 0.0266 because the 75 live columns, which hold most of the weight energy that matters, are GPTQ-adjusted for the calibration Hessian.

## NOT validated

- No Pi0 re-export, no bundle manifest or policy regenerated, no run of the real export pipeline with the option (too slow on the shared host). The option was exercised through `GptqAwq.transform_weight` on synthetic layers only, and through `gptq()` on the real layer.
- No action quality, no drift, no LIBERO / DROID rollouts. Only layer 23 fc2 (one of the nine layers 17 to 25), one frame, so the held-out evidence is a single image pair.
- The export's own 4177 zero columns were not reproduced (needs its 8 calibration frames).
- AWQ path (`method=awq_gptq`) with `rtn`: only the parsing is tested; AWQ never changes which columns are dead (a dead channel stays dead after the 1/s rescale of `H`).
- `calib=chip` / `target=fp` with `rtn` not run.
- Hardware: no check against the tile contract in an exported bundle (`verify_export`); the contract was checked on the code tensors from `e4m3_vpack` of the algorithm output.
