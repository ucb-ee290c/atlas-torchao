# Corrections to the v1.0 release records (2026-10-14)

The v1.0 records in this directory are kept as written; this note and
[`reported_results_corrected.json`](reported_results_corrected.json) correct them. No measurement
changes; labels, scope and status do.

| File | Status |
|---|---|
| `reported_results.json` | historical, as supplied for v1.0; labels as reported |
| `reported_results_corrected.json` | corrected copy: relabelled rows, flags below, `correction` block at the top |
| `RESULTS_SUMMARY.md`, `RELEASE_CONTENTS.md`, `SHARED_DOC_REPLACEMENT.md`, `presentation.html`, `release_verification.json` | v1.0 narrative; read with this note. Publication status and local paths updated in place. |

## 1. `atlas_v1` changed meaning

At `l40s-run-5` .. `l40s-run-5.3`, `--algo atlas_v1` was SmoothQuant+GPTQ+granularity+vadd `rn`.
At commit `5ff028e` (and on `main` at `2d4b187`) it was AWQ+GPTQ+granularity+vadd `rn` plus a BF16
exception for the Multi-Task DiT CLIP patch embedding. Every run-5.x row and every run-5.3 export
labelled `atlas_v1` in `reported_results.json` is the SmoothQuant stack, now named `atlas_v1_sq`;
the corrected file relabels them and keeps the original label in `recipe_as_reported`. The DiT row
`atlas_v1+patch_embed_bf16` is `atlas_v1_sq` with the patch embedding in BF16.

From v1.1 every recipe name is immutable (`atlas_v1_sq`, `atlas_v1_awq`, `atlas_v1_static`),
`atlas_v1` is an alias (currently of `atlas_v1_awq`), no recipe carries a BF16 exception, and
results and exports record the alias, resolved name and expanded stack. The full table by tag is
in [`../atlas_v1.md`](../atlas_v1.md#recipe-names-by-tag).

## 2. SmolVLA inputs were not normalized

**Every SmolVLA result before `--norm-stats dataset` (all of run 5.x and v1.0) was measured on
un-normalized inputs**: the checkpoint's saved processors store stats under keys no feature looks
up, so state was raw and actions were never unnormalized. Their ground-truth metrics (MSE / MAE)
are invalid and must not be cited. Their FP8-vs-BF16 fidelity (rel L2, cosine) compares two
quantizations of an off-distribution computation and is not comparable with
`--norm-stats dataset` results.

## 3. The AWQ-vs-SmoothQuant choice was within noise

AWQ's chip margins over SmoothQuant were 0.0159 rel L2 on SmolVLA (floor 0.020) and 0.0024 on
Pi0 (floor 0.0027): within noise. The TinyLlama margin (0.0477 PPL) exceeds the chip floor only
if the floor's `pt` unit means normalized percentage points. The Pi0 rows mixed commits: `none`
at run 5 (`0447807`), SmoothQuant and static at run 5.1 (`46b0b1d`), AWQ at run 5.3
(`a30779b`). The v1.0 choice is a numerical preference, not a resolved win. Run 6
(`l40s-run-6`) re-measures all recipes at one commit with seeds.

## 4. Status and citations

- The release was merged to `main` by pull request #1 (merge `54e1855`, tree of `2d4b187`).
  Earlier wording that publication was on hold, or that `main` was unchanged, is superseded.
- The `atlas-v1.0` tag exists only in a local checkout and was never published. Cite commit
  `5ff028ee58b28606873278236a379dcc09339b82` (`5ff028e`).
- Local absolute paths in `release_verification.json` were replaced by the placeholders
  `$GPU_ATLAS_ROOT` and `$LOCAL_ARTIFACTS`.
