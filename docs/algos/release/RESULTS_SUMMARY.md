# Atlas FP8 v1 results

> **Corrected 2026-10-14** ([CORRECTIONS.md](CORRECTIONS.md)): `atlas_v1` rows of run 5.x are the SmoothQuant stack (`atlas_v1_sq`); SmolVLA numbers predate input normalization (ground-truth metrics invalid); AWQ vs SmoothQuant was within noise on SmolVLA and Pi0, and the Pi0 rows mixed commits; `atlas-v1.0` is a local tag (cite `5ff028e`); the release was merged to `main` (`54e1855`).

Select AWQ plus GPTQ, token and channel granularity (`nc`), and rounded vector addition (`rn`) for `atlas_v1`. AWQ has the lowest measured chip error on TinyLlama, SmolVLA and Pi0. It ties the SmoothQuant candidate bit for bit on DiT at `ref_conv`; a DiT chip recipe result is unavailable. Keep `atlas_v1_static` as the fallback while the hardware team resolves runtime scales.

**NEW:** the canonical DiT recipe keeps the CLIP patch embedding in BF16. This composite choice combines the measured AWQ candidate with the measured patch exception. Its fresh full-checkpoint export passed an eight-frame, 96-tensor byte-identical round trip at `5ff028ee58b28606873278236a379dcc09339b82`. DiT chip accuracy is an unmeasured benchmark gap, not an export release gate. These results evaluate numerical simulation on L40S, not execution on Atlas hardware.

## TinyLlama

Lower perplexity is better. The full reference evaluation has 166 windows; the chip comparison has four mxq windows. Calibration uses 16 WikiText 2 training windows. Compare modes within the same sample set.

| Recipe | Full reference PPL | Chip set reference PPL | Chip PPL | Chip increase over BF16 | Chip KL / top 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 7.9745 | 8.0560 | 8.0560 | 0% | — |
| None | 8.1067 | 8.2092 | 8.2559 | 2.48% | 0.0259 / 92.4% |
| SmoothQuant candidate | 8.0662 | 8.1756 | 8.1921 | 1.69% | 0.0174 / 93.8% |
| Selected AWQ candidate | 8.0660 | 8.1523 | 8.1444 | 1.10% | 0.0169 / 93.8% |
| Static fallback | 8.0731 | 8.1661 | 8.1461 | 1.12% | 0.0173 / 94.1% |

The selected candidate reduces chip PPL by 0.1115 versus none, equivalent to 1.384 percentage points of BF16 normalized PPL increase. Its margin over SmoothQuant is 0.0477 raw PPL, or 0.592 percentage points. Full reference PPL improves by 0.0407 versus none. All recipe rows quantize 154/154 evaluated layers.

The supplied 2σ floors are 0.077 “pt” for reference and 0.244 “pt” for chip. The supplied classifications place all three recipes above the floor versus none. The source does not settle whether “pt” means raw PPL or normalized percentage points; these classifications must remain reported, not independently established. Under the percentage point interpretation AWQ exceeds the chip floor versus SmoothQuant; under the raw PPL interpretation it does not. The original noise protocol used 40 mxq windows, differing from both current sample counts.

## SmolVLA

Lower action relative L2 error versus BF16 is better; BF16 is zero by definition. Reference uses 200 frames, chip uses 32, and calibration uses 32 frames from non-test episodes.

| Recipe | Reference 200 frames | Reference 32 frames | Chip 32 frames | Chip change versus none |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0 | 0 | 0 | — |
| None | 0.1519 | 0.1458 | 0.1908 | — |
| SmoothQuant candidate | 0.1243 | 0.1361 | 0.1618 | −0.0290 |
| Selected AWQ candidate | 0.1271 | 0.1327 | 0.1459 | −0.0449 |
| Static fallback | 0.1251 | 0.1257 | 0.1796 | −0.0112 |

The reference 2σ floor is 0.0095; the chip floor is 0.020. AWQ improves on none beyond both supplied floors, but its 0.0159 chip margin over SmoothQuant is within the chip floor. Static's chip gain is also within that floor. All recipe rows quantize 303/303 evaluated layers.

## Pi0

Five DROID observations, seed 0; calibration uses eight observations from different episodes. The main value is absolute action relative L2 versus BF16, and the bracketed value is delta-space relative L2. BF16 is zero by definition.

| Recipe | Reference absolute [delta] | Chip absolute [delta] | Chip change versus none | Peak GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0 [0] | 0 [0] | — | — |
| None | 0.0120 [0.106] | 0.0144 [0.127] | — | 23.0 GB |
| SmoothQuant candidate | 0.0081 [0.071] | 0.0100 [0.088] | −0.0044 | 32.8 GB |
| Selected AWQ candidate | 0.0066 [0.058] | 0.0076 [0.067] | −0.0068 | 36.5 GB |
| Static fallback | 0.0106 [0.093] | 0.0082 [0.072] | −0.0062 | 32.8 GB |

The supplied absolute-error floors are 0.0056 for reference and 0.0027 for chip. Each recipe's reference gain is within its floor; all chip gains versus none exceed the chip floor. AWQ's 0.0024 chip margin over SmoothQuant is within the floor. All recipe rows quantize 420/420 evaluated layers. The original noise prompt specifies 32 observations, so transferring its floor to this five-observation evaluation is a qualification, not a significance test.

**NEW:** CPU Hessian storage enables the large-model recipe jobs. **NEW:** a recorded Cholesky damping retry makes the AWQ job complete; one down projection in language model layer 6 required damping 0.1. Earlier memory and Cholesky failures are superseded by these successful rows. They are not silently omitted from the run history.

## DiT with repaired CLIP weights

The checkpoint repair remaps 395 CLIP keys. Every earlier DiT result, including earlier BF16 baselines, is superseded. The recipe harness evaluates 200 frames from episodes 136, 363, 380, 464 and 474. Its BF16 ground truth MSE is 0.01822.

| Recipe | Reference relative L2 | Chip relative L2 | Cosine | Ground truth MSE | BF16 exceptions |
| --- | ---: | --- | ---: | ---: | --- |
| BF16 | 0 | Unavailable | 1 | 0.01822 | — |
| None | 0.1572 | Unavailable | 0.9876 | 0.01890 | 0 |
| SmoothQuant candidate | 0.1394 | Unavailable | 0.9902 | 0.02025 | 0 |
| AWQ candidate | 0.1394 | Unavailable | 0.9902 | 0.02025 | 0 |
| Static fallback | 0.1514 | Unavailable | 0.9885 | 0.01955 | 0 |
| Selected composite with patch embedding in BF16 | 0.0883* | Unavailable | 0.9961* | 0.01828* | 1 |

*Measured with the SmoothQuant candidate plus the patch exception. SmoothQuant and AWQ candidates without the exception are bit identical: smoothing folded nothing and `awq_folded` is empty. The canonical AWQ composite subsequently passed fresh export verification; its eight-frame replay check is not a new 200-frame accuracy measurement. All rows report 179/179 evaluated layers; the patch row additionally records one convolution exception.

The repaired Set A floor from the separate harness is 0.0559 relative L2 (mean 0.1768, standard deviation 0.0279). The patch recipe's 0.0689 gain versus none exceeds that floor. The incremental 0.0511 gain over the unpatched candidate is within it. Other recipe improvements are within the floor. The two harnesses have different BF16 MSE values (0.01822 and 0.01639 for the matching episode set), so their absolute errors must not be treated as paired measurements.

The earlier claim that convolution dominates DiT error no longer holds: convolution only yields 0.1427 and no convolution 0.1348, within the repaired floor of one another. SA's adaLN issue persists: all six adaLN layers have local relative RMSE 0.178–0.202, whereas IPT's worst layer is 0.048. Within each job, reference-to-matrix-unit error rises 0.1224→0.1586 for IPT and 0.1609→0.2853 for SA. The jobs cover 32 and 14 frames respectively, so these are paired within-job changes, not a controlled cross-job effect size.

## Vector arithmetic and implementation choices

Without recipes, moving from reference to chip on the same chip samples adds 0.0467 PPL for TinyLlama and 0.0450 action relative L2 for SmolVLA. With AWQ those changes are −0.0079 and +0.0132 respectively. These comparisons show aggregate profile cost; they do not isolate vector addition. The earlier rounded-add ablation reported a 0.51 percentage point TinyLlama chip improvement; that historical result is not re-estimated here.

**NEW:** the composite default, the DiT BF16 exception, `granularity_rescale` on the vector unit, and `vadd=rn` are explicit policies pending hardware rulings. CPU Hessians and recorded damping retries are implementation choices; keep all such choices visible in exported policy and result metadata. None establishes RTL support.

## Exports and verification

All 12 historical model and recipe exports passed bit-identical round trips and manifest hash checks at `a30779b`. TinyLlama uses two windows, SmolVLA and DiT eight frames, and Pi0 32 raw-action observations. Pi0 reuses each export's observation count because calibration follows the evaluation observations. The export site counts include attention sites and differ from evaluated-layer counts above.

| Model | Quantized sites | SmoothQuant / AWQ size | Static size | Skipped sites |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 176 | 1176.0 MiB | 1174.5 MiB | 1 |
| SmolVLA | 317 | 477.3 MiB | 475.9 MiB | 0 |
| DiT | 204 | 264.0 MiB | 263.3 MiB | 0 |
| Pi0 | 484 | 4099.7 MiB | 4095.1 MiB | 2 |

All historical DiT exports have zero BF16 exceptions. They do not implement the new canonical patch exception. Full historical manifest digests remain in the GPU result `results/l40s-run-5.3/exports_summary.json`; abbreviated digests are not full integrity proofs.

### Canonical DiT release verification

The GPU release report verifies the unchanged source at `5ff028ee58b28606873278236a379dcc09339b82`. The export is 277,366,330 bytes and its manifest SHA256 is `5cbb07ef1e892af7bd5acb56accd8b9e94c1bcc8bded1e8ce05e1454ed59b629`. All payload hashes match. The policy census is 204 total sites: 179 quantized Linear, 24 quantized attention, one BF16 Conv2d and zero skipped sites. The sole BF16 exception uses the exact patch-embedding path above and is marked NEW.

The recipe records AWQ plus GPTQ, channel weights and token activations with `nc`, and residual-add `rn`. Checkpoint repair remapped 395 keys with zero unresolved missing or unexpected keys; checkpoint and calibration fingerprints match run 5.3. Across eight frames, all 96 captured tensors are byte-identical between export replay and fresh quantization. Action statistics match exactly, outputs are finite, and repeats are deterministic.

The main release checks passed 55 tests with five OpenPI-dependent skips. The Pi0 environment passed six tests, including the skipped cases; additional CUDA/compilation checks passed 14 tests. Total: 75 passed, zero failed/errors; both compilation checks passed. These suites overlap earlier testing and are not a count of unique campaign tests. The separate GPU paths and complete supplied verification details are retained in `release_verification.json`.

Reported tests: run 5 full suite 446 passed / 22 skipped; run 5.1 GPTQ 26 passed; run 5.2 full suite 489 passed / 23 skipped, separate DiT suite 143 passed, Pi0 environment 99 passed / 3 skipped; run 5.3 GPTQ 28 passed. No reported failures; all CUDA tests ran and passed. All repeats were identical, chip cross-checks had zero mismatches, and successful runs had no untouched layers or non-finite values.

The local release full suite passed 484 tests with 44 skips: 20 CUDA-only, 14 requiring `OPENPI_DIR`, eight requiring Pi0's patched Transformers environment and two opt-in compilation tests. Supplemental release regressions passed six tests; static chip export checks passed three. The final GPU checks cover the requested release paths, not a claimed one-for-one rerun of all 44 local skips.

Evidence consists of the completed historical GPU campaign and the subsequent canonical release verification reports supplied on 2026-10-12, plus local test logs. The original GPU JSONs, exports and logs remain on the separate GPU machine; their hashes have not been independently recomputed locally. `reported_results.json` records the supplied measurements, source commits, units and limitations; `release_verification.json` records the final release check. The export verification gate is complete. FX graph export, compiler lowering, hardware execution and closed-loop task success are separate work.
