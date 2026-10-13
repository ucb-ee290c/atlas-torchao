# Atlas FP8 Quantization Algorithms

`atlas_v1` selects AWQ plus GPTQ with token and channel granularity (`nc`) and rounded vector addition (`rn`). AWQ has the lowest measured chip error on TinyLlama, SmolVLA and Pi0. Its chip margins over SmoothQuant are 0.0477 PPL, 0.0159 relative L2 and 0.0024 absolute relative L2 respectively. The latter two are within the supplied 2σ floors, so the choice is a consistent numerical preference rather than three individually resolved wins. DiT candidates tie at reference precision; no DiT chip recipe result is available.

## Recipe and fallback

| Policy | Configuration | Status |
| --- | --- | --- |
| Default | AWQ plus GPTQ, granularity token/channel `nc`, vector addition `rn` | Selected |
| DiT exception | Keep CLIP patch embedding in BF16 | **NEW**, canonical export verified at `5ff028e` |
| Fallback | SmoothQuant with static scales, GPTQ, `rn` | Retain until runtime-scale ruling |
| Large-model fitting | CPU Hessians and recorded Cholesky damping retry | **NEW**, successful Pi0 result |
| Vector scale adjustment | `granularity_rescale` | **NEW**, hardware ruling pending |

The selected recipe changes the prior SmoothQuant default. Explicit metadata must retain the algorithm, scale granularity, rounding and BF16 exceptions. Runtime scale support, exponent derivation and BF16 fallback placement require hardware-team rulings before claims about deployment on hardware.

## Measured outcomes

| Model | Unoptimized chip | Selected chip | Improvement versus none |
| --- | ---: | ---: | ---: |
| TinyLlama | 8.2559 PPL | 8.1444 PPL | 0.1115 PPL; 1.384 normalized percentage points |
| SmolVLA | 0.1908 relative L2 | 0.1459 relative L2 | 0.0449 |
| Pi0 | 0.0144 absolute relative L2 | 0.0076 absolute relative L2 | 0.0068 |
| DiT | Chip unavailable | Chip unavailable | Reference patch recipe 0.1572→0.0883 |

TinyLlama chip uses four mxq windows; SmolVLA chip uses 32 frames; Pi0 uses five DROID observations. Their wider reference evaluations are not interchangeable with chip samples. Supplied chip floors are 0.244 “pt”, 0.020 and 0.0027. TinyLlama's “pt” unit is unresolved between raw PPL and normalized percentage points. Pi0's original noise protocol uses 32 observations, so its five-observation comparison requires qualification. All floor comparisons describe spread across samples, not formal significance tests.

## DiT correction and matrix arithmetic

The CLIP checkpoint repair remaps 395 keys; all earlier DiT results are superseded. Under repaired weights, the claim that convolution dominates error no longer holds. Keeping patch embedding in BF16 nevertheless improves the recipe's reference relative L2 to 0.0883 and its ground truth MSE to 0.01828 versus BF16 0.01822. The 0.0689 gain versus no recipe exceeds the repaired 0.0559 floor; the 0.0511 incremental gain versus the unpatched recipe is within it.

SmoothQuant and AWQ DiT candidates without the exception are bit identical, with no folded smoothing or AWQ scales. The BF16 exception was measured on the SmoothQuant candidate. The canonical AWQ composite subsequently passed a full-checkpoint export and eight-frame round trip at `5ff028e`: 96 captured tensors were byte-identical. This is export verification, not a new 200-frame accuracy measurement. No additional benchmark campaign is required for the export gate.

SA remains problematic in adaLN: six layers show local relative RMSE 0.178–0.202 versus IPT's worst layer 0.048. Within-job reference-to-matrix changes are 0.1224→0.1586 for IPT and 0.1609→0.2853 for SA. The jobs contain 32 and 14 frames respectively; avoid claiming a controlled SA versus IPT output comparison.

## Export and compiler boundary

The 12 historical exports pass bit-identical round trips and manifest integrity checks. Their DiT policies have zero BF16 exceptions, so they do not represent the new default. The separately verified canonical DiT export has 204 total sites: 179 quantized Linear, 24 quantized attention and one BF16 Conv2d, with zero skipped. All payload hashes match; the final release checks report 75 passed and zero failures, with five main-environment skips covered in the Pi0 environment. Both compilation checks passed. Historical export counts are 176 TinyLlama sites, 317 SmolVLA, 204 DiT and 484 Pi0; they include attention sites and are not the evaluated-layer counts.

The export is a three-file bundle: `model.safetensors` stores FP8 byte codes, integer scale exponents and remaining native-dtype state; `policy.json` stores execution rules; `manifest.json` stores integrity and provenance. It is not a serialized FX graph or compiled executable. Canonical verification details and the full manifest digest are in `release_verification.json`.

The compiler must consume the existing quantization policy and export rather than create another quantization flow. Preserve FP8 encodings, power-of-two scale exponents, folded BF16 parameters, numeric policies and exceptions through capture. Reconcile exponent range, runtime scales, scale granularity, rounded vector addition, BF16 placement and SA accumulation with the target contract. Hardware, RTL and FPGA validation remain deferred until after the compiler handoff.

The GPU report records deterministic repeats, zero chip cross-check mismatches, no non-finite values and passing CUDA tests. These are simulation and round-trip results; they do not certify Atlas hardware execution. Detailed measurements and source commits are in `RESULTS_SUMMARY.md` and `reported_results.json`.
