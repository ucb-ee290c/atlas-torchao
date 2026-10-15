# Atlas v1 release contents

> **Corrected 2026-10-14** ([CORRECTIONS.md](CORRECTIONS.md)): `atlas_v1` rows of run 5.x are the SmoothQuant stack (`atlas_v1_sq`); SmolVLA numbers predate input normalization (ground-truth metrics invalid); AWQ vs SmoothQuant was within noise on SmolVLA and Pi0, and the Pi0 rows mixed commits; `atlas-v1.0` is a local tag (cite `5ff028e`); the release was merged to `main` (`54e1855`).

The local `atlas-v1.0` tag pins verified code at `5ff028ee58b28606873278236a379dcc09339b82`. It provides a reproducible BF16-to-FP8 quantization, numerical evaluation, and export workflow for TinyLlama, SmolVLA, Pi0 and Multi-Task DiT. The release history was merged into `main` by pull request #1 (merge `54e1855`), making these capabilities the repository's mainline implementation. The `atlas-v1.0` tag itself was not published.

## What users get

| Capability | Shipped behavior |
| --- | --- |
| Selected recipe | `--algo atlas_v1`: AWQ plus GPTQ, per-channel weight scales, per-token activation scales using `nc`, and rounded residual additions under chip profiles. This composition and its selection are NEW. |
| DiT exception | NEW: preserve exactly `observation_encoder.vision_encoder.model.vision_model.embeddings.patch_embedding` in BF16 for the Multi-Task DiT policy. Validate the site before calibration, record the exception, and preserve it through export and reload. |
| Static fallback and replay | `atlas_v1_static` keeps calibrated Linear input scales static. Attention still uses dynamic scales. `atlas_v1_sq` replays the previous SmoothQuant candidate; `atlas_v1_awq` retains historical behavior without the automatic DiT exception. Existing tags and exports keep their original meaning. |
| Algorithm toolkit | Stackable static calibration, granularity, SmoothQuant, mixed precision, and GPTQ/AWQ. CPU Hessians and recorded damping retries support fitting the larger Pi0 model. Rotation is excluded. |
| Numerical ladder | BF16, collaborator baseline, reference Linear/attention/convolution, matrix-unit accumulation emulation, and chip arithmetic emulation. IPT is the default; SA and vector rounding alternatives remain selectable. Native TorchAO FP8 is a GPU comparison profile. |
| Evaluation | Model-specific harnesses, calibration separated from evaluation data, perplexity, action fidelity, KL/top-1 where applicable, layer errors, determinism checks, chip coverage and cross-checks. |
| DiT checkpoint repair | Repair 395 misplaced CLIP checkpoint keys instead of evaluating stock CLIP weights. Measurements before that repair are superseded. |
| Portable quantized export | `--export-dir`, `--export-only`, `--from-export`: FP8 element codes and scale exponents in `model.safetensors`, execution rules in `policy.json`, and provenance/integrity metadata in `manifest.json`. Reload without recalibration. |
| Documentation and regression coverage | Recipe and export documentation, historical measurements, explicit NEW decisions and pending hardware rulings, plus tests for recipe identity, exact DiT exclusions and export replay. |

The release contains source and tools. The GPU machine's multi-gigabyte model exports and original benchmark logs are separate artifacts; the tag does not bundle them. Results, structured verification, the presentation and the shared-doc replacement are retained together in a local documentation commit under `docs/algos/release/`. The verified code tag is unchanged. Its original annotation predates the final GPU verification and remains a historical status snapshot; `release_verification.json` records the completed checks.

## Validation

Local CPU full suite: 484 passed, 44 skipped, zero failures. Supplemental release regressions: six passed; static chip export checks: three passed. The skips are 20 CUDA-only tests, 14 tests requiring `OPENPI_DIR`, eight requiring Pi0's patched Transformers environment, and two opt-in `torch.compile` tests.

The prior GPU campaign reports passing CUDA tests and bit-identical round trips for all 12 historical exports. The new canonical DiT BF16 exception has small-model regression coverage under reference and chip framework execution, including exact export/reload parity. Its full-checkpoint export subsequently passed the requested eight-frame GPU round trip: all 96 captured tensors are byte-identical, action statistics match, and outputs are finite and deterministic. Final release checks: 55 main-environment tests passed with five OpenPI skips, six Pi0-environment tests passed including those skipped cases, and 14 CUDA/compilation checks passed. Total: 75 passed, zero failures; both compilation checks passed. No benchmark campaign or 200-frame rerun is required for this export gate.

The canonical DiT export is 277,366,330 bytes with 204 total sites: 179 quantized Linear, 24 quantized attention, one exact BF16 Conv2d exception and zero skipped. Its manifest SHA256 is `5cbb07ef1e892af7bd5acb56accd8b9e94c1bcc8bded1e8ce05e1454ed59b629`; all payload hashes match in the GPU report. Checkpoint repair remapped 395 keys without unresolved missing or unexpected keys. Checkpoint and calibration fingerprints match run 5.3. The raw export and report remain on the separate GPU machine; see `release_verification.json` for their paths.

AWQ is numerically best among the candidates on the three supplied chip model comparisons and ties the unmodified DiT candidate at reference precision. This does not establish a statistically resolved win on every model or closed-loop policy success. See `RESULTS_SUMMARY.md` for measurement protocols and noise qualifications.

## What remained after merging

FX graph serialization, the external model2MLIR adapter, complete manifest-v2 contract integration and real-model FP8-preserving compiler capture remain compiler work. Separately committed compiler infrastructure is outside this release and does not establish real-model integration. The current export manifest is only partially aligned with m2m v2. No serialized FX graph, whole-model Atlas executable or hardware execution qualification ships in this tag.

The hardware team must still rule on runtime scales, exponent derivation and granularity, the 448-versus-240 operand-domain conflict, rounded vector additions, BF16 execution placement, SA precision, and memory constraints. Hardware, RTL and FPGA validation remain deferred until after the compiler handoff.

## Starting point

The unchanged handoff tree is `a30779b` (`l40s-run-5.3`). Local checkpoint commit `aa679e2`, on `handoff/atlas-v1-start` and tagged `atlas-v1-handoff`, has that same tree. The release change is `5ff028e`. Merlin integration starts at `7bf05ba63`, with C1–C5 already present. The handoff checkpoint branch and tag remain local.
