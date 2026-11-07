# Atlas FP8 v1 release records

> **Corrected 2026-10-14** ([CORRECTIONS.md](CORRECTIONS.md)): `atlas_v1` rows of run 5.x are the SmoothQuant stack (`atlas_v1_sq`); SmolVLA numbers predate input normalization (ground-truth metrics invalid); AWQ vs SmoothQuant was within noise on SmolVLA and Pi0, and the Pi0 rows mixed commits; `atlas-v1.0` is a local tag (cite `5ff028e`); the release was merged to `main` (`54e1855`).

The local `atlas-v1.0` tag pins verified quantization and export code at
`5ff028ee58b28606873278236a379dcc09339b82`. The canonical DiT export gate is complete:
eight frames, 96 byte-identical captured tensors, valid payload hashes and 75 passing
GPU test invocations with zero failures. Five main-environment OpenPI skips passed
in the Pi0 environment. These records follow the code commit without moving its tag.

- [Release contents](RELEASE_CONTENTS.md): capabilities, verification and scope.
- [Results summary](RESULTS_SUMMARY.md): accuracy measurements and qualifications.
- [Structured results](reported_results.json): historical measurements and source commits.
- [Release verification](release_verification.json): canonical export digest, census,
  test results and retained GPU/local artifact paths.
- [Presentation](presentation.html): four slides covering the recipe, results and next steps.
- [Shared document replacement](SHARED_DOC_REPLACEMENT.md): the matching concise narrative.

The multi-gigabyte exports and raw GPU reports remain on the GPU machine; the supplied
report is recorded here, not independently rehashed locally. Local test logs remain
in the artifact archive identified by the verification record, with SHA256 digests.

The bundle contains SafeTensors weights, JSON execution policy and a JSON integrity
manifest. It is not a serialized FX graph or device executable. DiT chip accuracy,
real-model compiler lowering, hardware performance and closed-loop robot success are
outside the verified claims. FX graph export and MLIR handoff follow release closure.

The release was merged to `main` by pull request #1 (merge `54e1855`). The `atlas-v1.0` tag
was not published; it exists only locally, so cite commit `5ff028e`.
[Corrections](CORRECTIONS.md) and the [corrected structured results](reported_results_corrected.json)
supersede labels and status in the v1.0 records, which are otherwise kept as written.
