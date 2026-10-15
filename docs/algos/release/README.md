# Atlas FP8 v1 release records

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

Publication is on hold. Local names and the existing tag are preserved; no change to
upstream `main`, branch push or tag republication is part of this documentation update.
