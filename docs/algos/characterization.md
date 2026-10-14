# Native action characterization

`characterize_actions.py` provides NEW metadata preflight, paired existing-export
recording and offline descriptive analysis. The initial adapter is SmolVLA on the
cached SO100 task: native 50-by-6 joint/gripper chunks, not XYZ plus rotation.
Physical units, executed action prefix, training-held-out generalization and simulator
success are not established by this adapter.

## Frozen pilot

Run `scripts/run_characterization_pilot.sh --preflight-only` with absolute
`MODEL_CONFIG`, `DATASET_ROOT`, `EXPORT_DIR`, `PILOT_DIR` and `PILOT_PYTHON` settings.
The directory must be new. Preflight freezes ten episodes and ten temporally stratified
frames each, source metadata/parquet hashes, native shape, normalization, seeds and
provisional existing-export calibration exclusions. It does not load weights or videos.
Videos are not fingerprinted at preflight; decoded/preprocessed inputs are hashed at
record time. An unbound local preflight is explicitly planned/unfitted and cannot record.

Recording reuses the existing inference and export loader, without recalibration or
quantizer changes. Original BF16 checkpoint weights must match the export's checkpoint
fingerprints. Before any evaluation forward, recording reconstructs the original indexed
preprocessed calibration batches and requires their digest to match the export. Only
then are calibration identity and evaluation disjointness verified; changed episode
lengths or source data refuse rather than silently changing the split. This integrity
check does not fit or recalibrate the model. Every observation is paired across BF16 and the selected reference export
with the same input and noise. Raw model-space and postprocessed dataset-space chunks,
ground truth and valid-target masks are retained separately. Only a completed recording
with matching payload/header hashes can be analyzed. Failed partial runs are retained.

The initial pilot compares BF16 and the selected export, not an unoptimized FP8 baseline.
The latter is a separate follow-up configuration. Reported bias, RMSE, MAE, standard
deviation and quantiles use explicit native component/horizon axes and separate
full-generated versus valid-target denominators. No bootstrap interval or acceptance
threshold is implied by those descriptive statistics.

## Optional layer diagnostics

On the `record` command, `--trace-sites` selects exact module FQNs validated on both
models. Defaults are 256 sampled values, the first frozen observation in each episode,
and 20,000 records maximum per mode. Hooks are removed outside selected observations.
Population counts/moments cover each observed tensor; paired errors and quantiles cover
the recorded deterministic samples unless explicitly marked full-tensor. Missing pairs,
wrong coordinates, exhausted budgets and nonfinite errors refuse rather than becoming
zero-error results. Module-call indices are not inferred sampling iterations.

These are paired whole-policy boundary diagnostics, not clean-input local-operation
replay. Observers transfer values and allocate temporaries; the entire run is marked
timing-ineligible. Clean latency, bandwidth, multirate scheduling and closed-loop
evaluation need separate measurement protocols and supported execution adapters.

## Journal and next work

The working journal is `out/artifacts/handoff/atlas-fp8/atlas_v1/next-phase/JOURNAL.md`
in the surrounding workspace. Update it at decisions, checks, failures and handoffs,
using short entries linked to detailed evidence. Remote GPU work should retain its own
short run journal and relay events/results for inclusion in the working journal.

The next compiler integration is a versioned replay adapter and exact saved-program
intake into Merlin, followed by quantization-aware reference MLIR. Preserve code bytes,
signed exponents, dynamic scales, attention bindings and explicit residual-domain guards.
Native lowering and hardware qualification require separate target legality and execution
evidence. Neither offline RMSE nor the meeting's 0.1/0.4 suggestions are acceptance gates.
