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

## Offline report draft

After a recording completes, create a new, separate report directory using the installed
wheel (or add `PYTHONPATH=src` when running from this source checkout):

```sh
python -m characterize_report --record-dir /absolute/completed-recording \
    --out-dir /absolute/new-report-directory
```

This command shares the analyzer's completion/header/action/calibration and frozen-ID
checks. It also checks the embedded manifest's canonical content digest when present.
It does not accept an unbound summary as input, load a model, fit, or rerun inference.
The header's original manifest-file byte hash is retained as provenance; the original
manifest bytes are not independently available in the recording. Payload hashes prove
internal binding, not authenticity of supplied completion files. Partial/failed reports
are retained and never marked complete; existing output directories are refused.

`summary.json` contains aggregate, task-ID and task/episode-group descriptive metrics:
signed bias, RMSE, MAE, population standard deviation/variance, maximum absolute error
and absolute-error quantiles, including each horizon/component. `per_chunk_rmse.csv`
contains one RMSE per chunk/component/comparison, with exact full/valid step counts;
`chunk_rmse_distributions.csv` reports their mean, population variance, standard deviation
and quantiles. These are distributions of chunk RMSE, not signed error bias or a
confidence interval. `metrics.csv` flattens aggregate/group component/horizon statistics.
Variance is in squared numerical units; RMSE, standard deviation and quantiles are in
the original numerical units. Pooled native-coordinate RMSE may mix incomparable
component units and must not be read as a physical distance or angle.

For each comparison the draft emits component-RMSE bars, a horizon/component heatmap
and per-component absolute-error ECDF as dependency-light SVG. `chart_data.csv` contains
their source values; every SVG and `report_manifest.json` bind its SHA256. ECDF plots
show at most 256 deterministic ranks per component by default, with the exact empirical
CDF at each shown value; this is visual subsampling, not an uncertainty estimate.
An empty valid-target population is missing/gray, never zero error. Labels are escaped
for XML and spreadsheet formula prefixes are quoted in CSV; raw labels remain in JSON.

Model-space fidelity and postprocessed native dataset-space fidelity remain separate,
with full-generated and valid-target denominators. Task grouping uses recorded IDs;
missing IDs remain explicitly unknown, and duplicate human names never merge distinct
recorded task IDs. The single-task pilot does not establish multi-task or closed-loop success. Overlapping
chunks are not treated as independent observations. Unknown joint/gripper units remain
unknown; no XYZ mapping, rotation-angle conversion, Z-axis verdict, bootstrap interval,
performance result or acceptance cutoff is inferred. This reporting draft has synthetic
CPU tests, not new measured model results.

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
