# Handoff: repeat the SmolVLA FP8 QAT study on `smolvla_libero`

**TL;DR:** Another agent has built and validated a QAT pipeline for SmolVLA against the repo's `ref_conv`
FP8 profile, on `lerobot/smolvla_base` + `lerobot/svla_so100_pickplace` ("model 1"). Your job is to run the
**same study on "model 2": `lerobot/smolvla_libero` + `lerobot/libero`**. That is the checkpoint the repo's
FX / tile_flow hardware tooling uses (`src/scripts/tile_flow/smolvla_phase_capture.py`). Reuse the scripts in
`src/qat/` unchanged where you can. **Do not redo any model-1 work.**

## 1. What already exists (do not repeat)

Model 1 (`smolvla_base` / `svla_so100_pickplace`) is done or finishing on another server:
- QAT seeds 0–2, a no-quant control, an AWQ+GPTQ bake, a 1,000-chunk test eval, and AWQ+GPTQ→QAT.
- Do not run anything on model 1. Do not copy model-1 numbers into model-2 tables.

Scripts, all in `src/qat/` (read `src/qat/README.md` first):

| file | what it does |
|---|---|
| `smolvla.py` | QAT trainer (torchrun, multi-GPU). The student is the real `quant_modes.quantize(..., "ref_conv")` model with its RefQuantLinear / RefQuantConv2d swapped for straight-through-estimator copies, plus the attention casts patched. Knowledge distillation from a frozen bf16 teacher. Checks bit-exact parity with `ref_conv` before training. `--init DIR` starts the student from another checkpoint (e.g. the AWQ+GPTQ bake); the teacher always stays `--policy`. |
| `eval_qat.py` | Final held-out evaluation (torchrun). Covers bf16, `ref_conv` and every `--ckpt NAME=DIR` (each evaluated in FP8 `ref_conv`) on the same frames and noise draws. Reports chunk rel_l2 vs the ORIGINAL bf16, % gap recovered (squared error, paired per frame, SE and bootstrap CI), the teacher-forced velocity MSE (low-noise metric), GT MAE, and per-episode / per-dim breakdowns. Writes `.json` + `.md`. Checks itself against `eval_policy.run_mode` (`meta.matches_eval_policy_run_mode` must be all true). |
| `bake_ptq.py` | Runs the repo's `gptq_awq:method=awq_gptq` under `ref_conv`, calibrated on TRAIN episodes only. Bakes the result into a plain bf16 checkpoint and checks bit for bit that plain `ref_conv` on it equals the PTQ model; refuses to save otherwise. |
| `plot_progress.py` | Plots from `qat_log.json` / `train.log`. **It hard-codes model-1 run directory names and 6 action dims (`j0..j4, grip`).** Copy and adapt it for model 2, which has 7 action dims. |

## 2. Model 2 facts (checked on the Hub 2026-11-19)

- `lerobot/smolvla_libero`:
  - state 8-dim, action **7-dim**, 2 cameras (`camera1`, `camera2`);
  - chunk 50, 10 Euler steps, 16 VLM layers;
  - its preprocessor already renames the dataset's `observation.images.image/image2` to `camera1/camera2` and stores normalizer stats.
- `lerobot/libero`: 1,693 episodes, 273,465 frames, 10 fps, about 1.9 GB. The model is 0.9 GB.
- **Normalization:** try `--norm-stats checkpoint` first. This checkpoint was fine-tuned on this dataset, so its own stats should be the right ones, unlike model 1 where `dataset` was required. The guard in `src/loaders/policy_norm.py` refuses missing stats. If it refuses, use `--norm-stats dataset` and write down why.
- **Test protocol:** match the FX capture protocol, `--n-test-episodes 32 --eval-seed 0` (`smolvla_phase_capture.py:37`). Pass the same `--n-test-episodes 32` to `smolvla.py`, `bake_ptq.py` and `eval_qat.py`, so the test, val and train splits agree everywhere.
- **Caveat to report:** the test episodes were probably in `smolvla_libero`'s own fine-tuning data. That's fine for a fidelity-to-bf16 study, but say so.

## 3. Lessons from model 1 (save yourself the time)

- **Parity first.** Run `smolvla.py ... --parity-only`. The log line `parity vs ref_conv: {"deterministic": true, ...}` must be true. Never train with `--allow-parity-fail`.
- **Speed.** Single GPU is about 24 s per step (8 frames). torchrun on 4 GPUs is about 6.5 s per step, and 500 steps (`--epochs 2`, 2,000 frames, `--accum 8`) take about 65 min. The student forward and backward is the bottleneck. Keep GPUs on one CPU socket per run where possible: crossing sockets slowed the all-reduce.
- **Data loading.** DataLoader workers need a fresh lerobot `VideoDecoderCache` (already done via `worker_init_fn=_fresh_decoder_cache`). If you see torchcodec "Could not push packet to decoder", that's the cause.
- **Val is noisy.** Even 128 val frames give ±10–25 points of SE on gap recovered. Use val only to watch training. Claims come from `eval_qat.py` with ≥500 frames × 2 draws.
- **Gap recovered is on squared error.** For example, 27.8% gap recovered was a 15% drop in rel_l2. Say so on every slide.
- **Hyperparameters that worked:** `--loss vel --lr 3e-6 --epochs 2 --accum 8`, with `embed_tokens` and `lm_head` frozen (default). Use these unchanged for the first pass so the two models compare directly.
- **The no-quant control matters.** On model 1 the control (`--no-quant`, same everything else) reached about +24% on val even though its own loss rose, so QAT's credit is only what it adds above the control.
- **Model 1 reference points** (for context, not for model-2 tables): `ref_conv` rel_l2 0.0558; QAT seed 0 test gap 27.8% ± 10.3; AWQ+GPTQ alone about +1% ± 13 (100 frames).

## 4. The work, in order

Run from the repo root. GPU ids are placeholders; use what's free on your server.

```bash
P=lerobot/smolvla_libero; D=lerobot/libero; O=src/out/qat_libero
COMMON="--policy $P --dataset $D --norm-stats checkpoint --n-test-episodes 32"

# 0. Parity + step-0 val (must say deterministic true)
torchrun --standalone --nproc_per_node 4 src/qat/smolvla.py $COMMON --out-dir $O/parity --parity-only

# 1. QAT "none" recipe (round-to-nearest start): 3 seeds + the no-quant control
torchrun ... src/qat/smolvla.py $COMMON --out-dir $O/vel_s0   --loss vel --lr 3e-6 --epochs 2 --seed 0 --train-seed 1234
torchrun ... src/qat/smolvla.py $COMMON --out-dir $O/vel_s1   --loss vel --lr 3e-6 --epochs 2 --seed 1 --train-seed 1235
torchrun ... src/qat/smolvla.py $COMMON --out-dir $O/vel_s2   --loss vel --lr 3e-6 --epochs 2 --seed 2 --train-seed 1236
torchrun ... src/qat/smolvla.py $COMMON --out-dir $O/noquant  --loss vel --lr 3e-6 --epochs 2 --seed 0 --train-seed 1234 --no-quant

# 2. AWQ+GPTQ bake (one GPU, ~15 min; must print "bake check ... deterministic: true")
python src/qat/bake_ptq.py $COMMON --out-dir $O/ptq_awq_gptq

# 3. AWQ+GPTQ -> QAT: 3 seeds, the same settings plus --init
torchrun ... src/qat/smolvla.py $COMMON --init $O/ptq_awq_gptq/checkpoint --out-dir $O/gptq_qat_s0 --loss vel --lr 3e-6 --epochs 2 --seed 0 --train-seed 1234
#   (and s1 / s2 with seeds 1/1235 and 2/1236)

# 4. One final eval of everything on the same frames
torchrun ... src/qat/eval_qat.py --policy $P --dataset $D --norm-stats checkpoint --n-test-episodes 32 \
  --ckpt vel_s0=$O/vel_s0/checkpoint --ckpt vel_s1=$O/vel_s1/checkpoint --ckpt vel_s2=$O/vel_s2/checkpoint \
  --ckpt noquant=$O/noquant/checkpoint --ckpt awq_gptq=$O/ptq_awq_gptq/checkpoint \
  --ckpt gptq_qat_s0=... --ckpt gptq_qat_s1=... --ckpt gptq_qat_s2=... \
  --n-frames 500 --draws 2 --out $O/eval_libero.json
```

Notes:
- Run training jobs in the background with logs (`> $O/<run>/train.log 2>&1`). Run two jobs side by side if you have 7–8 GPUs.
- With `--init`, val step 0 is the AWQ+GPTQ model, so that run's val "gap recovered" is relative to AWQ+GPTQ, not `ref_conv`. Only `eval_qat.py` compares everything against `ref_conv`.
- Before the full eval, smoke-test `eval_qat.py` with `--n-frames 6 --draws 1` on one checkpoint.

## 5. Deliverables

1. `src/out/qat_libero/RESULTS.md`, starting with a TL;DR, containing:
   - the `eval_qat` table (rel_l2, gap recovered ± SE and CI, frames improved, velocity gap, GT MAE);
   - QAT seed mean ± std;
   - QAT minus the control;
   - AWQ+GPTQ alone vs AWQ+GPTQ→QAT;
   - per-dim and per-episode tables;
   - the caveats (base vs fine-tuned, test episodes possibly seen in fine-tuning, squared-error definition).
2. The `eval_libero.json` / `.md`, every `qat_log.json` and `train.log`, `bake.json`.
3. Plots: val curves, linear training loss, and a "gap recovered vs default FP8" bar chart. Adapt `plot_progress.py` for 7 dims.

## 6. Rules

- Don't touch model-1 outputs (`src/out/qat/`) or the model-1 checkpoints.
- Don't change files outside `src/qat/` unless a model-2 bug forces it; then record the change in RESULTS.md.
- Don't commit or push unless the user asks.
- If parity or the bake check fails, stop and report. Don't work around it.
- Calibration and training data come from train episodes only. Val and test are never trained or calibrated on.
