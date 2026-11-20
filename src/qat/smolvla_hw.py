"""QAT of SmolVLA against a HARDWARE RECIPE (default: the FX export's atlas_v1_awq_tile_static).

TL;DR: a copy of smolvla.py (same data split, teacher, losses, optimizer, multi-GPU) whose student is a
bake_hw.py checkpoint quantized with hw_recipe.Pinned: every cast is the recipe's (pinned panel weight
exponents, static activation exponents, attention p static, q/k/v reference-dynamic). Weights are re-cast
at their pinned exponents every step (saturating, STE masked). The step-0 forward must equal the recipe's
model bit for bit (parity). The saved checkpoint + its hw_state.pt is the deployable model.
See src/qat/HW_EXPERIMENT.md. Original ref_conv script: smolvla.py (unchanged).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import eval_policy as ep  # noqa: E402
import loaders.policy_norm as policy_norm  # noqa: E402
import quant_modes  # noqa: E402
import reference.ref_attention as ra  # noqa: E402
import reference.ref_numerics as rn  # noqa: E402
from reference.ref_attention import RefQuantConv2d  # noqa: E402
from reference.ref_profile import RefQuantLinear  # noqa: E402

MODE = "ref_conv"
FQ_ENABLED = True       # --no-quant (diagnostic control): the same training with every cast off


# ---------------------------------------------------------------------------------------------
# Fake quant (STE) that reproduces the ref_conv casts
# ---------------------------------------------------------------------------------------------

from algos import base as algo_hooks  # noqa: E402
from algos.base import Scale  # noqa: E402
import algos  # noqa: E402
import hw_recipe as hw  # noqa: E402


class _STE(torch.autograd.Function):
    """forward: the given cast value (exactly what the recipe computes on); backward: identity, zero
    where the cast saturated."""

    @staticmethod
    def forward(ctx, x, xhat, keep):
        ctx.save_for_backward(keep)
        ctx.in_dtype = x.dtype
        return xhat

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return (g * keep).to(ctx.in_dtype), None, None


def _keep(x, exp, block):
    a = x.detach().to(torch.bfloat16).float().abs()
    lim = 448.0 * torch.exp2(exp.float())
    if block:
        return (a.unflatten(-1, (-1, block)) <= lim).flatten(-2)
    return a <= lim


def fq_act(algo, name, x, kind, dims=None):
    """The recipe's activation cast (algo_hooks.act_cast: static or reference-dynamic exponent) with an STE."""
    if not FQ_ENABLED:
        return x.float()
    with torch.no_grad():
        q, s, blk = algo_hooks.act_cast(algo, name, x.detach(), kind, dims)
        dq = algo_hooks.dequant(q, s, blk) if blk else rn.dequantize(q, s)
        keep = _keep(x, s, blk)
    return _STE.apply(x, dq, keep)


def fq_weight(w2d, exp, block):
    """The weight cast at its PINNED exponent (panel / per tensor), recomputed every call, with an STE."""
    if not FQ_ENABLED:
        return w2d.float()
    with torch.no_grad():
        q, _ = algo_hooks.cast_with(w2d.detach(), Scale(exp, "pinned", block=block, dynamic=False))
        dq = algo_hooks.dequant(q, exp, block)
        keep = _keep(w2d, exp, block)
    return _STE.apply(w2d, dq, keep)


class QATLinear(nn.Module):
    """RefQuantLinear._forward_algo (framework accumulation) with the weight cast on every call.
    Shares the Parameters, so state_dict keys equal the original policy's."""

    def __init__(self, ref: RefQuantLinear):
        super().__init__()
        if ref.fm is not None or ref.residual or ref.algo is None:
            raise ValueError(f"{ref.layer_name}: hw QAT needs a plain reference layer with a pinned recipe")
        self.weight, self.bias = ref.weight, ref.bias
        self.in_features, self.out_features = ref.in_features, ref.out_features
        self.layer_name, self.algo = ref.layer_name, ref.algo
        self.register_buffer("w_exp", ref.weight_scale_exp.clone(), persistent=False)
        self.w_block = ref.weight_block

    def forward(self, x):
        dtype = self.weight.dtype
        x = x.to(dtype)
        xh = fq_act(self.algo, self.layer_name, x, "linear")
        y = F.linear(xh.to(dtype), fq_weight(self.weight, self.w_exp, self.w_block).to(dtype))
        return y + self.bias if self.bias is not None else y


class QATConv2d(nn.Module):
    """RefQuantConv2d.forward + _mm_algo (im2col, the recipe's casts) with the weight cast every call."""

    def __init__(self, ref: RefQuantConv2d):
        super().__init__()
        if ref.fm is not None or ref.algo is None:
            raise ValueError(f"{ref.layer_name}: hw QAT needs a plain reference layer with a pinned recipe")
        self.weight, self.bias = ref.weight, ref.bias
        self.kernel_size, self.stride = ref.kernel_size, ref.stride
        self.padding, self.dilation = ref.padding, ref.dilation
        self.out_channels, self.layer_name, self.algo = ref.out_channels, ref.layer_name, ref.algo
        self.register_buffer("w_exp", ref.weight_scale_exp.clone(), persistent=False)
        self.w_block = ref.weight_block

    def forward(self, x):
        w = self.weight
        B, _, H, W = x.shape  # noqa: N806
        oh = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        ow = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        cols = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                        dilation=self.dilation)
        x_flat = cols.permute(0, 2, 1).reshape(B * oh * ow, -1).to(w.dtype)
        xh = fq_act(self.algo, self.layer_name, x_flat, "conv2d")
        wh = fq_weight(w.reshape(self.out_channels, -1), self.w_exp, self.w_block)
        y = F.linear(xh.to(w.dtype), wh.to(w.dtype))
        y = y.reshape(B, oh * ow, self.out_channels).permute(0, 2, 1).reshape(B, self.out_channels, oh, ow)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1).to(dtype=w.dtype, device=x.device)
        return y


_ORIG_CAST = ra._cast


def _qat_cast(algo, site, operand, x, dims):
    """ref_attention._cast replacement: the recipe's operand cast (dequantized float32) with an STE, and
    None, so ref_attention._mm runs torch.matmul in float32 on those values (= rn.dq_matmul_nt)."""
    if not FQ_ENABLED:
        return x.float(), None
    with torch.no_grad():
        q, s = _ORIG_CAST(algo, site, operand, x.detach(), dims)
        dq = rn.dequantize(q, s)
        keep = _keep(x, s, 0)
    return _STE.apply(x, dq, keep), None


@contextlib.contextmanager
def qat_numerics():
    """Route the reference attention's casts through the STE while the body runs (the bf16 teacher
    never calls ref_attention; the parity reference must run outside this context)."""
    ra._cast = _qat_cast
    try:
        yield
    finally:
        ra._cast = _ORIG_CAST


def to_qat(policy) -> dict:
    """Swap every recipe Linear / Conv2d for its STE copy (the attention patches stay)."""
    n_lin = n_conv = 0
    for name, m in list(policy.named_modules()):
        if isinstance(m, (RefQuantLinear, RefQuantConv2d)):
            parent, _, leaf = name.rpartition(".")
            new = QATLinear(m) if isinstance(m, RefQuantLinear) else QATConv2d(m)
            setattr(policy.get_submodule(parent) if parent else policy, leaf, new)
            n_lin += isinstance(m, RefQuantLinear)
            n_conv += isinstance(m, RefQuantConv2d)
    return {"linear": n_lin, "conv2d": n_conv}


@torch.no_grad()
def weight_saturation(policy):
    """Fraction of quantized weights beyond the pinned exponent's range (they saturate at 448 * 2^s)."""
    n = tot = 0
    for m in policy.modules():
        if isinstance(m, (QATLinear, QATConv2d)):
            w = m.weight.reshape(m.weight.shape[0], -1)
            n += int((~_keep(w, m.w_exp, m.w_block)).sum())
            tot += w.numel()
    return n / max(tot, 1)


# ---------------------------------------------------------------------------------------------
# The eval forward with gradients, and the teacher / student denoise interception
# ---------------------------------------------------------------------------------------------

def run_chunk(policy, batch):
    """SmolVLAPolicy.predict_action_chunk's body without @torch.no_grad / self.eval(): the same
    queue handling, prefix pass with KV cache and Euler loop. `batch` carries "_noise"."""
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION

    b = {k: v for k, v in batch.items() if k != "_noise"}
    policy.reset()
    b = policy._prepare_batch(b)
    policy._queues = populate_queues(policy._queues, b, exclude_keys=[ACTION])
    return policy._get_action_chunk(b, noise=batch["_noise"])


@contextlib.contextmanager
def intercept_denoise(model, fn):
    """sample_actions calls self.denoise_step(**kwargs) once per Euler step; route it through
    fn(orig, step_index, **kwargs) for the body."""
    orig = model.denoise_step
    count = [0]

    def wrapped(**kw):
        i = count[0]
        count[0] += 1
        return fn(orig, i, **kw)
    model.denoise_step = wrapped
    try:
        yield
    finally:
        del model.denoise_step                # back to the class method


class _StopSampling(Exception):
    """Raised from inside denoise_step to end sample_actions once the steps a loss needs are done."""


@torch.no_grad()
def teacher_trace(teacher, batch, seed, device, stop_at=None):
    """(chunk or None, [x_t], [v_t]) of the bf16 teacher per Euler step, under `seed`. stop_at=k
    ends after step k (the `vel` loss needs nothing later; on average 5.5 of 10 steps)."""
    xs, vs = [], []
    chunk = None

    def rec(orig, i, **kw):
        v = orig(**kw)
        xs.append(kw["x_t"].detach().clone())
        vs.append(v.detach().clone())
        if stop_at is not None and i >= stop_at:
            raise _StopSampling
        return v
    # suppress is outermost, so intercept_denoise and seeded_rng clean up before it swallows the stop
    with contextlib.suppress(_StopSampling), intercept_denoise(teacher.model, rec), ep.seeded_rng(seed, device):
        chunk = run_chunk(teacher, batch).float()
    return chunk, xs, vs


def student_loss(student, batch, seed, device, trace, loss, step_k, adim):
    chunk_t, xs, vs = trace
    if loss == "chunk":
        with ep.seeded_rng(seed, device):
            chunk = run_chunk(student, batch)
        return F.mse_loss(chunk.float(), chunk_t)
    steps = set(range(len(vs))) if loss == "vel-all" else {step_k}
    last = max(steps)
    preds = {}

    def forced(orig, i, **kw):
        if i in steps:
            kw["x_t"] = xs[i]                                  # teacher forcing: the teacher's x_t
            preds[i] = orig(**kw)
        if i >= last:
            raise _StopSampling                                # nothing after the last needed step
        return vs[i]                                           # keep integrating the teacher's path
    with contextlib.suppress(_StopSampling), intercept_denoise(student.model, forced), \
            ep.seeded_rng(seed, device):
        run_chunk(student, batch)
    return sum(F.mse_loss(preds[i][..., :adim].float(), vs[i][..., :adim].float()) for i in preds) / len(preds)


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def split_episodes(dataset, n_test, eval_seed, n_val):
    """eval_policy.make_data's test episodes, then n_val val episodes and the rest for training."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(dataset)
    eps = list(range(meta.total_episodes))
    random.Random(eval_seed).shuffle(eps)                       # eval_policy.make_data
    test, rest = sorted(eps[:n_test]), eps[n_test:]
    val, train = sorted(rest[:n_val]), sorted(rest[n_val:])
    if not train:
        raise SystemExit(f"{dataset}: no training episodes left ({meta.total_episodes} episodes)")
    return meta, test, val, train


def episode_frames(cfg, dataset, meta, episodes, n, seed):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset, episodes=episodes, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    idx = sorted(random.Random(seed).sample(range(len(ds)), min(n, len(ds))))
    return ds, idx


def make_batch(sample, preprocess, policy, seed, device):
    """eval_policy's preprocessed batch for a dataset sample, with the explicit initial noise of `seed`."""
    batch = ep.to_batch(sample, preprocess)
    batch["_noise"] = ep.noise_kwargs(policy, seed, device)["noise"]
    return batch


class _Frames(torch.utils.data.Dataset):
    """This rank's (step, seed, k, sample) items in schedule order; DataLoader workers decode the
    video frames ahead of the GPU (preprocessing, which moves tensors to the GPU, stays in the main process)."""

    def __init__(self, ds, items):
        self.ds, self.items = ds, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        s, i, seed, k = self.items[n]
        return s, seed, k, self.ds[i]


def _fresh_decoder_cache(_worker_id):
    """DataLoader worker_init_fn. lerobot keeps one module-wide cache of open torchcodec decoders
    and their file handles; a forked worker inherits the parent's, sharing file offsets with it
    and the other workers, and decoding fails ("Invalid data found when processing input").
    Each worker starts its own cache instead (the inherited entries are dropped, not closed)."""
    import lerobot.datasets.video_utils as vu

    vu._default_decoder_cache = vu.VideoDecoderCache()


# ---------------------------------------------------------------------------------------------
# Multi-GPU (torchrun): data parallel, every frame still at batch 1 (per-tensor FP8 scales are
# per frame, so frames must never share a tensor)
# ---------------------------------------------------------------------------------------------

def dist_setup(device):
    """(rank, world, device). Without torchrun: (0, 1, device)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, device
    import datetime

    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local}"


def dist_run(fn, idx, rank, world):
    """fn(sub_idx) -> per-frame list, run on this rank's share of idx; every rank gets the
    full list in idx order. Collective: every rank must call it."""
    mine = [(j, i) for j, i in enumerate(idx) if j % world == rank]
    out = list(zip([j for j, _ in mine], fn([i for _, i in mine])))
    if world > 1:
        import torch.distributed as dist

        parts = [None] * world
        dist.all_gather_object(parts, out)
        out = [x for p in parts for x in p]
    return [p for _, p in sorted(out, key=lambda t: t[0])]


def dist_bcast(obj, rank, world):
    if world == 1:
        return obj
    import torch.distributed as dist

    box = [obj if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


# ---------------------------------------------------------------------------------------------
# Evaluation (the eval forward; dataset units, as eval_policy reports)
# ---------------------------------------------------------------------------------------------

def per_frame_sq(preds, ref):
    return torch.tensor([float((p - r).pow(2).sum()) for p, r in zip(preds, ref)], dtype=torch.float64)


def agreement_vs(preds, ref, base_sq=None):
    """eval_policy.agreement against the ORIGINAL bf16 predictions, plus the paired gap recovered
    against base_sq (per-frame squared error of the step-0 ref_conv student)."""
    rec = ep.agreement(preds, ref)
    sq = per_frame_sq(preds, ref)
    rec["sq_err"] = float(sq.sum())
    rec["per_frame_sq"] = sq.tolist()
    if base_sq is not None:
        d = base_sq - sq                                       # > 0: closer to bf16 than ref_conv was
        rec["gap_recovered"] = float(d.sum() / base_sq.sum())
        rec["gap_recovered_se"] = float(d.std() * math.sqrt(len(d)) / base_sq.sum()) if len(d) > 1 else None
    return rec


def run_student(student, ds, idx, preprocess, postprocess, seed, device):
    """The student under ref_conv (casts always on, also under --no-quant: the control's weights
    are evaluated in FP8 like every other model)."""
    global FQ_ENABLED
    prev, FQ_ENABLED = FQ_ENABLED, True
    try:
        with qat_numerics():
            return ep.run_mode(student, ds, idx, preprocess, postprocess, seed, device)[0]
    finally:
        FQ_ENABLED = prev


class _Pick(torch.utils.data.Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, n):
        return self.idx[n], self.ds[self.idx[n]]


class Cached:
    """ds[i] from memory: frames decoded once (video decoding is ~5 s per frame here)."""

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]


def prefetch(ds, idx, workers):
    """Cached(ds restricted to idx), decoded in parallel DataLoader workers."""
    if not idx:
        return Cached({})
    loader = torch.utils.data.DataLoader(_Pick(ds, idx), batch_size=None, num_workers=workers,
                                         worker_init_fn=_fresh_decoder_cache if workers else None)
    return Cached(dict(loader))


def share(idx, rank, world):
    """The frames dist_run gives this rank."""
    return [i for j, i in enumerate(idx) if j % world == rank]


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--dataset", required=True, help="LeRobot dataset repo id")
    ap.add_argument("--out-dir", required=True, help="QAT checkpoint (lerobot policy) + qat_log.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default=None,
                    help="student start weights (default --policy), e.g. bake_ptq.py's AWQ+GPTQ checkpoint; the "
                         "teacher and every bf16 reference stay --policy, and val step 0 is then that PTQ model")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="as eval_policy --dtype; the eval must use the same")
    ap.add_argument("--skip", nargs="*", default=[], help="fqn substrings left unquantized (as eval_policy)")
    ap.add_argument("--rename-map", type=json.loads, default=None)
    ap.add_argument("--tokenizer-name", default=None)
    # data
    ap.add_argument("--n-test-episodes", type=int, default=5, help="as eval_policy (never trained on)")
    ap.add_argument("--eval-seed", type=int, default=0, help="eval_policy's --seed: fixes the test split")
    ap.add_argument("--n-frames", type=int, default=200, help="--test frames (as eval_policy --n-frames)")
    ap.add_argument("--n-val-episodes", type=int, default=4)
    ap.add_argument("--val-n", type=int, default=128, help="val frames from the val episodes")
    ap.add_argument("--val-seed", type=int, default=2000, help="val frame sample and noise seeds")
    ap.add_argument("--train-n", type=int, default=2000, help="train frames from the train episodes")
    ap.add_argument("--train-seed", type=int, default=1234)
    ap.add_argument("--parity-n", type=int, default=4, help="val frames for the bit-exact parity check")
    # training
    ap.add_argument("--loss", choices=["vel", "vel-all", "chunk"], default="vel")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="frames per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-6, help="TinyLlama: 1e-5 diverged, 3e-6 worked")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--freeze", nargs="*", default=["embed_tokens", "lm_head"],
                    help="parameter-name substrings kept frozen (unquantized embeddings, unused lm_head)")
    ap.add_argument("--train-only", default=None,
                    help="regex over parameter names: train only these (e.g. 'lm_expert|action_|state_proj')")
    ap.add_argument("--val-every", type=int, default=50, help="optimizer steps between val passes (0: per epoch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers per GPU (video decoding)")
    ap.add_argument("--no-quant", action="store_true", help="diagnostic control: the same training, casts off")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps; 0 = all")
    ap.add_argument("--parity-only", action="store_true", help="run the parity check (and step-0 val), then exit")
    ap.add_argument("--allow-parity-fail", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="after training: rel L2 vs the original bf16 on eval_policy's test frames")
    policy_norm.add_args(ap)
    return ap


def main():
    global FQ_ENABLED
    ap = build_parser()
    args = ap.parse_args()
    policy_norm.check_args(ap, args)
    FQ_ENABLED = not args.no_quant
    if not args.init or not (Path(args.init) / "hw_state.pt").exists():
        ap.error("--init must be a bake_hw.py checkpoint (with hw_state.pt)")
    if args.test:
        ap.error("--test is not supported here; evaluate with eval_hw.py")
    st = torch.load(Path(args.init) / "hw_state.pt")
    if list(args.skip) != list(st["skip"]):
        ap.error(f"--skip must equal the bake's {st['skip']}")
    rank, world, dev = dist_setup(args.device)
    args.device = dev
    main0 = rank == 0
    say = print if main0 else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = ep.torch_dtype(args.dtype)
    out = Path(args.out_dir)
    if main0:
        out.mkdir(parents=True, exist_ok=True)
    ep_args = argparse.Namespace(**{**vars(args), "seed": args.eval_seed, "policy": args.policy})

    def finish():
        if world > 1:
            import torch.distributed as dist

            dist.destroy_process_group()

    # -- data and normalization (eval_policy's protocol: guarded processors, probed on a test frame);
    #    every rank builds the same split from the same seeds
    cfg0 = ep.load_config(args.policy, dev)
    rename = ep.camera_rename_map(cfg0, args.dataset, args.rename_map)
    meta, test_eps, val_eps, train_eps = split_episodes(args.dataset, args.n_test_episodes, args.eval_seed,
                                                        args.n_val_episodes)
    ds_test, _, test_eps2, idx_test = ep.make_data(cfg0, args.dataset, args.n_test_episodes, args.n_frames,
                                                   args.eval_seed)
    assert test_eps2 == test_eps and not set(test_eps) & (set(val_eps) | set(train_eps))
    try:
        norm, norm_stats = ep.setup_normalization(ep_args, cfg0, rename, ds_test, idx_test)
    except policy_norm.NormalizationError as e:
        raise SystemExit(f"error: {e}")
    ds_val, idx_val = episode_frames(cfg0, args.dataset, meta, val_eps, args.val_n, args.val_seed)
    need = set(share(idx_val, rank, world)) | (set(idx_val[:args.parity_n]) if main0 else set())
    ds_val = prefetch(ds_val, sorted(need), args.workers)
    if args.test:
        ds_test = prefetch(ds_test, share(idx_test, rank, world), args.workers)
    ds_tr, idx_tr = episode_frames(cfg0, args.dataset, meta, train_eps, args.train_n, args.train_seed)
    say(f"{world} GPU(s); episodes: test {test_eps}, val {val_eps}, train {len(train_eps)}; "
        f"frames: val {len(idx_val)}, train {len(idx_tr)}", flush=True)

    # -- teacher (bf16, as eval_policy's bf16 mode) and its val predictions
    teacher, cfg = ep.load_policy(args.policy, dev, dtype)
    teacher.requires_grad_(False)
    preprocess, postprocess = ep.build_processors(cfg, ep_args, rename, norm_stats)
    ep.require_guarded_normalization("qat", cfg, preprocess, postprocess, norm)

    def run(policy, ds, seed):
        return lambda sub: ep.run_mode(policy, ds, sub, preprocess, postprocess, seed, dev)[0]

    def student_preds(ds, idx, seed):
        return dist_run(lambda sub: run_student(student, ds, sub, preprocess, postprocess, seed, dev),
                        idx, rank, world)

    val_ref = dist_run(run(teacher, ds_val, args.val_seed), idx_val, rank, world)

    # -- student: the real ref_conv model with STE casts
    hw_state = torch.load(Path(args.init) / "hw_state.pt")
    student, _ = ep.load_policy(args.init, dev, dtype)                 # bake_hw.py checkpoint
    qrep = algos.quantize(student, MODE, args.skip, options={}, algo=hw.Pinned(hw_state))
    n_qat = to_qat(student)

    # -- parity (rank 0): the real ref_conv model vs the QAT forward, bit for bit, on a few val frames
    parity = None
    if main0:
        par_idx = idx_val[:args.parity_n]
        ref, _ = ep.load_policy(args.init, dev, dtype)
        algos.quantize(ref, MODE, args.skip, options={}, algo=hw.Pinned(torch.load(Path(args.init) / "hw_state.pt")))
        ref_preds = run(ref, ds_val, args.val_seed)(par_idx)
        del ref
        torch.cuda.empty_cache()
        parity = ep.repeat_agreement(ref_preds, run_student(student, ds_val, par_idx, preprocess, postprocess,
                                                            args.val_seed, dev))
        parity["frames"] = len(par_idx)
        print(f"QAT modules {n_qat}, attention sites {len(qrep.get('attention_sites', []))}; "
              f"parity vs {MODE}: {json.dumps(parity)}", flush=True)
    parity = dist_bcast(parity, rank, world)
    if not parity["deterministic"] and not args.allow_parity_fail:
        finish()
        raise SystemExit("parity check failed: the QAT forward is not the ref_conv forward "
                         "(--allow-parity-fail to train anyway)")

    # -- step-0 val: the untrained student IS ref_conv (by parity)
    val0 = student_preds(ds_val, idx_val, args.val_seed)
    base_sq = per_frame_sq(val0, val_ref)
    log = {"args": vars(args), "world": world, "normalization": norm["status"], "parity": parity,
           "qat_modules": n_qat, "episodes": {"test": test_eps, "val": val_eps, "train": train_eps},
           "frames": {"val": idx_val, "train": idx_tr}, "val": [{"step": 0, **agreement_vs(val0, val_ref)}],
           "steps": []}
    say(f"val step 0 ({MODE}): rel_l2 vs bf16 {log['val'][0]['rel_l2']:.5f}", flush=True)
    if args.parity_only:
        if main0:
            (out / "qat_log.json").write_text(json.dumps(log, indent=2))
        finish()
        return

    # -- trainable parameters, fp32 masters (identical on every rank), one flat fp32 gradient buffer
    import re

    pat = re.compile(args.train_only) if args.train_only else None
    student.requires_grad_(False)
    params = []
    for n, p in student.named_parameters():
        if any(s in n for s in args.freeze) or (pat is not None and not pat.search(n)):
            continue
        p.requires_grad_(True)
        params.append(p)
    masters = [p.detach().float().clone().requires_grad_(True) for p in params]
    gflat = torch.zeros(sum(m.numel() for m in masters), dtype=torch.float32, device=dev)
    off = 0
    for m in masters:
        m.grad = gflat[off:off + m.numel()].view_as(m)        # all-reduced in one call per step
        off += m.numel()
    opt = torch.optim.AdamW(masters, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    # -- the schedule: drawn up front from one RNG, so every rank agrees on frames, noise seeds and k
    n_euler = cfg.num_steps
    rng = random.Random(args.seed)
    schedule = []
    for epoch in range(args.epochs):
        order = list(idx_tr)
        rng.shuffle(order)
        for j0 in range(0, len(order), args.accum):
            schedule.append((epoch, [(i, rng.randrange(1 << 31), rng.randrange(n_euler))
                                     for i in order[j0:j0 + args.accum]]))
    if args.max_steps:
        schedule = schedule[:args.max_steps]
    steps = len(schedule)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, steps // 10)) * 0.5 * (1 + math.cos(math.pi * i / steps)))
    mine = [(s, i, seed, k) for s, (_, items) in enumerate(schedule)
            for j, (i, seed, k) in enumerate(items) if j % world == rank]
    loader = torch.utils.data.DataLoader(_Frames(ds_tr, mine), batch_size=None, shuffle=False,
                                         num_workers=args.workers, worker_init_fn=_fresh_decoder_cache,
                                         prefetch_factor=8 if args.workers else None)
    log["trainable_params"] = sum(p.numel() for p in params)
    say(f"trainable {log['trainable_params'] / 1e6:.1f}M params in {len(params)} tensors; {steps} optimizer "
        f"steps of {args.accum} frames over {world} GPU(s), loss {args.loss}", flush=True)
    adim = cfg.action_feature.shape[0]

    last_val = [0]

    def validate(step):
        last_val[0] = step
        preds = student_preds(ds_val, idx_val, args.val_seed)  # collective: every rank calls it
        if main0:
            rec = {"step": step, **agreement_vs(preds, val_ref, base_sq)}
            log["val"].append(rec)
            print(f"val step {step}: rel_l2 {rec['rel_l2']:.5f}, gap recovered {100 * rec['gap_recovered']:.1f}% "
                  f"+- {100 * (rec['gap_recovered_se'] or 0):.1f}", flush=True)

    # -- training (student.eval(): no dropout; gradients still flow)
    student.eval()
    t_start = time.perf_counter()
    tm = dict.fromkeys(("data", "teacher", "student", "sync"), 0.0)
    it = iter(loader)

    def nxt():
        t = time.perf_counter()
        x = next(it, None)
        tm["data"] += time.perf_counter() - t
        return x

    pending = nxt()
    step = 0
    for s, (epoch, items) in enumerate(schedule):
        gflat.zero_()
        loss_sum, n_local = 0.0, 0
        while pending is not None and pending[0] == s:
            _, seed, k, sample = pending
            t = time.perf_counter()
            batch = make_batch(sample, preprocess, teacher, seed, dev)
            trace = teacher_trace(teacher, batch, seed, dev, stop_at=k if args.loss == "vel" else None)
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            tm["teacher"] += t1 - t
            with qat_numerics():
                loss = student_loss(student, batch, seed, dev, trace, args.loss, k, adim)
            (loss / len(items)).backward()                     # summed over ranks = mean over the step
            for m, p in zip(masters, params):                  # accumulate in fp32
                if p.grad is not None:
                    m.grad.add_(p.grad)
                    p.grad = None
            loss_sum += float(loss.detach())
            n_local += 1
            torch.cuda.synchronize(dev)
            tm["student"] += time.perf_counter() - t1
            pending = nxt()
        t = time.perf_counter()
        stats = torch.tensor([loss_sum, n_local], dtype=torch.float64, device=dev)
        if world > 1:
            import torch.distributed as dist

            dist.all_reduce(gflat)
            dist.all_reduce(stats)
        gn = float(torch.nn.utils.clip_grad_norm_(masters, args.clip))
        opt.step()
        with torch.no_grad():
            for m, p in zip(masters, params):
                p.copy_(m)                                     # round to the eval dtype
        sched.step()
        torch.cuda.synchronize(dev)
        tm["sync"] += time.perf_counter() - t
        step = s + 1
        if main0:
            rec = {"step": step, "epoch": epoch, "loss": float(stats[0] / stats[1]), "grad_norm": gn,
                   "lr": sched.get_last_lr()[0], "w_saturated": weight_saturation(student), "time": dict(tm)}
            log["steps"].append(rec)
            print(f"ep {epoch} step {step}/{steps} loss {rec['loss']:.3e} gn {gn:.3f} "
                  f"w_sat {100 * rec['w_saturated']:.4f}% | "
                  + " ".join(f"{k_} {v:.1f}s" for k_, v in tm.items())
                  + f" ({time.perf_counter() - t_start:.0f}s)", flush=True)
        tm = dict.fromkeys(tm, 0.0)
        end_of_epoch = s + 1 == steps or schedule[s + 1][0] != epoch
        if (args.val_every and step % args.val_every == 0) or (not args.val_every and end_of_epoch):
            validate(step)
    if last_val[0] != step:                                   # same decision on every rank
        validate(step)

    # -- save (rank 0): the trained values in a clean policy (no QAT modules, no attention patches)
    del opt, masters, gflat, loader, it
    torch.cuda.empty_cache()
    if main0:
        clean, _ = ep.load_policy(args.policy, "cpu", dtype)
        sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        missing, unexpected = clean.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"state_dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        ckpt = out / "checkpoint"
        clean.save_pretrained(ckpt)
        src = Path(args.policy)
        if not src.is_dir():
            from huggingface_hub import snapshot_download

            src = Path(snapshot_download(args.policy))
        for f in src.glob("policy_*processor*"):              # the saved pre/post pipelines
            shutil.copy2(f, ckpt / f.name)
        shutil.copy2(Path(args.init) / "hw_state.pt", ckpt / "hw_state.pt")   # the pinned recipe it deploys with
        del clean
        print("wrote", ckpt, flush=True)
        (out / "qat_log.json").write_text(json.dumps(log, indent=2))

    # -- test: eval_policy's held-out frames, both against the ORIGINAL bf16 (split over the ranks)
    if args.test:
        t_ref = dist_run(run(teacher, ds_test, args.eval_seed), idx_test, rank, world)
        del teacher
        torch.cuda.empty_cache()
        r0, _ = ep.load_policy(args.policy, dev, dtype)
        quant_modes.quantize(r0, MODE, args.skip, options={})
        t0_preds = dist_run(run(r0, ds_test, args.eval_seed), idx_test, rank, world)
        del r0
        t_qat = student_preds(ds_test, idx_test, args.eval_seed)
        if main0:
            b_sq = per_frame_sq(t0_preds, t_ref)
            log["test"] = {"frames": len(idx_test), MODE: agreement_vs(t0_preds, t_ref),
                           f"qat_{MODE}": agreement_vs(t_qat, t_ref, b_sq)}
            q = log["test"][f"qat_{MODE}"]
            print(f"test: {MODE} rel_l2 {log['test'][MODE]['rel_l2']:.5f} -> QAT {q['rel_l2']:.5f}, "
                  f"gap recovered {100 * q['gap_recovered']:.1f}% +- {100 * (q['gap_recovered_se'] or 0):.1f}",
                  flush=True)
    if main0:
        (out / "qat_log.json").write_text(json.dumps(log, indent=2))
        print("wrote", out / "qat_log.json")
    finish()


if __name__ == "__main__":
    main()
