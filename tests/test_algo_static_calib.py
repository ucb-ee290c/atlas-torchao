"""static_calib: static calibrated po2 activation scales (src/algos/static_calib.py)."""
import copy
import types

import pytest
import torch
import torch.nn as nn

import algos
import ref_numerics as rn
from algos import static_calib as sc
from refutil import need_refs
from test_ladder import FAST, tiny_llama


def _run(m, b):
    return m(b)


def _ids(seed, n=16):
    return torch.randint(0, 128, (1, n), generator=torch.Generator().manual_seed(seed))


def _calib(batches):
    return lambda: (batches, _run, {"source": "synthetic", "n_samples": len(batches), "seed": 0})


def _quantize(mode="ref_linear", spec="static_calib", batches=None, model=None, options=None):
    model = model or tiny_llama()
    algo = algos.make(spec)
    rep = algos.quantize(model, mode, ["lm_head"], dict(options or FAST), algo=algo,
                         calib=_calib(batches or [_ids(10), _ids(11)]))
    return model, algo, rep


def _heavy(seed, shape=(64, 96)):
    """BF16 activations spanning many binades, with a few outliers (massive-activation-like)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * torch.exp2(torch.randn(shape, generator=g) * 3)
    x.view(-1)[:3] = torch.tensor([300.0, -180.0, 95.0])
    return x.to(torch.bfloat16)


class _Probe(nn.Module):
    """One Linear whose calibration inputs are given directly (calibrate sees exactly them)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(96, 8).to(torch.bfloat16)

    def forward(self, x):
        return self.fc(x)


def _calibrate_probe(spec, xs):
    algo = algos.make(spec)
    rec = algo.calibrate(_Probe(), xs, _run)
    return algo.exps["fc"], rec


# --- exactness on synthetic tensors -------------------------------------------------------------

def test_absmax_equals_reference_rule_on_the_calibration_set():
    xs = [_heavy(s) for s in range(3)]
    e, _ = _calibrate_probe("static_calib:method=absmax", xs)
    assert e == int(rn.po2_scale_exp(torch.cat(xs)))
    # the histogram statistic is lossless: its amax is the tensor's BF16 amax
    h = sc.bf16_abs_hist(torch.cat(xs))
    assert sc.amax_of(h) == float(torch.cat(xs).abs().max())
    assert h.sum() == torch.cat(xs).numel()


def test_mse_equals_brute_force():
    xs = [_heavy(s) for s in range(2)]
    e, _ = _calibrate_probe("static_calib:method=mse", xs)
    x = torch.cat(xs).float().flatten()
    s0 = int(rn.po2_scale_exp(x))
    best, best_err = None, None
    for s in range(s0 - 8, s0 + 4):                     # same range and tie rule as the search
        err = float(((rn.dequantize(rn.e4m3_vpack(x, s), torch.tensor(s)) - x).double() ** 2).sum())
        if best_err is None or err <= best_err:
            best, best_err = s, err
    assert e == best


def test_percentile_and_kl_are_below_absmax_with_outliers():
    xs = [_heavy(s) for s in range(2)]
    e_abs, _ = _calibrate_probe("static_calib:method=absmax", xs)
    e_p, _ = _calibrate_probe("static_calib:method=percentile,p=99", xs)
    e_kl, _ = _calibrate_probe("static_calib:method=kl", xs)
    v = torch.cat(xs).abs().float().flatten().sort().values
    assert e_p == sc.ref_exp(float(v[int(torch.ceil(torch.tensor(0.99 * v.numel()))) - 1]))
    assert e_p < e_abs and e_kl <= e_abs


def test_headroom_adds_binades():
    xs = [_heavy(1)]
    for method in ("absmax", "mse"):
        e0, _ = _calibrate_probe(f"static_calib:method={method}", xs)
        e2, rec = _calibrate_probe(f"static_calib:method={method},headroom=2", xs)
        assert e2 == e0 + 2 and rec["headroom"] == 2


def test_weight_rules():
    w = _heavy(5)
    a = algos.make("static_calib")
    assert a.weight_scale("w", w, "linear") is None                          # ref rule
    s = algos.make("static_calib:weight=ceil448").weight_scale("w", w, "linear")
    assert int(s.exp) == int(torch.ceil(torch.log2(w.float().abs().max() / 448))) and not s.dynamic
    s = algos.make("static_calib:weight=mse").weight_scale("w", w, "linear")
    x = w.float().flatten()
    s0 = int(rn.po2_scale_exp(x))
    errs = {k: float(((rn.dequantize(rn.e4m3_vpack(x, k), torch.tensor(k)) - x) ** 2).double().sum())
            for k in range(s0 - 3, s0 + 3)}
    assert errs[int(s.exp)] == min(errs.values())


def test_bad_params():
    for bad in ("static_calib:method=foo", "static_calib:weight=foo", "static_calib:per_step=true,method=mse"):
        with pytest.raises(ValueError):
            algos.make(bad)


# --- calibration data and reproducibility -------------------------------------------------------

def test_calibration_reads_the_train_split_only_and_is_seeded(monkeypatch):
    import datasets

    from algos import calib

    splits = []
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda *a, split=None, **k: splits.append(split) or {"text": ["a b c"] * 100})
    tok = lambda text, return_tensors=None: types.SimpleNamespace(  # noqa: E731
        input_ids=torch.arange(5000)[None])
    b1, r1 = calib.wikitext2_train("m", 4, 64, 1234, tokenizer=tok)
    b2, r2 = calib.wikitext2_train("m", 4, 64, 1234, tokenizer=tok)
    b3, r3 = calib.wikitext2_train("m", 4, 64, 99, tokenizer=tok)
    assert set(splits) == {"train"}
    assert r1["window_starts"] == r2["window_starts"] != r3["window_starts"]
    assert all(torch.equal(x, y) for x, y in zip(b1, b2))


def test_exponents_come_only_from_calibration_and_are_reproducible():
    need_refs()
    m1, a1, r1 = _quantize("ref_attn")
    m2, a2, r2 = _quantize("ref_attn")
    assert a1.exps == a2.exps and a1.exps
    assert r1["algo"]["calibration"]["result"]["exponents"] == a1.exps
    frozen = dict(a1.exps)
    with torch.no_grad():                               # evaluation inputs do not move the scales
        m1(_ids(99, 32))
        m1(_ids(98, 8) * 0 + 5)
    assert a1.exps == frozen
    g = r1["algo"]["granularity"]["activation"]
    assert g["linear:static"] == len([n for n in frozen if "lm_head" not in n])   # lm_head skipped
    assert g["attention.p:static"] >= 1                 # P static, q/k/v dynamic per head
    assert any(k.startswith("attention.q:dynamic") for k in g)


def test_static_reproduces_dynamic_when_calibrated_on_the_eval_input():
    """Calibrated on the very input it then sees, absmax static == the reference dynamic rule on
    every Linear, so the outputs equal --algo none except at P (static -8 vs dynamic)."""
    need_refs()
    ids = _ids(7, 24)
    base = tiny_llama()
    m0 = copy.deepcopy(base)
    algos.quantize(m0, "ref_linear", ["lm_head"], dict(FAST), algo=None)
    m1, _, _ = _quantize("ref_linear", "static_calib:p_exp=dynamic", batches=[ids], model=copy.deepcopy(base))
    with torch.no_grad():
        y0, y1 = m0(ids).logits, m1(ids).logits
    assert torch.equal(y0.view(torch.int16), y1.view(torch.int16))


def test_p_exp_never_saturates_softmax():
    p = torch.softmax(torch.randn(2, 4, 8, 8) * 20, -1).to(torch.bfloat16)
    p[0, 0, 0] = 0
    p[0, 0, 0, 0] = 1.0                                     # fully concentrated row
    q = rn.dequantize(rn.e4m3_vpack(p, -8), torch.tensor(-8))
    assert float(q.max()) == 1.0 and int(rn.po2_scale_exp(p)) <= -8


# --- per-step tables (diffusion / flow loops) ---------------------------------------------------

class _Loop(nn.Module):
    """A layer applied once per 'denoising step' with a step-dependent input range."""

    def __init__(self, steps=3):
        super().__init__()
        self.fc = nn.Linear(16, 16).to(torch.bfloat16)
        self.steps = steps

    def forward(self, x):
        for k in range(self.steps):
            x = self.fc(x * (4.0 ** k))
        return x


def test_per_step_table_and_runtime_index():
    x = torch.randn(4, 16).to(torch.bfloat16)
    algo = algos.make("static_calib:per_step=true")
    m = _Loop()
    rec = algo.calibrate(m, [x, x * 0.5], _run)
    steps = algo.step_exps["fc"]
    assert rec["per_step_layers"] == {"fc": 3} and steps == sorted(steps) and steps[2] - steps[0] >= 3
    seen = [int(algo.act_scale("fc", x, "linear").exp) for _ in range(6)]
    assert seen == steps * 2                                # call index modulo calls per forward
    assert algo.exps["fc"] == steps[-1]                     # the per-layer fallback is the max


def test_per_step_falls_back_when_call_counts_differ():
    algo = algos.make("static_calib:per_step=true")
    m = _Loop()

    def run(model, b):
        model.steps = b.shape[0]                            # 2 then 3 calls per forward
        return model(b)

    rec = algo.calibrate(m, [torch.ones(2, 16, dtype=torch.bfloat16), torch.ones(3, 16, dtype=torch.bfloat16)], run)
    assert rec["per_step_fallback"] == ["fc"] and "fc" not in algo.step_exps
    assert algo.act_scale("fc", torch.ones(1), "linear").granularity == "static"


# --- chip coverage: static casts are vpack only -------------------------------------------------

def _cast_counts(rep):
    row = next(o for o in rep["coverage"]["ops"] if o["op"] == "fp8_operand_cast")
    return row["instructions"]


def test_chip_counts_static_casts_as_vpack_without_amax(monkeypatch):
    need_refs()
    import chip_profile

    calls = []
    orig = chip_profile.count_algo_cast
    monkeypatch.setattr(chip_profile, "count_algo_cast", lambda n, dyn: (calls.append((n, dyn)), orig(n, dyn)))
    ids = _ids(3, 16)
    opts = {**FAST, "chip_mxu": "framework"}
    base = tiny_llama()
    m0 = copy.deepcopy(base)
    rep0 = algos.quantize(m0, "chip", ["lm_head"], dict(opts), algo=None)
    m1, _, rep1 = _quantize("chip", "static_calib", model=copy.deepcopy(base), options=opts)
    with torch.no_grad():
        m0(ids)
        m1(ids)
    c0, c1 = _cast_counts(rep0), _cast_counts(rep1)
    static_n = sum(n for n, _ in calls)
    assert calls and not any(dyn for _, dyn in calls)
    assert c1["vpack.bf16.fp8"] == c0["vpack.bf16.fp8"]                 # same operands cast
    assert c0["vredmax.row.bf16"] - c1["vredmax.row.bf16"] == static_n   # no amax for static ones
    assert c1["vredmax.row.bf16"] > 0                                   # q / k / v stay dynamic
