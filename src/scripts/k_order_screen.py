"""CPU screen of the k_order step on real TinyLlama data (docs/algos/tile-aware.md#k-order).

For down_proj of layers 0, 7, 14, 21: 64 WikiText-2 test tokens x 16 random output channels.
Operands are the real FP8 codes (per-token nc activation cast, per-row nc weight cast of the BF16
checkpoint). Channel importance = mean|x_k| over 2048 WikiText-2 TRAIN tokens (not the evaluated
tokens) x L2 norm of the weight column. For each order the K axis of both code matrices is permuted
and the bit-exact IPT chain (and the SA chain, for reference) is compared with the float64 exact sum
of the same FP8 products (identical for every order; asserted).
Run: python -I src/scripts/k_order_screen.py   (CPU only)
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("HF_HOME", "/scratch2/lorenhung/hf_cache")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))
import torch
torch.set_num_threads(16)
import mxu.accum_torch as at
import hw_ipt_error as hw
from algos.k_order import order_from_importance, perm_hash

OUT = hw.OUT
LAYERS = (0, 7, 14, 21)
RULES = ("none", "asc", "desc", "rand")
RAND_SEEDS = (0, 1, 2)


def capture(ntok=64, ncal=2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(hw.CKPT)
    model = AutoModelForCausalLM.from_pretrained(hw.CKPT, torch_dtype=torch.bfloat16).eval()
    ds = lambda split: "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    test_ids = tok(ds("test")[:20000], return_tensors="pt").input_ids[:, :ntok]
    cal_ids = tok(ds("train")[:60000], return_tensors="pt").input_ids[:, :ncal]
    mods = {li: model.model.layers[li].mlp.down_proj for li in LAYERS}
    amean, test_x, hooks = {}, {}, []
    mode = {"cal": True}

    def hook(mod, args, li):
        x = args[0][0].float()
        if mode["cal"]:
            amean[li] = amean.get(li, 0) + x.abs().sum(0)
        else:
            test_x[li] = x.clone()
    for li, m in mods.items():
        hooks.append(m.register_forward_pre_hook(lambda mod, a, li=li: hook(mod, a, li)))
    with torch.no_grad():
        for i in range(0, ncal, 512):
            model(cal_ids[:, i:i + 512])
        mode["cal"] = False
        model(test_ids)
    for h in hooks:
        h.remove()
    n = (ncal // 512) * 512
    return {li: dict(x=test_x[li], w=mods[li].weight.detach().float().clone(), amean=amean[li] / n) for li in LAYERS}


def run(li, d, nch=16):
    x, w = d["x"], d["w"]
    colnorm = w.norm(dim=0)
    imp = d["amean"] * colnorm
    torch.manual_seed(0)
    sel = torch.randperm(w.shape[0])[:nch]
    ws = w[sel]
    xq, sx = hw.cast_act(x)
    wq, sw = hw.cast_w(ws)
    scale = torch.exp2(sx.double() + sw.double().T)
    ref_bf16 = x.to(torch.bfloat16).double() @ ws.to(torch.bfloat16).double().T
    exact = hw.products(xq, wq).sum(-1) * scale
    rows = {"layer": li, "K": x.shape[1], "fp8_input_err": hw.rel(exact, ref_bf16)}
    orders = {"none": order_from_importance(imp, "none"), "asc": order_from_importance(imp, "asc"),
              "desc": order_from_importance(imp, "desc")}
    for s in RAND_SEEDS:
        orders[f"rand{s}"] = order_from_importance(imp, "rand", s)
    res = {}
    for name, p in orders.items():
        xp, wp = xq[:, p], wq[:, p]
        e2 = hw.products(xp, wp).sum(-1) * scale
        assert torch.equal(e2, exact), "exact sum changed under a permutation"
        ipt = hw.decode(at.ipt_linear_bits(xp, wp)) * scale
        sa = hw.decode(at.sa_linear_bits(xp, wp, compile=False)) * scale
        res[name] = dict(ipt=hw.rel(ipt, exact), sa=hw.rel(sa, exact), hash=perm_hash(p))
        print(li, name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    rows["orders"] = res
    return rows


def main():
    cap = capture()
    allr = [run(li, d) for li, d in cap.items()]
    for r in allr:
        base = r["orders"]["none"]
        for k, v in r["orders"].items():
            v["ipt_pct_vs_none"] = 100 * (v["ipt"] / base["ipt"] - 1)
            v["sa_pct_vs_none"] = 100 * (v["sa"] / base["sa"] - 1)
        rn = [r["orders"][f"rand{s}"] for s in RAND_SEEDS]
        r["orders"]["rand_mean"] = {m: sum(v[m] for v in rn) / len(rn) for m in ("ipt", "sa", "ipt_pct_vs_none", "sa_pct_vs_none")}
    json.dump(dict(layers=allr, setup="TinyLlama-1.1B-Chat down_proj, 64 WikiText-2 test tokens x 16 random output channels, "
                   "importance from 2048 WikiText-2 train tokens, per-token nc act / per-row nc weight cast, errors vs float64 exact sum "
                   "of the same FP8 products"), open(os.path.join(OUT, "k_order_screen.json"), "w"), indent=1)
    hdr = f"{'layer':>5} {'fp8_in':>9} | " + " | ".join(f"{n:>17}" for n in ("none", "asc", "desc", "rand(mean3)"))
    for chain in ("ipt", "sa"):
        print(f"\nrel L2 error of the {chain.upper()} chain vs exact (pct change vs none)\n" + hdr)
        for r in allr:
            o = r["orders"]
            cells = [f"{o[n][chain]:.5f}" + ("" if n == "none" else f" ({o[n][chain + '_pct_vs_none']:+.0f}%)")
                     for n in ("none", "asc", "desc", "rand_mean")]
            print(f"{r['layer']:>5} {r['fp8_input_err']:>9.5f} | " + " | ".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
