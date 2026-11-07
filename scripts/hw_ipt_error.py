"""Where does the IPT arithmetic error come from? (truncation vs per-tile BF16 rounding.)

Four K-chain results on the SAME FP8 operands (TinyLlama, real activations and weights):
  a exact      float64 sum of the exact FP8 products
  b tile-exact exact tile sum + BF16 psum, one RNE per 32-wide K tile, no truncation
  c ipt        accum_torch.ipt_linear_bits (bit-exact IPT: 17 guard bits, truncating alignment)
  d sa         accum_torch.sa_linear_bits (per-add RNE)
Run: python -I scripts/hw_ipt_error.py   (CPU only)
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("HF_HOME", "/scratch2/lorenhung/hf_cache")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch
torch.set_num_threads(16)
import ref_numerics as rn
import accum_torch as at
from algos.granularity import scale_exp

CKPT = "/scratch/agarcia05/TinyLlama-1.1B-Chat-v1.0"
OUT = "/scratch2/lorenhung/atlas-fp8/results/hw-aware"
LAYERS = [(0, "self_attn.q_proj"), (0, "mlp.gate_proj"), (10, "self_attn.q_proj"), (10, "mlp.gate_proj"),
          (21, "self_attn.q_proj"), (21, "mlp.down_proj"), (10, "mlp.down_proj")]


def capture(layers=LAYERS, ntok=64, cache=os.path.join(OUT, "tinyllama_capture.pt")):
    """Inputs of the chosen Linear layers on the first `ntok` WikiText-2 test tokens, plus BF16 weights."""
    if os.path.exists(cache):
        return torch.load(cache)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text[:20000], return_tensors="pt").input_ids[:, :ntok]
    got, hooks = {}, []
    for li, name in layers:
        mod = model.model.layers[li].get_submodule(name)
        key = f"{li}.{name}"
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, key=key, mod=mod: got.__setitem__(key, (i[0][0].float().clone(), mod.weight.detach().float().clone()))))
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    torch.save(got, cache)
    return got


def cast_act(x):
    """Run-time per-token nc cast: bytes [M,K], exponent [M,1]."""
    xb = x.to(torch.bfloat16)
    s = scale_exp(xb, (-1,), "nc")
    return rn.e4m3_vpack(xb, s), s


def cast_w(w):
    """Offline per-channel nc cast: bytes [N,K], exponent [N,1]."""
    wb = w.to(torch.bfloat16)
    s = scale_exp(wb, (-1,), "nc")
    return rn.e4m3_vpack(wb, s), s


def bf16_rne_f64(v):
    """RNE of float64 values to BF16 (kept as float64), done on the bit pattern (no double rounding)."""
    b = v.contiguous().view(torch.int64)
    b = (b + ((1 << 44) - 1) + ((b >> 45) & 1)) & ~((1 << 45) - 1)
    return b.view(torch.float64)


def products(xq, wq):
    """Exact FP8 products [M, N, K] in float64 (E4M3 x E4M3 fits in a few bits)."""
    xv = rn.e4m3_unpack(xq).double()
    wv = rn.e4m3_unpack(wq).double()
    return xv[:, None, :] * wv[None, :, :]


def tile_exact_bf16(p, vec=32):
    """(b): per tile exact sum + BF16 psum, one RNE per tile. Returns result [M,N] and the psum before each tile."""
    M, N, K = p.shape
    acc = torch.zeros(M, N, dtype=torch.float64)
    psums = []
    for t in range(0, K, vec):
        psums.append(acc)
        acc = bf16_rne_f64(p[:, :, t:t + vec].sum(-1) + acc)
    return acc, psums


def tile_stats(p, psums, vec=32):
    """Per (m,n,tile): intra-tile spread log2(max/median |nonzero product|), IPT truncation error
    (anchor = max(product exp, psum exp) + 7, LSB = 2^(anchor-31), magnitudes truncated toward zero) and
    the BF16 ulp of the tile result, for the psum chain of (b)."""
    M, N, K = p.shape
    spread, trunc, ulp = [], [], []
    for i, t in enumerate(range(0, K, vec)):
        pt = p[:, :, t:t + vec]
        a = pt.abs()
        nz = a > 0
        mx = a.amax(-1)
        med = torch.where(nz, a, torch.full_like(a, float("nan"))).nanmedian(-1).values
        sp = torch.log2(mx / med)
        e_p = torch.floor(torch.log2(mx.clamp(min=1e-300)))
        ps = psums[i].abs()
        e_s = torch.floor(torch.log2(ps.clamp(min=1e-300)))
        e = torch.where(mx > 0, e_p, torch.full_like(e_p, -1e9))
        e = torch.maximum(e, torch.where(ps > 0, e_s, torch.full_like(e_s, -1e9)))
        lsb = torch.exp2(e - 24)[..., None]
        q = torch.floor(a / lsb) * lsb
        err = (a - q).sum(-1)       # signed-magnitude truncation: error is toward zero per product, sign applied below
        err = ((a - q) * torch.sign(pt)).sum(-1)
        tot = pt.sum(-1) + psums[i]
        u = torch.exp2(torch.floor(torch.log2(tot.abs().clamp(min=1e-300))) - 7)
        ok = (mx > 0) & torch.isfinite(sp) & (tot != 0)
        spread.append(sp[ok]); trunc.append(err.abs()[ok]); ulp.append(u[ok])
    return torch.cat(spread), torch.cat(trunc), torch.cat(ulp)


def spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def rel(x, ref):
    return float(((x - ref).norm() / ref.norm()))


def decode(bits):
    return at.decode_output_bits(bits.to(torch.int32)).double()


def run_chains(xq, wq, with_sa=True):
    """a, b, c, d on bytes xq [M,K], wq [N,K] (code domain, before the power-of-two output scale)."""
    p = products(xq, wq)
    a = p.sum(-1)
    b, psums = tile_exact_bf16(p)
    c = decode(at.ipt_linear_bits(xq, wq))
    d = decode(at.sa_linear_bits(xq, wq, compile=False)) if with_sa else None
    return p, a, b, c, d, psums


def analyse(key, x, w, nch=16, perm_test=True):
    torch.manual_seed(0)
    sel = torch.randperm(w.shape[0])[:nch]
    w = w[sel]
    xq, sx = cast_act(x)
    wq, sw = cast_w(w)
    scale = torch.exp2((sx.double() + sw.double().T))            # [M,N] exact output rescale
    p, a, b, c, d, psums = run_chains(xq, wq)
    ref_bf16 = (x.to(torch.bfloat16).double() @ w.to(torch.bfloat16).double().T)
    A, B, C, D = a * scale, b * scale, c * scale, d * scale
    r = dict(layer=key, K=x.shape[1], rel_fp8_input_vs_bf16=rel(A, ref_bf16),
             rel_b=rel(B, A), rel_c=rel(C, A), rel_d=rel(D, A), rel_trunc_c_vs_b=rel(C, B))
    r["trunc_over_ipt_err"] = r["rel_trunc_c_vs_b"] / r["rel_c"]
    r["trunc_var_share"] = max(0.0, (r["rel_c"] ** 2 - r["rel_b"] ** 2) / r["rel_c"] ** 2)
    r["ipt_err_over_fp8_err"] = r["rel_c"] / r["rel_fp8_input_vs_bf16"]
    r["frac_outputs_c_ne_b"] = float((c != b).double().mean())
    sp, tr, ul = tile_stats(p, psums)
    r["spread_log2_mean"] = float(sp.mean()); r["spread_log2_p90"] = float(sp.quantile(0.9))
    rr = tr / ul
    r["tile_trunc_over_ulp_mean"] = float(rr.mean()); r["tile_trunc_over_ulp_max"] = float(rr.max())
    r["spread_vs_trunc_pearson"] = float(torch.corrcoef(torch.stack([sp, rr]))[0, 1])
    r["spread_vs_trunc_spearman"] = spearman(sp, rr)
    if perm_test:
        # tile-aware K order: sort K channels by static magnitude (mean exponent of x code + mean of w code)
        mag = torch.log2(rn.e4m3_unpack(xq).abs().clamp(min=2 ** -9)).mean(0) + \
            torch.log2(rn.e4m3_unpack(wq).abs().clamp(min=2 ** -9)).mean(0)
        perm = mag.argsort()
        pp, a2, b2, c2, _, _ = run_chains(xq[:, perm], wq[:, perm], with_sa=False)
        A2, B2, C2 = a2 * scale, b2 * scale, c2 * scale
        r["sorted_K"] = dict(rel_b=rel(B2, A2), rel_c=rel(C2, A2), rel_trunc_c_vs_b=rel(C2, B2),
                             rel_c_over_orig=rel(C2, A2) / r["rel_c"],
                             trunc_over_orig=rel(C2, B2) / max(r["rel_trunc_c_vs_b"], 1e-30))
    return r


def main():
    cap = capture()
    res = []
    for key, (x, w) in cap.items():
        r = analyse(key, x, w)
        res.append(r)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "sorted_K"}), flush=True)
    summ = dict(max_trunc_over_ipt_err=max(r["trunc_over_ipt_err"] for r in res),
                max_trunc_var_share=max(r["trunc_var_share"] for r in res),
                note="sorted_K permutes K globally by static magnitude (free offline only for down_proj and o_proj, "
                     "whose input channel order can be folded into the preceding op's output order)")
    json.dump(dict(layers=res, summary=summ, setup="TinyLlama-1.1B-Chat, 64 WikiText-2 test tokens, 16 random output channels/layer, "
                   "per-token nc act cast, per-channel nc weight cast"), open(os.path.join(OUT, "hw_ipt_error.json"), "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
