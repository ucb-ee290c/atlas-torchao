"""Read-only probe on the TinyLlama static export (real text, 2 x 256 tokens, wikitext-2 test).
Per Linear input with a static code outside 122..128: (1) bit-exactness of the power-of-two fold, (2) the induced flush /
saturation / error of a plain clamp of the exponent. Attention: p (static -8) clamp, kit-style unnormalized p, and the dynamic q/k/v codes.
usage: pack_range_probe.py MODEL_DIR EXPORT_DIR OUT_JSON SEQLEN BATCH"""
import json, sys, torch
import load_quantized, ref_numerics as rn, ref_attention as ra
model_dir, export, out, T, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : B * T].reshape(B, T)
model, _, man = load_quantized.load_quantized(model_dir, export, "cpu", mode="ref_conv", seqlen=T)
pol = json.load(open(export + "/policy.json"))
LO, HI = 122, 128
def tgt(c): return min(max(c, LO), HI)
def stats(xb, s):
    """encode xb at exponent s (int): flushed (nonzero -> byte&0x7f==0), saturated (byte 0x7E), rel L2 vs xb, decoded."""
    q = rn.e4m3_vpack(xb, s)
    dec = rn.e4m3_unpack(q) * (2.0 ** s)
    nz = xb.float() != 0
    fl = ((q & 0x7F) == 0) & nz
    sat = ((q & 0x7F) == 0x7E)
    err = ((dec - xb.float()) ** 2).sum().sqrt() / (xb.float() ** 2).sum().sqrt().clamp_min(1e-30)
    return q, dec, int(fl.sum()), int(sat.sum()), int(nz.sum()), float(err)
rec = {}
def mk(n, s):
    def pre(m, inp):
        xb = inp[0].detach().to(torch.bfloat16)
        c = 127 + s; k = tgt(c) - c
        q0, d0, fl0, sat0, nz, e0 = stats(xb, s)
        r = {"static_exp": s, "code": c, "k": k, "n_nonzero": nz, "flush0": fl0, "sat0": sat0, "relL2_orig": e0}
        if k != 0:
            xk = (xb.float() * (2.0 ** k)).to(torch.bfloat16)
            exact_scale = bool((xk.float() == xb.float() * (2.0 ** k)).all())   # bf16 scaling by 2^k is exact
            qf = rn.e4m3_vpack(xk, s + k)                                        # fold: scaled value, shifted exponent
            r["fold_scale_exact_in_bf16"] = exact_scale
            r["fold_bytes_identical"] = bool((qf == q0).all())
            r["fold_bytes_differing"] = int((qf != q0).sum())
            qc, dc, flc, satc, _, ec = stats(xb, s + k)                          # clamp: same value, shifted exponent
            r.update({"clamp_flush": flc, "clamp_sat": satc, "relL2_clamp": ec, "clamp_bytes_changed": int((qc != q0).sum()),
                      "numel": xb.numel()})
        rec[n] = r
    return pre
for n, m in model.named_modules():
    if type(m).__name__ == "RefQuantLinear":
        a = pol["sites"][m.layer_name]["activation"]; assert a["mode"] == "static"
        m.register_forward_pre_hook(mk(m.layer_name, int(a["exp"])))
# attention
att = {"p": [], "dyn": {"q": [], "k": [], "v": []}}
orig = ra._cast
cur = {}
def wrap(algo, site, operand, x, dims):
    xb = x.to(torch.bfloat16)
    if operand in ("q", "k", "v"):
        s = rn.po2_scale_exp(xb, dims).flatten() + 127
        att["dyn"][operand].append((site, s.tolist()))
    if operand == "p":
        cur["p"] = xb
    if operand == "v":
        p = cur.pop("p"); v = xb; nz = p.float() > 0
        rec_p = {"site": site, "n_nonzero": int(nz.sum())}
        pf = p.float()
        exact = pf @ v.float()
        def rel(o): return float(((o - exact) ** 2).sum().sqrt() / (exact ** 2).sum().sqrt())
        for s in (-8, -7, -6, -5):
            q, d, fl, sat, nzc, e = stats(p, s)
            rec_p[f"s{s}"] = {"flush": fl, "flush_frac": fl / max(nzc, 1), "mass_lost": float(((pf * ((q & 0x7F) == 0)).sum()) / pf.sum()),
                              "relL2_p": e, "relL2_out": rel(d @ v.float())}
        # kit style: unnormalized p~ = exp(s - rowmax) ~ p / rowmax(p) (BF16), x256, pack at unit scale, normalize by the sum of decoded codes
        pm = pf.amax(-1, keepdim=True).clamp_min(1e-30)
        pt = (pf / pm).to(torch.bfloat16)
        ptq = (pt.float() * 256.0).to(torch.bfloat16)
        q = rn.e4m3_vpack(ptq, 0); d = rn.e4m3_unpack(q)
        ssum = d.sum(-1, keepdim=True).clamp_min(1e-30)
        o_kit = (d @ v.float()) / ssum
        rec_p["fold_p_x256_unit_bytes_identical"] = bool((rn.e4m3_vpack((pf * 256.0).to(torch.bfloat16), 0) == rn.e4m3_vpack(p, -8)).all())
        rec_p["kit_style_x256_unit"] = {"relL2_out": rel(o_kit), "flush_frac": float((((q & 0x7F) == 0) & (pt.float() > 0)).sum() / max(int((pt.float() > 0).sum()), 1))}
        att["p"].append(rec_p)
    return orig(algo, site, operand, x, dims)
ra._cast = wrap
with torch.no_grad():
    model(ids, use_cache=False)
ra._cast = orig
json.dump({"tokens": B * T, "linear": rec, "attention_p": att["p"], "dyn": att["dyn"]}, open(out, "w"))
print("done", len(rec), len(att["p"]))
