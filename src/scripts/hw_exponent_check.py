"""HW-D check: does a per-row weight exponent within +-2 of the nc rule beat nc on an H-weighted error?
Error per row = tr(dW H dW^T) = ||X dW_row||^2 with X the captured layer input (H = X^T X).
Run: python -I src/scripts/hw_exponent_check.py   (CPU only)"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "16")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))
import torch
torch.set_num_threads(16)
import reference.ref_numerics as rn
from algos.granularity import scale_exp
import hw_ipt_error as h

KEYS = ["10.self_attn.q_proj", "10.mlp.gate_proj", "21.mlp.down_proj"]
SHIFTS = [-2, -1, 0, 1, 2]


def main():
    cap = h.capture(ntok=128, cache=os.path.join(h.OUT, 'tinyllama_capture128.pt'))   # 64 fit H, 64 held out
    out = []
    for key in KEYS:
        x, w = cap[key]
        xh = x[64:].to(torch.bfloat16).double()                   # held-out tokens
        xd = x[:64].to(torch.bfloat16).double()
        wb = w.to(torch.bfloat16)
        s0 = scale_exp(wb, (-1,), "nc")                          # [N,1]
        errs, flush, herrs = [], [], []
        for d in SHIFTS:
            s = s0 + d
            q = rn.e4m3_vpack(wb, s)
            wd = rn.e4m3_unpack(q).double() * torch.exp2(s.double())
            dw = wd - wb.double()
            errs.append(((xd @ dw.T) ** 2).sum(0))               # [N]
            herrs.append(((xh @ dw.T) ** 2).sum(0))
            flush.append(((q & 0x7F) == 0) & (wb.float() != 0))  # nonzero weights flushed to zero
        E = torch.stack(errs)                                    # [5,N]
        best = E.argmin(0)
        e_nc, e_best = E[2], E.min(0).values
        Hh = torch.stack(herrs)                                  # held-out error of each candidate
        h_nc = Hh[2]; h_best = Hh[best, torch.arange(w.shape[0])]
        rows_diff = (best != 2) & (e_best < e_nc)                # strictly better only (ties keep nc)
        fl_nc = int(flush[2].sum())
        fl_best = int(torch.stack(flush)[best, torch.arange(w.shape[0])].sum())
        rec = dict(layer=key, rows=w.shape[0], K=w.shape[1],
                   frac_rows_best_differs_from_nc=float(rows_diff.double().mean()),
                   mean_rel_reduction_all_rows=float(((e_nc - e_best) / e_nc.clamp(min=1e-300)).mean()),
                   total_rel_reduction=float((e_nc.sum() - e_best.sum()) / e_nc.sum()),
                   heldout_total_rel_reduction=float((h_nc.sum() - h_best.sum()) / h_nc.sum()),
                   heldout_frac_rows_improved=float((h_best < h_nc).double().mean()),
                   heldout_frac_rows_worse=float((h_best > h_nc).double().mean()),
                   mean_rel_reduction_changed_rows=float(((e_nc - e_best) / e_nc)[rows_diff].mean()) if rows_diff.any() else 0.0,
                   flushed_elems_nc=fl_nc, flushed_elems_best=fl_best,
                   frac_flushed_nc=fl_nc / w.numel(),
                   best_shift_histogram={str(d): int((best == i).sum()) for i, d in enumerate(SHIFTS)})
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(dict(layers=out, setup="TinyLlama BF16 weights, Atlas cast RNE, per-row exponent nc+d, d in [-2,2], "
                   "H from 64 WikiText-2 test tokens; ties keep nc"),
              open(os.path.join(h.OUT, "hw_exponent_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
