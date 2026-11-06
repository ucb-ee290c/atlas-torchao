"""Static pack-scale census of the exported FX bundles: per quantize_act (static) node, the E8M0 code 127 + exp, the op that
produces the packed tensor, the distinct consumer sites, shared-input groups, and non-pack consumers of the producer.
Read-only: parses fx/graph.txt and export/policy.json of each bundle.  usage: python fx_pack_census.py [EXPORTS_ROOT]"""
import collections, json, re, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/scratch2/lorenhung/atlas-fp8/exports-tile"
LO, HI = 122, 128
PASS = {"aten.to.dtype", "aten.to.dtype_layout", "aten.view.default", "aten.reshape.default", "aten.contiguous.default",
        "aten.transpose.int", "aten.expand.default", "aten.unsqueeze.default", "aten.permute.default", "aten.clone.default",
        "aten._unsafe_view.default", "aten.flatten.using_ints", "aten.squeeze.dim", "aten.slice.Tensor", "aten.select.int",
        "aten.detach.default", "aten.alias.default", "aten.t.default", "operator.getitem", "aten.squeeze.dims",
        "aten.unflatten.int", "aten.narrow.default", "aten.cat.default"}
LINE = re.compile(r'^\s*%(\S+) : \[num_users=(\d+)\] = call_function\[target=([^\]]+)\]\(args = \((.*)\), kwargs = (.*)\)\s*$')

def parse(path):
    nodes, order = {}, []
    for ln in open(path):
        m = LINE.match(ln)
        if not m:
            continue
        name, _, tgt, args, _kw = m.groups()
        tgt = tgt.replace("torch.ops.", "")
        if tgt.startswith("atlas."):
            tgt = tgt.replace(".default", "")
        nodes[name] = dict(t=tgt, args=args, refs=re.findall(r"%([\w\.]+)", args))
        order.append(name)
    return nodes, order

def real(nodes, n):
    for _ in range(50):
        if n in nodes and nodes[n]["t"] in PASS and nodes[n]["refs"]:
            n = nodes[n]["refs"][0]
        else:
            break
    return n

def kind(nodes, r):
    rn = real(nodes, r)
    if rn not in nodes:
        return "param" if r.startswith(("p_", "b_")) else "input"
    t = nodes[rn]["t"]
    return t.split(".")[1] if t.startswith("atlas.") else t.replace("aten.", "").split(".")[0]

def desc(nodes, n):
    if n not in nodes:
        return "input/param:" + n
    d = nodes[n]; t = d["t"]
    if t in ("aten.mul.Tensor", "aten.add.Tensor"):
        ks = [kind(nodes, r) for r in d["refs"]] + (["const"] if len(d["refs"]) == 1 else [])
        return ("mul(" if t == "aten.mul.Tensor" else "add(") + ",".join(ks) + ")"
    return t

def category(pd):
    if pd in ("mul(param,mul)", "mul(mul,add)"): return "norm_affine"
    if pd == "aten.layer_norm.default": return "layer_norm"
    if pd == "aten.softmax.int": return "attn_p"
    if pd in ("atlas.pv_matmul",): return "attn_out_pv"
    if pd == "aten.scaled_dot_product_attention.default": return "attn_out_sdpa"
    if pd in ("mul(silu,qlinear)", "mul(gelu,qlinear)"): return "gated_product"
    if pd == "mul(qlinear,sigmoid)": return "x_sigmoid_product"
    if pd in ("aten.gelu.default", "aten.silu.default"): return "bare_activation"
    if pd == "atlas.qlinear": return "linear_to_linear"
    if pd.startswith("add("): return "modulation_add"
    if pd.startswith("input/param:") or pd == "aten.im2col.default": return "raw_input"
    return "other:" + pd

def leaves(nodes, users, n):
    out = []
    for u in users[n]:
        t = nodes[u]["t"]
        if t in PASS or t.startswith("aten._assert"):
            out += leaves(nodes, users, u)
        else:
            out.append(t)
    return out

for model in ["tinyllama", "smolvla", "multitask-dit", "pi0"]:
    d = f"{ROOT}/{model}/atlas_v1_awq_tile_static"
    nodes, order = parse(d + "/fx/graph.txt")
    users = collections.defaultdict(list)
    for n in order:
        for r in nodes[n]["refs"]:
            users[r].append(n)
    sites, groups, cat_out, cat_all, nonpack = {}, collections.defaultdict(list), collections.Counter(), collections.Counter(), collections.Counter()
    for n in order:
        if nodes[n]["t"] != "atlas.quantize_act":
            continue
        a = [x.strip() for x in nodes[n]["args"].split(",")]
        if a[1] != "static":
            continue
        code = 127 + int(a[4])
        src = real(nodes, nodes[n]["refs"][0]); pd = desc(nodes, src); cat = category(pd)
        key = n
        for gi in users[n]:
            for u in users[gi]:
                if nodes[u]["t"] in ("atlas.qlinear", "atlas.qconv2d"):
                    mm = re.findall(r"%b_(\w+?)_weight_codes", nodes[u]["args"]); key = mm[0] if mm else key
                if nodes[u]["t"] == "atlas.pv_matmul":
                    key = "P:" + (a[6] if len(a) > 6 else n)
        sites.setdefault(key, []).append((cat, code)); groups[src].append(code)
        extra = sorted(set(l for l in leaves(nodes, users, src) if l not in ("atlas.quantize_act", "aten._assert_tensor_metadata.default")))
        if extra:
            nonpack[(cat, tuple(extra))] += 1
    for key, l in sites.items():
        cat = l[0][0]; out = any(not LO <= c <= HI for _, c in l)
        cat_all[cat] += 1; cat_out[cat] += int(out)
    multi = [v for v in groups.values() if len(v) > 1]
    print(f"== {model}: static consumer sites {len(sites)}, outside {sum(cat_out.values())}; source tensors {len(groups)}; "
          f"multi-consumer groups {len(multi)} (identical codes {sum(len(set(v)) == 1 for v in multi)}); "
          f"distinct (tensor,code) packs {sum(len(set(v)) for v in groups.values())}, outside {sum(1 for v in groups.values() for c in set(v) if not LO <= c <= HI)}")
    for c, v in sorted(cat_all.items(), key=lambda kv: -kv[1]):
        print(f"   {c:20s} sites {v:4d}  outside {cat_out[c]:4d}")
    for k, v in nonpack.items():
        print("   producer with non-pack consumers:", v, k)
