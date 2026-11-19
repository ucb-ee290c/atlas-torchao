"""Atlas hardware cost model: per-layer shape, DMA, MXU and flush accounting, and pop-scale applicability.

Every hardware number comes from `hw_params` (RTL wins over the frozen spec); nothing is hardcoded
here except model dimensions. Labels used in outputs:

  MEASURED   read from an export (shapes, code histograms, scale exponents)
  DERIVED    arithmetic on hw_params constants (cycle counts, seconds at CLOCK_HZ)
  INFERRED   reasoning over RTL/spec text that no simulation confirmed
  NOT MEASURED  needs captured activations

Part 1 (shape/DMA/MXU): `layer_cost`, `aggregate`.
Part 2 (pop scale): `trace_layer_graph` records the real op graph of one HF Llama decoder layer on the
meta device; `consumer_class` and `rescale_plan` classify each linear's output consumer and cost the
rescale mechanism.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import chip.hw_params as hp

# ---- model dimensions (TinyLlama-1.1B-Chat config.json; dimensions are model facts, not hw constants)
TINYLLAMA = dict(hidden_size=2048, intermediate_size=5632, num_attention_heads=32,
                 num_key_value_heads=4, num_hidden_layers=22, vocab_size=32000)

QUANTIZED_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ------------------------------------------------------------------------------------------------
# Part 1: shape / DMA / MXU
# ------------------------------------------------------------------------------------------------
def pad_to(n: int, m: int = hp.TILE) -> int:
    """Smallest multiple of `m` that is >= n."""
    return -(-n // m) * m


def pad_waste(n: int, m: int = hp.TILE) -> tuple[int, float]:
    """(padded - n, (padded - n) / padded) for one dimension."""
    p = pad_to(n, m)
    return p - n, (p - n) / p


def layer_type(name: str) -> str:
    """'model.layers.3.mlp.down_proj' -> 'down_proj'."""
    return name.rsplit(".", 1)[-1]


def mxu_cycles(tiles: int, tokens: int, mxu: str = "ipt") -> int:
    """DERIVED: MXU cycles for `tiles` weight tiles and `tokens` activation rows, weights resident per tile.

    The sequencer iterates `tileRows` = 32 rows for every matmul whatever the token count
    (InnerProductTreesSequencer.scala:191-193), so the cost is ceil(tokens/32) blocks of MXU_ISSUE_GAP
    per tile, minimum one block: decode at T=1 costs as much MXU time as T=32. The pipeline fill
    (MXU_MATMUL_LATENCY) is paid once. `hw_params.mxu_tile_cycles` now always returns the 32-row issue gap and is
    NOT used here for that reason. Weight push (TILE cycles per tile, MXU_PUSH_POP_LATENCY fill) is
    assumed hidden behind the previous tile's matmul by the two weight slots (INFERRED).
    """
    blocks = max(1, -(-tokens // hp.TILE))
    return tiles * blocks * hp.MXU_ISSUE_GAP + hp.MXU_MATMUL_LATENCY[mxu]


def dma_cycles_tiled(nbytes: int) -> float:
    """DERIVED: weight bytes streamed in 4 KiB transfers (needs a tile-major re-layout, see layout notes)."""
    return hp.dma_load_cycles(nbytes)


def dma_cycles_strided(tiles: int) -> int:
    """DERIVED UPPER BOUND, serialized (one transfer in flight at the isolated 32 B latency): weights left
    row-major [N,K]; one tile is 32 pieces of 32 B (one K-slice of 32 output channels), each a separate 32 B
    transfer at DMA_LOAD_CYCLES[32]. The pipelined cost with several transfers outstanding is not measured
    (the port streams about 23 cycles per 32 B beat, which would put a tile near 736 cycles) and the per-transfer
    command overhead is unknown."""
    return tiles * hp.TILE * hp.DMA_LOAD_CYCLES[32]


def compute_bound_blocks(dma: float, tiles: int, mxu: str = "ipt") -> int:
    """Smallest block count b (32 tokens each) with mxu_cycles(tiles, 32*b) >= dma."""
    lat = hp.MXU_MATMUL_LATENCY[mxu]
    return max(1, math.ceil((dma - lat) / (tiles * hp.MXU_ISSUE_GAP)))


def compute_bound_blocks_with_stores(Kp: int) -> float:
    """DERIVED caveat to compute_bound_blocks: per output tile (32 channels) the weight column of Kt tiles is
    loaded once (Kt x 764 cycles) but every 32-token block also stores one BF16 output tile (2 KiB, half of a
    4 KiB store at DMA_STORE_CYCLES), on the same DMA port. Compute-bound needs
    b x Kt x MXU_ISSUE_GAP >= Kt x load_per_tile + b x store_per_tile, i.e. b >= Kt x load / (Kt x gap - store).
    Returns inf when the stores alone exceed the MXU time (Kt x 32 <= store cycles)."""
    kt = Kp // hp.TILE
    load = hp.DMA_LOAD_CYCLES[4 * hp.KIB] / (4 * hp.KIB // hp.WEIGHT_SLOT_BYTES)
    store = hp.DMA_STORE_CYCLES[4 * hp.KIB] / (4 * hp.KIB // (2 * hp.WEIGHT_SLOT_BYTES))
    den = kt * hp.MXU_ISSUE_GAP - store
    return math.inf if den <= 0 else kt * load / den


def vmem_activation_token_cap(Kp: int) -> int:
    """Tokens whose FP8 activations (K bytes each) alone fill VMEM; weights, outputs and scratch not counted."""
    return hp.VMEM_BYTES // Kp


@dataclass
class LayerCost:
    name: str
    type: str
    N: int
    K: int
    N_pad: int
    K_pad: int
    pad_waste_n: float            # fraction of the padded N that is padding
    pad_waste_k: float
    pad_waste_elems: float        # fraction of padded elements that are padding
    fp8_bytes: int                # N*K (1 byte per E4M3 code), unpadded
    fp8_bytes_padded: int
    tiles: int                    # 32x32 tiles after padding
    dma_cycles: float             # tiled layout, 4 KiB transfers
    dma_cycles_strided: int       # as-stored row-major, 32 B transfers (ASSUMED serialized)
    mxu_cycles_block: int         # one 32-token block
    mxu_cycles_T: int
    T: int
    reuse_blocks_continuous: float  # DMA == MXU, ignoring pipeline fill
    compute_bound_blocks: int
    compute_bound_tokens_min: int   # smallest token count that reaches the integer block count
    compute_bound_blocks_with_stores: float  # DERIVED: also charging the BF16 output stores to the DMA port
    vmem_activation_token_cap: int  # tokens of FP8 activations that fit VMEM at this K (nothing else resident)
    mxu_over_dma_T: float           # > 1 means compute bound at T
    fits_vmem: bool                 # the whole weight (not counting activations) fits in VMEM_BYTES
    vmem_fraction: float


def layer_cost(name: str, N: int, K: int, T: int = 32, mxu: str = "ipt") -> LayerCost:
    Np, Kp = pad_to(N), pad_to(K)
    tiles = (Np // hp.TILE) * (Kp // hp.TILE)
    nbytes = Np * Kp
    dma = dma_cycles_tiled(nbytes)
    blk = mxu_cycles(tiles, hp.TILE, mxu)
    mT = mxu_cycles(tiles, T, mxu)
    b = compute_bound_blocks(dma, tiles, mxu)
    return LayerCost(
        name=name, type=layer_type(name), N=N, K=K, N_pad=Np, K_pad=Kp,
        pad_waste_n=pad_waste(N)[1], pad_waste_k=pad_waste(K)[1],
        pad_waste_elems=1 - (N * K) / nbytes,
        fp8_bytes=N * K, fp8_bytes_padded=nbytes, tiles=tiles,
        dma_cycles=dma, dma_cycles_strided=dma_cycles_strided(tiles),
        mxu_cycles_block=blk, mxu_cycles_T=mT, T=T,
        reuse_blocks_continuous=dma / (tiles * hp.MXU_ISSUE_GAP),
        compute_bound_blocks=b, compute_bound_tokens_min=(b - 1) * hp.TILE + 1,
        compute_bound_blocks_with_stores=compute_bound_blocks_with_stores(Kp),
        vmem_activation_token_cap=vmem_activation_token_cap(Kp),
        mxu_over_dma_T=mT / dma, fits_vmem=nbytes <= hp.VMEM_BYTES, vmem_fraction=nbytes / hp.VMEM_BYTES,
    )


def flush_stats(hist) -> dict:
    """Weight code statistics from a 256-bin histogram of uint8 E4M3 codes (MEASURED inputs).

    zero_code_frac counts (code & 0x7F) == 0, i.e. +0 and -0. The encoder flushes values below 2^-6 to
    +0 and never emits 0x80, but a code of zero also arises from a true zero, so this is an UPPER BOUND
    on flushed mass (flushed + exact zeros). Subnormal exponent-0 codes with a non-zero mantissa
    (0x01..0x07, 0x81..0x87) are counted separately: the MXU multiplier treats them as zero but the
    encoder never produces them. 0x7F/0xFF are reserved (pack clamps them to 0x7E; the MXU multiplies
    them as 480).
    """
    total = int(sum(hist))
    zero = int(hist[0]) + int(hist[0x80])
    sub = sum(int(hist[c]) for c in list(range(1, 8)) + list(range(0x81, 0x88)))
    reserved = int(hist[0x7F]) + int(hist[0xFF])
    return dict(elements=total, zero_codes=zero, zero_code_frac_upper_bound_of_flushed=zero / total,
                subnormal_codes=sub, reserved_codes=reserved)


def aggregate(layers: list[LayerCost], T_list=(1, 32, 128, 2048), mxu: str = "ipt") -> dict:
    """Totals and per-layer-type summaries. Cycles/seconds are DERIVED."""
    tot_bytes = sum(l.fp8_bytes_padded for l in layers)
    tot_dma = sum(l.dma_cycles for l in layers)
    tot_tiles = sum(l.tiles for l in layers)
    out = dict(
        layers=len(layers), fp8_bytes=sum(l.fp8_bytes for l in layers), fp8_mib=sum(l.fp8_bytes for l in layers) / hp.MIB,
        fp8_mib_padded=tot_bytes / hp.MIB, tiles=tot_tiles,
        weight_pass_dma_cycles=tot_dma, weight_pass_seconds=tot_dma / hp.CLOCK_HZ,
        weight_pass_dma_cycles_strided=sum(l.dma_cycles_strided for l in layers),
        weight_pass_seconds_strided=sum(l.dma_cycles_strided for l in layers) / hp.CLOCK_HZ,
        label="DERIVED (hw_params DMA rate; 4 KiB transfers; CLOCK_HZ)", per_T={}, per_type={},
    )
    for T in T_list:
        mx = sum(mxu_cycles(l.tiles, T, mxu) for l in layers)
        out["per_T"][str(T)] = dict(mxu_cycles=mx, mxu_seconds=mx / hp.CLOCK_HZ,
                                    dma_cycles=tot_dma, step_cycles_if_overlapped=max(mx, tot_dma),
                                    bound="compute" if mx >= tot_dma else "dma")
    for ty in sorted({l.type for l in layers}):
        ls = [l for l in layers if l.type == ty]
        dma, tl = sum(l.dma_cycles for l in ls), sum(l.tiles for l in ls)
        out["per_type"][ty] = dict(
            count=len(ls), N=ls[0].N, K=ls[0].K, tiles_each=ls[0].tiles, fp8_mib=sum(l.fp8_bytes for l in ls) / hp.MIB,
            dma_cycles=dma, dma_cycles_per_tile=dma / tl,
            compute_bound_blocks=compute_bound_blocks(dma, tl, mxu),
            compute_bound_tokens_min=(compute_bound_blocks(dma, tl, mxu) - 1) * hp.TILE + 1,
            reuse_blocks_continuous=dma / (tl * hp.MXU_ISSUE_GAP),
            compute_bound_blocks_with_stores=ls[0].compute_bound_blocks_with_stores,
            vmem_activation_token_cap=ls[0].vmem_activation_token_cap,
            fits_vmem=ls[0].fits_vmem,
        )
    return out


def cost_dict(l: LayerCost) -> dict:
    return asdict(l)


# ------------------------------------------------------------------------------------------------
# Part 2: pop scale, from the actual module graph
# ------------------------------------------------------------------------------------------------
# Ops that only move or re-index data. The VPU/XLU may spend cycles on them but they do not change values,
# so a per-tile scale applied at the pop survives them.
LAYOUT_OPS = {"view", "reshape", "transpose", "contiguous", "expand", "__getitem__", "permute", "unsqueeze",
              "squeeze", "flatten", "_unsafe_view", "clone", "movedim", "t"}
MATMUL_OPS = {"matmul", "bmm", "linear"}


def _flat_tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for i in x:
            yield from _flat_tensors(i)
    elif isinstance(x, dict):
        for i in x.values():
            yield from _flat_tensors(i)


def trace_layer_graph(cfg: dict | None = None, seq: int = 96) -> dict:
    """Run one HF LlamaDecoderLayer (eager attention, meta device, no weights) under a TorchFunctionMode and
    return the op DAG plus each linear's terminal consumers (first non-layout op through view/transpose/...).

    `seq` is chosen different from head_dim so an operand's contraction axis can be read off its shape.
    Returns {"ops": [...], "linears": {name: {"out_shape":..., "consumers": [...]}}, "attn_matmuls": [...]}.
    """
    import torch
    from torch.overrides import TorchFunctionMode
    from transformers import LlamaConfig
    from transformers.models.llama import modeling_llama as m

    c = dict(TINYLLAMA, **(cfg or {}))
    hf = LlamaConfig(hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
                     num_attention_heads=c["num_attention_heads"], num_key_value_heads=c["num_key_value_heads"],
                     num_hidden_layers=1, vocab_size=c["vocab_size"], attn_implementation="eager")
    head_dim = c["hidden_size"] // c["num_attention_heads"]

    class Rec(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.ops, self.keep, self.cur = [], [], None

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            outs = list(_flat_tensors(out))
            if outs:
                ins = list(_flat_tensors([args, kwargs]))
                self.ops.append(dict(op=getattr(func, "__name__", str(func)), ins=[id(t) for t in ins],
                                     outs=[id(t) for t in outs], module=self.cur,
                                     in_shapes=[tuple(t.shape) for t in ins], out_shapes=[tuple(t.shape) for t in outs]))
                self.keep.append((ins, outs))
            return out

    rec = Rec()
    with torch.device("meta"):
        layer = m.LlamaDecoderLayer(hf, 0)
        x = torch.empty(1, seq, c["hidden_size"], dtype=torch.bfloat16)
        # the rotary tables only matter by shape here; calling LlamaRotaryEmbedding on meta needs
        # torch.get_autocast_dtype("meta"), which torch 2.7 (the Pi0 environment) does not support
        cos = torch.empty(1, seq, head_dim, dtype=torch.bfloat16)
        sin = torch.empty(1, seq, head_dim, dtype=torch.bfloat16)
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_pre_hook(lambda mod_, a, n=name: setattr(rec, "cur", n))
            mod.register_forward_hook(lambda mod_, a, o: setattr(rec, "cur", None))
    with rec:
        layer(x, attention_mask=None, position_embeddings=(cos, sin))

    ops = rec.ops
    layer_input = id(x)
    # a residual add is an add with the layer input, or a previous residual-add output, as an operand
    resid = {layer_input}
    for o in ops:
        if o["op"] == "add" and any(i in resid for i in o["ins"]):
            o["residual_add"] = True
            resid.update(o["outs"])

    def consumers_of(tid):
        return [o for o in ops if tid in o["ins"]]

    def terminals(tid, via, seen=frozenset()):
        res = []
        seen = seen | {tid}
        for o in consumers_of(tid):
            if o["op"] in LAYOUT_OPS:
                for out in o["outs"]:
                    if out not in seen:   # contiguous()/reshape() may return their input unchanged
                        res += terminals(out, via + [o["op"]], seen)
            else:
                pos = o["ins"].index(tid)
                role = "residual_add" if o.get("residual_add") else o["op"]
                res.append(dict(op=role, arg=pos, via_layout=via, operand_shape=o["in_shapes"][pos],
                                out_shape=o["out_shapes"][0], module=o["module"]))
        return res

    linears = {}
    for o in ops:
        if o["op"] == "linear" and o["module"]:
            linears[o["module"]] = dict(out_shape=o["out_shapes"][0], consumers=terminals(o["outs"][0], []))
    attn = []
    for o in ops:
        if o["op"] == "matmul":
            attn.append(dict(out_shape=o["out_shapes"][0], in_shapes=o["in_shapes"],
                             consumers=terminals(o["outs"][0], [])))
    return dict(ops=ops, linears=linears, attn_matmuls=attn, seq=seq, head_dim=head_dim)


def consumer_class(consumers: list[dict]) -> str:
    """'fp8_matmul_operand' if every terminal consumer is a matmul/linear operand, 'residual_add' if any is a
    residual add, otherwise 'vpu_op'. Mixed fan-out collapses to the most restrictive class
    (vpu_op > residual_add > fp8_matmul_operand)."""
    classes = set()
    for c in consumers:
        if c["op"] in MATMUL_OPS:
            classes.add("fp8_matmul_operand")
        elif c["op"] == "residual_add":
            classes.add("residual_add")
        else:
            classes.add("vpu_op")
    for k in ("vpu_op", "residual_add", "fp8_matmul_operand"):
        if k in classes:
            return k
    return "unconsumed"


def contracts_over_tokens(consumer: dict, seq: int) -> bool | None:
    """For a matmul consumer, True if this operand's contraction axis is the token axis (the operand is
    the right-hand [K=tokens, N] side, which the MXU weight push wants as [N, K] and so needs a transpose)."""
    if consumer["op"] != "matmul" or consumer["arg"] != 1:
        return None
    return consumer["operand_shape"][-2] == seq


# Pop-scale conditions (RTL: ScalarCore.scala:179,184-187 gives `scaleE8M0` one scalar-file register per
# vmatpop.fp8; OutputConvStage.scala:120-173 applies a single signed exponent to all 32x32 outputs; the
# BF16 pop (vmatpop.bf16) reads no scale at all).
def rescale_plan(consumer_cls: str, act_scale: str, weight_scale_group32_uniform: bool,
                 consumer_scale_static: bool | None = None, tokens: int = hp.TILE, N: int = hp.TILE) -> dict:
    """Which mechanism applies the 2^(sx+sw) output rescale, and what it costs (DERIVED from hw_params).

    act_scale: 'static' | 'per_head' | 'per_token' (the producer's own input scale sx: static is fixed at
    export; per_head is one dynamic scalar per attention head; per_token is one dynamic exponent per row,
    which a per-tile pop scale cannot carry). Only 'static' can feed the pop.
    weight_scale_group32_uniform: all sw in each group of 32 output channels are equal (pop carries one).
    consumer_scale_static: only for fp8_matmul_operand consumers: the consumer's operand scale must be known
    before the pop (static) and constant along the consumer's K axis; None means not applicable.

    Pop scale is free iff the output feeds an FP8 matmul operand directly AND the sx is static AND the
    weight scale is constant per 32-channel group AND the consumer's scale is static. Otherwise the pop is
    BF16 (two mregs) followed by vmul(s) of 2^(sx+sw) -- one vmul if sx is static or per-head (a per-column
    tile of 2^(sx+sw) built once per layer / per head), two if sx is per-token (row factor and column
    factor) -- and, for an FP8 consumer, a
    vpack to make the operand (assumed to cost one tile elementwise op; hw_params has no vpack latency).
    Costs are per 32x32 output tile; `layer_extra_cycles` multiplies by the tile count.
    """
    t = hp.VPU_TILE_ELEMENTWISE_CYCLES
    free = (consumer_cls == "fp8_matmul_operand" and act_scale == "static" and weight_scale_group32_uniform
            and bool(consumer_scale_static))
    if free:
        plan = dict(mechanism="pop_scale (vmatpop.fp8, one 2^e per 32x32 tile)", n_vmul=0, vpack=0)
    else:
        n_vmul = 2 if act_scale == "per_token" else 1
        plan = dict(mechanism=f"vmatpop.bf16 + {n_vmul} x vmul 2^(sx+sw)"
                              + (" + vpack" if consumer_cls == "fp8_matmul_operand" else ""),
                    n_vmul=n_vmul, vpack=1 if consumer_cls == "fp8_matmul_operand" else 0)
    plan["extra_cycles_per_tile"] = (plan["n_vmul"] + plan["vpack"]) * t
    plan["pop_scale_free"] = free
    out_tiles = max(1, -(-tokens // hp.TILE)) * (pad_to(N) // hp.TILE)
    plan["layer_extra_cycles"] = out_tiles * plan["extra_cycles_per_tile"]
    plan["why_not_free"] = [] if free else [
        r for ok, r in [
            (consumer_cls == "fp8_matmul_operand", f"consumer is {consumer_cls}, not an FP8 matmul operand"),
            (act_scale == "static", f"activation scale is not static ({act_scale})"),
            (weight_scale_group32_uniform, "weight scale exponents vary within a 32-channel group"),
            (bool(consumer_scale_static), "consumer operand scale is dynamic or unknown"),
        ] if not ok]
    if consumer_cls != "fp8_matmul_operand" and plan["why_not_free"]:
        plan["why_not_free"] = plan["why_not_free"][:1]   # the consumer alone rules it out
    return plan


def classify_linears(graph: dict | None = None) -> dict:
    """{site: dict(consumer_class, terminals, contracts_over_tokens)} for the seven linears of a Llama layer,
    plus the two attention matmuls under 'attn.qk' / 'attn.pv'."""
    g = graph or trace_layer_graph()
    out = {}
    for name, d in g["linears"].items():
        cons = d["consumers"]
        out[layer_type(name)] = dict(
            consumer_class=consumer_class(cons),
            terminals=sorted({(c["op"], c["arg"]) for c in cons}),
            via_layout=sorted({"/".join(c["via_layout"]) for c in cons}),
            contracts_over_tokens=[contracts_over_tokens(c, g["seq"]) for c in cons],
        )
    for tag, d in zip(("attn.qk", "attn.pv"), g["attn_matmuls"]):
        cons = d["consumers"]
        out[tag] = dict(consumer_class=consumer_class(cons), terminals=sorted({(c["op"], c["arg"]) for c in cons}),
                        via_layout=sorted({"/".join(c["via_layout"]) for c in cons}), contracts_over_tokens=[])
    return out


# ------------------------------------------------------------------------------------------------
# Part 3: two-term FP8 activations (src/chip/hw_two_term.py): the second MXU pass on the resident weight tile
# ------------------------------------------------------------------------------------------------
# Extra VPU instructions per ACTIVATION tile (32 tokens x 32 K, formed once and shared by all N tiles) and per
# OUTPUT tile (32 tokens x 32 channels). All elementwise ops cost VPU_TILE_ELEMENTWISE_CYCLES per BF16 tile and
# the two row reductions VREDMAX_ROW_CYCLES each (INFERRED: the fp8 unpack / pack are assumed to cost like a
# BF16 tile op; no simulation of this sequence exists).
TWO_TERM_ACT_TILE_INSTRS = {"vunpack.fp8.bf16": 1, "vmul.bf16 (x 2^sx, dequant)": 1, "vsub.bf16 (residual)": 1,
                            "vredmax.row.bf16": 1, "vredmin.row.bf16": 1, "vmul.bf16 (x 2^-sx2, pre-vpack)": 1,
                            "vpack.bf16.fp8": 1}
TWO_TERM_OUT_TILE_INSTRS = {"vmul.bf16 (x 2^(sx2+sw), pass-2 rescale)": 1, "vadd.bf16 (y1 + y2)": 1}


def two_term_layer_cost(N: int, K: int, tokens: int, mxu: str = "ipt") -> dict:
    """DERIVED cycles of one N x K linear per forward, one-term vs two-term activations.

    DMA: the weight tiles are loaded once either way (dma_cycles_tiled, about 764 cycles per 1 KiB tile).
    MXU: ceil(tokens/32) blocks x MXU_ISSUE_GAP per tile and per pass (the second pass reuses the resident tile).
    VPU extra (two-term only): residual formation + the residual's own cast per activation tile, pass-2 rescale
    and the final add per output tile. Layer time = max(DMA, MXU, extra VPU); the baseline VPU work (casts,
    output rescale of pass 1) is not modelled in either column, so the overhead is the DELTA of the extras
    against the same DMA/MXU bound. Inputs are padded to 32."""
    Np, Kp = pad_to(N), pad_to(K)
    nt, kt = Np // hp.TILE, Kp // hp.TILE
    tiles = nt * kt
    blocks = max(1, -(-tokens // hp.TILE))
    dma = dma_cycles_tiled(tiles * hp.WEIGHT_SLOT_BYTES)
    mxu1 = mxu_cycles(tiles, tokens, mxu)
    mxu2 = tiles * blocks * 2 * hp.MXU_ISSUE_GAP + hp.MXU_MATMUL_LATENCY[mxu]
    act_tile_cycles = (sum(v for k, v in TWO_TERM_ACT_TILE_INSTRS.items() if not k.startswith("vred"))
                       * hp.VPU_TILE_ELEMENTWISE_CYCLES
                       + sum(v for k, v in TWO_TERM_ACT_TILE_INSTRS.items() if k.startswith("vred"))
                       * hp.VREDMAX_ROW_CYCLES)
    out_tile_cycles = sum(TWO_TERM_OUT_TILE_INSTRS.values()) * hp.VPU_TILE_ELEMENTWISE_CYCLES
    vpu_extra = blocks * (kt * act_tile_cycles + nt * out_tile_cycles)
    vpu_instrs = blocks * (kt * sum(TWO_TERM_ACT_TILE_INSTRS.values()) + nt * sum(TWO_TERM_OUT_TILE_INSTRS.values()))
    t1, t2 = max(dma, mxu1), max(dma, mxu2, vpu_extra)
    return dict(N=N, K=K, tokens=tokens, blocks=blocks, tiles=tiles, dma_cycles=dma, mxu_cycles_1=mxu1,
                mxu_cycles_2=mxu2, vpu_extra_cycles=vpu_extra, vpu_extra_tile_instrs=vpu_instrs,
                cycles_1=t1, cycles_2=t2, overhead=t2 / t1 - 1.0,
                bound_1="dma" if dma >= mxu1 else "mxu", bound_2=max(("dma", dma), ("mxu", mxu2), ("vpu", vpu_extra),
                                                                     key=lambda p: p[1])[0],
                label="DERIVED (VPU part INFERRED)")


def two_term_free_blocks() -> float:
    """Largest block count (32 tokens each) for which the second pass is still hidden behind the weight DMA:
    2 x MXU_ISSUE_GAP x blocks <= DMA cycles per 1 KiB tile (latency and VPU terms ignored). About 11.9."""
    return hp.dma_load_cycles(4 * hp.KIB) / (4 * hp.KIB // hp.WEIGHT_SLOT_BYTES) / (2 * hp.MXU_ISSUE_GAP)


# ------------------------------------------------------------------------------------------------
# Part 3b: two-term attention operands (--act-two-term attnq | attn | all+attn; ref_attention two_term)
# ------------------------------------------------------------------------------------------------
def attention_two_term_cost(Sq: int, Skv: int, D: int, heads: int, kv_heads: int, variant: str = "qp",
                            kv: str = "vmem", mxu: str = "ipt") -> dict:
    """DERIVED cycles of one attention layer (S = Q K^T, O = P V; every query head its own pair of matmuls,
    full Sq x Skv square, no causal tile skipping, softmax and the single-term casts not modelled in either column).

    The resident operand is K (tiles of 32 keys x 32 head-dim) for S and V^T for O; the streamed operand is Q
    (Sq/32 blocks) or P. The second term only adds MXU passes against the same resident tile: matrix passes
    double, NO extra weight DMA. kv="vmem": K and V are already in VMEM (KV cache of one layer, FP8: 2 x kv_heads x
    Skv x D bytes; the 1.5 MiB VMEM holds it for TinyLlama at 2048 tokens only if little else is resident), so
    there is no DMA term. kv="hbm": the FP8 K and V tiles are loaded once per kv head (dma_cycles_tiled).
    variant "q": only S gets the second pass; "qp": S and O.
    VPU extras (INFERRED, same instruction sequence as two_term_layer_cost): per streamed tile (Q: Sq/32 x D/32;
    P: Sq/32 x Skv/32) the activation-side sequence, per output tile (S: Sq/32 x Skv/32; O: Sq/32 x D/32) the
    pass-2 rescale and the BF16 vadd. The per-(batch, head) exponent needs the max over the whole residual tile
    set, so the residual (bf16 Sq x D, and Sq x Skv for P) must be held until its reductions finish."""
    if variant not in ("q", "qp"):
        raise ValueError(variant)
    blocks = max(1, -(-Sq // hp.TILE))
    dt, st_ = pad_to(D) // hp.TILE, pad_to(Skv) // hp.TILE
    tiles = dt * st_                                     # resident tiles of K (S) and of V^T (O), per head
    act = TWO_TERM_ACT_TILE_INSTRS
    act_cyc = (sum(v for k, v in act.items() if not k.startswith("vred")) * hp.VPU_TILE_ELEMENTWISE_CYCLES
               + sum(v for k, v in act.items() if k.startswith("vred")) * hp.VREDMAX_ROW_CYCLES)
    out_cyc = sum(TWO_TERM_OUT_TILE_INSTRS.values()) * hp.VPU_TILE_ELEMENTWISE_CYCLES
    n_act, n_out = sum(act.values()), sum(TWO_TERM_OUT_TILE_INSTRS.values())
    lat = hp.MXU_MATMUL_LATENCY[mxu]
    mm1 = 2 * tiles * blocks * hp.MXU_ISSUE_GAP * heads + 2 * lat           # S and O, one pass, all heads
    mm_two = (2 if variant == "qp" else 1)                                  # matmuls that get a second pass
    mm2 = mm1 + mm_two * (tiles * blocks * hp.MXU_ISSUE_GAP * heads + lat)
    q_act, p_act = blocks * dt, blocks * st_                                 # streamed tiles per head
    s_out, o_out = blocks * st_, blocks * dt
    tiles_act = q_act + (p_act if variant == "qp" else 0)
    tiles_out = s_out + (o_out if variant == "qp" else 0)
    vpu = heads * (q_act * act_cyc + s_out * out_cyc
                   + ((p_act * act_cyc + o_out * out_cyc) if variant == "qp" else 0))
    instrs = heads * (tiles_act * n_act + tiles_out * n_out)
    dma = dma_cycles_tiled(2 * tiles * kv_heads * hp.WEIGHT_SLOT_BYTES) if kv == "hbm" else 0.0
    t1 = max(dma, mm1)
    t2 = max(dma, mm2, vpu)
    return dict(Sq=Sq, Skv=Skv, D=D, heads=heads, kv_heads=kv_heads, variant=variant, kv=kv, blocks=blocks,
                resident_tiles_per_head=tiles, dma_cycles=dma, mxu_cycles_1=mm1, mxu_cycles_2=mm2,
                vpu_extra_cycles=vpu, vpu_extra_tile_instrs=instrs, cycles_1=t1, cycles_2=t2,
                overhead=t2 / t1 - 1.0,
                bound_2=max(("dma", dma), ("mxu", mm2), ("vpu", vpu), key=lambda p: p[1])[0],
                label="DERIVED (VPU part INFERRED; softmax/casts/causal skipping not modelled)")
