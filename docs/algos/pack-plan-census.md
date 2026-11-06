| | tinyllama | smolvla | multitask-dit | pi0 |
|---|---|---|---|---|
| static sites | 176 | 316 | 203 | 483 |
| outside 122..128 | 114 | 244 | 88 | 127 |
| free fold (no added instruction) | 92 | 221 | 24 | 61 |
| vmul per p row block | 22 | 13 | 24 | 63 |
| vmul per tile | 0 | 2 | 9 | 0 |
| act-constant rewrite (unverified) | 0 | 6 | 30 | 1 |
| host | 0 | 2 | 1 | 2 |
| unknown producer | 0 | 0 | 0 | 0 |
| code range | 117..126 | 117..125 | 119..126 | 117..133 |
| k range (nearest in-range) | 1..5 | 1..5 | 1..3 | -5..5 |
| shared-input groups differing (of multi) | 0 of 44 | 2 of 76 | 0 of 24 | 0 of 99 |

Outside-range sites by producer class:

| producer class | tinyllama | smolvla | multitask-dit | pi0 |
|---|---|---|---|---|
| attn_out_pv | 22 | 44 | 24 | 27 |
| attn_out_sdpa | - | - | 6 | - |
| attn_p | 22 | 13 | 24 | 63 |
| bare_activation | - | 6 | 13 | 1 |
| gated_product | 11 | 24 | - | 7 |
| layer_norm | - | - | - | 3 |
| linear_to_linear | - | 9 | - | - |
| modulation_add | - | 1 | 1 | - |
| norm_affine | 59 | 145 | - | 24 |
| other:aten.index.Tensor | - | - | 1 | - |
| other:aten.sin.default | - | - | 1 | - |
| raw_input | - | 2 | 1 | 2 |
| x_sigmoid_product | - | - | 17 | - |

Mismatches against the research census: none

