import sys, numpy as np
sys.path.insert(0,'/scratch2/lorenhung/atlas-bringup/atlas-kernels'); sys.path.insert(0,'/scratch2/lorenhung/atlas-bringup/atlas-kernels/lib')
import atlas_kernels_path
import kernel_bench as bench
rows,keys,heads,kv=1,129,32,4
rng=np.random.default_rng(7)
words=[bench.bf16_words(rng.standard_normal(shape,dtype=np.float32)) for shape in ((heads,rows,64),(kv,keys,64),(kv,keys,64))]
q,k,v=(bench.bf16_values(x) for x in words)
print('q_exp',bench.fp8_scale(q),'k_exp',bench.fp8_scale(k),'v_exp',bench.fp8_scale(v))
