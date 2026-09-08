# Qwen3.8-Flash-Next on M5 Max: round 4 (prefill)

Base: upstream `qwen3.8-flash-next` at 18ca8ec plus the round-3 branch (PR #6).
Same machine, pack and harnesses as the earlier rounds.

## Where a prefill chunk goes (8192 tokens at prefix 0, encoder timeline)

| kernel | share |
| --- | ---: |
| `kernel_qwen4_moe_mm_mid` (routed Q4_K gate/up tiles) | 28.1% |
| `kernel_qwen4_moe_mm_down` (routed MXFP4 down tiles) | 17.5% |
| `kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128` (dense Q8 tensor-op GEMMs) | 15.4% |
| `kernel_qwen4_attn_mm` | 9.2% |
| `kernel_qwen4_gdn_scan_r4` | 5.0% |
| `kernel_mul_mm_f16_f32_mpp_direct_rhs_n128` (HC low-rank GEMMs) | 4.2% |
| moe_reduce, dense_mm, hc_mix_rows, hc_norm_reuse, swiglu, conv | 1.8-2.4% each |

## What the routed tiles are bound by

A production-shaped microbench (`QWEN4_BENCH=1 QWEN4_BENCH_ONLY="moe mm q4k"`: Q4_K
gate/up + MXFP4 down, 32 experts, 2048 tokens x 10 slots, lists+mid+down) runs in
16.0 ms.  Skip-variants of the kernel source (`DS4_METAL_QWEN4_SOURCE` override):

| variant | time | attribution |
| --- | ---: | --- |
| no weight dequant (A staging) | 14.45 ms | ~10% |
| no activation staging (B) | 14.96 ms | ~6% |
| no simdgroup matmuls (loads and staging dead-code-eliminated with them) | 0.70 ms | the fp16 `simdgroup_matrix` loop is ~80% |

At ~15.7 TFLOP/s on the matmul part the tiles sit near the practical rate of the
simdgroup path.  Load-side rewrites therefore move little; only the tile
geometry and the Metal 4 tensor-op route (drift class, not attempted) change the
picture.

## Adopted

### 64-token routed tiles (gate/up NT=8 beside the existing down NT=8)

`kernel_qwen4_moe_mm_mid<8>` is a new instantiation of the existing template (a
64-token tile per threadgroup: each decoded weight tile serves twice the
tokens; the 8/16/32-token remainder kernels take the tails, with a third tail
dispatch for the 32-token remainder).  Tile geometry only: every output keeps
its K order and simdgroup products; `test_moe_mm_tiles_exact` pins it against
the 32-token tiles with tails on, and `test_qwen4_moe_mm_specialize` passes.

| measurement | result |
| --- | ---: |
| microbench, 32 experts x 640 pairs, mid NT 4 -> 8 | 16.0 -> 15.66 ms (-2.1%) |
| same, mid and down NT 8 | 15.32 ms (-4.2%) |
| microbench at chunk density, 256 experts x 80 pairs, both NT 8 | 17.70 -> 17.43 ms (-1.5%) |
| prefill A/B, 8192-token chunks, 32K frontier (mid NT 8, down NT 8 both sides) | **+2.58%**, logits exact |
| prefill A/B, 4096-token chunks, 12K frontier | **+2.93%**, logits exact |
| confirmation, default on as control vs 32-token tiles, 32K | 32-token tiles **-3.41%**, logits exact |

M5 default for the Q4_K / MXFP4 pack at batches of 4096 tokens or more
(`DS4_QWEN4_MOE_MID_NT` / `DS4_QWEN4_MOE_DOWN_NT` still override).

## Rejected (all exact, all measured on the microbench first)

| variant | tile pair | note |
| --- | ---: | --- |
| single-barrier, row-coalesced epilogue | -4% (-1.6% prefill at 32K) | the per-fragment epilogue is already cheap |
| same, accumulators aliased onto the dead weight tiles | -1 to -2% | |
| 64-row tiles (two A fragments per simdgroup) | +1.3% | halves activation re-reads; too small alone and superseded by 64-token tiles |
| double-buffered staging (one barrier per K step, 26 KB tgmem) | -11% | threadgroup memory per tile decides residency |
| 32-wide K steps (8 KB tgmem, twice the barriers) | -9% | residency is not the limiter either |

### Prefill attention: query fragments in registers, tile-ahead key positions

`kernel_qwen4_attn_mm` keeps 27 KB of threadgroup memory and four barriers per
16-key tile.  A variant holding the row tile's 32 query fragments in registers
(queries staged in the K/V area) and preparing the next tile's key positions
during the current staging (three barriers, 19 KB) is byte-identical
(`test_attn_mm` pins it) but **1.9% slower** on the 32K prefill A/B: the extra
registers cost more residency than the barrier and memory savings return.
Rejected.

### Half activation operands for the routed tiles

`x` rounded to half once per layer (the same round-to-nearest-even the tiles
apply while staging) and the gate/up product carried as half into the down
tiles.  Final logits exact on the 32K prefill A/B, but only **+0.30%**: the
staging conversions are not where the tile time goes.  Rejected (patch
archived).

### Tensor-op dense GEMMs with 64-wide K steps

`kernel_mul_mm_mpp_direct_rhs` templated on the K step (half the stage calls
and barriers).  Byte-identical to the 32-wide steps on all 14 shapes of
`test_q8_prefill_variants` (so the cooperative matmul's accumulation does not
depend on the step partition), but no gain: the Q8 6144x2560 GEMM at 2048
tokens reads 1.46-1.49 ms either way (the F16 variant was not instantiated in
this probe; the Q8 GEMMs are the bulk of the share).  Rejected.

### GDN prefill scan with eight state rows per simdgroup

Same per-row chain as the four-row kernel (the compiled op mix per row is
identical), q/k/g/beta loaded once per eight rows.  Microbench at 1024 tokens:
1.11-1.13 ms -> 1.15-1.17 ms (+3%, slower).  The scan is bound by its
per-token dependent chain, not by the shared-operand re-reads; more rows per
simdgroup only lengthen the chain.  Rejected.

## What is left

Every exact lever on the routed tiles below the matmul loop itself has now
been measured; the loop runs the fp16 `simdgroup_matrix` path near its rate.
The remaining large prefill lever is the Metal 4 tensor-op route for the
routed experts (upstream ships `kernel_mul_mm_id_q4_K_f32_dbuf_mpp` and the
MXFP4 twin for the generic MoE path; the Qwen graph does not use them).  That
is a drift-class change: the M5 build already runs its dense prefill GEMMs on
tensor ops, so routed tiles on tensor ops would be consistent with the
device's existing practice, but it changes the M5 logits against the current
build and needs the user's decision and a quality gate.

## Post-round checks

- **262K frontier.** Per-frontier sweep of the upstream build vs this branch:
  frontier logits exact, decode 40.62 -> 43.51 t/s (+7.1%); its prefill column
  (792.7 -> 756.3) is thermally confounded (four 5-6 minute runs decline
  monotonically 856, 783, 730, 729 t/s).  The chunk-interleaved A/B at the
  same frontier (245K-token untimed prefix, 8192-token chunks) reads 32-token
  tiles 521.6 t/s vs the 64-token default 567.2 t/s: **+8.7%**, logits exact.
- **Prefill chunk width** at a 32K frontier (single-process runs, two passes):
  4096 -> 987 / 929 t/s, 8192 -> 1026 / 945, 16384 -> 896 / 800.  The 8192
  default stays; frontier logits are identical across the three widths.
- **Server `ignore_eos` token pick** (gap scan between the few stop ids with
  the unrolled argmax instead of a predicate per vocabulary entry): outputs
  identical, throughput unchanged within noise (56.4-56.6 vs 56.8 t/s on a
  125-token greedy completion), so the scalar loop is not on the critical
  path as estimated.  Not adopted.
