# Qwen3.8-Flash-Next on M5 Max: round 3 (compiler-replica prefetch)

Cumulative result, upstream 18ca8ec (its own build and Metal sources) against
this branch with default settings, CLI greedy decode, three prompts x three
interleaved repeats, outputs and MTP acceptance counters identical:

| prompt | plain decode | MTP |
| --- | ---: | ---: |
| Hamlet | 55.39 -> 56.87 t/s (**+2.67%**) | 67.29 -> 69.68 (**+3.55%**) |
| Fibonacci | 55.72 -> 57.05 (**+2.39%**) | 80.93 -> 83.76 (**+3.50%**) |
| explanation | 55.78 -> 57.14 (**+2.44%**) | 72.05 -> 74.48 (**+3.37%**) |

Prefill is unchanged in this round (every prefill candidate was rejected, see
below).  M3 Ultra is unchanged by construction: all new defaults are gated on
`ds4_gpu_device_is_m5_apple_silicon()`.

Base: upstream `qwen3.8-flash-next` at 18ca8ec (PR #5 merged plus the Q4_K
TensorOps prefill and MXFP4 decode specialization).  Same machine, pack and
harnesses as rounds 1 and 2 (`speed-bench/qwen38-m5-round1.md`, `round2.md`).

## Why the round-2 prefetch rewrites drifted

The Metal library is compiled with fast math (MTLMathModeFast), which lets the
compiler reassociate and contract.  Dumping the shipped library to AIR
(`xcrun metal -S -emit-llvm` on the concatenated runtime source with the
default macros) shows what each hot loop actually became:

| kernel | source loop | compiled loop |
| --- | --- | --- |
| `kernel_qwen4_hc_gate_mix_f16` | `acc += w * silu(l/hc)` | `x = l * (1/hc); sig = sigmoid(x); acc = acc + (x*w)*sig` — `w*silu` reassociated, no fma |
| `kernel_qwen4_hc_gate_mix_pair_f16` | `acc += w * activated[r]` (float2) | `acc = acc + w*act`, no fma |
| `kernel_mul_mv_q8_0_f32` | `sumq += q*y` x8, `sumf += sumq*d` | mid-end keeps the loop; the backend's own unrolling decides (see below) |

A rewrite that keeps the *source* order therefore does not reproduce the
shipped rounding.  The recipe that does: replicate the *compiled* op order
and pin it with `#pragma clang fp reassociate(off)` and `contract(off)` in the
new kernel, then let the per-step decode A/B prove byte-identity.

Probe on the single-token mixer (eight terms loaded ahead per lane):
`fma(w, silu, acc)` in source order: 247k/248k logits differ;
`acc + w*silu` with contraction off: 246k differ;
compiled-order replica with fma: 246k differ;
compiled-order replica without fma: **exact** (401 rows), +2.0% decode.

## Adopted

### F16 hyper-connection gate/mix, eight terms ahead (`kernel_qwen4_hc_gate_mix_f16_pf`)

M5 default, `DS4_QWEN4_HC_MIX_PREFETCH=0/1` overrides on any device.  Unit test
pins it against the plain kernel on ranks 8/72/136/200/320 x tokens 1/3.
Per-step A/B, 2048-token prefix, 384 steps: **+2.05%** plain decode
(53.55 -> 54.64 t/s), -1.99% with the variants swapped; 401 rows exact.

### Paired F16 gate/mix (T = 2 MTP verify), eight terms ahead (`kernel_qwen4_hc_gate_mix_pair_f16_pf`)

Same gate.  The backend runs the pair loop as an fma chain (a mul-then-add
replica is two ulps off; the fma replica is byte-identical), so the form has
to be found per kernel even inside one family.  CLI MTP compare, same build,
`--baseline-env` off vs `--candidate-env` on, three prompts x three
interleaved repeats, outputs and acceptance identical: Hamlet **+1.09%**
(66.72 -> 67.45 t/s), Fibonacci **+1.32%** (80.11 -> 81.17), explanation
**+1.19%** (71.33 -> 72.18).  The compare harness gained `--baseline-env`
for this kind of same-build A/B.  MTP harness: **+1.13%** (256 cycles, 273 rows exact).

### MXFP4 routed down rows, four blocks per lane ahead (`kernel_qwen4_moe_down_mxfp4_pf`)

Form sweep against the MV-exact fixture (E 2560/256, F 640/672, T 1/2/3/9,
shared Q8 slot, generic and specialized geometries): the four-term inner sum
is an fma chain; the outer `acc += s*d` form is invisible because e8m0
scales are powers of two (the product is exact either way).  Per-step decode A/B: **+0.33% / +0.41%** over 384 and 768 steps, -0.42% /
-0.43% swapped, 401 and 785 rows exact.  Small, but four runs of one sign
above the harness's ~0.3% floor; kept as an M5 default
(`DS4_QWEN4_MOE_DOWN_PREFETCH`).

### One-row Q4_K gate/up geometry for the two-row MTP passes (env screen from the round-3 plan)

The M5 single-token default (NR 1, NSG 4) stopped at `n_tokens == 1`, so the
two-row verify and batched MTP step still ran the two-row kernel at NR 2 /
NSG 2.  Geometry class, fixture-proven at T = 2.  CLI MTP compare, three
prompts x three repeats, outputs and acceptance identical:
NSG 4: Hamlet **+1.71%** (67.84 -> 69.00), Fibonacci **+2.80%** (80.76 -> 83.02), explanation **+1.04%** (72.81 -> 73.57).  NSG 8: +0.41% / +2.19% / +0.95%, so NSG 4 becomes the M5 default for `n_tokens <= 2` (`DS4_QWEN4_Q4K_MID_NR` / `_NSG` still override).  MTP harness confirmation on the long prompt: the old geometry is -1.73% against the new default (273 rows exact).

## Tooling

- `qwen38_decode_variant_bench --mtp`: one greedy speculative cycle per step on
  both sessions (same committed token lists required, logits compared after
  every cycle), reporting tokens/s, cycles and accepted tokens per cycle.
- `qwen38_mtp_compare.py --baseline-env`.
- `DS4_METAL_MV_EXT_NSG` (1..8, default 2): simdgroups per threadgroup for the
  2..16-row Q8/F16 matvecs that carry the verify rows; rows-per-threadgroup
  only, fixture-exact at 1/2/4/8 for T 2/3 and 640/641/2560 rows.  Swept with the MTP harness (256 cycles, 1.51 accepted per cycle, 273 rows exact): NSG 1/4/8 vs 2 = -0.14% / 0.00% / +0.06%, so the default stays 2.

## Prefill attribution (8192-token chunk at prefix 0, encoder timeline)

| kernel | share |
| --- | ---: |
| `kernel_qwen4_moe_mm_mid` (routed Q4_K gate/up tiles) | 28.1% |
| `kernel_qwen4_moe_mm_down` (routed MXFP4 down tiles) | 17.5% |
| `kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128` (dense Q8 tensor-op GEMMs) | 15.4% |
| `kernel_qwen4_attn_mm` | 9.2% |
| `kernel_qwen4_gdn_scan_r4` | 5.0% |
| `kernel_mul_mm_f16_f32_mpp_direct_rhs_n128` (HC low-rank GEMMs) | 4.2% |
| moe_reduce, dense_mm, hc_mix_rows, hc_norm_reuse, swiglu, conv | 1.8-2.4% each |

Routed tiles are 45.6% of the chunk and the tensor-op dense GEMMs 19.7%.

## Opt-in: MTP draft head over a vocabulary subset (MTPLX FR-Spec idea)

MTPLX drafts from the 64K most frequent tokens and verifies over the full
vocabulary; acceptance stays exact with respect to the target because the
draft is only a proposal.  In ds4 the draft head reads 636 MB of Q8 per cycle
for one argmax.  Two opt-in forms, verify rows untouched (all teacher-forced
logits and committed tokens unchanged):

- `DS4_QWEN4_MTP_DRAFT_ROWS=151936`: leading rows only (ids are assigned in
  merge order; every one of MTPLX's top-63K frequent tokens has an id below
  151,936).  CLI compare, three prompts x three repeats, outputs identical:
  **+2.0% / +1.9% / +1.8%** (Hamlet / Fibonacci / explanation); Hamlet
  accepts 36 drafts instead of 37 over 120 tokens, the others identical.
- `DS4_QWEN4_MTP_DRAFT_VOCAB=<ids>` with MTPLX's code-ranked 64K list:
  **+0.8% / +2.8% / +2.5%**, Hamlet acceptance differs (the list is
  code-ranked; prose tokens above the cut are missed more often).

Both stay off by default because the acceptance counters can move; the
outputs never did in these runs.

## Free env screens (not adopted)

- `DS4_QWEN4_MOE_DOWN_NT=8` (prefill, exact by fixture): +0.45% at a 32K frontier, inside the prefill harness's ~1% noise.
- `DS4_QWEN4_PREFILL_REUSE=1` (blocked GDN convolution): +0.15%, noise.

## Rejected

### Prefill-row tile maxima and prefiltered top-k (128K+)

The matrix-unit scorer can emit the same per-tile maxima as the decode vector
scorer (fixture-exact for T = 64 rows, ties included), which lets
`kernel_qwen4_idx_select_pre` run on prefill rows.  Chunk-interleaved A/B at
a 128K frontier (122880-token untimed prefix, two 8192-token timed chunks per
variant): **+0.11%**, final logits exact.  The selector is too small a share
of a prefill chunk to matter; not adopted (patch archived).

### Row-block-major threadgroup order for the tensor-op dense GEMMs

Index remap only (`FC_mul_mm_row_major`), fixture-exact.  Chunk-interleaved
prefill A/B at a 32K frontier (8192-token chunks, four timed runs each):
**-3.34%** with final logits exact.  Token tiles innermost stays: the
re-streamed weight blocks cost more than the activation tile re-reads they
save.

### Q4_K routed gate/up rows, next super-block requested ahead

Exact form found by the sweep: `t = ds*q - dmin` unfused, then
`sum = fma(t, y, sum)` (the third distinct pattern in this campaign).  The
one-row kernel with a 20-value staging struct rotated per block measured
**-1.83%** plain decode (+1.92% swapped, 401 rows exact): at 72% of peak
the extra register traffic costs more than the latency it hides.  Not
adopted; the exact form is recorded for a lighter word-load variant.

### Q8_0 dense matvec, packed loads and one-block-ahead prefetch

The mid-end keeps the shipped `kernel_mul_mv_q8_0_f32` loops as loops, so
the backend's own unrolling decides the rounding.  A nine-form sweep of the
replica against the exactness fixtures (serial mul/add, fma chains, pairwise
trees, split accumulators, with and without an fma on the block scale) found
exactly one byte-identical form: `sumq = fma(q, y, sumq)` over the eight
elements and `sumf = fma(sumq, d, sumf)`.  (The HC mixer is the opposite
case: no fma anywhere.  The form has to be found per kernel.)

Two exact prefetch variants both lost: packed `char4` weight loads with
one-block-ahead operands -0.66% (spilled to private memory), and 16-bit
weight loads in scalar registers -1.08%; both symmetric under swap, 401 rows
exact.  With two or three blocks per lane the rotating-register loop costs
more than it hides; the shipped loop stays.  Patch archived in the lab notes.

