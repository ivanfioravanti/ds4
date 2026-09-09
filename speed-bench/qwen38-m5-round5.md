# Qwen3.8-Flash-Next on M5 Max: round 5 (routed tiles on the tensor ops)

Base: upstream `qwen3.8-flash-next` at 6c1e836 (rounds 3 and 4 merged, PR #6).  Same machine,
pack (`qwen38-q4k`: Q4_K routed gate/up, MXFP4 routed down) and harnesses as the
earlier rounds.

Round 4 left the routed expert tiles MAC-bound: about 80% of a tile's time is the
fp16 `simdgroup_matrix` loop, and every exact lever below that loop is used up.
This round moves the two routed tile kernels onto the Metal 4 tensor ops (the M5
neural accelerators, `matmul2d` from MetalPerformancePrimitives), the same API the
dense Q8 GEMMs already use on M5.

**This is not bit-exact.**  The staged operands are unchanged (the same Qwen
dequantizers, activations rounded to half, exactly as the simdgroup tiles stage
them), but the cooperative matmul accumulates in its own order.  So the path is
opt-in (`DS4_QWEN4_MOE_MM_NAX=1` for 32-token tiles, `=2` for 64-token tiles),
off by default, and this report carries the drift and quality numbers that a
default decision needs.

## Kernels

`kernel_qwen4_moe_mm_mid_nax[64]` and `kernel_qwen4_moe_mm_down_nax[64]`
(`metal/qwen4.metal`, under `DS4_METAL_HAS_TENSOR`): 64 expert rows x 32 or 64
tokens per threadgroup, K in 32-wide steps, 128 threads.  Per K step the threads
stage the dequantized weight block (two 16-wide halves per row) and the token
block (8 floats per item, rounded to half) into threadgroup memory, then run
`matmul2d<...(NR1, 64, 32, false, true, false, multiply_accumulate),
execution_simdgroups<4>>` with the token block as the left operand and the weight
block as the transposed right operand.  Gate and up share one element layout, so
the mid epilogue applies `silu(gate) * up` on the cooperative tensors and stores
one float C tile; the scatter to the `mid`/`part` rows is the simdgroup kernels'.
Threadgroup memory: mid 10 KB (32 tokens) / 16 KB (64), down 8 KB / 16 KB.

Tried and dropped: double-buffered staging (stage step k+1 while the tensor op
consumes step k) and the descriptor's relaxed-precision accumulation.  Neither
moved the microbench (dense 9.2-9.5 ms vs 9.0-9.1 ms single-buffered; sparse 11.4
vs 10.6 ms), so the tiles are staging-bound on the tensor path, not MAC-bound.
The register-resident SiLU epilogue (one C tile instead of two, 16 KB -> 10 KB)
gained about 1.5%.

## Microbench (`QWEN4_BENCH=1 QWEN4_BENCH_ONLY=...`, mid + down, 3 runs each)

| case | simdgroup tiles | tensor tiles, 32 tokens | tensor tiles, 64 tokens |
| --- | ---: | ---: | ---: |
| `moe mm q4k/mxfp4 32` (32 experts, 2048 tokens x 10 slots) | 16.07-16.12 ms | 8.87-8.97 ms | 7.50-7.62 ms |
| `lo 256` (256 experts, sparse lists) | 17.99-18.13 ms | 10.63-10.79 ms | 11.37-11.43 ms |

## Prefill (chunk-interleaved harness, 8192-token chunks, `--tolerate-drift`)

| context | control (simdgroup) | tensor tiles | delta |
| --- | ---: | ---: | ---: |
| 8K -> 40K, 32-token tiles | 1148.0 tok/s | 1359.2 tok/s | +18.4% |
| 8K -> 40K, 64-token tiles vs 32-token tiles | 1376.0 tok/s (32) | 1422.2 tok/s (64) | +3.4% |
| 96K -> 128K, 64-token tiles (first repeat of two runs) | 1010.2 / 1023.2 tok/s | 1209.5 / 1234.6 tok/s | +19.7% / +20.7% |

The 64-token and 32-token tensor tiles produce bit-identical logits (each output
element accumulates over the same K order); the 64-token tiles are the opt-in
level 2.  At 128K each repeat first prefills a 96K prefix, and in both runs
(one after 15 minutes of continuous GPU load, one after a three-minute
cool-down) the second repeat throttled: both variants fell together and the
interleaved pairs stayed at +16 to +28%.  The table shows the first, unthrottled
repeat of each run; the two-repeat aggregates are +17.2% and +23.3%.

## Drift (full-vocabulary logits at the last position, candidate vs control, fresh sessions)

| prefix | max abs | mean abs | top-1 | reference margin |
| --- | ---: | ---: | --- | ---: |
| 2048 | 0.873 | 0.144 | agree (303) | 0.195 |
| 32768 | 0.653 | 0.094 | agree (85952) | 3.09 |
| 40960 (mixed histories, interleaved run) | 0.775 | 0.110 | flip on a 0.233 near-tie | 0.233 |
| 131072 (mixed histories, interleaved run) | 1.375 | 0.150 | agree (296) | 0.992 |

The perturbation does not grow with context: it is the same size at 2K, 32K and 128K.
Per tile, `test_moe_mm_tiles_exact` bounds the difference against the simdgroup
tiles at 7.7e-7 (mid, scale 1.3) and 1.5e-5 (down, scale 0.12) on a
production-shaped fixture (bound 2e-3 x scale in the test).

## Quality gate (BF16-reference continuation fixture, 99 aligned cases, 2376 target tokens)

`speed-bench/qwen38_q2down_compare.py`, production prefill of each prompt (the tile
path) and teacher-forced targets, same tokenizer for both runs:

| metric | simdgroup tiles | tensor tiles | delta |
| --- | ---: | ---: | ---: |
| target NLL | 0.20505 | 0.20503 | -0.00002 |
| top-1 agreement with BF16 | 96.34% | 96.25% | -0.08 pt (2 tokens) |
| first-token matches | 86 / 99 | 86 / 99 | 0 (one case each way) |
| logprob MAE | 0.04679 | 0.04659 | -0.00020 |

NLL moved in either direction on 51 vs 48 cases.  For scale, the Q4_K -> Q8 pack
spread on the same fixture is 0.015 NLL and 0.022 MAE (`qwen38-iq2-quality.md`);
the tensor-tile drift is two orders of magnitude below it.

## Verification

- `tests/test_qwen4_kernels`: all passes, including the bounded tensor-tile pass
  for both tile widths (skips when the tensor API is unavailable).
- Default path untouched: the simdgroup kernels and their dispatch are unchanged;
  the opt-in branch is taken only with `DS4_QWEN4_MOE_MM_NAX` set and the tensor
  API available.  Checked across binaries: the 99 fixture cases scored by the
  pre-round build and by this branch with the variable unset give byte-identical
  full-vocabulary logits (99/99 files) and the same metrics.
