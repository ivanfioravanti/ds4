# Qwen3.8 on Apple M5 Max, round 1: re-qualifying the M3 Ultra defaults

Hardware: Apple M5 Max, 128 GB unified memory, 40-core GPU, Metal 4.
Model: `Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf`
(the `qwen38-q4k` download) with the Q4_1 PLE sidecar. Baseline `82d0314`.
Every accepted change keeps full-vocabulary FP32 logits byte-identical.

## Why this round exists

Every Qwen kernel geometry tuned so far was gated on the M3 Ultra device name,
so an M5 ran the untuned defaults everywhere. This round measures each gated
default on the M5 with single-engine A/B harnesses that abort on any logit
difference, then turns the winners into M5 defaults.

## Measurement discipline on this machine

Two effects dominate small comparisons on the M5 Max and shaped the protocol:

- **Sustained-load drift.** Four back-to-back full `ds4-bench` sweeps of the
  same build measured 1030, 910, 870 and 850 prefill tokens/s at the same
  frontier, and 49.7 down to 45.0 decode tokens/s. Process-level ABBA over
  whole sweeps therefore mis-ranks a few-percent change. The lab sweep now runs
  four fresh processes per frontier in ABBA or BAAB order, and the decode
  harness interleaves variants per step inside one engine.
- **First timed run.** In the prefill harness the first timed 8192-token run
  of a process is 15-35% slower than the following ones even after a
  2048-token warmup. With that warmup a no-op candidate "gained" +1.6% and the
  MoE specialization "gained" +16.4%; with a full 8192-token warmup and four
  ABBA/BAAB repeats (16 timed runs) the no-op measures +1.3% and specialization
  +1.0%. All prefill numbers below use the full-warmup protocol unless marked
  otherwise, and deltas under about 2% are not treated as evidence.

## Baseline

First cold `ds4-bench` sweep after loading (8192-token chunks, 128 greedy
tokens): 612 to 690 prefill tokens/s and 45.4 to 48.0 decode tokens/s from 4K
to 128K. Warm and rested, the same sweep starts at 1004 to 1046 prefill and
49.4 to 49.8 decode tokens/s (936 / 46.3 at 128K). Short-prompt CLI plain
decode (Hamlet, `--nothink --temp 0`): 42.7 tokens/s cold, 52.3 warm.

## Tooling added

- `speed-bench/qwen38_decode_variant_bench` (`make qwen38-decode-variant-bench`):
  one engine, two sessions synced to the same prefix, every step decodes the
  same token under both variants with alternating order and pairing, and
  aborts unless every full-vocabulary logit row is bit-identical. Compares
  dispatch-time environment knobs in ~35 s per run; a no-op candidate measured
  -0.15% and -0.29%.
- `speed-bench/metal_prefill_variant_bench` gained `--extra-env` and
  `--control-env` so combinations can be measured against a tuned control.
- `speed-bench/qwen38_m5_lab.py`: teacher-forced parity, per-frontier
  interleaved ds4-bench sweeps with logit comparison, CLI plain/MTP decode
  comparison, long-prompt MTP, and a `gate` that runs them all.
- `speed-bench/qwen38_timeline_report.py`: per-kernel aggregation of
  `DS4_METAL_ENCODER_TIMELINE` logs.
- `DS4_QWEN4_GDN_PREFILL_NSG` exposes the large-prefill GDN scan grouping
  that was hard-wired to M3 Ultra.

## Where a decode token goes (encoder timeline, relative)

897 dispatches per token. GPU time by kernel: dense Q8_0 projections 41%
(138 dispatches, ~64% of peak bandwidth), Q4_K routed gate/up 14%, F16
hyper-connection gate/mix 9%, MXFP4 routed down 8%, F16 low-rank down 7%,
HC norm 4%, F32 router 3%, GDN front 2.4%. A standalone probe measured the
M5 Max per-dispatch cost at ~1.2 us inside one encoder, so dispatch count is
about 5% of the token; the rest is kernel latency and bandwidth.

## Plain decode A/B (single engine, 2048-token prefix, 384 measured steps, 401 exact logit rows each)

| candidate | change |
| --- | ---: |
| `DS4_QWEN4_Q4K_MID_NR=1 DS4_QWEN4_Q4K_MID_NSG=4` | +1.94% |
| `... NSG=8` (M3 Ultra default) | +1.62% |
| `... NSG=6` | +2.08% |
| `... NSG=2` | +1.36% |
| `NR=2 NSG=4` / `NR=2 NSG=8` | -0.40% / -0.21% |
| `DS4_QWEN4_MOE_MV_SPECIALIZE=1` (MXFP4 down, NSG 8) | +1.30% |
| `... DS4_QWEN4_MOE_MV_NSG=16` | +1.92% |
| `... NSG=16 NR=2` / `NSG=12` / `NSG=4` | +1.43% / +1.04% / +1.12% |
| Q4K NR1/NSG4 + MXFP4 specialized NSG16 | **+3.80%** |
| Q4K NR1/NSG8 + MXFP4 specialized NSG16 | +3.65% |
| `DS4_QWEN4_DECODE_FUSIONS=1` (both M3 Ultra decode fusions) | -0.81% |
| Q8 QKV+gate grid concatenation alone / combine+norm fusion alone | -0.31% / -0.29% |
| `DS4_QWEN4_GDN_NSG=2/4/8` | -0.13% / +0.10% / -0.48% |
| `DS4_QWEN4_FLUSH_LAYER=0/1/4/8` (default 2) | -2.38% / -0.58% / -0.42% / -0.53% |
| `DS4_QWEN4_NO_GDN_R4=1` / `DS4_QWEN4_NO_Q4K_MID=1` | -1.19% / -3.94% |
| four-row Q8_0 matvec tile (exact per row; temporary kernel) | -0.55%, removed |
| `DS4_METAL_Q8_MV_NSG=1/2/8` | rejected: logits differ (split-K reduction) |
| Q8_0 decode through the Metal 4 tensor kernel | rejected: logits differ |

## Prefill A/B (single engine, fresh 8192-token prefix, full warmup, 16 timed runs, 1,986,560 final logits compared per run)

Control is the tree with MoE specialization on; candidates add or remove knobs.

| candidate | change |
| --- | ---: |
| no-op | +1.3% (floor) |
| `DS4_QWEN4_MOE_MM_SPECIALIZE=0` (rollback of specialization) | -1.0% |
| `DS4_QWEN4_MOE_TAILS=1` | +2.3% |
| tails + `DS4_QWEN4_MOE_MID_TILES=32` | +4.4% |
| tails + mid tiles 32 + `DS4_QWEN4_HC_NORM_REUSE=1` | **+7.5%** |
| the above + down tiles 32 + GDN prefill NSG 4 | +7.4% |

Earlier single-knob screens with the short warmup (excluding the first timed
run of each): mid tiles 16 / 32 +0.3% / +0.2%, down tiles 16 / 32 -0.5% /
-0.1%, HC norm reuse +1.0%, tails +1.0%, GDN prefill NSG 2 / 4 / 8 -0.5% /
+0.4% / -0.8%, token tiles of 16 (`MOE_MID_NT=2`) -4.4%, down NT 2 -3.3%.
Specialization at 32K (24576-token untimed prefix, 4 runs) measured within
0.5% of generic.

## Adopted M5 defaults

- Single-token Q4_K gate/up decode rows: one row per SIMD group, four groups
  per threadgroup (`DS4_QWEN4_Q4K_MID_NR` / `_NSG` override).
- MXFP4 routed-down decode rows specialized with sixteen SIMD groups per
  threadgroup (`DS4_QWEN4_MOE_MV_SPECIALIZE` / `_NSG` override).
- Routed MoE prefill GEMM specialization (`DS4_QWEN4_MOE_MM_SPECIALIZE=0`
  restores generic kernels), remainder tiles for Q4_K gate/up and MXFP4 down
  (`DS4_QWEN4_MOE_TAILS=0`), gate/up tile caps of 16 from 4096 tokens and 32
  from 8192 (`DS4_QWEN4_MOE_MID_TILES`), and hyper-connection RMS reuse from
  8192 tokens (`DS4_QWEN4_HC_NORM_REUSE=0`).

The M3 Ultra ordinary-decode fusions stay off on M5.

## Lab gate

Decode defaults (`96dd6bf`) against the baseline:

- Teacher-forced parity: exact on 6 histories x 32 full-vocabulary rows.
- ds4-bench frontier logits at 4K, 8K, 16K, 32K, 64K and 128K: exact across
  every run of both builds.
- CLI greedy decode, three prompts, three interleaved repeats each, medians:

| prompt | plain baseline | plain candidate | change | MTP baseline | MTP candidate | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hamlet | 52.27 | 54.67 | +4.59% | 63.93 | 65.65 | +2.69% |
| Fibonacci | 52.23 | 54.50 | +4.35% | 77.42 | 79.55 | +2.75% |
| Explanation | 52.30 | 54.80 | +4.78% | 68.62 | 70.42 | +2.62% |

All outputs and MTP acceptance counters matched.

- Per-frontier interleaved sweep (four fresh processes per frontier, 128
  greedy tokens), decode defaults plus MoE specialization:

| ctx | prefill baseline | prefill candidate | change | decode baseline | decode candidate | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 906.6 | 938.4 | +3.51% | 46.18 | 47.49 | +2.84% |
| 131,072 | 864.1 | 868.1 | +0.47% | 42.97 | 44.38 | +3.31% |

Frontier logits exact across all four runs at both frontiers.

Prefill defaults (this tree) against their own rollback, single engine, full
warmup, 16 timed runs each, all final logits exact:

| candidate (rollback) | change |
| --- | ---: |
| 8192-token chunks: tails off, gate/up tile cap 8, HC RMS reuse off | -5.1% |
| 4096-token chunks: tails off | -4.1% |
| 4096-token chunks: gate/up tile cap 8 instead of 16 | -0.7% |

Per-frontier interleaved sweep, all M5 defaults against the baseline, fans at
maximum and no other model loaded:

| ctx | prefill baseline | prefill candidate | change | decode baseline | decode candidate | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1036.4 | 1168.0 | +12.70% | 49.70 | 52.16 | +4.93% |
| 32,768 | 946.2 | 1046.7 | +10.63% | 47.71 | 50.09 | +4.99% |
| 131,072 | 839.2 | 891.5 | +6.23% | 42.85 | 44.99 | +4.99% |

Frontier logits exact across all four runs at every frontier.

## Validation

`make test-qwen4-kernels`, `DS4_TEST_QWEN4_MV_EXACT=1` (extended to Q4_K
gate/up beside MXFP4 down), `DS4_TEST_QWEN4_Q4K_ORDERED_ONLY=1`,
`DS4_TEST_QWEN4_HC_NORM_REUSE_ONLY=1` and `make test-qwen4-moe-mm-specialize`
pass on the M5. Raw records are in `qwen38-m5-round1-results.json`.
