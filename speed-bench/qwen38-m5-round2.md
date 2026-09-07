# Qwen3.8 on Apple M5 Max, round 2: kernel geometry, tile order, MTP rollback

Hardware, model and rules as in [round 1](qwen38-m5-round1.md). Baseline for
this round: `1bc6e6d` (round-1 defaults). Every accepted change keeps
full-vocabulary FP32 logits byte-identical; every A/B below compared every
logit row (decode) or the final logits of every run or repeat (prefill).

## Decode (single engine, per-step interleaved, 2048-token prefix unless noted)

| change | measurement | result |
| --- | --- | ---: |
| F16/F32 matvecs one row per SIMD group (hc low-rank down, router) | rollback to two rows, two runs | -0.55% / -0.54% |
| GDN front threadgroup 1024 threads (was 256) | 512 / 1024 | +0.14% / +0.69% |
| MoE reduce threadgroup 64 / 128 threads | env | +0.05% / -0.02% (dropped) |
| HC combine threadgroup 64 / 128 threads | env | -0.05% / -0.07% (dropped) |
| shared expert slot dispatched first in the MoE grids | two runs | +0.10% / +0.08% (dropped) |
| attention layer q + k/v/indexer projections in one dispatch | two runs | +0.06% / +0.08% (dropped) |
| four-row Q8_0 matvec tile | env | -0.55% (dropped) |
| F16 HC gate/mix with eight terms loaded ahead per lane | env | rejected: logits differ (fast-math contraction changed with the staged loop) |
| vector indexer scorer with staged queries, 64K prefix | rollback | -0.80% |
| one-thread-per-dim split-attention merge, 64K prefix | rollback | +0.04% (kept: exact, no cost) |
| scorer + merge together, 64K / 2K prefix | rollback | -1.85% / -1.03% |
| merge loading sixteen splits ahead of its chain, 64K prefix | rollback | -0.80% |
| tile-max prefiltered top-k, 64K prefix (16K blocks) | rollback, two runs | -0.01% / -0.82% |
| no-op candidate, 2K / 64K prefix | env | -0.005% / +0.16% |

## MTP

Zero-copy rejection rollback (swap the live and snapshot state buffers instead
of 73 blits): CLI compare, three prompts, three interleaved repeats, identical
outputs and acceptance counters. Hamlet 65.81 -> 66.39 t/s, explanation
70.55 -> 71.07 t/s (rejection-heavy); Fibonacci within noise.

## Prefill

Expert-major tile order (`DS4_QWEN4_MOE_MM_ORDER`): +1.8% and +2.1% at
8192-token chunks (16 fresh-session runs each), +0.8% at 4096-token chunks,
+2.6% over the drift-free second pass of a chunk-interleaved 64K prefill.
The interleaved mode exposed the first-session penalty (every chunk after the
first of the first session at a new context length runs ~25% slower) and
the harness now walks an untimed pass first.

The prefiltered top-k is proven exact by a fixture (rows up to 65536 blocks,
ties at both ends, full and prefiltered selections compared byte for byte)
and targets 128K-262K contexts, where the full-row radix select reads its
256 KB row six times per layer.

## Where the long-context decode time goes (encoder timeline at a 128K prefix, relative)

911 dispatches per token. Indexer scoring 3.1%, radix top-k 2.4%, gathered
attention 1.8%, split merge 1.7%, attention prep 0.8% per token, against a
2K prompt where the whole attention family is under 2%. The prefiltered
top-k (next section of work) targets the radix select.

## Adopted M5 defaults this round

- Single-token F16/F32 matvecs with at most 1024 rows launch one row per
  SIMD group (`DS4_METAL_PLAIN_MV_NR0`).
- GDN front threadgroups of 1024 threads (`DS4_QWEN4_GDN_FRONT_THREADS`).
- Expert-major prefill tile order (`DS4_QWEN4_MOE_MM_ORDER`).
- Vector indexer scorer and one-thread-per-dim split merge
  (`DS4_QWEN4_IDX_SCORE_VEC`, `DS4_QWEN4_ATTN_MERGE_WIDE`).
- Prefiltered decode top-k (`DS4_QWEN4_IDX_PREFILTER`).
- MTP rejection rollback by buffer swap on every device
  (`DS4_QWEN4_MTP_SWAP_RESTORE=0` restores the copy).

## To measure on M3 Ultra

Everything above except the MTP rollback is gated on M5. The mechanisms
that do not depend on the M5's core count are worth a 35-second A/B each
with the decode harness (both variants share one engine, every logit row
compared):

```sh
make qwen38-decode-variant-bench
B="./speed-bench/qwen38_decode_variant_bench -m MODEL --ple PLE --prompt-file speed-bench/promessi_sposi.txt"
$B --candidate-env DS4_METAL_PLAIN_MV_NR0=1        # one-row F16/F32 matvecs
$B --candidate-env DS4_QWEN4_GDN_FRONT_THREADS=1024
$B --candidate-env DS4_QWEN4_IDX_SCORE_VEC=1 --candidate-env DS4_QWEN4_ATTN_MERGE_WIDE=1 --prefix-tokens 65536
$B --candidate-env DS4_QWEN4_IDX_PREFILTER=1 --candidate-env DS4_QWEN4_IDX_SCORE_VEC=1 --prefix-tokens 131072
./speed-bench/metal_prefill_variant_bench -m MODEL --ple PLE --prompt-file speed-bench/promessi_sposi.txt \
  --interleave --prefix-tokens 65536 --prefill-chunk 8192 --warmup-tokens 8192 --repeats 2 \
  --candidate-env DS4_QWEN4_MOE_MM_ORDER --candidate-value 1
```

The single-token Q4_K NSG 4 and MXFP4 NSG 16 geometries are M5-specific;
M3 Ultra keeps its measured NR1/NSG8 and low-bit defaults.

## Lab gate

Against the round-1 baseline `82d0314` (round-1 and round-2 defaults
together), teacher-forced parity exact on 6 histories x 32 rows; CLI greedy
decode, three prompts, three interleaved repeats each, medians, all outputs
and acceptance counters identical:

| prompt | plain baseline | plain candidate | change | MTP baseline | MTP candidate | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hamlet | 52.15 | 55.12 | +5.70% | 64.05 | 66.67 | +4.09% |
| Fibonacci | 52.23 | 55.24 | +5.76% | 77.65 | 80.11 | +3.17% |
| Explanation | 52.37 | 55.34 | +5.67% | 68.73 | 71.36 | +3.83% |

Per-frontier interleaved sweep (four fresh processes per frontier, 128 greedy
tokens), all round-1 and round-2 defaults against the baseline, frontier
logits exact across every run:

| ctx | prefill baseline | prefill candidate | change | decode baseline | decode candidate | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1027.9 | 1122.1 | +9.17% | 46.41 | 49.03 | +5.67% |
| 32,768 | 887.2 | 983.3 | +10.83% | 44.29 | 47.37 | +6.95% |
| 131,072 | 834.7 | 908.8 | +8.88% | 41.84 | 45.53 | +8.82% |

The 128K decode gain over round 1 (+5.0% then) is the long-context work:
vector scorer, prefetched split merge and prefiltered top-k.
