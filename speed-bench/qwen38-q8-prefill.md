# Qwen Q2 prefill: dequantize Q8 projections once

The retained local experiment improves warmed full-model prefill by about
1.4–1.8% at 8K and 0.4% at 1K on M3 Ultra, with bit-exact compared logits.
It materializes each eligible Q8 projection into a reusable FP16 scratch
buffer before the existing F16 matrix multiplication. Dequantization time
is included in every measurement; this is not a preconverted model.

## Implementation

The original Q8 matrix kernel repeats weight unpacking for each 32-token
tile. The candidate unpacks once per projection invocation and reuses the
half-rounded weights across those tiles. Both paths compute
`half(float(q) * float(scale))`, stage the same FP16 operands, and retain
the same K accumulation order. The original Q8 model bytes are unchanged.

Enabled by default for Qwen on M3 Ultra. Set
`DS4_QWEN4_Q8_PREFILL_UNPACK=0` to restore the original path, or `=1` to
force the path on other Metal configurations for experimentation. It applies
only to generic Q8 matrix batches of at least 32 tokens. Larger
projections whose decoded weights exceed 128 MiB use the original path.
The scratch allocation is reused, included in the GPU memory report, and
released at backend cleanup. For this model its steady capacity is 60 MiB;
older allocations can remain alive briefly while previously encoded work
completes during growth. This is not a total peak-memory measurement.

One- and two-token decode paths are unchanged. No MTP implementation changes
were made, and no model files were created or uploaded. The default is selected through a Qwen-specific entry point; other models
retain their existing default. Measurements below concern
Qwen on M3 Ultra only, not other models, GPUs, streaming, or physical 64 GB
machines. Nothing was committed or pushed.

## Measurements

Apple M3 Ultra, 512 GiB RAM. Base revision `b4c3550`, plus the previously
added local GDN profiler (disabled for throughput). Released IQ2_XXS gate/up,
padded Q2_K down model, SHA-256
`341c8d79468384a05e22998ae834a489145f6c385e276dcf79170a6b0d0ffd2d`,
and unchanged external Q4_1 PLE sidecar. Resident weights, no MTP.

Fresh sessions in one loaded engine; full-length warmups for both variants,
then ABBA and BAAB. Rates are total measured tokens divided by total time.
Every measured final vocabulary row is compared bit for bit. Chunk size
equals 1,024 or 8,192 respectively.

- 1K prose prefix: 1,168.35 → 1,173.35 tok/s (**+0.43%**).
- 8K prose prefix: 1,278.73 → 1,301.29 tok/s (**+1.76%**).
- Separate 8K code prefix: 1,282.44 → 1,304.68 tok/s (**+1.73%**).
- Append 8K after an untimed 8K prefix: 1,231.12 → 1,248.56 tok/s
  (**+1.42%**). Only the appended tokens are timed.

Prose uses `speed-bench/promessi_sposi.txt`; code uses `ds4.c`. These are
bounded local measurements, not a broad model/hardware benchmark.

A second candidate used the existing packed Q8 load helper inside the
original tiled kernel. It measured +0.48% at 1K, but -0.61% aggregate at 8K
with one unusually slow candidate run. All samples are retained. That
candidate and its dispatch mode were removed from the final implementation.

## Validation and reproduction

`make test-q8-prefill-variants` checks 14 shapes: small-batch fallbacks,
32-token dispatch boundaries, partial token/output tiles, large projection
shapes, and a decoded matrix exceeding the scratch cap. It compares two
different weight matrices queued back to back, toggles the candidate in one
backend, and checks exact finite outputs plus untouched trailing guards.

Screening used `DS4_QWEN4_Q8_PREFILL_VARIANT=2` for scratch dequantization.
The final implementation removes that experimental selector and uses the
boolean `DS4_QWEN4_Q8_PREFILL_UNPACK=1`. The final tests were rerun after
that cleanup. Source and executable hashes are in the results artifact.

Across screening and validation, 48 full-model vocabulary vectors matched
exactly (11,919,360 floats), including the rejected candidate. After final
cleanup, another 1K ABBA check matched four vectors / 993,280 floats exactly.

A separate ABBA run compared the final executable against the saved
pre-change executable with its original dense Metal source. Both processed
8K then another 8K; only prefill rates are considered. Arithmetic means:

- First 8K: 1,276.945 → 1,298.925 tok/s (**+1.72%**).
- Next 8K at the 16K frontier: 1,262.695 → 1,285.650 tok/s (**+1.82%**).

This confirms the speedup against the pre-change build rather than only an
in-process disabled branch. These independent-process runs supplement the
full-logit parity checks; their one-token generation timings are not a decode
or MTP benchmark.

To reproduce:

```sh
make ds4-bench
DS4_QWEN4_Q8_PREFILL_UNPACK=1 ./ds4-bench --metal \
  -m "$HOME/models/qwen38-q2down-experiment/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf" \
  --ple "$HOME/models/qwen38-q4k-imatrix/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 16384 --step-mul 2 \
  --prefill-chunk 8192 --gen-tokens 1
```

Use 1,024-token chunks for the existing smaller-workspace configuration.
The flag does not itself change chunk size or context allocation.

[Machine-readable results](qwen38-q8-prefill-results.json) retain all runs.
Raw commands, logs, the pre-change executable and dense source, rejected
packed-load patch, and original/final focused tests are in
`OUT/q8-prefill/`. The retained Q8 default was copied into the main checkout without the diagnostic GDN profiler.

## Enabled default and subsequent experiment

The default now uses a Qwen-specific entry point on M3 Ultra. Other model
entry points and non-Apple backends retain their existing defaults. Explicit
`DS4_QWEN4_Q8_PREFILL_UNPACK=0` provides rollback. The final focused test
checks explicit off/on and an unset environment variable across all 14 shapes.

A new 8K comparison with four balanced repeats (16 measured runs) produced
1,286.5467 tok/s enabled versus 1,268.6863 disabled: **+1.41%**.
All 16 final vocabulary vectors matched exactly. The first, shorter check
had a slow opening run and reversed the aggregate result; both complete
runs are retained rather than discarding the noisy measurement.

The next experiment omitted three no-memory SIMD barriers per K step only
in the F16 kernel used for unpacked Q8 weights. Threadgroup memory barriers
and arithmetic order were unchanged. It passed all 14 focused shapes and
full-model logit checks, but measured **-0.34% at 1K** and **-0.26% at 8K**
in the longer balanced repeat. The first noisy 8K screening was -5.49%.
There is no demonstrated benefit, so the specialization and its switch were
removed. The rejected patch remains at `OUT/q8-prefill/rejected-no-barriers.patch`
in the experiment checkout.

[Enabled-default and barrier results](qwen38-q8-prefill-enabled-results.json)
include all commands and raw logs: 56 measured vocabulary rows, 13,905,920
floats, all bit-identical. Nothing was committed or pushed.

After integration into the main checkout, the 14-shape focused test passed
again and a final 1K ABBA compared four more vocabulary rows exactly
(993,280 floats). That check measured 1,172.8472 tok/s enabled versus
1,168.2250 disabled (+0.40%). CLI, benchmark, and server binaries were rebuilt;
existing server processes were not restarted.
