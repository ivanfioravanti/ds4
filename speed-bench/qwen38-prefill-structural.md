# Qwen Q2 prefill: block convolution and wider down tiles

The retained candidate improves end-to-end prefill over the **enabled Q8
unpack baseline**. The requested 5% target was not reached. Full results and
the final integration checks are recorded in the accompanying JSON artifact.

## Changes and scope

GDN convolution previously ran the entire token sequence serially within
each channel. The new path snapshots the incoming history for each 64-token
block, then runs those blocks independently. Each channel retains exactly
the old tap order, FP32 accumulation, SiLU, and raw-input history. Only the
last block writes the final history; all blocks read their incoming windows
from the snapshot, so no block reads another block's overwritten input.

The Q2_K routed down projection uses 64-token tiles at batches of at least
8,192 tokens. This reuses each decoded weight tile for twice as many token
columns. The K traversal, FP16 operands, FP32 accumulation, logical 640-wide
input, and physical 768-wide weight rows stay unchanged. Separate 8-, 16-,
and 32-token tails avoid wasted work for small expert remainders; larger
remainders stay on the 64-token kernel. Gate/up tiles are unchanged.

Enabled by default for Qwen on M3 Ultra from 1,024-token batches; wider down
tiles additionally require Q2_K and at least 8,192 tokens. Smaller batches
retain their existing defaults. `DS4_QWEN4_PREFILL_REUSE=0` rolls both changes
back while retaining the existing Q8 optimization. `=1` is a diagnostic
opt-in for other Metal devices and smaller convolution batches. Batches of
at most eight tokens always retain the original convolution path. Decode
and MTP call sites, CPU/CUDA backends, and model files were not changed.

The block-history buffer uses `ceil(T/64) * (K-1) * channels * 4` bytes:
**15 MiB at 8K**, or **1.875 MiB at 1K**, for this model. It is reusable,
reported in backend scratch accounting, capped at 1 GiB, and freed at backend
cleanup. Older buffers can remain alive temporarily while queued work
finishes during growth. This is additional steady scratch capacity, not a
measurement of total peak memory. The wider down kernel adds 4 KiB of
threadgroup staging per active group and no persistent buffer. The retained
Q8 scratch remains separate.

## Measurement method

Apple M3 Ultra, 512 GiB unified memory, resident model weights, no MTP.
Main model: `/Users/ifioravanti/models/qwen38-q2down-experiment/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`.
PLE: `/Users/ifioravanti/models/qwen38-q4k-imatrix/Qwen3.8-Flash-Next-PLE-Q4_1.gguf`.
Base revision `b4c3550` plus the current uncommitted changes, copied into the
separate `ds4-prefill-structural` checkout before experimentation. A HEAD-only
checkout was not used as the benchmark baseline.

Each comparison loads one engine, warms both variants on a full-length
prefix, and creates a fresh session for each measured run. Final validation
uses four alternating ABBA/BAAB repeats, 16 measured runs per scenario.
Throughput is total measured tokens divided by total elapsed time, including
input staging and all GPU preprocessing. Every measured final vocabulary
vector is compared bit for bit. For append, an 8K live prefix is established
before timing the next 8K; both variants pay their own prefix setup outside
the timed interval.

The harness originally unset the candidate variable for control. It now
also accepts `--control-value`, so enabled defaults cannot silently make both
variants identical. Validation explicitly uses `--control-value 0` and
`--candidate-value 1`, with `DS4_QWEN4_Q8_PREFILL_UNPACK=1` for both. All
samples are retained. Processes and GPU utilization were inspected; existing
idle servers were left running. Model benchmarks and focused GPU tests ran
serially. CPU-only builds overlapped portions of validation; no timing
samples were excluded for this or any other reason.

## Full-model validation results

Each rate below aggregates eight runs per variant (16 measured runs):

- 1K prose: 1,173.5053 → 1,194.9359 tok/s (**+1.8262%**).
- 8K prose: 1,300.4796 → 1,338.8000 tok/s (**+2.9466%**).
- Independent 8K code prompt: 1,303.8942 → 1,340.0871 tok/s (**+2.7758%**).
- Append 8K after an 8K prose prefix: 1,284.8197 → 1,321.0206 tok/s (**+2.8176%**).

All 64 final vocabulary rows / **15,892,480 floats** matched bit for bit.
The repeatable improvement justifies retaining the smaller result; it is
not evidence that the 5% target was met. The accompanying
[results artifact](qwen38-prefill-structural-results.json) contains every
screening and validation sample, including all negative experiments.

## Experiments and decision

The current synchronized MoE profile attributes about 1.32 s per 8K chunk
to gate/up and 0.81 s to routed down, out of about 2.62 s for all MoE work.
Unprofiled baseline prefill takes about 6.30 s. These are synchronized
wall-time attributions, not GPU hardware counters; the profiler adds
overhead. Saving roughly 0.30 s would be needed to reach 5% throughput.

The initial ranking favored removing repeated IQ2 weight decoding, then
packing routed inputs once, then reducing GDN preparation serialization.
The first two had enough affected work to plausibly reach the target, but
the added traffic outweighed the saved work in the tested implementations:

- IQ2 gate/up unpacked once into FP16: **-6.95%** at 8K. Rejected.
- Routed inputs packed once into FP16 tiles: **-19.70%** at 8K. Rejected.
- 64-token gate/up and down tiles: **+0.74%**; gate/up alone **-0.09%**,
  down alone **+0.74%**. Retained only wider Q2 down tiles at large batches.
- Fully parallel convolution with a full raw-input snapshot: **+1.53%**.
  Superseded by the smaller block-history snapshot: **+2.10%**.

The failed implementations and their switches were removed. Screening
patches are retained under `OUT/structural/` in the experiment checkout;
patch names ending `with-baseline.patch` include the original uncommitted
baseline changes and are archival artifacts, not integration patches.

## Correctness and reproduction

`make test-qwen4-prefill-reuse` checks convolution outputs and final history,
nonzero incoming state, two appended calls queued back to back, scratch
growth/reuse, SiLU on/off, kernel widths 2–4, partial channels, block tails,
short-batch fallback, and explicit on/off/unset defaults. The MoE test retains
its original format/tile tests and adds 64-token generic/specialized kernels,
8/16/32-token tails, poisoned output guards, and 8K eligibility boundaries.

Full-model checks cover the final vocabulary row after each measured prefix,
not all intermediate activations, arbitrary continuations, or general model
accuracy. Arithmetic precision and reduction order were not relaxed.

Example (replace model variables with the paths above):

```sh
rtk make speed-bench/metal_prefill_variant_bench
rtk proxy env DS4_QWEN4_Q8_PREFILL_UNPACK=1 \
  ./speed-bench/metal_prefill_variant_bench -m "$MODEL" --ple "$PLE" \
  --prompt-file speed-bench/promessi_sposi.txt --prefill-chunk 8192 \
  --prefix-tokens 8192 --warmup-tokens 8192 --repeats 4 \
  --candidate-env DS4_QWEN4_PREFILL_REUSE --control-value 0 --candidate-value 1
```

Use 1,024 for the smaller-prefix case, `--prompt-file ds4.c` for the second
prompt, or `--prefix-tokens 16384 --initial-tokens 8192` for append.
The experiment's `OUT/structural/run.py` saves exact commands, controlled
environment, source/executable hashes, process snapshots, and raw logs.
Nothing was committed, pushed, uploaded, or written to the model files.

## Integration verification

The main checkout's target files were checked against the carried baseline
before applying only `OUT/structural/integration.patch`. Existing Q8 changes
and unrelated files were preserved. CLI, benchmark, and server binaries were
rebuilt; running servers were not restarted.

The final experiment build passed `make test-qwen4-prefill-reuse`,
`make test-q8-prefill-variants`, and `make test-qwen4-q2`. This includes
114 convolution cases, 132 MoE format/shape cases with 14 configurations each,
14 Q8 shapes, and the existing exact Qwen decode tests. The focused prefill
tests passed again in the main checkout.

A final main-build 8K ABBA/BAAB measured 1,301.9296 → 1,338.8552 tok/s
(**+2.8362%**) with eight exact vocabulary rows. A 1K ABBA checked the unset
default against rollback: enabled default 1,194.5299 tok/s, rollback
1,172.7283 tok/s, with four exact rows. That log intentionally labels the
enabled default as control and rollback as candidate; its negative candidate
delta is not a regression. These 12 additional rows / 2,979,840 floats
matched exactly. Final source and executable hashes are in the JSON artifact.
