# Qwen3.8 Q2 ordinary decode: exact operation fusions

The requested 5% decode gain was **not achieved**. After 104 exploratory
runs covering 36 candidate configurations, two small, exact fusions remain.
All other prototypes were removed.

Baseline: `01a9489`, on Apple M3 Ultra with 512 GiB unified memory.
The model is `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`
with the external Q4_1 PLE sidecar. Context capacity is 8192; generation
uses greedy sampling and `--nothink`.

## Ordinary decode results

Three measured repetitions per prompt, interleaving baseline and candidate
and reversing their order on alternate repetitions. Each binary receives
a separate Hamlet warmup before measured runs. Generation rates exclude
loading and prefill. The table uses median generation rates. MTP is off.

| Prompt | Baseline tokens/s | Candidate tokens/s | Gain |
|---|---:|---:|---:|
| Hamlet | 54.28 | 54.94 | 1.22% |
| Fibonacci | 54.31 | 54.92 | 1.12% |
| Explanation | 54.35 | 54.92 | 1.05% |

All measured outputs matched. These short-prompt measurements establish
a small local gain, not a 5% improvement or a claim about other hardware
or long contexts. Individual samples, ranges, prefill rates, commands, and
binary/runtime-source hashes are in the linked records.

## Implementation

- Combine the single-token residual update with the following F16
  hyper-connection normalization and injection projection. The old residual
  and injection stay readable throughout the dispatch. The next residual
  reuses the existing, otherwise idle `hc_u` buffer; a separate injection
  buffer prevents cross-threadgroup read/write races. That buffer costs
  512 bytes per graph-capacity token, or 4 MiB at capacity 8192.
- Concatenate the disjoint threadgroup grids for the Q8 GDN QKV and gate
  projections. Each output row retains the standalone kernel's input walk
  and reduction tree. Unsupported shapes fall back to separate projections.

Both paths default on for M3 Ultra ordinary-decode sessions and apply only
to single-token passes. MTP sessions retain the separate operations.
`DS4_QWEN4_DECODE_FUSIONS=0` restores separate operations. The injection
scratch allocation remains. Other devices default to the previous paths;
an explicit value of 1 opts into the new paths.

## Correctness and MTP regression

`make -j8 all test-qwen4-kernels test-qwen4-q2` passed. New kernel tests
compare residual, normalized values, and injection partials byte-for-byte
at embedding widths 64 and 2560, including output guards and alias rejection.
The Q8 test covers unequal output sizes with short and real projection
dimensions, output guards, and odd-shape fallback.

All **192 full-vocabulary FP32 vectors** were finite and byte-identical to
the saved baseline: 32 teacher-forced steps after each of 1, 2, 8, 9, 39,
and 128 prompt tokens. This establishes implementation parity on those
histories; it does not measure quantization quality against original weights.

The separate MTP regression uses the same three prompts and repetitions:

| Prompt | Baseline tokens/s | Candidate tokens/s | Gain |
|---|---:|---:|---:|
| Hamlet | 64.15 | 64.22 | 0.11% |
| Fibonacci | 79.12 | 78.98 | -0.18% |
| Explanation | 67.99 | 68.04 | 0.07% |

Generated output hashes and accepted-draft/verification-cycle counts
matched in every measured MTP run. The two-token verifier retains its
existing kernels. An earlier candidate enabled the fusions in MTP sessions
and measured 0.04–0.42% slower. A disabled-fusion control on Hamlet was
within 0.05% of baseline. The final implementation therefore restricts
the new operations to ordinary-decode sessions. Those earlier comparison
records are preserved separately.

## Rejected experiments

The exploratory runs used the 256-token networking explanation prompt.
Their raw rates are screening evidence, not final release measurements.

| Experiment | Result |
|---|---|
| Q8 SIMD-group and rows-per-group sweeps | No repeatable improvement; larger row tiles regressed substantially. |
| Fixed-width Q8 kernels and packed Q8/IQ2 reads | Flat or slower. |
| Q8 one-SIMD reductions | About 50 tokens/s versus 54.4 baseline. |
| Q8 FP32 SIMD-group matrix prototype | About 31.8 tokens/s versus 54.4 baseline. |
| Combined expert down/reduce/residual | Effectively flat. |
| Split hyper-connection low-rank projection | Slower. |
| Overlap GDN gate projection with front/scan | Slower with the additional encoder/barrier overhead. |
| Fixed GDN front and combined front/gate dispatch | Insufficient benefit. |
| Expanded IQ2 signed weights and scales | About 50.4 tokens/s, with roughly 90 GB extra storage. |
| Smaller expanded IQ2 metadata | About 53.6 tokens/s, with roughly 30 GB extra storage. |
| Cached hyper-connection activation | Small timing gain, rejected for full-model numerical drift. |

The cached activation passed local FP64-reference error checks and preserved
greedy output on screening prompts, but failed full-logit comparison.
One tested history reached 2.63 maximum absolute logit difference and
0.532 RMS difference despite matching top-1 choices. Local rounding changes
can be amplified in this routed model. The cache is absent from the final
implementation; matching generated text alone was not sufficient evidence.

Encoder-timeline profiling ranked Q8 dense projections as the largest
measured kernel family. That instrumentation reduced throughput to roughly
34 tokens/s, so its absolute times must not be treated as production timings.
The remaining 5% target needs a larger improvement than these dispatch
fusions demonstrated; the unsuccessful Q8 and expert approaches above
should not be repeated without a new hypothesis.

## Reproduction and records

Save the baseline binary plus `metal/qwen4.metal` and `metal/moe.metal`
before editing, because the binary loads Metal sources at runtime. Run
`speed-bench/qwen38_mtp_compare.py` with those paths, the model/PLE paths,
`--no-mtp --repeats 3`, and an output directory. Omit `--no-mtp` for the MTP
regression. Use `speed-bench/qwen38_q2_parity.py` for the full-logit checks.

[Complete records](qwen38-q2-decode-fusions-results.json) preserve final
measurements, hashes, test output, numerical checks, and exploratory scripts
and patch snapshots. The snapshots are cumulative experimental states,
not exact revisions of every incremental screening run; their flags are
not supported by the final implementation.
