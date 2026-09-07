# Qwen3.8 Q2 MTP: predictor token selection

Baseline: `0bb323a`, after the paired-mixer tuning round. Tests use an
Apple M3 Ultra with 512 GiB unified memory and
`Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf` plus the Q4_1
PLE sidecar from the [preceding measurements](qwen38-q2-mtp.md).

## Implementation

The MTP predictor previously copied 248,320 FP32 logits to the CPU and
scanned them to choose a draft. When only a draft token is requested,
the candidate uses two GPU reductions and reads back a single int32.
It reduces independent chunks of 4096 logits, then merges their winning
score/index pairs. The extra per-session scratch contains one pair per
chunk, plus the final index. It is allocated and freed with the MTP graph.

The full output projection still runs, and callers requesting logits keep
the full readback. Target verification and sampling use their existing
logits. Tie-breaking favors the lowest index; NaNs are ignored, and the
initial score of -1e30 with index zero matches CPU selection, including
the all-invalid fallback. `DS4_QWEN4_MTP_GPU_ARGMAX=0` restores CPU
selection for comparisons. The implementation changes are confined to
the Qwen Metal graph and kernels.

## Performance

Five measured repetitions per prompt, interleaved after separate warmups.
Rates exclude model loading and prefill. Outputs and acceptance counts
matched in every run. All fifteen measured candidate runs were faster
than their paired baselines, but the gains are small.

| Prompt | Baseline tokens/s | Candidate tokens/s | Median gain | Accepted / cycles |
|---|---:|---:|---:|---:|
| Hamlet | 64.01 | 64.15 | 0.22% | 44 / 71 |
| Fibonacci | 78.86 | 79.03 | 0.22% | 197 / 201 |
| Explanation | 67.98 | 68.03 | 0.07% | 106 / 148 |

These are local M3 Ultra measurements. The change does not establish a
large decode speedup or a performance gain on other hardware.

The ordinary-decode explanation regression measured 54.47 to 54.44
tokens/s (-0.06%), with overlapping ranges and identical output. Its
prefill rate was 233.30 to 233.13 tokens/s (-0.07%). Both are effectively
unchanged; this optimization targets predictor token selection.

## Numerical validation

All **547 full-vocabulary FP32 vectors** captured from target, verifier,
and predictor passes were finite and byte-identical to the saved baseline.
The candidate diagnostic also checks every GPU-selected draft against
CPU selection from its full logits. Generated outputs, acceptance counters,
and per-cycle decisions matched in all cases:

- 128-token greedy generation.
- 128-token exact sampling: temperature 0.7, top-p 0.8, seed 123.
- 64-token MTP generation following a 15,881-token chat prompt.
- Output budgets of one, two, and three tokens.

The baseline and candidate diagnostics use their respective engine and
Metal host sources, with the same trace hooks. Their timings are excluded
from performance measurements.

The kernel tests compare against CPU selection for vocabulary sizes at
SIMD, threadgroup, and 4096-value chunk boundaries, through the full
248,320-token vocabulary. Cases include ties across chunks, signed zeros,
NaNs, positive/negative infinity, scores at/below -1e30, invalid sizes,
and guarded output/scratch buffers. `make test-qwen4-q2`, the full
`make test-qwen4-kernels`, and `make -j8 all` passed.

## Expert-reuse experiments

Two-token kernels were tested for IQ2 gate/up and padded Q2 down weights,
matching shared experts even when their routing slots differed. They kept
the original per-token accumulation order and paired repeated IDs
bijectively to avoid multiple writers. Focused numerical checks passed,
but all configurations were slower than the existing kernels.

A second prototype computed the cross-slot matches once per dispatch.
It recovered some Q2 down performance but still did not beat the baseline.
Both prototypes were removed; quantized weights and expert dispatch remain
unchanged. Exploratory rates are retained separately from final results.

## Reproduction

Save the baseline executable and runtime `metal/qwen4.metal` and
`metal/moe.metal` before editing. Run `speed-bench/qwen38_mtp_compare.py`
with the model/PLE paths and these saved baseline paths. The final MTP
comparison uses five interleaved repetitions per prompt after separate
warmups. The ordinary-decode regression uses three repetitions of the
explanation prompt with `--no-mtp --case explanation`.

[Complete records](qwen38-q2-mtp-argmax-results.json) include commands,
source/binary hashes, individual samples, diagnostic scripts, full-logit
hashes, and exploratory screens. These checks establish implementation
parity on the tested histories, not quantization quality relative to BF16.
