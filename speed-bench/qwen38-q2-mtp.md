# Qwen3.8 padded Q2_K: MTP optimization round

Baseline: `b4c3550`, after the two Q2 prefill/decode rounds. Hardware:
Apple M3 Ultra, 512 GiB unified memory. Model:
`Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`, with the Q4_1
PLE sidecar used in the [previous round](qwen38-q2-round2.md).

## Selected change

The two-token hyper-connection mixer now uses sixteen SIMD groups per
threadgroup at the measured 2560-wide, rank-320 shape on M3 Ultra. This
shares activated low-rank inputs across sixteen independent output rows
instead of four, reducing threadgroup count. The existing Metal kernel
already supports this grouping; its arithmetic and reduction order are
unchanged. Other shapes and devices retain four groups. Ordinary
single-token decode and prefill retain their original dispatch.

`DS4_QWEN4_HC_PAIR_NSG=4` restores the previous paired layout. The override
accepts 1 through 16 and applies only when the two-token paired mixer is
enabled. `DS4_QWEN4_NO_HC_PAIR=1` still disables that mixer altogether.

## MTP measurements

Three interleaved repetitions per prompt, with separate excluded warmups.
Rates exclude model loading and prefill. Each measured candidate rate
exceeded its paired baseline rate. This is a small local improvement,
not evidence of a large or universal MTP speedup.

| Prompt | Baseline tokens/s | Candidate tokens/s | Gain | Accepted / cycles |
|---|---:|---:|---:|---:|
| Hamlet | 63.97 | 64.16 | 0.30% | 44 / 71 |
| Fibonacci | 78.71 | 78.89 | 0.23% | 197 / 201 |
| Explanation | 67.68 | 67.92 | 0.35% | 106 / 148 |

All generated outputs and acceptance counts matched exactly. The
[complete records](qwen38-q2-mtp-results.json) retain individual samples,
commands, binary/source hashes, and validation results.

Ordinary-decode medians changed by -0.15%, +0.24%, and -0.17% respectively,
with overlapping sample ranges and identical outputs. Its mixer dispatch
is unchanged; these measurements do not establish an ordinary-decode gain.

## Experiments not selected

- More SIMD groups in two-token Q8 dense projections did not give a
  reliable end-to-end gain.
- Larger routed-expert row groupings slowed MTP; the existing IQ2/Q2
  decode defaults remain in place.
- Sharing IQ2 weight unpacking when both verifier tokens select the same
  expert in the same slot preserved tested outputs but was slightly slower.
- GDN group sizes 2, 4, 8, and 16 did not establish a useful improvement
  over the existing eight-group verifier layout.

Exploratory screens used the explanation prompt and two observations per
setting in reversed orders. Their rates are selection evidence, not added
to the final gain. The temporary IQ2 kernel and Q8/GDN overrides were
removed. Diagnostic synchronization costs are excluded from reported rates.

## Validation and reproduction

The expanded kernel suite compares paired-mixer groups 1, 2, 8, and 16
against four groups, bit-for-bit, for F16, F32, and Q8 weights. Widths
9, 64, and 2560 cover partial groups and the production shape, with
guarded outputs and finite-value checks. Both `make test-qwen4-q2` and
`make test-qwen4-kernels` passed.

The ordinary full-logit regression checks matched all 192 teacher-forced
vectors and both next-token vectors following 1022- and 7952-token chat
prompts. These checks establish implementation parity on the tested
histories; they do not assess quantization quality against BF16.

An instrumented build captured 550 full-vocabulary FP32 vectors from
target passes (including both verifier rows) and predictor passes.
Every value was finite and byte-identical to the baseline. Cases cover
128-token greedy generation, 128-token exact sampling at temperature 0.7,
top-p 0.8 and seed 123, 64-token MTP after the 7952-token prompt, and
output budgets of 1, 2, and 3. Outputs, acceptance counters, and per-cycle
accept/reject traces also matched. Both diagnostic builds use the same
instrumented engine C object and their respective Metal host objects;
their timings are excluded. The portable records include the diagnostic
scripts and hashes. `make -j8 all` also passed.

For timings, save the baseline executable and runtime Metal sources, then
run `speed-bench/qwen38_mtp_compare.py` with `--model`, `--ple`,
`--baseline`, `--baseline-source`, `--baseline-moe-source`, `--repeats 3`,
and `--out`. The default measures MTP; `--no-mtp` runs ordinary decoding.
