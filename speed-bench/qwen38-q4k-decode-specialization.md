# Qwen Q4_K decode: specialize MXFP4 expert down projections

The Q4_K model uses Q4_K gate/up expert weights and MXFP4 down expert weights. On M3 Ultra, specializing the MXFP4 decode kernel removes runtime format/width branches and dispatches 16 SIMD groups with one output row each. The per-lane accumulation and reduction order are unchanged. Other devices retain their previous defaults; Q2 defaults are unchanged. `DS4_QWEN4_MOE_MV_SPECIALIZE=0` restores the generic expert kernel.

## Ordinary decode measurements

Apple M3 Ultra, 512 GiB, Metal. Original chart baseline: `1bf4842`; fresh control: `a30ed07`. Candidate applies the MXFP4 specialization change on `a30ed07`. Same Q4_K GGUF, external Q4_1 PLE, and frozen Promessi Sposi prompt as the chart. Two runs per build, candidate/control/control/candidate order, 128 generated tokens per frontier, 262273 allocated context positions. Prefill is incremental; snapshot/restore/replay are outside timing. All GPU work was serial.

| Context | Original t/s | Fresh control t/s | Candidate t/s | vs original | vs fresh control |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 48.745 | 47.995 | 49.860 | +2.29% | +3.89% |
| 8,192 | 48.575 | 47.925 | 49.825 | +2.57% | +3.96% |
| 16,384 | 48.455 | 47.800 | 49.695 | +2.56% | +3.96% |
| 32,768 | 48.190 | 47.615 | 49.405 | +2.52% | +3.76% |
| 65,536 | 47.945 | 47.315 | 49.050 | +2.30% | +3.67% |
| 131,072 | 47.125 | 46.425 | 48.220 | +2.32% | +3.87% |
| 262,144 | 44.915 | 44.440 | 46.095 | +2.63% | +3.72% |

Minimum measured gain across the seven contexts: **2.29%**. Two-run means describe these measurements, not confidence bounds. The 5% stretch target was not established by this experiment.

## Correctness

Build and full Qwen kernel suite passed. Expanded exact MoE tests cover Q4_K/MXFP4 and IQ2/Q2 combinations, 1/2/3/9 tokens, real 2560×640 dimensions, partial widths, shared experts, and nine dispatch geometries. All 192 teacher-forced full-vocabulary decode vectors were finite and byte-identical to the fresh control. All 28 full-vocabulary frontier dumps and generated continuations match between the four full sweeps; frontier dumps also match the previous chart.

## MTP regression

Two measured repeats per prompt after separate warmups. Outputs and acceptance counts match between builds.

| Prompt | Control t/s | Candidate t/s | Change |
|---|---:|---:|---:|
| hamlet | 60.82 | 63.71 | +4.75% |
| fibonacci | 73.76 | 77.53 | +5.10% |
| explanation | 65.39 | 68.50 | +4.76% |

## Other experiments

MXFP4 specialization with fewer SIMD groups or two rows per group was slower. Increasing Q4_K gate/up groups to 16, reducing them to four, and changing the early command submission to layer one or three did not justify further default changes. The temporary gate/up group-limit extension was removed. Raw screens are retained, including compilation-delayed first-token observations.

## Records

Local raw records, frozen runtimes and patch hashes: `OUT/qwen38-q4k-decode-goal/`. Full-sweep commands, stdout/stderr, CSV rows and logit dumps are in `validation/`; exact vectors are in `parity/`. The initial candidate sweep exited before recording a row because the diagnostic output directory was missing; its logs are retained separately.

Compact measurement and verification records are in [qwen38-q4k-decode-specialization-results.json](qwen38-q4k-decode-specialization-results.json).
