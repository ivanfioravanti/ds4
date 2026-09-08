# PR #5 integration validation on M3 Ultra

Integrated PR head `315cdfaf11af1d622439b90b8251bc79b6a1282b` with target
`881c678d1afd581b3668b199321c5195a538cca2`, 8 September 2026.
Hardware: Apple M3 Ultra, 512 GiB, Metal.

## Integration fixes

- Preserve the target branch's M3 Ultra MXFP4 specialization and 16-group
  down projection while adding the PR's M5 defaults. Keep the combined
  Q4_K/MXFP4 and IQ2_XXS/Q2_K exact fixtures without duplicate cases.
- Disable the prefiltered selector when its vector scorer is disabled or
  the batch is outside the decode path. The scalar/MM scorers do not write
  tile maxima. Added an exact-selection regression with deliberately stale
  maxima and independently overridden scorer/prefilter settings.
- Move shared PLE eviction and RAM-query helpers outside GPU-only guards.
  The latest target branch failed the CPU-only session-state build; this
  failure was reproduced on the untouched target before fixing it.
- Repair the archived Qwen image fixture to use image-token bounds and
  three-channel, non-temporally-duplicated patch strides. The old fixture's
  eight failures were reproduced on the untouched target.

## Passed validation

```
make -j8 all
make test DS4_TEST_ARGS="--server"
make test-qwen4-host
make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse
make test-q8-prefill-variants test-mxfp4-metal test-metal-moe-prefill
DS4_TEST_QWEN4_MV_EXACT=1 ./tests/test_qwen4_kernels
DS4_TEST_QWEN4_IDX_PREFILTER_ONLY=1 ./tests/test_qwen4_kernels
python -m unittest discover -s speed-bench -p 'test_qwen4_*.py'
```

The general target covers evaluation-case validation and extractor self-tests,
server/agent, CPU session state, TP commands, memory, layer packing, placement,
GPU argument parsing and CLI checks, prompt prefixes, sampling, and image
fixtures. All 40 Python benchmark tests passed. The model-dependent default
`ds4_test --all` includes unrelated DeepSeek/GLM/API-golden fixtures; Qwen
model tests were run explicitly below instead.

Both local Q4_K imatrix/MXFP4-down and IQ2_XXS/Q2_K-down models use the Q4_1
PLE sidecar. Against independently built target binaries and their own Metal
sources, each model passed six teacher-forced histories (1, 2, 8, 9, 39, 128
input tokens) times 32 steps: **384 full-vocabulary vectors total**, all
248320 values finite and byte-identical.

Each model passed ordinary and MTP greedy comparisons on Hamlet, Fibonacci,
and networking explanation prompts, two interleaved measured repetitions
per configuration plus excluded warmups. All text and MTP acceptance/cycle
counts matched. Both also passed `--session-snapshot --session-rewind`,
with `DS4_TEST_GLM_MTP=0` and `=1`.

The full Qwen kernel suite also passed with the selectable M5 paths forced
on, and MoE prefill specialization passed with expert-major ordering. A
32768-token full-model decode A/B with those overrides compared **145
full-vocabulary rows / 36006400 floats byte for byte**, including 16 warmup
steps and 128 measured steps. Its single timing comparison was +3.865%; this
is a portable-path diagnostic, not evidence to change M3 production defaults.

`test-metal-dense-mpp` skipped explicitly because it requires M5 tensor cores.
M5 hardware performance, CUDA/ROCm, and checkpoint-specific vision/API-golden
fixtures were not locally validated. No stopped quality-evaluation job was
resumed.

## Short-prompt timings

These are two-run means (also medians for two samples), not confidence bounds.
Ordinary decode differences under 0.4% are small; long-context chart sweeps
are a separate post-merge measurement.

| Model | Mode | Prompt | Baseline tok/s | Candidate tok/s | Change |
|---|---|---|---:|---:|---:|
| Q4 | plain | hamlet | 54.050 | 53.970 | -0.148% |
| Q4 | plain | fibonacci | 54.000 | 53.865 | -0.250% |
| Q4 | plain | explanation | 54.095 | 53.910 | -0.342% |
| Q4 | mtp | hamlet | 63.870 | 64.455 | +0.916% |
| Q4 | mtp | fibonacci | 77.685 | 77.775 | +0.116% |
| Q4 | mtp | explanation | 68.645 | 68.960 | +0.459% |
| Q2 | plain | hamlet | 54.860 | 54.710 | -0.273% |
| Q2 | plain | fibonacci | 54.780 | 54.730 | -0.091% |
| Q2 | plain | explanation | 54.925 | 54.915 | -0.018% |
| Q2 | mtp | hamlet | 64.360 | 65.065 | +1.095% |
| Q2 | mtp | fibonacci | 79.060 | 79.105 | +0.057% |
| Q2 | mtp | explanation | 68.270 | 68.735 | +0.681% |

Commands, hashes, model paths and individual measurements are retained in
[qwen38-pr5-m3-validation-results.json](qwen38-pr5-m3-validation-results.json).
Raw logs and orchestration scripts are under `OUT/pr5-validation/` in the
`ds4-pr5` integration worktree. Benchmark subprocesses use their respective
binary directories and pin the dense, Qwen and MoE Metal sources.
