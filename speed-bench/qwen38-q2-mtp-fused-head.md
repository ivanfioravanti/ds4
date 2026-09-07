# Qwen Q2-down MTP output-head fusion experiment

Baseline: `66b0e3f`, M3 Ultra with 512 GiB, IQ2_XXS gate/up and padded Q2_K down model.

Decision: reject the prototypes and retain the existing output projection plus GPU argmax. Removing one dispatch did not improve end-to-end decode performance.

The prototype preserved the Q8 output head's eight-SIMD-group K walk and reduction order, emitted one argmax partial per group of vocabulary rows, then selected the final token in one 256-thread group. Variants used two or sixteen rows per group. A third variant omitted the full-logit writes. All were opt-in with `DS4_QWEN4_MTP_FUSED_HEAD=1` during the experiment; no such flag remains in production.

| Prototype | Baseline tokens/s | Fused tokens/s | Change |
| --- | ---: | ---: | ---: |
| rows2 | 68.265 | 67.825 | -0.64% |
| rows16 | 68.115 | 67.760 | -0.52% |
| rows16-no-logits | 68.260 | 67.730 | -0.78% |

Values are means of two warm repetitions per configuration, interleaved in alternating order. An initial pair was excluded for shader warmup. This is a rejection screen on one 256-token networking explanation, not a broad performance study. Context capacity was 8192, prompt length 48 tokens, temperature zero, with MTP enabled. Both configurations used the same candidate executable and runtime Metal source, selected by the opt-in flag; the unfused path retained the existing output projection and argmax.

All eighteen runs produced byte-identical output, with 148 verification cycles and 106 accepted drafts each. The prototypes built and executed successfully, but full-logit parity and comprehensive kernel validation were not pursued after the performance rejection. The first two variants retained logits for possible diagnostics; the last skipped those stores.

The two-row design produces 124160 partial winners, versus 61 in the current argmax implementation. Sixteen-row grouping reduces that to 15520 but changes projection scheduling. These are plausible costs, not independently profiled explanations for the regression. Removing the logit stores also failed to recover a gain.

The accompanying JSON preserves raw rates, output hashes, runner, and all three prototype patches against the baseline. Production source was restored and all executables rebuilt after the screen.
