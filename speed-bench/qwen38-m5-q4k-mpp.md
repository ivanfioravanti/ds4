# Qwen3.8 M5 TensorOps: Q4_K routed MPP and prefill chunk width

Two measurements on Apple M5 Max (137 GB, Metal 4 tensor API enabled):

1. A new double-buffered TensorOps/MPP routed-expert kernel for Q4_K weights
   (`kernel_mul_mm_id_mpp_dbuf`), selected for Q4_K gate/up (f32 RHS) and
   Q4_K down (f16 RHS) in the mm_id prefill dispatch.
2. Prefill chunk width on this machine: the routed GEMMs are batch-width
   limited at the 1024-token chunks the release evals use.

## Change

`kernel_mul_mm_id_mpp_dbuf` stages the next k-step's weight and activation
tiles into alternate threadgroup buffers while the cooperative matmul reads
the current ones, paying one barrier per k-step instead of two. Only the
Q4_K instantiations use it; the iq2_xxs / q2_K / mxfp4 / cached paths keep
the original single-buffered kernel, which measured neutral-to-negative for
those dequant costs (IQ2_XXS grid lookups already hide staging latency;
Q4_K dequant is cheap, so the barrier mattered). Threadgroup allocation for
the affected encodes grows 8192 -> 12288 bytes.
`DS4_METAL_DISABLE_Q4_K_MM_ID_MPP=1` restores the simdgroup kernels.
Staged values and the mm.run accumulation order are unchanged, so outputs
are bit-identical to the simdgroup path only up to accumulation order (same
operand domains, f32 accumulation); the kernel suites verify exactness
against their CPU references.

## Methodology

Interleaved same-model pairs (A,B,A,B,...) at a single 16384-token
frontier, `--gen-tokens 128 --prefill-chunk 4096`, one pair entry per
adjacent runs so thermal drift divides out. Two contamination modes were
identified and excluded from the final numbers: first-run-after-model-switch
pays cold weight paging (~40% prefill penalty with a 65 GiB model), and
run order tracks heat soak (~10% across a session). Alternating models
inside a pair is invalid on this machine.

## Results (prefill t/s, 16384-token frontier, chunk 4096)

Q4 (Q4KImatrix-MTP qwen4exp) — five warm interleaved pairs across two
sessions:

| Pair | mpp | simdgroup | delta |
| --- | ---: | ---: | ---: |
| r3-1 | 1034.9 | 1057.1* | -2.1% |
| r3-2 | 1024.4 | 965.9 | +6.1% |
| r3-3 | 964.1 | 927.4 | +4.0% |
| r6-2 | 962.8 | 925.1 | +4.1% |
| r6-3 | 889.2 | 869.4 | +2.3% |

*r3-1's simdgroup run followed the mpp run while pages were still settling;
r6-1 (cold mpp first run) excluded for the same reason. Warm-pair mean:
**+3.8%, 5/5 positive**. An isolation variant (Q4_K MPP on the original
single-buffered kernel) measured -0.9% mean: the double-buffered staging is
the active ingredient, not the TensorOps swap alone.

Q2 (IQ2XXSImatrix-Q2KDownPad768-MTP) — identical code path before/after by
construction (original kernel retained); warm pairs measured +0.6%, +0.2%:
**neutral, no regression**.

Decode is untouched (M=1 routed and dense paths unchanged); decode t/s
moved within run-to-run noise for both models.

## Prefill chunk width (both models, 16384-token frontier)

| chunk | Q2 prefill | Q4 prefill |
| ---: | ---: | ---: |
| 1024 | 790, 488 | 466, 445 |
| 4096 | 1025, 995 | 970, 1005 |
| 8192 | 1059, 1069 | 535, 539 |

The routed mm_id GEMMs at 1024 tokens run ~20 rows per expert per tile
(NR1=32, 10 of 512 experts per token), leaving the TensorOps tiles mostly
idle. Q2 is fastest at `--prefill-chunk 8192` (~1064 t/s, +35% over 1024);
Q4 is fastest at 4096 (~1000 t/s, 2.2x over 1024) and regresses at 8192
(measured 537; not investigated further). The release docs' 1024-token
examples are an eval-comparison convention, not a throughput
recommendation for this machine class.

## Verification

`make -j8 ds4 ds4-bench`, `./tests/test_qwen4_kernels` ("all qwen4 kernel
tests passed"), `make test-qwen4-q2` (132 PASS lines, including Qwen MoE
specialization exactness for type=12/10/16/8/39/2 at T=8193 with padding
intact). Greedy smoke on both models returns correct one-shot answers
through the new path. Kill-switch verified (Q4 simdgroup restored).

[Machine-readable results](qwen38-m5-q4k-mpp-results.json)
