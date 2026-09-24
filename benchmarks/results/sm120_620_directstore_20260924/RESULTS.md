# SM120 FROST D128 output store, RTX 5090, 2026-09-24

## Change and scope

For dense, non-packed SM120 forward attention with a 128-column V tile, transpose each four-lane MMA output group with four butterfly shuffles per fragment pair. Each lane then writes one contiguous 16-byte output vector directly from registers. The other shapes retain the existing shared-memory epilogue. This follows the direct register-to-output direction in [gau-nernst's SM120 attention kernel](https://github.com/gau-nernst/gn-kernels/blob/main/gn_kernels/cutedsl/sm120/sm120_attn_bf16.py); the permutation here is specific to this FROST MMA layout.

The comparison is the parent report branch's unchanged kernel at upstream `20f696cbd0b928dd6bacc907466dc16ab7fbb9a8`, against that kernel plus this source change. The tested source file SHA256 is `5fc6e7ddfcb660a83cabc3a0ee8701166c4bb34b64d5c5d65bf0164b621bb1df`. Candidate and baseline were loaded from distinct paths in fresh Python processes; each JSON records the imported `cudnn` path and selected FROST plan.

## End-to-end graph timing

The [driver](graph_bench.py) builds and selects the native cuDNN FROST graph, checks O against a PyTorch FP32 reference, and requires candidate O to be bitwise equal to baseline O. Each process captures 64 graph executions, warms up four replays, then times 30 replays with CUDA events. GPU 0 (`GPU-a015762a-f0d5-9065-109d-37898af80ecf`) on the eight RTX 5090 host was used. Speedup is baseline latency divided by candidate latency. Values below are medians of the independent process medians; speedups are medians of adjacent baseline/candidate pair ratios, not ratios of the displayed rounded columns.

| Dtype, B=1 H=8 D=128, Q/KV | Baseline µs | Candidate µs | Paired speedup | Process pairs |
| --- | ---: | ---: | ---: | ---: |
| FP16, 64/64 | 4.4632 | 4.4003 | 1.0149× | 4 |
| BF16, 128/256 | 8.0809 | 7.9863 | 1.0129× | 4 |
| FP16, 128/128 | 5.1759 | 5.0660 | 1.0216× | 4 |
| FP16, 512/512 | 14.1326 | 14.0376 | 1.0068× | 2 |
| FP16, 1024/1024 | 26.9740 | 26.8661 | 1.0040× | 2 |
| BF16, 1024/1024 | 26.9346 | 26.7814 | 1.0057× | 2 |

Short cases used process order `B C C B C B B C` ([raw JSON](runs/)); long cases used `B C C B` ([raw JSON](long-runs/)). All 18 candidate outputs passed the bitwise check against the matching baseline tensor and the independent FP32 reference tolerance. The short-case four pair ratios, respectively, were `1.0139, 1.0111, 1.0159, 1.0160`; `1.0112, 1.0146, 1.0192, 1.0095`; and `1.0219, 1.0212, 1.0246, 1.0214`. Gains below 1% on the long shapes are directional only. These are selected SDPA graph measurements, not full-model throughput claims.

GPU 4 had an unrelated high-utilization workload during an earlier exploratory run, so its timings are excluded from this table. No other process was stopped.

## Correctness and generated code

The [targeted upstream run](pytest-targeted.log) passed 44 existing SM120 tests, including dense, long, mask, Stats, sink, and head-dimension cases. A new L0 test passed for padded Q/KV lengths with D=128 and V/O widths 120 and 128 ([log](pytest-d128-padded.log)); it covers the zero-filled Q rows and final partial output column group. The earlier graph smoke test passed FP16/BF16 × D128/D256.

For the FP16 D128 64/64 plan, the candidate `sm_120a` SASS has 8 `STG.E.128` output stores and no epilogue `STSM.16.M88.4` or `LDS.128`; the baseline had 8 of each. The extracted candidate section SHA256 is `7d73ea36f7978b2f71e87ed86f2d536364f2ee50f037203d89bb1cacbeae206f`. [Resource output](shuffle64.resources) reports 255 registers, zero local bytes, and zero stack bytes. [NCU](ncu.txt) reports 256 threads/CTA, 16 CTAs, 65.55 KiB dynamic shared memory/CTA, 170 SMs, 0.09 waves/SM, and zero local-memory spilling requests. Its total 8,192 excessive L2 sectors remain; this aggregate is not an O-store coalescing proof. The NCU log contains a Python startup traceback after the target printed `passed` and profiling disconnected; the profiler exited successfully, but the log is retained as-is.

The original `/home/gongji/0z5a` environment was not modified. The isolated `0z5a-ptx` environment used CUTLASS DSL 4.7.1, cuDNN Frontend 1.31.0, driver 580.82.07, CUDA 13.0.88, and PyTorch 2.13.0+cu130. No model weights were downloaded for this kernel test.
