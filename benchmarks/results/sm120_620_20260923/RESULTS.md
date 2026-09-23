# cuDNN Frontend #620 — RTX 5090 SM120 O epilogue (2026-09-23)

## Result

`baseline_executed_no_epilogue_candidate`: the fixed upstream `develop` SHA `20f696cbd0b928dd6bacc907466dc16ab7fbb9a8` was built as cuDNN Frontend 1.31.0 in a new `0z5a-ptx` environment with CUTLASS DSL 4.7.1. Its SM120 FROST forward engine executed on the original RTX 5090 host. FP16/BF16 × D128/D256 graph E2E checks and four upstream graph API cases passed. The earlier Frontend 1.29.0 / DSL 4.6.2 gate remains documented in [the original log](wheel129-smoke.log); the original `/home/gongji/0z5a` environment was not changed.

The short dense cases launch 16 CTAs across 170 SMs and use 241–255 registers per thread plus 65.55–81.94 KiB dynamic shared memory per CTA. SASS confirms the O epilogue stages through shared memory then emits 128-bit global stores. D128's O stores have zero excess L2 sectors in NCU's source counters; neither profile reports local-memory spilling requests. This is evidence against treating the current store sequence alone as the demonstrated full-SDPA bottleneck. A bulk/tensor-store replacement was not implemented, and no O-epilogue candidate speedup is claimed.

## End-to-end and contextual speed comparison

The [source-built smoke log](source131_sm120_smoke.log) records four selected `sdpa_fwd_prefill_sm120` plans and independent PyTorch numerical checks: FP16/BF16 × D128/D256, B=1, H=8, Q=KV=128. The [upstream graph API test log](pytest620-graph-api.log) records four passes at B=2, H=8, Q=KV=256, D256, dense/causal FP16/BF16. Test isolation and timeout were disabled (`CUDNN_TEST_NO_ISOLATION=1`, `CUDNN_TEST_TIMEOUT=0`) so the test runner would not terminate workers.

The repository's [benchmark driver](bench620_matrix.sh) used 8 warmups and 30 timed runs per case on GPU 4. `cudnn_oss` pins FROST; `cudnn` deselects it and uses the native backend. Speedup is native latency divided by FROST latency. The printed median has 1 µs precision. These ratios compare existing backends, **not** an O-epilogue code change. [Raw CSV](bench620_matrix.csv).

| B=1, H=8, dtype, D, Q/KV | Native cuDNN (µs) | FROST SM120 (µs) | FROST/native speedup |
| --- | ---: | ---: | ---: |
| FP16, D128, 64/64 | 7 | 6 | 1.17× |
| FP16, D256, 64/64 | 7 | 8 | 0.88× |
| BF16, D128, 128/256 | 19 | 11 | 1.73× |
| BF16, D256, 128/256 | 24 | 15 | 1.60× |

## Actual O-store path and profile

The [general template](https://github.com/NVIDIA/cudnn-frontend/blob/20f696cbd0b928dd6bacc907466dc16ab7fbb9a8/python/cudnn/sdpa/fwd/kernels/sm120/prefill_f16.py) and [D256 template](https://github.com/NVIDIA/cudnn-frontend/blob/20f696cbd0b928dd6bacc907466dc16ab7fbb9a8/python/cudnn/sdpa/fwd/kernels/sm120/prefill_d256_f16.py) both normalize and convert O, reuse `sKV` as `sO`, perform `stmatrix`, load 8 half elements per lane, and store a 16-byte vector to O. The [selected SASS](sm120_epilogue_sass.txt) was extracted from the compiled plan's `kernel.o` fatbinary for `sm_120a`; it contains no O tensor bulk-store instruction. This describes the generated path, not a claim about all SM120 hardware capabilities.

| FP16 Q=KV=128, B=1, H=8 | D128 general | D256 template |
| --- | ---: | ---: |
| CTA grid / block threads | 16 / 256 | 16 / 128 |
| Registers per thread | 255 | 241 |
| Dynamic shared memory per CTA | 65.55 KiB | 81.94 KiB |
| NCU kernel duration | 8.64 µs | 14.98 µs |
| Static O epilogue `STSM.16.M88.4` / `LDS.128` / `STG.E.128` | 8 / 8 / 8 | 16 / 16 / 16 |
| Local-memory spilling requests | 0 | 0 |

[D128 NCU launch/occupancy](ncu-d128-baseline.txt), [D256 NCU launch/occupancy](ncu-d256-baseline.txt), [D128 source counters](ncu-d128-deep.txt), [D256 source counters](ncu-d256-deep.txt), and [per-instruction epilogue counters](epilogue_ncu_source.csv) are attached. D128 O-store instructions account for 8,192 actual and 8,192 ideal L2 sectors, hence zero excess; its 8,192 excess sectors occur at global load instructions. D256's O-store `L2 Theoretical Sectors Global Ideal` field is empty in the source report, so its derived “excess” field cannot establish uncoalesced O stores. D256 shared-load ideal values are reported as zero, so their derived excess-wavefront count is not a reliable bank-conflict attribution. NCU reports 0.09 waves per SM for both shapes. The small grid and occupancy limits are observed; their relative contribution to latency is an inference, not an isolated epilogue timing measurement.

## Provenance and decision

- Host: 8× NVIDIA GeForce RTX 5090, GPU 4 selected, 170 SMs on that GPU; driver 580.82.07; CUDA toolkit 13.0.88; cuDNN backend 92000; PyTorch `2.13.0+cu130`. Python: `/home/gongji/0z5a-work/0z5a-ptx/bin/python` on local ext4. No access to `/home/lcpu` was used.
- Built wheel SHA256: `16da0d85cbf6b162532cf82b2d20cc9b9eabae4cd2bfcdbf6c8e1c29aa64b092`. CUTLASS DSL and `libs-core` are both 4.7.1. D128/D256 extracted fatbinary hashes are recorded in [the SASS excerpt](sm120_epilogue_sass.txt).
- The source wheel was built with a local copy of DLPack v1.3 because the remote CMake GitHub fetch stalled. The successful build and all tests used the fixed source SHA. The separate fetch-based build was never interrupted.
- The profile reproducer is [profile_sm120.py](profile_sm120.py). NCU 2025.3.1 profiled one warmed kernel after `cudaProfilerStart`; its replay durations are distinct from the benchmark driver's medians. No model weights were downloaded.

For this fixed source and short-shape evidence, replacing the existing epilogue would add descriptor/setup and completion costs without addressing the observed 16-CTA grid or shared-memory occupancy limit. The data does not establish an end-to-end gain for that change, so this Draft PR remains a measurement report with no production kernel patch.
