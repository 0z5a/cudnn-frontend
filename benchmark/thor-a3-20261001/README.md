# A3 256×128 plan validation — 2026-10-01

The existing 256×128 MMA tile improves GPU completion on the saved BF16-output workloads while preserving C, D and per-call AMAX exactly. On ragged8, paired compiled-execute time decreases 23.6% and graph-replay time decreases 24.2%. The public wrapper remains host-bound and shows no statistically clear improvement. This is a supported offline plan choice, not a production default change or model throughput result.

## Actual source and build

Latest upstream develop checked: b56af9d558e10bb20ecff3cf6e5879ba18134f04. The locally rebased unified SwiGLU plus lean-launch head is 531f7746255a55c97b9b7f3d0e41f8ef621711e4. Its CuTe GEMM sources match the previous lean candidate 0257acc7 byte for byte; upstream CuTe GEMM sources/tests match the earlier develop 14a1e999. Other upstream native binding changes were not assumed equivalent: a fresh private native editable build on Thor exited 0 in 225.817 seconds, and the actual cudnn import resolved to a3nculatest/python/cudnn/__init__.py. Both timed plans use this same rebuilt source and binding. Full production Python manifests match before and after all sixteen benchmark processes.

Environment: Thor SM110, 20 SMs, Torch 2.13.0+cu130, cuDNN 9.20, CuTe DSL 4.7.1, TVM FFI 0.1.11. All GPU work is serialized under /tmp/codex-thor-perf.lock; vLLM remains inactive. Clocks and power settings were not changed.

An additional CPU-only import check confirms the actual native binding is a3nculatest/python/cudnn/_compiled_module.cpython-312-aarch64-linux-gnu.so, 6532408 bytes, SHA256 74408ba755cc064b52c0d1f7dfd1940985da43e2c82a1178c7dd41916abeb167, returning backend version 92000. That provenance JSON is saved alongside the NCU analysis. The Thor transport contains the complete current head/tree; historical parent trees were not all transported, so historical git-show traversal is limited. The current runtime and its full before/after source manifests are unaffected.

## Scope and correctness

Both timed cases use N=512, K=1024, MXFP8 A/B with E8M0 scaling, BF16 C/D, FP32 accumulation, probability scaling, per-call AMAX, m_aligned=256, sf_vec_size=32 and cluster_shape_mn=(2,1). Control MMA tile is (256,256); candidate is (256,128). Four32 uses four 32-row groups. Ragged8 uses (0,32,64,0,128,256,32,64), including empty experts.

The first correctness run finished 18 passed / 4 skipped. The skips were the production support gate rejecting FP8 A/B with 128-wide tiles and FP8 D, not failed BF16 kernels. That restriction remains intact. The follow-up r2 replaces those four unsupported reference attempts with four direct public-wrapper rejection checks after a successful BF16 call. Actual pytest return code is 0, with 22 passed, zero errors/failures/skips (33.20 seconds in pytest; 35.556 seconds including process startup).

The final 22 cases cover:

- Eight default-tile Torch-reference cases: BF16/FP8 D, probability on/off, regular/ragged groups.
- Four narrow-tile BF16 Torch-reference cases with probability on/off and regular/ragged groups.
- Four narrow-tile reused-plan cases at expert counts 1/31/32/33, changing device offsets and alpha/probability, checking a side stream against a cold plan and empty-expert AMAX.
- Default/narrow per-call AMAX tests.
- Four explicit narrow-tile FP8-D rejection checks, including the warm BF16 memo path.

All sixteen ABBA processes exited 0. Independently audited C/D/AMAX dtype, shape and SHA256 bytes match across both tiles and all repetitions; each timing mode also checks its post-timing outputs against its own setup outputs. NCU captures are separately checked against these exact benchmark digests. This does not establish support for FP8 D on the narrow tile, nor generalize performance to other shapes, architectures or activations.

## Unprofiled paired performance

Each workload has four paired process observations in ABBA order repeated twice. Each process measures five groups of 1000 calls after a one-second warmup per mode. The table reports the median of four process medians and the geometric mean of paired candidate/control ratios. Bracketed intervals use a paired log t interval with three degrees of freedom; these are small-sample descriptive uncertainty, not independent-device replication.

| Workload | Mode | 256×256 µs | 256×128 µs | Paired time ratio [95% interval] |
|---|---|---:|---:|---|
| four32 | wrapper including allocation/reset | 48.846 | 48.528 | 0.9913 [0.9616, 1.0221] |
| four32 | compiled execute with AMAX reset | 14.733 | 13.549 | 0.9230 [0.9068, 0.9395] |
| four32 | graph replay with AMAX reset | 16.593 | 15.781 | 0.9496 [0.9423, 0.9569] |
| ragged8 | wrapper including allocation/reset | 48.569 | 47.965 | 0.9982 [0.9610, 1.0369] |
| ragged8 | compiled execute with AMAX reset | 21.932 | 16.779 | 0.7643 [0.7615, 0.7672] |
| ragged8 | graph replay with AMAX reset | 24.674 | 18.705 | 0.7584 [0.7549, 0.7619] |

These are wall completion intervals per call. CUDA event intervals include host submission gaps. Graph replay includes the required AMAX reset. Host enqueue does not clearly improve; ragged8 graph enqueue increases approximately 11% while its completion time decreases. First-call cache state is uncontrolled, and memory counters cover only the PyTorch allocator. cProfile diagnostics run separately from timing groups. No result is presented as standalone kernel time, public-wrapper speedup, model E2E throughput or cold-compile benefit.

## Profiling and reproduction

Read [NCU.md](NCU.md) for the profiler provenance and source correlation. The 256×128 follow-up uses the same latest source, line information and cache-control=none, with one warmed production execute captured per process. The profiler explicitly uses clock-control=none. AMAX reset is outside the NVTX range on the same stream. Profiler durations are diagnostic and do not supply the performance ratios above.

For this validated BF16-output path, pass mma_tiler_mn=(256,128) and cluster_shape_mn=(2,1) through the existing wrapper or compiled-plan construction. Keep production support checks enabled. FP8 D must retain a supported wider tile. A wider performance sweep is necessary before changing any default selection policy. PR1280 remains independent correctness work; no performance PR or default-selection patch has been published from this experiment.

## Published evidence

- [Paired process audit](paired-audit.json): all process observations, output digests and small-sample intervals.
- [NCU capture summary](ncu-captures.json): all 16 reports, exact source heads, report hashes, selected raw counters and source-correlated PC samples. Source paths are normalized to repository/package-relative paths; counter values are unchanged.
- [Final correctness XML](correctness.xml): 22 cases, zero failures/errors/skips.
- [Build and actual import provenance](build-and-import.json): fresh binding build return code and imported native module hash.

Full original .ncu-rep binaries, raw logs/metrics, source manifests and tests are archived locally. The latest 245-file archive has SHA256 acb393d00d3cb74781d19e160c84bf1b2feb4dfc0d8b520004c5c1cdbf70da4e, verified file by file after transport. Models, engines, caches and tensor .pt files are excluded. The current runtime head/tree is complete; the Thor transport does not contain every historical parent tree. Our GPU jobs exited; an unrelated job acquired the shared lock afterward and was left running. vLLM remains inactive.

Current #1233 head 82c25771e99e4504accefa7efa24241af04b313a was additionally fetched and compared with the rebased pre-fix control f8b193aa. Their python/cudnn/gemm/cutedsl and test/python/gemm/cutedsl trees are identical. The actual measured head nevertheless includes the separately disclosed Thor correctness and lean-launch changes. This evidence branch adds only reports/data above that measured head; it is not a production default-selection PR.
