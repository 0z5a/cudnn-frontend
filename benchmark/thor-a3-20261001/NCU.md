# A3 Nsight Compute capture — 2026-10-01

Sixteen production-kernel captures completed successfully. The first twelve compare dedicated SwiGLU, unified GLU and the lean-launch candidate on the saved four32/ragged8 inputs, first with cold-cache replay and then with source line information and no profiler cache flush. Four more compare 256×256/256×128 tiles on the freshly rebuilt latest-source candidate. All sixteen actual ncu return codes are zero. One target GEMM was captured per process after warmup. C, D and AMAX output bytes match before/after profiling and across each comparison; latest-tile capture digests also match the unprofiled ABBA benchmark. Production source manifests remain unchanged.

## Provenance and capture settings

Thor SM110, 20 SMs; ncu 2026.1.1.0, cuDNN 9.20, CuTe DSL 4.7.1, Torch 2.13.0+cu130. Root was used because `RmProfilingAdminOnly=1`. The shared `/tmp/codex-thor-perf.lock` serialized all GPU work; vLLM stayed inactive. No service restoration timer or clock/power changes were made. Explicit `--clock-control none` prevented the profiler's default boost-clock override.

Frozen measured heads:

- Dedicated: a3c7d901134615688f47db97987ddc773bef182a.
- Unified: 8efb996d16a4cf8692b137b788cbe868d768bbbe.
- Lean launch: 0257acc7584c3733f72b27fb3dbc87cdbcf14a63.

Latest develop was checked at b56af9d558e10bb20ecff3cf6e5879ba18134f04. Its CuTe GEMM sources and tests are unchanged from 14a1e999. The local candidate was separately rebased to 531f7746255a55c97b9b7f3d0e41f8ef621711e4; its CuTe GEMM files match the measured lean candidate exactly. The fresh latest-source binding/plan validation is recorded separately from these original-head captures.

The profiler includes the `a3_ncu_execute/` NVTX range and cudnn kernel-name filter, one launch, kernel replay. AMAX reset happens before the profiled range on the same stream. The captured kernel does not include the wrapper allocator or AMAX reset. Sections: SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats, MemoryWorkloadAnalysis, InstructionStats and SourceCounters. r1 uses `--cache-control all`; r2 uses `--cache-control none` and only `CUTE_DSL_LINEINFO=1` for source correlation. Python source files are imported in all six r2 reports. r2 PC sampling has two aggregated passes and zero dropped bytes or buffer overflows.

## Findings

All three implementations launch 224-thread CTAs in two-CTA clusters. Dynamic shared memory is 204800 bytes per CTA (200 KiB), plus driver reservation. Both register and shared-memory residency limits are one CTA per SM. Dedicated uses 149 registers/thread; unified and lean use 147. Local and shared spill requests are zero in every capture. This excludes spilling as an observed explanation for the earlier performance gap.

Source-correlated long-scoreboard samples on ragged8 place the largest PC group at the accumulator consumer wait: dedicated line 2336, unified line 2671, lean line 2723 in their respective frozen kernel files. Unified has 137 of 440 long-scoreboard samples at that source line; lean has 142 of 442. Tile-info/TMA and other pipeline wait locations also appear. The SASS at prominent PCs is polling/branching with `NANOSLEEP.SYNCS`, so a generic long-scoreboard label alone is not evidence that ordinary data loads or expert-offset global reads dominate.

Unified/lean execute the same number of instructions for four32 (382044); ragged8 is 561508/561520. Dedicated has 371168 and 551010 respectively. Source sampling does not establish the serial expert scan as the dominant cost, and the earlier offset-cache candidate still has no measured speed benefit. The lean FFI change reduces host overhead, while these captures show essentially the same device work.

These are short kernels, with a small PC sample and optimized DSL source attribution. PC counts describe where sampled warps wait; they do not measure the causal latency contribution of each function. Captured profiler durations vary substantially from the normal benchmark and across cache/line-info settings. They must not be used as paired speedup or full-model throughput results. Normal unprofiled ABBA timing remains the performance evidence.

The concrete follow-up tested existing smaller MMA tiles without changing support checks or operation semantics. The initial offline plan sweep is in a3-tile-sweep-r1-20261001. The latest-source reference/reuse and frozen ABBA comparison is complete: 22 final tests passed with zero skips, and all sixteen timing processes passed exact output audits. Read [README.md](README.md). No production default has been changed.

## Latest-source 256×128 comparison

All four r3 captures use actual head 531f7746255a55c97b9b7f3d0e41f8ef621711e4, based on latest develop b56af9d, through the fresh private native binding. The source remains frozen. These captures have line information, cache-control=none, clock-control=none and five imported Python source files each. Raw reports, metric tables, exact commands and source correlation are preserved. There are no dropped PC-sampling bytes or sampling-buffer overflows.

| Resource or metric | four32 256×256 | four32 256×128 | ragged8 256×256 | ragged8 256×128 |
|---|---:|---:|---:|---:|
| Registers/thread | 147 | 143 | 147 | 143 |
| Dynamic shared memory/CTA, bytes | 204800 | 221184 | 204800 | 221184 |
| Register/shared residency limit, CTAs/SM | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Executed instructions | 382044 | 373124 | 561518 | 547318 |
| Issue active, % | 29.643 | 29.929 | 30.884 | 34.148 |
| Eligible warps/active cycle | 0.2998 | 0.3103 | 0.3129 | 0.3596 |
| Local/shared spill requests | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| PC samples | 398 | 370 | 692 | 540 |

All launch 224-thread CTAs, grid size 20 and two-CTA clusters. The initial hypothesis that a narrower tile would reduce total per-CTA shared memory is disproved by the measured allocation: it increases from 200 to 216 KiB. The register decrease does not increase the measured residency limit. Source compute_stages selects one accumulator stage with overlapping accumulation for tile N=256 and two accumulator stages for tile N=128; A/B stages are separately computed from available storage. Tile width therefore changes pipeline structure and scheduling as well as dimensions. These data are consistent with changed device-work efficiency, but do not isolate a single causal mechanism for the unprofiled speed improvement.

On ragged8 the default capture has 412 correlated long-scoreboard samples; the narrow capture has 302. The narrow capture still has an accumulator-wait polling/branch group: 113 samples at source line 2720 and 33 at 2723, compared with 124 at default line 2723 and differently attributed polling/branch PCs. Moving samples between optimized source lines is not evidence that accumulator waiting was eliminated. Instruction counts decrease approximately 2.3% (four32) and 2.5% (ragged8); a source PC count or profiler duration cannot explain the approximately 24% normal ragged8 completion improvement by itself.

The normal ABBA benchmark preserves BF16 C/D, probability scaling, AMAX reset and the same saved inputs. Its paired ragged8 time ratios are 0.7643 for compiled execute and 0.7584 for graph replay; wrapper ratio 0.9982 has an interval spanning no change. The 128-wide plan continues to reject FP8 D. No full-model benefit or universal plan-selection policy is claimed.

## Evidence

See [ncu-captures.json](ncu-captures.json) for all 16 capture hashes, heads, selected metric values and source-correlated samples, and [README.md](README.md) for the separately measured unprofiled comparison. Original binary reports, exact commands and full raw logs remain archived locally.
