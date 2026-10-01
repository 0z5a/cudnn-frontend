# Private A3 C-store elision: protected TMA reuse and frozen validation

The production-arithmetic private candidate passes 79 GPU tests. On three frozen Granite first-layer inputs, captured wrapper execution is about 9–10% shorter than the unchanged production control, with identical D, column D, row/column SFD and AMAX. This is a private unused-output optimization, not a complete Granite precision pass or end-to-end throughput result.

The existing correctness PR remains head `741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1`, based on develop `e2bf967b17dae17a642102198e6d862d559655c9`. Production source is unchanged/clean. Fresh binding SHA `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a`; Thor SM110, PyTorch 2.13.0+cu130, CUDA Toolkit 13.2, cuDNN 9.20, CuTeDSL 4.7.1.

## Change and discovered dependency

The full model consumes D, while intermediate gate/up C is unused. A private constructor boolean skips only the C-store block. C storage remains allocated and is explicitly unavailable as an output in omit mode; this does not reduce allocation or shared-memory capacity. No public API flag, cache-key change, additional input tensor, support bypass or fallback is introduced.

Naively removing C stores corrupts D. The original C pipeline acquire waits on global hardware TMA store groups and incidentally protects D shared-memory reuse. The corrected candidate explicitly acquires the D pipeline immediately after its commit, before the next stage reuse. Both corrected emit/omit controls use that explicit acquire; a third unchanged control ensures a slower corrected emit arm cannot manufacture a speedup. This diagnosis is based on the failed native output assertion and corrected byte parity, not a claimed independent hardware race trace.

## Production-arithmetic correctness

Candidate source SHA `3543ffc3140058646845040a4924278ae2a323d049492d4b5383af0d6ff04482` derives production SHA `5fe58b7118d6cd82d822b82b8f01aa96665414d3e6499d4752cd3cb7f4095b1d`; FP32 activation, vector/scalar math, reductions, support checks, stream and AMAX semantics remain unchanged.

- Corrected emit run: 63 passed, pytest rc0, no failures/errors/skips. Includes 31 existing representative regressions plus 32 exact original/emit/omit comparisons across E=1/4/31/33, narrow geometry, canonical/legacy layouts, BF16/FP16 D and scalar/vector math.
- Omit run: 16 passed, pytest rc0, no failures/errors/skips. Existing independent activation/SFD poison/guard tests cover both widths, empty experts, cached changed-input calls, ownership of earlier outputs, changed-scale graph replay, per-call AMAX and cold/warm launch streams.
- Outer driver rc0, 253.274995274 s. Python execute counters 157/31 include captured graph nodes; they are not replay/kernel-count claims.
- Independent CPU audit rc0, 2.523144524 s: all 79 JUnit records/imports/source hashes and 54 saved performance tensor entries. Boundary assertions are GPU test assertions; boundary tensors were not separately saved for CPU numerical re-execution.

## Frozen production capture comparison

Same full pinned checkpoint first-layer weights and previously CPU-audited original input/route captures. Three arms: unchanged original emit-C, explicit-D-wait emit-C, explicit-D-wait omit-C. Fixed M8192/K4096/N1024, E32, matching physical inputs/scales/offset hashes, no probability input in these primitive comparisons. C poison sentinels confirm actual C-write omission. All five non-C outputs and original/protected emitted C match byte-for-byte, including after timed replay.

| Original tokens | Omit / original paired median | Reduction in captured execution time | Omit / protected emit |
|---:|---:|---:|---:|
| 73 | 0.896254970 | 10.375% | 0.898933616 |
| 70 | 0.905361180 | 9.464% | 0.878746999 |
| 256 | 0.907308446 | 9.269% | 0.891540841 |

Each case has eight balanced ABBA/BAAB blocks against both controls, 25 graph replays per phase, 1600 timed replays/case, 4800 total. GPU producer rc0/32.323995231 s. Python execute counter108 includes nine graph capture calls; model native calls0. CPU audit recomputes all balanced ratios and saved byte pairs, but does not independently remeasure GPU timing. Events measure the captured wrapper work, including its other captured operations, not an isolated kernel-only duration. Raw within-run variation is retained; clocks/cache/power were not changed. Formal model performance remains NOT_RUN.

## Separate experimental full-model parity

The prior private BF16-staged pure-MXFP8 epilogue is evaluated separately from production FP32 activation. Corrected emit/omit run rc0/25.626703245 s executes all24 native expert calls for each of six full-model cases (144 model calls). All six native and six oracle complete-logit tensors are byte-identical to the frozen unmodified K32/BF16-stage control. The full pinned Granite model/revision, 1,334,628,352 parameters, E32/top8, original BF16 baseline and 1% oracle/5% BF16 gates are preserved. Native/oracle errors remain about10.41%,10.50%,4.34% for73/70/256tokens; every oracle gate fails. No generation/KV or model speedup success is claimed.

The experimental third-arm run rc0/32.610186652 s yields paired omit/original ratios0.953017200,0.899730489,0.937928805. These results are distinct from the production table. Independent CPU audit rc0/4.997275062 s checks all102 saved entries (48 model/paired/sentinel plus54 third-arm), complete-logit metrics, sentinel nonfinites, byte pairs and timing. Combined CPU coverage is156 saved tensor entries.

## Preserved failed attempts

- Experimental r1: real rc1, loader registration failure after a first execution; no complete numerical/performance result.
- Experimental r2: real rc1, unsafe C omission changes model outputs and fails first primitive D byte assertion. Partial metadata/logs retained; full tensors were not saved, so those partial metrics are producer-only evidence.
- Production test r1: real outer/pytest rc1, 47 passes and16 failures. All failures are CPU raw-byte view exceptions on trailing singleton legacy strides (`stride(-1) must be 1`), before equality evaluation. Corrected r2 flattens the CPU tensors before the same exact-byte assertion; GPU source/inputs/tolerances do not change. Failed source, all63 JUnit cases and full logs are preserved.
- `audit_a3_production_c_store_20261001.py` was a prepared auditor targeting failed r1 and was not executed; the executed r2 auditor validates the successful result and all preserved failure causes.

## Evidence and remaining scope

All nine terminal driver statuses are retained, including three rc1 histories. The archive has108 files/all107 manifest entries size/SHA-verified on Mac; `.pt` tensors, models, cache, engines and native binaries remain on Thor. Source byte copies are published with `.py.txt` names and mapped in PUBLIC_FILES.json. Existing production checkout stays clean and vLLM inactive; no restoration timer or clock/power change.

No optional output contract/cache-key/API validation or public feature has been accepted. Full24 production-model candidate parity, model precision, generation/KV and meaningful model performance remain open. The original all24 recipe still fails; an unused-output optimization does not repair that numerical recipe.
