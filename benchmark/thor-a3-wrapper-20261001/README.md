# A3 public-wrapper optimization — 2026-10-01

This round is complete: the final candidate reduces public-wrapper completion time by 3.4–7.9% in six paired comparisons. It is an operator result on one Thor, not full-model throughput. Runtime kernel source and production default tile remain unchanged. Prepared execute/graph do not gain from this host change; one ragged8 narrow graph comparison is 0.24% slower (paired ratio 1.0024, interval [1.0003, 1.0045]).

## Source and change

Latest develop was fetched at the start and after timing began: `a4dcf87ca1b77fbc0ab62642e8cb7136a63baf73`. Control `83374af96ec3f8e22dd159d278bdf25acab17a4e` and candidate `209d9e61c6ca873ccf5b869e83db00eed9f8f966` both use that base. They include the unified #1233 implementation, the previously validated Thor correctness fixes and the dense-SwiGLU lean-launch experiment; neither is unmodified upstream/#1233.

The SwiGLU wrapper now retains a bounded (256-entry) memo of full operand metadata, compiled-cache keys and output shape/stride/dtype specifications. It retains no user tensor, pointer or stream in that memo. Every invocation still allocates independent public outputs, creates the AMAX -inf reduction identity and calls the existing checked execute path with live inputs. A compiled-cache clear forces support/compile work again; metadata M changes output geometry without adding M to the compiled-plan key. Device and overlap margin are also present in the compiled key.

Torch allocation/reset work is enclosed by the shared `cudnn._torch_stream.stream_context` on the launch stream and input device. The helper now skips context creation when the raw handle is already current, including 0, and checks a torch Stream's device before that shortcut. 0/1/2 still avoid ExternalStream. No execute allocations, device-to-host reads, conversion kernels, support spoofing or default-policy changes were introduced.

## Correctness and provenance

Final candidate: **67 passed, zero skips** (10 shared-stream checks, 22 wrapper/key checks, 31 reference/cache/layout cases, 4 public-wrapper graph captures). The real two-GPU stream test was explicitly deselected on this single-GPU host; the wrong-device/equal-default-handle host contract was tested with a synthetic stream. Control also passed four public-wrapper graph tests. The corrected behavioral detectors are RED on the actual control: 12 wrapper failures plus one current-default-stream failure, all at the expected assertions, no collection/context errors.

Coverage includes fresh pointers/data with stable metadata; canonical and legacy BF16/FP8 D output ownership and byte parity; unchanged prior returned outputs; M changes and cleared compile caches; probability/dtype signatures; partial/full dynamic modes; overlap margin; 1/31/32/33 experts with live routing changes, zero alpha, reused plans and an explicit side stream; 0/1/2/None streams; warm sync-debug checks; and wrapper CUDA Graph replay after routing/alpha/prob/norm changes. The narrow BF16-output plan still explicitly rejects FP8 D.

Three private native builds exited 0: control 217.911s, first candidate 222.975s, final candidate 179.599s. Every pytest worker and timing process checked actual Python/native paths. All production Python and native-binding hashes match before/after formal timing. The imports/final-provenance files preserve the actual paths, sizes and SHA256 values.

The initial candidate run is preserved: 20 passed and two side-stream checks failed because the test compared PyTorch Stream objects rather than CUDA handle/device (the logged raw handles were equal). The final tests compare handle/device and remain RED on the old wrapper. No failed test was skipped or represented as a pass.

## Frozen paired timing

MXFP8 A/B + E8M0 scales, BF16 C/D, N512/K1024, probability and AMAX enabled, cluster (2,1). Inputs are the previous saved CPU-generated four32/ragged8 datasets with identical SHA256 across all sources, tiles and stream modes. The control/candidate always use the same tile and stream mode in a comparison. `default` is 256x256; `narrow` is the already-supported 256x128 BF16-output plan. Names ending `none` pass current_stream=None; the other names pass the current raw CUstream.

Four paired process observations per comparison, ABBA repeated twice; five groups of 1000 calls per mode per process. **48/48 processes exited 0**. C/D/AMAX output byte hashes agree across both sources, both tiles and both stream entry forms, and every timing mode checks outputs against its setup result. GPU work used `/tmp/codex-thor-perf.lock`; no CPU native build overlapped formal timing. vLLM stayed inactive; our jobs ended and released the lock. No clock/power setting or service-restoration timer was changed/created.

| Case | Control wrapper µs | Candidate wrapper µs | Paired time ratio [95% small-sample interval] | Time reduction |
|---|---:|---:|---|---:|
| four32-default | 48.903 | 45.976 | 0.9377 [0.9181, 0.9578] | 6.2% |
| four32-narrow | 48.700 | 46.279 | 0.9480 [0.9250, 0.9715] | 5.2% |
| ragged8-default | 48.325 | 46.610 | 0.9662 [0.9390, 0.9942] | 3.4% |
| ragged8-narrow | 49.207 | 45.851 | 0.9296 [0.9151, 0.9444] | 7.0% |
| four32-narrow-none | 49.526 | 46.455 | 0.9419 [0.9032, 0.9823] | 5.8% |
| ragged8-narrow-none | 50.647 | 46.246 | 0.9213 [0.8897, 0.9540] | 7.9% |

These are wall completion intervals per call, including host submission and wrapper allocations. Ratios/intervals are computed from four paired process log-ratios (Student t, 3 degrees of freedom), not by treating within-process groups as independent. This is descriptive uncertainty from one device/session. Compiled execute and graph retain required AMAX resets. First-call compile/cache state is uncontrolled; memory counters cover only the Torch allocator. cProfile runs are separate diagnostics and their durations are not speedup evidence. Warm wrapper dtype derivation/stride sorting/SF permutes disappear, while eight generic metadata reads and output allocation remain.

No new NCU capture was needed for this host-only change. The earlier narrow-tile/NCU result remains a separate experiment: approximately 24% ragged8 prepared completion improvement, without the wrapper improvement measured here. Do not multiply or attribute these numbers to full-model inference, TE training, broader dimensions, output dtypes, or untested architectures. No new performance PR has been opened; correctness PR1280 remains independent.

## Raw evidence archive

Raw logs, XML, host profiles, import/freeze metadata, source snapshots and drivers were archived on the testing machine and transferred to the author’s Mac. The public audit and validation summaries are included here. Archive SHA256: `9ce06e972a03ebf01772fda2b6f44a53617148aa5600d6a0b0237ae55ae65d46`; all **454 files** were checked after transport. Tensor .pt files, cache, engines, models and native binaries were excluded. The paired audit, validation summary and combined candidate XML below contain the publishable evidence. Private raw paths are normalized in the validation summary; raw logs remain archived.

The graph probe in `public-wrapper-graph.py` was run from `test/python/gemm/cutedsl/` on both checkouts. Its original SHA256 and commands are recorded in `validation.json`.
