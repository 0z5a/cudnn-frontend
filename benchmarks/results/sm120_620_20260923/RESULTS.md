# cuDNN Frontend #620 — RTX 5090 continuation (2026-09-23)

## Result

`blocked_baseline`: the original 8× RTX 5090 0z5a environment can load cuDNN Frontend 1.29.0 and discover the SM120 FROST engine, but cannot select it. For a dense FP16 D128 SDPA graph, engine analysis reported `requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2`; only backend plans `eng8_k24=3` and `eng11_k24=3` were available. Therefore no FROST kernel, O epilogue, full E2E, SASS, or speedup was measured. The installed environment was not changed.

| Test path | Baseline | Candidate | Speedup | Outcome |
| --- | ---: | ---: | ---: | --- |
| Dense FP16 D128, 1×8 heads, Q/KV=128 | N/A | N/A | N/A | FROST plan absent; DSL 4.6.2 below required 4.7.0 |
| Dense FP16/BF16 D256 and remaining shape matrix | N/A | N/A | N/A | Not run after the same environment-wide gate |

## Evidence

- Target: original 8× NVIDIA GeForce RTX 5090, driver 580.82.07; test bound to GPU 4. Python: `/home/gongji/0z5a/bin/python`; `nvidia-cudnn-frontend 1.29.0`, `nvidia-cutlass-dsl 4.6.2`, `apache-tvm-ffi 0.1.11`, PyTorch `2.13.0+cu130`.
- Source under review: `NVIDIA/cudnn-frontend` `develop` SHA `20f696cbd0b928dd6bacc907466dc16ab7fbb9a8`. The current [general SM120 template](https://github.com/NVIDIA/cudnn-frontend/blob/20f696cbd0b928dd6bacc907466dc16ab7fbb9a8/python/cudnn/sdpa/fwd/kernels/sm120/prefill_f16.py) and [D256 template](https://github.com/NVIDIA/cudnn-frontend/blob/20f696cbd0b928dd6bacc907466dc16ab7fbb9a8/python/cudnn/sdpa/fwd/kernels/sm120/prefill_d256_f16.py) stage O through shared memory before vector global stores. This is source inspection, not an observed device instruction trace.
- [Wheel smoke script](wheel129_sm120_smoke.py) and [log](wheel129-smoke.log): FROST opt-in was true, but eligibility rejected the installed DSL before graph execution. The script stopped after the first D128 FP16 case; its `StopIteration` is a consequence of the missing FROST plan.
- The checked-out source tests could not import directly because the source tree had no built `cudnn._compiled_module`. Using the installed 1.29.0 wheel, 107 of 134 SM120 test cases collected and 27 were deselected by repository configuration; [collection log](collect-wheel.log). One selected dense FP16 graph test was [skipped](selected-wheel.log) because the DSL version failed the test utility's availability check. The generic skip text says `cutlass/dsl not installed`; the engine's detailed eligibility message above establishes that 4.6.2 is installed but too old.
- Test isolation and timeout were disabled (`CUDNN_TEST_NO_ISOLATION=1`, `CUDNN_TEST_TIMEOUT=0`) so the repository test runner would not terminate worker processes. No process was killed.

## Gate for further work

Use an already installed compatible 0z5a environment with `nvidia-cutlass-dsl >= 4.7.0` and a matching Frontend binary, then confirm the selected plan actually executes FROST SM120 before profiling the O epilogue or writing a candidate. Without that baseline, an epilogue optimization claim would be speculative. No model weights were needed or downloaded.
