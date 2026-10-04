# Thor A3 latest upstream: fresh native build and asynchronous queue

The grouped SwiGLU correctness branch is source head `6c252fab563ae531e92a7c8114de345217ea9761`, merging upstream develop `9ecb75e7771fcfb933e19136ace70ab8fade83fa`. The six PR code/test files retain the A24 GPU-qualified bytes. Latest-head GPU tests, the complete Granite run and new performance results are **NOT_RUN**. This report establishes fresh CPU native-build provenance and the execution/queue state; it does not establish a new kernel or model pass.

## Actual source and native build

Independent Thor checkout `/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928/a3sfdlatest9ecb_r2` and matching dedicated venv were used. Upstream changed 15 of 118 tracked native inputs, including headers and native SDPA bindings. A new editable installation compiled the native extension rather than reusing the previous binding.

- Native build: **actual EXITED0, signal null, 583.499847741 seconds**.
- Complete preparation: **actual EXITED0, signal null, 804.310140496 seconds**.
- Loaded package and binding resolve to the new checkout. Binding SHA256: `d6ed1ac6d7b43d024e48c5fd864678c0988bc5e5b8bfc8673c8a78c4c47dbd7e`.
- All 118 native input hashes were rechecked; the older source remained unchanged.
- CPU import used `CUDA_VISIBLE_DEVICES=""`, CUDA remained uninitialized, and there were zero new native GPU calls.
- Six PR Python files pass Black line-length160, SPDX and syntax checks. Source transport preserves the original shallow-history boundary; imported source matches all118 native hashes and has the exact tested head.

The original R1 attempt **actually exited1/null after12.906341068 seconds**, before compilation: a global Git URL rewrite sent the public fetch through gh-proxy.com, which returnedHTTP429. The failed source, harness and logs remain intact. R2 uses an independent checkout and output directory and bypasses global Git configuration for its public fetch command only. No global configuration was modified. The successful build is R2; R1 is not a build success.

## Actual asynchronous execution contract

A scoped `OBSERVE_ONLY_NO_SIGNALS` runner was independently CPU audited: **actual EXITED0/null/5.914410636 seconds**. Two intentional children exceeded their observation deadlines and then finished normally with actual returncodes7 and9, signal null; both completion markers were retained. Nested propagation with a replaced `PYTHONPATH`, policy-off control, unchanged legacy logger and all48 frozen A3 inputs were checked. The policy sent zero signals and did not import Torch or launch GPU work. Source files are immutable and their hashes are in the manifests.

Original R5 naturally ended **EXITED75/null/43207.693185506 seconds** without GPU admission, completed phases, Torch or CUDA; both original handles were confirmed absent. Its shared-lock wait result is preserved. One distinct R6 wrapper now reuses the frozen R5 mathematical source and unchanged numeric gates. At **2026-10-04T11:37:57Z**, its logger2802174/worker2802185 were actually live; there was no admission/completed/result record. It waits for `/tmp/codex-thor-perf.lock` and does not bypass other work. Its future tests explicitly have historical **AAB/A24/b57** scopes, not this latest6c head.

After genuine R6 qualification, a separately validated latest-head follow-on must first run the selected31 GPU tests/61 native calls, then validate native9 boundaries/reuse, complete24-layer Granite forwards and conditional paired generation. The old E491 prepared coordinator requires oldR5 success and cannot be run unchanged or counted as current-head validation. The latest C++/SDPA GPU tests and full upstream/other-architecture matrices are also NOT_RUN.

## Original Granite objective remains incomplete

Granite3.1 1B-A400M Instruct remains pinned to its original nine checkpoint files and1,334,628,352 parameters, all24 layers, E32/top8/H1024/I512, original input token IDs, oracle and BF16 down projection. Original1% fixed-oracle/5% BF16 quality gates are unchanged. Production-default K1024 still fails its original quality gate; the private K4096 repair control is not a production-default repair. Original passcode83 refusal/applicationfalse is retained. No accepted new optimization, stock throughput gain or overall A3 completion is claimed.

Last actual selected GPU qualification belongs to A24 `a24a7401bb7d26d42d88d98b316f634ef487301c`, with31 passed and61 actual native calls. The same six code/test bytes are preserved here, but that result is not a6c GPU execution. [Historical GPU/model evidence](https://github.com/0z5a/cudnn-frontend/blob/3783003f7e4cb0ee7e80e1fc00783ddd8633b63d/benchmark/thor-a3-guarded-rounding-20261002/RESULT.md).

## Raw evidence

Two closed evidence archives were copied to Mac and every payload size/SHA256 checked: 13 fresh-build payloads and41 execution-control payloads. Public source snapshots have `.py.txt` suffixes; raw file bytes remain exact. Models, caches, engines, `.pt` captures, native/JIT binaries and the binary Git bundle are excluded from publication. `PUBLIC_FILES.json` maps each original payload to its public path and hash.

- Native archive SHA256: `0b5960f8339ee88b6b9fabff9fbc293c324d93c1ca5913f0421e3dc4e99a2dd1`.
- Control archive SHA256: `e6c2569b5038989bc9db2ce095c471c3842d3d6a869817e6ca899fb2162d6448`.
- Exact source bundle SHA256 (stored on Mac, not published as binary): `3e42f37fa8fe542b533cc6fd65cee95f3f04d32fcb52d64bd06a41c84a561909`.

vLLM service remains inactive; no restoration timer, process termination, lock clearing, GPU-setting change or other-task interruption occurred. Existing human/maintainer PR ready/draft state is preserved. The asynchronous heartbeat remains active and stays quiet when state is unchanged. The original A3 goal is still open.
