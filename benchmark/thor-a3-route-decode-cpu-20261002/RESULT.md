# Private routed-A predecode: closed CPU qualification only

The new private native9 routed-A predecode compiles for SM110 and passes a CPU address/data audit. GPU boundary, stream, allocator reuse, full Granite forward/generation and paired performance validation have **NOT_RUN** for this candidate. It is not in any existing live GPU pipeline, public PR source, or accepted optimization. Overall A3 remains open.

Only A predecode storage/addressing changes: allocate `(live, 1024)` encoded-derived integer words, read source rows through the existing destinations, retain full padded M for physical scale addressing, and let the existing repair read its route row. The original source tokens are preserved apart from two helper functions and these two A changes. The C++ exports are byte-identical to native7; the Python AST differs only in module name. Native C lookup, input/support checks, integer target/guard, B decoder/cache ownership, current stream, activation and AMAX remain source-preserved. M0/live0 source control retains no unnecessary A decode; invalid routes take the original rejection path before A words are consumed. This is source evidence, not new GPU execution.

## Actual terminal results

All three original logger records are `EXITED`, real returncode0, signal null:

| Closed CPU task | Elapsed seconds | Actual evidence |
| --- | ---: | --- |
| Native SM110 compilation | 83.965318671 | CUDA hidden/uninitialized, native calls0; explicit compute110/sm110, CUDA13.2, `--ftz=false --fmad=false`, MAX_JOBS2 |
| Literal host address/data audit | 194.849008183 | 70 cases; 13,811,712 output word positions verified; 13,765,632 valid encoded reconstruction calls |
| Read-only compiled code inspection | 2.168096588 | Both cuobjdump commands0; SM110 and new routed predecode plus original predecode/rounding/activation/bounds kernel symbols present |

Native CUDA source SHA256 `10ba822eceef85dbb1716ac83c77311b285d8499e8eaa9f72427419c13491315`; binary SHA256 `a1efe51dd955c01bc8401adbab8058e4b8f0ad07214a36202bd33d3435306c7f`. Original native7 CUDA parent SHA256 `f02ca51bd028a800e4c851e2f318a30905e878782502476acb4419e23ce7c7b8`. Binary files remain on Thor and are excluded from this publication.

The auditor compiles literal candidate routing/SF-address/decode function bodies through a **CPU float/BF16 shim**. It compares five previously qualified A24 GPU decoded-A corpora to an independent CPU unpack, then checks actual routes, dense/all rows, 128-row tile boundaries, unsorted duplicates, changed destinations, random permutations, invalid int64 destinations and empty live rows: 40 captured-data cases. Another28 cases use independently permuted synthetic E4M3/UE8M0 physical storage at M128/256/384/2048, including negative zero and extreme finite scales. Two M0 cases check CPU/source no-decode control only. All45 mixed invalid destinations are never decoded and use zero scratch placeholders; three M0 invalid destinations retain an unread poison sentinel in the CPU control. Placeholder output positions are included in the 13,811,712 count; GPU M0 scratch/output behavior is not validated here. CPU arithmetic or host source execution does not prove native GPU arithmetic, concurrency, allocator or stream behavior.

For the saved true E32/top8 decoder geometry, full native7 A predecode uses2,097,152 words; route-only geometry uses8,192 words. These are work/allocation counts, **not measured GPU time or speedup**. Native main geometry, K4096 four-component recipe, E32/top8, all24 Granite layers, all1,334,628,352 parameters, checkpoint, token IDs, original BF16 baseline and unchanged1% fixed-oracle/5% BF16 gates remain required. No BF16/Torch gate/up dot or oracle output substitution is introduced.

## Existing numerical limits and next qualification

The original K1024 recipe remains failed and incompatible with both original oracle/BF16 gates on the saved cases. The previously qualified all24 K4096 encoded-only repair control is numerically valid but costly; its passcode83 application flag remains false. Saved actual cached7 full-forward flags show99.9973524%,99.9975659%,99.9983231% integer repair for73/70/256 tokens. The accompanying census reproduces these counts from immutable original metadata; it is not a profile or measured gain. This candidate preserves certification; removing it is only an unimplemented separate hypothesis.

Keep existing frozen core-r3 cached-generation/native31/integer8, current ff6 native31, and compact-A GPU queues unchanged. They wait behind the shared Thor GPU lock. The routed-A candidate has no GPU job queued and cannot be promoted without meaningful native boundaries/reuse/mutation/two-stream checks, original full-forward/generation/KV gates, independent saved-output audit and frozen paired wrapper/model timing. Current draft PR1280 at ff6 distinguishes CPU/latest source checks from the last qualified A24 GPU tests; these results do not close its pending GPU scope.

## Retention and transport

Archive SHA256 `59615f90a64c8a7e84c3fa4a1acd72de4da76defc899d9c07a33911a9b3c9704`; 31 original closed payloads plus manifest, all Mac byte/SHA256 verified. Original logs, frozen source, host generated source, build flags and terminal records are preserved. Mac extraction initially hit an unsupported Python `tarfile.extractall(filter=...)` API; the archive digest already matched, and explicit regular-member copying after path/link checks succeeded. This was a local transport API retry, not a numerical/build failure. Models, engines, cache, .pt outputs, native/host binaries and live GPU results are excluded. Source snapshots use `.txt` only to retain executed bytes under repository formatting hooks.

vLLM remains inactive; heartbeat remains paused. No other task is interrupted, no GPU lock is bypassed, no service restoration timer or device setting is changed. CPU qualification is complete; GPU candidate acceptance and overall A3 completion remain open.
