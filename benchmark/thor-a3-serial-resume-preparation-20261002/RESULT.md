# A3 closed waiters and serial continuation preparation

The three previous A3 waiters genuinely exited with code 75 and no signal. They did not import Torch or enter GPU validation. This is a scheduling result, not a numerical failure or a test pass. A new frozen serial queue reuses the completed native builds and preserves the original full Granite objective.

| Closed worker | Actual elapsed seconds | Terminal scope |
| --- | ---: | --- |
| Core b57 r3 | 7200.466909435003 | Shared GPU lock unavailable; prior CPU build reused; GPU not run |
| Frozen ff6 native regression | 9111.933101846007 | Core dependency plus shared lock wait; GPU not run |
| Compact A pipeline r1 | 12001.501442581 | Required cached-generation and b57-regression results absent; GPU not run |

Two new CPU checks actually completed with `EXITED`, return code 0 and no signal:

| CPU-only check | Actual seconds | Verified scope |
| --- | ---: | --- |
| Route input and main contract preparation | 1.7676965739956358 | Five qualified captured case geometries; 637 valid and 65 invalid int64 route inputs; independent padded expert association; original boundary main AST unchanged; mutation main precision, tiling and stream AST unchanged |
| Serial source and provenance preflight | 0.8651020340039395 | 46 source dependencies; all 114 current native input hashes; installed binding hash and source paths; Mac/Thor new source bytes identical |

Neither CPU check imported Torch or made native calls. These checks do not prove GPU correctness, streams, reuse, full-model quality or performance. The separate native9 compile and 70-case CPU address/decode audit are recorded in `benchmark/thor-a3-route-decode-cpu-20261002/RESULT.md`; they were not repeated here.

At 2026-10-02T06:54:21.675085Z, the new queue's logger 807143 and worker 807144 were alive without Torch/CUDA libraries. No GPU admission or terminal record existed. The shared-lock owner observed at 06:52:47Z was PID 545711; PID 555938 was another waiter, as shown by the arrow/blocked record. No other task was interrupted. The source is frozen at SHA256 `9303f04cae5032a50f5ba4d4d2c4698e472c0a807fe860f2867e2bc67d44797a`, waiting up to 43200 seconds under the existing shared lock. The outer logger deadline is 63000 seconds. Queue creation is not successful execution.

The serial order is current installed-source native31 at PR head `aab01504ad3ad97170d19087ce23bef48d95dfa0`, then A24 cached7 full32 generation and model timing with a CPU audit, b57 native31 and integer8 boundaries/pairs/full forwards with CPU audits, conditional integer8 full32/model pairs, compact A full-model/generation pairs with CPU audits, then current-head native9 route boundaries/reuse and independent CPU capture audits. Each GPU child verifies the owning parent's lock FD before importing Torch. Original A24 and b57 test scopes retain their actual heads; they are not relabeled as current-head results.

The routed-A probe is prepared for 46 records per arm: reordered and duplicated routes, SF tile edges, changed route storage, invalid int64 routes, empty live rows, M=0 cases, in-place A/physical/logical scale mutation with a new real MXFP8 main call, owner rejection, preserved old outputs, and a different stream with an explicit ready event. Expected 46 repair calls and 103 postprocess launches per arm are **not executed counts**. Reference C/D/AMAX use the original complete padded expert matrix geometry and remain outside the native repair branch. The later CPU audit uses the independently decoded physical operands and integer fixed-window arithmetic. Native9 full-model/generation/performance pairing is not part of this queue and remains required before accepting that optimization.

GitHub readback confirmed PR1280 still draft at aab0150 with current `develop` base `5f27ff4f023e23c1d36261e4a2300cc1e4a9a2c6`. Its original seven-file correctness patch remains unchanged. Current-head GPU31 is queued, not passed. The original K1024 recipe remains failed; the separate all24 K4096 control has prior numerical evidence but no accepted production optimization. All 24 layers, E32/top8/H1024/I512, 1,334,628,352 parameters, nine checkpoint files, original input IDs, the 1% fixed-oracle and 5% original-BF16 gates, and original BF16 downprojection are preserved. Original passcode83 application result remains false.

The closed-waiter archive contains 28 raw payload files, SHA256 `3f1d505cedd25e8c29351edca80e10f61878a0e450422a2b60f012f1fdb68613`. The closed CPU/source preparation archive contains 64 raw payload files, SHA256 `6a5a9f642facc4df3bf8842c78285b45301cec4f7c057adba8007ed43928694a`. Every payload's size and SHA256 were verified after transport to Mac. Models, caches, engines, binaries and `.pt` output files were excluded. Initial packaging/launch setup errors were preserved separately and corrected; none executed GPU tests.

vLLM remains inactive, heartbeat `pr229-a3` remains paused, and overall A3 completion remains false. PR229, PR224, A2, SeedVR2 and video artifacts retain their independently completed scopes.
