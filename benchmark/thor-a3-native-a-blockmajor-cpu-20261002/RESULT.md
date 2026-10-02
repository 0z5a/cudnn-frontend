# Thor A3: direct native-A block-major packing, CPU-only proof

A further private packing candidate has actual CPU logger EXITED0/signal null in 32.383308503 s. It preserves the frozen high/residual quantization and physical SFA packing, generating the existing K32 component order directly with two stack/reshape operations. **GPU boundary/stream/reuse/full-model correctness and paired performance have not run; no optimization is accepted.** Existing compact-A/core GPU experiments and public correctness source remain unchanged.

## Numerical and source scope

Original functions are loaded from frozen AST bodies of the actual model workers, without importing their CUDA/model setup. All 48 BF16/FP32 cases at128/256/512 rows and a leading batch dimension cover zero/signed zero, finite extrema/BF16 subnormals, changed noncontiguous inputs and repeated calls. All144 paired encoded-A/physical-SFA/logical-scale buffers have identical shape/dtype/bytes against the original function, with distinct output storage and unchanged inputs. Three additional K2048 quantizer cases/9 buffers match the original through the unchanged compact-function path; this is quantizer behavior preservation, not new native-kernel geometry admission. Invalid M127/K1000/empty M0/K512/K128 cases retain their original exception classes. B remains explicitly rejected by the same native-A-only guard; original B/reference function identities are unchanged.

The original gather maps all4096 encoded slots and128 logical-scale slots exactly to the direct block-major order. AST checks preserve the original high/residual equations, component selection and physical packing statements; the private transformation replaces only concatenation/index allocation/two gathers and omits the already-proven discarded effective operand. The helper inherits the existing K_BLOCK32 factory precondition and uses unchanged compact behavior outside the optimized K1024 path. Native support checks/kernel/stream/AMAX code are not edited.

A real CPU TorchDispatchMode trace at128x1024 records the following. These are CPU operator calls, not GPU launch counts or timing:

| Operator | Original | Discarded-result-only compact | Direct block-major |
|---|---:|---:|---:|
| aten.arange |1|1|0|
| aten.index_select |2|2|0|
| aten.cat |2|2|0|
| aten.stack |0|0|2|

The required residual dequantization remains; original effective-result dequantization count3 becomes1 as in the preceding compact candidate. No original B/reference/global function is patched. Torch 2.13.0+cu130/gitcf30153c4c131c8164ee7798e5022d810682e2cb, CUDA_VISIBLE_DEVICES empty, CUDA uninitialized, new GPU/native calls0. GPU allocation/copy/stream behavior and elapsed-time benefit cannot be inferred from this CPU trace.

## Status and evidence

Public correctness PR1280 remains draft atff6adf01/upstreamac89; only its CPU source/import/binding reuse is qualified. Last completed native31/full-model numerical qualification remains A24. Original full Granite/all24/E32/top8/H1024/I512/1,334,628,352 parameters, checkpoint/baseline/oracle/original token IDs and1%/5% gates stay pinned. Original K1024 precision remains failed, private K4096 integer controls remain expensive, passcode83 remains application false, and overall A3 optimization stays open. No native BF16/Torch gate-up replacement or output substitution is introduced.

All10 original executed source/log/metadata payloads plus the manifest are transferred to Mac and byte/SHA256 verified, archive `8729c87e108434be656813b8c91b492451c1a3c2bc5699f033edb9f2208d963a`. PUBLIC_FILES.json preserves original paths/digests; executed source text uses .txt suffixes so formatting hooks cannot alter it. Models/cache/engines/PT/native/JIT binaries are excluded. This candidate has no GPU job queued at this checkpoint; the earlier frozen compact-A GPU pipeline is unchanged. vLLM inactive, heartbeatPAUSED, no recovery timer/clock/power/fan changes or interference with other tasks.
