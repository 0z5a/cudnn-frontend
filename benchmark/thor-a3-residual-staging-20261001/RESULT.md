# Granite arithmetic and residual controls — 2026-10-01

The current all24 A3 task is still open. No production code or new native recipe has been accepted. New causal controls show that the short-input discrepancy already exists with original unquantized gate/up weights when their dot product is computed in FP32, even when C and activation are staged through BF16. Adding residual components alone cannot repair that demonstrated arithmetic difference. The unchanged 1% native/same-recipe oracle and 5% native/original-BF16 gates remain mandatory.

Runtime stays frozen independent PR1280 `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`; same pinned full Granite, all 1,334,628,352 original parameters, 24 layers, E32/top8, H1024/I512. Every control includes all positions and 49155 vocabulary logits of the original 73/70/256-token cases. Actual native calls in this continuation are **zero**. Torch-only reconstruction/arithmetic controls are not supported native features or accuracy successes. The earlier layer23-only hybrid keeps its explicitly limited scope.

## Complete arithmetic controls

All controls retain original down projection and top-k merge semantics. The stock arithmetic trace is byte-identical to the stored original BF16 math case (producer assertion). At each of its 24 layer inputs, the original BF16 adapter output is independently audited byte-identical to the captured stock expert output. Four new complete controls use either original weights/inputs or two reconstructed components, with fixed declared arithmetic. The previous expanded-K4096 BF16-stage oracle is reused rather than rerun.

| Torch-only control | 73-token / BF16 L2 | 70-token / BF16 L2 | 256-token / BF16 L2 |
|---|---:|---:|---:|
| original_fp32_dot_fp32_activation | 12.757296% | 11.165024% | 4.929253% |
| original_fp32_dot_bf16_stages | 10.335080% | 9.162215% | 4.633200% |
| double_effective_fp32_dot_bf16_stages | 10.335115% | 9.162254% | 4.633200% |
| double_recast_bf16_dot_bf16_stages | 0.000000% | 0.000000% | 4.001023% |

The original-FP32/BF16-stage and factorized reconstructed-FP32/BF16-stage complete outputs differ only slightly in the two short cases; the 256-token case matches their reported FP64 L2 exactly. Their short-case fixed-oracle compatibility bounds still exceed5%. The reconstructed-BF16-dot control matches the two short original baselines byte-for-byte but differs4.0010% on the long case. It is diagnostic only: production A3 still does not accept BF16 A/B or offer BF16 gate/activation staging. No native/oracle gate is claimed for these controls.

## Same-input numerical evidence

At layer0, original BF16 C and original-FP32 C rounded into BF16 differ in270/598016 values. BF16-stage D differs235/299008 and final expert output differs848/74752 values. Expert-output L2 is0.03582%. The reconstructed effective-FP32 control has exactly the same first-layer C/D/y errors. This local comparison plus full controls establishes an arithmetic sensitivity independent of large quantization loss; it does not identify one GPU instruction as the sole cause.

The independent CPU auditor reconstitutes every saved stock-path two-component input from its actual byte codes and logical SF32. Maximum input reconstruction L2 across24 stock-path inputs is approximately2.90e-8;53 BF16 recast elements differ. Producer complete-weight reconstruction has maxL2~1.51e-9 and27580 BF16 recast elements differ. Complete GPU weight component tensors were not saved; do not call that a GPU component-pack audit.

An additional independent CPU logical quantizer reads the exact original checkpoint (SHA256 ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1) and covers all805306368 gate/up elements, plus48 earlier native/reference input trajectories. This is a CPU reconstruction diagnostic, not a new native recipe or complete-model pass.

| Components | Complete checkpoint max L2 | Checkpoint FP32/BF16 recast mismatches | Earlier input trace max L2 | Input FP32 mismatches |
|---|---:|---:|---:|---:|
| 2 | 1.50563990892e-09 | 27580 | 2.87946602424e-08 | 139 |
| 3 | 7.94056215255e-12 | 533 | 2.84502625139e-11 | 2 |

Three components further reduce reconstruction error but have not been run as native A3; no expanded-K9216 recipe or benefit is asserted. The existing no-quantization FP32 control is the stronger evidence that more bits alone do not settle the short-case arithmetic issue.

## Verification and remaining scope

The arithmetic-control process returned0 after360.595s including shared-lock waiting, not a benchmark duration. CPU saved-tensor audit returned0 in6.560s, verified516 entries, exact producer FP32 metrics and separate complete FP64 metrics, input byte-code reconstruction, import/binding/source hashes, and kept CUDA uninitialized. CPU complete-component reconstruction returned0 in13.921s with CUDA uninitialized. Original baselines and gate limits were retained.

The initial separate GPU BF16-reduction-flag control was **NOT_RUN_SHARED_LOCK_TIMEOUT**: the process waited for another task's shared GPU lock until its600s deadline (launcher124, childSIGTERM/-15), with Torch CUDA uninitialized and without starting the model/GPU phase. Original r1 source, CPU partial, stdout/stderr and real timeout metadata are preserved. This is a lock-admission failure, not a native accuracy/build failure. No other task was interrupted and the original BF16 baseline was not replaced. Its prepared combined GPU auditor was not executed because no GPU result exists.

A separate new CPU-only revision saves the first-layer full FP64 dot and midpoint distances (2 tensors) before exit; it returns0 in14.083s. Independent CPU audit returns0 in6.311s and checks every differing C position directly from original checkpoint products with elementwise FP64 sums. All270 selected dots match the saved FP64 dot exactly. StockBF16 versus FP64-rounded C differs269 positions; FP32-dot/BF16-stage versus FP64-rounded C differs61. Only56/270 stock-versus-FP32 differing positions are within8 output-FP32 ULPs of their BF16 midpoint. Therefore this is not described as only exact ties or as one proved GPU instruction defect.

Every selected stock BF16 rounding cell is within the conservative generic FP32 summation bound `gamma_1024 * sum(abs(products))` of the independently calculated dot; maximum gap/bound ratio0.00261138 and absolute gap1.63075e-5. This is consistent with accumulation-order/rounding sensitivity; it does not prove the exact stock GPU instruction/reduction path. The BF16 reduction-flag effect remains untested because the lock was occupied. The next hardware experiment should check that flag against the original saved baseline with explicit lock admission; do not merely add component bits or rerun these completed controls.

Latest observed upstream develop160f322f adds immutable acquire reuse in SDPA and grouped WGrad. The broad GEMM comparison changes WGrad only; grouped SwiGLU/API/stream/native build paths remain unchanged against tested baseef85. This is an upstream observation/path comparison, not a rebase or hardware test at160f.

No new generation/KV/HTTP, native regression matrix, NCU, or formal paired model performance is claimed. PR1280 remains an independent correctness draft. Keep vLLM inactive, heartbeatpr229-a3 PAUSED, and future GPU work serialized through /tmp/codex-thor-perf.lock. PR229/A2/PR224 remain independent and completed.

## Preserved Mac evidence

Mac `a3-residual-staging-evidence-20261001/` contains39 files/614971 payload bytes. Every one of38 manifest entries matches its size and SHA256; archive80585 bytes/SHA256 b42295dae25dffabdda9c7c0c2542caa836a2237499d8012ef14156694a77400. Original source/run logs/timeout evidence are preserved. Model/cache/engine/.pt/native-binary payloads are excluded; original516+2 audited tensors remain on Thor. No source revision was promoted and no failed precision policy is presented as full all24 success.

## Subsequent GPU admission attempts

After a brief free-lock probe, the GPU-only r3 attempt returned75 because another task had acquired the lock before admission. A separate r4 source version acquires before heavy imports and waits at most45s; it also returned75, with Torch not imported and no model/GPU computation. Both frozen launch/status/log/source sets are preserved in [BF16_FLAG_ADMISSION.md](BF16_FLAG_ADMISSION.md). The flag effect remains **NOT_RUN_SHARED_LOCK_BUSY_AFTER_INITIAL_TIMEOUT**, no additional native calls or precision pass occurred, and no other task/service was interrupted. Final Mac admission archive13 files/all12 manifest entries verified, SHA256 afbb4cfc06f4ed8dd6b21a3f930b83f91ef40ecb4f0ff6439f755513a3327628. The previous39-file archive remains unchanged.

## Completed flag control (supersedes historical admission-only status)

The r5 GPU control and independent CPU audit both returned0. Turning off BF16 reduced-precision reduction preserves all three original BF16 complete-logit tensors byte-for-byte. Native A3 calls0; original baseline/precision gates unchanged; all24 MXFP8 remains open. See [BF16_FLAG_SUCCESS.md](BF16_FLAG_SUCCESS.md) and its immutable raw evidence. The earlier NOT_RUN statements above refer to the preserved r1/r3/r4 attempts.
