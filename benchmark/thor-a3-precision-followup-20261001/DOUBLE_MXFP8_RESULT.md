# Double-MXFP8 full-Granite diagnostic — 2026-10-01

The new explicit residual-component recipe still does **not** pass the unchanged full-model gates. It is a different recipe from the original single-component MXFP8 and from the ten earlier failed policies. Two supported variants each fail the short-input oracle/BF16 compatibility prefilter; their long-input native runs pass the 5% BF16 gate but fail the 1% same-recipe oracle gate. No variant is promoted. A separate BF16-stage reference is not a supported native policy.

Runtime is unchanged independent PR1280 head `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744` in Thor `a3sfdlatest4`. The same pinned full Granite checkpoint retains all 1,334,628,352 parameters, 24 layers and E32/top8. Pretrained weights, model hidden/intermediate sizes, original BF16 baseline and both gate limits are unchanged. Actual API/kernel/binding paths and hashes are checked again by the CPU auditor. Binding reuse and previously observed develop `3d1135e` retain their earlier documented scope; no source rebase or new build is claimed.

## Explicit recipe and filtering

Each routed input and gate/up weight is quantized into a high E4M3 component and a separately quantized residual, each with its own E8M0 SF32. Expanded native A uses `[A0,A0,A1,A1]`; expanded B uses `[B0,B1,B0,B1]`. One supported A3 invocation computes the four cross terms in GEMM, then runs its actual fused SwiGLU and consumes actual D/SFD. Original H1024 is retained, while the operator's expanded K is explicitly **4096**. This increases input/weight storage and GEMM work; no speedup is asserted. The saved producer reports maximum weight reconstruction L2 about 1.51e-9 and absolute error 2.98e-8, but component weight tensors were not saved for an independent reconstruction audit.

Two supported native semantics are checked: original BF16 D/down projection/merge, and supported FP16 D decoded into FP32 with FP32 down projection/merge. Both independent oracles compute the same expanded quantized dot products before activation. The FP16 oracle independently stages activation; it does not copy native D/SFD.

Every complete oracle tensor is first compared with the original BF16 tensor using the previously documented triangle-inequality bound. The arithmetic/sky cases have bounds above5%, so their native runs are explicitly NOT_RUN_FIXED_ORACLE_INCOMPATIBLE. This is a declared diagnostic prefilter, not a fallback inside a model request or an unreported native accuracy pass. Only the two 256-token cases pass the necessary prefilter and run 24 native expert layers each. Full-model success still requires every case to pass both actual gates.

The third policy explicitly rounds gate/up and activation through BF16 in a Torch-only reference. The production native API has no equivalent activation-staging semantics; its native runs are NOT_RUN_UNSUPPORTED_BF16_ACTIVATION_STAGE. Its results do not establish native quality or justify weakening the existing support checks.

## Independent CPU FP64 complete-logit results

Case0 is arithmetic, case1 sky and case2 the long note. All input positions and all 49155 vocabulary logits are included.

| Policy | Case / input tokens | Oracle / BF16 L2 | Minimum native / BF16 L2 under 1% oracle gate | Actual native / oracle and native / BF16 L2, or reason not run |
|---|---:|---:|---:|---|
| `double_mxfp8_bf16_d_down` | 0 / 73 | 11.6194% | 10.6228% | NOT_RUN_FIXED_ORACLE_INCOMPATIBLE |
| `double_mxfp8_bf16_d_down` | 1 / 70 | 10.8949% | 9.8972% | NOT_RUN_FIXED_ORACLE_INCOMPATIBLE |
| `double_mxfp8_bf16_d_down` | 2 / 256 | 4.7775% | 3.7788% | 3.2369% / 4.4868% |
| `double_mxfp8_fp16_d_fp32_down_merge` | 0 / 73 | 11.7736% | 10.7750% | NOT_RUN_FIXED_ORACLE_INCOMPATIBLE |
| `double_mxfp8_fp16_d_fp32_down_merge` | 1 / 70 | 10.5035% | 9.5024% | NOT_RUN_FIXED_ORACLE_INCOMPATIBLE |
| `double_mxfp8_fp16_d_fp32_down_merge` | 2 / 256 | 5.1407% | 4.1413% | 4.3563% / 4.9054% |
| `double_mxfp8_bf16_gate_activation_reference_only` | 0 / 73 | 10.2118% | 9.2163% | NOT_RUN_UNSUPPORTED_BF16_ACTIVATION_STAGE |
| `double_mxfp8_bf16_gate_activation_reference_only` | 1 / 70 | 9.9582% | 8.9626% | NOT_RUN_UNSUPPORTED_BF16_ACTIVATION_STAGE |
| `double_mxfp8_bf16_gate_activation_reference_only` | 2 / 256 | 4.5584% | 3.5594% | NOT_RUN_UNSUPPORTED_BF16_ACTIVATION_STAGE |

For the two executed native cases, both producer FP32 metrics and independent FP64 metrics agree: **oracle gate FAIL, BF16 gate PASS**. The gate bound is only necessary; passing that prefilter is not an accuracy pass. The earlier 33/33 incompatibility finding applies to the original recipe and ten earlier policies. These new long-input cases are compatible with the necessary bound but still fail actual native/oracle quality; do not extend the old 33/33 statement to every tested recipe.

The GPU process returns0 in26.523s with **48 actual native calls**, all in the two long-input forwards. The independent CPU process returns0 in3.030s, leaves CUDA uninitialized, verifies all11 saved full-logit tensors, exact producer FP32 metrics, separate FP64 metrics, every prefilter bound/skip reason, actual imported/executed files, and complete shapes/finite values. It verifies both gates against the unchanged BF16 baseline. These are diagnostic durations, not paired inference timings. No new generation/KV/HTTP, regression-matrix or formal performance claim is made.

## Preserved evidence and remaining work

Mac archive `a3-double-mxfp8-evidence-20261001/` contains12 files/63672 payload bytes. Every one of its11 manifest entries matches size and SHA256; archive SHA256 is `5af022346f2d4edd7ac0335a662e8cee8bc58ba5279bdaef092b435d25d3f549`. Models, caches, engines, .pt outputs and native binaries are excluded; the original11 tensors remain on Thor with archive/tensor hashes in the CPU audit. Existing helper harnesses and logger dependencies are preserved in the first precision-follow-up archive and public evidence.

Broader A3 all-layer precision/performance work remains open. The original recipe, ten earlier policies and both new supported double-component policies are not accepted. The explicitly declared earlier layer23-only hybrid remains the only passing tested policy here, with its separately reported limits and no formal model throughput result. Native rounding alone cannot satisfy the incompatible short-case fixed oracles. Any future all-layer precision recipe must pass complete oracle/BF16 compatibility and both actual native gates, then meaningful boundary/reuse checks and frozen paired performance. Do not repeat these completed diagnostics, use the reference-only BF16 stage as a native success, or describe a single long-case BF16 pass as full-model completion.

The existing PR1280 remains an independent correctness draft at `7bfeda6b`. Its earlier five repairs/31 zero-skip checks and PR229/A2/PR224 remain separate. Keep vLLM inactive, heartbeat paused, future GPU work serialized through `/tmp/codex-thor-perf.lock`, and frozen sources untouched.
