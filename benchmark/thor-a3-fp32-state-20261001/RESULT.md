# Full Granite FP32 state and router precision diagnosis

All five tested precision policies fail the complete three-case requirement. One policy, `fp32_non_quantized_math`, passes both original gates on the 256-token input (native/oracle about0.6281%, native/original BF16 about4.3503%). Its two short-input oracles differ from original BF16 by about17.04% and14.71%, so this is a single-case success, not a complete all24 recipe or E2E repair. No candidate is promoted and PR1280 remains draft.

The same full pinned Granite checkpoint retains all1,334,628,352 original parameters,24 layers,E32/top8,H1024/I512. The original saved BF16 baseline and unchanged1% native/same-declared-oracle and5% native/original-BF16 gates remain. All logit metrics cover every position and all49155 vocabulary entries. These are numerical diagnostics, not a formal performance test.

The first group tests FP32 router dot/probabilities, FP32 residuals with ordinary BF16 linears, and FP32 non-quantized arithmetic. The second group retains the original BF16 router input/weight dot, FP32 top-k/softmax and BF16 selected probabilities, isolating that change in the latter two policies. FP32 non-quantized parameters are exact promotions of original BF16 checkpoint values; no training or weight edits. Original embeddings are scaled in BF16 before the first decoder layer promotes hidden state. Native and oracle use the same declared policies.

All policies use three MXFP8 components for FP32 hidden inputs, two for original gate/up weights, and six cross terms at expandedK6144. This increases work/storage; no throughput benefit is claimed. Original H1024 is unchanged. Actual native fused FP16 D is decoded to FP32 and consumed by the original down weights promoted to FP32 with one FP32 route merge. No support relaxation, native-output replacement, failure fallback, capacity reduction, stream/AMAX change or original-baseline replacement occurs.

For each policy, three original-expert precision controls and three independent quantized-oracle full forwards run. All ten short-input oracle/BF16 necessary bounds exceed5%; their native arms are explicitly NOT_RUN_FIXED_ORACLE_INCOMPATIBLE. All five256-token native arms actually execute24 layers each: total120 native A3 calls. All five pass the5% BF16 gate; four fail the1% oracle gate. No policy passes all3 inputs.

| Policy | Oracle/BF16 L2 at73 tokens | Oracle/BF16 L2 at70 tokens | Native/oracle L2 at256 tokens | Native/BF16 L2 at256 tokens |
|---|---:|---:|---:|---:|
| `router_fp32` | 14.475773% | 12.935170% | 4.212313% | 4.460678% |
| `fp32_residual_bf16_linears` | 15.578304% | 14.861443% | 3.038305% | 4.490704% |
| `fp32_non_quantized_math` | 17.041825% | 14.713520% | 0.627827% | 4.348896% |
| `fp32_residual_bf16_linears_bf16_router` | 12.645984% | 13.248345% | 2.608513% | 3.996831% |
| `fp32_non_quantized_math_bf16_router` | 13.851168% | 12.072830% | 1.716574% | 4.430509% |

Both GPU processes return0: first571.136102s including a verified shared-lock wait, second38.386286s. Independent CPU audits return0 in4.621233s and3.889881s, CUDA uninitialized, checking21+14 saved tensors. They recompute exact producer FP32 metrics, independent FP64 complete-logit norms/bounds/gate classifications, shapes, skips, counts, baseline SHA and current source/import/binding/harness/launcher provenance. Complete component weights/packed inputs were not saved for a separate physical-pack audit; checkpoint promotion and D consumption are producer assertions/source evidence.

Runtime/source is frozen7bfeda6b on testedbaseef85, source remains clean, vLLM inactive. All GPU work used the shared lock; no other job was interrupted. No new native source repair is accepted. Generation/KV/HTTP, broader kernel boundary/reuse cases and frozen paired model performance are NOT_RUN for these policies, because full accuracy has not passed.

Raw logs/metadata and exact source bytes are transported toMac:31 files/all30 manifest entries, SHA256 `0d5011fbbfc857a127c00b96a78c648a1661683b9b9993d3ecf0ac30aaaec823` (38058 bytes). Model/cache/engine/.pt/native binaries stay onThor. Unexecuted initial45s-admission source is included separately from the actually executed queued source; only the queued source is the producer identified by the run/hash.

## CPU accumulation-order hypothesis control

A separate CPU-only probe returns0 in7.224768s, CUDA uninitialized, native calls0. It uses the original first-layer stock trajectory/checkpoint at all270 differing BF16 C coordinates and512 deterministic unconditioned random coordinates. Every sampled BF16 product is exactly representable in FP32, and independent elementwise FP64 sums exactly match the stored FP64 dot. Fifteen recorded dot results cover the saved GPU FP32 dot, FP64 rounded once into FP32, serial forward/reverse, pairwise tree and striped4/8/16/32/64 with serial/pairwise merging.

The best simulated order still differs from stock BF16 on234/270 selected positions; all512 random controls match BF16 rounding for all orders. No candidate reproduces the selected stock outputs, so these CPU orders are not an identified stock GPU reduction or an accepted correction. This conditioned first-layer diagnosis does not establish whole-model behavior or an ISA defect. Exact raw metadata/run/logs/source are included under `raw-dot-orders/`; five downloaded/source files have independently checked remote SHA256. GPU pack/native boundary/performance remains outside this CPU scope.
