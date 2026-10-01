# Thor A3: residual-first native dot ordering rejected

Runtime/import/binding head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1** on develop **e2bf967b17dae17a642102198e6d862d559655c9**. Same independently stored experimental native MXFP8 dot/BF16 epilogue source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3**. Production source/PR1280 implementation unchanged. No model quality or performance gain; both new whole-case policies are rejected.

Keep original pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352 parameters/all24/E32/top8/H1024/I512 and original saved BF16 baseline. Whole-logit1% native/declared independent oracle and5% native/original BF16 gates are unchanged. Actual native D is consumed in every layer; no unsupported gate bypass, fallback, Torch substitution or smaller model.

Two globally fixed K32 cross-term orders: `[LL,HL,LH,HH]` and `[LL,LH,HL,HH]`, each applied identically to matched A/B/logical-SF/physical-SF inputs. They reorder exact same four products, not learned parameters or empirical correction biases. Native compute remains MXFP8 tensor-core dot/FP32 accumulation, then explicit BF16 C/SiLU/product stages. The previously declared logical oracle is frozen unchanged: reconstructed values recast to BF16, BF16 dot/activation/down/merge. It matches original BF16 full logits byte-for-byte on73/70tokens;256-token oracle/BF16 L2 remains4.0010227%. Its definition is not changed after native failure.

|Policy|Tokens|Oracle / original BF16 L2|Native / oracle L2|Native / original BF16 L2|Gates|
|---|---:|---:|---:|---:|---|
|residual_terms_then_HH_bf16_stage|73|0.000000%|14.945853%|14.945853%|oracle=False, BF16=False|
|residual_terms_then_HH_bf16_stage|70|0.000000%|14.362749%|14.362749%|oracle=False, BF16=False|
|residual_terms_then_HH_bf16_stage|256|4.001023%|3.992296%|4.650711%|oracle=False, BF16=True|
|reversed_four_cross_terms_bf16_stage|73|0.000000%|14.032262%|14.032262%|oracle=False, BF16=False|
|reversed_four_cross_terms_bf16_stage|70|0.000000%|13.800488%|13.800488%|oracle=False, BF16=False|
|reversed_four_cross_terms_bf16_stage|256|4.001023%|3.724495%|4.076327%|oracle=False, BF16=True|

Actual producer rc0 in20.072492032 s,144 actual native calls (2policies×3completecases×24layers). Independent CPU audit rc0 in4.093550877 s, verifies all12 complete logit tensors, original baseline, producer FP32 metrics and independent FP64 metrics/gates, actual imports/fresh binding/experimental kernel/source hashes. CPU CUDA uninitialized/new native calls0. All six native/oracle comparisons fail1%; all four short native/BF16 comparisons fail5%. Two long BF16 passes do not establish either complete policy. No generation/KV, complete upstream regression matrix or frozen paired performance claim for these rejected candidates.

22-file archive SHA **b3e3c6f3c01d84f77ae84549cef17a687c5c165fa1eb26c9e144b8679bd8f9e4**, all21 manifest entries verified after Mac transport. `.py.txt` copies retain exact frozen source bytes/original manifest paths. Cache/model/engine/native-binary/.pt output files excluded; their provenance hashes remain. Candidate physical input buffers were not captured; full-logit audit does not constitute independent physical-pack proof. Shared GPU lock obeyed before imports; vLLM remains inactive, no timers/clock/power changes or other-task interruption. PR1280 remains draft at741e10ff; full24 A3 accuracy/performance remains open. Do not repeat these two completed order candidates as new evidence.
