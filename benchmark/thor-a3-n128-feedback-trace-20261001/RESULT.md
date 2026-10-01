# Thor A3: N128 full-model parity and actual feedback traces

The only changed execution geometry is M256/N256 to M256/N128; C/D stay BF16 and original padded capacity and K32 component order are retained. All3 complete native and oracle full-logit outputs are byte-identical to the previous frozen N256 control. No accuracy improvement. Actual native and independent reference feedback traces are captured separately; routes and later inputs are never held fixed.

Runtime head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh native binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Experimental native source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3** remains outside the clean production checkout. Actual MXFP8 tensor-core dot, FP32 accumulation, then BF16 C/SiLU/product rounding; no native BF16 GEMM or Torch replacement of native D.

Same original complete Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`, 1,334,628,352 parameters, all24 layers/E32/top8/H1024/I512. Original BF16 baseline and every-position/all49155-vocabulary 1% native/declared oracle and5% native/original-BF16 gates stay unchanged. The oracle is frozen reconstructed operands recast to BF16, BF16 dot/activation/down/merge. No support bypass, failure fallback, smaller model, threshold change or empirical output correction.

|Tokens|Native / oracle L2 (independent FP64)|Native / original BF16 L2|Native byte-identical to control|Both gates|
|---:|---:|---:|---|---|
|73|10.408587%|10.408587%|True|False|
|70|10.497938%|10.497938%|True|False|
|256|4.337401%|4.313020%|True|False|

Actual producer rc0 in 16.253541998s, 72 native calls (3 complete cases ×24 layers), actual native D consumed throughout. Independent CPU audit rc0 in 4.121219473s; checks original baseline, frozen control result/output hashes, imported clean head, fresh binding, prototype source, FP32 producer metrics and independent FP64 gates. No new GPU calls; CUDA uninitialized.

All582 tensors audited:6 full-logit tensors plus576 per-layer input/route/probability/expert-output tensors (3cases×2independent feedback paths×24layers×4fields). At layer0, input, route indices and probabilities are byte-identical, while expert output differs.

|Tokens|First ordered-route difference (zero-based layer)|First expert-set difference|Layer0 expert-output L2|Layer23 input L2|
|---:|---:|---:|---:|---:|
|73|1|3|0.032693%|7.952510%|
|70|1|3|0.007562%|7.908997%|
|256|1|2|0.015698%|5.873377%|

Ordered top-k permutations and expert-set changes are distinguished. These traces locate feedback divergence, consistent with rounding sensitivity and trajectory amplification; they do not prove that routing alone explains all error or establish a unique kernel defect. CPU audit does not recompute the whole model or the physical A/B pack.

22-file archive SHA **5c94dd527f9fb4400b8254983cc5428e4ee4b067d6fa0ae26f960be41d3ad1e9**, all21 manifest entries verified on Mac. Exact `.py.txt` copies preserve source bytes and original manifest paths. Model/cache/engine/.pt/native binaries excluded. Full tensor payloads remain on Thor with hashes. The existing latest production31 GPU regression passes are separate from these model failures. No experimental feature accepted, generation/KV or formal paired performance measured. PR1280 stays draft; full24 A3 accuracy and performance remain open. vLLM inactive; shared GPU lock obeyed; no timers, clock/power changes or other-task interruption.
