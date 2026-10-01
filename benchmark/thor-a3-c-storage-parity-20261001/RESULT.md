# Thor A3: full-model C storage parity

The only changed native configuration is C output storage BF16 to FP32 at M256/N256. C is never used as model input. All3 complete native and oracle full-logit outputs are byte-identical to the previous frozen BF16-C-storage K32 control. This rules out a C-storage-dependent model output change for these three cases; accuracy remains failed.

Runtime head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh native binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Experimental native source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3** remains outside the clean production checkout. Actual MXFP8 tensor-core dot, FP32 accumulation, then BF16 C/SiLU/product rounding; no native BF16 GEMM or Torch replacement of native D.

Same original complete Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`, 1,334,628,352 parameters, all24 layers/E32/top8/H1024/I512. Original BF16 baseline and every-position/all49155-vocabulary 1% native/declared oracle and5% native/original-BF16 gates stay unchanged. The oracle is frozen reconstructed operands recast to BF16, BF16 dot/activation/down/merge. No support bypass, failure fallback, smaller model, threshold change or empirical output correction.

|Tokens|Native / oracle L2 (independent FP64)|Native / original BF16 L2|Native byte-identical to control|Both gates|
|---:|---:|---:|---|---|
|73|10.408587%|10.408587%|True|False|
|70|10.497938%|10.497938%|True|False|
|256|4.337401%|4.313020%|True|False|

Actual producer rc0 in 14.963203192s, 72 native calls (3 complete cases ×24 layers), actual native D consumed throughout. Independent CPU audit rc0 in 3.693099882s; checks original baseline, frozen control result/output hashes, imported clean head, fresh binding, prototype source, FP32 producer metrics and independent FP64 gates. No new GPU calls; CUDA uninitialized.

All6 saved full-logit tensors independently audited; all3 oracles and all3 native logits byte-identical to the prior control. Prior control is not rerun in this diagnostic; exact frozen runtime/prototype/binding and control file hashes are verified.

22-file archive SHA **88ab0f3b4a93a04a6bbb69b38d26b4a98c909690ad4983e1f8a64fc9d6691c32**, all21 manifest entries verified on Mac. Exact `.py.txt` copies preserve source bytes and original manifest paths. Model/cache/engine/.pt/native binaries excluded. Full tensor payloads remain on Thor with hashes. The existing latest production31 GPU regression passes are separate from these model failures. No experimental feature accepted, generation/KV or formal paired performance measured. PR1280 stays draft; full24 A3 accuracy and performance remain open. vLLM inactive; shared GPU lock obeyed; no timers, clock/power changes or other-task interruption.
