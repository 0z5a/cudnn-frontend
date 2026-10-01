# Thor A3: SDPA-internal FP32 recipe rejected

Only existing BF16 SDPA Q/K/V operands (after unchanged BF16 projections/rotary) are exactly promoted to FP32 for attention computation. SDPA output is cast back to BF16 before the unchanged original output projection. Original BF16 router/probabilities/residual/non-attention linears/down/merge and native MXFP8/BF16-stage computation retained. Attention mask values/semantics, causal handling, dropout0 and original helper are preserved. This is a distinct recipe fixed before both oracle/native runs; the attention interface is restored afterward. No full-FP32-state or native BF16 GEMM mode.

Runtime/import/head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Same experimental native MXFP8 dot/FP32 accumulator/BF16 C-SiLU-product kernel SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3** remains outside clean production source and unaccepted. Actual native D is consumed; no native BF16 GEMM, Torch native-output replacement, support bypass or failure fallback.

Same original full Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352parameters/all24/E32/top8/H1024/I512. Original saved BF16 baseline and whole-position/all49155-vocabulary1% native/declared independent oracle and5% native/original BF16 gates retained.

Declared oracle uses that same SDPA-internal FP32 mode, then reconstructed logical operands recastBF16 for BF16 dot/activation/down/merge. Both short fixed oracles are incompatible with simultaneous1%/5% gates; no short native execution. The compatible long input executesall24 native layers but still fails1% oracle. Whole recipe rejected; no quality or performance fix.

|Tokens|Oracle / original BF16 L2 (independent FP64)|Necessary native/BF16 lower bound at1% oracle|Actual native calls|Native / oracle L2|Native / original BF16 L2|
|---:|---:|---:|---:|---:|---:|
|73|15.366186%|14.365438%|0|NOT_RUN|NOT_RUN|
|70|15.279858%|14.280206%|0|NOT_RUN|NOT_RUN|
|256|4.972367%|3.968146%|24|4.207944%|4.478271%|

Actual producer rc0/14.513240682s,24native calls, SDPA-internal FP32 callbacks reference72/native24. Independent CPU audit rc0/3.729417550s/all4saved full logits (3oracles+1native), original baseline, producer FP32/independent FP64 metrics and compatibility lower bounds, clean source/import/fresh binding/experimental kernel/harness/actual attention helper source hashes; CUDA uninitialized/newnativecalls0. Short lower bound>5% means no result within1% ofthat fixed oracle can also be within5% oforiginal BF16. This rejects only this complete recipe, not all possible attention/precision approaches.

23-file archive SHA **b92377f613c8e9511ea0b142cfb2518ef11d5a9185ae4fdf261c07df55225fe1**, all22rawmanifestentries verified on Mac/tracked publicly; exact original attention helper and `.py.txt` source bytes preserved. Models/cache/engines/.pt/native binaries excluded. No independent candidate physical-pack, generation/KV/boundary/reuse or frozen paired performance result. No support bypass/threshold change/smaller model/correction bias. vLLM inactive/shared GPUlock obeyed; no clock/power changes, restoration timers or other-task interruption. PR1280 remainsdraft at741e10ff; full24 A3 quality/performance remainsopen. Do not repeat this completed SDPA-internal FP32 recipe asnew evidence.
