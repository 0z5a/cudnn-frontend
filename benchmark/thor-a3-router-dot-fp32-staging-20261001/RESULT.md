# Thor A3: FP32 router-dot-only BF16-staged recipe rejected

Only the router dot is promoted to FP32: exact FP32 promotion of original BF16 hidden values/weights, FP32 top-k logits, then the original BF16 softmax probabilities. Residual/attention/non-router linears/down/merge keep original BF16 math. The distinct recipe is fixed before both reference and native execution. Gate/up remains the existing pure MXFP8 dot/FP32-accumulator/BF16 C-SiLU-product native prototype, M256/N256/K4096, K32 cross-term order; no native BF16 GEMM. All24 original routers restored after execution; production source unchanged. This differs from earlier FP32 model-state or unstaged native policies.

Runtime/import/head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Existing experimental native source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3**; actual native D consumed in every executed native layer. Actual upstream Granite router source is included byte-for-byte in the archive, with original imported path and SHA in producer metadata.

Original full Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352parameters/all24/E32/top8/H1024/I512 retained. The independently declared oracle uses the same FP32 router dot and BF16 probabilities, then reconstructed logical operands recast BF16 for BF16 dot/activation/down/merge. Original saved BF16 baseline and whole-position/all49155-vocabulary 1% oracle/5% BF16 gates unchanged. No output substitution, support bypass, threshold relaxation, smaller model or failure fallback.

|Tokens|Oracle / original BF16 L2 (independent FP64)|Necessary native/BF16 lower bound at1% oracle|Actual native calls|Native / oracle L2|Native / BF16 L2|
|---:|---:|---:|---:|---:|---:|
|73|16.669470%|15.674109%|0|NOT_RUN|NOT_RUN|
|70|16.646782%|15.652311%|0|NOT_RUN|NOT_RUN|
|256|4.952454%|3.951860%|24|3.703461%|4.643035%|

The fixed reference alone makes both short cases mathematically incompatible with simultaneous1%/5% gates: lower bound `max(0, ||r-b||/||b|| -0.01||r||/||b||)` exceeds5%; native short execution is explicitly NOT_RUN. Long input is compatible at this prefilter but its actual24-layer native output fails1% oracle. The complete recipe is rejected; one long BF16 gate pass is not complete success.

Actual producer rc0/14.137372522s,24 actual native calls. Independent CPU audit rc0/2.473037380s, all4 saved full-logit tensors (3oracles+1native), original baseline, FP32 producer and independent FP64 gates/bounds, clean head/import/fresh binding/experimental kernel/harness/router-source hashes. CUDA uninitialized/newnativecalls0. Candidate physical A/B buffers not captured/independently audited. No candidate generation/KV/boundary/reuse or formal frozen paired performance run, accepted feature, model-throughput or near-capacity claim.

23-file archive SHA **942969a7d55eb2df729d5c75998a7329d2f2ebf656bf521c9d55c46038715454**, all22manifest entries verified on Mac. Exact `.py.txt` source copies preserve original bytes/manifest paths. Model/cache/engine/.pt/native binaries remain excluded. Shared GPU lock obeyed, vLLM inactive, no clock/power changes or restoration timers, no other-task interruption. PR1280 stays draft at741e10ff; full24 A3 quality/performance goal remains open. Do not repeat this completed router-dot-only policy.
