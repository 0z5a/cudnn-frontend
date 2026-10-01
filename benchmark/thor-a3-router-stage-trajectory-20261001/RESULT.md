# Thor A3: actual router, attention and residual trajectory

Passive hooks capture actual full32 router logits and six layer/attention/residual stages atall24 layers, without replacing computation. All3 native AND independent oracle full-logit tensors remain byte-identical to the frozen uninstrumented K32/M256/N256 control. Native full quality still fails. This is new instrumentation and causal localization evidence, not a new quality recipe or performance result.

Runtime/import/head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Same experimental native MXFP8 dot/FP32 accumulator/BF16 C-SiLU-product kernel SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3** remains outside clean production source and unaccepted. Actual native D is consumed; no native BF16 GEMM, Torch native-output replacement, support bypass or failure fallback.

Same original full Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352parameters/all24/E32/top8/H1024/I512. Original saved BF16 baseline and whole-position/all49155-vocabulary1% native/declared independent oracle and5% native/original BF16 gates retained.

Actual producer rc0/17.318496141s,72 native calls (all3completecases×24layers). Independent CPU audit rc0/8.567235558s, all1590saved tensor entries:6full logits,576expert x/index/prob/y and1008router-logit/layer-stage captures. CPU verifies metrics and FP64 gates, exact reference/native full-logit parity to frozen control, actual top8 selected scores, score-gap perturbation bounds, strict reversal versus one/both-sided ties, probabilities aligned by expert ID, layer chaining, residual formulas, clean source/import/binding/prototype and actual Transformers source hashes. CUDA uninitialized/newnativecalls0. Original decoder/router source is archived byte-for-byte.

All144 captured attention/expert residual sums (24layers×2branches×3cases) match CPU reconstruction byte-for-byte using original0.22 multiplier and explicit BF16 multiplication/addition stages. Layer0 incoming hidden state, attention, normalized expert input, router logits/indices/probabilities match exactly, while actual expert output and outgoing residual already differ. Attention output differences appear atlayer1, before expert-set divergence.

|Tokens|Layer0 outgoing residual L2|Layer1 attention output L2|First expert-set change (zero-based layer)|Strict reversed pairs atfirst set change|One-sided tied pairs there|
|---:|---:|---:|---:|---:|---:|
|73|0.038949%|0.144012%|3|1|0|
|70|0.009307%|0.105667%|3|1|0|
|256|0.021914%|0.149060%|2|3|1|

|Tokens|All24 strict reversed expert pairs|Both-sided tied pairs|One-sided tied pairs|
|---:|---:|---:|---:|
|73|99|0|13|
|70|93|0|12|
|256|248|0|39|

Short cases first differ at token34/layer3: original selected expert12 exceeds unselected expert1 by0.01171875, whereas the native trajectory ranks expert1 over12 by0.00390625. These are strict score reversals, not arbitrary choices between equal boundary scores. Long case first differs atlayer2 (four tokens,3strict pairs/1one-sided tie). Dynamic trajectories include changed inputs afterlayer0; they do not freeze routing, independently recompute all model operators, or prove a unique cause. The attention-stage growth is localization evidence, not proof that the attention implementation is incorrect. Physical candidate A/B packs are not captured/audited here.

23-file archive SHA **6a35e96cced631de1bf10b6c8e182984d0e14e8833628e65e4a7f5d8da887a6b**, all22rawmanifest entries verified on Mac/tracked inpublic evidence. Exact `.py.txt` preserves source bytes/original manifest paths. Model/cache/engine/.pt/native binaries excluded. No new feature, generation/KV, candidate boundary/reuse or frozen paired performance claim. Latest31production GPUpasses are separate. vLLM inactive/shared GPU lock obeyed; no clock/power changes, restoration timers or other-task interruption. PR1280 stays draft; full24 A3 quality/performance remains open.
