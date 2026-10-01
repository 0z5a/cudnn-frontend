# Thor A3: K16 zero-gap dot grouping rejected

The new candidate retains original logical products, K32 cross-term order, model capacity and frozen oracle; appends16 exact zeros after every16 original component values in **both A and B**, duplicating each SF32 into two SF32 blocks with16 live and16 zero entries. Expanded native K changes4096→8192; M256/N256, BF16 C/D and experimental BF16 epilogue stay unchanged. This changes native summation grouping rather than input values, weights, routing, oracle or acceptance gates. It changes native full logits but fails all three 1% oracle gates; not accepted or promoted.

Runtime/import/head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, develop **e2bf967b17dae17a642102198e6d862d559655c9**, fresh binding SHA **37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a**. Existing experimental native source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3**; clean production source unchanged. Native MXFP8 tensor-core dot/FP32 accumulator, BF16 C/SiLU/product rounding, actual native D consumed throughout. No native BF16 GEMM, support bypass, failure fallback, Torch native-D replacement or empirically biased correction.

Original full pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352parameters/all24/E32/top8/H1024/I512 retained. Independent declared oracle reconstructs logical operands, recasts BF16, then stock BF16 dot/activation/down/merge. Oracle full logits byte-identical to the frozen K4096 control on all3 cases; original saved BF16 baseline and every-position/all49155-vocabulary 1% oracle/5% BF16 gates unchanged.

|Tokens|Native / oracle L2 (independent FP64)|Native / original BF16 L2|Oracle gate|BF16 gate|First route-set difference (zero-based layer)|
|---:|---:|---:|---|---|---:|
|73|10.425790%|10.425790%|False|False|4|
|70|8.291477%|8.291477%|False|False|3|
|256|4.569367%|4.837636%|False|True|2|

Actual producer rc0/16.940568657s,72 native calls (all3completecases×24layers). Independent CPU audit rc0/3.535480491s, all582 saved tensors:6full logits and576 actual-feedback layer input/route/probability/expert-output tensors. CPU checks unchanged original baseline, frozen control result/output hashes, byte-exact oracle parity, actual native metrics and independent FP64 gates, all24 feedback trace metrics/routing counts and clean source/import/binding/prototype provenance; CUDA uninitialized/new nativecalls0.

Native error on the70-token case drops from~10.50% to~8.29% (producer FP32 metrics), while73/256cases do not improve; all3native/oracle gates still fail. Isolated argmax or one-input improvement is not an accepted quality gain. Dynamic traces do not hold inputs/routes fixed after layer0, recompute the full model on CPU, or prove a unique cause. Candidate physical A/B input buffers were not captured; this does not establish independent physical-pack parity. Generation/KV, candidate boundary/reuse matrix and formal frozen paired performance NOT_RUN for this rejected policy. No throughput or near-capacity success claim.

22-file archive SHA **499f84c9a85150bf7cb784c1eded85469dd9cf860b0eab4d69d90d6b9a8693a8**, all21manifest entries verified on Mac; exact `.py.txt` preserves source bytes/original manifest paths. `.pt`/models/cache/engines/native binaries remain excluded. vLLM inactive, shared GPU lock obeyed, no clock/power changes or restoration timers, no other-task interruption. PR1280 stays draft at741e10ff; full24 A3 quality/performance goal remains open. Do not repeat this completed K16-zero-padding candidate as new evidence.
