# Full24 production C-store configuration parity; quality remains failed

All nine complete native model outputs match the unchanged production control byte-for-byte. This closes the full-model configuration-parity gap for the private C-write candidate; the original MXFP8 recipe still fails both unchanged accuracy gates. No public feature or usable model throughput is accepted.

Actual tested head `741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1`, develop base `e2bf967b17dae17a642102198e6d862d559655c9`, source clean. Fresh binding SHA `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a`; original production kernel SHA `5fe58b7118d6cd82d822b82b8f01aa96665414d3e6499d4752cd3cb7f4095b1d`; private kernel SHA `3543ffc3140058646845040a4924278ae2a323d049492d4b5383af0d6ff04482`.

Full pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`, 1,334,628,352 parameters/all24/E32/top8/H1024/I512. Original single-component MXFP8 K1024/SF32, original FP32 activation, native BF16 D/SFD decode/down/route merge and original independent quantized FP32 oracle are retained. No BF16-staged experimental epilogue, double-component K expansion or Torch native replacement is used.

Three configurations are tested on the same73/70/256-token complete inputs: original production emit-C, explicit D-pipeline-acquire emit-C, and explicit D-pipeline-acquire omit-C. All24 expert modules consume actual native D in every native case; exactly24 native calls/case,216 total. The oracle uses zero native calls. Original BF16 baseline saved at prior7bfeda6b remains unchanged and is never replaced.

Producer real rc0/21.967799716 s. All nine native full-logit outputs are byte-identical to the newly executed unchanged production control, including every token position and all49155 vocabulary logits. These are actual latest-head model runs, not relabeled old results.

Independent CPU audit real rc0/3.499796575 s, CUDA uninitialized/new native calls0. All12 saved complete tensors (9native +3oracle) are shape/dtype/SHA checked; source/import/kernel/binding and baseline hashes verified. Producer FP32 metrics reproduce exactly; FP64 metrics below preserve gate classifications.

| Tokens | Native/oracle L2 | Native/original BF16 L2 | 1% / 5% gates |
|---:|---:|---:|---|
| 73 | 19.155049% | 24.998460% | FAIL / FAIL |
| 70 | 18.657393% | 23.595795% | FAIL / FAIL |
| 256 | 7.215138% | 8.274963% | FAIL / FAIL |

The table applies identically to both private protected configurations. Configuration parity is successful; complete-model precision remains failed. No gate/oracle change, empirical bias, reduced capacity/model, failure fallback or layer23-only success claim is made.

The previous private performance evidence measures a different primitive K4096 workload made from double-component frozen first-layer inputs. Its9.3–10.4% captured-wrapper reduction is not an end-to-end timing result for this original K1024 model recipe. [Private boundary/reuse and captured-wrapper evidence](https://github.com/0z5a/cudnn-frontend/blob/3f64d8702cd601769da181c8da0739e10675e758/benchmark/thor-a3-optional-c-store-20261001/RESULT.md).

C remains an unavailable output in the private omit mode and its storage remains allocated. Public optional-output/cache-key API validation, generation/KV and formal frozen paired model performance are NOT_RUN. Production source/API are unchanged; this is not code in PR1280. VLLM stays inactive, shared GPU lock is honored and no clock/power/restoration timer change is made.

Mac archive29files/all28manifest entries size/SHA-verified. Raw driver logs, metadata, executed source/auditor, original production source and provenance are retained; model/cache/engine/PT/native binaries excluded. Published source byte copies use .py.txt and PUBLIC_FILES.json maps their exact hashes.
