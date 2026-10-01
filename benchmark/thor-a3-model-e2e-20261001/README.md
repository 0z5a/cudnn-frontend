# A3 full-model E2E — 2026-10-01

The complete trained-model source regression passed, but the model recipe failed its preregistered accuracy gates. This is **not** an E2E quality pass or evidence of model speedup. Formal paired performance was not started. The previously measured operator-only gains remain limited to their recorded workloads.

## Actual model, source and environment

The checkpoint is [IBM Granite 3.1 1B-A400M Instruct at the pinned revision](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/tree/0da7a48b0276d500ce5922fd2b33944091fc6c09). All **1,334,628,352 parameters** were loaded: 24 layers, 32 local experts, top-8 routing, hidden size 1024, intermediate size 512 and vocabulary 49155. There was no reduction of layers, experts, hidden size or vocabulary. Loading reported no missing, unexpected, mismatched or erroneous keys.

The full 2,669,283,096-byte checkpoint was verified against the official LFS SHA256 `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`. Tokenizer/config files were checked against official repository blob digests. Download retries and the first HTTP 403 are preserved. Model weights and tensor outputs remain on Thor.

Upstream develop was checked before and after the experiment and remains `a4dcf87ca1b77fbc0ab62642e8cb7136a63baf73`. The compared heads are:

- Control: `83374af96ec3f8e22dd159d278bdf25acab17a4e`.
- Wrapper-memo/shared-stream candidate: `209d9e61c6ca873ccf5b869e83db00eed9f8f966`.

Both sources contain the separate unified GLU, Thor correctness and lean-launch adaptations. Neither is unmodified upstream or the independent correctness PR1280 head. No production source was changed in this E2E round. Existing private native builds were reused, with actual imported Python/native paths and hashes verified. All production Python, native binding, harness and checkpoint hashes were frozen before/after the comparison. The actual grouped GLU kernel source SHA256 is `ac515fb8a81c89e13141fb788f6fd3e48b9722efd452d639a07b66ae938b3cf5`; the tested harness SHA256 is `6c96035a64e3906cd8f222e40ec507d13c2e1a647e287ec14828ea3d465111eb`.

Hardware/runtime: NVIDIA Thor SM110, 20 SMs; PyTorch 2.13.0+cu130, CUDA runtime 13.0, cuDNN 9.20, CuTe DSL 4.7.1, Transformers 5.17.0. TF32 was disabled. GPU work was serialized with `/tmp/codex-thor-perf.lock`. The existing vLLM service stayed inactive; no power/clock settings or restoration timers were changed.

## Integration and completed request coverage

The experimental adapter replaces each real expert gate/up projection with the checked public A3 wrapper using MXFP8 A/B and E8M0 block scales. Attention, router, normalization, residuals, embeddings, LM head and down projection use the original BF16 model. It retains live top-8 routing, sorts/compacts routes, pads live groups to 256 rows, quantizes activations, consumes the wrapper's row-SFD scale output, runs the original down projections and merges weighted expert outputs. These operations belong to model integration, outside compiled execute. No native failure falls back to another implementation.

This is batch-1 eager Python model inference with a real KV cache, **not** an HTTP/vLLM/TE-training test. Default tile `(256,256)`, cluster `(2,1)`, FP32 accumulation, BF16 C/D and scale-vector size 32 were held fixed between sources.

Each source completed three full-logit forwards (73, 70 and 256 input tokens), three 16-token greedy generations, and two full-model requests with **256/1024 input tokens plus 32 generated tokens**. Generation lengths are fixed for comparison and continue past EOS; short displayed answers stop at the first EOS. No long-context-capacity or production-serving claim follows from these tests.

Each process executed **2761 real A3 calls**: 1 operator-oracle call, 72 calls for three full-model forwards, 1152 calls for three 16-token generations and 1536 calls for the two 32-token requests. Every native model forward executed all 24 expert-layer calls. Both children returned 0. The CPU-only independent auditor loaded the saved `.pt` outputs directly and verified **20 tensor entries byte for byte**, including native/reference/stock full logits, generated token IDs and generation/request logits. Input token IDs, quantized weight payloads/scales and all five generated texts agree across sources. The auditor did not initialize CUDA.

## Accuracy and semantic findings

The exact-source comparison does not waive the failed model-quality gates. The independent oracle computes FP32 GEMM/SiLU on the identical dequantized MXFP8 operands, with BF16 activation storage and the same routing/merge code. The original BF16 model is a separate accuracy arm. They are not performance baselines.

The preregistered full-logit relative-L2 limits remain **1% versus the quantized oracle** and **5% versus original BF16**. Both sources have identical failures:

| Full forward | Input tokens | Native vs quantized oracle | Native vs original BF16 | Accuracy gate |
|---|---:|---:|---:|---|
| Arithmetic prompt | 73 | 20.1175% | 24.1755% | Failed |
| Sky prompt | 70 | 21.3416% | 23.8513% | Failed |
| Prefixed note | 256 | 7.0427% | 8.1727% | Failed |

The short arithmetic answer is `2 + 2 = 4`, which contains the expected value but does not follow the requested number-only format. The sky answer is truncated at 16 tokens. The short passcode prompt produces a refusal and does not contain `cobalt`; the two longer note requests do contain `cobalt`. Source parity is exact, but this is not an all-prompts semantic pass. Original-BF16 generations were not run, so the refusal is not attributed to quantization or the wrapper change.

For the first prompt, repeated independent-reference forwards and repeated native forwards are each stable and byte-identical. A same-input, same-route comparison at each of all 24 layers has maximum FFN relative L2 below 0.028%. In separate native/reference full forwards, small differences grow across layers and routing differs by 135 of 584 assignments in the final layer. This is diagnostic evidence, not proof of a single causal defect. A separate FP32 model diagnostic still gives 16.8929% native/oracle full-logit relative L2. Neither diagnostic establishes a passing accuracy recipe.

## Row-SFD integration defect and preserved failures

The initial adapter incorrectly consumed normalized BF16 D without its scale output. That adapter error was fixed by reconstructing D from row SFD before the ordinary down projection. The initial failed preflight and source snapshot are retained. Default-tile preflight then passed the independent FFN oracle for 13-token/all-expert and 1-token/empty-expert cases; the real-weight operator-oracle check also passed exactly.

The new model shape exposed an independent narrow-tile issue. A poisoned-output detector prefilled only the wrapper's output SF buffers with `0xFF`. For 13 tokens, top-8 routes, 32 experts, N1024/K1024, it checked 1664 live row-scale entries:

- `(256,256)`: zero poisoned entries remain; row SFD is written.
- `(256,128)`: **1664/1664** poisoned entries remain, despite finite D and production support acceptance.

The grouped GLU source flushes row/column SFD under `subtile_idx == 6`; that index is not reached for the narrow loop. This was not fixed or worked around in production. Narrow-tile model integration/performance is **NOT_MEASURED**. Earlier C/D/AMAX-only operator checks did not validate readable row-SFD output and cannot establish this model recipe's correctness. The existing narrow FP8-D rejection and default tile policy remain intact.

First strict runs also preserve two harness-only failures (the current Transformers tokenizer returns a BatchEncoding; runtime config contains JSON-incompatible sets). After those fixes, strict run r3 stopped on the actual 20.1175% accuracy failure. A separate diagnostic-completion run finished the full source-regression requests while retaining every failed threshold. Its final status and independent audit are `SOURCE_REGRESSION_PASS_REFERENCE_QUALITY_FAILED`. There are no formal timing processes or throughput estimates after that failure. The single per-request timings in raw metadata are explicitly diagnostic observations.

## Evidence and remaining work

All raw logs, failed attempts, metadata, the runtime model implementation and frozen Python source snapshots are saved on the Mac under `a3-model-e2e-evidence-20261001/`. All **1800 files** passed size/SHA256 checks after transfer. Archive SHA256: `a020fd5f632a8ec2d7adfd7234d8100c735befd75110b1859de5db622f9b6cba`. No model weights, `.pt` files, native binaries, engines or caches were copied. The audit is `results/a3-model-e2e-regression-20261001/independent-audit.json`; transport verification is `A3_MODEL_E2E_MAC_TRANSPORT_20261001.json`.

The requested E2E experiment is complete with the negative quality result preserved. A3 optimization still needs a valid full-model accuracy recipe, a separate row-SFD fix with proper output/scale detectors, and frozen paired end-to-end performance before any model speedup claim. PR1280 stays on its independent correctness-only head and remains draft. No performance PR or default-selection change is introduced here. Existing heartbeat `pr229-a3` remains paused and the chat remains open.
