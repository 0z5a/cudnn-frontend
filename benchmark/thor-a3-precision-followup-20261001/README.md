# A3 full-Granite precision follow-up — 2026-10-01

The original all-24-layer MXFP8 recipe remains **FAILED**. Two supported FP16-D/FP32-down variants and eight calibrated channel-rescaling variants also fail the unchanged 1% native/quantized-oracle and 5% native/original-BF16 complete-logit gates. No variant is promoted and no performance claim is made.

An independent CPU check identifies a stronger blocker: for all 33 full-logit cases across the original recipe and ten new policies, the fixed quantized oracle is too far from BF16 for any native output to satisfy both gates simultaneously. A native rounding correction alone cannot repair these fixed recipes. A different validated precision recipe is required. This conclusion does not relax either gate or justify the previously tested one-layer hybrid as all24 success.

## Exact source and scope

Both GPU processes run the unchanged dedicated PR1280 head `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`, on develop `ef85ae94d04b0a366ff6446b327e787d9af5eb85`, using private Thor checkout/venv `a3sfdlatest4`. Actual cuDNN API, executed-kernel and native-binding paths and hashes are verified again by the CPU audit. Binding SHA256 is `2edf36cdbdebe785ab48e30086d595c5799d5131ad3036d385771ca5f39763cc`, reused from the previously verified unchanged native inputs; no new build is claimed.

Latest observed develop is `3d1135e83573507b903e439bac28194a2b52e7a4` (SM90 D512 SDPA prefill). Git comparison verifies no intervening changes to GEMM, shared API/stream helpers, Python native sources, headers or native build configuration. This is an upstream observation, not a rebase or test of `3d1135e`. `A3_PRECISION_FOLLOWUP_UPSTREAM_20261001.json` records exact paths and commits.

The same pinned Granite 3.1 1B-A400M Instruct checkpoint is used: revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA256 `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`. All 1,334,628,352 parameters, 24 layers, 32 experts/top8, H1024/I512 and vocabulary 49155 remain; loading reports are empty. No pretrained parameters are edited. All native forwards use A3 in every expert layer with E4M3 A/B, E8M0 SF32, FP32 accumulation, default tile `(256,256)` and cluster `(2,1)`.

Three full-logit inputs are the unchanged arithmetic prompt (73 tokens), sky explanation (70 tokens) and 256-token long note. Every input position and all 49155 vocabulary logits are checked, including the prefix. These new diagnostics do not repeat generation, 1024-token KV requests, HTTP/vLLM serving or the earlier 31-case regression suite. Their earlier results retain their separately reported scope.

## Supported precision variants

The first probe keeps fused D in supported FP16, decodes actual row SFD directly into FP32, and uses an FP32 copy of the original BF16 down projection. `route_bf16` rounds weighted expert contributions to BF16 before the once-only top8 sum; `merge_fp32` keeps contributions and their sum in FP32 before a final BF16 conversion. The independent oracle computes gate/up from the same dequantized MXFP8 inputs/weights and independently stages activation in normalized FP16. It does not read native D or SFD. An auxiliary ideal FP32-activation oracle is also saved. FP32 fused D remains excluded by the production support check for FP8 operands; that check is not weakened.

The second probe applies the algebraic transform `x / s`, `W * s` before MXFP8 quantization. Four separate stock-BF16 calibration prompts record per-input-channel activation maxima, with four visits to each of 24 layers and zero native calls. Calibration prompts concern a compiler, vegetable soup, sorting numbers and a river/bridge/bicycle sentence; none are the evaluation messages. This small calibration set is not a broad model-quality calibration claim.

The experimental channel formula is inspired by the [primary SmoothQuant implementation](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py): `s = activation_amax**alpha / weight_amax**(1-alpha)`, with explicit lower clamps. This application to Granite MXFP8 is an experiment, not an official supported recipe. Schemes are a constant `1.1` and channel alpha 0, 0.5 and 1; each is tested with original BF16-D/down/merge and FP16-D/FP32-down/merge. Each layer shares its input-channel scale across all experts. Routing still uses original hidden states. The inverse pre-quantization weight transform has recorded L2 below 1e-6. Actual model parameters, support gates, launch-stream and AMAX semantics stay unchanged.

## Complete-logit results

Numbers below are independently recomputed in CPU FP64; each cell lists arithmetic, sky and long-note results in that order. Raw producer FP32 numbers are retained exactly in result JSON and reproduced exactly by the audit. FP32/FP64 norm accumulation differs slightly on these large tensors (maximum relative metric difference 0.166%); every pass/fail classification agrees.

| Policy | Native / same-policy oracle relative L2 | Native / original BF16 relative L2 | 1% / 5% gates |
|---|---|---|---|
| `route_bf16` | 20.4431%, 18.9380%, 6.4961% | 20.9365%, 20.1658%, 7.7578% | FAIL / FAIL |
| `merge_fp32` | 18.6555%, 18.4784%, 6.8472% | 22.7657%, 21.7288%, 8.9343% | FAIL / FAIL |
| `bf16_d_down_bf16/scalar_1p1` | 17.2659%, 17.4819%, 6.8762% | 24.0254%, 22.7938%, 8.4151% | FAIL / FAIL |
| `bf16_d_down_bf16/channel_alpha0` | 15.0472%, 15.2185%, 6.5611% | 24.2649%, 23.1782%, 9.0935% | FAIL / FAIL |
| `bf16_d_down_bf16/channel_alpha05` | 17.9126%, 15.5867%, 6.2879% | 27.0443%, 25.3556%, 10.9290% | FAIL / FAIL |
| `bf16_d_down_bf16/channel_alpha1` | 16.9839%, 16.1453%, 7.1706% | 22.6828%, 20.8599%, 8.7718% | FAIL / FAIL |
| `fp16_d_down_merge_fp32/scalar_1p1` | 16.7175%, 15.0318%, 6.4370% | 21.4784%, 20.7814%, 8.0453% | FAIL / FAIL |
| `fp16_d_down_merge_fp32/channel_alpha0` | 17.0919%, 15.8418%, 7.1339% | 23.3242%, 22.2144%, 9.1034% | FAIL / FAIL |
| `fp16_d_down_merge_fp32/channel_alpha05` | 16.1784%, 16.4091%, 6.1950% | 24.5916%, 23.9367%, 12.8372% | FAIL / FAIL |
| `fp16_d_down_merge_fp32/channel_alpha1` | 17.0821%, 18.1387%, 7.0420% | 20.5602%, 20.5530%, 9.7507% | FAIL / FAIL |

Both GPU processes have real exit code 0: precision probe 20.614 s with 168 actual native calls (24 baseline-trace calls plus 144 policy calls); channel-rescaling probe 36.769 s with 576 actual native calls. These are diagnostic process durations, not paired inference timings. Complete execution does not mean precision success.

The corrected CPU auditor returns 0 in 10.815 s with CUDA uninitialized. It audits 210 saved tensor entries from the first probe (192 layer-trace entries and 18 full-logit tensors), 48 full-logit tensors from the second, and 24 calibration vectors. It checks complete shapes, finite values, provenance/file hashes, all metrics, unchanged gate decisions and all 24 layer-trace counts. The first audit attempt returned 1 because its FP64/FP32 norm tolerance was too tight; its exact source and traceback are preserved. This is an audit-harness issue, not a GPU or model-process failure. The repaired auditor reproduces FP32 metrics exactly and uses FP64 independently for classification. Baseline trace full-logit byte parity and inverse transform quality remain producer assertions because their extra full-logit/scale tensors were not separately saved; this limitation is recorded in the audit.

## Why native accuracy alone cannot fix these recipes

Let `R` be the complete quantized-oracle logits, `B` original BF16 logits and `Y` any proposed native output. By the triangle inequality, the unchanged oracle gate implies:

```
||Y-B|| / ||B|| >= ||R-B|| / ||B|| - 0.01 * ||R|| / ||B||
```

If the right side exceeds 0.05, no output can meet both gates for that fixed oracle. This is a necessary-condition calculation on the saved full tensors, independent of the native output and without changing any tolerance.

| Original all24 case | Oracle / BF16 L2 | Minimum native / BF16 L2 if the 1% oracle gate passes |
|---|---:|---:|
| Arithmetic, 73 tokens | 22.4706% | 21.4644% |
| Sky, 70 tokens | 21.6594% | 20.6477% |
| Long note, 256 tokens | 8.4777% | 7.4726% |

The same necessary-condition check fails in **33/33 cases** across the original recipe and all ten new policies. The smallest bound across all 33 is **7.3407%**, above 5%. The CPU-only checker returns 0 in 3.435 s, does not initialize CUDA, and records the SHA256 of all three input tensor archives and producer metadata. It does not prove that the kernel has no remaining bugs, that all possible quantization recipes fail, or that a different recipe cannot work.

## Numerical trajectory

The original all24 arithmetic trace reproduces the prior full-logit bytes according to the producer assertion. Layer 0 sees identical hidden input and routing; its expert-output L2 differs by about 0.00630%. At layer 1, expert-input L2 is about 0.16433% and 1211 logical FP8 bytes differ, although the selected top8 expert sets still match. The first changed expert set is at layer 3. Independent CPU reconstruction confirms every trace metric/count from saved hidden inputs, expert outputs, route indices and probabilities.

This establishes the order in which differences appear. It is consistent with amplification through subsequent model/quantization operations; it is not a causal isolation of a single quantizer, router, attention operation or GEMM accumulation mode. Earlier dot/activation diagnostics and rejected register-offset-cache/NCU experiments are not repeated.

## Evidence, publication and remaining work

The verified Mac archive is `a3-precision-followup-evidence-20261001/`: 36 files, 723050 payload bytes, archive SHA256 `b82b7d5d87a9498a5fd10cc17bc676127d80cc0ab0a7a96c9dc3b4fd59afd5b5`. All 35 manifest entries match size and SHA256. It includes exact harnesses, raw logs, exit metadata, baseline input IDs, CPU audits, frozen API/kernel snapshots and the failed audit attempt. Models, engines, caches, .pt outputs/calibration and native binaries are excluded; original tensor archives remain on Thor. `A3_PRECISION_FOLLOWUP_MAC_TRANSPORT_20261001.json` records transport verification.

The existing independent correctness [PR1280](https://github.com/NVIDIA/cudnn-frontend/pull/1280) remains draft at tested head `7bfeda6b`. Its five production repairs and earlier 31 zero-skip checks remain separate from these unsuccessful model recipe experiments. The earlier explicitly declared layer23-only MXFP8 hybrid remains the only tested passing policy here; it is not all24 MXFP8 success and has no formal model speedup result. PR229/A2/PR224 remain independent.

Broader A3 work stays open: improve a declared all-layer precision recipe on this same full Granite checkpoint, first validate oracle/BF16 compatibility, then validate actual native/oracle and native/BF16 quality, boundary/reuse correctness and frozen paired performance. Do not promote these ten failed candidates or rerun their completed diagnostics. Keep vLLM inactive, serialize future GPU work with `/tmp/codex-thor-perf.lock`, and leave the existing heartbeat paused. No clocks, power settings, restoration timers, new chats or messages to other chats were changed.

## Published evidence usage

This is an evidence-only follow-up above previous evidence commit `957174c5db957711f75c45e135da82d5964cc602`. It does not change production files or become a new tested runtime head. Runtime remains `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`; the earlier correctness repairs/regressions and explicit one-layer hybrid are detailed in `../thor-a3-repair-20261001/`.

`PUBLISHED_FILE_MANIFEST.json` verifies every curated source/metadata/log byte against the verified Mac archive or local metadata. `RAW_ARCHIVE_MANIFEST.json` preserves the original transport inventory and private Thor origins. Logs include both successful GPU launches and the failed/repaired CPU audit attempts. `.pt` archives stay on Thor; SHA256 and per-tensor hashes are in the independent audit output. No model, engine, cache or native binary is published.

Exact tested harnesses live under `frozen-harness/*.py.txt`, and exact runtime source snapshots use `.py.txt` under `frozen-source/`. The suffix protects historical tested bytes from repository formatting hooks. For reproduction, copy harnesses together into a private tools directory and remove only the final `.txt` suffix. The base adapter and both probes require the matching private cuDNN environment and full verified pinned checkpoint. The CPU auditor and gate checker need the saved tensor archives at their recorded private paths and do not initialize CUDA. The failed r1 audit is preserved for provenance; use the successful r2 audit for verification. Prepare equivalent paths privately and choose new output directories; never overwrite frozen evidence.

`run_logged.py.txt` is the exact original launch logger. Commands, cwd, elapsed time and actual return codes are in each launch's `run.json`. GPU harnesses hold `/tmp/codex-thor-perf.lock` throughout execution, preserve inactive vLLM and do not change clocks/power. Formal paired model performance remains NOT_RUN.
