# A3 SwiGLU repair and full Granite validation — 2026-10-01

The independent kernel correctness repair passes on latest develop. Full Granite inference passes the original logit gates with an explicit experimental recipe that uses MXFP8 only in layer 23 and keeps layers 0–22 in original BF16. The original all-24-layer MXFP8 recipe still fails. Formal model performance was not measured.

## Source and native binding

The final PR1280 head is `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`, rebased onto develop `ef85ae94d04b0a366ff6446b327e787d9af5eb85`. Its code remains the dedicated SwiGLU implementation; the experimental unified/lean/wrapper stack is separate at tested head `ad33459c19969eff5170e32d4319ff27183b8796` on develop `ad741313`.

Fresh private native builds returned 0 at dedicated `052f88de` (213.502 s), unified `ad33459c` (225.441 s), and dedicated `5227f1e0` (216.392 s). The final upstream update changed only SM107 SDPA backward sources/tests. All 113 native build inputs match byte-for-byte between `5227f1e0` and final `7bfeda6b`; the final private environment therefore reuses that verified binding, copied into its own checkout with the same SHA256. This is recorded as binding reuse, not a new build. The final 31-case suite and both complete model recipes were rerun at `7bfeda6b` with verified Python, binding and executed-kernel paths.

Thor paths are `/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928/a3sfdlatest4` and `venvs/a3sfdlatest4`. The environment is Thor SM110/20 SMs, CUDA Toolkit 13.2, PyTorch 2.13.0+cu130, cuDNN 9.20, CuTe DSL 4.7.1 and Transformers 5.17.0. Runtime source and native binding hashes are in the import records and native provenance JSON.

## Correctness repairs

- SM110 float maxima use five shuffle/fmax stages instead of the assembly-rejected `redux.f32`; NaN propagation and the existing reduction on other compilation targets are retained.
- AMAX is a fresh invocation output initialized to `-inf`, while the compiled plan stays cached. A later call cannot overwrite the earlier returned AMAX or inherit its maximum. Empty experts keep the reduction identity.
- The accepted narrow `(256,128)` tile now flushes row/column scale outputs at its final GLU subtile and derives their tile geometry from the actual 32-column epilogue atoms. The previous hard-coded final index and width left live scale outputs unwritten. Wide-tile geometry remains independent of input scale-vector size.
- Output allocations and the real AMAX fill run on the requested launch stream through the existing common stream helper. A bounded delayed ambient-stream probe reproduces actual `-inf` AMAX corruption before the repair, including a cached plan launched on the default stream from a different ambient stream.
- The column-reference conversion temporary uses the actual output dtype, preventing the old uint8-backed reinterpretation from exceeding its allocation.

The final pytest summary is **31 passed in 58.98 s**, zero skips, real exit code 0. Coverage is 11 poisoned-scale/layout/independent-reference/reuse/graph cases, four cold/warm launch-stream cases, one AMAX ownership case, five canonical/legacy cases and ten explicitly selected original FP4/FP8 matrix cases. The matrix covers vector modes, discrete column scales, one/two-CTA valid configurations, SF16/SF32 and relevant output types. It is representative coverage, not the entire upstream matrix.

The 11 scale detectors fail on frozen control `83374af9`, with real pytest/conftest/worker isolation and exit code 1. That old checkout's unrelated preserved stream test modification is disclosed; its production files match its Git head. The four stream detectors fail on `052f88de`: two fail allocation-stream assertions and two observe actual nonempty AMAX values overwritten with `-inf`. They pass on the unified stack and the final independent PR head. Guard storage, changed alpha/scales, previous-output preservation and explicit-stream restoration are checked. Other GPU architectures have not been hardware-tested.

## Granite model contract and adapter repair

The model remains `ibm-granite/granite-3.1-1b-a400m-instruct`, revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, with verified checkpoint SHA256 `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`. All 1,334,628,352 parameters, 24 layers, 32 experts, top8 routing, H1024/I512 and vocabulary 49155 are retained. Loading has no missing/unexpected/mismatched/error entries.

The installed stock expert implementation is `grouped_mm`. It restores original top-k order and sums BF16 expert contributions once with FP32 accumulation. The old experimental adapter instead used repeated BF16 `index_add_`. The corrected adapter follows the installed stock merge contract. With quantization disabled, it reproduces all three original BF16 full-logit tensors byte-for-byte. This adapter repair is experimental validation code, separate from the frontend PR.

The MXFP8 boundary uses E4M3 A/B, E8M0 scale vectors of 32, FP32 accumulation and BF16 fused D reconstructed from the actual row SFD. Original BF16 down projections and top-k weights remain. The oracle independently computes gate/up dot products and SiLU from the same dequantized A/B. Sorting, padding, quantization, real native execution, down projection, weighted merge and KV-cache inference all occur in the model path.

## Original all-24-layer MXFP8 outcome

Five frozen source/tile runs each return 0 and execute **2761 real native calls**, with 24 calls per native model forward. The CPU-only independent audit verifies all **20 saved tensor entries per run** byte-for-byte across dedicated `052f88de`, unified `ad33459c` wide/narrow, dedicated `5227f1e0` and final dedicated `7bfeda6b`. Three full-logit forwards, three 16-step greedy generations and 256/1024-token prefill plus 32-step KV-cache requests complete. Their failed quality gates remain failed:

| Input tokens | Native / same-MXFP8 oracle L2 | Native / original BF16 L2 |
|---:|---:|---:|
| 73 | 19.1557% | 25.0032% |
| 70 | 18.6575% | 23.5998% |
| 256 | 7.2216% | 8.2849% |

The original limits remain 1% and 5%, applied to every input position and vocabulary logit. Source/tile parity does not establish recipe accuracy. Single-request durations in these diagnostic logs are not paired performance evidence.

Diagnostics show same-input native FP32 GEMM differences around `1e-7` relative L2 and few BF16 midpoint changes. Full-model routing/numerical effects amplify them. Removing D normalization produces byte-identical native full logits. FP16 D, Torch activation after native C, and activation-staging experiments do not pass the gates. Token errors occur throughout the inputs; truncating a prefix would not explain the failure. These diagnostics are retained and were not promoted into production workarounds.

## Explicit layer-23 mixed precision outcome

A declared layer-precision ablation finds that the last-layer-only policy passes the three initial cases; last4/8/12, middle12/16 and even12 policies fail. The accepted experimental policy has **one native MXFP8 expert layer (index 23), 23 original BF16 expert layers**, and no failure-triggered fallback. It constructs quantized weights only for that selected layer.

This recipe then passes ten complete full-logit cases: the three original cases, six additional prompts not used in the initial policy selection, and a 1024-token case. It also completes three 16-step generations and two 256/1024-token plus 32-step KV-cache requests in stock BF16, independent hybrid oracle and actual native hybrid modes. Each native run executes **122 actual A3 calls**, one per native model forward.

| Checked tensor scope | Maximum native / hybrid oracle L2 | Maximum native / original BF16 L2 |
|---|---:|---:|
| Ten complete input-logit tensors | 0.03138% | 4.69770% |
| Three full-vocabulary generation-logit traces | 0.01920% | 3.87000% |
| Two full-vocabulary KV-request-logit traces | 0.00641% | 2.74643% |

All five source/tile runs pass the unchanged 1%/5% gates. The independent CPU audit verifies **60 saved tensor entries per run** byte-for-byte across all five runs. All three backends generate the same token IDs for all five requests, so their decode-logit comparisons also use the same input histories. Arithmetic contains 4; both long notes return cobalt; the sky text matches original BF16. The short passcode prompt is refused identically by stock BF16, oracle and native. That model behavior was not repaired and is not attributed to A3.

This is batch-1 eager inference with real KV cache and fixed decode-step counts, including steps after EOS. It does not validate all-24-layer MXFP8 accuracy, HTTP/vLLM serving, TE training, broader-model quality, or an end-to-end speedup. Formal paired performance remains **NOT_RUN**.

## Evidence and final state

Full logs, failed attempts, metadata, JUnit output, executed-kernel import records, frozen Python/runtime sources and independent audits are saved on the Mac under `a3-repair-evidence-20261001/` and `a3-repair-evidence-latest-20261001/`. Their transport records verify archive SHA256, file size and every manifest entry. Models, engines, caches, `.pt` outputs, native binaries and Git object packs are excluded from these evidence archives. Checkpoints and saved tensor outputs remain on Thor.

Earlier narrow-tile timings only checked C/D/AMAX and did not establish valid SFD. They cannot support a usable narrow-plan/model performance claim. The repaired scale-writing and stream-ordering heads have not been formally timed. Earlier default-tile wrapper results retain their separately disclosed tested heads and scope.

PR1280 is updated as an independent correctness draft. PR229/A2 remain independent. All jobs from this repair round have ended, the shared GPU lock is released, vLLM is inactive, and heartbeat `pr229-a3` remains paused. No service-restoration timer or new chat was created. The remaining A3 work is a validated full-24-layer precision recipe and, after accuracy passes, frozen paired model performance.

## Published evidence usage

This directory is an evidence-only commit above the tested runtime head `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`. No production runtime source changes are included in the evidence commit. `PUBLISHED_FILE_MANIFEST.json` hashes each curated file and verifies archived copies against the Mac archive manifest. Full raw logs, setup/harness failures and frozen source trees remain in the verified Mac archives; this public subset is not their replacement.

The tested Python payloads are preserved byte-for-byte under `frozen-harness/*.py.txt`, with only the filename suffix changed. They are historical evidence, not maintained repository Python modules. For reproduction, copy them into a private `tools/` directory and remove the final `.txt` suffix together, preserving bytes, before running them. The model adapter/hybrid harness hashes exactly match the hashes in the final run metadata. They deliberately retain the actual Thor paths, pinned checkpoint manifest and source assertions; prepare equivalent private checkouts/environments and the verified full checkpoint before adapting paths. Do not overwrite frozen result directories.

`results/a3-latest4-correctness-tests-20261001/run.json` records the actual pytest command, working directory and real return code; the adjacent import records include the executed kernel and native call count. Final full24 and hybrid launch records show the actual model commands. `prepare_a3_latest4_20261001.py.txt` and the native provenance JSON record the binding reuse, with all 113 build inputs checked. CPU auditors consume the saved `.pt` tensors retained on Thor; tensor hashes and complete audit outcomes are published, but the `.pt` payloads are excluded.

No clocks or power settings are changed. GPU reproduction must serialize through `/tmp/codex-thor-perf.lock` and preserve inactive vLLM. The historical launcher and native-build log runner are environment infrastructure outside the frontend repository; their actual command/return-code metadata is included. No paired timing was run at the repaired heads.

Raw failed-run JUnit and stdout retain pytest traceback whitespace intentionally; they match the verified archive bytes. Formatting hooks have no applicable source files in this evidence-only commit. Production-source hooks were run separately on the actual PR diff.
