# Thor A3: latest-head native rounding controls and pinned Granite generation

Production correctness PR1280 is tested and pushed at `a24a7401bb7d26d42d88d98b316f634ef487301c`, integrating upstream develop `29a06d9cdc8843e33c5e374846be3bb0560a0711`. The latest run has an actual outer rc0, 31 GPU test passes, three complete-forward numerical passes and five cached-generation numerical passes. A separate private postfixed-bound integer control reduces complete 256/1024-token request time by 3.114/3.477 times against the earlier serial-bound integer control. Both controls use the same MXFP8 main and encoded-input integer repair; this is not a stock BF16 or default K1024 speedup. The improved control remains prohibitively costly, the original default recipe remains failed, and the 83-token application answer check still fails identically in stock, oracle and native. The A3 optimization goal remains open and PR1280 remains draft.

## Production correctness source and actual validation

Thor SM110, 20 SMs; Toolkit13.2; PyTorch2.13.0+cu130; cuDNN9.20; CuTeDSL4.7.1; Transformers5.17.0. Separate frozen checkout/venv `a3sfdlatest9018`; older741/5f checkouts remain unchanged. All six A3 code/test paths and correctness-patch bytes remain identical while upstream native headers changed, including `include/cudnn_frontend/node/sdpa_fp8_bwd.h`.

Fresh native build actual rc0 /178.666266712 s; preparation logger rc0 /184.564773525 s. Actual binding SHA256 `f8e1555dcba3cd56f4e1f27736d67431b164308c7252b12a86626c1a7f286563`; exact source/API/binding/executed-kernel paths independently checked. Production tests: 31 passed in59.51 s, zero errors/failures/skips, pytest rc0 /63.097036982 s, corrected outer logger rc0 /63.301579985 s, worker61 actual native calls/coordinator0. Covers 11 scale/layout/poison/guard/reuse/graph checks, four stream checks, one AMAX ownership check, five canonical/legacy cases and ten representative original FP4/FP8 matrix cases. This is not the whole upstream matrix or hardware coverage of other architectures. PR stays draft; no performance benefit is attributed to production correctness changes.

Those fresh-build/test numbers are historical ce0 measurements. At current a24, the next upstream commit changes nine Python SDPA-backward/docs/test paths; all114 tracked native inputs and all six A3 code/test paths match ce0 byte-for-byte. The new isolated checkout/venv reuses the above native binding only after checking all114 hashes. Preparation actual rc0 /5.229112465 s; no fresh native compile at a24 is claimed. Latest actual test summary is **31 passed in59.80 s**, pytest rc0 /63.554816203 s, zero failures/errors/skips, worker61/coordinator0. The frozen latest pipeline logger exits0 /609.849433092 s after tests, forwards, generation and paired timing. Exact imports, source head, API/kernel paths and binding digest are recorded in its evidence.

Intermediate5f build/tests also passed their inner31 checks but the old outer driver returned1 for an invalid coordinator-positive-count assertion. Independent CPU finalization rc0 verified those31 actual results without repeating GPU work. Its failed outer source/logs are preserved rather than relabelled rc0. Current ce0 uses a corrected driver and has an actual outer0.

## Private actual native control, original capacity and gates

Pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`; original nine checkpoint-file hashes unchanged; weights SHA256 `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`. Retains1,334,628,352 parameters, all24 layers, E32/top8/H1024/I512, vocabulary49155. Original whole-logit thresholds remain1% native/fixed oracle and5% native/original BF16. Original baseline7b (`a997dac0...`) and frozen oracle741 (`96dcdce4...`) are never replaced.

Main dot executes the frozen four-component K32-block-major K4096 MXFP8 Tensor Core prototype (`68d7c926...`), exposing actual FP32 C. A separate CUDA integer control retains C when its input-derived enclosure certifies the BF16 rounding cell; otherwise it reconstructs logical BF16 words solely from E4M3/UE8M0 bytes and emulates the predeclared25-bit raw-exponent/RTZ K16 arithmetic. A native kernel emits BF16-rounded C/SiLU/product, compact plain D and fresh AMAX; all24 layers consume that actual D. Native gate/up executes no BF16 Tensor Core GEMM or Torch gate/up dot and receives no original BF16/oracle execution tensors. Original BF16 down projection remains unchanged. The arithmetic model matches observed Thor data; this does not establish a universal ISA contract.

The r3 native control build has actual rc0 /28.281621812 s, binary SHA256 `bfb40e36c3dd692004c9beee6d47b6f16f2f13d7f50707738aed2810ed7c2d0b`. Actual boundary tests were at5f, not relabelled ce0: E1/N1024/36 routes, changed inputs/37, E31/N128/ragged4 and E33/N256/ragged18. All79,872 C-BF16, D and AMAX outputs match declared references byte-for-byte; actual4 main/15 repair/28 postprocess calls. Cold/warm explicit-stream checks require side completion before delayed default completion after priming only Torch allocations; cover ownership, zero live/empty AMAX, eight invalid int64 routes, nonfinite C/sticky-NaN AMAX and five static declines. Independent CPU audit rc0 /4.184318056 s verifies physical codes/scales, all74,749 repaired FP32 words with independent integer arithmetic,5,123 retained actual native C words, activation/AMAX and64 fresh scalar coordinates.

Full24 current-ce0 control logger rc0 /53.501960768 s; independent CPU audit rc0 /3.543094080 s checks all12 saved tensors, original token/oracle/baseline hashes, metrics, gates, counts, source/binding and flags.

| Original input | Native/frozen oracle | Native/original BF16, independent FP64 L2 | Repair fraction | Result |
|---|---:|---:|---:|---|
|73 tokens|byte-exact,0|byte-exact,0|98.0488%|both gates pass|
|70 tokens|byte-exact,0|byte-exact,0|98.0792%|both gates pass|
|256 tokens|byte-exact,0|4.0010227%|97.7107%|both gates pass|

Actual72 MXFP8 main/72 integer repair/144 postprocess launches, zero invalid flags. Original default K1024 single-component production recipe remains a separately documented failed recipe; it is not declared fixed by this K4096 private control.

## Input-date trap, preserved failures and reproducibility

The pinned chat template injects the current date. Regenerating prompts after midnight changed exactly one token at[0,23],35→36 (October1→2), while comparing prior logits. The failed full-model attempt stopped before its native stage and preserved the new oracle and IDs. CPU input audit rc0 /8.421583755 s verifies all nine original model-file sizes/SHA256, the original five ID tensors and the single date-token change. Corrected g4 consumes original saved IDs, whose JSON SHA256 is `0983f2e3e173eb5d6dd1fdb286b1bd96b513bff54b1f1a2724cc76bdc82122dc`, and still requires newly executed oracle bytes to equal frozen741 bytes. No gate/oracle/checkpoint adjustment.

Preserved failures also include missing `math_constants.h` in the initial native compile, three lock-busy rc75 admissions before Torch, an invalid short cold-stream detector despite matching first-case values, and an independent decoder audit that initially lost negative zero. The corrected sign-after-float decoder verifies the same saved GPU evidence. Prior CPU postfixed enclosures prove conservative bounds but reject nearly every certificate; they alone are not performance evidence.

## Actual greedy generation and KV reuse

At ce0, five original73/70/83/256/1024-token inputs each run32 greedy tokens with real retained KV cache in independent oracle/native/original-stock branches, plus original-stock teacher forcing at exactly the oracle continuation for same-context numeric comparison. All five native/oracle logits and tokens are byte-exact; all five native/original-stock greedy tokens are also byte-exact. Native counts3840 main/3840 repair/7680 postprocess. Cache objects are reused for31 incremental calls; final lengths104/101/114/287/1055. Producer logger rc0 /193.298865365 s; independent CPU audit rc0 /5.941168619 s verifies all50 saved tensors, every greedy argmax/decoded answer/flag/count/context/hash, gates and KV metadata, CUDA uninitialized/new native0.

|Input|Native/oracle L2|Native/BF16 same-context independent FP64 L2|Expected answer|
|---|---:|---:|---|
|math73|0|0|`2 + 2 = 4`, pass|
|sky70|0|0|Rayleigh scattering explanation, nonempty|
|passcode83|0|0|fails `cobalt`: all three implementations give the same refusal|
|prefix256|0|2.6817428%|`The passcode is cobalt.`, pass|
|prefix1024|0|3.3239166%|`The passcode is cobalt.`, pass|

The producer's overall application flag remains false for83; logger/auditor rc0 means the run/evidence was completed, not that every application answer passed. Formal paired whole-model timing was not started because that predeclared application check failed. Single unpaired generation observations already show prohibitive prototype cost (native prefill ~21.5 s at256 and85.4 s at1024, versus stock ~0.061/0.134 s); these include diagnostic flag reductions and are not a warm paired performance or speedup claim. Neither model refusal nor these timings are hidden or used to replace the original forward baseline.

## Postfixed native candidate and independent boundary/pair verification

Private CUDA control r4 replaces the192 serial directed-FP64 enclosure recurrences with an upward-rounded postfixed enclosure, checking its sufficient condition and failing conservatively to an infinite enclosure if no certificate is established. The integer repair arithmetic, activation, routing, stream and AMAX behavior stay unchanged. CPU native build exits0 /24.528166949 s; actual binary SHA256 `e692d0783bf0339d62f736b7bb7a3d247dbdc0c23e4d7d8cba9d3e3f54315fd4`.

At frozen ce0, actual producer exits0 /58.553684912 s. The same four boundary/reuse/stream/ownership/invalid-route cases pass. All584 actual CUDA bounds match an independent CPU implementation and enclose the serial model. All65,280 finite BF16 words validate the existing F32-derived exponent expression, including subnormals; this is an expression check, not a new native subnormal-arithmetic test. All80 warmed CUDA-event wrapper observations compare actual C-BF16/D/AMAX outputs, using ten observations per arm per case with alternating ABBA/BAAB order. Median serial/postfixed wrapper times in milliseconds are15.654592/4.298224,16.056960/4.399632,0.316368/0.147840 and1.996048/0.610224, ratios3.642/3.650/2.140/3.271. These are repair-wrapper timings, not full-model timings.

Three independent CPU audits exit0 in3.335278493/2.484020463/2.768151163 s. They verify all79,872 boundary outputs, all12 complete-forward tensors, all584 bounds, all360 paired tensors and1,597,440 C words against independently reconstructed integer results, source/binary hashes and raw timing order/medians. The looser bound repairs about99.997–99.998% of full-model C, so its gain comes from cheaper enclosure evaluation, not a reduction in repair frequency.

## Latest-head whole-model generation and paired timing

At current a24/upstream29, all three original complete-forward gates pass again, native/frozen oracle byte-exact with original BF16 errors0/0/4.0010227% in independent FP64. Actual72 main/72 repair/144 postprocess calls. Five32-token cached generations retain the original token IDs, independent oracle and original-stock teacher-forcing references: every native/oracle logit and greedy token is byte-exact, native/original-stock greedy tokens are byte-exact, and unchanged1%/5% numerical gates pass. The83-token answer still gives the original refusal and its application flag remains false. No threshold, oracle, date token, model capacity or answer flag is changed.

The latest separately declared performance qualification uses numerical gates plus equality with original-stock behavior. It preserves the earlier r3 producer's false application flag and NOT_RUN timing status; it does not relabel that old run. After one untimed warmup per arm and length, four timed requests per length run in ABBA order: two observations each for the original serial-bound control and new postfixed-bound control. Both arms execute the same all24 model, MXFP8 Tensor Core main, encoded-only integer repair and diagnostic flag reductions. Each request runs32 greedy tokens with real retained KV, and all warm/timed continuation logits and IDs equal the fixed oracle.

|Input length|Serial median prefill/decode/request ms|Postfixed median prefill/decode/request ms|Private-control request ratio|
|---|---:|---:|---:|
|256|21507.646657 /5283.722704 /26791.369361|6056.659767 /2545.868500 /8602.528267|3.114359933|
|1024|85350.579833 /5353.416954 /90703.996787|23544.992353 /2545.099370 /26090.091724|3.476568720|

Five quality requests plus12 warm/timed requests give17 actual requests, child13056 main/13056 repair/26112 postprocess calls, in addition to the separate72 complete-forward calls. The previous unpaired original-stock observation is much faster; there is no new paired stock/BF16/default K1024 throughput claim and this control is not an accepted production optimization.

Latest independent CPU forward audit exits0 /2.377843649 s and verifies all12 saved complete-forward tensors. Latest generation/pair audit exits0 /6.136482919 s, checking all56 tensors, all17 requests, every greedy argmax/answer/cache identity/final length/flag/gate/count, raw ABBA timing, both actual extension digests and source/binding provenance. CUDA is uninitialized and new native calls0 in both audits. The extracted runtime factory inherited a `native_forward_byte_identical_to_g4:true` metadata label; literal source differs by the explicit `self.repair_module` selection handle. The auditor records literal equality false and all forward math except that module handle byte-identical. The stale raw label/source is preserved, not silently rewritten.

## Evidence and remaining work

Four Mac archives exclude model/cache/engines/PT/native/JIT binaries/Git bundles. Forward archive190 files/all189 manifest size/SHA entries, tar SHA256 `e56589cf6a7c89ed0740c6ebdf9bcd7972e012e72f44981102075d5d3e4f8780`, includes earlier preflight under its separate historical scope. Generation archive22 files/21 manifest entries, SHA `0b7ca6045655b43a4dad408cbc1aaa45c6a2c0485d12be88ca6d47ad945eea3e`. Postfixed boundary/pair archive47 files/46 entries, SHA `61d9b2b27ec5e0b8b64318e8be5035b43e342cb985763037191e44bfcd90087d`. Latest a24 pipeline archive46 files/45 entries, SHA `4d4a1f418d1203132cae4cdae7cf92779dc6569e23ec0670561f2c817a4398ce`. Dedicated Mac transport checks accompany this report. Exact executed source payloads are published as `.py.txt`/`.cu.txt`/`.cpp.txt` with original-byte manifest mapping; failed sources and stale raw labels are preserved without formatter rewrites. Raw tensors and binaries stay on Thor with recorded hashes.

Original A3 optimization goal remains open. The postfixed candidate has now passed boundary/reuse correctness, unchanged complete numerical gates, generated/KV parity and frozen paired performance against its private predecessor. Its prohibitive remaining cost and the failed original default recipe prevent declaring A3 complete or promoting it into the production correctness PR. Further work will check low-token padding/host submission and quantization overhead using supported configurations, original capacities/semantics and the same frozen references. Shared `/tmp/codex-thor-perf.lock` is mandatory; vLLM stays inactive, heartbeatpr229-a3 stays PAUSED, no restoration timer/clock/power change or other-job interruption. PR229/224/A2/SeedVR2 evidence remains independent.
