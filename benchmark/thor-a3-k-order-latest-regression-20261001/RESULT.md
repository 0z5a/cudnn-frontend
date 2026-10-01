# Thor A3: latest-head native regressions and rejected model candidates

Actual tested runtime head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, based on upstream develop **e2bf967b17dae17a642102198e6d862d559655c9**. This report separates the old-head local K-order study from latest-head hardware/model evidence. The dedicated correctness PR remains draft; full24 Granite accuracy and paired model performance remain open.

## Latest native build and 31 GPU regressions

A fresh native build returned **0 in 196.111348067 s** (preparation wrapper 0 in 201.915368709 s). All 113 native build inputs were compared with the previous checkout: `python/pygraph/sdpa_thd_binding.cpp` changed, so the old binding was not reused. Actual new binding SHA256: `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a`. The Python/kernel and binding imports resolve to the private latest checkout.

Actual pytest returned **0**, with **31 passed in 59.99 s**, zero failures/errors/skips. Inner logged elapsed time is 63.495093092 s. The single `gw0` worker logged 61 actual native calls and exit0. The coordinator logged 0 native calls and exit0, as expected. JUnit contains all31 distinct cases: 11 scale/reference/reuse/graph, 4 stream, 1 AMAX, 5 canonical/legacy and 10 representative original FP4/FP8 cases; this is not the entire upstream matrix.

The original outer driver **returned1** after successful pytest because it incorrectly asserted positive native-call counts for *every* import record, including the coordinator. Its raw failure, source and logs are retained. A separate CPU finalizer returned0 in 0.164904616 s, independently verified real pytest/JUnit/import/source/native-build evidence, and did not rerun GPU tests. Therefore the tests are accepted as31 passes; the outer driver is never reported as rc0.

Runtime: Thor SM110/20 SMs; Torch2.13+cu130, Toolkit13.2, cuDNN9.20, CuTeDSL4.7.1, Transformers5.17. Original SFD/AMAX/launch-stream repairs are unchanged by the rebase. Other architectures are not hardware-tested.

## Local K-order experiment at prior head7bfeda6b

Five actual native calls at **7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744** used the original73-token first-layer trajectory,584 live rows and double-MXFP8 expandedK4096. Paired A/B and scale groups were reordered in blocks0(identity),32,64,128,256. Actual native FP32 C and decoded FP16 D were saved/consumed. Identity C/D reproduce the prior saved outputs byte-for-byte.

| K block | Native C cast to BF16 vs stock C: differing values | FP32 reference C cast to BF16 vs stock C: differing values |
|---:|---:|---:|
| identity |285|263|
|32|82|279|
|64|91|272|
|128|93|276|
|256|114|274|

Native C/reference L2 improves from2.8039e-6 to1.50086e-6 for block32, but native D/original BF16 D stays approximately0.3189% across all five arms. Local C rounding agreement does not establish usable model accuracy or speedup. GPU producer rc0; CPU audit rc0 in9.025305287 s independently rebuilt each permutation, recomputed CPU dot values from prior saved inputs, verified35 saved tensors and byte-exact direct-address D/SFD decode. Candidate physical input buffers were not captured; the CPU recomputation does not prove each candidate's physical input packing. No new GPU calls occurred during this CPU audit.

## Full24 K32 candidate at latest741e10ff

Same original pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352 parameters/all24/E32/top8/H1024/I512. Original BF16 baseline from7bfeda6b is retained unchanged. Whole-logit gates are1% native/same declared quantized oracle and5% native/original BF16, for every input position and all49155 vocabulary entries. No capacity reduction, layer23 hybrid, fallback or Torch substitution is accepted as full24 success.

K32 interleaves four matched cross-component pairs in original32-column blocks, with explicit corresponding logical/physical scale packing. All three cases were checked for each supported policy. Four short fixed-oracle cases violate the necessary joint-gate bound; native execution was correctly not attempted for those cases. Both256-token cases actually executed24 native calls each; actual native D was consumed.

| Policy |Tokens|Oracle / original BF16 L2|Native / oracle L2|Native / original BF16 L2|Outcome|
|---|---:|---:|---:|---:|---|
|k32_double_mxfp8_bf16_d_down|73|9.582791%|Not run|Not run|Fixed oracle incompatible|
|k32_double_mxfp8_bf16_d_down|70|9.549737%|Not run|Not run|Fixed oracle incompatible|
|k32_double_mxfp8_bf16_d_down|256|4.774713%|3.410810%|4.623739%|Fails oracle gate|
|k32_double_mxfp8_fp16_d_fp32_down_merge|73|11.862574%|Not run|Not run|Fixed oracle incompatible|
|k32_double_mxfp8_fp16_d_fp32_down_merge|70|11.246371%|Not run|Not run|Fixed oracle incompatible|
|k32_double_mxfp8_fp16_d_fp32_down_merge|256|4.718186%|4.083181%|5.087097%|Fails oracle gate and BF16 gate|

Actual producer rc0 in24.079368575 s including admission; CPU audit rc0 in4.208582043 s. All eight complete saved logit tensors, FP64 metrics, producer FP32 metrics, source/import/binding hashes, original baseline and48 observed native calls audited. Both whole-three-case policies are rejected. Generation/KV and paired performance were not run for rejected candidates.

## Twelve new BF16-activation / FP32-state reference cases at741e10ff

Separate reference-only experiment: A3/B2 MXFP8 reconstruction, expandedK6144, FP32-input/output dot; C/SiLU/product each round to BF16, followed by normalized FP16 staging. Original BF16 router retained, FP32 residual, two nonquantized-math policies and two down-projection policies. Native staged mode is unimplemented, native calls0, and these Torch reference values never substitute native execution.

|Complete policy|73-token oracle/BF16 L2|70-token oracle/BF16 L2|256-token oracle/BF16 L2|All-case necessary compatibility|
|---|---:|---:|---:|---|
|fp32_residual_bf16_linears/bf16_activation/torch.bfloat16|11.860441%|11.495701%|4.041110%|False|
|fp32_residual_bf16_linears/bf16_activation/torch.float32|13.440216%|12.924793%|5.484596%|False|
|fp32_non_quantized_math/bf16_activation/torch.bfloat16|13.103665%|11.643302%|4.412514%|False|
|fp32_non_quantized_math/bf16_activation/torch.float32|16.180784%|11.589949%|4.530497%|False|

All eight short cases violate necessary compatibility (minimum lower bound10.5102107489%, exceeding5%). All four long cases satisfy only the necessary bound; no complete policy is eligible and there is no native pass. Producer rc0; CPU audit rc0 in4.228876371 s verifies all12 complete saved logit tensors/24-layer oracle-call records/provenance without initializing CUDA. This unsupported staging feature is not promoted.

## Evidence and final state

Raw source bytes are stored as `.py.txt`; manifests preserve their original `.py` paths/hashes. All source/log/JUnit/metadata entries were verified after transport to Mac. Models, cache, engines, native binaries and `.pt` tensor outputs remain excluded; their metadata hashes are retained. The combined50-file archive SHA is `dc02da110656da73a5c80073870862a5ed6e6bca6ce77ebb8a52ff7c0daf4565`; separate K32 21-file archive SHA is `4575a7d9b6b34eeac54c6148351f2b215d4615a1a09e8af5a80c4026f5abf8e5`. All49+20 manifest entries verified.

GPU work obeyed `/tmp/codex-thor-perf.lock`; waits are not inference timings. vLLM remains inactive. No service restoration timer, clock/power setting or other task was changed/interrupted. PR1280 runtime source is741e10ff; this evidence-only branch does not alter production code. No full24 accuracy or throughput success is claimed.
