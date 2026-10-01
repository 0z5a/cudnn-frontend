# Thor A3: native BF16 epilogue staging remains rejected

Runtime imports/build head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1** on develop **e2bf967b17dae17a642102198e6d862d559655c9**. Production checkout and PR1280 code remain unchanged by these experiments. An explicit independently stored experimental native kernel implements BF16 C/SiLU/product rounding while retaining **MXFP8 tensor-core dot with FP32 accumulation**. It does not implement native BF16/FP64 matrix multiplication. No experiment below fixes the full24 model; no new feature or optimization is accepted.

## Preserved model and accuracy contract

Original pinned Granite3.1 1B-A400M Instruct revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, checkpoint SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`;1,334,628,352 parameters,24 layers/E32/top8/H1024/I512. Original saved BF16 baseline at7bfeda6b remains immutable. Cases73/70/256 tokens use every position and all49155 logits. Gates remain1% native/declared independent oracle and5% native/original BF16. No layer23 hybrid, smaller capacity, fallback, support bypass or Torch replacement of failed native output counts as success.

Latest fresh native binding SHA `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a` is loaded from the verified private latest environment. The experimental kernel lives outside that checkout, under the tools directory, and is installed into the API class only inside this diagnostic process. Actual native-plan records resolve to its path/SHA, not silently to the production kernel.

## FP64 oracle control with original BF16 state

No native execution. Two policies accumulate either original BF16 values or two-component reconstructed effective MXFP8 values in FP64, then round C/SiLU/product to BF16. Router, residual, down projection and merge retain original BF16 behavior. This oracle-only arithmetic control tests whether improving dot accumulation alone restores model agreement; it is not a proposed native FP64 mode.

|Policy|Tokens|Oracle / original BF16 L2|Joint-gate necessary bound|Compatible bound?|
|---|---:|---:|---:|---|
|original_fp64_dot_bf16_stages|73|9.974721%|8.976221%|False|
|original_fp64_dot_bf16_stages|70|9.396419%|8.398296%|False|
|original_fp64_dot_bf16_stages|256|4.579419%|3.578900%|True|
|double_effective_fp64_dot_bf16_stages|73|9.721817%|8.721651%|False|
|double_effective_fp64_dot_bf16_stages|70|8.495447%|7.498024%|False|
|double_effective_fp64_dot_bf16_stages|256|4.769763%|3.769509%|True|

Both complete policies fail short-case necessary compatibility. The original-weight/no-quantization control also differs: more accurate accumulation does not imply matching the original BF16 rounding trajectory. This observation is not proof that every possible pure-MXFP8 recipe must fail.

Original r1 returned1 in41.860836794 s after computing six references because a post-reference assertion compared quantizer function identity across two independently imported helper modules. Its frozen source/partial/logs are retained; no saved r1 tensors exist. Separate r2 fixes only that bookkeeping check, verifies both native counters remain0 and original quantizer unchanged, and returned0 in42.673177020 s. Independent CPU audit returned0 in3.444257538 s and verifies all6 complete saved r2 tensors/bounds/provenance. r1 is not reported as rc0 or fully tensor-audited.

## Actual native BF16-staged MXFP8 epilogue

Experimental source SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3**. The existing persistent blockscaled MMA/TMA/AMAX/scale/stream paths are retained. Scalar epilogue adds explicit BF16 casts at C, SiLU and product; vectorized epilogue is explicitly declined for this prototype. No extra input tensor, separate activation fallback or native BF16 GEMM is introduced. The prototype is not promoted and has not undergone the boundary/reuse/performance checks required for acceptance.

Frozen declared oracle recasts reconstructed logical operands to BF16 and performs BF16 dot plus BF16 activation staging, before original BF16 down/merge. This oracle definition is fixed before native execution; it is not swapped after a failure. It is byte-identical to original BF16 for both short cases, with4.0010227% oracle/BF16 L2 for256 tokens. Both native K orders share that logical oracle. Actual native D and row SFD are decoded and consumed throughout all24 layers.

|Native K order|Tokens|Oracle / BF16 L2|Native / oracle L2|Native / BF16 L2|Gates|
|---|---:|---:|---:|---:|---|
|component_major_mxfp8_dot_bf16_stage|73|0.000000%|11.478155%|11.478155%|oracle=False, BF16=False|
|component_major_mxfp8_dot_bf16_stage|70|0.000000%|11.199077%|11.199077%|oracle=False, BF16=False|
|component_major_mxfp8_dot_bf16_stage|256|4.001023%|4.711875%|4.753035%|oracle=False, BF16=True|
|k32_mxfp8_dot_bf16_stage|73|0.000000%|10.408587%|10.408587%|oracle=False, BF16=False|
|k32_mxfp8_dot_bf16_stage|70|0.000000%|10.497938%|10.497938%|oracle=False, BF16=False|
|k32_mxfp8_dot_bf16_stage|256|4.001023%|4.337401%|4.313020%|oracle=False, BF16=True|

Producer realrc0 in19.146416332 s, **144 actual native calls** (all6 cases×24 layers). CPU audit rc0 in3.079021760 s verifies all12 full-logit tensors, exact producer FP32 metrics, independent FP64 metrics, gate decisions, original baseline, actual imports/binding/prototype source. Both whole-three-case policies fail1%; short cases also fail5%. No generation/KV or performance test is represented as passing.

## Common scale / per-element four-cross-term order

A second frozen worker quantizes the residual with the high component's common SF32, allowing the four cross products of each original K element to be contiguous in expandedK4096. Physical scales are packed for either component-major or element-major order. This changes declared reconstruction, so it has its own frozen logical oracle. Candidate physical input buffers were not saved; CPU logit/provenance audit is not a complete independent physical-pack audit of these new buffers.

|Native order|Tokens|Oracle / BF16 L2|Native / oracle L2|Native / BF16 L2|Outcome|
|---|---:|---:|---:|---:|---|
|common_sf_component_major_mxfp8_bf16_stage|73|10.771147%|Not run|Not run|Fixed oracle incompatible|
|common_sf_component_major_mxfp8_bf16_stage|70|10.565141%|Not run|Not run|Fixed oracle incompatible|
|common_sf_component_major_mxfp8_bf16_stage|256|4.474529%|4.105209%|4.681538%|Fails1% oracle|
|common_sf_element_major_mxfp8_bf16_stage|73|10.771147%|Not run|Not run|Fixed oracle incompatible|
|common_sf_element_major_mxfp8_bf16_stage|70|10.565141%|Not run|Not run|Fixed oracle incompatible|
|common_sf_element_major_mxfp8_bf16_stage|256|4.474529%|3.899610%|4.302574%|Fails1% oracle|

Producer rc0 in16.302130477 s. Four short fixed oracles are incompatible; only two256-token cases execute24 calls each, **48 actual native calls**. CPU audit rc0 in2.774404631 s verifies all8 full-logit tensors/bounds/provenance. Neither complete policy is accepted.

## Evidence and state

- `a3-fp64-dot-bf16-state-evidence-20261001.tar.gz`: SHA `a39e1a0dd54469c622e2503f797d458531fa67afd9c5da509447a2dc0d41687e`, all25 manifest entries verified after Mac transport.
- `a3-mxfp8-bf16-staged-native-evidence-20261001.tar.gz`: SHA `ac0e2526ef60d9909521d25bae461a0a98560a12faa2673f845a0ef7bd5c603a`, all21 manifest entries verified after Mac transport.
- `a3-common-sf-mxfp8-bf16-staged-evidence-20261001.tar.gz`: SHA `cd118703615b7ef944d85715378ea1315cb2b75ead71b805fcf31c95c98a7cbc`, all21 manifest entries verified after Mac transport.

Raw `.py.txt` payloads preserve exact source bytes and original `.py` manifest paths; raw failures/logs/metadata remain tracked. Model/cache/engine/native-binary/.pt outputs are excluded; output hashes remain in audits. GPU admission uses `/tmp/codex-thor-perf.lock` before library imports. vLLM remains inactive. No timers, clock/power changes or other-task interruption. Formal paired model performance is NOT_RUN. The existing correctness PR remains draft at741e10ff; the broader full24 A3 goal remains active.
