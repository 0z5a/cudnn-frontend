# Thor A3: all24 actual-C epilogue isolation

Runtime/private import and fresh-binding head **741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1**, based on **e2bf967b17dae17a642102198e6d862d559655c9**. Experimental pure-MXFP8 dot/BF16 staged epilogue SHA **68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3**. Production checkout remains unchanged, PR1280 remains draft, and the complete24-layer model recipe still fails previously recorded1%/5% gates. This diagnostic does not change that result.

## Scope and execution

Use the original saved BF1673-token trajectory separately for each of24 layers, with its own real top8 routes and original checkpoint gate/up weights, E32/H1024/I512. Two-component K32-interleaved MXFP8 expands dotK to4096 without changing original model width/parameters. Both BF16 and FP16 output-storage modes run for each layer: **48 actual native calls**,584 live routed rows per case, with changed route counts and padded rows7680/7936/8192. Each actual native FP32 C is rounded to BF16 and used only to compute an independent GPU SiLU/product reference. Actual native D plus row SFD are decoded/compared, never substituted by reference outputs. This is48 isolated fixed-input executions, not full-model feedback or generation/KV/performance success.

Original checkpoint revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`,1,334,628,352 parameters. Original baseline/weights/gates/capacity retained. Native binding SHA `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a`; native-plan records verify actual experimental kernel path/hash. The production API's support/stream/AMAX semantics are retained.

Originalr1 returned1 in12.056812239 s after the first BF16D native call because the reused diagnostic decoder explicitly accepted only FP16 output. Its frozen source/logs are preserved; it saved no output tensors and is not a successful48-case run. New independentr2 adds a BF16/FP16 actual raw-D/SFD decoder, returns0 in18.642515409 s, and saves288 tensor entries. Independent CPU audit returns0 in5.104088182 s, verifies every saved tensor shape/hash, independently reconstructs direct physical scale addresses and decoded D byte-for-byte, recomputes producer FP32 and independent FP64 metrics, checks actual-C CPU activation vs saved GPU references, original routes/counts/weights/source/import/binding/prototype hashes. CPU CUDA is uninitialized and new native calls0. New candidate physical input buffers and runtime compiled-plan identities are not captured; this is not independent physical-pack or compiled-plan reuse identity proof.

## Results

|Storage|Total native D / actual-C GPU reference differing values across24 cases|Maximum relative L2|Maximum absolute delta|CPU actual-C staging / native D maximum relative L2|
|---|---:|---:|---:|---:|
|BF16|17|6.5137911429e-43|7.3468396926e-40|6.5137911429e-43|
|FP16|59|5.4694231957e-9|3.0517578125e-5|0|

BF16 CPU activation stages match the saved GPU activation stages exactly. The17 tiny BF16 differences are in the subnormal range after output normalization. FP16 CPU reference independently chooses normalization with FP32 `frexp`; it matches actual native D exactly across all24 cases. The previous GPU helper computed normalization from a BF16 activation tensor, yielding59 tiny differences, separately retained instead of silently zeroed or reclassified. These local normalized-output differences do not establish a complete-model1% pass.

Native FP32 C cast to BF16 differs from original BF16 dot C by **62–102 values per layer**. Maximum native D/original saved D relative L2 is approximately9.40634e-5 across this fixed-input dataset. The full-model native short-case errors remain approximately10–11% in the independently audited feedback experiments. **Inference:** these fixed-input tests favor further investigation of dot rounding and trajectory/routing amplification over a large epilogue arithmetic defect. They do not prove a causal explanation for every full-model error or rule out all other cases.

A preceding first-layer identity/K32 producer check returned0 with two native calls and zero actual-C GPU-reference differences. Its source/logs/metadata are included; its16 saved tensors have not received a separate independent CPU tensor audit in this archive. The288-tensor all24r2 audit is the independently verified scope.

## Evidence and state

34-file archive SHA **713c083fe759ddb3ad6678810f0d2d5cb1043e723a424cb75ee74985da607c51**, all33 manifest entries verified after transport to Mac. `.py.txt` source copies retain exact tested bytes and original `.py` manifest paths. Cache/engines/checkpoints/native binaries/.pt outputs remain excluded. Real failures, partial/status logs and metadata remain available.

GPU admission obeyed `/tmp/codex-thor-perf.lock` before GPU-library imports; no other job was interrupted. vLLM remains inactive, no timers or clock/power changes. Full24 accuracy/generation/KV and formal frozen paired model performance remain open. No new native feature or optimization is accepted; the A3 goal remains active.
