# Thor accumulation arithmetic reproduced; full-model precision still open

An independent CPU implementation of a published raw-exponent arithmetic formula reproduces both saved original BF16-input/FP32-output GPU results and saved actual MXFP8 native FP32 C, bit-for-bit, at the original first-layer expert. Its original-BF16 predictions also match all 224,256 sampled stock outputs across all 24 layers, including 223,680 held-out outputs. This identifies a concrete accumulation-rounding contribution to the remaining numerical differences. It is a diagnosis, not a repaired native model or a new production feature.

All work in this round is CPU-only: CUDA stays uninitialized, new GPU/native calls are zero, frozen production source remains unchanged. The full pinned Granite 3.1 1B-A400M Instruct retains all 1,334,628,352 parameters/24 layers/E32/top8/H1024/I512 and checkpoint revision `0da7a48b0276d500ce5922fd2b33944091fc6c09`, SHA `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`. Neither original BF16 baseline nor declared oracle/1% and 5% gates is changed. The latest actual original-recipe model result remains failed; [latest complete-model configuration comparison](https://github.com/0z5a/cudnn-frontend/blob/2466c034c7d38bd119277d7333ccd2545b476ea6/benchmark/thor-a3-production-c-store-full24-pair-20261001/RESULT.md).

## Predeclared formula and source

The independently written implementation tests the formula in [sf-tensor's original B200 research model, pinned at d96b82180ac9041c09295f3a6a3c2c4e592ddce6](https://github.com/sf-tensor/tcgen05/blob/d96b82180ac9041c09295f3a6a3c2c4e592ddce6/tcgen05_model/MODEL.md). Products and incoming C share a quantum derived from their maximum raw exponent minus 25. Each contribution is aligned by truncating its magnitude, signed integer units are summed, and the result is normalized to FP32 toward zero. Scaled FP8 product exponents include their power-of-two scales. The source validates B200 and explicitly leaves other architectures unverified; applicability to these Thor datasets is tested here, not assumed. The pinned document and its MIT license are preserved.

Before seeing these CPU results, the main hypothesis was fixed to ascending contiguous K16 updates for original BF16, with 24/26-bit, K32 grouping, exact-dot RN/RZ and four IEEE group/add rounding controls. First 18 and last 18 rows were separated without selecting output mismatch coordinates. For native MXFP8, the separately fixed hypothesis uses actual K32 scale blocks and frozen HH/HL/LH/LL block-interleaved packing. Logical quantization is independently recomputed from saved BF16 operands, without importing any model/native kernel or external research package.

## Original BF16 FP32 validation

All 36 actual rows × 1024 columns are used; 220 padded zero rows do not inflate matches. Saved actual36 and padded256 GPU FP32 outputs agree on all live rows. The K16/25-bit prediction matches all 36,864 FP32 words and their stock BF16 casts, including all 18,432 held-out words. An independent scalar implementation decodes input bits, shifts integer magnitudes and packs FP32 words; it validates 64 unconditioned coordinates at each of 24/25/26 bits.

| Fixed candidate | FP32 mismatches / 36,864 | Stock BF16 mismatches |
|---|---:|---:|
| Raw window25, K16 | 0 | 0 |
| Raw window24, K16 | 33,136 | 3 |
| Raw window26, K16 | 25,335 | 1 |
| Raw window25, K32 control | 35,434 | 7 |
| Exact dot, FP32 RN | 36,498 | 20 |
| IEEE K16 group RN / add RZ | 28,483 | 0 |

The last control demonstrates why matching only a final BF16 cast does not identify the underlying FP32 arithmetic. All other predefined control results and split counts are retained in raw metadata.

## Held-out original all24 stock data

Every layer uses all 584 original expert routes and 16 fixed, unconditioned packed gate/up columns `[0,1,15,16,31,32,33,63,64,127,255,511,512,767,991,1023]`. Original hidden states/routes and checkpoint weight packing are recomputed. Only layer0/expert0's 576 sampled values overlap the discovery scope; all remaining 223,680 values are held out. Eight independent scalar coordinates per layer additionally verify integer/vector implementation agreement.

| Fixed candidate | BF16 mismatches / 224,256 | Held-out BF16 mismatches / 223,680 |
|---|---:|---:|
| Raw window25, K16 | 0 | 0 |
| Raw window24, K16 | 31 | 31 |
| Raw window26, K16 | 14 | 14 |
| IEEE K16 group RN / add RZ | 21 | 21 |

These saved all24 targets are original stock **BF16** C. Underlying stock FP32 words were not saved for all24; the table does not claim that stronger match. No full model is rerun here and no sampled target is substituted into a native trajectory.

## Actual MXFP8 native C validation and a rounding crossing

The separately saved native K4096 two-component experiment at production runtime head `741e10ffa5e93dbc0a8a41e06e66e0c81c1f85b1` uses private BF16-staged kernel SHA `68d7c926dec0fbdf7bb0015a80df605d4ba8045af83929db5376b3b66b89feb3`. The native run predates this CPU round; it is not reported as a new GPU execution. Both saved BF16-D and FP16-D configurations have byte-identical first-expert C.

The K32/25-bit prediction matches all 36,864 actual native FP32 words and all 18,432 held-out words. Its 24/26-bit controls differ in 33,018/25,347 FP32 words. Independent raw E4M3/scaling/integer normalization reproduces 32 unconditioned vector coordinates. Original input reconstruction is exact; the maximum reconstructed-weight difference is `7.450580596923828e-09`. These are logical reconstruction checks, not a new audit of all physical buffers.

Comparing the original BF16 and actual MXFP8 predictions gives 31,821 FP32 word differences. Of the 35,895 coordinates whose exact FP64 dots agree between the two operand representations, 30,977 FP32 results still differ. Thus input reconstruction error alone cannot explain these differences in this tested scope.

One BF16 crossing occurs at original first-expert row15/packed-column695. Both exact dots equal `0.8847652553231455`. The saved original GPU FP32 word is `1063419908`; actual native MXFP8 word is `1063419903`, five FP32 ULPs apart. They round to `0.88671875` and `0.8828125` respectively. This disagreement remains with an identical exact dot, providing a direct finite-data witness of accumulation/grouping affecting the BF16 result. It does not prove that one coordinate alone causes the complete model failure.

## Evidence, return codes and limits

Five terminal CPU runs have real logged returncode0, no signal: initial BF16 prediction10.216923519 s, all24 holdout22.965645401 s, MXFP8 prediction17.303956531 s, independent audit r1/r2 4.327275150/4.289232863 s. These are CPU diagnostic elapsed times, not a kernel or model performance comparison. The audit independently checks original/saved GPU bytes, source and input hashes, split/control counts and all160 saved prediction tensor entries. All independent integer evaluations total416 (192 initial bit-width cases +192 all24 +32 MXFP8).

Raw logged stdout/stderr/run records, executed source, source provenance and prior input metadata are transported to the Mac. All35 archive manifest entries and archive SHA `3c37bae080a9d35ade3f9c5065a74654aebb7691172d7553431ac5237726ea4c` are independently size/hash checked. Two pinned reference files are additionally hash checked. Models/cache/engines/PT/native binaries/NCU binaries are excluded. One archive setup invocation preceded completion of its yielded SCP copy and exited2 with missing-file error; it is retained separately, the copy completed0 and the identical archiver then succeeded. No GPU test failure is hidden.

Original stock input provenance remains historical head `7bfeda6bfeaa63a1e59fadc5d9fb48676f03f744`; actual native C provenance is latest production runtime head741 with fresh binding SHA `37e7c4e0d16bd0a952ade77ccd577ccadae1888f166a4c897c9e98a104606c2a`. No old whole-model result is relabeled as a new latest-head execution. The original BF16-output NCU SASS is retained, but the BF16-input/FP32-output control was not profiled; arithmetic agreement is not a proof of identical kernel selection or all undocumented ISA behavior.

The next precision work can use this validated arithmetic model to evaluate a principled MXFP8 representation/accumulation change before consuming GPU time. A future repair must be computed from actual input operands and tested on independent boundary/reuse cases and complete native Granite feedback. An empirical output bias, relaxed gate, changed oracle, padded-zero match count, native BF16 substitution or software prediction standing in for native output is not accepted. The separate unused-C-write timing remains a K4096 primitive result, not usable full-model throughput. PR1280 stays draft; full24 accuracy/generation/KV/formal paired model performance remain open. No service/clock/power/restoration timer change or other task interruption is made.
