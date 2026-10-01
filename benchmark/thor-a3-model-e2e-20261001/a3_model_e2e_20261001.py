# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full trained Granite MoE inference through the checked A3 public wrapper.

This experimental adapter changes expert gate/up precision to MXFP8. Stock BF16
and an independent FP32 oracle are correctness arms, not performance baselines.
The performance A/B changes only frontend source, holding the model adapter,
checkpoint, precision, padding, routes, tile and input token IDs fixed.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6")
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time

try:
    import transformer_engine
except (ImportError, OSError):
    pass

import torch
import torch.nn.functional as F
import transformers
import cudnn
from cudnn import _compiled_module
from cudnn.gemm.cutedsl.grouped.swiglu import api
from cuda.bindings import driver as cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts

MODEL = Path("/home/jwipc/models/granite-3.1-1b-a400m-instruct-0da7a48b")
ROOT = Path("/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928")
WEIGHT_SHA = "ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1"
NATIVE_CALLS = 0
NATIVE_PLANS = {}
ORIGINAL_EXECUTE = api.GroupedGemmSwigluSm100.execute


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def tensor_info(x):
    c = x.detach().cpu().contiguous()
    return {"shape": list(c.shape), "dtype": str(c.dtype), "sha256": hashlib.sha256(c.flatten().view(torch.uint8).numpy().tobytes()).hexdigest()}


def json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (torch.dtype, Path)):
        return str(value)
    raise TypeError(f"Unsupported metadata type {type(value).__name__}")


def counted_execute(self, *args, **kwargs):
    global NATIVE_CALLS
    result = ORIGINAL_EXECUTE(self, *args, **kwargs)
    NATIVE_CALLS += 1
    kernel = self._kernel
    name = kernel.__module__ + "." + kernel.__name__
    if name not in NATIVE_PLANS:
        module = importlib.import_module(kernel.__module__)
        NATIVE_PLANS[name] = {"path": module.__file__, "sha256": sha_file(module.__file__)}
    return result


api.GroupedGemmSwigluSm100.execute = counted_execute


def quantize_mxfp8(x):
    """BF16/FP32 [..., M,K] -> E4M3, physical F8_128x4 E8M0, logical scale.

    Quantization and scale packing are intentionally ordinary Torch operations
    at the model boundary; all their cost is inside end-to-end timing.
    """
    assert x.shape[-1] % 128 == 0 and x.shape[-2] % 128 == 0
    m, k = x.shape[-2:]
    leading = x.shape[:-2]
    blocks = x.float().reshape(*leading, m, k // 32, 32)
    maximum = blocks.abs().amax(dim=-1)
    exponent = torch.where(maximum > 0, torch.ceil(torch.log2(maximum * (1.0 / 448.0))), 0).clamp(-126, 127)
    scale = torch.pow(2.0, exponent)
    q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
    codes = (exponent + 127).to(torch.uint8)
    # [..., M/128, r_hi(4), r_lo(32), K/128, c_lo(4)]
    raw = codes.reshape(-1, m // 128, 4, 32, k // 128, 4)
    packed = raw.permute(0, 1, 4, 3, 2, 5).contiguous().view(torch.float8_e8m0fnu)
    return q, packed, scale


def dequantize(q, scale):
    return (q.float().reshape(*scale.shape, 32) * scale.unsqueeze(-1)).reshape(q.shape)


def dequantize_wrapper_d(result, norm):
    """The MXFP8 wrapper returns normalized D even when D storage is BF16.

    Consume its row SFD output before a plain BF16 down projection. Scale bytes
    follow the same physical F8_128x4 layout as SFA/SFB; column SFD is not used.
    """
    d, sf = result["d_tensor"], result["sfd_row_tensor"]
    assert sf is not None and d.ndim == 2
    rows, cols = d.shape
    codes = sf.view(torch.uint8).reshape(1, rows // 128, cols // 128, 32, 4, 4)
    logical = codes.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(rows, cols // 32)
    scale = torch.pow(2.0, logical.int() - 127)
    return (d.float().reshape(rows, cols // 32, 32) * scale.unsqueeze(-1) / norm).reshape(rows, cols).to(torch.bfloat16)


def pack_contract_check():
    # Vary every logical scale, including both row subaxes and K subaxes.
    x = torch.pow(2.0, (torch.arange(256 * 1024).reshape(256, 1024) // 32 % 17 - 8).float())
    q, packed, scale = quantize_mxfp8(x)
    codes = packed.view(torch.uint8).reshape(1, 2, 8, 32, 4, 4)
    logical = codes.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(256, 32)
    assert torch.equal(torch.pow(2.0, logical.int() - 127), scale)
    assert torch.equal(dequantize(q, scale), x)
    # A plain flatten must not be mistaken for the swizzled physical buffer.
    assert not torch.equal(codes.flatten(), logical.flatten())
    return "EXACT_VARYING_SCALE_PACK_AND_DEQUANTIZATION"


class A3ExpertAdapter:
    def __init__(self, expert, index, tile):
        self.expert = expert
        self.original = expert.forward
        self.index = index
        self.tile = tile
        self.mode = "native"
        self.calls = 0
        self.last_counts = None
        e, n, h = expert.gate_up_proj.shape
        assert (e, n, h) == (32, 1024, 1024)
        self.e, self.i, self.h = e, n // 2, h
        gate, up = expert.gate_up_proj.detach().chunk(2, dim=1)
        # Production code/reference: 32 gate columns followed by 32 up columns.
        weight = torch.stack((gate.reshape(e, -1, 32, h), up.reshape(e, -1, 32, h)), dim=2).reshape(e, n, h)
        self.b, self.sfb, self.bscale = quantize_mxfp8(weight)
        self.alpha = torch.ones(e, dtype=torch.float32, device="cuda")
        self.norm = torch.ones(1, dtype=torch.float32, device="cuda")
        self.bref = None

    def attach(self, mode):
        self.mode = mode
        self.expert.forward = self.original if mode == "stock" else self.forward

    def routing(self, x, index):
        flat = index.flatten()
        order = torch.argsort(flat, stable=True)
        expert_ids = flat[order]
        counts = torch.bincount(flat, minlength=self.e)
        # Explicit experimental integration sync, included in model latency.
        sizes = counts.cpu().tolist()
        aligned = [((n + 255) // 256) * 256 for n in sizes]
        offsets, running = [], 0
        for n in aligned:
            running += n
            offsets.append(running)
        assert running > 0
        ends = torch.tensor(offsets, dtype=torch.int32, device=x.device)
        starts = torch.tensor([0] + offsets[:-1], dtype=torch.long, device=x.device)
        actual_start = counts.cumsum(0) - counts
        destinations = torch.arange(flat.numel(), device=x.device) - actual_start[expert_ids] + starts[expert_ids]
        token_ids = order // index.shape[-1]
        padded = torch.zeros((running, self.h), dtype=x.dtype, device=x.device)
        padded.index_copy_(0, destinations, x[token_ids])
        return padded, ends, destinations, order, token_ids, sizes, aligned

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.calls += 1
        x, ends, destinations, order, token_ids, sizes, aligned = self.routing(hidden_states, top_k_index)
        self.last_counts = sizes
        a, sfa, ascale = quantize_mxfp8(x)
        if self.mode == "native":
            result = api.grouped_gemm_swiglu_wrapper_sm100(
                a_tensor=a,
                b_tensor=self.b,
                sfa_tensor=sfa,
                sfb_tensor=self.sfb,
                padded_offsets=ends,
                alpha_tensor=self.alpha,
                norm_const_tensor=self.norm,
                prob_tensor=None,
                acc_dtype=torch.float32,
                c_dtype=torch.bfloat16,
                d_dtype=torch.bfloat16,
                mma_tiler_mn=self.tile,
                cluster_shape_mn=(2, 1),
                sf_vec_size=32,
                m_aligned=256,
                current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
            d = dequantize_wrapper_d(result, self.norm)
        elif self.mode == "reference":
            if self.bref is None:
                self.bref = dequantize(self.b, self.bscale)
            adeq = dequantize(a, ascale)
            d = torch.zeros((x.shape[0], self.i), dtype=torch.bfloat16, device=x.device)
            start = 0
            for expert, n in enumerate(aligned):
                if n:
                    c = F.linear(adeq[start : start + n], self.bref[expert]).reshape(n, -1, 2, 32)
                    activation = F.silu(c[:, :, 0]) * c[:, :, 1]
                    d[start : start + n] = activation.reshape(n, self.i).to(torch.bfloat16)
                start += n
        else:
            raise RuntimeError(f"Unexpected adapter mode {self.mode}")
        live_d = d[destinations]
        result = torch.zeros_like(hidden_states)
        start = 0
        sorted_weight = top_k_weights.flatten()[order]
        for expert, n in enumerate(sizes):
            if n:
                y = F.linear(live_d[start : start + n], self.expert.down_proj[expert])
                y = y * sorted_weight[start : start + n, None]
                result.index_add_(0, token_ids[start : start + n], y.to(result.dtype))
            start += n
        return result


def difference(actual, ref):
    a, r = actual.float(), ref.float()
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(r).all())
    diff = a - r
    return {
        "max_abs": diff.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(r).clamp_min(1e-12)).item(),
        "argmax_equal_fraction": (a.argmax(-1) == r.argmax(-1)).float().mean().item(),
    }


def attach(adapters, mode):
    for adapter in adapters:
        adapter.attach(mode)


def request(model, ids, new_tokens=32):
    """Greedy full-model inference with real KV cache and fixed token count."""
    before = NATIVE_CALLS
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    outputs = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_done = time.perf_counter_ns()
    next_id = outputs.logits[:, -1].argmax(-1, keepdim=True)
    generated = [next_id]
    logits = [outputs.logits[:, -1].detach()]
    cache = outputs.past_key_values
    for _ in range(new_tokens - 1):
        outputs = model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
        next_id = outputs.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_id)
        logits.append(outputs.logits[:, -1].detach())
        cache = outputs.past_key_values
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    return {
        "ids": torch.cat(generated, dim=-1).cpu(),
        "logits": torch.stack(logits, dim=1).cpu(),
        "prefill_ms": (prefill_done - begin) / 1e6,
        "decode_ms": (end - prefill_done) / 1e6,
        "request_ms": (end - begin) / 1e6,
        "native_calls": NATIVE_CALLS - before,
        "forwards": new_tokens,
    }


def load_inputs(tokenizer):
    messages = [
        "What is 2 + 2? Answer with just the number.",
        "Explain in one short sentence why the sky appears blue.",
        "Read this note and repeat the passcode exactly: the passcode is cobalt. What is the passcode?",
    ]
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)[
            "input_ids"
        ].cpu()
        for t in messages
    ]
    filler = tokenizer.encode("The laboratory recorded temperature, pressure and humidity. ", add_special_tokens=False)
    prefix = prompts[2][0].tolist()
    workloads = {}
    for length in (256, 1024):
        body = (filler * (length // len(filler) + 2))[: length - len(prefix)]
        ids = torch.tensor([body + prefix], dtype=torch.long)
        assert ids.shape == (1, length)
        workloads[str(length)] = ids
    return messages, prompts, workloads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("validate", "measure", "diagnostic"), required=True)
    p.add_argument("--expected-source", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, choices=(128, 256), default=256)
    p.add_argument("--length", type=int, choices=(256, 1024), default=256)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    meta = {"stage": "STARTED", "pid": os.getpid(), "mode": args.mode, "tile": [256, args.tile], "scope": __doc__}

    def save():
        (out / "status.json").write_text(json.dumps(meta, indent=2, default=json_default))

    save()
    try:
        source = Path(args.expected_source).resolve()
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        assert actual == args.expected_head
        for module in (cudnn, api, _compiled_module):
            assert Path(module.__file__).resolve().is_relative_to(source / "python"), module.__file__
        service = subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip()
        assert service == "inactive"
        assert torch.cuda.get_device_capability() == (11, 0)
        checkpoint = json.loads((MODEL / "thor-checkpoint-manifest.json").read_text())
        assert checkpoint["stage"] == "VERIFIED_COMPLETE"
        assert next(v for v in checkpoint["files"] if v["name"] == "model.safetensors")["sha256"] == WEIGHT_SHA
        model_impl = importlib.import_module(GraniteMoeExperts.__module__)
        meta.update(
            source_head=actual,
            source=str(source),
            imports={m.__name__: {"path": m.__file__, "sha256": sha_file(m.__file__)} for m in (cudnn, api, _compiled_module, model_impl)},
            torch=torch.__version__,
            transformers=transformers.__version__,
            torch_cuda=torch.version.cuda,
            cudnn_backend=cudnn.backend_version(),
            device=torch.cuda.get_device_name(),
            multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
            service=service,
            harness_sha256=sha_file(__file__),
            checkpoint=checkpoint,
            sf_contract=pack_contract_check(),
        )
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        messages, prompts, workloads = load_inputs(tokenizer)
        meta["inputs"] = {"messages": messages, "prompts": [tensor_info(t) for t in prompts], "workloads": {k: tensor_info(v) for k, v in workloads.items()}}
        (out / "input-token-ids.json").write_text(
            json.dumps({"prompts": [t.tolist() for t in prompts], "workloads": {k: t.tolist() for k, t in workloads.items()}}, indent=2)
        )
        started = time.perf_counter()
        model, loading = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False, output_loading_info=True
        )
        assert not loading.get("unexpected_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs"), loading
        assert not [k for k in loading.get("missing_keys", []) if k != "lm_head.weight"], loading
        parameters = sum(p.numel() for p in model.parameters())
        assert parameters == 1334628352, parameters
        assert (
            model.config.num_hidden_layers,
            model.config.num_local_experts,
            model.config.num_experts_per_tok,
            model.config.hidden_size,
            model.config.intermediate_size,
            model.config.vocab_size,
        ) == (24, 32, 8, 1024, 512, 49155)
        model = model.eval().cuda()
        experts = [m for m in model.modules() if isinstance(m, GraniteMoeExperts)]
        assert len(experts) == 24
        adapters = [A3ExpertAdapter(e, i, (256, args.tile)) for i, e in enumerate(experts)]
        meta.update(
            parameters=parameters,
            full_config=model.config.to_dict(),
            loading_info=loading,
            checkpoint_load_and_weight_quant_seconds=time.perf_counter() - started,
            adapters=24,
            weight_digests=[{"layer": a.index, "data": tensor_info(a.b), "scales": tensor_info(a.sfb)} for a in adapters],
        )
        save()
        print(json.dumps({"stage": "MODEL_LOADED", "parameters": parameters, "source_head": actual}), flush=True)
        tensors = {}
        with torch.inference_mode():
            if args.mode in ("validate", "diagnostic"):
                generator = torch.Generator().manual_seed(31012026)
                x = (torch.randn((13, 1024), generator=generator) * 0.3).to(torch.bfloat16).cuda()
                index = (torch.arange(13 * 8).reshape(13, 8) % 32).cuda()
                prob = torch.softmax(torch.randn((13, 8), generator=generator), dim=-1).to(torch.bfloat16).cuda()
                a = adapters[0]
                a.mode = "reference"
                ref = a.forward(x, index, prob).cpu()
                before = NATIVE_CALLS
                a.mode = "native"
                result = a.forward(x, index, prob).cpu()
                assert NATIVE_CALLS - before == 1
                diff = difference(result, ref)
                assert diff["relative_l2"] <= 0.01 and diff["max_abs"] <= 0.125, diff
                meta["operator_oracle"] = diff
                tensors["operator_native"] = result
                cases = [prompts[0], prompts[1], workloads["256"]]
                checks = []
                for i, ids in enumerate(cases):
                    outputs = {}
                    for backend in ("stock", "reference", "native"):
                        attach(adapters, backend)
                        before = NATIVE_CALLS
                        y = model(input_ids=ids.cuda(), use_cache=False).logits.cpu()
                        assert NATIVE_CALLS - before == (24 if backend == "native" else 0)
                        assert bool(torch.isfinite(y).all())
                        outputs[backend] = y
                        tensors[f"forward_{i}_{backend}"] = y
                    oracle = difference(outputs["native"], outputs["reference"])
                    bf16 = difference(outputs["native"], outputs["stock"])
                    oracle_pass = oracle["relative_l2"] <= 0.01
                    bf16_pass = bf16["relative_l2"] <= 0.05
                    checks.append(
                        {
                            "case": i,
                            "shape": list(ids.shape),
                            "quantized_oracle": oracle,
                            "original_bf16": bf16,
                            "oracle_gate_pass": oracle_pass,
                            "bf16_gate_pass": bf16_pass,
                            "native_logits": tensor_info(outputs["native"]),
                            "native_calls": 24,
                        }
                    )
                    meta["forward_checks"] = checks
                    save()
                    print(json.dumps({"stage": "FORWARD_CHECKED", "case": i, "oracle": oracle, "bf16": bf16}), flush=True)
                    if args.mode == "validate":
                        assert oracle_pass, oracle
                        assert bf16_pass, bf16
                attach(adapters, "native")
                generated = []
                for i, ids in enumerate(prompts):
                    r = request(model, ids.cuda(), 16)
                    assert r["native_calls"] == 24 * r["forwards"]
                    assert bool(torch.isfinite(r["logits"]).all())
                    tokens = r["ids"][0].tolist()
                    if tokenizer.eos_token_id in tokens:
                        tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    if args.mode == "validate":
                        assert text, ("empty generation", i)
                    generated.append(
                        {
                            "case": i,
                            "prompt": messages[i],
                            "text": text,
                            "tokens": r["ids"].tolist(),
                            "native_calls": r["native_calls"],
                            "logits": tensor_info(r["logits"]),
                            "quality_expected": "4" if i == 0 else "cobalt" if i == 2 else None,
                            "quality_contains_expected": ("4" in text) if i == 0 else ("cobalt" in text.lower()) if i == 2 else None,
                        }
                    )
                    tensors[f"generation_{i}_ids"] = r["ids"]
                    tensors[f"generation_{i}_logits"] = r["logits"]
                meta["generations"] = generated
                if args.mode == "diagnostic":
                    # Compatibility/source regression only. Failed strict quality
                    # gates remain failed; no formal timing or speedup evidence.
                    meta["reference_quality_pass"] = all(c["oracle_gate_pass"] and c["bf16_gate_pass"] for c in checks)
                    requests = []
                    for length, ids in workloads.items():
                        r = request(model, ids.cuda(), 32)
                        assert r["native_calls"] == 24 * 32
                        assert bool(torch.isfinite(r["logits"]).all())
                        requests.append(
                            {
                                "input_tokens": int(length),
                                "new_tokens": 32,
                                "ids": tensor_info(r["ids"]),
                                "logits": tensor_info(r["logits"]),
                                "token_ids": r["ids"].tolist(),
                                "text": tokenizer.decode(r["ids"][0], skip_special_tokens=True),
                                "native_calls": r["native_calls"],
                                "request_ms_diagnostic_only": r["request_ms"],
                                "prefill_ms_diagnostic_only": r["prefill_ms"],
                                "decode_ms_diagnostic_only": r["decode_ms"],
                            }
                        )
                        tensors[f"request_{length}_ids"] = r["ids"]
                        tensors[f"request_{length}_logits"] = r["logits"]
                    meta["e2e_requests"] = requests
                    meta["timing_scope"] = (
                        "Single diagnostic observation per workload; not formal paired performance, no gain/throughput claim while reference quality gates fail"
                    )
                torch.save(tensors, out / "outputs.pt")
            else:
                attach(adapters, "native")
                ids = workloads[str(args.length)].cuda()
                for _ in range(2):
                    warm = request(model, ids, 32)
                    assert warm["native_calls"] == 24 * 32
                reference_ids = warm["ids"].clone()
                reference_logits = warm["logits"].clone()
                torch.cuda.reset_peak_memory_stats()
                rows = []
                for i in range(args.repeats):
                    r = request(model, ids, 32)
                    assert r["native_calls"] == 24 * 32
                    assert torch.equal(r["ids"], reference_ids) and torch.equal(r["logits"], reference_logits), "Output drift within a frozen process"
                    assert bool(torch.isfinite(r["logits"]).all())
                    rows.append({k: r[k] for k in ("prefill_ms", "decode_ms", "request_ms", "native_calls", "forwards")})
                    print(json.dumps({"stage": "TIMED_REQUEST", "index": i, **rows[-1]}), flush=True)
                meta.update(
                    length=args.length,
                    new_tokens=32,
                    rows=rows,
                    medians={k: statistics.median(v[k] for v in rows) for k in ("prefill_ms", "decode_ms", "request_ms")},
                    ids=tensor_info(reference_ids),
                    logits=tensor_info(reference_logits),
                    generated_text=tokenizer.decode(reference_ids[0], skip_special_tokens=True),
                    peak_torch_allocated=torch.cuda.max_memory_allocated(),
                    peak_torch_reserved=torch.cuda.max_memory_reserved(),
                )
                torch.save({"ids": reference_ids, "logits": reference_logits}, out / "outputs.pt")
        meta.update(
            stage="DIAGNOSTIC_COMPLETE" if args.mode == "diagnostic" else "PASS",
            total_native_calls=NATIVE_CALLS,
            native_plans=NATIVE_PLANS,
            adapter_calls=[a.calls for a in adapters],
            compile_cache_entries=len(api._cache_of_GroupedGemmSwigluSm100Objects),
        )
        save()
        (out / "result.json").write_text(json.dumps(meta, indent=2, default=json_default))
        print(json.dumps({"stage": meta["stage"], "mode": args.mode, "native_calls": NATIVE_CALLS}), flush=True)
    except BaseException as exc:
        meta.update(stage="FAILED", error=repr(exc), total_native_calls=NATIVE_CALLS)
        save()
        raise


if __name__ == "__main__":
    main()
