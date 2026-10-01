# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native and independent reference on each identical real-model layer input."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("a3_e2e", Path(__file__).with_name("a3_model_e2e_20261001.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
t = m.torch
t.set_num_threads(4)
t.backends.cuda.matmul.allow_tf32 = False
t.backends.cudnn.allow_tf32 = False
t.set_float32_matmul_precision("highest")
tokenizer = m.AutoTokenizer.from_pretrained(m.MODEL, local_files_only=True, trust_remote_code=False)
model = (
    m.AutoModelForCausalLM.from_pretrained(m.MODEL, dtype=t.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).eval().cuda()
)
adapters = [m.A3ExpertAdapter(e, i, (256, 256)) for i, e in enumerate(model.modules()) if isinstance(e, m.GraniteMoeExperts)]
captured, records = {}, []
original = m.api.grouped_gemm_swiglu_wrapper_sm100


def capture(**kw):
    output = original(**kw)
    captured.update(result=output, args=kw)
    return output


m.api.grouped_gemm_swiglu_wrapper_sm100 = capture


def comparison(adapter):
    def call(x, index, prob):
        adapter.mode = "native"
        native = adapter.forward(x, index, prob)
        args, output = captured["args"], captured["result"]
        adapter.mode = "reference"
        ref = adapter.forward(x, index, prob)
        diff = m.difference(native.cpu(), ref.cpu())
        padded, _, _, _, _, sizes, aligned = adapter.routing(x, index)
        aq, _, ascale = m.quantize_mxfp8(padded)
        adeq = m.dequantize(aq, ascale)
        dnative = m.dequantize_wrapper_d(output, adapter.norm)
        groups, start = [], 0
        for expert, rows in enumerate(sizes):
            if rows:
                cref = m.F.linear(adeq[start : start + rows], adapter.bref[expert])
                pairs = cref.reshape(rows, -1, 2, 32)
                dref = (m.F.silu(pairs[:, :, 0]) * pairs[:, :, 1]).reshape(rows, 512)
                dc = m.difference(output["c_tensor"][start : start + rows].cpu(), cref.cpu())
                dd = m.difference(dnative[start : start + rows].cpu(), dref.cpu())
                groups.append({"expert": expert, "rows": rows, "c": dc, "d": dd})
            start += aligned[expert]
        record = {
            "layer_module_index": adapter.index,
            "ffn": diff,
            "input_max": x.abs().max().item(),
            "input_shape": list(x.shape),
            "counts": sizes,
            "groups": groups,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "layer": adapter.index,
                    "ffn": diff,
                    "worst_c_l2": max(g["c"]["relative_l2"] for g in groups),
                    "worst_d_l2": max(g["d"]["relative_l2"] for g in groups),
                }
            ),
            flush=True,
        )
        adapter.mode = "native"
        return native

    return call


for adapter in adapters:
    adapter.expert.forward = comparison(adapter)
messages, prompts, workloads = m.load_inputs(tokenizer)
with t.inference_mode():
    logits = model(input_ids=prompts[0].cuda(), use_cache=False).logits.cpu()
out = m.ROOT / "results/a3-e2e-real-layer-diagnosis-20261001"
out.mkdir(exist_ok=False)
(out / "result.json").write_text(
    json.dumps({"scope": __doc__, "stage": "DIAGNOSTIC_COMPLETE", "layers": records, "logits": m.tensor_info(logits), "native_plans": m.NATIVE_PLANS}, indent=2)
)
