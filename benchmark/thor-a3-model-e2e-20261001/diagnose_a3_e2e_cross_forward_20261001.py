# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate the first cross-forward difference and test each oracle's self stability."""

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
experts = [e for e in model.modules() if isinstance(e, m.GraniteMoeExperts)]
adapters = [m.A3ExpertAdapter(e, i, (256, 256)) for i, e in enumerate(experts)]
messages, prompts, _ = m.load_inputs(tokenizer)
saved, records = {}, []


def traced(adapter, label):
    def call(x, index, prob):
        saved[(label, adapter.index)] = {"x": x.detach().cpu().clone(), "index": index.detach().cpu().clone(), "prob": prob.detach().cpu().clone()}
        y = adapter.forward(x, index, prob)
        saved[(label, adapter.index)]["y"] = y.detach().cpu().clone()
        return y

    return call


with t.inference_mode():
    outputs = {}
    for label, backend in (("stock", "stock"), ("reference1", "reference"), ("reference2", "reference"), ("native1", "native"), ("native2", "native")):
        for adapter in adapters:
            adapter.attach(backend)
            if backend != "stock":
                adapter.expert.forward = traced(adapter, label)
        outputs[label] = model(input_ids=prompts[0].cuda(), use_cache=False).logits.cpu().clone()
        if label != "stock":
            print(json.dumps({"label": label, "reference1_difference": m.difference(outputs[label], outputs["reference1"])}), flush=True)
    for index in range(24):
        r, n = saved[("reference1", index)], saved[("native1", index)]
        records.append(
            {
                "layer": index,
                "x": m.difference(n["x"], r["x"]),
                "y": m.difference(n["y"], r["y"]),
                "changed_routes": (r["index"] != n["index"]).sum().item(),
                "total_routes": r["index"].numel(),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    report = {
        "stage": "DIAGNOSTIC_COMPLETE",
        "scope": __doc__,
        "layers": records,
        "full_outputs": {k: m.difference(v, outputs["reference1"]) for k, v in outputs.items()},
    }
out = m.ROOT / "results/a3-e2e-cross-forward-diagnosis-20261001"
out.mkdir(exist_ok=False)
(out / "result.json").write_text(json.dumps(report, indent=2))
t.save(outputs, out / "outputs.pt")
