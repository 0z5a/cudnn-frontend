# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic poisoned-output detector for the model adapter's row-SFD need."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("a3_e2e", Path(__file__).with_name("a3_model_e2e_20261001.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
t = m.torch
t.set_num_threads(4)
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig

t.manual_seed(31012026)
e = m.GraniteMoeExperts(GraniteMoeConfig(hidden_size=1024, intermediate_size=512, num_local_experts=32, num_experts_per_tok=8)).to(dtype=t.bfloat16)
with t.no_grad():
    e.gate_up_proj.normal_(std=0.02)
    e.down_proj.normal_(std=0.02)
e = e.cuda()
captured = {}
original = m.api.grouped_gemm_swiglu_wrapper_sm100


def capture(**kw):
    output = original(**kw)
    captured["result"] = output
    return output


def poison_allocation(original):
    def allocate(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("dtype") == t.float8_e8m0fnu:
            result.view(t.uint8).fill_(255)
        return result

    return allocate


m.api.grouped_gemm_swiglu_wrapper_sm100 = capture
m.dequantize_wrapper_d = lambda result, norm: result["d_tensor"]
records = []
with t.inference_mode():
    x = (t.randn((13, 1024)) * 0.3).to(t.bfloat16).cuda()
    index = (t.arange(104).reshape(13, 8) % 32).cuda()
    prob = t.softmax(t.randn((13, 8)), dim=-1).to(t.bfloat16).cuda()
    for tile in (256, 128):
        adapter = m.A3ExpertAdapter(e, 0, (256, tile))
        _, _, destinations, _, _, _, _ = adapter.routing(x, index)
        empty, empty_strided = t.empty, t.empty_strided
        try:
            t.empty, t.empty_strided = poison_allocation(empty), poison_allocation(empty_strided)
            result = adapter.forward(x, index, prob)
            t.cuda.synchronize()
        finally:
            t.empty, t.empty_strided = empty, empty_strided
        sf = captured["result"]["sfd_row_tensor"]
        rows = captured["result"]["d_tensor"].shape[0]
        logical = sf.view(t.uint8).reshape(1, rows // 128, 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(rows, 16)
        live = logical[destinations]
        remaining = (live == 255).sum().item()
        if tile == 256:
            assert remaining == 0
        else:
            assert remaining == live.numel(), (remaining, live.numel())
        records.append(
            {
                "tile": [256, tile],
                "live_scale_elements": live.numel(),
                "unwritten_poison_elements": remaining,
                "d_finite": bool(t.isfinite(captured["result"]["d_tensor"]).all()),
                "production_support_accepted": True,
                "model_row_sfd": "VALID" if remaining == 0 else "UNWRITTEN_DECLINE_MODEL_INTEGRATION",
            }
        )
out = m.ROOT / "results/a3-e2e-sfd-poison-20261001"
out.mkdir(exist_ok=False)
report = {"stage": "DETECTOR_CONFIRMED", "scope": __doc__, "records": records, "source": m.cudnn.__file__, "harness_sha256": m.sha_file(m.__file__)}
(out / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
