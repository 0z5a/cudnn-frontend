# SPDX-FileCopyrightText: Copyright (c) 2026 0z5a
# SPDX-License-Identifier: Apache-2.0

import math
import logging

import cudnn
import torch
from cudnn.engines import manifest
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, analyze_for, engine_name

logging.basicConfig(level=logging.INFO)


def run(d, dtype):
    torch.manual_seed(620)
    scale = 1 / math.sqrt(d)
    q = torch.randn(1, 128, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(1, 128, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(1, 128, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    o = torch.empty(1, 128, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tq = graph.tensor_like(q, name="q")
    tk = graph.tensor_like(k, name="k")
    tv = graph.tensor_like(v, name="v")
    to, _ = graph.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=scale)
    to.set_output(True).set_dim(o.shape).set_stride(o.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    name = engine_name(arch="sm120")
    plans = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    print("available_plans", plans, flush=True)
    spec = next(spec for spec in ENGINE_SPECS if spec.name == name)
    _, reason = analyze_for(spec, graph)
    print("frost_opt_in", manifest.opt_in_engines_enabled(), "eligibility", reason,
          "candidates", [engine.name for engine in manifest.engines_for(graph)], flush=True)
    index = next(i for i, plan in enumerate(plans) if plan == name or plan.startswith(name + "["))
    graph.select_plan(index)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    graph.execute({tq: q, tk: k, tv: v, to: o}, workspace)
    torch.cuda.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    torch.testing.assert_close(o.float(), expected, atol=0.1, rtol=0.05)
    print(dtype, d, plans[index], "passed", "workspace", graph.get_workspace_size(), flush=True)


for dtype in (torch.float16, torch.bfloat16):
    for d in (128, 256):
        run(d, dtype)
