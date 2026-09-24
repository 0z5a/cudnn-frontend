# SPDX-FileCopyrightText: Copyright (c) 2026 0z5a
# SPDX-License-Identifier: Apache-2.0

import json
import math
import statistics
import sys

import cudnn
import torch
from cudnn.sdpa.fwd.engines import engine_name


mode = sys.argv[1]
out_dir = sys.argv[2]
cases = [(torch.float16, 64, 64), (torch.bfloat16, 128, 256), (torch.float16, 128, 128)]
if len(sys.argv) > 3 and sys.argv[3] == "long":
    cases = [(torch.float16, 512, 512), (torch.float16, 1024, 1024), (torch.bfloat16, 1024, 1024)]
records = []
for dtype, sq, skv in cases:
    torch.manual_seed(620 + sq + skv + (1 if dtype == torch.bfloat16 else 0))
    d = 128
    q = torch.randn(1, sq, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(1, skv, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(1, skv, 8, d, device="cuda", dtype=dtype).transpose(1, 2)
    o = torch.empty_like(q)
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = (graph.tensor_like(t, name=name) for t, name in ((q, "q"), (k, "k"), (v, "v")))
    to, _ = graph.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=1 / math.sqrt(d))
    to.set_output(True).set_dim(o.shape).set_stride(o.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    name = engine_name(arch="sm120")
    index = next(i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).startswith(name + "["))
    graph.select_plan(index)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    pack = {tq: q, tk: k, tv: v, to: o}
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=1 / math.sqrt(d))
    torch.testing.assert_close(o.float(), expected, atol=0.1, rtol=0.05)
    tag = f"{dtype}-{sq}-{skv}"
    if mode == "baseline":
        torch.save(o.cpu(), f"{out_dir}/{tag}.pt")
    else:
        baseline = torch.load(f"{out_dir}/{tag}.pt", weights_only=True)
        assert torch.equal(o.cpu(), baseline), tag
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        for _ in range(64):
            graph.execute(pack, workspace)
    for _ in range(4):
        captured.replay()
    torch.cuda.synchronize()
    samples_us = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        captured.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / 64)
    records.append({"tag": tag, "plan": graph.get_plan_name_at_index(index), "median_us": statistics.median(samples_us), "samples_us": samples_us})
print(json.dumps({"mode": mode, "cudnn_path": cudnn.__file__, "records": records}, indent=2))
