# SPDX-FileCopyrightText: Copyright (c) 2026 0z5a
# SPDX-License-Identifier: Apache-2.0

"""Profile one compiled FROST SDPA launch after graph setup and warmup."""

import math
import sys

import cudnn
import torch
from cudnn.sdpa.fwd.engines import engine_name


d = int(sys.argv[1])
torch.manual_seed(620)
q = torch.randn(1, 128, 8, d, device="cuda", dtype=torch.float16).transpose(1, 2)
k = torch.randn(1, 128, 8, d, device="cuda", dtype=torch.float16).transpose(1, 2)
v = torch.randn(1, 128, 8, d, device="cuda", dtype=torch.float16).transpose(1, 2)
o = torch.empty_like(q)
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)
tq = graph.tensor_like(q, name="q")
tk = graph.tensor_like(k, name="k")
tv = graph.tensor_like(v, name="v")
to, _ = graph.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=1 / math.sqrt(d))
to.set_output(True).set_dim(o.shape).set_stride(o.stride())
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
name = engine_name(arch="sm120")
index = next(i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).startswith(name + "["))
print(graph.get_plan_name_at_index(index), flush=True)
graph.select_plan(index)
graph.check_support()
graph.build_plans()
workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
pack = {tq: q, tk: k, tv: v, to: o}
for _ in range(5):
    graph.execute(pack, workspace)
torch.cuda.synchronize()
torch.cuda.profiler.start()
graph.execute(pack, workspace)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=1 / math.sqrt(d))
torch.testing.assert_close(o.float(), expected, atol=0.1, rtol=0.05)
print("passed", flush=True)
