# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-wrapper capture must rebind live routing and reset its captured AMAX."""

import pytest
import torch
from cuda.bindings import driver as cuda

from gemm.cutedsl.test_grouped_swiglu_wrapper_memo import api, assert_same, call, operands, written

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_wrapper_capture_replays_live_inputs_and_amax(api, canonical, d_dtype):
    inputs = operands(canonical)
    eager = call(api, inputs, d_dtype=d_dtype)
    eager_snapshot = written(eager, 2048)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        call(api, inputs, d_dtype=d_dtype, current_stream=cuda.CUstream(side.cuda_stream))
        with torch.cuda.graph(graph, stream=side):
            captured = call(api, inputs, d_dtype=d_dtype, current_stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.current_stream().wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
    assert_same(written(captured, 2048), eager_snapshot)
    if captured["amax_tensor"] is not None:
        assert bool((captured["amax_tensor"] > 0).all())

    inputs["padded_offsets_tensor"].copy_(torch.tensor([0, 1024, 1536, 2048], dtype=torch.int32, device="cuda"))
    inputs["alpha_tensor"].zero_()
    inputs["prob_tensor"].fill_(0.5)
    inputs["norm_const_tensor"].mul_(2.0)
    for _ in range(3):
        graph.replay()
    cold = call(api, inputs, d_dtype=d_dtype)
    torch.cuda.synchronize()
    assert_same(written(captured, 2048), written(cold, 2048))
    assert_same(written(eager, 2048), eager_snapshot)
    if captured["amax_tensor"] is not None:
        assert bool(torch.isneginf(captured["amax_tensor"][0]).all())
        assert torch.count_nonzero(captured["amax_tensor"][1:]) == 0
