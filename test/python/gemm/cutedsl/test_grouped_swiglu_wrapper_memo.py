# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU metadata reuse must preserve live operands, output ownership and streams."""

import pytest
import torch
from cuda.bindings import driver as cuda

from gemm.cutedsl.test_grouped_gemm_wrapper_memo import block_scaled_result, mxfp8_inputs, raw_bytes

pytestmark = pytest.mark.L0


@pytest.fixture
def api(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ is required")
    from cudnn.gemm.cutedsl.grouped.swiglu import api as module

    monkeypatch.setattr(module, "_cache_of_GroupedGemmSwigluSm100Objects", {})
    # Also allows the two behavioral detectors to run against the old wrapper.
    monkeypatch.setattr(module, "_swiglu_wrapper_memo", {}, raising=False)
    return module


def operands(canonical=False, tensor_m=2048, experts=4):
    inputs = mxfp8_inputs([tensor_m // experts] * experts, l=experts, tensor_m=tensor_m)
    if canonical:
        inputs["a_tensor"] = inputs["a_tensor"].squeeze(-1)
        inputs["b_tensor"] = inputs["b_tensor"].permute(2, 0, 1)
        for key in ("sfa_tensor", "sfb_tensor"):
            inputs[key] = inputs[key].permute(5, 2, 4, 0, 1, 3).view(-1)
        inputs["prob_tensor"] = inputs["prob_tensor"].view(-1)
    inputs["alpha_tensor"].fill_(1.0)
    inputs["prob_tensor"].fill_(0.75)
    return inputs


def call(api, inputs, **options):
    kwargs = {key: inputs[key] for key in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "alpha_tensor", "prob_tensor", "norm_const_tensor")}
    kwargs.update(padded_offsets=inputs["padded_offsets_tensor"], sf_vec_size=32, c_dtype=torch.bfloat16, d_dtype=torch.bfloat16)
    kwargs.update(options)
    return api.grouped_gemm_swiglu_wrapper_sm100(**kwargs)


def written(outputs, valid_m):
    return {name: tensor.clone() for name, tensor in block_scaled_result(outputs, valid_m).items()}


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(raw_bytes(actual[name]), raw_bytes(expected[name])), name


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_hit_binds_fresh_inputs_and_preserves_returned_outputs(api, monkeypatch, canonical, d_dtype):
    first_inputs = operands(canonical)
    second_inputs = operands(canonical)
    second_inputs["alpha_tensor"].fill_(0.25)
    second_inputs["prob_tensor"].fill_(0.5)
    first = call(api, first_inputs, d_dtype=d_dtype)
    snapshot = written(first, 2048)

    # An identity cache or a wrapper that repeats dtype/geometry derivation is
    # independently detected, even if it produces numerically correct outputs.
    def no_derivation(*args, **kwargs):
        raise AssertionError("warm wrapper repeated dtype derivation")

    with monkeypatch.context() as patch:
        patch.setattr(api, "_convert_to_cutlass_data_type", no_derivation)
        second = call(api, second_inputs, d_dtype=d_dtype)
    assert tuple(second.keys()) == tuple(first.keys())
    assert tuple(second) == tuple(second.values())
    for name in first.keys():
        a, b = first[name], second[name]
        if a is None:
            assert b is None
        else:
            assert (a.shape, a.stride(), a.dtype) == (b.shape, b.stride(), b.dtype), name
            assert a.data_ptr() != b.data_ptr(), name
    warm = written(second, 2048)
    api._swiglu_wrapper_memo.clear()
    cold = written(call(api, second_inputs, d_dtype=d_dtype), 2048)
    torch.cuda.synchronize()
    assert_same(warm, cold)
    assert_same(written(first, 2048), snapshot)
    assert not torch.equal(raw_bytes(snapshot["d_tensor"]), raw_bytes(warm["d_tensor"]))


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("handle", ["side", 0, 1, 2, None])
def test_allocations_and_amax_reset_use_launch_stream(api, monkeypatch, warm, handle):
    inputs = operands()
    if warm:
        call(api, inputs)
    origin = torch.cuda.Stream()
    origin.wait_stream(torch.cuda.current_stream())
    target = torch.cuda.Stream() if handle == "side" else (origin if handle is None else torch.cuda.default_stream())
    target.wait_stream(torch.cuda.current_stream())
    launch = cuda.CUstream(target.cuda_stream) if handle == "side" else (None if handle is None else cuda.CUstream(handle))
    allocations = []

    def guard(function):
        def allocate(*args, **kwargs):
            # A pool Stream and an ExternalStream can wrap the same CUDA handle
            # while comparing unequal as torch objects. CUDA ordering is by the
            # underlying handle on its device, not torch's internal stream ID.
            actual = torch.cuda.current_stream(inputs["a_tensor"].device)
            assert (actual.cuda_stream, actual.device) == (target.cuda_stream, target.device), "allocation/reset is unordered with launch"
            allocations.append(function.__name__)
            return function(*args, **kwargs)

        return allocate

    with monkeypatch.context() as patch:
        for name in ("empty", "empty_strided", "full"):
            patch.setattr(torch, name, guard(getattr(torch, name)))
        if handle != "side":

            def no_external(*args, **kwargs):
                raise AssertionError("default/current stream must not use ExternalStream")

            patch.setattr(torch.cuda, "ExternalStream", no_external)
        with torch.cuda.stream(origin):
            old_mode = torch.cuda.get_sync_debug_mode()
            if warm:
                torch.cuda.set_sync_debug_mode("error")
            try:
                result = call(api, inputs, current_stream=launch)
            finally:
                torch.cuda.set_sync_debug_mode(old_mode)
    assert "full" in allocations
    torch.cuda.current_stream().wait_stream(target)
    reference = call(api, inputs)
    torch.cuda.synchronize()
    assert_same(written(result, 2048), written(reference, 2048))


@pytest.mark.parametrize("experts", [1, 31, 32, 33])
def test_routing_reuse_and_per_call_amax_on_side_stream(api, experts):
    inputs = operands(tensor_m=256 * experts, experts=experts)
    first = call(api, inputs, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1))
    snapshot = written(first, 256 * experts)
    plans = tuple(id(entry[0]) for entry in api._cache_of_GroupedGemmSwigluSm100Objects.values())
    if experts > 1:
        offsets = torch.tensor([0] + [256 * (i + 1) for i in range(1, experts)], dtype=torch.int32, device="cuda")
        inputs["padded_offsets_tensor"].copy_(offsets)
    inputs["alpha_tensor"].zero_()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    warm = call(api, inputs, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1), current_stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.current_stream().wait_stream(side)
    assert tuple(id(entry[0]) for entry in api._cache_of_GroupedGemmSwigluSm100Objects.values()) == plans
    api._swiglu_wrapper_memo.clear()
    cold = call(api, inputs, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1))
    torch.cuda.synchronize()
    assert_same(written(warm, 256 * experts), written(cold, 256 * experts))
    assert_same(written(first, 256 * experts), snapshot)
    if experts > 1:
        assert bool(torch.isneginf(warm["amax_tensor"][0]).all())
    assert torch.count_nonzero(warm["amax_tensor"][1:] if experts > 1 else warm["amax_tensor"]) == 0


def test_m_change_reuses_compiled_plan_and_cache_clear_rebuilds(api, monkeypatch):
    first = call(api, operands(tensor_m=1024))
    plans = tuple(id(entry[0]) for entry in api._cache_of_GroupedGemmSwigluSm100Objects.values())
    inputs = operands(tensor_m=2048)
    second = call(api, inputs)
    assert first["d_tensor"].shape[0] == 1024 and second["d_tensor"].shape[0] == 2048
    assert tuple(id(entry[0]) for entry in api._cache_of_GroupedGemmSwigluSm100Objects.values()) == plans
    api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    calls = []
    original = api.GroupedGemmSwigluSm100.check_support

    def support(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(api.GroupedGemmSwigluSm100, "check_support", support)
    third = call(api, inputs)
    assert calls, "clearing compiled plans must repeat support checking"
    torch.cuda.synchronize()
    assert_same(written(second, 2048), written(third, 2048))


def test_configuration_change_does_not_bypass_support(api):
    inputs = operands()
    call(api, inputs, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1))
    with pytest.raises(NotImplementedError, match="fp8 ab_dtype with mma_tiler_mn.*128 and fp8 d_dtype is not supported"):
        call(api, inputs, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1), d_dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="cd_major must be 'n'"):
        call(api, inputs, cd_major="m")


def test_overlap_margin_rebuilds_plan(api, monkeypatch):
    inputs = operands()
    plans = []
    for margin in ("0", "1", "0"):
        monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", margin)
        call(api, inputs)
        key, _ = next(value for key, value in api._swiglu_wrapper_memo.items() if key[-2] == margin)
        plan = api._cache_of_GroupedGemmSwigluSm100Objects[key][0]
        assert plan.num_cluster_overlap_margin == int(margin)
        plans.append(plan)
    assert plans[0] is plans[2] and plans[0] is not plans[1]
