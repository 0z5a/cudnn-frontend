# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn._torch_stream``: the one place a raw stream handle becomes a torch stream (Rule 5).

Default-stream sentinels (0, cudaStreamLegacy, cudaStreamPerThread) and torch's own
default stream map to ``torch.cuda.default_stream``; torch's current stream maps to
itself; only a genuine side stream is wrapped in ``torch.cuda.ExternalStream``.
"""

import pytest
import torch

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn._torch_stream import DEFAULT_STREAM_HANDLES, as_torch_stream, stream_context

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.fixture
def no_external_stream(monkeypatch):
    def refuse(handle, device=None):
        raise AssertionError(f"torch.cuda.ExternalStream({handle}) built for a default-stream handle")

    monkeypatch.setattr(torch.cuda, "ExternalStream", refuse)


@pytest.mark.parametrize("handle", sorted(DEFAULT_STREAM_HANDLES) + ["torch_default"])
def test_default_stream_handles_map_to_torch_default_stream(no_external_stream, handle):
    default = torch.cuda.default_stream()
    if handle == "torch_default":
        handle = default.cuda_stream
    assert as_torch_stream(handle) == default
    assert as_torch_stream(handle, torch.device("cuda", torch.cuda.current_device())) == default
    with stream_context(handle):
        assert torch.cuda.current_stream() == default
    with stream_context(handle, verify_current=True):
        assert torch.cuda.current_stream() == default


def test_side_stream_handle_is_that_stream():
    side = torch.cuda.Stream()
    assert as_torch_stream(side.cuda_stream).cuda_stream == side.cuda_stream
    assert as_torch_stream(side) is side
    with stream_context(side.cuda_stream):
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
    with stream_context(side.cuda_stream, verify_current=True):
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
    assert torch.cuda.current_stream() == torch.cuda.default_stream()


def test_current_stream_handle_needs_no_wrapper(no_external_stream):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert as_torch_stream(side.cuda_stream) == side
        with stream_context(side.cuda_stream):
            assert torch.cuda.current_stream() == side


@pytest.mark.parametrize("use_default", [False, True])
def test_current_raw_handle_needs_no_stream_context(monkeypatch, use_default):
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        pytest.skip("torch raw stream getter is unavailable")
    current = torch.cuda.default_stream() if use_default else torch.cuda.Stream()
    with torch.cuda.stream(current):

        def no_context(*args, **kwargs):
            raise AssertionError("the current raw stream should not rebuild a torch stream context")

        monkeypatch.setattr(torch.cuda, "stream", no_context)
        with stream_context(current.cuda_stream, current.device):
            assert torch.cuda.current_stream() == current


def test_matching_default_handle_does_not_hide_wrong_stream_device(monkeypatch):
    # The device check is a host contract: a synthetic stream lets a single-GPU
    # worker exercise equal default handles on different devices without launch.
    class OtherDeviceStream:
        cuda_stream = 0
        device = torch.device("cuda", 1)

    monkeypatch.setattr(torch.cuda, "Stream", OtherDeviceStream)
    with pytest.raises(ValueError, match="stream must be on cuda:0, got cuda:1"):
        with stream_context(OtherDeviceStream(), torch.device("cuda", 0)):
            pytest.fail("wrong-device stream was admitted")


def test_none_is_a_no_op():
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), stream_context(None):
        assert torch.cuda.current_stream() == side


def test_torch_stream_on_another_device_is_rejected():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two devices")
    other = torch.cuda.Stream(device=1)
    with pytest.raises(ValueError, match="must be on cuda:0"):
        as_torch_stream(other, torch.device("cuda", 0))
