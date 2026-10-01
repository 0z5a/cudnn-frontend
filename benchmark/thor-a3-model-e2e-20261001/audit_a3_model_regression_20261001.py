# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit saved CPU tensors; preserve failed model-quality gates and timing scope."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
from pathlib import Path
import subprocess

import torch

ROOT = Path("/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928")
OUT = ROOT / "results/a3-model-e2e-regression-20261001"
HEADS = {"control": "83374af96ec3f8e22dd159d278bdf25acab17a4e", "candidate": "209d9e61c6ca873ccf5b869e83db00eed9f8f966"}
torch.set_num_threads(4)


def read(path):
    return json.loads(path.read_text())


def info(tensor):
    assert tensor.device.type == "cpu"
    raw = tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def difference(actual, reference):
    a, r = actual.float(), reference.float()
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    diff = a - r
    return {"max_abs": diff.abs().max().item(), "relative_l2": (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(r).clamp_min(1e-12)).item()}


def main():
    assert read(OUT / "status.json")["stage"] == "SOURCE_REGRESSION_PASS_REFERENCE_QUALITY_FAILED"
    runs = read(OUT / "runs.json")
    assert len(runs) == 2 and all(row["returncode"] == 0 for row in runs)
    pre, post = read(OUT / "source-pre.json"), read(OUT / "source-post.json")
    assert pre == post and {label: value["head"] for label, value in pre.items()} == HEADS
    metadata = {label: read(OUT / label / "result.json") for label in HEADS}
    saved = {label: torch.load(OUT / label / "outputs.pt", map_location="cpu", weights_only=True) for label in HEADS}
    a, b = metadata["control"], metadata["candidate"]
    assert read(OUT / "control/input-token-ids.json") == read(OUT / "candidate/input-token-ids.json")
    for key in ("inputs", "weight_digests", "checkpoint", "full_config", "parameters", "harness_sha256"):
        assert a[key] == b[key], key
    assert a["parameters"] == 1334628352 and a["adapters"] == b["adapters"] == 24
    assert a["checkpoint"]["revision"] == "0da7a48b0276d500ce5922fd2b33944091fc6c09"
    assert a["tile"] == b["tile"] == [256, 256]
    assert saved["control"].keys() == saved["candidate"].keys() and len(saved["control"]) == 20
    tensor_checks = []
    for name in sorted(saved["control"]):
        x, y = saved["control"][name], saved["candidate"][name]
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8)), name
        tensor_checks.append({"name": name, "same_bytes": True, **info(x)})
    gates = []
    for label, meta in metadata.items():
        assert meta["source_head"] == HEADS[label] and meta["stage"] == "DIAGNOSTIC_COMPLETE"
        assert meta["total_native_calls"] == 1 + 3 * 24 + 3 * 16 * 24 + 2 * 32 * 24 == 2761
        assert meta["reference_quality_pass"] is False
        assert not any(meta["loading_info"].values())
        assert meta["service"] == "inactive"
        for name, entry in meta["imports"].items():
            if name.startswith("cudnn"):
                assert Path(entry["path"]).is_relative_to(Path(meta["source"]) / "python")
                assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        assert len(meta["native_plans"]) == 1
        for name, entry in meta["native_plans"].items():
            assert name.endswith(".BlockScaledMoEGroupedGemmGluBiasKernel")
            assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        for check in meta["forward_checks"]:
            i = check["case"]
            native = saved[label][f"forward_{i}_native"]
            assert info(native) == check["native_logits"] and check["native_calls"] == 24
            q = difference(native, saved[label][f"forward_{i}_reference"])
            stock = difference(native, saved[label][f"forward_{i}_stock"])
            assert q["max_abs"] == check["quantized_oracle"]["max_abs"]
            assert stock["max_abs"] == check["original_bf16"]["max_abs"]
            # CPU/GPU reduction ordering can differ; the preregistered quality
            # thresholds stay 1% and 5%, with no tolerance waiver.
            assert abs(q["relative_l2"] - check["quantized_oracle"]["relative_l2"]) < 1e-5
            assert abs(stock["relative_l2"] - check["original_bf16"]["relative_l2"]) < 1e-5
            assert q["relative_l2"] > 0.01 and stock["relative_l2"] > 0.05
            assert check["oracle_gate_pass"] is False and check["bf16_gate_pass"] is False
            gates.append(
                {"source": label, "case": i, "quantized_oracle": q, "original_bf16": stock, "oracle_threshold": 0.01, "bf16_threshold": 0.05, "passed": False}
            )
        for row in meta["generations"]:
            i = row["case"]
            assert saved[label][f"generation_{i}_ids"].tolist() == row["tokens"]
            assert info(saved[label][f"generation_{i}_logits"]) == row["logits"]
            assert row["native_calls"] == 384
        for row in meta["e2e_requests"]:
            length = row["input_tokens"]
            assert row["new_tokens"] == 32 and row["native_calls"] == 768
            assert info(saved[label][f"request_{length}_ids"]) == row["ids"]
            assert info(saved[label][f"request_{length}_logits"]) == row["logits"]
            assert saved[label][f"request_{length}_ids"].tolist() == row["token_ids"]
    assert [(x["tokens"], x["text"]) for x in a["generations"]] == [(x["tokens"], x["text"]) for x in b["generations"]]
    assert [(x["token_ids"], x["text"]) for x in a["e2e_requests"]] == [(x["token_ids"], x["text"]) for x in b["e2e_requests"]]
    assert subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip() == "inactive"
    assert not torch.cuda.is_initialized()
    result = {
        "stage": "SOURCE_REGRESSION_PASS_REFERENCE_QUALITY_FAILED",
        "source_heads": HEADS,
        "actual_child_return_codes": [r["returncode"] for r in runs],
        "same_saved_tensor_bytes": tensor_checks,
        "same_all_generated_text": True,
        "full_model_parameters": a["parameters"],
        "native_calls_per_source": 2761,
        "quality_checks": gates,
        "quality_pass": False,
        "formal_performance": "NOT_RUN_AFTER_QUALITY_GATE_FAILURE",
        "service": "inactive",
        "cuda_initialized_by_auditor": False,
    }
    (OUT / "independent-audit.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "same_saved_tensors": len(tensor_checks),
                "native_calls_per_source": 2761,
                "quality_pass": False,
                "cuda_initialized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
