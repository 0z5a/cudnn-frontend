# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete same-source-recipe E2E comparison without waiving failed quality gates."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path("/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928")
OUT = ROOT / "results/a3-model-e2e-regression-20261001"
MODEL = Path("/home/jwipc/models/granite-3.1-1b-a400m-instruct-0da7a48b")
SOURCES = {
    "control": ("a3wrappercontrol", "83374af96ec3f8e22dd159d278bdf25acab17a4e"),
    "candidate": ("a3wrappermemo2", "209d9e61c6ca873ccf5b869e83db00eed9f8f966"),
}
HARNESS = ROOT / "tools/a3_model_e2e_20261001.py"
OUT.mkdir(exist_ok=False)
records = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def state(stage, **details):
    s = {"stage": stage, "pid": os.getpid(), "time": time.time(), "completed_children": len(records), **details}
    (OUT / "status.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s), flush=True)


def freeze(source):
    return {
        "head": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "production": {str(p.relative_to(source)): sha(p) for p in sorted((source / "python/cudnn").rglob("*.py")) if "__pycache__" not in p.parts},
        "native": {str(p.relative_to(source)): sha(p) for p in (source / "python/cudnn").glob("_compiled_module*.so")},
    }


try:
    (OUT / "harness.py").write_bytes(HARNESS.read_bytes())
    (OUT / "driver.py").write_bytes(Path(__file__).read_bytes())
    (OUT / "amendment.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "original_strict_gate": "FAILED at full-model relative L2=0.2011746 vs quantized oracle; threshold stays0.01. Narrow tile row SFD detector leaves1664/1664 live scales poisoned.",
                "reason": "Finish user-requested complete model source-regression E2E at the same recipe. Compare exact full logits, native routes, prompt text, and256/1024-prefill+32 real KV-cache generated tokens. Preserve reference gate failures and do not run formal performance or declare recipe quality/throughput success.",
                "sources": SOURCES,
            },
            indent=2,
        )
    )
    harness_hash = sha(HARNESS)
    state("WAITING_GPU_LOCK")
    with open("/tmp/codex-thor-perf.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip() == "inactive"
        frozen = {label: freeze(ROOT / source) for label, (source, _) in SOURCES.items()}
        assert all(frozen[label]["head"] == head for label, (_, head) in SOURCES.items())
        (OUT / "source-pre.json").write_text(json.dumps(frozen, indent=2))
        weight_hash = sha(MODEL / "model.safetensors")
        assert weight_hash == "ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1"
        results = {}
        for label, (source, head) in SOURCES.items():
            state("RUNNING_" + label)
            cmd = [
                str(ROOT / "venvs" / source / "bin/python"),
                str(HARNESS),
                "--mode",
                "diagnostic",
                "--tile",
                "256",
                "--expected-source",
                str(ROOT / source),
                "--expected-head",
                head,
                "--out",
                str(OUT / label),
            ]
            started = time.monotonic()
            with (OUT / (label + ".log")).open("w") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            records.append({"source": label, "command": cmd, "returncode": r.returncode, "seconds": time.monotonic() - started})
            (OUT / "runs.json").write_text(json.dumps(records, indent=2))
            r.check_returncode()
            result = json.loads((OUT / label / "result.json").read_text())
            assert result["stage"] == "DIAGNOSTIC_COMPLETE" and result["harness_sha256"] == harness_hash
            assert result["total_native_calls"] == 2761
            results[label] = result
        a, b = results["control"], results["candidate"]
        assert a["inputs"] == b["inputs"] and a["weight_digests"] == b["weight_digests"]
        assert [c["native_logits"] for c in a["forward_checks"]] == [c["native_logits"] for c in b["forward_checks"]]
        assert [(c["tokens"], c["logits"], c["text"]) for c in a["generations"]] == [(c["tokens"], c["logits"], c["text"]) for c in b["generations"]]
        assert [(c["ids"], c["logits"]) for c in a["e2e_requests"]] == [(c["ids"], c["logits"]) for c in b["e2e_requests"]]
        post = {label: freeze(ROOT / source) for label, (source, _) in SOURCES.items()}
        (OUT / "source-post.json").write_text(json.dumps(post, indent=2))
        assert post == frozen and sha(HARNESS) == harness_hash and sha(MODEL / "model.safetensors") == weight_hash
        assert subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip() == "inactive"
        result = {
            "stage": (
                "SOURCE_REGRESSION_PASS_REFERENCE_QUALITY_FAILED"
                if not all(c["reference_quality_pass"] for c in results.values())
                else "SOURCE_REGRESSION_AND_QUALITY_PASS"
            ),
            "scope": __doc__,
            "reference_quality_pass": {k: v["reference_quality_pass"] for k, v in results.items()},
            "same_full_logits": True,
            "same_generated_tokens_and_text": True,
            "same_e2e_logits_and_tokens": True,
            "native_calls_per_source": 2761,
            "children": 2,
            "actual_rc_all_zero": True,
            "source_native_harness_checkpoint_frozen": True,
            "service": "inactive",
            "formal_performance": "NOT_RUN_AFTER_QUALITY_GATE_FAILURE",
        }
        (OUT / "audit.json").write_text(json.dumps(result, indent=2))
        state(result["stage"])
except BaseException as exc:
    (OUT / "failure.txt").write_text(traceback.format_exc())
    state("FAILED", error=repr(exc))
    raise
