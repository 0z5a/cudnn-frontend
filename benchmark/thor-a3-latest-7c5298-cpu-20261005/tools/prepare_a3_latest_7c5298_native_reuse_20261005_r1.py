# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare current review head without changing any frozen GPU execution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928")
HEAD = "6850a911a4e18a5e01dc191b65e341ad33548893"
OLD_HEAD = "2fc0bc608c19c75311a405b12cc57374741f9df1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and "torch" not in sys.modules
    assert os.environ["A3_NO_SIGNAL_POLICY"] == "1"
    policy = ROOT / "tools/a3_async_policy_20261004"
    manifest = json.loads((policy / "manifest.json").read_text())
    assert manifest["policy"] == "OBSERVE_ONLY_NO_SIGNALS"
    assert all(sha(policy / path) == digest for path, digest in manifest["files"].items())
    old = ROOT / "a3sfdlatest51d9_r1"
    source = ROOT / "a3sfdlatest7c5298_r1"
    envdir = ROOT / "venvs/a3sfdlatest7c5298_r1"
    original = json.loads((ROOT / "results/a3-latest-51d9-native-provenance-20261005-r1.json").read_text())
    assert original["head"] == OLD_HEAD and original["native_build"] == "FRESH_RC0"
    assert original["build_run"]["status"] == "EXITED" and original["build_run"]["returncode"] == 0 and original["build_run"]["signal"] is None
    assert not original["cuda_initialized"] and len(original["native_input_digests"]) == 118
    original_hashes = {item["path"]: item["sha256"] for item in original["native_input_digests"]}
    assert all(sha(old / p) == digest for p, digest in original_hashes.items())
    assert sha(original["binding"]["path"]) == original["binding"]["sha256"]
    frozen = json.loads((ROOT / "results/a3-serial-iteration-20261002-r5/frozen_inputs.json").read_text())["files"]
    assert len(frozen) == 48 and all(sha(path) == digest for path, digest in frozen.items())
    assert not source.exists() and not envdir.exists()
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(old), str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "fetch", str(ROOT / "results/a3-latest-7c5298-preparation-background-20261005-r1/source.bundle"), "HEAD"], check=True
    )
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "FETCH_HEAD"], text=True).strip() == HEAD
    subprocess.run(["git", "-C", str(source), "checkout", "-b", "codex/a3-latest-7c5298-20261005-r1", HEAD], check=True)
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^"], text=True).strip() == OLD_HEAD
    changed = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", OLD_HEAD, HEAD], text=True).splitlines()
    assert changed == [
        "python/cudnn/AGENTS.md",
        "python/cudnn/frost/buffers.py",
        "python/cudnn/frost/workspace.py",
        "test/python/core/cutedsl/test_workspace_device.py",
    ]
    pr_paths = subprocess.check_output(
        ["git", "-C", str(old), "diff", "--name-only", "9ecb75e7771fcfb933e19136ace70ab8fade83fa", "c7841117c988fc09cad1df415f10e3ce7d6fa50e"], text=True
    ).splitlines()
    assert len(pr_paths) == 7 and all((old / path).read_bytes() == (source / path).read_bytes() for path in pr_paths)
    assert (
        subprocess.check_output(["git", "-C", str(source), "merge-base", "HEAD", "7c5298ecea940698590b8dd8d55adadc8a02f71e"], text=True).strip()
        == "7c5298ecea940698590b8dd8d55adadc8a02f71e"
    )
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines()
    native = [
        p
        for p in tracked
        if p.startswith(("include/", "cmake/"))
        or p in ("CMakeLists.txt", "setup.py", "pyproject.toml")
        or (p.startswith("python/") and Path(p).suffix in (".h", ".hpp", ".cpp", ".cc", ".cu", ".c"))
    ]
    assert set(native) == set(original_hashes) and all(sha(source / p) == digest for p, digest in original_hashes.items())
    oldbinding = Path(original["binding"]["path"])
    binding = source / oldbinding.relative_to(old)
    assert not binding.exists()
    shutil.copyfile(oldbinding, binding)
    assert sha(binding) == original["binding"]["sha256"]
    subprocess.run(["python3", "-m", "venv", str(envdir)], check=True)
    oldsite = ROOT / "venvs/a3sfdlatest51d9_r1/lib/python3.12/site-packages"
    site = envdir / "lib/python3.12/site-packages"
    for p in oldsite.iterdir():
        destination = site / p.name
        if destination.exists() or p.name == "__pycache__":
            continue
        if p.name.startswith("__editable__") or p.suffix == ".pth":
            assert p.is_file()
            destination.write_text(p.read_text().replace(str(old), str(source)))
        elif p.name.startswith("nvidia_cudnn_frontend"):
            if p.is_dir():
                shutil.copytree(p, destination, symlinks=True)
                url = destination / "direct_url.json"
                if url.exists():
                    data = json.loads(url.read_text())
                    data["url"] = source.as_uri()
                    url.write_text(json.dumps(data) + "\n")
            else:
                shutil.copyfile(p, destination)
        else:
            destination.symlink_to(p, target_is_directory=p.is_dir())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "tools/a3_async_policy_20261004")
    code = (
        "import os,json,torch,cudnn;from cudnn import _compiled_module as b;"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized();"
        "print(json.dumps({'torch':torch.__version__,'cudnn_path':cudnn.__file__,'binding_path':b.__file__,"
        "'cuda_initialized':torch.cuda.is_initialized(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}))"
    )
    validation = json.loads(subprocess.check_output([str(envdir / "bin/python"), "-c", code], cwd=source, env=env, text=True))
    assert Path(validation["cudnn_path"]).resolve().is_relative_to(source / "python")
    assert Path(validation["binding_path"]).resolve() == binding.resolve()
    assert not validation["cuda_initialized"]
    assert all(sha(source / p) == digest for p, digest in original_hashes.items())
    assert all(sha(path) == digest for path, digest in frozen.items())
    assert subprocess.check_output(["git", "-C", str(old), "rev-parse", "HEAD"], text=True).strip() == OLD_HEAD
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip() == "inactive"
    result = {
        "stage": "LATEST_UPSTREAM_CPU_SOURCE_AND_NATIVE_REUSE_PROVENANCE_VERIFIED",
        "head": HEAD,
        "base": "7c5298ecea940698590b8dd8d55adadc8a02f71e",
        "prior_base": original["base"],
        "PR_paths_preserved_byte_exact": pr_paths,
        "changed_native_inputs": [],
        "source": str(source),
        "venv": str(envdir),
        "parent": OLD_HEAD,
        "changed_source_paths": [{"path": p, "sha256": sha(source / p)} for p in changed],
        "native_build": "REUSED_ONLY_AFTER_ALL118_INPUTS_EXACT",
        "native_compiled_head": OLD_HEAD,
        "native_build_run_at_compiled_head": original["build_run"],
        "native_input_digests": original["native_input_digests"],
        "binding": {"path": str(binding), "sha256": sha(binding)},
        "import_validation": validation,
        "new_compile_invoked": False,
        "new_native_calls": 0,
        "cuda_initialized": False,
        "R7_original48_frozen_sources_unchanged": True,
        "service_final": "inactive",
        "GPU_tests": "NOT_RUN",
        "new_mock_precision_or_performance_test": "NOT_RUN_NO_DUPLICATE_EXTERNAL_MOCK_CLAIM",
        "script_sha256": sha(__file__),
        "policy_manifest_sha256": sha(policy / "manifest.json"),
        "workspace_multi_GPU_tests": "NOT_RUN_REQUIRES_TWO_VISIBLE_GPUS",
    }
    result_path = ROOT / "results/a3-latest-7c5298-native-provenance-20261005-r1.json"
    assert not result_path.exists()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "native_input_digests"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
