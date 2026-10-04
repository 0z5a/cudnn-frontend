# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare independent latest upstream and fresh native binding without GPU admission."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928")
OLD_HEAD = "c7841117c988fc09cad1df415f10e3ce7d6fa50e"
BASE = "51d9d06b574222378a3d806009accab098e73705"
OLD_BASE = "9ecb75e7771fcfb933e19136ace70ab8fade83fa"
SOURCE = ROOT / "a3sfdlatest51d9_r1"
ENVDIR = ROOT / "venvs/a3sfdlatest51d9_r1"
RESULT = ROOT / "results/build-a3-latest-51d9-20261005-r1"
PROVENANCE = ROOT / "results/a3-latest-51d9-native-provenance-20261005-r1.json"
POLICY = ROOT / "tools/a3_async_policy_20261004"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and "torch" not in sys.modules
    assert os.environ["A3_NO_SIGNAL_POLICY"] == "1"
    manifest = json.loads((POLICY / "manifest.json").read_text())
    assert manifest["policy"] == "OBSERVE_ONLY_NO_SIGNALS"
    assert all(sha(POLICY / path) == digest for path, digest in manifest["files"].items())
    old = ROOT / "a3sfdreviewc784"
    previous = json.loads((ROOT / "results/a3-review-c784-native-provenance-20261004-r1.json").read_text())
    assert previous["head"] == OLD_HEAD and previous["base"] == OLD_BASE
    assert git(old, "rev-parse", "HEAD") == OLD_HEAD
    assert sha(previous["binding"]["path"]) == previous["binding"]["sha256"]
    frozen = json.loads((ROOT / "results/a3-serial-iteration-20261002-r5/frozen_inputs.json").read_text())["files"]
    assert len(frozen) == 48 and all(sha(path) == digest for path, digest in frozen.items())
    disk_before = shutil.disk_usage(ROOT).free
    assert disk_before >= 6 * 1024**3, f"Insufficient build headroom: {disk_before} bytes"
    assert not any(path.exists() for path in [SOURCE, ENVDIR, RESULT, PROVENANCE])
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(old), str(SOURCE)], check=True)
    public_env = os.environ.copy()
    public_env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0")
    subprocess.run(
        ["git", "-C", str(SOURCE), "-c", "credential.helper=", "fetch", "https://github.com/NVIDIA/cudnn-frontend.git", "develop"],
        env=public_env,
        check=True,
    )
    assert git(SOURCE, "rev-parse", "FETCH_HEAD") == BASE
    subprocess.run(
        ["git", "-C", str(SOURCE), "-c", "credential.helper=", "fetch", "https://github.com/0z5a/cudnn-frontend.git", "codex/a3-sm110-correctness"],
        env=public_env,
        check=True,
    )
    assert git(SOURCE, "rev-parse", "FETCH_HEAD") == OLD_HEAD
    subprocess.run(["git", "-C", str(SOURCE), "checkout", "-b", "codex/a3-sfd-latest-51d9-20261005-r1", OLD_HEAD], check=True)
    author = git(old, "show", "-s", "--format=%an%n%ae", "HEAD").splitlines()
    subprocess.run(["git", "-C", str(SOURCE), "-c", "user.name=" + author[0], "-c", "user.email=" + author[1], "merge", "--no-edit", BASE], check=True)
    head = git(SOURCE, "rev-parse", "HEAD")
    assert git(SOURCE, "merge-base", "HEAD", BASE) == BASE
    assert git(SOURCE, "rev-parse", "HEAD^1") == OLD_HEAD
    pr_paths = git(old, "diff", "--name-only", OLD_BASE, OLD_HEAD).splitlines()
    assert pr_paths and all((old / path).read_bytes() == (SOURCE / path).read_bytes() for path in pr_paths)
    tracked = git(SOURCE, "ls-files").splitlines()
    native = [
        path
        for path in tracked
        if path.startswith(("include/", "cmake/"))
        or path in ("CMakeLists.txt", "setup.py", "pyproject.toml")
        or (path.startswith("python/") and Path(path).suffix in (".h", ".hpp", ".cpp", ".cc", ".cu", ".c"))
    ]
    changed = [path for path in native if not (old / path).exists() or (old / path).read_bytes() != (SOURCE / path).read_bytes()]
    assert len(native) >= 118 and changed
    assert not list((SOURCE / "python").rglob("_compiled_module*.so"))
    subprocess.run(["python3", "-m", "venv", str(ENVDIR)], check=True)
    oldsite = ROOT / "venvs/a3sfdreviewc784/lib/python3.12/site-packages"
    site = ENVDIR / "lib/python3.12/site-packages"
    for path in oldsite.iterdir():
        destination = site / path.name
        if destination.exists() or path.name == "__pycache__":
            continue
        if path.name.startswith("__editable__") or path.suffix == ".pth":
            assert path.is_file()
            destination.write_text(path.read_text().replace(str(old), str(SOURCE)))
        elif path.name.startswith("nvidia_cudnn_frontend"):
            if path.is_dir():
                shutil.copytree(path, destination, symlinks=True)
                url = destination / "direct_url.json"
                if url.exists():
                    data = json.loads(url.read_text())
                    data["url"] = SOURCE.as_uri()
                    url.write_text(json.dumps(data) + "\n")
            else:
                shutil.copyfile(path, destination)
        else:
            destination.symlink_to(path, target_is_directory=path.is_dir())
    shared = Path("/home/jwipc/projects/host-gateway/.vllm")
    env = os.environ.copy()
    env.update(
        PATH=str(shared / "bin") + ":" + env["PATH"],
        CUDAToolkit_ROOT="/usr/local/cuda-13.2",
        CUDA_PATH="/usr/local/cuda-13.2",
        CUDNN_PATH=str(shared / "lib/python3.12/site-packages/nvidia/cudnn"),
        FETCHCONTENT_SOURCE_DIR_DLPACK="/home/jwipc/experiments/sm110-execution-20260928/sources/dlpack",
        CMAKE_BUILD_PARALLEL_LEVEL="1",
        CMAKE_PREFIX_PATH=str(site / "pybind11/share/cmake/pybind11"),
    )
    record = {
        "head": head,
        "base": BASE,
        "source": str(SOURCE),
        "venv": str(ENVDIR),
        "prior_public_head": OLD_HEAD,
        "prior_base": OLD_BASE,
        "PR_paths_preserved_byte_exact": pr_paths,
        "changed_native_inputs": changed,
        "native_input_digests": [{"path": path, "sha256": sha(SOURCE / path)} for path in native],
        "script_sha256": sha(__file__),
        "policy_manifest_sha256": sha(POLICY / "manifest.json"),
        "disk_available_before_bytes": disk_before,
        "public_fetch_command_scoped_global_config": "/dev/null",
        "user_global_git_config_modified": False,
        "GPU_tests": "NOT_RUN",
        "complete_Granite_e2e": "NOT_RUN",
    }
    prebuild = ROOT / "results/a3-latest-51d9-prebuild-20261005-r1.json"
    assert not prebuild.exists()
    prebuild.write_text(json.dumps(record, indent=2) + "\n")
    python = ENVDIR / "bin/python"
    rc = subprocess.run(
        [
            str(python),
            str(POLICY / "run_logged_observe.py"),
            "--out",
            str(RESULT),
            "--cwd",
            str(SOURCE),
            "--timeout",
            "3600",
            "--",
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(SOURCE),
        ],
        env=env,
    ).returncode
    if rc:
        return rc
    run = json.loads((RESULT / "run.json").read_text())
    assert run["status"] == "EXITED" and run["returncode"] == 0 and run["signal"] is None and run["signals_sent"] == 0
    bindings = list((SOURCE / "python").rglob("_compiled_module*.so"))
    assert len(bindings) == 1
    code = (
        "import os,json,torch,cudnn;from cudnn import _compiled_module as b;"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized();"
        "print(json.dumps({'torch':torch.__version__,'cudnn_path':cudnn.__file__,'binding_path':b.__file__,"
        "'cuda_initialized':torch.cuda.is_initialized(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}))"
    )
    validation = json.loads(subprocess.check_output([str(python), "-c", code], cwd=SOURCE, env=env, text=True))
    assert Path(validation["cudnn_path"]).resolve().is_relative_to(SOURCE / "python")
    assert Path(validation["binding_path"]).resolve() == bindings[0].resolve()
    assert not validation["cuda_initialized"] and validation["cuda_visible_devices"] == ""
    assert all(sha(SOURCE / item["path"]) == item["sha256"] for item in record["native_input_digests"])
    assert all(sha(path) == digest for path, digest in frozen.items())
    assert all(sha(POLICY / path) == digest for path, digest in manifest["files"].items())
    assert git(old, "rev-parse", "HEAD") == OLD_HEAD and sha(previous["binding"]["path"]) == previous["binding"]["sha256"]
    assert not git(SOURCE, "status", "--porcelain", "--untracked-files=no")
    assert subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True).stdout.strip() == "inactive"
    record.update(
        native_build="FRESH_RC0",
        build_run=run,
        binding={"path": str(bindings[0]), "sha256": sha(bindings[0])},
        import_validation=validation,
        new_native_calls=0,
        cuda_initialized=False,
        original_source_and_binding_unchanged=True,
        R6_original48_frozen_sources_unchanged=True,
        service_final="inactive",
        disk_available_after_bytes=shutil.disk_usage(ROOT).free,
    )
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"head": head, "base": BASE, "fresh_native_build_rc": 0, "binding": record["binding"], "GPU_tests": "NOT_RUN"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
