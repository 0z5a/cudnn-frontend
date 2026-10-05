# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped redirection of only the frozen timeout logger; math files unchanged."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def install():
    if os.environ.get("A3_NO_SIGNAL_POLICY") != "1":
        return
    if getattr(subprocess.Popen, "_a3_observe_only_installed", False):
        return
    directory = Path(os.environ["A3_NO_SIGNAL_POLICY_DIR"]).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    original = subprocess.Popen.__init__
    legacy = "/home/jwipc/experiments/sm110-validation-round2-20260928/pack/scripts/run_logged.py"

    def observed_init(self, args, *positional, **kwargs):
        if os.environ.get("A3_NO_SIGNAL_POLICY") != "1":
            return original(self, args, *positional, **kwargs)
        environment = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        paths = environment.get("PYTHONPATH", "").split(os.pathsep)
        environment["PYTHONPATH"] = os.pathsep.join([str(directory), *[x for x in paths if x and x != str(directory)]])
        environment.update(A3_NO_SIGNAL_POLICY="1", A3_NO_SIGNAL_POLICY_DIR=str(directory))
        kwargs["env"] = environment
        if isinstance(args, (list, tuple)) and len(args) > 1 and str(args[1]) == legacy:
            assert not kwargs.get("shell", False)
            args = list(args)
            args[1] = str(directory / "run_logged_observe.py")
            audit = Path(environment["A3_POLICY_AUDIT_DIR"])
            audit.mkdir(parents=True, exist_ok=True)
            record = {
                "parent_pid": os.getpid(),
                "original_logger": legacy,
                "replacement_logger": args[1],
                "policy_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                "signals_sent": 0,
            }
            (audit / ("redirect-" + str(os.getpid()) + "-" + uuid.uuid4().hex + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        return original(self, args, *positional, **kwargs)

    subprocess.Popen.__init__ = observed_init
    subprocess.Popen._a3_observe_only_installed = True
