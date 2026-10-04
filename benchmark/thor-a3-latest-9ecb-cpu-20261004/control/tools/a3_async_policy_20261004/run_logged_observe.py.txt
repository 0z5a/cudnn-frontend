# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe commands and real terminals; deadlines never send process signals."""

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cwd", type=Path, default=Path.cwd())
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error("Command and positive observation deadline required")
    a.out.mkdir(parents=True, exist_ok=False)
    meta = {
        "argv": cmd,
        "cwd": str(a.cwd.resolve()),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "deadline_seconds": a.timeout,
        "status": "NOT_STARTED",
        "runner_policy": "OBSERVE_ONLY_NO_SIGNALS",
        "signals_sent": 0,
        "selected_env": {k: os.environ[k] for k in ["CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING"] if k in os.environ},
    }
    started = time.monotonic()
    result = 125
    child = None
    try:
        with (a.out / "stdout.log").open("wb") as output, (a.out / "stderr.log").open("wb") as error:
            child = subprocess.Popen(cmd, cwd=a.cwd, stdout=output, stderr=error, start_new_session=True)
            meta["pid"] = child.pid
            live = {
                "status": "RUNNING",
                "pid": child.pid,
                "logger_pid": os.getpid(),
                "argv": cmd,
                "started_utc": meta["started_utc"],
                "runner_policy": meta["runner_policy"],
                "deadline_observed": False,
                "signals_sent": 0,
            }
            while child.poll() is None:
                live.update(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
                live["deadline_observed"] = live["elapsed_seconds"] > a.timeout
                write(a.out / "run.live.json", live)
                time.sleep(min(1.0, max(0.01, a.timeout / 4)))
            code = child.wait()
            meta.update(status="EXITED", returncode=code, signal=(-code if code < 0 else None), deadline_observed=time.monotonic() - started > a.timeout)
            result = code if code >= 0 else min(255, 128 - code)
    except (OSError, ValueError) as exc:
        meta["observation_error"] = f"{type(exc).__name__}: {exc}"
        if child is None:
            meta.update(status="LAUNCH_ERROR")
        else:
            code = child.wait()
            meta.update(status="EXITED", returncode=code, signal=(-code if code < 0 else None), pid=child.pid)
            result = code if code >= 0 else min(255, 128 - code)
    meta["elapsed_seconds"] = time.monotonic() - started
    write(a.out / "run.live.json", dict(meta, logger_pid=os.getpid(), child_alive=False))
    write(a.out / "run.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
