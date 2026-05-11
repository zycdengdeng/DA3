"""Subprocess + multi-GPU sharding helpers used by the upstream-stage
wrappers (``depth``, ``mask``, ``seg``, ``calib``, ``object-accum``).

The upstream stages are still implemented as standalone scripts in
``scripts/``; this module thinly wraps them so each new ``lad <stage>``
subcommand can:

* place the script's outputs under the standard
  ``outputs/<scene>/<stage>/<run_id>/`` directory,
* tee stdout/stderr into a per-run ``run.log``,
* shard across multiple GPUs when the underlying script has a
  ``--ts-shard k/N`` argument.

A later refactor can lift each script's body into a proper stage
implementation; the subprocess wrapper keeps the architectural CLI
working in the meantime without forking the script logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def _resolve_script(script_name: str) -> Path:
    """Resolve a script path under the project's ``scripts/`` dir.

    Looks up to a few directory levels above the package root.
    """
    # This file lives at src/lidar_anchored_depth/engine/script_runner.py
    here = Path(__file__).resolve()
    project_root = here.parents[3]   # .../projects/lidar_anchored_depth
    script_path = project_root / "scripts" / script_name
    if not script_path.is_file():
        raise SystemExit(
            f"upstream script not found: {script_path} "
            f"(looked relative to {here})"
        )
    return script_path


def run_script_subprocess(
    script_name: str,
    argv: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``python scripts/<script_name> <argv>`` and tee everything
    to ``log_path``. Returns the subprocess exit code (0 = success)."""
    script_path = _resolve_script(script_name)
    cmd = [sys.executable, str(script_path), *argv]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the command we ran at the top of the log for reproducibility.
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("# cmd\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    return proc.returncode


def run_script_sharded_by_ts(
    script_name: str,
    base_argv: list[str],
    gpu_ids: Iterable[int],
    *,
    log_dir: Path,
) -> int:
    """Spawn one subprocess per GPU, each pinned to its card and
    handed ``--ts-shard k/N``. The underlying script must accept
    ``--ts-shard`` (DA3 + SegFormer do).

    Returns 0 if every shard exited 0, else the first non-zero rc.
    """
    gpu_list = list(gpu_ids)
    if not gpu_list:
        # Serial: one process, no sharding, no CUDA env override.
        return run_script_subprocess(
            script_name, base_argv, log_path=log_dir / "run.log",
        )

    script_path = _resolve_script(script_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    log_files = []
    for k, gpu in enumerate(gpu_list):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        argv = [
            *base_argv,
            "--ts-shard", f"{k}/{len(gpu_list)}",
        ]
        cmd = [sys.executable, str(script_path), *argv]
        log_path = log_dir / f"shard_{k}_gpu{gpu}.log"
        f = log_path.open("w", encoding="utf-8")
        f.write("# cmd\n")
        f.write(" ".join(cmd) + "\n")
        f.write(f"# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
        )
        procs.append(p)
        log_files.append(f)
        print(
            f"  [shard {k}/{len(gpu_list)}] pid={p.pid} gpu={gpu} "
            f"-> {log_path}",
            flush=True,
        )

    rc_final = 0
    t0 = time.time()
    for k, (p, f) in enumerate(zip(procs, log_files)):
        rc = p.wait()
        f.close()
        if rc != 0 and rc_final == 0:
            rc_final = rc
        print(
            f"  [shard {k}] exit={rc}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return rc_final


def run_script_sharded_by_cam(
    script_name: str,
    base_argv: list[str],
    cams: Iterable[str],
    gpu_ids: Iterable[int],
    *,
    log_dir: Path,
) -> int:
    """For scripts that don't have ``--ts-shard`` but do accept
    ``--cam <id>`` (e.g. ``run_sam_inference.py``), shard by camera
    instead. Round-robin assigns each cam to a GPU.

    With ``gpu_ids`` empty, runs ``--cam all`` serially.
    """
    cam_list = [str(c) for c in cams]
    gpu_list = list(gpu_ids)
    if not gpu_list:
        return run_script_subprocess(
            script_name, [*base_argv, "--cam", "all"],
            log_path=log_dir / "run.log",
        )

    script_path = _resolve_script(script_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    log_files = []
    for i, cam in enumerate(cam_list):
        gpu = gpu_list[i % len(gpu_list)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        argv = [*base_argv, "--cam", cam]
        cmd = [sys.executable, str(script_path), *argv]
        log_path = log_dir / f"cam{cam}_gpu{gpu}.log"
        f = log_path.open("w", encoding="utf-8")
        f.write("# cmd\n")
        f.write(" ".join(cmd) + "\n")
        f.write(f"# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
        )
        procs.append(p)
        log_files.append(f)
        print(
            f"  [cam {cam}] pid={p.pid} gpu={gpu} -> {log_path}",
            flush=True,
        )

    rc_final = 0
    t0 = time.time()
    for cam, p, f in zip(cam_list, procs, log_files):
        rc = p.wait()
        f.close()
        if rc != 0 and rc_final == 0:
            rc_final = rc
        print(f"  [cam {cam}] exit={rc}  ({time.time() - t0:.0f}s)", flush=True)
    return rc_final


__all__ = [
    "run_script_sharded_by_cam",
    "run_script_sharded_by_ts",
    "run_script_subprocess",
]
