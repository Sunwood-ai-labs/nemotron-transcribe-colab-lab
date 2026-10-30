#!/usr/bin/env python3
"""Capture the Colab runtime and install the minimal HF inference stack."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
MODEL_REVISION = "c17667641642c4265ff2b7037dd101b7d4567634"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION = "e82b00dd04d71b2e2801ecf3029eea3858868091"
OUTPUT_DIR = Path(os.environ.get("NEMOTRON_OUT_DIR", "/content/nemotron-transcribe"))
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/content/hf-cache"))
RUNTIME_JSON = OUTPUT_DIR / "runtime.json"

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "soundfile",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "pyarrow",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def read_memory_facts() -> dict[str, int | None]:
    facts: dict[str, int | None] = {"total_bytes": None, "available_bytes": None}
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return facts
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, remainder = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            fields = remainder.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    facts["total_bytes"] = values.get("MemTotal")
    facts["available_bytes"] = values.get("MemAvailable")
    return facts


def collect_environment() -> dict[str, Any]:
    import torch

    gpu_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(total_bytes),
                    "free_memory_bytes": int(free_bytes),
                }
            )

    disk_path = Path("/content") if Path("/content").exists() else OUTPUT_DIR.parent
    disk = shutil.disk_usage(disk_path)
    nvidia_smi: dict[str, Any] = {"available": shutil.which("nvidia-smi") is not None}
    if nvidia_smi["available"]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            nvidia_smi.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-2000:],
                }
            )
        except Exception as exc:  # Runtime fact collection is best-effort at this boundary.
            nvidia_smi["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "captured_at_utc": utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": gpu_devices,
        },
        "ram": read_memory_facts(),
        "disk": {
            "path": str(disk_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "nvidia_smi": nvidia_smi,
    }


def write_report(report: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=OUTPUT_DIR,
            prefix="runtime.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(RUNTIME_JSON)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def run_install(torch_version: str) -> dict[str, Any]:
    constraint_file = OUTPUT_DIR / "torch-constraint.txt"
    constraint_file.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        "--constraint",
        str(constraint_file),
        f"git+{TRANSFORMERS_REPOSITORY}@{TRANSFORMERS_REVISION}",
        "accelerate>=1.0",
        "datasets>=4.4",
        "soundfile>=0.13",
        "bitsandbytes",
    ]
    log("Installing the pinned Transformers source and audio/data dependencies; torch is constrained to its existing version.")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    install_result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"pip install exited with code {completed.returncode}")
    return install_result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "transformers_source": {
            "repository": TRANSFORMERS_REPOSITORY,
            "revision": TRANSFORMERS_REVISION,
        },
        "paths": {
            "output_dir": str(OUTPUT_DIR),
            "hf_cache_dir": str(HF_CACHE_DIR),
            "weights_are_inside_output_dir": False,
        },
        "packages_before": package_versions(),
        "environment_before": None,
        "install": None,
        "packages_after": None,
        "environment_after": None,
        "stages": [],
        "error": None,
    }
    current_stage: dict[str, Any] | None = None

    try:
        log("Collecting the current runtime facts.")
        environment_before = collect_environment()
        report["environment_before"] = environment_before
        report["stages"].append(
            {
                "name": "collect_runtime_before",
                "status": "complete",
                "finished_at_utc": utc_now(),
            }
        )
        write_report(report)

        if not environment_before["torch"]["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; refusing to prepare a CPU-only experiment.")

        current_stage = {
            "name": "install_dependencies",
            "status": "running",
            "started_at_utc": utc_now(),
        }
        report["stages"].append(current_stage)
        write_report(report)
        report["install"] = run_install(environment_before["torch"]["version"])
        importlib.invalidate_caches()
        report["packages_after"] = package_versions()
        environment_after = collect_environment()
        report["environment_after"] = environment_after

        if environment_after["torch"]["version"] != environment_before["torch"]["version"]:
            raise RuntimeError(
                "The dependency installer changed torch from "
                f"{environment_before['torch']['version']} to {environment_after['torch']['version']}."
            )
        if not environment_after["torch"]["cuda_available"]:
            raise RuntimeError("CUDA became unavailable after dependency installation.")

        current_stage["status"] = "complete"
        current_stage["finished_at_utc"] = utc_now()
        report["status"] = "complete"
        report["finished_at_utc"] = utc_now()
        write_report(report)
        log(f"Bootstrap complete. Runtime report: {RUNTIME_JSON}")
        return 0
    except Exception as exc:
        if current_stage is not None and current_stage.get("status") == "running":
            current_stage["status"] = "failed"
            current_stage["finished_at_utc"] = utc_now()
        report["status"] = "failed"
        report["finished_at_utc"] = utc_now()
        report["error"] = {
            "stage": current_stage.get("name") if current_stage else "collect_runtime_before",
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-12000:],
        }
        try:
            report["packages_after"] = package_versions()
            if report["environment_after"] is None:
                report["environment_after"] = collect_environment()
        except Exception as capture_exc:
            report["error"]["post_failure_capture_error"] = f"{type(capture_exc).__name__}: {capture_exc}"
        try:
            write_report(report)
        except Exception as write_exc:
            print(f"[bootstrap] Could not write {RUNTIME_JSON}: {write_exc}", file=sys.stderr, flush=True)
        log(f"Bootstrap failed during {report['error']['stage']}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
