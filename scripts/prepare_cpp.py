#!/usr/bin/env python3
"""Prepare pinned NeMo-Speech.cpp, build T4 CUDA SALM, and convert a local checkpoint."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REVISION = "d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14"
REPO = Path("/content/nemo-speech")
SOURCE = Path("/content/hf-cache/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/snapshots/c17667641642c4265ff2b7037dd101b7d4567634")
OUT = Path("/content/nemotron-gguf")
REPORT_DIR = Path("/content/nemotron-transcribe")
STATUS = REPORT_DIR / "cpp-status.json"
LOG = REPORT_DIR / "prepare_cpp.log"
PATCH_FILE = REPORT_DIR / "n_gpu_layers.patch"
DECODER = OUT / "decoder-Q4_0.gguf"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_status() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=REPORT_DIR, delete=False) as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    tmp.replace(STATUS)


def run_stage(name: str, command: list[str], cwd: Path | None = None) -> None:
    step = {"status": "running", "started_at_utc": utc(), "command": command}
    state["phase"] = name
    state["stages"][name] = step
    save_status()
    print(f"[prepare_cpp] start {name}: {' '.join(command)}", flush=True)
    try:
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{utc()}] {name}: {' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        step["status"] = "complete"
        step["finished_at_utc"] = utc()
        save_status()
    except Exception as exc:
        step.update({"status": "failed", "finished_at_utc": utc(), "error": f"{type(exc).__name__}: {exc}"})
        save_status()
        raise


def patch_gpu_layers() -> None:
    name = "patch_gpu_layers"
    state["phase"] = name
    state["stages"][name] = {"status": "running", "started_at_utc": utc()}
    save_status()
    path = REPO / "src/speechlm/salm_engine.cpp"
    source = path.read_text(encoding="utf-8")
    old = "        mp.n_gpu_layers = cfg.gpu >= 0 ? 999 : 0;"
    new = '''        const char* gpu_layers_env = std::getenv("NEMOTRON_GPU_LAYERS");
        if (gpu_layers_env) {
            char* parsed_end = nullptr;
            const long gpu_layers = std::strtol(gpu_layers_env, &parsed_end, 10);
            if (parsed_end == gpu_layers_env || *parsed_end != '\\0' || gpu_layers < 0 || gpu_layers > 999)
                throw std::invalid_argument("NEMOTRON_GPU_LAYERS must be an integer from 0 to 999");
            mp.n_gpu_layers = static_cast<int>(gpu_layers);
        } else {
            mp.n_gpu_layers = cfg.gpu >= 0 ? 999 : 0;
        }'''
    if "NEMOTRON_GPU_LAYERS" not in source:
        if source.count(old) != 1:
            raise RuntimeError("Pinned SALM source no longer has exactly one expected n_gpu_layers assignment.")
        tmp = path.with_suffix(".cpp.tmp")
        tmp.write_text(source.replace(old, new, 1), encoding="utf-8")
        tmp.replace(path)
    elif "mp.n_gpu_layers = static_cast<int>(gpu_layers);" not in source:
        raise RuntimeError("Existing NEMOTRON_GPU_LAYERS patch does not match the expected implementation.")
    diff = subprocess.run(["git", "diff", "--binary", "--", "src/speechlm/salm_engine.cpp"],
                          cwd=REPO, check=True, capture_output=True, text=True).stdout
    if not diff:
        raise RuntimeError("No source diff was produced for the GPU-layer override.")
    PATCH_FILE.write_text(diff, encoding="utf-8")
    state["patch_file"] = str(PATCH_FILE)
    state["stages"][name].update({"status": "complete", "finished_at_utc": utc(), "diff_bytes": len(diff.encode())})
    save_status()
    print(f"[prepare_cpp] saved source diff: {PATCH_FILE}", flush=True)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    global state
    state = {"status": "running", "phase": "startup", "started_at_utc": utc(), "revision": REVISION,
             "checkpoint": str(SOURCE), "quantization": "experimental q4_0 experts / q8_0 dense", "decoder": str(DECODER),
             "log": str(LOG), "stages": {}, "error": None}
    save_status()
    try:
        if REPO.exists() and not (REPO / ".git").exists():
            raise RuntimeError(f"Refusing to overwrite non-git path: {REPO}")
        if not REPO.exists():
            run_stage("clone", ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo-Speech.cpp.git", str(REPO)])
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if head != REVISION:
            dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet"], check=False).returncode
            if dirty:
                raise RuntimeError(f"Refusing to checkout {REVISION}; existing tracked changes would be lost.")
            run_stage("fetch_revision", ["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", REVISION])
            run_stage("checkout_revision", ["git", "-C", str(REPO), "checkout", "--detach", REVISION])
        run_stage("submodule", ["git", "-C", str(REPO), "submodule", "update", "--init", "--depth", "1", "llama.cpp"])
        run_stage("system_dependencies", ["apt-get", "install", "-y", "build-essential", "cmake", "ninja-build", "git", "pkg-config", "libsentencepiece-dev"])
        if not shutil.which("nvcc"):
            raise RuntimeError("nvcc is missing; the CUDA toolkit must be available in the Colab runtime.")
        if not SOURCE.is_dir():
            raise FileNotFoundError(f"Expected existing BF16 snapshot; refusing model download: {SOURCE}")
        torch_version = importlib.metadata.version("torch")
        if torch_version != "2.11.0+cu130":
            raise RuntimeError(f"Expected existing torch 2.11.0+cu130; found {torch_version}. No pip install was attempted.")
        torchaudio_version = importlib.metadata.version("torchaudio")
        requirements = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
        filtered = ["gguf>=0.10.0", "sentencepiece>=0.1.98,<0.3.0", "PyYAML>=6,<7"]
        req_file = REPORT_DIR / "cpp-requirements.txt"
        constraint_file = REPORT_DIR / "torch-constraints.txt"
        req_file.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        constraint_file.write_text(f"torch=={torch_version}\ntorchaudio=={torchaudio_version}\nnumpy=={importlib.metadata.version('numpy')}\ntransformers=={importlib.metadata.version('transformers')}\n", encoding="utf-8")
        run_stage("converter_dependencies", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                  "--upgrade-strategy", "only-if-needed", "--constraint", str(constraint_file), "-r", str(req_file)], cwd=REPO)
        if importlib.metadata.version("torch") != torch_version:
            raise RuntimeError("Converter dependency install changed the existing torch version.")
        patch_gpu_layers()
        configure = ["scripts/configure.sh", "cuda-asr", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
                     "-DNEMO_SPEECH_GGML_PATCHED=OFF", "-DNEMO_SPEECH_BUILD_ASR=ON", "-DNEMO_SPEECH_BUILD_SPEECHLM=ON",
                     "-DNEMO_SPEECH_BUILD_DIAR=OFF", "-DNEMO_SPEECH_BUILD_TTS=OFF", "-DNEMO_SPEECH_BUILD_NMT=OFF",
                     "-DNEMO_SPEECH_BUILD_CLI=ON", "-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF", "-DNEMO_SPEECH_BUILD_HTTP=OFF",
                     "-DNEMO_SPEECH_BUILD_GRPC=OFF", "-DNEMO_SPEECH_BUILD_S2S=OFF"]
        run_stage("configure", configure, cwd=REPO)
        run_stage("build", ["cmake", "--build", "--preset", "cuda-asr", "--parallel", "8"], cwd=REPO)
        adapter = REPORT_DIR / "convert_q4_profile.py"
        if not adapter.is_file():
            raise FileNotFoundError(f"Write the supplied experimental Q4_0 adapter first: {adapter}")
        run_stage("convert_q4_0", [sys.executable, str(adapter), str(SOURCE), str(DECODER)], cwd=REPO)
        encoders = list(OUT.glob("*-Encoder-F16.gguf"))
        if not DECODER.is_file() or not encoders:
            raise RuntimeError("Converter exited successfully but decoder or F16 speech encoder is missing.")
        state.update({"status": "complete", "phase": "complete", "finished_at_utc": utc(),
                      "encoder_files": [str(path) for path in encoders]})
        save_status()
        print(f"[prepare_cpp] complete: {DECODER}", flush=True)
        return 0
    except Exception as exc:
        state.update({"status": "failed", "phase": state.get("phase", "unknown"), "finished_at_utc": utc(),
                      "error": {"type": type(exc).__name__, "message": str(exc)}})
        save_status()
        print(f"[prepare_cpp] failed in {state['phase']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
