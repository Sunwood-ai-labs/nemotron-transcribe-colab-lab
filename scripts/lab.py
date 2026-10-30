"""Run the pinned Nemotron small-GPU preparation and transcription pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("NEMOTRON_OUT_DIR", "/content/nemotron-transcribe"))
HF_HOME = Path(os.environ.get("HF_HOME", "/content/hf-cache"))
MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
MODEL_REVISION = "c17667641642c4265ff2b7037dd101b7d4567634"
DECODER_PATH = Path("/content/nemotron-gguf/decoder-Q4_0.gguf")
NEMO_REPO = Path("/content/nemo-speech")
MIN_RAM_GIB = 50
MIN_FREE_DISK_GB = 180
MIN_FREE_VRAM_GIB = 4
INSTRUCTION = (
    "Transcribe this Japanese audio with punctuation. Omit verbal fillers, repeated words, "
    "and false starts while preserving the speaker's intended meaning. "
    "Write numbers in their standard written form."
)
SAMPLES = (
    {
        "id": "1519",
        "path": REPO_ROOT / "audio" / "fleurs-ja-01.wav",
        "reference": "多くの場合、海外のギャップイヤーコースに入学することで、帰国後に実際に大学に進学しやすくなります。",
    },
    {
        "id": "1607",
        "path": REPO_ROOT / "audio" / "fleurs-ja-02.wav",
        "reference": "興味をそそられる村を30分ほど散策するのは価値があります。",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, remainder = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(remainder.split()[0]) * 1024
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def check_runtime() -> dict[str, Any]:
    """Check the T4 High-RAM, compiler, package, and free-space prerequisites."""
    if not Path("/proc/meminfo").is_file() or not Path("/content").is_dir():
        raise RuntimeError("このノートブックはLinux版Google Colabで実行してください。")

    total_ram, available_ram = _read_memory()
    disk = shutil.disk_usage("/content")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("GPUが見つかりません。ColabのランタイムをT4 GPUに変更してください。")
    gpu = subprocess.run(
        [smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()[0].split(",")
    gpu_name, total_vram_mib, free_vram_mib = (part.strip() for part in gpu)
    facts = {
        "gpu": gpu_name,
        "gpu_total_vram_gib": round(int(total_vram_mib) / 1024, 2),
        "gpu_free_vram_gib": round(int(free_vram_mib) / 1024, 2),
        "ram_total_gib": round(total_ram / 1024**3, 2),
        "ram_available_gib": round(available_ram / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1_000_000_000, 2),
        "torch": importlib.metadata.version("torch"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "nvcc": shutil.which("nvcc"),
    }
    problems = []
    if "T4" not in gpu_name:
        problems.append(f"T4が必要です。現在: {gpu_name}")
    if total_ram < MIN_RAM_GIB * 1024**3:
        problems.append(f"High-RAMランタイムが必要です。RAM合計が{MIN_RAM_GIB} GiB未満です。")
    if disk.free < MIN_FREE_DISK_GB * 1_000_000_000:
        problems.append(f"/contentに空きディスクが{MIN_FREE_DISK_GB} GB必要です。")
    if int(free_vram_mib) < MIN_FREE_VRAM_GIB * 1024:
        problems.append(f"GPUに空きVRAMが{MIN_FREE_VRAM_GIB} GiB以上必要です。")
    if facts["torch"] != "2.11.0+cu130":
        problems.append(f"PyTorch 2.11.0+cu130が必要です。現在: {facts['torch']}")
    if not facts["nvcc"]:
        problems.append("nvccがありません。CUDA対応のT4ランタイムを選択してください。")
    if problems:
        raise RuntimeError("\n".join(problems))
    return facts


def _run_logged(
    name: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    log_dir = OUT_DIR / "notebook-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[{name}] 開始。ログ: {log_path}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    chunks: queue.Queue[bytes | None] = queue.Queue()

    def read_output() -> None:
        while data := proc.stdout.read(4096):
            chunks.put(data)
        chunks.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    pending = bytearray()
    tail: deque[str] = deque(maxlen=10)
    shown_progress: set[int] = set()
    started = last_heartbeat = time.monotonic()
    saw_eof = False
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def show_line(raw: bytes) -> None:
        line = ansi.sub("", raw.decode("utf-8", errors="replace")).strip()
        if not line:
            return
        tail.append(line)
        percent = re.search(r"(?<!\d)(\d{1,3})%", line)
        if percent:
            bucket = min(100, int(percent.group(1)) // 10 * 10)
            if bucket in shown_progress or (bucket == 0 and shown_progress):
                return
            shown_progress.add(bucket)
            print(f"[{name}] {bucket}% 進行", flush=True)
            return
        if any(word in line.lower() for word in ("start", "complete", "failed", "error", "install", "cuda", "checkpoint", "revision")):
            print(f"[{name}] {line[:180]}", flush=True)

    with log_path.open("wb") as log:
        while not saw_eof:
            try:
                chunk = chunks.get(timeout=15)
            except queue.Empty:
                chunk = b""
            if chunk is None:
                saw_eof = True
            elif chunk:
                log.write(chunk)
                pending.extend(chunk)
                while True:
                    delimiters = [idx for idx in (pending.find(b"\n"), pending.find(b"\r")) if idx >= 0]
                    if not delimiters:
                        break
                    end = min(delimiters)
                    show_line(bytes(pending[:end]))
                    del pending[: end + 1]
            if time.monotonic() - last_heartbeat >= 60:
                elapsed = int((time.monotonic() - started) / 60)
                last = tail[-1][:120] if tail else "ログに新しい出力はありません"
                print(f"[{name}] 実行中 {elapsed}分。直近: {last}", flush=True)
                log.flush()
                last_heartbeat = time.monotonic()
        if pending:
            show_line(bytes(pending))
        log.flush()
    code = proc.wait()
    reader.join(timeout=5)
    if code:
        excerpt = "\n".join(tail)[-2000:]
        raise RuntimeError(f"{name}が終了コード{code}で失敗しました。ログ: {log_path}\n{excerpt}")
    print(f"[{name}] 完了。", flush=True)
    return log_path


def prepare() -> dict[str, Any]:
    """Install the pinned stack, download BF16 weights, build CUDA CLI, and convert."""
    runtime = check_runtime()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "NEMOTRON_OUT_DIR": str(OUT_DIR),
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
        }
    )
    scripts = REPO_ROOT / "scripts"
    for name in ("bootstrap.py", "prepare_cpp.py", "convert_q4_profile.py"):
        if not (scripts / name).is_file():
            raise FileNotFoundError(f"公開リポジトリに必要な準備スクリプトがありません: {scripts / name}")

    _run_logged("bootstrap", [sys.executable, str(scripts / "bootstrap.py")], env=env)
    download_code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, cache_dir={str(HF_HOME)!r}), flush=True)"
    )
    _run_logged("model-download", [sys.executable, "-c", download_code], env=env)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scripts / "convert_q4_profile.py", OUT_DIR / "convert_q4_profile.py")
    _run_logged("prepare-cpp", [sys.executable, str(scripts / "prepare_cpp.py")], env=env)
    if not DECODER_PATH.is_file() or not list(DECODER_PATH.parent.glob("*-Encoder-F16.gguf")):
        raise RuntimeError("変換後のdecoderまたはF16 encoderが見つかりません。prepare_cpp.logを確認してください。")
    result = {
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "decoder": str(DECODER_PATH),
        "encoder_files": [str(path) for path in DECODER_PATH.parent.glob("*-Encoder-F16.gguf")],
        "runtime": runtime,
        "logs": [str(path) for path in (OUT_DIR / "notebook-logs").glob("*.log")],
    }
    print("モデルの準備が完了しました。推論はまだ実行していません。")
    return result


def _validate_audio(path: Path) -> tuple[int, float]:
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise FileNotFoundError(f"WAV音声ファイルが見つかりません: {path}")
    try:
        import soundfile

        info = soundfile.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"WAV音声を読み取れません: {path}: {exc}") from exc
    if info.format != "WAV" or info.channels != 1:
        raise ValueError("入力はモノラルWAVにしてください。PCMとfloat WAVを利用できます。")
    return info.samplerate, info.duration


def _find_binary() -> Path:
    candidates = sorted(path for path in (NEMO_REPO / "build").rglob("nemo-speech") if path.is_file() and os.access(path, os.X_OK))
    if not candidates:
        raise FileNotFoundError("nemo-speech実行ファイルがありません。先にprepare()を実行してください。")
    return candidates[0]


def transcribe(wav: str | Path, gpu_layers: int = 0) -> dict[str, Any]:
    """Transcribe one mono PCM WAV with the native CLI and save its logs and result."""
    if not isinstance(gpu_layers, int) or not 0 <= gpu_layers <= 999:
        raise ValueError("gpu_layersは0から999までの整数にしてください。")
    audio_path = Path(wav).resolve()
    sample_rate, duration = _validate_audio(audio_path)
    binary = _find_binary()
    if not DECODER_PATH.is_file():
        raise FileNotFoundError(f"decoderがありません。先にprepare()を実行してください: {DECODER_PATH}")

    result_dir = OUT_DIR / "transcripts"
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}-{audio_path.stem}-layers{gpu_layers}"
    stdout_path = result_dir / f"{stem}.txt"
    stderr_path = result_dir / f"{stem}.log"
    command = [
        str(binary), "--verbose", "transcribe", str(audio_path),
        "--model", str(DECODER_PATH), "--device", "cuda:0", "-l", "ja",
        "--instruction", INSTRUCTION, "--no-warmup",
        "--asr.backend.threads", str(min(8, os.cpu_count() or 1)),
        "--asr.salm.max_sequences", "1", "--asr.salm.n_ctx_per_sequence", "1024",
        "--asr.salm.max_context", "1024", "--asr.salm.n_batch", "256",
        "--asr.salm.encoder_max_batch", "1", "--asr.salm.mtp_draft", "0",
        "--asr.salm.max_new_tokens", "128",
    ]
    env = os.environ.copy()
    env.update({"NEMOTRON_GPU_LAYERS": str(gpu_layers), "OMP_NUM_THREADS": str(min(8, os.cpu_count() or 1))})
    started = time.perf_counter()
    print(f"音声 {audio_path.name} を文字起こしします。decoder GPU layers={gpu_layers}。")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE, env=env, bufsize=0)
        assert process.stderr is not None

        def drain_stderr() -> None:
            while data := process.stderr.read(4096):
                stderr_file.write(data)
                stderr_file.flush()

        reader = threading.Thread(target=drain_stderr, daemon=True)
        reader.start()
        while process.poll() is None:
            time.sleep(15)
            if process.poll() is None:
                print("推論中です。出力ログは保存しています。", flush=True)
        reader.join(timeout=10)
    elapsed = time.perf_counter() - started
    if process.returncode:
        excerpt = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"文字起こしが失敗しました。ログ: {stderr_path}\n{excerpt}")
    transcript = re.sub(r"\s*<\|im_end\|>\s*$", "", stdout_path.read_text(encoding="utf-8", errors="replace")).strip()
    if not transcript:
        raise RuntimeError(f"文字起こし結果が空です。標準出力を確認してください: {stdout_path}")
    result = {
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "audio": str(audio_path),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "audio_sample_rate_hz": sample_rate,
        "audio_duration_seconds": round(duration, 3),
        "gpu_layers": gpu_layers,
        "quantization": "experimental Q4_0 routed experts / Q8_0 dense; F16 speech encoder",
        "transcript": transcript,
        "elapsed_seconds_including_model_load": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        "finished_at_utc": _utc_now(),
    }
    result_path = result_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_json"] = str(result_path)
    return result


def compare_samples(gpu_layers: int = 0) -> list[dict[str, Any]]:
    """Run each bundled Japanese sample once and return the fresh transcripts."""
    return [transcribe(sample["path"], gpu_layers=gpu_layers) for sample in SAMPLES]


def export_results(runs: list[dict[str, Any]], source_commit: str) -> Path:
    """Write a small report/log ZIP. Model weights and input audio are excluded."""
    report_dir = OUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "nemotron-transcribe-report.json"
    report = {
        "schema_version": 1,
        "status": "complete" if runs and all(item.get("status") == "complete" for item in runs) else "incomplete",
        "source_commit": source_commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "created_at_utc": _utc_now(),
        "runs": runs,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = report_dir / f"nemotron-transcribe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    allowlist = [
        OUT_DIR / "runtime.json",
        OUT_DIR / "cpp-status.json",
        OUT_DIR / "prepare_cpp.log",
        report_path,
    ]
    for run in runs:
        allowlist.extend(Path(run[key]) for key in ("stdout_log", "stderr_log", "result_json") if run.get(key))
    allowlist.extend((OUT_DIR / "notebook-logs").glob("*.log"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        seen: set[Path] = set()
        for path in allowlist:
            path = path.resolve()
            if path.is_file() and path not in seen:
                archive.write(path, arcname=str(path.relative_to(OUT_DIR.resolve())))
                seen.add(path)
    return zip_path
