#!/usr/bin/env python3
"""Generate the short, source-driven Google Colab notebook."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Nemotron_T4_Colab.ipynb"
CODE_COMMIT = "33a38c8faaed84eb8a4c781f35327d2c21a9e2f2"

CELLS = [
    (
        "markdown",
        """# Nemotron TranscribeをT4で試す

このノートブックはNeMo-Speech.cppを使い、CPUとGPUを併用して日本語音声を文字起こしします。最初に公開コードとモデルの固定リビジョンを取得し、T4上でビルド・変換してから音声を処理します。

## 0. 実行前に確認する

- Colabのランタイムを **T4 GPU / High RAM** にします。検証環境のRAMは約51 GiBです。事前チェックではRAM50 GiB以上、空きディスク180 GB以上を要求します。
- 変換中はBF16モデル約67 GB、変換用データ約66 GB、量子化出力約22 GBを使います。処理後もモデル重みは `/content` に残り、ZIPには入りません。
- 既定の `GPU_LAYERS = 0` は言語モデルのdecoderをCPUで実行し、音声encoderはGPUを使います。decoderは実験的なQ4_0 experts / Q8_0 dense構成です。事前チェックでは空きVRAM4 GiB以上を要求します。物理的な4 GB GPUは未検証です。
- 既定の0層を使ってください。8層は前回約5 GiBのGPUメモリを使ったため、空きVRAMは6 GiB以上を目安にしてください。自動の事前確認は4 GiBの共通条件だけを調べます。
- 初回のセットアップはモデル取得、ビルド、変換を行うため時間がかかります。各段階のログは `/content/nemotron-transcribe` に保存します。""",
    ),
    (
        "code",
        """#@title 実行設定
from pathlib import Path

REPO_URL = "https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab.git"
CODE_REVISION = "main"  # 公開前に固定コミットSHAへ置き換える
REPO_ROOT = Path("/content/nemotron-transcribe-colab-lab")
GPU_LAYERS = 0  #@param [0, 8] {type:"integer"}
USE_UPLOAD = False  #@param {type:"boolean"}
DOWNLOAD_REPORT = False  #@param {type:"boolean"}

if GPU_LAYERS not in (0, 8):
    raise ValueError("GPU_LAYERSは0または8にしてください。")
print(f"取得元: {REPO_URL}\\nコード指定: {CODE_REVISION}\\ndecoder GPU layers: {GPU_LAYERS}")""",
    ),
    (
        "markdown",
        """## 1. 公開コードを固定して取得する

コードはリポジトリの scripts/lab.py から読み込みます。mainを使う公開準備中は固定SHAへ差し替えます。固定版では取得したコミットをノートブックと結果ZIPに記録します。""",
    ),
    (
        "code",
        """import re
import subprocess
import sys

if not re.fullmatch(r"main|[0-9a-f]{40}", CODE_REVISION):
    raise ValueError("CODE_REVISIONはmainまたは40桁のコミットSHAにしてください。")
if REPO_ROOT.exists() and not (REPO_ROOT / ".git").exists():
    raise RuntimeError(f"保存先がGitリポジトリではありません: {REPO_ROOT}")
if not REPO_ROOT.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", REPO_URL, str(REPO_ROOT)], check=True)
remote = subprocess.check_output(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"], text=True).strip()
if remote.removesuffix(".git").lower() != REPO_URL.removesuffix(".git").lower():
    raise RuntimeError(f"保存先のoriginが想定と異なります: {remote}")
if subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip():
    raise RuntimeError("取得先に未保存の変更があります。内容を退避してから再実行してください。")
subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", CODE_REVISION], check=True)
subprocess.run(["git", "-C", str(REPO_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
SOURCE_COMMIT = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip().lower()
if re.fullmatch(r"[0-9a-f]{40}", CODE_REVISION) and SOURCE_COMMIT != CODE_REVISION:
    raise RuntimeError(f"取得したSHAが一致しません: {SOURCE_COMMIT}")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import lab
print(f"取得したコードSHA: {SOURCE_COMMIT}")""",
    ),
    (
        "markdown",
        """## 2. ランタイムを確認する

この確認はPyTorchをimportせず、GPU・RAM・空き容量・`nvcc`・既存PyTorchのバージョンを読みます。条件を満たさない場合は、数十GBのモデルを取得する前に停止します。""",
    ),
    (
        "code",
        """from IPython.display import display

runtime = lab.check_runtime()
display(runtime)""",
    ),
    (
        "markdown",
        """## 3. モデルと実行ファイルを準備する

この段階で依存関係を整え、BF16モデルを取得し、T4向けネイティブ実行ファイルをビルドしてQ4_0 / Q8_0に変換します。ログは保存しながら、セルには節目と簡潔な進捗だけを表示します。""",
    ),
    (
        "code",
        """prepared = lab.prepare()
display({key: value for key, value in prepared.items() if key != "runtime"})""",
    ),
    (
        "markdown",
        """## 4. サンプル音声を聞く

同梱のGoogle FLEURS日本語音声と参照文です。各音声はここで一度だけ再生できます。次のセルはリポジトリ内に保存したT4の実行結果を読みます。今回の実行結果とは分けて表示します。""",
    ),
    (
        "code",
        """from IPython.display import Audio, display
import soundfile

for sample in lab.SAMPLES:
    print(f"ID {sample['id']} / 参照文: {sample['reference']}")
    audio, sample_rate = soundfile.read(str(sample["path"]), dtype="float32")
    display(Audio(audio, rate=sample_rate, autoplay=False))""",
    ),
    (
        "markdown",
        """### 前回保存済みのT4・Q4_0 / CPU併用の出力

元データは [保存済み結果JSON](https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/evidence/2026-10-30/small-gpu-results.json) です。T4上でdecoderのGPU layer数を0にした保存済み出力だけを表示します。4 GiBの空きVRAMを残すよう同じT4上でメモリを確保した実験で、4 GiB GPUでの実行ではありません。2音声の参考例で、今回の結果や一般的な日本語精度を示すものではありません。""",
    ),
    (
        "code",
        """import json

saved_path = REPO_ROOT / "evidence" / "2026-10-30" / "small-gpu-results.json"
saved_evidence = json.loads(saved_path.read_text(encoding="utf-8"))
saved_runs = [run for run in saved_evidence["runs"] if run["gpu_layers"] == 0]
for run in saved_runs:
    transcript = run["transcript"].removesuffix("<|im_end|>").strip()
    print(f"ID {run['sample_id']} / T4保存済み出力: {transcript}")""",
    ),
    (
        "markdown",
        """## 5. 今回の音声を文字起こしする

既定では2件を `GPU_LAYERS` の値で1回ずつ処理します。各出力に音声SHA256、設定、経過秒、ログの保存先を記録します。量子化は実験的な経路なので、出力を正解文と見比べてください。""",
    ),
    (
        "code",
        """RUNS = lab.compare_samples(gpu_layers=GPU_LAYERS)
for run in RUNS:
    print(f"ID {Path(run['audio']).name} / decoder GPU layers={run['gpu_layers']}")
    print(run["transcript"])
    print(f"経過時間: {run['elapsed_seconds_including_model_load']:.1f}秒 / JSON: {run['result_json']}\\n")""",
    ),
    (
        "markdown",
        """## 6. 自分のWAVを追加する（任意）

アップロードは初期設定で無効です。使う場合は設定セルで `USE_UPLOAD = True` を選び、設定セルを再実行してから、この下のアップロード用セルを実行してください。入力はモノラルWAVで、PCM16またはfloat32、サンプルレート8〜96 kHzにしてください。長時間音声は未検証です。""",
    ),
    (
        "code",
        """if USE_UPLOAD:
    from google.colab import files

    uploaded = files.upload()
    if uploaded:
        upload_path = Path("/content") / next(iter(uploaded))
        RUNS.extend([lab.transcribe(upload_path, gpu_layers=GPU_LAYERS)])
        print(RUNS[-1]["transcript"])
else:
    print("アップロードは無効です。設定セルでUSE_UPLOAD=Trueにすると選択できます。")""",
    ),
    (
        "markdown",
        """## 7. 結果とログを書き出す

ZIPには実行レポートと必要なログを含めます。認識文、音声ファイル名、実行設定も保存されます。モデル重みと入力音声ファイルは含みません。ブラウザーからダウンロードする場合は、設定セルで `DOWNLOAD_REPORT = True` を選び、設定セルを再実行してから、この下の書き出し用セルを実行してください。Colab CLIや自動実行では初期設定のまま使います。""",
    ),
    (
        "code",
        """report_path = lab.export_results(RUNS, source_commit=SOURCE_COMMIT)
print(f"保存したレポート: {report_path}")
from IPython.display import FileLink, display
display(FileLink(str(report_path)))
if DOWNLOAD_REPORT:
    from google.colab import files

    files.download(str(report_path))""",
    ),
]


def _cell(kind: str, source: str, index: int) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": kind,
        "id": f"nemotron-{index:02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build(code_revision: str = CODE_COMMIT) -> Path:
    if code_revision != "main" and (len(code_revision) != 40 or any(ch not in "0123456789abcdef" for ch in code_revision.lower())):
        raise ValueError("code_revision must be 'main' or a 40-character commit SHA")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        if kind == "code" and 'CODE_REVISION = "main"  # 公開前に固定コミットSHAへ置き換える' in source:
            comment = "公開準備中の仮値。公開版では固定SHA" if code_revision == "main" else "固定コミットSHA"
            source = source.replace(
                'CODE_REVISION = "main"  # 公開前に固定コミットSHAへ置き換える',
                f'CODE_REVISION = "{code_revision}"  # {comment}',
            )
        if kind == "markdown" and source.startswith("## 1. 公開コードを固定して取得する") and code_revision != "main":
            source = """## 1. 公開コードを固定して取得する

このノートブックは固定コミットから scripts/lab.py を読み込みます。取得したSHAは下に表示され、結果ZIPにも保存されます。"""
        cells.append(_cell(kind, source, index))
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": [], "gpuType": "T4", "toc_visible": True},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", default=CODE_COMMIT, help="main or a 40-character source commit SHA")
    arguments = parser.parse_args()
    print(build(arguments.code_revision))
