<div align="center">
  <h1>Nemotron Transcribe Colab Lab</h1>
  <p>固定したNeMo-Speech.cppを使い、Google ColabでNVIDIA Nemotron 3.5 Transcribe 30B-A4Bを動かします。</p>
  <p><a href="README.md">English</a> · <strong>日本語</strong></p>
</div>

<p align="center">
  <a href="https://colab.research.google.com/github/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/notebooks/Nemotron_T4_Colab.ipynb"><img alt="Google Colabで開く" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab/actions/workflows/quality.yml"><img alt="ノートブックとソースの品質チェック" src="https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab/actions/workflows/quality.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="オリジナルの補助コードのライセンス" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Colabで実行する

1. Google Colabにログインし、[公開ノートブックを開きます](https://colab.research.google.com/github/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/notebooks/Nemotron_T4_Colab.ipynb)。
2. Tesla T4 High RAMランタイムを選びます。
3. 既定では同梱のFLEURS音声を実行します。手元の音声を文字起こしする場合は、任意のアップロード設定（`USE_UPLOAD=True`）を有効にしてWAVファイルをアップロードします。
4. 実行セルの下に表示される音声と文字起こし結果を確認します。

ノートブックの事前確認では、T4、4 GiB以上の空きVRAM、約50 GiBのCPUメモリ、180 GB以上の空きディスク、`nvcc`、`torch==2.11.0+cu130`を要求します。これらは今回の準備で確認した環境条件です。一般的な最低ハードウェア要件や、Colabが必ず割り当てる構成を示すものではありません。初回セットアップではモデルの重みを取得し、固定したC++ランタイムをビルドしてから重みを変換するため、数十分かかる場合があります。このリポジトリにはモデルの重みを含めていません。モデルとソースの利用条件は[第三者ライセンスの一覧](THIRD_PARTY_NOTICES.md)で確認できます。

## T4で記録した結果を確認する

記録上は、日本語FLEURS音声2件をそれぞれの構成で処理できました。結果は、1台のTesla T4で実行したColab CLIの4回分です。

| GPUに配置した言語モデルの層数 | 文字起こしプロセスのGPU使用量の観測最大 | 文字起こしプロセスのRSS観測最大 | 完了した音声 |
| --- | ---: | ---: | ---: |
| 8 | 5.0 GiB | 16.0 GiB | 2件中2件 |
| 0 | 2.1 GiB | 18.8 GiB | 2件中2件 |

言語モデルを0層にした構成でも、音声エンコーダはGPUを使います。ネイティブ文字起こしの開始前に、ワーカーがT4のGPUメモリを予約し、空き容量を4,294,770,688 bytesにしました。2.1 GiBは文字起こしプロセス単体の観測値です。物理的な4 GB GPUでは試していません。監視は0.5秒間隔のため、短いピークは捉えられない場合があります。RSSにはOSと監視プロセスを含みません。

2音声のうち、1件は参照文と一致しました。もう1件は、参照文の「帰国後」に対して「企画後」と出力しました。この2件だけでは、一般的な文字起こし精度や会議音声での品質を判断できません。

[記録した結果JSON](evidence/2026-10-30/small-gpu-results.json)、[Colab CLIの記録ノートブック](evidence/2026-10-30/executed-history.ipynb)、[出力とログ](evidence/2026-10-30/runs/)、[検証範囲](evidence/2026-10-30/verification.json)に、元の実験記録を保存しています。CLI記録ノートブックは履歴であり、利用者向けノートブックではありません。リポジトリ版ノートブックを新しいColabランタイムで最初から最後まで実行した証拠でもありません。

## モデルとソースのリビジョンを確認する

元の実験では、`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` のリビジョン `c17667641642c4265ff2b7037dd101b7d4567634` と、NeMo-Speech.cppのコミット `d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14` を使いました。

実験用プロファイルでは、routed expertをQ4_0、dense linearをQ8_0、埋め込みと出力headをBF16、音声エンコーダをF16にしました。upstreamのIQ4_NLプロファイルとは異なります。また、GPUに載せる言語モデルの層数を指定するパッチを使いました。NeMo-Speech.cppを変更せず全層をGPUへ載せた実行とは報告していません。

元の実験では、物理的な4 GB GPU、長時間音声、話者分離、翻訳を確認していません。日本語音声2件を使い、メモリを0.5秒間隔で観測しました。リポジトリ版ノートブックの新規ランタイムでの実行結果は、追加された新しい実行記録で確認してください。

## リポジトリを検査する

Python 3.10以降でローカルの構造チェックを実行します。

```console
python -m pip install nbformat
python tools/check_repository.py
```

GitHub Actionsでも同じチェックを実行します。CIはノートブック構造、スクリプト構文、記録済みの根拠、ローカルリンク、公開ファイルを確認します。Colabを起動したりGPUを確保したりはしません。

## ライセンスを確認する

ルートのMITライセンスは、このリポジトリ向けに作った補助コードだけを対象にします。モデル、ソース、データセットには別の利用条件があります。利用・再配布の前に[第三者ライセンスの一覧](THIRD_PARTY_NOTICES.md)を確認してください。
