<div align="center">
  <h1>Nemotron Transcribe Colab Lab</h1>
  <p>Run NVIDIA Nemotron 3.5 Transcribe 30B-A4B on Google Colab with a pinned NeMo-Speech.cpp build.</p>
  <p><strong>English</strong> · <a href="README.ja.md">日本語</a></p>
</div>

<p align="center">
  <a href="https://colab.research.google.com/github/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/notebooks/Nemotron_T4_Colab.ipynb"><img alt="Open in Google Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab/actions/workflows/quality.yml"><img alt="Notebook and source quality" src="https://github.com/Sunwood-ai-labs/nemotron-transcribe-colab-lab/actions/workflows/quality.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Original glue code license" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Run it in Colab

1. Sign in to Google Colab and [open the public notebook](https://colab.research.google.com/github/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/notebooks/Nemotron_T4_Colab.ipynb).
2. Select a Tesla T4 High RAM runtime.
3. Run the bundled FLEURS examples by default. To transcribe your own recording, enable the optional upload control (`USE_UPLOAD=True`) and upload a WAV file.
4. Read the audio and transcript output below the run cell.

The notebook preflight requires a T4, at least 4 GiB of free VRAM, about 50 GiB of CPU memory, at least 180 GB of free disk, `nvcc`, and `torch==2.11.0+cu130`. These checks describe the preparation environment used for this run. They are not a general minimum-hardware claim or a guarantee that Colab will assign that runtime. The first setup downloads the model weights, builds the pinned C++ runtime, and converts the weights. Allow tens of minutes. This repository does not contain model weights. The model and source terms are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

## Read the executed notebook

[Open the saved T4 notebook in Colab](https://colab.research.google.com/github/Sunwood-ai-labs/nemotron-transcribe-colab-lab/blob/main/notebooks/results/T4_2026-10-30.ipynb).

All nine processing cells were executed with Google Colab CLI on a newly allocated T4 High RAM runtime. The notebook downloaded the checkpoint, built and converted it, transcribed both bundled clips with the CPU decoder and GPU audio encoder, and wrote the report ZIP. One transcript matched its reference; the other retained `企画後` instead of `帰国後`.

The saved notebook keeps preparation logs in a collapsible block. Its processing cells are identical to the raw execution capture and the current public notebook. Only explanatory Markdown and log presentation changed after execution. [Source bytes, outputs, hashes, and verification scope](evidence/notebook-validation-2026-10-30/README.md) are preserved separately. The task T4 has been stopped. Optional upload and browser-download controls were skipped during the automated run.

## Read the earlier small-GPU memory experiment

The recorded run completed both Japanese FLEURS clips in each configuration. The results come from four successful Colab CLI invocations on one Tesla T4.

| Language-model layers on GPU | Native transcriber GPU peak, sampled | Native transcriber RSS peak, sampled | Completed clips |
| --- | ---: | ---: | ---: |
| 8 | 5.0 GiB | 16.0 GiB | 2 of 2 |
| 0 | 2.1 GiB | 18.8 GiB | 2 of 2 |

The zero-layer configuration still uses the GPU for the audio encoder. The worker reserved GPU memory on the T4 so that 4,294,770,688 bytes remained free before the native transcriber started. The 2.1 GiB figure is the transcriber process's sampled GPU use. This was not a test on a physical 4 GB GPU. The monitor sampled every 0.5 seconds, so it may miss short peaks. RSS excludes the operating system and the monitor process.

One clip matched its reference exactly. The other produced `企画後` where the reference has `帰国後`. Two short recordings do not establish general transcription accuracy or meeting-audio quality.

The [recorded result JSON](evidence/2026-10-30/small-gpu-results.json), [captured CLI notebook](evidence/2026-10-30/executed-history.ipynb), [transcripts and logs](evidence/2026-10-30/runs/), and [verification scope](evidence/2026-10-30/verification.json) preserve the earlier experiment. The fresh public-notebook execution has its own proof record linked above.

## Review the model and source revisions

Both experiments used `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` at revision `c17667641642c4265ff2b7037dd101b7d4567634` and NeMo-Speech.cpp at commit `d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14`.

The experimental profile uses Q4_0 routed experts, Q8_0 dense linears, BF16 embeddings and output head, and an F16 speech encoder. It is different from the upstream IQ4_NL profile. The run also used a small patch to choose the number of language-model layers on the GPU.

Neither experiment tested a physical 4 GB GPU, long recordings, speaker diarization, or translation. The earlier memory experiment sampled every 0.5 seconds. The fresh notebook run used normal free VRAM on the T4 and did not repeat the memory-reservation experiment.

## Check the repository

Run the local structural checks with Python 3.12:

```console
python -m pip install nbformat
python tools/check_repository.py
python -m unittest discover -s tests -v
python tools/verify_execution.py
```

GitHub Actions runs the same checks. CI validates notebook structure, script syntax, saved execution proof, local links, and the public payload. CI does not run Colab or allocate a GPU.

## Review the licenses

The root MIT license covers only original glue code. Model, source, and dataset files use separate terms. Read [third-party notices](THIRD_PARTY_NOTICES.md) before using or redistributing them.
