# Third-party notices

The root `LICENSE` applies only to original glue code in this repository. It does not change the terms for model materials, dataset files, upstream source, or dependencies.

## NVIDIA Nemotron 3.5 Transcribe

The notebook downloads `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` at revision [`c17667641642c4265ff2b7037dd101b7d4567634`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/tree/c17667641642c4265ff2b7037dd101b7d4567634). NVIDIA's model card identifies the license as [OpenMDW-1.1](https://openmdw.ai/license/1-1/). The repository does not include model weights.

## NVIDIA NeMo-Speech.cpp

Colab fetches [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14) at commit `d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14`. NVIDIA's [Apache-2.0 license](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14/LICENSE), [NOTICE](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14/NOTICE), and [third-party notices](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14/THIRD_PARTY_NOTICES.md) apply to that source and its dependencies. The notebook applies the GPU-layer change recorded in [`n_gpu_layers.patch`](evidence/2026-10-30/n_gpu_layers.patch). That patch and the source fragment in `scripts/prepare_cpp.py` are NeMo-Speech.cpp modifications and are not relicensed by the root MIT license.

## Google FLEURS

`audio/fleurs-ja-01.wav` and `audio/fleurs-ja-02.wav` are Japanese validation examples with dataset IDs 1519 and 1607 from the [Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs). The dataset card lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset audio was exported as mono, 16 kHz, float32 WAV. The speech content was not edited. The notebook also accepts user-provided WAV files.
