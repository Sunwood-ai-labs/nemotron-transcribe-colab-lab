# Evidence recorded on 2026-10-30

`small-gpu-results.json` records four successful Colab CLI transcriptions on one Tesla T4. The four runs cover two Japanese FLEURS clips at 8 and 0 language-model layers on the GPU. The audio encoder still used the GPU in the 0-layer configuration.

`verification.json` records the checks applied to those results. The 0-layer run reserved memory on the T4 to leave 4,294,770,688 bytes free before native inference. It did not run on a physical 4 GB GPU.

`executed-history.ipynb` preserves the historical Colab CLI command and output capture. It has stored outputs but no execution-count metadata. It is not the public run notebook and does not prove that a fresh repository-backed notebook run completed.

`runs/` contains the four transcript outputs and their short native-process logs. `n_gpu_layers.patch` records the one-line GPU-layer control change applied to the pinned NeMo-Speech.cpp checkout.

`source-map.json` records source hashes for the files copied from the experiment workspace. Script hashes describe the initial copied versions; the notebook integration may change those files. Audio and evidence hashes describe immutable source records.
