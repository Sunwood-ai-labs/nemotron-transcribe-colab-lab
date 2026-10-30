# Public notebook execution

The nine processing cells of the published notebook were executed in order with Google Colab CLI on a newly allocated T4 High RAM runtime. The run downloaded the pinned checkpoint, built the native runtime, converted its weights, transcribed both bundled recordings with `GPU_LAYERS=0`, and created the report ZIP. The task runtime was then stopped.

- [Readable saved notebook](../../notebooks/results/T4_2026-10-30.ipynb).
- [Original source bytes](raw-source.ipynb) and [unmodified execution capture](raw-executed.ipynb).
- [Execution log](execution.log), [remote model and run facts](notebook-validation.json), and [artifact hashes](execution-provenance.json).
- [Runtime stop verification](runtime-stop.json).

The executed source was repository revision `d16e3035b9fe5f278cfee8291fe717666e611f37`; it loaded scripts from `33a38c8faaed84eb8a4c781f35327d2c21a9e2f2`. The current public and readable notebooks have the same nine processing cells. After execution, explanatory Markdown was corrected and preparation stdout was wrapped in a collapsible HTML block in the readable result. The raw capture retains the original outputs.

Both new transcriptions match the earlier cleaned-prompt outputs. One recording matches its reference; the other retains `企画後` in place of `帰国後`. The new runs used normal free VRAM on the T4. The earlier 4 GiB-free reservation experiment remains separate under [historical evidence](../2026-10-30/README.md).

The fresh decoder GGUF is 20,352,806,816 bytes. The earlier experiment's file was 20,352,806,848 bytes. They are not claimed to be byte-identical. The tensor-type counts are unchanged: 48 Q4_0 expert tensors, 122 Q8_0 tensors, four BF16 tensors, and 243 F32 tensors. The F16 speech encoder is 1,670,410,080 bytes.

The displayed durations include model loading and the Python monitor's waits. They do not measure decoder computation alone. Optional upload and browser-download controls were skipped during the automated run. Two valid mono 16 kHz PCM16 audio players are embedded in the saved notebook, but browser interaction and listening acceptance were not checked because local browser use is disabled.

Run `python tools/verify_execution.py` from the repository root to validate the saved proof without allocating a GPU or running inference.
