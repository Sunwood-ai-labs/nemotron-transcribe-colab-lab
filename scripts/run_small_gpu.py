import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/content/nemotron-transcribe')
repo = Path('/content/nemo-speech')
build = repo / 'build/cuda-asr'
binaries = [path for path in (repo / 'build').rglob('nemo-speech') if path.is_file() and os.access(path, os.X_OK)]
assert binaries, 'nemo-speech executable not found'
binary = binaries[0]
model = Path(os.environ.get('NEMOTRON_DECODER_PATH', '/content/nemotron-gguf/decoder-Q4_0.gguf'))
assert model.is_file()
state = {'status': 'running', 'binary': str(binary), 'decoder': str(model), 'decoder_bytes': model.stat().st_size, 'encoder_files': {str(file): file.stat().st_size for file in model.parent.glob('*-Encoder-F16.gguf')}, 'quantization': 'Q4_0 routed experts, Q8_0 dense linears, BF16 embeddings/head, F16 speech encoder; experimental profile', 'gpu': 'Tesla T4', 'patch': 'NEMOTRON_GPU_LAYERS override in salm_engine.cpp', 'runs': []}
path = root / 'small-gpu-results.json'
path.write_text(json.dumps(state, indent=2))

sys.path.insert(0, str(repo / 'llama.cpp/gguf-py'))
import gguf
reader = gguf.GGUFReader(str(model))
quant_types = {}
expert_types = {}
for tensor in reader.tensors:
    kind = gguf.GGMLQuantizationType(tensor.tensor_type).name
    quant_types[kind] = quant_types.get(kind, 0) + 1
    if 'exps' in tensor.name:
        expert_types[kind] = expert_types.get(kind, 0) + 1
state['gguf_tensor_types'] = quant_types
state['gguf_expert_tensor_types'] = expert_types
del reader

instruction = 'Transcribe this Japanese audio with punctuation. Omit verbal fillers, repeated words, and false starts while preserving the speaker\'s intended meaning. Write numbers in their standard written form.'
jobs = [('1607', '/content/fleurs-ja-02.wav', 8), ('1519', '/content/fleurs-ja-01.wav', 8), ('1607', '/content/fleurs-ja-02.wav', 0), ('1519', '/content/fleurs-ja-01.wav', 0)]
for sample_id, audio, layers in jobs:
    state['current_sample'] = sample_id
    state['current_gpu_layers'] = layers
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    command = [str(binary), '--verbose', 'transcribe', audio, '--model', str(model), '--device', 'cuda:0', '-l', 'ja', '--instruction', instruction, '--no-warmup', '--asr.backend.threads', str(min(8, os.cpu_count() or 1)), '--asr.salm.max_sequences', '1', '--asr.salm.n_ctx_per_sequence', '1024', '--asr.salm.max_context', '1024', '--asr.salm.n_batch', '256', '--asr.salm.encoder_max_batch', '1', '--asr.salm.mtp_draft', '0', '--asr.salm.max_new_tokens', '128']
    environment = os.environ.copy()
    environment['NEMOTRON_GPU_LAYERS'] = str(layers)
    environment['OMP_NUM_THREADS'] = str(min(8, os.cpu_count() or 1))
    reservation = None
    gpu_budget = None
    if layers == 0:
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        reserve_bytes = max(0, free_bytes - 4 * 1024**3)
        reservation = torch.empty(reserve_bytes, dtype=torch.uint8, device='cuda')
        gpu_budget = {'method': 'Reserve memory in this worker while native transcription runs; this is not a test on a physical 4GB GPU.', 'reserved_bytes': reserve_bytes, 'free_bytes_before_native': torch.cuda.mem_get_info()[0], 'total_device_bytes': total_bytes}
    stdout_path, stderr_path = root / f'iq4-{sample_id}-layers{layers}.txt', root / f'iq4-{sample_id}-layers{layers}.log'
    started = time.perf_counter()
    gpu_peak_mib, native_gpu_peak_mib, rss_peak_kib = 0, 0, 0
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        while process.poll() is None:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if gpu.returncode == 0:
                gpu_peak_mib = max(gpu_peak_mib, int(gpu.stdout.strip().splitlines()[0]))
            gpu_processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            for row in gpu_processes.stdout.splitlines():
                values = [value.strip() for value in row.split(',')]
                if len(values) == 2 and values[0] == str(process.pid) and values[1].isdigit():
                    native_gpu_peak_mib = max(native_gpu_peak_mib, int(values[1]))
            status = Path(f'/proc/{process.pid}/status')
            if status.exists():
                try:
                    process_status = status.read_text()
                except FileNotFoundError:
                    process_status = ''
                for line in process_status.splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        rss_peak_kib = max(rss_peak_kib, int(line.split()[1]))
            if time.perf_counter() - started > 1200:
                process.terminate()
                process.wait(timeout=30)
                state['status'] = 'failed'
                state['error'] = 'Single-sample inference exceeded 1200 seconds'
                path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
                raise RuntimeError('Single-sample inference exceeded 1200 seconds')
            time.sleep(0.5)
    run = {'sample_id': sample_id, 'gpu_layers': layers, 'gpu_budget': gpu_budget, 'command': command, 'returncode': process.returncode, 'transcript': stdout_path.read_text().strip(), 'wall_seconds_including_model_load': time.perf_counter() - started, 'gpu_peak_device_used_mib_sampled_500ms': gpu_peak_mib, 'native_process_gpu_peak_mib_sampled_500ms': native_gpu_peak_mib, 'process_rss_peak_kib_sampled_500ms': rss_peak_kib, 'audio_sha256': hashlib.sha256(Path(audio).read_bytes()).hexdigest(), 'stderr_file': str(stderr_path)}
    if reservation is not None:
        del reservation
        torch.cuda.empty_cache()
    state['runs'].append(run)
    state['status'] = 'running' if process.returncode == 0 and run['transcript'] else 'failed'
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(json.dumps(run, ensure_ascii=False), flush=True)
    if state['status'] == 'failed':
        print(stderr_path.read_text(errors='replace')[-5000:], flush=True)
        raise RuntimeError('Native transcription failed; preserved logs')
state['status'] = 'complete'
path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
print('Two Japanese recordings completed on T4; also checked a CPU-decoder configuration with the speech encoder on GPU.', flush=True)
