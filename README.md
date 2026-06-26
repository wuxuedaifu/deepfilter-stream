# deepfilter-stream

Real-time streaming noise cancellation with DeepFilterNet3 on ONNX Runtime.

`deepfilter-stream` wraps the [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet)
model (arXiv:2110.05588) in a minimal, dependency-light Python API. It ships the model as an
ONNX graph so that inference runs entirely on CPU via ONNX Runtime — no PyTorch required.
The library is designed for low-latency, real-time pipelines: a single stream runs at roughly
**7x faster than real-time** on a modern CPU core, measured end-to-end algorithmic latency is
**~32 ms**, and a single machine can run **16+ concurrent real-time streams** on a typical CPU.

## Installation

```bash
pip install deepfilter-stream
```

The ONNX model (~13 MB) downloads automatically on first use and is cached in the user's
platform cache directory. See [Model download](#model-download) below.

## Quickstart

### Library API

```python
from deepfilter_stream import DeepFilterModel, Denoiser

# Option A: shared model, one stream per thread
model = DeepFilterModel()
stream = model.new_stream(atten_lim_db=None)  # or atten_lim_db=20.0 to blend in dry signal

# Feed arbitrary-length chunks at any sample rate; returns enhanced float32 mono
import numpy as np
noisy = np.random.randn(9600).astype(np.float32)  # 0.2 s at 48 kHz
enhanced = stream.process(noisy, sr=48000)

# Flush any buffered tail at end of file
tail = stream.flush()

# Reset stream state without reloading the model (e.g. next utterance)
stream.reset()

# Option B: standalone Denoiser (loads its own model internally)
denoiser = Denoiser()
enhanced = denoiser.process(noisy, sr=48000)
```

### Frame-by-frame API (lowest latency)

```python
model = DeepFilterModel()
stream = model.new_stream()

frame_size = stream.frame_size   # 512 samples at 48 kHz
frame = np.zeros(frame_size, dtype=np.float32)
enhanced_frame = stream.process_frame(frame)
```

### Live demo (mic -> denoise -> speakers)

> **Warning: wear headphones** to avoid feedback when using the live demo.

```bash
deepfilter-stream                          # use default mic/speakers
deepfilter-stream --list-devices           # list audio devices
deepfilter-stream --input-device 2 --output-device 4
deepfilter-stream --atten-lim-db 20       # blend 20 dB noise reduction with dry signal
```

## API reference

| Symbol | Description |
|---|---|
| `DeepFilterModel(model_path=None, providers=None, intra_op_num_threads=None, inter_op_num_threads=None)` | Loads the ONNX session. Thread-safe; share across threads. |
| `model.new_stream(atten_lim_db=None)` | Returns a new `Denoiser` stream bound to this model. |
| `stream.process(samples, sr)` | Process a chunk of audio (any length, any sample rate). Returns float32 mono at `sr`. |
| `stream.flush()` | Drain buffered tail; call at end of file/clip. |
| `stream.process_frame(frame)` | Process exactly one 512-sample frame at 48 kHz. |
| `stream.reset()` | Reset stream state (GRU hidden states + buffers). |
| `stream.sample_rate` | `48000` |
| `stream.frame_size` | `512` |
| `stream.latency_ms` | Per-frame STFT minimum (~10.7 ms for 512 samples at 48 kHz). |

## Latency

- **Hop size:** 512 samples @ 48 kHz = **10.67 ms** per frame.
- **`latency_ms` property:** reports the per-frame STFT minimum (~10.7 ms), which is the
  minimum algorithmic offset introduced by framing alone.
- **Measured end-to-end algorithmic latency:** ~**32 ms** (≈ 3 hops), measured via impulse
  benchmark. This reflects the actual signal delay through the STFT/iSTFT and GRU pipeline.
- **Real-world latency** adds device/buffer round-trip time on top of the algorithmic offset.

## Performance

Measured on a single CPU core (no GPU required):

| Metric | Value |
|---|---|
| Single-stream RTF | ~0.145 (~7x faster than real-time) |
| End-to-end algorithmic latency | ~32 ms (impulse benchmark, ~3 hops) |
| Concurrent real-time streams | 16+ on a typical CPU |
| Hop duration | 10.67 ms (512 samples @ 48 kHz) |

## Concurrency

The `DeepFilterModel` session is **thread-safe and shareable**. Each `Denoiser` stream holds
its own GRU hidden state and is **not** thread-safe — use **one stream per thread**.

```python
from deepfilter_stream import DeepFilterModel

# For servers with many concurrent streams, limit ONNX intra-op threads
# so that thread contention across streams does not reduce throughput:
model = DeepFilterModel(intra_op_num_threads=1)

# Each worker thread gets its own stream; model is shared
def worker():
    stream = model.new_stream()
    # ... process audio
```

## Model download

The ONNX model is downloaded automatically on first use and cached under the platform cache
directory (e.g. `~/.cache/deepfilter-stream/dfn3-512-v1/` on Linux).

To use a local copy, set the environment variable:

```bash
export DEEPFILTER_STREAM_MODEL_DIR=/path/to/folder
# folder must contain: denoiser_model.onnx, initial_states.npz, meta.json
```

## Attribution

`deepfilter-stream` bundles the DeepFilterNet3 model weights. The original work:

- Repository: <https://github.com/Rikorose/DeepFilterNet>
- Paper: arXiv:2110.05588 — "DeepFilterNet: A Low Complexity Speech Enhancement Framework
  for Full-Band Audio based on Deep Filtering"

Dual-licensed under **MIT** and **Apache 2.0** (your choice). See `LICENSE` and `NOTICE`.
