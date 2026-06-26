import time
import numpy as np
from deepfilter_stream import Denoiser


def test_single_stream_realtime_factor(require_model):
    den = Denoiser()
    f = den.frame_size
    frames = [np.random.randn(f).astype(np.float32) * 0.1 for _ in range(300)]
    # warmup
    for fr in frames[:20]:
        den.process_frame(fr)
    t0 = time.perf_counter()
    for fr in frames:
        den.process_frame(fr)
    dt = time.perf_counter() - t0
    audio_s = len(frames) * f / den.sample_rate
    rtf = dt / audio_s
    assert rtf < 0.5, f"RTF too high: {rtf:.3f}"
