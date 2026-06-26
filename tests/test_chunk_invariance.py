import numpy as np
import soundfile as sf
from deepfilter_stream import Denoiser
from deepfilter_stream._meta import SAMPLE_RATE


def _run_in_chunks(noisy, chunk):
    den = Denoiser()
    out = [den.process(noisy[i:i + chunk], SAMPLE_RATE) for i in range(0, len(noisy), chunk)]
    out.append(den.flush())
    return np.concatenate(out)


def test_chunk_size_does_not_change_output(data_dir, require_model):
    noisy, sr = sf.read(data_dir / "noisy_2s_48k.wav", dtype="float32")
    a = _run_in_chunks(noisy, 9600)   # 200 ms chunks
    b = _run_in_chunks(noisy, 512)    # one hop
    c = _run_in_chunks(noisy, len(noisy))  # one shot
    m = min(len(a), len(b), len(c))
    assert np.max(np.abs(a[:m] - b[:m])) < 1e-5
    assert np.max(np.abs(a[:m] - c[:m])) < 1e-5


def test_total_output_length_tracks_input_at_48k(data_dir, require_model):
    noisy, sr = sf.read(data_dir / "noisy_2s_48k.wav", dtype="float32")
    y = _run_in_chunks(noisy, 9600)
    # 48k passthrough: streaming output length == padded multiple of frame_size.
    assert 0 <= len(y) - len(noisy) < 512
