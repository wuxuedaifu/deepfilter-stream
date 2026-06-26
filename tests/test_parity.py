import numpy as np
import soundfile as sf
from deepfilter_stream import Denoiser
from deepfilter_stream._meta import SAMPLE_RATE


def _frame_by_frame_offline(den, noisy):
    """Replicate the reference generator: pad, run all frames, trim by d=frame_size."""
    f = den.frame_size
    orig = noisy.shape[0]
    pad_div = (f - orig % f) % f
    orig_p = orig + pad_div
    fft = 2 * f
    x = np.pad(noisy, (0, fft + pad_div)).astype(np.float32)
    out = []
    for k in range(x.shape[0] // f):
        out.append(den.process_frame(x[k * f:(k + 1) * f]))
    y = np.concatenate(out)
    d = fft - f
    return y[d:orig_p + d]


def test_offline_parity_with_reference(data_dir, require_model):
    noisy, sr = sf.read(data_dir / "noisy_2s_48k.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    ref, _ = sf.read(data_dir / "reference_2s_48k.wav", dtype="float32")
    den = Denoiser()
    out = _frame_by_frame_offline(den, noisy)
    m = min(len(out), len(ref))
    assert np.max(np.abs(out[:m] - ref[:m])) < 1e-4
