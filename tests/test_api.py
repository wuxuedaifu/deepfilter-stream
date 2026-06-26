import deepfilter_stream


def test_version_string():
    assert isinstance(deepfilter_stream.__version__, str)
    assert deepfilter_stream.__version__.count(".") >= 2


def test_flush_drains_input_resampler(require_model):
    import numpy as np
    from deepfilter_stream import Denoiser
    sr = 44100
    sig = np.sin(2 * np.pi * 300 * np.arange(sr) / sr).astype(np.float32)  # 1.0s @44.1k
    den = Denoiser()
    out = []
    for i in range(0, len(sig), 4410):  # 100ms chunks
        out.append(den.process(sig[i:i+4410], sr))
    out.append(den.flush())
    y = np.concatenate(out)
    # output duration should track input duration within ~20ms (no dropped resampler tail)
    assert abs(len(y) / sr - 1.0) < 0.02, f"got {len(y)/sr:.4f}s"
