import deepfilter_stream
import numpy as np
from deepfilter_stream import DeepFilterModel


def test_version_string():
    assert isinstance(deepfilter_stream.__version__, str)
    assert deepfilter_stream.__version__.count(".") >= 2


def test_atten_lim_none_is_exact(require_model):
    m = DeepFilterModel()
    a = m.new_stream(atten_lim_db=None)
    f = np.random.randn(512).astype(np.float32) * 0.1
    out = a.process_frame(f)
    assert out.shape == (512,)


def test_atten_lim_limits_suppression_energy(require_model):
    m = DeepFilterModel()
    plain = m.new_stream(atten_lim_db=None)
    limited = m.new_stream(atten_lim_db=12.0)
    rng = np.random.default_rng(0)
    e_plain = e_lim = 0.0
    for _ in range(40):
        f = (rng.standard_normal(512) * 0.1).astype(np.float32)
        e_plain += float(np.sum(plain.process_frame(f) ** 2))
        e_lim += float(np.sum(limited.process_frame(f.copy()) ** 2))
    # limited keeps more (dry-mixed) energy than fully-suppressed plain output
    assert e_lim >= e_plain


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
