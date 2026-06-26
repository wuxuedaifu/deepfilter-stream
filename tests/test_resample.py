import numpy as np
from deepfilter_stream.resample import StreamResampler


def test_passthrough_when_rates_equal():
    rs = StreamResampler(48000, 48000)
    x = np.random.randn(100).astype(np.float32)
    np.testing.assert_array_equal(rs.process(x), x)


def test_upsample_total_length_matches_ratio():
    rs = StreamResampler(44100, 48000)
    sig = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    out = []
    for i in range(0, len(sig), 441):
        out.append(rs.process(sig[i:i + 441], last=(i + 441 >= len(sig))))
    y = np.concatenate(out)
    assert abs(len(y) - 48000) <= 2  # ~1s at 48k


def test_streaming_matches_oneshot_no_boundary_artifacts():
    sig = np.sin(2 * np.pi * 300 * np.arange(16000) / 16000).astype(np.float32)
    one = StreamResampler(16000, 48000).process(sig, last=True)
    rs = StreamResampler(16000, 48000)
    blk = [rs.process(sig[i:i + 160], last=(i + 160 >= len(sig))) for i in range(0, len(sig), 160)]
    streamed = np.concatenate(blk)
    m = min(len(one), len(streamed))
    # Block boundaries must not introduce discontinuities: streamed == one-shot.
    assert np.max(np.abs(one[:m] - streamed[:m])) < 1e-4


def test_reset_restarts_state():
    rs = StreamResampler(16000, 48000)
    x = np.ones(160, dtype=np.float32)
    a = rs.process(x)
    rs.reset()
    b = rs.process(x)
    np.testing.assert_allclose(a, b, atol=1e-6)
