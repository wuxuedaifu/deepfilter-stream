"""Per-frame latency, RTF, and empirical algorithmic offset (impulse method)."""
import time
import numpy as np
from deepfilter_stream import Denoiser


def per_frame_latency(n=2000):
    den = Denoiser(); f = den.frame_size
    frames = [np.random.randn(f).astype(np.float32) * 0.1 for _ in range(n)]
    for fr in frames[:50]:
        den.process_frame(fr)
    ts = []
    for fr in frames:
        t0 = time.perf_counter(); den.process_frame(fr); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    audio_ms = f / den.sample_rate * 1e3
    print(f"frame={f} ({audio_ms:.2f} ms audio)")
    for q in (50, 95, 99):
        print(f"  p{q} per-frame: {np.percentile(ts, q):.3f} ms")
    print(f"  RTF (mean): {ts.mean()/audio_ms:.4f}")


def algorithmic_offset():
    den = Denoiser(); f = den.frame_size
    sig = np.zeros(f * 40, dtype=np.float32); sig[f * 5] = 1.0  # impulse
    out = np.concatenate([den.process_frame(sig[i:i+f]) for i in range(0, len(sig), f)])
    # Background is silent except the single impulse, so the output is essentially
    # just the (attenuated) impulse response: argmax of |out| locates it directly.
    # (Cross-correlating against a unit-impulse reference reduces to this same argmax,
    # so it would add cost without adding robustness here.)
    peak_in = int(np.argmax(np.abs(sig)))
    peak_out = int(np.argmax(np.abs(out)))
    lag = peak_out - peak_in
    print(f"impulse offset: {lag} samples ({lag / den.sample_rate * 1e3:.2f} ms)")


if __name__ == "__main__":
    per_frame_latency(); algorithmic_offset()
