"""Scale concurrent streams; report aggregate RTF and max real-time stream count."""
import threading, time
import numpy as np
from deepfilter_stream import DeepFilterModel


def run_n(n, seconds=5.0):
    model = DeepFilterModel(intra_op_num_threads=1)
    f = model.frame_size
    n_frames = int(seconds * model.sample_rate / f)
    frames = [np.random.randn(f).astype(np.float32) * 0.1 for _ in range(n_frames)]
    times = {}

    def worker(idx):
        s = model.new_stream()
        for fr in frames[:20]:
            s.process_frame(fr)
        t0 = time.perf_counter()
        for fr in frames:
            s.process_frame(fr)
        times[idx] = time.perf_counter() - t0

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    audio_s = n_frames * f / model.sample_rate
    agg_rtf = max(times.values()) / audio_s
    print(f"N={n:2d}  wall={wall:.2f}s  per-stream-RTF(max)={agg_rtf:.3f}  "
          f"realtime={'OK' if agg_rtf < 1 else 'OVER'}")


if __name__ == "__main__":
    for n in (1, 2, 4, 8, 16):
        run_n(n)
