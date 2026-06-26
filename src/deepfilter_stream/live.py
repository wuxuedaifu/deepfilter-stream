"""Real-time mic -> denoise -> speakers demo. Inference runs on a worker thread."""
from __future__ import annotations

import argparse
import queue
import sys
import threading

import numpy as np

from . import DeepFilterModel
from ._meta import SAMPLE_RATE


def _worker(stream, in_q: queue.Queue, out_buf, stop: threading.Event):
    while not stop.is_set():
        try:
            block = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        y = stream.process(block, stream.sample_rate)
        if y.size:
            with out_buf["lock"]:
                out_buf["data"] = np.concatenate([out_buf["data"], y])


def main(argv=None) -> int:
    import sounddevice as sd

    p = argparse.ArgumentParser(prog="deepfilter-stream")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--input-device", default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--atten-lim-db", type=float, default=None)
    p.add_argument("--blocksize", type=int, default=512, help="device frames per callback")
    p.add_argument("--samplerate", type=int, default=SAMPLE_RATE)
    a = p.parse_args(argv)

    if a.list_devices:
        print(sd.query_devices())
        return 0

    print("WARNING: use headphones to avoid feedback (mic picking up the denoised output).")
    model = DeepFilterModel()
    stream = model.new_stream(atten_lim_db=a.atten_lim_db)

    in_q: queue.Queue = queue.Queue(maxsize=64)
    out_buf = {"data": np.zeros(0, np.float32), "lock": threading.Lock()}
    stop = threading.Event()
    t = threading.Thread(target=_worker, args=(stream, in_q, out_buf, stop), daemon=True)
    t.start()

    def callback(indata, outdata, frames, time, status):  # noqa: ANN001
        if status:
            print(status, file=sys.stderr)
        try:
            in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass
        with out_buf["lock"]:
            avail = out_buf["data"]
            if avail.shape[0] >= frames:
                outdata[:, 0] = avail[:frames]
                out_buf["data"] = avail[frames:]
            else:
                outdata[:, 0] = 0.0  # underflow -> silence, never block

    try:
        with sd.Stream(
            samplerate=a.samplerate, blocksize=a.blocksize, dtype="float32",
            channels=1, callback=callback,
            device=(a.input_device, a.output_device),
        ):
            print(f"Running at {a.samplerate} Hz, blocksize {a.blocksize}. Ctrl+C to stop.")
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
