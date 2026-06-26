"""Per-stream denoiser: framing, state threading, resampling, flush."""
from __future__ import annotations

import threading
import numpy as np

from .ringbuffer import RingBuffer
from .resample import StreamResampler


class Denoiser:
    def __init__(self, model=None, atten_lim_db: float | None = None) -> None:
        if model is None:
            from .model import DeepFilterModel
            model = DeepFilterModel()
        self._model = model
        self.frame_size = model.frame_size
        self.sample_rate = model.sample_rate  # 48000
        self.atten_lim_db = atten_lim_db
        self._lock = threading.Lock()
        self.reset()

    # ---- lifecycle ----
    def reset(self) -> None:
        self._states = self._model.initial_states()
        self._in_buf = RingBuffer()
        self._out_buf = RingBuffer()
        self._in_rs: StreamResampler | None = None
        self._out_rs: StreamResampler | None = None
        self._cur_sr: int | None = None
        self._dry_delay = RingBuffer()
        if self.atten_lim_db is not None:
            self._dry_delay.write(np.zeros(self.frame_size, dtype=np.float32))  # d-sample delay
            self._lim = float(10.0 ** (-self.atten_lim_db / 20.0))

    def _guard(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Denoiser stream used from two threads at once. Use one stream per thread; "
                "call DeepFilterModel.new_stream() per thread (the model/session is shared)."
            )

    @property
    def latency_ms(self) -> float:
        # algorithmic minimum: trim offset d == frame_size samples.
        return 1000.0 * self.frame_size / self.sample_rate

    # ---- hot path ----
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != self.frame_size:
            raise ValueError(f"frame must be {self.frame_size} samples, got {frame.shape[0]}")
        feeds = {"input_frame": frame}
        feeds.update(self._states)
        res = self._model.session.run(self._model.output_names, feeds)
        enhanced = np.asarray(res[0], dtype=np.float32).reshape(-1)
        for name, val in zip(self._model.input_names[1:], res[1:]):
            self._states[name] = val
        if self.atten_lim_db is not None:
            self._dry_delay.write(frame)
            dry = self._dry_delay.pop(self.frame_size)
            enhanced = self._lim * dry + (1.0 - self._lim) * enhanced
        return enhanced

    # ---- block API ----
    def _ensure_resamplers(self, sr: int) -> None:
        if sr == self._cur_sr:
            return
        self._cur_sr = sr
        if sr == self.sample_rate:
            self._in_rs = self._out_rs = None
        else:
            self._in_rs = StreamResampler(sr, self.sample_rate)
            self._out_rs = StreamResampler(self.sample_rate, sr)

    def _run_available_frames(self) -> None:
        f = self.frame_size
        while self._in_buf.available >= f:
            enh = self.process_frame(self._in_buf.pop(f))
            self._out_buf.write(enh)

    def process(self, samples: np.ndarray, sr: int) -> np.ndarray:
        self._guard()
        try:
            x = np.ascontiguousarray(samples, dtype=np.float32)
            if x.ndim == 2:  # [frames, channels] or [channels, frames] -> mono
                x = x.mean(axis=1) if x.shape[1] <= x.shape[0] else x.mean(axis=0)
            x = x.reshape(-1)
            self._ensure_resamplers(sr)
            if self._in_rs is not None:
                x = self._in_rs.process(x)
            self._in_buf.write(x)
            self._run_available_frames()
            y = self._out_buf.pop_all()
            if self._out_rs is not None:
                y = self._out_rs.process(y)
            return y
        finally:
            self._lock.release()

    def flush(self) -> np.ndarray:
        self._guard()
        try:
            # drain any tail held by the input resampler's filter first.
            if self._in_rs is not None:
                tail = self._in_rs.process(np.zeros(0, dtype=np.float32), last=True)
                if tail.size:
                    self._in_buf.write(tail)
            # pad the final partial input frame with zeros, drain everything.
            f = self.frame_size
            rem = self._in_buf.available % f
            if rem:
                self._in_buf.write(np.zeros(f - rem, dtype=np.float32))
            self._run_available_frames()
            y = self._out_buf.pop_all()
            if self._out_rs is not None:
                y = np.concatenate([self._out_rs.process(y), self._out_rs.process(
                    np.zeros(0, dtype=np.float32), last=True)])
            return y
        finally:
            self._lock.release()
