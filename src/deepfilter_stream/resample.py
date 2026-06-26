"""Stateful streaming resampler (soxr) that preserves filter state across blocks."""
import numpy as np
import soxr


class StreamResampler:
    def __init__(self, in_rate: int, out_rate: int, quality: str = "VHQ") -> None:
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.quality = quality
        self._rs = None if self.in_rate == self.out_rate else self._make()

    def _make(self):
        return soxr.ResampleStream(
            self.in_rate, self.out_rate, 1, dtype="float32", quality=self.quality
        )

    def process(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if self._rs is None:
            return x
        return np.asarray(self._rs.resample_chunk(x, last=last), dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        if self._rs is not None:
            self._rs = self._make()
