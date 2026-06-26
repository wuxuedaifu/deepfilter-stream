"""Minimal float32 FIFO buffer that decouples variable-size writes from fixed-size reads."""
import numpy as np


class RingBuffer:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._size = 0

    @property
    def available(self) -> int:
        return self._size

    def write(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if x.size:
            self._chunks.append(x)
            self._size += x.size

    def _coalesce(self) -> None:
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]

    def pop(self, n: int) -> np.ndarray:
        if n > self._size:
            raise ValueError(f"pop({n}) but only {self._size} available")
        self._coalesce()
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        buf = self._chunks[0]
        out, rest = buf[:n].copy(), buf[n:]
        self._chunks = [rest] if rest.size else []
        self._size -= n
        return out

    def pop_all(self) -> np.ndarray:
        return self.pop(self._size)

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0
