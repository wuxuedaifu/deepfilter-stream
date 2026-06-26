import numpy as np
import pytest
from deepfilter_stream.ringbuffer import RingBuffer


def test_write_then_pop_exact():
    rb = RingBuffer()
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.available == 10
    out = rb.pop(4)
    np.testing.assert_array_equal(out, np.arange(4, dtype=np.float32))
    assert rb.available == 6


def test_pop_across_multiple_writes():
    rb = RingBuffer()
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.write(np.array([4, 5], dtype=np.float32))
    np.testing.assert_array_equal(rb.pop(5), np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert rb.available == 0


def test_pop_more_than_available_raises():
    rb = RingBuffer()
    rb.write(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        rb.pop(3)


def test_pop_all_and_clear():
    rb = RingBuffer()
    rb.write(np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(rb.pop_all(), np.ones(7, dtype=np.float32))
    assert rb.available == 0
    rb.write(np.ones(3, dtype=np.float32))
    rb.clear()
    assert rb.available == 0


def test_write_casts_to_float32():
    rb = RingBuffer()
    rb.write(np.arange(4, dtype=np.float64))
    assert rb.pop(4).dtype == np.float32
