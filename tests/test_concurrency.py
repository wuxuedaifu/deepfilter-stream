import threading
import numpy as np
import soundfile as sf
from deepfilter_stream import DeepFilterModel
from deepfilter_stream._meta import SAMPLE_RATE


def _full(stream, noisy):
    out = [stream.process(noisy[i:i + 9600], SAMPLE_RATE) for i in range(0, len(noisy), 9600)]
    out.append(stream.flush())
    return np.concatenate(out)


def test_shared_model_many_streams_match_single(data_dir, require_model):
    noisy, _ = sf.read(data_dir / "noisy_2s_48k.wav", dtype="float32")
    model = DeepFilterModel(intra_op_num_threads=1)
    ref = _full(model.new_stream(), noisy)

    results = {}

    def worker(idx):
        results[idx] = _full(model.new_stream(), noisy)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(4):
        m = min(len(ref), len(results[i]))
        assert np.max(np.abs(ref[:m] - results[i][:m])) < 1e-5  # no state bleed


def test_concurrent_calls_on_one_stream_raise(data_dir, require_model):
    import pytest
    stream = DeepFilterModel().new_stream()
    noisy = np.zeros(9600, dtype=np.float32)
    # Simulate a concurrent in-progress call by holding the stream's lock,
    # then assert a second entry is rejected deterministically.
    assert stream._lock.acquire(blocking=False) is True
    try:
        with pytest.raises(RuntimeError):
            stream.process(noisy, SAMPLE_RATE)
        with pytest.raises(RuntimeError):
            stream.flush()
    finally:
        stream._lock.release()
    # After releasing, the stream is usable again.
    out = stream.process(noisy, SAMPLE_RATE)
    assert out.dtype == np.float32
