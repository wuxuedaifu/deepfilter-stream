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
    for t in threads: t.start()
    for t in threads: t.join()

    for i in range(4):
        m = min(len(ref), len(results[i]))
        assert np.max(np.abs(ref[:m] - results[i][:m])) < 1e-5  # no state bleed


def test_concurrent_calls_on_one_stream_raise(data_dir, require_model):
    noisy, _ = sf.read(data_dir / "noisy_2s_48k.wav", dtype="float32")
    stream = DeepFilterModel().new_stream()
    errors = []
    barrier = threading.Barrier(2)

    def hammer():
        try:
            barrier.wait()
            for _ in range(50):
                stream.process(noisy[:9600], SAMPLE_RATE)
        except RuntimeError as e:
            errors.append(e)

    a = threading.Thread(target=hammer); b = threading.Thread(target=hammer)
    a.start(); b.start(); a.join(); b.join()
    assert errors, "expected RuntimeError on concurrent single-stream use"
