import numpy as np
from deepfilter_stream import assets


def test_env_dir_resolves_without_network(require_model, data_dir, monkeypatch):
    monkeypatch.setenv("DEEPFILTER_STREAM_MODEL_DIR", str(data_dir))
    paths = assets.ensure_assets()
    assert paths["onnx"].exists()
    assert paths["initial_states"].exists()
    assert paths["meta"].exists()


def test_initial_states_npz_has_12_named_arrays(require_model, data_dir, monkeypatch):
    monkeypatch.setenv("DEEPFILTER_STREAM_MODEL_DIR", str(data_dir))
    p = assets.ensure_assets()["initial_states"]
    with np.load(p) as z:
        assert len(z.files) == 12
        assert "erb_norm_state" in z.files
        assert z["band_unit_norm_state"].shape == (1, 96, 1)


def test_sha256_file(data_dir):
    h = assets.sha256_file(data_dir / "meta.json")
    assert len(h) == 64
    assert h == "f069011a01849629ad23fbb1d00f4417cf106d5e316e3f7fbcba65cce3440818"
