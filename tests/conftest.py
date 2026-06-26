import pathlib
import pytest

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session", autouse=True)
def _use_local_assets():
    """Point the library at staged test assets so no network/download happens in CI."""
    import os
    key = "DEEPFILTER_STREAM_MODEL_DIR"
    old = os.environ.get(key)
    os.environ[key] = str(DATA)
    yield
    if old is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old


@pytest.fixture
def require_model():
    if not (DATA / "denoiser_model.onnx").exists():
        pytest.skip("denoiser_model.onnx not staged in tests/data (fetch from model Release)")
