import pathlib
import pytest

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session", autouse=True)
def _use_local_assets(tmp_path_factory):
    """Point the library at staged test assets so no network/download happens in CI."""
    import os
    os.environ["DEEPFILTER_STREAM_MODEL_DIR"] = str(DATA)
    yield
