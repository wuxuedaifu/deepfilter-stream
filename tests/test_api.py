import deepfilter_stream


def test_version_string():
    assert isinstance(deepfilter_stream.__version__, str)
    assert deepfilter_stream.__version__.count(".") >= 2
