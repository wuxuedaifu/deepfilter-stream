import numpy as np
from deepfilter_stream.model import DeepFilterModel


def test_model_loads_and_reports_contract(require_model):
    m = DeepFilterModel()
    assert m.sample_rate == 48000
    assert m.frame_size == 512
    assert m.input_names[0] == "input_frame"
    assert len(m.input_names) == 13 and len(m.output_names) == 13


def test_initial_states_are_fresh_copies(require_model):
    m = DeepFilterModel()
    a = m.initial_states()
    a["erb_norm_state"][:] = 999.0
    b = m.initial_states()
    assert not np.allclose(b["erb_norm_state"], 999.0)
    assert b["band_unit_norm_state"].shape == (1, 96, 1)


def test_new_stream_returns_denoiser(require_model):
    m = DeepFilterModel()
    s = m.new_stream()
    assert s.frame_size == 512
    assert s.sample_rate == 48000
