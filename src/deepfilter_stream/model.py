"""Shared, thread-safe DeepFilterNet ONNX session; mints per-stream Denoiser objects."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from . import assets
from ._meta import SAMPLE_RATE


class DeepFilterModel:
    def __init__(
        self,
        model_path: str | None = None,
        providers: list | None = None,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        paths = assets.ensure_assets()
        onnx_path = model_path or str(paths["onnx"])

        so = ort.SessionOptions()
        if intra_op_num_threads is not None:
            so.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads is not None:
            so.inter_op_num_threads = inter_op_num_threads
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.frame_size = int(self.session.get_inputs()[0].shape[0])
        self.sample_rate = SAMPLE_RATE

        with np.load(paths["initial_states"]) as z:
            # keep in input order (skip input_frame at index 0)
            self._init = {name: z[name].astype(np.float32) for name in self.input_names[1:]}

    def initial_states(self) -> dict:
        return {k: v.copy() for k, v in self._init.items()}

    def new_stream(self, atten_lim_db: float | None = None):
        from .denoiser import Denoiser
        return Denoiser(model=self, atten_lim_db=atten_lim_db)
