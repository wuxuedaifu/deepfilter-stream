"""Offline file -> file using the streaming API (with whole-file pad/trim)."""
import argparse
import numpy as np
import soundfile as sf
from deepfilter_stream import Denoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--atten-lim-db", type=float, default=None)
    a = ap.parse_args()
    x, sr = sf.read(a.input, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    den = Denoiser(atten_lim_db=a.atten_lim_db)
    y = np.concatenate([den.process(x, sr), den.flush()])
    sf.write(a.output, y, sr, subtype="PCM_16")
    print(f"wrote {a.output} ({len(y)/sr:.2f}s)")


if __name__ == "__main__":
    main()
