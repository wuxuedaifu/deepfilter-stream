"""Regenerate model assets (initial_states.npz, meta.json) and a reference clip.

Pure onnxruntime + numpy; no torch/df/libdf needed. Run from repo root:
    python scripts/export_assets.py --onnx tests/data/denoiser_model.onnx \
        --wav tests/data/noisy_2s_48k.wav --out tests/data --ref-name reference_2s_48k.wav
"""
import argparse, json, hashlib, pathlib
import numpy as np
import onnxruntime as ort
import soundfile as sf

LINSPACE_ERB = (-60.0, -90.0)
LINSPACE_DF = (0.001, 0.0001)


def init_states(inputs, nb_bands, nb_df):
    st = {}
    for name, meta in inputs.items():
        if name == "input_frame":
            continue
        shape = [int(x) for x in meta.shape]
        if name == "erb_norm_state":
            st[name] = np.linspace(*LINSPACE_ERB, nb_bands, dtype=np.float32).reshape(shape)
        elif name == "band_unit_norm_state":
            st[name] = np.linspace(*LINSPACE_DF, nb_df, dtype=np.float32).reshape(shape)
        else:
            st[name] = np.zeros(shape, dtype=np.float32)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-name", default="reference_48k.wav")
    a = ap.parse_args()
    out = pathlib.Path(a.out)

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    out_names = [o.name for o in sess.get_outputs()]
    fsz = int(inputs["input_frame"].shape[0]); fft = 2 * fsz
    nb_bands = int(np.prod(inputs["erb_norm_state"].shape))
    nb_df = int(np.prod(inputs["band_unit_norm_state"].shape))

    states = init_states(inputs, nb_bands, nb_df)
    np.savez(out / "initial_states.npz", **states)

    noisy, sr = sf.read(a.wav, dtype="float32")
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    orig = noisy.shape[0]; pad_div = (fsz - orig % fsz) % fsz; orig_p = orig + pad_div
    x = np.pad(noisy, (0, fft + pad_div)).astype(np.float32)
    st = init_states(inputs, nb_bands, nb_df)
    buf = np.empty((x.shape[0] // fsz) * fsz, dtype=np.float32)
    for k in range(x.shape[0] // fsz):
        feeds = {"input_frame": x[k * fsz:(k + 1) * fsz], **st}
        res = sess.run(out_names, feeds)
        buf[k * fsz:(k + 1) * fsz] = res[0]
        for nm, val in zip(list(inputs)[1:], res[1:]):
            st[nm] = val
    enh = buf[fft - fsz: orig_p + (fft - fsz)]
    sf.write(out / a.ref_name, enh, sr, subtype="PCM_16")

    def h(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    (out / "meta.json").write_text(json.dumps({
        "model": "DeepFilterNet3_torchDF", "sample_rate": sr,
        "frame_size": fsz, "fft_size": fft, "nb_bands": nb_bands, "nb_df": nb_df,
        "input_names": list(inputs), "output_names": out_names,
        "sha256": {"onnx": h(a.onnx), "initial_states": h(out / "initial_states.npz")},
    }, indent=2))
    print("wrote", a.ref_name)


if __name__ == "__main__":
    main()
