"""Resolve model assets: env override -> cache -> download from GitHub Release."""
from __future__ import annotations

import hashlib
import os
import pathlib
import urllib.request

from platformdirs import user_cache_dir

from . import _meta

_FILES = {
    "onnx": "denoiser_model.onnx",
    "initial_states": "initial_states.npz",
    "meta": "meta.json",
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_root(cache_root: str | None) -> pathlib.Path:
    root = pathlib.Path(cache_root or user_cache_dir("deepfilter-stream"))
    return root / _meta.MODEL_VERSION


def _complete(d: pathlib.Path) -> bool:
    return all((d / fn).exists() for fn in _FILES.values())


def ensure_assets(cache_root: str | None = None) -> dict[str, pathlib.Path]:
    # 1. explicit override directory
    env = os.environ.get("DEEPFILTER_STREAM_MODEL_DIR")
    if env:
        d = pathlib.Path(env)
        if _complete(d):
            return {k: d / fn for k, fn in _FILES.items()}
        raise FileNotFoundError(
            f"DEEPFILTER_STREAM_MODEL_DIR={env} is missing one of {list(_FILES.values())}"
        )

    # 2. cache (verify sha256)
    d = _cache_root(cache_root)
    if _complete(d) and all(
        sha256_file(d / fn) == _meta.ASSETS[fn] for fn in _FILES.values()
    ):
        return {k: d / fn for k, fn in _FILES.items()}

    # 3. download
    d.mkdir(parents=True, exist_ok=True)
    for fn, want in _meta.ASSETS.items():
        dst = d / fn
        if dst.exists() and sha256_file(dst) == want:
            continue
        url = f"{_meta.RELEASE_BASE_URL}/{fn}"
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download {fn} from {url}: {e}\n"
                f"Download manually and set DEEPFILTER_STREAM_MODEL_DIR to its folder."
            ) from e
        got = sha256_file(dst)
        if got != want:
            raise RuntimeError(f"sha256 mismatch for {fn}: got {got}, want {want}")
    return {k: d / fn for k, fn in _FILES.items()}
