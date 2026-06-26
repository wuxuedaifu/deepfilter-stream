"""Package version + model release metadata (single source of truth)."""

__version__ = "0.1.0"

# Fixed audio contract for the shipped DeepFilterNet3 torchDF model.
SAMPLE_RATE = 48000

# Model release. Assets are attached to a GitHub Release on this repo.
# Filled in at first release (Task 11). sha256 verified on download.
MODEL_VERSION = "dfn3-512-v1"
RELEASE_TAG = "model-dfn3-512-v1"
RELEASE_BASE_URL = (
    "https://github.com/wuxuedaifu/deepfilter-stream/releases/download/" + RELEASE_TAG
)
ASSETS = {
    "denoiser_model.onnx": "b758c49d6708a5b7979e3de185705a8a4915076c862fb17b1b304d9a72b75cdc",
    # initial_states / meta sha256 are stamped during Task 4 asset staging.
    "initial_states.npz": "REPLACE_WITH_SHA256",
    "meta.json": "REPLACE_WITH_SHA256",
}
