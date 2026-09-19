import builtins

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "safetensors", "huggingface_hub", "torchvision", "scipy"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_portrait(*, seed: int = 0, size: int = 512):
    """A cheap stand-in for the renderer: a coloured disc with a soft edge on a gradient, and its exact alpha."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    cx, cy, r = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65), rng.uniform(0.15, 0.3)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip((r + 0.02 - dist) / 0.04, 0.0, 1.0).astype(np.float32)
    background = np.stack([120 + 100 * x, 80 + 120 * y, 160 - 60 * x], axis=-1)
    foreground = np.asarray(rng.integers(0, 255, 3), dtype=np.float32)[None, None, :]
    image = alpha[..., None] * foreground + (1 - alpha[..., None]) * background
    return np.clip(image, 0, 255).astype(np.uint8), alpha


def synthetic_records(n: int = 6, *, seed: int = 0, labels: bool = True, size: int = 512):
    out = []
    for i in range(n):
        image, alpha = synthetic_portrait(seed=seed + i, size=size)
        record = {"id": f"disc-{i:03d}", "image": image}
        if labels:
            record["alpha"] = alpha
        out.append(record)
    return out
