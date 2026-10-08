"""The isolated worker's google.colab stubs must carry a module spec, in every generated notebook.

On Colab, accelerate/transformers call importlib.util.find_spec("google.colab"), which
raises `ValueError: google.colab.__spec__ is None` on a spec-less stub (language-model-pipeline
T4 run of f2053aa; reference fixes language-model-pipeline d185817,
chronos-2-forecasting-pipeline 33a2f53, mobilenetv4-classification-pipeline a8a888d
moment-pipeline 3b0cee5 and maskrcnn-instance-segmentation-pipeline a72753a).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("tutorials/modnet_matting_colab.ipynb",)
_MARKER = '_WORKER_SOURCE = r"""'


def _colab_shim(notebook: str) -> str:
    nb = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    router = next(s for s in sources if _MARKER in s)
    worker = router[router.index(_MARKER) + len(_MARKER) :]
    worker = worker[: worker.index('"""')]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    return worker[start : worker.index('_main = types.ModuleType("__main__")', start)]


@pytest.mark.parametrize("real_google", [False, True])
@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_worker_colab_stubs_have_specs(
    notebook: str, real_google: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = _colab_shim(notebook)
    names = ("google", "google.colab", "google.colab.files")
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for n in names:
            sys.modules.pop(n, None)
        # Both branches: no importable `google` (stub created) and an existing namespace package.
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        namespace = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
        exec(compile(shim, f"{notebook}-colab-shim", "exec"), namespace)
        for n in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(n)  # raised ValueError before the fix
            assert spec is not None and spec.name == n
        assert sys.modules["google.colab"].__path__ == []
        assert callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
