"""Model-backed checks on the real converted weights (skipped when the snapshot is not staged): strict load and
parameter count, a matte of a rendered portrait, one bounded adaptation epoch and exact reload parity — on CUDA
when it is visible, else on the CPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from modnet_matting_pipeline import (  # noqa: E402
    CONVERTED_WEIGHTS_NAME,
    DEFAULT_WEIGHTS_DIR,
    PARAMETER_COUNT,
    ModNetMattingPipeline,
    render_portrait,
    verify_converted,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_NAME).is_file(), reason="converted weights not staged locally"
)


def test_load_matte_adapt_and_reload_on_the_available_device(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verify_converted(DEFAULT_WEIGHTS_DIR)
    pipe = ModNetMattingPipeline.from_pretrained(device=device, require_source=False)
    assert pipe.device == device and sum(p.numel() for p in pipe.model.parameters()) == PARAMETER_COUNT
    records = [render_portrait(s) for s in range(4)]
    held_out = [render_portrait(9_000)]
    frozen = pipe.evaluate(held_out)
    assert 0.0 < frozen["model"]["mad"] < frozen["baselines"]["all_background"]["mad"]
    matte = pipe.predict([{"id": "wide", "image": np.asarray(records[0]["image"][:256], dtype=np.uint8)}])["predictions"][0]
    assert matte["alpha"].shape == (256, 512) and matte["model_size"] == [256, 512]
    result = pipe.adapt(records, held_out, epochs=1, lr=1e-4, batch_size=2)
    assert result["n_trainable"] == 4_263_203 and result["n_total"] == PARAMETER_COUNT and result["n_steps"] == 2
    adapted = pipe.evaluate(held_out)["model"]
    pipe.save_artifact(tmp_path / "adapter")
    reloaded = ModNetMattingPipeline.from_artifact(tmp_path / "adapter", device=device, require_source=False)
    assert reloaded.evaluate(held_out)["model"] == adapted
    before = pipe.predict(held_out)["predictions"][0]["alpha"]
    after = reloaded.predict(held_out)["predictions"][0]["alpha"]
    assert float(np.abs(before - after).max()) < 1e-4
