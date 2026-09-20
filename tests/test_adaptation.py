"""Adaptation, evaluation and artifact tests on a stub model (torch required, no weights): the scope of the
trainable tensors, epoch selection, the transactional guarantee (including the aliased backbone keys) and the
artifact round trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from modnet_matting_pipeline import ADAPTATION_MODES, ModNetMattingPipeline  # noqa: E402
from modnet_matting_pipeline import pipeline as pl  # noqa: E402


class _StubModel(torch.nn.Module):
    """Parameter names follow the MODNet layout, including the shared backbone registered twice; every output
    depends on the branch tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.scale = torch.nn.Parameter(torch.ones(1))
        self.lr_branch = torch.nn.Module()
        self.lr_branch.backbone = self.backbone  # the alias upstream creates
        self.lr_branch.bias = torch.nn.Parameter(torch.zeros(1))
        self.hr_branch = torch.nn.Module()
        self.hr_branch.weight = torch.nn.Parameter(torch.tensor([0.2, 0.2, 0.2]))
        self.f_branch = torch.nn.Module()
        self.f_branch.bias = torch.nn.Parameter(torch.zeros(1))
        self.norm = torch.nn.BatchNorm2d(1, affine=False)

    def forward(self, img, inference=True):
        feature = (img * self.hr_branch.weight[None, :, None, None]).sum(dim=1, keepdim=True) * self.backbone.scale
        matte = torch.sigmoid(feature * 4 + self.f_branch.bias)
        if inference:
            return None, None, matte
        semantic = torch.nn.functional.avg_pool2d(matte, 16) + self.lr_branch.bias
        detail = torch.sigmoid(feature * 4 + self.f_branch.bias + self.lr_branch.bias)
        return semantic, detail, matte

    def freeze_norm(self) -> None:
        self.norm.eval()


def _pipeline() -> ModNetMattingPipeline:
    return ModNetMattingPipeline(model=_StubModel(), device="cpu", weights_dir=Path("unused"), source="stub")


def test_trainable_scopes():
    pipe = _pipeline()
    assert pipe._trainable("branches") == ["f_branch.bias", "hr_branch.weight", "lr_branch.bias"]
    assert pipe._trainable("full") == ["backbone.scale", "f_branch.bias", "hr_branch.weight", "lr_branch.bias"]
    assert ADAPTATION_MODES == ("branches", "full")
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe._trainable("everything")
    assert set(pipe.model.state_dict()) >= {"backbone.scale", "lr_branch.backbone.scale"}


def test_predict_and_evaluate_shapes():
    pipe = _pipeline()
    records = synthetic_records(3)
    result = pipe.predict(records)
    assert len(result["predictions"]) == 3 and result["predictions"][0]["alpha"].shape == (512, 512)
    assert result["predictions"][0]["alpha"].dtype == np.float32 and result["predictions"][0]["model_size"] == [512, 512]
    wide = pipe.predict([{"id": "wide", "image": np.zeros((300, 900, 3), dtype=np.uint8)}])["predictions"][0]
    assert wide["alpha"].shape == (300, 900) and wide["model_size"] == [
        288,
        896,
    ]  # straddles 512: kept, floored to multiples of 32
    tall = pipe.predict([{"id": "tall", "image": np.zeros((1000, 700, 3), dtype=np.uint8)}])["predictions"][0]
    assert tall["alpha"].shape == (1000, 700) and tall["model_size"] == [704, 512]
    report = pipe.evaluate(records)
    assert set(report["model"]) == {"n_images", "mad", "mse", "sad", "mad_unknown"} and len(report["per_image"]) == 3
    assert set(report["baselines"]) == {"all_background", "all_foreground"} and report["adapted"] is False


def test_adapt_trains_only_the_scope_and_keeps_the_best_epoch():
    pipe = _pipeline()
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    backbone_before = pipe.model.backbone.scale.clone()
    before = pipe.evaluate(val)["model"]["mad"]
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, batch_size=2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"] == "frozen model" and seen[0]["val"]["mad"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == 5 and result["n_steps"] == 9 and result["precision"] == "float32"
    assert set(result["history"][1]["train_loss_parts"]) == {"semantic", "detail", "matte"}
    assert torch.equal(pipe.model.backbone.scale, backbone_before)
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.model.parameters())
    assert pipe.evaluate(val)["model"] == result["history"][result["best_epoch"]]["val"]
    assert result["history"][-1]["train_loss"] < result["history"][1]["train_loss"] or result["best_epoch"] >= 1


def test_adapt_is_transactional_when_the_progress_callback_raises():
    pipe = _pipeline()
    initial = {k: v.clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, lr=1e-2, trainable="full", progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.model.state_dict().items())
    assert torch.equal(pipe.model.backbone.scale, torch.ones(1))  # the aliased backbone was restored, not the trained copy
    assert all(not p.requires_grad for p in pipe.model.parameters())


def test_adapt_refusals():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe.adapt(synthetic_records(4), trainable="all")
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path):
    pipe = _pipeline()
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2, trainable="full")
    adapted = pipe.evaluate(records)["model"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tensors"] == ["backbone.scale", "f_branch.bias", "hr_branch.weight", "lr_branch.bias"]
    assert manifest["adapter"]["trainable"] == "full" and manifest["metadata"] == {"tutorial": "test"}
    fresh = _pipeline()
    assert fresh.evaluate(records)["model"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records)["model"] == adapted and fresh.adapter["best_epoch"] == 1
    assert (
        torch.equal(fresh.model.backbone.scale, pipe.model.backbone.scale)
        and fresh.model.lr_branch.backbone is fresh.model.backbone
    )
    # a scope narrower than the tensor list is refused before deserialising
    narrowed = dict(manifest, adapter={**manifest["adapter"], "trainable": "branches"})
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(narrowed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline().load_artifact(out)
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline().load_artifact(out)
    assert np.isfinite(adapted["mad"])
