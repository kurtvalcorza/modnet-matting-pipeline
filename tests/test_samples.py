"""Offline tests for the synthetic portrait renderer, the sample splits, the photograph loader, the split helpers
and the BYOD zip loader. No model library is imported."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_records
from modnet_matting_pipeline import (
    SAMPLE_COUNTS,
    SAMPLE_SEEDS,
    SAMPLE_SIZE,
    check_split_disjoint,
    dataset_manifest,
    load_byod_dataset,
    load_photo,
    render_portrait,
    sample_dataset,
    split_dataset,
    write_dataset_csv,
    write_sample_pair,
)
from modnet_matting_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]


def test_render_portrait_is_deterministic_with_fractional_alpha(forbid_model_imports):
    a, b = render_portrait(7), render_portrait(7)
    assert a["id"] == "portrait-00007" and a["image"].shape == (SAMPLE_SIZE, SAMPLE_SIZE, 3) and a["image"].dtype == np.uint8
    assert a["alpha"].shape == (SAMPLE_SIZE, SAMPLE_SIZE) and a["alpha"].dtype == np.float32
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["alpha"], b["alpha"])
    assert not np.array_equal(a["image"], render_portrait(8)["image"])
    assert 0.15 < float(a["alpha"].mean()) < 0.6
    fractional = ((a["alpha"] > 0.02) & (a["alpha"] < 0.98)).mean()
    assert 0.005 < fractional < 0.08  # hair strands and silhouette edges, not a binary mask
    assert a["alpha"][-1].min() == 1.0 or a["alpha"][-1].max() == 1.0  # the torso reaches the bottom edge
    assert a["meta"]["seed"] == 7 and a["meta"]["strands"] >= 40
    small = render_portrait(3, size=128)
    assert small["image"].shape == (128, 128, 3)


def test_sample_dataset_splits_and_manifest(forbid_model_imports):
    splits = sample_dataset(counts={"train": 4, "validation": 2, "test": 3})
    assert [len(v) for v in splits.values()] == [4, 2, 3]
    assert splits["train"][0]["id"] == f"portrait-{SAMPLE_SEEDS['train']:05d}"
    assert splits["test"][0]["id"] == f"portrait-{SAMPLE_SEEDS['test']:05d}"
    report = dataset_manifest(splits)
    assert report["disjoint"] is True and len(report["digest"]) == 64
    assert report["splits"]["train"]["n_records"] == 4 and report["splits"]["train"]["foreground_fraction"] > 0
    assert SAMPLE_COUNTS == {"train": 48, "validation": 12, "test": 20}
    with pytest.raises(ValueError, match="unknown split"):
        sample_dataset(counts={"holdout": 2})
    with pytest.raises(ValueError, match="1..500"):
        sample_dataset(counts={"train": 0})


def test_split_helpers(forbid_model_imports):
    records = synthetic_records(10)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 10 and all(splits.values())
    assert check_split_disjoint(splits) == {"disjoint": True, "n_ids": 10}
    with pytest.raises(ValueError, match="is in both"):
        check_split_disjoint({"train": records[:2], "test": records[1:3]})
    twin = {**records[5], "id": "twin"}
    with pytest.raises(ValueError, match="duplicates an image"):
        check_split_disjoint({"train": records[:6], "test": [twin]})
    with pytest.raises(ValueError, match="at least three"):
        split_dataset(records[:2])


def test_load_photo_downscales_and_records_sizes(tmp_path, forbid_model_imports):
    from PIL import Image

    Image.new("RGB", (4000, 3000), (200, 150, 120)).save(tmp_path / "big.jpg", quality=80)
    record = load_photo(tmp_path / "big.jpg", record_id="own")
    assert record["id"] == "own" and record["image"].shape == (1152, 1536, 3) and record["image"].dtype == np.uint8
    assert record["original_size"] == [4000, 3000] and record["loaded_size"] == [1536, 1152] and record["source"] == "big.jpg"
    Image.new("L", (300, 200), 90).save(tmp_path / "grey.png")
    small = load_photo(tmp_path / "grey.png")
    assert small["image"].shape == (200, 300, 3) and small["loaded_size"] == [300, 200]
    with pytest.raises(FileNotFoundError):
        load_photo(tmp_path / "nowhere.jpg")


def test_byod_zip_loader_and_writers(tmp_path, forbid_model_imports):
    records = synthetic_records(3, seed=20)
    pair = write_sample_pair(records[0], tmp_path / "out" / "p.png", tmp_path / "out" / "a.png")
    assert Path(pair["image"]).is_file() and Path(pair["alpha"]).is_file()
    csv_path = write_dataset_csv(records, tmp_path / "out" / "pairs.csv")
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "id,image,alpha"
    zip_path = tmp_path / "byod.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "data/pairs.csv", "id,image,alpha\n" + "".join(f"{r['id']},{r['id']}.png,{r['id']}_alpha.png\n" for r in records)
        )
        for r in records:
            write_sample_pair(r, tmp_path / f"{r['id']}.png", tmp_path / f"{r['id']}_alpha.png")
            archive.write(tmp_path / f"{r['id']}.png", f"data/{r['id']}.png")
            archive.write(tmp_path / f"{r['id']}_alpha.png", f"data/{r['id']}_alpha.png")
        archive.writestr("data/decoy.txt", "ignored")
    loaded = load_byod_dataset(zip_path)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert loaded[0]["image"].shape == (512, 512, 3) and loaded[0]["alpha"].shape == (512, 512)
    assert np.abs(loaded[1]["alpha"] - records[1]["alpha"]).max() < 2 / 255
    assert loaded[0]["original_size"] == [512, 512]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("pairs.csv", "id,image,alpha\nx,missing.png,missing_alpha.png\n")
    with pytest.raises(ValueError, match="not in the archive"):
        load_byod_dataset(bad)
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("readme.txt", "no csv")
    with pytest.raises(ValueError, match="pairs.csv"):
        load_byod_dataset(bad)
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "nowhere.zip")
    assert json.dumps(sm.SAMPLE_LABEL_SOURCE)
