"""Offline tests for the snapshot manifest, staging, the static pickle audit (zip, plain and legacy torch layouts),
the record contract, the trimap, the metrics and the artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_portrait, synthetic_records
from modnet_matting_pipeline import (
    CKPT_ALLOWED_GLOBALS,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    TRAIN_SIZE,
    ModNetMattingPipeline,
    audit_pickle,
    constant_baselines,
    dataset_digest,
    matte_errors,
    matting_metrics,
    read_alpha,
    read_image,
    stage_missing_files,
    trimap_from_alpha,
    validate_dataset,
    validate_inputs,
    verify_converted,
    verify_snapshot,
)
from modnet_matting_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes], *, pin_source: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if pin_source and pl.SOURCE_CKPT_NAME not in files:
        entries.append({"path": pl.SOURCE_CKPT_NAME, "bytes": pl.SOURCE_CKPT_BYTES, "sha256": pl.SOURCE_CKPT_SHA256})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": pl.MODEL_KEY,
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and the committed manifest ------------------------------------------------------------------


def test_identity_is_immutable_and_the_manifest_agrees():
    assert len(MODEL_REVISION) == 40 and len(pl.UPSTREAM_CODE_COMMIT) == 40
    manifest = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, pl.MODEL_KEY)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    assert [e["path"] for e in manifest["files"]] == [pl.SOURCE_CKPT_NAME]
    source = manifest["files"][0]
    assert (source["bytes"], source["sha256"]) == (pl.SOURCE_CKPT_BYTES, pl.SOURCE_CKPT_SHA256)
    assert len(pl.CONVERTED_SHA256) == 64 and pl.CONVERTED_BYTES > 0 and len(pl.PICKLE_AUDIT_SHA256) == 64
    assert pl.PARAMETER_COUNT == 6_487_075 and pl.STATE_TENSORS == 439 and pl.CHECKPOINT_TENSORS == 751
    assert abs(sum(sum(row) for row in pl.SEMANTIC_BLUR_KERNEL) - 1.0) < 1e-9


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {pl.SOURCE_CKPT_NAME: b"ckpt"}, pin_source=False)
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).write_bytes(b"ckpt-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"}, pin_source=False)
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_converted_checks_the_pinned_digest(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / pl.CONVERTED_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="size"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"fetched")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [pl.SOURCE_CKPT_NAME]
    assert calls == [pl.SOURCE_CKPT_NAME]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    _write_snapshot(root, "someone/else", MODEL_REVISION, {})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=downloader)


# --- static pickle audit ----------------------------------------------------------------------------------


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def _torch_like_archive(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def _legacy_like(obj, *, keys=("0",), magic: int = pl.LEGACY_MAGIC_NUMBER) -> bytes:
    """The five header streams of torch's legacy format followed by fake storage bytes (which start with a byte
    that would be read as a pickle PROTO opcode if the audit overran the header)."""
    return (
        pickle.dumps(magic, protocol=2)
        + pickle.dumps(1001, protocol=2)
        + pickle.dumps(
            {"protocol_version": 1001, "little_endian": True, "type_sizes": {"short": 2, "int": 4, "long": 4}}, protocol=2
        )
        + pickle.dumps(obj, protocol=2)
        + pickle.dumps(list(keys), protocol=2)
        + b"\x80\x00\x00\x00\x00\x00\x00\x00"
        + b"\x00" * 64
    )


def test_audit_pickle_lists_globals_and_refuses_code(tmp_path, forbid_model_imports):
    benign = tmp_path / "benign.pt"
    benign.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": {"a": 1}, "epoch": 3})))
    report = audit_pickle(benign)
    assert report["layout"] == "torch_zip" and report["pickles"] == 1 and report["globals"] == [] and report["violations"] == []
    evil = tmp_path / "evil.pt"
    evil.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": _Evil()})))
    with pytest.raises(ValueError, match="pickle audit failed.*os.system|posix.system|nt.system"):
        audit_pickle(evil)
    plain = tmp_path / "plain.pickle"
    plain.write_bytes(pickle.dumps(complex(1, 2)))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(plain)
    assert "collections.OrderedDict" in CKPT_ALLOWED_GLOBALS and len(CKPT_ALLOWED_GLOBALS) == 4
    with pytest.raises(FileNotFoundError):
        audit_pickle(tmp_path / "missing.pt")


def test_audit_pickle_reads_every_header_stream_of_a_legacy_file(tmp_path, forbid_model_imports):
    """The legacy torch layout is five consecutive pickles: a global hidden in the fourth (the object) or the
    fifth (the storage keys) must be found, and the raw storage bytes after them must not be parsed."""
    from collections import OrderedDict

    benign = tmp_path / "legacy.ckpt"
    benign.write_bytes(_legacy_like(OrderedDict(a=1)))
    report = audit_pickle(benign)
    assert report["layout"] == "torch_legacy" and report["pickles"] == 5
    assert report["globals"] == ["collections.OrderedDict"] and report["violations"] == []
    assert report["header_bytes"] < benign.stat().st_size
    evil = tmp_path / "legacy-evil.ckpt"
    evil.write_bytes(_legacy_like(OrderedDict(a=_Evil())))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(evil)
    evil_keys = tmp_path / "legacy-evil-keys.ckpt"
    evil_keys.write_bytes(_legacy_like(OrderedDict(a=1), keys=(_Evil(),)))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(evil_keys)
    # a file that opens with another integer is a plain pickle of that integer, not a legacy archive
    other = tmp_path / "int.pickle"
    other.write_bytes(pickle.dumps(12345, protocol=2))
    assert audit_pickle(other)["layout"] == "pickle"


# --- record contract --------------------------------------------------------------------------------------


def test_validate_dataset_reports_fractions_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_labelled"] == 6 and report["sizes"] == [[TRAIN_SIZE, TRAIN_SIZE]]
    assert 0.05 < report["foreground_fraction"] < 0.5 and 0.0 < report["fractional_alpha_fraction"] < 0.1
    assert report["digest"] == dataset_digest(records) and len(report["digest"]) == 64
    first = validate_inputs(records[0])
    assert (first["id"], first["shape"], first["has_alpha"]) == ("disc-000", (512, 512, 3), True) and 0.05 < first[
        "foreground_fraction"
    ] < 0.5
    unlabelled = validate_dataset(synthetic_records(2, labels=False), min_records=1, require_labels=False)
    assert unlabelled["n_labelled"] == 0 and unlabelled["foreground_fraction"] is None
    assert "validation" in INPUT_SCHEMA and INPUT_SCHEMA["records"] == [4, 2000]


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(4)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'image'"):
        validate_dataset([{"id": "x"}, *records[1:]])
    with pytest.raises(ValueError, match="id must be"):
        validate_dataset([{**records[0], "id": ""}, *records[1:]])
    with pytest.raises(ValueError, match="uint8"):
        validate_dataset([{**records[0], "image": records[0]["image"].astype(np.float32)}, *records[1:]])
    with pytest.raises(ValueError, match="must have shape \\(H, W, 3\\)"):
        validate_dataset([{**records[0], "image": records[0]["image"][..., 0]}, *records[1:]])
    with pytest.raises(ValueError, match="512 × 512"):
        validate_dataset([{**records[0], "image": records[0]["image"][:256], "alpha": records[0]["alpha"][:256]}, *records[1:]])
    with pytest.raises(ValueError, match="has no alpha"):
        validate_dataset([{"id": "x", "image": records[0]["image"]}, *records[1:]])
    with pytest.raises(ValueError, match="alpha must have shape"):
        validate_dataset([{**records[0], "alpha": records[0]["alpha"][:100]}, *records[1:]])
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        validate_dataset([{**records[0], "alpha": records[0]["alpha"] * 2}, *records[1:]])
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset([{**records[0], "alpha": np.full_like(records[0]["alpha"], np.nan)}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], records[0], *records[1:3]])
    with pytest.raises(ValueError, match="all background"):
        validate_dataset([{**r, "alpha": np.zeros_like(r["alpha"])} for r in records])
    with pytest.raises(ValueError, match="sides must be in"):
        validate_inputs({"id": "x", "image": np.zeros((32, 512, 3), dtype=np.uint8)})
    with pytest.raises(ValueError, match="file not found"):
        validate_inputs({"id": "x", "image": "nowhere.png"})
    assert validate_inputs({"id": "x", "image": np.zeros((300, 700, 3), dtype=np.uint8)})["shape"] == (300, 700, 3)


def test_image_and_alpha_files_round_trip(tmp_path, forbid_model_imports):
    from PIL import Image

    image, alpha = synthetic_portrait(seed=3)
    Image.fromarray(image).save(tmp_path / "p.png")
    Image.fromarray((alpha * 255 + 0.5).astype(np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(image).convert("RGBA").save(tmp_path / "rgba.png")
    assert np.array_equal(read_image(tmp_path / "p.png"), image)
    assert np.abs(read_alpha(tmp_path / "a.png") - alpha).max() < 1 / 255 + 1e-6
    assert read_image(tmp_path / "rgba.png").shape == (512, 512, 3)
    record = validate_dataset(
        [{"id": "f", "image": str(tmp_path / "p.png"), "alpha": str(tmp_path / "a.png")}, *synthetic_records(3, seed=9)]
    )["records"][0]
    assert record["image"].dtype == np.uint8 and record["alpha"].dtype == np.float32


def test_trimap_marks_the_fractional_band(forbid_model_imports):
    _image, alpha = synthetic_portrait(seed=1)
    trimap = trimap_from_alpha(alpha, radius=4)
    assert set(np.unique(trimap).tolist()) == {0.0, 0.5, 1.0}
    unknown = trimap == 0.5
    fractional = (alpha > 0.02) & (alpha < 0.98)
    assert fractional[unknown].sum() == fractional.sum()  # every fractional pixel is unknown
    assert unknown.sum() > fractional.sum()  # and the band was grown
    assert np.all(trimap[alpha >= 0.999] != 0.0) and np.all(trimap[alpha <= 0.001] != 1.0)
    assert np.array_equal(trimap_from_alpha(alpha, radius=0) == 0.5, fractional)


# --- metrics ---------------------------------------------------------------------------------------------


def test_matte_metrics_and_constant_baselines(forbid_model_imports):
    _image, alpha = synthetic_portrait(seed=5)
    exact = matte_errors(alpha, alpha)
    assert exact == {"mad": 0.0, "mse": 0.0, "sad": 0.0, "mad_unknown": None}
    off = matte_errors(np.clip(alpha + 0.1, 0, 1), alpha, unknown=alpha > 0.5)
    assert 0.0 < off["mad"] <= 0.1 and off["mse"] <= 0.01 and off["sad"] > 0 and off["mad_unknown"] is not None
    with pytest.raises(ValueError, match="shape"):
        matte_errors(alpha[:100], alpha)
    with pytest.raises(ValueError, match="unknown mask"):
        matte_errors(alpha, alpha, unknown=np.ones((3, 3), dtype=bool))
    summary = matting_metrics([alpha, alpha], [alpha, np.zeros_like(alpha)])
    assert summary["n_images"] == 2 and summary["mad"] == round(float(alpha.mean()) / 2, 6) and len(summary["per_image"]) == 2
    baselines = constant_baselines([alpha])
    assert abs(baselines["all_background"]["mad"] - float(alpha.mean())) < 1e-5
    assert abs(baselines["all_foreground"]["mad"] - float(1 - alpha.mean())) < 1e-5
    with pytest.raises(ValueError, match="equal length"):
        matting_metrics([alpha], [])


# --- artifact-manifest static checks ----------------------------------------------------------------------


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "converted_sha256": pl.CONVERTED_SHA256},
        "adapter": {"trainable": "branches"},
        "tensors": ["f_branch.conv_f.0.layers.0.weight"],
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
    }
    path, mode = ModNetMattingPipeline.check_artifact_manifest(tmp_path, good)
    assert path == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve() and mode == "branches"
    bad_cases = [
        ({"format": "other"}, "artifact format"),
        ({"format_version": "9.9"}, "format_version"),
        ({"base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        ({"base_model": {**good["base_model"], "converted_sha256": "1" * 64}}, "converted-base digest"),
        ({"files": []}, "exactly one weights file"),
        ({"files": [{"path": "other.safetensors", "bytes": 1, "sha256": "0" * 64}]}, "must be named"),
        ({"files": [{"path": "../adapter.safetensors", "bytes": 1, "sha256": "0" * 64}]}, "must be named"),
        ({"adapter": {"trainable": "everything"}}, "must be one of"),
        ({"tensors": "not-a-list"}, "list its tensors"),
    ]
    for patch, message in bad_cases:
        with pytest.raises(ValueError, match=message):
            ModNetMattingPipeline.check_artifact_manifest(tmp_path, {**good, **patch})
