"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import modnet_matting_pipeline as pkg
from conftest import synthetic_records
from modnet_matting_pipeline import INPUT_SCHEMA, MAX_SIDE, MIN_SIDE, REF_SIZE, TRAIN_SIZE, validate_inputs


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["records"] == [4, 2000] and TRAIN_SIZE == REF_SIZE == 512 and (MIN_SIDE, MAX_SIDE) == (64, 4096)
    assert "any RGB image is matted without complaint" in INPUT_SCHEMA["validation"]
    assert str(TRAIN_SIZE) in INPUT_SCHEMA["training_size"] and str(REF_SIZE) in INPUT_SCHEMA["inference_size"]


def test_validate_inputs_reports_the_record():
    report = validate_inputs(synthetic_records(1)[0])
    assert report["id"] == "disc-000" and report["shape"] == (512, 512, 3) and report["has_alpha"]


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "ModNetMattingPipeline" in pkg.__all__ and "audit_pickle" in pkg.__all__ and "render_portrait" in pkg.__all__
