"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot is verified and the checkpoint is statically
audited before torch is imported.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import MODEL_LIBRARIES, synthetic_records
from modnet_matting_pipeline import validate_dataset, validate_inputs

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "modnet_matting_pipeline"


def test_package_import_does_not_import_model_libraries(forbid_model_imports):
    import importlib

    import modnet_matting_pipeline

    importlib.reload(modnet_matting_pipeline)


def test_no_module_level_model_imports():
    """Every torch / safetensors / huggingface_hub import is inside a function body (modeling.py excepted: it is
    the vendored architecture and is only ever imported lazily by build_model)."""
    for path in PACKAGE.glob("*.py"):
        if path.name == "modeling.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_records_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="must have shape"):
        validate_inputs({"id": "x", "image": synthetic_records(1)[0]["image"][..., :2]})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(synthetic_records(2))
