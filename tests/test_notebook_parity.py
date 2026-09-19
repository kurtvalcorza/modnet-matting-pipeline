"""NOTEBOOK_SPEC 2.0 parity tests (PAR1–PAR3) for the standalone tutorial notebook.

The notebook carries `src/<package>/pipeline.py` verbatim; these tests fail whenever the carried
cell, the inline manifest, or the inline pins diverge from the repository at HEAD.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
PKG_DIR = ROOT / TEMPLATE.get("package_dir", f"src/{TEMPLATE['package']}")
MODULE = PKG_DIR / TEMPLATE.get("entry_module", "pipeline.py")
MANIFEST = ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(notebook: dict, cell_type: str) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == cell_type]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def test_par1_embedded_modules_equal_repository_modules(notebook: dict) -> None:
    """One tagged cell per carried module, in dependency order, each equal to its module after rewrites."""
    tagged = [c for c in _cells(notebook, "code") if c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, TEMPLATE, recorded)
    assert [c["metadata"]["dimer"]["embedded_module"] for c in tagged] == ctx["module_rels"]
    for cell, module in zip(tagged, ctx["modules"], strict=True):
        rel = f"{ctx['pkg_rel']}/{module}"
        assert cell["metadata"]["dimer"]["module_sha256"] == ctx["per_module_sha256"][rel]
        drifted = f"embedded module cell for {rel} drifted from the package; regenerate the notebook"
        assert _source(cell).rstrip("\n") + "\n" == ctx["embedded"][module], drifted


REWRITES = TEMPLATE.get("rewrites", build.REWRITES)  # a template may declare its own rules (generator /2)


def test_par1_rewrite_rules_are_the_only_difference() -> None:
    """Every line the generator changed in a carried module is a documented rewrite: the template's
    `__file__` rules (each exactly once across modules), a removed package-relative import, or a
    disabled `__main__` guard. Compared with difflib because a multi-line import collapses to one
    marker line."""
    import difflib

    ctx = build.load_context(ROOT, TEMPLATE)
    rule_hits = 0
    for module, original in ctx["texts"].items():
        a, b = original.splitlines(), ctx["embedded"][module].splitlines()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            replaced = b[j1:j2]
            assert replaced and all("standalone rewrite" in line for line in replaced), (
                module,
                a[i1:i2],
                replaced,
            )
            rule_hits += sum("__file__" in line for line in a[i1:i2])
    assert rule_hits == len(REWRITES)


def test_par2_inline_manifest_and_pins_match_repository(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    assert inline, "model cell must carry MANIFEST = {...}"
    assert json.loads(inline.group(1)) == manifest
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    assert pins_block, "install cell must carry PINS = [...]"
    inline_pins = re.findall(r"'([^']+)'", pins_block.group(1))
    assert inline_pins == build._pins(ROOT)
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC
    pkg_dir = TEMPLATE.get("package_dir", f"src/{TEMPLATE['package']}")
    entry = TEMPLATE.get("entry_module", "pipeline.py")
    assert meta["generated_from"]["module"] == f"{pkg_dir}/{entry}"
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, TEMPLATE)["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    # The recorded revision is a provenance label carried through the check (see build_notebook.py
    # --check); content drift is what fails this comparison.
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "git" not in re.findall(r"subprocess\.run\(\[([^\]]*)\]", code).__str__()
    assert f"import {TEMPLATE['package']}" not in code
    assert f"from {TEMPLATE['package']}" not in code
    assert "github.com" not in code
