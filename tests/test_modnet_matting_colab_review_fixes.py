"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on modnet_matting_colab (MOD-M1..M3,
MOD-m1..m3). They need only CI's dependencies (numpy, Pillow): notebook cells are executed with stand-in pipelines,
and the one torch-only check skips where torch is absent."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modnet_matting_pipeline.pipeline import MIN_RECORDS, validate_dataset
from modnet_matting_pipeline.samples import SAMPLE_SEEDS, minimum_pairs, render_portrait, split_dataset, split_sizes

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "modnet_matting_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [s for s in _code(nb) if marker in s]
    assert len(found) == 1, marker
    return found[0]


def _functions(source: str, names: tuple[str, ...], namespace: dict) -> None:
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<helpers>", "exec"), namespace)


# --- MOD-M3: BYOD minimum, refusal probes, small test split ----------------------------------------------------


def _old_split(n: int, fractions=(0.7, 0.15)) -> tuple[int, int, int]:
    n_train = max(1, int(round(n * fractions[0])))
    n_val = max(1, int(round(n * fractions[1])))
    if n_train + n_val >= n:
        n_train = n - n_val - 1
    return n_train, n_val, n - n_train - n_val


def test_mod_m3_split_sizes_unchanged_and_minimum_is_six() -> None:
    for n in range(3, 60):
        assert split_sizes(n) == _old_split(n), n
    assert minimum_pairs(MIN_RECORDS) == 6
    assert split_sizes(5)[0] < MIN_RECORDS <= split_sizes(6)[0]


def test_mod_m3_too_few_pairs_refused_at_split_naming_the_minimum() -> None:
    records = [{"id": f"r{i}"} for i in range(5)]
    with pytest.raises(ValueError, match=r"5 labelled pairs give a training split of 3; .*at least 4 training records, so supply at least 6 pairs"):
        split_dataset(records, seed=0, min_train=MIN_RECORDS)
    splits = split_dataset([{"id": f"r{i}"} for i in range(6)], seed=0, min_train=MIN_RECORDS)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [4, 1, 1]
    # the shuffle itself is unchanged (no min_train given keeps the old behaviour)
    assert split_dataset(records, seed=0)["train"] == [dict(records[i]) for i in np.random.default_rng(0).permutation(5).tolist()[:3]]


def test_mod_m3_byod_cell_uses_the_minimum_before_any_model_runs(nb: dict) -> None:
    section4 = _cell_with(nb, "splits = sample_dataset()")
    assert "split_dataset(load_byod_dataset(byod_path), seed=0, min_train=MIN_RECORDS)" in section4
    assert "pipe." not in section4, "Section 4 must not run the model"
    md = _markdown(nb)
    assert "at least four pairs" not in md and "at least six pairs" in md


def test_mod_m3_probes_refused_for_their_own_reason_under_byod(nb: dict) -> None:
    section4 = _cell_with(nb, "splits = sample_dataset()")
    tail = section4[section4.index("probe_records = ") :]
    tiny_test = [render_portrait(SAMPLE_SEEDS["test"] + 50)]  # a 6-pair BYOD split has one test portrait
    ns = {"USE_BYOD": True, "test_records": tiny_test, "render_portrait": render_portrait, "SAMPLE_SEEDS": SAMPLE_SEEDS, "np": np, "validate_dataset": validate_dataset}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(tail, "<section 4 probes>", "exec"), ns)
    lines = out.getvalue().splitlines()
    assert len(lines) == 3 and all("'rejected'" in line for line in lines), lines
    assert not any("records;" in line for line in lines), "a probe was refused for the record count"
    assert "[0, 1]" in lines[0] or "alpha" in lines[0]
    assert "512" in lines[1]
    assert "all background" in lines[2]


def test_mod_m3_small_test_split_warns(nb: dict) -> None:
    section4 = _cell_with(nb, "splits = sample_dataset()")
    start = section4.index("if len(test_records) < 5:")
    snippet = section4[start : section4.index("\n\n", start)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(snippet, {"test_records": [1]})
        exec(snippet, {"test_records": list(range(5))})
    assert out.getvalue().count("warning") == 1 and "1 portrait(s)" in out.getvalue()


# --- MOD-M2: Section 5 and Section 6 always start from the frozen model ----------------------------------------


class _Tensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype="float32")

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.values

    def __deepcopy__(self, memo):
        return _Tensor(self.values.copy())


class _Model:
    def __init__(self) -> None:
        self.weights = {"w": _Tensor([0.08])}

    def state_dict(self):
        return dict(self.weights)

    def load_state_dict(self, state, strict=True):
        assert strict and set(state) == set(self.weights)
        self.weights = {k: _Tensor(v.values.copy()) for k, v in state.items()}

    def eval(self):
        return self


class _Pipe:
    """Stand-in whose test MAD is its weight; `adapt` trains in place, as the real pipeline does."""

    def __init__(self) -> None:
        self.model = _Model()
        self.adapter = None

    def _mad(self) -> float:
        return float(self.model.weights["w"].values[0])

    def evaluate(self, records):
        stats = {"mad": self._mad(), "mse": 0.0, "sad": 0.0, "mad_unknown": 0.0}
        base = {"mad": 0.34, "mse": 0.34, "sad": 1.0, "mad_unknown": 0.5}
        return {"metric": "mad", "baselines": {"all_background": base, "all_foreground": base}, "model": stats, "per_image": [{"id": r["id"], **stats} for r in records], "adapted": self.adapter is not None}

    def predict(self, records):
        preds = [{"alpha": np.full(r["alpha"].shape, 0.5, dtype="float32"), "model_size": [32, 32], "foreground_fraction": 0.5} for r in records]
        return {"predictions": preds, "output": "alpha", "seconds": 0.0}

    def adapt(self, *args, **kwargs):
        self.model.weights["w"] = _Tensor([0.004])
        self.adapter = {"trained": True}
        return {"n_trainable": 1, "n_total": 2, "n_steps": 1, "best_epoch": 1, "losses": {}, "batchnorm": "frozen", "history": []}


def _records(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    return [{"id": f"p{i}", "image": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8), "alpha": rng.random((32, 32), dtype=np.float32)} for i in range(n)]


def test_mod_m2_rerun_from_section4_scores_the_frozen_model(nb: dict, tmp_path, monkeypatch) -> None:
    import time

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    section5 = _cell_with(nb, "frozen_test = pipe.evaluate(test_records)")
    section6 = _cell_with(nb, "adapt_result = pipe.adapt(")
    shown: list = []
    pipe = _Pipe()
    ns = {"pipe": pipe, "np": np, "test_records": _records(3), "val_records": _records(2), "train_records": _records(4), "display": shown.append, "time": time}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(section5, "<section 5>", "exec"), ns)
        first = ns["frozen_test"]["model"]["mad"]
        exec(compile(section6, "<section 6>", "exec"), ns)
        assert pipe._mad() == pytest.approx(0.004) and pipe.adapter is not None
        exec(compile(section5, "<section 5 rerun>", "exec"), ns)  # the BYOD instruction: re-run from Section 4
    assert first == pytest.approx(0.08)
    assert ns["frozen_test"]["model"]["mad"] == pytest.approx(first), "the rerun scored the adapted model as frozen"
    assert ns["frozen_test"]["adapted"] is False
    assert len(shown) == 4 and all(isinstance(img, Image.Image) for img in shown), "Section 5 shows one strip per shown portrait"


def test_mod_m2_restore_refuses_an_already_adapted_pipeline(nb: dict) -> None:
    section5 = _cell_with(nb, "frozen_test = pipe.evaluate(test_records)")
    pipe = _Pipe()
    pipe.adapter = {"trained": True}
    ns = {"pipe": pipe, "copy": copy, "hashlib": hashlib}
    _functions(section5, ("state_digest", "restore_frozen_weights"), ns)
    with pytest.raises(RuntimeError, match="adapted before the frozen snapshot"):
        ns["restore_frozen_weights"]()
    assert section5.index("restore_frozen_weights()\n") < section5.index("frozen_test = pipe.evaluate(test_records)")


# --- MOD-m2: mattes rendered inline ----------------------------------------------------------------------------


def test_mod_m2_minor_mattes_are_displayed_in_sections_5_and_8(nb: dict) -> None:
    section5 = _cell_with(nb, "frozen_test = pipe.evaluate(test_records)")
    section8 = _cell_with(nb, "adapted_shown = pipe.predict(shown_records)")
    assert "display(matte_strip(record, [record['alpha'], pred['alpha']]))" in section5
    assert "display(matte_strip(record, [record['alpha'], before['alpha'], after['alpha']]))" in section8
    assert "display(matte_strip(photo, [frozen_photo['alpha'], adapted_photo['alpha']]))" in section8
    ns = {"np": np, "Image": Image}
    _functions(section5, ("matte_strip",), ns)
    record = _records(1)[0]
    strip = ns["matte_strip"](record, [record["alpha"], record["alpha"]], tile=64)
    assert strip.size == (4 * 64 + 3 * 6, 64) and strip.mode == "RGB"


# --- MOD-m3 / MOD-M1: declarations, documentation, warning, record wording -------------------------------------


def test_mod_m3_declarations_and_documentation(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    md = _markdown(nb)
    assert "{{id, image, alpha}}" not in md and "{id, image, alpha}" in md
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" in md


def test_mod_m3_to_tensor_does_not_warn_on_read_only_images() -> None:
    pytest.importorskip("torch")
    from modnet_matting_pipeline.pipeline import ModNetMattingPipeline

    pipe = ModNetMattingPipeline.__new__(ModNetMattingPipeline)
    pipe.device = "cpu"
    images = np.asarray(Image.new("RGB", (32, 32), (10, 20, 30)), dtype=np.uint8)[None]
    assert not images.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batch = pipe._to_tensor(images)
    assert tuple(batch.shape) == (1, 3, 32, 32)


def test_mod_m1_record_does_not_count_a_restart_dependent_run_as_a_pass() -> None:
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    assert "restart after the install is expected" not in verification
    assert "**PASSED** — 11/11 code cells ok (1 restart after install cell)" not in verification
    assert "Restart-dependent — not a one-pass Run all" in verification
