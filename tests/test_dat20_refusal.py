"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

from modnet_matting_pipeline import pipeline as pipeline_module
from modnet_matting_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/modnet_matting_colab.ipynb"
BYOD_CELL = "modnet_matting-18"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


PHOTO_CELL = "modnet_matting-30"
STEM = "modnet_matting"
EARLIER = {"splits": "old", "test_records": "old", "frozen_test": "old", "adapt_result": "old", "evaluation_report": "old", "result_payload": "old"}
PHOTO_EARLIER = {"photo": "old", "photo_report": "old", "adapted_photo": "old", "result_payload": "old"}


def _base_namespace() -> dict:
    ns: dict = {}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    return ns


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True)
    for name in ("sample_pairs.csv", "evaluation_report.json", "result.json", "predictions.json", "matte_adapted_own-photograph.png", "cutout_frozen_p1.png"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    source = source[: source.index("else:\n    splits = sample_dataset()")]
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = {**_base_namespace(), **EARLIER, "pipe": "kept"}
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "nopairs.zip").write_bytes(_zip({"a.png": b"x"}))
    (tmp_path / "badcols.zip").write_bytes(_zip({"pairs.csv": "a,b\n1,2\n"}))
    return {
        "not a .zip (the scripted /etc/hostname run)": ("hostname", "expected a labelled-portrait .zip archive"),
        "corrupt zip": ("fake.zip", "BadZipFile"),
        "no pairs.csv": ("nopairs.zip", "must contain pairs.csv"),
        "wrong columns": ("badcols.zip", "pairs.csv must have the columns id, image, alpha"),
        "missing": ("nope.zip", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["not a .zip (the scripted /etc/hostname run)", "corrupt zip", "no pairs.csv", "wrong columns", "missing"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_dataset_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL)
    source = source[: source.index("else:\n    splits = sample_dataset()")].replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.zip'", 1)
    ns = {**_base_namespace(), **EARLIER, "pipe": "kept"}
    with pytest.raises(ValueError, match="^nope.zip: BYOD path"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0 files"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip", "received 2 files")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one") and rule in str(caught.value)


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "ok.zip").write_bytes(_zip({"pairs.csv": "id,image,alpha\n"}))
    source = _cell(BYOD_CELL).replace("        try:\n            byod_records = load_byod_dataset(byod_path)", "        raise RuntimeError('internal bug')\n        try:\n            byod_records = load_byod_dataset(byod_path)", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.zip"), source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file present in every fresh runtime, not a .zip."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: expected a labelled-portrait .zip archive")


# --- the catches are narrow: only bad input is refused; an internal TypeError/KeyError/OSError keeps its traceback -----


def test_the_byod_catches_are_narrow_in_the_notebook_source():
    dataset = _cell(BYOD_CELL)
    dataset = dataset[dataset.index("if USE_BYOD:\n") : dataset.index("else:\n    splits = sample_dataset()")]
    photo = _cell(PHOTO_CELL)
    photo = photo[photo.index("if USE_BYOD_PHOTO:\n") : photo.index("    frozen_photo = ")]
    caught = {}
    for name, branch in (("dataset", dataset), ("photo", photo)):
        caught[name] = [line.strip() for line in branch.splitlines() if line.strip().startswith("except ")]
        assert not [c for c in caught[name] if "KeyError" in c], caught
        assert branch.count("    except ValueError as exc:\n") == 1
    assert [c for c in caught["dataset"] if "TypeError" in c or "OSError" in c] == ["except (OSError, zipfile.BadZipFile) as exc:", "except TypeError as exc:"]
    assert [c for c in caught["photo"] if "TypeError" in c or "OSError" in c] == ["except OSError as exc:"]
    assert "        try:\n            byod_records = load_byod_dataset(byod_path)\n        except (OSError, zipfile.BadZipFile) as exc:\n" in dataset
    assert "        except TypeError as exc:\n            raise BYODRefusal(byod_name, f'a pairs.csv row has fewer cells than the header" in dataset
    assert "        try:\n            photo = load_photo(photo_path, record_id='own-photograph')\n        except OSError as exc:\n" in photo


def _png(mode: str, size=(32, 32), value=128) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, value).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("entries", "rule"),
    [
        ({"pairs.csv": "id,image,alpha\np1,p1.png,p1_a.png\n", "p1.png": b"not a picture", "p1_a.png": b"x"}, "the archive cannot be read (UnidentifiedImageError: "),
        ({"pairs.csv": "id,image,alpha\np1,p1.png\n", "p1.png": b"x"}, "a pairs.csv row has fewer cells than the header (TypeError: "),
    ],
    ids=["undecodable image", "short row"],
)
def test_a_bad_archive_member_is_still_one_refused_line(monkeypatch, tmp_path, entries, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "bad.zip").write_bytes(_zip(entries))
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "bad.zip"))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: bad.zip: {rule}"), line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def _good_zip(n: int = 12) -> bytes:
    rows = ["id,image,alpha"] + [f"p{i},p{i}.png,p{i}_a.png" for i in range(n)]
    entries: dict = {"pairs.csv": "\n".join(rows) + "\n"}
    for i in range(n):
        entries[f"p{i}.png"] = _png("RGB", value=(10 * i, 20, 30))
        entries[f"p{i}_a.png"] = _png("L", value=128 + i)
    return _zip(entries)


@pytest.mark.parametrize("error", [TypeError("internal helper bug"), KeyError("internal key"), OSError("internal I/O bug")], ids=["TypeError", "KeyError", "OSError"])
def test_an_internal_error_after_the_loader_keeps_its_traceback_and_clears_nothing(monkeypatch, tmp_path, error):
    import traceback

    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "ok.zip").write_bytes(_good_zip())
    seen: list = []

    def broken_split(records, **kwargs):
        seen.append(len(records))
        raise error

    monkeypatch.setattr(samples_module, "split_dataset", broken_split)
    with pytest.raises(type(error)) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.zip"))
    assert seen == [12]  # the loader accepted the archive; the error comes from the helper after it
    assert caught.value is error and not getattr(caught.value, "dimer_refusal", False)
    rendered = "".join(traceback.format_exception(caught.value))
    assert rendered.startswith("Traceback") and "broken_split" in rendered and "Refused:" not in rendered
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


@pytest.mark.parametrize("error", [TypeError("byod_file bug"), OSError("byod_file I/O bug")], ids=["TypeError", "OSError"])
def test_an_internal_error_in_the_photo_branch_keeps_its_traceback(monkeypatch, tmp_path, error):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (40, 30)).save(tmp_path / "me.png")

    def buggy_byod_file(path, kind, suffixes=()):
        raise error

    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    helpers = _cell(BYOD_CELL)
    ns = {**_base_namespace(), **PHOTO_EARLIER, "splits": "kept"}
    exec(compile(helpers[: helpers.index("os.makedirs('outputs'")], "<section 4 helpers>", "exec"), ns)
    ns.update(USE_BYOD_PHOTO=True, BYOD_PHOTO_PATH=str(tmp_path / "me.png"), byod_file=buggy_byod_file)
    source = _cell(PHOTO_CELL)
    branch = source[source.index("if USE_BYOD_PHOTO:\n") : source.index("    frozen_photo = ")]
    with pytest.raises(type(error)) as caught:
        exec(compile(branch, "<section 10>", "exec"), ns)
    assert caught.value is error and not getattr(caught.value, "dimer_refusal", False)
    assert ns["photo_report"] == "old" and (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is cleared


# --- Section 8 own photograph ------------------------------------------------------------------------------------


def _photo(monkeypatch, tmp_path, path: str) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    helpers = _cell(BYOD_CELL)
    ns = {**_base_namespace(), **PHOTO_EARLIER, "splits": "kept"}
    exec(compile(helpers[: helpers.index("os.makedirs('outputs'")], "<section 4 helpers>", "exec"), ns)
    source = _cell(PHOTO_CELL)
    branch = source[source.index("if USE_BYOD_PHOTO:\n") : source.index("    frozen_photo = ")]
    ns.update(USE_BYOD_PHOTO=True, BYOD_PHOTO_PATH=path)
    exec(compile(branch, "<section 10>", "exec"), ns)
    return ns


@pytest.mark.parametrize(("name", "rule"), [("hostname", "expected a photograph (JPEG or PNG) ending in"), ("fake.png", "UnidentifiedImageError"), ("nope.jpg", "does not exist or is not a file")])
def test_photo_problems_are_one_refused_line(monkeypatch, tmp_path, name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.png").write_text("not a picture", encoding="utf-8")
    holder: dict = {}
    with pytest.raises(ValueError) as caught:
        holder.update(_photo(monkeypatch, tmp_path, str(tmp_path / name)))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {name}: ") and rule in line
    remaining = sorted(p.name for p in (tmp_path / "outputs").iterdir())
    assert remaining == [f"{STEM}_cutout_frozen_p1.png", f"{STEM}_evaluation_report.json", f"{STEM}_sample_pairs.csv"], remaining


def test_good_photo_still_loads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (40, 30)).save(tmp_path / "me.png")
    ns = _photo(monkeypatch, tmp_path, str(tmp_path / "me.png"))
    assert ns["photo_name"] == "me.png" and ns["photo"]["id"] == "own-photograph"
