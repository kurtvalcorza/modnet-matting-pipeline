#!/usr/bin/env python3
"""Generate a STANDALONE DIMER tutorial notebook (NOTEBOOK_SPEC 2.0 §4) from repository sources — /2.2.

/2 adds to /1: multi-module packages (one tagged cell per module, topologically ordered, package-relative
imports removed), template-declared rewrite rules, and extra pinned snapshots (`extra_weights`) for packages
that stage more than one manifest. Single-module templates render as in /1 except for the generator version.

/2.2 adds three opt-in template keys (2026-10-05 notebook-review fix cycle):
- ``isolated_runtime``: nothing is pip-installed into the notebook kernel. One kernel cell downloads a pinned `uv`
  wheel (size + SHA-256), builds a managed-CPython environment from a hash-locked lock (`--require-hashes
  --only-binary :all:`) and routes every later cell to one persistent worker process in that environment, so a
  hosted runtime's preloaded packages are never replaced and no restart is needed (NOTEBOOK_SPEC 2.2 §5, RUN1,
  RUN10, ENV6). The environment is keyed on the lock digest and reused by a re-run or a second Run all; re-running
  the cell keeps a live worker (and every variable later cells created) instead of replacing it. The worker
  forces `MPLBACKEND=Agg` and drops `PYTHONPATH`, `PYTHONHOME` and `PYTHONSTARTUP`. Mechanism: the fleet's
  isolated-runtime carrier (bioclip2-biodiversity-pipeline, build_notebook.py/2.1).
- ``infrastructure_labels``: setup, carrier and snapshot cells are labelled **Infrastructure** and collapsed
  (`cellView: form`, Jupyter `source_hidden`) (GDL11).
- ``guided``: ``{'opening': [markdown cells after the header]}`` for the guided layer's orientation (GDL1–GDL4).

Usage (from the repository root, or with --repo):
    python tools/build_notebook.py            # write tutorials/<notebook_name>
    python tools/build_notebook.py --check    # exit 1 if the committed notebook differs (PAR3)
    python tools/build_notebook.py --out PATH # write elsewhere (review copies)

The per-repository template is ``tools/notebook_template.py`` and exposes ``TEMPLATE`` (see
``template_contract`` below). This file is vendored per repository; the fleet copy lives in the
relay ``shared/`` directory and is the one to edit first.
"""
# ruff: noqa: E501  -- learner-facing prose is kept on single lines so the rendered markdown stays readable
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "build_notebook.py/2.2"
NOTEBOOK_SPEC = "2.2"

# ST2: default rewrite rule; a template may replace it with its own `rewrites` list. Every rule must
# match exactly once across the embedded modules, so a silent no-op is impossible.
DEFAULT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        r"^DEFAULT_WEIGHTS_DIR = Path\(__file__\)[^\n]*$",
        'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / MODEL_KEY'
        "  # standalone rewrite (build_notebook.py): working-directory-relative",
    ),
)
# Package-relative imports are removed: in the notebook every module's names are already globals of the
# kernel, and the cells are emitted in dependency order so each name exists before it is used.
_REL_IMPORT_MULTI = re.compile(r"^(?P<indent>[ \t]*)from \.(\w+) import \((?P<names>[^)]*)\)[ \t]*$", re.M | re.S)
_REL_IMPORT_LINE = re.compile(r"^(?P<indent>[ \t]*)from \.(\w+) import (?P<names>[^\n(]+)$", re.M)

_INSTALL_GUARD = '''
def _installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

if not SKIP_INSTALL:
    # Capture every distribution already imported in this runtime, whatever its module name
    # (PIL -> pillow), so a pinned install that replaces a loaded package is detected and the
    # notebook stops with a restart instruction instead of continuing with mixed versions.
    _module_dists = importlib.metadata.packages_distributions()
    _loaded = sorted({d for m in list(sys.modules) for d in _module_dists.get(m.partition('.')[0], ())})
    loaded = {distribution: _installed_version(distribution) for distribution in _loaded}
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)
    importlib.invalidate_caches()
    stale = []
    for distribution, before in loaded.items():
        installed = _installed_version(distribution)
        if before is not None and before != installed:
            stale.append(f'{distribution}: loaded={before}, installed={installed}')
    if stale:
        raise RuntimeError('Core dependencies changed while older modules were loaded: ' + '; '.join(stale) + '. Restart the runtime, then rerun from the top.')
'''


# /2.2 isolated runtime (template key `isolated_runtime`). The kernel only bootstraps: this one cell builds (or reuses)
# the hash-locked environment and starts (or keeps) the worker that runs every later cell. It is the only cell that
# runs in the kernel, and it is safe to re-run on its own.
_ISOLATED_INSTALL = """# @title Infrastructure: build (or reuse) the isolated environment and route later cells to it
# dimer: kernel cell (runs in the notebook kernel, not in the isolated environment)
import hashlib
import io
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

{pins_literal}
MANAGED_PYTHON = {python!r}
UV_URL = {uv_url!r}
UV_BYTES = {uv_bytes}
UV_SHA256 = {uv_sha256!r}
LOCK_NAME = {lock_name!r}
LOCK_SHA256 = {lock_sha256!r}
LOCKED_PACKAGES = {n_locked}
# The hash-locked requirements, compiled from the PINS above with `uv pip compile --generate-hashes` for manylinux x86_64.
LOCK_TEXT = r'''{lock_text}'''

SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'
# One environment per lock digest: a re-run of this cell, or a second Run all in the same runtime, reuses a complete
# environment built from this exact lock instead of rebuilding it; a different lock gets a different folder.
ISOLATED_ENV = Path(os.environ.get('DIMER_ISOLATED_ENV', 'dimer_isolated_env_' + LOCK_SHA256[:12])).resolve()
ISOLATED_PYTHON = ISOLATED_ENV / 'bin' / 'python'
ISOLATED_TOOLS = ISOLATED_ENV.with_name(ISOLATED_ENV.name + '_tools')
ISOLATED_READY = ISOLATED_ENV / '.dimer-lock-sha256'


def _isolated_environment_ready():
    return ISOLATED_PYTHON.is_file() and ISOLATED_READY.is_file() and ISOLATED_READY.read_text(encoding='utf-8').strip() == LOCK_SHA256


if SKIP_INSTALL:
    print('DIMER_NOTEBOOK_CI_PREINSTALLED=1: the pins are already installed; the notebook runs in this kernel.')
elif _isolated_environment_ready():
    print({{'isolated_environment': str(ISOLATED_ENV), 'reused': True, 'lock_sha256': LOCK_SHA256[:16] + '...', 'locked_packages': LOCKED_PACKAGES, 'kernel_python': platform.python_version()}})
else:
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle or Linux Jupyter): its locked environment is built for manylinux x86_64.')
    setup_started = time.perf_counter()
    if hashlib.sha256(LOCK_TEXT.encode('utf-8')).hexdigest() != LOCK_SHA256:
        raise RuntimeError('The carried lock does not match its digest: regenerate the notebook from the repository instead of editing this cell.')
    ISOLATED_TOOLS.mkdir(parents=True, exist_ok=True)
    lock_path = ISOLATED_TOOLS / LOCK_NAME
    lock_path.write_text(LOCK_TEXT, encoding='utf-8', newline='\\n')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UV_URL, timeout=90) as response:
                wheel = response.read(UV_BYTES + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
        raise RuntimeError('The pinned uv wheel failed its size/SHA-256 check: refusing to run it. Run this cell again; if it repeats, the download is being altered.')
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = next(name for name in archive.namelist() if name.endswith('.data/scripts/uv'))
        uv = ISOLATED_TOOLS / 'uv'
        uv.write_bytes(archive.read(member))
    uv.chmod(0o700)
    # uv gets no kernel Python path; the managed interpreter is downloaded once and reused on a re-run.
    uv_env = dict(os.environ, MPLBACKEND='Agg')
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
        uv_env.pop(name, None)
    if not ISOLATED_PYTHON.is_file():
        subprocess.run([str(uv), 'venv', '--quiet', '--managed-python', '--python', MANAGED_PYTHON, str(ISOLATED_ENV)], env=uv_env, check=True)
    isolated_version = subprocess.run([str(ISOLATED_PYTHON), '-c', 'import platform; print(platform.python_version())'], env=uv_env, check=True, capture_output=True, text=True).stdout.strip()
    if isolated_version != MANAGED_PYTHON:
        raise RuntimeError(f'{{ISOLATED_ENV}} holds Python {{isolated_version}}, not {{MANAGED_PYTHON}}: delete that folder (or start a fresh runtime) and run this cell again.')
    ISOLATED_READY.unlink(missing_ok=True)
    subprocess.run([str(uv), 'pip', 'install', '--quiet', '--python', str(ISOLATED_PYTHON), '--require-hashes', '--only-binary', ':all:', '--index-url', 'https://pypi.org/simple', '-r', str(lock_path)], env=uv_env, check=True)
    ISOLATED_READY.write_text(LOCK_SHA256 + '\\n', encoding='utf-8')
    print({{'isolated_environment': str(ISOLATED_ENV), 'reused': False, 'isolated_python': isolated_version, 'kernel_python': platform.python_version(), 'locked_packages': LOCKED_PACKAGES, 'setup_seconds': round(time.perf_counter() - setup_started)}})

# The worker runs in the isolated environment. It executes each routed cell in one persistent namespace and sends
# back printed text, displayed objects and matplotlib figures, so every cell behaves as it would in the kernel.
"""

_ISOLATED_ROUTER = (
    r'''_WORKER_SOURCE = r"""
import ast, base64, builtins, io, linecache, os, signal, sys, traceback, types
from multiprocessing.connection import Connection

_send = Connection(int(sys.argv[1]), readable=False)
_recv = Connection(int(sys.argv[2]), writable=False)


class _Stream(io.TextIOBase):
    def __init__(self, name):
        self._name = name

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, text):
        if text:
            _send.send(("stream", self._name, str(text)))
        return len(text)


sys.stdout, sys.stderr = _Stream("stdout"), _Stream("stderr")


def _figure_bundle(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": repr(fig)}


def _flush_figures():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for number in plt.get_fignums():
        _send.send(("display", _figure_bundle(plt.figure(number))))
    plt.close("all")


def _mimebundle(obj):
    if hasattr(obj, "savefig"):
        return _figure_bundle(obj)
    data = {"text/plain": repr(obj)}
    for method, mime in (("_repr_html_", "text/html"), ("_repr_markdown_", "text/markdown"), ("_repr_png_", "image/png")):
        render = getattr(obj, method, None)
        if callable(render):
            try:
                value = render()
            except Exception:
                value = None
            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")
            if value is not None:
                data[mime] = value
    return data


def display(*objects, **kwargs):
    for obj in objects:
        _send.send(("display", _mimebundle(obj)))


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    matplotlib.pyplot.show = lambda *args, **kwargs: _flush_figures()
except ImportError:
    pass

if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":
    # google.colab only exists in the kernel; forward the BYOD upload dialog to it.
    def _upload():
        _send.send(("upload",))
        reply = _recv.recv()
        if reply[1] is None:
            raise RuntimeError("The notebook kernel could not open the upload dialog.")
        return reply[1]

    import importlib.machinery

    def _stub(name, package):
        # A spec on every stub: importlib.util.find_spec("google.colab") (accelerate does this) raises on a None __spec__.
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=package)
        if package:
            module.__path__ = []
        return module

    try:
        import google
    except ImportError:
        google = _stub("google", True)
        sys.modules["google"] = google
    _colab = _stub("google.colab", True)
    _files = _stub("google.colab.files", False)
    _files.upload = _upload
    _colab.files = _files
    google.colab = _colab
    sys.modules["google.colab"] = _colab
    sys.modules["google.colab.files"] = _files

_main = types.ModuleType("__main__")
_main.__dict__.update(__builtins__=builtins, display=display)
sys.modules["__main__"] = _main
_count = 0
while True:
    # An interrupt only lands inside a running cell; between cells it is ignored.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        message = _recv.recv()
    except EOFError:
        break
    if message[0] != "run":
        continue
    _count += 1
    filename = f"<isolated cell {_count}>"
    source = message[1]
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        tree = ast.parse(source, filename)
        tail = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, filename, "exec"), _main.__dict__)
        if tail is not None:
            value = eval(compile(tail, filename, "eval"), _main.__dict__)
            if value is not None:
                display(value)
        _flush_figures()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _send.send(("done",))
    except BaseException as exc:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        frames = exc.__traceback__.tb_next if exc.__traceback__ is not None else None
        _send.send(("error", "".join(traceback.format_exception(type(exc), exc, frames)), f"{type(exc).__name__}: {exc}"))
"""


class IsolatedCellError(RuntimeError):
    """A routed cell raised inside the isolated environment; its traceback is printed above."""


class IsolatedRuntime:
    """One persistent worker process in the isolated environment, fed one cell at a time."""

    def __init__(self, python, display=None):
        self.python = str(python)
        to_kernel_r, to_kernel_w = os.pipe()
        to_worker_r, to_worker_w = os.pipe()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", DIMER_NOTEBOOK_CI_PREINSTALLED="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(name, None)
        env["DIMER_KERNEL_IS_COLAB"] = "1" if "google.colab" in sys.modules else "0"
        self.proc = subprocess.Popen(
            [str(python), "-c", _WORKER_SOURCE, str(to_kernel_w), str(to_worker_r)],
            pass_fds=(to_kernel_w, to_worker_r),
            env=env,
            start_new_session=True,  # interrupts reach the worker only through run(), exactly once
        )
        os.close(to_kernel_w)
        os.close(to_worker_r)
        self._recv = Connection(to_kernel_r, writable=False)
        self._send = Connection(to_worker_w, readable=False)
        if display is None:
            from IPython.display import display
        self._display = display

    def alive(self):
        return self.proc.poll() is None

    def _exited(self):
        return RuntimeError(
            f"The isolated environment's Python process exited (code {self.proc.wait()}); a crash of this kind is "
            "usually running out of memory. Restart the session and choose Run all again."
        )

    def run(self, source):
        try:
            self._send.send(("run", source))
        except OSError:
            raise self._exited() from None
        while True:
            try:
                message = self._recv.recv()
            except EOFError:
                raise self._exited() from None
            except KeyboardInterrupt:
                self.proc.send_signal(signal.SIGINT)
                continue
            kind = message[0]
            if kind == "stream":
                (sys.stdout if message[1] == "stdout" else sys.stderr).write(message[2])
            elif kind == "display":
                self._display(message[1], raw=True)
            elif kind == "upload":
                self._send.send(("upload", self._colab_upload()))
            elif kind == "error":
                sys.stderr.write(message[1])
                raise IsolatedCellError(message[2]) from None
            elif kind == "done":
                return

    @staticmethod
    def _colab_upload():
        try:
            from google.colab import files
        except ImportError:
            return None
        return files.upload()

    def close(self):
        self._send.close()
        self.proc.wait(timeout=30)


def _is_user_cell():
    # ipykernel transforms a cell before executing it; its caller knows whether this is a silent frontend request.
    frame = sys._getframe(2)
    while frame is not None:
        local = frame.f_locals
        if "silent" in local and "store_history" in local:
            return bool(local["store_history"]) and not bool(local["silent"])
        frame = frame.f_back
    return True


def _route_to_isolated_runtime(lines):
    source = "".join(lines)
    if not source.strip() or "# dimer: kernel cell" in source or not _is_user_cell():
        return lines
    return [f"_DIMER_ISOLATED_RUNTIME.run({source!r})\n"]'''
    + """

if SKIP_INSTALL:
    print('Routing disabled: the notebook runs in this kernel.')
else:
    from IPython import get_ipython as _kernel_shell

    _ip = _kernel_shell()
    _ip.input_transformers_cleanup[:] = [t for t in _ip.input_transformers_cleanup if getattr(t, '__name__', '') != '_route_to_isolated_runtime']
    # Idempotent: a live worker on this environment is kept, with every variable the later cells created, so
    # re-running this cell alone never strands the cells after it. A dead or different worker is replaced.
    _previous = globals().get('_DIMER_ISOLATED_RUNTIME')
    _reuse = _previous is not None and getattr(_previous, 'python', None) == str(ISOLATED_PYTHON) and _previous.proc.poll() is None
    if _reuse:
        _DIMER_ISOLATED_RUNTIME = _previous
    else:
        if _previous is not None and _previous.proc.poll() is None:
            _previous.close()
        _DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)
    _ip.input_transformers_cleanup.append(_route_to_isolated_runtime)
    print({'routed_to': str(ISOLATED_PYTHON), 'worker_pid': _DIMER_ISOLATED_RUNTIME.proc.pid, 'worker_reused': _reuse})"""
)


def template_contract() -> dict[str, str]:
    """Keys ``TEMPLATE`` must define (documentation for template authors). Optional keys are marked."""
    return {
        "package": "import name of the repository package, e.g. resnet50_classification_pipeline",
        "repo_name": "GitHub repository name",
        "stem": "output file stem, e.g. resnet50_classification (notebook, outputs/ files)",
        "notebook_name": "tutorials/<notebook_name>",
        "profile": "TASK-INFERENCE | MULTI-CAPABILITY | E2E | ARTIFACT-INFERENCE",
        "pipeline_class": "public class exposing from_pretrained(weights_dir=...)",
        "runtime_imports": "list of principal libraries whose versions the runtime cell prints, e.g. ['torch', 'timm']",
        "title": "H1 text",
        "badges": "list of (alt, image_url, link_url)",
        "capability": "one-line capability statement",
        "intro": "markdown paragraphs after the header block (no heading)",
        "learning_objectives": "one markdown paragraph starting after the bold label",
        "exclusions": "one markdown sentence after the bold label",
        "prerequisites": "list of markdown bullets; the generator appends the External access bullet",
        "cells": "list of {'md': str, 'code': str} stage cells inserted after the model cell; may use {stem}, {MODEL_ID}, {MODEL_REVISION}",
        "closing": "markdown for Interpretation and limits + References",
        "weights_key": "MODEL_KEY value (weights/<key>/dimer-base-manifest.json)",
        # optional:
        "modules": "OPTIONAL list of module files under src/<package>/ to embed (default ['pipeline.py']); dependency order is computed",
        "entry_module": "OPTIONAL module that defines MODEL_ID/MODEL_REVISION/MODEL_LICENSE/MODEL_KEY (default 'pipeline.py')",
        "rewrites": "OPTIONAL list of [regex, replacement] applied to the embedded modules, each matching exactly once (default DEFAULT_REWRITES)",
        "extra_weights": "OPTIONAL list of {key, var, dir, identity: [ID_CONST, REV_CONST], stage, verify} for additional pinned snapshots",
        "model_load": "OPTIONAL replacement for the default `<pipeline_class>.from_pretrained(weights_dir=WEIGHTS_DIR)` expression",
        "package_dir": "OPTIONAL repository-relative directory of the package (default 'src/<package>'; e.g. 'mitra_pipeline' for a root-level package)",
        "isolated_runtime": "OPTIONAL bool (default False): install the hash-locked pins into a separate uv environment (managed CPython) and route every later cell to a persistent worker there; nothing is installed into the kernel and no restart is needed (NOTEBOOK_SPEC 2.2 §5). Needs managed_python, uv and lock",
        "managed_python": "OPTIONAL (required with isolated_runtime): exact CPython version uv installs for the isolated environment",
        "uv": "OPTIONAL (required with isolated_runtime): {'version', 'url', 'bytes', 'sha256'} of the pinned manylinux x86_64 uv wheel",
        "lock": "OPTIONAL (required with isolated_runtime): repository-relative hash-locked requirements compiled from the pins",
        "infrastructure_labels": "OPTIONAL bool (default False): label the setup, carrier and snapshot sections Infrastructure and collapse their cells (NOTEBOOK_SPEC 2.2 GDL11)",
        "guided": "OPTIONAL {'opening': [markdown cells inserted after the header]} (NOTEBOOK_SPEC 2.2 GDL1-GDL4)",
    }


REQUIRED_KEYS = [k for k, v in template_contract().items() if not v.startswith("OPTIONAL")]


def load_template(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("notebook_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.TEMPLATE
    missing = [k for k in REQUIRED_KEYS if k not in template]
    if missing:
        raise SystemExit(f"template missing keys: {missing}")
    return template


def _relative_imports(text: str, module: str) -> set[str]:
    """Names of sibling modules imported at module top level; `if TYPE_CHECKING:` imports are ignored
    (never executed), any other nested relative import is refused (it would fail inside a notebook)."""
    import ast

    tree = ast.parse(text)
    top: set[str] = set()
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = test.id if isinstance(test, ast.Name) else (test.attr if isinstance(test, ast.Attribute) else "")
            if name == "TYPE_CHECKING":
                guarded.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.level > 1 or node.module is None:
                raise SystemExit(f"{module}: `from . import x` / nested relative imports are not embeddable")
            if node in tree.body:
                top.add(node.module)
            # Nested runtime imports (lazy imports that break cycles) are rewritten to `pass` by
            # _strip_relative_imports: in the notebook the imported names are kernel globals, defined
            # by the time any function body runs. TYPE_CHECKING-guarded ones never execute and stay.
        if isinstance(node, ast.Import) and any(a.name.startswith(".") for a in node.names):
            raise SystemExit(f"{module}: `import .x` is not embeddable")
    return top


def _module_order(pkg_dir: Path, modules: list[str]) -> list[str]:
    """Topological order of the embedded modules by their package-relative imports (Kahn)."""
    stems = {m: Path(m).stem for m in modules}
    deps: dict[str, set[str]] = {}
    for m in modules:
        text = (pkg_dir / m).read_text(encoding="utf-8")
        found = _relative_imports(text, m)
        unknown = found - set(stems.values())
        if unknown:
            raise SystemExit(f"{m}: imports modules not listed in template['modules']: {sorted(unknown)}")
        deps[m] = {mm for mm in modules if stems[mm] in found and mm != m}
    order: list[str] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(m for m, d in remaining.items() if not (d - set(order)))
        if not ready:
            raise SystemExit(f"circular package-relative imports among {sorted(remaining)}")
        order.extend(ready)
        for m in ready:
            remaining.pop(m)
    return order


def _strip_relative_imports(text: str, module: str) -> str:
    def _check_names(match: re.Match[str]) -> str:
        names = match.group("names")
        if " as " in names:
            raise SystemExit(f"{module}: aliased relative import cannot be embedded: {match.group(0)[:60]}")
        indent = match.group("indent")
        first = match.group(0).splitlines()[0].strip()
        note = f"# standalone rewrite (build_notebook.py): `{first}` removed — names are kernel globals defined by the carried modules"
        # An indented import sits in a function/class/`if TYPE_CHECKING:` body: keep the block valid.
        return f"{indent}pass  {note}" if indent else note

    text = _REL_IMPORT_MULTI.sub(_check_names, text)
    text = _REL_IMPORT_LINE.sub(_check_names, text)
    return text


REWRITES = DEFAULT_REWRITES  # /1-compatible name used by parity tests


def apply_rewrites(
    texts: dict[str, str] | str,
    rewrites: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
) -> dict[str, str] | str:
    """Apply each rule exactly once across all modules, then strip package-relative imports.

    Accepts a single module text (returns text — the /1 contract used by parity tests) or a
    {module: text} mapping (returns the mapping)."""
    if isinstance(texts, str):
        return apply_rewrites({"pipeline.py": texts}, rewrites)["pipeline.py"]
    rewrites = DEFAULT_REWRITES if rewrites is None else rewrites
    out = dict(texts)
    for pattern, replacement in rewrites:
        total = 0
        for name, text in out.items():
            text, n = re.subn(pattern, replacement, text, flags=re.M)
            out[name] = text
            total += n
        if total != 1:
            raise SystemExit(f"rewrite rule matched {total} times across modules (expected 1): {pattern}")
    return {name: _strip_relative_imports(text, name).rstrip("\n") + "\n" for name, text in out.items()}


def _pins(repo: Path) -> list[str]:
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not block:
        raise SystemExit("pyproject.toml: dependencies block not found")
    pins = re.findall(r'"([^"]+)"', block.group(1))
    bad = [p for p in pins if "==" not in p]
    if bad:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {bad}")
    return pins


def _canonical(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_packages(lock_text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock."""
    return {_canonical(m.group(1)): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


def check_lock(pins: list[str], lock_text: str) -> None:
    """Every direct pin must appear in the lock at the same version, and every lock entry must carry a hash."""
    locked = lock_packages(lock_text)
    for pin in pins:
        if pin.startswith("--") or "==" not in pin:
            raise SystemExit(f"isolated_runtime supports `name==version` pins only, found {pin!r}")
        name, version = pin.split("==", 1)
        name = re.split(r"[\[;\s]", name, maxsplit=1)[0]
        if locked.get(_canonical(name)) != version.strip():
            raise SystemExit(f"lock does not pin {pin} (found {locked.get(_canonical(name))}); recompile the lock")
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock_text.split("\n", 2)[-1])
    unhashed = [b.split("==", 1)[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")
    if "'''" in lock_text:
        raise SystemExit("lock text cannot be carried in a raw triple-quoted literal")


def _isolated_install(ctx: dict[str, Any], template: dict[str, Any], pins_literal: str) -> str:
    lock_text = ctx["lock_text"]
    uv = template["uv"]
    return _ISOLATED_INSTALL.format(
        pins_literal=pins_literal,
        python=template["managed_python"],
        uv_url=uv["url"],
        uv_bytes=int(uv["bytes"]),
        uv_sha256=uv["sha256"],
        lock_name=Path(template["lock"]).name,
        lock_sha256=hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        n_locked=len(lock_packages(lock_text)),
        lock_text=lock_text,
    ) + _ISOLATED_ROUTER


def _head_revision(repo: Path) -> str:
    """The repository revision the notebook is generated from (ST5): HEAD at generation time.

    A provenance label only; the parity anchor is the module SHA-256, so a later commit that carries
    the regenerated notebook does not invalidate it (see ``--check``).
    """
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or "uncommitted"


def recorded_revision(notebook_path: Path) -> str | None:
    """`generated_from.revision` of an existing notebook, or None."""
    if not notebook_path.exists():
        return None
    try:
        meta = json.loads(notebook_path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
        return str(meta["revision"])
    except (KeyError, ValueError, TypeError):
        return None


def _read_manifest(repo: Path, key: str) -> dict[str, Any]:
    path = repo / "weights" / key / "dimer-base-manifest.json"
    if not path.is_file():
        raise SystemExit(f"manifest missing: {path} (MOD13: standalone needs a committed manifest)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    pkg = template["package"]
    pkg_rel = template.get("package_dir", f"src/{pkg}")
    pkg_dir = repo / pkg_rel
    modules = list(template.get("modules", ["pipeline.py"]))
    entry = template.get("entry_module", "pipeline.py")
    if entry not in modules:
        raise SystemExit(f"entry_module {entry!r} must be listed in modules {modules}")
    order = _module_order(pkg_dir, modules)
    texts = {m: (pkg_dir / m).read_text(encoding="utf-8") for m in order}
    entry_text = texts[entry]
    # `identity_names` lets a package that spells a constant differently (e.g. DEFAULT_MODEL_KEY)
    # map it onto the fleet name; the notebook still refers to the package's own spelling.
    names = {**{k: k for k in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")}, **template.get("identity_names", {})}
    ident: dict[str, str] = {}
    for k, src_name in names.items():
        m = re.search(rf'^{src_name} = "([^"]+)"$', entry_text, re.M)
        if not m:
            raise SystemExit(f"{entry}: {src_name} not found as a top-level string constant")
        ident[k] = m.group(1)
    ident_expr = dict(names)  # constant names as they appear in the carried code (used by the model cell)
    manifest = _read_manifest(repo, template["weights_key"])
    if manifest["modelId"] != ident["MODEL_ID"] or manifest["revision"] != ident["MODEL_REVISION"]:
        raise SystemExit("manifest identity != module identity")
    if template["weights_key"] != ident["MODEL_KEY"]:
        raise SystemExit("template weights_key != MODEL_KEY")
    extra = []
    for spec in template.get("extra_weights", []):
        em = _read_manifest(repo, spec["key"])
        id_const, rev_const = spec["identity"]
        eid = re.search(rf'^{id_const} = "([^"]+)"$', entry_text, re.M)
        erev = re.search(rf'^{rev_const} = "([^"]+)"$', entry_text, re.M)
        if not (eid and erev):
            raise SystemExit(f"{entry}: {id_const}/{rev_const} not found for extra snapshot {spec['key']}")
        if (em["modelId"], em["revision"]) != (eid.group(1), erev.group(1)):
            raise SystemExit(f"extra manifest {spec['key']} identity != module constants")
        extra.append({**spec, "manifest": em})
    rewrites = template.get("rewrites", DEFAULT_REWRITES)
    pins = _pins(repo)
    lock_text = None
    if template.get("isolated_runtime"):
        missing = [k for k in ("managed_python", "uv", "lock") if k not in template]
        if missing:
            raise SystemExit(f"isolated_runtime needs template keys {missing}")
        # Read as text (universal newlines), so a CRLF checkout carries the same LF lock and digest.
        lock_text = (repo / template["lock"]).read_text(encoding="utf-8")
        check_lock(pins, lock_text)
    rel = [f"{pkg_rel}/{m}" for m in order]
    return {
        "pkg": pkg,
        "pkg_rel": pkg_rel,
        "modules": order,
        "module_rels": rel,
        "entry_rel": f"{pkg_rel}/{entry}",
        "texts": texts,
        "embedded": apply_rewrites(texts, rewrites),
        "module_sha256": hashlib.sha256("".join(texts[m] for m in order).encode("utf-8")).hexdigest(),
        "per_module_sha256": {f"{pkg_rel}/{m}": hashlib.sha256(texts[m].encode("utf-8")).hexdigest() for m in order},
        "module_revision": revision or _head_revision(repo),
        "manifest": manifest,
        "extra_weights": extra,
        "pins": pins,
        "lock_text": lock_text,
        "n_rewrites": len(rewrites),
        "ident_expr": ident_expr,
        **ident,
    }


def _md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": "", "metadata": metadata or {}, "outputs": [], "source": source.rstrip("\n")}

# NOTEBOOK_SPEC 2.0 §3.4/§28 declarations. A template MAY override `mode`, `run_all` and `byod`;
# E2E and ARTIFACT-INFERENCE templates MUST state `run_all` themselves (their default paths differ).
MODES = ("REFERENCE", "GUIDED", "WORKSHOP")
_RUN_ALL_DEFAULT = {
    "TASK-INFERENCE": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned snapshot, obtains the tutorial sample automatically, validates it into an input manifest before the model "
        "runs, runs the task locally in this kernel, writes the evaluation report, and exports machine-readable outputs "
        "with provenance. The default path needs no repository clone, no DIMER worker or service, no credential, no upload "
        "dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "MULTI-CAPABILITY": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned snapshot, obtains the tutorial sample automatically, validates it into an input manifest before the model "
        "runs, runs every demonstrated capability locally in this kernel with its own input/output contract, writes the "
        "evaluation report, and exports machine-readable outputs with provenance. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5)."
    ),
}
_BYOD_DEFAULT = (
    "After the sample workflow completes, set `USE_BYOD = True` in the sample cell and re-run from that cell to supply "
    "your own input. It passes through the same notebook-local validation, task, evaluation-report and export cells as "
    "the sample; the expected input format, the ceilings and the privacy guidance are stated in the Prerequisites and "
    "in the sample cell, and the upload stays inside this runtime. BYOD is optional and never part of the default path."
)


def _declarations(template: dict[str, Any]) -> tuple[str, str, str]:
    mode = template.get("mode", "GUIDED")
    if mode not in MODES:
        raise SystemExit(f"template mode {mode!r} is not one of {MODES}")
    run_all = template.get("run_all") or _RUN_ALL_DEFAULT.get(template["profile"])
    if not run_all:
        raise SystemExit(f"template must state `run_all` for profile {template['profile']}")
    return mode, run_all.strip(), (template.get("byod") or _BYOD_DEFAULT).strip()


def render(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    ctx = load_context(repo, template, revision)
    mode, run_all, byod = _declarations(template)
    stem = template["stem"]
    fmt = {"stem": stem, **{k: ctx[k] for k in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")}}
    cells: list[dict[str, Any]] = []

    def add(cell: dict[str, Any]) -> None:
        cell["id"] = f"{stem}-{len(cells):02d}"
        cells.append(cell)

    badges = " ".join(f"[![{alt}]({img})]({link})" for alt, img, link in template["badges"])
    total_mb = (ctx["manifest"]["totalBytes"] + sum(e["manifest"]["totalBytes"] for e in ctx["extra_weights"])) / 1e6
    n_mod = len(ctx["modules"])
    carried = (
        f"the repository's pipeline module (`{ctx['entry_rel']}` at revision `{ctx['module_revision'][:12]}`) verbatim in Section 2"
        if n_mod == 1
        else f"the repository's package ({n_mod} modules under `{ctx['pkg_rel']}/`, at revision `{ctx['module_revision'][:12]}`) verbatim in Section 2"
    )
    header = (
        f"# {template['title']}\n\n{badges}\n\n"
        f"**Profile:** `{template['profile']}`  \n"
        f"**Mode:** `{mode}`  \n"
        f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)  \n"
        f"**Capability:** {template['capability']}\n\n"
        f"**This notebook is standalone.** It carries {carried}, the pinned model identity and the per-file SHA-256 manifest in Section 3, "
        f"and the exact runtime pins in Section 1, so it keeps working after export even if the repository changes or disappears. Its only "
        f"external dependencies are the pinned PyPI distributions and the Hugging Face Hub at the immutable revision `{ctx['MODEL_REVISION']}` "
        f"(~{total_mb:.0f} MB, digest-verified before loading). It was generated by `tools/build_notebook.py` ({GENERATOR_VERSION}); edit the "
        f"repository and regenerate rather than editing cells.\n\n"
        f"**Run all:** {run_all}\n\n"
        f"**Bring Your Own Data:** {byod}\n\n"
        f"{template['intro'].strip()}\n\n"
        f"**Learning objectives:** {template['learning_objectives'].strip()}\n\n"
        f"**This notebook does not demonstrate:** {template['exclusions'].strip()}"
    )
    add(_md(header))
    for opening in (template.get("guided") or {}).get("opening", []):
        add(_md(opening.format(**fmt)))

    isolated = bool(template.get("isolated_runtime"))
    # GDL11: infrastructure label and collapsed cells (Colab form view; Jupyter source_hidden).
    labels = bool(template.get("infrastructure_labels"))
    infra = (
        "> **Infrastructure.** You may run this section without studying its implementation; the learning activities "
        "start after Section 3.\n\n"
        if labels
        else ""
    )
    collapsed: dict[str, Any] = {"cellView": "form", "jupyter": {"source_hidden": True}} if labels else {}
    isolated_access = (
        f" The isolated environment also needs PyPI (`pypi.org`, `files.pythonhosted.org`) for the pinned `uv` wheel and the "
        f"{len(lock_packages(ctx['lock_text']))} hash-locked packages, and the managed CPython {template['managed_python']} build "
        "(python-build-standalone) that `uv` downloads."
        if isolated
        else ""
    )
    prereq = list(template["prerequisites"]) + [
        f"- **External access:** the Hugging Face Hub only, to fetch the pinned `{ctx['MODEL_ID']}` snapshot (~{total_mb:.0f} MB in total) "
        f"at revision `{ctx['MODEL_REVISION'][:12]}…`. No GitHub access and no credentials are required; nothing is installed from this repository."
        + isolated_access
    ]
    add(_md("## Prerequisites\n\n" + "\n".join(prereq)))

    pins_literal = "PINS = [\n" + "".join(f"    {p!r},\n" for p in ctx["pins"]) + "]"
    imports = template["runtime_imports"]
    ident_print = ", ".join(f"'{m}': {m}.__version__" for m in imports)
    source_literal = (
        "NOTEBOOK_SOURCE = {\n"
        f"    'repository': {template['repo_name']!r},\n"
        f"    'repository_revision': {ctx['module_revision']!r},\n"
        f"    'embedded_module': {ctx['entry_rel']!r},\n"
        f"    'embedded_modules': {ctx['module_rels']!r},\n"
        f"    'module_sha256': {ctx['module_sha256']!r},\n"
        f"    'generator': {GENERATOR_VERSION!r},\n"
        f"    'notebook_spec': {NOTEBOOK_SPEC!r},\n"
        "}\n"
    )
    if isolated:
        n_locked = len(lock_packages(ctx["lock_text"]))
        add(
            _md(
                "## 1. Install the pinned runtime (isolated environment)\n\n"
                + infra
                + "Hosted runtimes such as Colab and Kaggle import some packages, NumPy and often PyTorch among them, before the "
                "first cell runs, and Python cannot swap a module that is already loaded. Installing the pins into the notebook's "
                "own Python would therefore leave mixed versions or require a manual restart. Instead, the next cell builds a "
                f"separate environment (`dimer_isolated_env_<lock digest>/`) and leaves the kernel's packages untouched: it "
                f"downloads the pinned `uv` {template['uv']['version']} wheel and refuses it unless its size and SHA-256 match, has "
                f"`uv` install the managed CPython **{template['managed_python']}** (whatever Python the kernel itself runs), and "
                f"installs the {n_locked} packages of the carried hash-locked requirements (`{template['lock']}`, compiled from the "
                "pins in the cell) with `--require-hashes --only-binary :all:`, so every package, direct or transitive, is the exact "
                "file that was locked. No repository code is installed. The lock holds manylinux x86_64 wheels, so the notebook "
                "supports **Linux x86_64 runtimes only** (Google Colab, Kaggle or Linux Jupyter); on any other platform the cell "
                "stops with that message.\n\n"
                "The same cell then starts one Python process in that environment and routes **every later code cell** to it. "
                "Printed output comes back to the notebook as usual, variables persist from cell to cell, and an error stops "
                "**Run all** as it would in the kernel. It is the only cell that runs in the kernel (it is marked "
                "`# dimer: kernel cell`), and it is safe to re-run: a complete environment built from this exact lock is reused "
                "rather than rebuilt, and a live worker is kept with every variable created so far, so the cells after it keep "
                "working. To start over, restart the session and choose **Run all**. `DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an "
                "executor that has already installed exactly these pins run every cell in its own kernel instead."
            )
        )
        add(_code(_isolated_install(ctx, template, pins_literal), {"cellView": "form", **({"jupyter": {"source_hidden": True}} if labels else {})}))
        add(
            _md(
                "### Record the runtime\n\n"
                "This is the first cell that runs in the isolated environment. Nothing is installed here. It records the "
                "notebook's source revision and the versions actually imported. Look for a dictionary reporting the notebook's "
                f"source revision, the isolated Python ({template['managed_python']}), "
                + ", ".join(f"`{m}`" for m in imports)
                + " versions, and whether CUDA is available."
            )
        )
        add(
            _code(
                "# @title Infrastructure: record the runtime\n"
                "import importlib\nimport os\nimport platform\nimport sys\n\n"
                f"{source_literal}"
                f"import {', '.join(imports)}\n"
                f"print({{'notebook_source': NOTEBOOK_SOURCE, 'python': platform.python_version(), 'executable': sys.executable, {ident_print}, 'cuda': torch.cuda.is_available()}})",
                dict(collapsed),
            )
        )
    else:
        add(
            _md(
                "## 1. Install the pinned runtime\n\n"
                "The dependency set is pinned exactly (the same `==` pins as the repository's `pyproject.toml` at the generating revision) and "
                "installed directly — there is no repository clone and no package install. If a pin replaces a distribution this runtime has already "
                "imported, the cell stops with a restart instruction rather than continuing with mixed versions. Look for a dictionary reporting the "
                "notebook's source revision, Python, " + ", ".join(f"`{m}`" for m in imports) + " versions, and whether CUDA is available."
            )
        )
        add(
            _code(
                "import importlib\nimport importlib.metadata\nimport os\nimport platform\nimport subprocess\nimport sys\n\n"
                f"{pins_literal}\n"
                f"{source_literal}"
                "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'\n"
                f"{_INSTALL_GUARD}\n"
                f"import {', '.join(imports)}\n"
                f"print({{'notebook_source': NOTEBOOK_SOURCE, 'python': platform.python_version(), {ident_print}, 'cuda': torch.cuda.is_available()}})"
            )
        )

    for i, m in enumerate(ctx["modules"]):
        rel = f"{ctx['pkg_rel']}/{m}"
        if i == 0:
            title = f"## 2. Pipeline code (carried verbatim from `{ctx['pkg_rel']}/` @ `{ctx['module_revision'][:12]}`)"
            intro = (
                f"\n\nThe next {n_mod} cell(s) **are** the repository's package, module by module in dependency order: the pinned identity constants, "
                "snapshot verification (`verify_snapshot`), staged download (`stage_missing_files`), the named operational ceilings, the public "
                "validation and evaluation helpers, and the pipeline class. The text is the modules', byte for byte, except for the rewrite rules "
                f"listed in `tools/build_notebook.py` ({ctx['n_rewrites']} rule(s), plus the removal of package-relative `from .x import` lines, whose "
                "names are already defined by the preceding cells). The repository's parity test (`tests/test_notebook_parity.py`) fails whenever "
                "these cells and the modules diverge, so what you run here is what the repository tests. Nothing in these cells runs a model yet."
            )
            label = "\n\n" + infra.rstrip("\n") if infra else ""
            add(_md(title + label + intro + f"\n\n**Module {i + 1}/{n_mod}:** `{rel}`"))
        else:
            add(_md(f"**Module {i + 1}/{n_mod}:** `{rel}` (carried verbatim; see the note above)"))
        add(_code(ctx["embedded"][m], {**collapsed, "dimer": {"embedded_module": rel, "module_sha256": ctx["per_module_sha256"][rel]}}))

    manifest_literal = json.dumps(ctx["manifest"], indent=2, ensure_ascii=False)
    n_files = len(ctx["manifest"]["files"])
    extra_note = ""
    if ctx["extra_weights"]:
        extra_note = " The package also pins " + ", ".join(
            f"a second snapshot `{e['key']}` ({len(e['manifest']['files'])} files)" for e in ctx["extra_weights"]
        ) + ", carried and verified the same way."
    load_expr = template.get("model_load") or f"{template['pipeline_class']}.from_pretrained(weights_dir=WEIGHTS_DIR)"
    add(
        _md(
            "## 3. Pin, stage and verify the model\n\n"
            + infra
            + f"The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the module above and the `{n_files}`-file manifest below (paths, "
            "byte sizes, SHA-256) — and the cell first asserts they agree. It writes the manifest into the working-directory snapshot, then "
            f"`stage_missing_files(..., allow_download=True)` fetches exactly the entries that are absent from the Hugging Face Hub **at revision "
            f"`{ctx['MODEL_REVISION'][:12]}…`** (never `main`), `verify_snapshot` re-hashes every file and raises on the first size or digest mismatch, "
            f"and only then does `{load_expr}` load the verified files. There is no fallback to a different download and no remote model code is "
            f"executed.{extra_note} The effective identity, device and weight source are printed before any inference."
        )
    )
    ie = ctx["ident_expr"]
    model_code = (
        ("# @title Infrastructure: stage and digest-verify the pinned snapshot, then load the model\n" if labels else "")
        + "import json\n\n"
        f"MANIFEST = {manifest_literal}\n\n"
        f"if (MANIFEST['modelId'], MANIFEST['revision']) != ({ie['MODEL_ID']}, {ie['MODEL_REVISION']}):\n"
        "    raise RuntimeError('inline manifest does not name the identity carried by the pipeline module; the notebook was not regenerated after a change')\n"
        "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR\n"
        "WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "with open(WEIGHTS_DIR / MANIFEST_NAME, 'w', encoding='utf-8') as handle:\n"
        "    json.dump(MANIFEST, handle, indent=2)\n"
        f"print({{'model_id': {ie['MODEL_ID']}, 'revision': {ie['MODEL_REVISION']}, 'license': {ie['MODEL_LICENSE']}, 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes']}})\n"
        "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)\n"
        "print({'weights_dir': str(WEIGHTS_DIR), 'fetched': fetched})\n"
        "snapshot = verify_snapshot(WEIGHTS_DIR)\n"
        "_files = snapshot.get('files', []) if isinstance(snapshot, dict) else []\n"
        "print({'verified_files': len(_files) if isinstance(_files, list) else _files, 'revision': snapshot.get('revision', MODEL_REVISION) if isinstance(snapshot, dict) else MODEL_REVISION})\n"
    )
    for e in ctx["extra_weights"]:
        lit = json.dumps(e["manifest"], indent=2, ensure_ascii=False)
        id_const, rev_const = e["identity"]
        model_code += (
            f"\n{e['var']} = {lit}\n\n"
            f"if ({e['var']}['modelId'], {e['var']}['revision']) != ({id_const}, {rev_const}):\n"
            f"    raise RuntimeError('inline {e['key']} manifest does not name the identity carried by the pipeline module')\n"
            f"{e['dir']}.mkdir(parents=True, exist_ok=True)\n"
            f"with open({e['dir']} / MANIFEST_NAME, 'w', encoding='utf-8') as handle:\n"
            f"    json.dump({e['var']}, handle, indent=2)\n"
            f"fetched_{e['key'].replace('-', '_')} = {e['stage']}({e['dir']}, allow_download=True)\n"
            f"print({{'weights_dir': str({e['dir']}), 'fetched': fetched_{e['key'].replace('-', '_')}}})\n"
            f"_extra = {e['verify']}({e['dir']})\n"
            f"_extra_files = _extra.get('files', []) if isinstance(_extra, dict) else []\n"
            f"print({{'verified_files_{e['key'].replace('-', '_')}': len(_extra_files) if isinstance(_extra_files, list) else _extra_files}})\n"
        )
    model_code += (
        f"pipe = {load_expr}\n"
        "print({'device': getattr(pipe, 'device', None), 'source': getattr(pipe, 'source', 'local-snapshot')})"
    )
    add(_code(model_code, dict(collapsed)))

    for stage in template["cells"]:
        add(_md(stage["md"].format(**fmt)))
        if stage.get("code"):
            add(_code(stage["code"].format(**fmt)))
    add(_md(template["closing"].format(**fmt)))

    return {
        "cells": cells,
        "metadata": {
            "dimer": {
                "notebook_profile": template["profile"],
                "notebook_mode": mode,
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                **({"environment": "isolated hash-locked uv environment; nothing installed into the kernel"} if isolated else {}),
                "generated_from": {
                    "repository": template["repo_name"],
                    "revision": ctx["module_revision"],
                    "module": ctx["entry_rel"],
                    "modules": ctx["module_rels"],
                    "module_sha256": ctx["module_sha256"],
                    "generator": GENERATOR_VERSION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, default=None, help="default: <repo>/tools/notebook_template.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/tutorials/<notebook_name>")
    parser.add_argument("--check", action="store_true", help="exit 1 if the existing notebook differs (PAR3)")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    template = load_template(args.template or repo / "tools" / "notebook_template.py")
    out = args.out or repo / "tutorials" / template["notebook_name"]
    if args.check:
        # PAR3/PAR4: the recorded revision is a provenance label and is carried through the check;
        # drift is caught by content (module text, manifest, pins) — a changed module changes the
        # rendered cell and its SHA-256, so the byte comparison fails regardless of the label.
        rendered = to_bytes(render(repo, template, recorded_revision(out)))
        # Compare on LF: a Windows checkout with core.autocrlf rewrites the file to CRLF.
        current = out.read_bytes().replace(b"\r\n", b"\n") if out.exists() else b""
        if current != rendered:
            print(f"STALE: {out} differs from the generator output; run tools/build_notebook.py", file=sys.stderr)
            return 1
        print(f"OK: {out} is up to date ({len(rendered)} bytes)")
        return 0
    rendered = to_bytes(render(repo, template))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rendered)
    print(f"wrote {out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
