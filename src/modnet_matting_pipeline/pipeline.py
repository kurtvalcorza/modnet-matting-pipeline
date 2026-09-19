"""MODNet photographic portrait matting (`ZHKKKe/MODNet`, `modnet_photographic_portrait_matting.ckpt`) DIMER
pipeline: verified snapshot, static audit and one-time conversion of the legacy torch pickle into safetensors,
trimap-free alpha-matte prediction, held-out evaluation against constant baselines, and bounded fine-tuning of the
matting branches to labelled portrait/alpha pairs with a portable adapter.

MODNet (Ke et al., AAAI 2022) predicts a portrait's alpha matte from an RGB image alone — no trimap — with a
MobileNetV2 low-resolution semantic branch, a high-resolution detail branch and a fusion branch. The authors
publish the photographic checkpoint on Google Drive; the byte-identical file is mirrored on the Hugging Face Hub
(`XM5354/Modnet_models`), which is what this package pins at an immutable revision with the digest of the Drive
original recorded beside it (docs/WEIGHTS.md).

The asset is a **legacy torch.save file** (not a zip archive): five pickle streams — the magic number, the protocol
version, the system record, the `OrderedDict` state dict whose tensors are persistent-id references to
`torch.FloatStorage` / `torch.LongStorage`, and the storage keys — followed by raw storage bytes. Under the fleet
asset specification (§11) that is executable serialization, so this package converts it once — every header
stream statically audited against an allow-list, then `torch.load(weights_only=True)`, the DataParallel `module.`
prefix stripped, a strict load into the vendored architecture — into safetensors with a pinned digest, and serves
only the converted file. The architecture is vendored in `modeling.py` from the upstream repository at a pinned
commit; nothing is fetched from the Hub at load time except the manifest-listed file.

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input validation
run (and can refuse) before `torch` is imported (fleet RTM-001). `numpy` and Pillow are used for images and are
imported freely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "XM5354/Modnet_models"
MODEL_REVISION = "71aca6d04ed0267b4b12bde776868f2b9fb1d06f"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "modnet-photographic-portrait-matting"
ARTIFACT_FORMAT = "org.valcorza.modnet-matting.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (the authors' Google Drive file; the Hub mirror is byte-identical, docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "modnet_photographic_portrait_matting.ckpt"
SOURCE_CKPT_BYTES = 26_255_603
SOURCE_CKPT_SHA256 = "7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8"
SOURCE_DRIVE_FILE_ID = "1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz"  # in the authors' folder 1umYmlCulvIFNaqPjwod1SayFmSRHziyR
UPSTREAM_CODE_COMMIT = "28165a451e4610c9d77cfdf925a94610bb2810fb"  # ZHKKKe/MODNet, the vendored architecture
# Code-free serving file produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "modnet-photographic-portrait-matting.safetensors"
CONVERTED_SHA256 = "0ec6d832a873ea38974077b23920da806cc2e44e5553c7ee12e7cb7b921bcc6b"
CONVERTED_BYTES = 26_135_396
# Static-audit digest of the source pickle (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = "5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10"
CKPT_ALLOWED_GLOBALS = frozenset(
    {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage", "torch.LongStorage"}
)
LEGACY_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C  # torch's legacy serialization header
LEGACY_HEADER_STREAMS = 5  # magic, protocol version, sys_info, the object, the storage keys
STATE_DICT_PREFIX = "module."  # the checkpoint was saved from an nn.DataParallel wrapper
ALIAS_PREFIX = "lr_branch.backbone."  # the LR branch holds the same backbone module; these keys alias `backbone.`

# Architecture and data-contract facts.
HR_CHANNELS = 32
PARAMETER_COUNT = 6_487_075  # nn.Parameters of the vendored MODNet (shared backbone counted once)
STATE_TENSORS = 439  # tensors in the converted file (the state dict without the 312 aliased LR-branch backbone keys)
STATE_NUMEL = 6_521_976  # elements in the converted file (parameters + BatchNorm buffers)
CHECKPOINT_TENSORS = 751  # tensors in the source state dict (aliases included)
REF_SIZE = 512  # upstream inference resizes the short side to 512 (and both sides to multiples of 32)
TRAIN_SIZE = 512  # labelled records are exactly this square size
MIN_SIDE, MAX_SIDE = 64, 4_096  # inference images
MIN_RECORDS = 4
MAX_RECORDS = 2_000
ADAPTATION_MODES = ("branches", "full")  # the only scopes an adapter may declare
FROZEN_PREFIXES: dict[str, tuple[str, ...]] = {"branches": ("backbone.", ALIAS_PREFIX), "full": ()}
TRIMAP_RADIUS = 8  # pixels of unknown band grown around the alpha transition (at TRAIN_SIZE)
# Upstream's `GaussianBlurLayer(1, 3)` kernel: scipy.ndimage.gaussian_filter of a 3 × 3 delta with sigma 0.8
# (0.3 · ((3 − 1) · 0.5 − 1) + 0.8), reflect mode — hard-coded so scipy is not a dependency.
SEMANTIC_BLUR_KERNEL: tuple[tuple[float, ...], ...] = (
    (0.06261056416945447, 0.12499990229092141, 0.06261056416945447),
    (0.12499990229092141, 0.24955813415849687, 0.12499990229092141),
    (0.06261056416945447, 0.12499990229092141, 0.06261056416945447),
)
LOSS_SCALES = {"semantic": 10.0, "detail": 10.0, "matte": 1.0}  # upstream `supervised_training_iter` defaults


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_CKPT_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_CKPT_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_CKPT_NAME and (size, digest) != (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256):
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving file (safetensors) against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    file_path = root / CONVERTED_WEIGHTS_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"converted file missing: {file_path}")
    size = file_path.stat().st_size
    if size != CONVERTED_BYTES:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: size {size} != pinned {CONVERTED_BYTES}")
    digest = _sha256_file(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return {"files": [{"path": CONVERTED_WEIGHTS_NAME, "bytes": size, "sha256": digest}]}


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when the
    converted serving file is present, that against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = (root / CONVERTED_WEIGHTS_NAME).is_file()
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores the
    25 MB checkpoint and the safetensors it converts to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_stream_globals(stream: io.BytesIO) -> tuple[dict[str, int], Any]:
    """Globals of one pickle stream read from the current position (stops at STOP, leaving the stream after it),
    plus the value of a single LONG1/LONG/INT top-level constant (for the legacy magic number) when the stream is
    that simple. Collected with `pickletools.genops` — no execution."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    constant = None
    n_ops = 0
    for op, arg, _pos in pickletools.genops(stream):
        n_ops += 1
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("LONG1", "LONG4", "LONG", "INT", "BININT", "BININT1", "BININT2"):
            constant = arg
            stack.append(None)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
        if op.name == "STOP":
            break
    return found, (constant if n_ops <= 3 else None)


def _pickle_globals(data: bytes) -> dict[str, int]:
    found, _ = _pickle_stream_globals(io.BytesIO(data))
    return found


def _legacy_globals(data: bytes) -> tuple[dict[str, int], int]:
    """Globals of the five header streams of a legacy torch.save file (the raw storage bytes after them are not
    pickles and are not read)."""
    stream = io.BytesIO(data)
    found: dict[str, int] = {}
    for index in range(LEGACY_HEADER_STREAMS):
        part, constant = _pickle_stream_globals(stream)
        if index == 0 and constant != LEGACY_MAGIC_NUMBER:
            raise ValueError("not a legacy torch.save file: the first pickle stream is not the magic number")
        for key, count in part.items():
            found[key] = found.get(key, 0) + count
    return found, stream.tell()


def audit_pickle(path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, inside a torch zip archive, or the header streams of a legacy
    torch.save file) would import and refuse any outside `allowed`. Executes nothing. Returns the sorted globals
    and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    layout = "pickle"
    pickles = 1
    header_bytes = len(data)
    if data[:4] == b"PK\x03\x04":
        layout = "torch_zip"
        pickles = 0
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                pickles += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        _first, constant = _pickle_stream_globals(io.BytesIO(data))
        if constant == LEGACY_MAGIC_NUMBER:
            layout = "torch_legacy"
            pickles = LEGACY_HEADER_STREAMS
            found, header_bytes = _legacy_globals(data)
        else:
            found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "layout": layout,
        "pickles": pickles,
        "header_bytes": header_bytes,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_CKPT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_CKPT_BYTES:
        raise ValueError(f"{SOURCE_CKPT_NAME}: size {size} != pinned {SOURCE_CKPT_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_CKPT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: sha256 {digest} != pinned {SOURCE_CKPT_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}")
    return {"path": SOURCE_CKPT_NAME, "bytes": size, "sha256": digest, "audit": audit}


def build_model() -> Any:
    """Instantiate the vendored MODNet architecture (random initialisation; no download, no pickle)."""
    from .modeling import MODNet

    return MODNet(in_channels=3, hr_channels=HR_CHANNELS)


def _expand_aliases(state: Mapping[str, Any]) -> dict[str, Any]:
    """The model's state dict lists the shared backbone twice (`backbone.*` and `lr_branch.backbone.*`); the
    converted file stores it once and the aliases are re-created here."""
    out = dict(state)
    for key, value in state.items():
        if key.startswith("backbone."):
            out[ALIAS_PREFIX + key[len("backbone.") :]] = value
    return out


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned legacy checkpoint into safetensors, deterministically, after size, digest and
    static-audit checks: torch's weights-only unpickler, the `module.` prefix stripped, aliased backbone keys
    checked equal and dropped, a strict load into the vendored architecture, and the model's own state dict
    (backbone once) saved."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    payload = torch.load(root / SOURCE_CKPT_NAME, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(not isinstance(v, torch.Tensor) for v in payload.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to a state dict of tensors")
    if len(payload) != CHECKPOINT_TENSORS:
        raise ValueError(f"{SOURCE_CKPT_NAME}: {len(payload)} tensors, expected {CHECKPOINT_TENSORS}")
    stripped = {}
    for key, value in payload.items():
        if not key.startswith(STATE_DICT_PREFIX):
            raise ValueError(f"{SOURCE_CKPT_NAME}: unexpected state-dict key {key!r} outside {STATE_DICT_PREFIX!r}")
        stripped[key[len(STATE_DICT_PREFIX) :]] = value
    for key, value in stripped.items():
        if key.startswith(ALIAS_PREFIX):
            twin = "backbone." + key[len(ALIAS_PREFIX) :]
            if twin not in stripped or not torch.equal(stripped[twin], value):
                raise ValueError(f"{SOURCE_CKPT_NAME}: aliased key {key!r} differs from {twin!r}")
    model = build_model()
    model.load_state_dict(stripped, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith(ALIAS_PREFIX)}
    n_elements = sum(v.numel() for v in canonical.values())
    if len(canonical) != STATE_TENSORS or n_elements != STATE_NUMEL:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors / {n_elements} elements; expected {STATE_TENSORS} / {STATE_NUMEL}"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "tensors": len(payload),
            "prefix": STATE_DICT_PREFIX,
            "aliased_backbone_tensors": len(payload) - STATE_TENSORS,
        },
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# images, mattes and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": (
        "{id, image, alpha?}: image = (H, W, 3) uint8 RGB array or an image file path (JPEG/PNG; greyscale and RGBA are "
        "converted to RGB); alpha = (H, W) float in [0, 1] (or an 8-bit greyscale PNG path, scaled by 1/255), optional"
    ),
    "inference_size": (
        f"any image with both sides in [{MIN_SIDE}, {MAX_SIDE}]; resized as upstream (short side {REF_SIZE}, multiples of 32) "
        "and the matte returned at the input size"
    ),
    "training_size": f"labelled records must be exactly {TRAIN_SIZE} × {TRAIN_SIZE} (the BYOD loader resizes pairs)",
    "records": [MIN_RECORDS, MAX_RECORDS],
    "validation": (
        "record shape, dtype, size range, finiteness and alpha range only. Nothing checks that the image shows a "
        "portrait, that the alpha was drawn for that image, or that the alpha marks the person rather than something "
        "else -- any RGB image is matted without complaint"
    ),
}


def read_image(path: str | Path) -> Any:
    """Load an image file with Pillow as (H, W, 3) uint8 RGB."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_alpha(path: str | Path) -> Any:
    """Load an 8-bit greyscale alpha PNG as (H, W) float32 in [0, 1]."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _check_record(record: Any, index: int, *, training: bool) -> dict[str, Any]:
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image[/alpha]")
    for key in ("id", "image"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image = record["id"], record["image"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    if isinstance(image, str | Path):
        if not Path(image).is_file():
            raise ValueError(f"{label_name}: image file not found: {image}")
        image = read_image(image)
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: image must be an array") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{label_name}: image must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{label_name}: image must be uint8 RGB, got dtype {array.dtype}")
    h, w = array.shape[:2]
    if training:
        if (h, w) != (TRAIN_SIZE, TRAIN_SIZE):
            raise ValueError(f"{label_name}: a labelled record must be {TRAIN_SIZE} × {TRAIN_SIZE}, got {h} × {w}")
    elif not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise ValueError(f"{label_name}: image sides must be in [{MIN_SIDE}, {MAX_SIDE}], got {h} × {w}")
    item: dict[str, Any] = {"id": rid, "image": np.ascontiguousarray(array)}
    alpha = record.get("alpha")
    if alpha is not None:
        if isinstance(alpha, str | Path):
            if not Path(alpha).is_file():
                raise ValueError(f"{label_name}: alpha file not found: {alpha}")
            alpha = read_alpha(alpha)
        try:
            matte = np.asarray(alpha, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name}: alpha must be a numeric array") from exc
        if matte.shape != (h, w):
            raise ValueError(f"{label_name}: alpha must have shape {(h, w)}, got {matte.shape}")
        if not np.all(np.isfinite(matte)):
            raise ValueError(f"{label_name}: alpha contains non-finite values")
        if float(matte.min()) < 0.0 or float(matte.max()) > 1.0:
            raise ValueError(
                f"{label_name}: alpha values must lie in [0, 1], got [{float(matte.min()):.3f}, {float(matte.max()):.3f}]"
            )
        item["alpha"] = np.ascontiguousarray(matte)
    elif training:
        raise ValueError(f"{label_name} has no alpha; every record of a labelled dataset needs one")
    for key in ("title", "source", "license", "meta", "original_size", "loaded_size"):
        if key in record:
            item[key] = record[key]
    return item


def check_record(record: Mapping[str, Any], *, training: bool = False) -> dict[str, Any]:
    """Validate one record and return its normalised copy (uint8 RGB image, float32 alpha)."""
    return _check_record(record, 0, training=training)


def record_digest(record: Mapping[str, Any], *, training: bool = False) -> str:
    checked = _check_record(record, 0, training=training)
    digest = hashlib.sha256(checked["image"].tobytes())
    if "alpha" in checked:
        digest.update(checked["alpha"].tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]], *, training: bool = True) -> str:
    payload = [[r["id"], record_digest(r, training=training)] for r in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a portrait dataset; raises ValueError before any model import. Labelled datasets
    (`require_labels=True`) must be `TRAIN_SIZE` squares with an alpha per record."""

    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, alpha} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    alpha_sum = 0.0
    fractional = 0
    pixels = 0
    for index, record in enumerate(records):
        item = _check_record(record, index, training=require_labels)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if "alpha" in item:
            alpha_sum += float(item["alpha"].sum())
            fractional += int(((item["alpha"] > 0.02) & (item["alpha"] < 0.98)).sum())
            pixels += item["alpha"].size
        checked.append(item)
    labelled = sum("alpha" in r for r in checked)
    if require_labels and alpha_sum == 0.0:
        raise ValueError("every alpha is all background; nothing to learn or evaluate")
    sizes = sorted({(int(r["image"].shape[0]), int(r["image"].shape[1])) for r in checked})
    return {
        "records": checked,
        "n_records": len(checked),
        "n_labelled": labelled,
        "sizes": [list(s) for s in sizes],
        "foreground_fraction": round(alpha_sum / pixels, 4) if pixels else None,
        "fractional_alpha_fraction": round(fractional / pixels, 4) if pixels else None,
        "digest": dataset_digest(checked, training=require_labels),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record (inference contract); returns its id, size and, with an alpha, the foreground fraction."""
    item = _check_record(record, 0, training=False)
    report: dict[str, Any] = {"id": item["id"], "shape": tuple(int(x) for x in item["image"].shape), "has_alpha": "alpha" in item}
    if "alpha" in item:
        report["foreground_fraction"] = round(float(item["alpha"].mean()), 4)
    return report


def trimap_from_alpha(alpha: Any, *, radius: int = TRIMAP_RADIUS) -> Any:
    """Upstream-style trimap from a ground-truth matte: 1 = foreground, 0 = background, 0.5 = unknown, the unknown
    band being every pixel whose alpha is fractional grown by `radius` pixels (a max filter) — numpy only."""
    import numpy as np

    a = np.asarray(alpha, dtype=np.float32)
    unknown = (a > 0.02) & (a < 0.98)
    if radius > 0:
        padded = np.pad(unknown, radius, mode="constant", constant_values=False)
        grown = np.zeros_like(unknown)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    grown |= padded[radius + dy : radius + dy + a.shape[0], radius + dx : radius + dx + a.shape[1]]
        unknown = grown
    trimap = np.where(a >= 0.5, 1.0, 0.0).astype(np.float32)
    trimap[unknown] = 0.5
    return trimap


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _inference_size(h: int, w: int) -> tuple[int, int]:
    """Upstream `inference.py`: short side to REF_SIZE unless the image already straddles it; multiples of 32."""
    if max(h, w) < REF_SIZE or min(h, w) > REF_SIZE:
        if w >= h:
            rh, rw = REF_SIZE, int(w / h * REF_SIZE)
        else:
            rw, rh = REF_SIZE, int(h / w * REF_SIZE)
    else:
        rh, rw = h, w
    return max(32, rh - rh % 32), max(32, rw - rw % 32)


@dataclass
class ModNetMattingPipeline:
    """Trimap-free portrait matting and bounded fine-tuning on top of the verified MODNet checkpoint."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> ModNetMattingPipeline:
        """Verify, convert if needed, build the vendored architecture and strictly load. With `require_source=False`
        the checkpoint may be absent (the DIMER-hosted case) as long as the converted file verifies. `report`
        receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted file already present and digest-verified"})
            source = "converted from the manifest-verified source checkpoint"
        else:
            verify_converted(root)
            source = "converted file, pinned digest (source checkpoint not required)"
        import torch
        from safetensors.torch import load_file

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if chosen.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        model = build_model()
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        if len(state) != STATE_TENSORS:
            raise ValueError(f"converted file holds {len(state)} tensors, expected {STATE_TENSORS}")
        model.load_state_dict(_expand_aliases(state), strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(chosen)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, device=chosen, weights_dir=root, source=source)

    # ---- forward ---------------------------------------------------------------------------------------

    def _to_tensor(self, images: Any) -> Any:
        """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 in [-1, 1] on the device (upstream normalisation)."""
        import numpy as np
        import torch

        batch = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).permute(0, 3, 1, 2).float()
        return (batch / 255.0 - 0.5) / 0.5

    def _matte_one(self, image: Any) -> Any:
        """One (H, W, 3) uint8 image -> (H, W) float32 matte at the input size (upstream resize rules)."""
        import torch
        import torch.nn.functional as F

        h, w = int(image.shape[0]), int(image.shape[1])
        rh, rw = _inference_size(h, w)
        batch = self._to_tensor(image[None])
        if (rh, rw) != (h, w):
            batch = F.interpolate(batch, size=(rh, rw), mode="area")
        with torch.inference_mode():
            _, _, matte = self.model(batch, True)
            if (rh, rw) != (h, w):
                matte = F.interpolate(matte, size=(h, w), mode="area")
        return matte[0, 0].clamp(0.0, 1.0).float().cpu().numpy()

    # ---- inference -------------------------------------------------------------------------------------

    def predict(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Matte images one at a time: per record the alpha matte (H, W) float32 in [0, 1] at the input size, the
        resized model input size, and the foreground fraction (mean alpha). The matte is the model's sigmoid output,
        not a calibrated probability."""
        checked = validate_dataset(records, min_records=1, require_labels=False)["records"]
        started = time.perf_counter()
        predictions = []
        for record in checked:
            matte = self._matte_one(record["image"])
            h, w = record["image"].shape[:2]
            predictions.append(
                {
                    "id": record["id"],
                    "alpha": matte,
                    "input_size": [int(h), int(w)],
                    "model_size": list(_inference_size(int(h), int(w))),
                    "foreground_fraction": round(float(matte.mean()), 4),
                }
            )
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "output": "alpha matte in [0, 1] per pixel (sigmoid of the fusion branch; no threshold applied)",
            "predictions": predictions,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Matte errors on labelled records: MAD, MSE, SAD and the MAD over the trimap's unknown band, with the
        all-background and all-foreground constant mattes scored on the same pixels."""
        from .metrics import constant_baselines, matting_metrics

        checked = validate_dataset(records, min_records=1)["records"]
        started = time.perf_counter()
        result = self.predict(checked)
        predicted = [p["alpha"] for p in result["predictions"]]
        reference = [r["alpha"] for r in checked]
        unknown = [trimap_from_alpha(r["alpha"]) == 0.5 for r in checked]
        metrics = matting_metrics(predicted, reference, unknown=unknown)
        return {
            "n_records": len(checked),
            "metric": (
                "per-image alpha-matte MAD / MSE / SAD (÷1000) averaged over the records, and MAD over the trimap's unknown band"
            ),
            "model": {k: v for k, v in metrics.items() if k != "per_image"},
            "per_image": [{"id": r["id"], **row} for r, row in zip(checked, metrics["per_image"], strict=True)],
            "baselines": constant_baselines(reference, unknown=unknown),
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def _trainable(self, mode: str) -> list[str]:
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"trainable must be one of {ADAPTATION_MODES}")
        frozen = FROZEN_PREFIXES[mode]
        return sorted(
            name
            for name, _param in self.model.named_parameters()
            if not name.startswith(ALIAS_PREFIX) and not name.startswith(frozen)
        )

    def _losses(self, batch: Any, trimap: Any, gt_matte: Any) -> dict[str, Any]:
        """Upstream `supervised_training_iter` losses: semantic MSE against the blurred 1/16 matte, detail L1 in the
        unknown band, and matte L1 + compositional L1 (with the unknown band weighted 4×)."""
        import torch
        import torch.nn.functional as F

        pred_semantic, pred_detail, pred_matte = self.model(batch, False)
        boundaries = (trimap < 0.5) | (trimap > 0.5)
        gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode="bilinear")
        kernel = torch.tensor(SEMANTIC_BLUR_KERNEL, dtype=gt_semantic.dtype, device=gt_semantic.device)[None, None]
        gt_semantic = F.conv2d(F.pad(gt_semantic, (1, 1, 1, 1), mode="reflect"), kernel)
        semantic_loss = LOSS_SCALES["semantic"] * F.mse_loss(pred_semantic, gt_semantic)
        pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
        gt_detail = torch.where(boundaries, trimap, gt_matte)
        detail_loss = LOSS_SCALES["detail"] * F.l1_loss(pred_boundary_detail, gt_detail)
        pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
        matte_l1 = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
        matte_comp = F.l1_loss(batch * pred_matte, batch * gt_matte) + 4.0 * F.l1_loss(
            batch * pred_boundary_matte, batch * gt_matte
        )
        matte_loss = LOSS_SCALES["matte"] * (matte_l1 + matte_comp)
        return {
            "semantic": semantic_loss,
            "detail": detail_loss,
            "matte": matte_loss,
            "total": semantic_loss + detail_loss + matte_loss,
        }

    def _batch_tensors(self, records: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
        import numpy as np
        import torch

        images = np.stack([r["image"] for r in records])
        mattes = np.stack([r["alpha"] for r in records]).astype(np.float32)
        trimaps = np.stack([r["trimap"] if "trimap" in r else trimap_from_alpha(r["alpha"]) for r in records]).astype(np.float32)
        return (
            self._to_tensor(images),
            torch.from_numpy(trimaps).to(self.device)[:, None],
            torch.from_numpy(mattes).to(self.device)[:, None],
        )

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-4,
        batch_size: int = 4,
        trainable: str = "branches",
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the matting branches (`trainable="branches"`: everything except the MobileNetV2
        backbone; `"full"` unfreezes the backbone too) on labelled portrait/alpha pairs with the upstream supervised
        losses, Adam at a fixed learning rate, seeded horizontal flips, and BatchNorm statistics frozen. Epoch 0
        records the frozen model; the epoch with the lowest validation loss is kept."""
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        names = self._trainable(trainable)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        for record in train_checked + (val_checked or []):  # trimaps once per record, not once per step
            record["trimap"] = trimap_from_alpha(record["alpha"])
        import numpy as np
        import torch

        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.model
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
        rng = np.random.default_rng(seed)

        def val_loss() -> float | None:
            if val_checked is None:
                return None
            model.eval()
            losses = []
            with torch.inference_mode():
                for start in range(0, len(val_checked), batch_size):
                    batch, trimap, matte = self._batch_tensors(val_checked[start : start + batch_size])
                    losses.append(float(self._losses(batch, trimap, matte)["total"]))
            return sum(losses) / len(losses)

        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val_loss": val_loss(), "note": "frozen model"}
            if val_checked is not None:
                entry["val"] = self.evaluate(val_checked)["model"]
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            if progress:
                progress(entry)
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                model.freeze_norm()  # BatchNorm statistics stay frozen (upstream SOC practice): small batches would corrupt them
                order = rng.permutation(len(train_checked)).tolist()
                losses = []
                parts = {"semantic": 0.0, "detail": 0.0, "matte": 0.0}
                for start in range(0, len(order), batch_size):
                    records = [train_checked[i] for i in order[start : start + batch_size]]
                    if rng.random() < 0.5:
                        records = [
                            {**r, "image": r["image"][:, ::-1], "alpha": r["alpha"][:, ::-1], "trimap": r["trimap"][:, ::-1]}
                            for r in records
                        ]
                    batch, trimap, matte = self._batch_tensors(records)
                    with torch.enable_grad():
                        loss = self._losses(batch, trimap, matte)
                        optimiser.zero_grad(set_to_none=True)
                        loss["total"].backward()
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimiser.step()
                    losses.append(float(loss["total"].detach()))
                    for key in parts:
                        parts[key] += float(loss[key].detach())
                    n_steps += 1
                model.eval()
                entry = {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "train_loss_parts": {k: round(v / len(losses), 4) for k, v in parts.items()},
                    "val_loss": val_loss(),
                }
                if val_checked is not None:
                    entry["val"] = self.evaluate(val_checked)["model"]
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the model as it
            # was before adapt() (trained tensors restored), frozen, with no adapter attached.
            restore = dict(model.state_dict())
            restore.update(_expand_aliases(initial_state))
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(_expand_aliases(best_state))
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "losses": (
                "upstream supervised losses: semantic (×10), detail (×10), matte L1 + compositional (×1); "
                f"trimap grown {TRIMAP_RADIUS} px from the fractional alpha"
            ),
            "augmentation": "seeded horizontal flips",
            "batchnorm": "running statistics frozen (eval mode) during adaptation",
            "precision": "float32",
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "converted_sha256": CONVERTED_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
        """Static checks on an adapter manifest, before any model or weights work: format and version, the pinned
        base and converted digest, exactly one weights entry named `adapter.safetensors` inside the artifact
        directory, and an adaptation mode that is one of the declared scopes. Returns the weights path and mode."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records a different converted-base digest")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        mode = adapter.get("trainable") if isinstance(adapter, Mapping) else None
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"artifact adapter.trainable must be one of {ADAPTATION_MODES}")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path, mode

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the tensors the scope allows."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, mode = self.check_artifact_manifest(root, manifest)
        expected = self._trainable(mode)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(
                f"artifact tensor list does not match the {len(expected)} tensors that trainable={mode!r} may change"
            )
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update(_expand_aliases({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}))
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> ModNetMattingPipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
