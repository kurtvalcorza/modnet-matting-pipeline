"""Tutorial data for the MODNet matting pipeline: a seeded synthetic portrait generator with exact alpha mattes (the
labelled sample dataset), four digest-pinned CC0 portrait photographs for label-free inference, the BYOD zip
loader, and the writers that put a sample pair on disk in the BYOD shape.

Why synthetic labels: no portrait-matting dataset with per-pixel alpha mattes is both permissively licensed and free
of personal-data concerns (P3M-10k, PPM-100 and AIM-500 are research-only), so the labelled records are drawn
figures — head, neck, shoulders, a hair cap and dozens of thin hair strands with fractional coverage — composited
over generated backgrounds with an exact alpha. They are out of MODNet's photographic training domain on purpose:
the frozen model's error on them and the adapted model's error are the tutorial's paired comparison, and the four
photographs (CC0, Pixabay via Wikimedia Commons) show the frozen and adapted models on real portraits without a
label. Everything here uses numpy and Pillow only; no model library is imported.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SAMPLE_SIZE = 512  # the tutorial's training resolution (MODNet's reference size)
SUPERSAMPLE = 2  # the figures are drawn at 1024 × 1024 and box-filtered down, which is where fractional alpha comes from
SAMPLE_COUNTS = {"train": 48, "validation": 12, "test": 20}
SAMPLE_SEEDS = {"train": 0, "validation": 1_000, "test": 2_000}  # disjoint seed ranges; ids carry the seed
SAMPLE_LABEL_SOURCE = "in-code synthetic portraits with exact alpha mattes (seeded numpy + Pillow renderer)"
DEFAULT_PORTRAIT_DIR = Path(__file__).resolve().parents[2] / "weights" / "portraits"
PORTRAIT_LICENSE = "CC0 1.0 (Pixabay photographs re-hosted on Wikimedia Commons)"
PORTRAIT_USER_AGENT = "modnet-matting-pipeline/0.1 (DIMER fleet tutorial by kurtvalcorza; digest-pinned fetch of four CC0 files)"
# Four CC0 stock portraits, pinned by URL, byte size and SHA-256; a re-upload under the same name changes the bytes
# and is refused. They are inference-only (no matte exists for them).
PORTRAIT_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "bearded-man-pipe",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/49/Bearded_man_smoking_pipe-3013924.jpg",
        "bytes": 4_702_901,
        "sha256": "aca3b45787b17c66309eaa10483d7c29d5689fbd82e773dce0f6fa299f735203",
        "title": "Bearded man smoking pipe (Pixabay 3013924)",
        "width": 5500,
        "height": 3667,
    },
    {
        "id": "portrait-in-hijab",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Portrait_in_hijab%2C_3064633.jpg",
        "bytes": 1_241_106,
        "sha256": "4cf790351645a6300c5529450eaa2da0581e8ed38e1171e60374049472ad190f",
        "title": "Portrait in hijab (Pixabay 3064633)",
        "width": 4256,
        "height": 2832,
    },
    {
        "id": "close-up-old-woman",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Close-up_portrait_of_an_old_woman.jpg",
        "bytes": 1_607_542,
        "sha256": "03f80f34457f0f15c20251de8ca93fc9c97e379c6ebcee815ef64fe5fde03108",
        "title": "Close-up portrait of an old woman (Pixabay)",
        "width": 3008,
        "height": 2000,
    },
    {
        "id": "womans-close-portrait",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/70/Woman%27s_close_portrait%2C_3096664.jpg",
        "bytes": 1_718_396,
        "sha256": "eb77b3e953813c97978a70892d2646b88e21e227ae5bef9e6e81d310df67869b",
        "title": "Woman's close portrait (Pixabay 3096664)",
        "width": 3021,
        "height": 2351,
    },
)
PORTRAIT_MAX_SIDE = 1536  # the photographs are 3–5.5 k pixels wide; they are downscaled once on load (records say so)

# --------------------------------------------------------------------------------------------------
# synthetic portraits
# --------------------------------------------------------------------------------------------------

_SKIN = ((246, 219, 196), (226, 189, 158), (200, 152, 116), (168, 116, 84), (128, 80, 56), (92, 58, 40), (240, 205, 190))
_HAIR = (
    (28, 22, 20),
    (60, 40, 28),
    (110, 70, 40),
    (170, 120, 60),
    (210, 180, 120),
    (120, 120, 125),
    (230, 225, 220),
    (150, 40, 30),
)
_CLOTH = (
    (40, 60, 120),
    (150, 30, 40),
    (30, 110, 70),
    (230, 230, 225),
    (40, 40, 45),
    (200, 150, 40),
    (120, 60, 140),
    (90, 130, 180),
)
_BACK = (
    (200, 210, 225),
    (235, 225, 205),
    (120, 140, 160),
    (90, 100, 90),
    (220, 190, 170),
    (60, 65, 80),
    (170, 200, 180),
    (240, 240, 240),
)


def _jitter(rng: Any, colour: Sequence[int], spread: int = 18) -> tuple[int, int, int]:
    return tuple(int(min(255, max(0, c + rng.integers(-spread, spread + 1)))) for c in colour)  # type: ignore[return-value]


def _background(rng: Any, size: int) -> Any:
    """A gradient between two colours, a few soft blobs, optional wall/furniture rectangles, and grain."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    c1 = np.asarray(_jitter(rng, _BACK[int(rng.integers(len(_BACK)))]), dtype=np.float32)
    c2 = np.asarray(_jitter(rng, _BACK[int(rng.integers(len(_BACK)))]), dtype=np.float32)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    t = (0.6 * y + 0.4 * x) if rng.random() < 0.5 else y
    base = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    canvas = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for _ in range(int(rng.integers(0, 4))):  # furniture-like rectangles
        x0, y0 = rng.integers(0, size, 2)
        w, h = rng.integers(size // 8, size // 2, 2)
        draw.rectangle((int(x0), int(y0), int(x0 + w), int(y0 + h)), fill=_jitter(rng, _BACK[int(rng.integers(len(_BACK)))], 40))
    blobs = Image.new("RGB", (size, size), (0, 0, 0))
    bd = ImageDraw.Draw(blobs)
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.integers(0, size, 2)
        r = int(rng.integers(size // 6, size // 2))
        bd.ellipse(
            (int(cx - r), int(cy - r), int(cx + r), int(cy + r)), fill=_jitter(rng, _BACK[int(rng.integers(len(_BACK)))], 50)
        )
    blobs = blobs.filter(ImageFilter.GaussianBlur(size / 12))
    mixed = np.asarray(canvas, dtype=np.float32) * 0.65 + np.asarray(blobs, dtype=np.float32) * 0.35
    grain = rng.normal(0, 4.0, mixed.shape).astype(np.float32)
    return np.clip(mixed + grain, 0, 255)


def _figure(rng: Any, size: int) -> tuple[Any, Any, dict[str, Any]]:
    """Draw one portrait figure at `size` × `size`: returns the foreground RGB (float32), the alpha (float32 in
    [0, 1], still at drawing resolution) and a record of what was drawn."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    fg = Image.new("RGB", (size, size), (0, 0, 0))
    alpha = Image.new("L", (size, size), 0)
    fd, ad = ImageDraw.Draw(fg), ImageDraw.Draw(alpha)
    skin = _jitter(rng, _SKIN[int(rng.integers(len(_SKIN)))], 10)
    hair = _jitter(rng, _HAIR[int(rng.integers(len(_HAIR)))], 12)
    cloth = _jitter(rng, _CLOTH[int(rng.integers(len(_CLOTH)))], 20)
    cx = size * float(rng.uniform(0.38, 0.62))
    cy = size * float(rng.uniform(0.34, 0.48))
    rx = size * float(rng.uniform(0.12, 0.18))
    ry = rx * float(rng.uniform(1.2, 1.4))
    shoulder_y = cy + ry + size * float(rng.uniform(0.10, 0.18))
    shoulder_w = size * float(rng.uniform(0.28, 0.42))
    neck_w = rx * float(rng.uniform(0.55, 0.8))
    # torso: a rounded trapezoid from the shoulders to the bottom edge
    torso = [
        (cx - shoulder_w, size + 10),
        (cx - shoulder_w, shoulder_y + shoulder_w * 0.35),
        (cx - shoulder_w * 0.6, shoulder_y),
        (cx + shoulder_w * 0.6, shoulder_y),
        (cx + shoulder_w, shoulder_y + shoulder_w * 0.35),
        (cx + shoulder_w, size + 10),
    ]
    for canvas, fill in ((fd, cloth), (ad, 255)):
        canvas.polygon(torso, fill=fill)
        canvas.ellipse(
            (cx - shoulder_w, shoulder_y - shoulder_w * 0.1, cx - shoulder_w * 0.3, shoulder_y + shoulder_w * 0.6), fill=fill
        )
        canvas.ellipse(
            (cx + shoulder_w * 0.3, shoulder_y - shoulder_w * 0.1, cx + shoulder_w, shoulder_y + shoulder_w * 0.6), fill=fill
        )
    # collar / neckline in the skin colour, then the neck
    for canvas, fill in ((fd, skin), (ad, 255)):
        canvas.rectangle((cx - neck_w, cy + ry * 0.6, cx + neck_w, shoulder_y + neck_w * 0.4), fill=fill)
        canvas.ellipse((cx - neck_w * 1.3, shoulder_y - neck_w * 0.5, cx + neck_w * 1.3, shoulder_y + neck_w * 0.8), fill=fill)
    # ears, head
    ear = rx * 0.22
    for canvas, fill in ((fd, skin), (ad, 255)):
        canvas.ellipse((cx - rx - ear, cy - ear * 1.3, cx - rx + ear, cy + ear * 1.3), fill=fill)
        canvas.ellipse((cx + rx - ear, cy - ear * 1.3, cx + rx + ear, cy + ear * 1.3), fill=fill)
        canvas.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)
    # hair cap over the top of the head, optionally long hair falling beside the neck
    long_hair = rng.random() < 0.5
    cap_top = cy - ry * float(rng.uniform(1.05, 1.25))
    for canvas, fill in ((fd, hair), (ad, 255)):
        canvas.chord((cx - rx * 1.08, cap_top, cx + rx * 1.08, cy + ry * 0.35), 180, 360, fill=fill)
        if long_hair:
            canvas.polygon(
                [
                    (cx - rx * 1.05, cy - ry * 0.2),
                    (cx - rx * 1.25, shoulder_y + rx * 0.4),
                    (cx - rx * 0.9, shoulder_y + rx * 0.3),
                    (cx - rx * 0.95, cy),
                ],
                fill=fill,
            )
            canvas.polygon(
                [
                    (cx + rx * 1.05, cy - ry * 0.2),
                    (cx + rx * 1.25, shoulder_y + rx * 0.4),
                    (cx + rx * 0.9, shoulder_y + rx * 0.3),
                    (cx + rx * 0.95, cy),
                ],
                fill=fill,
            )
    # face features are drawn on the foreground only (they do not change the alpha)
    eye_y = cy - ry * 0.12
    for sx in (-1, 1):
        ex = cx + sx * rx * 0.42
        fd.ellipse((ex - rx * 0.16, eye_y - rx * 0.09, ex + rx * 0.16, eye_y + rx * 0.09), fill=(250, 250, 250))
        fd.ellipse((ex - rx * 0.07, eye_y - rx * 0.07, ex + rx * 0.07, eye_y + rx * 0.07), fill=_jitter(rng, (40, 30, 30), 20))
        fd.line((ex - rx * 0.2, eye_y - rx * 0.24, ex + rx * 0.2, eye_y - rx * 0.26), fill=hair, width=max(2, int(rx * 0.05)))
    fd.line((cx, cy - ry * 0.05, cx - rx * 0.08, cy + ry * 0.28), fill=_jitter(rng, skin, 30), width=max(2, int(rx * 0.04)))
    fd.arc(
        (cx - rx * 0.3, cy + ry * 0.35, cx + rx * 0.3, cy + ry * 0.62),
        10,
        170,
        fill=_jitter(rng, (150, 70, 80), 20),
        width=max(2, int(rx * 0.06)),
    )
    glasses = rng.random() < 0.3
    if glasses:
        for sx in (-1, 1):
            ex = cx + sx * rx * 0.42
            fd.rectangle(
                (ex - rx * 0.24, eye_y - rx * 0.16, ex + rx * 0.24, eye_y + rx * 0.16),
                outline=(30, 30, 30),
                width=max(2, int(rx * 0.04)),
            )
    # hair strands: thin lines from the hair boundary outward, with a partial-coverage brush, so the alpha at the
    # silhouette is fractional after the box filter — the part of a matte that matting exists for
    n_strands = int(rng.integers(40, 120))
    strand_sigma = float(rng.uniform(0.0, 1.2))
    for _ in range(n_strands):
        angle = float(rng.uniform(np.pi * 1.05, np.pi * 1.95))  # over the cap (upper half)
        if long_hair and rng.random() < 0.4:
            side = -1.0 if rng.random() < 0.5 else 1.0  # flyaway hair off the outer edge of the long-hair strips
            angle = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi)) + (np.pi if side < 0 else 0.0)
            x0, y0 = cx + side * rx * 1.2, float(rng.uniform(cy, shoulder_y + rx * 0.3))
        else:
            x0, y0 = cx + np.cos(angle) * rx * 1.05, cy + ry * 0.2 + np.sin(angle) * ry * 1.0
        length = size * float(rng.uniform(0.015, 0.08))
        bend = float(rng.uniform(-0.6, 0.6))
        pts = [(x0, y0)]
        for k in (0.5, 1.0):
            a = angle + bend * k
            pts.append((x0 + np.cos(a) * length * k, y0 + np.sin(a) * length * k))
        width = int(rng.integers(1, 4))
        coverage = int(255 * float(rng.uniform(0.35, 1.0)))
        fd.line(pts, fill=hair, width=width)
        ad.line(pts, fill=coverage, width=width)
    if strand_sigma > 0.2:
        alpha = alpha.filter(ImageFilter.GaussianBlur(strand_sigma))
    record = {
        "skin": skin,
        "hair": hair,
        "cloth": cloth,
        "head_center": (round(cx / size, 3), round(cy / size, 3)),
        "long_hair": long_hair,
        "glasses": glasses,
        "strands": n_strands,
    }
    return np.asarray(fg, dtype=np.float32), np.asarray(alpha, dtype=np.float32) / 255.0, record


def render_portrait(seed: int, *, size: int = SAMPLE_SIZE) -> dict[str, Any]:
    """One synthetic portrait: `{id, image (size, size, 3) uint8, alpha (size, size) float32 in [0, 1], meta}`,
    deterministic in `seed`. Drawn at `SUPERSAMPLE` × size and box-filtered down, so the alpha is fractional
    along every hair strand and silhouette edge; composited as alpha · foreground + (1 − alpha) · background."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    big = size * SUPERSAMPLE
    background = _background(rng, big)
    fg, alpha, meta = _figure(rng, big)
    composite = alpha[..., None] * fg + (1.0 - alpha[..., None]) * background
    image = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.BOX)
    matte = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.BOX)
    return {
        "id": f"portrait-{seed:05d}",
        "image": np.asarray(image, dtype=np.uint8),
        "alpha": np.asarray(matte, dtype=np.float32) / 255.0,
        "meta": {"seed": seed, **meta},
    }


def sample_dataset(*, counts: Mapping[str, int] | None = None, size: int = SAMPLE_SIZE) -> dict[str, list[dict[str, Any]]]:
    """The tutorial's labelled splits (48 / 12 / 20 by default), each drawn from its own seed range."""
    counts = dict(SAMPLE_COUNTS if counts is None else counts)
    out: dict[str, list[dict[str, Any]]] = {}
    for split, n in counts.items():
        if split not in SAMPLE_SEEDS:
            raise ValueError(f"unknown split {split!r}; expected {sorted(SAMPLE_SEEDS)}")
        if not isinstance(n, int) or not 1 <= n <= 500:
            raise ValueError(f"{split}: count must be an int in 1..500")
        base = SAMPLE_SEEDS[split]
        out[split] = [render_portrait(base + i, size=size) for i in range(n)]
    return out


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = 0, fractions: tuple[float, float] = (0.7, 0.15)
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of user records into train / validation / test (validation and test hold at least one
    record each when there are three or more records)."""
    import numpy as np

    if len(records) < 3:
        raise ValueError("at least three labelled records are needed to form train / validation / test splits")
    order = np.random.default_rng(seed).permutation(len(records)).tolist()
    n_train = max(1, int(round(len(records) * fractions[0])))
    n_val = max(1, int(round(len(records) * fractions[1])))
    if n_train + n_val >= len(records):
        n_train = len(records) - n_val - 1
    shuffled = [dict(records[i]) for i in order]
    return {"train": shuffled[:n_train], "validation": shuffled[n_train : n_train + n_val], "test": shuffled[n_train + n_val :]}


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Refuse a record (by id, and by image bytes) present in more than one split."""
    seen_ids: dict[str, str] = {}
    seen_digests: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            rid = str(record["id"])
            if rid in seen_ids and seen_ids[rid] != split:
                raise ValueError(f"record {rid!r} is in both {seen_ids[rid]!r} and {split!r}")
            seen_ids[rid] = split
            digest = hashlib.sha256(_image_bytes(record["image"])).hexdigest()
            if digest in seen_digests and seen_digests[digest] != split:
                raise ValueError(f"record {rid!r} duplicates an image in {seen_digests[digest]!r}")
            seen_digests[digest] = split
    return {"disjoint": True, "n_ids": len(seen_ids)}


def _image_bytes(image: Any) -> bytes:
    import numpy as np

    if isinstance(image, str | Path):
        return Path(image).read_bytes()
    return np.ascontiguousarray(np.asarray(image)).tobytes()


# --------------------------------------------------------------------------------------------------
# pinned photographs (inference only)
# --------------------------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_portraits(cache_dir: str | Path | None = None, *, fetcher: Any | None = None) -> list[dict[str, Any]]:
    """Fetch the four pinned CC0 photographs into `cache_dir` (default `weights/portraits/`, git-ignored), each
    refused on a byte-size or SHA-256 mismatch, and return inference records `{id, image, title, source}`. The
    photographs are downscaled on load to at most `PORTRAIT_MAX_SIDE` pixels on the long side (recorded per
    record) — MODNet resizes to 512 on the short side anyway, and the originals are 3–5.5 k pixels wide."""
    import numpy as np
    from PIL import Image

    root = Path(cache_dir) if cache_dir is not None else DEFAULT_PORTRAIT_DIR
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for pin in PORTRAIT_RECORDS:
        target = root / f"{pin['id']}.jpg"
        if not (
            target.is_file() and target.stat().st_size == pin["bytes"] and _sha256_bytes(target.read_bytes()) == pin["sha256"]
        ):
            data = (fetcher or _http_get)(pin["url"])
            if len(data) != pin["bytes"] or _sha256_bytes(data) != pin["sha256"]:
                raise ValueError(f"{pin['id']}: downloaded bytes do not match the pinned size/SHA-256; refusing")
            target.write_bytes(data)
        with Image.open(target) as im:
            im = im.convert("RGB")
            original = im.size
            if max(im.size) > PORTRAIT_MAX_SIDE:
                scale = PORTRAIT_MAX_SIDE / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
            image = np.asarray(im, dtype=np.uint8)
        records.append(
            {
                "id": pin["id"],
                "image": image,
                "title": pin["title"],
                "source": pin["url"],
                "license": PORTRAIT_LICENSE,
                "original_size": list(original),
                "loaded_size": [int(image.shape[1]), int(image.shape[0])],
            }
        )
    return records


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": PORTRAIT_USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - pinned https URL, digest-verified
        return response.read()


# --------------------------------------------------------------------------------------------------
# BYOD: zip of image / alpha pairs + pairs.csv
# --------------------------------------------------------------------------------------------------

BYOD_MAX_MEMBERS = 2_000
BYOD_MAX_BYTES = 2_000_000_000


def load_byod_dataset(path: str | Path, *, size: int = SAMPLE_SIZE) -> list[dict[str, Any]]:
    """Read a zip holding `pairs.csv` (columns `id`, `image`, `alpha`) beside RGB images and single-channel alpha
    PNGs (0 = background, 255 = foreground). Members are read through the archive API by the names the CSV lists
    (no `extractall`, no paths from the archive are written anywhere); every pair is resized to `size` × `size`
    (recorded in the record) so it fits the training contract."""
    import numpy as np
    from PIL import Image

    zip_path = Path(path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"BYOD zip not found: {zip_path}")
    if zip_path.stat().st_size > BYOD_MAX_BYTES:
        raise ValueError(f"BYOD zip larger than {BYOD_MAX_BYTES} bytes")
    with zipfile.ZipFile(zip_path) as archive:
        names = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if len(names) > BYOD_MAX_MEMBERS:
            raise ValueError(f"BYOD zip holds more than {BYOD_MAX_MEMBERS} members")
        csv_name = next((n for n in names if n.endswith("pairs.csv")), None)
        if csv_name is None:
            raise ValueError("BYOD zip must contain pairs.csv")
        prefix = csv_name[: -len("pairs.csv")]
        rows = list(csv.DictReader(io.StringIO(archive.read(csv_name).decode("utf-8"))))
        if not rows or not {"id", "image", "alpha"} <= set(rows[0]):
            raise ValueError("pairs.csv must have the columns id, image, alpha")
        records = []
        for row in rows:
            image_name, alpha_name = prefix + row["image"], prefix + row["alpha"]
            for name in (image_name, alpha_name):
                if name not in names:
                    raise ValueError(f"pairs.csv names a member that is not in the archive: {name}")
            with Image.open(io.BytesIO(archive.read(image_name))) as im:
                original = im.size
                image = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
            with Image.open(io.BytesIO(archive.read(alpha_name))) as am:
                alpha = np.asarray(am.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            records.append(
                {"id": str(row["id"]), "image": image, "alpha": alpha, "source": row["image"], "original_size": list(original)}
            )
    return records


def write_sample_pair(record: Mapping[str, Any], image_path: str | Path, alpha_path: str | Path) -> dict[str, Any]:
    """Write one record as the BYOD pair shape: an RGB PNG and an 8-bit alpha PNG."""
    import numpy as np
    from PIL import Image

    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(record["image"], dtype=np.uint8)).save(image_path)
    Image.fromarray(np.clip(np.asarray(record["alpha"], dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
        alpha_path
    )
    return {"image": str(image_path), "alpha": str(alpha_path), "id": record["id"]}


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """The `pairs.csv` a BYOD zip is expected to carry, listing the records by id (image/alpha names by convention)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "image", "alpha"])
        for record in records:
            writer.writerow([record["id"], f"{record['id']}.png", f"{record['id']}_alpha.png"])
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split with the pipeline's checker, refuse overlaps, and record per-split facts and a digest
    (the pipeline module is imported lazily so this module stays model-free)."""
    from .pipeline import validate_dataset

    reports = {name: validate_dataset(records, min_records=1) for name, records in splits.items()}
    disjoint = check_split_disjoint(splits)
    payload = json.dumps({name: report["digest"] for name, report in reports.items()}, sort_keys=True)
    return {
        "splits": {name: {k: v for k, v in report.items() if k != "records"} for name, report in reports.items()},
        "disjoint": disjoint["disjoint"],
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
