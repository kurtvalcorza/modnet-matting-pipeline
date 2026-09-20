"""Alpha-matte metrics (numpy only): MAD, MSE and SAD over whole mattes, MAD over the trimap's unknown band, and
the two constant baselines (all background, all foreground) scored on the same pixels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def matte_errors(predicted: Any, reference: Any, *, unknown: Any | None = None) -> dict[str, float]:
    """Errors of one predicted alpha matte against its reference (both (H, W) float in [0, 1]).

    `mad` and `mse` are means over every pixel; `sad` is the summed absolute difference divided by 1,000 (the
    matting-literature convention); `mad_unknown` is the MAD over the pixels of the trimap's unknown band when
    `unknown` (a boolean (H, W) mask) is given and non-empty, else None.
    """
    import numpy as np

    p = np.asarray(predicted, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    if p.shape != r.shape or p.ndim != 2:
        raise ValueError(f"predicted and reference mattes must share one (H, W) shape, got {p.shape} and {r.shape}")
    diff = np.abs(p - r)
    out = {
        "mad": float(diff.mean()),
        "mse": float(np.square(p - r).mean()),
        "sad": float(diff.sum() / 1000.0),
    }
    if unknown is not None:
        band = np.asarray(unknown, dtype=bool)
        if band.shape != p.shape:
            raise ValueError(f"unknown mask must have shape {p.shape}, got {band.shape}")
        out["mad_unknown"] = float(diff[band].mean()) if band.any() else None
    else:
        out["mad_unknown"] = None
    return out


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(sum(present) / len(present)) if present else None


def matting_metrics(
    predicted: Sequence[Any], reference: Sequence[Any], *, unknown: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Per-image errors averaged over a set of mattes (each image weighs the same, whatever its size)."""
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("predicted and reference must be non-empty sequences of equal length")
    if unknown is not None and len(unknown) != len(predicted):
        raise ValueError("unknown must have one mask per matte")
    rows = [
        matte_errors(p, r, unknown=None if unknown is None else unknown[i])
        for i, (p, r) in enumerate(zip(predicted, reference, strict=True))
    ]
    return {
        "n_images": len(rows),
        "mad": round(_mean_of([row["mad"] for row in rows]), 6),
        "mse": round(_mean_of([row["mse"] for row in rows]), 6),
        "sad": round(_mean_of([row["sad"] for row in rows]), 4),
        "mad_unknown": (lambda v: None if v is None else round(v, 6))(_mean_of([row["mad_unknown"] for row in rows])),
        "per_image": [{k: (None if v is None else round(v, 6)) for k, v in row.items()} for row in rows],
    }


def constant_baselines(reference: Sequence[Any], *, unknown: Sequence[Any] | None = None) -> dict[str, dict[str, Any]]:
    """The all-background (alpha 0) and all-foreground (alpha 1) mattes scored on the same references: what a
    model that never separates the subject from the scene would score."""
    import numpy as np

    out = {}
    for name, value in (("all_background", 0.0), ("all_foreground", 1.0)):
        constants = [np.full(np.asarray(r).shape, value, dtype=np.float32) for r in reference]
        metrics = matting_metrics(constants, reference, unknown=unknown)
        out[name] = {k: v for k, v in metrics.items() if k != "per_image"}
    return out
