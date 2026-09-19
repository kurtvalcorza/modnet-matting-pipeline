# MODNet Matting Pipeline

DIMER-oriented pipeline for **MODNet photographic portrait matting** (`ZHKKKe/MODNet`, `modnet_photographic_portrait_matting.ckpt` — the authors' trimap-free portrait matting model), pinned to an immutable Hugging Face revision of the byte-identical Hub mirror `XM5354/Modnet_models`. The repository exposes trimap-free alpha-matte prediction for RGB portraits of any size, MAD / MSE / SAD and unknown-band MAD against the constant baselines, a labelled-portrait contract with explicit ceilings, a bounded fine-tuning contract for the matting branches with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `XM5354/Modnet_models` (Hub mirror of the authors' Google Drive release; file id `1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz`)
- Revision: `71aca6d04ed0267b4b12bde776868f2b9fb1d06f`
- Source asset: `modnet_photographic_portrait_matting.ckpt` (26,255,603 bytes, SHA-256 `7c22235f…`, equal on Drive and on the mirror) — a legacy torch pickle, converted once to safetensors and never served (see below)
- Upstream code: `ZHKKKe/MODNet` at `28165a451e4610c9d77cfdf925a94610bb2810fb`, vendored as `src/modnet_matting_pipeline/modeling.py`
- Upstream weight license: Apache-2.0 (the upstream README covers "the code, models, and demos")
- Upstream task: trimap-free portrait matting — a MobileNetV2 low-resolution semantic branch, a high-resolution detail branch and a fusion branch that outputs the alpha matte (Ke et al., AAAI 2022)
- Runtime: `torch==2.14.0` + `numpy` + `pillow` + `safetensors` + `huggingface-hub` — **no Hub-hosted code, no torchvision, no served pickle**
- Repository adaptation: **E2E** (bounded fine-tuning of the matting branches — optionally the backbone — on labelled portrait/alpha pairs with the upstream supervised losses, with a portable safetensors adapter)

## Two things to know before you start

**The checkpoint is a pickle, and the pipeline converts it once.** The upstream file is a legacy `torch.save` file: five pickle streams followed by raw tensor bytes. `audit_pickle` parses exactly those streams with `pickletools` (no execution), refuses any global outside `collections.OrderedDict` / `torch._utils._rebuild_tensor_v2` / `torch.FloatStorage` / `torch.LongStorage`, and `convert_model` unpickles it once through `torch.load(weights_only=True)`, strips the DataParallel `module.` prefix, loads the tensors strictly into the vendored architecture and writes `modnet-photographic-portrait-matting.safetensors` (26,135,396 bytes, SHA-256 `0ec6d832…`), the only file the model is ever loaded from.

**The labelled portraits are drawings.** No portrait-matting dataset with per-pixel alpha mattes is both permissively licensed and free of personal-data concerns, so `sample_dataset()` renders 80 figures with exact mattes in code (hair strands and silhouette edges with fractional alpha), and four digest-pinned CC0 photographs are fetched for label-free inference. The frozen model's error on the drawings (MAD 0.079), the adapted model's error (0.004) and the model's behaviour on the photographs before and after adaptation are the evidence; none of it is a claim about matting quality on photographs.

## Quick start

```python
from modnet_matting_pipeline import ModNetMattingPipeline, fetch_portraits, sample_dataset

pipe = ModNetMattingPipeline.from_pretrained(allow_download=True)  # stages + verifies the snapshot, audits + converts the pickle once, loads safetensors
splits = sample_dataset()                                           # 48 / 12 / 20 rendered portraits with exact alpha mattes
print(pipe.evaluate(splits["test"])["model"])                       # frozen model: MAD, MSE, SAD, unknown-band MAD (baselines under ["baselines"])
pipe.adapt(splits["train"], splits["validation"], epochs=6)         # bounded fine-tuning of the matting branches, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["model"])                       # adapted model, same portraits
mattes = pipe.predict(fetch_portraits())["predictions"]            # four CC0 photographs: alpha (H, W) float32 in [0, 1] at the input size
pipe.save_artifact("outputs/adapter")
```

`predict()` takes records `{id, image}` with an RGB uint8 array (any size with both sides in [64, 4096]) or an image path; `evaluate()` and `adapt()` take `{id, image, alpha}` records at exactly 512 × 512 with an alpha in [0, 1] (or an 8-bit PNG path). Images are normalised and resized as the upstream inference script does; validation is structural: nothing checks that an image shows a person or that an alpha belongs to its image.

## Weights layout

```
weights/modnet-photographic-portrait-matting/   modnet_photographic_portrait_matting.ckpt  (git-ignored, the pinned source)
                                                dimer-base-manifest.json
                                                modnet-photographic-portrait-matting.safetensors  (git-ignored, converted)
weights/portraits/                              the four CC0 photographs, fetched by pinned URL + SHA-256 (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of the manifest entry and of the converted file when present), converts the pickle when the safetensors file is absent, and refuses on the first mismatch. With `require_source=False` the digest-verified converted file is accepted without the checkpoint — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance (Drive origin, Hub mirror, the second mirror), the audit, the conversion, the vendored code and the DIMER hosting notes.

## Sample data

`sample_dataset()` renders the labelled splits in code — deterministic in the seed, drawn at 1024 × 1024 and box-filtered to 512 × 512 — and `dataset_manifest` validates them, refuses a portrait in two splits and records a digest. `fetch_portraits()` downloads the four CC0 photographs (Pixabay uploads re-hosted on Wikimedia Commons, 9.3 MB) and refuses any on a size or SHA-256 mismatch. Nothing is committed under `weights/`.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the matting branches, about 17 MB; the whole model, about 26 MB, with `trainable="full"`) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digest, the adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `ModNetMattingPipeline.from_artifact(dir)` re-verifies the base file, checks the manifest, scope and digest before deserialising, rebuilds the model and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles in all three layouts (zip, plain, legacy multi-stream), temporary manifests, synthetic portraits, an injected photograph fetcher, a BYOD zip with a decoy member and a stub model with the aliased backbone, never the weights; `tests/test_model_backed.py` runs the real converted weights when they are staged (load, matte, one adaptation epoch, reload parity) and skips otherwise. The model-backed smoke is recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/modnet-matting-pipeline/blob/main/tutorials/modnet_matting_colab.ipynb)

`tutorials/modnet_matting_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 4 package modules (`metrics.py`, `modeling.py`, `pipeline.py`, `samples.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It runs on a CPU or a GPU. It downloads and converts the pinned checkpoint in the runtime (the audit and conversion records are printed before the model loads), renders the labelled portraits, fetches the pinned photographs, and runs the sample path: validation, the frozen model against the constant baselines and on the photographs, bounded fine-tuning, held-out evaluation, the photographs again, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the `E2E` notebook has executed top-to-bottom on the local pre-flight harness only (WSL, RTX 5070 Ti, weights pre-staged; 11/11 cells, 35.5 s); the clean-runtime Kaggle execution that promotes it is pending and will be recorded in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but are not the evidence; the hosted run is.

## Licensing

- Upstream weights: Apache-2.0 (`ZHKKKe/MODNet`: "the code, models, and demos … are released under the Apache License 2.0"), staged from the pinned Hub mirror whose bytes equal the authors' Google Drive release, and converted, not modified, into the served safetensors.
- Upstream code: Apache-2.0, vendored as `modeling.py` with the commit and file digests recorded in `docs/WEIGHTS.md`.
- Tutorial photographs: CC0 1.0 (Pixabay portraits re-hosted on Wikimedia Commons); fetched at run time, never committed. Labelled portraits: rendered in code, this repository's licence.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
