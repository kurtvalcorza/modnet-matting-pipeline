# Weight provenance, the pickle audit, the conversion, the vendored architecture, the tutorial data and DIMER hosting

This repository pins **one** model snapshot with its own `dimer-base-manifest.json` and **one** upstream code commit (the vendored architecture). The checkpoint is a legacy torch pickle, which this pipeline audits and converts but never serves; the labelled portraits are rendered in code and no photograph is fetched.

## MODNet photographic portrait matting weights

- Upstream authors' release: `modnet_photographic_portrait_matting.ckpt` in the Google Drive folder linked from `ZHKKKe/MODNet` (`pretrained/README.md`): folder `1umYmlCulvIFNaqPjwod1SayFmSRHziyR`, file id `1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz`. Downloaded on 2026-09-20: 26,255,603 bytes, SHA-256 `7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8`. Google Drive offers no immutable revision and no credential-free API contract, so it is recorded as the origin, not used as the staging source.
- Pinned staging source (Hugging Face Hub mirror): `XM5354/Modnet_models` at the immutable revision `71aca6d04ed0267b4b12bde776868f2b9fb1d06f` (2023-04-16, "Delete README.md"; the file was uploaded in `b186f59a` the same day). Its LFS pointer's `oid sha256` for `modnet_photographic_portrait_matting.ckpt` equals the Drive digest above, and the downloaded bytes re-hash to it — the mirror is byte-identical to the authors' release. A second mirror, `frkesk/modnet` at `f69229190e22bdca5a028f19255e7fbd00f8da4d` (2022-12-28), carries the same LFS digest; it is not used. The mirror repository also holds `modnet_webcam_portrait_matting.ckpt` (SHA-256 `913b82b6…`), which is not staged.
- Source format: a **legacy `torch.save` file** (not a zip archive): five consecutive pickle streams — the magic number `0x1950a86a20f9469cfc6c`, the protocol version, the system record, the `collections.OrderedDict` state dict whose 751 tensors are persistent-id references to `torch.FloatStorage` / `torch.LongStorage`, and the list of storage keys (163,911 bytes in total) — followed by the raw storage bytes. The keys carry the `module.` prefix of an `nn.DataParallel` wrapper; 630 tensors are float32 and 121 are int64 `num_batches_tracked` counters.
- Upstream weight license: Apache-2.0 — the upstream README states that "the code, models, and demos in this repository … are released under the Apache License 2.0" (`ZHKKKe/MODNet` at `28165a45…`, *License*). The Hub mirror carries no licence metadata of its own (the second mirror is tagged `other`); the licence recorded here is the authors' statement about their models.
- Local layout: `weights/modnet-photographic-portrait-matting/` holds the single manifest entry (the checkpoint, git-ignored) with its byte size and SHA-256, plus the converted file described below. `verify_snapshot()` in `src/modnet_matting_pipeline/pipeline.py` checks the manifest entry, asserts the checkpoint's digest against the package constant, and checks the converted file against its pinned digest when present.

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` recognises three layouts — a torch zip archive (every `.pkl` member), a plain pickle, and the legacy torch layout (exactly the five header streams; the storage bytes after them are not pickles and are not read) — disassembles each stream with `pickletools.genops`, collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Layout | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|---|
| `modnet_photographic_portrait_matting.ckpt` | legacy torch (5 header streams) | `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` | exactly those four | `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256` (the digest covers the sorted set of global names, so it equals the digest of any torch state-dict checkpoint that names the same four); `convert_model()` refuses a file whose audit digest differs. Tests craft a torch zip archive carrying `os.system`, a plain pickle of a `complex` number, and legacy-layout files with `os.system` hidden in the fourth and in the fifth stream, and assert that the audit refuses each before anything is constructed; a legacy file whose first stream is not the magic number is treated as a plain pickle. A first draft of the audit, written for zip archives and single pickles, saw only the 15-byte magic-number stream of this file and reported zero globals — the multi-stream parser exists because of that.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict of exactly 751 tensors whose every key starts with `module.`;
- the prefix is stripped; the 312 keys under `lr_branch.backbone.` are checked equal, tensor by tensor, to their `backbone.` twins (MODNet registers the one MobileNetV2 backbone twice, under `backbone` and inside the low-resolution branch) and are dropped from the serving file, because safetensors refuses shared tensors;
- the 751 tensors are loaded with `strict=True` into `MODNet(in_channels=3, hr_channels=32)` from the vendored `modeling.py`; the model's own state dict, minus the aliased keys, is saved as safetensors. At every later load `_expand_aliases` re-creates the twin keys so the load is again `strict=True`.

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `modnet-photographic-portrait-matting.safetensors` | 26,135,396 | 439 (6,521,976 elements: 6,487,075 parameters + 34,901 BatchNorm buffer elements) | `0ec6d832a873ea38974077b23920da806cc2e44e5553c7ee12e7cb7b921bcc6b` | no (regenerated) |

The conversion is deterministic: the digest was reproduced by two consecutive conversions on the build machine and by the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count.

## The vendored architecture

`src/modnet_matting_pipeline/modeling.py` is the MODNet architecture from `ZHKKKe/MODNet` at commit `28165a451e4610c9d77cfdf925a94610bb2810fb` (Apache-2.0): `src/models/modnet.py` (SHA-256 `2f26f5f0915d28bd059a998462b53195dfd33d11e78bd536c632aa1bb8885004`), `src/models/backbones/mobilenetv2.py` (`e3cc8ad6a9933ba18a17a62d5f887c64e0721240871ea8b48742fb9a8a2c3199`) and `src/models/backbones/wrapper.py` (`41197be7eb96b8a60dc034b55d8c9340dd682a41441dcf2ce67238955dfa5607`). Parameter and buffer names are unchanged, so the checkpoint loads strictly. Deliberate differences: no `torch.load` of a backbone checkpoint (`backbone_pretrained` is gone; the matting checkpoint carries the backbone), no `exit()` calls, no `nn.DataParallel`, type hints and the `nn.ReLU6` / `IBNorm` / `SEBlock` / branch semantics exactly as upstream; `forward(img, inference)` keeps its signature. The supervised losses of `src/trainer.py` (`ced99145de65eb2d78fb7ab2c13f9f94070096475a5acec855adea2b9e629eab`) are re-implemented in `pipeline.py` (`_losses`): the 3 × 3 Gaussian of the semantic target is the kernel `scipy.ndimage.gaussian_filter` produces for a delta with sigma 0.8 (hard-coded to full precision so scipy is not a dependency), the trimap is derived from the reference alpha by growing the fractional band 8 pixels, and the loss scales are the upstream defaults (10 / 10 / 1). The self-supervised SOC adaptation of the paper is not vendored.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match against the vendored architecture (0 missing, 0 unexpected, 0 shape mismatches), the 312 alias tensors found equal, and sane mattes on the four CC0 photographs: foreground fractions 0.396 (the man with the pipe, whose hand and pipe are kept), 0.445 (the veiled portrait), 0.717 (the elderly woman) and 0.838 (the close-up), with the silhouettes and loose hair where a viewer would put them (`MODEL_CARD.md`, *Runtime*). Behaviour under the exact torch version the authors used was not measured; the upstream ONNX and TorchScript exports were not compared.

## Runtime facts

- The model is float32 as shipped and runs in float32 on CPU and CUDA alike (no autocast: the model is small, and float32 keeps the CPU and GPU numbers close); `predict` runs under `torch.inference_mode()` one image at a time and moves results to the CPU; `adapt` trains with gradients only on the selected tensors, with BatchNorm running statistics frozen (`freeze_norm`, as the upstream SOC code does).
- Inputs are normalised to [−1, 1] ((x / 255 − 0.5) / 0.5) and resized as the upstream inference script does: when an image is entirely below or entirely above 512 pixels its short side becomes 512, otherwise its size is kept; both sides are floored to multiples of 32 and the matte is resized back to the input size with area interpolation.
- Only `torch`, `numpy`, Pillow, `safetensors` and `huggingface_hub` are needed; no torchvision, no scipy, no Hub-hosted module.

## The tutorial data

- **Labelled portraits** are rendered in code (`samples.render_portrait`): a seeded numpy + Pillow renderer draws a head, ears, neck, shoulders, a hair cap, optional long hair and glasses, face features, and 40–120 thin hair strands with a partial-coverage brush, at 1024 × 1024, and box-filters everything to 512 × 512, so the alpha is fractional along every strand and edge (about 2.3 % of pixels; foreground fraction about 0.34). The default splits are 48 / 12 / 20 portraits from three disjoint seed ranges (0…, 1000…, 2000…), rendered in the runtime and never downloaded or committed. Why drawings: no portrait-matting dataset with per-pixel alpha mattes is both permissively licensed and free of personal-data concerns — P3M-10k, PPM-100 and AIM-500 are research-only.
- **No photograph is fetched.** The row first pinned four CC0 Pixabay portraits re-hosted on Wikimedia Commons (URL + byte size + SHA-256); the first clean-room Kaggle run failed with `HTTP 429 Too many requests` from `upload.wikimedia.org` on the third download (2026-09-20) — Commons throttles shared cloud runtimes — and the maintainer ruled the photographs out of the default path rather than re-host them. A real portrait enters only as the user's own file through `samples.load_photo` (JPEG/PNG, downscaled once to at most 1536 pixels on the long side, original and loaded sizes recorded) and the notebook's `USE_BYOD_PHOTO` gate, and never leaves the runtime.

## Files deliberately not staged

The mirror's `modnet_webcam_portrait_matting.ckpt` (the video/webcam checkpoint) is not listed in the manifest and not fetched. The upstream repository's demo scripts, ONNX and TorchScript exports, and the `pretrained/mobilenetv2_human_seg.ckpt` backbone checkpoint that `backbone_pretrained=True` would load are neither vendored nor executed.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `modnet-photographic-portrait-matting.safetensors` (26,135,396 bytes). **The `.ckpt` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class is the vendored `modeling.py`, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: a matte needs the 25 MB weights and one RGB image; on an RTX 5070 Ti laptop GPU four 1536 × ~1100 photographs took 0.12 s during the build, on a laptop CPU 0.66 s. An adapted profile needs the weights plus a 17 MB adapter (`branches`) or a 26 MB one (`full`).
- Provenance note for the profile: the authoritative release is the authors' Google Drive file; the Hub mirror is a third-party re-upload whose bytes were verified equal to it on 2026-09-20 (SHA-256 above). A profile should cite both.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
