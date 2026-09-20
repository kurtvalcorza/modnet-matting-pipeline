# Release verification

`tutorials/modnet_matting_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`metrics.py`, `modeling.py`, `pipeline.py`, `samples.py`), each equal to its
  source after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions; the upstream code
  commit `28165a45…` and the second mirror's revision `f6922919…` are the only other 40-hex commits the documents
  may name;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ModNetMattingPipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the pickle audit and the
  conversion are printed before the model loads, `sample_dataset`, `load_byod_dataset`, `dataset_manifest`,
  `write_sample_pair`, `validate_dataset` with the refusal probes, `pipe.evaluate` on the frozen model and after
  adaptation with the assertions, `pipe.predict` on two held-out portraits before and after adaptation with their
  mattes and cut-outs written, `load_photo` behind the photograph gate, `pipe.adapt` with its explicit hyperparameters,
  `pipe.save_artifact`,
  `ModNetMattingPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `served_from_pickle: False`, `remote_code_executed: False`, the Drive file id, the upstream code commit, the
  labelled-data source and `photographs_downloaded: 0`), the expected `outputs/` paths, the learner-facing statements (the asset is a pickle
  unpickled once, the portraits are drawn, the default path downloads no image, the constant baselines, uncalibrated
  mattes, frozen BatchNorm, watch a real photograph after adaptation) and the two gated-off BYOD defaults; forbidden patterns (credential-in-URL,
  any `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `huggingface_hub` / `safetensors` / `urllib` / `zipfile` / `Unpickler` use, `MODNet(` construction or
  `torch.load(` / `pickle.load` **outside the carried module cells**, `trust_remote_code=True`, `pickle.load` or
  `torch.load(` without `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub model, skipped without torch),
`tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`,
`tests/test_model_backed.py` (skipped without the staged weights); crafted pickles in three layouts, temporary
manifests, synthetic portraits, a synthetic photograph through the loader, a BYOD zip with a decoy member, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab runtime, CPU or GPU (T4 or better) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU, sequential cell executor with a `google.colab` shim, pre-staged pins and weights | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/modnet-photographic-portrait-matting/` (the standalone path writes the manifest itself, stages the
   checkpoint from the Hub, audits and converts it and renders the labelled portraits, so the directory may not be
   seeded); the runtime needs about 100 MB of free disk;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 6`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 4`, `TRAINABLE = 'branches'`,
   `USE_BYOD_PHOTO = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `numpy==2.5.3`, `pillow==11.3.0`, `safetensors==0.8.0`,
   `huggingface-hub==1.32.0` (an interpreter restart after the install is expected where the runtime's preinstalled
   torch, numpy or Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (defining `ModNetMattingPipeline`, `MODNet`, `audit_pickle`, `convert_model`,
     `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`, `validate_inputs`,
     `validate_dataset`, `trimap_from_alpha`, `render_portrait`, `sample_dataset`, `load_photo`,
     `load_byod_dataset`, `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`, `matting_metrics`,
     `constant_baselines`) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the checkpoint fetched from `XM5354/Modnet_models` at the immutable revision and `verify_snapshot`
     reporting 1 verified file;
   - the model cell printing the **conversion record** with the static audit (layout `torch_legacy`, 5 pickles,
     globals `collections.OrderedDict` / `torch.FloatStorage` / `torch.LongStorage` / `torch._utils._rebuild_tensor_v2`,
     0 violations, audit digest `5b9f0ba0…`), the checkpoint record (751 tensors, prefix `module.`, 312 aliased backbone
     tensors) and the converted file (`0ec6d832…`, 26,135,396 bytes), then the load report with source "converted
     from the manifest-verified source checkpoint";
   - the dataset manifest with 48 / 12 / 20 portraits (foreground fractions about 0.34, fractional-alpha fractions
     about 0.02), the written sample pair and `outputs/modnet_matting_sample_pairs.csv`, and three refusals (alpha
     outside [0, 1], labelled record not 512 × 512, all-background alphas);
   - the constant baselines and the frozen model on the test portraits (on the sample: all-background MAD ≈ 0.34,
     all-foreground ≈ 0.66, frozen MAD ≈ 0.08, unknown-band MAD ≈ 0.085) and the validation portraits (MAD ≈ 0.064),
     and the frozen and reference mattes and cut-outs of the first two test portraits written;
   - `pipe.adapt` printing epoch 0 as the frozen model, 4,263,203 trainable of 6,487,075 parameters, 72 steps,
     frozen BatchNorm statistics, and a six-epoch history with validation loss ≈ 0.80 → ≈ 0.06 at the kept epoch;
   - `pipe.evaluate` on the test portraits with the four-way comparison and
     `outputs/modnet_matting_evaluation_report.json` written (the cell asserts the kept epoch's validation loss is no
     higher than the frozen model's, that the validation MAD matches the history within 0.001, and that the adapted
     test MAD is below the frozen one — on the sample ≈ 0.004 against ≈ 0.08);
   - the two portraits matted again by the adapted model with their per-portrait MAD before and after printed, the
     photograph gate skipped (`own_photograph: null`), and `outputs/modnet_matting_predictions.json` written;
   - `pipe.save_artifact` writing `outputs/modnet_matting_adapter/{adapter.safetensors,manifest.json}`
     (about 17 MB), and `ModNetMattingPipeline.from_artifact` reloading it with held-out metrics and mattes matching
     the adapted model (the cell asserts a MAD difference below 10⁻⁴ and a maximum matte difference below 10⁻³);
   - `outputs/modnet_matting_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the provenance block
     (`served_from_pickle: false`, `remote_code_executed: false`, the Drive file id, the upstream code commit, the
     audit digest, the converted digest, the labelled-data source and `photographs_downloaded: 0`), the runtime
     versions, the comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the photograph cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `modnet_matting_colab.ipynb` | generated, pre-commit | 2026-09-20 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti, `google.colab` shim, pins pre-installed) | PASS — pre-flight of the first blob, which still fetched four Commons photographs; pre-flight only, **not** promotion evidence |
| `modnet_matting_colab.ipynb` | generated, pre-commit (photograph-free) | 2026-09-20 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/modnet_matting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/modnet_matting_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | generated, pre-commit (first blob, with the Commons photographs) | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti, `pillow 11.3.0`, `numpy 2.5.3`) | Default sample path of the first blob (stage → verify → load the already-converted file → render 80 portraits → validate → refusal probes → four photographs → baselines + frozen evaluation → branches adapt → evaluate → photographs again → export → reload); the checkpoint, the converted safetensors and the photographs were pre-staged | 35.5 s | **PASSED** — 11/11 code cells; probes refused; test MAD 0.0795 (frozen) → 0.0042 (adapted), MSE 0.0750 → 0.0015, SAD 20.84 → 1.09, unknown-band MAD 0.0855 → 0.0249 (all-background baseline 0.341 / all-foreground 0.659); validation loss 0.8008 → 0.0592 (best epoch 4), validation MAD 0.0640 → 0.0040; adaptation 13.4 s / 72 steps; photograph drift (mean abs matte difference frozen vs adapted) 0.052 / 0.004 / 0.016 / 0.006; adapter 17,060,652 bytes; reload parity identical (mad_diff 0.0, max_abs_matte_diff 0.0) |
| 2026-09-20 | generated, pre-commit (photograph-free) | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti, `pillow 11.3.0`, `numpy 2.5.3`) | Default sample path (stage → verify → load the already-converted file → render 80 portraits → validate → refusal probes → baselines + frozen evaluation with written mattes → branches adapt → evaluate → adapted mattes, photograph gate off → export → reload); the checkpoint and the converted safetensors were pre-staged, so `stage_missing_files` fetched 0 of 1 entries | 30.8 s | **PASSED** — 11/11 code cells; probes refused; test MAD 0.0795 (frozen) → 0.0042 (adapted), unknown-band MAD 0.0855 → 0.0249; validation loss 0.8008 → 0.0601 (best epoch 4), validation MAD 0.0640 → 0.0041; adaptation 72 steps; reload parity identical |
| 2026-09-20 | `dd05b34` / `99d5ff30` (first blob) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-modnet-matting` v1, image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda`) | Default sample path of the first blob from a fresh interpreter with an empty Hugging Face cache and no repository checkout; install, audit, conversion, load and the 80 rendered portraits succeeded | 193.5 s | **FAILED** — 6/11 code cells; `HTTP Error 429: Too many requests` from `upload.wikimedia.org` on the third of four photograph downloads (Wikimedia Commons throttles shared cloud runtimes). Finding, not a model defect: the photographs were removed from the default path (this blob) rather than re-hosted |

## Current status

**Candidate.** The notebook has run top-to-bottom on the local pre-flight harness only. The clean-runtime execution of
the committed notebook blob (REL1/REL10) is pending; when it is recorded here the registry moves to
**Release-grade**. Any later change to the carried modules or to the notebook produces a new blob, and the registry
returns to **Candidate** until a clean run of that blob is recorded here.
