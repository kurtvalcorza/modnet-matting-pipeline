"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py, modeling.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E portrait-matting workflow: the pinned MODNet checkpoint (a legacy torch pickle)
is digest-verified, statically audited and converted once into safetensors, 80 synthetic portraits with exact alpha
mattes are rendered in the kernel and four digest-pinned CC0 photographs are fetched, the frozen model is scored
against the constant baselines and shown on the photographs, a bounded fine-tuning of the matting branches runs in
the kernel, the held-out portraits are scored again, the photographs are matted again, and the adapter is exported
and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "modnet-matting-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/modnet_matting_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-XM5354%2FModnet__models-ffcc4d?style=flat",
        "https://huggingface.co/XM5354/Modnet_models",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ZHKKKe%2FMODNet-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ZHKKKe/MODNet",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2011.11961-b31b1b.svg", "https://arxiv.org/abs/2011.11961"),
]

TEMPLATE = {
    "package": "modnet_matting_pipeline",
    "repo_name": REPO,
    "stem": "modnet_matting",
    "notebook_name": "modnet_matting_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh runtime (CPU or GPU) installs the pinned dependencies (torch, numpy, Pillow, safetensors, "
        "huggingface-hub), stages and digest-verifies the pinned MODNet checkpoint (25 MB) from the Hub, statically audits the legacy "
        "torch pickle against an allow-list, converts it once into safetensors with a pinned digest, builds the vendored architecture and "
        "loads it strictly, renders 80 synthetic portraits with exact alpha mattes in the kernel (48 training, 12 validation, 20 test), "
        "fetches four digest-pinned CC0 photographs (9.3 MB, no credential), scores the frozen model against the all-background and "
        "all-foreground baselines and mattes the photographs, runs a bounded fine-tuning of the matting branches, scores the same held-out "
        "portraits again, mattes the photographs again, exports the adapter as safetensors with a manifest, and reloads that artifact into "
        "a fresh pipeline to verify prediction parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a T4 the model time is under a minute; on a "
        "CPU the adaptation takes a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own labelled "
        "portraits as a zip holding `pairs.csv` (columns `id`, `image`, `alpha`) beside RGB images and 8-bit greyscale alpha PNGs "
        "(0 = background, 255 = subject); at least four pairs. Pairs are resized to 512 × 512, split by seed into training, validation "
        "and test sets and flow through the same contract — validation, frozen baseline, adaptation, held-out evaluation, inference, "
        "artifact export and reload parity. The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites "
        "and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "ModNetMattingPipeline",
    "model_load": "ModNetMattingPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "modnet-photographic-portrait-matting",
    "modules": ["pipeline.py", "samples.py", "metrics.py", "modeling.py"],
    "entry_module": "pipeline.py",
    # ST2: both `__file__`-relative directories become working-directory-relative in the standalone kernel.
    "rewrites": [
        [
            r"^DEFAULT_WEIGHTS_DIR = Path\(__file__\)[^\n]*$",
            'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory-relative',
        ],
        [
            r"^DEFAULT_PORTRAIT_DIR = Path\(__file__\)[^\n]*$",
            'DEFAULT_PORTRAIT_DIR = Path.cwd() / "weights" / "portraits"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ],
    ],
    "runtime_imports": ["torch", "PIL"],
    "title": "MODNet portrait matting — DIMER E2E matting fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "trimap-free portrait alpha matting with MODNet (MobileNetV2 semantic branch, detail branch, fusion branch), held-out MAD / MSE / SAD against constant baselines, and bounded fine-tuning of the matting branches to labelled portrait/alpha pairs",
    "intro": (
        "MODNet (Ke et al., AAAI 2022) predicts a portrait's alpha matte from an RGB image alone — no trimap — with three branches trained "
        "together: a MobileNetV2 low-resolution branch for the coarse semantics, a high-resolution branch for the boundary detail, and a "
        "fusion branch that produces the matte. The photographic checkpoint packaged here is the authors' release, published on Google "
        "Drive; the byte-identical file is mirrored on the Hugging Face Hub, which is what Section 3 pins at an immutable revision, with "
        "the Drive digest recorded beside it.\n\n"
        "Two things about this row are handled in the open. **The upstream asset is a pickle** — a legacy `torch.save` file made of five "
        "pickle streams followed by raw tensor bytes. Section 3 downloads and digest-verifies it, statically lists every global those "
        "streams would import (a state dict of tensors and nothing else), refuses anything outside that allow-list, unpickles it exactly "
        "once through torch's weights-only loader, and writes a safetensors file whose digest is pinned in the carried module; the model "
        "you run is the architecture vendored in the carried `modeling.py` and loads that file strictly. **The labelled portraits are "
        "drawn, not photographed**: no portrait-matting dataset with per-pixel alpha mattes is both permissively licensed and free of "
        "personal-data concerns, so Section 4 renders figures — head, shoulders, a hair cap and dozens of hair strands with fractional "
        "coverage — over generated backgrounds with an exact alpha. They are out of the photographic training domain on purpose: the "
        "frozen model's error on them, the adapted model's error, and the model's behaviour on four real CC0 photographs before and after "
        "adaptation are the tutorial's evidence; the point of the contract is the same recipe on *your* labelled portraits."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset, metrics and model modules; stage and digest-verify a legacy "
        "pickled checkpoint, read its static audit and see it converted into safetensors; render labelled portraits with exact mattes and "
        "validate them with refusal probes; read MAD, MSE, SAD and the unknown-band MAD against constant baselines; run a bounded "
        "fine-tuning with the upstream semantic / detail / matte losses, explicit hyperparameters and frozen BatchNorm statistics; compare "
        "the adapted and frozen models on the same held-out portraits and on real photographs; and export a safetensors adapter that "
        "reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "video matting, the self-supervised SOC adaptation of the paper, trimap-based matting, background replacement quality beyond a "
        "simple composite, the published PPM-100 benchmark scores, face detection or recognition, and any claim that 20 drawn portraits "
        "stand in for an evaluation on photographs. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime — Google Colab (CPU or T4), Kaggle, or a Jupyter kernel with Python 3.12. The model has 6.5 M parameters: a 512 × 512 matte takes well under a second on a CPU, the default adaptation about 2 s per epoch on a T4 and about half a minute per epoch on a laptop CPU. About 200 MB of disk is needed for the checkpoint, its conversion and the photographs.",
        "- **Knowledge:** what an alpha matte is (per-pixel opacity of the subject, fractional along hair and soft edges), what a trimap's unknown band is, and how MAD / MSE / SAD are read against a constant baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a legacy torch pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; the architecture is carried verbatim from the repository (`modeling.py`, vendored from the upstream repository at a pinned commit).",
        "- **Data contract:** a record is `{{id, image, alpha}}` — an RGB uint8 image (any size with both sides in [64, 4096] for inference; exactly 512 × 512 for labelled records) and an alpha in [0, 1] of the same size. Validation is structural: nothing checks that the image shows a person, that the alpha belongs to the image, or that the alpha marks the subject rather than something else.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people you have no consent to process are exactly that. The default path uploads nothing; its four photographs are CC0 stock portraits.",
        "- **External access (data):** besides the model snapshot, the default path fetches four pinned objects over HTTPS — CC0 portrait photographs from Wikimedia Commons (Pixabay uploads; 9.3 MB in total), each refused on a size or SHA-256 mismatch. The labelled portraits are rendered in the kernel and need no download.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample portraits, validation and roles\n\n"
                "The default labelled dataset is rendered here, in the kernel, by `sample_dataset`: 48 training, 12 validation and 20 test "
                "portraits from three disjoint seed ranges, each drawn at 1024 × 1024 and box-filtered to 512 × 512 so that every hair "
                "strand and silhouette edge carries fractional alpha, composited as alpha · figure + (1 − alpha) · background. "
                "`dataset_manifest` validates every split with the same checker the model path uses, refuses a portrait present in two "
                "splits and records a digest; `fetch_portraits` downloads the four CC0 photographs (each refused on a size or digest "
                "mismatch) and downscales them once for the record.\n\n"
                "Look for: 48 / 12 / 20 records with foreground fractions around 0.34 and 2 % fractional-alpha pixels, a written "
                "sample pair (`outputs/{stem}_sample_portrait.png` + `_sample_alpha.png`, the BYOD shape), the four photographs with "
                "their original and loaded sizes, and three refusal probes — an alpha outside [0, 1], a labelled record that is not "
                "512 × 512, and a dataset whose alphas are all background — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = sample_dataset()\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    print({{name: {{'foreground_fraction': part['foreground_fraction'], 'fractional_alpha_fraction': part['fractional_alpha_fraction'], 'sizes': part['sizes']}}}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_portrait.png', 'outputs/{stem}_sample_alpha.png')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n\n"
                "portraits = fetch_portraits(cache_dir='weights/portraits')\n"
                "for record in portraits:\n"
                "    print({{'photograph': record['id'], 'title': record['title'], 'original_size': record['original_size'], 'loaded_size': record['loaded_size'], 'license': record['license']}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'alpha outside [0, 1]': [{{**test_records[0], 'alpha': test_records[0]['alpha'] * 1.5}}, *test_records[1:4]],\n"
                "    'labelled record not 512 x 512': [{{**test_records[0], 'image': test_records[0]['image'][:256], 'alpha': test_records[0]['alpha'][:256]}}, *test_records[1:4]],\n"
                "    'all-background alphas': [{{**r, 'alpha': np.zeros_like(r['alpha'])}} for r in test_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. The frozen model: constant baselines, held-out errors and the photographs\n\n"
                "`pipe.predict` normalises each image to [−1, 1], resizes it as the upstream inference script does (short side 512, both "
                "sides multiples of 32), runs the three branches and returns the fusion branch's sigmoid output as the matte at the input "
                "size — the model's output, not a calibrated probability. `pipe.evaluate` scores labelled records per image and averages: "
                "MAD and MSE over every pixel, SAD (the summed absolute difference ÷ 1000), and the MAD over the trimap's unknown band "
                "(every fractional-alpha pixel grown by 8 pixels — the part of a matte that matting exists for); the two **constant "
                "baselines** — every pixel background, every pixel subject — are scored on the same references, so all-background MAD "
                "equals the foreground fraction.\n\n"
                "Look for: a frozen test MAD near 0.08 (in the build record 0.079, against 0.341 for all-background and 0.659 for "
                "all-foreground) — the photographic model finds the drawn heads but drops parts of the drawn clothing and misses strands — "
                "and, on the four photographs, mattes whose foreground fractions run from about 0.40 (the man with the pipe against a "
                "dark background) to 0.84 (a face filling the frame); the frozen mattes are written to `outputs/` as PNGs beside a cut-out "
                "on white. These are sample-sanity numbers on 20 and 12 drawn portraits, not a benchmark."
            ),
            "code": (
                "import time\n\n"
                "from PIL import Image\n\n"
                "def write_matte(record, alpha, tag):\n"
                "    matte = Image.fromarray(np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))\n"
                "    matte.save(f'outputs/{stem}_matte_' + tag + '_' + record['id'] + '.png')\n"
                "    cutout = (alpha[..., None] * record['image'].astype(np.float32) + (1.0 - alpha[..., None]) * 255.0)\n"
                "    Image.fromarray(np.clip(cutout + 0.5, 0, 255).astype(np.uint8)).save(f'outputs/{stem}_cutout_' + tag + '_' + record['id'] + '.png')\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baselines_test': {{k: {{m: v[m] for m in ('mad', 'mse', 'sad', 'mad_unknown')}} for k, v in frozen_test['baselines'].items()}}}})\n"
                "print({{'frozen_test': frozen_test['model']}})\n"
                "print({{'frozen_validation': frozen_val['model']}})\n"
                "for row in frozen_test['per_image'][:5]:\n"
                "    print({{'portrait': row['id'], 'mad': row['mad'], 'mad_unknown': row['mad_unknown']}})\n"
                "frozen_photos = pipe.predict(portraits)\n"
                "for record, pred in zip(portraits, frozen_photos['predictions']):\n"
                "    write_matte(record, pred['alpha'], 'frozen')\n"
                "    print({{'photograph': record['id'], 'model_size': pred['model_size'], 'foreground_fraction': pred['foreground_fraction'], 'note': 'no label; sanity check'}})\n"
                "print({{'output': frozen_photos['output'], 'alpha_shape': frozen_photos['predictions'][0]['alpha'].shape, 'seconds': frozen_photos['seconds']}})"
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the matting branches\n\n"
                "`pipe.adapt` trains the low-resolution branch's SE block and convolutions, the high-resolution branch and the fusion "
                "branch (4.26 M parameters — 66 % of the model) and nothing else: the MobileNetV2 backbone is frozen (no gradient is "
                "stored for it), and every BatchNorm layer keeps its running statistics, as the upstream adaptation code does. Each step "
                "takes four portraits with a seeded horizontal flip, derives the trimap from the reference alpha, and minimises the "
                "upstream supervised objective — the semantic loss (MSE against the blurred 1/16 matte, ×10), the detail loss (L1 in the "
                "unknown band, ×10) and the matte loss (L1 plus a compositional L1 with the unknown band weighted 4×) — with Adam at a "
                "small fixed learning rate and gradient-norm clipping. Epoch 0 records the frozen model's validation loss and metrics; the "
                "epoch with the lowest validation loss is kept.\n\n"
                "Watch the validation loss: in the build record it fell from 0.80 to about 0.06 by epoch 4 and drifted afterwards, and "
                "the validation MAD from 0.064 to about 0.004. Six epochs (72 steps) take about 15 s on a T4 and about 3 minutes on a "
                "laptop CPU. `TRAINABLE = 'full'` also unfreezes the backbone (6.49 M parameters)."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE = 'branches'  # @param [\"branches\", \"full\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'train_loss_parts' in entry:\n"
                "        row['parts'] = entry['train_loss_parts']\n"
                "    if 'val' in entry:\n"
                "        row['val_mad'] = entry['val']['mad']\n"
                "        row['val_mad_unknown'] = entry['val']['mad_unknown']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'losses': adapt_result['losses'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test portraits were never used for training or epoch selection (they come from their own seed range). The adapted "
                "model is scored exactly as the frozen model was in Section 5, and the table puts the baselines, the frozen and the "
                "adapted numbers side by side. The cell asserts what the procedure guarantees — the kept epoch's validation loss is no "
                "higher than the frozen model's, and re-scoring the validation portraits reproduces the kept epoch's MAD within 0.001 — "
                "and it also asserts that the adapted test MAD is below the frozen one: on drawn portraits the domain shift is large "
                "enough that the build record moved the test MAD from 0.079 to 0.004 (unknown-band MAD 0.085 → 0.025) on every "
                "hyperparameter probe, so a failure here is a finding, not noise. The size of the gain, and its reading — the model "
                "learned the drawing style, on 48 portraits, with one seed and no dispersion estimate — is not a quality claim about "
                "photographs; with your own portraits the gap between frozen and adapted is the number to watch."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('mad', 'mse', 'sad', 'mad_unknown'):\n"
                "    comparison[key] = {{'all_background': frozen_test['baselines']['all_background'][key], 'all_foreground': frozen_test['baselines']['all_foreground'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "print({{'validation_mad': {{'frozen': frozen_val['model']['mad'], 'adapted': adapted_val['model']['mad']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': best['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['model']['mad'] - best['val']['mad']) < 1e-3\n"
                "assert adapted_test['model']['mad'] < frozen_test['model']['mad']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 8. The photographs again, artifact export and fresh reload\n\n"
                "The adapted model mattes the same four photographs; they carry no label, so the comparison is the mean absolute "
                "difference between the frozen and the adapted matte per photograph and the foreground fractions side by side — a "
                "sanity check on whether learning the drawing style moved the model on real portraits (the build record: differences of "
                "0.004–0.05; on the man with the pipe the adapted matte took in dark background beside the body, the silhouettes "
                "themselves stayed), not an evaluation. The adapted mattes and cut-outs are written next to the "
                "frozen ones.\n\n"
                "`pipe.save_artifact` writes the trained tensors (about 17 MB) as `adapter.safetensors`, with a `manifest.json` recording "
                "the artifact format, the base model id and revision, the digest of the converted base file, the adaptation scope, the "
                "tensor names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`ModNetMattingPipeline.from_artifact` re-verifies the base file, checks the artifact manifest, scope and digest "
                "**before** deserialising, rebuilds the model and overlays the tensors — a fresh object from files, not the in-memory "
                "model (VER2). The cell asserts the same held-out MAD within 0.0001 and mattes within 0.001 (VER4)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "adapted_photos = pipe.predict(portraits)\n"
                "photo_drift = []\n"
                "for record, before, after in zip(portraits, frozen_photos['predictions'], adapted_photos['predictions']):\n"
                "    write_matte(record, after['alpha'], 'adapted')\n"
                "    drift = {{'photograph': record['id'], 'foreground_fraction_frozen': before['foreground_fraction'], 'foreground_fraction_adapted': after['foreground_fraction'], 'mean_abs_matte_difference': round(float(np.abs(after['alpha'] - before['alpha']).mean()), 4)}}\n"
                "    photo_drift.append(drift)\n"
                "    print({{**drift, 'note': 'no label; sanity check'}})\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': adapted_photos['model'], 'output': adapted_photos['output'], 'photographs': [{{'id': r['id'], 'title': r['title'], 'source': r['source'], 'license': r['license']}} for r in portraits], 'frozen': [{{k: v for k, v in p.items() if k != 'alpha'}} for p in frozen_photos['predictions']], 'adapted': [{{k: v for k, v in p.items() if k != 'alpha'}} for p in adapted_photos['predictions']], 'drift': photo_drift}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = ModNetMattingPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'mad_diff': round(abs(reloaded_test['model']['mad'] - adapted_test['model']['mad']), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_matte_diff': max(float(np.abs(a['alpha'] - b['alpha']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['mad_diff'] < 1e-4 and parity['max_abs_matte_diff'] < 1e-3\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'source_drive_file_id': SOURCE_DRIVE_FILE_ID,\n"
                "        'upstream_code_commit': UPSTREAM_CODE_COMMIT,\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'photographs': [{{'id': p['id'], 'url': p['url'], 'bytes': p['bytes'], 'sha256': p['sha256']}} for p in PORTRAIT_RECORDS],\n"
                "        'photograph_license': PORTRAIT_LICENSE,\n"
                "        'labelled_data': SAMPLE_LABEL_SOURCE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'pillow': PIL.__version__, 'numpy': np.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'photograph_drift': photo_drift,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 20 held-out drawn portraits the photographic MODNet checkpoint reaches a MAD near 0.08 against constant baselines of 0.34 "
        "and 0.66, and a bounded fine-tuning of its matting branches on 48 drawn portraits, selected by validation loss with the frozen "
        "model as a candidate, brings it near 0.004. That is the claim: the adaptation contract runs end to end on labelled "
        "portrait/alpha pairs, the pickle is audited and converted rather than served, and the artifact that carries the change is "
        "about 17 MB and reloads with the same outputs. It is not a claim about matting quality on photographs — the labelled portraits "
        "are drawings, chosen because no photographic matting dataset with alpha mattes is both permissively licensed and free of "
        "personal-data concerns — and the four photographs are a sanity check without labels, not an evaluation.\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 20 test portraits from one renderer, no dispersion estimate, and a "
        "domain gap (drawn figures) that makes the gain large by construction. Nothing here measures the model on the PPM-100 "
        "benchmark, on video, on group portraits, on hands and objects held in front of the body, or on the hair detail that matting "
        "is judged on in practice.\n\n"
        "Three things to carry to real data. **The alpha is the contract, and it must belong to the image:** an alpha drawn for another "
        "crop, or a binary mask passed off as a matte, is trained on without complaint. **Watch the photographs after adaptation:** "
        "fine-tuning on a narrow domain moves the model everywhere, and the frozen-versus-adapted drift on held-out photographs is the "
        "early warning. **Read the baselines first:** on a close-up where the subject fills 84 % of the frame the all-foreground matte "
        "already scores a MAD of 0.16; only the unknown-band MAD says whether the model resolved the boundary.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, can "
        "acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without executing anything "
        "outside the audited allow-list, build the vendored architecture and load it strictly, render and validate labelled portraits, "
        "fetch digest-pinned photographs, execute bounded fine-tuning, evaluate against constant baselines and the frozen model on "
        "held-out portraits, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, production fitness, or matting skill on photographs beyond the checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE = 'full'` and compare the photograph drift; raise "
        "`EPOCHS` and watch the validation loss drift; try `LEARNING_RATE = 5e-5`; or bring your own labelled portraits through BYOD and "
        "read the baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/modnet-matting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/modnet-matting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights, provenance and conversion notes: https://github.com/kurtvalcorza/modnet-matting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face mirror of the checkpoint: https://huggingface.co/XM5354/Modnet_models (revision `{MODEL_REVISION}`; byte-identical to the authors' Google Drive release)\n"
        "- Upstream repository (code, models and demos, Apache-2.0): https://github.com/ZHKKKe/MODNet\n"
        "- Ke, Z., Sun, J., Li, K., Yan, Q., Lau, R. W. H. (2022). MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition. AAAI 2022. arXiv:2011.11961: https://arxiv.org/abs/2011.11961\n"
        "- Photographs: CC0 Pixabay portraits re-hosted on Wikimedia Commons (URLs pinned in `samples.py`)\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
