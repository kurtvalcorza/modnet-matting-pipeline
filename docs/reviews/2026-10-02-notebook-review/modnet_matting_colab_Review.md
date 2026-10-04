# MODNet Portrait Matting E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/modnet-matting-pipeline`  
**Notebook:** `tutorials/modnet_matting_colab.ipynb`  
**Reviewed commit:** `d74036be500995a26caa29083c192445689fc1b4` (`main`, confirmed with `gh api repos/kurtvalcorza/modnet-matting-pipeline/commits/main`)  
**Notebook Git blob:** `ab5bea5843e2665a7f5b5600c43fe4cf4c9a7be5`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `aa26e44`). The notebook has not changed since; the carried modules are those of `125c23a86d78`.  
**Finding prefix:** `MOD`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The default path holds up well. The notebook digest-verifies a legacy pickled checkpoint, audits it statically, converts it once into safetensors and says so. It renders 80 drawn portraits with exact alpha mattes, scores the frozen model against both constant baselines, runs a seeded bounded fine-tune of the matting branches with BatchNorm statistics frozen and validation-loss epoch selection, scores the held-out portraits again, and exports an adapter that reloads with identical outputs. Its interpretation section is unusually honest: the gain comes from learning a drawing style, the sample is one seed and 20 drawings, and the photograph gate is a sanity check, not an evaluation. This review's clean CPU run reproduced the recorded numbers (test MAD 0.0795 frozen → 0.0042 adapted, baselines 0.341 / 0.659, reload parity 0.0).

The defects sit outside the default `Run all`, in the paths a learner is told to take next, plus the install step:

1. **No one-pass `Run all` (MOD-M1).** The recorded Kaggle run stopped at the install guard (`cuda-bindings 12.9.4→13.4.2`, `numpy 2.0.2→2.5.3`) and passed only after a restart. The release record and `STATUS.md` still report PASSED / Release-grade, and `release-verification.md` step 4 calls the restart "expected".
2. **The documented reruns reuse the adapted model and label it "frozen" (MOD-M2).** `pipe.adapt` changes `pipe` in place. If the learner follows the BYOD instruction and re-runs from Section 4, Section 5 reports the drawing-adapted model as the frozen model (test MAD 0.0034 instead of 0.0795). If the learner follows an "Optional experiment" and re-runs Section 6, training continues from the adapted weights, and epoch 0 is still labelled `frozen model`.
3. **The BYOD contract fails as documented (MOD-M3).** "At least four pairs" is wrong: 4 or 5 pairs fail in Section 6 only after the frozen evaluation, and 6 is the real minimum. With 6 pairs the test split holds one portrait, all three refusal probes are rejected for the wrong reason (record count), and the bare `assert adapted < frozen` in Section 7 stops the notebook before predictions, export and reload whenever validation keeps epoch 0.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites name knowledge of alpha mattes, trimap unknown bands and MAD/MSE/SAD against a constant baseline |
| Supported runtime | Colab (CPU or T4), Kaggle, or a Jupyter kernel with Python 3.12; float32 |
| Promised outcomes | pinned install; stage + digest-verify the checkpoint; static pickle audit and one-time conversion; strict load of the vendored architecture; 80 rendered portraits (48/12/20) validated with three refusal probes; frozen model vs constant baselines with mattes/cut-outs written; bounded fine-tune of the matting branches; held-out four-way comparison; adapted mattes; adapter export + fresh reload parity; optional BYOD pairs (≥4) through the same contract; optional own photograph |
| Learning objectives | install; inspect the carried modules; stage/verify/audit/convert a pickle; render and validate labelled portraits; read MAD/MSE/SAD/unknown-band MAD against baselines; run bounded fine-tuning; compare adapted vs frozen; write mattes/cut-outs; export and reload an adapter |

### Existing execution evidence

- `docs/release-verification.md`: Kaggle Tesla T4, commit `aa26e44` / blob `ab5bea58` (= the reviewed blob), 2026-09-20, **PASSED — 11/11 code cells ok (1 restart after install cell)**. The archived `run_summary.json` (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-modnet-matting/v2/evidence/`) shows pass 1 failing in cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`, and pass 2 succeeding after the restart. Test MAD 0.0794 → 0.0035, best epoch 5, reload parity 0.0.
- No hosted run covers BYOD, the photograph gate, or any optional experiment (REL12).

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`), build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, numpy 2.5.3, pillow 11.3.0, safetensors 0.8.0 = `PINS`; huggingface-hub 0.36.2 instead of the pinned 1.32.0). The install cell ran with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so nothing was installed. `HF_HOME` and the working directory were empty scratch directories, so the checkpoint was fetched from the Hub at the pinned revision, audited and converted fresh. `google.colab.files.upload` was a shim returning prepared bytes. **This is not a Colab run.** The first attempt stopped in Section 4 with `MemoryError` because the shared host was out of commit memory (0.9 GB free); the retry ran clean.
- **P1 static:** JSON parses, 11 code cells compile, no persisted outputs, the blob equals the recorded-run blob, and the carried modules are unchanged since `125c23a86d78`.
- **P2 default path (direct, CPU):** 11/11 cells ok, 101.8 s. Test MAD 0.0795 → 0.0042, unknown-band MAD 0.0855 → 0.0249, SAD 20.84 → 1.11. Validation loss 0.8008 → 0.0599 (best epoch 4 of 6), 4,263,203 trainable parameters, reload parity `mad_diff 0.0, max_abs_matte_diff 0.0`. 20 output files written.
- **P3 rerun from Section 4 (the BYOD instruction, sample data):** Section 5 "frozen" test MAD 0.003409 = the adapted model's; `frozen_test['adapted'] == True` (not printed); epoch 0 `note: 'frozen model'` with validation loss 0.0393; Section 7 shows `frozen 0.003409 → adapted 0.003228`.
- **P4 exercise rerun (Section 6 with `TRAINABLE='full'`, `EPOCHS=1`, then 7 and 8):** epoch 0 validation loss 0.059943 = the previous kept epoch (`epoch0_equals_previous_kept_not_frozen: true`), labelled `frozen model`. Section 7 prints `validation_loss.frozen = 0.0599`. Export/reload parity still 0.0.
- **P5 BYOD arithmetic (no model):** `split_dataset` sizes for N = 3…10 and `validate_dataset(train)` as `pipe.adapt` calls it. N = 4 → 2/1/1, rejected `2 records; 4..2000 are required`; N = 5 → 3/1/1, rejected; N = 6 → 4/1/1, accepted. For every N ≤ 10 the test split has 1–2 records, so the Section 4 refusal probes are rejected for record count. Zips without `pairs.csv`, or whose CSV names a missing member, are rejected with actionable messages.
- **P6 BYOD, 6 pairs (640 × 480 rendered portraits) + own photograph, after a fresh Section 3:** Sections 4–6 ran (**`EPOCHS` overridden to 2** to save time). Validation kept epoch 0, and Section 7 failed on the bare `assert adapted_test MAD < frozen_test MAD` (0.003301 vs 0.003301). Continuing past it, the photograph gate (900 × 1200 JPEG → model size 672 × 512), export and reload all ran, with parity 0.0.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | The default path is correct and reproducible (CPU numbers within 0.001 MAD of the T4 record). The install guard forces a restart on hosted images (MOD-M1). `adapt` mutates shared state, so the documented reruns produce mislabelled results (MOD-M2). The BYOD minimum and the Section 7 assertion are wrong for small user datasets (MOD-M3). |
| Promise fulfilment | Default promises are delivered and evidenced. The "≥4 pairs through the same contract" BYOD promise is not (MOD-M3). "Run all" is not one pass (MOD-M1). |
| Scientific validity | Held-out test from a disjoint seed range; validation-only epoch selection with the frozen model as a candidate; both constant baselines on the same pixels; limits stated plainly (domain gap, one seed, no dispersion). After a rerun, the "frozen" baseline is no longer frozen, which invalidates the comparison the notebook tells BYOD users to watch (MOD-M2). |
| Learner experience | Good "Look for" notes in Sections 4–6 and a strong interpretation section. No stated audience, roadmap, prediction prompts, checkpoints, troubleshooting or conclusion template, and the mattes are written to files but never shown (MOD-m1, MOD-m2). |
| Spec conformance | Fails RUN1/RUN10/ENV6 (restart), REL2/REL11 (restart-dependent run recorded as a pass), REL12 (no BYOD verification), DAT12/DAT14/DAT19 (BYOD minimum, late failure), SRC2 (hidden state dependency on rerun). SHOULD gaps: GDL1–4, GDL7, GDL9, GDL13, GDL14, EXE5, UX11. Declares spec 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes in a fresh runtime | cell 3 pip install + stale-import guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 ok | Learner must restart; opening paragraph does not say so | **Not met** (MOD-M1) |
| Pinned, digest-verified, audited, converted checkpoint | cell 13 → `stage_missing_files`, `verify_snapshot`, `from_pretrained(report=print)` | audit: 4 globals, 0 violations, audit digest `5b9f0ba0…`; converted `0ec6d832…` | Printed before load | Met (documented + direct) |
| 80 portraits, validated, three refusals | cell 15 | 48/12/20, fg ≈ 0.34, three refusals with the intended reasons | "Look for" note matches | Met (default only; see MOD-M3) |
| Frozen model vs constant baselines | cell 17 `pipe.evaluate` | 0.0795 vs 0.341 / 0.659 | Explained, including all-background MAD = foreground fraction | Met on the first pass; **wrong after a rerun** (MOD-M2) |
| Bounded fine-tune of the branches, frozen BN | cell 19 `pipe.adapt` | 4.26 M of 6.49 M trainable, 72 steps, seed 0, history printed | Explained; epoch 0 labelled frozen | Met on the first pass; **mislabelled on reruns** (MOD-M2) |
| Held-out four-way comparison | cell 21 | adapted 0.0042 | Domain-gap caveat stated | Met |
| Adapter export + fresh reload parity | cell 23 `save_artifact`, `from_artifact` | parity 0.0 | VER2/VER4 explained | Met |
| BYOD ≥4 pairs through every stage | cells 15–23, `load_byod_dataset`, `split_dataset(seed=0)` | 4–5 pairs fail in Section 6; 6 pairs stop at the Section 7 assertion | — | **Not met** (MOD-M3) |
| Own photograph gate | cell 23 `load_photo` | ran via shim (P6) | "Sanity check, not evaluation" | Met (direct, shim) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Read MAD/MSE/SAD/unknown-band MAD vs baselines | Read the printed table | Explained in prose; no prediction or checkpoint question |
| Run a bounded fine-tune with explicit hyperparameters | Optional experiments (`TRAINABLE`, `EPOCHS`, `LEARNING_RATE`) | The rerun does not start from the base model, so the comparison it invites is confounded (MOD-M2) |
| Compare adapted vs frozen | Section 7 table | Valid on the first pass only |
| Export and reload | Section 8 | Parity asserted |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Oriented by strong section prose and "Look for" notes. Missing audience, roadmap, prediction prompts, checkpoints, troubleshooting and on-screen mattes (MOD-m1, MOD-m2). The install restart is not mentioned where Run all is promised (MOD-M1). |
| Clean default | Documented (Kaggle T4, reviewed blob) + direct (local CPU) | Passes after one restart (documented). Passes 11/11 on CPU with the install skipped (direct, not Colab). |
| Active learning | Direct (local CPU) | The `TRAINABLE='full'` exercise runs, but continues from the adapted weights, with epoch 0 labelled `frozen model` (MOD-M2). |
| Reuse and recovery | Direct (local CPU, upload shim) | BYOD 4–5 pairs fail late; 6 pairs stop at the Section 7 assertion (with `EPOCHS=2`); the photograph gate, export and reload work; malformed zips are rejected clearly (MOD-M3). Not verified on a hosted runtime. |

## 5. Findings

### Major

#### MOD-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run as a pass

- **Cell/section:** Section 1 (cell 3); opening "Run all" paragraph; `docs/release-verification.md` (step 4, Current status); `STATUS.md`; `tutorials/README.md`.
- **Observed issue:** cell 3 `pip install`s `torch==2.14.0`, `numpy==2.5.3` … into the running kernel and raises `RuntimeError … Restart the runtime, then rerun from the top` when a loaded distribution changed. On the recorded Kaggle T4 run, pass 1 failed this way (`cuda-bindings 12.9.4→13.4.2`, `numpy 2.0.2→2.5.3`) and only pass 2 completed. The release record nevertheless says PASSED / Release-grade, and step 4 of the procedure calls the restart "expected". The opening Run-all paragraph promises completion without mentioning it.
- **Consequence:** a learner's first Run all on a hosted image stops at cell 3. A run that depends on a restart is reported as a release pass.
- **Evidence:** documented: `run_summary.json` passes 1–2, `restarted_after_install_cell: true`. Source: cell 3; `tools/build_notebook.py` lines ~48–61.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**. The setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` (`uv pip install --require-hashes --only-binary :all:`), and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (also `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`). Implement it in `tools/build_notebook.py` (install cell) and regenerate. Do not add another in-kernel guard. Correct `release-verification.md` and `STATUS.md` so a restart-dependent run is not recorded as a Run-all PASS, and return the status to Candidate until a one-pass run is recorded.
- **Acceptance check:** a fresh hosted Colab or Kaggle runtime executes all code cells in one pass with no restart and no error output. The release record cites that run's blob, and step 4 no longer calls a restart expected.

#### MOD-M2 — Documented reruns reuse the already-adapted model and report it as "frozen"

- **Cell/section:** opening BYOD paragraph ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Sections 5–7 (cells 17, 19, 21); "Optional experiments" (Interpretation and limits); `pipeline.py` `adapt` (lines ~771–915).
- **Observed issue:** `pipe.adapt` trains `pipe.model` in place and does not restore the base weights. Re-running from Section 4, as the BYOD instruction says, makes Section 5's `frozen_test` score the drawing-adapted model: test MAD **0.003409 instead of 0.079514**. `evaluate` returns `adapted: True`, but the cell does not print it. Re-running Section 6, as the exercises invite (`TRAINABLE = 'full'`, more `EPOCHS`, `LEARNING_RATE = 5e-5`), continues from the adapted weights. Epoch 0, labelled `frozen model`, has validation loss 0.0599 (the previous kept epoch, not the frozen 0.8008), and Section 7 prints it as `validation_loss.frozen`. The "full" result then reflects 6 branch epochs plus the new run.
- **Consequence:** BYOD users are told that "the gap between frozen and adapted is the number to watch", but after the documented rerun that gap compares the adapted model with itself. Each exercise comparison is confounded. Both are silent: nothing on screen says the baseline is not frozen.
- **Evidence:** direct (P3, P4). Source: `adapt` has no reset; cell 23 itself rebuilds a fresh `from_pretrained` pipeline for the photo's frozen matte, which shows the author knew `pipe` is mutated.
- **Spec:** SRC2, DAT13/DAT14, GDL10, UX7, EVAL14.
- **Recommended correction:** make the frozen model a stable reference. Either (a) Section 5 and `adapt` operate on a pipeline rebuilt from the verified base (`ModNetMattingPipeline.from_pretrained` at the top of Section 5, or `adapt` restoring the base trainable tensors before epoch 0), or (b) the BYOD and exercise instructions say "re-run from Section 3", and Sections 5 and 6 refuse with an actionable message when `pipe.adapter is not None`. Fix in `tools/notebook_template.py` (byod text ~line 62, Section 5 cell ~line 198, Section 6 cell ~line 246, exercises ~line 405) and/or `pipeline.py`, then regenerate.
- **Acceptance check:** after a full default run, re-running from Section 4 prints a Section 5 test MAD equal to the first pass's frozen value (±0.001). Re-running Section 6 with `TRAINABLE='full'` prints epoch 0 validation loss equal to the frozen model's (≈0.80 on the sample). Or, under option (b), both reruns stop with a message naming Section 3.

#### MOD-M3 — The BYOD contract fails as documented: the minimum is wrong, the probes misfire, and a bare assertion stops the run

- **Cell/section:** opening BYOD paragraph ("at least four pairs"); Section 4 (cell 15) refusal probes; Section 6 (`pipe.adapt` → `validate_dataset(train)`); Section 7 (cell 21) `assert adapted_test['model']['mad'] < frozen_test['model']['mad']`.
- **Observed issue:**
  - `split_dataset` (70/15/15) gives 2/1/1 for 4 pairs and 3/1/1 for 5. `adapt` then rejects the training split (`2 records; 4..2000 are required`), but only in Section 6, after the frozen evaluation has run. The real minimum is 6 pairs (4/1/1).
  - With any N ≤ 10, the test split holds 1–2 records, so all three Section 4 refusal probes are rejected with `1 records; 4..2000 are required` instead of the alpha-range, size and all-background conditions they claim to demonstrate.
  - Section 7's unconditional assertion fails whenever validation selection keeps epoch 0, which the procedure explicitly allows ("the frozen model as a candidate"). With 6 pairs it fired (adapted = frozen = 0.003301) with a bare `AssertionError`, so the predictions, export and reload in Section 8 never ran. The test estimate is a single portrait, with no warning.
- **Consequence:** the promised BYOD route (validate → split → adapt → evaluate → infer → export) fails for the stated minimum, fails late for 4–5 pairs, and on a legitimate "adaptation did not help" outcome stops with no actionable message.
- **Evidence:** direct (P5, P6; P6 used `EPOCHS=2`). Source: `samples.py` `split_dataset` (lines 268–283), `pipeline.py` `MIN_RECORDS = 4`, template line 289.
- **Spec:** DAT12, DAT14, DAT19, UX10, REL12.
- **Recommended correction:** validate the BYOD split sizes in Section 4, before any model runs, with a message naming the minimum (or derive the minimum from `MIN_RECORDS` and the split fractions and state it correctly). Run the refusal probes on fixed sample records, not on `test_records[1:4]`. Replace the Section 7 test assertion with a printed verdict (for example, "adaptation kept the frozen model" or "adapted worse than frozen") on the BYOD path, keeping the assertion only for the sample path if wanted. Warn when the test split has fewer than ~5 portraits. Record a hosted BYOD run in `release-verification.md` (REL12).
- **Acceptance check:** a 4-pair zip is refused in Section 4 with a message stating the minimum. The three probes print their intended reasons under BYOD. A 6-pair BYOD run whose validation keeps epoch 0 completes through export and reload with a printed verdict instead of an `AssertionError`.

### Minor

#### MOD-m1 — Guided layer is partial

- **Cell/section:** opening cells; Sections 4–8; Interpretation and limits.
- **Observed issue:** "Look for" notes and the interpretation section are strong. Missing: an intended-audience statement and **How to use this notebook** (GDL1–2), a roadmap (GDL3), an Input → Model → Output contract near the top (GDL4), prediction prompts before Sections 5 and 7 (GDL7), interpretation checkpoints with sample answers (GDL9), troubleshooting (GDL13), and a conclusion template (GDL14). The 1,850 lines of carried modules in Section 2 are introduced but not labelled as optional **Infrastructure** (GDL11).
- **Consequence:** self-paced learners get explanation but few points where they have to think before reading the answer.
- **Evidence:** source inspection; P1 keyword scan.
- **Recommended correction:** add these in `tools/notebook_template.py` (intro, cell markdown, closing) and regenerate.
- **Acceptance check:** the regenerated notebook has an audience/how-to-use block, a roadmap, at least two prediction prompts with collapsible answers, a troubleshooting section, and a conclusion template.

#### MOD-m2 — The mattes that "show the failure" are never shown

- **Cell/section:** Sections 5 and 8 (`write_matte`).
- **Observed issue:** Section 5 says the frozen mattes and cut-outs are written "so the failure can be seen", but no cell displays them. The learner has to open PNGs from the file browser.
- **Consequence:** the visual evidence for the core comparison is easy to miss.
- **Evidence:** source inspection (no `display`/`imshow` in any code cell).
- **Spec:** UX3, UX11.
- **Recommended correction:** show a small reference / frozen / adapted grid for the two held-out portraits (and the photo, when gated on), while still writing the files.
- **Acceptance check:** Sections 5 and 8 render the mattes inline.

#### MOD-m3 — Declarations and documentation drift

- **Observed issue:** the notebook declares NOTEBOOK_SPEC 2.0 (current 2.2). The Prerequisites render a literal `{{id, image, alpha}}` (template line 115 doubles braces in a string that is not `.format()`-ed). The install cell reads `DIMER_NOTEBOOK_CI_PREINSTALLED` without documenting it (EXE5). A `UserWarning` from `torch.from_numpy` in `_to_tensor` appears in Section 5 output on the T4 record with no explanation.
- **Evidence:** source inspection (P1); documented (`run_summary.json` cell 17 stdout).
- **Recommended correction:** update the spec declaration after a 2.2 conformance pass, fix the braces, document the variable in Section 1, and make the array writable (or narrowly filter that warning).
- **Acceptance check:** the regenerated notebook shows `{id, image, alpha}`, names the environment variable, and Section 5 prints no warning.

### Suggestions

- **MOD-S1:** print `frozen_test['adapted']` and `pipe.adapter is not None` in Sections 5 and 6, so any future state reuse is visible.
- **MOD-S2:** add a second seed, or the per-image MAD spread, to the comparison, so learners see variability rather than one number.
- **MOD-S3:** in BYOD mode, avoid overwriting the sample-pair files in `outputs/` with the user's first test portrait, or name them after the data source.

## 6. Readiness

**Needs revision.** Three Major findings are open: a restart-dependent Run all (MOD-M1), mislabelled frozen results on the documented reruns (MOD-M2), and a BYOD contract that fails as documented (MOD-M3). The applicable MUSTs RUN1/RUN10/ENV6/REL2/REL11/REL12/DAT12/DAT14/DAT19 are unmet. The default sample path itself is technically sound and evidenced. Remaining gates after fixes: a one-pass hosted Run all of the new blob, and a recorded hosted BYOD run.

## 7. Verified versus inferred

- **Verified (direct, local CPU, not Colab):** default path 11/11 with the recorded metrics; rerun-from-Section-4 and exercise-rerun state reuse; BYOD minimum arithmetic; 6-pair BYOD stopping at the Section 7 assertion (with `EPOCHS=2`); photo gate, export and reload; malformed-zip refusals.
- **Verified (documented):** the Kaggle T4 run of the reviewed blob needed a restart after cell 3.
- **Inferred:** that hosted Colab images also trigger the restart (Colab preloads NumPy and the CUDA bindings like Kaggle, but no Colab run of this blob is recorded); that the Section 7 assertion fires at the default 6 epochs on real BYOD data (it fires whenever epoch 0 is kept, which depends on the data).
- **Only Kurt can confirm:** whether BYOD is meant to be release-gated for this row (REL12), and learner-facing effectiveness (no learner observation).
- **Most likely to be wrong:** MOD-M3's severity. Its 6-pair failure was observed with `EPOCHS` reduced to 2, so with the default 6 epochs on real portraits the assertion may fire less often, and the remaining items (minimum, probes) could be argued Minor.

*Probes: `modnet_matting_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).*
