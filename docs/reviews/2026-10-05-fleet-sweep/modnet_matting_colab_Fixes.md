# modnet_matting_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`d74036b`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check) | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 1 of 9 guided markers. Added a model-specific guided layer: audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, four **Predict before running** prompts with **Check your reasoning** answers (Sections 4–7) taken from the repository's build record numbers already quoted in the notebook (frozen test MAD 0.079 vs 0.341 / 0.659 baselines, adapted 0.004, unknown-band 0.085 → 0.025, validation loss 0.80 → about 0.06), Troubleshooting, Glossary and a Conclusion template. Sections 1–3 labelled Infrastructure and collapsed. A literal `{{id, image, alpha}}` in the data-contract prerequisite (prerequisites are not formatted) now renders as `{id, image, alpha}`. | Template opening, prerequisites, Sections 4–7, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Fixed | Confirmed in Section 7: `assert adapted_test['model']['mad'] < frozen_test['model']['mad']` aborted any run that did not improve, before Section 8's export and reload. Replaced with a recorded verdict (`improved` / `no gain` / `worse`, with the MAD delta) in `evaluation_report['verdicts']` and a printed note; the procedure guarantees (kept epoch's validation loss ≤ frozen, validation MAD reproduces the kept epoch) and the reload-parity assert stay as contract checks. | Section 7; validator marker | `test_swp_a_no_quality_assert_and_verdict_recorded`, `test_swp_a_verdict_does_not_abort_when_adaptation_is_worse` |
| SWP-F (frozen re-run) | Fixed | Confirmed: `pipe.adapt` trains `pipe.model` in place, so re-running Section 6 with other hyperparameters (the closing's suggested experiments) continued from the adapted weights while epoch 0 was still labelled "frozen model". Section 6 now snapshots the pinned state once, before any training (refusing if the pipeline was already adapted), restores it at the start of every run, clears the adapter and checks the restored state digest. | Section 6; closing experiments note; validator marker | `test_swp_f_rerunning_section6_restarts_from_frozen_weights` (executes the Section 6 cell twice with a stand-in pipeline that trains in place) |
| SWP-B (BYOD) | Fixed | Both gates (labelled zip, photograph) worked only through `files.upload()`, and an empty upload raised a bare `StopIteration`. Added `BYOD_PATH` and `BYOD_PHOTO_PATH` form fields that work on Colab, Kaggle and Jupyter, with the Colab upload as a guarded fallback; refusals name the path, the expected suffix and the rule (the package's own loaders already name the CSV row and member). | Sections 4 and 8, BYOD declaration | `test_swp_b_byod_path_reads_file_and_refuses_with_names`, `test_swp_b_cancelled_colab_upload_gives_a_clear_message`, `test_swp_b_byod_path_fields_default_off` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run) or reuses `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes only.
- Section 7 prints and records an adaptation verdict instead of stopping on a non-improving run.
- Section 6 restarts from the frozen weights on every run (prints `restarted_from_frozen_weights`).
- New form fields `BYOD_PATH` (Section 4) and `BYOD_PHOTO_PATH` (Section 8); on Colab an empty path still opens the upload dialog.
- Guided material added; Sections 1–3 collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Stand-ins: the kernel-cell test runs the generated bootstrap against a pre-built environment folder whose `python` is the test interpreter (routing, reuse and idempotence are real; the managed CPython and locked packages are stand-ins); the SWP-F test runs the Section 6 cell with a numpy-backed stand-in pipeline.
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies (torch absent; the two torch modules skip cleanly): 33 passed, 2 skipped before → 46 passed, 2 skipped after.
- Every code cell of the regenerated notebook parses.

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
