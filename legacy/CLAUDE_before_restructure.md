# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Post-hoc analysis of HQQ weight/activation quantization accuracy for FGIR/UFGIR vision
models. The repo reads W&B run config/summary, saved logits and feature exports; it never
runs training and does not modify TGDA. Docs are mostly Traditional Chinese.

`AGENTS.md` is the operating contract (seven data/artifact rules, migration status per
module, the mandatory Slurm header). Read it, then the owning module's `README.md`, before
changing or running an analysis. `CONTEXT.md` fixes the vocabulary (decision stability vs.
accuracy robustness, normalized decision margin, vulnerable-tail margin, the gates for a
"supported" mechanism); use those terms and their thresholds as defined, including the
"avoid" synonyms.

## Running

Everything runs as a module from the repo root; there is no build, lint or test suite.
Verification is done by the `verify*` modules and `--verify-run` against real data.

```bash
conda run -n quant_analysis python -m activation_quant.run --input-id wandb_s20008_20261104
conda run -n quant_analysis python -m activation_quant.run --verify-run RUN_ID   # read-only recheck of one run
conda run -n quant_analysis python -m legacy_hqq.run --stage all --run-name main
conda run -n quant_analysis python -m hqq_confidence.run --stage all --run-name main
conda run -n quant_analysis python -m dataset_ranking.analyze --results-dir dataset_ranking/results/<new_dir>
conda run -n quant_analysis python -m dataset_ranking.verify  --results-dir dataset_ranking/results/<new_dir>
conda run -n quant_analysis python -m granularity_robustness.analyze && conda run -n quant_analysis python -m granularity_robustness.verify
```

- On `server-3090` use the conda env `quant_analysis` (Python 3.11; numpy 2.4.6,
  pandas 3.1.1, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.17.1, scikit-learn 1.9.1,
  threadpoolctl 3.8.0, wandb 0.31.0, Pillow 13.0.0, openpyxl 3.1.5, Jinja2 3.1.6; no
  torch). The READMEs and `AGENTS.md` say `opencode_env`, the original author's env, which
  is not on this box: substitute `quant_analysis`. The nano4 launchers use the `tgda` env
  on nano4. `requirements.txt` is unpinned.
- `data/`, `result/`, `results/` are gitignored everywhere, so a fresh checkout has no
  inputs. Analyses reuse local snapshots; a W&B download (project `nycu_pcs/TiT`, needs
  `WANDB_API_KEY` or `wandb login`) happens only when input is missing or `--refresh` /
  a new `--input-id` is given.
- `dataset_features` and the margin-matching / bias-competitor parts of `hqq_confidence`
  run on NCHC nano4 through the `run_*_nano4.sh` sbatch launchers
  (`--stage preflight` first). Any sbatch script carries the header in `AGENTS.md`
  (partition `8gpus`, account `MST114495`). Some raw arrays exist only on nano4.

## Architecture

One top-level package per analysis, each owning its code, data and results. Common stages:
`acquire → prepare (validate/pair) → analyze → plot`.

| Package | Does | Layout |
|---|---|---|
| `activation_quant` | W/A-bit accuracy ratio vs. matched unquantized AMP baseline; W+A minus weight-only deltas | new contract |
| `legacy_hqq` | original weight-only HQQ ratios; `recover_baselines.py` holds the corrected baseline pairing | old (`results/`) |
| `dataset_ranking` | dataset difficulty ranking and loss-profile PCA from a frozen legacy snapshot | old |
| `granularity_robustness` | FGIR vs. UFGIR comparison | old |
| `hqq_confidence` | matched-sample logit analyses: overconfidence, margin, swap mechanism, margin matching | mixed |
| `dataset_features` | paired FP/W3 feature geometry: CKA, PCA, t-SNE, class distances | old (`analysis/`, `results/`) |

- **New layout contract** (`activation_quant`, target for all): immutable
  `data/<input_id>/` with `source_manifest.json` (sizes, SHA-256), full output in
  `result/<run_id>/` (singular) with `run.json`, and exactly one tracked
  `reports/<run_id>/REPORT.md` whose `figures/` are hash-identical copies of the ones it
  cites. Existing input/run/report directories are never overwritten: a rerun or a formula
  change gets a new ID. Unmigrated modules keep their current paths until migrated and
  verified; path moves are recorded in a `relocations/` receipt.
- **Cross-package imports exist and are intended**: `activation_quant` and
  `tools/baseline_recovery` import `context`, `differences`, `DATASETS`, `MODELS`, `MODES`
  from `legacy_hqq.recover_baselines`; `granularity_robustness` and `dataset_ranking`
  verifiers import `dataset_ranking.analyze` / `pca_profiles`. A change there affects
  every consumer.
- **Baseline pairing** matches the full checkpoint path, dataset, model, mode, seed,
  classifier, precision and recorded eval settings, never a basename or serial. The
  baseline is unquantized AMP fp16; `fp32_top1` is a legacy column name, not pure FP32.
  Missing, ambiguous or duplicate matches stay N/A with a reason; no latest-run pick, no
  ratio clipping. Quantized accuracy is `summary.hqq_top1`, baseline `summary.top1`, both
  in percent.
- **Cohort**: 16 datasets × 3 models × FT/FZ/CAL = 144 contexts. Coverage is checked per
  condition against that matrix; extra seeds do not add coverage. A condition ID hashes
  the full quant config, so two "W3A4" runs with different scope/group/exclude stay
  separate. `activation_quant/run.py` dispatches schema-2 inputs to `workflow.py` and the
  historical schema-1 W6A4 input (serial 20007) to `analyze.py`.
- **No tests or synthetic fixtures in the repo** (AGENTS.md rule 5): probes and mocks run
  outside it and are cleaned up.
- `archive/five_dataset_features_legacy_20261101/` is hashed, non-executable source;
  restore it before running its commands. `docs/` holds layout plans, baseline provenance
  and ADRs; `docs/analysis_history_20261021.md` is the entry for old results.
