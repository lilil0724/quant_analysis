# Legacy

The earlier analysis packages and documents, kept at their former paths under `legacy/`. The paper
pipeline at the repo root reads only the run bundles listed in `data/MANIFEST.json`.

| Path | Holds |
|---|---|
| `quant_acc/` | Weight-only HQQ accuracy ratios and baseline pairing; `acquire_snapshot.py` downloads serials 10000–10016, `rerun_serial_grid.py` selects the grid from a download, `check_duplicates.py` compares duplicate runs; `20261117_legacy_hqq_s10000_10016_cal10016_v2/` (untracked) is the run that produced `data/grid.csv`, `legacy/` (untracked) the older 141-context grid |
| `activation_quantization/` | Weight and activation (W/A) quantization grids, serials 20007 and 20008 |
| `dataset_features/` | Paired baseline/W3 feature geometry (CKA, PCA, t-SNE, class distances); nano4 launchers |
| `dataset_confidence/` | Logit analyses: overconfidence, margin, swap mechanism, margin matching, bias/competitors; nano4 launchers |
| `dataset_ranking_robustness/` | Dataset difficulty ranking and loss-profile PCA |
| `dataset_static_feature/` | FGIR vs. UFGIR group comparison |
| `archive/` | Hashed, non-executable source of the five-dataset feature analysis |
| `docs/` | Plans, ADRs, baseline-recovery records and research notes of the earlier analyses |
| `AGENTS.md`, `CONTEXT.md` | The earlier operating rules and vocabulary for these packages |
| `CLAUDE_before_restructure.md`, `README_before_restructure.md`, `requirements.txt` | The repo's agent notes, README and (unpinned) requirements before the restructure |
| `paper_draft/` | The paper's working notes: handoff brief, intro drafts, citation map, `new_draft_guideline.md`, `pdfs/` (cited papers, untracked), `data/` (serial 20004 run bundles, untracked), `reports/`, the WACV author kit; `paper/` holds the old `build.sh`, `EVIDENCE.md`, `TBD_zh.md` and README |
| `paper_versions/` | Source copies of paper versions v1–v4; their PDFs are in `drafts/` |
| `pr/` | Empty folder from the earlier layout |
| `RELOCATION.json` | Every untracked file before the move (path, size, SHA-256); tag `pre-restructure-20261120` has the tracked ones |

Run a package from this folder, e.g. `cd legacy && python -m quant_acc.run --stage all --run-name main`.
The nano4 launchers are submitted from `legacy/` (`sbatch dataset_features/run_nano4.sh ...`).
