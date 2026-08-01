# AGENTS.md — quant_analysis

Standalone analysis scripts that download WandB runs to `data/` and summarise /
plot them. No package, no tests, no build step. Each script is a CLI entrypoint
that exits 0 on success and writes under `results_all/<sub>/` (created on demand).

## Hard prerequisite: the `utils` module is NOT in this repo

`summarize_acc.py`, `summarize_cost.py`, `summarize_quant.py`, and `plot.py` all
do `from utils import ...`. `utils.py` lives in the sibling repo
`../reference/BackbonesAnalysis/utils.py` (relative to this dir's parent
`my_project`). It is not vendored here. To run any of those scripts you must put
that dir on `PYTHONPATH`, e.g. from this dir:

```powershell
conda activate opencode_env  # parent AGENTS.md machine convention
$env:PYTHONPATH = "$env:PYTHONPATH;C:\iCloudDrive\project_2\reference\BackbonesAnalysis"
python summarize_quant.py
```

`utils` provides the canonical `rename_swin` / `add_setting` / `diff_setting` /
`diff_384_224` / `highlight_top_k` / `min_max_*_better` helpers, the
`MODELS_OG` / `DATASETS_REDUCED` / `MODELS_DIC` / `SETTINGS_DIC*` / `VAR_DIC`
constants, and the `input_output_args` / `filtering_args` / `plot_args`
argparse factories that `plot.py` uses (which is why `plot.py` exposes flags like
`--input_file_cost`, `--input_file_stats_models`, `--host` that don't appear in
the summarizers' argparse). Any change to one of those helpers' signatures will
break every summarizer at once — check all call sites.

## Inputs (per script)

Default paths are under `data/` and are not committed. Without the CSVs the
summarizers fail on `pd.read_csv` / `pd.merge`:

- `summarize_acc.py`: `data/backbones_stage2.csv`, `data/backbones_accuracy_cal_ap.csv`, `data/stats_datasets.csv`
- `summarize_cost.py`: `data/backbones_train_cost_cub_3090.csv`, `data/backbones_inference_cost.csv`, plus the three above (it imports `load_acc_df` / `aggregate_results` from `summarize_acc`)
- `summarize_quant.py`: `data/backbones_quant.csv`, optional `data/stats_datasets.csv`
- `plot.py`: cost CSV + `data/stats_datasets.csv` + stats_models CSV (paths are CLI args via `utils`)

To regenerate these CSVs run `download_save_wandb_data.py`. Requires `wandb`
login (env `WANDB_API_KEY` or `wandb login`).

## `download_save_wandb_data.py`: 3-way default selection (gotcha)

Defaults are selected in `parse_args` — not by argparse `set_defaults` — because
the choice is 3-way. `default=None` flags let the user-passed value win via `or`:

| Flags | WandB project | output CSV (under `data/`) | column set |
|---|---|---|---|
| (none) | `nycu_pcs/FGIRFT` | `fgirft_stage2.csv` | `CONFIG_COLS` / `SUMMARY_COLS` / `SORT_COLS` (training runs) |
| `--quant` | `nycu_pcs/TiT` | `backbones_quant.csv` | `QUANT_*` (eval_quant.py HQQ / native / fp32 runs) |
| `--vit_only` | `nycu_pcs/Backbones`, `serial=1`, `model_name=vit_b16` | `backbones_vit_stage2.csv` | training cols |

`QUANT_SUMMARY_COLS` is the **bare** set `top1, loss, vram_gb, params_m,
time_total_s` — post-cleanup. The earlier `hqq_top1 / hqq_loss /
hqq_vram_gb / hqq_params_m` keys plus the `--hqq_compare_fp32` parasite set
(`original_*`, `acc_drop`, `params_reduction_m`) have been removed from
`eval_quant.py`'s wandb log; do NOT re-add them here unless the sibling TGDA-2
repo's `eval_quant.py` is reverted. See `fix_eval_quant.md` for the upstream
cleanup checklist.

`--serials` only applies in the default and `--vit_only` paths (forces `[1]`).
`--project_name` is ignored when `--vit_only` is set (hard-coded to `Backbones`).

## Quant serials (the `4a0c` scan — `summarize_quant.py`)

`--main_serials` default is `4000..4003, 4100..4103, ..., 4700..4703, 4999`.
Format `4a0c`: prefix `4`, `a` = nbits code (0..7), fixed `0`, `c` = group_size
code (0..3). `4999` is the fp32 baseline. The **code → nbits value** and
**code → group_size value** mappings are NOT in this repo — they live in the
sibling TGDA project's `eval_quant.py` (see `nycu_pcs/TiT`). Do not invent them;
fetch from TGDA before reasoning about actual bit widths.

Per-run potion is reconstructed from the WandB row by `summarize_quant.py`:

- `hqq=True` → `quant_setting = hqq_n{nbits}_g{group_size}[_nozero][_noscale][_offload]`
- `hqq` omitted + model in `QUANT_MODELS_NATIVE` (`BHViT`, `BinaryViT`, `GSB`,
  `SI_BiViT`, `158b_ViT`) → `quant_setting = native_<model_name>`
- `hqq` omitted + non-native model → `quant_setting = fp32` (the 4999 path)

All three long-format rows (`_long_from_hqq` / `_long_from_native` /
`_long_from_plain_fp32`) read the **bare** `top1 / loss / vram_gb / params_m`
summary keys — the earlier `hqq_*` and `original_*` keys are gone (see
`fix_eval_quant.md`). The only fp32 baseline is the independent 4999 run;
`--hqq_compare_fp32`'s parasitic in-run fp32 record has been removed upstream
and not regenerated here. Missing 4999 baseline for a (dataset, model) →
`diff_fp32_*` is NaN for all that group's rows (intentional; do NOT fall back
to a fabricated baseline).

WandB config booleans can arrive as Python `bool` **or** `'True'` / `'False'` /
`'true'` / `1` strings — `summarize_quant.py` tolerates both explicitly. When
adding branches keyed on a config flag, check both shapes or you'll silently
misclassify.

`summarize_quant.py` / `summarize_acc.py` write `*_counts.txt` by redirecting
`sys.stdout` to a file inside `count_best_per_ds`/`count_best_per_host` — the
function returns through a `with open(...) as sys.stdout:` block, so any `print`
you add during that call will land in the file. Each also writes one per-dataset
`.xlsx` via `openpyxl` → `openpyxl` is a runtime dep, not optional.

## Settings vocabulary (shared across scripts)

Setting keys are `<kind>_<image_size>` where kind ∈
`fz, ft, cal, cal_ap, cal_cm, cal_cm_ap` and image_size ∈ `224, 384, 448`
(not every combination exists). The full canonical list is hard-coded in
`summarize_acc.py`, `summarize_cost.py`, and `plot.py`. When adding a new kind
or size, update the list in **all three** plus the relevant `SETTINGS_DIC*` in
`utils` — a stale list silently drops rows in `filter_df` without warning.

`DATASETS_REDUCED = ['aircraft', 'cub', 'soygene', 'soylocal']` is the canonical
4-dataset subset selected by the non-`--filter_og_only` branch of `filter_df`
across scripts. `MODELS_OG` (9 original backbones) gates the
`--filter_og_only` branch. Both constants also exist in `utils`; the local
copies in the summarizers drift independently — change both if you need them
to agree.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl` (xlsx export), `wandb`
(download only). Install on demand into `opencode_env`:

```powershell
conda activate opencode_env; pip install wandb pandas numpy matplotlib seaborn openpyxl
```

## No lint / typecheck / tests

There is no `pytest`, `ruff`, `mypy`, or CI config. Verification = run the
script you touched and check the CSV/xlsx under `results_all/`. Do not add a
test framework unless asked.

## Cross-repo references

The quant runs analysed here come from the sibling TGDA project
(`../../project_2/my_project/TGDA-2` and the `reference/TGDA` snapshot); native
low-bit backbones live in `../ViT_integration` / `../reference/BHViT` /
`BinaryViT` / `GSB-Vision-Transformer` / `SI-BiViT`. If a quant result seems
wrong, the bug is almost always in the upstream `eval_quant.py` config, not
here — verify the `quant_setting` reconstruction here against the original run
in `nycu_pcs/TiT` before suspecting this repo.