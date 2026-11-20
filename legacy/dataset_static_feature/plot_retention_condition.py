"""Draw a compact FGIR / Ultra-FGIR retention contrast for selected HQQ bits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = ROOT / "results/main/retention_level_contrasts.csv"
DEFAULT_PLOT_DIR = ROOT / "results/main/plots"
MODELS = (
    ("beitv2_base_patch16_224_in22k", "BEiTv2"),
    ("swin_base_patch4_window7_224_in22k", "Swin"),
    ("vit_b16", "ViT"),
)
CHECKPOINTS = ("cal", "ft", "fz")


def read_values(table: Path, bits: list[int], group_size: int) -> dict[tuple[int, str, str], float]:
    values: dict[tuple[int, str, str], float] = {}
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bit_width = int(float(row["hqq_nbits"]))
            if bit_width not in bits or int(row["hqq_group_size"]) != group_size:
                continue
            if row["comparable"].lower() != "true" or row["status"] != "ok":
                raise ValueError(f"Selected cell is not comparable: {row}")
            key = (bit_width, row["model_name"], row["ckpt_kind"])
            if key in values:
                raise ValueError(f"Duplicate selected cell: {key}")
            values[key] = float(row["retention_gap"])

    expected = {(bit, model, ckpt) for bit in bits for model, _ in MODELS
                for ckpt in CHECKPOINTS}
    if set(values) != expected:
        raise ValueError(f"Missing cells: {sorted(expected - set(values))}; "
                         f"unexpected cells: {sorted(set(values) - expected)}")
    return values


def draw(table: Path, bits: list[int], group_size: int, output_dir: Path) -> Path:
    values = read_values(table, bits, group_size)
    fig = plt.figure(figsize=(6.8 * len(bits) + 1.3, 5.3), facecolor="white")
    grid = fig.add_gridspec(1, len(bits) + 1, width_ratios=[1] * len(bits) + [0.045],
                            left=0.13 if len(bits) == 1 else 0.09, right=0.91,
                            top=0.78, bottom=0.17, wspace=0.25)
    norm = TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75)
    image = None
    for index, bit in enumerate(bits):
        axis = fig.add_subplot(grid[0, index])
        matrix = [[values[(bit, model, ckpt)] for ckpt in CHECKPOINTS]
                  for model, _ in MODELS]
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="equal")
        axis.set_xticks(range(3), CHECKPOINTS, fontsize=12)
        axis.set_yticks(range(3), [label for _, label in MODELS], fontsize=12)
        axis.tick_params(length=0)
        axis.set_title(f"{bit}-bit", fontsize=15, pad=12)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                axis.text(col_index, row_index, f"{value:+.2f}", ha="center", va="center",
                          fontsize=14, fontweight="bold",
                          color="white" if abs(value) >= 0.42 else "#1f2937")
        for boundary in (0.5, 1.5):
            axis.axhline(boundary, color="white", linewidth=3)
            axis.axvline(boundary, color="white", linewidth=3)
        for spine in axis.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, -1]))
    colorbar.set_label("Ultra-FGIR − FGIR", fontsize=11, labelpad=11)
    colorbar.ax.tick_params(labelsize=10)
    bit_label = ", ".join(str(bit) for bit in bits)
    fig.suptitle(f"Matched-condition accuracy retention  |  group size {group_size}",
                 fontsize=17, fontweight="bold", y=0.96)
    fig.text(0.5, 0.875, f"HQQ {bit_label}-bit · median accuracy_ratio gap",
             ha="center", fontsize=12, color="#475569")
    fig.text(0.5, 0.055, "Negative values mean lower Ultra-FGIR retention",
             ha="center", fontsize=11, color="#475569")

    output_dir.mkdir(parents=True, exist_ok=True)
    bit_slug = "-".join(map(str, bits))
    output = output_dir / f"granularity_retention_{bit_slug}bit_g{group_size}.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": output.name,
        "source_table": str(table.resolve()),
        "source_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "bits": bits,
        "group_size": group_size,
        "metric": "median accuracy_ratio gap (Ultra-FGIR minus FGIR)",
        "cells": [
            {"bits": bit, "model": model, "checkpoint": ckpt,
             "retention_gap": values[(bit, model, ckpt)]}
            for bit in bits for model, _ in MODELS for ckpt in CHECKPOINTS
        ],
    }
    output.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--bits", nargs="+", type=int, default=[3])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()
    bits = list(dict.fromkeys(args.bits))
    print(draw(args.table, bits, args.group_size, args.output_dir))


if __name__ == "__main__":
    main()
