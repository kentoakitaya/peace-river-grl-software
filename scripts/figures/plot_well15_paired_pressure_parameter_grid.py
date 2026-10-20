#!/usr/bin/env python3
"""Plot the Well 15 paired-pressure difference across the diffusion grid.

The script joins the public paired-pressure data to the permeability
design by ensemble member ID. It validates the complete 10 x 10 x 10 design
and the audited Well 15 sign counts before drawing any output.

Example
-------
python scripts/figures/plot_well15_paired_pressure_parameter_grid.py \
    --config config/local_paths.local.yml \
    --output-dir outputs/figures/paired_component_differences
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, NullLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import paired_pressure_data


DATA_PACKAGE_ROOT = (
    REPO_ROOT / "zenodo_release" / "data_package"
    if (REPO_ROOT / "src" / "peace_river_ogs").is_dir()
    else REPO_ROOT.parent / "data_package"
)
DEFAULT_PAIRED_CSV = DATA_PACKAGE_ROOT / "paired_pressure_differences.csv"
DEFAULT_PARAMS_CSV = DATA_PACKAGE_ROOT / "params_design_reference.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "figures" / "paired_component_differences"
DEFAULT_OUTPUT_STEM = "well15_paired_pressure_parameter_grid"

REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY

EXPECTED_CASES = 1000
EXPECTED_AXIS_VALUES = 10
EXPECTED_POSITIVE = 266
EXPECTED_NEGATIVE = 734
EXPECTED_ZERO = 0
EXPECTED_POSITIVE_HIGH_FGZ = 259
EXPECTED_POSITIVE_LEDUC_DISPLAY = ["2.00", "3.34", "5.57", "9.28"]


@dataclass(frozen=True)
class ValidationSummary:
    n_cases: int
    n_positive: int
    n_negative: int
    n_zero: int
    n_positive_high_fgz: int
    minimum_kpa: float
    maximum_kpa: float
    symmetric_limit_kpa: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the 2 by 5 Well 15 paired-pressure parameter grid from "
            "existing ensemble outputs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional local YAML configuration, such as config/local_paths.local.yml.",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=None,
        help=(
            "Public paired_pressure_differences.csv. Defaults to the flat data package."
        ),
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=None,
        help=(
            "Public params_design_reference.csv containing case_id and permeability "
            "axes. Defaults to the flat data package."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for PDF and PNG files.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename without extension.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution; minimum 600.")
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional explicit sans-serif font file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the audit summary without writing figures.",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used.") from exc

    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"--config must contain a YAML mapping: {path}")
    return config


def config_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def resolve_params_csv(argument: Path | None, config: dict[str, object]) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    return DEFAULT_PARAMS_CSV.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_join(paired_csv: Path, params_csv: Path) -> tuple[pd.DataFrame, ValidationSummary]:
    paired = paired_pressure_data.load_paired_pressure_differences(paired_csv)
    params = paired_pressure_data.load_parameter_design(params_csv)
    joined = paired_pressure_data.join_parameter_design(
        paired,
        params,
        source_group="well15",
        add_case_name=True,
    )

    axis_columns = ["leduc_perm_mD", "dz_ratio", "gouge_ratio"]
    axis_values = [np.sort(joined[column].unique()) for column in axis_columns]
    axis_counts = {column: len(values) for column, values in zip(axis_columns, axis_values)}
    if any(count != EXPECTED_AXIS_VALUES for count in axis_counts.values()):
        raise RuntimeError(f"Expected 10 distinct values on every parameter axis: {axis_counts}")
    if (joined[axis_columns] <= 0.0).any().any():
        raise RuntimeError("All three logarithmic parameter axes must be strictly positive.")
    if joined.duplicated(axis_columns).any():
        raise RuntimeError("Duplicate permeability parameter combinations were found.")

    actual_combinations = set(map(tuple, joined[axis_columns].to_numpy()))
    expected_combinations = set(itertools.product(*axis_values))
    if actual_combinations != expected_combinations:
        missing = len(expected_combinations.difference(actual_combinations))
        extra = len(actual_combinations.difference(expected_combinations))
        raise RuntimeError(f"Incomplete Cartesian grid: missing={missing}, unexpected={extra}.")

    differences = joined["paired_difference_kPa"].to_numpy(dtype=float)
    positive = differences > 0.0
    negative = differences < 0.0
    zero = differences == 0.0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    n_zero = int(zero.sum())
    sign_counts = (n_positive, n_negative, n_zero)
    expected_sign_counts = (EXPECTED_POSITIVE, EXPECTED_NEGATIVE, EXPECTED_ZERO)
    if sign_counts != expected_sign_counts:
        raise RuntimeError(
            "Expected positive/negative/zero counts "
            f"{expected_sign_counts}, found {sign_counts}."
        )

    positive_leduc = np.sort(joined.loc[positive, "leduc_perm_mD"].unique())
    positive_leduc_display = [f"{value:.2f}" for value in positive_leduc]
    if positive_leduc_display != EXPECTED_POSITIVE_LEDUC_DISPLAY:
        raise RuntimeError(
            "Positive cases do not occur only at the audited first four Leduc values: "
            f"{positive_leduc_display}"
        )
    if not np.array_equal(positive_leduc, axis_values[0][:4]):
        raise RuntimeError("Positive cases are not restricted to the four lowest Leduc values.")
    if (positive & (joined["leduc_perm_mD"] >= axis_values[0][4])).any():
        raise RuntimeError("A positive case occurs at or above the fifth Leduc value.")

    n_positive_high_fgz = int(
        (positive & (joined["gouge_ratio"].to_numpy(dtype=float) >= 1.0e-4)).sum()
    )
    if n_positive_high_fgz != EXPECTED_POSITIVE_HIGH_FGZ:
        raise RuntimeError(
            "Expected 259 positive cases with r_FGZ >= 1e-4, found "
            f"{n_positive_high_fgz}."
        )

    symmetric_limit_kpa = float(np.max(np.abs(differences)))
    if not math.isfinite(symmetric_limit_kpa) or symmetric_limit_kpa <= 0.0:
        raise RuntimeError("The symmetric colour limit is not finite and positive.")

    summary = ValidationSummary(
        n_cases=len(joined),
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        n_positive_high_fgz=n_positive_high_fgz,
        minimum_kpa=float(np.min(differences)),
        maximum_kpa=float(np.max(differences)),
        symmetric_limit_kpa=symmetric_limit_kpa,
    )
    return joined, summary


def find_font(font_file: Path | None) -> str | None:
    candidates: list[str] = []
    if font_file is not None:
        candidates.append(str(font_file.expanduser()))
    if shutil.which("fc-match") is not None:
        try:
            result = subprocess.run(
                ["fc-match", "-f", "%{file}\n", REQUESTED_FONT_FAMILY],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                candidates.append(result.stdout.strip())
        except Exception:
            pass
    try:
        candidates.append(fm.findfont(REQUESTED_FONT_FAMILY, fallback_to_default=False))
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def configure_matplotlib(font_file: Path | None) -> None:
    global SELECTED_FONT_FAMILY

    font_path = find_font(font_file)
    if font_path is not None:
        try:
            fm.fontManager.addfont(font_path)
            family = fm.FontProperties(fname=font_path).get_name()
            SELECTED_FONT_FAMILY = family or REQUESTED_FONT_FAMILY
        except Exception:
            SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
    else:
        SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [SELECTED_FONT_FAMILY, FALLBACK_FONT_FAMILY, "sans-serif"],
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "mathtext.fontset": "stixsans",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def geometric_edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(values <= 0.0):
        raise ValueError("Logarithmic cell centers must be a positive one-dimensional array.")
    log_values = np.log(values)
    log_edges = np.empty(values.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_values[:-1] + log_values[1:])
    log_edges[0] = log_values[0] - 0.5 * (log_values[1] - log_values[0])
    log_edges[-1] = log_values[-1] + 0.5 * (log_values[-1] - log_values[-2])
    return np.exp(log_edges)


def scientific_math(value: float) -> str:
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    if math.isclose(mantissa, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def concise_tick(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 1.0:
        return f"{value:.3g}"
    return f"{value:.3g}"


def positive_boundary_segments(
    positive: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> list[list[tuple[float, float]]]:
    mask = np.asarray(positive, dtype=bool)
    n_rows, n_columns = mask.shape
    segments: list[list[tuple[float, float]]] = []
    for row in range(n_rows):
        for column in range(n_columns):
            if not mask[row, column]:
                continue
            if column == 0 or not mask[row, column - 1]:
                segments.append(
                    [(x_edges[column], y_edges[row]), (x_edges[column], y_edges[row + 1])]
                )
            if column == n_columns - 1 or not mask[row, column + 1]:
                segments.append(
                    [
                        (x_edges[column + 1], y_edges[row]),
                        (x_edges[column + 1], y_edges[row + 1]),
                    ]
                )
            if row == 0 or not mask[row - 1, column]:
                segments.append(
                    [(x_edges[column], y_edges[row]), (x_edges[column + 1], y_edges[row])]
                )
            if row == n_rows - 1 or not mask[row + 1, column]:
                segments.append(
                    [
                        (x_edges[column], y_edges[row + 1]),
                        (x_edges[column + 1], y_edges[row + 1]),
                    ]
                )
    return segments


def draw_figure(
    joined: pd.DataFrame,
    summary: ValidationSummary,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    font_file: Path | None,
) -> tuple[Path, Path, int, int]:
    if dpi < 600:
        raise RuntimeError("--dpi must be at least 600.")
    configure_matplotlib(font_file)

    leduc_values = np.sort(joined["leduc_perm_mD"].unique())
    fdz_values = np.sort(joined["dz_ratio"].unique())
    fgz_values = np.sort(joined["gouge_ratio"].unique())
    x_edges = geometric_edges(leduc_values)
    y_edges = geometric_edges(fdz_values)

    low_colour = cmc.vik(0.0)
    high_colour = cmc.vik(1.0)
    if not (low_colour[2] > low_colour[0] and high_colour[0] > high_colour[2]):
        raise RuntimeError("cmcrameri vik orientation is not blue-negative/red-positive.")

    norm = TwoSlopeNorm(
        vmin=-summary.symmetric_limit_kpa,
        vcenter=0.0,
        vmax=summary.symmetric_limit_kpa,
    )

    figure = plt.figure(figsize=(8.5, 4.45), facecolor="white")
    grid = GridSpec(
        2,
        6,
        figure=figure,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.065],
        left=0.075,
        right=0.925,
        bottom=0.145,
        top=0.92,
        wspace=0.14,
        hspace=0.30,
    )

    mesh = None
    marker_count = 0
    boundary_segment_count = 0
    tick_indices = np.array([0, 4, 9], dtype=int)
    x_ticks = leduc_values[tick_indices]
    y_ticks = fdz_values[tick_indices]

    for panel_index, fgz_value in enumerate(fgz_values):
        row, column = divmod(panel_index, 5)
        ax = figure.add_subplot(grid[row, column])

        panel = joined.loc[np.isclose(joined["gouge_ratio"], fgz_value)].pivot(
            index="dz_ratio", columns="leduc_perm_mD", values="paired_difference_kPa"
        )
        panel = panel.reindex(index=fdz_values, columns=leduc_values)
        if panel.isna().any().any() or panel.shape != (10, 10):
            raise RuntimeError(f"Incomplete 10 by 10 panel for r_FGZ={fgz_value}.")
        values = panel.to_numpy(dtype=float)
        positive = values > 0.0

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=cmc.vik,
            norm=norm,
            shading="flat",
            antialiased=False,
            rasterized=False,
            zorder=1,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xticks(x_ticks, labels=[concise_tick(value) for value in x_ticks])
        ax.set_yticks(y_ticks, labels=[concise_tick(value) for value in y_ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            labelbottom=row == 1,
            labelleft=column == 0,
            length=2.5,
            width=0.7,
            pad=2.0,
        )
        ax.set_title(
            rf"({chr(ord('a') + panel_index)}) $r_{{\mathrm{{FGZ}}}}={scientific_math(fgz_value)}$",
            loc="left",
            pad=3.0,
        )

        segments = positive_boundary_segments(positive, x_edges, y_edges)
        boundary_segment_count += len(segments)
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors="black",
                    linewidths=0.8,
                    capstyle="butt",
                    joinstyle="miter",
                    zorder=4,
                )
            )

        for row_index, column_index in np.argwhere(positive):
            marker_count += 1
            text = ax.text(
                leduc_values[column_index],
                fdz_values[row_index],
                "+",
                ha="center",
                va="center",
                color="black",
                fontsize=8.0,
                fontweight="bold",
                zorder=5,
                clip_on=True,
            )
            text.set_path_effects(
                [path_effects.Stroke(linewidth=1.5, foreground="white"), path_effects.Normal()]
            )

    if mesh is None:
        raise RuntimeError("No panels were drawn.")
    if marker_count != summary.n_positive:
        raise RuntimeError(
            f"Positive-marker count {marker_count} does not match {summary.n_positive}."
        )

    figure.supxlabel(r"$k_{\mathrm{Leduc}}$ (mD)", x=0.47, y=0.035, fontsize=10.5)
    figure.supylabel(r"$r_{\mathrm{FDZ}}$", x=0.015, y=0.53, fontsize=10.5)

    colourbar_axis = figure.add_subplot(grid[:, 5])
    colourbar = figure.colorbar(mesh, cax=colourbar_axis, orientation="vertical")
    colourbar_ticks = np.array(
        [
            -summary.symmetric_limit_kpa,
            -0.5 * summary.symmetric_limit_kpa,
            0.0,
            0.5 * summary.symmetric_limit_kpa,
            summary.symmetric_limit_kpa,
        ]
    )
    colourbar.set_ticks(colourbar_ticks)
    colourbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    colourbar.ax.tick_params(direction="in", length=2.5, width=0.7, labelsize=9.0)
    colourbar.set_label(r"$\Delta p_{15}^{(m)}$ (kPa)", fontsize=10.5, labelpad=4.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path, marker_count, boundary_segment_count


def print_summary(
    paired_csv: Path,
    params_csv: Path,
    summary: ValidationSummary,
    joined: pd.DataFrame,
) -> None:
    positive = joined["paired_difference_kPa"] > 0.0
    positive_leduc = np.sort(joined.loc[positive, "leduc_perm_mD"].unique())
    fifth_leduc = np.sort(joined["leduc_perm_mD"].unique())[4]
    print(f"paired_csv={paired_csv}")
    print(f"paired_csv_sha256={sha256(paired_csv)}")
    print(f"params_csv={params_csv}")
    print(f"params_csv_sha256={sha256(params_csv)}")
    print(f"cmcrameri_version={version('cmcrameri')}")
    print(f"cases={summary.n_cases}")
    print("axis_cardinality=k_Leduc:10,r_FDZ:10,r_FGZ:10")
    print("grid_combinations=1000,missing=0,duplicates=0")
    print(
        f"sign_counts=positive:{summary.n_positive},negative:{summary.n_negative},zero:{summary.n_zero}"
    )
    print(
        "positive_k_Leduc_mD=" + ",".join(f"{value:.15g}" for value in positive_leduc)
    )
    print(f"no_positive_at_or_above_k_Leduc_mD={fifth_leduc:.15g}")
    print(
        f"positive_with_r_FGZ_ge_1e-4={summary.n_positive_high_fgz}/{summary.n_positive}"
    )
    print(
        "paired_difference_range_kPa="
        f"{summary.minimum_kpa:.15g},{summary.maximum_kpa:.15g}"
    )
    print(f"symmetric_colour_limit_kPa={summary.symmetric_limit_kpa:.15g}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paired_csv = (args.paired_csv or DEFAULT_PAIRED_CSV).expanduser().resolve()
    params_csv = resolve_params_csv(args.params_csv, config)
    joined, summary = validate_and_join(paired_csv, params_csv)
    print_summary(paired_csv, params_csv, summary, joined)

    if args.validate_only:
        print("validation_only=true")
        return 0

    pdf_path, png_path, marker_count, boundary_segment_count = draw_figure(
        joined=joined,
        summary=summary,
        output_dir=args.output_dir.expanduser().resolve(),
        output_stem=args.output_stem,
        dpi=args.dpi,
        font_file=args.font_file,
    )
    print(f"positive_markers_drawn={marker_count}")
    print(f"positive_boundary_segments_drawn={boundary_segment_count}")
    print(f"output_pdf={pdf_path}")
    print(f"output_png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
