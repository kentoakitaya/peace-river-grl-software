#!/usr/bin/env python3
"""
Plot ECDFs of the public paired pressure differences.

The seven ECDFs are calculated directly from paired_pressure_differences.csv.
Seven panels show individual source-group ECDFs; the lower-right panel is left
empty.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
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

QUANTITY_PRESSURE = "pressure_Pa"
QUANTITY_CHOICES = [QUANTITY_PRESSURE]

DEFAULT_COMPONENT_LABELS: Dict[str, str] = {
    "other_wells": "Remaining wells",
    "remaining_nine_wells": "Remaining wells",
}

DEFAULT_COMPONENT_ORDER: List[str] = [
    "well6",
    "well10",
    "well11",
    "well12",
    "well14",
    "well15",
    "remaining_nine_wells",
]

ECDF_COLOR = "black"

DISPLAY_CONFIG = {
    QUANTITY_PRESSURE: {
        "scale": 1.0,
        "unit": "kPa",
        "xlabel": r"$\Delta p_i$ (kPa)",
        "filename": "paired_difference_ecdf_pressure.pdf",
        "title": None,
    },
}


def load_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config; install pyyaml or omit --config.") from exc

    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"--config must contain a YAML mapping: {path}")
    return data


def config_path(config: Dict[str, object], key: str) -> Optional[Path]:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def default_output_dir_from_config(config: Dict[str, object]) -> Path:
    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is None:
        local_output_dir = REPO_ROOT / "outputs"
    return local_output_dir / "figures" / "paired_component_differences"


def find_liberation_sans_path(font_file: Optional[Path] = None) -> Optional[str]:
    candidates: List[str] = []
    if font_file is not None:
        candidates.append(str(font_file))

    if shutil.which("fc-match") is not None:
        try:
            result = subprocess.run(
                ["fc-match", "-f", "%{file}\n", REQUESTED_FONT_FAMILY],
                check=True,
                capture_output=True,
                text=True,
            )
            path = result.stdout.strip()
            if path:
                candidates.append(path)
        except Exception:
            pass

    try:
        candidates.append(fm.findfont(REQUESTED_FONT_FAMILY, fallback_to_default=False))
    except Exception:
        pass

    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def configure_matplotlib(font_file: Optional[Path] = None) -> str:
    global SELECTED_FONT_FAMILY

    font_path = find_liberation_sans_path(font_file)
    if font_path is not None:
        try:
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            selected_family = font_prop.get_name()
            SELECTED_FONT_FAMILY = selected_family if selected_family else REQUESTED_FONT_FAMILY
        except Exception:
            SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
    else:
        SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [SELECTED_FONT_FAMILY, FALLBACK_FONT_FAMILY, "sans-serif"],
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "mathtext.fontset": "custom",
            "mathtext.rm": SELECTED_FONT_FAMILY,
            "mathtext.it": f"{SELECTED_FONT_FAMILY}:italic",
            "mathtext.bf": f"{SELECTED_FONT_FAMILY}:bold",
            "mathtext.sf": SELECTED_FONT_FAMILY,
            "mathtext.tt": FALLBACK_FONT_FAMILY,
            "mathtext.cal": FALLBACK_FONT_FAMILY,
            "mathtext.fallback": "stixsans",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return SELECTED_FONT_FAMILY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 4 by 2 ECDF panels for paired component differences."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file, such as an untracked copy of config/local_paths.example.yml.",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=None,
        help=(
            "Public paired_pressure_differences.csv. Defaults to the flat sibling "
            "data package."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Only valid when --quantity is not both.",
    )
    parser.add_argument(
        "--quantity",
        choices=QUANTITY_CHOICES,
        default=QUANTITY_PRESSURE,
        help="Quantity to plot. The public file contains pressure only.",
    )
    parser.add_argument(
        "--component-order",
        nargs="*",
        default=None,
        help="Optional explicit component order.",
    )
    parser.add_argument(
        "--label-map",
        nargs="*",
        default=None,
        help="Optional component label mapping as KEY=LABEL pairs.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional common x-axis limits in displayed units.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.6, 9.4),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional path to a local font file.",
    )
    return parser.parse_args()


def parse_label_map(values: Sequence[str] | None) -> Dict[str, str]:
    mapping = dict(DEFAULT_COMPONENT_LABELS)
    if not values:
        return mapping
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --label-map entry: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def prettify_component(name: str, mapping: Dict[str, str]) -> str:
    if name in mapping:
        return mapping[name]
    if name in {"other_wells", "remaining_nine_wells"}:
        return "Remaining wells"
    if name.startswith("well"):
        suffix = name[4:]
        if suffix.isdigit():
            return f"Well {int(suffix)}"
    return name.replace("_", " ").title()


def order_components(available: Iterable[str], explicit_order: Sequence[str] | None) -> List[str]:
    available_list = list(dict.fromkeys(available))
    if explicit_order:
        chosen = [name for name in explicit_order if name in available_list]
        if not chosen:
            raise ValueError("None of the requested --component-order entries were found in the input file")
        return chosen

    chosen: List[str] = []
    for name in DEFAULT_COMPONENT_ORDER:
        if name in available_list and name not in chosen:
            chosen.append(name)
    for name in available_list:
        if name not in chosen:
            chosen.append(name)
    return chosen


def read_ecdf(path: Path) -> pd.DataFrame:
    paired = paired_pressure_data.load_paired_pressure_differences(path)
    out = paired_pressure_data.build_ecdf(paired).rename(
        columns={
            "source_group": "component",
            "paired_difference_kPa": "paired_difference",
        }
    )
    out["quantity"] = QUANTITY_PRESSURE
    return out


def compute_common_xlim(values: np.ndarray, requested_xlim: Sequence[float] | None) -> tuple[float, float]:
    if requested_xlim is not None:
        xmin, xmax = float(requested_xlim[0]), float(requested_xlim[1])
        if xmax <= xmin:
            raise ValueError("--xlim requires XMAX > XMIN")
        return xmin, xmax

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite paired_difference values are available for x-axis limits.")

    xmin = float(np.min(finite))
    xmax = float(np.max(finite))
    if xmin == xmax:
        pad = max(abs(xmin) * 0.05, 1.0)
    else:
        pad = 0.05 * (xmax - xmin)
    xmin -= pad
    xmax += pad

    if xmin > 0.0:
        xmin = 0.0
    if xmax < 0.0:
        xmax = 0.0
    return xmin, xmax


def apply_tick_label_font(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)


def style_ecdf_axis(ax: plt.Axes, xlim: tuple[float, float], show_xlabel: bool, xlabel: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(False)
    ax.axvline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=5,
        width=1.0,
        pad=6,
    )
    ax.tick_params(which="minor", length=3)
    ax.set_xlabel(xlabel if show_xlabel else "", fontfamily=SELECTED_FONT_FAMILY)
    if show_xlabel:
        ax.tick_params(axis="x", labelbottom=True)
        for label in ax.get_xticklabels():
            label.set_visible(True)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    secondary_y = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    secondary_y.set_ylim(0.0, 1.0)
    secondary_y.set_yticks(np.linspace(0.0, 1.0, 6))
    secondary_y.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    secondary_y.set_ylabel("")
    secondary_y.tick_params(
        direction="in",
        which="both",
        right=True,
        labelright=False,
        length=5,
        width=1.0,
        pad=6,
    )
    secondary_y.tick_params(which="minor", length=3)
    apply_tick_label_font(ax)


def add_panel_label(ax: plt.Axes, label: str, component_label: str) -> None:
    ax.text(
        0.03,
        0.94,
        f"{label} {component_label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontfamily=SELECTED_FONT_FAMILY,
    )


def plot_single_component(
    ax: plt.Axes,
    df: pd.DataFrame,
    component: str,
) -> None:
    work = df[df["component"] == component].sort_values("paired_difference")
    if work.empty:
        raise ValueError(f"Component {component} not found for selected quantity")

    x = work["plot_difference"].to_numpy(dtype=float)
    y = work["ecdf_probability"].to_numpy(dtype=float)
    ax.step(x, y, where="post", color=ECDF_COLOR, linewidth=1.8, zorder=3)


def panel_label(index: int) -> str:
    return f"({chr(ord('a') + index)})"


def plot_quantity(
    ecdf: pd.DataFrame,
    quantity: str,
    component_order: Sequence[str],
    label_map: Dict[str, str],
    output: Path,
    figsize: Sequence[float],
    requested_xlim: Sequence[float] | None,
) -> None:
    config = DISPLAY_CONFIG[quantity]
    work = ecdf[ecdf["quantity"] == quantity].copy()
    if work.empty:
        raise ValueError(f"Quantity {quantity} not found in ECDF file")

    work["plot_difference"] = work["paired_difference"] * float(config["scale"])
    component_order = [name for name in component_order if name in set(work["component"])]
    if len(component_order) < 1:
        raise ValueError("No requested components were found for selected quantity")

    xlim = compute_common_xlim(work["plot_difference"].to_numpy(dtype=float), requested_xlim)

    fig, axes = plt.subplots(4, 2, figsize=tuple(figsize), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())

    for idx, component in enumerate(component_order[:7]):
        ax = axes_flat[idx]
        row = idx // 2
        show_xlabel = row == 3 or idx == 5
        style_ecdf_axis(ax, xlim, show_xlabel=show_xlabel, xlabel=str(config["xlabel"]))
        plot_single_component(
            ax,
            work,
            component,
        )
        add_panel_label(ax, panel_label(idx), prettify_component(component, label_map))

    axes_flat[7].axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel(
            "Cumulative fraction\nof realizations (%)",
            fontfamily=SELECTED_FONT_FAMILY,
        )

    title = config.get("title")
    if title:
        fig.suptitle(str(title), y=0.992, fontfamily=SELECTED_FONT_FAMILY, fontsize=13)
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.065,
        top=0.965,
        wspace=0.12,
        hspace=0.12,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    font_file = args.font_file.expanduser().resolve() if args.font_file else None
    selected_font = configure_matplotlib(font_file)

    paired_csv = (args.paired_csv or DEFAULT_PAIRED_CSV).expanduser().resolve()
    output_dir = (args.output_dir or default_output_dir_from_config(config)).resolve()
    ecdf = read_ecdf(paired_csv)

    label_map = parse_label_map(args.label_map)
    available_components = ecdf["component"].dropna().astype(str).tolist()
    component_order = order_components(available_components, args.component_order)

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else output_dir / str(DISPLAY_CONFIG[args.quantity]["filename"])
    )
    plot_quantity(
        ecdf,
        args.quantity,
        component_order,
        label_map,
        output,
        args.figsize,
        args.xlim,
    )

    print(f"Configured font family: {selected_font}")


if __name__ == "__main__":
    main()
