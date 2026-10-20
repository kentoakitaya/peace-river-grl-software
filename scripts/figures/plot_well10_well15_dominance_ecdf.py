#!/usr/bin/env python3
"""Plot public paired-pressure ECDFs for Wells 15 and 10.

Manuscript panels:
    (a) Well 15 paired pressure difference
    (b) Well 10 paired pressure difference

The public input contains the November 2022 pressure differences in kPa.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import paired_pressure_data


REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
FIGURE_FONT_SIZE = 12
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PACKAGE_ROOT = (
    REPO_ROOT / "zenodo_release" / "data_package"
    if (REPO_ROOT / "src" / "peace_river_ogs").is_dir()
    else REPO_ROOT.parent / "data_package"
)
DEFAULT_PAIRED_CSV = DATA_PACKAGE_ROOT / "paired_pressure_differences.csv"

COMPONENT_CONFIG = {
    "well10": {
        "label": "Well 10",
        "color": "#0072B2",
        "linestyle": "--",
    },
    "well15": {
        "label": "Well 15",
        "color": "#D55E00",
        "linestyle": "-",
    },
}

QUANTITY_PRESSURE = "pressure_Pa"
EXPECTED_MANUSCRIPT_PRESSURE_FRACTIONS = {
    "well15": 0.266,
    "well10": 0.679,
}

QUANTITY_CONFIG = {
    QUANTITY_PRESSURE: {
        "scale": 1.0,
        "xlabel": r"Paired pressure difference, $\Delta p_i$ (kPa)",
        "column_title": "Pressure",
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


def default_output_path(config: Dict[str, object]) -> Path:
    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is None:
        local_output_dir = REPO_ROOT / "outputs"
    return (
        local_output_dir
        / "figures"
        / "paired_component_differences"
        / "well10_well15_dominance_ecdf.pdf"
    )


def find_liberation_sans_path(font_file: Optional[Path] = None) -> Optional[str]:
    candidates: list[str] = []
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
            "font.size": FIGURE_FONT_SIZE,
            "axes.linewidth": 1.2,
            "axes.labelsize": FIGURE_FONT_SIZE,
            "axes.titlesize": FIGURE_FONT_SIZE,
            "xtick.labelsize": FIGURE_FONT_SIZE,
            "ytick.labelsize": FIGURE_FONT_SIZE,
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


def use_font_family(font_family: str) -> None:
    """Use an already configured family when panels are drawn in another figure."""

    global SELECTED_FONT_FAMILY
    SELECTED_FONT_FAMILY = font_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Well 15 and Well 10 ECDF dominance-fraction figures."
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
        "--output",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(8.6, 6.2),
        metavar=("W", "H"),
        help="Manuscript figure size in inches.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional path to a local font file.",
    )
    return parser.parse_args()


def read_dataset(
    paired_csv: Path,
    quantities: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quantities is not None and set(quantities) != {QUANTITY_PRESSURE}:
        raise ValueError(
            "The public paired CSV contains pressure only; request pressure_Pa only."
        )
    paired = paired_pressure_data.load_paired_pressure_differences(paired_csv)
    ecdf = paired_pressure_data.build_ecdf(paired).rename(
        columns={
            "source_group": "component",
            "paired_difference_kPa": "paired_difference",
        }
    )
    summary = paired_pressure_data.sign_summary(paired).rename(
        columns={"source_group": "component"}
    )
    ecdf["quantity"] = QUANTITY_PRESSURE
    summary["quantity"] = QUANTITY_PRESSURE
    return ecdf, summary


def select_ecdf(
    ecdf: pd.DataFrame,
    component: str,
    quantity: str,
    scale: float,
) -> pd.DataFrame:
    out = ecdf[
        (ecdf["component"] == component) & (ecdf["quantity"] == quantity)
    ].copy()
    if out.empty:
        raise ValueError(f"No ECDF rows found for component={component}, quantity={quantity}")
    out = out[np.isfinite(out["paired_difference"]) & np.isfinite(out["ecdf_probability"])].copy()
    out = out.sort_values("paired_difference")
    out["plot_difference"] = out["paired_difference"] * scale
    return out


def select_summary(summary: pd.DataFrame, component: str, quantity: str) -> pd.Series:
    rows = summary[
        (summary["component"] == component) & (summary["quantity"] == quantity)
    ]
    if rows.empty:
        raise ValueError(f"No summary row found for component={component}, quantity={quantity}")
    if len(rows) != 1:
        raise ValueError(
            f"Expected one summary row for component={component}, quantity={quantity}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def compute_column_xlim(
    panel_data: Dict[tuple[str, str], pd.DataFrame],
    quantity: str,
) -> tuple[float, float]:
    values = [
        panel_data[(component, quantity)]["plot_difference"].to_numpy(dtype=float)
        for component in COMPONENT_CONFIG
    ]
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for x-axis limits for {quantity}")

    xmin = min(float(np.min(finite)), 0.0)
    xmax = max(float(np.max(finite)), 0.0)
    if xmin == xmax:
        pad = max(abs(xmin) * 0.05, 0.01)
    else:
        pad = 0.05 * (xmax - xmin)
    return xmin - pad, xmax + pad


def load_manuscript_pressure_panels(
    paired_csv: Path,
) -> tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.Series],
    tuple[float, float],
]:
    """Load and validate the pressure-only inputs shared by manuscript figures."""

    ecdf, summary = read_dataset(paired_csv, quantities=(QUANTITY_PRESSURE,))
    components = ("well15", "well10")
    panel_data = {
        component: select_ecdf(
            ecdf,
            component,
            QUANTITY_PRESSURE,
            float(QUANTITY_CONFIG[QUANTITY_PRESSURE]["scale"]),
        )
        for component in components
    }
    summary_data = {
        component: select_summary(summary, component, QUANTITY_PRESSURE)
        for component in components
    }
    for component, expected in EXPECTED_MANUSCRIPT_PRESSURE_FRACTIONS.items():
        actual = float(summary_data[component]["fraction_positive"])
        if not np.isclose(actual, expected, rtol=0.0, atol=5.0e-13):
            raise ValueError(
                "Pressure dominance fraction does not match the required manuscript "
                f"value for {component}: actual={actual:.17g}, expected={expected:.17g}"
            )

    panel_data_by_quantity = {
        (component, QUANTITY_PRESSURE): data
        for component, data in panel_data.items()
    }
    xlim = compute_column_xlim(panel_data_by_quantity, QUANTITY_PRESSURE)
    return panel_data, summary_data, xlim


def apply_tick_label_font(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)


def style_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    show_ylabel: bool,
    show_xlabel: bool,
    xlabel: str,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(False)
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
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(
        "Cumulative fraction of realizations (%)" if show_ylabel else "",
        fontfamily=SELECTED_FONT_FAMILY,
    )
    apply_tick_label_font(ax)


def finite_support_ecdf_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return an ECDF polyline that begins and ends at the observed support."""
    support = (
        df.groupby("plot_difference", sort=True, as_index=False)["ecdf_probability"]
        .max()
        .sort_values("plot_difference")
    )
    x_values = support["plot_difference"].to_numpy(dtype=float)
    y_values = support["ecdf_probability"].to_numpy(dtype=float)
    previous_y = np.concatenate(([0.0], y_values[:-1]))
    x = np.repeat(x_values, 2)
    y = np.column_stack((previous_y, y_values)).ravel()
    return x, y


def add_dominance_annotation(
    ax: plt.Axes,
    summary_row: pd.Series,
    xlim: tuple[float, float],
    fraction_text_prefix: Optional[str] = None,
) -> None:
    fraction_positive = float(summary_row["fraction_positive"])
    intersection_y = float(summary_row["fraction_negative"]) + float(
        summary_row["fraction_zero"]
    )

    xmin, xmax = xlim
    xspan = xmax - xmin
    arrow_x = max(xmin + 0.04 * xspan, -0.025 * xspan)

    if fraction_positive > 0.0:
        arrow = FancyArrowPatch(
            (arrow_x, intersection_y),
            (arrow_x, 1.0),
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=1.0,
            color="black",
            zorder=5,
        )
        ax.add_patch(arrow)

    if arrow_x > xmin + 0.60 * xspan:
        text_x = arrow_x - 0.035 * xspan
        horizontal_alignment = "right"
    else:
        text_x = arrow_x + 0.035 * xspan
        horizontal_alignment = "left"

    text_y = min(
        max(intersection_y + 0.5 * max(fraction_positive, 0.055), 0.08),
        0.95,
    )
    fraction_text = f"{100.0 * fraction_positive:.1f}%"
    if fraction_text_prefix:
        fraction_text = f"{fraction_text_prefix}\n{fraction_text}"
    ax.text(
        text_x,
        text_y,
        fraction_text,
        ha=horizontal_alignment,
        va="center",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
        zorder=6,
    )


def draw_panel(
    ax: plt.Axes,
    ecdf_by_component: Dict[str, pd.DataFrame],
    summary_by_component: Dict[str, pd.Series],
    xlim: tuple[float, float],
    panel_label: str,
    show_ylabel: bool,
    show_xlabel: bool,
    xlabel: str,
    show_dominance_arrow: bool = False,
    show_fraction_text: bool = True,
    line_color: Optional[str] = None,
    line_style: Optional[str] = None,
    fraction_text_prefix: Optional[str] = None,
) -> None:
    style_axis(ax, xlim, show_ylabel=show_ylabel, show_xlabel=show_xlabel, xlabel=xlabel)

    for zorder, (component, ecdf) in enumerate(ecdf_by_component.items(), start=2):
        component_config = COMPONENT_CONFIG[component]
        x, y = finite_support_ecdf_arrays(ecdf)
        ax.plot(
            x,
            y,
            color=line_color or str(component_config["color"]),
            linestyle=line_style or str(component_config["linestyle"]),
            linewidth=2.0,
            zorder=zorder,
        )

    ax.axvline(0.0, color="0.40", linewidth=1.0, linestyle="--", zorder=1)

    for component, summary_row in summary_by_component.items():
        component_config = COMPONENT_CONFIG[component]
        boundary_ecdf = float(summary_row["fraction_negative"]) + float(
            summary_row["fraction_zero"]
        )
        ax.plot(
            [0.0],
            [boundary_ecdf],
            linestyle="none",
            marker="_",
            markersize=9,
            markeredgewidth=2.0,
            color=line_color or str(component_config["color"]),
            clip_on=False,
            zorder=6,
        )

    if show_dominance_arrow:
        if len(summary_by_component) != 1:
            raise ValueError("A dominance arrow requires exactly one component per panel")
        add_dominance_annotation(
            ax,
            next(iter(summary_by_component.values())),
            xlim,
            fraction_text_prefix=fraction_text_prefix,
        )

    if show_fraction_text:
        fraction_text = "\n".join(
            ["Dominance fraction"]
            + [
                f"{COMPONENT_CONFIG[component]['label']}  "
                f"{100.0 * float(summary_row['fraction_positive']):.1f}%"
                for component, summary_row in summary_by_component.items()
            ]
        )
        ax.text(
            0.04,
            0.84,
            fraction_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FIGURE_FONT_SIZE - 1,
            fontfamily=SELECTED_FONT_FAMILY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            zorder=5,
        )

    if panel_label:
        ax.text(
            0.03,
            0.94,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    font_file = args.font_file.expanduser().resolve() if args.font_file else None
    selected_font = configure_matplotlib(font_file)

    requested_quantities = (QUANTITY_PRESSURE,)

    paired_csv = (args.paired_csv or DEFAULT_PAIRED_CSV).expanduser().resolve()
    ecdf, summary = read_dataset(paired_csv, quantities=requested_quantities)

    panel_data: Dict[tuple[str, str], pd.DataFrame] = {}
    summary_data: Dict[tuple[str, str], pd.Series] = {}
    for component in COMPONENT_CONFIG:
        for quantity in requested_quantities:
            quantity_config = QUANTITY_CONFIG[quantity]
            panel_data[(component, quantity)] = select_ecdf(
                ecdf,
                component,
                quantity,
                float(quantity_config["scale"]),
            )
            summary_data[(component, quantity)] = select_summary(
                summary,
                component,
                quantity,
            )

    pressure_xlim = compute_column_xlim(panel_data, QUANTITY_PRESSURE)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=tuple(args.figsize),
        sharex=True,
        sharey=True,
    )

    panel_specs = [
        (0, "well15", "(a)", False),
        (1, "well10", "(b)", True),
    ]
    for row, component, panel_label, show_xlabel in panel_specs:
        draw_panel(
            axes[row],
            {component: panel_data[(component, QUANTITY_PRESSURE)]},
            {component: summary_data[(component, QUANTITY_PRESSURE)]},
            pressure_xlim,
            panel_label,
            show_ylabel=False,
            show_xlabel=show_xlabel,
            xlabel=str(QUANTITY_CONFIG[QUANTITY_PRESSURE]["xlabel"]),
            show_dominance_arrow=True,
            show_fraction_text=False,
            line_color="black",
            line_style="-",
            fraction_text_prefix=(
                "Dominance fraction" if panel_label == "(a)" else None
            ),
        )

    fig.text(
        0.085,
        0.50,
        "Cumulative fraction of realizations (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
    )
    fig.text(
        0.045,
        0.715,
        "Well 15",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
    )
    fig.text(
        0.045,
        0.285,
        "Well 10",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
    )

    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        bottom=0.12,
        top=0.98,
        hspace=0.12,
    )

    output = (args.output or default_output_path(config)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")
    print(f"Configured font family: {selected_font}")


if __name__ == "__main__":
    main()
