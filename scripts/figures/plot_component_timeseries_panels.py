#!/usr/bin/env python3
"""Plot the Supporting Information pressure histories as a 4 by 2 panel figure."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


DEFAULT_COMPONENT_LABELS: Dict[str, str] = {
    "other_wells": "Remaining nine wells",
    "remaining_nine_wells": "Remaining nine wells",
    "all_wells": "All-wells case",
}

DEFAULT_COMPONENT_ORDER: List[str] = [
    "all_wells",
    "well6",
    "well10",
    "well11",
    "well12",
    "well14",
    "well15",
    "remaining_nine_wells",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIRNAME = "well_component_contribution"
DEFAULT_OUTPUT_NAME = "component_timeseries_panels_pressure.pdf"

YEAR_TICKS = [2, 122, 242, 362, 482]
YEAR_TICKLABELS = ["1980", "1990", "2000", "2010", "2020"]

REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY

BAND_COLOR = "0.85"
LINE_COLOR = "black"

YLABEL_PRESSURE = r"$\langle p \rangle(t)$ (kPa)"


def find_liberation_sans_path() -> str | None:
    candidates: List[str] = []

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

    common_paths = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    ]
    candidates.extend(common_paths)

    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def configure_matplotlib() -> str:
    global SELECTED_FONT_FAMILY

    font_path = find_liberation_sans_path()
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
            "font.size": 11,
            "axes.linewidth": 1.2,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "mathtext.fontset": "dejavusans",
        }
    )
    return SELECTED_FONT_FAMILY


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


def resolve_analysis_dir(
    analysis_dir_arg: Optional[Path],
    config: Dict[str, object],
) -> Path:
    if analysis_dir_arg is not None:
        return analysis_dir_arg.expanduser().resolve()

    postprocessed_dir = config_path(config, "ogs_results_postprocessed_dir")
    if postprocessed_dir is not None:
        return (postprocessed_dir / DEFAULT_ANALYSIS_DIRNAME).resolve()

    raise RuntimeError(
        "--analysis-dir is required unless --config provides ogs_results_postprocessed_dir."
    )


def resolve_output_path(output_arg: Optional[Path], config: Dict[str, object]) -> Path:
    if output_arg is not None:
        return output_arg.expanduser().resolve()

    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is None:
        local_output_dir = REPO_ROOT / "outputs"
    return (local_output_dir / "figures" / "component_contribution" / DEFAULT_OUTPUT_NAME).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pressure histories for the all-wells case and seven source groups "
            "as a 4 by 2 panel figure."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file, such as an untracked copy of config/local_paths.example.yml.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Q_timeseries_pressure.csv or "
            "Q_timeseries_component_pressure.csv."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output figure path. Defaults to local_output_dir/figures/component_contribution/ "
            "when --config provides local_output_dir, otherwise outputs/figures/component_contribution/."
        ),
    )
    parser.add_argument(
        "--label-map",
        nargs="*",
        default=None,
        help="Optional component label mapping as KEY=LABEL pairs.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.72, 8.90),
        metavar=("W", "H"),
        help=(
            "Figure size in inches. The default preserves the approximately "
            "4.2 by 1.8 inch panel size of the previous 4 by 2 figures."
        ),
    )
    parser.add_argument(
        "--panel-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help=(
            "Optional size in inches for each panel. "
            "When provided, the total figure size is computed from the number of rows and columns."
        ),
    )
    return parser.parse_args()


def required_pressure_file(analysis_dir: Path) -> Path:
    candidates = [
        analysis_dir / "Q_timeseries_pressure.csv",
        analysis_dir / "Q_timeseries_component_pressure.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required pressure CSV not found; expected {expected}")


def read_pressure_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "component" not in df.columns:
        raise ValueError(f"Missing required column 'component' in pressure CSV: {path}")
    out = df.copy()
    out["component"] = out["component"].astype(str).replace(
        {"other_wells": "remaining_nine_wells"}
    )
    return out


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
        return "Remaining nine wells"
    if name.startswith("well"):
        suffix = name[4:]
        if suffix.isdigit():
            return f"Well {int(suffix)}"
    return name.replace("_", " ").title()


def validate_required_components(df: pd.DataFrame, path: Path) -> None:
    available = set(df["component"].dropna().astype(str))
    missing = [name for name in DEFAULT_COMPONENT_ORDER if name not in available]
    if missing:
        raise ValueError(
            f"Pressure CSV {path} is missing required cases: {', '.join(missing)}. "
            f"Required cases are: {', '.join(DEFAULT_COMPONENT_ORDER)}"
        )


def to_kpa(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col] / 1000.0
    return out


def prepare_component(df: pd.DataFrame, component: str) -> pd.DataFrame:
    required = ["component", "time_month", "q10", "q50", "q90"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    out = df[df["component"].astype(str) == component].copy()
    if out.empty:
        raise ValueError(f"Component {component} not found in dataframe")
    out = out.sort_values("time_month")
    return to_kpa(out, ["q10", "q50", "q90", "mean", "std"])


def apply_tick_label_font(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)


def style_time_axis(ax: plt.Axes, show_xticklabels: bool) -> None:
    ax.set_xlim(2, 540)
    ax.set_xticks(YEAR_TICKS)
    if show_xticklabels:
        ax.set_xticklabels(YEAR_TICKLABELS)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(False)
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        length=4,
        width=0.9,
        pad=5,
    )
    ax.tick_params(which="minor", length=2)


def draw_component_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    panel_label: str,
    component_label: str,
    show_xticklabels: bool,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: Sequence[float],
) -> None:
    x = df["time_month"].to_numpy(dtype=float)
    q10 = df["q10"].to_numpy(dtype=float)
    q50 = df["q50"].to_numpy(dtype=float)
    q90 = df["q90"].to_numpy(dtype=float)

    ax.fill_between(x, q10, q90, color=BAND_COLOR, edgecolor="none", zorder=1)
    ax.plot(x, q50, color=LINE_COLOR, linewidth=1.8, zorder=2)
    style_time_axis(ax, show_xticklabels=show_xticklabels)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.tick_params(axis="y", labelleft=True)
    ax.set_xlabel(xlabel, fontfamily=SELECTED_FONT_FAMILY)
    ax.set_ylabel(ylabel, fontfamily=SELECTED_FONT_FAMILY)
    ax.text(
        0.02,
        0.92,
        f"{panel_label} {component_label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="normal",
        fontfamily=SELECTED_FONT_FAMILY,
    )
    apply_tick_label_font(ax)


def main() -> None:
    selected_font = configure_matplotlib()
    args = parse_args()
    config = load_config(args.config)

    analysis_dir = resolve_analysis_dir(args.analysis_dir, config)
    output = resolve_output_path(args.output, config)
    pressure_file = required_pressure_file(analysis_dir)
    pressure_df = read_pressure_csv(pressure_file)
    validate_required_components(pressure_df, pressure_file)

    label_map = parse_label_map(args.label_map)
    pressure_rows = [prepare_component(pressure_df, comp) for comp in DEFAULT_COMPONENT_ORDER]
    labels = [prettify_component(comp, label_map) for comp in DEFAULT_COMPONENT_ORDER]

    n_rows = 4
    n_cols = 2
    if args.panel_size is not None:
        panel_w, panel_h = float(args.panel_size[0]), float(args.panel_size[1])
        # Match the fixed subplot adjustment below so panel sizes stay consistent
        left, right, bottom, top = 0.11, 0.98, 0.05, 0.98
        wspace, hspace = 0.22, 0.20
        fig_w = panel_w * (n_cols + wspace * (n_cols - 1)) / (right - left)
        fig_h = panel_h * (n_rows + hspace * (n_rows - 1)) / (top - bottom)
        figsize = (fig_w, fig_h)
    else:
        figsize = tuple(args.figsize)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, (row, label) in enumerate(zip(pressure_rows, labels)):
        show_x = i // n_cols == n_rows - 1
        draw_component_panel(
            axes.flat[i],
            row,
            f"({chr(ord('a') + i)})",
            label,
            show_xticklabels=show_x,
            xlabel="Year" if show_x else "",
            ylabel=YLABEL_PRESSURE,
            ylim=(-30.0, 240.0),
            yticks=[0.0, 60.0, 120.0, 180.0, 240.0],
        )

    handles = [
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.8, label="Median"),
        Patch(facecolor=BAND_COLOR, edgecolor="none", label="10th to 90th percentile"),
    ]
    legend = axes[0, 1].legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
    )
    for text in legend.get_texts():
        text.set_fontfamily(SELECTED_FONT_FAMILY)

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.05,
        top=0.98,
        wspace=0.22,
        hspace=0.20,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")
    print(f"Configured font family: {selected_font}")


if __name__ == "__main__":
    main()
