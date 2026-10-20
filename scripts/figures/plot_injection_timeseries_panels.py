#!/usr/bin/env python3
"""
Plot monthly injection volumes as stacked bar time series grouped by Layer.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
REPO_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_FIGSIZE = (10.5, 15.0)
REFERENCE_SUBPLOTS = {
    "left": 0.11,
    "right": 0.98,
    "bottom": 0.05,
    "top": 0.98,
    "wspace": 0.22,
    "hspace": 0.20,
}

FIGURE_SUBPLOTS = {
    "left": 0.12,
    "right": 0.76,
    "bottom": 0.08,
    "top": 0.98,
    "hspace": 0.12,
}

VOLUME_SCALE = 1.0e5
Y_LIM = (0.0, 4.0)
Y_TICKS = [0.0, 1.0, 2.0, 3.0, 4.0]
X_LIM = (pd.Timestamp(1975, 1, 1), pd.Timestamp(2024, 12, 1))
X_TICKS = [pd.Timestamp(year, 1, 1) for year in range(1980, 2030, 10) if year <= 2020]

GROUP_SPECS = [
    {
        "title": "Permian, Stoddart, and Rundle (grouped)",
        "layers": {"Permian", "Stoddart, Rundle"},
        "panel_label": "(a)",
    },
    {
        "title": "Wabamun",
        "layers": {"Wabamun"},
        "panel_label": "(b)",
    },
    {
        "title": "Leduc",
        "layers": {"Leduc"},
        "panel_label": "(c)",
    },
]

WELL_COLORS: Dict[str, str] = {
    "Well1": "#4477AA",
    "Well2": "#66CCEE",
    "Well3": "#228833",
    "Well4": "#CCBB44",
    "Well5": "#EE6677",
    "Well6": "#AA3377",
    "Well7": "#BBBBBB",
    "Well8": "#332288",
    "Well9": "#88CCEE",
    "Well10": "#44AA99",
    "Well11": "#117733",
    "Well12": "#999933",
    "Well13": "#DDCC77",
    "Well14": "#CC6677",
    "Well15": "#882255",
}


def repo_root() -> Path:
    return REPO_ROOT


def default_output_path() -> Path:
    return (
        repo_root()
        / "outputs"
        / "figures"
        / "injection"
        / "monthly_injection_stacked_panels.pdf"
    )


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


def output_path_from_config(config: Dict[str, object]) -> Path:
    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is None:
        return default_output_path()
    return local_output_dir / "figures" / "injection" / "monthly_injection_stacked_panels.pdf"


def well_number(tag: str) -> int:
    if not tag.startswith("Well"):
        raise ValueError(f"Unexpected well tag: {tag}")
    return int(tag[4:])


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
            "font.size": 11,
            "axes.linewidth": 1.2,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "mathtext.fontset": "dejavusans",
            "savefig.dpi": 600,
        }
    )
    return SELECTED_FONT_FAMILY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot monthly injection volumes as stacked bar time series grouped by Layer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file, such as an untracked copy of config/local_paths.example.yml.",
    )
    parser.add_argument(
        "--injection-csv",
        type=Path,
        default=None,
        help="Input injection CSV file supplied locally or from an approved archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution in dots per inch. Ignored by vector outputs such as PDF.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional path to a local font file.",
    )
    return parser.parse_args()


def parse_month_label(text: str) -> pd.Timestamp:
    month_str, year_str = text.split("-")
    month = pd.to_datetime(month_str, format="%b").month
    year_2digit = int(year_str)
    year = 1900 + year_2digit if year_2digit >= 70 else 2000 + year_2digit
    return pd.Timestamp(year, month, 1)


def read_injection_csv(path: Path) -> tuple[Dict[str, str], pd.DataFrame]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 7:
        raise ValueError(f"Injection CSV looks too short: {path}")

    tag_row = rows[0]
    layer_row = rows[1]
    date_row = rows[5]
    if not tag_row or tag_row[0] != "Tag":
        raise ValueError("First row must start with 'Tag'")
    if not layer_row or layer_row[0] != "Layer":
        raise ValueError("Second row must start with 'Layer'")
    if not date_row or date_row[0] != "Date":
        raise ValueError("Sixth row must start with 'Date'")

    well_tags = tag_row[1:]
    layer_map = {tag: layer.strip() for tag, layer in zip(well_tags, layer_row[1:])}

    records: List[Dict[str, float | pd.Timestamp]] = []
    for row in rows[6:]:
        if not row or not row[0].strip():
            continue
        record: Dict[str, float | pd.Timestamp] = {"Date": parse_month_label(row[0].strip())}
        for tag, value in zip(well_tags, row[1:]):
            record[tag] = float(value) / VOLUME_SCALE
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No monthly data rows found in {path}")

    df = df.sort_values("Date").reset_index(drop=True)
    return layer_map, df


def month_widths(dates: Sequence[pd.Timestamp]) -> List[float]:
    widths: List[float] = []
    for date in dates:
        next_month = date + pd.offsets.MonthBegin(1)
        widths.append(mdates.date2num(next_month.to_pydatetime()) - mdates.date2num(date.to_pydatetime()))
    return widths


def prettify_well(tag: str) -> str:
    return f"Well {well_number(tag)}"


def apply_tick_label_font(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)


def reference_panel_size() -> tuple[float, float]:
    ref_left = REFERENCE_SUBPLOTS["left"]
    ref_right = REFERENCE_SUBPLOTS["right"]
    ref_bottom = REFERENCE_SUBPLOTS["bottom"]
    ref_top = REFERENCE_SUBPLOTS["top"]
    ref_wspace = REFERENCE_SUBPLOTS["wspace"]
    ref_hspace = REFERENCE_SUBPLOTS["hspace"]
    fig_w, fig_h = REFERENCE_FIGSIZE

    panel_w = fig_w * (ref_right - ref_left) / (2.0 + ref_wspace)
    panel_h = fig_h * (ref_top - ref_bottom) / (7.0 + 6.0 * ref_hspace)
    return panel_w, panel_h


def figure_size_from_reference() -> tuple[float, float]:
    panel_w, panel_h = reference_panel_size()
    left = FIGURE_SUBPLOTS["left"]
    right = FIGURE_SUBPLOTS["right"]
    bottom = FIGURE_SUBPLOTS["bottom"]
    top = FIGURE_SUBPLOTS["top"]
    hspace = FIGURE_SUBPLOTS["hspace"]

    fig_w = panel_w / (right - left)
    fig_h = panel_h * (3.0 + 2.0 * hspace) / (top - bottom)
    return fig_w, fig_h


def wells_for_group(layer_map: Dict[str, str], layers: set[str]) -> List[str]:
    selected = [tag for tag, layer in layer_map.items() if layer in layers]
    if not selected:
        raise ValueError(f"No wells found for layers: {sorted(layers)}")
    return sorted(selected, key=well_number, reverse=True)


def style_axis(ax: plt.Axes, show_xticklabels: bool) -> None:
    ax.set_xlim(X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks(X_TICKS)
    if show_xticklabels:
        ax.set_xticklabels([str(tick.year) for tick in X_TICKS])
        ax.set_xlabel("Year", fontfamily=SELECTED_FONT_FAMILY)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=4,
        width=0.9,
        pad=5,
    )
    apply_tick_label_font(ax)


def add_panel_header(ax: plt.Axes, panel_label: str, title: str) -> None:
    ax.text(
        0.02,
        0.94,
        f"{panel_label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontfamily=SELECTED_FONT_FAMILY,
    )


def add_group_legend(ax: plt.Axes, wells_descending: Sequence[str]) -> None:
    legend_wells = sorted(wells_descending, key=well_number)
    handles = [
        Patch(facecolor=WELL_COLORS[tag], edgecolor="none", label=prettify_well(tag))
        for tag in legend_wells
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
    )
    for text in legend.get_texts():
        text.set_fontfamily(SELECTED_FONT_FAMILY)


def plot_group_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    wells_descending: Sequence[str],
    panel_label: str,
    title: str,
    show_xticklabels: bool,
) -> None:
    x = mdates.date2num([date.to_pydatetime() for date in df["Date"]])
    widths = month_widths(df["Date"].tolist())
    bottom = pd.Series(0.0, index=df.index)

    for tag in wells_descending:
        values = df[tag].astype(float)
        ax.bar(
            x,
            values,
            width=widths,
            bottom=bottom.to_numpy(),
            align="edge",
            color=WELL_COLORS[tag],
            edgecolor="none",
            linewidth=0.0,
        )
        bottom = bottom + values

    style_axis(ax, show_xticklabels=show_xticklabels)
    add_panel_header(ax, panel_label, title)
    add_group_legend(ax, wells_descending)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    font_file = args.font_file.expanduser().resolve() if args.font_file else None
    selected_font = configure_matplotlib(font_file)

    injection_csv_arg = args.injection_csv or config_path(config, "injection_csv")
    if injection_csv_arg is None:
        raise RuntimeError("--injection-csv is required unless --config provides injection_csv.")

    injection_csv = injection_csv_arg.resolve()
    output = (args.output or output_path_from_config(config)).resolve()
    if not injection_csv.exists():
        raise FileNotFoundError(f"Input injection CSV not found: {injection_csv}")

    layer_map, df = read_injection_csv(injection_csv)
    figsize = figure_size_from_reference()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

    for index, (ax, spec) in enumerate(zip(axes, GROUP_SPECS)):
        wells = wells_for_group(layer_map, spec["layers"])
        plot_group_panel(
            ax=ax,
            df=df,
            wells_descending=wells,
            panel_label=spec["panel_label"],
            title=spec["title"],
            show_xticklabels=index == len(GROUP_SPECS) - 1,
        )

    axes[1].set_ylabel(
        r"Monthly injection volume ($10^5$ m$^3$/month)",
        fontfamily=SELECTED_FONT_FAMILY,
    )

    fig.subplots_adjust(
        left=FIGURE_SUBPLOTS["left"],
        right=FIGURE_SUBPLOTS["right"],
        bottom=FIGURE_SUBPLOTS["bottom"],
        top=FIGURE_SUBPLOTS["top"],
        hspace=FIGURE_SUBPLOTS["hspace"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    for spec in GROUP_SPECS:
        wells = wells_for_group(layer_map, spec["layers"])
        print(f"{spec['title']}: {', '.join(prettify_well(tag) for tag in sorted(wells, key=well_number))}")
    print(f"Saved figure to {output}")
    print(f"Configured font family: {selected_font}")


if __name__ == "__main__":
    main()
