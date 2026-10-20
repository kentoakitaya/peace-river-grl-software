#!/usr/bin/env python3
"""
Compute paired component-vs-rest differences for fault responses.

The canonical default evaluates the November 2022 monthly output at
``time_month = 516.0``. Custom inclusive windows remain available through the
command-line arguments.

For each diffusion realization and decomposition component, the script computes

    Delta_i = mean_window(component_i) - mean_window(all_wells - component_i)

for pressure and pressure rate. For the canonical single-output run,
``mean_window`` is the value at the November 2022 output. Positive values
indicate realizations in which the selected component exceeds the combined
contribution from all other components at the selected output or window.

Input CSV format
    case,time_month,pressure_Pa,pressure_rate_Pa_per_month
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "case",
    "time_month",
    "pressure_Pa",
    "pressure_rate_Pa_per_month",
]

TIME_COLUMN = "time_month"
ALL_WELLS_FILE = "fault_pressure_timeseries_all_wells.csv"
FILE_GLOB = "fault_pressure_timeseries_*.csv"
DEFAULT_OUTPUT_DIRNAME = "paired_component_differences"
REPO_ROOT = Path(__file__).resolve().parents[3]

LEGACY_OTHER_WELLS_COMPONENT = "other_wells"
REMAINING_NINE_WELLS_COMPONENT = "remaining_nine_wells"
ALL_WELLS_COMPONENT = "all_wells"

QUANTILES = (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired differences and ECDFs comparing each well component "
            "with the sum of all other components. The canonical default is the "
            "November 2022 monthly output at time_month 516."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file, such as an untracked copy of config/local_paths.example.yml.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing fault_pressure_timeseries_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate output CSV files.",
    )
    parser.add_argument(
        "--window-start-time",
        type=float,
        default=516.0,
        help=f"Inclusive analysis-window start {TIME_COLUMN}. Canonical default: 516.",
    )
    parser.add_argument(
        "--window-end-time",
        type=float,
        default=516.0,
        help=f"Inclusive analysis-window end {TIME_COLUMN}. Canonical default: 516.",
    )
    parser.add_argument(
        "--case-summary-csv",
        type=Path,
        default=None,
        help="Optional case summary table containing case_name and leduc_perm_mD for subset filtering.",
    )
    parser.add_argument(
        "--leduc-min",
        type=float,
        default=None,
        help="Optional minimum Leduc permeability in mD for included cases.",
    )
    parser.add_argument(
        "--leduc-max",
        type=float,
        default=None,
        help="Optional maximum Leduc permeability in mD for included cases.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=50,
        help="Minimum number of finite paired differences required for distribution summaries.",
    )
    return parser.parse_args()


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


def default_input_dir_from_config(config: Dict[str, object]) -> Optional[Path]:
    postprocessed_dir = config_path(config, "ogs_results_postprocessed_dir")
    if postprocessed_dir is None:
        return None
    return postprocessed_dir / "pressure_timeseries"


def default_output_dir_from_config(config: Dict[str, object]) -> Path:
    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is None:
        local_output_dir = REPO_ROOT / "outputs"
    return local_output_dir / "analysis" / DEFAULT_OUTPUT_DIRNAME


def normalize_component_name(name: str) -> str:
    if name == LEGACY_OTHER_WELLS_COMPONENT:
        return REMAINING_NINE_WELLS_COMPONENT
    return name


def component_sort_key(name: str) -> Tuple[int, int | float, str]:
    name = normalize_component_name(name)
    match = re.fullmatch(r"well(\d+)", name)
    if match:
        return (1, int(match.group(1)), name)
    if name == REMAINING_NINE_WELLS_COMPONENT:
        return (2, math.inf, name)
    if name == ALL_WELLS_COMPONENT:
        return (3, math.inf, name)
    return (4, math.inf, name)


def component_name_from_path(path: Path) -> str:
    prefix = "fault_pressure_timeseries_"
    name = path.stem
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected file name: {path.name}")
    return normalize_component_name(name[len(prefix) :])


def discover_input_files(input_dir: Path) -> Tuple[Path, List[Path]]:
    csv_files = sorted(input_dir.glob(FILE_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"No files matching {FILE_GLOB} were found in {input_dir}")

    all_path = input_dir / ALL_WELLS_FILE
    if not all_path.exists():
        raise FileNotFoundError(f"Required file not found: {all_path}")

    component_paths = [p for p in csv_files if p.name != ALL_WELLS_FILE]
    if not component_paths:
        raise FileNotFoundError("No component CSV files were found.")

    component_paths = sorted(component_paths, key=lambda p: component_sort_key(component_name_from_path(p)))
    return all_path, component_paths


def read_and_pivot_csv(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    work = df[["case", TIME_COLUMN, value_col]].copy()
    if work.duplicated(subset=["case", TIME_COLUMN]).any():
        dup = work.loc[work.duplicated(subset=["case", TIME_COLUMN], keep=False)]
        raise ValueError(
            f"Duplicate case and {TIME_COLUMN} rows found in {path}. Example rows:\n{dup.head()}"
        )

    work[TIME_COLUMN] = pd.to_numeric(work[TIME_COLUMN], errors="coerce")
    if work[TIME_COLUMN].isna().any():
        raise ValueError(f"Non numeric values found in {TIME_COLUMN} in {path}")

    pivot = work.pivot(index="case", columns=TIME_COLUMN, values=value_col)
    return pivot.sort_index().sort_index(axis=1)


def align_frames(named_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_cases: Optional[pd.Index] = None
    common_times: Optional[pd.Index] = None

    for frame in named_frames.values():
        common_cases = frame.index if common_cases is None else common_cases.intersection(frame.index)
        common_times = frame.columns if common_times is None else common_times.intersection(frame.columns)

    assert common_cases is not None
    assert common_times is not None
    if len(common_cases) == 0:
        raise ValueError("No common cases across all CSV files.")
    if len(common_times) == 0:
        raise ValueError(f"No common {TIME_COLUMN} values across all CSV files.")

    return {name: frame.loc[common_cases, common_times].copy() for name, frame in named_frames.items()}


def load_selected_cases(case_summary_csv: Path, leduc_min: float | None, leduc_max: float | None) -> set[str]:
    df = pd.read_csv(case_summary_csv)
    if "case_name" not in df.columns or "leduc_perm_mD" not in df.columns:
        raise ValueError("case summary csv must contain case_name and leduc_perm_mD")

    leduc = pd.to_numeric(df["leduc_perm_mD"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if leduc_min is not None:
        mask &= leduc >= leduc_min
    if leduc_max is not None:
        mask &= leduc <= leduc_max

    selected = set(df.loc[mask, "case_name"].dropna().astype(str).tolist())
    if not selected:
        if leduc_min is None:
            raise ValueError(f"No cases satisfy leduc_perm_mD <= {leduc_max}")
        if leduc_max is None:
            raise ValueError(f"No cases satisfy leduc_perm_mD >= {leduc_min}")
        raise ValueError(f"No cases satisfy {leduc_min} <= leduc_perm_mD <= {leduc_max}")
    return selected


def maybe_filter_cases(
    frames: Dict[str, pd.DataFrame],
    case_summary_csv: Path | None,
    leduc_min: float | None,
    leduc_max: float | None,
) -> Dict[str, pd.DataFrame]:
    if case_summary_csv is None:
        return frames

    selected_cases = load_selected_cases(case_summary_csv, leduc_min, leduc_max)
    common_selected_cases = frames[ALL_WELLS_COMPONENT].index.intersection(pd.Index(sorted(selected_cases)))
    if len(common_selected_cases) == 0:
        raise ValueError("No selected cases were found in the pressure timeseries tables.")
    return {name: frame.loc[common_selected_cases].copy() for name, frame in frames.items()}


def compute_window_average(frame: pd.DataFrame, start_time: float, end_time: float) -> pd.Series:
    time_values = np.asarray(frame.columns, dtype=float)
    mask = (time_values >= start_time) & (time_values <= end_time)
    if not mask.any():
        raise ValueError(f"No {TIME_COLUMN} values fall within the requested window {start_time} to {end_time}.")
    return frame.loc[:, frame.columns[mask]].mean(axis=1)


def build_paired_difference_rows(
    frames: Dict[str, pd.DataFrame],
    component_names: Sequence[str],
    quantity: str,
    unit: str,
    window_start_time: float,
    window_end_time: float,
) -> pd.DataFrame:
    all_window = compute_window_average(frames[ALL_WELLS_COMPONENT], window_start_time, window_end_time)
    component_windows = {
        name: compute_window_average(frames[name], window_start_time, window_end_time)
        for name in component_names
    }

    rows: List[pd.DataFrame] = []
    for component in component_names:
        component_value = component_windows[component]
        other_value = all_window - component_value
        paired_difference = component_value - other_value
        finite_mask = np.isfinite(component_value) & np.isfinite(other_value) & np.isfinite(paired_difference)

        out = pd.DataFrame(
            {
                "case": component_value.index.to_numpy(),
                "component": component,
                "quantity": quantity,
                "unit": unit,
                "window_start_time": window_start_time,
                "window_end_time": window_end_time,
                "component_window_mean": component_value.to_numpy(dtype=float),
                "other_components_window_mean": other_value.to_numpy(dtype=float),
                "all_wells_window_mean": all_window.to_numpy(dtype=float),
                "paired_difference": paired_difference.to_numpy(dtype=float),
                "component_exceeds_others": paired_difference.to_numpy(dtype=float) > 0.0,
                "finite": finite_mask.to_numpy(dtype=bool),
            }
        )
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def make_ecdf(paired_df: pd.DataFrame, n_min: int) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["component", "quantity", "unit", "window_start_time", "window_end_time"]

    for group_key, group in paired_df.groupby(group_cols, sort=False):
        values = group.loc[group["finite"], "paired_difference"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        values.sort()
        n_valid = int(values.size)

        if n_valid < n_min:
            continue

        for rank, value in enumerate(values, start=1):
            rows.append(
                {
                    "component": group_key[0],
                    "quantity": group_key[1],
                    "unit": group_key[2],
                    "window_start_time": group_key[3],
                    "window_end_time": group_key[4],
                    "n_valid": n_valid,
                    "n_min_required": n_min,
                    "rank": rank,
                    "ecdf_probability": rank / n_valid,
                    "paired_difference": value,
                }
            )

    return pd.DataFrame(rows)


def summarize_paired_differences(paired_df: pd.DataFrame, n_min: int) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["component", "quantity", "unit", "window_start_time", "window_end_time"]

    for group_key, group in paired_df.groupby(group_cols, sort=False):
        values = group.loc[group["finite"], "paired_difference"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n_valid = int(values.size)

        row = {
            "component": group_key[0],
            "quantity": group_key[1],
            "unit": group_key[2],
            "window_start_time": group_key[3],
            "window_end_time": group_key[4],
            "n_valid": n_valid,
            "n_min_required": n_min,
            "n_positive": int(np.sum(values > 0.0)),
            "n_zero": int(np.sum(values == 0.0)),
            "n_negative": int(np.sum(values < 0.0)),
            "fraction_positive": np.nan,
            "fraction_zero": np.nan,
            "fraction_negative": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

        for q in QUANTILES:
            row[f"q{int(round(q * 100)):02d}"] = np.nan

        if n_valid >= n_min:
            row["fraction_positive"] = row["n_positive"] / n_valid
            row["fraction_zero"] = row["n_zero"] / n_valid
            row["fraction_negative"] = row["n_negative"] / n_valid
            row["mean"] = float(np.mean(values))
            row["std"] = float(np.std(values, ddof=0))
            row["min"] = float(np.min(values))
            row["max"] = float(np.max(values))
            qvals = np.quantile(values, QUANTILES)
            for q, qval in zip(QUANTILES, qvals):
                row[f"q{int(round(q * 100)):02d}"] = float(qval)

        rows.append(row)

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_dir_arg = args.input_dir or default_input_dir_from_config(config)
    if input_dir_arg is None:
        raise RuntimeError("--input-dir is required unless --config provides ogs_results_postprocessed_dir.")

    input_dir = input_dir_arg.resolve()
    output_dir = (args.output_dir or default_output_dir_from_config(config)).resolve()

    if args.window_end_time < args.window_start_time:
        raise ValueError("window_end_time must be greater than or equal to window_start_time.")
    if args.case_summary_csv is None and (args.leduc_min is not None or args.leduc_max is not None):
        raise ValueError("--case-summary-csv is required when using --leduc-min or --leduc-max.")
    if args.case_summary_csv is not None and args.leduc_min is None and args.leduc_max is None:
        raise ValueError("At least one of --leduc-min or --leduc-max must be provided with --case-summary-csv.")
    if args.leduc_min is not None and args.leduc_max is not None and args.leduc_min > args.leduc_max:
        raise ValueError("leduc_min must be less than or equal to leduc_max.")

    all_path, component_paths = discover_input_files(input_dir)
    component_names = [component_name_from_path(path) for path in component_paths]

    pressure_frames: Dict[str, pd.DataFrame] = {
        ALL_WELLS_COMPONENT: read_and_pivot_csv(all_path, "pressure_Pa")
    }
    rate_frames: Dict[str, pd.DataFrame] = {
        ALL_WELLS_COMPONENT: read_and_pivot_csv(all_path, "pressure_rate_Pa_per_month")
    }

    for path in component_paths:
        component = component_name_from_path(path)
        pressure_frames[component] = read_and_pivot_csv(path, "pressure_Pa")
        rate_frames[component] = read_and_pivot_csv(path, "pressure_rate_Pa_per_month")

    pressure_frames = align_frames(pressure_frames)
    rate_frames = align_frames(rate_frames)
    n_cases_before_filter = len(pressure_frames[ALL_WELLS_COMPONENT].index)

    case_summary_csv = args.case_summary_csv.resolve() if args.case_summary_csv is not None else None
    pressure_frames = maybe_filter_cases(pressure_frames, case_summary_csv, args.leduc_min, args.leduc_max)
    rate_frames = maybe_filter_cases(rate_frames, case_summary_csv, args.leduc_min, args.leduc_max)

    if not pressure_frames[ALL_WELLS_COMPONENT].index.equals(rate_frames[ALL_WELLS_COMPONENT].index):
        raise ValueError("Pressure and pressure-rate case indices do not match after filtering.")

    paired_pressure = build_paired_difference_rows(
        pressure_frames,
        component_names,
        "pressure_Pa",
        "Pa",
        args.window_start_time,
        args.window_end_time,
    )
    paired_rate = build_paired_difference_rows(
        rate_frames,
        component_names,
        "pressure_rate_Pa_per_month",
        "Pa/month",
        args.window_start_time,
        args.window_end_time,
    )
    paired = pd.concat([paired_pressure, paired_rate], ignore_index=True)

    summary = summarize_paired_differences(paired, args.n_min)
    ecdf = make_ecdf(paired, args.n_min)

    write_csv(paired, output_dir / "paired_window_differences.csv")
    write_csv(ecdf, output_dir / "paired_difference_ecdf.csv")
    write_csv(summary, output_dir / "paired_difference_summary.csv")

    metadata_rows = [
        {"key": "input_dir", "value": str(input_dir)},
        {"key": "all_wells_file", "value": ALL_WELLS_FILE},
        {"key": "component_files", "value": ",".join(path.name for path in component_paths)},
        {"key": "component_order", "value": ",".join(component_names)},
        {"key": "definition", "value": "paired_difference = component_window_mean - (all_wells_window_mean - component_window_mean)"},
        {"key": "time_column", "value": TIME_COLUMN},
        {"key": "window_start_time", "value": str(args.window_start_time)},
        {"key": "window_end_time", "value": str(args.window_end_time)},
        {
            "key": "window_definition",
            "value": (
                f"single monthly output at {TIME_COLUMN}={args.window_start_time}"
                if args.window_start_time == args.window_end_time
                else (
                    f"inclusive arithmetic mean from {TIME_COLUMN}="
                    f"{args.window_start_time} to {args.window_end_time}"
                )
            ),
        },
        {
            "key": "canonical_default",
            "value": "November 2022 monthly output at time_month=516.0",
        },
        {"key": "n_cases_before_filter", "value": str(n_cases_before_filter)},
        {"key": "n_cases", "value": str(len(pressure_frames[ALL_WELLS_COMPONENT].index))},
        {"key": "case_summary_csv", "value": "not requested" if case_summary_csv is None else str(case_summary_csv)},
        {"key": "leduc_min_mD", "value": "not requested" if args.leduc_min is None else str(args.leduc_min)},
        {"key": "leduc_max_mD", "value": "not requested" if args.leduc_max is None else str(args.leduc_max)},
        {"key": "n_min", "value": str(args.n_min)},
    ]
    write_csv(pd.DataFrame(metadata_rows), output_dir / "analysis_metadata.csv")

    print(f"Paired difference analysis completed. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
