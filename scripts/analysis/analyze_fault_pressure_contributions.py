#!/usr/bin/env python3
"""
Analyze well contribution time series from OGS fault pressure CSV exports.

This script assumes one CSV for the all wells case and one CSV for each
component case such as individual wells and a remaining_nine_wells aggregate.

Input CSV format
    case,time_month,pressure_Pa,pressure_rate_Pa_per_month

Main tasks
    1. Check numerical superposition consistency for pressure and pressure rate.
    2. Compute per case contribution ratios with validity masks.
    3. Compute ensemble quantiles for raw values and ratios.
    4. Compute dominance frequencies using valid cases only.
    5. Optionally compute event window summaries.

Notes
    * Raw all wells statistics are computed directly from the all wells CSV.
    * Component raw statistics are computed from each component CSV.
    * Ratio outputs are explicitly labeled with _ratio.
    * The superposition mismatch uses no denominator floor or threshold;
      selected exact-zero or nonfinite all-wells responses are errors.
    * Other contribution ratios are set to NaN when the denominator
      magnitude is below a dataset scale based threshold.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
OUTPUT_DIRNAME = "contribution_analysis"
REPO_ROOT = Path(__file__).resolve().parents[3]

LEGACY_OTHER_WELLS_COMPONENT = "other_wells"
REMAINING_NINE_WELLS_COMPONENT = "remaining_nine_wells"
ALL_WELLS_COMPONENT = "all_wells"
ALL_EXCEPT_WELL15_COMPONENT = "all_except_well15"
WELL15_COMPONENT = "well15"

TIME_MONTH_ORIGIN_YEAR = 1979
TIME_MONTH_ORIGIN_MONTH = 11
CONTRIBUTION_ANALYSIS_TIME_MONTHS = (516.0, 517.0, 518.0)
CANONICAL_REALIZATION_COUNT = 1000
CANONICAL_TARGET_TIME_COUNT = 3
CANONICAL_EVALUATION_POINT_COUNT = 3000
SUPERPOSITION_POINTWISE_OUTPUT = "superposition_mismatch_pointwise.csv"
SUPERPOSITION_SUMMARY_OUTPUT = "superposition_mismatch_summary.csv"
EXPECTED_SOURCE_GROUPS = (
    "well6",
    "well10",
    "well11",
    "well12",
    "well14",
    "well15",
    REMAINING_NINE_WELLS_COMPONENT,
)


def time_month_to_calendar_month(time_month: float) -> str:
    month_index = int(time_month)
    if float(month_index) != float(time_month):
        raise ValueError(f"{TIME_COLUMN} must be an integer monthly output index: {time_month}")
    zero_based_month = (
        TIME_MONTH_ORIGIN_YEAR * 12
        + (TIME_MONTH_ORIGIN_MONTH - 1)
        + month_index
    )
    year, month_zero_based = divmod(zero_based_month, 12)
    return f"{year:04d}-{month_zero_based + 1:02d}"


CONTRIBUTION_ANALYSIS_CALENDAR_MONTHS = tuple(
    time_month_to_calendar_month(value) for value in CONTRIBUTION_ANALYSIS_TIME_MONTHS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fault pressure contribution decomposition across parameter cases."
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
        help="Directory for output CSV files. Default is outputs/analysis/contribution_analysis.",
    )
    parser.add_argument(
        "--window-start-time",
        type=float,
        default=None,
        help=f"Inclusive event window start {TIME_COLUMN}.",
    )
    parser.add_argument(
        "--window-end-time",
        type=float,
        default=None,
        help=f"Inclusive event window end {TIME_COLUMN}.",
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
        "--alpha-pressure",
        type=float,
        default=1e-6,
        help="Relative threshold coefficient for pressure ratios.",
    )
    parser.add_argument(
        "--alpha-rate",
        type=float,
        default=1e-4,
        help="Relative threshold coefficient for pressure rate ratios.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=50,
        help="Minimum number of valid cases required for quantile and dominance summaries.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-30,
        help="Small numerical tolerance used only to avoid exact zero scale thresholds.",
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
    return local_output_dir / "analysis" / OUTPUT_DIRNAME


def component_sort_key(name: str) -> Tuple[int, int | float, str]:
    name = normalize_component_name(name)
    if name == REMAINING_NINE_WELLS_COMPONENT:
        return (2, math.inf, name)
    if name == ALL_EXCEPT_WELL15_COMPONENT:
        return (3, math.inf, name)
    if name == ALL_WELLS_COMPONENT:
        return (4, math.inf, name)
    match = re.fullmatch(r"well(\d+)", name)
    if match:
        return (1, int(match.group(1)), name)
    return (5, math.inf, name)


def normalize_component_name(name: str) -> str:
    if name == LEGACY_OTHER_WELLS_COMPONENT:
        return REMAINING_NINE_WELLS_COMPONENT
    return name


def series_kind_for_component(name: str) -> str:
    if name in {ALL_WELLS_COMPONENT, ALL_EXCEPT_WELL15_COMPONENT}:
        return "aggregate"
    return "component"


def add_series_kind_column(df: pd.DataFrame) -> pd.DataFrame:
    if "component" not in df.columns:
        return df

    out = df.copy()
    series_kind = out["component"].astype(str).map(series_kind_for_component)
    if "series_kind" in out.columns:
        out["series_kind"] = series_kind
        return out

    insert_at = out.columns.get_loc("component") + 1
    out.insert(insert_at, "series_kind", series_kind)
    return out


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


def component_name_from_path(path: Path) -> str:
    name = path.stem
    prefix = "fault_pressure_timeseries_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected file name: {path.name}")
    return normalize_component_name(name[len(prefix) :])


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
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def identifier_examples(values: Sequence[object], limit: int = 5) -> str:
    examples = [str(value) for value in list(values)[:limit]]
    return "[" + ", ".join(examples) + "]"


def validate_case_identifier_sets(
    named_frames: Mapping[str, pd.DataFrame],
    reference_name: str = ALL_WELLS_COMPONENT,
) -> pd.Index:
    if not named_frames:
        raise ValueError("No input frames were provided for case identifier validation.")
    if reference_name not in named_frames:
        raise ValueError(f"Reference input '{reference_name}' was not provided.")

    reference_cases = named_frames[reference_name].index.copy()
    if len(reference_cases) == 0:
        raise ValueError(f"Reference input '{reference_name}' contains no case identifiers.")
    reference_unique_cases = reference_cases.unique()
    for input_name, frame in named_frames.items():
        duplicate_mask = frame.index.duplicated(keep=False)
        duplicate_identifiers = frame.index[duplicate_mask].unique()
        input_unique_cases = frame.index.unique()
        missing_cases = reference_unique_cases.difference(input_unique_cases, sort=False)
        extra_cases = input_unique_cases.difference(reference_unique_cases, sort=False)
        if len(duplicate_identifiers) or len(missing_cases) or len(extra_cases):
            raise ValueError(
                f"Case identifier validation failed for input '{input_name}' relative to "
                f"'{reference_name}': "
                f"duplicate_identifier_count={len(duplicate_identifiers)}, "
                f"missing_count={len(missing_cases)}, "
                f"extra_count={len(extra_cases)}, "
                "representative_duplicate_identifiers="
                f"{identifier_examples(duplicate_identifiers)}, "
                f"representative_missing_identifiers={identifier_examples(missing_cases)}, "
                f"representative_extra_identifiers={identifier_examples(extra_cases)}."
            )
    return reference_cases


def align_frames(named_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    reference_cases = validate_case_identifier_sets(named_frames)
    common_times = named_frames[ALL_WELLS_COMPONENT].columns.copy()
    for frame in named_frames.values():
        common_times = common_times.intersection(frame.columns, sort=False)

    if len(common_times) == 0:
        raise ValueError(f"No common {TIME_COLUMN} values across all CSV files.")

    aligned: Dict[str, pd.DataFrame] = {}
    for name, frame in named_frames.items():
        aligned[name] = frame.loc[reference_cases, common_times].copy()
    return aligned


def build_component_cube(frames: Dict[str, pd.DataFrame], component_names: Sequence[str]) -> np.ndarray:
    arrays = [frames[name].to_numpy(dtype=float) for name in component_names]
    return np.stack(arrays, axis=2)


def add_derived_aggregate_frame(named_frames: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    derived_names: List[str] = []
    out = dict(named_frames)
    if ALL_WELLS_COMPONENT in out and WELL15_COMPONENT in out:
        out[ALL_EXCEPT_WELL15_COMPONENT] = out[ALL_WELLS_COMPONENT] - out[WELL15_COMPONENT]
        derived_names.append(ALL_EXCEPT_WELL15_COMPONENT)
    return out, derived_names


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

    subset = df.loc[mask, "case_name"].dropna().astype(str)
    selected = set(subset.tolist())
    if not selected:
        if leduc_min is None:
            raise ValueError(f"No cases satisfy leduc_perm_mD <= {leduc_max}")
        if leduc_max is None:
            raise ValueError(f"No cases satisfy leduc_perm_mD >= {leduc_min}")
        raise ValueError(f"No cases satisfy {leduc_min} <= leduc_perm_mD <= {leduc_max}")
    return selected


def target_time_positions(
    available_time_months: Sequence[float],
    target_time_months: Sequence[float] = CONTRIBUTION_ANALYSIS_TIME_MONTHS,
    input_name: str | None = None,
) -> List[int]:
    available = np.asarray(available_time_months, dtype=float)
    input_context = "" if input_name is None else f" in input '{input_name}'"
    positions: List[int] = []
    for target in target_time_months:
        matches = np.flatnonzero(available == float(target))
        if matches.size != 1:
            if matches.size == 0:
                raise ValueError(
                    f"Required {TIME_COLUMN}={target:g} is missing{input_context}; "
                    "occurrence_count=0."
                )
            raise ValueError(
                f"Required {TIME_COLUMN}={target:g} occurs more than once{input_context}; "
                f"occurrence_count={matches.size}."
            )
        positions.append(int(matches[0]))
    return positions


def validate_target_times_in_frames(
    named_frames: Mapping[str, pd.DataFrame],
    target_time_months: Sequence[float] = CONTRIBUTION_ANALYSIS_TIME_MONTHS,
) -> None:
    for input_name, frame in named_frames.items():
        target_time_positions(frame.columns.to_numpy(dtype=float), target_time_months, input_name)


def count_finite_nonzero_opposite_sign(left: np.ndarray, right: np.ndarray) -> int:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        raise ValueError("Opposite-sign diagnostic arrays must have identical shapes.")

    nonfinite_left_count = int(np.count_nonzero(~np.isfinite(left_values)))
    nonfinite_right_count = int(np.count_nonzero(~np.isfinite(right_values)))
    if nonfinite_left_count or nonfinite_right_count:
        raise ValueError(
            "Opposite-sign diagnostic requires finite values: "
            f"left_nonfinite_count={nonfinite_left_count}, "
            f"right_nonfinite_count={nonfinite_right_count}."
        )

    both_nonzero = (left_values != 0.0) & (right_values != 0.0)
    opposite_sign = both_nonzero & np.not_equal(
        np.signbit(left_values),
        np.signbit(right_values),
    )
    return int(np.count_nonzero(opposite_sign))


def validate_canonical_input_cardinality(
    n_realizations: int,
    n_target_times: int,
) -> None:
    errors: List[str] = []
    if n_realizations != CANONICAL_REALIZATION_COUNT:
        errors.append(
            f"realization_count={n_realizations}, expected={CANONICAL_REALIZATION_COUNT}"
        )
    if n_target_times != CANONICAL_TARGET_TIME_COUNT:
        errors.append(
            f"target_time_count={n_target_times}, expected={CANONICAL_TARGET_TIME_COUNT}"
        )

    point_count_from_dimensions = n_realizations * n_target_times
    if point_count_from_dimensions != CANONICAL_EVALUATION_POINT_COUNT:
        errors.append(
            "point_count_from_dimensions="
            f"{point_count_from_dimensions}, expected={CANONICAL_EVALUATION_POINT_COUNT}"
        )

    if errors:
        raise ValueError(
            "Canonical superposition input cardinality validation failed: "
            + "; ".join(errors)
        )


def validate_canonical_superposition_cardinality(
    n_realizations: int,
    n_target_times: int,
    evaluation_point_counts: Mapping[str, int],
) -> None:
    validate_canonical_input_cardinality(n_realizations, n_target_times)
    point_count_from_dimensions = n_realizations * n_target_times
    errors: List[str] = []
    for quantity_name in ("pressure", "pressure_rate"):
        if quantity_name not in evaluation_point_counts:
            errors.append(f"{quantity_name}_point_count=missing")
            continue
        actual_point_count = int(evaluation_point_counts[quantity_name])
        if (
            actual_point_count != point_count_from_dimensions
            or actual_point_count != CANONICAL_EVALUATION_POINT_COUNT
        ):
            errors.append(
                f"{quantity_name}_point_count={actual_point_count}, "
                f"dimension_product={point_count_from_dimensions}, "
                f"expected={CANONICAL_EVALUATION_POINT_COUNT}"
            )

    if errors:
        raise ValueError(
            "Canonical superposition cardinality validation failed: " + "; ".join(errors)
        )


def pointwise_superposition_mismatch(
    all_frame: pd.DataFrame,
    component_cube: np.ndarray,
    quantity_name: str,
    unit: str,
    target_time_months: Sequence[float] = CONTRIBUTION_ANALYSIS_TIME_MONTHS,
    target_calendar_months: Sequence[str] = CONTRIBUTION_ANALYSIS_CALENDAR_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all-wells and summed source-group responses at exact output times."""
    all_values = all_frame.to_numpy(dtype=float)
    if component_cube.ndim != 3 or component_cube.shape[:2] != all_values.shape:
        raise ValueError("component_cube must match the case and time dimensions of all_frame.")
    if len(target_time_months) != len(target_calendar_months):
        raise ValueError("Each target time_month must have one calendar-month label.")

    positions = target_time_positions(all_frame.columns.to_numpy(dtype=float), target_time_months)
    selected_all = all_values[:, positions]
    selected_source_sum = component_cube[:, positions, :].sum(axis=2)

    zero_denominator_count = int(np.count_nonzero(selected_all == 0.0))
    nonfinite_denominator_count = int(np.count_nonzero(~np.isfinite(selected_all)))
    if zero_denominator_count or nonfinite_denominator_count:
        raise ValueError(
            f"{quantity_name}: pointwise mismatch cannot be computed because the selected "
            f"all-wells responses include {zero_denominator_count} exact zero and "
            f"{nonfinite_denominator_count} nonfinite values."
        )

    nonfinite_source_sum_count = int(np.count_nonzero(~np.isfinite(selected_source_sum)))
    if nonfinite_source_sum_count:
        raise ValueError(
            f"{quantity_name}: the selected source-group sums include "
            f"{nonfinite_source_sum_count} nonfinite values."
        )

    signed_residual = selected_all - selected_source_sum
    absolute_residual = np.abs(signed_residual)
    absolute_all = np.abs(selected_all)
    mismatch_fraction = absolute_residual / absolute_all
    mismatch_percent = 100.0 * mismatch_fraction

    n_cases, n_times = selected_all.shape
    cases = np.repeat(all_frame.index.to_numpy(), n_times)
    time_months = np.tile(np.asarray(target_time_months, dtype=float), n_cases)
    calendar_months = np.tile(np.asarray(target_calendar_months, dtype=object), n_cases)
    pointwise = pd.DataFrame(
        {
            "quantity": quantity_name,
            "unit": unit,
            "case": cases,
            TIME_COLUMN: time_months,
            "calendar_month": calendar_months,
            "all_wells_response": selected_all.ravel(),
            "source_group_response_sum": selected_source_sum.ravel(),
            "signed_residual": signed_residual.ravel(),
            "absolute_residual": absolute_residual.ravel(),
            "absolute_all_wells_response": absolute_all.ravel(),
            "pointwise_relative_mismatch_fraction": mismatch_fraction.ravel(),
            "pointwise_relative_mismatch_percent": mismatch_percent.ravel(),
        }
    )

    maximum_flat_index = int(np.argmax(mismatch_fraction))
    maximum_case_index, maximum_time_index = np.unravel_index(maximum_flat_index, mismatch_fraction.shape)
    opposite_sign_count = count_finite_nonzero_opposite_sign(selected_all, selected_source_sum)
    summary = pd.DataFrame(
        [
            {
                "quantity": quantity_name,
                "unit": unit,
                "source_group_count": int(component_cube.shape[2]),
                "case_count": n_cases,
                "output_time_count": n_times,
                "point_count": int(mismatch_fraction.size),
                "time_months": ",".join(f"{float(value):g}" for value in target_time_months),
                "calendar_months": ",".join(target_calendar_months),
                "exact_zero_all_wells_response_count": zero_denominator_count,
                "nonfinite_all_wells_response_count": nonfinite_denominator_count,
                "negative_all_wells_response_count_before_abs": int(np.count_nonzero(selected_all < 0.0)),
                "opposite_sign_all_wells_and_source_group_sum_count": opposite_sign_count,
                "point_count_above_0_01_percent": int(np.count_nonzero(mismatch_percent > 0.01)),
                "point_count_above_0_1_percent": int(np.count_nonzero(mismatch_percent > 0.1)),
                "maximum_pointwise_relative_mismatch_fraction": float(
                    mismatch_fraction[maximum_case_index, maximum_time_index]
                ),
                "maximum_pointwise_relative_mismatch_percent": float(
                    mismatch_percent[maximum_case_index, maximum_time_index]
                ),
                "maximum_case": all_frame.index[maximum_case_index],
                "maximum_time_month": float(target_time_months[maximum_time_index]),
                "maximum_calendar_month": target_calendar_months[maximum_time_index],
                "all_wells_response_at_maximum_mismatch": float(
                    selected_all[maximum_case_index, maximum_time_index]
                ),
                "source_group_response_sum_at_maximum_mismatch": float(
                    selected_source_sum[maximum_case_index, maximum_time_index]
                ),
                "signed_residual_at_maximum_mismatch": float(
                    signed_residual[maximum_case_index, maximum_time_index]
                ),
                "absolute_residual_at_maximum_mismatch": float(
                    absolute_residual[maximum_case_index, maximum_time_index]
                ),
                "absolute_all_wells_response_at_maximum_mismatch": float(
                    absolute_all[maximum_case_index, maximum_time_index]
                ),
            }
        ]
    )
    return pointwise, summary


def compute_scale_threshold(values_2d: np.ndarray, alpha: float, eps: float) -> Tuple[float, float]:
    scale = float(np.nanmedian(np.abs(values_2d)))
    threshold = max(alpha * scale, eps)
    return scale, threshold


def safe_signed_ratio(component_cube: np.ndarray, total_2d: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    denom = total_2d[:, :, None]
    out = np.full_like(component_cube, np.nan, dtype=float)
    valid_2d = np.abs(total_2d) >= threshold
    np.divide(component_cube, denom, out=out, where=valid_2d[:, :, None])
    return out, valid_2d


def positive_only_ratio(component_cube: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    positive_cube = np.maximum(component_cube, 0.0)
    denom = positive_cube.sum(axis=2, keepdims=True)
    out = np.full_like(positive_cube, np.nan, dtype=float)
    denom_2d = denom[:, :, 0]
    valid_2d = denom_2d >= threshold
    np.divide(positive_cube, denom, out=out, where=valid_2d[:, :, None])
    return out, valid_2d


def quantile_timeseries_ratio(
    ratio_cube: np.ndarray,
    valid_2d: np.ndarray,
    component_names: Sequence[str],
    time_values: Sequence[float],
    n_min: int,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    rows: List[dict] = []
    for comp_idx, comp_name in enumerate(component_names):
        values = ratio_cube[:, :, comp_idx]
        for time_idx, t in enumerate(time_values):
            valid_mask = valid_2d[:, time_idx] & np.isfinite(values[:, time_idx])
            valid_values = values[valid_mask, time_idx]
            n_valid = int(valid_values.size)
            row = {
                "component": comp_name,
                TIME_COLUMN: float(t),
                "n_valid": n_valid,
                "n_min_required": n_min,
            }
            if n_valid >= n_min:
                qvals = np.quantile(valid_values, quantiles)
                for q, qval in zip(quantiles, qvals):
                    row[f"q{int(round(q * 100)):02d}"] = qval
                row["mean"] = float(np.mean(valid_values))
                row["std"] = float(np.std(valid_values, ddof=0))
            else:
                for q in quantiles:
                    row[f"q{int(round(q * 100)):02d}"] = np.nan
                row["mean"] = np.nan
                row["std"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_timeseries_raw_components(
    raw_cube: np.ndarray,
    component_names: Sequence[str],
    time_values: Sequence[float],
    n_min: int,
    quantity_label: str,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    rows: List[dict] = []
    for comp_idx, comp_name in enumerate(component_names):
        values = raw_cube[:, :, comp_idx]
        for time_idx, t in enumerate(time_values):
            finite_mask = np.isfinite(values[:, time_idx])
            valid_values = values[finite_mask, time_idx]
            n_valid = int(valid_values.size)
            row = {
                "component": comp_name,
                "quantity": quantity_label,
                TIME_COLUMN: float(t),
                "n_valid": n_valid,
                "n_min_required": n_min,
            }
            if n_valid >= n_min:
                qvals = np.quantile(valid_values, quantiles)
                for q, qval in zip(quantiles, qvals):
                    row[f"q{int(round(q * 100)):02d}"] = qval
                row["mean"] = float(np.mean(valid_values))
                row["std"] = float(np.std(valid_values, ddof=0))
            else:
                for q in quantiles:
                    row[f"q{int(round(q * 100)):02d}"] = np.nan
                row["mean"] = np.nan
                row["std"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_timeseries_total(
    total_2d: np.ndarray,
    time_values: Sequence[float],
    n_min: int,
    quantity_label: str,
    component_name: str = ALL_WELLS_COMPONENT,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    rows: List[dict] = []
    for time_idx, t in enumerate(time_values):
        finite_mask = np.isfinite(total_2d[:, time_idx])
        valid_values = total_2d[finite_mask, time_idx]
        n_valid = int(valid_values.size)
        row = {
            "component": component_name,
            "quantity": quantity_label,
            TIME_COLUMN: float(t),
            "n_valid": n_valid,
            "n_min_required": n_min,
        }
        if n_valid >= n_min:
            qvals = np.quantile(valid_values, quantiles)
            for q, qval in zip(quantiles, qvals):
                row[f"q{int(round(q * 100)):02d}"] = qval
            row["mean"] = float(np.mean(valid_values))
            row["std"] = float(np.std(valid_values, ddof=0))
        else:
            for q in quantiles:
                row[f"q{int(round(q * 100)):02d}"] = np.nan
            row["mean"] = np.nan
            row["std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dominance_frequency(
    ratio_cube: np.ndarray,
    valid_2d: np.ndarray,
    component_names: Sequence[str],
    time_values: Sequence[float],
    n_min: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for time_idx, t in enumerate(time_values):
        step_values = ratio_cube[:, time_idx, :]
        valid_cases_mask = valid_2d[:, time_idx]
        n_valid = int(valid_cases_mask.sum())
        row = {
            TIME_COLUMN: float(t),
            "n_valid": n_valid,
            "n_min_required": n_min,
        }
        if n_valid < n_min:
            for comp_name in component_names:
                row[comp_name] = np.nan
            rows.append(row)
            continue

        winner_weights = {comp_name: 0.0 for comp_name in component_names}
        for case_idx in np.flatnonzero(valid_cases_mask):
            case_row = step_values[case_idx]
            finite_mask = np.isfinite(case_row)
            if not finite_mask.any():
                continue
            max_val = np.nanmax(case_row)
            winners = np.flatnonzero(np.isfinite(case_row) & np.isclose(case_row, max_val, rtol=0.0, atol=0.0))
            if winners.size == 0:
                continue
            share = 1.0 / winners.size
            for winner in winners:
                winner_weights[component_names[winner]] += share

        for comp_name in component_names:
            row[comp_name] = winner_weights[comp_name] / n_valid
        rows.append(row)

    return pd.DataFrame(rows, columns=[TIME_COLUMN, "n_valid", "n_min_required", *component_names])


def compute_window_average(cube: np.ndarray, time_values: Sequence[float], start_time: float, end_time: float) -> np.ndarray:
    time_arr = np.asarray(time_values, dtype=float)
    mask = (time_arr >= start_time) & (time_arr <= end_time)
    if not mask.any():
        raise ValueError(f"No {TIME_COLUMN} values fall within the requested window {start_time} to {end_time}.")
    return cube[:, mask, :].mean(axis=1)


def compute_window_average_total(total_2d: np.ndarray, time_values: Sequence[float], start_time: float, end_time: float) -> np.ndarray:
    time_arr = np.asarray(time_values, dtype=float)
    mask = (time_arr >= start_time) & (time_arr <= end_time)
    if not mask.any():
        raise ValueError(f"No {TIME_COLUMN} values fall within the requested window {start_time} to {end_time}.")
    return total_2d[:, mask].mean(axis=1)


def window_ratio_from_raw(window_raw: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    denom = window_raw.sum(axis=1, keepdims=True)
    out = np.full_like(window_raw, np.nan, dtype=float)
    valid = np.abs(denom[:, 0]) >= threshold
    np.divide(window_raw, denom, out=out, where=valid[:, None])
    return out, valid


def window_positive_ratio_from_raw(window_raw: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    positive_raw = np.maximum(window_raw, 0.0)
    denom = positive_raw.sum(axis=1, keepdims=True)
    out = np.full_like(positive_raw, np.nan, dtype=float)
    valid = denom[:, 0] >= threshold
    np.divide(positive_raw, denom, out=out, where=valid[:, None])
    return out, valid


def summarize_window_distribution(
    values_2d: np.ndarray,
    valid: np.ndarray,
    component_names: Sequence[str],
    quantity_label: str,
    start_time: float,
    end_time: float,
    n_min: int,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    rows: List[dict] = []
    n_valid = int(valid.sum())
    for comp_idx, comp_name in enumerate(component_names):
        vals = values_2d[valid, comp_idx]
        row = {
            "component": comp_name,
            "quantity": quantity_label,
            "window_start_time": start_time,
            "window_end_time": end_time,
            "n_valid": n_valid,
            "n_min_required": n_min,
        }
        if n_valid >= n_min and vals.size > 0:
            qvals = np.quantile(vals, quantiles)
            for q, qval in zip(quantiles, qvals):
                row[f"q{int(round(q * 100)):02d}"] = qval
            row["mean"] = float(np.mean(vals))
            row["std"] = float(np.std(vals, ddof=0))
        else:
            for q in quantiles:
                row[f"q{int(round(q * 100)):02d}"] = np.nan
            row["mean"] = np.nan
            row["std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_window_total(
    values_1d: np.ndarray,
    quantity_label: str,
    start_time: float,
    end_time: float,
    n_min: int,
    component_name: str = ALL_WELLS_COMPONENT,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    vals = values_1d[np.isfinite(values_1d)]
    n_valid = int(vals.size)
    row = {
        "component": component_name,
        "quantity": quantity_label,
        "window_start_time": start_time,
        "window_end_time": end_time,
        "n_valid": n_valid,
        "n_min_required": n_min,
    }
    if n_valid >= n_min:
        qvals = np.quantile(vals, quantiles)
        for q, qval in zip(quantiles, qvals):
            row[f"q{int(round(q * 100)):02d}"] = qval
        row["mean"] = float(np.mean(vals))
        row["std"] = float(np.std(vals, ddof=0))
    else:
        for q in quantiles:
            row[f"q{int(round(q * 100)):02d}"] = np.nan
        row["mean"] = np.nan
        row["std"] = np.nan
    return pd.DataFrame([row])


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
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_summary_csv is not None:
        args.case_summary_csv = args.case_summary_csv.resolve()
    if args.leduc_min is not None and args.leduc_max is not None and args.leduc_min > args.leduc_max:
        raise ValueError("leduc_min must be less than or equal to leduc_max.")
    if args.case_summary_csv is None and (args.leduc_min is not None or args.leduc_max is not None):
        raise ValueError("--case-summary-csv is required when using --leduc-min or --leduc-max.")
    if args.case_summary_csv is not None and args.leduc_min is None and args.leduc_max is None:
        raise ValueError("At least one of --leduc-min or --leduc-max must be provided with --case-summary-csv.")

    all_path, component_paths = discover_input_files(input_dir)
    component_names = [component_name_from_path(p) for p in component_paths]
    if tuple(component_names) != EXPECTED_SOURCE_GROUPS:
        raise ValueError(
            "Expected exactly these seven source groups in this order: "
            f"{', '.join(EXPECTED_SOURCE_GROUPS)}; found {', '.join(component_names)}."
        )

    pressure_frames: Dict[str, pd.DataFrame] = {"all_wells": read_and_pivot_csv(all_path, "pressure_Pa")}
    rate_frames: Dict[str, pd.DataFrame] = {"all_wells": read_and_pivot_csv(all_path, "pressure_rate_Pa_per_month")}

    for path in component_paths:
        name = component_name_from_path(path)
        pressure_frames[name] = read_and_pivot_csv(path, "pressure_Pa")
        rate_frames[name] = read_and_pivot_csv(path, "pressure_rate_Pa_per_month")

    validate_target_times_in_frames(pressure_frames)
    validate_target_times_in_frames(rate_frames)
    pressure_frames = align_frames(pressure_frames)
    rate_frames = align_frames(rate_frames)

    n_cases_before_filter = len(pressure_frames[ALL_WELLS_COMPONENT].index)
    validate_canonical_input_cardinality(
        n_cases_before_filter,
        len(CONTRIBUTION_ANALYSIS_TIME_MONTHS),
    )
    pressure_mismatch_cube = build_component_cube(pressure_frames, component_names)
    rate_mismatch_cube = build_component_cube(rate_frames, component_names)
    pressure_mismatch_points, pressure_mismatch_summary = pointwise_superposition_mismatch(
        pressure_frames[ALL_WELLS_COMPONENT],
        pressure_mismatch_cube,
        "pressure",
        "Pa",
    )
    rate_mismatch_points, rate_mismatch_summary = pointwise_superposition_mismatch(
        rate_frames[ALL_WELLS_COMPONENT],
        rate_mismatch_cube,
        "pressure_rate",
        "Pa per month",
    )
    evaluation_point_counts = {
        "pressure": len(pressure_mismatch_points),
        "pressure_rate": len(rate_mismatch_points),
    }
    validate_canonical_superposition_cardinality(
        n_cases_before_filter,
        len(CONTRIBUTION_ANALYSIS_TIME_MONTHS),
        evaluation_point_counts,
    )
    print(
        "Canonical superposition cardinality: "
        f"inputs={len(pressure_frames)}, realizations={n_cases_before_filter}, "
        f"target_times={len(CONTRIBUTION_ANALYSIS_TIME_MONTHS)}, "
        f"pressure_points={evaluation_point_counts['pressure']}, "
        f"pressure_rate_points={evaluation_point_counts['pressure_rate']}"
    )
    mismatch_points = pd.concat([pressure_mismatch_points, rate_mismatch_points], ignore_index=True)
    mismatch_summary = pd.concat([pressure_mismatch_summary, rate_mismatch_summary], ignore_index=True)
    write_csv(mismatch_points, output_dir / SUPERPOSITION_POINTWISE_OUTPUT)
    write_csv(mismatch_summary, output_dir / SUPERPOSITION_SUMMARY_OUTPUT)
    for row in mismatch_summary.itertuples(index=False):
        print(
            f"{row.quantity} superposition mismatch: "
            f"maximum={row.maximum_pointwise_relative_mismatch_percent:.12g}%, "
            f"case={row.maximum_case}, time_month={row.maximum_time_month:g}, "
            f"calendar_month={row.maximum_calendar_month}; "
            f"exact_zero_all_wells_response_count={row.exact_zero_all_wells_response_count}, "
            f"nonfinite_all_wells_response_count={row.nonfinite_all_wells_response_count}, "
            "negative_all_wells_response_count_before_abs="
            f"{row.negative_all_wells_response_count_before_abs}"
        )

    selected_cases: set[str] | None = None
    if args.case_summary_csv is not None:
        selected_cases = load_selected_cases(args.case_summary_csv, args.leduc_min, args.leduc_max)
        common_selected_cases = pressure_frames[ALL_WELLS_COMPONENT].index.intersection(pd.Index(sorted(selected_cases)))
        if len(common_selected_cases) == 0:
            raise ValueError("No selected cases were found in the pressure timeseries tables.")
        pressure_frames = {name: frame.loc[common_selected_cases].copy() for name, frame in pressure_frames.items()}
        rate_frames = {name: frame.loc[common_selected_cases].copy() for name, frame in rate_frames.items()}

    pressure_frames, derived_component_names = add_derived_aggregate_frame(pressure_frames)
    rate_frames, derived_component_names_rate = add_derived_aggregate_frame(rate_frames)
    if derived_component_names != derived_component_names_rate:
        raise ValueError("Derived component lists do not match between pressure and rate frames.")

    cases = pressure_frames[ALL_WELLS_COMPONENT].index.to_list()
    time_values = [float(x) for x in pressure_frames[ALL_WELLS_COMPONENT].columns.to_list()]

    pressure_cube = build_component_cube(pressure_frames, component_names)
    rate_cube = build_component_cube(rate_frames, component_names)
    pressure_all_values = pressure_frames[ALL_WELLS_COMPONENT].to_numpy(dtype=float)
    rate_all_values = rate_frames[ALL_WELLS_COMPONENT].to_numpy(dtype=float)

    pressure_scale, pressure_threshold = compute_scale_threshold(pressure_all_values, args.alpha_pressure, args.eps)
    rate_scale, rate_threshold = compute_scale_threshold(rate_all_values, args.alpha_rate, args.eps)

    pressure_ratio_cube, pressure_valid = safe_signed_ratio(pressure_cube, pressure_all_values, pressure_threshold)
    rate_ratio_signed_cube, rate_signed_valid = safe_signed_ratio(rate_cube, rate_all_values, rate_threshold)
    rate_ratio_positive_cube, rate_positive_valid = positive_only_ratio(rate_cube, rate_threshold)

    pressure_component_q = quantile_timeseries_raw_components(pressure_cube, component_names, time_values, args.n_min, "pressure_Pa")
    rate_component_q = quantile_timeseries_raw_components(rate_cube, component_names, time_values, args.n_min, "pressure_rate_Pa_per_month")
    pressure_total_q = quantile_timeseries_total(pressure_all_values, time_values, args.n_min, "pressure_Pa")
    rate_total_q = quantile_timeseries_total(rate_all_values, time_values, args.n_min, "pressure_rate_Pa_per_month")

    pressure_combined_frames = [pressure_component_q]
    rate_combined_frames = [rate_component_q]
    if ALL_EXCEPT_WELL15_COMPONENT in pressure_frames:
        pressure_combined_frames.append(
            quantile_timeseries_total(
                pressure_frames[ALL_EXCEPT_WELL15_COMPONENT].to_numpy(dtype=float),
                time_values,
                args.n_min,
                "pressure_Pa",
                component_name=ALL_EXCEPT_WELL15_COMPONENT,
            )
        )
    if ALL_EXCEPT_WELL15_COMPONENT in rate_frames:
        rate_combined_frames.append(
            quantile_timeseries_total(
                rate_frames[ALL_EXCEPT_WELL15_COMPONENT].to_numpy(dtype=float),
                time_values,
                args.n_min,
                "pressure_rate_Pa_per_month",
                component_name=ALL_EXCEPT_WELL15_COMPONENT,
            )
        )
    pressure_combined_frames.append(pressure_total_q)
    rate_combined_frames.append(rate_total_q)

    write_csv(add_series_kind_column(pd.concat(pressure_combined_frames, ignore_index=True)), output_dir / "Q_timeseries_pressure.csv")
    write_csv(add_series_kind_column(pd.concat(rate_combined_frames, ignore_index=True)), output_dir / "Q_timeseries_pressure_rate.csv")

    write_csv(
        quantile_timeseries_ratio(pressure_ratio_cube, pressure_valid, component_names, time_values, args.n_min),
        output_dir / "Q_timeseries_pressure_ratio.csv",
    )
    write_csv(
        quantile_timeseries_ratio(rate_ratio_signed_cube, rate_signed_valid, component_names, time_values, args.n_min),
        output_dir / "Q_timeseries_pressure_rate_ratio_signed.csv",
    )
    write_csv(
        quantile_timeseries_ratio(rate_ratio_positive_cube, rate_positive_valid, component_names, time_values, args.n_min),
        output_dir / "Q_timeseries_pressure_rate_ratio_positive.csv",
    )

    write_csv(
        dominance_frequency(pressure_ratio_cube, pressure_valid, component_names, time_values, args.n_min),
        output_dir / "D_timeseries_pressure_ratio.csv",
    )
    write_csv(
        dominance_frequency(rate_ratio_signed_cube, rate_signed_valid, component_names, time_values, args.n_min),
        output_dir / "D_timeseries_pressure_rate_ratio_signed.csv",
    )
    write_csv(
        dominance_frequency(rate_ratio_positive_cube, rate_positive_valid, component_names, time_values, args.n_min),
        output_dir / "D_timeseries_pressure_rate_ratio_positive.csv",
    )

    if (args.window_start_time is None) ^ (args.window_end_time is None):
        raise ValueError("Both --window-start-time and --window-end-time must be provided together.")

    if args.window_start_time is not None and args.window_end_time is not None:
        if args.window_end_time < args.window_start_time:
            raise ValueError("window_end_time must be greater than or equal to window_start_time.")

        window_pressure_raw = compute_window_average(pressure_cube, time_values, args.window_start_time, args.window_end_time)
        window_rate_raw = compute_window_average(rate_cube, time_values, args.window_start_time, args.window_end_time)
        window_pressure_all = compute_window_average_total(pressure_all_values, time_values, args.window_start_time, args.window_end_time)
        window_rate_all = compute_window_average_total(rate_all_values, time_values, args.window_start_time, args.window_end_time)

        window_pressure_ratio, window_pressure_valid = window_ratio_from_raw(window_pressure_raw, pressure_threshold)
        window_rate_ratio_signed, window_rate_signed_valid = window_ratio_from_raw(window_rate_raw, rate_threshold)
        window_rate_ratio_positive, window_rate_positive_valid = window_positive_ratio_from_raw(window_rate_raw, rate_threshold)

        window_pressure_component_summary = summarize_window_distribution(
            window_pressure_raw,
            np.isfinite(window_pressure_raw[:, 0]),
            component_names,
            "pressure_Pa",
            args.window_start_time,
            args.window_end_time,
            args.n_min,
        )
        window_rate_component_summary = summarize_window_distribution(
            window_rate_raw,
            np.isfinite(window_rate_raw[:, 0]),
            component_names,
            "pressure_rate_Pa_per_month",
            args.window_start_time,
            args.window_end_time,
            args.n_min,
        )
        window_pressure_total_summary = summarize_window_total(
            window_pressure_all,
            "pressure_Pa",
            args.window_start_time,
            args.window_end_time,
            args.n_min,
        )
        window_rate_total_summary = summarize_window_total(
            window_rate_all,
            "pressure_rate_Pa_per_month",
            args.window_start_time,
            args.window_end_time,
            args.n_min,
        )

        window_pressure_combined_frames = [window_pressure_component_summary]
        window_rate_combined_frames = [window_rate_component_summary]
        if ALL_EXCEPT_WELL15_COMPONENT in pressure_frames:
            window_pressure_combined_frames.append(
                summarize_window_total(
                    compute_window_average_total(
                        pressure_frames[ALL_EXCEPT_WELL15_COMPONENT].to_numpy(dtype=float),
                        time_values,
                        args.window_start_time,
                        args.window_end_time,
                    ),
                    "pressure_Pa",
                    args.window_start_time,
                    args.window_end_time,
                    args.n_min,
                    component_name=ALL_EXCEPT_WELL15_COMPONENT,
                )
            )
        if ALL_EXCEPT_WELL15_COMPONENT in rate_frames:
            window_rate_combined_frames.append(
                summarize_window_total(
                    compute_window_average_total(
                        rate_frames[ALL_EXCEPT_WELL15_COMPONENT].to_numpy(dtype=float),
                        time_values,
                        args.window_start_time,
                        args.window_end_time,
                    ),
                    "pressure_rate_Pa_per_month",
                    args.window_start_time,
                    args.window_end_time,
                    args.n_min,
                    component_name=ALL_EXCEPT_WELL15_COMPONENT,
                )
            )
        window_pressure_combined_frames.append(window_pressure_total_summary)
        window_rate_combined_frames.append(window_rate_total_summary)

        write_csv(add_series_kind_column(pd.concat(window_pressure_combined_frames, ignore_index=True)), output_dir / "event_window_summary_pressure.csv")
        write_csv(add_series_kind_column(pd.concat(window_rate_combined_frames, ignore_index=True)), output_dir / "event_window_summary_pressure_rate.csv")
        write_csv(
            summarize_window_distribution(window_pressure_ratio, window_pressure_valid, component_names, "pressure_ratio", args.window_start_time, args.window_end_time, args.n_min),
            output_dir / "event_window_summary_pressure_ratio.csv",
        )
        write_csv(
            summarize_window_distribution(window_rate_ratio_signed, window_rate_signed_valid, component_names, "pressure_rate_ratio_signed", args.window_start_time, args.window_end_time, args.n_min),
            output_dir / "event_window_summary_pressure_rate_ratio_signed.csv",
        )
        write_csv(
            summarize_window_distribution(window_rate_ratio_positive, window_rate_positive_valid, component_names, "pressure_rate_ratio_positive", args.window_start_time, args.window_end_time, args.n_min),
            output_dir / "event_window_summary_pressure_rate_ratio_positive.csv",
        )

    metadata_rows = [
        {"key": "input_dir", "value": str(input_dir)},
        {"key": "all_wells_file", "value": ALL_WELLS_FILE},
        {"key": "time_column", "value": TIME_COLUMN},
        {"key": "component_files", "value": ",".join(p.name for p in component_paths)},
        {"key": "n_cases_before_filter", "value": str(n_cases_before_filter)},
        {"key": "n_cases", "value": str(len(cases))},
        {"key": "superposition_mismatch_case_count", "value": str(n_cases_before_filter)},
        {"key": "n_time_values", "value": str(len(time_values))},
        {"key": "component_order", "value": ",".join(component_names)},
        {
            "key": "superposition_mismatch_time_months",
            "value": ",".join(f"{value:g}" for value in CONTRIBUTION_ANALYSIS_TIME_MONTHS),
        },
        {
            "key": "superposition_mismatch_calendar_months",
            "value": ",".join(CONTRIBUTION_ANALYSIS_CALENDAR_MONTHS),
        },
        {"key": "superposition_mismatch_pointwise_output", "value": SUPERPOSITION_POINTWISE_OUTPUT},
        {"key": "superposition_mismatch_summary_output", "value": SUPERPOSITION_SUMMARY_OUTPUT},
        {"key": "derived_series", "value": ",".join(derived_component_names) if derived_component_names else "none"},
        {
            "key": "combined_series_order",
            "value": ",".join(component_names + derived_component_names + [ALL_WELLS_COMPONENT]),
        },
        {
            "key": "case_summary_csv",
            "value": "not requested" if args.case_summary_csv is None else str(args.case_summary_csv),
        },
        {
            "key": "leduc_min_mD",
            "value": "not requested" if args.leduc_min is None else str(args.leduc_min),
        },
        {
            "key": "leduc_max_mD",
            "value": "not requested" if args.leduc_max is None else str(args.leduc_max),
        },
        {"key": "alpha_pressure", "value": f"{args.alpha_pressure:.3e}"},
        {"key": "alpha_rate", "value": f"{args.alpha_rate:.3e}"},
        {"key": "pressure_scale_median_abs", "value": f"{pressure_scale:.16e}"},
        {"key": "pressure_threshold", "value": f"{pressure_threshold:.16e}"},
        {"key": "rate_scale_median_abs", "value": f"{rate_scale:.16e}"},
        {"key": "rate_threshold", "value": f"{rate_threshold:.16e}"},
        {"key": "n_min", "value": str(args.n_min)},
        {"key": "eps", "value": f"{args.eps:.3e}"},
        {
            "key": "event_window",
            "value": "not requested" if args.window_start_time is None else f"{args.window_start_time} to {args.window_end_time}",
        },
    ]
    write_csv(pd.DataFrame(metadata_rows), output_dir / "analysis_metadata.csv")
    print(f"Analysis completed. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
