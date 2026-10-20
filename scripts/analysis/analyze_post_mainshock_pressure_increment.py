#!/usr/bin/env python3
"""Compare modeled post-mainshock pressure increments with Sun et al. (2026).

The calculation is paired within each hydraulic-property realization.  It uses
the existing Fault S area-averaged pressure exports for the all-wells case and
the seven nonoverlapping source groups.  Monthly model outputs are linearly
interpolated to the earthquake origin times; adjacent monthly outputs are also
summarized as a time-discretization sensitivity check.

No OGS input or simulation output is modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .analyze_fault_pressure_contributions import (
        ALL_WELLS_COMPONENT,
        EXPECTED_SOURCE_GROUPS,
        REMAINING_NINE_WELLS_COMPONENT,
        align_frames,
        component_name_from_path,
        default_input_dir_from_config,
        discover_input_files,
        load_config,
        read_and_pivot_csv,
        validate_case_identifier_sets,
    )
except ImportError:  # Direct execution from this directory.
    from analyze_fault_pressure_contributions import (  # type: ignore
        ALL_WELLS_COMPONENT,
        EXPECTED_SOURCE_GROUPS,
        REMAINING_NINE_WELLS_COMPONENT,
        align_frames,
        component_name_from_path,
        default_input_dir_from_config,
        discover_input_files,
        load_config,
        read_and_pivot_csv,
        validate_case_identifier_sets,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "postprocess_validation" / "pressure_timeseries"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "post_mainshock_pressure_increment"
DEFAULT_FIGURE = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "post_mainshock_pressure_increment"
    / "post_mainshock_pressure_increment_diagnostic.pdf"
)

PRESSURE_COLUMN = "pressure_Pa"
PA_PER_KPA = 1_000.0
SUN_REFERENCE_KPA = 0.5
SUBSECTION_3_4_PRESSURE_RESIDUAL_PERCENT = 0.030
CANONICAL_REALIZATION_COUNT = 1_000
CANONICAL_TIME_INDICES = np.arange(0.0, 541.0, dtype=float)
SUBSECTION_3_4_TIME_INDICES = (516.0, 517.0, 518.0)

# Model time 0 is the initial boundary immediately before the December 1979
# injection interval.  Thus time 1 is the December 1979 output, time 516 is the
# November 2022 output, and time 520 is the March 2023 output.
TIME_ZERO_BOUNDARY_UTC = datetime(1979, 12, 1, tzinfo=timezone.utc)

# Times in the processed catalog used by the current study.  The first of the
# two ML 4.8+ events on 16 March is used for t1; the ML 5.0 event followed only
# 13 minutes later, which is negligible relative to monthly output spacing.
MAINSHOCK_TIME_UTC = datetime(2022, 11, 30, 0, 55, 56, 400_000, tzinfo=timezone.utc)
MARCH_FIRST_EVENT_TIME_UTC = datetime(2023, 3, 16, 14, 46, 18, 500_000, tzinfo=timezone.utc)
MARCH_SECOND_EVENT_TIME_UTC = datetime(2023, 3, 16, 14, 59, 9, 900_000, tzinfo=timezone.utc)

PRIMARY_SCENARIO = "event_time_linear_interpolation"
COMPONENT_ORDER = (ALL_WELLS_COMPONENT, *EXPECTED_SOURCE_GROUPS)
DISPLAY_NAMES = {
    ALL_WELLS_COMPONENT: "All wells",
    "well6": "Well 6",
    "well10": "Well 10",
    "well11": "Well 11",
    "well12": "Well 12",
    "well14": "Well 14",
    "well15": "Well 15 (well 14-18)",
    REMAINING_NINE_WELLS_COMPONENT: "Remaining nine wells",
}


@dataclass(frozen=True)
class TimeScenario:
    name: str
    description: str
    t0_position: float
    t1_position: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired Fault S pressure increments from the 30 November 2022 "
            "mainshock to the 16 March 2023 earthquakes."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional local YAML config. If --input-dir is omitted, "
            "ogs_results_postprocessed_dir/pressure_timeseries is used."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the eight fault_pressure_timeseries_*.csv files. "
            "Fallback: outputs/postprocess_validation/pressure_timeseries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for numerical CSV outputs.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Diagnostic figure path (PDF).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this analysis's existing CSV and figure outputs.",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Write numerical outputs without producing the diagnostic figure.",
    )
    return parser


def shift_month_start(origin: datetime, months: int) -> datetime:
    """Shift a first-of-month UTC timestamp by an integer number of months."""

    if origin.day != 1 or any((origin.hour, origin.minute, origin.second, origin.microsecond)):
        raise ValueError("origin must be midnight on the first day of a month")
    total = origin.year * 12 + (origin.month - 1) + int(months)
    year, month_zero = divmod(total, 12)
    return origin.replace(year=year, month=month_zero + 1)


def output_boundary_utc(time_index: int) -> datetime:
    if int(time_index) != time_index:
        raise ValueError(f"time_index must be an integer, got {time_index}")
    return shift_month_start(TIME_ZERO_BOUNDARY_UTC, int(time_index))


def output_month_label(time_index: int) -> str:
    """Calendar month whose end is represented by a saved output index."""

    return (output_boundary_utc(time_index) - timedelta(microseconds=1)).strftime("%Y-%m")


def time_position_for_event(event_time: datetime) -> float:
    """Return fractional model-month position using actual calendar duration."""

    if event_time.tzinfo is None:
        raise ValueError("event_time must be timezone-aware")
    event_utc = event_time.astimezone(timezone.utc)
    lower_index = (event_utc.year - TIME_ZERO_BOUNDARY_UTC.year) * 12 + (
        event_utc.month - TIME_ZERO_BOUNDARY_UTC.month
    )
    lower_time = output_boundary_utc(lower_index)
    upper_time = output_boundary_utc(lower_index + 1)
    fraction = (event_utc - lower_time) / (upper_time - lower_time)
    if not 0.0 <= fraction < 1.0:
        raise RuntimeError("event-time month interpolation fraction is outside [0, 1)")
    return float(lower_index + fraction)


def nearest_output_index(event_time: datetime) -> int:
    position = time_position_for_event(event_time)
    lower_index = math.floor(position)
    candidates = (lower_index, lower_index + 1)
    return min(
        candidates,
        key=lambda index: abs(event_time.astimezone(timezone.utc) - output_boundary_utc(index)),
    )


def build_time_scenarios() -> tuple[TimeScenario, ...]:
    t0 = time_position_for_event(MAINSHOCK_TIME_UTC)
    t1 = time_position_for_event(MARCH_FIRST_EVENT_TIME_UTC)
    t0_lower = math.floor(t0)
    t1_lower = math.floor(t1)
    return (
        TimeScenario(
            PRIMARY_SCENARIO,
            "Linear interpolation of both monthly pressure series to the event origin times.",
            t0,
            t1,
        ),
        TimeScenario(
            "nearest_monthly_outputs",
            "Nearest saved monthly output for each event.",
            float(nearest_output_index(MAINSHOCK_TIME_UTC)),
            float(nearest_output_index(MARCH_FIRST_EVENT_TIME_UTC)),
        ),
        TimeScenario(
            "second_march_event_linear_interpolation",
            "Linear interpolation to the second ML 4.8+ event, 13 minutes after the first.",
            t0,
            time_position_for_event(MARCH_SECOND_EVENT_TIME_UTC),
        ),
        TimeScenario(
            "t0_previous_adjacent_output",
            "Previous saved output for t0; t1 remains event-time interpolated.",
            float(t0_lower),
            t1,
        ),
        TimeScenario(
            "t0_next_adjacent_output",
            "Next saved output for t0; t1 remains event-time interpolated.",
            float(t0_lower + 1),
            t1,
        ),
        TimeScenario(
            "t1_previous_adjacent_output",
            "t0 remains event-time interpolated; previous saved output for t1.",
            t0,
            float(t1_lower),
        ),
        TimeScenario(
            "t1_next_adjacent_output",
            "t0 remains event-time interpolated; next saved output for t1.",
            t0,
            float(t1_lower + 1),
        ),
        TimeScenario(
            "adjacent_outputs_t0_previous_t1_previous",
            "Previous adjacent saved output for both t0 and t1.",
            float(t0_lower),
            float(t1_lower),
        ),
        TimeScenario(
            "adjacent_outputs_t0_previous_t1_next",
            "Previous adjacent saved output for t0 and next adjacent output for t1.",
            float(t0_lower),
            float(t1_lower + 1),
        ),
        TimeScenario(
            "adjacent_outputs_t0_next_t1_previous",
            "Next adjacent saved output for t0 and previous adjacent output for t1.",
            float(t0_lower + 1),
            float(t1_lower),
        ),
    )


def required_indices_for_position(position: float) -> tuple[int, ...]:
    rounded = round(position)
    if math.isclose(position, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        return (int(rounded),)
    lower = math.floor(position)
    return (lower, lower + 1)


def interpolate_frame(
    frame: pd.DataFrame, position: float, input_name: str = "input"
) -> np.ndarray:
    """Interpolate each realization at one fractional model-month position."""

    needed = required_indices_for_position(position)
    missing = [float(index) for index in needed if float(index) not in frame.columns]
    if missing:
        raise ValueError(f"Missing time indices {missing} in '{input_name}'")
    if len(needed) == 1:
        values = frame[float(needed[0])].to_numpy(dtype=float)
    else:
        lower, upper = needed
        weight = position - lower
        lower_values = frame[float(lower)].to_numpy(dtype=float)
        upper_values = frame[float(upper)].to_numpy(dtype=float)
        values = lower_values + weight * (upper_values - lower_values)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Nonfinite interpolated pressure in '{input_name}' at {position:g}")
    return values


def paired_increments_kpa(
    frames: Mapping[str, pd.DataFrame], scenario: TimeScenario
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    t0_values: dict[str, np.ndarray] = {}
    t1_values: dict[str, np.ndarray] = {}
    increments: dict[str, np.ndarray] = {}
    for component in COMPONENT_ORDER:
        frame = frames[component]
        p0 = interpolate_frame(frame, scenario.t0_position, component) / PA_PER_KPA
        p1 = interpolate_frame(frame, scenario.t1_position, component) / PA_PER_KPA
        t0_values[component] = p0
        t1_values[component] = p1
        increments[component] = p1 - p0
    return t0_values, t1_values, increments


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot summarize an empty or nonfinite distribution")
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return {
        "n_realizations": int(finite.size),
        "median_kPa": float(quantiles[2]),
        "q25_kPa": float(quantiles[1]),
        "q75_kPa": float(quantiles[3]),
        "q05_kPa": float(quantiles[0]),
        "q95_kPa": float(quantiles[4]),
        "minimum_kPa": float(np.min(finite)),
        "maximum_kPa": float(np.max(finite)),
        "fraction_positive": float(np.mean(finite > 0.0)),
        "percent_positive": float(100.0 * np.mean(finite > 0.0)),
    }


def primary_realization_table(
    cases: pd.Index,
    t0_values: Mapping[str, np.ndarray],
    t1_values: Mapping[str, np.ndarray],
    increments: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for component in COMPONENT_ORDER:
        rows.append(
            pd.DataFrame(
                {
                    "case": cases.astype(str),
                    "component": component,
                    "display_name": DISPLAY_NAMES[component],
                    "series_kind": (
                        "all_wells" if component == ALL_WELLS_COMPONENT else "source_group"
                    ),
                    "pressure_t0_kPa": t0_values[component],
                    "pressure_t1_kPa": t1_values[component],
                    "pressure_increment_kPa": increments[component],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def primary_summary_table(increments: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        stats = summarize_values(increments[component])
        row: dict[str, object] = {
            "component": component,
            "display_name": DISPLAY_NAMES[component],
            "series_kind": "all_wells" if component == ALL_WELLS_COMPONENT else "source_group",
            **stats,
            "sun_reference_kPa": SUN_REFERENCE_KPA if component == ALL_WELLS_COMPONENT else np.nan,
            "sun_within_q05_q95": (
                bool(stats["q05_kPa"] <= SUN_REFERENCE_KPA <= stats["q95_kPa"])
                if component == ALL_WELLS_COMPONENT
                else pd.NA
            ),
            "sun_within_full_range": (
                bool(stats["minimum_kPa"] <= SUN_REFERENCE_KPA <= stats["maximum_kPa"])
                if component == ALL_WELLS_COMPONENT
                else pd.NA
            ),
            "median_to_sun_ratio": (
                float(stats["median_kPa"] / SUN_REFERENCE_KPA)
                if component == ALL_WELLS_COMPONENT
                else np.nan
            ),
            "median_same_positive_order_as_sun_factor_10": (
                bool(0.1 <= float(stats["median_kPa"]) / SUN_REFERENCE_KPA <= 10.0)
                if component == ALL_WELLS_COMPONENT
                else pd.NA
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_summary_table(
    scenario_results: Mapping[str, Mapping[str, np.ndarray]],
    scenarios: Sequence[TimeScenario],
) -> pd.DataFrame:
    primary = scenario_results[PRIMARY_SCENARIO]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for component in COMPONENT_ORDER:
            stats = summarize_values(scenario_results[scenario.name][component])
            rows.append(
                {
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "t0_position_month": scenario.t0_position,
                    "t1_position_month": scenario.t1_position,
                    "t0_sampling": describe_position(scenario.t0_position),
                    "t1_sampling": describe_position(scenario.t1_position),
                    "component": component,
                    "display_name": DISPLAY_NAMES[component],
                    **stats,
                    "median_difference_from_primary_kPa": float(
                        stats["median_kPa"] - np.median(primary[component])
                    ),
                    "sun_reference_kPa": (
                        SUN_REFERENCE_KPA if component == ALL_WELLS_COMPONENT else np.nan
                    ),
                    "sun_within_q05_q95": (
                        bool(stats["q05_kPa"] <= SUN_REFERENCE_KPA <= stats["q95_kPa"])
                        if component == ALL_WELLS_COMPONENT
                        else pd.NA
                    ),
                    "sun_within_full_range": (
                        bool(
                            stats["minimum_kPa"]
                            <= SUN_REFERENCE_KPA
                            <= stats["maximum_kPa"]
                        )
                        if component == ALL_WELLS_COMPONENT
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)


def safe_relative_percent(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(numerator, dtype=float), np.nan)
    denominator_values = np.asarray(denominator, dtype=float)
    mask = np.isfinite(numerator) & np.isfinite(denominator_values) & (denominator_values != 0.0)
    out[mask] = 100.0 * np.abs(np.asarray(numerator, dtype=float)[mask]) / np.abs(
        denominator_values[mask]
    )
    return out


def superposition_increment_table(
    cases: pd.Index,
    t0_values: Mapping[str, np.ndarray],
    t1_values: Mapping[str, np.ndarray],
    increments: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    source_t0 = np.sum([t0_values[name] for name in EXPECTED_SOURCE_GROUPS], axis=0)
    source_t1 = np.sum([t1_values[name] for name in EXPECTED_SOURCE_GROUPS], axis=0)
    source_increment = np.sum([increments[name] for name in EXPECTED_SOURCE_GROUPS], axis=0)
    residual_t0 = t0_values[ALL_WELLS_COMPONENT] - source_t0
    residual_t1 = t1_values[ALL_WELLS_COMPONENT] - source_t1
    residual_increment = increments[ALL_WELLS_COMPONENT] - source_increment
    endpoint_scale = np.maximum(
        np.abs(t0_values[ALL_WELLS_COMPONENT]), np.abs(t1_values[ALL_WELLS_COMPONENT])
    )
    source_absolute_increment_sum = np.sum(
        [np.abs(increments[name]) for name in EXPECTED_SOURCE_GROUPS], axis=0
    )
    return pd.DataFrame(
        {
            "case": cases.astype(str),
            "all_wells_increment_kPa": increments[ALL_WELLS_COMPONENT],
            "source_group_sum_increment_kPa": source_increment,
            "increment_residual_kPa": residual_increment,
            "increment_absolute_relative_residual_percent": safe_relative_percent(
                residual_increment, increments[ALL_WELLS_COMPONENT]
            ),
            "increment_residual_scaled_to_endpoint_pressure_percent": safe_relative_percent(
                residual_increment, endpoint_scale
            ),
            "increment_residual_scaled_to_sum_absolute_components_percent": safe_relative_percent(
                residual_increment, source_absolute_increment_sum
            ),
            "t0_superposition_residual_kPa": residual_t0,
            "t1_superposition_residual_kPa": residual_t1,
            "t0_absolute_relative_residual_percent": safe_relative_percent(
                residual_t0, t0_values[ALL_WELLS_COMPONENT]
            ),
            "t1_absolute_relative_residual_percent": safe_relative_percent(
                residual_t1, t1_values[ALL_WELLS_COMPONENT]
            ),
        }
    )


def finite_max(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def existing_window_max_relative_residual_percent(
    frames: Mapping[str, pd.DataFrame],
) -> float:
    indices = list(SUBSECTION_3_4_TIME_INDICES)
    all_values = frames[ALL_WELLS_COMPONENT].loc[:, indices].to_numpy(dtype=float)
    source_sum = np.sum(
        [frames[name].loc[:, indices].to_numpy(dtype=float) for name in EXPECTED_SOURCE_GROUPS],
        axis=0,
    )
    residual = all_values - source_sum
    relative = safe_relative_percent(residual, all_values)
    return finite_max(relative)


def superposition_summary_table(
    residuals: pd.DataFrame,
    existing_window_max_percent: float,
) -> pd.DataFrame:
    endpoint_relative_max = max(
        finite_max(residuals["t0_absolute_relative_residual_percent"]),
        finite_max(residuals["t1_absolute_relative_residual_percent"]),
    )
    metrics: list[tuple[str, object, str]] = [
        ("n_realizations", int(len(residuals)), "count"),
        (
            "maximum_absolute_increment_residual",
            float(np.max(np.abs(residuals["increment_residual_kPa"]))),
            "kPa",
        ),
        (
            "median_absolute_increment_residual",
            float(np.median(np.abs(residuals["increment_residual_kPa"]))),
            "kPa",
        ),
        (
            "root_mean_square_increment_residual",
            float(np.sqrt(np.mean(np.square(residuals["increment_residual_kPa"])))),
            "kPa",
        ),
        (
            "maximum_increment_relative_residual",
            finite_max(residuals["increment_absolute_relative_residual_percent"]),
            "percent of absolute all-wells increment",
        ),
        (
            "q95_increment_relative_residual",
            float(
                np.nanquantile(
                    residuals["increment_absolute_relative_residual_percent"], 0.95, method="linear"
                )
            ),
            "percent of absolute all-wells increment",
        ),
        (
            "maximum_increment_residual_scaled_to_endpoint_pressure",
            finite_max(residuals["increment_residual_scaled_to_endpoint_pressure_percent"]),
            "percent",
        ),
        (
            "maximum_increment_residual_scaled_to_sum_absolute_components",
            finite_max(
                residuals["increment_residual_scaled_to_sum_absolute_components_percent"]
            ),
            "percent",
        ),
        (
            "maximum_event_endpoint_relative_residual",
            endpoint_relative_max,
            "percent",
        ),
        (
            "recomputed_subsection_3_4_pressure_residual",
            existing_window_max_percent,
            "percent",
        ),
        (
            "reported_subsection_3_4_pressure_residual",
            SUBSECTION_3_4_PRESSURE_RESIDUAL_PERCENT,
            "percent",
        ),
        (
            "recomputed_subsection_3_4_rounds_to_reported_value",
            bool(round(existing_window_max_percent, 3) == SUBSECTION_3_4_PRESSURE_RESIDUAL_PERCENT),
            "boolean",
        ),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "unit_or_note"])


def monthly_output_superposition_summary(
    frames: Mapping[str, pd.DataFrame],
    time_indices: Sequence[int] = (515, 516, 519, 520),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_index in time_indices:
        all_values_kpa = (
            frames[ALL_WELLS_COMPONENT][float(time_index)].to_numpy(dtype=float) / PA_PER_KPA
        )
        source_sum_kpa = np.sum(
            [
                frames[name][float(time_index)].to_numpy(dtype=float) / PA_PER_KPA
                for name in EXPECTED_SOURCE_GROUPS
            ],
            axis=0,
        )
        residual_kpa = all_values_kpa - source_sum_kpa
        rows.append(
            {
                "time_month": time_index,
                "calendar_month": output_month_label(time_index),
                "output_boundary_utc": output_boundary_utc(time_index).isoformat(),
                "n_realizations": len(all_values_kpa),
                "maximum_absolute_residual_kPa": float(np.max(np.abs(residual_kpa))),
                "root_mean_square_residual_kPa": float(
                    np.sqrt(np.mean(np.square(residual_kpa)))
                ),
                "maximum_absolute_relative_residual_percent": finite_max(
                    safe_relative_percent(residual_kpa, all_values_kpa)
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_input_collection(frames: Mapping[str, pd.DataFrame]) -> pd.Index:
    cases = validate_case_identifier_sets(frames)
    if len(cases) != CANONICAL_REALIZATION_COUNT:
        raise ValueError(
            f"Expected {CANONICAL_REALIZATION_COUNT} realizations, found {len(cases)}"
        )
    expected_cases = pd.Index(
        [f"case_{index:06d}" for index in range(1, CANONICAL_REALIZATION_COUNT + 1)]
    )
    if not cases.equals(expected_cases):
        missing = expected_cases.difference(cases)
        extra = cases.difference(expected_cases)
        raise ValueError(
            "Canonical case identifiers are not in the expected all-wells order: "
            f"missing={missing[:5].tolist()}, extra={extra[:5].tolist()}"
        )
    reference_times = frames[ALL_WELLS_COMPONENT].columns.to_numpy(dtype=float)
    if not np.array_equal(reference_times, CANONICAL_TIME_INDICES):
        raise ValueError("The all-wells time axis is not the canonical 0..540 monthly axis")
    for name, frame in frames.items():
        if not np.array_equal(frame.columns.to_numpy(dtype=float), reference_times):
            raise ValueError(f"Time axis for '{name}' does not exactly match all_wells")
    return cases


def load_pressure_frames(
    input_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], pd.Index]:
    all_path, component_paths = discover_input_files(input_dir)
    paths_by_component: dict[str, Path] = {ALL_WELLS_COMPONENT: all_path}
    for path in component_paths:
        name = component_name_from_path(path)
        if name in paths_by_component:
            raise ValueError(f"Duplicate normalized component input '{name}'")
        paths_by_component[name] = path
    source_names = tuple(name for name in EXPECTED_SOURCE_GROUPS if name in paths_by_component)
    if source_names != EXPECTED_SOURCE_GROUPS or set(paths_by_component) != set(COMPONENT_ORDER):
        raise ValueError(
            "Expected exactly all_wells and these seven source groups: "
            f"{', '.join(EXPECTED_SOURCE_GROUPS)}; found {sorted(paths_by_component)}"
        )
    frames = {
        component: read_and_pivot_csv(paths_by_component[component], PRESSURE_COLUMN)
        for component in COMPONENT_ORDER
    }
    cases = validate_input_collection(frames)
    aligned = align_frames(dict(frames))
    # align_frames is retained to use the repository's audited realization-ID
    # alignment, while the stricter checks above ensure no time is discarded.
    return aligned, paths_by_component, cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_position(position: float) -> str:
    indices = required_indices_for_position(position)
    if len(indices) == 1:
        index = indices[0]
        return (
            f"saved output {index} ({output_month_label(index)}; "
            f"boundary {output_boundary_utc(index).isoformat()})"
        )
    lower, upper = indices
    return (
        f"linear interpolation between {lower} ({output_month_label(lower)}) and "
        f"{upper} ({output_month_label(upper)}), upper weight={position - lower:.12g}"
    )


def metadata_table(
    input_dir: Path,
    paths: Mapping[str, Path],
    primary: TimeScenario,
    scenarios: Sequence[TimeScenario],
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("analysis", "post-mainshock Fault S pressure increment comparison"),
        ("input_directory", str(input_dir.resolve())),
        ("input_quantity", "area-averaged pressure perturbation on Fault S Gamma"),
        (
            "spatial_statistic",
            "nodal-area-weighted mean over the FGZ intermediate plane 2-7 km below sea level",
        ),
        (
            "calculation",
            "within-realization pressure(t1) minus pressure(t0), then ensemble summary",
        ),
        ("pressure_input_unit", "Pa"),
        ("reported_unit", "kPa"),
        ("realization_count", CANONICAL_REALIZATION_COUNT),
        ("quantile_method", "numpy linear empirical quantiles"),
        ("positive_definition", "pressure_increment_kPa > 0"),
        ("model_time_zero_boundary_utc", TIME_ZERO_BOUNDARY_UTC.isoformat()),
        (
            "model_output_interpretation",
            "index n is the end boundary of the calendar month named by output_month_label(n)",
        ),
        ("t0_event_utc", MAINSHOCK_TIME_UTC.isoformat()),
        ("t1_event_utc", MARCH_FIRST_EVENT_TIME_UTC.isoformat()),
        ("t1_second_event_utc", MARCH_SECOND_EVENT_TIME_UTC.isoformat()),
        (
            "event_time_source",
            "retained Vasyura-Bathke et al. (2023) relocated southern-cluster catalog",
        ),
        ("t1_event_choice", "first of the two ML 4.8+ events; ML 5.0 followed 13 minutes later"),
        ("primary_time_method", primary.description),
        ("primary_t0_position_month", primary.t0_position),
        ("primary_t1_position_month", primary.t1_position),
        ("primary_t0_outputs", describe_position(primary.t0_position)),
        ("primary_t1_outputs", describe_position(primary.t1_position)),
        ("sun_reference_kPa", SUN_REFERENCE_KPA),
        (
            "sun_reference_definition",
            "modeled p(16 Mar 2023)-p(30 Nov 2022) fault-zone pressure increment; not absolute cumulative pressure",
        ),
        ("sun_reference_doi", "10.1029/2025GL118405"),
        ("sensitivity_scenarios", "; ".join(scenario.name for scenario in scenarios)),
    ]
    for component in COMPONENT_ORDER:
        path = paths[component].resolve()
        try:
            relative = path.relative_to(REPO_ROOT)
            path_text = str(relative)
        except ValueError:
            path_text = str(path)
        rows.extend(
            [
                (f"input_file_{component}", path_text),
                (f"input_sha256_{component}", sha256_file(path)),
            ]
        )
    return pd.DataFrame(rows, columns=["key", "value"])


def use_symlog_axis(axis, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    scale = max(float(np.max(np.abs(finite))), SUN_REFERENCE_KPA)
    if scale > 10.0 * SUN_REFERENCE_KPA:
        axis.set_xscale("symlog", linthresh=SUN_REFERENCE_KPA, linscale=1.0)


def plot_diagnostic(
    increments: Mapping[str, np.ndarray],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_values = np.asarray(increments[ALL_WELLS_COMPONENT], dtype=float)
    sorted_values = np.sort(all_values)
    ecdf = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    all_summary = summary.loc[summary["component"] == ALL_WELLS_COMPONENT].iloc[0]

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    ax_ecdf, ax_groups = axes

    ax_ecdf.axvspan(all_summary.q05_kPa, all_summary.q95_kPa, color="0.88", label="5th-95th")
    ax_ecdf.axvspan(all_summary.q25_kPa, all_summary.q75_kPa, color="0.72", label="25th-75th")
    ax_ecdf.plot(sorted_values, ecdf, color="black", linewidth=1.8, label="All wells ECDF")
    ax_ecdf.axvline(0.0, color="0.45", linewidth=0.8)
    ax_ecdf.axvline(all_summary.median_kPa, color="#2166ac", linewidth=1.5, label="Median")
    ax_ecdf.axvline(
        SUN_REFERENCE_KPA,
        color="#b2182b",
        linestyle="--",
        linewidth=1.7,
        label="Sun et al. (2026): ~0.5 kPa",
    )
    ax_ecdf.set_ylim(0.0, 1.0)
    ax_ecdf.set_xlabel("Post-mainshock pressure increment (kPa)")
    ax_ecdf.set_ylabel("Empirical cumulative probability")
    ax_ecdf.set_title("(a) All wells")
    ax_ecdf.grid(True, which="both", color="0.9", linewidth=0.6)
    use_symlog_axis(ax_ecdf, sorted_values)
    ax_ecdf.legend(loc="best", frameon=False, fontsize=8)

    group_order = [
        ALL_WELLS_COMPONENT,
        "well15",
        "well6",
        "well10",
        "well11",
        "well12",
        "well14",
        REMAINING_NINE_WELLS_COMPONENT,
    ]
    plot_rows = summary.set_index("component").loc[group_order]
    positions = np.arange(len(group_order), dtype=float)
    colors = ["black", "#b2182b", *(["#2166ac"] * 6)]
    for y, (_, row), color in zip(positions, plot_rows.iterrows(), colors):
        ax_groups.hlines(y, row.minimum_kPa, row.maximum_kPa, color="0.82", linewidth=1.0)
        ax_groups.hlines(y, row.q05_kPa, row.q95_kPa, color=color, linewidth=1.4, alpha=0.65)
        ax_groups.hlines(y, row.q25_kPa, row.q75_kPa, color=color, linewidth=5.0)
        ax_groups.plot(row.median_kPa, y, marker="o", color=color, markersize=4.5)
    ax_groups.axvline(0.0, color="0.45", linewidth=0.8)
    ax_groups.axvline(SUN_REFERENCE_KPA, color="#b2182b", linestyle="--", linewidth=1.2)
    ax_groups.set_yticks(positions)
    ax_groups.set_yticklabels([DISPLAY_NAMES[name] for name in group_order], fontsize=8)
    ax_groups.invert_yaxis()
    ax_groups.set_xlabel("Post-mainshock pressure increment (kPa)")
    ax_groups.set_title("(b) Source-group distributions")
    ax_groups.grid(True, axis="x", which="both", color="0.9", linewidth=0.6)
    combined = np.concatenate([np.asarray(increments[name], dtype=float) for name in group_order])
    use_symlog_axis(ax_groups, combined)
    legend_handles = [
        Line2D([0], [0], color="0.4", linewidth=1.0, label="min-max"),
        Line2D([0], [0], color="#2166ac", linewidth=1.5, label="5th-95th"),
        Line2D([0], [0], color="#2166ac", linewidth=5.0, label="25th-75th"),
        Line2D([0], [0], marker="o", color="#2166ac", linewidth=0, label="median"),
    ]
    ax_groups.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    figure.suptitle(
        "Fault S area-averaged pressure change: 30 Nov 2022 to 16 Mar 2023\n"
        "Paired within each hydraulic realization; monthly outputs linearly interpolated",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Analysis outputs already exist; pass --overwrite to replace only these files:\n"
            f"{formatted}"
        )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    configured_input_dir = default_input_dir_from_config(config)
    selected_input_dir = args.input_dir or configured_input_dir or DEFAULT_INPUT_DIR
    input_dir = selected_input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figure_path = args.figure.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_paths = {
        "realizations": output_dir / "pressure_increment_by_realization.csv",
        "summary": output_dir / "pressure_increment_summary.csv",
        "sensitivity": output_dir / "time_discretization_sensitivity.csv",
        "residuals": output_dir / "superposition_increment_residuals.csv",
        "residual_summary": output_dir / "superposition_increment_summary.csv",
        "monthly_residual_summary": output_dir / "superposition_monthly_output_summary.csv",
        "metadata": output_dir / "analysis_metadata.csv",
    }
    guarded_paths = list(output_paths.values())
    if not args.no_figure:
        guarded_paths.append(figure_path)
    ensure_outputs_available(guarded_paths, args.overwrite)

    frames, input_paths, cases = load_pressure_frames(input_dir)
    scenarios = build_time_scenarios()
    primary_scenario = scenarios[0]
    if primary_scenario.name != PRIMARY_SCENARIO:
        raise RuntimeError("Primary time scenario is not first")

    scenario_results: dict[str, Mapping[str, np.ndarray]] = {}
    primary_t0: dict[str, np.ndarray] | None = None
    primary_t1: dict[str, np.ndarray] | None = None
    for scenario in scenarios:
        t0_values, t1_values, increments = paired_increments_kpa(frames, scenario)
        scenario_results[scenario.name] = increments
        if scenario.name == PRIMARY_SCENARIO:
            primary_t0 = t0_values
            primary_t1 = t1_values
    if primary_t0 is None or primary_t1 is None:
        raise RuntimeError("Primary event-time results were not calculated")
    primary_increments = scenario_results[PRIMARY_SCENARIO]

    realizations = primary_realization_table(
        cases, primary_t0, primary_t1, primary_increments
    )
    summary = primary_summary_table(primary_increments)
    sensitivity = sensitivity_summary_table(scenario_results, scenarios)
    residuals = superposition_increment_table(
        cases, primary_t0, primary_t1, primary_increments
    )
    existing_window_max = existing_window_max_relative_residual_percent(frames)
    residual_summary = superposition_summary_table(residuals, existing_window_max)
    monthly_residual_summary = monthly_output_superposition_summary(frames)
    metadata = metadata_table(input_dir, input_paths, primary_scenario, scenarios)

    write_csv(realizations, output_paths["realizations"])
    write_csv(summary, output_paths["summary"])
    write_csv(sensitivity, output_paths["sensitivity"])
    write_csv(residuals, output_paths["residuals"])
    write_csv(residual_summary, output_paths["residual_summary"])
    write_csv(monthly_residual_summary, output_paths["monthly_residual_summary"])
    write_csv(metadata, output_paths["metadata"])
    if not args.no_figure:
        plot_diagnostic(primary_increments, summary, figure_path)

    all_row = summary.loc[summary["component"] == ALL_WELLS_COMPONENT].iloc[0]
    print(f"Wrote {len(realizations)} realization-component rows to {output_dir}")
    print(
        "All wells: "
        f"median={all_row.median_kPa:.6g} kPa, "
        f"q05-q95={all_row.q05_kPa:.6g} to {all_row.q95_kPa:.6g} kPa, "
        f"positive={all_row.percent_positive:.3f}%"
    )
    print(
        f"Sun reference {SUN_REFERENCE_KPA:g} kPa within q05-q95: "
        f"{all_row.sun_within_q05_q95}; within full range: {all_row.sun_within_full_range}"
    )
    print(
        "Maximum absolute increment superposition residual: "
        f"{np.max(np.abs(residuals['increment_residual_kPa'])):.6g} kPa"
    )
    if not args.no_figure:
        print(f"Wrote diagnostic figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
