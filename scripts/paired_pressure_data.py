"""Load and validate the public paired-pressure and parameter-design tables.

This module contains data-only helpers shared by the public-facing analysis and
figure scripts.  It intentionally depends only on NumPy and pandas so that its
validation and transformations can be tested without plotting dependencies.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PUBLIC_PAIRED_COLUMNS = (
    "case_id",
    "source_group",
    "paired_difference_kPa",
)
PUBLIC_PARAMETER_COLUMNS = (
    "case_id",
    "leduc_perm_mD",
    "dz_ratio",
    "gouge_ratio",
)
SOURCE_GROUPS = (
    "well6",
    "well10",
    "well11",
    "well12",
    "well14",
    "well15",
    "remaining_nine_wells",
)
EXPECTED_SIGN_COUNTS = {
    "well6": (103, 897, 0),
    "well10": (679, 321, 0),
    "well11": (103, 897, 0),
    "well12": (103, 897, 0),
    "well14": (103, 897, 0),
    "well15": (266, 734, 0),
    "remaining_nine_wells": (103, 897, 0),
}

EXPECTED_CASE_COUNT = 1000
EXPECTED_PAIRED_ROW_COUNT = EXPECTED_CASE_COUNT * len(SOURCE_GROUPS)
EXPECTED_AXIS_VALUE_COUNT = 10
EXPECTED_CASE_IDS = tuple(range(1, EXPECTED_CASE_COUNT + 1))


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    source: str,
) -> None:
    actual = tuple(str(column) for column in frame.columns)
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        raise ValueError(
            f"{source} must have exactly the columns {expected_tuple}; found {actual}"
        )


def _coerce_case_ids(series: pd.Series, source: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{source} case_id values must be integers, not booleans")
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} contains non-numeric case_id values") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite case_id values")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{source} contains non-integer case_id values")
    return pd.Series(values.astype(np.int64), index=series.index, name="case_id")


def _validate_paired_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    _require_exact_columns(frame, PUBLIC_PAIRED_COLUMNS, source)
    if len(frame) != EXPECTED_PAIRED_ROW_COUNT:
        raise ValueError(
            f"{source} must contain {EXPECTED_PAIRED_ROW_COUNT} rows; found {len(frame)}"
        )

    validated = frame.copy()
    validated["case_id"] = _coerce_case_ids(validated["case_id"], source)
    actual_case_ids = tuple(sorted(validated["case_id"].unique().tolist()))
    if actual_case_ids != EXPECTED_CASE_IDS:
        missing = sorted(set(EXPECTED_CASE_IDS).difference(actual_case_ids))
        unexpected = sorted(set(actual_case_ids).difference(EXPECTED_CASE_IDS))
        raise ValueError(
            f"{source} case IDs are not exactly 1..{EXPECTED_CASE_COUNT}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    if validated["source_group"].isna().any():
        raise ValueError(f"{source} contains missing source_group values")
    validated["source_group"] = validated["source_group"].astype(str)
    actual_groups = set(validated["source_group"].unique())
    expected_groups = set(SOURCE_GROUPS)
    if actual_groups != expected_groups:
        missing = sorted(expected_groups.difference(actual_groups))
        unexpected = sorted(actual_groups.difference(expected_groups))
        raise ValueError(
            f"{source} source groups differ from the expected seven; "
            f"missing={missing}, unexpected={unexpected}"
        )

    try:
        paired_values = pd.to_numeric(
            validated["paired_difference_kPa"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} contains non-numeric paired_difference_kPa values"
        ) from exc
    values = paired_values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite paired_difference_kPa values")
    validated["paired_difference_kPa"] = values

    key_columns = ["case_id", "source_group"]
    duplicate = validated.duplicated(key_columns, keep=False)
    if duplicate.any():
        examples = (
            validated.loc[duplicate, key_columns]
            .head()
            .to_dict(orient="records")
        )
        raise ValueError(f"{source} contains duplicate case/source keys: {examples}")

    expected_keys = pd.MultiIndex.from_product(
        [EXPECTED_CASE_IDS, SOURCE_GROUPS], names=key_columns
    )
    actual_keys = pd.MultiIndex.from_frame(validated[key_columns])
    missing_keys = expected_keys.difference(actual_keys)
    unexpected_keys = actual_keys.difference(expected_keys)
    if len(missing_keys) or len(unexpected_keys):
        raise ValueError(
            f"{source} does not contain the complete 1000 x 7 Cartesian product; "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )

    group_sizes = validated.groupby("source_group", sort=False).size().to_dict()
    if any(group_sizes.get(group, 0) != EXPECTED_CASE_COUNT for group in SOURCE_GROUPS):
        raise ValueError(
            f"{source} must contain {EXPECTED_CASE_COUNT} rows per source group; "
            f"found={group_sizes}"
        )
    return validated


def _observed_sign_counts(
    validated: pd.DataFrame, source_group: str
) -> tuple[int, int, int]:
    values = validated.loc[
        validated["source_group"] == source_group, "paired_difference_kPa"
    ].to_numpy(dtype=float)
    return (
        int(np.count_nonzero(values > 0.0)),
        int(np.count_nonzero(values < 0.0)),
        int(np.count_nonzero(values == 0.0)),
    )


def _validate_sign_regression(validated: pd.DataFrame, source: str) -> None:
    observed = {
        group: _observed_sign_counts(validated, group) for group in SOURCE_GROUPS
    }
    if observed != EXPECTED_SIGN_COUNTS:
        differences = {
            group: {"observed": observed[group], "expected": EXPECTED_SIGN_COUNTS[group]}
            for group in SOURCE_GROUPS
            if observed[group] != EXPECTED_SIGN_COUNTS[group]
        }
        raise ValueError(
            f"{source} sign counts differ from the canonical pressure results: {differences}"
        )


def load_paired_pressure_differences(path: Path) -> pd.DataFrame:
    """Read the canonical public paired-pressure CSV with strict validation."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Public paired-pressure CSV not found: {resolved}")
    frame = pd.read_csv(resolved, float_precision="round_trip")
    validated = _validate_paired_frame(frame, str(resolved))
    _validate_sign_regression(validated, str(resolved))
    return validated


def case_name(case_id: int) -> str:
    """Convert a publication case ID to its deterministic OGS directory name."""

    if isinstance(case_id, bool) or not isinstance(case_id, Integral):
        raise ValueError(f"case_id must be an integer in 1..{EXPECTED_CASE_COUNT}: {case_id!r}")
    value = int(case_id)
    if value < 1 or value > EXPECTED_CASE_COUNT:
        raise ValueError(f"case_id must be in 1..{EXPECTED_CASE_COUNT}: {value}")
    return f"case_{value:06d}"


def select_source_group(
    frame: pd.DataFrame,
    source_group: str,
    add_case_name: bool = False,
) -> pd.DataFrame:
    """Return one complete source group ordered by integer case ID."""

    validated = _validate_paired_frame(frame, "paired-pressure frame")
    _validate_sign_regression(validated, "paired-pressure frame")
    if source_group not in SOURCE_GROUPS:
        raise ValueError(
            f"Unknown source group {source_group!r}; expected one of {SOURCE_GROUPS}"
        )
    selected = (
        validated.loc[validated["source_group"] == source_group]
        .sort_values("case_id", kind="stable")
        .reset_index(drop=True)
    )
    if add_case_name:
        selected.insert(
            1,
            "case",
            [case_name(value) for value in selected["case_id"].to_numpy()],
        )
    return selected


def sign_counts(frame: pd.DataFrame, source_group: str) -> tuple[int, int, int]:
    """Return positive, negative, and zero counts for one source group."""

    selected = select_source_group(frame, source_group)
    return _observed_sign_counts(selected, source_group)


def sign_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical sign counts and fractions for all seven source groups."""

    validated = _validate_paired_frame(frame, "paired-pressure frame")
    _validate_sign_regression(validated, "paired-pressure frame")
    rows: list[dict[str, int | float | str]] = []
    for group in SOURCE_GROUPS:
        positive, negative, zero = _observed_sign_counts(validated, group)
        rows.append(
            {
                "source_group": group,
                "n_positive": positive,
                "n_negative": negative,
                "n_zero": zero,
                "fraction_positive": positive / EXPECTED_CASE_COUNT,
                "fraction_negative": negative / EXPECTED_CASE_COUNT,
                "fraction_zero": zero / EXPECTED_CASE_COUNT,
            }
        )
    return pd.DataFrame(rows)


def build_ecdf(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the seven ECDFs using ranks 1..1000 divided by 1000."""

    validated = _validate_paired_frame(frame, "paired-pressure frame")
    _validate_sign_regression(validated, "paired-pressure frame")
    rows: list[pd.DataFrame] = []
    for group in SOURCE_GROUPS:
        values = validated.loc[
            validated["source_group"] == group, "paired_difference_kPa"
        ].to_numpy(dtype=float)
        values = np.sort(values)
        count = len(values)
        rows.append(
            pd.DataFrame(
                {
                    "source_group": group,
                    "rank": np.arange(1, count + 1, dtype=np.int64),
                    "ecdf_probability": np.arange(1, count + 1, dtype=float) / count,
                    "paired_difference_kPa": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            "source_group",
            "rank",
            "ecdf_probability",
            "paired_difference_kPa",
        ],
    ]


def _validate_parameter_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    _require_exact_columns(frame, PUBLIC_PARAMETER_COLUMNS, source)
    if len(frame) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"{source} must contain {EXPECTED_CASE_COUNT} rows; found {len(frame)}"
        )

    validated = frame.copy()
    validated["case_id"] = _coerce_case_ids(validated["case_id"], source)
    actual_case_ids = tuple(sorted(validated["case_id"].unique().tolist()))
    if actual_case_ids != EXPECTED_CASE_IDS:
        raise ValueError(f"{source} case IDs are not exactly 1..{EXPECTED_CASE_COUNT}")
    if validated["case_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate case_id values")

    axes = list(PUBLIC_PARAMETER_COLUMNS[1:])
    for column in axes:
        try:
            numeric = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} contains non-numeric {column} values") from exc
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains non-finite {column} values")
        if np.any(values <= 0.0):
            raise ValueError(f"{source} contains non-positive {column} values")
        validated[column] = values

    axis_values = [np.sort(validated[column].unique()) for column in axes]
    cardinalities = {
        column: len(values) for column, values in zip(axes, axis_values)
    }
    if any(count != EXPECTED_AXIS_VALUE_COUNT for count in cardinalities.values()):
        raise ValueError(
            f"{source} must have 10 unique values on every parameter axis; "
            f"found={cardinalities}"
        )
    if validated.duplicated(axes).any():
        raise ValueError(f"{source} contains duplicate parameter combinations")

    actual_combinations = set(map(tuple, validated[axes].to_numpy(dtype=float)))
    expected_combinations = {
        (float(leduc), float(dz), float(gouge))
        for leduc in axis_values[0]
        for dz in axis_values[1]
        for gouge in axis_values[2]
    }
    if actual_combinations != expected_combinations:
        raise ValueError(
            f"{source} is not the complete 10 x 10 x 10 Cartesian design; "
            f"missing={len(expected_combinations.difference(actual_combinations))}, "
            f"unexpected={len(actual_combinations.difference(expected_combinations))}"
        )
    return validated


def load_parameter_design(path: Path) -> pd.DataFrame:
    """Read and validate the four-column public parameter design."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Public parameter-design CSV not found: {resolved}")
    frame = pd.read_csv(resolved, float_precision="round_trip")
    return _validate_parameter_frame(frame, str(resolved))


def join_parameter_design(
    paired: pd.DataFrame,
    params: pd.DataFrame,
    source_group: str = "well15",
    add_case_name: bool = True,
) -> pd.DataFrame:
    """Join one paired source group to the complete parameter design by case ID."""

    selected = select_source_group(
        paired,
        source_group,
        add_case_name=add_case_name,
    )
    parameters = _validate_parameter_frame(params, "parameter-design frame")
    joined = parameters.merge(
        selected,
        on="case_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    join_counts = joined["_merge"].value_counts().to_dict()
    if join_counts.get("both", 0) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Incomplete parameter/paired-data join: {join_counts}")
    return (
        joined.drop(columns="_merge")
        .sort_values("case_id", kind="stable")
        .reset_index(drop=True)
    )
