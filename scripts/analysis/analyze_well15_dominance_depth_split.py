#!/usr/bin/env python3
"""Reproduce the manuscript's conditional upper/lower Fault S pressures.

The calculation is intentionally limited to the four all-wells pressure
medians reported in the Results.  The fixed Well 15 dominance classes are read
from the public paired-pressure table. For each realization, nodal pressure at
model output 516 (November 2022) is averaged with the stored lumped nodal
weights over the stored Gamma mask split at z = -4 km:

    upper Gamma: z > -4 km
    lower Gamma: z <= -4 km

No triangle clipping, interpolation, geometric redefinition, depth profile,
threshold coverage, source-group map, or new case classification is performed.
Large OGS arrays remain read-only local inputs and generated tables are written
under the ignored outputs directory by default.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_ROOT_CANDIDATE = Path(__file__).resolve().parents[3]
REPO_ROOT = (
    CANONICAL_ROOT_CANDIDATE
    if (CANONICAL_ROOT_CANDIDATE / "src" / "peace_river_ogs").is_dir()
    else Path(__file__).resolve().parents[2]
)
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import paired_pressure_data


DATA_PACKAGE_ROOT = (
    REPO_ROOT / "zenodo_release" / "data_package"
    if REPO_ROOT == CANONICAL_ROOT_CANDIDATE
    else REPO_ROOT.parent / "data_package"
)
DEFAULT_CONFIG = REPO_ROOT / "config" / "local_paths.local.yml"
DEFAULT_PAIRED_CSV = DATA_PACKAGE_ROOT / "paired_pressure_differences.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "analysis" / "well15_dominance_depth_split"
)

ALL_WELLS_COLLECTION = "cases_pr_all_wells"
EXPECTED_CASES = tuple(f"case_{index:06d}" for index in range(1, 1001))
MODEL_TIMES = (516.0,)
CALENDAR_MONTHS = ("November 2022",)
EXPECTED_CLASS_COUNTS = {"M_plus": 266, "M_minus": 734}
EXPECTED_PRESSURE_SHAPE = (541, 1240)
EXPECTED_PRESSURE_DTYPE = np.dtype("float32")
EXPECTED_GAMMA_NODES = 821
SPLIT_ELEVATION_KM = -4.0
REGION_ORDER = ("full_gamma", "upper_gamma", "lower_gamma")

CASE_METRICS_NAME = "well15_dominance_depth_split_case_metrics.csv"
SUMMARY_NAME = "well15_dominance_depth_split_summary.csv"
METADATA_NAME = "well15_dominance_depth_split_metadata.json"

# Full-precision values from the controlled analysis that supplied the Results.
# They are regression targets, not inputs to the calculation.
MANUSCRIPT_REFERENCE_MEDIANS_PA = {
    ("M_plus", "upper_gamma"): 12795.841518859281,
    ("M_plus", "lower_gamma"): 61.484982425012355,
    ("M_minus", "upper_gamma"): 115596.61011171665,
    ("M_minus", "lower_gamma"): 51119.349574457963,
}
MANUSCRIPT_DISPLAY_KPA = {
    ("M_plus", "upper_gamma"): "12.8",
    ("M_plus", "lower_gamma"): "0.061",
    ("M_minus", "upper_gamma"): "115.6",
    ("M_minus", "lower_gamma"): "51.1",
}

AREA_RTOL = 1.0e-14
AREA_ATOL_M2 = 1.0e-8
RECOMBINATION_RTOL = 1.0e-12
RECOMBINATION_ATOL_PA = 1.0e-8
REFERENCE_RTOL = 1.0e-13
REFERENCE_ATOL_PA = 1.0e-10


@dataclass(frozen=True)
class Region:
    """A nodal subset of Gamma with its inherited lumped nodal weights."""

    name: str
    mask: np.ndarray
    weights_m2: np.ndarray
    node_count: int
    area_m2: float
    elevation_rule: str


@dataclass(frozen=True)
class ReferenceInputs:
    """Mesh and time arrays shared by all all-wells realizations."""

    points: np.ndarray
    time: np.ndarray
    depth_km: np.ndarray
    weights_m2: np.ndarray
    gamma_mask: np.ndarray
    time_indices: tuple[int, ...]
    regions: dict[str, Region]
    array_sha256: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the four conditional upper/lower all-wells pressure "
            "medians reported for the Well 15 dominance analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional untracked local YAML path configuration, such as "
            f"{DEFAULT_CONFIG.relative_to(REPO_ROOT)}."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="OGS results_raw directory; overrides ogs_results_raw_dir in --config.",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=DEFAULT_PAIRED_CSV,
        help="Public paired-pressure table defining the fixed classes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory; defaults below local_output_dir or outputs/.",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Config must contain a YAML mapping: {path}")
        return data

    # The local path template is deliberately a flat mapping.  Supporting it
    # without PyYAML keeps this numerical reproducer runnable in the minimal
    # manuscript environment.
    data: dict[str, object] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in line:
            raise RuntimeError(
                "PyYAML is unavailable and the config is not a flat mapping at "
                f"{path}:{line_number}"
            )
        key, raw_value = line.split(":", 1)
        try:
            value: object = ast.literal_eval(raw_value.strip())
        except (SyntaxError, ValueError):
            value = raw_value.strip()
        data[key.strip()] = value
    return data


def configured_path(config: Mapping[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_paths(
    args: argparse.Namespace, config: Mapping[str, object]
) -> tuple[Path, Path]:
    raw_root = (
        args.raw_root.expanduser().resolve()
        if args.raw_root is not None
        else configured_path(config, "ogs_results_raw_dir")
    )
    if raw_root is None:
        raise RuntimeError("Provide --raw-root or ogs_results_raw_dir in --config")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"OGS results_raw directory not found: {raw_root}")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        local_output = configured_path(config, "local_output_dir")
        output_dir = (
            local_output / "analysis" / "well15_dominance_depth_split"
            if local_output is not None
            else DEFAULT_OUTPUT_DIR
        )
    return raw_root, output_dir.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_fixed_classification(paired_csv: Path) -> pd.DataFrame:
    """Load the public 266/734 class membership without reclassification."""

    paired_csv = paired_csv.expanduser().resolve()
    frame = paired_pressure_data.load_paired_pressure_differences(paired_csv)
    public_selected = paired_pressure_data.select_source_group(
        frame,
        "well15",
        add_case_name=True,
    )
    paired_kpa = public_selected["paired_difference_kPa"].to_numpy(dtype=float)
    selected = public_selected.loc[:, ["case"]].copy()
    # Retain the existing local case-metrics output in Pa.
    selected["paired_difference"] = paired_kpa * 1.0e3
    if tuple(selected["case"]) != EXPECTED_CASES:
        raise RuntimeError("Classification is not the complete case_000001..case_001000 set")
    expected_dominance = paired_kpa > 0.0
    if np.any(paired_kpa == 0.0):
        raise RuntimeError("Expected no zero paired differences in the fixed classification")

    selected["fixed_class"] = np.where(expected_dominance, "M_plus", "M_minus")
    counts = selected["fixed_class"].value_counts().to_dict()
    observed = {name: int(counts.get(name, 0)) for name in EXPECTED_CLASS_COUNTS}
    if observed != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            f"Fixed class counts differ from {EXPECTED_CLASS_COUNTS}: observed={observed}"
        )
    return selected


def pressure_dir(all_wells_root: Path, case: str) -> Path:
    return all_wells_root / case / "post" / "resample" / "fault"


def dieterich_dir(all_wells_root: Path, case: str) -> Path:
    return all_wells_root / case / "post" / "dieterich"


def build_regions(
    gamma_mask: np.ndarray, elevation_km: np.ndarray, weights_m2: np.ndarray
) -> dict[str, Region]:
    """Partition the stored Gamma nodes without modifying their nodal weights."""

    gamma = np.asarray(gamma_mask, dtype=bool)
    elevation = np.asarray(elevation_km, dtype=np.float64)
    weights = np.asarray(weights_m2, dtype=np.float64)
    if gamma.shape != elevation.shape or gamma.shape != weights.shape:
        raise ValueError("Gamma mask, elevation, and weights must have the same shape")
    if np.any(~np.isfinite(elevation)) or np.any(~np.isfinite(weights)):
        raise ValueError("Elevation and weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("Lumped nodal weights must be strictly positive")

    masks = {
        "full_gamma": gamma.copy(),
        "upper_gamma": gamma & (elevation > SPLIT_ELEVATION_KM),
        "lower_gamma": gamma & (elevation <= SPLIT_ELEVATION_KM),
    }
    if np.any(masks["upper_gamma"] & masks["lower_gamma"]):
        raise RuntimeError("Upper and lower Gamma overlap")
    if not np.array_equal(
        masks["full_gamma"], masks["upper_gamma"] | masks["lower_gamma"]
    ):
        raise RuntimeError("Upper and lower Gamma do not completely cover Gamma")

    rules = {
        "full_gamma": "stored canonical Gamma mask",
        "upper_gamma": "stored Gamma mask and elevation_km > -4",
        "lower_gamma": "stored Gamma mask and elevation_km <= -4",
    }
    regions: dict[str, Region] = {}
    for name in REGION_ORDER:
        mask = masks[name]
        region_weights = np.where(mask, weights, 0.0)
        area = float(np.sum(region_weights, dtype=np.float64))
        if not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(f"Region has nonpositive or non-finite area: {name}")
        regions[name] = Region(
            name=name,
            mask=mask,
            weights_m2=region_weights,
            node_count=int(np.count_nonzero(mask)),
            area_m2=area,
            elevation_rule=rules[name],
        )

    full_area = regions["full_gamma"].area_m2
    partition_area = (
        regions["upper_gamma"].area_m2 + regions["lower_gamma"].area_m2
    )
    if not np.isclose(
        full_area, partition_area, rtol=AREA_RTOL, atol=AREA_ATOL_M2
    ):
        raise RuntimeError(
            "Discrete nodal area conservation failed: "
            f"full={full_area}, upper_plus_lower={partition_area}"
        )
    return regions


def load_reference_inputs(all_wells_root: Path) -> ReferenceInputs:
    case = EXPECTED_CASES[0]
    pdir = pressure_dir(all_wells_root, case)
    ddir = dieterich_dir(all_wells_root, case)
    points = np.load(pdir / "points.npy", allow_pickle=False)
    time = np.load(pdir / "time.npy", allow_pickle=False)
    depth_km = np.load(ddir / "depth_km.npy", allow_pickle=False)
    weights_m2 = np.load(ddir / "weights.npy", allow_pickle=False)
    gamma_raw = np.load(ddir / "mask_seismo.npy", allow_pickle=False)
    dieterich_time = np.load(ddir / "time.npy", allow_pickle=False)
    pressure = np.load(pdir / "pressure.npy", mmap_mode="r", allow_pickle=False)

    if (
        pressure.shape != EXPECTED_PRESSURE_SHAPE
        or pressure.dtype != EXPECTED_PRESSURE_DTYPE
    ):
        raise RuntimeError(
            f"Unexpected pressure array: shape={pressure.shape}, dtype={pressure.dtype}"
        )
    if points.shape != (EXPECTED_PRESSURE_SHAPE[1], 3):
        raise RuntimeError(f"Unexpected points shape: {points.shape}")
    if time.ndim != 1 or not np.isfinite(time).all():
        raise RuntimeError("Reference time vector must be one-dimensional and finite")
    if not np.array_equal(time, dieterich_time):
        raise RuntimeError("Resampled-pressure and Dieterich time vectors differ")
    node_shape = (EXPECTED_PRESSURE_SHAPE[1],)
    if depth_km.shape != node_shape or weights_m2.shape != node_shape or gamma_raw.shape != node_shape:
        raise RuntimeError("Depth, weight, or Gamma-mask shape differs from the node axis")
    if not np.isfinite(points).all() or not np.isfinite(depth_km).all():
        raise RuntimeError("Reference points and depth must be finite")
    unique_mask = np.unique(gamma_raw)
    if not np.all(np.isin(unique_mask, [0, 1])):
        raise RuntimeError(f"Stored Gamma mask is not binary: {unique_mask.tolist()}")
    gamma_mask = gamma_raw.astype(bool)
    if int(gamma_mask.sum()) != EXPECTED_GAMMA_NODES:
        raise RuntimeError(
            f"Expected {EXPECTED_GAMMA_NODES} Gamma nodes; found {int(gamma_mask.sum())}"
        )
    if not np.array_equal(gamma_mask, (depth_km >= 2.0) & (depth_km <= 7.0)):
        raise RuntimeError("Stored Gamma mask differs from the canonical 2..7 km depth mask")

    elevation_km = np.asarray(points[:, 2], dtype=np.float64) / 1000.0
    if not np.array_equal(
        gamma_mask, (elevation_km >= -7.0) & (elevation_km <= -2.0)
    ):
        raise RuntimeError("Stored Gamma mask differs from canonical model elevations")
    time_indices: list[int] = []
    for model_time in MODEL_TIMES:
        positions = np.flatnonzero(time == model_time)
        if positions.size != 1:
            raise RuntimeError(
                f"Model time {model_time:g} occurs {positions.size} times"
            )
        time_indices.append(int(positions[0]))

    regions = build_regions(gamma_mask, elevation_km, weights_m2)
    hashes = {
        "points": sha256_array(points),
        "time": sha256_array(time),
        "depth_km": sha256_array(depth_km),
        "weights_m2": sha256_array(weights_m2),
        "gamma_mask": sha256_array(gamma_raw),
    }
    return ReferenceInputs(
        points=np.asarray(points),
        time=np.asarray(time),
        depth_km=np.asarray(depth_km),
        weights_m2=np.asarray(weights_m2),
        gamma_mask=gamma_mask,
        time_indices=tuple(time_indices),
        regions=regions,
        array_sha256=hashes,
    )


def mean_selected_outputs(selected_pressure: np.ndarray) -> np.ndarray:
    """Average the selected nodal fields in float64.

    For the canonical November 2022 analysis, this returns the single selected
    output after conversion to float64.
    """

    values = np.asarray(selected_pressure, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(MODEL_TIMES):
        raise ValueError(
            f"Expected {len(MODEL_TIMES)} selected output rows; got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Selected pressure contains non-finite values")
    return np.mean(values, axis=0, dtype=np.float64)


def lumped_area_mean(field: np.ndarray, region: Region) -> float:
    values = np.asarray(field, dtype=np.float64)
    if values.shape != region.weights_m2.shape:
        raise ValueError("Field and region weights have different shapes")
    if not np.isfinite(values).all():
        raise ValueError("Pressure field contains non-finite values")
    return float(values @ region.weights_m2 / region.area_m2)


def validate_case_static_inputs(
    all_wells_root: Path, case: str, reference: ReferenceInputs
) -> np.memmap:
    pdir = pressure_dir(all_wells_root, case)
    ddir = dieterich_dir(all_wells_root, case)
    required = (
        pdir / "pressure.npy",
        pdir / "points.npy",
        pdir / "time.npy",
        ddir / "depth_km.npy",
        ddir / "weights.npy",
        ddir / "mask_seismo.npy",
        ddir / "time.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for {case}: {missing}")

    pressure = np.load(required[0], mmap_mode="r", allow_pickle=False)
    if (
        pressure.shape != EXPECTED_PRESSURE_SHAPE
        or pressure.dtype != EXPECTED_PRESSURE_DTYPE
    ):
        raise RuntimeError(
            f"Unexpected pressure array for {case}: shape={pressure.shape}, dtype={pressure.dtype}"
        )
    candidate_mask_raw = np.load(required[5], allow_pickle=False)
    unique_mask = np.unique(candidate_mask_raw)
    if not np.all(np.isin(unique_mask, [0, 1])):
        raise RuntimeError(
            f"Stored Gamma mask is not binary in {case}: {unique_mask.tolist()}"
        )
    comparisons = (
        ("points", np.load(required[1], allow_pickle=False), reference.points),
        ("pressure time", np.load(required[2], allow_pickle=False), reference.time),
        ("depth", np.load(required[3], allow_pickle=False), reference.depth_km),
        ("weights", np.load(required[4], allow_pickle=False), reference.weights_m2),
        ("Gamma mask", candidate_mask_raw.astype(bool), reference.gamma_mask),
        ("Dieterich time", np.load(required[6], allow_pickle=False), reference.time),
    )
    for label, candidate, expected in comparisons:
        if not np.array_equal(candidate, expected):
            raise RuntimeError(f"{label} differs from the reference in {case}")
    return pressure


def analyze_cases(
    all_wells_root: Path,
    reference: ReferenceInputs,
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    actual_cases = sorted(
        path.name
        for path in all_wells_root.glob("case_[0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    if tuple(actual_cases) != EXPECTED_CASES:
        missing = sorted(set(EXPECTED_CASES).difference(actual_cases))
        extra = sorted(set(actual_cases).difference(EXPECTED_CASES))
        raise RuntimeError(
            "All-wells case inventory differs from the canonical 1000 cases: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    class_index = classification.set_index("case")
    rows: list[dict[str, object]] = []
    selected_pressure_digest = hashlib.sha256()
    recombination_differences: list[float] = []

    full = reference.regions["full_gamma"]
    upper = reference.regions["upper_gamma"]
    lower = reference.regions["lower_gamma"]
    for case_index, case in enumerate(EXPECTED_CASES, start=1):
        pressure = validate_case_static_inputs(all_wells_root, case, reference)
        selected_source = np.asarray(pressure[list(reference.time_indices), :])
        if selected_source.shape != (len(MODEL_TIMES), EXPECTED_PRESSURE_SHAPE[1]):
            raise RuntimeError(
                f"Selected pressure rows have an unexpected shape in {case}: {selected_source.shape}"
            )
        if not np.isfinite(selected_source).all():
            raise RuntimeError(f"Selected pressure contains non-finite values in {case}")
        selected_pressure_digest.update(case.encode("ascii"))
        selected_pressure_digest.update(np.ascontiguousarray(selected_source).view(np.uint8))

        mean_field = mean_selected_outputs(selected_source)
        means = {
            name: lumped_area_mean(mean_field, reference.regions[name])
            for name in REGION_ORDER
        }
        recombined = (
            upper.area_m2 * means["upper_gamma"]
            + lower.area_m2 * means["lower_gamma"]
        ) / full.area_m2
        difference = abs(means["full_gamma"] - recombined)
        recombination_differences.append(difference)
        if not np.isclose(
            means["full_gamma"],
            recombined,
            rtol=RECOMBINATION_RTOL,
            atol=RECOMBINATION_ATOL_PA,
        ):
            raise RuntimeError(
                f"Upper/lower recombination failed for {case}: "
                f"full={means['full_gamma']}, recombined={recombined}"
            )

        fixed = class_index.loc[case]
        rows.append(
            {
                "case": case,
                "fixed_class": str(fixed["fixed_class"]),
                "paired_difference_Pa": float(fixed["paired_difference"]),
                "full_gamma_pressure_Pa": means["full_gamma"],
                "upper_gamma_pressure_Pa": means["upper_gamma"],
                "lower_gamma_pressure_Pa": means["lower_gamma"],
            }
        )
        if case_index == 1 or case_index % 100 == 0:
            print(f"[INFO] analyzed all-wells cases {case_index}/1000")

    metrics = pd.DataFrame(rows)
    if not np.isfinite(
        metrics[
            [
                "paired_difference_Pa",
                "full_gamma_pressure_Pa",
                "upper_gamma_pressure_Pa",
                "lower_gamma_pressure_Pa",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite values reached the case metrics table")
    validation = {
        "case_inventory_count": len(metrics),
        "selected_pressure_finite": True,
        "selected_pressure_sha256": selected_pressure_digest.hexdigest(),
        "upper_lower_recombination_max_abs_difference_Pa": float(
            max(recombination_differences)
        ),
    }
    return metrics, validation


def summarize_case_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fixed_class in ("M_plus", "M_minus"):
        subset = metrics.loc[metrics["fixed_class"] == fixed_class]
        expected_count = EXPECTED_CLASS_COUNTS[fixed_class]
        if len(subset) != expected_count:
            raise RuntimeError(
                f"Unexpected {fixed_class} count in case metrics: {len(subset)}"
            )
        for region in ("upper_gamma", "lower_gamma"):
            values = subset[f"{region}_pressure_Pa"].to_numpy(dtype=float)
            median_pa = float(np.quantile(values, 0.5, method="linear"))
            key = (fixed_class, region)
            rows.append(
                {
                    "fixed_class": fixed_class,
                    "class_definition": (
                        "Well 15 dominance: paired difference > 0"
                        if fixed_class == "M_plus"
                        else "remaining realizations: paired difference < 0"
                    ),
                    "case_count": expected_count,
                    "time_aggregation": "November 2022 monthly output 516",
                    "region": region,
                    "region_elevation_rule": (
                        "stored Gamma nodes with elevation_km > -4"
                        if region == "upper_gamma"
                        else "stored Gamma nodes with elevation_km <= -4"
                    ),
                    "statistic": "conditional ensemble median",
                    "median_pressure_Pa": median_pa,
                    "median_pressure_kPa": median_pa / 1000.0,
                    "manuscript_display_kPa": MANUSCRIPT_DISPLAY_KPA[key],
                }
            )
    return pd.DataFrame(rows)


def validate_reference_medians(summary: pd.DataFrame) -> float:
    keys = list(
        zip(
            summary["fixed_class"].astype(str),
            summary["region"].astype(str),
            strict=True,
        )
    )
    expected_keys = set(MANUSCRIPT_REFERENCE_MEDIANS_PA)
    if len(keys) != len(expected_keys) or len(set(keys)) != len(keys):
        raise RuntimeError(
            "Manuscript regression summary must contain each of the four class/region keys once"
        )
    if set(keys) != expected_keys:
        raise RuntimeError(
            "Manuscript regression summary keys differ from the expected four: "
            f"observed={sorted(keys)}, expected={sorted(expected_keys)}"
        )
    maximum_difference = 0.0
    for row in summary.itertuples(index=False):
        key = (str(row.fixed_class), str(row.region))
        expected = MANUSCRIPT_REFERENCE_MEDIANS_PA[key]
        observed = float(row.median_pressure_Pa)
        difference = abs(observed - expected)
        maximum_difference = max(maximum_difference, difference)
        if not np.isclose(
            observed,
            expected,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL_PA,
        ):
            raise RuntimeError(
                f"Manuscript regression value changed for {key}: "
                f"observed={observed}, expected={expected}"
            )
    return maximum_difference


def relative_identifier(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return "<LOCAL_PATH_EXCLUDED>"


def write_outputs(
    output_dir: Path,
    paired_csv: Path,
    reference: ReferenceInputs,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    validation: Mapping[str, object],
    reference_median_max_difference_pa: float,
    script_sha256: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / CASE_METRICS_NAME
    summary_path = output_dir / SUMMARY_NAME
    metadata_path = output_dir / METADATA_NAME
    metrics.to_csv(metrics_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")

    regions = reference.regions
    metadata = {
        "schema_version": 1,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script": relative_identifier(Path(__file__)),
        "script_sha256": script_sha256,
        "classification": {
            "source": relative_identifier(paired_csv),
            "source_sha256": sha256_file(paired_csv),
            "rule": "M_plus if Well 15 paired_difference_kPa > 0; M_minus if < 0",
            "counts": EXPECTED_CLASS_COUNTS,
            "recalculated": False,
        },
        "raw_input": {
            "identifier": "<LOCAL_OGS_RESULTS_RAW_DIR>/cases_pr_all_wells",
            "stored_in_repository": False,
            "selected_pressure_sha256": validation["selected_pressure_sha256"],
            "reference_array_sha256": reference.array_sha256,
        },
        "time_aggregation": {
            "model_outputs": list(MODEL_TIMES),
            "calendar_months": list(CALENDAR_MONTHS),
            "operation_order": (
                "use pressure at output 516; then compute the lumped-nodal area mean"
            ),
        },
        "gamma_partition": {
            "definition": "stored Gamma mask with stored full-surface lumped nodal weights",
            "split_elevation_km": SPLIT_ELEVATION_KM,
            "upper_rule": "elevation_km > -4",
            "lower_rule": "elevation_km <= -4",
            "triangle_clipping_or_boundary_interpolation": False,
            "regions": {
                name: {
                    "node_count": region.node_count,
                    "area_m2": region.area_m2,
                    "area_km2": region.area_m2 / 1.0e6,
                    "elevation_rule": region.elevation_rule,
                }
                for name, region in regions.items()
            },
            "area_conservation_residual_m2": (
                regions["full_gamma"].area_m2
                - regions["upper_gamma"].area_m2
                - regions["lower_gamma"].area_m2
            ),
        },
        "validation": {
            **validation,
            "fixed_class_counts": EXPECTED_CLASS_COUNTS,
            "area_conservation_pass": True,
            "upper_lower_recombination_pass": True,
            "manuscript_reference_medians_pass": True,
            "manuscript_reference_median_max_abs_difference_Pa": (
                reference_median_max_difference_pa
            ),
        },
        "summary": summary.to_dict(orient="records"),
        "scope": {
            "performed": [
                "fixed-class conditional all-wells pressure medians",
                "stored nodal Gamma split at elevation -4 km",
                "casewise November 2022 output 516",
            ],
            "not_performed": [
                "case reclassification",
                "source-group pressure maps",
                "threshold coverage",
                "pressure ratios",
                "sign transitions",
                "depth profiles or boundary sensitivity",
                "triangle clipping or geometric Gamma redefinition",
                "figure generation",
            ],
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics_path, summary_path, metadata_path


def main() -> int:
    script_path = Path(__file__).resolve()
    script_sha_start = sha256_file(script_path)
    args = parse_args()
    config = load_config(args.config)
    raw_root, output_dir = resolve_paths(args, config)
    paired_csv = args.paired_csv.expanduser().resolve()
    classification = load_fixed_classification(paired_csv)
    all_wells_root = raw_root / ALL_WELLS_COLLECTION
    if not all_wells_root.is_dir():
        raise FileNotFoundError(f"All-wells collection not found: {all_wells_root}")

    reference = load_reference_inputs(all_wells_root)
    metrics, validation = analyze_cases(all_wells_root, reference, classification)
    summary = summarize_case_metrics(metrics)
    reference_difference = validate_reference_medians(summary)
    script_sha_end = sha256_file(script_path)
    if script_sha_end != script_sha_start:
        raise RuntimeError("The analysis script changed while the run was in progress")
    written = write_outputs(
        output_dir,
        paired_csv,
        reference,
        metrics,
        summary,
        validation,
        reference_difference,
        script_sha_start,
    )

    print("[OK] fixed classification: M_plus=266, M_minus=734, zero=0")
    print(
        "[OK] Gamma nodes: "
        f"full={reference.regions['full_gamma'].node_count}, "
        f"upper={reference.regions['upper_gamma'].node_count}, "
        f"lower={reference.regions['lower_gamma'].node_count}"
    )
    for row in summary.itertuples(index=False):
        print(
            f"[OK] {row.fixed_class} {row.region}: "
            f"{row.median_pressure_Pa:.17g} Pa "
            f"({row.manuscript_display_kPa} kPa in manuscript)"
        )
    for path in written:
        print(f"[OK] wrote: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
