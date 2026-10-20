#!/usr/bin/env python3
"""Plot the combined pressure-dominance Figure 3 and its linear diagnostic.

The case groups are defined only from the paired differences stored in the
public paired-pressure CSV.  The displayed fields are the all-wells
monthly pressure perturbations at model time 516 (November 2022).  The script
never recomputes the grouping variable from the spatial pressure arrays.

The canonical manuscript output combines pressure-only ECDFs for Wells 15 and
10 with the logarithmic conditional pressure maps.  The map-only linear figure
is retained as a local diagnostic.

The raw OGS arrays and retained postprocessed products are read-only inputs.
Pressure snapshots are copied to short-lived float32 memory maps, and nodewise
medians are calculated in chunks without loading all cases and times into RAM.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "peace_river_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

if __package__:
    from . import plot_well10_well15_dominance_ecdf as dominance_ecdf
else:
    import plot_well10_well15_dominance_ecdf as dominance_ecdf


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import paired_pressure_data


CANONICAL_REPO_ROOT = (
    WORKFLOW_ROOT
    if (WORKFLOW_ROOT / "src" / "peace_river_ogs").is_dir()
    else None
)
REPO_ROOT = WORKFLOW_ROOT
DATA_PACKAGE_ROOT = (
    REPO_ROOT / "zenodo_release" / "data_package"
    if CANONICAL_REPO_ROOT is not None
    else REPO_ROOT.parent / "data_package"
)
DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "local_paths.local.yml"
    if CANONICAL_REPO_ROOT is not None
    else None
)
DEFAULT_PAIRED_CSV = DATA_PACKAGE_ROOT / "paired_pressure_differences.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "figures" / "well15_paired_sign_pressure_maps"
)
COMBINED_OUTPUT_NAME = "paired_pressure_dominance_and_fields.pdf"
REFERENCE_TIME_SCRIPT = (
    REPO_ROOT / "src" / "peace_river_ogs" / "analysis"
    if CANONICAL_REPO_ROOT is not None
    else REPO_ROOT / "scripts" / "analysis"
) / "analyze_post_mainshock_pressure_increment.py"
REFERENCE_GEOMETRY_SCRIPT = (
    REPO_ROOT / "banff_workshop2026" / "scripts" / "make_surface_pressure_animation.py"
)
REFERENCE_GEOMETRY_WORKFLOW = (
    str(REFERENCE_GEOMETRY_SCRIPT)
    if CANONICAL_REPO_ROOT is not None
    else (
        "not included in the code package; canonical repository path: "
        "banff_workshop2026/scripts/make_surface_pressure_animation.py"
    )
)
REFERENCE_AREA_SCRIPT = (
    REPO_ROOT / "src" / "peace_river_ogs" / "postprocess" / "case_level"
    if CANONICAL_REPO_ROOT is not None
    else REPO_ROOT / "scripts" / "postprocess" / "case_level"
) / "batch_depthband_area_average_from_file.py"
CASE_COLLECTION = "cases_pr_all_wells"
SURFACES = ("leduc_top", "fault")
REQUIRED_ARRAY_NAMES = ("pressure.npy", "points.npy", "time.npy")
EXPECTED_CASES = tuple(f"case_{index:06d}" for index in range(1, 1001))
DISPLAY_TIME = 516.0
DISPLAY_MONTH = "November 2022"
EXPECTED_POSITIVE = 266
EXPECTED_NEGATIVE = 734
EXPECTED_ZERO = 0
REQUESTED_FONT_FAMILY = "Liberation Sans"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
FIGURE_FONT_SIZE = 12.0
FAULT_ORIENTATION_FONT_SIZE = 9.0
PANEL_HEADING_LINE_SPACING = 1.2
PANEL_HEADING_X_AXES = -0.10
PANEL_HEADING_Y_AXES = 1.02
ECDF_PANEL_HEADING_Y_AXES = 1.06
PANEL_HEADING_SEPARATOR = " "
PANEL_HEADING_CONTINUATION_INDENT = "      "
ECDF_XLIM_KPA = (-200.0, 100.0)
MAP_SHARED_VERTICAL_LABEL = (
    "November 2022 nodewise median pressure perturbation from the all-wells case"
)
MAP_PANEL_TITLES = (
    (
        "With Well 15 pressure dominance\n266 realizations",
        "Without Well 15 pressure dominance\n734 realizations",
    ),
    (
        "With Well 15 pressure dominance\n266 realizations",
        "Without Well 15 pressure dominance\n734 realizations",
    ),
)
COMBINED_GRIDSPEC_WIDTH_RATIOS = (0.035, 0.035, 1.0, 1.0, 0.075)
COMBINED_GRIDSPEC_HEIGHT_RATIOS = (1.0, 0.05, 1.0, 1.0)
MAP_DECORATION_GAP_FIGURE = 0.008
MAP_ROW_TITLE_WIDTH_FIGURE = 0.026
MAP_SHARED_LABEL_WIDTH_FIGURE = 0.026
MAP_FIGURE_SIZE_INCHES = (9.6, 7.7)
COMBINED_FIGURE_SIZE_INCHES = (9.6, 10.4)
LEDUC_CLOCKWISE_ROTATION_DEG = 49.43
LEDUC_MAX_EDGE_KM = 2.5
FAULT_MAX_EDGE_KM = 2.0
PRESSURE_TO_KPA = 1.0e-3
LOG_VMIN_KPA = 1.0
LOG_VMAX_KPA = 5000.0
COLORMAP_NAME = "viridis"
CONTOUR_LEVELS_KPA = (1.0, 10.0, 100.0, 1000.0)
COLORBAR_TICKS_KPA = (1.0, 10.0, 100.0, 1000.0, 5000.0)
FAULT_S_DOMAIN_MATERIAL_ID = 15
EXPECTED_FAULT_S_ALONG_STRIKE_LENGTH_KM = 25.08
EXPECTED_FAULT_S_OPENING_WALL_PAIR_COUNT = 17
EXPECTED_FAULT_S_OPENING_WALL_OFFSET_M = 129.903810568
FAULT_S_OPENING_CENTER_EXCLUSION_M = 1.0
FAULT_S_OPENING_WALL_TOLERANCE_M = 1.0e-3
FAULT_S_CENTER_PLANE_PHYSICAL_TAG = 110
FAULT_S_LEDUC_GEOMETRICAL_TAG = 596
FAULT_S_BASEMENT_GEOMETRICAL_TAG = 1220
EXPECTED_FAULT_S_CENTER_PLANE_TRIANGLE_COUNT = 2351
EXPECTED_FAULT_S_CENTER_PLANE_NODE_COUNT = 1240
EXPECTED_FAULT_S_CENTER_PLANE_GEOMETRICAL_TAGS = (
    489,
    596,
    765,
    862,
    951,
    1033,
    1125,
    1220,
)
EXPECTED_FAULT_S_LEDUC_TRIANGLE_COUNT = 82
EXPECTED_FAULT_S_BASEMENT_TRIANGLE_COUNT = 1791
EXPECTED_FAULT_S_LEDUC_OUTLINE_EDGE_COUNT = 40
EXPECTED_FAULT_S_BASEMENT_TOP_EDGE_COUNT = 27
EXPECTED_FAULT_S_LEDUC_DISPLAY_EXTENT_KM = (
    -12.2235502308,
    3.9583476301,
    1.0903395456,
    1.4523933221,
)
EXPECTED_FAULT_S_BASEMENT_TOP_DISPLAY_EXTENT_KM = (
    -12.2227753006,
    12.5567071354,
    1.3005133999,
    1.4523933221,
)
FAULT_STRATIGRAPHY_LABEL_X_KM = -11.25
MANUAL_CONTOUR_LABEL_POSITIONS_KM = {
    ("leduc_top", "positive", 1.0): ((-13.3, 1.3),),
    ("leduc_top", "positive", 10.0): ((-5.6, 5.2),),
    ("leduc_top", "positive", 100.0): ((3.7, 9.9),),
    ("leduc_top", "positive", 1000.0): ((15.2, 15.6),),
    ("leduc_top", "negative", 100.0): ((-10.6, 2.7),),
    ("leduc_top", "negative", 1000.0): ((15.8, 15.9),),
    ("fault", "positive", 10.0): (
        (-7.05, -2.06),
        (-6.87, -0.50),
        (8.09, -0.28),
    ),
    ("fault", "positive", 100.0): ((3.08, -2.39),),
}
DEFAULT_CHUNK_NODES = 1024
PNG_DPI = 600
PDF_RASTER_DPI = 300

# Constants copied from the canonical Banff display transform.
EPSG3400_A = 6378137.0
EPSG3400_INV_F = 298.257222101
EPSG3400_LON0_DEG = -115.0
EPSG3400_K0 = 0.9992
EPSG3400_FALSE_EASTING = 500000.0
EPSG3400_FALSE_NORTHING = 0.0
DEFAULT_MESH_X_CORNER_M = 262051.263
DEFAULT_MESH_Y_CORNER_M = 6245295.891
DEFAULT_MESH_STRIKE_DEG = 49.43

FIELD_KEYS = {
    ("leduc_top", "positive"): "leduc_top_positive_median_pressure_Pa",
    ("leduc_top", "negative"): "leduc_top_negative_median_pressure_Pa",
    ("fault", "positive"): "fault_positive_median_pressure_Pa",
    ("fault", "negative"): "fault_negative_median_pressure_Pa",
}


@dataclass(frozen=True)
class SurfaceInventory:
    surface: str
    pressure_shape: tuple[int, int]
    pressure_dtype: str
    node_count: int
    time_count: int
    display_time_index: int
    points: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class FaultProjection:
    x_km: np.ndarray
    elevation_km: np.ndarray
    xy_center_m: np.ndarray
    along_axis: np.ndarray


@dataclass(frozen=True)
class DisplayGeometry:
    surface: str
    x_km: np.ndarray
    y_km: np.ndarray
    tri: mtri.Triangulation
    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    max_edge_km: float


@dataclass(frozen=True)
class WellOverlay:
    labels: tuple[str, ...]
    x_km: np.ndarray
    y_km: np.ndarray
    source_xyz_m: np.ndarray
    fault_x_km: np.ndarray
    fault_y_km: np.ndarray
    fault_cross_axis_offset_km: np.ndarray
    fault_inside: np.ndarray
    reference_csv: Path
    mesh_dir: Path


@dataclass(frozen=True)
class EarthquakeOverlay:
    leduc_x_km: np.ndarray
    leduc_y_km: np.ndarray
    fault_x_km: np.ndarray
    fault_y_km: np.ndarray
    fault_inside: np.ndarray
    catalog_path: Path


@dataclass(frozen=True)
class FaultGuideOverlay:
    leduc_x_km: np.ndarray
    leduc_y_km: np.ndarray


@dataclass(frozen=True)
class FaultStratigraphyOverlay:
    leduc_x_km: np.ndarray
    leduc_y_km: np.ndarray
    basement_top_x_km: np.ndarray
    basement_top_y_km: np.ndarray
    leduc_label_xy_km: tuple[float, float]
    basement_label_xy_km: tuple[float, float]


class ValidationProblems:
    """Count all validation failures while retaining a small set of examples."""

    def __init__(self, sample_limit: int = 8) -> None:
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[str]] = {}
        self.sample_limit = sample_limit

    def add(self, category: str, message: str) -> None:
        self.counts[category] += 1
        samples = self.samples.setdefault(category, [])
        if len(samples) < self.sample_limit:
            samples.append(message)

    def any(self) -> bool:
        return bool(self.counts)

    def format(self) -> str:
        lines = ["Input validation failed:"]
        for category in sorted(self.counts):
            lines.append(f"  {category}: {self.counts[category]}")
            lines.extend(f"    example: {item}" for item in self.samples[category])
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the combined paired-pressure dominance manuscript figure "
            "and map-only linear diagnostic."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paired-csv", type=Path, default=DEFAULT_PAIRED_CSV)
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help=(
            "Combined Figure 3 PDF path. Defaults to "
            "paired_pressure_dominance_and_fields.pdf inside --output-dir."
        ),
    )
    parser.add_argument(
        "--temporary-parent",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Existing directory that will contain one automatically removed temporary directory.",
    )
    parser.add_argument("--chunk-nodes", type=int, default=DEFAULT_CHUNK_NODES)
    parser.add_argument("--png-dpi", type=int, default=PNG_DPI)
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional path to a Liberation Sans font file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run classification and complete 6000-file inventory checks, then exit without writing outputs.",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Disable wells, earthquakes, and model-geometry overlays.",
    )
    args = parser.parse_args()
    if args.chunk_nodes <= 0:
        parser.error("--chunk-nodes must be positive")
    if args.png_dpi < 600:
        parser.error("--png-dpi must be at least 600")
    return args


def find_liberation_sans_path(font_file: Path | None = None) -> str | None:
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
        candidates.append(
            font_manager.findfont(REQUESTED_FONT_FAMILY, fallback_to_default=False)
        )
    except Exception:
        pass
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def configure_matplotlib(font_file: Path | None = None) -> str:
    global SELECTED_FONT_FAMILY

    font_path = find_liberation_sans_path(font_file)
    if font_path is not None:
        try:
            font_manager.fontManager.addfont(font_path)
            font_prop = font_manager.FontProperties(fname=font_path)
            SELECTED_FONT_FAMILY = font_prop.get_name() or REQUESTED_FONT_FAMILY
        except Exception:
            SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
    else:
        SELECTED_FONT_FAMILY = FALLBACK_FONT_FAMILY
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                SELECTED_FONT_FAMILY,
                FALLBACK_FONT_FAMILY,
                "sans-serif",
            ],
            "font.size": FIGURE_FONT_SIZE,
            "axes.linewidth": 1.2,
            "axes.labelsize": FIGURE_FONT_SIZE,
            "axes.titlesize": FIGURE_FONT_SIZE,
            "xtick.labelsize": FIGURE_FONT_SIZE,
            "ytick.labelsize": FIGURE_FONT_SIZE,
            "legend.fontsize": FIGURE_FONT_SIZE,
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


def load_flat_yaml(path: Path) -> dict[str, object]:
    """Load the repository's flat path YAML without requiring PyYAML."""

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

    data: dict[str, object] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in line:
            raise RuntimeError(
                f"PyYAML is unavailable and config is not a flat mapping at {path}:{line_number}"
            )
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        try:
            value: object = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value
        data[key.strip()] = value
    return data


def configured_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_case_root(args: argparse.Namespace, config: dict[str, object]) -> tuple[Path, str]:
    if args.case_root is not None:
        root = args.case_root.expanduser().resolve()
        source = "command_line"
    else:
        raw_root = configured_path(config, "ogs_results_raw_dir")
        primary = raw_root / CASE_COLLECTION if raw_root is not None else None
        if primary is not None and primary.is_dir():
            root = primary
            source = "ogs_results_raw_dir"
        else:
            old_root = configured_path(config, "old_workspace_root")
            fallback = (
                old_root / "ogs" / "results_raw" / CASE_COLLECTION
                if old_root is not None
                else None
            )
            if fallback is None or not fallback.is_dir():
                raise FileNotFoundError(
                    "Could not resolve cases_pr_all_wells from ogs_results_raw_dir or "
                    "old_workspace_root."
                )
            root = fallback
            source = "old_workspace_root_fallback"
    if not root.is_dir():
        raise FileNotFoundError(f"Case root not found: {root}")
    return root, source


def resolve_timeseries_dir(config: dict[str, object]) -> Path | None:
    explicit = configured_path(config, "pressure_timeseries_dir")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    postprocessed = configured_path(config, "ogs_results_postprocessed_dir")
    if postprocessed is not None:
        candidates.append(postprocessed / "pressure_timeseries")
    old_root = configured_path(config, "old_workspace_root")
    if old_root is not None:
        candidates.append(old_root / "ogs" / "results_postprocessed" / "pressure_timeseries")
    return next((path for path in candidates if path.is_dir()), None)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_repository_root() -> Path | None:
    if CANONICAL_REPO_ROOT is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=CANONICAL_REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    repository_root = Path(result.stdout.strip()).resolve()
    if repository_root != CANONICAL_REPO_ROOT.resolve():
        return None
    return repository_root


def git_commit(ref: str = "HEAD") -> str | None:
    repository_root = git_repository_root()
    if repository_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=repository_root,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def assert_tracked_input_matches_current_head(path: Path) -> bool:
    repository_root = git_repository_root()
    if repository_root is None:
        print(
            "[WARN] Git worktree unavailable; paired input provenance is recorded "
            "by SHA-256 and numerical validation only"
        )
        return False
    try:
        relative = path.resolve().relative_to(repository_root)
    except ValueError as exc:
        raise RuntimeError(f"Required tracked input is outside the repository: {path}") from exc
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=repository_root,
            capture_output=True,
            text=True,
        )
    except OSError:
        print(
            "[WARN] Git executable unavailable; paired input provenance is recorded "
            "by SHA-256 and numerical validation only"
        )
        return False
    if tracked.returncode != 0:
        raise RuntimeError(f"Required paired CSV is not tracked by Git: {relative}")
    try:
        comparison = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
            cwd=repository_root,
        )
    except OSError:
        print(
            "[WARN] Git executable unavailable; paired input provenance is recorded "
            "by SHA-256 and numerical validation only"
        )
        return False
    if comparison.returncode == 1:
        raise RuntimeError(
            f"Required paired CSV does not match the tracked content in current HEAD: {relative}"
        )
    if comparison.returncode != 0:
        raise RuntimeError(f"Could not compare paired CSV with current HEAD: {relative}")
    return True


def classify_cases(paired_csv: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    paired_csv = paired_csv.expanduser().resolve()
    frame = paired_pressure_data.load_paired_pressure_differences(paired_csv)
    public_selected = paired_pressure_data.select_source_group(
        frame,
        "well15",
        add_case_name=True,
    )
    values_kpa = public_selected["paired_difference_kPa"].to_numpy(dtype=float)
    selected = public_selected.loc[:, ["case"]].copy()
    # Preserve the existing local diagnostic output schema and its Pa values.
    selected["paired_difference"] = values_kpa / PRESSURE_TO_KPA
    actual_cases = tuple(selected["case"])
    if actual_cases != EXPECTED_CASES:
        missing_cases = sorted(set(EXPECTED_CASES).difference(actual_cases))
        unexpected = sorted(set(actual_cases).difference(EXPECTED_CASES))
        raise RuntimeError(
            "Selected paired rows are not the complete unique case_000001..case_001000 set. "
            f"rows={len(selected)}, missing={len(missing_cases)}, unexpected={len(unexpected)}, "
            f"missing_examples={missing_cases[:5]}, unexpected_examples={unexpected[:5]}"
        )
    values = values_kpa
    positive = values > 0.0
    negative = values < 0.0
    zero = values == 0.0
    nearest = selected.iloc[np.argsort(np.abs(values), kind="stable")[:5]]
    nearest_records = [
        {
            "case": str(public_selected.iloc[index]["case"]),
            "paired_difference_kPa": float(values_kpa[index]),
        }
        for index in nearest.index
    ]
    counts = (int(np.count_nonzero(positive)), int(np.count_nonzero(negative)), int(np.count_nonzero(zero)))
    expected = (EXPECTED_POSITIVE, EXPECTED_NEGATIVE, EXPECTED_ZERO)
    if counts != expected:
        raise RuntimeError(
            "Paired-sign counts do not match the required classification; plotting is stopped. "
            f"positive={counts[0]}, negative={counts[1]}, zero={counts[2]}, expected={expected}; "
            f"five_smallest_absolute_paired_differences={nearest_records}"
        )
    selected["sign"] = np.where(positive, "positive", np.where(negative, "negative", "zero"))
    selected["set"] = np.where(positive, "M_plus", np.where(negative, "M_minus", "zero"))
    return selected, nearest_records


def surface_dir(case_root: Path, case: str, surface: str) -> Path:
    return case_root / case / "post" / "resample" / surface


def validate_inventory(
    case_root: Path, cases: Sequence[str]
) -> tuple[dict[str, SurfaceInventory], dict[str, object]]:
    required_paths = [
        surface_dir(case_root, case, surface) / name
        for case in cases
        for surface in SURFACES
        for name in REQUIRED_ARRAY_NAMES
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(required_paths)} required NPY files. "
            f"Representative paths:\n{examples}"
        )

    problems = ValidationProblems()
    references: dict[str, dict[str, Any]] = {}
    for case in cases:
        for surface in SURFACES:
            directory = surface_dir(case_root, case, surface)
            pressure_path = directory / "pressure.npy"
            points_path = directory / "points.npy"
            time_path = directory / "time.npy"
            try:
                pressure = np.load(pressure_path, mmap_mode="r", allow_pickle=False)
                points = np.load(points_path, mmap_mode="r", allow_pickle=False)
                time = np.load(time_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                problems.add("npy_load_error", f"{directory}: {exc}")
                continue

            if pressure.ndim != 2:
                problems.add("pressure_not_2d", f"{pressure_path}: shape={pressure.shape}")
            if points.ndim != 2 or points.shape[1:] != (3,):
                problems.add("points_not_N_by_3", f"{points_path}: shape={points.shape}")
            elif not np.all(np.isfinite(points)):
                problems.add("points_nonfinite", str(points_path))
            if time.ndim != 1:
                problems.add("time_not_1d", f"{time_path}: shape={time.shape}")
            if pressure.ndim == 2 and time.ndim == 1 and pressure.shape[0] != time.size:
                problems.add(
                    "pressure_time_length_mismatch",
                    f"{directory}: pressure={pressure.shape}, time={time.shape}",
                )
            if (
                pressure.ndim == 2
                and points.ndim == 2
                and points.shape[1:] == (3,)
                and pressure.shape[1] != points.shape[0]
            ):
                problems.add(
                    "pressure_node_length_mismatch",
                    f"{directory}: pressure={pressure.shape}, points={points.shape}",
                )
            if time.ndim == 1 and not np.all(np.isfinite(time)):
                problems.add("time_nonfinite", str(time_path))

            display_indices = (
                np.flatnonzero(np.asarray(time) == DISPLAY_TIME) if time.ndim == 1 else np.array([], dtype=int)
            )
            if display_indices.size != 1:
                problems.add(
                    "display_time_not_unique",
                    f"{time_path}: occurrences_of_{DISPLAY_TIME:g}={display_indices.size}",
                )
            elif pressure.ndim == 2 and display_indices[0] < pressure.shape[0]:
                snapshot = np.asarray(pressure[int(display_indices[0]), :])
                if not np.all(np.isfinite(snapshot)):
                    problems.add("display_pressure_nonfinite", str(pressure_path))

            if surface not in references and pressure.ndim == 2 and points.ndim == 2 and time.ndim == 1:
                references[surface] = {
                    "pressure_shape": tuple(int(v) for v in pressure.shape),
                    "pressure_dtype": str(pressure.dtype),
                    "points": np.array(points, copy=True),
                    "time": np.array(time, copy=True),
                    "display_time_index": int(display_indices[0]) if display_indices.size == 1 else -1,
                    "case": case,
                }
                continue

            reference = references.get(surface)
            if reference is None:
                continue
            if pressure.ndim == 2 and tuple(pressure.shape) != reference["pressure_shape"]:
                problems.add(
                    "pressure_shape_mismatch",
                    f"{pressure_path}: {pressure.shape} != {reference['pressure_shape']}",
                )
            if pressure.ndim == 2 and str(pressure.dtype) != reference["pressure_dtype"]:
                problems.add(
                    "pressure_dtype_mismatch",
                    f"{pressure_path}: {pressure.dtype} != {reference['pressure_dtype']}",
                )
            if points.ndim == 2 and not np.array_equal(np.asarray(points), reference["points"]):
                problems.add(
                    "points_or_node_order_mismatch",
                    f"{points_path} differs from {reference['case']}",
                )
            if time.ndim == 1 and not np.array_equal(np.asarray(time), reference["time"]):
                problems.add(
                    "time_vector_mismatch",
                    f"{time_path} differs from {reference['case']}",
                )

    if problems.any():
        raise RuntimeError(problems.format())
    if set(references) != set(SURFACES):
        raise RuntimeError(f"Could not establish surface references: {sorted(references)}")

    inventory: dict[str, SurfaceInventory] = {}
    surface_metadata: dict[str, object] = {}
    for surface in SURFACES:
        reference = references[surface]
        points = reference["points"]
        time = reference["time"]
        item = SurfaceInventory(
            surface=surface,
            pressure_shape=reference["pressure_shape"],
            pressure_dtype=reference["pressure_dtype"],
            node_count=int(points.shape[0]),
            time_count=int(time.size),
            display_time_index=int(reference["display_time_index"]),
            points=points,
            time=time,
        )
        inventory[surface] = item
        surface_metadata[surface] = {
            "pressure_shape": list(item.pressure_shape),
            "pressure_dtype": item.pressure_dtype,
            "node_count": item.node_count,
            "time_count": item.time_count,
            "display_time_index": item.display_time_index,
            "points_sha256": hashlib.sha256(points.tobytes(order="C")).hexdigest(),
            "time_sha256": hashlib.sha256(time.tobytes(order="C")).hexdigest(),
            "points_identical_all_cases": True,
            "node_order_identical_all_cases": True,
            "time_identical_all_cases": True,
            "pressure_dtype_identical_all_cases": True,
            "pressure_node_axis_matches_points_all_cases": True,
            "display_pressure_finite_all_cases": True,
        }
    metadata = {
        "expected_case_count": len(cases),
        "expected_npy_file_count": len(required_paths),
        "missing_npy_file_count": 0,
        "surface_checks": surface_metadata,
    }
    return inventory, metadata


def required_temporary_bytes(
    inventory: dict[str, SurfaceInventory], positive_count: int, negative_count: int
) -> int:
    return int(
        sum(
            (positive_count + negative_count) * item.node_count * np.dtype(np.float32).itemsize
            for item in inventory.values()
        )
    )


def compute_conditional_medians(
    case_root: Path,
    classification: pd.DataFrame,
    inventory: dict[str, SurfaceInventory],
    temporary_parent: Path,
    chunk_nodes: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    positive_cases = classification.loc[classification["sign"] == "positive", "case"].tolist()
    negative_cases = classification.loc[classification["sign"] == "negative", "case"].tolist()
    required_bytes = required_temporary_bytes(
        inventory, len(positive_cases), len(negative_cases)
    )
    temporary_parent = temporary_parent.expanduser().resolve()
    if not temporary_parent.is_dir():
        raise FileNotFoundError(f"Temporary parent directory does not exist: {temporary_parent}")
    available_bytes = int(shutil.disk_usage(temporary_parent).free)
    safety_bytes = int(math.ceil(required_bytes * 1.05)) + 1024 * 1024
    if available_bytes < safety_bytes:
        raise RuntimeError(
            "Insufficient temporary storage; median calculation is stopped. "
            f"required_data_bytes={required_bytes}, required_with_margin={safety_bytes}, "
            f"available_bytes={available_bytes}, parent={temporary_parent}"
        )

    medians: dict[str, np.ndarray] = {}
    temp_metadata: dict[str, object] = {
        "parent": str(temporary_parent),
        "required_data_bytes": required_bytes,
        "required_with_margin_bytes": safety_bytes,
        "available_bytes_before_run": available_bytes,
        "storage_dtype": "float32",
        "chunk_nodes": chunk_nodes,
        "temporary_directory_removed_after_use": True,
    }

    with tempfile.TemporaryDirectory(
        prefix="well15_paired_sign_", dir=temporary_parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        for surface in SURFACES:
            item = inventory[surface]
            group_maps: dict[str, np.memmap] = {}
            group_cases = {"positive": positive_cases, "negative": negative_cases}
            try:
                for sign, cases in group_cases.items():
                    path = temp_dir / f"{surface}_{sign}_snapshots.float32"
                    mmap = np.memmap(
                        path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(cases), item.node_count),
                    )
                    for row_index, case in enumerate(cases):
                        pressure_path = surface_dir(case_root, case, surface) / "pressure.npy"
                        pressure = np.load(pressure_path, mmap_mode="r", allow_pickle=False)
                        snapshot = np.asarray(
                            pressure[item.display_time_index, :], dtype=np.float32
                        )
                        if not np.all(np.isfinite(snapshot)):
                            raise RuntimeError(
                                f"Non-finite pressure found during median pass: {pressure_path}"
                            )
                        mmap[row_index, :] = snapshot
                    mmap.flush()
                    group_maps[sign] = mmap

                for sign, mmap in group_maps.items():
                    output = np.empty(item.node_count, dtype=np.float64)
                    for start in range(0, item.node_count, chunk_nodes):
                        stop = min(start + chunk_nodes, item.node_count)
                        block = np.asarray(mmap[:, start:stop], dtype=np.float64)
                        output[start:stop] = np.median(block, axis=0)
                    medians[FIELD_KEYS[(surface, sign)]] = output
            finally:
                for mmap in group_maps.values():
                    mmap.flush()
                group_maps.clear()
    return medians, temp_metadata


def fixed_node_median_checks(
    case_root: Path,
    classification: pd.DataFrame,
    inventory: dict[str, SurfaceInventory],
    saved_npz_path: Path,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    with np.load(saved_npz_path, allow_pickle=False) as saved_arrays:
        for surface in SURFACES:
            item = inventory[surface]
            nodes = sorted({0, item.node_count // 4, item.node_count // 2, 3 * item.node_count // 4, item.node_count - 1})
            surface_checks: dict[str, object] = {}
            for sign in ("positive", "negative"):
                cases = classification.loc[classification["sign"] == sign, "case"].tolist()
                direct = np.empty((len(cases), len(nodes)), dtype=np.float64)
                for row_index, case in enumerate(cases):
                    pressure = np.load(
                        surface_dir(case_root, case, surface) / "pressure.npy",
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    direct[row_index, :] = pressure[item.display_time_index, nodes]
                recomputed = np.median(direct, axis=0)
                saved = saved_arrays[FIELD_KEYS[(surface, sign)]][nodes]
                difference = np.abs(recomputed - saved)
                exact = bool(np.array_equal(recomputed, saved))
                if not exact:
                    raise RuntimeError(
                        f"Fixed-node direct median check failed for {surface}/{sign}; "
                        f"max_abs_difference_Pa={float(np.max(difference))}"
                    )
                surface_checks[sign] = {
                    "node_indices": nodes,
                    "saved_npz_reloaded": True,
                    "direct_recalculation_exact": exact,
                    "max_abs_difference_Pa": float(np.max(difference)),
                }
            checks[surface] = surface_checks
    return checks


def rotate_clockwise(
    x: np.ndarray, y: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    return (
        x * np.cos(theta) + y * np.sin(theta),
        -x * np.sin(theta) + y * np.cos(theta),
    )


def fault_projection(points: np.ndarray) -> FaultProjection:
    xy_center = np.mean(points[:, :2], axis=0)
    xy = points[:, :2] - xy_center
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    along_axis = vh[0]
    return FaultProjection(
        x_km=(xy @ along_axis) / 1000.0,
        elevation_km=points[:, 2] / 1000.0,
        xy_center_m=xy_center,
        along_axis=along_axis,
    )


def masked_triangulation(
    x_km: np.ndarray, y_km: np.ndarray, max_edge_km: float
) -> mtri.Triangulation:
    tri = mtri.Triangulation(x_km, y_km)
    triangles = tri.triangles
    xtri = x_km[triangles]
    ytri = y_km[triangles]
    edges = np.stack(
        [
            np.hypot(xtri[:, 0] - xtri[:, 1], ytri[:, 0] - ytri[:, 1]),
            np.hypot(xtri[:, 1] - xtri[:, 2], ytri[:, 1] - ytri[:, 2]),
            np.hypot(xtri[:, 2] - xtri[:, 0], ytri[:, 2] - ytri[:, 0]),
        ],
        axis=1,
    )
    tri.set_mask(np.max(edges, axis=1) > max_edge_km)
    return tri


def build_display_geometry(
    inventory: dict[str, SurfaceInventory]
) -> dict[str, DisplayGeometry]:
    output: dict[str, DisplayGeometry] = {}
    for surface in SURFACES:
        points = inventory[surface].points
        if surface == "leduc_top":
            x_centered = (points[:, 0] - np.mean(points[:, 0])) / 1000.0
            y_centered = (points[:, 1] - np.mean(points[:, 1])) / 1000.0
            x_km, y_km = rotate_clockwise(
                x_centered, y_centered, LEDUC_CLOCKWISE_ROTATION_DEG
            )
            max_edge = LEDUC_MAX_EDGE_KM
            xlabel, ylabel = "Relative x (km)", "Relative y (km)"
        else:
            projection = fault_projection(points)
            x_km, y_km = projection.x_km, projection.elevation_km
            max_edge = FAULT_MAX_EDGE_KM
            xlabel, ylabel = "Along fault (km)", "Depth below sea level (km)"
        tri = masked_triangulation(x_km, y_km, max_edge)
        tiny = 1.0e-6
        output[surface] = DisplayGeometry(
            surface=surface,
            x_km=x_km,
            y_km=y_km,
            tri=tri,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=(float(np.min(x_km)) - tiny, float(np.max(x_km)) + tiny),
            ylim=(float(np.min(y_km)) - tiny, float(np.max(y_km)) + tiny),
            max_edge_km=max_edge,
        )
    return output


def latlon_to_epsg3400(
    longitude_deg: np.ndarray, latitude_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    f = 1.0 / EPSG3400_INV_F
    e2 = 2.0 * f - f * f
    e4, e6 = e2 * e2, e2 * e2 * e2
    ep2 = e2 / (1.0 - e2)
    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    lon0 = np.deg2rad(EPSG3400_LON0_DEG)
    sin_lat, cos_lat, tan_lat = np.sin(lat), np.cos(lat), np.tan(lat)
    n = EPSG3400_A / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t, c = tan_lat * tan_lat, ep2 * cos_lat * cos_lat
    a = (lon - lon0) * cos_lat
    arc = EPSG3400_A * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * np.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * np.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * np.sin(6.0 * lat)
    )
    easting = EPSG3400_FALSE_EASTING + EPSG3400_K0 * n * (
        a
        + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * ep2) * a**5 / 120.0
    )
    northing = EPSG3400_FALSE_NORTHING + EPSG3400_K0 * (
        arc
        + n
        * tan_lat
        * (
            a * a / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * a**4 / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * ep2) * a**6 / 720.0
        )
    )
    return easting, northing


def epsg3400_to_mesh_local(
    easting_m: np.ndarray, northing_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(DEFAULT_MESH_STRIKE_DEG)
    dx = easting_m - DEFAULT_MESH_X_CORNER_M
    dy = northing_m - DEFAULT_MESH_Y_CORNER_M
    return (
        dx * np.cos(theta) - dy * np.sin(theta),
        dx * np.sin(theta) + dy * np.cos(theta),
    )


def active_triangle_indices(tri: mtri.Triangulation) -> np.ndarray:
    indices = np.arange(tri.triangles.shape[0], dtype=int)
    return indices if tri.mask is None else indices[~np.asarray(tri.mask, dtype=bool)]


def points_inside_triangulation(
    tri: mtri.Triangulation, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    for index in active_triangle_indices(tri):
        vertices = tri.triangles[index]
        x1, x2, x3 = tri.x[vertices]
        y1, y2, y3 = tri.y[vertices]
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(float(denominator)) < 1.0e-12:
            continue
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1.0 - a - b
        inside |= (a >= -1.0e-10) & (b >= -1.0e-10) & (c >= -1.0e-10)
    return inside


def load_well_overlay(
    config: dict[str, object],
    leduc_points: np.ndarray,
    fault_points: np.ndarray,
    fault_geometry: DisplayGeometry,
) -> WellOverlay:
    old_root = configured_path(config, "old_workspace_root")
    if old_root is None:
        raise FileNotFoundError("old_workspace_root is not configured")
    input_root = old_root / "ogs" / "zenodo_staging" / "data" / "ogs_input"
    reference_csv = input_root / "well_reference.csv"
    mesh_dir = input_root / "mesh" / "vtu"
    if not reference_csv.is_file() or not mesh_dir.is_dir():
        raise FileNotFoundError(
            f"Well reference inputs not found: {reference_csv}, {mesh_dir}"
        )
    try:
        import meshio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("meshio is unavailable; run in the repository OGS environment") from exc
    reference = pd.read_csv(reference_csv).sort_values("well_number")
    if tuple(reference["well_number"].astype(int)) != tuple(range(1, 16)):
        raise RuntimeError("well_reference.csv does not contain unique Well 1..15 rows")
    coordinates = []
    labels = []
    for row in reference.itertuples(index=False):
        path = mesh_dir / str(row.mesh_vtu_basename)
        if not path.is_file():
            raise FileNotFoundError(f"Well mesh not found: {path}")
        mesh = meshio.read(path)
        points = np.asarray(mesh.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise RuntimeError(f"Invalid well mesh points: {path}, shape={points.shape}")
        coordinates.append(np.mean(points, axis=0))
        labels.append(str(row.manuscript_label))
    xyz = np.asarray(coordinates, dtype=float)
    centered_x = (xyz[:, 0] - np.mean(leduc_points[:, 0])) / 1000.0
    centered_y = (xyz[:, 1] - np.mean(leduc_points[:, 1])) / 1000.0
    x_km, y_km = rotate_clockwise(
        centered_x, centered_y, LEDUC_CLOCKWISE_ROTATION_DEG
    )
    projection = fault_projection(fault_points)
    fault_xy = xyz[:, :2] - projection.xy_center_m
    fault_x_km = (fault_xy @ projection.along_axis) / 1000.0
    fault_y_km = xyz[:, 2] / 1000.0
    cross_axis = np.array(
        [-projection.along_axis[1], projection.along_axis[0]], dtype=float
    )
    fault_cross_axis_offset_km = (fault_xy @ cross_axis) / 1000.0
    fault_inside = points_inside_triangulation(
        fault_geometry.tri, fault_x_km, fault_y_km
    )
    return WellOverlay(
        tuple(labels),
        x_km,
        y_km,
        xyz,
        fault_x_km,
        fault_y_km,
        fault_cross_axis_offset_km,
        fault_inside,
        reference_csv,
        mesh_dir,
    )


def load_earthquake_overlay(
    config: dict[str, object],
    inventory: dict[str, SurfaceInventory],
    geometry: dict[str, DisplayGeometry],
) -> EarthquakeOverlay:
    catalog_dir = configured_path(config, "processed_seismic_catalog_dir")
    if catalog_dir is None:
        raise FileNotFoundError("processed_seismic_catalog_dir is not configured")
    path = catalog_dir / "vasyura_bathke2023_relocated_southern_cluster.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Relocated southern-cluster catalog not found: {path}")
    catalog = pd.read_csv(path)
    required = {"Latitude", "Longitude", "Depth"}
    if not required.issubset(catalog.columns):
        raise RuntimeError(f"Relocated catalog lacks columns: {sorted(required - set(catalog.columns))}")
    latitude = pd.to_numeric(catalog["Latitude"], errors="raise").to_numpy(dtype=float)
    longitude = pd.to_numeric(catalog["Longitude"], errors="raise").to_numpy(dtype=float)
    elevation_m = pd.to_numeric(catalog["Depth"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(np.column_stack([latitude, longitude, elevation_m]))):
        raise RuntimeError("Relocated catalog contains non-finite coordinates")
    easting, northing = latlon_to_epsg3400(longitude, latitude)
    local_x, local_y = epsg3400_to_mesh_local(easting, northing)

    leduc_points = inventory["leduc_top"].points
    centered_x = (local_x - np.mean(leduc_points[:, 0])) / 1000.0
    centered_y = (local_y - np.mean(leduc_points[:, 1])) / 1000.0
    leduc_x, leduc_y = rotate_clockwise(
        centered_x, centered_y, LEDUC_CLOCKWISE_ROTATION_DEG
    )

    fault_points = inventory["fault"].points
    projection = fault_projection(fault_points)
    earthquake_xy = np.column_stack([local_x, local_y]) - projection.xy_center_m
    fault_x = (earthquake_xy @ projection.along_axis) / 1000.0
    fault_y = elevation_m / 1000.0
    inside = points_inside_triangulation(geometry["fault"].tri, fault_x, fault_y)
    return EarthquakeOverlay(leduc_x, leduc_y, fault_x, fault_y, inside, path)


def load_fault_s_guide_overlay(
    config: dict[str, object],
    inventory: dict[str, SurfaceInventory],
) -> tuple[FaultGuideOverlay, dict[str, object]]:
    """Build an approximate map-view Fault S guide from the model geometry."""

    old_root = configured_path(config, "old_workspace_root")
    if old_root is None:
        raise FileNotFoundError("old_workspace_root is not configured")
    domain_path = (
        old_root
        / "ogs"
        / "zenodo_staging"
        / "data"
        / "ogs_input"
        / "mesh"
        / "vtu"
        / "mesh_rescaled_domain.vtu"
    )
    if not domain_path.is_file():
        raise FileNotFoundError(f"Domain mesh not found: {domain_path}")
    try:
        import meshio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "meshio is unavailable; run in the repository OGS environment"
        ) from exc

    domain = meshio.read(domain_path)
    if "tetra" not in domain.cells_dict:
        raise RuntimeError(f"Domain mesh has no tetra cells: {domain_path}")
    material_data = domain.cell_data_dict.get("MaterialIDs", {})
    if "tetra" not in material_data:
        raise RuntimeError(f"Domain mesh has no tetra MaterialIDs: {domain_path}")
    tetra = np.asarray(domain.cells_dict["tetra"], dtype=int)
    material_ids = np.asarray(material_data["tetra"], dtype=int)
    f1d_cells = tetra[material_ids == FAULT_S_DOMAIN_MATERIAL_ID]
    if f1d_cells.size == 0:
        raise RuntimeError(
            f"No tetra cells have Fault S F1_D MaterialID "
            f"{FAULT_S_DOMAIN_MATERIAL_ID}: {domain_path}"
        )
    f1d_node_ids = np.unique(f1d_cells.ravel())
    f1d_points = np.asarray(domain.points[f1d_node_ids], dtype=float)

    leduc_points = inventory["leduc_top"].points
    fault_points = inventory["fault"].points
    fault_center = np.mean(fault_points, axis=0)
    _, _, vh = np.linalg.svd(fault_points - fault_center, full_matrices=False)
    plane_normal = vh[-1]
    plane_residual_m = np.abs((fault_points - fault_center) @ plane_normal)
    if float(np.max(plane_residual_m)) > 1.0e-5:
        raise RuntimeError(
            "Fault S centre-plane points are not planar enough to define the guide"
        )
    strike_axis = np.cross(plane_normal, np.array([0.0, 0.0, 1.0]))
    strike_norm = float(np.linalg.norm(strike_axis))
    if strike_norm <= 0.0:
        raise RuntimeError("Fault S centre plane has no horizontal strike direction")
    strike_axis /= strike_norm
    if strike_axis[0] < 0.0:
        strike_axis *= -1.0

    fault_along_m = (fault_points - fault_center) @ strike_axis
    f1d_along_m = (f1d_points - fault_center) @ strike_axis
    f1d_min_m = float(np.min(f1d_along_m))
    f1d_max_m = float(np.max(f1d_along_m))
    f1d_length_km = (f1d_max_m - f1d_min_m) / 1000.0
    if not math.isclose(
        f1d_length_km,
        EXPECTED_FAULT_S_ALONG_STRIKE_LENGTH_KM,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError(
            f"Unexpected Fault S F1_D along-strike length: {f1d_length_km:g} km"
        )
    leduc_relative = leduc_points - fault_center
    leduc_along_m = leduc_relative @ strike_axis
    signed_distance_m = leduc_relative @ plane_normal
    within_fault_plane_length = (
        (leduc_along_m >= float(np.min(fault_along_m)) - 1.0e-6)
        & (leduc_along_m <= float(np.max(fault_along_m)) + 1.0e-6)
    )
    negative_candidates = within_fault_plane_length & (
        signed_distance_m < -FAULT_S_OPENING_CENTER_EXCLUSION_M
    )
    positive_candidates = within_fault_plane_length & (
        signed_distance_m > FAULT_S_OPENING_CENTER_EXCLUSION_M
    )
    if not np.any(negative_candidates) or not np.any(positive_candidates):
        raise RuntimeError("No Leduc/Fault S opening-wall candidates were found")
    negative_offset_m = float(
        np.min(np.abs(signed_distance_m[negative_candidates]))
    )
    positive_offset_m = float(
        np.min(np.abs(signed_distance_m[positive_candidates]))
    )
    if not math.isclose(
        negative_offset_m,
        positive_offset_m,
        rel_tol=0.0,
        abs_tol=FAULT_S_OPENING_WALL_TOLERANCE_M,
    ):
        raise RuntimeError(
            "The two Leduc/Fault S opening walls are not symmetric about the "
            f"centre plane: {negative_offset_m:g} and {positive_offset_m:g} m"
        )
    opening_offset_m = 0.5 * (negative_offset_m + positive_offset_m)
    if not math.isclose(
        opening_offset_m,
        EXPECTED_FAULT_S_OPENING_WALL_OFFSET_M,
        rel_tol=0.0,
        abs_tol=FAULT_S_OPENING_WALL_TOLERANCE_M,
    ):
        raise RuntimeError(
            f"Unexpected Leduc/Fault S opening-wall offset: {opening_offset_m:g} m"
        )
    negative_side = leduc_points[
        within_fault_plane_length
        & np.isclose(
            signed_distance_m,
            -negative_offset_m,
            rtol=0.0,
            atol=FAULT_S_OPENING_WALL_TOLERANCE_M,
        )
    ]
    positive_side = leduc_points[
        within_fault_plane_length
        & np.isclose(
            signed_distance_m,
            positive_offset_m,
            rtol=0.0,
            atol=FAULT_S_OPENING_WALL_TOLERANCE_M,
        )
    ]
    negative_side = negative_side[
        np.argsort((negative_side - fault_center) @ strike_axis)
    ]
    positive_side = positive_side[
        np.argsort((positive_side - fault_center) @ strike_axis)
    ]
    if (
        negative_side.shape != positive_side.shape
        or negative_side.shape[0] != EXPECTED_FAULT_S_OPENING_WALL_PAIR_COUNT
    ):
        raise RuntimeError(
            "Cannot pair the two Leduc/Fault S opening walls: "
            f"negative={negative_side.shape}, positive={positive_side.shape}"
        )

    opening_midpoints = 0.5 * (negative_side + positive_side)
    midpoint_plane_residual_m = np.abs(
        (opening_midpoints - fault_center) @ plane_normal
    )
    if (
        float(np.max(midpoint_plane_residual_m))
        > FAULT_S_OPENING_WALL_TOLERANCE_M
    ):
        raise RuntimeError("Leduc opening-wall midpoints do not lie on the centre plane")
    midpoint_along_m = (opening_midpoints - fault_center) @ strike_axis
    if (
        float(midpoint_along_m[0]) < f1d_min_m - 1.0e-6
        or float(midpoint_along_m[-1]) > f1d_max_m + 1.0e-6
    ):
        raise RuntimeError("Leduc/Fault S opening interval lies outside F1_D")
    full_start = opening_midpoints[0] + (
        f1d_min_m - float(midpoint_along_m[0])
    ) * strike_axis
    full_end = opening_midpoints[-1] + (
        f1d_max_m - float(midpoint_along_m[-1])
    ) * strike_axis
    guide_xyz = np.vstack([full_start, opening_midpoints, full_end])
    if guide_xyz.shape[0] != opening_midpoints.shape[0] + 2:
        raise RuntimeError("Approximate Fault S guide has an invalid vertex count")

    leduc_center_xy = np.mean(leduc_points[:, :2], axis=0)
    guide_x_km, guide_y_km = rotate_clockwise(
        (guide_xyz[:, 0] - leduc_center_xy[0]) / 1000.0,
        (guide_xyz[:, 1] - leduc_center_xy[1]) / 1000.0,
        LEDUC_CLOCKWISE_ROTATION_DEG,
    )
    opening_x_km, opening_y_km = rotate_clockwise(
        (opening_midpoints[:, 0] - leduc_center_xy[0]) / 1000.0,
        (opening_midpoints[:, 1] - leduc_center_xy[1]) / 1000.0,
        LEDUC_CLOCKWISE_ROTATION_DEG,
    )
    if not np.all(np.isfinite(np.column_stack([guide_x_km, guide_y_km]))):
        raise RuntimeError("Approximate Fault S guide contains non-finite coordinates")

    metadata: dict[str, object] = {
        "included": True,
        "label": "Fault S",
        "domain_mesh": str(domain_path),
        "domain_mesh_sha256": sha256_file(domain_path),
        "f1d_material_id": FAULT_S_DOMAIN_MATERIAL_ID,
        "f1d_tetra_count": int(f1d_cells.shape[0]),
        "f1d_unique_node_count": int(f1d_node_ids.size),
        "modelled_along_strike_length_km": f1d_length_km,
        "opening_wall_pair_count": int(opening_midpoints.shape[0]),
        "polyline_vertex_count": int(guide_xyz.shape[0]),
        "opening_wall_absolute_plane_offset_m": opening_offset_m,
        "opening_along_strike_length_km": (
            float(midpoint_along_m[-1]) - float(midpoint_along_m[0])
        )
        / 1000.0,
        "extension_beyond_leduc_opening_km": (
            f1d_max_m - float(midpoint_along_m[-1])
        )
        / 1000.0,
        "extension_before_leduc_opening_km": (
            float(midpoint_along_m[0]) - f1d_min_m
        )
        / 1000.0,
        "display_start_end_xy_km": [
            [float(guide_x_km[0]), float(guide_y_km[0])],
            [float(guide_x_km[-1]), float(guide_y_km[-1])],
        ],
        "leduc_opening_start_end_xy_km": [
            [float(opening_x_km[0]), float(opening_y_km[0])],
            [float(opening_x_km[-1]), float(opening_y_km[-1])],
        ],
        "method": (
            "Midpoints of the two modelled Leduc/FDZ opening walls, extended "
            "along the horizontal strike of the Fault S centre plane to the "
            "along-strike limits of the F1_D model region."
        ),
        "interpretation": (
            "Approximate map-view spatial guide; not a mapped surface trace or "
            "an exact Leduc-top/Fault S intersection."
        ),
        "style": {
            "colour": "0.10 (dark grey)",
            "linestyle": "dashed",
            "dash_pattern": [5, 3],
            "linewidth_pt": 1.15,
            "white_halo_linewidth_pt": 2.6,
            "white_halo_alpha": 0.88,
            "zorder": 3.5,
            "panels": ["a", "b"],
        },
    }
    return FaultGuideOverlay(guide_x_km, guide_y_km), metadata


def boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """Return undirected edges that occur once in a triangular patch."""

    edges = np.sort(
        np.vstack(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def order_polyline_edges(
    edges: np.ndarray,
    points: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    """Order a single non-branching edge chain and validate its topology."""

    adjacency: dict[int, list[int]] = {}
    for first, second in np.asarray(edges, dtype=int):
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency:
        raise RuntimeError("Cannot order an empty polyline")
    degrees = {node: len(neighbours) for node, neighbours in adjacency.items()}
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    if closed:
        if endpoints or any(degree != 2 for degree in degrees.values()):
            raise RuntimeError("Expected one closed, non-branching boundary")
        start = min(
            adjacency,
            key=lambda node: tuple(float(value) for value in points[node]),
        )
    else:
        if len(endpoints) != 2 or any(
            degree not in (1, 2) for degree in degrees.values()
        ):
            raise RuntimeError("Expected one open, non-branching boundary")
        start = min(
            endpoints,
            key=lambda node: tuple(float(value) for value in points[node]),
        )

    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = [node for node in adjacency[current] if node != previous]
        if previous is None:
            next_node = min(
                candidates,
                key=lambda node: tuple(float(value) for value in points[node]),
            )
        else:
            if closed and start in candidates and len(ordered) < len(adjacency):
                candidates.remove(start)
            if not candidates:
                break
            next_node = candidates[0]
        if closed and next_node == start:
            ordered.append(start)
            break
        ordered.append(next_node)
        previous, current = current, next_node
        if not closed and len(adjacency[current]) == 1:
            break
        if len(ordered) > len(adjacency) + 1:
            raise RuntimeError("Polyline traversal did not terminate")

    expected_node_count = len(adjacency) + (1 if closed else 0)
    if len(ordered) != expected_node_count:
        raise RuntimeError(
            f"Polyline traversal returned {len(ordered)} nodes; "
            f"expected {expected_node_count}"
        )
    traversed_edges = {
        tuple(sorted((first, second)))
        for first, second in zip(ordered[:-1], ordered[1:])
    }
    expected_edges = {
        tuple(sorted((int(first), int(second)))) for first, second in edges
    }
    if traversed_edges != expected_edges:
        raise RuntimeError("Polyline traversal did not use every boundary edge once")
    return np.asarray(ordered, dtype=int)


def vertical_polyline_intersections(
    x_km: np.ndarray,
    y_km: np.ndarray,
    target_x_km: float,
) -> np.ndarray:
    """Return y coordinates where a polyline crosses a vertical display line."""

    intersections: list[float] = []
    for x0, x1, y0, y1 in zip(x_km[:-1], x_km[1:], y_km[:-1], y_km[1:]):
        lower, upper = sorted((float(x0), float(x1)))
        if target_x_km < lower or target_x_km > upper:
            continue
        if math.isclose(float(x0), float(x1), abs_tol=1.0e-12):
            continue
        fraction = (target_x_km - float(x0)) / (float(x1) - float(x0))
        intersections.append(float(y0) + fraction * (float(y1) - float(y0)))
    if not intersections:
        return np.asarray([], dtype=float)
    return np.unique(np.round(np.asarray(intersections, dtype=float), decimals=12))


def load_fault_stratigraphy_overlay(
    config: dict[str, object],
    fault_points: np.ndarray,
) -> tuple[FaultStratigraphyOverlay, dict[str, object]]:
    """Extract modelled stratigraphic boundaries on the Fault S FGZ midplane."""

    old_root = configured_path(config, "old_workspace_root")
    if old_root is None:
        raise FileNotFoundError("old_workspace_root is not configured")
    mesh_path = old_root / "ogs" / "input" / "mesh" / "msh" / "mesh_rescaled.msh"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Retained Gmsh mesh not found: {mesh_path}")
    try:
        import meshio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "meshio is unavailable; run in the repository OGS environment"
        ) from exc

    mesh = meshio.read(mesh_path)
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    geometrical_blocks = mesh.cell_data.get("gmsh:geometrical")
    if physical_blocks is None or geometrical_blocks is None:
        raise RuntimeError(f"Gmsh physical/geometrical tags are missing: {mesh_path}")
    if not (
        len(mesh.cells) == len(physical_blocks) == len(geometrical_blocks)
    ):
        raise RuntimeError(f"Gmsh cell-data blocks are not aligned: {mesh_path}")
    centre_group = np.asarray(
        mesh.field_data.get("F1_gouge_center_plane", []), dtype=int
    )
    if not np.array_equal(
        centre_group,
        np.asarray([FAULT_S_CENTER_PLANE_PHYSICAL_TAG, 2], dtype=int),
    ):
        raise RuntimeError(
            "F1_gouge_center_plane does not have the expected physical tag and "
            f"dimension: {centre_group.tolist()}"
        )
    triangle_blocks: list[np.ndarray] = []
    triangle_physical: list[np.ndarray] = []
    triangle_geometrical: list[np.ndarray] = []
    for block_index, cell_block in enumerate(mesh.cells):
        if not (
            len(cell_block.data) == len(physical_blocks[block_index])
            == len(geometrical_blocks[block_index])
        ):
            raise RuntimeError(
                f"Gmsh cell-data length mismatch in block {block_index}"
            )
        if cell_block.type != "triangle":
            continue
        triangle_blocks.append(np.asarray(cell_block.data, dtype=int))
        triangle_physical.append(
            np.asarray(physical_blocks[block_index], dtype=int)
        )
        triangle_geometrical.append(
            np.asarray(geometrical_blocks[block_index], dtype=int)
        )
    if not triangle_blocks:
        raise RuntimeError(f"Gmsh mesh contains no triangles: {mesh_path}")
    triangles = np.vstack(triangle_blocks)
    physical = np.concatenate(triangle_physical)
    geometrical = np.concatenate(triangle_geometrical)
    center_mask = physical == FAULT_S_CENTER_PLANE_PHYSICAL_TAG
    center_triangles = triangles[center_mask]
    center_geometrical = geometrical[center_mask]
    if center_triangles.size == 0:
        raise RuntimeError(
            f"No triangles have Fault S centre-plane physical tag "
            f"{FAULT_S_CENTER_PLANE_PHYSICAL_TAG}"
        )
    center_node_ids = np.unique(center_triangles)
    center_tags = tuple(int(tag) for tag in np.unique(center_geometrical))
    if (
        center_triangles.shape != (
            EXPECTED_FAULT_S_CENTER_PLANE_TRIANGLE_COUNT,
            3,
        )
        or center_node_ids.size != EXPECTED_FAULT_S_CENTER_PLANE_NODE_COUNT
        or center_tags != EXPECTED_FAULT_S_CENTER_PLANE_GEOMETRICAL_TAGS
    ):
        raise RuntimeError(
            "Unexpected Fault S centre-plane partition: "
            f"triangles={center_triangles.shape}, nodes={center_node_ids.size}, "
            f"geometrical_tags={center_tags}"
        )

    raw_center_points = np.asarray(mesh.points[center_node_ids], dtype=float)
    raw_sorted = raw_center_points[
        np.lexsort(
            (raw_center_points[:, 2], raw_center_points[:, 1], raw_center_points[:, 0])
        )
    ]
    fault_sorted = np.asarray(fault_points, dtype=float)[
        np.lexsort((fault_points[:, 2], fault_points[:, 1], fault_points[:, 0]))
    ]
    if raw_sorted.shape != fault_sorted.shape:
        raise RuntimeError(
            "Retained Gmsh centre plane and plotted fault surface have different "
            f"node counts: {raw_sorted.shape[0]} and {fault_sorted.shape[0]}"
        )
    coordinate_mismatch_m = float(np.max(np.abs(raw_sorted - fault_sorted)))
    if coordinate_mismatch_m > 1.0e-6:
        raise RuntimeError(
            "Retained Gmsh centre plane does not match the plotted fault surface: "
            f"maximum coordinate difference={coordinate_mismatch_m:g} m"
        )

    leduc_triangles = center_triangles[
        center_geometrical == FAULT_S_LEDUC_GEOMETRICAL_TAG
    ]
    basement_triangles = center_triangles[
        center_geometrical == FAULT_S_BASEMENT_GEOMETRICAL_TAG
    ]
    if (
        leduc_triangles.shape
        != (EXPECTED_FAULT_S_LEDUC_TRIANGLE_COUNT, 3)
        or basement_triangles.shape
        != (EXPECTED_FAULT_S_BASEMENT_TRIANGLE_COUNT, 3)
    ):
        raise RuntimeError(
            "Unexpected Leduc/basement centre-plane triangle counts: "
            f"Leduc={leduc_triangles.shape}, basement={basement_triangles.shape}"
        )
    leduc_edges = boundary_edges(leduc_triangles)
    if leduc_edges.shape[0] != EXPECTED_FAULT_S_LEDUC_OUTLINE_EDGE_COUNT:
        raise RuntimeError(
            f"Unexpected Leduc outline edge count: {leduc_edges.shape[0]}"
        )
    leduc_node_ids = order_polyline_edges(
        leduc_edges,
        np.asarray(mesh.points, dtype=float),
        closed=True,
    )

    edge_geometries: dict[tuple[int, int], list[int]] = {}
    for triangle, geometrical_tag in zip(center_triangles, center_geometrical):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_geometries.setdefault(edge, []).append(int(geometrical_tag))
    basement_top_edge_list: list[tuple[int, int]] = []
    for edge, owners in edge_geometries.items():
        tags = set(owners)
        if FAULT_S_BASEMENT_GEOMETRICAL_TAG not in tags or not any(
            tag != FAULT_S_BASEMENT_GEOMETRICAL_TAG for tag in tags
        ):
            continue
        if len(owners) != 2 or len(tags) != 2:
            raise RuntimeError(
                f"Non-manifold Precambrian basement-top edge {edge}: "
                f"owners={owners}"
            )
        basement_top_edge_list.append(edge)
    basement_top_edges = np.asarray(basement_top_edge_list, dtype=int)
    if basement_top_edges.shape != (EXPECTED_FAULT_S_BASEMENT_TOP_EDGE_COUNT, 2):
        raise RuntimeError(
            f"Unexpected Precambrian basement-top edge shape: "
            f"{basement_top_edges.shape}"
        )
    basement_node_ids = order_polyline_edges(
        basement_top_edges,
        np.asarray(mesh.points, dtype=float),
        closed=False,
    )

    projection = fault_projection(fault_points)

    def display_coordinates(node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(mesh.points[node_ids], dtype=float)
        x_km = ((xyz[:, :2] - projection.xy_center_m) @ projection.along_axis) / 1000.0
        y_km = xyz[:, 2] / 1000.0
        return x_km, y_km

    leduc_x_km, leduc_y_km = display_coordinates(leduc_node_ids)
    basement_x_km, basement_y_km = display_coordinates(basement_node_ids)
    if basement_x_km[0] > basement_x_km[-1]:
        basement_x_km = basement_x_km[::-1]
        basement_y_km = basement_y_km[::-1]
    if not np.all(np.diff(basement_x_km) > 0.0):
        raise RuntimeError("Precambrian basement-top boundary is not monotonic along fault")

    leduc_extent = np.asarray(
        [
            np.min(leduc_x_km),
            np.max(leduc_x_km),
            np.min(-leduc_y_km),
            np.max(-leduc_y_km),
        ],
        dtype=float,
    )
    basement_extent = np.asarray(
        [
            np.min(basement_x_km),
            np.max(basement_x_km),
            np.min(-basement_y_km),
            np.max(-basement_y_km),
        ],
        dtype=float,
    )
    if not np.allclose(
        leduc_extent,
        np.asarray(EXPECTED_FAULT_S_LEDUC_DISPLAY_EXTENT_KM),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise RuntimeError(f"Unexpected Leduc display extent: {leduc_extent.tolist()}")
    if not np.allclose(
        basement_extent,
        np.asarray(EXPECTED_FAULT_S_BASEMENT_TOP_DISPLAY_EXTENT_KM),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise RuntimeError(
            f"Unexpected Precambrian basement-top display extent: "
            f"{basement_extent.tolist()}"
        )

    leduc_crossings_y_km = vertical_polyline_intersections(
        leduc_x_km,
        leduc_y_km,
        FAULT_STRATIGRAPHY_LABEL_X_KM,
    )
    if leduc_crossings_y_km.size != 2:
        raise RuntimeError(
            f"Expected two Leduc-outline crossings at x="
            f"{FAULT_STRATIGRAPHY_LABEL_X_KM:g} km; found "
            f"{leduc_crossings_y_km.size}"
        )
    leduc_label_y_km = float(np.mean(leduc_crossings_y_km))
    basement_top_at_label_y_km = float(
        np.interp(FAULT_STRATIGRAPHY_LABEL_X_KM, basement_x_km, basement_y_km)
    )
    basement_label_y_km = basement_top_at_label_y_km - 0.22

    overlay = FaultStratigraphyOverlay(
        leduc_x_km=leduc_x_km,
        leduc_y_km=leduc_y_km,
        basement_top_x_km=basement_x_km,
        basement_top_y_km=basement_y_km,
        leduc_label_xy_km=(
            FAULT_STRATIGRAPHY_LABEL_X_KM,
            leduc_label_y_km,
        ),
        basement_label_xy_km=(
            FAULT_STRATIGRAPHY_LABEL_X_KM,
            basement_label_y_km,
        ),
    )
    metadata: dict[str, object] = {
        "included": True,
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": sha256_file(mesh_path),
        "centre_plane_physical_tag": FAULT_S_CENTER_PLANE_PHYSICAL_TAG,
        "leduc_geometrical_tag": FAULT_S_LEDUC_GEOMETRICAL_TAG,
        "basement_geometrical_tag": FAULT_S_BASEMENT_GEOMETRICAL_TAG,
        "centre_plane_triangle_count": int(center_triangles.shape[0]),
        "centre_plane_node_count": int(center_node_ids.size),
        "centre_plane_maximum_coordinate_difference_from_plotted_surface_m": coordinate_mismatch_m,
        "leduc_outline_edge_count": int(leduc_edges.shape[0]),
        "basement_top_edge_count": int(basement_top_edges.shape[0]),
        "leduc_display_x_range_km": [
            float(np.min(leduc_x_km)),
            float(np.max(leduc_x_km)),
        ],
        "leduc_display_depth_range_km": [
            float(-np.max(leduc_y_km)),
            float(-np.min(leduc_y_km)),
        ],
        "basement_top_display_x_range_km": [
            float(np.min(basement_x_km)),
            float(np.max(basement_x_km)),
        ],
        "basement_top_display_depth_range_km": [
            float(-np.max(basement_y_km)),
            float(-np.min(basement_y_km)),
        ],
        "label_positions_xy_km": {
            "Leduc": list(overlay.leduc_label_xy_km),
            "Precambrian basement": list(overlay.basement_label_xy_km),
        },
        "method": (
            "Boundaries of the retained Gmsh geometrical partitions on physical "
            "group 110, which is the same Fault S FGZ midplane used for the "
            "plotted pressure field."
        ),
        "interpretation": (
            "Approximate modelled stratigraphic boundaries on the Fault S FGZ "
            "midplane."
        ),
        "style": {
            "leduc_outline": "solid dark grey, 0.75 pt",
            "basement_top": "dash-dotted dark grey, 0.75 pt",
            "coincident_boundary_rendering": (
                "Shared Leduc-outline/basement-top segments use the solid "
                "Leduc-outline style."
            ),
            "white_halo_linewidth_pt": 1.45,
            "white_halo_alpha": 0.78,
            "direct_labels_panel": "d",
            "direct_label_font_size_pt": 9.5,
            "legend": False,
            "zorder": [3.05, 3.1],
        },
        "panels": ["c", "d"],
    }
    return overlay, metadata


def load_optional_overlays(
    config: dict[str, object],
    inventory: dict[str, SurfaceInventory],
    geometry: dict[str, DisplayGeometry],
    disabled: bool,
) -> tuple[
    WellOverlay | None,
    EarthquakeOverlay | None,
    FaultGuideOverlay | None,
    FaultStratigraphyOverlay | None,
    dict[str, object],
    list[str],
]:
    metadata: dict[str, object] = {}
    warnings: list[str] = []
    if disabled:
        return None, None, None, None, {"disabled_by_user": True}, warnings
    well_overlay: WellOverlay | None = None
    earthquake_overlay: EarthquakeOverlay | None = None
    fault_s_overlay: FaultGuideOverlay | None = None
    fault_stratigraphy_overlay: FaultStratigraphyOverlay | None = None
    try:
        well_overlay = load_well_overlay(
            config,
            inventory["leduc_top"].points,
            inventory["fault"].points,
            geometry["fault"],
        )
        projected_source_metadata: dict[str, object] = {}
        for label in ("Well 6", "Well 15"):
            index = well_overlay.labels.index(label)
            if not bool(well_overlay.fault_inside[index]):
                raise RuntimeError(
                    f"Projected model source for {label} lies outside the active fault triangulation"
                )
            projected_source_metadata[label] = {
                "source_xyz_m": well_overlay.source_xyz_m[index].tolist(),
                "fault_display_xy_km": [
                    float(well_overlay.fault_x_km[index]),
                    float(well_overlay.fault_y_km[index]),
                ],
                "absolute_cross_axis_offset_km": float(
                    abs(well_overlay.fault_cross_axis_offset_km[index])
                ),
                "inside_active_fault_triangulation": True,
            }
        metadata["wells"] = {
            "included": True,
            "count": len(well_overlay.labels),
            "reference_csv": str(well_overlay.reference_csv),
            "mesh_dir": str(well_overlay.mesh_dir),
            "well10_display_xy_km": [
                float(well_overlay.x_km[9]),
                float(well_overlay.y_km[9]),
            ],
            "well15_display_xy_km": [
                float(well_overlay.x_km[14]),
                float(well_overlay.y_km[14]),
            ],
            "fault_projection_method": (
                "model source-node plan-view coordinates projected onto the "
                "PCA-defined central plane while retaining source-node elevation"
            ),
            "fault_projected_model_sources": projected_source_metadata,
        }
    except Exception as exc:
        warnings.append(f"Well overlay omitted: {exc}")
        metadata["wells"] = {"included": False, "reason": str(exc)}
    try:
        earthquake_overlay = load_earthquake_overlay(config, inventory, geometry)
        leduc_inside = points_inside_triangulation(
            geometry["leduc_top"].tri,
            earthquake_overlay.leduc_x_km,
            earthquake_overlay.leduc_y_km,
        )
        metadata["relocated_southern_cluster"] = {
            "included": True,
            "catalog": str(earthquake_overlay.catalog_path),
            "event_count": int(earthquake_overlay.fault_x_km.size),
            "leduc_mesh_inside_count": int(np.count_nonzero(leduc_inside)),
            "leduc_displayed_count": int(earthquake_overlay.leduc_x_km.size),
            "fault_mesh_inside_and_displayed_count": int(
                np.count_nonzero(earthquake_overlay.fault_inside)
            ),
            "fault_mesh_outside_and_omitted_count": int(
                np.count_nonzero(~earthquake_overlay.fault_inside)
            ),
        }
    except Exception as exc:
        warnings.append(f"Relocated southern-cluster overlay omitted: {exc}")
        metadata["relocated_southern_cluster"] = {"included": False, "reason": str(exc)}
    fault_s_overlay, fault_s_metadata = load_fault_s_guide_overlay(
        config, inventory
    )
    metadata["approximate_fault_s_guide"] = fault_s_metadata
    fault_stratigraphy_overlay, fault_stratigraphy_metadata = (
        load_fault_stratigraphy_overlay(config, inventory["fault"].points)
    )
    metadata["fault_stratigraphy"] = fault_stratigraphy_metadata
    return (
        well_overlay,
        earthquake_overlay,
        fault_s_overlay,
        fault_stratigraphy_overlay,
        metadata,
        warnings,
    )


def field_statistics(values_pa: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values_pa)),
        "median": float(np.median(values_pa)),
        "percentile_95": float(np.percentile(values_pa, 95.0)),
        "percentile_99_5": float(np.percentile(values_pa, 99.5)),
        "max": float(np.max(values_pa)),
    }


def calculate_plot_limits(
    medians: dict[str, np.ndarray]
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    statistics: dict[str, dict[str, float]] = {}
    for key, values_pa in medians.items():
        stats_pa = field_statistics(values_pa)
        statistics[key] = {
            "Pa": stats_pa,
            "kPa": {name: value * PRESSURE_TO_KPA for name, value in stats_pa.items()},
        }

    clipping: dict[str, object] = {}
    for key, values_pa in medians.items():
        values_kpa = values_pa * PRESSURE_TO_KPA
        nonpositive = int(np.count_nonzero(values_kpa <= 0.0))
        below = int(
            np.count_nonzero((values_kpa > 0.0) & (values_kpa < LOG_VMIN_KPA))
        )
        above = int(np.count_nonzero(values_kpa > LOG_VMAX_KPA))
        total = int(values_kpa.size)
        clipping[key] = {
            "node_count": total,
            "nonpositive_masked_count": nonpositive,
            "nonpositive_masked_fraction": nonpositive / total,
            "below_vmin_count": below,
            "below_vmin_fraction": below / total,
            "above_vmax_count": above,
            "above_vmax_fraction": above / total,
        }

    linear_ranges: dict[str, list[float]] = {}
    for surface in SURFACES:
        combined = np.concatenate(
            [
                medians[FIELD_KEYS[(surface, "positive")]] * PRESSURE_TO_KPA,
                medians[FIELD_KEYS[(surface, "negative")]] * PRESSURE_TO_KPA,
            ]
        )
        vmin, vmax = float(np.min(combined)), float(np.max(combined))
        if vmax <= vmin:
            vmax = vmin + 1.0
        linear_ranges[surface] = [vmin, vmax]
    plotting = {
        "logarithmic": {
            "cmap": COLORMAP_NAME,
            "norm": "LogNorm",
            "shared_color_range_kPa": [LOG_VMIN_KPA, LOG_VMAX_KPA],
            "upper_limit_rule": "fixed at 5 MPa to resolve the spatial extent of sub-MPa pressure while clipping localized higher values",
            "contour_levels_kPa": list(CONTOUR_LEVELS_KPA),
            "contour_labels": {
                "enabled": True,
                "format": "numeric values with kPa or MPa units",
            },
            "colorbar_ticks_kPa": list(COLORBAR_TICKS_KPA),
            "clipping": clipping,
        },
        "linear_diagnostic": {
            "cmap": COLORMAP_NAME,
            "norm": "Normalize",
            "row_shared_color_ranges_kPa": linear_ranges,
            "clipping": "none; each row spans the combined exact min-to-max range of its two fields",
        },
    }
    return plotting, statistics


def draw_wells(
    ax: plt.Axes, overlay: WellOverlay, show_labels: bool
) -> None:
    ax.scatter(
        overlay.x_km,
        overlay.y_km,
        s=18,
        marker="o",
        facecolors="0.85",
        edgecolors="0.2",
        linewidths=0.55,
        zorder=7,
    )
    emphasis = {"Well 10": "#E69F00", "Well 15": "#CC79A7"}
    for label, color in emphasis.items():
        index = overlay.labels.index(label)
        x, y = float(overlay.x_km[index]), float(overlay.y_km[index])
        ax.scatter(
            [x], [y], s=46, facecolors=color, edgecolors="black", linewidths=0.8, zorder=9
        )
        if not show_labels:
            continue
        text = ax.annotate(
            label.replace("Well ", "Well "),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            weight="bold",
            color="black",
            zorder=10,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])


def draw_projected_well_sources(
    ax: plt.Axes, overlay: WellOverlay, show_labels: bool
) -> None:
    styles = {
        "Well 6": {
            "facecolor": "0.85",
            "xytext": (5, -6),
            "verticalalignment": "top",
        },
        "Well 15": {
            "facecolor": "#CC79A7",
            "xytext": (5, -7),
            "verticalalignment": "top",
        },
    }
    for label, style in styles.items():
        index = overlay.labels.index(label)
        if not bool(overlay.fault_inside[index]):
            continue
        x = float(overlay.fault_x_km[index])
        y = float(overlay.fault_y_km[index])
        ax.scatter(
            [x],
            [y],
            s=46,
            marker="o",
            facecolors=str(style["facecolor"]),
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
        )
        if not show_labels:
            continue
        text = ax.annotate(
            label,
            (x, y),
            xytext=style["xytext"],
            textcoords="offset points",
            ha="left",
            va=str(style["verticalalignment"]),
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            weight="bold",
            color="black",
            zorder=10,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.2, foreground="white")]
        )


def draw_earthquakes(
    ax: plt.Axes, surface: str, overlay: EarthquakeOverlay
) -> None:
    if surface == "leduc_top":
        x, y = overlay.leduc_x_km, overlay.leduc_y_km
        size, alpha = 9, 0.38
    else:
        mask = overlay.fault_inside
        x, y = overlay.fault_x_km[mask], overlay.fault_y_km[mask]
        size, alpha = 12, 0.58
    ax.scatter(
        x,
        y,
        s=size,
        marker="o",
        facecolors=(1.0, 1.0, 1.0, 0.0),
        edgecolors=(1.0, 1.0, 1.0, alpha),
        linewidths=0.55,
        zorder=6,
    )


def draw_fault_s_guide(ax: plt.Axes, overlay: FaultGuideOverlay) -> None:
    (line,) = ax.plot(
        overlay.leduc_x_km,
        overlay.leduc_y_km,
        color="0.10",
        linewidth=1.15,
        linestyle=(0, (5, 3)),
        dash_capstyle="butt",
        zorder=3.5,
    )
    line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=2.6,
                foreground="white",
                alpha=0.88,
            ),
            path_effects.Normal(),
        ]
    )


def draw_fault_stratigraphy(
    ax: plt.Axes,
    overlay: FaultStratigraphyOverlay,
    *,
    include_labels: bool,
) -> None:
    (basement_line,) = ax.plot(
        overlay.basement_top_x_km,
        overlay.basement_top_y_km,
        color="0.12",
        linewidth=0.75,
        linestyle=(0, (6.0, 2.5, 1.4, 2.5)),
        dash_capstyle="butt",
        zorder=3.05,
    )
    basement_line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=1.45,
                foreground="white",
                alpha=0.78,
            ),
            path_effects.Normal(),
        ]
    )
    (leduc_line,) = ax.plot(
        overlay.leduc_x_km,
        overlay.leduc_y_km,
        color="0.12",
        linewidth=0.75,
        linestyle="solid",
        zorder=3.1,
    )
    leduc_line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=1.45,
                foreground="white",
                alpha=0.78,
            ),
            path_effects.Normal(),
        ]
    )
    if not include_labels:
        return
    for label, position in (
        ("Leduc", overlay.leduc_label_xy_km),
        ("Precambrian basement", overlay.basement_label_xy_km),
    ):
        text = ax.text(
            *position,
            label,
            ha="left",
            va="center",
            color="0.08",
            fontsize=9.5,
            fontfamily=SELECTED_FONT_FAMILY,
            fontweight="normal",
            clip_on=True,
            zorder=5.5,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=1.45, foreground="white")]
        )


def configure_axes(ax: plt.Axes, geometry: DisplayGeometry) -> None:
    ax.set_xlim(*geometry.xlim)
    ax.set_ylim(*geometry.ylim)
    ax.set_aspect("auto")
    ax.set_box_aspect(1.0)
    ax.set_xlabel(geometry.xlabel)
    ax.set_ylabel(geometry.ylabel)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0,
        pad=6,
    )
    ax.tick_params(which="minor", length=3)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    if geometry.surface == "leduc_top":
        ax.yaxis.set_major_locator(MultipleLocator(20.0))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_formatter(
            FuncFormatter(depth_below_sea_level_tick_formatter)
        )


def hide_redundant_shared_y_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("")
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=False,
        labelright=False,
    )


def depth_below_sea_level_tick_formatter(
    elevation_km: float, _position: int | None = None
) -> str:
    depth_km = -elevation_km
    if math.isclose(depth_km, 0.0, abs_tol=1.0e-12):
        depth_km = 0.0
    return f"{depth_km:g}"


def draw_panel(
    ax: plt.Axes,
    geometry: DisplayGeometry,
    values_pa: np.ndarray,
    norm: LogNorm | Normalize,
    logarithmic: bool,
    well_overlay: WellOverlay | None,
    earthquake_overlay: EarthquakeOverlay | None,
    fault_s_overlay: FaultGuideOverlay | None,
    fault_stratigraphy_overlay: FaultStratigraphyOverlay | None,
    classification_sign: str,
) -> object:
    values_kpa = values_pa * PRESSURE_TO_KPA
    if logarithmic:
        plotted = np.ma.masked_less_equal(values_kpa, 0.0)
        plotted = np.ma.clip(plotted, float(norm.vmin), float(norm.vmax))
    else:
        plotted = values_kpa
    mesh = ax.tripcolor(
        geometry.tri,
        plotted,
        shading="gouraud",
        cmap=COLORMAP_NAME,
        norm=norm,
        rasterized=True,
        zorder=1,
    )
    if geometry.surface == "leduc_top" and fault_s_overlay is not None:
        draw_fault_s_guide(ax, fault_s_overlay)
    if geometry.surface == "fault" and fault_stratigraphy_overlay is not None:
        draw_fault_stratigraphy(
            ax,
            fault_stratigraphy_overlay,
            include_labels=classification_sign == "negative",
        )
    if logarithmic:
        available_levels = [
            level
            for level in CONTOUR_LEVELS_KPA
            if float(np.min(values_kpa)) < level < float(np.max(values_kpa))
        ]
        if available_levels:
            label_format = {
                level: format_pressure_value_kpa(level)
                for level in available_levels
            }
            manual_levels = [
                level
                for level in available_levels
                if (geometry.surface, classification_sign, level)
                in MANUAL_CONTOUR_LABEL_POSITIONS_KM
            ]
            automatic_levels = [
                level for level in available_levels if level not in manual_levels
            ]
            labels: list[plt.Text] = []
            if automatic_levels:
                automatic_contours = ax.tricontour(
                    geometry.tri,
                    values_kpa,
                    levels=automatic_levels,
                    colors="white",
                    linewidths=0.75,
                    alpha=0.9,
                    zorder=4,
                )
                labels.extend(
                    ax.clabel(
                        automatic_contours,
                        levels=automatic_levels,
                        inline=True,
                        inline_spacing=3,
                        fmt=label_format,
                        fontsize=FIGURE_FONT_SIZE,
                        colors="white",
                    )
                )
            for level in manual_levels:
                manual_contour = ax.tricontour(
                    geometry.tri,
                    values_kpa,
                    levels=[level],
                    colors="white",
                    linewidths=0.75,
                    alpha=0.9,
                    zorder=4,
                )
                labels.extend(
                    ax.clabel(
                        manual_contour,
                        levels=[level],
                        manual=[
                            *MANUAL_CONTOUR_LABEL_POSITIONS_KM[
                                (geometry.surface, classification_sign, level)
                            ]
                        ],
                        inline=True,
                        inline_spacing=3,
                        fmt=label_format,
                        fontsize=FIGURE_FONT_SIZE,
                        colors="white",
                    )
                )
            for label in labels:
                label.set_fontfamily(SELECTED_FONT_FAMILY)
                label.set_path_effects(
                    [path_effects.withStroke(linewidth=1.5, foreground="black")]
                )
                label.set_zorder(5)
                x, _ = label.get_position()
                x_fraction = (x - geometry.xlim[0]) / (
                    geometry.xlim[1] - geometry.xlim[0]
                )
                if x_fraction < 0.035 or x_fraction > 0.965:
                    label.set_visible(False)
    show_well_labels = classification_sign == "negative"
    if geometry.surface == "leduc_top" and well_overlay is not None:
        draw_wells(ax, well_overlay, show_labels=show_well_labels)
    if geometry.surface == "fault" and well_overlay is not None:
        draw_projected_well_sources(
            ax, well_overlay, show_labels=show_well_labels
        )
    if earthquake_overlay is not None:
        draw_earthquakes(ax, geometry.surface, earthquake_overlay)
    configure_axes(ax, geometry)
    return mesh


def format_pressure_value_kpa(value_kpa: float) -> str:
    if value_kpa >= 1000.0:
        return f"{value_kpa / 1000.0:g} MPa"
    return f"{value_kpa:g} kPa"


def pressure_tick_formatter(value: float, _position: int | None = None) -> str:
    return format_pressure_value_kpa(value)


def add_panel_label(ax: plt.Axes, panel_label: str) -> None:
    ax.text(
        PANEL_HEADING_X_AXES,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
        fontweight="normal",
        clip_on=False,
        zorder=12,
    )


def add_panel_heading(
    ax: plt.Axes,
    panel_label: str,
    title: str,
    *,
    y_axes: float = PANEL_HEADING_Y_AXES,
) -> None:
    indented_title = title.replace(
        "\n", f"\n{PANEL_HEADING_CONTINUATION_INDENT}"
    )
    heading = f"{panel_label}{PANEL_HEADING_SEPARATOR}{indented_title}"
    ax.text(
        PANEL_HEADING_X_AXES,
        y_axes,
        heading,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
        fontweight="normal",
        linespacing=PANEL_HEADING_LINE_SPACING,
        clip_on=False,
        zorder=12,
    )


def align_map_group_decorations(
    fig: plt.Figure,
    map_axes: np.ndarray,
    shared_label_axis: plt.Axes,
    row_title_axes: tuple[plt.Axes, plt.Axes],
    colorbar_axis: plt.Axes,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    map_positions = [[ax.get_position() for ax in row] for row in map_axes]
    left_tight_boxes = [
        map_axes[row, 0]
        .get_tightbbox(renderer)
        .transformed(fig.transFigure.inverted())
        for row in range(2)
    ]
    right_tight_boxes = [
        map_axes[row, 1]
        .get_tightbbox(renderer)
        .transformed(fig.transFigure.inverted())
        for row in range(2)
    ]

    fig.set_layout_engine(None)

    for row, title_axis in enumerate(row_title_axes):
        map_position = map_positions[row][0]
        title_axis.set_position(
            [
                left_tight_boxes[row].x0
                - MAP_DECORATION_GAP_FIGURE
                - MAP_ROW_TITLE_WIDTH_FIGURE,
                map_position.y0,
                MAP_ROW_TITLE_WIDTH_FIGURE,
                map_position.height,
            ]
        )

    shared_label_right = (
        min(title_axis.get_position().x0 for title_axis in row_title_axes)
        - MAP_DECORATION_GAP_FIGURE
    )
    map_group_bottom = float(map_positions[1][0].y0)
    map_group_top = float(map_positions[0][0].y1)
    shared_label_axis.set_position(
        [
            shared_label_right - MAP_SHARED_LABEL_WIDTH_FIGURE,
            map_group_bottom,
            MAP_SHARED_LABEL_WIDTH_FIGURE,
            map_group_top - map_group_bottom,
        ]
    )

    colorbar_position = colorbar_axis.get_position()
    colorbar_axis.set_position(
        [
            max(box.x1 for box in right_tight_boxes) + MAP_DECORATION_GAP_FIGURE,
            map_group_bottom,
            colorbar_position.width,
            map_group_top - map_group_bottom,
        ]
    )
    fig.canvas.draw()


def add_leduc_overlay_legend(
    ax: plt.Axes, include_wells: bool, include_fault_s: bool
) -> None:
    handles: list[Line2D] = []
    if include_wells:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6.5,
                markerfacecolor="0.85",
                markeredgecolor="0.2",
                markeredgewidth=0.8,
                label="Modelled wells",
            )
        )
    if include_fault_s:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.10",
                linewidth=1.15,
                linestyle=(0, (5, 3)),
                dash_capstyle="butt",
                label="Fault S",
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="0.25",
        framealpha=0.88,
        borderpad=0.3,
        handletextpad=0.4,
        handlelength=2.0,
        labelspacing=0.35,
        borderaxespad=0.5,
        prop={"family": SELECTED_FONT_FAMILY, "size": FIGURE_FONT_SIZE},
    )
    legend.set_zorder(12)


def add_north_arrow(ax: plt.Axes) -> None:
    x = 0.075
    ax.annotate(
        "",
        xy=(x, 0.15),
        xytext=(x, 0.055),
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linewidth": 1.2,
            "mutation_scale": 12,
        },
        annotation_clip=False,
        zorder=12,
    )
    ax.text(
        x,
        0.17,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
        color="black",
        zorder=12,
    )


def add_fault_orientation_labels(ax: plt.Axes) -> None:
    for x, label, horizontal_alignment in (
        (0.035, "←West", "left"),
        (0.965, "East→", "right"),
    ):
        ax.text(
            x,
            0.035,
            label,
            transform=ax.transAxes,
            ha=horizontal_alignment,
            va="bottom",
            fontsize=FAULT_ORIENTATION_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            color="white",
            clip_on=True,
            zorder=12,
        )


def write_combined_figure(
    medians: dict[str, np.ndarray],
    geometry: dict[str, DisplayGeometry],
    plotting: dict[str, object],
    ecdf_panel_data: dict[str, pd.DataFrame],
    ecdf_summary_data: dict[str, pd.Series],
    ecdf_xlim: tuple[float, float],
    well_overlay: WellOverlay | None,
    earthquake_overlay: EarthquakeOverlay | None,
    fault_s_overlay: FaultGuideOverlay | None,
    fault_stratigraphy_overlay: FaultStratigraphyOverlay | None,
    pdf_path: Path,
    local_output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    """Write the pressure-only ECDF and logarithmic conditional-map Figure 3."""

    fig = plt.figure(figsize=COMBINED_FIGURE_SIZE_INCHES, constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        5,
        width_ratios=COMBINED_GRIDSPEC_WIDTH_RATIOS,
        height_ratios=COMBINED_GRIDSPEC_HEIGHT_RATIOS,
        wspace=0.0,
        hspace=0.0,
    )

    ecdf_left = fig.add_subplot(grid[0, 2])
    ecdf_axes = (
        ecdf_left,
        fig.add_subplot(grid[0, 3], sharex=ecdf_left, sharey=ecdf_left),
    )
    ecdf_specs = (
        (
            ecdf_axes[0],
            "well15",
            "(a)",
            "Well 15 ECDF",
            True,
            "Dominance fraction",
            r"$\Delta p_{15}^{(m)}$ (kPa)",
        ),
        (
            ecdf_axes[1],
            "well10",
            "(b)",
            "Well 10 ECDF",
            False,
            None,
            r"$\Delta p_{10}^{(m)}$ (kPa)",
        ),
    )
    for (
        ax,
        component,
        panel_label,
        title,
        show_ylabel,
        fraction_prefix,
        xlabel,
    ) in ecdf_specs:
        ax.set_box_aspect(1.0)
        dominance_ecdf.draw_panel(
            ax,
            {component: ecdf_panel_data[component]},
            {component: ecdf_summary_data[component]},
            ecdf_xlim,
            panel_label="",
            show_ylabel=show_ylabel,
            show_xlabel=True,
            xlabel=xlabel,
            show_dominance_arrow=True,
            show_fraction_text=False,
            line_color="black",
            line_style="-",
            fraction_text_prefix=fraction_prefix,
        )
        if show_ylabel:
            ax.set_ylabel(
                "Cumulative fraction of\nrealizations (%)",
                fontfamily=SELECTED_FONT_FAMILY,
            )
        add_panel_heading(
            ax,
            panel_label,
            title,
            y_axes=ECDF_PANEL_HEADING_Y_AXES,
        )

    shared_map_label_axis = fig.add_subplot(grid[2:, 0])
    shared_map_label_axis.set_axis_off()
    shared_map_label_axis.text(
        0.5,
        0.5,
        MAP_SHARED_VERTICAL_LABEL,
        transform=shared_map_label_axis.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=FIGURE_FONT_SIZE,
        fontfamily=SELECTED_FONT_FAMILY,
        fontweight="normal",
        linespacing=PANEL_HEADING_LINE_SPACING,
    )

    row_title_axes = (
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 1]),
    )
    map_axes = np.empty((2, 2), dtype=object)
    map_axes[0, 0] = fig.add_subplot(grid[2, 2])
    map_axes[0, 1] = fig.add_subplot(grid[2, 3], sharey=map_axes[0, 0])
    map_axes[1, 0] = fig.add_subplot(grid[3, 2])
    map_axes[1, 1] = fig.add_subplot(grid[3, 3], sharey=map_axes[1, 0])
    for title_axis, title in zip(
        row_title_axes, ("Leduc top", "Fault S FGZ midplane")
    ):
        title_axis.set_axis_off()
        title_axis.text(
            0.5,
            0.5,
            title,
            transform=title_axis.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            fontweight="normal",
        )

    log_range = plotting["logarithmic"]["shared_color_range_kPa"]  # type: ignore[index]
    shared_norm = LogNorm(
        vmin=float(log_range[0]), vmax=float(log_range[1]), clip=True
    )
    meshes: dict[str, object] = {}
    panel_labels = (("(c)", "(d)"), ("(e)", "(f)"))
    for row, surface in enumerate(SURFACES):
        for column, sign in enumerate(("positive", "negative")):
            mesh = draw_panel(
                ax=map_axes[row, column],
                geometry=geometry[surface],
                values_pa=medians[FIELD_KEYS[(surface, sign)]],
                norm=shared_norm,
                logarithmic=True,
                well_overlay=well_overlay,
                earthquake_overlay=earthquake_overlay,
                fault_s_overlay=fault_s_overlay,
                fault_stratigraphy_overlay=fault_stratigraphy_overlay,
                classification_sign=sign,
            )
            meshes[surface] = mesh
            add_panel_heading(
                map_axes[row, column],
                panel_labels[row][column],
                MAP_PANEL_TITLES[row][column],
            )

    for ax in (ecdf_axes[1], map_axes[0, 1], map_axes[1, 1]):
        hide_redundant_shared_y_axis(ax)
    for left_ax, right_ax in (
        ecdf_axes,
        (map_axes[0, 0], map_axes[0, 1]),
        (map_axes[1, 0], map_axes[1, 1]),
    ):
        left_ax.set_anchor("E")
        right_ax.set_anchor("W")

    add_north_arrow(map_axes[0, 0])
    add_fault_orientation_labels(map_axes[1, 0])
    if well_overlay is not None or fault_s_overlay is not None:
        add_leduc_overlay_legend(
            map_axes[0, 0],
            include_wells=well_overlay is not None,
            include_fault_s=fault_s_overlay is not None,
        )

    cax = fig.add_subplot(grid[2:, 4])
    colorbar = fig.colorbar(meshes["fault"], cax=cax)
    vmin, vmax = log_range
    ticks = [
        level for level in COLORBAR_TICKS_KPA if float(vmin) <= level <= float(vmax)
    ]
    ticks.extend([float(vmin), float(vmax)])
    ticks = sorted(set(ticks))
    if ticks:
        colorbar.locator = FixedLocator(ticks)
        colorbar.formatter = FuncFormatter(pressure_tick_formatter)
        colorbar.update_ticks()
    colorbar.set_label("Pressure perturbation")
    colorbar.ax.tick_params(
        which="both",
        direction="in",
        length=5,
        width=1.0,
        pad=6,
        labelsize=FIGURE_FONT_SIZE,
    )
    colorbar.ax.yaxis.label.set_fontfamily(SELECTED_FONT_FAMILY)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)

    align_map_group_decorations(
        fig,
        map_axes,
        shared_map_label_axis,
        row_title_axes,
        cax,
    )

    pdf_path = pdf_path.expanduser().resolve()
    png_path = local_output_dir / "paired_pressure_dominance_and_fields.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=PDF_RASTER_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_linear_diagnostic_figure(
    medians: dict[str, np.ndarray],
    geometry: dict[str, DisplayGeometry],
    plotting: dict[str, object],
    well_overlay: WellOverlay | None,
    earthquake_overlay: EarthquakeOverlay | None,
    fault_s_overlay: FaultGuideOverlay | None,
    fault_stratigraphy_overlay: FaultStratigraphyOverlay | None,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=MAP_FIGURE_SIZE_INCHES, constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(0.06, 1.0, 1.0, 0.075))
    row_title_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]))
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 1])
    axes[0, 1] = fig.add_subplot(grid[0, 2])
    axes[1, 0] = fig.add_subplot(grid[1, 1])
    axes[1, 1] = fig.add_subplot(grid[1, 2])
    columns = (
        "Well 15 pressure dominance "
        + r"($\Delta p_{15}^{(m)} > 0$)"
        + "\n266 realizations",
        "Remaining realizations "
        + r"($\Delta p_{15}^{(m)} < 0$)"
        + "\n734 realizations",
    )
    for column, title in enumerate(columns):
        axes[0, column].set_title(
            title,
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            fontweight="normal",
            pad=8,
        )

    row_labels = ("Leduc top", "Fault S FGZ midplane")
    for title_axis, title in zip(row_title_axes, row_labels):
        title_axis.set_axis_off()
        title_axis.text(
            0.5,
            0.5,
            title,
            transform=title_axis.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=FIGURE_FONT_SIZE,
            fontfamily=SELECTED_FONT_FAMILY,
            fontweight="normal",
        )

    meshes: dict[str, object] = {}
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    for row, surface in enumerate(SURFACES):
        linear_range = plotting["linear_diagnostic"]["row_shared_color_ranges_kPa"][surface]  # type: ignore[index]
        norm = Normalize(
            vmin=float(linear_range[0]), vmax=float(linear_range[1]), clip=False
        )
        for column, sign in enumerate(("positive", "negative")):
            mesh = draw_panel(
                ax=axes[row, column],
                geometry=geometry[surface],
                values_pa=medians[FIELD_KEYS[(surface, sign)]],
                norm=norm,
                logarithmic=False,
                well_overlay=well_overlay,
                earthquake_overlay=earthquake_overlay,
                fault_s_overlay=fault_s_overlay,
                fault_stratigraphy_overlay=fault_stratigraphy_overlay,
                classification_sign=sign,
            )
            meshes[surface] = mesh
            add_panel_label(axes[row, column], panel_labels[row][column])

    add_north_arrow(axes[0, 0])
    if well_overlay is not None or fault_s_overlay is not None:
        add_leduc_overlay_legend(
            axes[0, 0],
            include_wells=well_overlay is not None,
            include_fault_s=fault_s_overlay is not None,
        )

    colorbars = []
    for row, surface in enumerate(SURFACES):
        cax = fig.add_subplot(grid[row, 3])
        colorbars.append(fig.colorbar(meshes[surface], cax=cax))
    for colorbar in colorbars:
        colorbar.ax.tick_params(direction="in")
        colorbar.set_label("Pressure perturbation (kPa)")
    colorbar = colorbars[0]
    colorbar.ax.tick_params(
        which="both",
        direction="in",
        length=5,
        width=1.0,
        pad=6,
        labelsize=FIGURE_FONT_SIZE,
    )
    colorbar.ax.yaxis.label.set_fontfamily(SELECTED_FONT_FAMILY)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(SELECTED_FONT_FAMILY)

    pdf_path = output_dir / "well15_paired_sign_pressure_maps_nov2022_linear_diagnostic.pdf"
    png_path = output_dir / "well15_paired_sign_pressure_maps_nov2022_linear_diagnostic.png"
    fig.savefig(pdf_path, dpi=PDF_RASTER_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def load_timeseries_snapshot(path: Path, model_time: float) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    pieces = []
    for chunk in pd.read_csv(
        path,
        usecols=["case", "time_month", "pressure_Pa"],
        chunksize=100_000,
    ):
        selected = chunk.loc[chunk["time_month"] == model_time, ["case", "pressure_Pa"]]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise RuntimeError(f"No time {model_time:g} rows found in {path}")
    frame = pd.concat(pieces, ignore_index=True)
    if frame["case"].duplicated().any():
        raise RuntimeError(f"Duplicate case/time rows in {path} at time {model_time:g}")
    series = frame.set_index("case")["pressure_Pa"].sort_index()
    if tuple(series.index.astype(str)) != EXPECTED_CASES:
        raise RuntimeError(f"Incomplete cases in {path} at time {model_time:g}")
    if np.any(~np.isfinite(series.to_numpy(dtype=float))):
        raise RuntimeError(f"Non-finite pressure in {path} at time {model_time:g}")
    return series


def single_snapshot_sign_diagnostic(
    timeseries_dir: Path | None, classification: pd.DataFrame
) -> tuple[dict[str, object], pd.Series | None]:
    if timeseries_dir is None:
        return {"available": False, "reason": "pressure time-series directory not found"}, None
    all_path = timeseries_dir / "fault_pressure_timeseries_all_wells.csv"
    well15_path = timeseries_dir / "fault_pressure_timeseries_well15.csv"
    try:
        all_516 = load_timeseries_snapshot(all_path, DISPLAY_TIME)
        well15_516 = load_timeseries_snapshot(well15_path, DISPLAY_TIME)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}, None
    delta = 2.0 * well15_516 - all_516
    stored = classification.set_index("case")["paired_difference"].reindex(delta.index)
    window_sign = np.sign(stored.to_numpy(dtype=float))
    snapshot_sign = np.sign(delta.to_numpy(dtype=float))
    agreement = window_sign == snapshot_sign
    mismatch_records = []
    for case in delta.index[~agreement]:
        mismatch_records.append(
            {
                "case": str(case),
                "classification_window_paired_difference_Pa": float(stored.loc[case]),
                "time_516_paired_difference_Pa": float(delta.loc[case]),
            }
        )
    diagnostic = {
        "available": True,
        "definition": "2 * well15_pressure_Pa(time=516) - all_wells_pressure_Pa(time=516)",
        "classification_basis": "stored November 2022 output 516 paired_difference",
        "all_wells_csv": str(all_path),
        "well15_csv": str(well15_path),
        "all_wells_csv_sha256": sha256_file(all_path),
        "well15_csv_sha256": sha256_file(well15_path),
        "positive_count": int(np.count_nonzero(snapshot_sign > 0.0)),
        "negative_count": int(np.count_nonzero(snapshot_sign < 0.0)),
        "zero_count": int(np.count_nonzero(snapshot_sign == 0.0)),
        "agreement_count": int(np.count_nonzero(agreement)),
        "total_count": int(agreement.size),
        "agreement_fraction": float(np.mean(agreement)),
        "mismatch_cases": mismatch_records,
    }
    return diagnostic, all_516


def area_average_diagnostic(
    case_root: Path,
    inventory: dict[str, SurfaceInventory],
    merged_all_516: pd.Series | None,
) -> dict[str, object]:
    if merged_all_516 is None:
        return {"available": False, "reason": "merged all-wells time-series snapshot unavailable"}
    try:
        spec = importlib.util.spec_from_file_location(
            "peace_river_reference_area_average", REFERENCE_AREA_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load module specification: {REFERENCE_AREA_SCRIPT}")
        area_average_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(area_average_module)
        weighted_mean = area_average_module.weighted_mean
    except Exception as exc:
        return {"available": False, "reason": f"canonical weighted_mean import failed: {exc}"}
    representative_cases = (
        "case_000001",
        "case_000006",
        "case_000015",
        "case_000124",
        "case_000253",
        "case_001000",
    )
    records = []
    try:
        for case in representative_cases:
            post = case_root / case / "post"
            pressure = np.load(
                post / "resample" / "fault" / "pressure.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            time = np.load(post / "resample" / "fault" / "time.npy", allow_pickle=False)
            depth = np.load(post / "dieterich" / "depth_km.npy", allow_pickle=False).astype(float)
            weights = np.load(post / "dieterich" / "weights.npy", allow_pickle=False).astype(float)
            index = np.flatnonzero(time == DISPLAY_TIME)
            if index.size != 1:
                raise RuntimeError(f"time 516 is not unique for area diagnostic: {case}")
            mask = (
                (depth >= 2.0)
                & (depth <= 7.0)
                & np.isfinite(depth)
                & np.isfinite(weights)
                & (weights > 0.0)
            )
            direct = float(
                weighted_mean(
                    np.asarray(pressure[int(index[0]), mask], dtype=float)[None, :],
                    weights[mask],
                )[0]
            )
            band_path = post / "band_stats" / "area_average_z_2.000_7.000_km.csv"
            band = pd.read_csv(band_path)
            band_row = band.loc[band["time_month"] == DISPLAY_TIME]
            if len(band_row) != 1:
                raise RuntimeError(f"Band CSV time 516 is not unique: {band_path}")
            band_value = float(band_row.iloc[0]["pressure_Pa"])
            merged_value = float(merged_all_516.loc[case])
            direct_band_difference = abs(direct - band_value)
            band_merged_difference = abs(band_value - merged_value)
            if not np.isclose(direct, band_value, rtol=1.0e-12, atol=1.0e-8):
                raise RuntimeError(
                    f"Canonical area-average mismatch for {case}: direct={direct}, band={band_value}"
                )
            if not np.isclose(band_value, merged_value, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"Band/merged time-series mismatch for {case}: band={band_value}, merged={merged_value}"
                )
            records.append(
                {
                    "case": case,
                    "selected_node_count": int(np.count_nonzero(mask)),
                    "weight_sum": float(np.sum(weights[mask])),
                    "direct_from_pressure_npy_Pa": direct,
                    "retained_band_csv_Pa": band_value,
                    "merged_timeseries_Pa": merged_value,
                    "direct_vs_band_abs_difference_Pa": direct_band_difference,
                    "band_vs_merged_abs_difference_Pa": band_merged_difference,
                }
            )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "partial_records": records}
    return {
        "available": True,
        "canonical_workflow": str(REFERENCE_AREA_SCRIPT),
        "definition": "2..7 km finite positive-weight nodal area mean, (x @ w) / sum(w)",
        "representative_cases": records,
        "max_direct_vs_band_abs_difference_Pa": float(
            max(record["direct_vs_band_abs_difference_Pa"] for record in records)
        ),
        "max_band_vs_merged_abs_difference_Pa": float(
            max(record["band_vs_merged_abs_difference_Pa"] for record in records)
        ),
    }


def save_numerical_outputs(
    output_dir: Path,
    classification: pd.DataFrame,
    inventory: dict[str, SurfaceInventory],
    medians: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    classification_path = output_dir / "well15_paired_sign_case_classification.csv"
    classification.to_csv(classification_path, index=False, float_format="%.17g")
    npz_path = output_dir / "well15_paired_sign_conditional_medians.npz"
    np.savez_compressed(
        npz_path,
        leduc_top_points_m=inventory["leduc_top"].points,
        fault_points_m=inventory["fault"].points,
        **medians,
    )
    return classification_path, npz_path


def main() -> int:
    args = parse_args()
    font_file = args.font_file.expanduser().resolve() if args.font_file else None
    selected_font = configure_matplotlib(font_file)
    dominance_ecdf.use_font_family(selected_font)
    script_path = Path(__file__).resolve()
    script_sha256_at_start = sha256_file(script_path)
    ecdf_script_path = Path(dominance_ecdf.__file__).resolve()
    ecdf_script_sha256_at_start = sha256_file(ecdf_script_path)
    if args.config is None:
        config_path = None
        config: dict[str, object] = {}
    else:
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Local config not found: {config_path}")
        config = load_flat_yaml(config_path)
    paired_csv = args.paired_csv.expanduser().resolve()
    ecdf_panel_data, ecdf_summary_data, _ = (
        dominance_ecdf.load_manuscript_pressure_panels(paired_csv)
    )
    ecdf_xlim = ECDF_XLIM_KPA
    print(
        "[OK] pressure ECDF fractions: "
        f"well15={float(ecdf_summary_data['well15']['fraction_positive']):.3f}, "
        f"well10={float(ecdf_summary_data['well10']['fraction_positive']):.3f}"
    )
    case_root, case_root_source = resolve_case_root(args, config)
    base_commit = git_commit("HEAD")
    origin_main_commit = git_commit("origin/main")
    paired_csv_matches_current_head = assert_tracked_input_matches_current_head(paired_csv)

    classification, nearest_zero = classify_cases(paired_csv)
    positive_count = int((classification["sign"] == "positive").sum())
    negative_count = int((classification["sign"] == "negative").sum())
    zero_count = int((classification["sign"] == "zero").sum())
    print(
        f"[OK] classification: cases={len(classification)}, positive={positive_count}, "
        f"negative={negative_count}, zero={zero_count}"
    )
    print("[INFO] five smallest absolute paired differences (kPa):")
    for record in nearest_zero:
        print(f"  {record['case']}: {record['paired_difference_kPa']:+.17g}")

    inventory, inventory_metadata = validate_inventory(
        case_root, classification["case"].tolist()
    )
    print(
        f"[OK] inventory: {inventory_metadata['expected_npy_file_count']} / "
        f"{inventory_metadata['expected_npy_file_count']} required NPY files"
    )
    for surface in SURFACES:
        item = inventory[surface]
        print(
            f"[OK] {surface}: pressure_shape={item.pressure_shape}, nodes={item.node_count}, "
            f"times={item.time_count}, time_516_index={item.display_time_index}"
        )
    if args.validate_only:
        print("[DONE] validate-only mode; no output files were written")
        return 0

    geometry = build_display_geometry(inventory)
    (
        well_overlay,
        earthquake_overlay,
        fault_s_overlay,
        fault_stratigraphy_overlay,
        overlay_metadata,
        overlay_warnings,
    ) = load_optional_overlays(config, inventory, geometry, args.no_overlays)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_output = (
        args.combined_output.expanduser().resolve()
        if args.combined_output is not None
        else output_dir / COMBINED_OUTPUT_NAME
    )
    medians, temporary_metadata = compute_conditional_medians(
        case_root=case_root,
        classification=classification,
        inventory=inventory,
        temporary_parent=args.temporary_parent,
        chunk_nodes=args.chunk_nodes,
    )
    classification_path, npz_path = save_numerical_outputs(
        output_dir, classification, inventory, medians
    )
    fixed_node_checks = fixed_node_median_checks(
        case_root, classification, inventory, npz_path
    )
    plotting, statistics = calculate_plot_limits(medians)
    plotting["style"] = {
        "font_family": selected_font,
        "font_size_pt": FIGURE_FONT_SIZE,
        "fault_orientation_font_size_pt": FAULT_ORIENTATION_FONT_SIZE,
        "figure_size_inches": list(COMBINED_FIGURE_SIZE_INCHES),
        "linear_diagnostic_figure_size_inches": list(MAP_FIGURE_SIZE_INCHES),
        "combined_gridspec_width_ratios": list(COMBINED_GRIDSPEC_WIDTH_RATIOS),
        "combined_gridspec_height_ratios": list(COMBINED_GRIDSPEC_HEIGHT_RATIOS),
        "combined_gridspec_wspace": 0.0,
        "combined_gridspec_hspace": 0.0,
        "shared_y_axes_by_row": True,
        "right_column_y_ticks_visible": True,
        "right_column_y_tick_labels_visible": False,
        "right_column_y_axis_label_visible": False,
        "paired_panel_anchors": ["E", "W"],
        "ecdf_panel_titles_follow_labels": True,
        "map_panel_titles": [list(row) for row in MAP_PANEL_TITLES],
        "map_shared_vertical_label": MAP_SHARED_VERTICAL_LABEL,
        "map_decoration_gap_figure": MAP_DECORATION_GAP_FIGURE,
        "map_row_title_width_figure": MAP_ROW_TITLE_WIDTH_FIGURE,
        "map_shared_label_width_figure": MAP_SHARED_LABEL_WIDTH_FIGURE,
        "map_decorations_aligned_to_rendered_axes": True,
        "panel_heading_single_text_object": True,
        "panel_heading_x_axes": PANEL_HEADING_X_AXES,
        "panel_heading_y_axes": PANEL_HEADING_Y_AXES,
        "ecdf_panel_heading_y_axes": ECDF_PANEL_HEADING_Y_AXES,
        "panel_heading_separator": "one space",
        "panel_heading_line_spacing": PANEL_HEADING_LINE_SPACING,
        "ecdf_xlim_kPa": list(ecdf_xlim),
        "panel_labels_outside_axes": True,
        "row_titles_outside_axes": True,
        "combined_north_arrow_panel": "c",
        "combined_leduc_overlay_legend_panel": "c",
        "linear_diagnostic_north_arrow_panel": "a",
        "linear_diagnostic_leduc_overlay_legend_panel": "a",
    }
    combined_pdf, combined_png = write_combined_figure(
        medians=medians,
        geometry=geometry,
        plotting=plotting,
        ecdf_panel_data=ecdf_panel_data,
        ecdf_summary_data=ecdf_summary_data,
        ecdf_xlim=ecdf_xlim,
        well_overlay=well_overlay,
        earthquake_overlay=earthquake_overlay,
        fault_s_overlay=fault_s_overlay,
        fault_stratigraphy_overlay=fault_stratigraphy_overlay,
        pdf_path=combined_output,
        local_output_dir=output_dir,
        png_dpi=args.png_dpi,
    )
    linear_pdf, linear_png = write_linear_diagnostic_figure(
        medians,
        geometry,
        plotting,
        well_overlay,
        earthquake_overlay,
        fault_s_overlay,
        fault_stratigraphy_overlay,
        output_dir,
        png_dpi=args.png_dpi,
    )

    timeseries_dir = resolve_timeseries_dir(config)
    sign_diagnostic, merged_all_516 = single_snapshot_sign_diagnostic(
        timeseries_dir, classification
    )
    area_diagnostic = area_average_diagnostic(
        case_root, inventory, merged_all_516
    )
    warnings = list(overlay_warnings)
    if not sign_diagnostic.get("available", False):
        warnings.append(f"Single-snapshot sign diagnostic unavailable: {sign_diagnostic.get('reason')}")
    if not area_diagnostic.get("available", False):
        warnings.append(f"Area-average diagnostic unavailable: {area_diagnostic.get('reason')}")

    geometry_metadata: dict[str, object] = {}
    for surface, item in geometry.items():
        geometry_metadata[surface] = {
            "x_range_km": list(item.xlim),
            "y_range_km": list(item.ylim),
            "axis_labels": [item.xlabel, item.ylabel],
            "box_aspect": 1.0,
            "triangles_total": int(item.tri.triangles.shape[0]),
            "triangles_masked": int(
                np.count_nonzero(item.tri.mask) if item.tri.mask is not None else 0
            ),
            "triangle_edge_mask_threshold_km": item.max_edge_km,
            "left_right_axes_identical": True,
        }
    projection = fault_projection(inventory["fault"].points)
    geometry_metadata["fault"]["pca_along_axis_xy"] = projection.along_axis.tolist()  # type: ignore[index]
    geometry_metadata["fault"]["pca_xy_center_m"] = projection.xy_center_m.tolist()  # type: ignore[index]
    geometry_metadata["leduc_top"]["clockwise_rotation_deg"] = LEDUC_CLOCKWISE_ROTATION_DEG  # type: ignore[index]

    output_paths = [
        classification_path,
        npz_path,
        combined_pdf,
        combined_png,
        linear_pdf,
        linear_png,
    ]
    metadata_path = output_dir / "well15_paired_sign_run_metadata.json"
    script_sha256_at_end = sha256_file(script_path)
    if script_sha256_at_end != script_sha256_at_start:
        raise RuntimeError("The plotting script changed while the run was in progress")
    if sha256_file(ecdf_script_path) != ecdf_script_sha256_at_start:
        raise RuntimeError("The imported ECDF script changed while the run was in progress")
    metadata = {
        "schema_version": 1,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit_hash": base_commit,
        "origin_main_commit_hash": origin_main_commit,
        "script": str(script_path),
        "script_sha256": script_sha256_at_start,
        "reference_scripts": {
            "ecdf_input_and_panels": str(ecdf_script_path),
            "ecdf_input_and_panels_sha256": ecdf_script_sha256_at_start,
            "time_definition": str(REFERENCE_TIME_SCRIPT),
            "display_geometry": REFERENCE_GEOMETRY_WORKFLOW,
            "area_average": str(REFERENCE_AREA_SCRIPT),
        },
        "inputs": {
            "local_config": str(config_path) if config_path is not None else None,
            "paired_csv": str(paired_csv),
            "paired_csv_sha256": sha256_file(paired_csv),
            "paired_csv_matches_current_head": paired_csv_matches_current_head,
            "head_matches_origin_main": (
                base_commit is not None
                and origin_main_commit is not None
                and base_commit == origin_main_commit
            ),
            "ecdf_derived_from_paired_csv": True,
            "ecdf_quantity": dominance_ecdf.QUANTITY_PRESSURE,
            "ecdf_components": ["well15", "well10"],
            "case_root": str(case_root),
            "case_root_resolution": case_root_source,
            "pressure_timeseries_dir": str(timeseries_dir) if timeseries_dir is not None else None,
        },
        "classification": {
            "source": "stored paired_difference_kPa; no spatial-field recomputation and no tolerance",
            "source_group": "well15",
            "unit": "kPa",
            "window_model_times": [DISPLAY_TIME],
            "window_months": [DISPLAY_MONTH],
            "window_definition": "November 2022 monthly output 516",
            "case_count": len(classification),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "five_smallest_absolute_paired_differences": nearest_zero,
        },
        "display_field": {
            "simulation": "all 15 wells active (cases_pr_all_wells)",
            "model_time": DISPLAY_TIME,
            "calendar_label": DISPLAY_MONTH,
            "meaning": "cumulative pressure perturbation at the November 2022 monthly output; not an October-to-November increment and not reset at the mainshock",
            "source_unit": "Pa",
            "median_computation_unit": "Pa",
            "display_unit": "kPa",
            "Pa_to_kPa_factor": PRESSURE_TO_KPA,
            "classification_display_time_alignment": "Classification and displayed pressure field both use the November 2022 monthly output 516.",
        },
        "inventory": inventory_metadata,
        "temporary_storage": temporary_metadata,
        "median_fields": statistics,
        "plotting": plotting,
        "display_geometry": geometry_metadata,
        "overlays": overlay_metadata,
        "validation": {
            "complete_unique_case_set": True,
            "required_sign_counts_reproduced": True,
            "ecdf_pressure_only": True,
            "ecdf_pressure_fraction_positive": {
                "well15": float(
                    ecdf_summary_data["well15"]["fraction_positive"]
                ),
                "well10": float(
                    ecdf_summary_data["well10"]["fraction_positive"]
                ),
            },
            "surface_mesh_points_node_order_and_time_identical_all_cases": True,
            "display_time_unique_all_cases": True,
            "display_pressure_finite_all_cases": True,
            "pressure_kept_in_Pa_until_plotting": True,
            "fixed_node_direct_median_checks": fixed_node_checks,
            "left_right_coordinate_and_color_ranges_identical": True,
            "input_node_remapping_reordering_or_off_surface_interpolation": False,
            "rendering_shading": "Gouraud interpolation only within the existing display triangulation; no input remapping or extrapolation",
            "median_smoothing_normalization_reweighting_or_preclipping": False,
            "single_snapshot_sign_diagnostic": sign_diagnostic,
            "canonical_area_average_diagnostic": area_diagnostic,
        },
        "figure_caption_note": (
            "Nodewise medians are conditional summaries across different all-wells "
            "realizations and are not a single model solution or a realization with "
            "median hydraulic properties. Groups and mapped all-wells fields use "
            "the same November 2022 monthly output (516)."
        ),
        "warnings": warnings,
        "outputs": [str(path.resolve()) for path in output_paths] + [str(metadata_path.resolve())],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] time-516 sign agreement: {sign_diagnostic.get('agreement_fraction', 'unavailable')}")
    print(f"[OK] configured font family: {selected_font}")
    print(f"[OK] wrote metadata: {metadata_path}")
    for path in output_paths:
        print(f"[OK] wrote: {path}")
    if warnings:
        for warning in warnings:
            print(f"[WARN] {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
