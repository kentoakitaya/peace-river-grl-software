#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


DEFAULT_RELATIVE_CSV = Path("post") / "band_stats" / "area_average_z_2.000_7.000_km.csv"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_config(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Config file must contain a YAML mapping: {path}")
    return config


def config_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_cases_root(cases_root_arg: str | None, case_collection: str | None, config: dict[str, object]) -> Path:
    if cases_root_arg:
        return Path(cases_root_arg).expanduser().resolve()

    postprocessed_root = config_path(config, "ogs_results_postprocessed_dir")
    postprocessed_candidate = None
    if postprocessed_root is not None:
        postprocessed_candidate = postprocessed_root
        if case_collection:
            postprocessed_candidate = postprocessed_root / case_collection
        if postprocessed_candidate.exists() or not case_collection:
            return postprocessed_candidate

    raw_root = config_path(config, "ogs_results_raw_dir")
    if raw_root is not None and case_collection:
        raw_candidate = raw_root / case_collection
        if raw_candidate.exists():
            return raw_candidate

    if postprocessed_candidate is not None:
        return postprocessed_candidate

    raise RuntimeError(
        "--cases_root is required, or pass --config with ogs_results_postprocessed_dir "
        "or ogs_results_raw_dir and optionally --case_collection."
    )


def resolve_output_csv(out_csv_arg: str | None, config: dict[str, object]) -> Path:
    if out_csv_arg:
        return Path(out_csv_arg).expanduser().resolve()

    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is not None:
        return local_output_dir / "postprocess" / "analysis" / "fault_pressure_timeseries.csv"

    return repo_root() / "outputs" / "postprocess" / "analysis" / "fault_pressure_timeseries.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect depth band area averaged pressure time series from many case directories "
            "and write one long table for correlation analysis."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional local YAML config file.",
    )
    parser.add_argument(
        "--cases_root",
        default=None,
        help=(
            "Directory that contains case_XXXXXX folders. If omitted, this may be "
            "resolved from --config ogs_results_postprocessed_dir."
        ),
    )
    parser.add_argument(
        "--case_collection",
        default=None,
        help=(
            "Optional subdirectory under ogs_results_postprocessed_dir, such as "
            "cases_pr_all_wells. If that path does not exist, ogs_results_raw_dir "
            "is checked for legacy per-case post outputs."
        ),
    )
    parser.add_argument(
        "--case_glob",
        default="case_*",
        help="Glob used to find case directories under cases_root",
    )
    parser.add_argument(
        "--relative_csv",
        default=str(DEFAULT_RELATIVE_CSV),
        help=(
            "Path to the target CSV relative to each case directory. "
            "Default is post/band_stats/area_average_z_2.000_7.000_km.csv"
        ),
    )
    parser.add_argument(
        "--params_csv",
        default=None,
        help=(
            "Optional parameter table to merge by case name. "
            "The table must contain a column named case or case_name"
        ),
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path for the merged long table",
    )
    parser.add_argument(
        "--out_missing_csv",
        default=None,
        help="Optional CSV path for a list of cases whose target file was missing",
    )
    parser.add_argument(
        "--sort_by",
        choices=["case", "time"],
        default="case",
        help="Row ordering in the output table",
    )
    return parser


REQUIRED_SOURCE_COLUMNS = [
    "month_index",
    "pressure_Pa",
    "pressure_rate_Pa_per_month",
]

OPTIONAL_SOURCE_COLUMNS = [
    "time_raw",
    "zmin_km",
    "zmax_km",
    "n_nodes",
    "sum_weights",
    "R",
    "chi",
    "chiR",
]


def normalise_case_column(df: pd.DataFrame) -> pd.DataFrame:
    if "case" in df.columns:
        return df
    if "case_name" in df.columns:
        return df.rename(columns={"case_name": "case"})
    raise ValueError("params_csv must contain a column named case or case_name")


def read_case_timeseries(case_dir: Path, relative_csv: Path) -> pd.DataFrame:
    csv_path = case_dir / relative_csv
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}. Missing columns are {missing}"
        )

    out = pd.DataFrame()
    out["case"] = [case_dir.name] * len(df)
    out["month_index"] = pd.to_numeric(df["month_index"], errors="coerce").astype("Int64")
    out["time_month"] = pd.to_numeric(df["month_index"], errors="coerce")
    out["pressure_Pa"] = pd.to_numeric(df["pressure_Pa"], errors="coerce")
    out["pressure_rate_Pa_per_month"] = pd.to_numeric(
        df["pressure_rate_Pa_per_month"], errors="coerce"
    )

    if "pressure_Pa" in out.columns:
        out["pressure_MPa"] = out["pressure_Pa"] / 1.0e6
    if "pressure_rate_Pa_per_month" in out.columns:
        out["pressure_rate_MPa_per_month"] = out["pressure_rate_Pa_per_month"] / 1.0e6

    for col in OPTIONAL_SOURCE_COLUMNS:
        if col in df.columns:
            out[col] = df[col]

    out["source_csv"] = str(csv_path)
    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    cases_root = resolve_cases_root(args.cases_root, args.case_collection, config)
    relative_csv = Path(args.relative_csv)
    out_csv = resolve_output_csv(args.out_csv, config)
    out_missing_csv = Path(args.out_missing_csv).expanduser().resolve() if args.out_missing_csv else None

    if not cases_root.exists():
        raise FileNotFoundError(
            f"cases_root not found {cases_root}. Pass --cases_root explicitly, or set "
            "ogs_results_postprocessed_dir or ogs_results_raw_dir in --config."
        )

    case_dirs = sorted([p for p in cases_root.glob(args.case_glob) if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(
            f"No case directories found under {cases_root} with glob {args.case_glob}"
        )

    frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, str]] = []
    error_messages: list[str] = []

    for case_dir in case_dirs:
        try:
            frames.append(read_case_timeseries(case_dir, relative_csv))
        except FileNotFoundError as exc:
            missing_rows.append(
                {
                    "case": case_dir.name,
                    "missing_file": str(exc),
                }
            )
        except Exception as exc:
            error_messages.append(f"{case_dir.name}  {exc}")

    if error_messages:
        joined = "\n".join(error_messages[:20])
        if len(error_messages) > 20:
            joined += f"\n... plus {len(error_messages) - 20} more errors"
        raise RuntimeError(f"Errors were found while reading case files\n{joined}")

    if not frames:
        raise RuntimeError("No readable case time series were found")

    merged = pd.concat(frames, ignore_index=True)

    if args.params_csv:
        params_df = pd.read_csv(Path(args.params_csv).expanduser().resolve())
        params_df = normalise_case_column(params_df)
        merged = merged.merge(params_df, on="case", how="left", validate="many_to_one")

    sort_cols = ["case", "month_index"] if args.sort_by == "case" else ["month_index", "case"]
    merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    columns_to_keep = [
        "case",
        "time_month",
        "pressure_Pa",
        "pressure_rate_Pa_per_month",
    ]
    merged = merged[columns_to_keep]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    if out_missing_csv is not None:
        missing_df = pd.DataFrame(missing_rows)
        out_missing_csv.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(out_missing_csv, index=False)

    n_cases_total = len(case_dirs)
    n_cases_used = merged["case"].drop_duplicates().shape[0]
    n_rows = len(merged)

    print(f"[OK] wrote merged table  {out_csv}")
    print(f"[INFO] total case directories  {n_cases_total}")
    print(f"[INFO] cases included  {n_cases_used}")
    print(f"[INFO] output rows  {n_rows}")

    if missing_rows:
        print(f"[WARN] missing target file for {len(missing_rows)} cases")
        if out_missing_csv is not None:
            print(f"[INFO] missing case list  {out_missing_csv}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
