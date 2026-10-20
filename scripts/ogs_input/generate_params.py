#!/usr/bin/env python3
"""
Generate params.csv for the Peace River permeability ensemble.

Design
- Leduc permeability k_L is log spaced from 2 to 200 mD with 10 points
- Damage zone contrast r_DZ = k_DZ / k_L is log spaced from 1e-2 to 1 with 10 points
- Gouge contrast r_G = k_G / k_L is log spaced from 1e-5 to 1e-2 with 10 points
- Total cases = 10 x 10 x 10 = 1000

Output units
- params.csv stores permeability in m/month for direct use in OGS
- params_design_reference.csv also stores mD values and contrast ratios

Conversion
- 1 mD = 0.0254 m/month
  This is the operational factor used here to match the current model setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N_LEDUC = 10
N_DZ = 10
N_G = 10

LEDUC_RANGE_MD = (2.0, 200.0)
DZ_RATIO_RANGE = (1.0e-2, 1.0)
GOUGE_RATIO_RANGE = (1.0e-5, 1.0e-2)

MD_TO_M_PER_MONTH = 0.0254


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def resolve_out_dir(out_dir_arg: str | None, config: dict[str, object]) -> Path:
    if out_dir_arg:
        return Path(out_dir_arg).expanduser().resolve()

    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is not None:
        return local_output_dir / "ogs_input" / "params"

    return repo_root() / "outputs" / "ogs_input" / "params"


def build_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    kL_md = np.geomspace(*LEDUC_RANGE_MD, N_LEDUC)
    rDZ = np.geomspace(*DZ_RATIO_RANGE, N_DZ)
    rG = np.geomspace(*GOUGE_RATIO_RANGE, N_G)

    params_rows = []
    ref_rows = []

    case_id = 1
    for kL in kL_md:
        for dz_ratio in rDZ:
            kDZ = kL * dz_ratio
            for g_ratio in rG:
                kG = kL * g_ratio

                params_rows.append(
                    {
                        "case_id": case_id,
                        "leduc_perm": kL * MD_TO_M_PER_MONTH,
                        "fdz_perm": kDZ * MD_TO_M_PER_MONTH,
                        "gouge_perm": kG * MD_TO_M_PER_MONTH,
                    }
                )

                ref_rows.append(
                    {
                        "case_id": case_id,
                        "leduc_perm_mD": kL,
                        "fdz_perm_mD": kDZ,
                        "gouge_perm_mD": kG,
                        "dz_ratio": dz_ratio,
                        "gouge_ratio": g_ratio,
                        "leduc_perm_m_per_month": kL * MD_TO_M_PER_MONTH,
                        "fdz_perm_m_per_month": kDZ * MD_TO_M_PER_MONTH,
                        "gouge_perm_m_per_month": kG * MD_TO_M_PER_MONTH,
                    }
                )

                case_id += 1

    return pd.DataFrame(params_rows), pd.DataFrame(ref_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Peace River OGS permeability ensemble parameter tables."
    )
    parser.add_argument("--config", default=None, help="Optional local YAML config file.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to local_output_dir/ogs_input/params when --config is used.",
    )
    parser.add_argument("--params_filename", default="params.csv")
    parser.add_argument("--reference_filename", default="params_design_reference.csv")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the ensemble in memory and print planned output paths without writing CSV files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = resolve_out_dir(args.out_dir, config)

    params_df, ref_df = build_ensemble()

    params_csv = out_dir / args.params_filename
    reference_csv = out_dir / args.reference_filename

    if args.dry_run:
        print("[DRY-RUN] no files written")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_csv, index=False)
        ref_df.to_csv(reference_csv, index=False)

    print("Generated:" if not args.dry_run else "Would generate:")
    print(f"  {params_csv}")
    print(f"  {reference_csv}")
    print(f"  Number of cases = {len(params_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
